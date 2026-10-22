"""Independent oracle, actual upstream tail substitution and fail-closed controls."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from models.execution import apply_model_policy, configure_execution
from models.gradient_attribution import (difference, fp64_reference, load_observed, measure,
                                         pixel_vjp, residual_changes, tail_derivative, verify_identity)
from models.native_diagnostic import tiny_fixture
from models.native_patch import NativePatchEmbed3D
from models.training import rng_state

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def case(upstream):
    configure_execution('cudnn-fp32', device='cpu')
    torch.manual_seed(20261021)
    model, pixels, masks_for = tiny_fixture(upstream)
    apply_model_policy(model, 'cudnn-fp32')
    model.eval()
    masks = masks_for(0)
    targets = model.clean_targets(pixels)
    return model, pixels, targets, masks


def test_patch_tail_chain_rule_matches_direct_upstream_derivative_and_restores(case):
    model, pixels, targets, masks = case
    identities = tuple(id(p) for p in model.parameters())
    modes = tuple(m.training for m in model.modules())
    patch = model.encoder.patch_embed
    hooks = (dict(patch._forward_hooks), dict(model.predictor._forward_hooks))
    rng = rng_state('cpu')
    with torch.no_grad():
        tokens = patch(model.normalize(pixels))
    tail = tail_derivative(model, pixels, targets, masks, tokens)
    x = pixels.clone().requires_grad_(True)
    direct_loss = model(x, targets, masks)
    direct, = torch.autograd.grad(direct_loss, x)
    rebuilt = pixel_vjp(model, patch, pixels, tail['cotangents'])
    assert torch.equal(tail['loss'], direct_loss.detach())
    assert difference(rebuilt, direct)['passed']
    assert tail['patch_calls'] == len(masks) and tail['leaf_stride'] == list(tokens.stride())
    assert (dict(patch._forward_hooks), dict(model.predictor._forward_hooks)) == hooks
    assert tuple(id(p) for p in model.parameters()) == identities
    assert tuple(m.training for m in model.modules()) == modes
    assert torch.equal(rng_state('cpu')['torch_cpu'], rng['torch_cpu'])
    assert not targets.requires_grad and not any(p.grad is not None for p in model.parameters())
    assert all(not p.requires_grad for p in model.target_encoder.parameters())


def test_hooks_and_rng_restore_on_prediction_failure(case):
    model, pixels, targets, masks = case
    patch = model.encoder.patch_embed
    with torch.no_grad():
        tokens = patch(model.normalize(pixels))
    def fail(module, inputs):
        torch.rand(5)
        raise RuntimeError('deliberate predictor failure')
    handle = model.predictor.register_forward_pre_hook(fail)
    before = rng_state('cpu')
    try:
        with pytest.raises(RuntimeError, match='deliberate'):
            tail_derivative(model, pixels, targets, masks, tokens)
        assert not patch._forward_hooks and not model.predictor._forward_hooks
        assert torch.equal(before['torch_cpu'], rng_state('cpu')['torch_cpu'])
    finally:
        handle.remove()


def test_trained_weight_shaped_oracle_includes_channel_normalization_and_family_sum(case):
    model, pixels, targets, masks = case
    with torch.no_grad():
        tokens = model.encoder.patch_embed(model.normalize(pixels))
    generator = torch.Generator().manual_seed(99)
    cots = [torch.randn(tokens.shape, generator=generator) for _ in masks]
    oracle = fp64_reference(model, pixels, cots)
    for patch in (model.encoder.patch_embed, NativePatchEmbed3D(model.encoder.patch_embed)):
        actual = pixel_vjp(model, patch, pixels, cots)
        # Random large cotangents amplify rounding, so verify in true FP64 too.
        model.double()
        actual64 = pixel_vjp(model, patch, pixels.double(), [v.double() for v in cots])
        torch.testing.assert_close(actual64, oracle['raw_pixel_vjp'], rtol=1e-12, atol=1e-12)
        assert torch.isfinite(actual).all()
        model.float()


def test_control_measures_both_cotangents_and_both_backward_paths(case):
    model, pixels, targets, masks = case
    artifacts = {}
    result = measure(model, pixels, targets, masks, retain=lambda n, v: artifacts.update({n: v}))
    assert result['control_checks_passed'] and result['identical_token_tail_exact']
    assert result['fixed_cotangent_oracles_agree'] and result['preservation_checks']
    assert len(artifacts) == 9 and len(result['residual_changes']) == 2
    assert not result['training_or_backend_adoption'] and not result['faithful_reconstruction']
    # Supply independent direct full-model gradients to check reconstruction.
    x = pixels.clone().requires_grad_(True)
    loss = model(x, targets, masks)
    gradient, = torch.autograd.grad(loss, x)
    observed = {p: {'gradient': gradient, 'loss': loss.detach()} for p in ('cudnn-fp32', 'native-fp32')}
    rebuilt = measure(model, pixels, targets, masks, observed=observed)
    assert rebuilt['faithful_reconstruction']
    observed['native-fp32']['gradient'] = gradient + 1
    rejected = measure(model, pixels, targets, masks, observed=observed)
    assert not rejected['faithful_reconstruction'] and not rejected['control_checks_passed']


def test_residual_sign_accounting_distinguishes_zeros_and_crossings():
    a, b = torch.tensor([-1., 0., 2., 3.]), torch.tensor([1., 1., 0., 3.])
    row, = residual_changes([a], [b])
    assert row['sign_or_zero_changes'] == 3 and row['nonzero_sign_crossings'] == 1
    assert row['left_zeros'] == row['right_zeros'] == 1
    assert row['largest_magnitude_at_changed_sign'] == 2
    with pytest.raises(ValueError):
        residual_changes([a], [b[:2]])


@pytest.mark.parametrize('defect', ['passed', 'source'])
def test_observed_failed_source_manifest_rejects_before_missing_assets(tmp_path, defect):
    spec = {'schema_version': 1, 'prior_attempt_passed': False, 'provenance': {'source': 'expected'}}
    if defect == 'passed':
        spec['prior_attempt_passed'] = True
    else:
        spec['provenance']['source'] = 'changed'
    path = tmp_path / 'observed.json'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='source identity'):
        load_observed(path, tmp_path, {'source': 'expected'})


@pytest.mark.parametrize('flags', [[], ['--diagnostic', '--tiny'],
                                 ['--diagnostic', '--tiny', '--device', 'cpu', '--max-seconds', '301']])
def test_cli_rejects_implicit_execution_or_unbounded_run(tmp_path, flags):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/check_gradient_attribution.py'),
                             '--output', str(tmp_path / 'unused'), *flags], capture_output=True, text=True)
    assert result.returncode == 2 and not (tmp_path / 'unused').exists()


@pytest.fixture
def observed_files(tmp_path, case):
    from models.cuda_profile import file_hash, tensor_signature
    from models.training import runtime_contract
    from models.diagnostics import state_digest
    model, pixels, targets, masks = case
    directory = tmp_path / 'failed' / 'results'
    directory.mkdir(parents=True)
    spec = {'schema_version': 1, 'prior_attempt_passed': False, 'provenance': {'source': 'fixture'}, 'workers': {}}
    base = {'runtime': runtime_contract('cpu', model=model), 'data': {'fixture': True},
            'pixels': tensor_signature(pixels), 'weights_sha256': state_digest(model), 'targets': tensor_signature(targets),
            'masks': [[tensor_signature(m.context), tensor_signature(m.target)] for m in masks], 'mask_seeds': [0, 100000]}
    for policy in ('cudnn-fp32', 'native-fp32'):
        contract = copy.deepcopy(base)
        contract['execution_policy'] = policy
        if policy == 'native-fp32':
            execution = contract['runtime']['model_execution']
            execution.update(policy=policy, native_backward='aten.convolution_backward-first-order-v1')
            execution['context_patch']['module'] = 'models.native_patch.NativePatchEmbed3D'
        tensors = {'input_gradient': pixels * 0 + .1, 'clean_loss': torch.tensor(.5)}
        tensor_path = directory / (policy + '.pt')
        torch.save(tensors, tensor_path)
        meta = {'finished': True, 'passed': True, 'provenance': spec['provenance'], 'contract': contract,
                'tensors_sha256': file_hash(tensor_path),
                'scientific': {'input_gradient': tensor_signature(tensors['input_gradient']), 'clean_loss': .5}}
        path = directory / (policy + '.json')
        path.write_text(json.dumps(meta))
        spec['workers'][policy] = {'json_file': path.name, 'json_sha256': file_hash(path),
                                  'tensor_file': tensor_path.name, 'tensor_sha256': file_hash(tensor_path), 'contract': contract}
    (directory / 'session.json').write_text(json.dumps({'complete': True, 'passed': False}))
    spec['prior_session_sha256'] = file_hash(directory / 'session.json')
    observed = tmp_path / 'observed.json'
    observed.write_text(json.dumps(spec))
    return observed, directory.parent, base, spec


def test_observed_contract_loads_and_matches_current_fixture(case, observed_files):
    model, pixels, targets, masks = case
    path, run, base, spec = observed_files
    observed = load_observed(path, run, spec['provenance'])
    verify_identity(base['runtime'], base['data'], pixels, base['weights_sha256'], targets, masks, observed)


@pytest.mark.parametrize('defect', ['session_hash', 'json_hash', 'tensor_hash', 'path_escape', 'runtime', 'mask', 'weights'])
def test_observed_hashes_and_shared_identity_reject_mixed_runs(case, observed_files, defect):
    model, pixels, targets, masks = case
    path, run, base, spec = observed_files
    if defect in ('session_hash', 'json_hash', 'tensor_hash', 'path_escape'):
        row = spec['workers']['cudnn-fp32']
        if defect == 'session_hash':
            spec['prior_session_sha256'] = '0' * 64
        elif defect == 'path_escape':
            row['json_file'] = '../outside.json'
        else:
            row['json_sha256' if defect == 'json_hash' else 'tensor_sha256'] = '0' * 64
        path.write_text(json.dumps(spec))
        with pytest.raises(ValueError):
            load_observed(path, run, spec['provenance'])
    else:
        observed = load_observed(path, run, spec['provenance'])
        if defect == 'runtime':
            base['runtime']['matmul_tf32'] = True
        elif defect == 'weights':
            base['weights_sha256'] = '0' * 64
        else:
            observed['cudnn-fp32']['contract']['masks'][0][0]['sha256'] = '0' * 64
        with pytest.raises(ValueError):
            verify_identity(base['runtime'], base['data'], pixels, base['weights_sha256'], targets, masks, observed)
