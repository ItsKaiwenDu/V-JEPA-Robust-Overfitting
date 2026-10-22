"""Independent FP64 derivatives, separate identities and bounded worker replay."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.cuda_convolution import analytical_reference
from models.cuda_profile import file_hash
from models.execution import apply_model_policy, configure_execution, model_execution_contract, precision_contract
from models.native_diagnostic import tiny_fixture
from models.precision_diagnostic import projection_check, summarize
from models.precision_patch import PrecisionPatchEmbed3D, _PrecisionProjection, precision_scope

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def case(upstream):
    configure_execution('cudnn-fp32', device='cpu')
    torch.manual_seed(20261021)
    model, pixels, masks_for = tiny_fixture(upstream)
    apply_model_policy(model, 'cudnn-fp32')
    model.eval()
    return model, pixels, model.clean_targets(pixels), masks_for(0)


def test_shared_fp64_forward_rounds_once_and_preserves_projection_and_layout():
    torch.manual_seed(12)
    original = SimpleNamespace(proj=nn.Conv3d(3, 7, (2, 4, 4), stride=(2, 4, 4)), patch_size=4, tubelet_size=2, training=False)
    x = torch.randn(2, 3, 4, 8, 8)
    cot = torch.randn(2, 8, 7)
    oracle = analytical_reference(x, original.proj.weight, original.proj.bias, cot)
    values = []
    for backend in ('cudnn', 'native'):
        layer = PrecisionPatchEmbed3D(original, backend)
        assert layer.proj is original.proj and list(layer.state_dict()) == ['proj.weight', 'proj.bias']
        values.append(layer(x))
        assert values[-1].dtype == torch.float32
        assert values[-1].stride() == (56, 1, 8)
        assert torch.equal(values[-1], oracle['output'].float())
    assert torch.equal(*values)


@pytest.mark.parametrize('backend', ['cudnn', 'native'])
def test_fp64_first_order_numeric_derivatives(backend):
    torch.manual_seed(73)
    x = torch.randn(1, 1, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(2, 1, 2, 2, 2, dtype=torch.float64, requires_grad=True)
    b = torch.randn(2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x,w,b: _PrecisionProjection.apply(x,w,b,backend), (x,w,b), fast_mode=True)


@pytest.mark.parametrize('backend', ['cudnn', 'native'])
def test_raw_pixel_weight_bias_vjps_and_full_chain_rule(case, backend):
    model, pixels, targets, masks = case
    ids = tuple(id(p) for p in model.parameters())
    weights = copy.deepcopy(model.state_dict())
    original = model.encoder.patch_embed
    with precision_scope(model, backend, 'cpu') as guard:
        with pytest.raises(ValueError, match='Unknown model execution policy'):
            model_execution_contract(model) # Legacy caches/training must not accept this as FP32.
        validation, tensors = projection_check(model, pixels, targets, masks)
        assert all(v['passed'] for v in validation.values())
        x = pixels.clone().requires_grad_(True)
        loss = model(x, targets, masks)
        gradient, = torch.autograd.grad(loss, x)
        torch.testing.assert_close(gradient, tensors['patch_input_gradient'], rtol=.002, atol=2e-7)
        assert torch.equal(loss.detach(), tensors['tail_loss'])
        guard.check()
    assert model.encoder.patch_embed is original and tuple(id(p) for p in model.parameters()) == ids
    assert model_execution_contract(model)['policy'] == 'cudnn-fp32'
    assert all(torch.equal(v, weights[n]) for n, v in model.state_dict().items())
    assert not any(p.grad is not None for p in model.parameters()) and not targets.requires_grad


def test_scope_restores_original_module_and_policy_on_error(case):
    model, _, _, _ = case
    original = model.encoder.patch_embed
    with pytest.raises(RuntimeError, match='deliberate'):
        with precision_scope(model, 'native', 'cpu'):
            raise RuntimeError('deliberate')
    assert model.encoder.patch_embed is original and model_execution_contract(model)['policy'] == 'cudnn-fp32'


@pytest.mark.parametrize('defect', ['forward', 'backward', 'weight', 'target', 'precision'])
def test_guard_rejects_precision_or_tensor_drift(case, defect):
    model, _, _, _ = case
    with pytest.raises(ValueError):
        with precision_scope(model, 'native', 'cpu') as guard:
            if defect == 'forward': model.encoder.patch_embed.forward_id = 'wrong'
            elif defect == 'backward': model.encoder.patch_embed.backward_backend = 'cudnn'
            elif defect == 'precision': torch.set_float32_matmul_precision('medium')
            else:
                module = model.target_encoder if defect == 'target' else model.encoder
                with torch.no_grad():next(module.parameters()).add_(1)
            guard.check()


@pytest.mark.parametrize('backend', ['cudnn', 'native'])
def test_backward_dispatch_scopes_flags_and_restores_them(case, monkeypatch, backend):
    model, pixels, _, _ = case
    operation = torch.ops.aten.convolution_backward
    flags = []
    def observe(*a, **kw):
        flags.append((torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32))
        return operation(*a, **kw)
    monkeypatch.setattr(torch.ops.aten, 'convolution_backward', observe)
    before = precision_contract()
    layer = PrecisionPatchEmbed3D(model.encoder.patch_embed, backend)
    x = pixels.clone().requires_grad_(True)
    torch.autograd.grad(layer(x).square().sum(), x)
    assert flags == [(backend == 'cudnn', False)] and precision_contract() == before


def test_bias_free_parameter_only_and_rejected_second_derivatives():
    original = SimpleNamespace(proj=nn.Conv3d(1, 2, 2, stride=2, bias=False), patch_size=2, tubelet_size=2, training=False)
    layer = PrecisionPatchEmbed3D(original, 'native')
    layer(torch.ones(1,1,2,4,4)).sum().backward()
    assert layer.proj.weight.grad is not None
    x = torch.rand(1,1,2,4,4, requires_grad=True)
    first, = torch.autograd.grad(layer(x).square().sum(), x, create_graph=True)
    with pytest.raises(RuntimeError):torch.autograd.grad(first.sum(),x)
    with pytest.raises(ValueError):layer(x.double())


@pytest.fixture(scope='module')
def tiny_run(tmp_path_factory):
    out = tmp_path_factory.mktemp('precision-workers') / 'results'
    result = subprocess.run([sys.executable, str(ROOT/'scripts/check_precision.py'), '--diagnostic', '--tiny', '--device', 'cpu',
                             '--output', str(out), '--max-seconds', '90', '--worker-seconds', '20'],
                            cwd=ROOT, capture_output=True, text=True, timeout=100)
    assert result.returncode == 0, result.stdout+result.stderr
    return out, json.loads((out/'session.json').read_text())


def test_four_actual_upstream_workers_have_shared_tokens_and_exact_replay(tiny_run):
    directory, session = tiny_run
    assert session['complete'] and session['passed'] and len(session['tasks']) == 4
    assert session['comparison']['shared_forward_tokens_bitwise_equal']
    assert not session['comparison']['full_model_optimizer_or_recovery_validated']
    assert summarize(directory, session['tasks'], session['provenance'], tiny=True) == session['comparison']


@pytest.mark.parametrize('defect', ['partial', 'timeout', 'source', 'forward', 'reference', 'hash', 'runtime', 'false_parameter_gate'])
def test_incomplete_mixed_or_altered_evidence_rejects(tiny_run, tmp_path, defect):
    import shutil
    directory, session = tiny_run
    copied = tmp_path/'copy'
    shutil.copytree(directory,copied)
    tasks, expected = copy.deepcopy(session['tasks']),copy.deepcopy(session['provenance'])
    if defect == 'partial':tasks.pop()
    elif defect == 'timeout':tasks[0]['status']='timeout'
    elif defect == 'source':expected['precision_sources_sha256']['models/precision_patch.py']='0'*64
    elif defect == 'hash':tasks[0]['result_sha256']='0'*64
    else:
        path=copied/tasks[0]['result'];value=json.loads(path.read_text())
        if defect == 'forward':value['contract']['precision']['forward']='FP32'
        elif defect == 'runtime':value['contract']['precision']['runtime']['cudnn_tf32']=True
        elif defect == 'false_parameter_gate':value['oracle_validation']['weight']['failed_elements']=1
        else:value['contract']['reference']='historical FP32'
        path.write_text(json.dumps(value));tasks[0]['result_sha256']=file_hash(path)
    with pytest.raises(ValueError):summarize(copied,tasks,expected,tiny=True)


@pytest.mark.parametrize('flags', [[], ['--diagnostic','--tiny'], ['--diagnostic','--tiny','--device','cpu','--worker-seconds','181']])
def test_cli_explicit_scope_and_bounds(tmp_path, flags):
    out=tmp_path/'unused'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/check_precision.py'),'--output',str(out),*flags],capture_output=True,text=True)
    assert result.returncode == 2 and not out.exists()


@pytest.mark.parametrize('backend', ['cudnn', 'native'])
def test_full_model_parameter_gradients_match_independent_double_conv_autograd(case, backend):
    model, pixels, targets, masks = case
    reference = copy.deepcopy(model)
    class AutogradPatch(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.proj = original.proj
        def forward(self, x):
            p = self.proj
            return F.conv3d(x.double(),p.weight.double(),p.bias.double(),stride=p.stride).float().flatten(2).transpose(1,2)
    reference.encoder.patch_embed = AutogradPatch(reference.encoder.patch_embed)
    def gradients(m):
        x = pixels.clone().requires_grad_(True)
        named = [(n,p) for n,p in m.named_parameters() if p.requires_grad]
        loss = m(x,targets,masks)
        grads = torch.autograd.grad(loss,[x,*[p for n,p in named]],allow_unused=True)
        return loss,dict(zip(['input',*[n for n,p in named]],grads))
    ref_loss, ref_grads = gradients(reference)
    with precision_scope(model,backend,'cpu'):
        loss, grads = gradients(model)
    torch.testing.assert_close(loss,ref_loss,rtol=1e-5,atol=2e-6)
    for name, expected in ref_grads.items():
        if expected is None:assert grads[name] is None
        else:torch.testing.assert_close(grads[name],expected,rtol=2e-4,atol=2e-6)
