"""Native backend integration on CPU fixtures; CUDA claims require actual CUDA."""

import copy
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from attacks.pgd import native_objective, pgd_linf
from models.diagnostics import state_digest
from models.execution import (ExecutionGuard, apply_model_policy, configure_execution,
                              model_execution_contract, precision_contract)
from models.native_patch import NativePatchEmbed3D, _NativeConv3d
from models.training import runtime_contract
from models.workflow_diagnostic import cached_training_loader
from test_training import assert_tree_equal, make_engine
from test_vjepa2 import model, pixels, masks
from test_evaluation import clip


@pytest.fixture(autouse=True)
def restore_flags(monkeypatch):
    previous = precision_contract()
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    yield
    torch.use_deterministic_algorithms(previous['deterministic_algorithms'],
                                     warn_only=previous['deterministic_warn_only'])
    torch.set_float32_matmul_precision(previous['matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32 = previous['matmul_tf32']
    for key, value in [('enabled', 'cudnn_enabled'), ('allow_tf32', 'cudnn_tf32'),
                       ('benchmark', 'cudnn_benchmark'), ('deterministic', 'cudnn_deterministic')]:
        setattr(torch.backends.cudnn, key, previous[value])


def native(model):
    configure_execution('native-fp32', device='cpu')
    return apply_model_policy(model, 'native-fp32')


def test_default_configuration_does_not_change_flags_or_weights(model):
    before, weights = precision_contract(), state_digest(model)
    configure_execution('original', device='cpu')
    assert apply_model_policy(model) is model
    assert precision_contract() == before and state_digest(model) == weights
    assert model_execution_contract(model)['policy'] == 'original'


def test_native_reuses_original_projection_parameters_and_checkpoint_keys(model):
    projection, target = model.encoder.patch_embed.proj, model.target_encoder.patch_embed
    state = copy.deepcopy(model.state_dict())
    ids = [id(p) for p in model.parameters()]
    native(model)
    assert model.encoder.patch_embed.proj is projection
    assert model.target_encoder.patch_embed is target
    assert [id(p) for p in model.parameters()] == ids
    assert_tree_equal(model.state_dict(), state)
    model.load_state_dict(state, strict=True)


def test_forward_and_backward_explicitly_disable_cudnn_and_restore_flags(model, pixels, monkeypatch):
    native(model)
    forward, backward = torch.nn.functional.conv3d, torch.ops.aten.convolution_backward
    observed = []

    def check(call):
        def wrapped(*args, **kwargs):
            observed.append((torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32))
            return call(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(torch.nn.functional, 'conv3d', check(forward))
    monkeypatch.setattr(torch.ops.aten, 'convolution_backward', check(backward))
    before = precision_contract()
    x = pixels[:1].clone().requires_grad_(True)
    y = model.encoder.patch_embed(x)
    assert precision_contract() == before  # Backward occurs outside forward's context.
    y.square().sum().backward()
    assert observed == [(False, False), (False, False)]
    assert precision_contract() == before and torch.isfinite(x.grad).all()


def test_native_conv_first_order_derivatives_match_numeric_fp64():
    torch.manual_seed(81)
    x = torch.randn(1, 2, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(3, 2, 2, 2, 2, dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, w, b: _NativeConv3d.apply(
        x, w, b, (2, 2, 2), (0, 0, 0), (1, 1, 1), 1), (x, w, b), fast_mode=True)


def test_bias_free_projection_and_parameter_only_gradient():
    patch = SimpleNamespace(proj=nn.Conv3d(2, 3, 2, stride=2, bias=False),
                            patch_size=2, tubelet_size=2, training=True)
    wrapper = NativePatchEmbed3D(patch)
    x = torch.randn(1, 2, 2, 4, 4)
    wrapper(x).square().sum().backward()
    assert wrapper.proj.weight.grad is not None and torch.isfinite(wrapper.proj.weight.grad).all()


def test_native_rejects_higher_order_differentiation():
    x = torch.randn(1, 1, 2, 2, 2, requires_grad=True)
    w = torch.ones(1, 1, 2, 2, 2, requires_grad=True)
    out = _NativeConv3d.apply(x, w, None, (2, 2, 2), (0, 0, 0), (1, 1, 1), 1)
    first, = torch.autograd.grad(out.square().sum(), x, create_graph=True)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(first.sum(), x)


def test_full_small_model_loss_input_and_parameter_gradients_match(model, pixels, masks):
    alternative = native(copy.deepcopy(model))
    reference_targets = model.clean_targets(pixels)
    assert torch.equal(reference_targets, alternative.clean_targets(pixels))
    values = []
    for m in (model, alternative):
        m.encoder.use_activation_checkpointing = True
        m.predictor.use_activation_checkpointing = True
        x = pixels.clone().requires_grad_(True)
        loss = m(x, reference_targets, masks)
        names, parameters = zip(*[(n, p) for n, p in m.named_parameters() if p.requires_grad])
        gradients = torch.autograd.grad(loss, (x, *parameters), allow_unused=True)
        values.append((loss, dict(zip(('input', *names), gradients))))
    torch.testing.assert_close(values[0][0], values[1][0], rtol=1e-5, atol=1e-7)
    for key, expected in values[0][1].items():
        actual = values[1][1][key]
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
    assert all(p.grad is None for p in alternative.parameters())


def test_native_attack_preserves_clean_target_and_never_accumulates_parameter_gradients(model, pixels, masks):
    native(model)
    model.encoder.use_activation_checkpointing = True
    model.predictor.use_activation_checkpointing = True
    targets = model.clean_targets(pixels)
    before = state_digest(model)
    objective = native_objective(model, targets, [masks])
    attack = pgd_linf(objective, pixels, epsilon=4/255, step_size=1/255, steps=5, seed=17)
    assert (attack.adversarial - pixels).abs().max() <= 4/255 + 1e-7
    assert attack.adversarial.min() >= 0 and attack.adversarial.max() <= 1
    assert (attack.losses >= objective(pixels)).all()
    assert not targets.requires_grad and state_digest(model) == before
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('condition', ['clean-finetune', 'adversarial-finetune'])
def test_native_small_model_exact_recovery_with_activation_checkpointing(model, condition):
    native(model)
    model.encoder.use_activation_checkpointing = True
    model.predictor.use_activation_checkpointing = True
    full = make_engine(copy.deepcopy(model), condition=condition)
    while not full.finished:
        full.update()
    split = make_engine(copy.deepcopy(model), condition=condition)
    split.update()
    saved = split.state()
    resumed = make_engine(copy.deepcopy(model), condition=condition)
    resumed.restore(saved)
    while not resumed.finished:
        resumed.update()
    assert_tree_equal(full.state(), resumed.state())


@pytest.mark.parametrize('change', ['backend', 'tf32', 'target'])
def test_cached_targets_reject_policy_precision_or_target_changes(model, pixels, change):
    native(model)
    panel = SimpleNamespace(rows=[{'video_id': 'a'}], pixels=lambda i: pixels[:1])
    loader = cached_training_loader(model, panel, 'cpu')
    loader(0)
    if change == 'tf32':
        torch.backends.cudnn.allow_tf32 = True
    elif change == 'backend':
        model._diagnostic_execution_policy = 'original'
    else:
        with torch.no_grad():
            next(model.target_encoder.parameters()).add_(.01)
    with pytest.raises(ValueError):
        loader(0)


def test_resume_and_anchor_reject_different_backend_before_weight_mutation(model, tmp_path):
    from models.evaluation_checkpoint import load_anchor
    from test_evaluation_checkpoint import trained_anchor
    original = copy.deepcopy(model)
    trained = trained_anchor(original, tmp_path)
    saved = trained.state()
    native(model)
    before = state_digest(model)
    engine = make_engine(model, provenance={'checkpoint_sha256': 'original'})
    with pytest.raises(ValueError, match='Resume contract differs'):
        engine.restore(saved)
    with pytest.raises(ValueError, match='execution runtime/backend'):
        load_anchor(model, tmp_path, 1, initial_checkpoint_sha256='original')
    assert state_digest(model) == before


def test_runtime_and_parameter_identity_guard_reject_mid_run_change(model):
    native(model)
    engine = make_engine(model)
    before = state_digest(model)
    torch.backends.cudnn.allow_tf32 = True
    for operation in (engine.update, engine.state, lambda: engine.restore({})):
        with pytest.raises(ValueError):
            operation()
    assert state_digest(model) == before and engine.step == 0


def test_completed_calibration_and_evaluation_cannot_cross_backend(model, clip, tmp_path):
    from models.calibration import load_calibration, run_calibration
    from models.evaluation import EvaluationSettings, model_identity
    from models.evaluation_store import EvaluationStore, evaluate_panel
    from test_evaluation import Panel, tiny_masks
    panel = Panel(clip)
    identity = model_identity(model)
    checkpoint = dict(epoch=0, condition='pretrained', seed=None,
                      training_contract_sha256=None, file_sha256='original')
    run_calibration(model, panel, tiny_masks, tmp_path / 'calibration',
                    checkpoint=checkpoint, provenance={}, diagnostic=True)
    store = EvaluationStore(tmp_path / 'evaluation')
    evaluate_panel(model, panel, tiny_masks, EvaluationSettings.for_method('fgsm'), store,
                   checkpoint=checkpoint, provenance={})
    native(model)
    assert model_identity(model) == identity  # Same learned tensors are insufficient.
    with pytest.raises(ValueError, match='Calibration execution'):
        load_calibration(tmp_path / 'calibration', panel.rows, original_identity=identity,
                         diagnostic=True, model=model)
    with pytest.raises(ValueError, match='resume contract changed'):
        evaluate_panel(model, panel, tiny_masks, EvaluationSettings.for_method('fgsm'),
                       EvaluationStore(tmp_path / 'evaluation', resume=True),
                       checkpoint=checkpoint, provenance={})


def test_backend_policy_refuses_mps_and_late_cuda_before_selection(monkeypatch):
    with pytest.raises(ValueError, match='CPU fixtures and CUDA'):
        configure_execution('native-fp32', device='mps')
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    with pytest.raises(RuntimeError, match='before CUDA'):
        configure_execution('native-fp32', device='cuda')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual CUDA needed for native backward kernel evidence')
def test_cuda_backward_outside_forward_context_uses_native_operator():
    from test_cuda_reproducibility import isolated
    result = isolated('''
from models.execution import configure_execution
from models.native_probe import native_backward_probe, validate_probe
configure_execution('native-fp32', device='cuda')
evidence = native_backward_probe('cuda')
assert evidence['actual_cuda_kernel_evidence']
assert evidence['kernel_boundary_checks']['passed']
assert validate_probe(evidence, device='cuda')
''')
    assert result.returncode == 0, result.stderr + result.stdout
