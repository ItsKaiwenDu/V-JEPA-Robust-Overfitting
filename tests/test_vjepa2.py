"""Scientific plumbing checks on small upstream transformers, not pretrained results."""

import copy
import hashlib
import importlib
from functools import partial

import pytest
import torch
from torch import nn

from models.diagnostics import run_smoke_checks, state_digest, upstream_loss_reference
from models.vjepa2 import (FixedTargetVJEPA, MaskPair, canonical_state, load_checkpoint,
                          load_components, sample_masks, use_upstream)


@pytest.fixture
def model(upstream):
    torch.manual_seed(7)
    vit = importlib.import_module("src.models.vision_transformer")
    pred = importlib.import_module("src.models.predictor")
    common = dict(img_size=16, patch_size=4, num_frames=4, tubelet_size=2,
                  use_rope=True, num_heads=2, depth=2, use_sdpa=True)
    # Preserve the released attention head dimensions (64 encoder / 32
    # predictor); smaller rotary dimensions are not supported upstream.
    encoder = vit.VisionTransformer(embed_dim=128, norm_layer=partial(nn.LayerNorm, eps=1e-6), **common)
    predictor = pred.vit_predictor(embed_dim=128, predictor_embed_dim=64,
                                   use_mask_tokens=True, num_mask_tokens=10, **common)
    target = copy.deepcopy(encoder)
    # Deliberately distinguish the target from the context encoder.
    with torch.no_grad():
        target.patch_embed.proj.weight.add_(0.01)
    return FixedTargetVJEPA(encoder, predictor, target)


@pytest.fixture
def pixels():
    return 0.2 + 0.6 * torch.rand(2, 3, 4, 16, 16, generator=torch.Generator().manual_seed(17))


@pytest.fixture
def masks():
    def indices(start, stop):
        return torch.arange(start, stop).expand(2, -1).clone()
    # Unequal family sizes catch accidentally weighting families by token count.
    return [MaskPair(indices(0, 8), indices(8, 32)),
            MaskPair(indices(0, 24), indices(24, 32))]


def checkpoint_fixture(model):
    return {name: {f"module.backbone.{key}": value.clone()
                   for key, value in getattr(model, name).state_dict().items()}
            for name in ("encoder", "target_encoder", "predictor")}


def test_native_loss_matches_actual_upstream_function(model, pixels, masks, upstream):
    targets = model.clean_targets(pixels)
    actual = model(pixels, targets, masks)
    reference = upstream_loss_reference(model, pixels, masks, upstream)
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual, model.loss_per_example(pixels, targets, masks).mean())


def test_smoke_checks_on_small_upstream_models(model, pixels, masks, upstream):
    model.encoder.use_activation_checkpointing = True
    model.predictor.use_activation_checkpointing = True
    checks = run_smoke_checks(model, pixels, masks, upstream)
    assert checks["passed"] and checks["input_gradient_abs_max"] > 0
    assert all(value > 0 for value in checks["parameter_gradient_l2"].values())


def test_input_gradients_do_not_reach_hidden_pixels_or_clean_targets(model, pixels, masks):
    clean = pixels.clone().requires_grad_(True)
    targets = model.clean_targets(clean)
    context = pixels.clone().requires_grad_(True)
    gradient, = torch.autograd.grad(model(context, targets, masks[:1]), context)
    visible = torch.zeros(2, 32, dtype=torch.bool).scatter_(1, masks[0].context, True)
    visible = visible.view(2, 2, 4, 4).repeat_interleave(2, 1).repeat_interleave(4, 2).repeat_interleave(4, 3)
    pixel_grad = gradient.permute(0, 2, 3, 4, 1)
    assert torch.count_nonzero(pixel_grad[~visible]) == 0
    assert torch.count_nonzero(pixel_grad[visible]) > 0
    assert clean.grad is None and not targets.requires_grad
    assert all(p.grad is None for p in model.parameters())


def test_fp32_input_gradient_against_finite_difference(model, pixels, masks):
    targets = model.clean_targets(pixels)
    context = pixels.clone().requires_grad_(True)
    gradient, = torch.autograd.grad(model(context, targets, masks), context)
    index = gradient.abs().argmax().item()
    direction = torch.zeros_like(pixels)
    direction.flatten()[index] = 1
    step = 0.002
    with torch.no_grad():
        numeric = (model(pixels + step * direction, targets, masks)
                   - model(pixels - step * direction, targets, masks)) / (2 * step)
    torch.testing.assert_close(numeric, gradient.flatten()[index], rtol=0.05, atol=2e-5)


def test_optimizer_can_update_context_and_predictor_but_not_target(model, pixels, masks):
    before = {name: state_digest(getattr(model, name)) for name in ("encoder", "predictor", "target_encoder")}
    model.train()
    assert model.encoder.training and model.predictor.training and not model.target_encoder.training
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.01)
    model(pixels, model.clean_targets(pixels), masks).backward()
    optimizer.step()
    assert state_digest(model.encoder) != before["encoder"]
    assert state_digest(model.predictor) != before["predictor"]
    assert state_digest(model.target_encoder) == before["target_encoder"]
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_strict_loading_preserves_all_three_distinct_states(model):
    checkpoint = checkpoint_fixture(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    counts = load_components(model, checkpoint)
    for name in checkpoint:
        expected = canonical_state(checkpoint[name])
        for key, value in getattr(model, name).state_dict().items():
            torch.testing.assert_close(value, expected[key])
        assert counts[name]["state_tensors"] == len(expected)
    assert not torch.equal(model.encoder.patch_embed.proj.weight, model.target_encoder.patch_embed.proj.weight)
    assert len(model.predictor.mask_tokens) == 10


@pytest.mark.parametrize("defect", ["missing_component", "missing_tensor", "wrong_shape", "unexpected_tensor"])
def test_bad_checkpoints_fail_before_mutating_model(model, defect):
    checkpoint = checkpoint_fixture(model)
    if defect == "missing_component":
        del checkpoint["target_encoder"]
    elif defect == "missing_tensor":
        checkpoint["predictor"].pop("module.backbone.mask_tokens.9")
    elif defect == "wrong_shape":
        checkpoint["predictor"]["module.backbone.predictor_embed.weight"] = torch.zeros(1, 1)
    else:
        checkpoint["predictor"]["module.backbone.unexpected"] = torch.zeros(1)
    before = state_digest(model)
    with pytest.raises(ValueError):
        load_components(model, checkpoint)
    assert state_digest(model) == before


def test_colliding_prefixes_are_rejected():
    with pytest.raises(ValueError, match="collide"):
        canonical_state({"module.backbone.x": torch.zeros(1), "x": torch.zeros(1)})


def test_safe_checkpoint_loading_and_hash_check(model, tmp_path):
    path = tmp_path / "fixture.pt"
    torch.save({**checkpoint_fixture(model), "epoch": 3}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = load_checkpoint(model, path, digest)
    assert result["sha256"] == digest and result["epoch"] == 3
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_checkpoint(model, path, "0" * 64)


def test_mask_sampling_replays_without_changing_callers_rng(upstream):
    rng = torch.random.get_rng_state().clone()
    first = sample_masks(upstream, batch_size=2, seed=3)
    assert torch.equal(rng, torch.random.get_rng_state())
    second = sample_masks(upstream, batch_size=2, seed=3)
    assert len(first) == len(second) == 2
    for a, b in zip(first, second):
        assert torch.equal(a.context, b.context) and torch.equal(a.target, b.target)
        for context, target in zip(a.context, a.target):
            assert not torch.isin(context, target).any()
    different = sample_masks(upstream, batch_size=2, seed=4)
    assert any(not torch.equal(a.context, b.context) for a, b in zip(first, different))


@pytest.mark.parametrize("defect", ["overlap", "out_of_range", "duplicate", "float_indices", "no_masks"])
def test_invalid_masks_are_rejected(model, pixels, masks, defect):
    context, target = masks[0].context.clone(), masks[0].target.clone()
    if defect == "overlap":
        target[:, 0] = context[:, 0]
    elif defect == "out_of_range":
        target[:, 0] = 32
    elif defect == "duplicate":
        target[:, 0] = target[:, 1]
    elif defect == "float_indices":
        target = target.float()
    invalid = [] if defect == "no_masks" else [MaskPair(context, target)]
    with pytest.raises(ValueError):
        model(pixels, model.clean_targets(pixels), invalid)


def test_raw_pixel_space_and_detached_targets_are_enforced(model, pixels, masks):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.clean_targets(pixels + 1)
    with pytest.raises(ValueError, match="detached"):
        model(pixels, model.clean_targets(pixels).requires_grad_(True), masks)


def test_wrong_upstream_revision_is_rejected(upstream):
    with pytest.raises(ValueError, match="revision mismatch"):
        use_upstream(upstream, "0" * 40)


def test_native_pgd_increases_loss_without_changing_model_or_targets(model, pixels, masks):
    from attacks.pgd import native_objective, pgd_linf

    targets = model.clean_targets(pixels)
    reference_targets = targets.clone()
    before = state_digest(model)
    objective = native_objective(model, targets, [masks])
    with torch.no_grad():
        clean_loss = objective(pixels)
    result = pgd_linf(objective, pixels, epsilon=4/255, step_size=1/255, steps=3)
    assert (result.losses >= clean_loss).all() and (result.losses > clean_loss).any()
    assert (result.adversarial - pixels).abs().max() <= 4/255 + 1e-7
    assert state_digest(model) == before and torch.equal(targets, reference_targets)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('adversarial', [False, True])
def test_pilot_update_changes_only_trainable_components(model, pixels, masks, adversarial):
    from models.pilot import effective_update, optimizer_for

    before = state_digest(model.target_encoder)
    context_before = state_digest(model.encoder)
    targets = model.clean_targets(pixels)
    optimizer = optimizer_for(model)
    loss = effective_update(model, optimizer, lambda i: (pixels, targets), lambda i: masks,
                            adversarial=adversarial, accumulation=2)
    assert loss > 0 and state_digest(model.target_encoder) == before
    assert state_digest(model.encoder) != context_before
    assert all(p.grad is None for p in model.parameters())
