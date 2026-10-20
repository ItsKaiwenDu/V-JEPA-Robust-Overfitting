"""One-batch correctness diagnostics; these are not research measurements."""

from __future__ import annotations

import ast
import hashlib
import importlib
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from models.vjepa2 import FixedTargetVJEPA, MaskPair


def state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def upstream_loss_reference(model: FixedTargetVJEPA, pixels: torch.Tensor,
                            masks: list[MaskPair], upstream: Path) -> torch.Tensor:
    """Execute the pinned upstream loss function with its own wrapper path.

    Extracting this function avoids importing the entire training launcher
    and its CUDA/distributed/video-decoding dependencies. The source revision
    must already have passed use_upstream().
    """
    source = ast.parse((upstream / "app/vjepa/train.py").read_text())
    functions = [node for node in ast.walk(source)
                 if isinstance(node, ast.FunctionDef) and node.name == "loss_fn"]
    if len(functions) != 1:
        raise RuntimeError("Cannot identify the pinned upstream native loss function.")
    namespace = {"torch": torch, "loss_exp": 1.0,
                 "apply_masks": importlib.import_module("src.masks.utils").apply_masks,
                 "masks_pred": [[pair.target for pair in masks]]}
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(upstream / "app/vjepa/train.py"), "exec"), namespace)
    wrappers = importlib.import_module("src.utils.wrappers")
    # Independent upstream wrappers and loss reduction, using the same raw
    # clip and checkpoint. Constants are from app/vjepa/transforms.py.
    mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    clips = [(pixels - mean) / std]
    context_masks = [[pair.context for pair in masks]]
    target_masks = [[pair.target for pair in masks]]
    with torch.no_grad():
        targets = wrappers.MultiSeqWrapper(model.target_encoder)(clips)
        targets = [F.layer_norm(h, (h.shape[-1],)) for h in targets]
        context = wrappers.MultiSeqWrapper(model.encoder)(clips, context_masks)
        predictions = wrappers.PredictorMultiSeqWrapper(model.predictor)(context, context_masks, target_masks)
        return namespace["loss_fn"](predictions, targets)


def run_smoke_checks(model: FixedTargetVJEPA, clean: torch.Tensor,
                     masks: list[MaskPair], upstream: Path) -> dict:
    """Check native parity, gradients, deterministic replay, and target freezing."""
    started = time.perf_counter()
    model.train()
    model.zero_grad(set_to_none=True)
    target_before = state_digest(model.target_encoder)
    targets = model.clean_targets(clean)
    assert not targets.requires_grad and not model.target_encoder.training
    pixels = clean.detach().clone().requires_grad_(True)
    loss = model(pixels, targets, masks)
    if not torch.isfinite(loss):
        raise RuntimeError("Native loss is not finite.")
    reference = upstream_loss_reference(model, clean, masks, upstream)
    torch.testing.assert_close(loss.detach(), reference, rtol=1e-5, atol=1e-6)
    gradient, = torch.autograd.grad(loss, pixels, retain_graph=True)
    if not torch.isfinite(gradient).all() or gradient.abs().max() == 0:
        raise RuntimeError("Raw-pixel input gradients must be finite and nonzero.")
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Input-only differentiation unexpectedly accumulated parameter gradients.")
    loss.backward()
    gradient_norms = {}
    for name in ("encoder", "predictor"):
        gradients = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise RuntimeError(f"Missing or nonfinite {name} parameter gradients.")
        # MPS does not support float64. Diagnostic reduction happens on CPU;
        # model forward/backward computation stays on the requested device.
        norm = sum(g.detach().cpu().double().square().sum().item() for g in gradients) ** 0.5
        if norm == 0:
            raise RuntimeError(f"Zero {name} parameter gradient norm.")
        gradient_norms[name] = norm
    if any(p.requires_grad or p.grad is not None for p in model.target_encoder.parameters()):
        raise RuntimeError("The clean target encoder is not frozen.")
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        replay = model(clean, model.clean_targets(clean), masks)
    torch.testing.assert_close(loss.detach(), replay, rtol=1e-5, atol=1e-6)
    if state_digest(model.target_encoder) != target_before:
        raise RuntimeError("Target encoder weights changed during the diagnostic.")
    return {"passed": True, "loss": loss.item(), "upstream_loss": reference.item(),
            "replay_loss": replay.item(), "input_gradient_abs_max": gradient.abs().max().item(),
            "parameter_gradient_l2": gradient_norms,
            "target_state_sha256": target_before, "elapsed_seconds": time.perf_counter() - started,
            "limitations": ["Synthetic input; no video decoding or dataset validation.",
                            "No attack, optimizer run, training curve, or cost estimate."]}
