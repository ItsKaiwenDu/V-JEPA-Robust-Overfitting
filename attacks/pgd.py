"""L-infinity projected ascent with explicit per-example candidate selection."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class AttackResult:
    adversarial: Tensor
    losses: Tensor


def _losses(objective: Callable[[Tensor], Tensor], pixels: Tensor) -> Tensor:
    values = objective(pixels)
    if values.shape != (pixels.shape[0],):
        raise ValueError("Attack objective must return one loss per example, shape [B].")
    if not torch.isfinite(values).all():
        raise ValueError("Attack objective returned nonfinite losses.")
    return values


def pgd_linf(objective: Callable[[Tensor], Tensor], clean: Tensor, *, epsilon: float,
             step_size: float, steps: int, restarts: int = 1, seed: int = 0,
             random_start: bool = True) -> AttackResult:
    """Maximize a deterministic per-example objective in raw [0, 1] pixels.

    Epsilon/step size are fractions, e.g. 4/255 and 1/255. Keep the best
    candidate per example over clean input, all starts, and all iterates,
    using only the attack objective. Independent scoring masks belong to
    subsequent evaluation, never candidate selection. Input differentiation
    does not populate or clear model parameter gradients. Model mode is
    controlled by the caller; attacks never change it.
    """
    if clean.ndim < 2 or clean.shape[0] == 0 or clean.dtype != torch.float32:
        raise ValueError("Expected a nonempty batch of float32 raw pixels.")
    if not torch.isfinite(clean).all() or clean.min() < 0 or clean.max() > 1:
        raise ValueError("Clean pixels must be finite and inside [0, 1].")
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("Epsilon must be finite and in [0, 1].")
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("Step size must be finite and positive.")
    if type(steps) is not int or steps < 1 or type(restarts) is not int or restarts < 1:
        raise ValueError("Steps and restarts must be positive integers.")
    center = clean.detach()
    lower = (center - epsilon).clamp(0, 1)
    upper = (center + epsilon).clamp(0, 1)
    best = center.clone()
    with torch.no_grad():
        best_loss = _losses(objective, best).detach().clone()
    if epsilon == 0:
        return AttackResult(best, best_loss)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (len(center),) + (1,) * (center.ndim - 1)

    def consider(candidate: Tensor, values: Tensor) -> None:
        nonlocal best, best_loss
        improved = values.detach() > best_loss
        best = torch.where(improved.view(shape), candidate.detach(), best)
        best_loss = torch.where(improved, values.detach(), best_loss)

    for _ in range(restarts):
        if random_start:
            noise = torch.rand(center.shape, generator=generator, dtype=torch.float32).to(center.device)
            current = (center + (2 * noise - 1) * epsilon).clamp(lower, upper)
        else:
            current = center.clone()
        for _ in range(steps):
            current = current.detach().requires_grad_(True)
            with torch.enable_grad():
                losses = _losses(objective, current)
                consider(current, losses)
                gradient, = torch.autograd.grad(losses.sum(), current)
            if not torch.isfinite(gradient).all():
                raise ValueError("Attack input gradient is not finite.")
            current = (current.detach() + step_size * gradient.sign()).clamp(lower, upper)
        with torch.no_grad():
            consider(current, _losses(objective, current))
    return AttackResult(best.detach(), best_loss.detach())


def fgsm(objective: Callable[[Tensor], Tensor], clean: Tensor, *, epsilon: float) -> AttackResult:
    """One sign-gradient step from clean input, retaining clean if it scores higher."""
    return pgd_linf(objective, clean, epsilon=epsilon, step_size=max(epsilon, 1e-12),
                    steps=1, random_start=False)


def native_objective(model, clean_targets: Tensor, mask_bank: Sequence[Sequence]):
    """Bind clean detached targets and a fixed mask bank for PGD/EOT.

    Each bank entry contains the upstream mask families for one draw. Targets
    are computed once from the clean clip outside the attack. The bank is
    reused for every step and can be independently resampled for the next
    batch or next checkpoint evaluation.
    """
    if clean_targets.requires_grad or not mask_bank or any(not masks for masks in mask_bank):
        raise ValueError("Use detached clean targets and a nonempty mask bank.")
    banks = tuple(tuple(masks) for masks in mask_bank)

    def objective(pixels):
        # Accumulate sequentially so only one scalar per example is stacked;
        # the autograd graph still includes every EOT draw.
        return torch.stack([model.loss_per_example(pixels, clean_targets, masks)
                            for masks in banks]).mean(dim=0)

    return objective
