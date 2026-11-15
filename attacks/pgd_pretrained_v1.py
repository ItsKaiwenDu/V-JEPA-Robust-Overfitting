"""Opt-in pretrained GPU PGD2/R2 diagnostic; no training or policy registration."""
from collections.abc import Callable
import math
import platform
import torch
from torch import Tensor
from attacks.pgd import AttackResult, _losses

IDENTITY = 'pretrained-pgd2-r2-eot8-native-input-context-diagnostic-v1'
CPU_PGD_SHA256 = '34d310b4518ff4e5c85ce722d1e331e7a9e40b56db1f3c5e10fe3378ffa8c552'
START_SEED = 4417571294737150623


def target_domain(clean, epsilon, step_size, steps, restarts, seed, random_start):
    if platform.node().split('.')[0] != 'linux10603' or platform.system() != 'Linux':
        raise ValueError('Pretrained diagnostic requires linux10603; no local/CUDA fallback.')
    if (torch.is_inference_mode_enabled() or torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda')
            or not isinstance(clean, Tensor) or clean.device != torch.device('cuda:0')
            or clean.dtype != torch.float32 or clean.shape != (1, 3, 16, 256, 256)
            or epsilon != 4/255 or step_size != 1/255 or type(steps) is not int or steps != 2
            or type(restarts) is not int or restarts != 2 or type(seed) is not int or seed != START_SEED
            or random_start is not True):
        raise ValueError('Only declared first-order FP32 GPU PGD2/R2 diagnostic domain.')


def checked_gradient(gradient, current):
    if (not isinstance(gradient, Tensor) or gradient.shape != current.shape or gradient.device != current.device
            or gradient.dtype != current.dtype or gradient.requires_grad or not torch.isfinite(gradient).all()):
        raise ValueError('Full finite first-order input gradient required.')
    return gradient


def pgd_linf_pretrained_v1(objective: Callable[[Tensor], Tensor], clean: Tensor, *, epsilon: float,
             step_size: float, steps: int, restarts: int = 1, seed: int = 0,
             random_start: bool = True, gradient_context) -> AttackResult:
    """Maximize a deterministic per-example objective in raw [0, 1] pixels.

    Epsilon/step size are fractions, e.g. 4/255 and 1/255. Keep the best
    candidate per example over clean input, all starts, and all iterates,
    using only the attack objective. Independent scoring masks belong to
    subsequent evaluation, never candidate selection. Input differentiation
    does not populate or clear model parameter gradients. Model mode is
    controlled by the caller; attacks never change it.
    """
    target_domain(clean, epsilon, step_size, steps, restarts, seed, random_start)
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
                with gradient_context(current) as step:
                    losses = _losses(lambda _: step.losses, current)
                    consider(current, losses)
                    step.note_consider(current, losses)
                    gradient = checked_gradient(step.gradient(), current)
            if not torch.isfinite(gradient).all():
                raise ValueError("Attack input gradient is not finite.")
            current = (current.detach() + step_size * gradient.sign()).clamp(lower, upper)
        with torch.no_grad():
            consider(current, _losses(objective, current))
    return AttackResult(best.detach(), best_loss.detach())
