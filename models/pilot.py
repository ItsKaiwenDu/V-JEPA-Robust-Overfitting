"""A measured effective-batch update for feasibility, not a training runner."""

from collections.abc import Callable

import torch

from attacks.pgd import native_objective, pgd_linf


def optimizer_for(model):
    return torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-5},
        {'params': model.predictor.parameters(), 'lr': 1e-4},
    ], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.04, foreach=False)


def effective_update(model, optimizer, load_example: Callable, masks_for: Callable, *,
                     adversarial: bool, accumulation: int = 8, seed: int = 0):
    """Accumulate microbatch-one gradients, preserving the clean target cache.

    The caller's load_example returns raw [1,3,T,H,W] pixels and detached
    clean targets. masks_for(step) gives the two upstream mask families.
    This pilot uses FP32; precision selection still needs measured evidence.
    """
    if accumulation < 1:
        raise ValueError('Accumulation must be positive.')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    for index in range(accumulation):
        clean, targets = load_example(index)
        masks = masks_for(index)
        pixels = clean
        if adversarial:
            objective = native_objective(model, targets, [masks])
            pixels = pgd_linf(objective, clean, epsilon=4/255, step_size=1/255,
                              steps=5, seed=seed + index).adversarial
        loss = model(pixels, targets, masks)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite pilot loss.')
        (loss / accumulation).backward()
        total += loss.detach().item() / accumulation
    if any(p.grad is not None for p in model.target_encoder.parameters()):
        raise RuntimeError('Pilot target encoder acquired gradients.')
    if any(p.grad is not None and not torch.isfinite(p.grad).all()
           for group in optimizer.param_groups for p in group['params']):
        raise RuntimeError('Nonfinite pilot parameter gradient.')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return total
