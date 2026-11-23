"""Frozen evaluator flow with explicit local ownership and no-grad observation bridges.

The caller restricts entry to primary/convergence. Other branches are preserved
for an exact source inverse and remain unreachable through LocalUnit.
"""
from dataclasses import asdict
import math
import time
import torch
from attacks.pgd import fgsm
from attacks.pgd_workload_v1 import pgd_linf_workload_v1
from models.candidate_evaluation import seeds_for
from models.runtime import synchronize
from models.training import rng_state, restore_rng

def evaluate_local_unit(model, clean, masks_for, settings, *, video_id, base_seed=20261028,
                  sensitivity_banks=0, binding):
    """Microbatch one. No gradient or checkpoint mutations; scoring never selects attacks.

    A single attack bank is fixed across all steps/restarts, matching the pilot.
    Common seeds pair checkpoints/conditions; perturbations are freshly optimized.
    The random control is scored as sampled, with no loss-based selection.
    """
    if clean.shape[0] != 1 or clean.dtype != torch.float32:
        raise ValueError('Evaluation requires one FP32 clip at a time.')
    if sensitivity_banks not in (0, 2):
        raise ValueError('Use zero or the two declared extra scoring banks.')
    if (any(p.grad is not None for p in model.parameters())
            or any(p.requires_grad for p in model.target_encoder.parameters())):
        raise ValueError('Evaluate at a gradient-free boundary with a frozen target.')
    device = clean.device.type
    saved_rng = rng_state(device)
    modes = [(module, module.training) for module in model.modules()]
    synchronize(device)
    started = time.perf_counter()
    seeds = seeds_for(settings, video_id, base_seed=base_seed, sensitivity_banks=sensitivity_banks)
    # Check the actual upstream seed space as well as the full hashes.
    attack_seeds = {s % (2**31 - 1) for s in seeds['attack']}
    score_seeds = {s % (2**31 - 1) for s in seeds['score']}
    if (len(attack_seeds) != len(seeds['attack']) or len(score_seeds) != len(seeds['score'])
            or attack_seeds & score_seeds):
        raise ValueError('Mask seed collision; revise the seed contract before any trajectory.')

    def bank(values):
        return [[pair.to(device) for pair in masks_for(seed)] for seed in values]

    try:
        model.eval()
        targets = model.clean_targets(clean)
        binding.prepare(model, clean, targets)
        attack = binding.objective(model, targets, bank(seeds['attack']))
        score = binding.objective(model, targets, bank(seeds['score']))
        with torch.no_grad():
            clean_loss = score(clean).item()
            attack_clean_loss = attack(clean).item()
        if settings.method == 'fgsm':
            result = fgsm(attack, clean, epsilon=settings.epsilon_255 / 255)
            adversarial, attack_loss = result.adversarial, result.losses.item()
        elif settings.method == 'random':
            generator = torch.Generator(device='cpu').manual_seed(seeds['start'])
            noise = torch.rand(clean.shape, generator=generator, dtype=torch.float32).to(device)
            adversarial = (clean.detach() + (noise * 2 - 1) * settings.epsilon_255 / 255).clamp(0, 1)
            with torch.no_grad():
                attack_loss = attack(adversarial).item()
        else:
            result = pgd_linf_workload_v1(attack, clean, epsilon=settings.epsilon_255 / 255,
                              step_size=settings.step_size_255 / 255, steps=settings.steps,
                              restarts=settings.restarts, seed=seeds['start'], gradient_context=binding.context)
            adversarial, attack_loss = result.adversarial, result.losses.item()
        with torch.no_grad():
            adversarial_loss = score(adversarial).item()
            extra = []
            for audit in range(sensitivity_banks):
                values = seeds[f'audit_{audit}']
                audit_seeds = {s % (2**31 - 1) for s in values}
                if len(audit_seeds) != len(values) or audit_seeds & (attack_seeds | score_seeds):
                    raise ValueError('Audit scoring seed collision.')
                score_seeds.update(audit_seeds)
                seeds[f'audit_{audit}'] = values
                audit_score = binding.objective(model, targets, bank(values))
                extra.append({'clean_loss': audit_score(clean).item(),
                              'adversarial_loss': audit_score(adversarial).item()})
        delta = (adversarial - clean).abs().max().item()
        metrics = {'video_id': video_id, 'method': settings.method, 'clean_loss': clean_loss,
                   'adversarial_loss': adversarial_loss, 'attack_clean_loss': attack_clean_loss,
                   'attack_loss': attack_loss, 'max_pixel_delta': delta,
                   'scoring_audits': extra, 'seeds': seeds, 'settings': asdict(settings)}
        values = [clean_loss, adversarial_loss, attack_clean_loss, attack_loss, delta]
        values.extend(value for audit in extra for value in audit.values())
        if not all(math.isfinite(v) for v in values) or delta > settings.epsilon_255 / 255 + 1e-7:
            raise ValueError('Evaluation has nonfinite losses or violates its pixel bound.')
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Evaluation accumulated parameter gradients.')
        synchronize(device)
        metrics['seconds'] = time.perf_counter() - started
        return metrics, adversarial.detach()
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(saved_rng, device)
