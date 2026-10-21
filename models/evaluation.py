"""Fresh native-loss attacks with separate, reproducible scoring mask banks."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time

import torch

from attacks.pgd import fgsm, native_objective, pgd_linf
from models.diagnostics import state_digest
from models.runtime import synchronize
from models.training import TrainingMaskSampler, restore_rng, rng_state

SEED_SCHEMA = 'evaluation-v1-json-sha256-63bit-mask-mod31'


def evaluation_seed(base, namespace, video_id, restart=0, draw=0):
    if (type(base) is not int or not 0 <= base < 2**63 or not isinstance(namespace, str)
            or not namespace or not isinstance(video_id, str) or not video_id
            or any(type(n) is not int or n < 0 for n in (restart, draw))):
        raise ValueError('Invalid evaluation seed fields.')
    data = json.dumps(['evaluation-v1', base, namespace, video_id, restart, draw],
                      separators=(',', ':'), ensure_ascii=True).encode('ascii')
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'big') % 2**63


class EvaluationMaskSampler:
    """Map the 63-bit hash to the pinned sampler's signed 32-bit counter range."""

    def __init__(self, upstream, **kwargs):
        self.sampler = TrainingMaskSampler(upstream, **kwargs)

    def __call__(self, seed):
        return self.sampler(seed % (2**31 - 1))


@dataclass(frozen=True)
class EvaluationSettings:
    method: str
    steps: int
    restarts: int
    attack_masks: int
    scoring_masks: int
    epsilon_255: float = 4.
    step_size_255: float = 1.

    def __post_init__(self):
        expected = {'routine': (5, 1, 1, 1), 'primary': (20, 2, 4, 4),
                    'convergence': (50, 2, 8, 4), 'fgsm': (1, 1, 4, 4), 'random': (1, 1, 4, 4)}
        if (self.method not in expected or any(type(n) is not int for n in
                (self.steps, self.restarts, self.attack_masks, self.scoring_masks))
                or (self.steps, self.restarts, self.attack_masks, self.scoring_masks) != expected[self.method]):
            raise ValueError('Evaluation strength differs from the declared staged method.')
        if self.epsilon_255 != 4 or self.step_size_255 not in (.5, 1, 2):
            raise ValueError('Use epsilon 4/255 and a declared calibration step size.')
        if self.method == 'routine' and self.step_size_255 != 1:
            raise ValueError('Routine PGD-5 retains step size 1/255.')

    @classmethod
    def for_method(cls, method, step_size_255=1.):
        dimensions = {'routine': (5, 1, 1, 1), 'primary': (20, 2, 4, 4),
                      'convergence': (50, 2, 8, 4), 'fgsm': (1, 1, 4, 4), 'random': (1, 1, 4, 4)}
        if method not in dimensions:
            raise ValueError('Unknown evaluation method.')
        return cls(method, *dimensions[method], step_size_255=step_size_255)


def model_identity(model):
    return {name: state_digest(getattr(model, name)) for name in ('encoder', 'predictor', 'target_encoder')}


def evaluate_clip(model, clean, masks_for, settings, *, video_id, base_seed=20261020,
                  sensitivity_banks=0):
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
    score_namespace = 'routine_score' if settings.method == 'routine' else 'primary_score'
    attack_namespace = settings.method + '_attack'
    seeds = {'attack': [evaluation_seed(base_seed, attack_namespace, video_id, draw=i)
                        for i in range(settings.attack_masks)],
             'score': [evaluation_seed(base_seed, score_namespace, video_id, draw=i)
                       for i in range(settings.scoring_masks)],
             'start': evaluation_seed(base_seed, settings.method + '_start', video_id)}
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
        attack = native_objective(model, targets, bank(seeds['attack']))
        score = native_objective(model, targets, bank(seeds['score']))
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
            result = pgd_linf(attack, clean, epsilon=settings.epsilon_255 / 255,
                              step_size=settings.step_size_255 / 255, steps=settings.steps,
                              restarts=settings.restarts, seed=seeds['start'])
            adversarial, attack_loss = result.adversarial, result.losses.item()
        with torch.no_grad():
            adversarial_loss = score(adversarial).item()
            extra = []
            for audit in range(sensitivity_banks):
                values = [evaluation_seed(base_seed, f'primary_score_audit_{audit}', video_id, draw=i)
                          for i in range(settings.scoring_masks)]
                audit_seeds = {s % (2**31 - 1) for s in values}
                if len(audit_seeds) != len(values) or audit_seeds & (attack_seeds | score_seeds):
                    raise ValueError('Audit scoring seed collision.')
                score_seeds.update(audit_seeds)
                seeds[f'audit_{audit}'] = values
                audit_score = native_objective(model, targets, bank(values))
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
