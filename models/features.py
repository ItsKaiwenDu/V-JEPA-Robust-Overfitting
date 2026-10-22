"""Clean context features, train-only fixed ridge probes, and raw feature diagnostics."""

import copy
import hashlib
import time

import torch

from models.execution import ExecutionGuard
from models.evaluation import model_identity
from models.evaluation_store import json_hash
from models.runtime import synchronize
from models.training import restore_rng, rng_state, runtime_contract

FEATURE_RECIPE = 'clean-context-all-tokens-mean-fp32-v1'
PROBE_RECIPE = 'train-population-standardize-sum-squares-ridge1-intercept-fp64-v1'
STD_FLOOR = 1e-12


def validate_features(bundle):
    rows, features = bundle['rows'], bundle['features']
    if (not rows or features.ndim != 2 or len(features) != len(rows) or features.shape[1] < 1
            or features.device.type != 'cpu' or features.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(features).all()
            or len({r['video_id'] for r in rows}) != len(rows)):
        raise ValueError('Invalid or duplicate feature rows.')
    if bundle['contract']['recipe'] != FEATURE_RECIPE:
        raise ValueError('Unknown feature recipe.')
    for row in rows:
        if not row['source_group_id'] or not row['video_id'] or int(row['class_id']) < 0:
            raise ValueError('Invalid feature labels or group metadata.')


def feature_digest(bundle):
    validate_features(bundle)
    metadata = json_hash({'rows': bundle['rows'], 'contract': bundle['contract'],
                          'shape': list(bundle['features'].shape), 'dtype': str(bundle['features'].dtype)})
    digest = hashlib.sha256(metadata.encode())
    digest.update(bundle['features'].contiguous().numpy().tobytes())
    return digest.hexdigest()


def extract_features(model, panel, *, provenance):
    if not panel.rows or panel.panel not in ('train', 'train_monitor', 'exploration_validation'):
        raise ValueError('Confirmation and test feature extraction remains closed.')
    if (any(p.grad is not None for p in model.parameters())
            or any(p.requires_grad for p in model.target_encoder.parameters())):
        raise ValueError('Extract features only at a gradient-free boundary with a frozen target.')
    device = next(model.encoder.parameters()).device.type
    guard = ExecutionGuard(model, device, target_versions=True)
    identity, modes, rng = model_identity(model), [(m, m.training) for m in model.modules()], rng_state(device)
    synchronize(device)
    started = time.perf_counter()
    result = []
    try:
        model.eval()
        with torch.no_grad():
            for index in range(len(panel.rows)):
                pixels = panel.pixels(index).to(device)
                if (pixels.ndim != 5 or pixels.shape[0] != 1 or pixels.dtype != torch.float32
                        or not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1):
                    raise ValueError('Expected one FP32 clean clip.')
                tokens = model.encoder(model.normalize(pixels))
                if tokens.ndim != 3 or tokens.shape[0] != 1 or not torch.isfinite(tokens).all():
                    raise ValueError('Invalid context token features.')
                guard.check()
                result.append(tokens.mean(dim=1).float().cpu())
        if model_identity(model) != identity:
            raise RuntimeError('Feature extraction changed model or target state.')
        synchronize(device)
        bundle = {'rows': copy.deepcopy(panel.rows), 'features': torch.cat(result),
                  'contract': {'recipe': FEATURE_RECIPE, 'panel': panel.panel,
                               'panel_identity': copy.deepcopy(panel.identity),
                               'encoder_sha256': identity['encoder'], 'target_sha256': identity['target_encoder'],
                               'runtime': runtime_contract(device, model=model), 'provenance': copy.deepcopy(provenance)},
                  'seconds': time.perf_counter() - started}
        validate_features(bundle)
        return bundle
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(rng, device)


def raw_feature_diagnostics(features):
    """Population variance and entropy effective rank of the covariance spectrum.

    These are measurements, not an automatic scientific collapse threshold.
    """
    if features.ndim != 2 or len(features) < 2 or not torch.isfinite(features).all():
        raise ValueError('Need at least two finite feature vectors.')
    x = features.detach().to(device='cpu', dtype=torch.float64)
    centered = x - x.mean(0)
    variance = centered.square().mean(0)
    singular = torch.linalg.svdvals(centered)
    tolerance = singular.max() * max(centered.shape) * torch.finfo(torch.float64).eps
    active = singular[singular > tolerance]
    energy = active.square()
    if len(energy) and energy.sum() > 0:
        probabilities = energy / energy.sum()
        effective = torch.exp(-(probabilities * probabilities.log()).sum()).item()
    else:
        effective = 0.
    return {'examples': len(x), 'dimensions': x.shape[1],
            'population_variance_mean': variance.mean().item(),
            'population_variance_min': variance.min().item(), 'population_variance_max': variance.max().item(),
            'dimensions_below_std_floor': int((variance.sqrt() <= STD_FLOOR).sum()),
            'numerical_centered_rank': len(active), 'covariance_entropy_effective_rank': effective,
            'rank_upper_bound': min(len(x) - 1, x.shape[1]), 'computed_before_standardization': True}


def fit_ridge_probe(training):
    """Minimize sum ||Z W + b - one_hot(y)||^2 + ||W||^2 on training only.

    Lambda is exactly 1 under the SUM loss convention. Use a dual solve for
    N << D. The intercept and centering/scaling are fitted from training only.
    """
    validate_features(training)
    if training['contract']['panel'] != 'train' or len(training['rows']) < 2:
        raise ValueError('Fit the probe only on actual training features.')
    x = training['features'].double()
    labels = torch.tensor([int(r['class_id']) for r in training['rows']], dtype=torch.long)
    classes, indices = labels.unique(sorted=True, return_inverse=True)
    if len(classes) < 2:
        raise ValueError('A classification probe requires at least two training classes.')
    mean = x.mean(0)
    std = (x - mean).square().mean(0).sqrt()
    scale = torch.where(std > STD_FLOOR, std, torch.ones_like(std))
    z = (x - mean) / scale
    # Account explicitly for small floating-point residuals in centered Z.
    zmean = z.mean(0)
    centered = z - zmean
    y = torch.nn.functional.one_hot(indices, len(classes)).double()
    ymean = y.mean(0)
    weights = centered.T @ torch.linalg.solve(centered @ centered.T + torch.eye(len(x), dtype=torch.float64), y - ymean)
    intercept = ymean - zmean @ weights
    if not torch.isfinite(weights).all() or not torch.isfinite(intercept).all():
        raise RuntimeError('Ridge solution is nonfinite.')
    return {'recipe': PROBE_RECIPE, 'lambda': 1., 'loss_reduction': 'sum',
            'classes': classes, 'mean': mean, 'scale': scale, 'weights': weights, 'intercept': intercept,
            'training_feature_sha256': feature_digest(training), 'training_rows': copy.deepcopy(training['rows']),
            'representation': {k: copy.deepcopy(training['contract'][k]) for k in
                               ('recipe', 'encoder_sha256', 'target_sha256', 'runtime', 'provenance')},
            'raw_training_diagnostics': raw_feature_diagnostics(x),
            'standardization_std_floor': STD_FLOOR, 'validation_used_for_fit': False}


def score_ridge_probe(probe, evaluation):
    validate_features(evaluation)
    if probe['recipe'] != PROBE_RECIPE or probe['lambda'] != 1 or probe['loss_reduction'] != 'sum':
        raise ValueError('Probe configuration differs from the fixed recipe.')
    panel = evaluation['contract']['panel']
    if panel not in ('train', 'train_monitor', 'exploration_validation'):
        raise ValueError('Confirmation/test probe scoring remains closed.')
    if any(evaluation['contract'][k] != v for k, v in probe['representation'].items()):
        raise ValueError('Refit a separate probe for each representation checkpoint.')
    training_ids = {r['video_id'] for r in probe['training_rows']}
    if panel == 'exploration_validation':
        training_groups = {r['source_group_id'] for r in probe['training_rows']}
        if (training_ids & {r['video_id'] for r in evaluation['rows']}
                or training_groups & {r['source_group_id'] for r in evaluation['rows']}):
            raise ValueError('Held-out probe panel overlaps the training videos or groups.')
    else:
        expected = {r['video_id']: r for r in probe['training_rows']}
        if any(expected.get(r['video_id']) != r for r in evaluation['rows']):
            raise ValueError('Training-monitor identities are not from probe training.')
    x = evaluation['features'].double()
    if x.shape[1] != len(probe['mean']):
        raise ValueError('Probe feature dimensionality differs.')
    labels = torch.tensor([int(r['class_id']) for r in evaluation['rows']], dtype=torch.long)
    if not torch.isin(labels, probe['classes']).all():
        raise ValueError('Evaluation has a class absent from training.')
    scores = ((x - probe['mean']) / probe['scale']) @ probe['weights'] + probe['intercept']
    if not torch.isfinite(scores).all():
        raise ValueError('Probe scores are nonfinite.')
    predicted = probe['classes'][scores.argmax(dim=1)]
    correct = predicted == labels
    by_class = {str(c.item()): correct[labels == c].double().mean().item() for c in labels.unique(sorted=True)}
    return {'panel': panel, 'examples': len(labels), 'correct': int(correct.sum()),
            'accuracy': correct.double().mean().item(),
            'balanced_accuracy_over_present_classes': sum(by_class.values()) / len(by_class),
            'class_accuracy': by_class, 'predictions': predicted.tolist(),
            'video_ids': [r['video_id'] for r in evaluation['rows']],
            'feature_sha256': feature_digest(evaluation),
            'training_feature_sha256': probe['training_feature_sha256'],
            'raw_feature_diagnostics': raw_feature_diagnostics(x),
            'diagnostic_only': True, 'adversarial_classification_robustness_claim': False}
