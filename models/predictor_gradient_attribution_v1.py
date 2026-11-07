"""Saved FP32-gradient analysis and a local CPU prediction-boundary observer.

No live entry point, alternate objective, backend change or execution grant.
The observer is deliberately restricted to the existing small CPU fixture.
"""
from contextlib import AbstractContextManager
import math
import platform
import threading

import torch

LOSS_BOUNDS = {'rtol': 1e-5, 'atol': 2e-6}
INPUT_BOUNDS = {'rtol': 2e-3, 'atol': 2e-7}
# Fixed before inspecting the target distribution; groups partition all elements.
REFERENCE_MAGNITUDE_EDGES = (0., 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, math.inf)


def _saved(t):
    if (not isinstance(t, torch.Tensor) or t.device.type != 'cpu'
            or t.dtype != torch.float32 or t.layout != torch.strided
            or t.requires_grad or t.numel() == 0 or not torch.isfinite(t).all()):
        raise ValueError('Finite detached nonempty CPU FP32 saved values required.')
    return t


def compare_saved(actual, reference, *, kind):
    """Apply only the original bound, with the original as denominator/reference."""
    _saved(actual); _saved(reference)
    if actual.shape != reference.shape or kind not in ('loss', 'input'):
        raise ValueError('Matching shapes and explicit comparison domain required.')
    if kind == 'loss' and actual.ndim != 0:
        raise ValueError('Loss is the actual rank-zero return.')
    bounds = LOSS_BOUNDS if kind == 'loss' else INPUT_BOUNDS
    a, b = actual.double(), reference.double()
    delta = (a-b).abs(); limit = bounds['atol'] + bounds['rtol'] * b.abs()
    ratio = delta/limit; index = int(ratio.reshape(-1).argmax())
    coordinates = []; remainder = index
    for dim in reversed(actual.shape):
        coordinates.insert(0, remainder % dim); remainder //= dim
    result = {'kind': kind, **bounds, 'elements': actual.numel(),
              'violations': int((delta > limit).sum()), 'exact': torch.equal(actual, reference),
              'max_absolute_error': float(delta.max()), 'max_threshold_ratio': float(ratio.max()),
              'worst_ratio_element': {'flat_index': index, 'coordinates': coordinates,
                  'actual': float(a.reshape(-1)[index]), 'reference': float(b.reshape(-1)[index]),
                  'absolute_error': float(delta.reshape(-1)[index]), 'threshold': float(limit.reshape(-1)[index])}}
    result['passed'] = result['violations'] == 0
    return result


def gradient_distribution(actual, reference):
    """Describe all channels/frames and all fixed magnitude bins; never select a subset."""
    _saved(actual); _saved(reference)
    if actual.shape != reference.shape or actual.ndim != 5 or actual.shape[:2] != (1, 3):
        raise ValueError('Matching [1,3,T,H,W] saved gradients required.')
    gate = compare_saved(actual, reference, kind='input')
    delta = (actual.double()-reference.double()).abs()
    limit = INPUT_BOUNDS['atol'] + INPUT_BOUNDS['rtol'] * reference.double().abs()
    bad = delta > limit; sign = actual.sign() != reference.sign()
    a, b = actual.reshape(-1), reference.reshape(-1)
    opposite = ((a > 0) & (b < 0)) | ((a < 0) & (b > 0))
    zero = ((a == 0) != (b == 0))
    per_frame_channel = []
    for c in range(3):
        for t in range(actual.shape[2]):
            mask = bad[0,c,t]; direction = sign[0,c,t]
            per_frame_channel.append({'channel': c, 'frame': t, 'elements': mask.numel(),
                'violations': int(mask.sum()), 'sign_disagreements': int(direction.sum()),
                'max_absolute_error': float(delta[0,c,t].max()),
                'max_threshold_ratio': float((delta[0,c,t]/limit[0,c,t]).max())})
    strata = []; counts = torch.zeros_like(b, dtype=torch.int8)
    for low, high in zip(REFERENCE_MAGNITUDE_EDGES[:-1], REFERENCE_MAGNITUDE_EDGES[1:]):
        mask = (b.abs().double() >= low) & (b.abs().double() < high)
        counts += mask.to(torch.int8)
        strata.append({'lower_inclusive': low, 'upper_exclusive': None if math.isinf(high) else high,
            'elements': int(mask.sum()), 'violations': int((mask & bad.reshape(-1)).sum()),
            'sign_disagreements': int((mask & sign.reshape(-1)).sum())})
    if not torch.all(counts == 1):
        raise ValueError('Magnitude strata did not partition the complete tensor.')
    assert sum(r['violations'] for r in per_frame_channel) == gate['violations']
    assert sum(r['violations'] for r in strata) == gate['violations']
    return {'gate': gate, 'per_frame_channel': per_frame_channel, 'reference_magnitude_strata': strata,
        'sign_disagreements': int(sign.sum()), 'opposite_nonzero_signs': int(opposite.sum()),
        'one_zero_sign_changes': int(zero.sum()), 'sign_disagreements_outside_original_bound': int((sign & bad).sum()),
        'sign_comparison_is_not_attack_validation': True, 'root_cause_established': False}


def l1_cotangent(prediction, selected_targets, *, draws=8, families=2):
    """CPU reconstruction of the native B=1 sum/mean L1 cotangent; no oracle claim."""
    _saved(prediction); _saved(selected_targets)
    if prediction.shape != selected_targets.shape or prediction.ndim != 3 or prediction.shape[0] != 1:
        raise ValueError('Matching native [1,target_tokens,features] predictions required.')
    if type(draws) is not int or type(families) is not int or draws != 8 or families != 2:
        raise ValueError('Original eight-draw/two-family weighting required.')
    # Retain FP32 division order from outer draw mean, family mean, feature mean.
    scale = torch.ones((), dtype=torch.float32) / draws / families / (prediction.shape[1]*prediction.shape[2])
    return (prediction-selected_targets).sign() * scale


def vjp_decomposition(g_oo, g_mo, g_om, g_mm):
    """Describe four saved J-path/cotangent cells, without computing a derivative.

    g_ab = J_a^T c_b. Jacobian-path change includes internal forward rounding;
    this identity does not isolate pure backward arithmetic or prove causation.
    """
    for t in (g_oo, g_mo, g_om, g_mm): _saved(t)
    if any(t.shape != g_oo.shape for t in (g_mo, g_om, g_mm)):
        raise ValueError('Four identically shaped saved VJP cells required.')
    oo, mo, om, mm = (t.double() for t in (g_oo, g_mo, g_om, g_mm))
    delta = mm-oo
    path_first = (mo-oo) + (mm-mo)
    cotangent_first = (om-oo) + (mm-om)
    return {'total_delta': delta, 'jacobian_change_at_original_cotangent': mo-oo,
            'cotangent_change_at_candidate_jacobian': mm-mo,
            'cotangent_change_at_original_jacobian': om-oo,
            'jacobian_change_at_candidate_cotangent': mm-om,
            'interaction': mm-mo-om+oo,
            'path_first_reconstruction_error': float((path_first-delta).abs().max()),
            'cotangent_first_reconstruction_error': float((cotangent_first-delta).abs().max()),
            'pure_backward_cause_established': False}


class LocalPredictionObserver(AbstractContextManager):
    """Non-replacing hooks, supported only for the existing small CPU fixture.

    Collect actual output and its actual incoming cotangent. No surrogate loss,
    autograd call, SDPA replacement, output/gradient rewrite or target entry.
    """
    def __init__(self, model, targets, banks):
        if platform.node() in ('linux10603', 'linux10609'):
            raise ValueError('Local observer proof only; no university execution grant.')
        _saved(targets)
        if tuple(targets.shape) != (1,32,128) or len(banks) != 8 or any(len(b) != 2 for b in banks):
            raise ValueError('Only the existing tiny CPU EOT8/two-family fixture is supported.')
        if any(t.device.type != 'cpu' for t in (*model.parameters(), *model.buffers())):
            raise ValueError('Local observer proof cannot use CUDA or another device.')
        self.model, self.targets = model, targets
        self.pairs = [pair for draw in banks for pair in draw]
        for pair in self.pairs:
            if pair.target.device.type != 'cpu' or pair.context.device.type != 'cpu':
                raise ValueError('CPU mask operands required.')
        self.records = []; self.active = False; self.closed = False
        self.handles = []; self.module_hook = None; self.lock = threading.RLock()

    def __enter__(self):
        if self.active or self.closed:
            raise ValueError('Single-use local observer; never reset or reuse it.')
        self.active = True
        try:self.module_hook = self.model.predictor.register_forward_hook(self._forward, with_kwargs=True)
        except BaseException:
            self.active = False; self.closed = True; raise
        return self

    def _forward(self, module, args, kwargs, output):
        with self.lock:
            i = len(self.records)
            if not self.active or i >= 16 or module is not self.model.predictor:
                raise ValueError('Unexpected extra/inactive predictor call.')
            pair = self.pairs[i]
            if (len(args) != 3 or kwargs != {'mask_index': 0}
                    or not torch.equal(args[1],pair.context) or not torch.equal(args[2],pair.target)):
                raise ValueError('Original predictor call order/masks changed.')
            if (not isinstance(output,torch.Tensor) or output.device.type != 'cpu'
                    or output.dtype != torch.float32 or not output.requires_grad
                    or list(output.shape) != [1,pair.target.shape[1],128]):
                raise ValueError('Unexpected native predictor return.')
            selected = self.targets.gather(1, pair.target.unsqueeze(-1).expand(-1,-1,128))
            row = {'draw':i//2, 'family':i%2, 'prediction':output.detach().clone(),
                   'selected_targets':selected.detach().clone(), 'cotangent':None}
            self.records.append(row)
            def incoming(gradient):
                with self.lock:
                    if not self.active or row['cotangent'] is not None:
                        raise ValueError('Duplicate/out-of-scope prediction cotangent.')
                    _saved(gradient.detach())
                    if gradient.shape != row['prediction'].shape:
                        raise ValueError('Prediction cotangent shape changed.')
                    row['cotangent'] = gradient.detach().clone()
                return None
            self.handles.append(output.register_hook(incoming))
        return None

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.module_hook is not None:self.module_hook.remove()
            for handle in self.handles:handle.remove()
        finally:self.active = False; self.closed = True
        if exc_type is None and (len(self.records) != 16 or any(r['cotangent'] is None for r in self.records)):
            raise ValueError('All sixteen native predictions and cotangents are required.')
        return False
