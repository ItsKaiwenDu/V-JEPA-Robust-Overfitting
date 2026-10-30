"""Separately identified projection-reuse diagnostic; no study-policy integration."""
from contextlib import contextmanager

import torch
from torch import nn

from attacks.pgd import native_objective
from models.execution import ExecutionGuard
from models.precision_patch import PrecisionPatchEmbed3D
from models.projection_reuse import BankProjectionReuse, _IndependentVJP, _bytes

KIND = 'bank-projection-reuse-diagnostic-v1'
MODES = ('recompute', 'reuse')
ACTIVE_POLICIES = {mode: KIND + '/' + mode for mode in MODES}


class DiagnosticProjectionBank(BankProjectionReuse):
    """New diagnostic adapter; the original CPU prototype remains CPU-only."""
    def __init__(self, projection, *, expected_calls, device, diagnostic_kind):
        if (diagnostic_kind != KIND or device not in ('cpu', 'cuda')
                or type(projection) is not PrecisionPatchEmbed3D
                or projection.backward_backend != 'native'
                or projection.proj.weight.device.type != device
                or type(expected_calls) is not int or expected_calls < 1):
            raise ValueError('Require the explicit native diagnostic and matching operands.')
        if device == 'cpu':
            super().__init__(projection, expected_calls=expected_calls)
        else:
            if torch.__version__.split('+')[0] != '2.8.0':
                raise ValueError('CUDA diagnostic requires the existing PyTorch 2.8.0 runtime.')
            self.projection, self.expected_calls = projection, expected_calls
            self.requests = self.forward_computations = 0
            self._active = self._entered = False
            self._input = self._tokens = self._parameter_copies = None
        self.device = device

    def _check(self):
        if torch.is_autocast_enabled(self.device):
            raise ValueError('Autocast is not supported by this diagnostic.')
        super()._check()

    def __call__(self, x):
        if not self._active or self.requests >= self.expected_calls:
            raise ValueError('Bank is closed or declared request count was exceeded.')
        self._check()
        if x.device.type != self.device or x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError('Require finite FP32 input on the declared diagnostic device.')
        signature = (tuple(x.shape), tuple(x.stride()), x.requires_grad)
        if self._input is None:
            self._input_signature, self._input = signature, _bytes(x).clone()
            with torch.no_grad():
                self._tokens = self.projection(x).detach()
            self.forward_computations += 1
        elif signature != self._input_signature or not torch.equal(_bytes(x), self._input):
            raise ValueError('Diagnostic input changed within a bank.')
        self.requests += 1
        p = self.projection.proj
        return _IndependentVJP.apply(x, p.weight, p.bias, self.projection.backward_backend, self._tokens)


class DirectProjectionBank:
    def __init__(self, projection, expected_calls):
        self.projection, self.expected_calls = projection, expected_calls
        self.requests = self.forward_computations = 0

    def __enter__(self):
        return self

    def __call__(self, x):
        if self.requests >= self.expected_calls:
            raise ValueError('Reference exceeded its declared projection requests.')
        self.requests += 1
        self.forward_computations += 1
        return self.projection(x)

    def __exit__(self, typ, value, traceback):
        if typ is None and self.requests != self.expected_calls:
            raise ValueError('Reference did not execute all declared projection requests.')
        return False


class DiagnosticPatchAdapter(nn.Module):
    """Unsupported by old provenance; installed only inside an explicit bank."""
    def __init__(self, original, bank):
        super().__init__()
        self.proj = original.proj
        self.patch_size, self.tubelet_size = original.patch_size, original.tubelet_size
        self.bank = bank
        self.train(original.training)

    def forward(self, x, **kwargs):
        return self.bank(x)


@contextmanager
def diagnostic_bank(model, mode, expected_calls, device):
    if (mode not in MODES or getattr(model, '_diagnostic_execution_policy', None) != 'shared-fp64-native'
            or type(model.encoder.patch_embed) is not PrecisionPatchEmbed3D
            or any(module.training for module in model.modules())
            or any(p.requires_grad or p.grad is not None for p in model.target_encoder.parameters())):
        raise ValueError('Require the original eval-mode shared-native model and frozen target.')
    base = ExecutionGuard(model, device, target_versions=True)
    original, policy = model.encoder.patch_embed, model._diagnostic_execution_policy
    bank = (DirectProjectionBank(original, expected_calls) if mode == 'recompute' else
            DiagnosticProjectionBank(original, expected_calls=expected_calls, device=device, diagnostic_kind=KIND))
    try:
        with bank:
            model.encoder.patch_embed = DiagnosticPatchAdapter(original, bank)
            model._diagnostic_execution_policy = ACTIVE_POLICIES[mode]
            yield bank
    finally:
        model.encoder.patch_embed = original
        model._diagnostic_execution_policy = policy
        base.check()


def diagnostic_objective(model, targets, mask_bank, *, mode, receipts):
    """The unchanged native objective; all families/draws and reductions remain."""
    objective = native_objective(model, targets, mask_bank)
    expected = sum(len(families) for families in mask_bank)
    device = next(model.encoder.parameters()).device.type

    def call(pixels):
        with diagnostic_bank(model, mode, expected, device) as bank:
            result = objective(pixels)
        receipts.append({'requests': bank.requests, 'forward_computations': bank.forward_computations})
        return result
    return call
