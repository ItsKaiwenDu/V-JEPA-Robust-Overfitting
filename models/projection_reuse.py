"""CPU-only, opt-in projection-reuse proof; no model-policy or runner integration.

Reuse only FP64 forward values within one identical-input bank. Each request
still owns a distinct first-order FP32 convolution VJP and normalization graph.
This plain object cannot replace a registered model module or qualify CUDA work.
"""
from contextlib import AbstractContextManager

import torch
from torch.autograd.function import once_differentiable

from models.precision_patch import PrecisionPatchEmbed3D, _PrecisionProjection
from models.execution import precision_contract

PROTOTYPE = 'identical-bank-projection-reuse-cpu-v1'


def _bytes(tensor):
    return tensor.detach().contiguous().view(torch.uint8)


class _IndependentVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, backend, tokens):
        ctx.save_for_backward(x, weight)
        ctx.backend = backend
        ctx.bias_sizes = None if bias is None else list(bias.shape)
        # Own storage and node for every draw, including the first. No shared
        # backward or changed reduction order; preserve original token strides.
        return tokens.clone(memory_format=torch.preserve_format)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return (*_PrecisionProjection.backward(ctx, gradient), None)


class BankProjectionReuse(AbstractContextManager):
    """One CPU bank, then close permanently; rejects drift rather than reusing it.

    `expected_calls` counts projection requests, including mask families, not
    attack iterations. It is not an evaluation-strength setting. Callers must
    open a fresh bank for each objective invocation/input, scoring bank or update.
    CUDA, inference mode and autocast are deliberately unsupported by this proof.
    """
    def __init__(self, projection, *, expected_calls):
        if (type(projection) is not PrecisionPatchEmbed3D
                or type(expected_calls) is not int or expected_calls < 1
                or projection.proj.weight.device.type != 'cpu'):
            raise ValueError('Require the original CPU precision projection and a positive call count.')
        self.projection = projection
        self.expected_calls = expected_calls
        self.requests = self.forward_computations = 0
        self._active = self._entered = False
        self._input = self._tokens = None
        self._parameter_copies = None

    def _signature(self):
        p, layer = self.projection.proj, self.projection
        return (id(p), p.training, layer.training, layer.forward_id, layer.backward_backend,
                p.kernel_size, p.stride, p.padding, p.dilation, p.groups, p.padding_mode,
                layer.patch_size, layer.tubelet_size,
                id(_PrecisionProjection.forward), id(_PrecisionProjection.backward),
                id(type(layer).forward), layer.__dict__.get('forward'),
                torch.get_num_threads(), torch.is_grad_enabled(),
                tuple(sorted(precision_contract().items())),
                tuple((id(t), t._version, t.requires_grad, t.dtype, t.device,
                       tuple(t.shape), tuple(t.stride())) for t in (p.weight, p.bias) if t is not None))

    def _check(self):
        p = self.projection.proj
        if (torch.is_inference_mode_enabled() or torch.is_autocast_enabled('cpu')
                or self._signature() != self._initial):
            raise ValueError('Projection identity, geometry, precision or parameters changed.')
        values = (p.weight,) if p.bias is None else (p.weight, p.bias)
        if any(not torch.equal(_bytes(t), snapshot) for t, snapshot in zip(values, self._parameter_copies)):
            raise ValueError('Projection parameter values changed.')

    def __enter__(self):
        if self._entered:
            raise ValueError('A bank cannot be nested or reopened.')
        self._entered = True
        layer, p = self.projection, self.projection.proj
        if (layer.training or p.training or p.weight.dtype != torch.float32
                or p.bias is not None and p.bias.dtype != torch.float32):
            raise ValueError('Require eval-mode FP32 projection tensors.')
        self._initial = self._signature()
        values = (p.weight,) if p.bias is None else (p.weight, p.bias)
        self._parameter_copies = tuple(_bytes(t).clone() for t in values)
        try:
            self._check()
        except BaseException:
            self._parameter_copies = None
            raise
        self._active = True
        return self

    def __call__(self, x):
        if not self._active or self.requests >= self.expected_calls:
            raise ValueError('Bank is closed or its declared request count was exceeded.')
        self._check()
        if x.device.type != 'cpu' or x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError('Require finite CPU FP32 input.')
        signature = (tuple(x.shape), tuple(x.stride()), x.requires_grad)
        if self._input is None:
            self._input_signature = signature
            self._input = _bytes(x).clone()
            with torch.no_grad():
                # The unchanged original forward supplies its validation, FP64
                # expression, single FP32 cast and transposed token layout.
                self._tokens = self.projection(x).detach()
            self.forward_computations += 1
        elif signature != self._input_signature or not torch.equal(_bytes(x), self._input):
            raise ValueError('Input changed within the bank; use a fresh bank.')
        self.requests += 1
        p = self.projection.proj
        return _IndependentVJP.apply(x, p.weight, p.bias, self.projection.backward_backend, self._tokens)

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                self._check()
                if self.requests != self.expected_calls:
                    raise ValueError('Bank did not execute every declared projection request.')
        finally:
            self._active = False
            self._input = self._tokens = self._parameter_copies = None
        return False
