"""Separate first-order diagnostic: FP64 patch accumulation, FP32 tokens/VJPs."""

from contextlib import contextmanager

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from models.execution import ExecutionGuard
from models.training import runtime_contract

BACKWARDS = ('cudnn', 'native')
FORWARD_ID = 'nonoverlapping-unfold-fp64-mm-separate-bias-cast-fp32-transposed-v1'


def backward_context(backend):
    if backend not in BACKWARDS:
        raise ValueError('Unknown precision-control backward.')
    return torch.backends.cudnn.flags(enabled=backend == 'cudnn', benchmark=False,
                                     deterministic=True, allow_tf32=False)


class _PrecisionProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, backend):
        ctx.save_for_backward(x, weight)
        ctx.backend = backend
        ctx.bias_sizes = None if bias is None else list(bias.shape)
        batch, channels, frames, height, width = x.shape
        outputs, _, kt, kh, kw = weight.shape
        nt, nh, nw = frames // kt, height // kh, width // kw
        patches = x.unfold(2, kt, kt).unfold(3, kh, kh).unfold(4, kw, kw)
        matrix = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, channels * kt * kh * kw).double()
        # Both arms execute this identical FP64 expression. No bias fusion or
        # backend-dependent forward dispatch. FP64 is transient, parameters stay FP32.
        tokens = matrix @ weight.double().flatten(1).T
        if bias is not None:
            tokens = tokens + bias.double()
        tokens = tokens.reshape(batch, nt * nh * nw, outputs).to(x.dtype)
        # Match the original Conv3d flatten/transpose token layout.
        return tokens.transpose(1, 2).contiguous().transpose(1, 2)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, weight = ctx.saved_tensors
        kt, kh, kw = weight.shape[2:]
        batch, _, frames, height, width = x.shape
        g = gradient.transpose(1, 2).reshape(batch, weight.shape[0], frames // kt, height // kh, width // kw)
        with backward_context(ctx.backend):
            grads = torch.ops.aten.convolution_backward(
                g, x, weight, ctx.bias_sizes, (kt, kh, kw), (0, 0, 0), (1, 1, 1),
                False, (0, 0, 0), 1, tuple(ctx.needs_input_grad[:3]))
        return (*grads, None)


class PrecisionPatchEmbed3D(nn.Module):
    """Reuse original tensors/state keys; no target or other layer precision change.

    First-order FP32 convolution VJPs approximate the real affine derivative;
    their rounding is explicitly compared with independent CPU FP64 derivatives.
    This is not the derivative of discrete rounding and supports no double backward.
    """
    def __init__(self, original, backward):
        super().__init__()
        p = original.proj
        if (backward not in BACKWARDS or not isinstance(p, nn.Conv3d)
                or p.stride != p.kernel_size or p.padding != (0, 0, 0)
                or p.dilation != (1, 1, 1) or p.groups != 1 or p.padding_mode != 'zeros'):
            raise ValueError('Precision control requires the original nonoverlapping Conv3d geometry.')
        self.proj, self.backward_backend = p, backward
        self.patch_size, self.tubelet_size = original.patch_size, original.tubelet_size
        self.forward_id = FORWARD_ID
        self.train(original.training)

    def forward(self, x, **kwargs):
        p = self.proj
        if (x.device.type not in ('cpu', 'cuda') or x.dtype != torch.float32
                or p.weight.dtype != torch.float32 or (p.bias is not None and p.bias.dtype != torch.float32)
                or x.ndim != 5 or x.shape[1] != p.in_channels
                or any(n % k for n, k in zip(x.shape[2:], p.kernel_size))
                or torch.is_autocast_enabled(x.device.type) or self.forward_id != FORWARD_ID):
            raise ValueError('Precision control needs FP32 CPU/CUDA operands, divisible geometry and no autocast.')
        return _PrecisionProjection.apply(x, p.weight, p.bias, self.backward_backend)


def precision_identity(model, device):
    patch, target = model.encoder.patch_embed, model.target_encoder.patch_embed
    if (not isinstance(patch, PrecisionPatchEmbed3D) or isinstance(target, PrecisionPatchEmbed3D)
            or patch.backward_backend not in BACKWARDS or patch.forward_id != FORWARD_ID
            or model._diagnostic_execution_policy != 'shared-fp64-forward-' + patch.backward_backend
            or any(p.dtype != torch.float32 for p in model.parameters())
            or any(p.requires_grad for p in model.target_encoder.parameters())
            or any(m.training for m in model.modules()) or torch.is_autocast_enabled(device)):
        raise ValueError('Actual precision-control execution drifted.')
    return {'schema_version': 1, 'diagnostic_only': True, 'study_adopted': False,
            'forward': FORWARD_ID, 'backward': patch.backward_backend,
            'backward_accumulation': 'FP32 aten.convolution_backward-first-order-v1',
            'parameters_and_tokens': 'torch.float32', 'transient_patch_accumulation': 'torch.float64',
            'normalization_and_tail': 'original FP32', 'target': 'original frozen FP32 Conv3d',
            'context_module': type(patch).__module__ + '.' + type(patch).__qualname__,
            'target_module': type(target).__module__ + '.' + type(target).__qualname__,
            'kernel': list(patch.proj.kernel_size), 'stride': list(patch.proj.stride),
            'padding': list(patch.proj.padding), 'weight_shape': list(patch.proj.weight.shape),
            'activation_checkpointing': {n: getattr(getattr(model, n), 'use_activation_checkpointing', None)
                                         for n in ('encoder', 'predictor', 'target_encoder')},
            'runtime': runtime_contract(device)}


class PrecisionGuard:
    def __init__(self, model, device):
        self.model, self.device = model, device
        self.identity = precision_identity(model, device)
        self.patch = model.encoder.patch_embed
        self.tensors = self._tensors()

    def _tensors(self):
        return tuple((n, id(v), v._version, v.requires_grad, str(v.dtype), str(v.device))
                     for n, v in (*self.model.named_parameters(), *self.model.named_buffers()))

    def check(self):
        if (precision_identity(self.model, self.device) != self.identity
                or self.model.encoder.patch_embed is not self.patch or self._tensors() != self.tensors):
            raise ValueError('Precision runtime, module, parameter or target identity changed.')


@contextmanager
def precision_scope(model, backend, device):
    """Separate diagnostic identity; legacy caches/training reject while active."""
    base = ExecutionGuard(model, device, target_versions=True)
    original, policy = model.encoder.patch_embed, model._diagnostic_execution_policy
    if policy != 'cudnn-fp32' or type(original) is not type(model.target_encoder.patch_embed):
        raise ValueError('Start from the explicit original cuDNN FP32 context/target pair.')
    wrapper = PrecisionPatchEmbed3D(original, backend)
    try:
        model.encoder.patch_embed = wrapper
        # Existing execution/cache/training APIs fail closed on this unknown
        # policy. Only this separate CLI/identity supports it; no study integration.
        model._diagnostic_execution_policy = 'shared-fp64-forward-' + backend
        guard = PrecisionGuard(model, device)
        yield guard
        guard.check()
    finally:
        model.encoder.patch_embed = original
        model._diagnostic_execution_policy = policy
        base.check()
