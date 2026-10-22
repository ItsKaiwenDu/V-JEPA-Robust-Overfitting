"""Explicit native Conv3d execution for first-order diagnostic gradients."""

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


def native_context():
    return torch.backends.cudnn.flags(enabled=False, benchmark=False,
                                     deterministic=True, allow_tf32=False)


class _NativeConv3d(torch.autograd.Function):
    """Delegate both directions to PyTorch convolution kernels, without cuDNN.

    In PyTorch 2.8 convolution backward queries the current global flags again;
    a forward-only context does not select its backend. This first-order-only
    delegator scopes both calls. PGD is detached before the outer update.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups):
        ctx.save_for_backward(x, weight)
        ctx.geometry = stride, padding, dilation, groups
        ctx.bias_sizes = None if bias is None else list(bias.shape)
        with native_context():
            return F.conv3d(x, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, weight = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.geometry
        mask = tuple(ctx.needs_input_grad[:3])
        with native_context():
            grads = torch.ops.aten.convolution_backward(
                gradient, x, weight, ctx.bias_sizes, stride, padding, dilation,
                False, (0, 0, 0), groups, mask)
        return (*grads, None, None, None, None)


class NativePatchEmbed3D(nn.Module):
    """Keep the original projection object, learned tensors and state keys.

    Target encoding remains on the original module. This wrapper changes only
    context patch execution and supports first-order CPU/CUDA FP32 diagnostics
    (CPU FP64 is allowed for numerical derivative tests).
    """

    def __init__(self, original):
        super().__init__()
        projection = original.proj
        if not isinstance(projection, nn.Conv3d):
            raise ValueError('Native patch projection requires the original Conv3d.')
        if (projection.stride != projection.kernel_size or projection.padding != (0, 0, 0)
                or projection.dilation != (1, 1, 1) or projection.groups != 1
                or projection.padding_mode != 'zeros'):
            raise ValueError('Native patch execution requires nonoverlapping original patch geometry.')
        self.proj = projection
        self.patch_size = original.patch_size
        self.tubelet_size = original.tubelet_size
        self.train(original.training)

    def forward(self, x, **kwargs):
        if (x.device.type not in ('cpu', 'cuda')
                or x.dtype not in ((torch.float32, torch.float64) if x.device.type == 'cpu' else (torch.float32,))
                or torch.is_autocast_enabled(x.device.type)):
            raise ValueError('Native patch diagnostics require CPU/CUDA without autocast; CUDA uses FP32.')
        y = _NativeConv3d.apply(x, self.proj.weight, self.proj.bias, self.proj.stride,
                              self.proj.padding, self.proj.dilation, self.proj.groups)
        return y.flatten(2).transpose(1, 2)
