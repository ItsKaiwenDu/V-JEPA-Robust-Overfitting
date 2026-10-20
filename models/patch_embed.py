"""Optional linear implementation of the same nonoverlapping 3-D patch map."""

from torch import nn
from torch.nn import functional as F


class LinearPatchEmbed3D(nn.Module):
    """Reuse the original Conv3d parameters and state keys, with patchwise GEMM.

    For kernel=stride, no padding/dilation/groups, each patch is independent.
    Flattening each [C, kT, kH, kW] patch and applying the reshaped convolution
    weight is the identical affine map, up to floating-point reduction order.
    This is an explicit performance option, not a new learned projection.
    """

    def __init__(self, original):
        super().__init__()
        projection = original.proj
        if not isinstance(projection, nn.Conv3d):
            raise ValueError('The original patch projection must be Conv3d.')
        if (projection.stride != projection.kernel_size or projection.padding != (0, 0, 0)
                or projection.dilation != (1, 1, 1) or projection.groups != 1
                or projection.padding_mode != 'zeros'):
            raise ValueError('Linear patch mapping requires nonoverlapping, unpadded, undilated patches.')
        self.proj = projection
        self.patch_size = original.patch_size
        self.tubelet_size = original.tubelet_size

    def forward(self, x, **kwargs):
        batch, channels, time, height, width = x.shape
        kt, kh, kw = self.proj.kernel_size
        if time % kt or height % kh or width % kw:
            raise ValueError('Clip dimensions must be divisible by the patch dimensions.')
        patches = x.reshape(batch, channels, time//kt, kt, height//kh, kh, width//kw, kw)
        patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(batch, -1, channels*kt*kh*kw)
        return F.linear(patches, self.proj.weight.reshape(self.proj.out_channels, -1), self.proj.bias)
