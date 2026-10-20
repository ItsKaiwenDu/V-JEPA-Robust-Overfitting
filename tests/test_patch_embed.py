import copy
from types import SimpleNamespace

import pytest
import torch

from models.patch_embed import LinearPatchEmbed3D


@pytest.mark.parametrize('shape,kernel', [((2,3,4,16,16),(2,4,4)), ((1,3,6,8,12),(3,2,3))])
def test_patch_map_preserves_outputs_input_and_parameter_gradients(shape, kernel):
    torch.manual_seed(20261018)
    native = torch.nn.Conv3d(3, 16, kernel_size=kernel, stride=kernel)
    equivalent = LinearPatchEmbed3D(SimpleNamespace(proj=copy.deepcopy(native), patch_size=kernel[1], tubelet_size=kernel[0]))
    x = torch.randn(shape, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    expected = native(x).flatten(2).transpose(1,2)
    actual = equivalent(y)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    upstream = torch.randn_like(actual)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(native.weight.grad, equivalent.proj.weight.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(native.bias.grad, equivalent.proj.bias.grad, rtol=1e-4, atol=1e-5)
    assert list(equivalent.state_dict()) == ['proj.weight', 'proj.bias']


def test_patch_map_retains_parameter_identity_and_rejects_unsupported_geometry():
    projection = torch.nn.Conv3d(3, 16, kernel_size=(2,4,4), stride=(2,4,4))
    original = SimpleNamespace(proj=projection, patch_size=4, tubelet_size=2)
    equivalent = LinearPatchEmbed3D(original)
    assert equivalent.proj.weight is projection.weight
    assert equivalent.proj.bias is projection.bias
    with pytest.raises(ValueError, match='divisible'):
        equivalent(torch.zeros(1,3,3,16,16))
    projection.stride = (1,1,1)
    with pytest.raises(ValueError, match='nonoverlapping'):
        LinearPatchEmbed3D(original)
