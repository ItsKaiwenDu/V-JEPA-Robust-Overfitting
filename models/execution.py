"""Opt-in diagnostic execution policies and identities; no study adoption."""

import os

import torch
from torch import nn

from models.native_patch import NativePatchEmbed3D
from models.patch_embed import LinearPatchEmbed3D
from models.precision_patch import FORWARD_ID, PrecisionPatchEmbed3D
from models.runtime import configure_cuda_reproducibility

# The temporary shared-fp64-forward-* scope keeps its separate, unsupported
# identity. These explicit integration policies do not admit research runs.
SHARED_POLICIES = ('shared-fp64-cudnn', 'shared-fp64-native')
POLICIES = ('original', 'cudnn-fp32', 'native-fp32', *SHARED_POLICIES)


def configure_execution(policy, *, device, strict_original=False):
    """Call before CUDA initialization. The default preserves existing flags."""
    if policy not in POLICIES or device not in ('cpu', 'cuda', 'mps'):
        raise ValueError('Unknown diagnostic execution policy or device.')
    if policy != 'original' and device == 'mps':
        raise ValueError('The FP32 backend diagnostic supports CPU fixtures and CUDA only.')
    if policy in SHARED_POLICIES and device == 'cuda' and torch.__version__.split('+')[0] != '2.8.0':
        raise ValueError('Shared-forward CUDA integration requires the validated PyTorch 2.8.0 boundary.')
    if policy != 'original' or strict_original:
        configure_cuda_reproducibility()
    if policy != 'original':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False


def precision_contract():
    return {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'matmul_precision': torch.get_float32_matmul_precision(),
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_enabled': torch.backends.cudnn.enabled,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def _validate_fp32():
    flags = precision_contract()
    if (not flags['deterministic_algorithms'] or flags['deterministic_warn_only']
            or flags['matmul_precision'] != 'highest' or flags['matmul_tf32']
            or not flags['cudnn_enabled'] or flags['cudnn_tf32'] or flags['cudnn_benchmark']
            or not flags['cudnn_deterministic'] or flags['cublas_workspace_config'] != ':4096:8'):
        raise ValueError('Diagnostic FP32 execution policy changed or was not configured before CUDA.')


def apply_model_policy(model, policy='original'):
    if policy not in POLICIES:
        raise ValueError('Unknown diagnostic execution policy.')
    if getattr(model, '_diagnostic_execution_policy', 'original') != 'original':
        raise ValueError('Execution policy already applied; reconstruct from original weights.')
    if policy != 'original':
        _validate_fp32()
        context, target = model.encoder.patch_embed, model.target_encoder.patch_embed
        if (type(context) is not type(target) or not isinstance(context.proj, nn.Conv3d)
                or isinstance(context, (NativePatchEmbed3D, LinearPatchEmbed3D, PrecisionPatchEmbed3D))):
            raise ValueError('Apply the diagnostic to original context and target Conv3d modules only.')
        if policy == 'native-fp32':
            model.encoder.patch_embed = NativePatchEmbed3D(context)
        elif policy in SHARED_POLICIES:
            model.encoder.patch_embed = PrecisionPatchEmbed3D(context, policy.removeprefix('shared-fp64-'))
    model._diagnostic_execution_policy = policy
    model_execution_contract(model)
    return model


def _patch_contract(encoder):
    patch = getattr(encoder, 'patch_embed', None)
    result = {'module': None if patch is None else type(patch).__module__ + '.' + type(patch).__qualname__}
    projection = getattr(patch, 'proj', None)
    if isinstance(projection, nn.Conv3d):
        result.update(projection='Conv3d', kernel=list(projection.kernel_size), stride=list(projection.stride),
                      padding=list(projection.padding), dilation=list(projection.dilation), groups=projection.groups,
                      padding_mode=projection.padding_mode, weight_shape=list(projection.weight.shape),
                      bias=projection.bias is not None)
    return result


def _shared_precision_contract(model, policy):
    """Actual execution, including transient arithmetic; FP32 storage is insufficient."""
    patch, target = model.encoder.patch_embed, model.target_encoder.patch_embed
    backend = policy.removeprefix('shared-fp64-')
    if (type(patch) is not PrecisionPatchEmbed3D or patch.backward_backend != backend
            or patch.forward_id != FORWARD_ID
            or isinstance(target, (NativePatchEmbed3D, LinearPatchEmbed3D, PrecisionPatchEmbed3D))
            or not isinstance(getattr(target, 'proj', None), nn.Conv3d)):
        raise ValueError('Actual shared-forward module/algorithm/backward differs from the declared policy.')
    p = patch.proj
    if (not isinstance(p, nn.Conv3d) or p.kernel_size != tuple(p.weight.shape[2:])
            or p.stride != p.kernel_size or p.padding != (0, 0, 0)
            or p.dilation != (1, 1, 1) or p.groups != 1 or p.padding_mode != 'zeros'
            or patch.patch_size != target.patch_size or patch.tubelet_size != target.tubelet_size
            or any(getattr(p, n) != getattr(target.proj, n) for n in
                   ('kernel_size', 'stride', 'padding', 'dilation', 'groups', 'padding_mode'))
            or p.weight.shape != target.proj.weight.shape):
        raise ValueError('Shared-forward integration requires original nonoverlapping patch geometry.')
    parameters = list(model.parameters())
    devices = {p.device.type for p in parameters}
    if (len(devices) != 1 or not devices <= {'cpu', 'cuda'}
            or any(p.dtype != torch.float32 for p in parameters)
            or any(b.is_floating_point() and b.dtype != torch.float32 for b in model.buffers())
            or any(torch.is_autocast_enabled(d) for d in ('cpu', 'cuda'))
            or any(p.requires_grad or p.grad is not None for p in model.target_encoder.parameters())
            or any(m.training for m in model.target_encoder.modules())):
        raise ValueError('Shared-forward execution requires FP32 tensors, no autocast and the frozen eval target.')
    if 'cuda' in devices and torch.__version__.split('+')[0] != '2.8.0':
        raise ValueError('Shared-forward CUDA integration requires the validated PyTorch 2.8.0 boundary.')
    for name, expected in (('pixel_mean', [.485, .456, .406]), ('pixel_std', [.229, .224, .225])):
        value = getattr(model, name)
        if not torch.equal(value, torch.tensor(expected, device=value.device, dtype=torch.float32).view(1, 3, 1, 1, 1)):
            raise ValueError('Shared-forward execution requires the original RGB normalization.')
    return {'forward': FORWARD_ID,
            'patch_accumulation': 'torch.float64', 'patch_tokens': 'torch.float32',
            'parameters': 'torch.float32', 'backward': backend,
            'backward_algorithm': 'aten.convolution_backward-first-order-v1',
            'backward_accumulation': 'torch.float32', 'higher_order': False, 'autocast': False,
            'normalization': {'algorithm': '(raw_rgb_fp32 - mean_fp32) / std_fp32',
                              'mean': model.pixel_mean.flatten().tolist(),
                              'std': model.pixel_std.flatten().tolist()},
            'target': 'original frozen FP32 Conv3d and FP32 layer_norm',
            'tail': 'original FP32 encoder/predictor and masked L1',
            'model_modules': {n: type(getattr(model, n)).__module__ + '.' + type(getattr(model, n)).__qualname__
                              for n in ('encoder', 'predictor', 'target_encoder')}}


def model_execution_contract(model):
    policy = getattr(model, '_diagnostic_execution_policy', 'original')
    if policy not in POLICIES:
        raise ValueError('Unknown model execution policy.')
    context = getattr(model.encoder, 'patch_embed', None)
    target = getattr(model.target_encoder, 'patch_embed', None)
    shared = policy in SHARED_POLICIES
    if policy != 'original':
        _validate_fp32()
        if shared:
            precision = _shared_precision_contract(model, policy)
        elif (isinstance(context, NativePatchEmbed3D) != (policy == 'native-fp32')
                or isinstance(context, LinearPatchEmbed3D)
                or isinstance(context, PrecisionPatchEmbed3D)
                or not isinstance(getattr(context, 'proj', None), nn.Conv3d)
                or isinstance(target, (NativePatchEmbed3D, LinearPatchEmbed3D, PrecisionPatchEmbed3D))
                or not isinstance(getattr(target, 'proj', None), nn.Conv3d)):
            raise ValueError('Actual patch backend differs from its declared diagnostic policy.')
    elif isinstance(context, (NativePatchEmbed3D, PrecisionPatchEmbed3D)) or isinstance(target, (NativePatchEmbed3D, PrecisionPatchEmbed3D)):
        raise ValueError('Native patch backend requires an explicit diagnostic policy.')
    result = {'schema_version': 1, 'policy': policy, 'context_patch': _patch_contract(model.encoder),
            'target_patch': _patch_contract(model.target_encoder),
            'native_backward': 'aten.convolution_backward-first-order-v1' if policy == 'native-fp32' else None,
            'activation_checkpointing': {name: getattr(getattr(model, name), 'use_activation_checkpointing', None)
                                         for name in ('encoder', 'predictor', 'target_encoder')}}
    if shared:
        # Leave all historical policy contracts unchanged. A schema-1 or
        # precision-free artifact cannot match this explicit schema-2 identity.
        result.update(schema_version=2, precision=precision, diagnostic_only=True, study_adopted=False)
    return result


class ExecutionGuard:
    """Reject runtime/backend drift and optionally in-place/replaced target tensors.

    Cache guards are process-local; targets are freshly computed, never loaded
    from an external cache. Tensor versions detect ordinary in-place updates;
    save/restore still hashes the complete fixed target state independently.
    """

    def __init__(self, model, device, *, target_versions=False):
        from models.training import runtime_contract
        self.model, self.device, self.with_target = model, device, target_versions
        self.runtime = runtime_contract(device, model=model)
        self.versions = self._versions()
        self.parameters = self._parameters()
        self.modules = self._modules()

    def _modules(self):
        if getattr(self.model, '_diagnostic_execution_policy', 'original') not in SHARED_POLICIES:
            return None
        return tuple((name, id(module)) for name, module in self.model.named_modules())

    def _parameters(self):
        return tuple((name, id(t), str(t.dtype), str(t.device), t.requires_grad)
                     for name, t in self.model.named_parameters())

    def _versions(self):
        if not self.with_target:
            return None
        target = self.model.target_encoder
        return (id(target), tuple((n, id(t), t._version, str(t.dtype), str(t.device), t.requires_grad)
                for n, t in (*target.named_parameters(), *target.named_buffers())))

    def check(self):
        from models.training import runtime_contract
        if (runtime_contract(self.device, model=self.model) != self.runtime or self._versions() != self.versions
                or self._parameters() != self.parameters or self._modules() != self.modules):
            raise ValueError('Execution runtime/backend or fixed target changed; rebuild this diagnostic/cache.')
