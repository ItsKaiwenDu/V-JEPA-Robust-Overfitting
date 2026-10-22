"""Retained, device-typed evidence for the scoped first-order CUDA backward."""

import math
from types import SimpleNamespace

import torch
from torch import nn

from models.execution import precision_contract
from models.native_patch import NativePatchEmbed3D
from models.runtime import synchronize

PROBE_SCHEMA = 2


def profile_records(profile):
    """Keep raw CPU/CUDA events, including device type and positive duration.

    Do not infer a CUDA launch from a CPU operator's name or aggregated device
    time. PyTorch 2.8's col2vol helper launches at::native::vol2im_kernel, as
    recorded in the earlier validated university convolution profiles.
    """
    rows = []
    for event in profile.events():
        device = ('cuda' if event.device_type == torch.autograd.DeviceType.CUDA
                  else 'cpu' if event.device_type == torch.autograd.DeviceType.CPU
                  else str(event.device_type))
        rows.append({'name': event.name, 'device_type': device,
                     'duration_us': float(event.time_range.end - event.time_range.start)})
    return rows


def kernel_boundary_checks(rows):
    if (not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get('name'), str)
            or not isinstance(r.get('device_type'), str)
            or not isinstance(r.get('duration_us'), (float, int))
            or not math.isfinite(r['duration_us']) or r['duration_us'] < 0 for r in rows)):
        raise ValueError('Invalid native backward profiler records.')
    kernels = {r['name'] for r in rows if r['device_type'] == 'cuda' and r['duration_us'] > 0}
    operators = {r['name'] for r in rows if r['device_type'] == 'cpu'}
    native = sorted(name for name in kernels if 'at::native::vol2im_kernel<' in name)
    cudnn = sorted(name for name in kernels if 'cudnn' in name.lower())
    cudnn_ops = sorted(name for name in operators if name.startswith('aten::cudnn_convolution'))
    return {'convolution_backward_observed': 'aten::convolution_backward' in operators,
            'cuda_events_observed': bool(kernels), 'native_reconstruction_kernels': native,
            'cudnn_kernels': cudnn, 'cudnn_convolution_operators': cudnn_ops,
            'passed': bool('aten::convolution_backward' in operators and native and not cudnn and not cudnn_ops)}


def validate_probe(evidence, *, device):
    """Recompute the gate from retained observations, including on export."""
    if evidence.get('schema_version') != PROBE_SCHEMA or evidence.get('device') != device:
        raise ValueError('Missing or incompatible native backward probe evidence.')
    checks = kernel_boundary_checks(evidence['profiler_events'])
    if checks != evidence['kernel_boundary_checks']:
        raise ValueError('Native backward gate disagrees with retained profiler records.')
    if (not evidence['backward_outside_forward_context']
            or not evidence['parameter_and_input_gradients_finite']
            or evidence['flags_before'] != evidence['flags_after_forward']
            or evidence['flags_before'] != evidence['flags_after_backward']
            or not evidence['flags_before']['cudnn_enabled']):
        raise ValueError('Native backward flags/gradient boundary failed.')
    actual = device == 'cuda' and checks['passed']
    if evidence['actual_cuda_kernel_evidence'] != actual or (device == 'cuda' and not actual):
        raise ValueError('Native backward CUDA kernel boundary was not observed; retained probe contains details.')
    if device == 'cpu' and checks['cuda_events_observed']:
        raise ValueError('CPU fixture unexpectedly reports CUDA events.')
    return True


def native_backward_probe(device, *, on_evidence=None):
    """Profile only backward; persist evidence before rejecting a failed gate."""
    before = precision_contract()
    projection = nn.Conv3d(3, 16, (2, 4, 4), stride=(2, 4, 4)).to(device)
    patch = NativePatchEmbed3D(SimpleNamespace(proj=projection, patch_size=4, tubelet_size=2, training=True))
    x = torch.ones(1, 3, 4, 16, 16, device=device, requires_grad=True)
    y = patch(x)
    after_forward = precision_contract()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profile:
        y.square().mean().backward()
        synchronize(device)
    rows = profile_records(profile)
    checks = kernel_boundary_checks(rows)
    evidence = {'schema_version': PROBE_SCHEMA, 'device': device,
                'backward_outside_forward_context': True,
                'parameter_and_input_gradients_finite': bool(x.grad is not None and projection.weight.grad is not None
                    and torch.isfinite(x.grad).all() and torch.isfinite(projection.weight.grad).all()),
                'flags_before': before, 'flags_after_forward': after_forward,
                'flags_after_backward': precision_contract(), 'profiler_events': rows,
                'kernel_boundary_checks': checks,
                'actual_cuda_kernel_evidence': device == 'cuda' and checks['passed'],
                'operator_and_kernel_names': sorted({r['name'] for r in rows})}
    if on_evidence is not None:
        on_evidence(evidence)
    validate_probe(evidence, device=device)
    return evidence
