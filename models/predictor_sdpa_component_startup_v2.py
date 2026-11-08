"""One configuration owner before CUDA initialization and all component operands."""
import json
import os
from models.execution import configure_execution, precision_contract

KIND = 'native-SDPA-component-startup-v2'
GPU_BYTES = 7 * 2**30


def initialize(device):
    import numpy
    import torch
    if device not in ('cpu', 'cuda'):
        raise ValueError('Explicit CPU fixture or authorized CUDA component required.')
    if device == 'cuda' and (torch.__version__ != '2.8.0+cu128' or torch.version.cuda != '12.8'
                              or numpy.__version__ != '2.4.6'):
        raise ValueError('Exact installed target runtime required; no fallback.')
    before = torch.cuda.is_initialized()
    # This is the only configuration call. Its original strict pre-init guard
    # remains active, including refusal of a pre-existing CUDA context.
    configure_execution('shared-fp64-native', device=device)
    total = fraction = None
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; no component work.')
        total = torch.cuda.get_device_properties(0).total_memory
        if type(total) is not int or total <= GPU_BYTES:
            raise ValueError('Insufficient GPU memory for the existing allocator cap.')
        fraction = GPU_BYTES / total
        torch.cuda.set_per_process_memory_fraction(fraction)
        torch.cuda.reset_peak_memory_stats()
    value = {'kind': KIND, 'pid': os.getpid(), 'device': device, 'policy': 'shared-fp64-native',
             'configure_calls': 1, 'cuda_initialized_before': before,
             'cuda_initialized_after': torch.cuda.is_initialized(),
             'allocator_cap_bytes': GPU_BYTES if device == 'cuda' else None,
             'device_total_bytes': total, 'allocator_fraction': fraction,
             'precision': precision_contract()}
    validate(value, pid=os.getpid(), device=device)
    return value


def validate(value, *, pid, device):
    if device not in ('cpu','cuda') or type(pid) is not int or pid<=0:
        raise ValueError('Exact component process/device required.')
    expected = {'kind': KIND, 'pid': pid, 'device': device, 'policy': 'shared-fp64-native',
                'configure_calls': 1, 'cuda_initialized_before': False,
                'cuda_initialized_after': device == 'cuda'}
    if (not isinstance(value, dict) or any(type(value.get(k)) is not type(v) or value.get(k) != v for k,v in expected.items())
            or set(value) != {*expected, 'allocator_cap_bytes', 'device_total_bytes', 'allocator_fraction', 'precision'}):
        raise ValueError('Exact component startup receipt required.')
    precision = value['precision']
    required = {'deterministic_algorithms': True, 'deterministic_warn_only': False,
                'matmul_precision': 'highest', 'matmul_tf32': False,
                'cudnn_enabled': True, 'cudnn_tf32': False, 'cudnn_benchmark': False,
                'cudnn_deterministic': True, 'cublas_workspace_config': ':4096:8'}
    if json.dumps(precision,sort_keys=True)!=json.dumps(required,sort_keys=True):
        raise ValueError('Original deterministic/precision identity changed.')
    if device == 'cuda':
        total, fraction = value['device_total_bytes'], value['allocator_fraction']
        if (type(total) is not int or total <= GPU_BYTES or type(fraction) is not float
                or fraction != GPU_BYTES / total or type(value['allocator_cap_bytes']) is not int
                or value['allocator_cap_bytes'] != GPU_BYTES):
            raise ValueError('Original component allocator cap changed.')
    elif any(value[k] is not None for k in ('allocator_cap_bytes','device_total_bytes','allocator_fraction')):
        raise ValueError('CPU fixture cannot claim GPU allocator evidence.')
