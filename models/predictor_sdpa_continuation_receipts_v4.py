"""Cold v7 resource receipt adapter. No scientific imports or execution.

Validator bodies match frozen startup/v7 resource readers; only the transitive
startup-module import is removed. Deep numerical/source/backend gates still apply.
"""
import json,math,os,stat
from pathlib import Path
from models.predictor_sdpa_numerical_receipts_v1 import POLICY as NUMERICAL_POLICY
KIND='native-SDPA-component-startup-v2'
GPU_BYTES=7*2**30

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


def load_resource(path,*,plan_sha256,index,worker):
    """Bounded receipt readback; no CUDA, tensor or scientific work."""
    from models import predictor_sdpa_component_records_v2 as rec
    path=Path(path);rec.folder(path);file=path/'component-resource.json';s=file.lstat()
    if (file.is_symlink() or not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=os.getuid()
            or stat.S_IMODE(s.st_mode)!=0o600 or not 0<s.st_size<=16*1024):
        raise ValueError('Private bounded component resource receipt required.')
    with os.fdopen(os.open(file,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        before=os.fstat(f.fileno());data=f.read(16*1024+1);after=os.fstat(f.fileno())
    if ((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
            or (after.st_size,after.st_mtime_ns,after.st_nlink)!=(s.st_size,s.st_mtime_ns,1) or len(data)!=s.st_size):
        raise ValueError('Component resource receipt changed during read.')
    r=json.loads(data)
    if (not isinstance(r,dict) or set(r)!={'kind','numerical_receipt_policy','plan_sha256','worker_sha256','pid','index','peak_allocator_bytes',
            'startup','post_entry_wall_seconds','research_admitted','performance_qualified'}
            or r['kind']!='native-fixed-SDPA-component-resource-v7' or r['plan_sha256']!=plan_sha256
            or r['numerical_receipt_policy']!=NUMERICAL_POLICY or worker.get('numerical_receipt_policy')!=NUMERICAL_POLICY
            or type(index) is not int or index not in range(4) or type(r['index']) is not int or r['index']!=index
            or worker.get('complete') is not True or worker.get('index')!=index or worker.get('plan_sha256')!=plan_sha256
            or type(r['pid']) is not int or r['pid']<=0 or r['pid']!=worker.get('pid')
            or r['worker_sha256']!=rec.sha(path/'worker.json') or r['startup']!=worker.get('startup')
            or type(r['peak_allocator_bytes']) is not int or not 0<r['peak_allocator_bytes']<=7*2**30
            or type(r['post_entry_wall_seconds']) not in (int,float) or not math.isfinite(r['post_entry_wall_seconds'])
            or not 0<=r['post_entry_wall_seconds']<=120 or r['research_admitted'] is not False or r['performance_qualified'] is not False):
        raise ValueError('Exact completed-worker resource receipt required.')
    validate(r['startup'],pid=r['pid'],device='cuda')
    return r
