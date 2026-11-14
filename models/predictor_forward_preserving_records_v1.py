"""Bounded one-EOT8 records, cold receipt validation and conditional replay gates."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
from types import FunctionType
from models import predictor_eot8_record_v2 as atomic
from models import predictor_boundary_record_v1 as native_record

WORKER_KIND = 'forward-preserving-native-EOT8-worker-v1'
IDENTITY = ('kind', 'execution_host', 'local_tiny', 'mode', 'replay', 'index', 'pid', 'control_sha256')
LIMITS = {'worker.json': 256*2**10, 'native-calls.jsonl': 8*2**20,
          'VJP-calls.jsonl': 8*2**20, 'scopes.json': 64*2**10,
          'success.json': 16*2**10, 'failure.json': 16*2**10}


def identity(state):
    value = {k: state.get(k) for k in IDENTITY}
    i, tiny = value['index'], value['local_tiny']
    if (value['kind'] != WORKER_KIND or type(i) is not int or i not in range(4)
            or type(tiny) is not bool or value['replay'] != i//2
            or value['mode'] != ('original' if i % 2 == 0 else 'candidate')
            or type(value['pid']) is not int or value['pid'] <= 0
            or not isinstance(value['control_sha256'], str) or len(value['control_sha256']) != 64
            or not isinstance(value['execution_host'], str)
            or (not tiny and value['execution_host'] != 'linux10603')
            or (tiny and value['execution_host'] in ('linux10603', 'linux10609'))):
        raise ValueError('Exact worker/control/domain/replay identity required.')
    return value


_native_scope = FunctionType(native_record.verify_scope.__code__,
                            dict(native_record.verify_scope.__globals__, identity=identity))


def read(folder, name, limit):
    folder = Path(folder); path = folder/name
    if Path(name).name != name or folder.is_symlink() or path.is_symlink():
        raise ValueError('Unsafe evidence filename.')
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) not in (0o400, 0o600) or not 0 <= before.st_size <= limit):
        raise ValueError('Evidence ownership, type or byte cap changed.')
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        now = os.fstat(f.fileno()); data = f.read(limit+1); after = os.fstat(f.fileno())
    if (len(data) != before.st_size or (before.st_ino, before.st_dev, before.st_mtime_ns, before.st_size)
            != (now.st_ino, now.st_dev, after.st_mtime_ns, after.st_size)):
        raise ValueError('Evidence changed while reading.')
    return data


def sha_file(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check_scope(state, calls, scopes, refusals, vjps, borrowed):
    result = _native_scope(state, calls, scopes, refusals)
    n = 2 if state['local_tiny'] else 12
    outer = scopes[0]['graph_id']
    if state['mode'] == 'original':
        if vjps or borrowed is not None:
            raise ValueError('Original worker may not contain a custom derivative.')
        return {'native': result, 'custom_VJPs': 0, 'local_or_target': 'local' if state['local_tiny'] else 'target'}
    initial = [r for r in vjps if r.get('forward_graph_id') == -1]
    recompute = [r for r in vjps if r.get('forward_graph_id') == outer]
    shape, device = ([1, 2, 32, 32], 'cpu') if state['local_tiny'] else ([1, 12, 2048, 32], 'cuda:0')
    if (len(initial) != n*16 or len(recompute) != n*16 or len(vjps) != n*32
            or len({r.get('inner_graph_id') for r in initial}) != n*16
            or borrowed is None or borrowed.get('state') != 'complete'
            or borrowed.get('identity_source') != 'unchanged-native-root'
            or borrowed.get('graph_id') != outer or borrowed.get('new_outer_hook_added') is not False
            or borrowed.get('native_root_hook_removed') is not True):
        raise ValueError('Complete declared predictor VJPs/checkpoints/actual native root required.')
    for i, row in enumerate(vjps):
        if (row.get('index') != i or row.get('owner') not in {'predictor.'+str(i) for i in range(n)}
                or not row.get('forward_complete') or not row.get('original_forward_grad_enabled')
                or row['entry_flags'] != row['exit_flags']
                or any(t['shape'] != shape or t['device'] != device or t['dtype'] != 'torch.float32' for t in row['inputs'])):
            raise ValueError('Actual original forward/input/context evidence changed.')
    for row in initial:
        hooks = row.get('leaf_hooks', [])
        if (not row.get('backward_complete') or not row.get('hooks_removed') or row.get('outer_graph_id') != outer
                or row.get('inner_graph_id') == outer or row.get('inner_root_calls') != 1
                or sorted(h.get('index') for h in hooks) != [0, 1, 2]
                or any(h.get('graph_id') != row['inner_graph_id'] for h in hooks)
                or row.get('math_flags') != {'flash': False, 'mem_efficient': False, 'math': True, 'cudnn': False}):
            raise ValueError('Owned public math/root/leaf/context evidence incomplete.')
    if Counter(r['owner'] for r in initial) != Counter({'predictor.'+str(i): 16 for i in range(n)}):
        raise ValueError('Every predictor block and mask family must run all eight draws.')
    return {'native': result, 'custom_VJPs': len(initial), 'predictor_forward_cap': n*32,
            'custom_identity': 'predictor-original-forward-public-math-vjp-diagnostic-v1',
            'public_math_arithmetic_frozen': True, 'CUDA_kernel_or_performance_qualified': False}


def write_final(folder, state, *, calls, scopes, refusals, vjps, borrowed):
    folder = Path(folder); identity(state)
    proof = {}
    for name, rows, cap, row_cap in [('native-calls.jsonl', calls, 4096, 2048), ('VJP-calls.jsonl', vjps, 384, 8192)]:
        if len(rows) > cap:
            raise ValueError('Evidence call-row cap.')
        chunks = []
        for row in rows:
            line = atomic._json(row)
            if len(line) > row_cap:
                raise ValueError('Evidence call-row byte cap.')
            chunks.append(line)
        proof[name] = atomic._atomic(folder, name, chunks, LIMITS[name])
    payload = {'scopes': scopes, 'refusals': refusals, 'borrowed': borrowed}
    proof['scopes.json'] = atomic._atomic(folder, 'scopes.json', [atomic._json(payload)], LIMITS['scopes.json'])
    state['proof_files'] = proof
    atomic._atomic(folder, 'worker.json', [atomic._json(state)], LIMITS['worker.json'])
    marker = 'success.json' if state['complete'] and state['passed'] else 'failure.json'
    commit = {'identity': identity(state), 'worker_sha256': sha_file(folder/'worker.json'), 'passed': state['passed'],
              'proof_files': proof, 'artifacts': {k: {'file': v['file'], 'sha256': v['sha256']} for k, v in state['artifacts'].items()}}
    atomic._atomic(folder, marker, [atomic._json(commit)], LIMITS[marker])


def load(folder, *, control_sha256, index, live):
    folder = Path(folder)
    state = atomic._parse(read(folder, 'worker.json', LIMITS['worker.json']))
    key = identity(state)
    if (key['control_sha256'] != control_sha256 or key['index'] != index or key['local_tiny'] is not (not live)
            or state.get('complete') is not True or state.get('passed') is not True
            or not state.get('model_unchanged') or not state.get('adapter_restored')
            or not 0 < state.get('wall_seconds', 0) <= 480
            or (folder/'failure.json').exists()):
        raise ValueError('New complete same-domain passing worker required.')
    bounds = {'loss': {'rtol': 1e-5, 'atol': 2e-6}, 'input': {'rtol': .002, 'atol': 2e-7}}
    if state.get('numerical_bounds') != bounds or state.get('research_admitted') is not False:
        raise ValueError('Original numerical policy and diagnostic-only receipt required.')
    for field in (['anchor_checks'] if live else []) + (['original_checks'] if index % 2 else []):
        checks = state.get(field)
        if not isinstance(checks, dict) or set(checks) != set(bounds):
            raise ValueError('Complete original numerical receipt required before reuse.')
        for name, bound in bounds.items():
            check = checks[name]
            if (check.get('passed') is not True or check.get('failed_elements') != 0
                    or any(check.get(k) != v for k, v in bound.items())
                    or not isinstance(check.get('max_tolerance_ratio'), (int, float))
                    or not 0 <= check['max_tolerance_ratio'] <= 1):
                raise ValueError('Passing original numerical receipt required before reuse.')
    if index >= 2 and state.get('replay_checks') != {}:
        raise ValueError('Exact completed fresh replay receipt required before reuse.')
    commit = atomic._parse(read(folder, 'success.json', LIMITS['success.json']))
    if commit.get('identity') != key or commit.get('worker_sha256') != sha_file(folder/'worker.json') or not commit.get('passed'):
        raise ValueError('Final completion marker changed.')
    proof = {}
    for name in ('native-calls.jsonl', 'VJP-calls.jsonl', 'scopes.json'):
        data = read(folder, name, LIMITS[name]);item = state['proof_files'][name]
        if item != commit['proof_files'][name] or len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Call/scope proof changed.')
        proof[name] = data
    scope = atomic._parse(proof['scopes.json'])
    actual = check_scope(state, [atomic._parse(b) for b in proof['native-calls.jsonl'].splitlines()],
                         scope['scopes'], scope['refusals'], [atomic._parse(b) for b in proof['VJP-calls.jsonl'].splitlines()], scope['borrowed'])
    if actual != state.get('ownership'):
        raise ValueError('Recomputed graph proof differs.')
    expected = {'loss', 'input', 'targets', 'masks'} | {kind+'-'+str(i) for i in range(16) for kind in ('prediction', 'cotangent')}
    if set(state['artifacts']) != expected or set(commit['artifacts']) != expected:
        raise ValueError('Full actual EOT8 returns/predictions/cotangents required.')
    for name, item in state['artifacts'].items():
        file = item['file'];cap = 65536 if name == 'loss' else 2**20 if name == 'masks' else 12*2**20 if name == 'targets' else 16*2**20
        data = read(folder, file, cap)
        input_shape, target_shape, source_device = (([1, 3, 16, 256, 256], [1, 2048, 1024], 'cuda:0') if live
                                                    else ([1, 3, 4, 16, 16], [1, 32, 128], 'cpu'))
        shape = item.get('shape')
        expected_shape = ([] if name == 'loss' else input_shape if name == 'input' else target_shape if name == 'targets'
                          else [8, 2, 1, target_shape[1]] if name == 'masks' else None)
        if (item.get('dtype') != ('int64' if name == 'masks' else 'float32') or item.get('source_device') != source_device
                or expected_shape is not None and shape != expected_shape
                or expected_shape is None and (not isinstance(shape, list) or len(shape) != 3 or shape[0] != 1
                                                or not 1 <= shape[1] <= target_shape[1] or shape[2] != target_shape[2])):
            raise ValueError('Actual typed scientific domain/shape changed.')
        if (len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']
                or commit['artifacts'][name] != {'file': file, 'sha256': item['sha256']}
                or item.get('saved_before_gate') is not True):
            raise ValueError('Actual saved tensor changed.')
    return state
