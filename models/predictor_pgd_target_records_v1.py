"""New perturbed-input records; frozen native proof code and saved-only validation."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from types import FunctionType

from models import predictor_forward_preserving_records_v1 as frozen
from models import predictor_boundary_record_v1 as native_record

KIND = 'pretrained-pgd2-r2-iterated-worker-v1'
OBJECTIVE = 'pretrained-pgd2-r2-owned-objective-v1'
ALGORITHM = 'pretrained-pgd2-r2-eot8-native-input-context-diagnostic-v1'
BOUNDS = {'loss': {'rtol': 1e-5, 'atol': 2e-6}, 'input': {'rtol': .002, 'atol': 2e-7}}
LIMITS = frozen.LIMITS


def identity(s):
    i = s.get('index')
    if (s.get('kind') != OBJECTIVE or type(i) is not int or not 0 <= i < 8
            or s.get('step') != i//2 or s.get('mode') != ('original' if i % 2 == 0 else 'candidate')
            or s.get('execution_host') != 'linux10603' or s.get('local_tiny') is not False
            or type(s.get('pid')) is not int or s['pid'] <= 0
            or not isinstance(s.get('control_sha256'), str) or len(s['control_sha256']) != 64):
        raise ValueError('Exact new perturbed objective/control identity required.')
    return {k: s[k] for k in ('kind', 'index', 'step', 'mode', 'execution_host', 'local_tiny', 'pid', 'control_sha256')}


# Keep both frozen proof functions' code objects and predicates. Their identity
# readers admit this NEW record kind; no historical state/header is synthesized.
_native_scope = FunctionType(native_record.verify_scope.__code__, dict(native_record.verify_scope.__globals__, identity=identity))
_scope = FunctionType(frozen.check_scope.__code__, dict(frozen.check_scope.__globals__, _native_scope=_native_scope))
write_final = FunctionType(frozen.write_final.__code__, dict(frozen.write_final.__globals__, identity=identity))


def check_scope(s, calls, scopes, refusals, vjps, borrowed):
    checked = _scope(s, calls, scopes, refusals, vjps, borrowed)
    owners = {r+'.'+str(i) for r, n in (('encoder', 24), ('predictor', 12)) for i in range(n)}
    for backward in (False, True):
        if Counter(r['owner'] for r in calls if r['during_backward'] is backward) != Counter({o: 16 for o in owners}):
            raise ValueError('Each original owner requires 16 primary and 16 checkpoint calls.')
    return checked


def save_json(folder, name, value, *, replace=False, cap=256*1024):
    p = Path(folder)/name
    if p.exists() and not replace:
        raise FileExistsError('Preserve prior record: '+str(p))
    return frozen.atomic._atomic(folder, name, [frozen.atomic._json(value)], cap)


def save_tensor(folder, state, name, tensor):
    import numpy as np
    a = tensor.detach().cpu().contiguous().numpy().copy()
    cap = 65536 if 'loss' in name or 'score' in name and 'mask' not in name else 2**20 if 'masks' in name else 12*2**20 if name == 'targets' else 16*2**20
    p = Path(folder)/(name+'.npy');partial = p.with_name(p.name+'.partial')
    if p.exists() or a.nbytes+4096 > cap:
        raise ValueError('Fresh bounded typed array required: '+name)
    with partial.open('xb') as f:
        import os
        os.fchmod(f.fileno(), 0o600);np.save(f, a, allow_pickle=False);f.flush();os.fsync(f.fileno())
    if partial.stat().st_size > cap:
        raise ValueError('Serialized typed file cap; preserve partial.')
    partial.rename(p)
    state['artifacts'][name] = {'file': p.name, 'bytes': p.stat().st_size, 'sha256': frozen.sha_file(p),
                               'shape': list(a.shape), 'dtype': str(a.dtype), 'source_device': str(tensor.device),
                               'saved_before_gate': True}
    if not np.isfinite(a).all():
        raise ValueError('Actual saved nonfinite return rejected.')
    return a


def arrays(folder, state):
    import io
    import numpy as np
    out = {}
    for name, d in state['artifacts'].items():
        if d.get('saved_before_gate') is not True or d.get('source_device') not in ('cuda:0', 'cpu'):
            raise ValueError('Actual pre-gate observation required.')
        cap = 16*2**20
        data = frozen.read(folder, d['file'], cap)
        if len(data) != d['bytes'] or hashlib.sha256(data).hexdigest() != d['sha256']:
            raise ValueError('Saved typed hash/extent changed.')
        stream = io.BytesIO(data);a = np.load(stream, allow_pickle=False, max_header_size=1024)
        if stream.read(1) or list(a.shape) != d['shape'] or str(a.dtype) != d['dtype'] or not np.isfinite(a).all():
            raise ValueError('Saved dtype/shape/finite/header changed.')
        out[name] = a
    return out


def compare(actual, expected):
    import numpy as np
    if set(actual) != set(expected):
        raise ValueError('Complete same-input original/candidate returns required.')
    result = {}
    for n, a in actual.items():
        e = expected[n]
        if a.shape != e.shape or a.dtype != e.dtype or not np.isfinite(a).all() or not np.isfinite(e).all():
            raise ValueError('Same full finite typed shape required.')
        if n in BOUNDS:
            error = np.abs(a.astype(np.float64)-e.astype(np.float64))
            allowed = BOUNDS[n]['atol']+BOUNDS[n]['rtol']*np.abs(e.astype(np.float64))
            check = dict(passed=bool(np.all(error <= allowed)), failed_elements=int(np.count_nonzero(error > allowed)),
                         max_absolute_error=float(error.max()), max_tolerance_ratio=float((error/allowed).max()), **BOUNDS[n])
            result[n] = check
            if not check['passed']:
                raise ValueError('Same-input numerical rejection: '+n+' '+json.dumps(check))
        elif a.tobytes() != e.tobytes():
            raise ValueError('Original forward/native-L1/input/mask value changed: '+n)
    return result


def profile_summary(profile, role):
    """Aggregate actual profiler ancestry and linked CUDA kernels; no new work."""
    groups = {}
    events = list(profile.events())
    if not 0 < len(events) <= 1_000_000:
        raise ValueError('Bounded nonempty profiler event list required.')
    for e in events:
        if str(e.device_type).split('.')[-1].lower() != 'cpu':
            continue
        chain, p = [], e
        for _ in range(128):
            if p is None:
                break
            chain.append(p.name);p = p.cpu_parent
        else:
            raise ValueError('Profiler ancestry cap exceeded.')
        native_owner = next((n.rsplit('/', 1)[-1] for n in chain if n.startswith('predictor-eot8/call/')), None)
        math = 'aten::_scaled_dot_product_attention_math' in chain
        backward = 'aten::_scaled_dot_product_efficient_attention_backward' in chain
        efficient = 'aten::_scaled_dot_product_efficient_attention' in chain
        kind = ('inner-math' if math else 'efficient-backward' if backward else 'efficient-forward' if efficient else None)
        if kind is None:
            continue
        if math and not any('_AttentionBackward' in n for n in chain):
            raise ValueError('Math SDPA outside actual candidate custom backward refused.')
        owner = native_owner or ('custom-predictor' if math else 'native-autograd')
        key = (kind, owner, e.name)
        row = groups.setdefault(key, {'kind': kind, 'owner': owner, 'operator': e.name, 'calls': 0, 'kernels': {}})
        row['calls'] += 1
        for k in e.kernels:
            duration = float(k.duration)
            if not math_is_finite(duration) or duration < 0:
                raise ValueError('Invalid linked CUDA duration.')
            r = row['kernels'].setdefault(k.name, {'calls': 0, 'duration_us': 0.})
            r['calls'] += 1;r['duration_us'] += duration
    rows = [groups[k] for k in sorted(groups)]
    result = {'kind': 'pretrained-pgd-attention-backend-aggregate-v1', 'role': role, 'event_count': len(events),
              'rows': rows, 'options': {'record_shapes': False, 'with_stack': False, 'profile_memory': False},
              'extra_objectives': 0, 'warmups': 0, 'performance_qualified': False}
    if len(frozen.atomic._json(result)) > 4*2**20:
        raise ValueError('Bounded backend aggregate exceeded.')
    return result


def math_is_finite(value):
    return math.isfinite(value)


def validate_backend(value, role):
    if (role not in ('original', 'candidate') or value.get('role') != role
            or value.get('kind') != 'pretrained-pgd-attention-backend-aggregate-v1'
            or value.get('options') != {'record_shapes': False, 'with_stack': False, 'profile_memory': False}
            or value.get('extra_objectives') != 0 or value.get('warmups') != 0
            or value.get('performance_qualified') is not False):
        raise ValueError('Exact diagnostic-only backend observation required.')
    rows = value['rows']
    op = Counter()
    names = {'efficient-forward': set(), 'efficient-backward': set(), 'inner-math': set()}
    forward_owners = Counter()
    for row in rows:
        if row['kind'] not in names or type(row['calls']) is not int or row['calls'] < 1:
            raise ValueError('Typed aggregate row required.')
        op[(row['kind'], row['operator'])] += row['calls']
        if row['operator'] == 'aten::_scaled_dot_product_efficient_attention':
            forward_owners[row['owner']] += row['calls']
        for name, k in row['kernels'].items():
            if not isinstance(name, str) or not name or type(k['calls']) is not int or k['calls'] < 1 or not math.isfinite(k['duration_us']) or k['duration_us'] < 0:
                raise ValueError('Actual finite typed linked CUDA kernel rows required.')
            if k['duration_us'] > 0:
                names[row['kind']].add(name)
    owners = {r+'.'+str(i) for r, n in (('encoder', 24), ('predictor', 12)) for i in range(n)}
    if (forward_owners != Counter({o: 32 for o in owners})
            or op[('efficient-forward', 'aten::_scaled_dot_product_efficient_attention')] != 1152
            or op[('efficient-backward', 'aten::_scaled_dot_product_efficient_attention_backward')] != (576 if role == 'original' else 384)
            or op[('inner-math', 'aten::_scaled_dot_product_attention_math')] != (0 if role == 'original' else 192)):
        raise ValueError('Actual native-owner efficient and predictor-math operator counts required.')
    if (not any('fmha_cutlassF' in n or 'AttentionKernel<' in n for n in names['efficient-forward'])
            or not any('fmha_cutlassB' in n or 'AttentionBackwardKernel<' in n for n in names['efficient-backward'])):
        raise ValueError('Actual efficient forward/backward CUDA kernels required.')
    if role == 'candidate' and (not any('gemm' in n.lower() for n in names['inner-math'])
                               or not any('softmax' in n.lower() for n in names['inner-math'])):
        raise ValueError('Actual math GEMM and softmax CUDA kernels required.')
    return {'passed': True, 'role': role, 'representative_perturbed_objective_only': True, 'performance_qualified': False}


def load_objective(folder, sha, index):
    s = json.loads(frozen.read(folder, 'worker.json', LIMITS['worker.json']))
    identity(s)
    if (s['index'] != index or s['control_sha256'] != sha or not s.get('complete') or not s.get('passed')
            or not s.get('model_unchanged') or not s.get('adapter_restored')
            or s.get('parameter_gradients_absent') is not True or s.get('input_dot_grad_absent') is not True
            or s.get('numerical_bounds') != BOUNDS or s.get('research_admitted') is not False
            or s.get('events') != (['forward', 'consider', 'backward', 'exit'] if index % 2 else ['forward', 'backward', 'exit'])):
        raise ValueError('New complete same-input objective with restoration required.')
    marker = json.loads(frozen.read(folder, 'success.json', LIMITS['success.json']))
    if marker['identity'] != identity(s) or marker['worker_sha256'] != frozen.sha_file(Path(folder)/'worker.json') or marker['passed'] is not True:
        raise ValueError('Objective completion changed.')
    proof = {}
    for name in ('native-calls.jsonl', 'VJP-calls.jsonl', 'scopes.json'):
        data = frozen.read(folder, name, LIMITS[name]);d = s['proof_files'][name]
        if d != marker['proof_files'][name] or len(data) != d['bytes'] or hashlib.sha256(data).hexdigest() != d['sha256']:
            raise ValueError('Actual call proof changed.')
        proof[name] = data
    scoped = json.loads(proof['scopes.json'])
    checked = check_scope(s, [json.loads(b) for b in proof['native-calls.jsonl'].splitlines()], scoped['scopes'], scoped['refusals'],
                          [json.loads(b) for b in proof['VJP-calls.jsonl'].splitlines()], scoped['borrowed'])
    if checked != s['ownership']:
        raise ValueError('Actual per-objective proof differs.')
    a = arrays(folder, s)
    names = {'pixels', 'loss', 'input', 'targets', 'masks'} | {k+'-'+str(i) for i in range(16) for k in ('prediction', 'cotangent')}
    if set(a) != names or set(marker['artifacts']) != names:
        raise ValueError('37 actual original/candidate returns required.')
    import numpy as np
    for name, value in a.items():
        if value.dtype != (np.int64 if name == 'masks' else np.float32) or s['artifacts'][name]['source_device'] != 'cuda:0':
            raise ValueError('Original FP32 CUDA return types required.')
        if marker['artifacts'][name] != {'file': s['artifacts'][name]['file'], 'sha256': s['artifacts'][name]['sha256']}:
            raise ValueError('Committed array changed.')
    if a['pixels'].shape != (1,3,16,256,256) or a['input'].shape != a['pixels'].shape or a['targets'].shape != (1,2048,1024) or a['masks'].shape != (8,2,1,2048) or a['loss'].shape != () or not a['input'].any():
        raise ValueError('Full original typed dimensions/nonzero input derivative required.')
    lengths = s['mask_context_lengths']
    losses = []
    for i in range(16):
        nc = lengths[i//2][i % 2];packed = a['masks'][i//2,i % 2,0]
        if not np.array_equal(np.sort(packed), np.arange(2048)):
            raise ValueError('Original complementary disjoint masks required.')
        selected = a['targets'][:,packed[nc:],:]
        pred,cot = a['prediction-'+str(i)],a['cotangent-'+str(i)]
        if pred.shape != selected.shape or cot.shape != pred.shape:
            raise ValueError('Actual predictor/L1 geometry changed.')
        want = np.sign(pred-selected).astype(np.float32)/np.float32(8)/np.float32(2)/np.float32(pred.size)
        if cot.tobytes() != want.tobytes():
            raise ValueError('Native mean-L1 incoming cotangent changed.')
        losses.append(np.abs(pred-selected).mean(dtype=np.float32))
    value = np.asarray(losses,dtype=np.float32).reshape(8,2).mean(axis=1,dtype=np.float32).mean(dtype=np.float32)
    # CPU reduction order may differ from CUDA; use the unchanged original loss bound.
    if abs(float(a['loss'])-float(value)) > BOUNDS['loss']['atol']+BOUNDS['loss']['rtol']*abs(float(value)):
        raise ValueError('Native L1 saved loss disagrees.')
    return s,a


def worker_header(s, control_sha256):
    if (s.get('kind') != KIND or s.get('algorithm') != ALGORITHM or s.get('control_sha256') != control_sha256
            or s.get('execution_host') != 'linux10603' or s.get('complete') is not True or s.get('passed') is not True
            or s.get('research_admitted') is not False or s.get('performance_qualified') is not False
            or s.get('gradient_objectives_started') != 8 or s.get('candidate_points') != 4
            or not 0 < s.get('wall_seconds',0) <= 1920
            or any(s.get(k) is not True for k in ('model_unchanged','global_rng_unchanged','all_methods_hooks_flags_restored'))
            or not 0 < s.get('peak_allocator_bytes',0) <= 7*2**30):
        raise ValueError('Complete new bounded iterated GPU worker required.')
    if hashlib.sha256(frozen.atomic._json(s['baseline'])).hexdigest() != s['baseline_sha256']:
        raise ValueError('Actual baseline hash changed.')
    return True


def verify_worker(folder, *, control_sha256):
    folder = Path(folder);s = json.loads(frozen.read(folder,'worker.json',256*1024))
    worker_header(s, control_sha256)
    import numpy as np
    root = arrays(folder,s);points=[];checks=[]
    for step in range(4):
        original,o = load_objective(folder/('objective-'+str(step*2)),control_sha256,step*2)
        candidate,c = load_objective(folder/('objective-'+str(step*2+1)),control_sha256,step*2+1)
        if original['pid'] != s['pid'] or candidate['pid'] != s['pid'] or original['baseline_sha256'] != s['baseline_sha256'] or candidate['baseline_sha256'] != s['baseline_sha256']:
            raise ValueError('Same actual process/model/target/bank baseline required.')
        check = compare(c,o)
        if check != candidate['comparison']:
            raise ValueError('Same-input numerical receipt changed.')
        checks.append(check);points.append(c)
        if step == 0:
            for role, objective in (('original', original),('candidate',candidate)):
                path=folder/('objective-'+str(0 if role=='original' else 1))/'backend.json'
                profile=json.loads(frozen.read(path.parent,path.name,4*2**20))
                if frozen.sha_file(path) != objective['backend_sha256']:
                    raise ValueError('Backend aggregate hash changed.')
                validate_backend(profile,role)
    names={'clean','adversarial','attack-loss','clean-attack-loss','clean-score','adversarial-score','restart-noise-0','restart-noise-1','scoring-masks'}
    if set(root) != names or any(v.dtype != (np.int64 if 'masks' in n else np.float32) for n,v in root.items()):
        raise ValueError('Complete typed rollout/scoring/noise returns required.')
    if root['clean'].shape != (1,3,16,256,256) or root['adversarial'].shape != root['clean'].shape or root['scoring-masks'].shape != (4,2,1,2048):
        raise ValueError('Full clean/adversary/independent scoring mask geometry required.')
    for d in range(4):
        for family in range(2):
            packed=root['scoring-masks'][d,family,0];nc=s['score_mask_context_lengths'][d][family]
            if not np.array_equal(np.sort(packed),np.arange(2048)):
                raise ValueError('Independent scoring mask complement changed.')
            for i,a in enumerate((packed[:nc].reshape(1,-1),packed[nc:].reshape(1,-1))):
                actual={'shape':list(a.shape),'dtype':'torch.int64','sha256':hashlib.sha256(a.tobytes()).hexdigest(),'abs_max':float(np.abs(a).max())}
                if actual!=s['score_mask_signatures'][d][family][i]:
                    raise ValueError('Independent scoring tensor identity changed.')
    logs=s['no_grad_calls']
    if len(logs)!=6 or [r['draws'] for r in logs]!=[8,4,8,8,8,4] or [r['label'] for r in logs]!=['attack','score-clean','attack','attack','attack','score-adversarial'] or any(r['custom_route_active'] or r['grad_enabled'] for r in logs):
        raise ValueError('Original no-grad scoring/selection counts changed.')
    center=root['clean'];eps=np.float32(4/255);alpha=np.float32(1/255)
    lower=np.maximum(center-eps,np.float32(0));upper=np.minimum(center+eps,np.float32(1))
    clamp=lambda v:np.minimum(np.maximum(v,lower),upper)
    def pixhash(v):return hashlib.sha256(v.tobytes()).hexdigest()
    best=center.copy();loss=np.asarray(logs[2]['loss'],dtype=np.float32);selected=[]
    def consider(p,v,label):
        nonlocal best,loss
        v=np.asarray(v,dtype=np.float32).reshape(1);improved=bool(v[0]>loss[0])
        if improved:best,loss=p.copy(),v.copy()
        selected.append({'point':label,'loss':float(v[0]),'selected':improved})
    for restart in range(2):
        noise=root['restart-noise-'+str(restart)]
        if noise.shape!=center.shape or noise.min()<0 or noise.max()>=1:raise ValueError('Fixed raw uniform restart draws required.')
        current=clamp(center+(np.float32(2)*noise-np.float32(1))*eps)
        for step in range(2):
            i=2*restart+step;p=points[i]
            if p['pixels'].tobytes()!=current.tobytes() or p['targets'].tobytes()!=points[0]['targets'].tobytes() or p['masks'].tobytes()!=points[0]['masks'].tobytes():
                raise ValueError('Iterate/start/target/bank changed.')
            consider(current,p['loss'],'iterate-'+str(i));current=clamp(current+alpha*np.sign(p['input']))
        if pixhash(current)!=logs[3+restart]['pixels']:raise ValueError('Final iterate scoring input changed.')
        consider(current,logs[3+restart]['loss'],'final-'+str(restart))
    if root['adversarial'].tobytes()!=best.tobytes() or root['attack-loss'].tobytes()!=loss.tobytes():raise ValueError('All-candidate strict maximum selection changed.')
    for i,name in ((0,'clean-attack-loss'),(1,'clean-score'),(5,'adversarial-score')):
        if root[name].tobytes()!=np.asarray(logs[i]['loss'],dtype=np.float32).tobytes():raise ValueError('Scoring output changed.')
    if any(logs[i]['pixels']!=pixhash(center) for i in (0,1,2)) or logs[5]['pixels']!=pixhash(best) or logs[0]['loss']!=logs[2]['loss']:
        raise ValueError('Clean/selected scoring boundaries changed.')
    return {'worker_sha256':frozen.sha_file(folder/'worker.json'),'pid':s['pid'],'gradient_objectives':8,
            'same_input_checks':checks,'selection':selected,'numerical_and_backend_scoped':True,
            'fresh_rollout_replay_qualified':False,'performance_qualified':False,'research_admitted':False}
