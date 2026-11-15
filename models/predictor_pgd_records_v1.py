"""Saved tiny PGD evidence reader: NumPy and standard library only; no model work."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

ALGORITHM = 'tiny-cpu-pgd2-r2-eot8-gradient-context-v1'
INPUT_BOUND = {'rtol': .002, 'atol': 2e-7}
LOSS_BOUND = {'rtol': 1e-5, 'atol': 2e-6}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check_proof(proof, candidate, pid):
    """Validate actual tiny root/call/inner-task observations, never construct them."""
    scopes, rows = proof['native_scopes'], proof['native_calls']
    if proof['native_refusals'] or len(scopes) != 1 or len(rows) != 128:
        raise ValueError('One complete 128-call native EOT8 scope required.')
    s = scopes[0]
    if (s.get('phase') != 'EOT8-before' or s.get('index') != 0 or s.get('state') != 'complete'
            or s.get('hook_removed') is not True or s.get('pid') != pid
            or type(s.get('graph_id')) is not int or s['graph_id'] < 0
            or any(type(s.get(k)) is not int or s[k] <= 0 for k in
                   ('creator_thread', 'runner_thread', 'root_hook_thread', 'loss_id', 'input_id'))
            or not s.get('call_threads') or len(set(s['call_threads'])) != len(s['call_threads'])
            or s['root_hook_thread'] not in s['call_threads']):
        raise ValueError('Actual native root/hook/thread completion required.')
    owners = {'encoder.0', 'encoder.1', 'predictor.0', 'predictor.1'}
    counts = [Counter(), Counter()]
    for i, r in enumerate(rows):
        if (r.get('index') != i or r.get('owner') not in owners
                or r['owner'] != str(r.get('role'))+'.'+str(r.get('block'))
                or r.get('phase') != 'EOT8-before' or r.get('pid') != pid
                or r.get('creator_thread') != s['creator_thread']
                or r.get('entry_flags') != r.get('exit_flags')
                or r.get('effective_flags') != r.get('entry_flags')
                or type(r.get('during_backward')) is not bool):
            raise ValueError('Original native ownership, flags and restoration required.')
        back = r['during_backward']
        if back:
            if (r.get('graph_id') != s['graph_id'] or r.get('backward_scope') != 0
                    or r.get('thread') not in s['call_threads']
                    or not (r.get('outcome') == 'returned' or r.get('outcome') == 'interrupted'
                            and r.get('error_type') == '_StopRecomputationError')):
                raise ValueError('Actual owned checkpoint recomputation required.')
        elif (r.get('graph_id') != -1 or r.get('backward_scope') is not None
              or r.get('thread') != s['creator_thread'] or r.get('outcome') != 'returned'):
            raise ValueError('Native forward ownership changed.')
        counts[int(back)][r['owner']] += 1
    if counts != [Counter({o: 16 for o in owners})]*2:
        raise ValueError('All four owners require 16 primary and 16 checkpoint calls.')
    if set(s['call_threads']) != {r['thread'] for r in rows if r['during_backward']}:
        raise ValueError('Native checkpoint-thread observations changed.')
    if candidate:
        calls, borrowed = proof['prototype_calls'], proof['borrowed_scope']
        initial = [c for c in calls if c.get('forward_graph_id') == -1]
        back = [c for c in calls if c.get('forward_graph_id') == s['graph_id']]
        if (len(calls) != 64 or len(initial) != 32 or len(back) != 32
                or len({c.get('inner_graph_id') for c in initial}) != 32
                or borrowed.get('state') != 'complete' or borrowed.get('graph_id') != s['graph_id']
                or borrowed.get('new_outer_hook_added') is not False
                or borrowed.get('identity_source') != 'unchanged-native-root'
                or borrowed.get('native_root_hook_removed') is not True
                or borrowed.get('forward_cap') != 64 or borrowed.get('expected_initial_VJPs') != 32):
            raise ValueError('Bounded borrowed native root and 32 distinct inner tasks required.')
        for i, c in enumerate(calls):
            if (c.get('index') != i or c.get('forward_complete') is not True
                    or c.get('exit_flags') != c.get('entry_flags')
                    or c.get('original_forward_grad_enabled') is not True):
                raise ValueError('Original grad-enabled forward values/flags required.')
        for c in initial:
            if (c.get('backward_complete') is not True or c.get('hooks_removed') is not True
                    or c.get('outer_graph_id') != s['graph_id'] or c.get('inner_root_calls') != 1
                    or type(c.get('inner_graph_id')) is not int or c['inner_graph_id'] < 0
                    or c['inner_graph_id'] == s['graph_id']
                    or c.get('math_flags') != {'flash': False, 'mem_efficient': False, 'math': True, 'cudnn': False}
                    or sorted(h['index'] for h in c.get('leaf_hooks', [])) != [0, 1, 2]
                    or any(h['graph_id'] != c['inner_graph_id'] for h in c['leaf_hooks'])):
                raise ValueError('Actual first-order math root and three leaf hooks required.')
    return {'native_calls': 128, 'primary_calls': 64, 'checkpoint_calls': 64,
            'owners': 4, 'inner_tasks': 32 if candidate else 0, 'passed': True}



def agreement(actual, expected, bound):
    a, e = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != e.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(e).all():
        raise ValueError('Same finite full shapes required.')
    error = np.abs(a-e)
    allowed = bound['atol']+bound['rtol']*np.abs(e)
    return dict(passed=bool(np.all(error <= allowed)), failed_elements=int(np.count_nonzero(error > allowed)),
                elements=a.size, max_absolute_error=float(error.max()), max_tolerance_ratio=float((error/allowed).max()),
                **bound)


def load_arrays(folder, receipt):
    result = {}
    for key, d in receipt['files'].items():
        p = folder/d['file']
        if (p.parent != folder or p.is_symlink() or p.stat().st_nlink != 1
                or p.stat().st_mode & 0o777 != 0o600 or p.stat().st_size != d['bytes']
                or digest(p) != d['sha256']):
            raise ValueError('Saved array identity/hash/extent/private file changed.')
        with p.open('rb') as f:
            a = np.load(f, allow_pickle=False, max_header_size=1024)
            if f.read(1):
                raise ValueError('Unexpected trailing array bytes.')
        if list(a.shape) != d['shape'] or str(a.dtype) != d['dtype'] or not np.isfinite(a).all():
            raise ValueError('Array shape/dtype/finite contract changed.')
        result[key] = a
    return result


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def read_json(path):
    p = Path(path)
    if p.is_symlink() or p.stat().st_nlink != 1 or p.stat().st_mode & 0o777 != 0o600:
        raise ValueError('Private regular single-link JSON required.')
    return json.loads(p.read_bytes())


def review(folder, paired=None):
    """Read actual saved iterates and reconstruct ascent/projection/all-candidate selection."""
    folder = Path(folder)
    receipt = read_json(folder/'receipt.json')
    mode = receipt.get('mode')
    if (not receipt.get('complete') or receipt.get('kind') != 'tiny-pgd-context-rollout-v1'
            or receipt.get('algorithm') != ALGORITHM or receipt.get('CPU_only') is not True
            or receipt.get('CUDA_qualified') is not False or receipt.get('training_or_research_qualified') is not False
            or mode not in ('paired', 'replay') or receipt.get('candidate_objectives') != 4
            or receipt.get('original_objectives') != (4 if mode == 'paired' else 0)):
        raise ValueError('Exact complete local-only bounded rollout required.')
    arrays = load_arrays(folder, receipt)
    required = {'clean', 'adversarial', 'attack-loss', 'clean-attack-loss', 'clean-score', 'adversarial-score', 'restart-noise'}
    if set(arrays) != required or arrays['clean'].shape != (1, 3, 4, 16, 16):
        raise ValueError('Exact seven tiny rollout returns required.')
    starts = sorted(folder.glob('started-*.json'))
    expected = [('original', i//2) if i % 2 == 0 else ('candidate', i//2) for i in range(8)] if mode == 'paired' else [('candidate', i) for i in range(4)]
    if len(starts) != len(expected):
        raise ValueError('Exact consumed objective ledger count required.')
    for i, (p, (role, step)) in enumerate(zip(starts, expected)):
        if read_json(p) != {'index': i, 'pid': receipt['pid'], 'role': role, 'step': step}:
            raise ValueError('Consumed objective ordering changed.')
    points, checks, proofs = [], [], []
    for i in range(4):
        per_role = {}
        for role in (('original', 'candidate') if mode == 'paired' else ('candidate',)):
            d = folder/f'{role}-{i}'
            r = read_json(d/'receipt.json')
            if (r.get('kind') != 'tiny-pgd-owned-objective-v1' or r.get('algorithm') != ALGORITHM
                    or not r.get('complete') or r.get('candidate') != (role == 'candidate') or r.get('step') != i
                    or r.get('pid') != receipt['pid'] or r.get('model_state_sha256') != receipt['model_state_sha256']
                    or any(r.get(k) is not True for k in ('restoration_passed', 'parameter_gradients_absent', 'input_gradient_absent'))
                    or r.get('events') != (['forward', 'consider', 'backward', 'exit'] if role == 'candidate' else ['forward', 'backward', 'exit'])):
                raise ValueError('Native objective identity/order/restoration changed.')
            proofs.append(check_proof(r['proof'], role == 'candidate', receipt['pid']))
            a = load_arrays(d, r)
            names = {'pixels', 'targets', 'masks', 'loss', 'input'} | {f'{kind}-{j}' for kind in ('prediction', 'cotangent') for j in range(16)}
            if (set(a) != names or any(v.dtype != (np.int64 if k == 'masks' else np.float32) for k, v in a.items())
                    or a['loss'].shape != () or a['pixels'].shape != (1, 3, 4, 16, 16)
                    or a['input'].shape != a['pixels'].shape or a['targets'].shape != (1, 32, 128)
                    or a['masks'].shape != (16, 32)):
                raise ValueError('All actual objective returns required.')
            # Each row is original context then target indices, with 8/24 or 24/8 split.
            for j, packed in enumerate(a['masks']):
                nc = (8, 24)[j % 2]
                if not same(np.sort(packed), np.arange(32, dtype=np.int64)):
                    raise ValueError('Complementary original masks required.')
                target = a['targets'][:, packed[nc:], :]
                pred, cot = a[f'prediction-{j}'], a[f'cotangent-{j}']
                if pred.dtype != np.float32 or cot.dtype != np.float32 or pred.shape != target.shape or cot.shape != target.shape:
                    raise ValueError('Actual prediction/cotangent shape/dtype changed.')
                # Native L1 / mean features / 2 families / 8 draws: multiply in the
                # actual gradient order (rather than rounding a decimal reciprocal).
                want = np.sign(pred-target).astype(np.float32) / np.float32(8)
                want = want / np.float32(2) / np.float32(pred.size)
                if not same(cot, want):
                    raise ValueError('Actual incoming cotangent differs from native L1 reduction.')
            family_losses = [np.abs(a[f'prediction-{j}']-a['targets'][:, a['masks'][j][(8, 24)[j % 2]:], :]).mean(dtype=np.float32) for j in range(16)]
            native_loss = np.asarray(family_losses, dtype=np.float32).reshape(8, 2).mean(axis=1, dtype=np.float32).mean(dtype=np.float32)
            if not agreement(a['loss'], native_loss, LOSS_BOUND)['passed']:
                raise ValueError('Saved loss differs from original L1 family/draw reduction.')
            per_role[role] = (r, a)
        candidate_receipt, c = per_role['candidate']
        if mode == 'paired':
            _, o = per_role['original']
            check = {'input': agreement(c['input'], o['input'], INPUT_BOUND),
                     'loss': agreement(c['loss'], o['loss'], LOSS_BOUND),
                     'forward_and_native_L1_bitwise': all(same(v, o[k]) for k, v in c.items() if k not in ('input', 'loss'))}
            reported = candidate_receipt['comparison']
            if reported['forward_and_native_L1_bitwise'] != check['forward_and_native_L1_bitwise'] or any(
                    reported[k][field] != check[k][field] for k in ('input', 'loss')
                    for field in ('passed', 'failed_elements', 'max_absolute_error', 'max_tolerance_ratio', 'rtol', 'atol')):
                raise ValueError('Numerical metadata changed.')
            # Runtime comparison includes a relative L2 metric; cold reader computes
            # the full per-element gate independently instead of trusting that metric.
            if (not check['input']['passed'] or not check['loss']['passed']
                    or not check['forward_and_native_L1_bitwise']):
                raise ValueError('Candidate rejected at a matched iterate; stop.')
            check['gradient_sign_differences'] = int(np.count_nonzero(np.sign(c['input']) != np.sign(o['input'])))
            checks.append(check)
        points.append(c)
    if any(not same(p['targets'], points[0]['targets']) or not same(p['masks'], points[0]['masks']) for p in points):
        raise ValueError('Clean target or attack bank changed across iterates.')
    logs = receipt['no_grad_calls']
    if (len(logs) != 6 or [r['label'] for r in logs] != ['attack', 'score-clean', 'attack', 'attack', 'attack', 'score-adversarial']
            or [r['draws'] for r in logs] != [8, 4, 8, 8, 8, 4]
            or any(r['grad_enabled'] or r['custom_route_active'] for r in logs)):
        raise ValueError('All original no-grad forwards must remain outside custom routes.')
    center = arrays['clean']
    if center.dtype != np.float32 or center.min() < 0 or center.max() > 1:
        raise ValueError('Original raw clean FP32 pixels required.')
    eps, alpha = np.float32(4/255), np.float32(1/255)
    lower, upper = np.maximum(center-eps, np.float32(0)), np.minimum(center+eps, np.float32(1))
    def clamp(v):
        return np.minimum(np.maximum(v, lower), upper)
    def hash_pixels(p):
        return hashlib.sha256(p.tobytes()).hexdigest()
    if arrays['restart-noise'].shape != (2, 1, 3, 4, 16, 16) or arrays['restart-noise'].dtype != np.float32:
        raise ValueError('Two fixed raw restart draws required.')
    best, loss = center.copy(), np.asarray(logs[2]['loss'], dtype=np.float32)
    considered = [{'kind': 'clean', 'loss': float(loss[0]), 'selected': True}]
    def consider(p, v, name):
        nonlocal best, loss
        v = np.asarray(v, dtype=np.float32).reshape(1)
        improved = v > loss
        if improved[0]:
            best, loss = p.copy(), v.copy()
        considered.append({'kind': name, 'loss': float(v[0]), 'selected': bool(improved[0])})
    for restart in range(2):
        noise = arrays['restart-noise'][restart]
        if noise.min() < 0 or noise.max() >= 1:
            raise ValueError('Raw uniform restart draws must lie in [0,1).')
        current = clamp(center+(np.float32(2)*noise-np.float32(1))*eps)
        for step in range(2):
            i = restart*2+step
            p = points[i]
            if not same(current, p['pixels']):
                raise ValueError('Random start or sign/projection update changed.')
            consider(current, p['loss'], f'iterate-{i}')
            current = clamp(current+alpha*np.sign(p['input']))
        if hash_pixels(current) != logs[3+restart]['pixels']:
            raise ValueError('Final restart no-grad score used different pixels.')
        consider(current, logs[3+restart]['loss'], f'final-{restart}')
    if not same(best, arrays['adversarial']) or not same(loss, arrays['attack-loss']):
        raise ValueError('Best-over-clean/starts/all-iterates strict per-example selection changed.')
    for i, name in ((0, 'clean-attack-loss'), (1, 'clean-score'), (5, 'adversarial-score')):
        if not same(arrays[name], np.asarray(logs[i]['loss'], dtype=np.float32)):
            raise ValueError('Scoring output was substituted.')
    if (logs[0]['pixels'] != hash_pixels(center) or logs[1]['pixels'] != hash_pixels(center)
            or logs[2]['pixels'] != hash_pixels(center) or logs[5]['pixels'] != hash_pixels(best)
            or logs[0]['loss'] != logs[2]['loss']):
        raise ValueError('Clean/selected scoring identities changed.')
    replay = None
    if mode == 'replay':
        if paired is None:
            raise ValueError('Conditional replay requires its complete paired rollout.')
        paired = Path(paired)
        before = read_json(paired/'receipt.json')
        if (not before.get('complete') or before.get('mode') != 'paired' or before['pid'] == receipt['pid']
                or before['sources'] != receipt['sources'] or before['model_state_sha256'] != receipt['model_state_sha256']
                or receipt.get('paired_receipt_sha256') != digest(paired/'receipt.json')):
            raise ValueError('Fresh-process replay and exact source-bound paired evidence required.')
        previous = load_arrays(paired, before)
        compared = 0
        for name, a in arrays.items():
            if not same(a, previous[name]):
                raise ValueError('Fresh rollout output mismatch: '+name)
            compared += 1
        for i, p in enumerate(points):
            prior_receipt = read_json(paired/f'candidate-{i}'/'receipt.json')
            prior = load_arrays(paired/f'candidate-{i}', prior_receipt)
            for name, a in p.items():
                if not same(a, prior[name]):
                    raise ValueError('Fresh per-iterate replay mismatch: '+str(i)+'/'+name)
                compared += 1
        replay = {'fresh_process': True, 'all_returns_bitwise': True, 'arrays_compared': compared}
    return {'kind': 'tiny-pgd-saved-review-v1', 'mode': mode, 'passed': True,
            'same_input_checks': checks, 'proofs': proofs, 'all_candidate_selection': considered,
            'projection_max_abs': float(np.abs(best-center).max()), 'fresh_rollout_replay': replay,
            'gradient_objectives': len(expected), 'CPU_only': True, 'CUDA_qualified': False,
            'research_attack_optimizer_performance_or_training_qualified': False}
