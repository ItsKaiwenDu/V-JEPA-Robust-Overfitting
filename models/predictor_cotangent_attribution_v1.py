"""Saved-value sparse cotangent patches and a strictly local tiny CPU prototype.

There is no university entry point or resource authorization in this module.
Cross cotangents are interventions for attribution, not native-loss gradients.
"""
import copy
import hashlib
import json
import os
import platform
import threading
import numpy as np

KIND = 'predictor-cotangent-sparse-values-v1'
DOMAINS = {'target-saved-only': (2048, 1024), 'local-tiny-CPU': (32, 128)}


def digest_array(value):
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or not value.size or not np.isfinite(value).all():
        raise ValueError('Nonempty finite saved FP32 array required.')
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Exact SHA256 required.')
    return value


def mask_digest(masks):
    return hashlib.sha256(json.dumps(masks, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def build_bundle(original, candidate, targets, masks, *, baseline_sha256, domain):
    if domain not in DOMAINS:
        raise ValueError('Explicit saved-target or local-tiny domain required.')
    tokens, features = DOMAINS[domain]
    if targets.shape != (1, tokens, features) or len(original) != 16 or len(candidate) != 16:
        raise ValueError('Full sixteen-pair domain required.')
    target_sha = digest_array(targets); _sha(baseline_sha256)
    if len(masks) != 8 or any(len(d) != 2 for d in masks):
        raise ValueError('All eight draws and two families required.')
    rows = []
    for k, (a, b) in enumerate(zip(original, candidate)):
        d, f = divmod(k, 2); pair = masks[d][f]
        if set(pair) != {'context', 'target'}:
            raise ValueError('Exact context/target indices required.')
        for ids in pair.values():
            if (not isinstance(ids, list) or len(ids) != 1 or not ids[0]
                    or any(type(v) is not int or not 0 <= v < tokens for v in ids[0])
                    or len(set(ids[0])) != len(ids[0])):
                raise ValueError('Mask geometry or unique indices changed.')
        if set(pair['context'][0]) & set(pair['target'][0]):
            raise ValueError('Context and target must be disjoint.')
        ids = np.asarray(pair['target'], dtype=np.int64)
        selected = targets[:, ids[0], :]
        shape = (1, ids.shape[1], features)
        values = []
        for value in (a, b):
            if set(value) != {'prediction', 'cotangent'} or any(x.shape != shape for x in value.values()):
                raise ValueError('Actual prediction/cotangent geometry required.')
            p, c = value['prediction'], value['cotangent']
            ps, cs = digest_array(p), digest_array(c)
            scale = np.float32(1)/np.float32(8)/np.float32(2)/np.float32(ids.size*features)
            expected = np.sign(np.subtract(p, selected, dtype=np.float32))*scale
            if not np.array_equal(c.view(np.uint32), expected.view(np.uint32)):
                raise ValueError('Saved cotangent disagrees with original native L1 tail.')
            values.append({'prediction_sha256': ps, 'cotangent_sha256': cs})
        a_bits = np.ascontiguousarray(a['cotangent']).view(np.uint32)
        b_bits = np.ascontiguousarray(b['cotangent']).view(np.uint32)
        edits = [{'coordinate': [int(x) for x in idx], 'bits': [f'{int(a_bits[tuple(idx)]):08x}', f'{int(b_bits[tuple(idx)]):08x}']}
                 for idx in np.argwhere(a_bits != b_bits)]
        rows.append({'index': k, 'draw': d, 'family': f, 'shape': list(shape), 'arms': values, 'edits': edits})
    bundle = {'kind': KIND, 'domain': domain, 'baseline_sha256': baseline_sha256,
        'targets_sha256': target_sha, 'masks_sha256': mask_digest(masks), 'pairs': rows,
        'changed_entries': sum(len(row['edits']) for row in rows), 'authorized': False,
        'research_admitted': False, 'new_target_execution_permitted': False}
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle):
    keys = {'kind', 'domain', 'baseline_sha256', 'targets_sha256', 'masks_sha256', 'pairs',
            'changed_entries', 'authorized', 'research_admitted', 'new_target_execution_permitted'}
    if (not isinstance(bundle, dict) or set(bundle) != keys or bundle['kind'] != KIND
            or bundle['domain'] not in DOMAINS or any(bundle[x] is not False for x in
            ('authorized', 'research_admitted', 'new_target_execution_permitted'))):
        raise ValueError('Inert exact sparse-value bundle required.')
    for key in ('baseline_sha256', 'targets_sha256', 'masks_sha256'):
        _sha(bundle[key])
    tokens, features = DOMAINS[bundle['domain']]
    if len(bundle['pairs']) != 16:
        raise ValueError('All sixteen pairs required.')
    count = 0
    for k, row in enumerate(bundle['pairs']):
        if (set(row) != {'index', 'draw', 'family', 'shape', 'arms', 'edits'}
                or any(type(row[x]) is not int for x in ('index', 'draw', 'family'))
                or (row['index'], row['draw'], row['family']) != (k, k//2, k%2)
                or len(row['shape']) != 3 or any(type(x) is not int for x in row['shape'])
                or row['shape'][0] != 1 or not 0 < row['shape'][1] < tokens or row['shape'][2] != features
                or len(row['arms']) != 2):
            raise ValueError('Pair identity or geometry changed.')
        for arm in row['arms']:
            if set(arm) != {'prediction_sha256', 'cotangent_sha256'}:
                raise ValueError('Whole native prediction/cotangent hashes required.')
            for want in arm.values():
                _sha(want)
        seen = set()
        for edit in row['edits']:
            if set(edit) != {'coordinate', 'bits'} or len(edit['coordinate']) != 3:
                raise ValueError('Exact sparse edit required.')
            index = tuple(edit['coordinate'])
            if any(type(v) is not int or not 0 <= v < n for v, n in zip(index, row['shape'])) or index in seen:
                raise ValueError('Invalid or repeated sparse coordinate.')
            bits = edit['bits']
            if (len(bits) != 2 or bits[0] == bits[1] or any(not isinstance(v, str) or len(v) != 8
                    or any(c not in '0123456789abcdef' for c in v) for v in bits)):
                raise ValueError('Two distinct exact FP32 bit patterns required.')
            if not np.isfinite(np.array([int(v, 16) for v in bits], dtype=np.uint32).view(np.float32)).all():
                raise ValueError('Finite cotangent bit patterns required.')
            seen.add(index)
        count += len(seen)
    if type(bundle['changed_entries']) is not int or count != bundle['changed_entries']:
        raise ValueError('Sparse coverage count changed.')
    return bundle


def replay_pair(native, row, source, desired):
    if type(source) is not int or type(desired) is not int or source not in (0, 1) or desired not in (0, 1):
        raise ValueError('Exact source/desired arm index required.')
    if list(native.shape) != row['shape'] or digest_array(native) != row['arms'][source]['cotangent_sha256']:
        raise ValueError('Whole incoming native cotangent changed.')
    result = np.ascontiguousarray(native).copy(); bits = result.view(np.uint32)
    for edit in row['edits']:
        index = tuple(edit['coordinate'])
        if int(bits[index]) != int(edit['bits'][source], 16):
            raise ValueError('Sparse source entry changed.')
        bits[index] = int(edit['bits'][desired], 16)
    if digest_array(result) != row['arms'][desired]['cotangent_sha256']:
        raise ValueError('Replayed whole cotangent fails the saved destination hash.')
    return result


class LocalCotangentReplay:
    """One tiny CPU intervention only; never a target/native-loss qualification."""
    def __init__(self, model, targets, banks, bundle, *, path, cotangent):
        if platform.node() in ('linux10603', 'linux10609'):
            raise ValueError('Local CPU prototype has no university execution grant.')
        import torch
        validate_bundle(bundle)
        if (bundle['domain'] != 'local-tiny-CPU' or targets.device.type != 'cpu'
                or targets.dtype != torch.float32 or targets.requires_grad or list(targets.shape) != [1, 32, 128]
                or any(t.device.type != 'cpu' for t in (*model.parameters(), *model.buffers()))):
            raise ValueError('Only the local tiny CPU domain is supported.')
        if type(path) is not int or type(cotangent) is not int or path not in (0, 1) or cotangent not in (0, 1):
            raise ValueError('Exact path and cotangent arm required.')
        if len(banks) != 8 or any(len(d) != 2 for d in banks):
            raise ValueError('Eight draws/two families required.')
        masks = []
        for draw in banks:
            pairs = []
            for pair in draw:
                if any(t.device.type != 'cpu' or t.dtype != torch.int64 for t in (pair.context, pair.target)):
                    raise ValueError('Original CPU masks required.')
                pairs.append({'context': pair.context.tolist(), 'target': pair.target.tolist()})
            masks.append(pairs)
        if mask_digest(masks) != bundle['masks_sha256'] or digest_array(targets.detach().numpy()) != bundle['targets_sha256']:
            raise ValueError('Bound mask or clean target bytes changed.')
        if model.predictor._forward_hooks or model.predictor._forward_pre_hooks:
            raise ValueError('Refuse a previously observed predictor.')
        self.model, self.bundle = model, copy.deepcopy(bundle)
        self.path, self.cotangent = path, cotangent
        self.pairs = [p for draw in banks for p in draw]
        self.pid, self.creator = os.getpid(), threading.get_ident()
        self.active = self.closed = self.restored = False
        self.rows, self.handles, self.module_hook = [], [], None
        self.lock = threading.RLock()

    def __enter__(self):
        if self.active or self.closed:
            raise ValueError('The local prototype is single use.')
        self.active = True
        try:
            self.module_hook = self.model.predictor.register_forward_hook(self._forward, with_kwargs=True)
        except BaseException:
            self.active = False; self.closed = True
            raise
        return self

    def _forward(self, module, args, kwargs, output):
        import torch
        with self.lock:
            k = len(self.rows)
            if not self.active or k >= 16 or module is not self.model.predictor or self.pid != os.getpid() or self.creator != threading.get_ident():
                raise ValueError('Unowned, inactive or additional forward call.')
            pair, expected = self.pairs[k], self.bundle['pairs'][k]
            if (len(args) != 3 or kwargs != {'mask_index': 0} or not torch.equal(args[1], pair.context)
                    or not torch.equal(args[2], pair.target) or output.device.type != 'cpu'
                    or output.dtype != torch.float32 or not output.requires_grad
                    or list(output.shape) != expected['shape']
                    or digest_array(output.detach().numpy()) != expected['arms'][self.path]['prediction_sha256']):
                raise ValueError('Original prediction/order/mask identity changed.')
            row = {'index': k, 'native': None, 'applied': None, 'hook_removed': False}
            self.rows.append(row)
            def incoming(gradient):
                with self.lock:
                    if (not self.active or self.pid != os.getpid() or row['native'] is not None
                            or gradient.device.type != 'cpu' or gradient.dtype != torch.float32 or gradient.requires_grad):
                        raise ValueError('Unexpected or repeated native incoming cotangent.')
                    native = gradient.detach().numpy()
                    patched = replay_pair(native, expected, self.path, self.cotangent)
                    row['native'] = gradient.detach().clone()
                    row['applied'] = torch.from_numpy(patched)
                    if digest_array(gradient.detach().numpy()) != expected['arms'][self.path]['cotangent_sha256']:
                        raise ValueError('Intervention mutated the native cotangent.')
                    return None if self.path == self.cotangent else row['applied'].clone()
            self.handles.append(output.register_hook(incoming))
        return None

    def __exit__(self, typ, error, tb):
        errors = []
        for handle in ([self.module_hook] if self.module_hook is not None else []) + self.handles:
            try:
                handle.remove()
            except BaseException as exc:
                errors.append(exc)
        self.active = False; self.closed = True
        self.restored = not errors and not self.model.predictor._forward_hooks and not self.model.predictor._forward_pre_hooks
        for row in self.rows:
            row['hook_removed'] = self.restored
        if typ is None and (not self.restored or len(self.rows) != 16 or any(r['native'] is None for r in self.rows)):
            raise ValueError('Incomplete local intervention or hook restoration.')
        return False


def analyze_saved_target(root):
    """Read the sealed target capture, prove reversible patches, infer no VJP."""
    from pathlib import Path
    import torch
    from models import predictor_boundary_record_v1 as records
    root = Path(root)
    read = lambda p: json.loads(Path(p).read_text())
    def file_sha(p):
        with Path(p).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    relative = 'report/checks/2026-11-06-predictor-boundary-v1-target-review.json'
    review_path = root/relative
    if file_sha(review_path) != 'b309378641673e390d6e63e521229c663453c45dffa5e8df30a797afc2b64051':
        raise ValueError('Frozen target review changed.')
    review = read(review_path)
    for field in ('raw_review', 'sealed_manifest', 'review_source'):
        item = review[field]
        if file_sha(root/item['path']) != item['sha256']:
            raise ValueError('Frozen review/manifest binding changed.')
    pins = read(root/review['sealed_manifest']['path'])
    if len(pins) != 10665:
        raise ValueError('Frozen target coverage changed.')
    for name, want in pins.items():
        if file_sha(root/name) != want:
            raise ValueError('Frozen source/value changed: '+name)
    handoff = root/'runs/pretrained/diagnostic/predictor-boundary-v1-20261106/linux10603-final'
    claim = handoff/'ATTEMPT.json'
    if file_sha(claim) != '701c196c878eea81f6b6810fc0c2bcde5468c257a111d224e6dd6d14e930fa7e':
        raise ValueError('Consumed target claim changed.')
    saved = handoff/'attended-uwpuxma2/saved'
    workers = [records.load(saved/'results'/str(i), control_sha256=review['control_sha256'], require_passed=(i==0)) for i in (0,1)]
    if workers[0]['baseline'] != workers[1]['baseline'] or workers[1]['passed'] or not workers[1]['capture_complete']:
        raise ValueError('Saved native comparison domain changed.')
    def array(folder, item):
        p=folder/item['file']
        if file_sha(p) != item['sha256'] or p.stat().st_size != item['bytes'] or not item['saved_before_numerical_gate']:
            raise ValueError('Actual saved tensor changed.')
        t=torch.load(p, weights_only=True, map_location='cpu')
        if t.requires_grad or t.dtype != torch.float32 or str(t.device) != 'cpu':
            raise ValueError('Detached actual CPU-loaded FP32 values required.')
        a=t.numpy()
        if list(a.shape) != item['source']['shape'] or digest_array(a) != item['source']['sha256']:
            raise ValueError('Typed tensor signature changed.')
        return a
    targets = [array(saved/'results'/str(i), w['artifacts']['targets']) for i,w in enumerate(workers)]
    if not np.array_equal(targets[0].view(np.uint32), targets[1].view(np.uint32)):
        raise ValueError('Clean targets differ.')
    masks = [read(saved/'results'/str(i)/'mask-indices.json') for i in (0,1)]
    if masks[0] != masks[1]:
        raise ValueError('Native mask indices differ.')
    pairs = [[{kind:array(saved/'results'/str(i), row[kind]) for kind in ('prediction','cotangent')}
              for row in w['observations']] for i,w in enumerate(workers)]
    baseline_sha = hashlib.sha256((json.dumps(workers[0]['baseline'],sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
    bundle = build_bundle(*pairs, targets[0], masks[0], baseline_sha256=baseline_sha, domain='target-saved-only')
    if bundle['changed_entries'] != 11:
        raise ValueError('Previously reviewed cotangent-change count changed.')
    for k, row in enumerate(bundle['pairs']):
        for source, desired in ((0,0),(1,1),(0,1),(1,0)):
            result = replay_pair(pairs[source][k]['cotangent'], row, source, desired)
            if not np.array_equal(result.view(np.uint32),pairs[desired][k]['cotangent'].view(np.uint32)):
                raise ValueError('Whole sparse replay disagrees with the actual saved destination.')
    summary = {'kind':'predictor-cotangent-offline-analysis-v1', 'prior_review':{'path':relative,'sha256':file_sha(review_path)},
        'prior_inputs_reverified':len(pins), 'consumed_attempt_sha256':file_sha(claim),
        'changed_entries':bundle['changed_entries'],'pairs_with_changes':sum(bool(r['edits']) for r in bundle['pairs']),
        'whole_array_patch_comparisons':64,'bitwise_reversible_sparse_values':True,
        'all_actual_saved_L1_cotangents_reconstructed':True,'saved_target_Jacobians_present':False,
        'target_cross_VJP_cells_present':False,'quantitative_causal_share_identified':False,
        'candidate_remains_rejected':True,'original_input_bounds':{'rtol':.002,'atol':2e-7},
        'new_model_forward_backward_SDPA_CUDA_or_remote_work':False,'paid_compute':False,
        'research_admitted':False,'financial_state_unchanged':review['financial_state_unchanged']}
    return bundle, summary
