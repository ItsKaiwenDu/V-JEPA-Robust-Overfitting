"""Immutable reference I/O for one missing replay; never an execution grant.

No tensors are computed. The original cold verifier runs unchanged code with a
private, content-addressed I/O namespace. Its source control remains historical.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import FunctionType

from models import predictor_forward_preserving_records_v1 as records

MiB = 2**20
KIND = 'forward-preserving-replay-reference-capsule-v1'
PLAN = 'configs/predictor_missing_replay_plan_v1.json'
PLAN_SHA = '2ca7e4f278198e483cf28ba8d8b35b715bf466cea37ff845f0238888c021c509'
SOURCE_CONTROL = '8b22bf0905ec0c87f6551376275e5c34542b8136cfdd2f555e0bbc39209c461e'
BASE = 'runs/pretrained/diagnostic/predictor-forward-preserving-handoff-v2-20261114'
SAVED = BASE + '/linux10603-final/attended-yo4uf0wc/saved'
ROLES = {'same_replay_original': 2, 'fresh_process_candidate_reference': 1}
PROOFS = {'worker.json', 'success.json', 'native-calls.jsonl', 'VJP-calls.jsonl', 'scopes.json'}
REVIEW = 'report/checks/2026-11-14-forward-preserving-v2-target-review.json'
LINEAGE = {
    'plan.json': (PLAN, PLAN_SHA),
    'target-review.json': (REVIEW, '597536c9ea803ad2f6144e99a23c834c7af5404ba29dfc47cee3741b574d4e0b'),
    'target-pins.json': (BASE + '/target-review/saved-target-pins.json', 'bfae52f42370f1e05abc0248f2cf173d5effc5ada31a68f0f8f440b563cbfc34'),
    'source-control.json': (SAVED + '/control.json', SOURCE_CONTROL),
    'source-authorization.json': (SAVED + '/authorization.json', '09eb6a19731061c3a9b8e16af5bad51a698082a3cd3b9f7a02c990ecf41082ee'),
    'source-claim.json': (BASE + '/linux10603-final/ATTEMPT.json', '1b0ea09c124f73c61330a30a796b25d0ade1a7dae18f8742babdd49b40ae7296'),
    'parent-session.json': (SAVED + '/session.json', 'c7a8def566bfefab23cbbe4c04698a9fb41e4099549d5611a2cfcc329650cc43'),
}
FROZEN_CODE = {
    'models/predictor_forward_preserving_records_v1.py': '24639c3b3d306cc67a02a2327ce2c40b4829788e2da48e373c6edab86255eb0f',
    'models/predictor_boundary_record_v1.py': 'c76c6e1564b2fc6d14c47b44f926b7fb2b12bc288e1fe7e1714cd5ca82dd484b',
    'models/predictor_eot8_record_v2.py': 'c75455b9a08f1b25bbb7269d4ae3c1fa34826656ba1800c0a45fdf9433765dd9',
    'models/predictor_boundary_contract_v1.py': 'f05cb2ca07aade34a95801bc97a07f9044de0493a2acc016d703749e812810aa',
    'models/predictor_eot8_owned_receipt_v1.py': '55005fd96e994c660e693e0347a337d01826eaf5224ecf6f38329a8905538531',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def parse(data):
    return records.atomic._parse(data)


def relative(name):
    p = PurePosixPath(name)
    if not isinstance(name, str) or not name or str(p) != name or p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('Canonical relative capsule path required.')
    return p.parts


def npy_type(data):
    """Decode only a bounded literal NPY header, never pickle or array arithmetic."""
    if data[:6] != b'\x93NUMPY' or data[6:8] not in (b'\x01\x00', b'\x02\x00'):
        raise ValueError('Only plain NPY v1/v2 values are allowed.')
    width = 2 if data[6] == 1 else 4
    start = 8 + width
    length = int.from_bytes(data[8:start], 'little')
    if not 1 <= length <= 4096 or start + length > len(data) or not data[start:start+length].endswith(b'\n'):
        raise ValueError('Bounded complete NPY header required.')
    try:
        node = ast.parse(data[start:start+length].decode('latin1').strip(), mode='eval').body
        if not isinstance(node, ast.Dict) or len(node.keys) != 3 or len({ast.literal_eval(k) for k in node.keys}) != 3:
            raise ValueError('Unique literal NPY keys required.')
        head = ast.literal_eval(node)
    except (SyntaxError, TypeError, UnicodeError) as error:
        raise ValueError('Literal NPY header required.') from error
    shape = head.get('shape')
    if (set(head) != {'descr', 'fortran_order', 'shape'} or head['descr'] not in ('<f4', '<i8')
            or head['fortran_order'] is not False or not isinstance(shape, tuple) or len(shape) > 5
            or any(type(n) is not int or not 1 <= n <= 4096 for n in shape)):
        raise ValueError('Exact non-object C-order FP32/int64 NPY required.')
    size = 4 if head['descr'] == '<f4' else 8
    if start + length + math.prod(shape)*size != len(data):
        raise ValueError('NPY extent differs from declared shape/dtype.')
    return {'dtype': 'float32' if size == 4 else 'int64', 'shape': list(shape)}


def recipe(plan, pins):
    """All logical aliases are fixed by the old plan and original target pins."""
    if set(plan['references']) != set(ROLES) or len(plan['deduplicated_array_blobs']) != 37:
        raise ValueError('Only the two complete source roles may be imported.')
    selected, aliases = {}, {}
    for sha, item in plan['deduplicated_array_blobs'].items():
        pin = pins[item['source']]
        if pin['sha256'] != sha or pin['bytes'] != item['bytes']:
            raise ValueError('Original blob pin changed.')
        selected['arrays/' + sha + '.npy'] = pin
    for role, index in ROLES.items():
        ref = plan['references'][role]
        if (ref['source_index'] != index or ref['source_control_sha256'] != SOURCE_CONTROL
                or ref['replay'] != index//2 or ref['mode'] != ('candidate' if index == 1 else 'original')
                or len(ref['artifacts']) != 36 or set(ref['proof_records']) != PROOFS):
            raise ValueError('Source mode/replay/control/completion identity changed.')
        mapping = {}
        for name, item in ref['artifacts'].items():
            pin = pins[item['path']]
            if any(item[k] != pin[k] for k in ('path', 'bytes', 'sha256')) or item['path'] != SAVED + f'/results/{index}/{name}.npy':
                raise ValueError('Logical array/source lineage changed.')
            blob = 'arrays/' + item['sha256'] + '.npy'
            if blob not in selected or selected[blob]['bytes'] != item['bytes']:
                raise ValueError('Missing exact content-addressed blob.')
            mapping[name + '.npy'] = {'member': blob, 'source': pin}
        for name, item in ref['proof_records'].items():
            pin = pins[item['path']]
            if any(item[k] != pin[k] for k in ('path', 'bytes', 'sha256')) or item['path'] != SAVED + f'/results/{index}/' + name:
                raise ValueError('Original proof/marker lineage changed.')
            member = 'references/' + role + '/' + name
            selected[member] = pin
            mapping[name] = {'member': member, 'source': pin}
        aliases[role] = mapping
    return selected, aliases


class _Folder:
    def __init__(self, reader, role):
        self.reader, self.role = reader, role

    def __truediv__(self, name):
        return _File(self, name)


class _File:
    def __init__(self, folder, name):
        self.folder, self.name = folder, name

    def exists(self):
        if self.name != 'failure.json':
            raise ValueError('Only the frozen failure-marker absence query is supported.')
        self.folder.reader._tree()  # no hidden failure/partial/extra file can be ignored
        return False


class Capsule:
    """Strict closed-tree reader anchored to directory descriptors; stdlib only."""
    def __init__(self, root, *, manifest_sha256):
        if not isinstance(manifest_sha256, str) or not re.fullmatch('[0-9a-f]{64}', manifest_sha256):
            raise ValueError('Separately bound manifest hash required.')
        self.root = Path(root)
        before = self.root.lstat()
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.closed = False
        try:
            self.anchor = os.fstat(self.fd)
            self._directory(self.anchor)
            if (before.st_dev, before.st_ino) != (self.anchor.st_dev, self.anchor.st_ino):
                raise ValueError('Capsule root changed during open.')
            self.manifest_sha256 = manifest_sha256
            raw = self._read('manifest.json', 2*MiB)
            if digest(raw) != manifest_sha256:
                raise ValueError('Separately bound capsule manifest changed.')
            self.manifest = parse(raw)
            m = self.manifest
            if (set(m) != {'kind', 'reference_only', 'execution_permitted', 'source_control_sha256', 'files', 'aliases', 'frozen_code'}
                    or m['kind'] != KIND or m['reference_only'] is not True or m['execution_permitted'] is not False
                    or m['source_control_sha256'] != SOURCE_CONTROL or m['frozen_code'] != FROZEN_CODE):
                raise ValueError('Reference-only manifest identity required.')
            for path, expected in FROZEN_CODE.items():
                if digest((Path(__file__).resolve().parents[1]/path).read_bytes()) != expected:
                    raise ValueError('Frozen cold-verifier source changed.')
            self._tree()
            lineage = {}
            for name, (path, sha) in LINEAGE.items():
                member = 'lineage/' + name
                data = self.read_member(member)
                if digest(data) != sha or m['files'][member]['source']['path'] != path:
                    raise ValueError('Pinned historical lineage changed.')
                lineage[name] = parse(data)
            self.plan = lineage['plan.json']
            self._lineage(lineage)
            expected, aliases = recipe(self.plan, lineage['target-pins.json'])
            if m['aliases'] != aliases or set(m['files']) != set(expected) | {'lineage/'+n for n in LINEAGE}:
                raise ValueError('Exact deduplicated member/alias coverage required.')
            for name, pin in expected.items():
                if m['files'][name]['source'] != pin:
                    raise ValueError('Original source descriptor changed.')
            for name, item in m['files'].items():
                source = item['source']
                if (set(item) != {'bytes', 'sha256', 'mode', 'mtime_ns', 'source'} or item['mode'] != 0o600
                        or any(item[k] != source[k] for k in ('bytes', 'sha256', 'mtime_ns'))):
                    raise ValueError('Canonical physical/source descriptor differs.')
                self.read_member(name)
            self._folders = {r: _Folder(self, r) for r in ROLES}
            self.states = {r: self.load_reference(r) for r in ROLES}
        except BaseException:
            self.close()
            raise

    def _directory(self, info):
        if (not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700
                or info.st_uid != os.getuid() or info.st_dev != self.anchor.st_dev):
            raise ValueError('Private owned directory required.')

    def _regular(self, info, cap):
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid()
                or info.st_dev != self.anchor.st_dev
                or stat.S_IMODE(info.st_mode) != 0o600 or not 0 <= info.st_size <= cap):
            raise ValueError('Owned single-link private regular file and byte cap required.')

    def close(self):
        if not self.closed:
            os.close(self.fd)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def _root_check(self):
        if self.closed:
            raise ValueError('Closed capsule reader.')
        now = self.root.lstat()
        self._directory(now)
        if (now.st_dev, now.st_ino) != (self.anchor.st_dev, self.anchor.st_ino):
            raise ValueError('Capsule root identity changed.')

    def _read(self, name, cap):
        parts = relative(name)
        self._root_check()
        parent = os.dup(self.fd)
        try:
            for component in parts[:-1]:
                nxt = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                os.close(parent); parent = nxt
                self._directory(os.fstat(parent))
            before = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            self._regular(before, cap)
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            with os.fdopen(descriptor, 'rb') as f:
                opened = os.fstat(f.fileno()); self._regular(opened, cap)
                data = f.read(cap+1); after = os.fstat(f.fileno())
            named = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            signature = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if any(signature(s) != signature(before) for s in (opened, after, named)) or len(data) != before.st_size:
                raise ValueError('Capsule file changed during read.')
            return data
        finally:
            os.close(parent)

    def _tree(self):
        self._root_check()
        expected = {'manifest.json'} | set(self.manifest['files'])
        directories = set()
        for name in expected:
            parts = relative(name)
            directories.update('/'.join(parts[:i]) for i in range(1, len(parts)))
        seen, seen_dirs = set(), set()
        def walk(fd, prefix=''):
            for name in os.listdir(fd):
                relative(name)
                key = prefix + name
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if key in directories:
                    self._directory(info)
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        self._directory(os.fstat(child)); walk(child, key + '/')
                    finally:
                        os.close(child)
                    seen_dirs.add(key)
                elif key in expected:
                    self._regular(info, 16*MiB if key.startswith('arrays/') else 8*MiB)
                    seen.add(key)
                else:
                    raise ValueError('Unexpected capsule entry: ' + key[:200])
        walk(self.fd)
        if seen != expected or seen_dirs != directories:
            raise ValueError('Exact capsule directory/member coverage required.')

    def read_member(self, name):
        item = self.manifest['files'][name]
        cap = 16*MiB if name.startswith('arrays/') else 8*MiB
        data = self._read(name, cap)
        if len(data) != item['bytes'] or digest(data) != item['sha256']:
            raise ValueError('Capsule member bytes/hash changed.')
        # Full nanosecond source timestamps are restored by the local builder;
        # tar stores seconds, so any future extractor must explicitly restore ns.
        parts = relative(name); parent = os.dup(self.fd)
        try:
            for part in parts[:-1]:
                nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                os.close(parent); parent = nxt
            info = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            if info.st_mtime_ns != item['mtime_ns']:
                raise ValueError('Capsule member nanosecond timestamp changed.')
        finally:
            os.close(parent)
        return data

    def _lineage(self, lineage):
        review = lineage['target-review.json']; parent = lineage['parent-session.json']
        claim = lineage['source-claim.json']; auth = lineage['source-authorization.json']
        if (review['initial_original_and_candidate_full_size_CUDA_numerical_gate_passed'] is not True
                or review['original_fresh_process_replay_exact'] is not True
                or review['candidate_fresh_process_replay_complete'] is not False
                or review['partial_candidate_replay_not_admitted'] is not True
                or review['complete_accepted_workers'] != 3 or review['parent_passed'] is not False
                or parent['complete'] is not False or parent['passed'] is not False
                or [t['index'] for t in parent['tasks']] != [0, 1, 2]
                or claim['consumed'] is not True or claim['control_sha256'] != SOURCE_CONTROL
                or auth['control_sha256'] != SOURCE_CONTROL or auth['maximum_new_attempts'] != 1):
            raise ValueError('Historical failure/consumed grant/initial qualification lineage changed.')
        pins_item = review['saved_target_pin_manifest']
        if pins_item['sha256'] != LINEAGE['target-pins.json'][1]:
            raise ValueError('Historical review-to-target descriptor binding changed.')
        self.parent_tasks = {r['index']: r for r in parent['tasks']}

    def _folder(self, value):
        if not isinstance(value, _Folder) or value.reader is not self or self._folders.get(value.role) is not value:
            raise ValueError('Only a registered historical reference folder is accepted.')
        return value

    def _mapped_read(self, folder, name, limit):
        folder = self._folder(folder)
        item = self.manifest['aliases'][folder.role].get(name)
        if item is None:
            raise ValueError('Unknown/partial historical reference filename.')
        data = self.read_member(item['member'])
        if len(data) > limit:
            raise ValueError('Frozen cold-verifier file cap exceeded.')
        if name.endswith('.npy'):
            array = npy_type(data)
            state = parse(self.read_member('references/' + folder.role + '/worker.json'))
            expected = state['artifacts'][name[:-4]]
            if array != {k: expected[k] for k in ('dtype', 'shape')}:
                raise ValueError('NPY header differs from the historical typed artifact.')
        return data

    def _mapped_sha(self, value):
        if not isinstance(value, _File):
            raise ValueError('Only mapped historical files may be hashed.')
        return digest(self._mapped_read(value.folder, value.name, 8*MiB))

    def load_reference(self, role):
        if role not in ROLES:
            raise ValueError('Only complete original-1 and candidate-0 may be references.')
        self._tree()
        namespace = dict(records.load.__globals__, Path=self._folder, read=self._mapped_read, sha_file=self._mapped_sha)
        verifier = FunctionType(records.load.__code__, namespace)
        verifier.__kwdefaults__ = records.load.__kwdefaults__
        state = verifier(self._folders[role], control_sha256=SOURCE_CONTROL, index=ROLES[role], live=True)
        task = self.parent_tasks[state['index']]
        if (task['worker_sha256'] != self.manifest['aliases'][role]['worker.json']['source']['sha256']
                or task['completion_sha256'] != self.manifest['aliases'][role]['success.json']['source']['sha256']
                or task['pid'] != state['pid'] or task['numerical_passed'] is not True):
            raise ValueError('Parent completion receipt differs from the imported worker.')
        return state

    def execution_binding(self, new_execution_control_sha256):
        if (not isinstance(new_execution_control_sha256, str)
                or not re.fullmatch('[0-9a-f]{64}', new_execution_control_sha256)
                or new_execution_control_sha256 == SOURCE_CONTROL):
            raise ValueError('A distinct new execution control is required; old grant is consumed.')
        # Bind only references that still pass their original byte/type/proof gates.
        for role in ROLES:
            self.load_reference(role)
        return {'kind': 'missing-candidate-replay-reference-binding-v1',
                'new_execution_control_sha256': new_execution_control_sha256,
                'capsule_manifest_sha256': self.manifest_sha256,
                'historical_reference_control_sha256': SOURCE_CONTROL,
                'references': {role: {'source_index': index,
                                     'worker_sha256': self.manifest['aliases'][role]['worker.json']['source']['sha256'],
                                     'completion_sha256': self.manifest['aliases'][role]['success.json']['source']['sha256']}
                               for role, index in ROLES.items()},
                'new_worker': {'index': 3, 'mode': 'candidate', 'replay': 1},
                'execution_permitted': False, 'authorization_required': True,
                'old_grant_or_partial_replay_reused': False}
