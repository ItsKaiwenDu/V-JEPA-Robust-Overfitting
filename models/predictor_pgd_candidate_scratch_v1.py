"""LOCAL scratch metadata/byte guard; no creation, proof, launch or cleanup API.

Callers still need a separately reviewed source bootstrap, phase coordinator and
exact authorization. Scans are observations, not quotas or atomic snapshots.
"""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import threading
from models import predictor_pgd_split_plan_v1 as plan
MiB = 2 ** 20
GiB = 2 ** 30
CONTRACT_SHA = 'fb4090047f0b9d08ae4be0d518d71bbb17a8953864746c02fdc74d3506a1fde9'
DESIGN_SHA = 'c5daae30cbe4043cf4d5265de2a2a7420d015ab1e244cc3d74fcc93b5f0186af'
SUPERVISOR_NAMES = ('bootstrap.py', 'predictor_pgd_split_scratch_v1.py', 'predictor_pgd_split_coordinator_v1.py', 'predictor_pgd_split_collection_v1.py', 'predictor_pgd_split_watch_v1.py', 'scratch_allocation_v2.py')
PINNED_SUPPORT = {'predictor_pgd_split_watch_v1.py': '07237d77806aba1c129e0e93520a2ceedd3ff2b06001f391d1cdaa46e55fae06', 'scratch_allocation_v2.py': '6b0503b8ab2c88a44a4d95c8fafbad52d1c0414293379406520a22f67700e874'}
PHASES = ('setup', 'proof', 'capture', 'collection')
ROOT_META = {'control.json': (256 * 1024, False, True), 'authorization.json': (64 * 1024, False, True), 'proposal.json': (256 * 1024, False, True), 'storage-before.json': (16 * 1024, False, True), 'session.json': (128 * 1024, True, True), 'runtime-source-manifest.json': (128 * 1024, False, True), 'installation.json': (64 * 1024, False, True), 'storage-status.json': (64 * 1024, True, True), 'allocation-events-v2.jsonl': (MiB, True, False), 'allocation-result-v2.json': (16 * 1024, False, False), 'worker-plan-numerical.json': (16 * 1024, False, True), 'worker-plan-native.json': (16 * 1024, False, True), 'worker-plan-forward-preserving.json': (16 * 1024, False, True), 'parent-numerical.json': (128 * 1024, False, True), 'parent-native.json': (128 * 1024, False, True), 'parent-forward-preserving.json': (128 * 1024, False, True), 'gpu-samples.jsonl': (512 * 1024, True, False)}
CAPS = {'raw': 1792 * MiB, 'input': 128 * MiB, 'code': 128 * MiB, 'cache': 32 * MiB, 'emergency': 16 * MiB, 'metadata': 64 * MiB, 'trace': 8 * MiB, 'proof': 5 * GiB // 2}

def encoded(v):
    return (json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()

def sha(v):
    return type(v) is str and len(v) == 64 and all((c in '0123456789abcdef' for c in v))

def natural(v):
    return type(v) is int and v >= 0

def parse_json(data):

    def pairs(rows):
        out = {}
        for key, value in rows:
            if key in out:
                raise ValueError('Duplicate storage metadata key.')
            out[key] = value
        return out

    def bad(value):
        raise ValueError('Nonfinite storage metadata: ' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)

def safe(name):
    if type(name) is not str or not name or len(name) > 768 or ('\\' in name) or any((c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/' for c in name)):
        raise ValueError('Bounded canonical ASCII scratch path required.')
    p = PurePosixPath(name)
    if p.is_absolute() or p.as_posix() != name or any((v in ('.', '..') or len(v) > 128 for v in p.parts)):
        raise ValueError('Canonical scratch path required.')
    return name

def directory_id(s):
    return (s.st_dev, s.st_ino, s.st_uid, stat.S_IMODE(s.st_mode))

def file_id(s):
    return (*directory_id(s), s.st_nlink, s.st_size, s.st_mtime_ns, s.st_blocks * 512)

def identity_binding(root):
    """Read-only metadata for the future setup owner; not host authorization."""
    root = Path(root)
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise ValueError('Canonical absolute scratch root required.')
    return {'root': str(root), 'root_identity': list(directory_id(root.lstat())), 'parent': str(root.parent), 'parent_identity': list(directory_id(root.parent.lstat()))}

class _OriginalLayout:

    def __init__(self, contract_bytes, base_bytes, design_bytes, installation_bytes, installation_sha256):
        if type(contract_bytes) is not bytes or hashlib.sha256(contract_bytes).hexdigest() != CONTRACT_SHA or type(design_bytes) is not bytes or (hashlib.sha256(design_bytes).hexdigest() != DESIGN_SHA) or (type(installation_bytes) is not bytes) or (not sha(installation_sha256)) or (len(installation_bytes) > 64 * 1024) or (hashlib.sha256(installation_bytes).hexdigest() != installation_sha256):
            raise ValueError('Exact source/storage/installation bytes required.')
        self.contract = parse_json(contract_bytes)
        self.result = plan.result_specs(plan.base_contract(base_bytes))
        design = parse_json(design_bytes)
        if len(self.result) != 419 or hashlib.sha256(plan.encoded(self.result)).hexdigest() != design['inventory']['canonical_result_sha256'] or encoded(self.contract['category_caps']) != encoded(CAPS) or (encoded(self.contract['root_metadata']) != encoded(ROOT_META)):
            raise ValueError('Frozen result namespace or whole-scratch policy changed.')
        self.installation = parse_json(installation_bytes)
        self.installation_sha256 = installation_sha256
        i = self.installation
        if type(i) is not dict or set(i) != {'kind', 'control_sha256', 'files'} or i['kind'] != 'pgd-split-scratch-installation-v1' or (not sha(i['control_sha256'])) or (type(i['files']) is not dict) or (set(i['files']) != set(SUPERVISOR_NAMES)):
            raise ValueError('Exact future source installation descriptor required; no caller callbacks.')
        for name, d in i['files'].items():
            if type(d) is not dict or set(d) != {'bytes', 'sha256'} or (not natural(d['bytes'])) or (not 0 < d['bytes'] <= 2 * MiB) or (not sha(d['sha256'])) or (name in PINNED_SUPPORT and d['sha256'] != PINNED_SUPPORT[name]):
                raise ValueError('Bounded supervisor source descriptor changed: ' + name)
        if sum((d['bytes'] for d in i['files'].values())) > 2 * MiB:
            raise ValueError('Shared supervisor code ceiling.')
        self.assets = {'original-packet.tar.gz': self.contract['packet']['archive']}
        self.assets.update({'code/' + p: d for p, d in self.contract['packet']['files'].items()})
        self.assets.update({'overlay/' + p: d for p, d in self.contract['runtime_files'].items()})
        self.assets.update({'supervisor/' + p: d for p, d in i['files'].items()})
        self.assets.update({'configs/base.json': {'bytes': len(base_bytes), 'sha256': hashlib.sha256(base_bytes).hexdigest()}, 'configs/design.json': {'bytes': len(design_bytes), 'sha256': DESIGN_SHA}})
        self.assets['installation.json'] = {'bytes': len(installation_bytes), 'sha256': installation_sha256}
        self.directories = {'.', 'capture', 'cache', 'tmp', 'cache/cuda'}
        for name in (*self.assets, *ROOT_META, *('capture/' + p for p in self.result)):
            safe(name)
            self.directories.update((str(p) for p in PurePosixPath(name).parents))

    def classify(self, name):
        safe(name)
        if name.startswith('capture/'):
            d, plain, partial = plan.classify(name[len('capture/'):], self.result)
            return dict(d, plain='capture/' + plain, partial=partial, space='result')
        if name == 'space-proof-v2.bin.partial':
            return dict(category='proof', max_bytes=5 * GiB // 2, plain=name, partial=False, mutable=False, partial_allowed=False, space='proof')
        if name == 'emergency-space.bin':
            return dict(category='emergency', max_bytes=16 * MiB, plain=name, partial=False, mutable=False, partial_allowed=False, space='emergency')
        if name.startswith(('cache/', 'tmp/')):
            if len(PurePosixPath(name).parts) > 8:
                raise ValueError('Cache directory depth ceiling.')
            return dict(category='cache', max_bytes=32 * MiB, plain=name, partial=False, mutable=True, partial_allowed=False, space='cache')
        partial = name.endswith('.partial')
        plain = name.removesuffix('.partial')
        if plain in ROOT_META:
            limit, mutable, allow = ROOT_META[plain]
            if partial and (not allow):
                raise ValueError('Disallowed supervisor metadata partial.')
            return dict(category='metadata', max_bytes=limit, plain=plain, partial=partial, mutable=mutable, partial_allowed=allow, space='metadata')
        if name in self.assets or (partial and plain == 'original-packet.tar.gz'):
            d = self.assets[plain]
            return dict(category='input' if plain == 'original-packet.tar.gz' else 'code', max_bytes=d['bytes'], plain=plain, partial=partial, mutable=False, partial_allowed=plain == 'original-packet.tar.gz', space='asset')
        raise ValueError('Unknown scratch path: ' + name)

    def directory(self, name):
        safe(name) if name != '.' else None
        if name in self.directories:
            return
        if name.startswith(('cache/', 'tmp/')) and len(PurePosixPath(name).parts) < 8:
            return
        raise ValueError('Unknown scratch directory: ' + name)

    def assess(self, files, directories, capacity, *, phase, initial=False, settled=False):
        """Pure metadata assessment; fake observations cannot authenticate execution."""
        if phase not in PHASES or type(initial) is not bool or type(settled) is not bool or (initial and phase != 'setup'):
            raise ValueError('Exact scratch phase required.')
        if type(files) is not dict or type(directories) is not dict or len(files) > 1024 or (len(directories) > 256):
            raise ValueError('Scratch entry-count ceiling.')
        if set(capacity) != {'available_bytes', 'available_inodes'} or not all((natural(v) for v in capacity.values())) or capacity['available_bytes'] < (8 * GiB if initial else GiB) or (capacity['available_inodes'] < (2048 if initial else 1024)):
            raise ValueError('Observed free-space/inode floor reached; no reservation or retry.')
        root = directories.get('.')
        if type(root) is not dict or type(root.get('identity')) not in (list, tuple) or len(root['identity']) != 4 or any((not natural(v) for v in root['identity'])):
            raise ValueError('Bounded root identity required.')
        device, _, uid, _ = root['identity']
        info = {}
        inodes = {}
        pairs = []
        for name, row in files.items():
            d = self.classify(name)
            info[name] = d
            if type(row) is not dict or set(row) != {'identity', 'bytes', 'allocated_bytes', 'mtime_ns'} or (not natural(row['bytes'])) or (not natural(row['allocated_bytes'])) or (row['bytes'] > d['max_bytes']) or (type(row['identity']) not in (list, tuple)) or (len(row['identity']) != 5) or any((not natural(v) for v in row['identity'])) or (row['identity'][3] != 384) or (row['identity'][4] not in (1, 2)) or (row['identity'][0] != device) or (row['identity'][2] != uid) or (type(row['mtime_ns']) is not int):
                raise ValueError('Typed scratch file observation refused: ' + name)
            if phase != 'proof' and d['category'] == 'proof':
                raise ValueError('Proof file outside its one proof phase.')
            if phase in ('setup', 'proof') and d['category'] in ('raw', 'trace'):
                raise ValueError('No capture arrays/trace during setup or proof.')
            if settled and d['partial']:
                raise ValueError('Unfinished publication in settled scratch.')
            inode = tuple(row['identity'][:2])
            inodes.setdefault(inode, []).append(name)
        canonical = dict(files)
        for inode, names in inodes.items():
            row = files[names[0]]
            links = row['identity'][4]
            if links == 1:
                if len(names) != 1:
                    raise ValueError('Repeated single-link inode observation.')
                continue
            if settled or len(names) != 2:
                raise ValueError('External/incomplete hardlink pair refused.')
            a, b = sorted(names, key=len)
            d = info[a]
            other = info[b]
            if b != a + '.partial' or other['plain'] != a or (not other['partial']) or d['partial'] or (not d['partial_allowed']) or (d['space'] not in ('result', 'metadata')) or (files[a] != files[b]):
                raise ValueError('Only one exact owned publication inode pair is permitted.')
            canonical.pop(a)
            pairs.append([a, b])
        if sum((d['space'] == 'cache' for d in info.values())) > 48:
            raise ValueError('Combined cache/tmp file-count ceiling.')
        if sum((not n.startswith('capture/') for n in files)) > 256:
            raise ValueError('Reserved non-result file-count ceiling.')
        result_entries = {n[len('capture/'):]: v['bytes'] for n, v in canonical.items() if n.startswith('capture/')}
        result_totals = plan.result_totals(result_entries, self.result)
        counts = {'raw': 0, 'metadata': 0, 'input': 0}
        extra_metadata = 0
        categories = dict.fromkeys(CAPS, 0)
        for name, row in canonical.items():
            d = info[name]
            categories[d['category']] += row['bytes']
            if d['partial']:
                label = 'raw' if d['category'] == 'raw' else 'input' if d['category'] == 'input' else 'metadata'
                counts[label] += 1
                if d['plain'] in canonical:
                    if not d['mutable']:
                        raise ValueError('Exclusive publication cannot replace completed bytes.')
                    extra_metadata += row['bytes']
        if counts['raw'] > 1 or counts['metadata'] > 2 or counts['input'] > 1 or (extra_metadata > 512 * 1024):
            raise ValueError('Shared scratch partial-count/extra-metadata ceiling.')
        for name in (*files, *directories):
            if name != '.' and str(PurePosixPath(name).parent) not in directories:
                raise ValueError('Every observed scratch parent must be included.')
        logical = sum(categories.values())
        allocated = 0
        for name, row in canonical.items():
            allocated += row['allocated_bytes']
        seen_dirs = set()
        for name, row in directories.items():
            self.directory(name)
            if type(row) is not dict or set(row) != {'identity', 'allocated_bytes'} or (not natural(row['allocated_bytes'])) or (type(row['identity']) not in (list, tuple)) or (len(row['identity']) != 4) or any((not natural(v) for v in row['identity'])) or (row['identity'][3] != 448) or (row['identity'][0] != device) or (row['identity'][2] != uid):
                raise ValueError('Private directory accounting required.')
            inode = tuple(row['identity'][:2])
            if inode in seen_dirs or inode in inodes:
                raise ValueError('Directory inode repeated or aliased.')
            seen_dirs.add(inode)
            allocated += row['allocated_bytes']
        if '.' not in directories:
            raise ValueError('Root allocation must be counted.')
        limit = 3 * GiB if phase in ('setup', 'proof') else 9 * GiB // 4
        if logical > limit or allocated > limit or categories['metadata'] + categories['trace'] > 64 * MiB or any((categories[k] > v for k, v in CAPS.items())):
            raise ValueError('Shared logical/allocated/category ceiling reached.')
        return {'kind': 'pgd-split-whole-scratch-observation-v1', 'phase': phase, 'initial': initial, 'settled': settled, 'files': len(files), 'directories': len(directories), 'logical_bytes': logical, 'named_bytes': sum((v['bytes'] for v in files.values())), 'allocated_bytes': allocated, 'categories': categories, 'result_categories': result_totals, 'ceiling_bytes': limit, 'capacity': dict(capacity), 'publication_pairs': pairs, 'partials': counts, 'extra_metadata_bytes': extra_metadata, 'sampling_is_not_atomic_or_kernel_quota': True, 'scientific_qualified': False, 'proof_execution_admitted': False, 'training_admitted': False}

class Guard:

    def __init__(self, root, binding, layout):
        self.root = Path(root)
        self.binding = json.loads(json.dumps(binding))
        self.layout = layout
        if type(binding) is not dict or set(binding) != {'root', 'root_identity', 'parent', 'parent_identity'} or binding['root'] != str(self.root) or (binding['parent'] != str(self.root.parent)) or (not self.root.is_absolute()) or (self.root.resolve(strict=True) != self.root):
            raise ValueError('Explicit canonical scratch identity binding required.')
        for key in ('root_identity', 'parent_identity'):
            if type(binding[key]) is not list or len(binding[key]) != 4 or (not all((natural(v) for v in binding[key]))):
                raise ValueError('Typed directory identity binding required.')
        if binding['root_identity'][2:] != [os.getuid(), 448] or binding['root_identity'][0] != binding['parent_identity'][0]:
            raise ValueError('Owned private same-device scratch binding required.')
        self.owner = (os.getpid(), threading.get_ident())
        self.pins = {}
        self.check_root()

    def check_root(self):
        if (os.getpid(), threading.get_ident()) != self.owner or self.root.resolve(strict=True) != self.root or list(directory_id(self.root.lstat())) != self.binding['root_identity'] or (list(directory_id(self.root.parent.lstat())) != self.binding['parent_identity']):
            raise ValueError('Scratch creator/root/parent identity changed.')

    def open_root(self):
        self.check_root()
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        if list(directory_id(os.fstat(fd))) != self.binding['root_identity']:
            os.close(fd)
            raise ValueError('Held scratch root changed.')
        return fd

    def scan(self, *, phase, initial=False, settled=False):
        """One anchored observation; two mutable-JSON and one immutable-raw resolution.

        No directory rescan, polling, experiment retry or content read. A missing
        listed partial must join its one current private final by cached inode.
        Collection/setup/proof/settled scans retain the original strict behavior.
        """
        files = {}
        dirs = {}
        rootfd = self.open_root()
        device = self.binding['root_identity'][0]
        uid = os.getuid()
        nodes = 0
        publications = {}
        raw_publications = {}
        enumerated = set()

        def eligible(path):
            d = self.layout.classify(path)
            return phase == 'capture' and initial is False and (settled is False) and (d['partial'] is True) and (d['partial_allowed'] is True) and (d['mutable'] is True) and (d['category'] == 'metadata') and (d['space'] in ('metadata', 'result')) and d['plain'].endswith('.json') and (type(d['max_bytes']) is int) and (0 < d['max_bytes'] <= 256 * 1024)

        def row(s, path):
            d = self.layout.classify(path)
            if s.st_dev != device or s.st_uid != uid or (not stat.S_ISREG(s.st_mode)) or (stat.S_IMODE(s.st_mode) != 384) or (not 0 <= s.st_size <= d['max_bytes']):
                raise ValueError('Private bounded same-device scratch file required: ' + path)
            return {'identity': (*directory_id(s), s.st_nlink), 'bytes': s.st_size, 'allocated_bytes': s.st_blocks * 512, 'mtime_ns': s.st_mtime_ns}

        def signature(v):
            return (tuple(v['identity']), v['bytes'], v['mtime_ns'])

        def absent(fd, leaf):
            try:
                os.stat(leaf, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                return
            raise ValueError('Partial reappeared or remains during JSON publication resolution.')

        def raw_eligible(path):
            d = self.layout.classify(path)
            return phase == 'capture' and initial is False and (settled is False) and (d['partial'] is True) and (d['partial_allowed'] is True) and (d['mutable'] is False) and (d['category'] == 'raw') and (d['space'] == 'result') and d['plain'].endswith(('.npy', '.pt')) and (type(d['max_bytes']) is int) and (0 < d['max_bytes'] <= 16 * MiB)

        def raw_signature(value):
            return (tuple(value['identity'][:4]), value['bytes'], value['mtime_ns'])

        def raw_checked(pub, value):
            if value['identity'][4] not in (1, 2) or raw_signature(value) != raw_signature(pub['row']):
                raise ValueError('Immutable raw publication identity/extent/time/links changed.')
            pub['reserve_allocated_bytes'] = max(pub['reserve_allocated_bytes'], value['allocated_bytes'])

        def refresh_raw(pub, fd):
            held = row(os.fstat(pub['fd']), pub['plain'])
            named = row(os.stat(pub['leaf'], dir_fd=fd, follow_symlinks=False), pub['plain'])
            raw_checked(pub, held)
            raw_checked(pub, named)
            if named['identity'][4] > held['identity'][4]:
                raise ValueError('Raw final link count increased after publication.')
            try:
                partial = row(os.stat(pub['partial_leaf'], dir_fd=fd, follow_symlinks=False), pub['partial'])
            except FileNotFoundError:
                pub['absence_seen'] = True
                held = row(os.fstat(pub['fd']), pub['plain'])
                named = row(os.stat(pub['leaf'], dir_fd=fd, follow_symlinks=False), pub['plain'])
                raw_checked(pub, held)
                raw_checked(pub, named)
                if held['identity'][4] != 1 or named['identity'][4] != 1:
                    raise ValueError('Absent raw partial requires one single-link final.')
                files.pop(pub['partial'], None)
                files[pub['plain']] = named
                pub['state'] = 'single-final'
            else:
                raw_checked(pub, partial)
                if pub['absence_seen'] or any((v['identity'][4] != 2 for v in (held, named, partial))):
                    raise ValueError('Only the exact two-link raw final/partial pair may remain.')
                files[pub['plain']] = named
                files[pub['partial']] = named
                pub['state'] = 'two-owned-names'

        def resolve_raw(fd, prefix, leaf, inode):
            partial = (prefix + '/' if prefix else '') + leaf
            plain = partial.removesuffix('.partial')
            if not raw_eligible(partial) or plain in self.pins or partial in self.pins or raw_publications or (type(inode) is not int) or (inode <= 0):
                raise ValueError('One distinct unpinned catalogued immutable raw publication required.')
            desc = self.layout.classify(partial)
            old_final = files.get(plain)
            old_partial = files.get(partial)
            finalfd = parentfd = None
            try:
                finalfd = os.open(leaf.removesuffix('.partial'), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                base = row(os.fstat(finalfd), plain)
                if base['identity'][1] != inode or base['identity'][4] not in (1, 2):
                    raise ValueError('Listed raw partial inode must join its bounded final.')
                for previous in (old_final, old_partial):
                    if previous is not None and (previous['identity'][4] not in (1, 2) or previous['identity'][:4] != base['identity'][:4]):
                        raise ValueError('Previously observed raw identity/links cannot be erased.')
                if old_final is not None and raw_signature(old_final) != raw_signature(base):
                    raise ValueError('Observed immutable final changed.')
                if old_partial is not None:
                    if old_partial['identity'][4] == 2 and raw_signature(old_partial) != raw_signature(base):
                        raise ValueError('Already linked raw partial changed.')
                    if old_partial['bytes'] > base['bytes']:
                        raise ValueError('Observed raw partial shrank before publication.')
                parentfd = os.dup(fd)
                pub = {'fd': finalfd, 'parent_fd': parentfd, 'prefix': prefix, 'leaf': leaf.removesuffix('.partial'), 'partial_leaf': leaf, 'partial': partial, 'plain': plain, 'listed_inode': inode, 'row': base, 'max_bytes': desc['max_bytes'], 'reserve_allocated_bytes': max([desc['max_bytes'], base['allocated_bytes']] + [v['allocated_bytes'] for v in (old_final, old_partial) if v is not None]), 'absence_seen': False}
                raw_publications[plain] = pub
                finalfd = parentfd = None
                refresh_raw(pub, fd)
            finally:
                if finalfd is not None:
                    os.close(finalfd)
                if parentfd is not None:
                    os.close(parentfd)

        def resolve(fd, prefix, leaf, inode):
            partial = (prefix + '/' if prefix else '') + leaf
            if not eligible(partial):
                raise ValueError('Only catalogued mutable JSON publication can resolve a missing name.')
            plain = partial.removesuffix('.partial')
            final_leaf = leaf.removesuffix('.partial')
            if plain in self.pins or partial in self.pins or plain in publications or (len(publications) >= 2) or (type(inode) is not int) or (inode <= 0):
                raise ValueError('Bounded distinct unpinned JSON publication required.')
            desc = self.layout.classify(partial)
            old = files.get(plain)
            prior_partial = files.get(partial)
            if old is not None and old['identity'][4] != 1:
                raise ValueError('Cannot erase a previously observed non-single-link final.')
            if prior_partial is not None and prior_partial['identity'][4] != 1:
                raise ValueError('Only a one-link stale partial can resolve after unlink.')
            finalfd = parentfd = None
            try:
                finalfd = os.open(final_leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                held = row(os.fstat(finalfd), plain)
                named = row(os.stat(final_leaf, dir_fd=fd, follow_symlinks=False), plain)
                if held['identity'][1] != inode or held['identity'][4] != 1 or signature(held) != signature(named):
                    raise ValueError('Listed partial inode must be the exact private single-link final.')
                if prior_partial is not None and prior_partial['identity'][:4] != held['identity'][:4]:
                    raise ValueError('Observed partial identity changed before final publication.')
                absent(fd, leaf)
                parentfd = os.dup(fd)
                reserve = max([desc['max_bytes'], held['allocated_bytes'], named['allocated_bytes']] + [v['allocated_bytes'] for v in (old, prior_partial) if v is not None])
                publications[plain] = {'fd': finalfd, 'parent_fd': parentfd, 'prefix': prefix, 'leaf': final_leaf, 'partial_leaf': leaf, 'listed_inode': inode, 'row': named, 'max_bytes': desc['max_bytes'], 'reserve_allocated_bytes': reserve, 'superseded_row': old}
                finalfd = parentfd = None
                files.pop(partial, None)
                files[plain] = named
            finally:
                if finalfd is not None:
                    os.close(finalfd)
                if parentfd is not None:
                    os.close(parentfd)

        def walk(fd, prefix):
            nonlocal nodes
            before = os.fstat(fd)
            name = prefix or '.'
            self.layout.directory(name)
            if not stat.S_ISDIR(before.st_mode) or before.st_uid != uid or before.st_dev != device or (stat.S_IMODE(before.st_mode) != 448):
                raise ValueError('Private same-device scratch directory required.')
            dirs[name] = {'identity': directory_id(before), 'allocated_bytes': before.st_blocks * 512}
            if len(dirs) > 256:
                raise ValueError('Directory-count ceiling.')
            with os.scandir(fd) as entries:
                for entry in entries:
                    nodes += 1
                    if nodes > 1280:
                        raise ValueError('Bounded scratch traversal required.')
                    path = (prefix + '/' if prefix else '') + entry.name
                    safe(path)
                    if path in enumerated:
                        raise ValueError('Repeated named entry during one scratch walk.')
                    enumerated.add(path)
                    listed_inode = entry.inode()
                    try:
                        observed = os.stat(entry.name, dir_fd=fd, follow_symlinks=False)
                    except FileNotFoundError:
                        if path.endswith('.partial') and raw_eligible(path):
                            plain = path.removesuffix('.partial')
                            if plain in raw_publications:
                                pub = raw_publications[plain]
                                if listed_inode != pub['listed_inode']:
                                    raise ValueError('Listed raw inode changed.')
                                refresh_raw(pub, fd)
                            else:
                                resolve_raw(fd, prefix, entry.name, listed_inode)
                            continue
                        if not path.endswith('.partial') or not eligible(path):
                            raise
                        resolve(fd, prefix, entry.name, listed_inode)
                        continue
                    if observed.st_dev != device or observed.st_uid != uid:
                        raise ValueError('Scratch device/owner changed.')
                    if stat.S_ISDIR(observed.st_mode):
                        self.layout.directory(path)
                        child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                        try:
                            if directory_id(os.fstat(child)) != directory_id(observed):
                                raise ValueError('Named/held scratch directory changed.')
                            walk(child, path)
                            if directory_id(os.stat(entry.name, dir_fd=fd, follow_symlinks=False)) != directory_id(observed):
                                raise ValueError('Directory replaced during walk.')
                        finally:
                            os.close(child)
                    else:
                        value = row(observed, path)
                        raw_plain = path.removesuffix('.partial')
                        if raw_plain in raw_publications:
                            pub = raw_publications[raw_plain]
                            if listed_inode != pub['listed_inode']:
                                raise ValueError('Cached raw entry inode changed.')
                            raw_checked(pub, value)
                            if path.endswith('.partial') and (pub['absence_seen'] or value['identity'][4] != 2):
                                raise ValueError('Raw partial reappeared or lost its owned pair.')
                            continue
                        if path in publications:
                            pub = publications[path]
                            if signature(value) != signature(pub['row']):
                                raise ValueError('Another JSON publication replaced the pinned final during scan.')
                            pub['reserve_allocated_bytes'] = max(pub['reserve_allocated_bytes'], value['allocated_bytes'])
                        files[path] = value
                        if len(files) > 1024:
                            raise ValueError('File-count ceiling.')
                        plain = path.removesuffix('.partial')
                        partial = plain + '.partial'
                        if plain in files and partial in files and eligible(partial) and (files[plain]['identity'][:2] == files[partial]['identity'][:2]) and (not (files[plain] == files[partial] and files[plain]['identity'][4] == 2)):
                            resolve(fd, prefix, Path(partial).name, files[partial]['identity'][1])
                        if plain in files and partial in files and raw_eligible(partial) and (files[plain]['identity'][:2] == files[partial]['identity'][:2]) and (not (files[plain] == files[partial] and files[plain]['identity'][4] == 2)):
                            resolve_raw(fd, prefix, Path(partial).name, files[partial]['identity'][1])
            if directory_id(os.fstat(fd)) != directory_id(before):
                raise ValueError('Directory identity changed during scan.')

        def verify_publications():
            for plain, pub in publications.items():
                fd = self.open_root()
                prefix = ''
                parents = []
                try:
                    for part in PurePosixPath(pub['prefix']).parts:
                        prefix = (prefix + '/' if prefix else '') + part
                        want = dirs[prefix]['identity']
                        named = os.stat(part, dir_fd=fd, follow_symlinks=False)
                        if directory_id(named) != want or not stat.S_ISDIR(named.st_mode):
                            raise ValueError('Publication ancestry changed.')
                        child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                        if directory_id(os.fstat(child)) != want:
                            os.close(child)
                            raise ValueError('Publication parent changed before open.')
                        parents.append((fd, part, want))
                        fd = child
                    if directory_id(os.fstat(fd)) != directory_id(os.fstat(pub['parent_fd'])):
                        raise ValueError('Held publication parent is no longer named.')
                    held = row(os.fstat(pub['fd']), plain)
                    named = row(os.stat(pub['leaf'], dir_fd=fd, follow_symlinks=False), plain)
                    if signature(held) != signature(pub['row']) or signature(named) != signature(pub['row']):
                        raise ValueError('Pinned JSON publication changed before observation completed.')
                    absent(fd, pub['partial_leaf'])
                    pub['reserve_allocated_bytes'] = max(pub['reserve_allocated_bytes'], held['allocated_bytes'], named['allocated_bytes'])
                    files[plain] = named
                    for parent, part, want in parents:
                        if directory_id(os.stat(part, dir_fd=parent, follow_symlinks=False)) != want:
                            raise ValueError('Publication ancestry moved during final verification.')
                finally:
                    os.close(fd)
                    for parent, _, _ in reversed(parents):
                        os.close(parent)

        def verify_raw_publications():
            for plain, pub in raw_publications.items():
                fd = self.open_root()
                prefix = ''
                parents = []
                try:
                    for part in PurePosixPath(pub['prefix']).parts:
                        prefix = (prefix + '/' if prefix else '') + part
                        want = dirs[prefix]['identity']
                        named = os.stat(part, dir_fd=fd, follow_symlinks=False)
                        if directory_id(named) != want or not stat.S_ISDIR(named.st_mode):
                            raise ValueError('Raw publication ancestry changed.')
                        child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                        if directory_id(os.fstat(child)) != want:
                            os.close(child)
                            raise ValueError('Raw parent changed before open.')
                        parents.append((fd, part, want))
                        fd = child
                    if directory_id(os.fstat(fd)) != directory_id(os.fstat(pub['parent_fd'])):
                        raise ValueError('Raw held parent is no longer named.')
                    refresh_raw(pub, fd)
                    for parent, part, want in parents:
                        if directory_id(os.stat(part, dir_fd=parent, follow_symlinks=False)) != want:
                            raise ValueError('Raw ancestry moved during verification.')
                finally:
                    os.close(fd)
                    for parent, _, _ in reversed(parents):
                        os.close(parent)
        try:
            walk(rootfd, '')
            verify_publications()
            verify_raw_publications()
            fs = os.fstatvfs(rootfd)
            capacity = {'available_bytes': fs.f_bavail * fs.f_frsize, 'available_inodes': fs.f_favail}
            report = self.layout.assess(files, dirs, capacity, phase=phase, initial=initial, settled=settled)
            for path, want in self.pins.items():
                if files.get(path) != want:
                    raise ValueError('Verified source asset changed after sealing: ' + path)
            if raw_publications:
                pub = next(iter(raw_publications.values()))
                raw_reserve = pub['max_bytes']
                json_reserve = sum((v['max_bytes'] for v in publications.values()))
                allocated = sum((v['reserve_allocated_bytes'] for v in (*publications.values(), *raw_publications.values())))
                virtual_raw_slot = int(pub['partial'] not in files)
                metadata_slots = len(publications)
                categories = dict(report['categories'])
                categories['raw'] += raw_reserve
                categories['metadata'] += json_reserve
                partials = dict(report['partials'])
                partials['raw'] += virtual_raw_slot
                partials['metadata'] += metadata_slots
                logical = report['logical_bytes'] + raw_reserve + json_reserve
                accounted_allocated = report['allocated_bytes'] + allocated
                accounted_files = report['files'] + virtual_raw_slot + metadata_slots
                if partials['raw'] > 1 or partials['metadata'] > 2 or accounted_files > 1024 or (report['extra_metadata_bytes'] + json_reserve > 512 * 1024) or (logical > report['ceiling_bytes']) or (accounted_allocated > report['ceiling_bytes']) or (categories['metadata'] + categories['trace'] > 64 * MiB) or any((categories[k] > v for k, v in CAPS.items())):
                    raise ValueError('Raw/JSON publication reserves exceed original shared limits.')
                result_categories = dict(report['result_categories'])
                result_categories['raw'] += raw_reserve
                result_categories['metadata'] += sum((v['max_bytes'] for name, v in publications.items() if name.startswith('capture/')))
                accounting = {'kind': 'bounded-json-and-immutable-raw-publication-v1', 'observed_logical_bytes': report['logical_bytes'], 'observed_allocated_bytes': report['allocated_bytes'], 'observed_categories': report['categories'], 'observed_partials': report['partials'], 'observed_files': report['files'], 'logical_reserve_bytes': raw_reserve + json_reserve, 'allocated_reserve_bytes': allocated, 'raw_partial_slots_reserved': 1, 'raw_virtual_partial_slots_charged': virtual_raw_slot, 'metadata_partial_slots_reserved': metadata_slots, 'raw_reserve_bytes': raw_reserve, 'metadata_reserve_bytes': json_reserve, 'raw_observation': {'final': pub['plain'], 'listed_inode': pub['listed_inode'], 'final_identity': list(files[pub['plain']]['identity']), 'state': pub['state'], 'absence_seen': pub['absence_seen'], 'logical_reserve_bytes': raw_reserve, 'allocated_reserve_bytes': pub['reserve_allocated_bytes'], 'completion_admitted': False}, 'atomic_snapshot_or_continuous_allocation_claimed': False}
                report = dict(report, publication_accounting=accounting, logical_bytes=logical, allocated_bytes=accounted_allocated, categories=categories, result_categories=result_categories, partials=partials, files=accounted_files, extra_metadata_bytes=report['extra_metadata_bytes'] + json_reserve)
            elif publications:
                reserve = sum((v['max_bytes'] for v in publications.values()))
                allocated = sum((v['reserve_allocated_bytes'] for v in publications.values()))
                if report['partials']['metadata'] + len(publications) > 2 or report['extra_metadata_bytes'] + reserve > 512 * 1024 or report['files'] + len(publications) > 1024 or (report['logical_bytes'] + reserve > report['ceiling_bytes']) or (report['allocated_bytes'] + allocated > report['ceiling_bytes']) or (report['categories']['metadata'] + report['categories']['trace'] + reserve > 64 * MiB):
                    raise ValueError('Resolved JSON publication reserve exceeds original shared limits.')
                report = dict(report, publication_accounting={'kind': 'bounded-mutable-json-publication-v1', 'observed_logical_bytes': report['logical_bytes'], 'observed_allocated_bytes': report['allocated_bytes'], 'logical_reserve_bytes': reserve, 'allocated_reserve_bytes': allocated, 'metadata_partial_slots_reserved': len(publications), 'remaining_metadata_partial_slots': 2 - report['partials']['metadata'] - len(publications), 'observations': [{'final': name, 'listed_inode': v['listed_inode'], 'final_identity': list(v['row']['identity']), 'logical_reserve_bytes': v['max_bytes'], 'allocated_reserve_bytes': v['reserve_allocated_bytes']} for name, v in publications.items()], 'atomic_snapshot_or_continuous_allocation_claimed': False}, logical_bytes=report['logical_bytes'] + reserve, allocated_bytes=report['allocated_bytes'] + allocated)
            self.check_root()
            return (report, files, dirs)
        finally:
            for pub in (*publications.values(), *raw_publications.values()):
                os.close(pub['fd'])
                os.close(pub['parent_fd'])
            os.close(rootfd)

    def read(self, name, cap):
        safe(name)
        d = self.layout.classify(name)
        if type(cap) is not int or cap < 0 or cap > d['max_bytes']:
            raise ValueError('Bounded source read cap required.')
        parts = PurePosixPath(name).parts
        fd = self.open_root()
        held = []
        try:
            for part in parts[:-1]:
                expected = os.stat(part, dir_fd=fd, follow_symlinks=False)
                if not stat.S_ISDIR(expected.st_mode) or expected.st_dev != self.binding['root_identity'][0] or expected.st_uid != os.getuid() or (stat.S_IMODE(expected.st_mode) != 448):
                    raise ValueError('Private source parent required.')
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                if directory_id(os.fstat(child)) != directory_id(expected):
                    os.close(child)
                    raise ValueError('Source parent changed before open.')
                held.append((fd, part, expected))
                fd = child
            expected = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
            if not stat.S_ISREG(expected.st_mode) or expected.st_nlink != 1 or expected.st_uid != os.getuid() or (stat.S_IMODE(expected.st_mode) != 384) or (expected.st_dev != self.binding['root_identity'][0]) or (not 0 <= expected.st_size <= cap):
                raise ValueError('Private bounded single-link source required.')
            data = bytearray()
            filefd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                if file_id(os.fstat(filefd)) != file_id(expected):
                    raise ValueError('Held source changed before read.')
                while len(data) <= cap:
                    chunk = os.read(filefd, min(MiB, cap + 1 - len(data)))
                    if not chunk:
                        break
                    data.extend(chunk)
                if file_id(os.fstat(filefd)) != file_id(expected):
                    raise ValueError('Source changed during read.')
            finally:
                os.close(filefd)
            if len(data) != expected.st_size or file_id(os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)) != file_id(expected):
                raise ValueError('Named source changed during read.')
            for parent, part, want in held:
                if directory_id(os.stat(part, dir_fd=parent, follow_symlinks=False)) != directory_id(want):
                    raise ValueError('Source ancestry replaced during read.')
            self.check_root()
            return bytes(data)
        finally:
            os.close(fd)
            for parent, _, _ in reversed(held):
                os.close(parent)

    @staticmethod
    def require_emergency(files):
        item = files.get('emergency-space.bin')
        if type(item) is not dict or item.get('bytes') != 16 * MiB or type(item.get('allocated_bytes')) is not int or (item['allocated_bytes'] < 16 * MiB):
            raise ValueError('Full allocated emergency extent required; sparse length is insufficient.')

    def proof_readiness(self):
        report, files, _ = self.scan(phase='setup', initial=True, settled=True)
        if any((n in files for n in ('space-proof-v2.bin.partial', 'allocation-events-v2.jsonl', 'allocation-result-v2.json'))):
            raise ValueError('Preserve previous proof artifacts; no retry.')
        self.require_emergency(files)
        if any((not natural(report.get(k)) for k in ('logical_bytes', 'allocated_bytes'))) or max(report['logical_bytes'], report['allocated_bytes']) + 5 * GiB // 2 + 32 * MiB > 3 * GiB:
            raise ValueError('Full proof plus original 32 MiB allocation reserve does not fit setup.')
        return dict(report, proof_can_be_proposed_only=True, proof_launch_authorized=False)

    def verify_assets(self, *, phase='capture'):
        before, files, _ = self.scan(phase=phase, settled=True)
        self.require_emergency(files)
        for name, want in self.layout.assets.items():
            if name not in files or files[name]['bytes'] != want['bytes']:
                raise ValueError('Missing/exact-extent source asset required: ' + name)
            data = self.read(name, want['bytes'])
            if hashlib.sha256(data).hexdigest() != want['sha256']:
                raise ValueError('Source asset digest changed: ' + name)
        after, current, _ = self.scan(phase=phase, settled=True)
        selected = {p: current[p] for p in self.layout.assets}
        if any((files[p] != row for p, row in selected.items())):
            raise ValueError('Source asset changed around verification.')
        self.pins = selected
        return {'kind': 'pgd-split-scratch-source-metadata-v1', 'control_sha256': self.layout.installation['control_sha256'], 'assets': len(selected), 'installation_sha256': self.layout.installation_sha256, 'source_names_and_hashes_verified': True, 'execution_or_authorization_authenticated': False, 'scientific_qualified': False}

    def verify_proof(self, expected_sha256):
        if not sha(expected_sha256):
            raise ValueError('Parent-bound proof receipt digest required.')
        data = self.read('allocation-result-v2.json', 16 * 1024)
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError('Proof receipt digest changed.')
        p = parse_json(data)
        if type(p) is not dict or p.get('kind') != 'scratch-allocation-proof-v2' or any((p.get(k) is not True for k in ('passed', 'complete', 'proof_removed', 'fsync_succeeded'))) or (p.get('preallocator_injected_for_test') is not False) or any((type(p.get(k)) is not int or p[k] != 5 * GiB // 2 for k in ('requested_bytes', 'bytes_written', 'bytes_read_back'))) or (type(p.get('setup_ceiling_bytes')) is not int) or (p['setup_ceiling_bytes'] != 3 * GiB) or (type(p.get('peak_observed_allocated_bytes')) is not int) or (not 5 * GiB // 2 <= p['peak_observed_allocated_bytes'] <= 3 * GiB) or (type(p.get('wall_seconds')) not in (int, float)) or (not math.isfinite(p['wall_seconds'])) or (not 0 < p['wall_seconds'] <= 60) or (not sha(p.get('sha256'))):
            raise ValueError('Original full fixed-extent proof receipt did not pass; no fallback.')
        self.scan(phase='capture', settled=True)
        return {'kind': 'pgd-split-storage-proof-metadata-v1', 'receipt_sha256': expected_sha256, 'requested_bytes': 5 * GiB // 2, 'original_proof_predicates_retained': True, 'proof_execution_authenticated': False, 'scientific_qualified': False}
STATIC = {'assets': {'component-inputs.tar': {'bytes': 27279360, 'sha256': '82a30efedfb0e85fd39c634c5bd0a8b82bd839b91bc7259841e2d517d495011e'}, 'configs/base.json': {'bytes': 17050, 'sha256': 'c7bb7f3af5e6b9e65ac746177b657ae966e2bc50171a6e5b516e30b848252960'}, 'configs/design.json': {'bytes': 10616, 'sha256': 'c5daae30cbe4043cf4d5265de2a2a7420d015ab1e244cc3d74fcc93b5f0186af'}, 'source/MANIFEST.json': {'bytes': 13480, 'sha256': 'a7e6740c69f5f3bf8c5fc6b71683296f2c9ebc818b644814b2f37eaf983c0e24'}, 'source/boundary/attention-calls.jsonl': {'bytes': 949546, 'sha256': '6f2ccd095b107eeaf59ba872be04de80abdfe348f4495243f1c8f19da293e126'}, 'source/boundary/attention-scopes.json': {'bytes': 312, 'sha256': 'fe85b6c1e54097cae93a1891d9ced17f4694f8de4f4754cb398fd1b33ab9d043'}, 'source/boundary/boundary-complete.json': {'bytes': 273, 'sha256': '8a7233c4138b87eca2b16b3e28c6412d2d664bd221c50cb494d8569e89bf524e'}, 'source/boundary/boundary.json': {'bytes': 9234, 'sha256': '9e4758d2209b9fd989926ee40d2ad92cc4b99aa98c86e6cec0722eaf6a83308d'}, 'source/boundary/dk.pt': {'bytes': 3147305, 'sha256': '00288e2c75abd4eef65a1e97b00f4fe25546ab20df640702375a5d7d2d15bf8c'}, 'source/boundary/dq.pt': {'bytes': 3147305, 'sha256': '6973339c06e07ba92e2c79f04aee4bbb982a3642cc4895d6c974c30840c15b09'}, 'source/boundary/dv.pt': {'bytes': 3147305, 'sha256': 'deead61cbb8cf5b6f7462219907082b6cc17db99af67734e170d6c3b35d9ffa5'}, 'source/boundary/grad_output.pt': {'bytes': 3147305, 'sha256': 'a673aabb28ca9e181c2426ee4ce81062b1fcecdc0609d31754996efdf45fcb7c'}, 'source/boundary/k.pt': {'bytes': 3147305, 'sha256': 'f80ea46c5e6bea6b117763c3551a5320b92aa45f65916d2c451553d2083aad37'}, 'source/boundary/output.pt': {'bytes': 3147305, 'sha256': '33cf9f63b54b285ee5c6fa1ed3dac252f7edcb40c4f5aa3a9d574ba686c8209e'}, 'source/boundary/q.pt': {'bytes': 3147305, 'sha256': 'eede3a77598c212faa77ca935e61622aa698fd4ffdad658e717cd80fed968866'}, 'source/boundary/v.pt': {'bytes': 3147305, 'sha256': '8881fa0dcbc8b653e5547ca2ef5533e3bfb436cfdabfe5025dd437792a7e30da'}, 'source/provenance/accepted-numerical-review.json': {'bytes': 7914, 'sha256': '13d2a40d2a989d9433f042dcf59187f84850ea8294e4ad8f494accda00074c8b'}, 'source/provenance/objective-0/VJP-calls.jsonl': {'bytes': 0, 'sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'}, 'source/provenance/objective-0/native-calls.jsonl': {'bytes': 949546, 'sha256': '6f2ccd095b107eeaf59ba872be04de80abdfe348f4495243f1c8f19da293e126'}, 'source/provenance/objective-0/numerical-complete.json': {'bytes': 5105, 'sha256': '9c688ffa5a5e34c0c3a2e6d13fb2b74780b443e2d893b35e52db045f9c677cbd'}, 'source/provenance/objective-0/numerical.json': {'bytes': 9661, 'sha256': '348f5ee5843a42a63c924ce19944228ff1cb9aed4bf288876de6a7298d8921e2'}, 'source/provenance/objective-0/scopes.json': {'bytes': 328, 'sha256': '539deace994dc1809bd5265c69be4a91d037f58440c73606aaf658e8b18d0b58'}, 'source/provenance/objective-1/numerical-complete.json': {'bytes': 5111, 'sha256': '22761fc13f36d293356e6662d7157adbb308f50fc1cbc251518e9020041d5e83'}, 'source/provenance/objective-1/numerical.json': {'bytes': 10130, 'sha256': '8af887389465c399cfb0e1b1bcf5b9e24a4fe2431c2cfad0cccf49838bdea1dc'}, 'source/provenance/objective-2/numerical-complete.json': {'bytes': 5105, 'sha256': 'b7077961078735f1407b8ce265ae9507ba33719ed186b606e71030431fc46fa9'}, 'source/provenance/objective-2/numerical.json': {'bytes': 9661, 'sha256': '8f8dbd56c57b7d8ed5b1e8eff6b62a95c880a3151c1b1fa0237d457200fe0a66'}, 'source/provenance/objective-3/numerical-complete.json': {'bytes': 5111, 'sha256': 'fafb91946d2ca6b12b241bbde81ca9f4089eb8996cd12f1a86661d978e822908'}, 'source/provenance/objective-3/numerical.json': {'bytes': 10131, 'sha256': '183023510653f86119b51811e1403cca9a7f1aa3db588afd444fe1592367a624'}, 'source/provenance/objective-4/numerical-complete.json': {'bytes': 5105, 'sha256': '9a5a3fef8ce4ba4bdd961215774ae235e2fc2f09cd7a4272abff5c8e1ee06ef8'}, 'source/provenance/objective-4/numerical.json': {'bytes': 9661, 'sha256': 'aa9fba339232bab8fac9137bd16c38e5cea4d6a63bef95f8f82a9bff8b59fc5d'}, 'source/provenance/objective-5/numerical-complete.json': {'bytes': 5111, 'sha256': 'bca25f28404c198eadeaa0471eba65e1499c153269a3936e154f641ec7f7ef5c'}, 'source/provenance/objective-5/numerical.json': {'bytes': 10130, 'sha256': 'b27fc9fe8d08935d1b954350a39865a0b5250d8f42a022176f848a26c22cb376'}, 'source/provenance/objective-6/numerical-complete.json': {'bytes': 5105, 'sha256': '277e8292d0d478482ba5385a5fbd722720f663988352624e3fc3686be5779119'}, 'source/provenance/objective-6/numerical.json': {'bytes': 9661, 'sha256': 'f9f58ec5ed398318f7ae32fb59e465afa288c6c38b9347904e088f81c01df7d3'}, 'source/provenance/objective-7/numerical-complete.json': {'bytes': 5111, 'sha256': '812ed32dda099acf9b175d6c8e11613e02357dbae93ff16e2905bb5bb7eb465e'}, 'source/provenance/objective-7/numerical.json': {'bytes': 10131, 'sha256': 'e260795efb3902fce97b860e5f90f58dc8b95a29f2d2f0e71cc51e81adf4be10'}, 'source/provenance/parent-numerical.json': {'bytes': 481, 'sha256': 'c422f69479a94734b9ec7c3d3065335d0c53bd440907935a79cc51af43856761'}, 'source/provenance/run-status.json': {'bytes': 13337, 'sha256': '56a6f931c8c9a4eb5367ef85244f497d8584eccd484a004190d1c459c2b5553d'}, 'source/provenance/session.json': {'bytes': 5177, 'sha256': '6cd4d8b18cd2a542ffb6c7c1285611e2ff7ab2cae3cc2fa7c53c248e1ec68697'}, 'overlay/attacks/__init__.py': {'bytes': 69, 'sha256': '41a090d550527278b09bfb62fbed810805b379151b7a5ab7cb5469bc12ee01cb'}, 'overlay/attacks/pgd.py': {'bytes': 5089, 'sha256': '947c3e4aa275fa98b5f84a2a7a89f74b8161d6ba498b948862f7d488712c2d81'}, 'overlay/models/__init__.py': {'bytes': 86, 'sha256': '1d079e9ff2633dfad73abff31961c9e81b7aff65e44ec6bedc60a32189630ff5'}, 'overlay/models/_pgd_component_contract_v1.py': {'bytes': 1188, 'sha256': '918854511a73fda012e1aad0703821a9e9793c5af62d3ede8b137e9f9a5f8401'}, 'overlay/models/_pgd_component_entry_v1.py': {'bytes': 5914, 'sha256': 'c59538e1114d0ee3337c161d0365571963785ea0c445709853e477cbb028ee9f'}, 'overlay/models/_pgd_component_io_v1.py': {'bytes': 32614, 'sha256': '36d70563fb2e14618a5f052e09b21b3aee089f894e17f55efd24c2c6211e4265'}, 'overlay/models/_pgd_component_job_v1.py': {'bytes': 27999, 'sha256': 'a8820eacf15c6e64c327ec85c3514bc91aa416f3a69a954645df7f658c50b180'}, 'overlay/models/_pgd_component_providers_v1.py': {'bytes': 4036, 'sha256': '57636cb1935b64576eaffc8c2015fd091ebd41b8b4efec2d2a3933ea94de380c'}, 'overlay/models/_pgd_component_source_v1.py': {'bytes': 32731, 'sha256': 'ba2c71deae006d6c3d1e9e523c4b6c4ef8f3aa14257fdc27cb88a90b903af940'}, 'overlay/models/_pgd_component_store_v1.py': {'bytes': 10349, 'sha256': '838785328996a0377492082aae2eb80a0cffcd89ab5247685cf29f39bc2631bd'}, 'overlay/models/_pgd_split_boundary_v1.py': {'bytes': 17495, 'sha256': '95237713919109f3e2c7d1ae4a3acdeb3967b1f2ef703ae675ae1489938d0980'}, 'overlay/models/_pgd_split_raw_v1.py': {'bytes': 4174, 'sha256': '78c9fcd2244d756e6c1aa5ad73c6c9f2907e27f6074156a06a5fec2a746c8663'}, 'overlay/models/_pgd_split_route_v1.py': {'bytes': 25865, 'sha256': 'fccad384e34147b615ca37a67f5c5ace9424270710f9b65a15977c30e44ca203'}, 'overlay/models/_pgd_split_startup_v1.py': {'bytes': 3449, 'sha256': '2deac57c321daeb7fcbc1979322a2f6eb50dbc7c852e19a562afd41937f84b68'}, 'overlay/models/checkpointing.py': {'bytes': 6645, 'sha256': 'edc5940bc73db42c89b3494be9b0f2098680762f4fa972305f51abe47c3d8373'}, 'overlay/models/diagnostics.py': {'bytes': 5555, 'sha256': '12185169ca1ebe2564bc0b1815ec19b8599c08f625af056bc30c25fca6f390d0'}, 'overlay/models/execution.py': {'bytes': 12043, 'sha256': 'e665f7c842c0d26733bd6c4d16887b10e2590b76f15e4fb1981b304ff37903ba'}, 'overlay/models/native_patch.py': {'bytes': 3250, 'sha256': '8f5ede9ea8ba61773ca11dffac5a2bdba01a62affc497d8327b3e1ede31fa5e7'}, 'overlay/models/patch_embed.py': {'bytes': 1827, 'sha256': '64838d575479e067ffecb8352efa7b51fcfafeabeb234e295605d87c1033b0ec'}, 'overlay/models/pilot.py': {'bytes': 2119, 'sha256': 'a8e947c5ce58c6497b653b3bdc9ac1051b045d07fe3f280f9c04daf44858ae82'}, 'overlay/models/precision_patch.py': {'bytes': 7991, 'sha256': '30e8a6b28b8f082fd771aaeab5a70a73cf200714ef2b24416a55938f61e9c550'}, 'overlay/models/predictor_eot8_adapter_v1.py': {'bytes': 7839, 'sha256': '89b7e0c67e88f2f776bb0bb6ad786a2202115e64ffc6c6f9a962381cc2e9a6b8'}, 'overlay/models/predictor_eot8_capture_v1.py': {'bytes': 2194, 'sha256': 'fcf582f0152f9ad1d93d1bf038895ba6c6c84d75a4fdfc02d5d4b2ea96e0d3e4'}, 'overlay/models/predictor_eot8_owned_receipt_v1.py': {'bytes': 4599, 'sha256': '55005fd96e994c660e693e0347a337d01826eaf5224ecf6f38329a8905538531'}, 'overlay/models/predictor_eot8_record_v2.py': {'bytes': 14128, 'sha256': 'c75455b9a08f1b25bbb7269d4ae3c1fa34826656ba1800c0a45fdf9433765dd9'}, 'overlay/models/predictor_pgd_candidate_accuracy_v1.py': {'bytes': 7105, 'sha256': '9e8bac5b01bcb9721218bc28f499ad6484a6e05396ea6ae383b69eb549c1d13e'}, 'overlay/models/predictor_pgd_candidate_reference_v1.py': {'bytes': 15140, 'sha256': 'e7933921c0143ff7b7e779c703da9062c1c954ec333a79f426ae6d1ecf6b2c13'}, 'overlay/models/predictor_pgd_component_capsule_v1.py': {'bytes': 14106, 'sha256': '453dea53b4e33b2187b8ffc248546aaba9c011e535d11dbc285a3c4fd15dcd37'}, 'overlay/models/predictor_pgd_split_backend_v4.py': {'bytes': 24232, 'sha256': '0a5ef4a5dc99b55eb5df6a3af8d11f99dbb18352b4ffab55c310a3f5a0f62622'}, 'overlay/models/predictor_pgd_split_lineage_v1.py': {'bytes': 8431, 'sha256': 'a53103bfc665b30930af2b6046a347f81369b0de3a1b72ff2743bc5f3c643647'}, 'overlay/models/predictor_pgd_split_plan_v1.py': {'bytes': 15697, 'sha256': 'cdd2729d2f05782d269a45aa4c22412dcff7b9c5d684267550eb16f59c838ba6'}, 'overlay/models/predictor_sdpa_component_events_v1.py': {'bytes': 8268, 'sha256': 'c1397e06ca89d1299a7d853e6c88124d1cc7874aae3f5763f018fbc67bd195fc'}, 'overlay/models/predictor_sdpa_component_events_v2.py': {'bytes': 15023, 'sha256': 'ebde9c0f32078979cfa5920503004bb79ca697fdc05d7adacbfc59bde41f515a'}, 'overlay/models/predictor_sdpa_launch_evidence_v1.py': {'bytes': 17552, 'sha256': 'c9749407f3d237050252ec7e9d4831c901df813a35b3b4a38ad8d693e0e1899f'}, 'overlay/models/predictor_sdpa_launch_evidence_v2.py': {'bytes': 19485, 'sha256': '05f99a96ca34d65c4167cdbd37f97600b14468aa08fbfe5aa9b581b72e5fdddb'}, 'overlay/models/predictor_sdpa_launch_evidence_v3.py': {'bytes': 19602, 'sha256': 'd8ca7092b346d3e1ccd9c43a99a4379eba21914a08ced173b692148015010675'}, 'overlay/models/runtime.py': {'bytes': 4827, 'sha256': '0764f5c63384882e06f1d409535e96b3423861866c16bc8ba5e3fa71955f83de'}, 'overlay/models/training.py': {'bytes': 16803, 'sha256': '15e280dfcc409a94f36defa42c59304e96dc198c8e4e063a9da6f4adf4f03679'}, 'overlay/models/vjepa2.py': {'bytes': 12926, 'sha256': '5501eb4ce79317d8ebab591d7fe4d8d0b74b1b5b692d2f7513d8c583d188debf'}, 'native-reference.tar': {'bytes': 13281280, 'sha256': 'dbaee50b765273f530976e9b8a8301ee055f4e2dc76b006239b4864b306b7d75'}, 'reference/native/backend.json': {'bytes': 3268, 'sha256': '4bd060d01524a8381c0dc978fd0ff248d3f1e25742df0c581b40ef47ddaa9bc5'}, 'reference/native/complete.json': {'bytes': 632, 'sha256': '51407b1c1569e6e870695b0f2ddaa6d79bb0a20a69b3d59ee3181b84e91a6b98'}, 'reference/native/dk.npy': {'bytes': 3145856, 'sha256': 'c31f6d5017e237d9488ad8ba96f745ddbcf1e5e21eb5224d618c7998f92fb49b'}, 'reference/native/dq.npy': {'bytes': 3145856, 'sha256': '218dc436f9795ef1014e910c990534eb75a06676b28145595964f0653d864afb'}, 'reference/native/dv.npy': {'bytes': 3145856, 'sha256': '46e728741ea905d43a9ebb3e0ea44f9bd7e0bf7750b6b5641e81b8f6e4d09134'}, 'reference/native/events.json': {'bytes': 569123, 'sha256': '97139d0db676bf14e2e14a0b294611687cec3e0860279e4755067c1c13b2c916'}, 'reference/native/output.npy': {'bytes': 3145856, 'sha256': '47f0ac9fbba30e14e9b82011f062596c32c05c6fa152aa89d5c1168e73b1c618'}, 'reference/native/ownership.json': {'bytes': 2845, 'sha256': '2bae7b2feb0c26f3238db3b07dfbee04114e2f5a77ebc7b191e3aa24f7cf2536'}, 'reference/native/parent.json': {'bytes': 435, 'sha256': 'cc57544291e2896af1996a1c05e20715fa9bde424c8b705c84c1a067198cb12e'}, 'reference/native/progress.jsonl': {'bytes': 29650, 'sha256': 'e13db72771087f38fb6cdbc6e7fa87a70f3b2fb03b80d35739ae0c0ecb82a134'}, 'reference/native/resource.json': {'bytes': 1197, 'sha256': 'aa5917e82ed385bd748fa878cf97e30a3b163f2b6dc7b49c248396bd03e9aa36'}, 'reference/native/worker.json': {'bytes': 6329, 'sha256': '6f1cdea1d114cc605103c645cd379f3c12867ae88855c211e97d0fd26e3fea29'}, 'reference/reviews/component.json': {'bytes': 9305, 'sha256': '45318db20a419f3f2671be590b3b3fe218f35a2a2ca0f935a37de727eceaa3f2'}, 'reference/reviews/metadata.json': {'bytes': 8346, 'sha256': '332667b3c9a6902464f75a920fb8c45bed5c37fc0387e4e9e45554de257dc56b'}, 'reference/reviews/pgd.json': {'bytes': 7914, 'sha256': '13d2a40d2a989d9433f042dcf59187f84850ea8294e4ad8f494accda00074c8b'}, 'reference/reviews/values.json': {'bytes': 7963, 'sha256': '1c632f09f34e05e37516df19c27d855780738c1208c42e3691890665f9622b53'}, 'reference/source/boundary.json': {'bytes': 9234, 'sha256': '9e4758d2209b9fd989926ee40d2ad92cc4b99aa98c86e6cec0722eaf6a83308d'}, 'reference/source/capsule-manifest.json': {'bytes': 13480, 'sha256': 'a7e6740c69f5f3bf8c5fc6b71683296f2c9ebc818b644814b2f37eaf983c0e24'}, 'reference/MANIFEST.json': {'bytes': 3289, 'sha256': '3dc2484c07c04bb72593c2ee386824edf52b8d80ec0fe3bb803e645714e5d932'}}, 'base_text': '{\n  "algorithm": "pretrained-pgd2-r2-eot8-native-input-context-diagnostic-v1",\n  "authorized": false,\n  "automatic_retries": 0,\n  "cleanup": false,\n  "date": "2026-11-14",\n  "financial_state_unchanged": {\n    "compute_cap_USD": 160,\n    "conservative_encumbrance_USD": 9.8116,\n    "full_account_reconciliation_complete": false,\n    "historical_inclusive_hold_USD": 2,\n    "historical_unassigned_gap_USD": 0.0792,\n    "last_user_reported_wallet_USD": 6.56,\n    "new_paid_hold_USD": 0,\n    "noncompute_reserve_USD": 40,\n    "project_cap_USD": 200,\n    "single_diagnostic_cap_USD": 10,\n    "unreserved_diagnostic_USD": 0.1884\n  },\n  "held_out_pixels": false,\n  "host": "linux10603",\n  "kind": "pretrained-pgd2-r2-iterated-validation-proposal-v1",\n  "new_target_execution_permitted": false,\n  "numerical_bounds": {\n    "input": {\n      "atol": 2e-07,\n      "rtol": 0.002\n    },\n    "loss": {\n      "atol": 2e-06,\n      "rtol": 1e-05\n    }\n  },\n  "paid_compute": false,\n  "parent": "/var/tmp",\n  "policy": "One new pretrained iterated-input numerical/native ownership diagnostic. Four fresh original checks at candidate iterates precede their candidate gradients. Original no-grad clean/final/scoring remain outside custom scopes. Profile the first already-counted matched pair only. Refusal stops remaining work; no clean gradient/cohort reruns, retry, optimizer, PGD20/PGD50 or performance/adoption claim.",\n  "prior_consumed_files": {\n    "runs/pretrained/diagnostic/predictor-forward-preserving-handoff-v1-20261114/linux10603-final/ATTEMPT.json": "4166295462bfe1af5f4273b35c314e7ccad8570e208428952bc3c9b32d3c290d",\n    "runs/pretrained/diagnostic/predictor-forward-preserving-handoff-v1-20261114/linux10603-final/AUTHORIZATION.json": "b8c8de17b40c575ecc5a174557d69c904f1a6b6d800f6fb4251044fbbbadd238",\n    "runs/pretrained/diagnostic/predictor-forward-preserving-handoff-v1-20261114/linux10603-final/CONTROL.json": "f6c9cf497d8bb10d080f295ba0946b14305092189d7d1ec03c6b1077f1f69a61",\n    "runs/pretrained/diagnostic/predictor-forward-preserving-handoff-v2-20261114/linux10603-final/ATTEMPT.json": "1b0ea09c124f73c61330a30a796b25d0ade1a7dae18f8742babdd49b40ae7296",\n    "runs/pretrained/diagnostic/predictor-forward-preserving-handoff-v2-20261114/linux10603-final/AUTHORIZATION.json": "09eb6a19731061c3a9b8e16af5bad51a698082a3cd3b9f7a02c990ecf41082ee",\n    "runs/pretrained/diagnostic/predictor-forward-preserving-handoff-v2-20261114/linux10603-final/CONTROL.json": "8b22bf0905ec0c87f6551376275e5c34542b8136cfdd2f555e0bbc39209c461e",\n    "runs/pretrained/diagnostic/predictor-missing-replay-handoff-v1-20261114/linux10603-ready/ATTEMPT.json": "e854df5342067a1b046f652c94a7ad72ebea1edd586b9ce1cbde7f23986f01a2",\n    "runs/pretrained/diagnostic/predictor-missing-replay-handoff-v1-20261114/linux10603-ready/AUTHORIZATION.json": "44dbc6568e0c7088a46d2c041ce3b49e0bfc564ef02e08e2db520570ae281679",\n    "runs/pretrained/diagnostic/predictor-missing-replay-handoff-v1-20261114/linux10603-ready/CONTROL.json": "af2c495cb86ab78b48e20a700358f6964182f015939fa18c5f7566aa5cdf0842",\n    "runs/pretrained/diagnostic/predictor-sdpa-continuation-v6-20261113/linux10603-final/ATTEMPT.json": "f1193936d4bb415b8ca5c793e34798b8e185b82eabd00c2f7423de26599d1bef",\n    "runs/pretrained/diagnostic/predictor-sdpa-continuation-v6-20261113/linux10603-final/AUTHORIZATION.json": "2786356c842a52e206b5005a1bf8bc16ca99de16548d7c4fe5ab47f5ddfc7c6d",\n    "runs/pretrained/diagnostic/predictor-sdpa-continuation-v6-20261113/linux10603-final/CONTROL.json": "4881d61dfa76d81fc6dcc624b1d78b29165aa8a077cc6ef5315c3d95a0c94c81"\n  },\n  "prior_inputs_manifest": "runs/pretrained/diagnostic/predictor-pgd-context-cpu-v1-20261114/sealed-inputs.json",\n  "prior_sealed_files_count": 4105,\n  "private_key_upload": false,\n  "profiling": {\n    "actual_existing_objective_indices": [\n      0,\n      1\n    ],\n    "additional_objectives": 0,\n    "performance_qualification": false,\n    "profile_memory": false,\n    "record_shapes": false,\n    "saved_format": "bounded operator/kernel aggregates with original native-owner and math-operator ancestry",\n    "warmups": 0,\n    "with_stack": false\n  },\n  "proposal_only": true,\n  "reference": {\n    "attack_seeds": [\n      865311023795641958,\n      7399188755883649554,\n      867422081190991898,\n      3902520512841466707,\n      6032771523782279135,\n      3779409266305138859,\n      169826197422819580,\n      8283481338430586684\n    ],\n    "clean_pixels": {\n      "abs_max": 1.0,\n      "dtype": "torch.float32",\n      "sha256": "d497eb48187685856e993af2cc010468be3f63b2a1125958bcbd2f3991685a3c",\n      "shape": [\n        1,\n        3,\n        16,\n        256,\n        256\n      ]\n    },\n    "clean_targets": {\n      "abs_max": 13.502695083618164,\n      "dtype": "torch.float32",\n      "sha256": "62049a4d393a59867090f34263f96cd655ca5e6b2e56890dcb2ceaa811680fd1",\n      "shape": [\n        1,\n        2048,\n        1024\n      ]\n    },\n    "confirmation_or_test_pixels_in_payload": false,\n    "data": {\n      "checkpoint_sha256": "5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012",\n      "kind": "original-training",\n      "pixel_sha256": "d497eb48187685856e993af2cc010468be3f63b2a1125958bcbd2f3991685a3c",\n      "video_id": "v_BreastStroke_g15_c01"\n    },\n    "encoder_depth": 24,\n    "existing_packet_archive": "runs/pretrained/diagnostic/eot-work-profile-university-20261029/packet/vjepa-eot-profile.tar.gz",\n    "existing_packet_bytes": 17192095,\n    "existing_packet_is_new_launcher": false,\n    "existing_packet_logical_bytes": 26446958,\n    "existing_packet_members": 99,\n    "existing_packet_sha256": "2278bce2b519943c0f83a49c556bb73dd928fe880fba0f6c56de2a4c3c2ec0da",\n    "full_mask_tensor_identities": [\n      [\n        [\n          {\n            "abs_max": 2047.0,\n            "dtype": "torch.int64",\n            "sha256": "b705ba23102114c7d545b36fc3ff4f79228cb60de031a58f621d8d6adf961e49",\n            "shape": [\n              1,\n              656\n            ]\n          },\n          {\n            "abs_max": 2006.0,\n            "dtype": "torch.int64",\n            "sha256": "c6259a21534aa66ccbf5e42ce93d97b1c312cd821eb0333a2342163ef39a4e35",\n            "shape": [\n              1,\n              1392\n            ]\n          }\n        ],\n        [\n          {\n            "abs_max": 2032.0,\n            "dtype": "torch.int64",\n            "sha256": "6875f919d9c7f175f2d0b192481cae958898f9cef9664b6a08e83158f4942543",\n            "shape": [\n              1,\n              400\n            ]\n          },\n          {\n            "abs_max": 2047.0,\n            "dtype": "torch.int64",\n            "sha256": "dc5ae5ebcb6ec09ed55d245e0a67ebc7a2a11bf64ffbb89f6290210fb8f8bcc8",\n            "shape": [\n              1,\n              1648\n            ]\n          }\n        ]\n      ],\n      [\n        [\n          {\n            "abs_max": 2047.0,\n            "dtype": "torch.int64",\n            "sha256": "0a75d82e582c3075535a601a9371ee09e29d5fd303d1e02cb1eeb3a2939a29cd",\n            "shape": [\n              1,\n              816\n            ]\n          },\n          {\n            "abs_max": 1998.0,\n            "dtype": "torch.int64",\n            "sha256": "7d8235b3daf60c716cb2402147983ad84c0ba2354f905ad987bd63f09baf4d45",\n            "shape": [\n              1,\n              1232\n            ]\n          }\n        ],\n        [\n          {\n            "abs_max": 2034.0,\n            "dtype": "torch.int64",\n            "sha256": "ecad33c4480b39adf4be51b86f7501b9d4ae83b930fd1adedd3e11e6433eb10e",\n            "shape": [\n              1,\n              288\n            ]\n          },\n          {\n            "abs_max": 2047.0,\n            "dtype": "torch.int64",\n            "sha256": "c72767770ac5b7e8f830e353caadbe3560069d45cfb7280388df77e0efe6e9ff",\n            "shape": [\n              1,\n              1760\n            ]\n          }\n        ]\n      ],\n      [\n        [\n          {\n            "abs_max": 2047.0,\n            "dtype": "torch.int64",\n            "sha256": "44714ba687857c5263210a4623a41034252b99e0d6c92d590a0bc29408d3462b",\n            "shape": [\n              1,\n              944\n            ]\n          },\n          {\n            "abs_max": 2041.0,\n            "dtype": "torch.int64",\n            "sha256": "29f45e3716f630ca93fa4c196b9cea8f6d369314ac1a2796e5c2f7ef9e3cc1e9",\n            "shape": [\n              1,\n              1104\n            ]\n          }\n        ],\n        [\n          {\n            "abs_max": 2047.0,\n            "dtype": "torch.int64",\n            "sha256": "283bf46434c78ee5482f4407ddc0959e765eb92d63cdadd2a9274382aeeec29d",\n            "shape": [\n              1,\n              288\n            ]\n          },\n          {\n            "abs_max": 2045.0,\n            "dtype": "torch.int64",\n            "sha256": "e809d9d40b743bbd943b17672f36fde9c4dd221792f651a217e89d3acc30bccf",\n            "shape": [\n              1,\n              1760\n            ]\n          }\n        ]\n      ],\n      [\n        [\n          {\n            "abs_max": 2047.0,\n            "dtype": "torch.int64",\n            "sha256": "3c19dd2767362d53ae7b9f2f90dfaffa1e3850ebdd64689609e3f6e60f004c84",\n            "shape": [\n              1,\n              880\n            ]\n          },\n          {\n            "abs_max": 2039.0,\n            "dtype": "torch.int64",\n            "sha256": "c530589c13b2e541d599c5d9198fa181a80c8b906ff98ceac474314225195fbd",\n            "shape": [\n              1,\n              1168\n            ]\n          }\n        ],\n        [\n          {\n            "abs_max": 2047.0,\n            "dtype": "torch.int64",\n            "sha256": "6e493b8bd855746fe195c88613591725a6200e950b03fd0d5cbfa61ab2b266dc",\n            "shape": [\n              1,\n              512\n            ]\n          },\n          {\n            "abs_max": 2046.0,\n            "dtype": "torch.int64",\n            "sha256": "b81f2e6e4a66ebbe94b6fab7f3d3ff585d2d3c13280b718218ec506a09c9020f",\n            "shape": [\n              1,\n              1536\n            ]\n          }\n        ]\n      ],\n      [\n        [\n          {\n            "abs_max": 2047.0,\n            "dtype": "torch.int64",\n            "sha256": "a794b33755f9c8f81a3fe94c21f0d33f310532ac41b20f575f64e16236dde17a",\n            "shape": [\n              1,\n              1192\n            ]\n          },\n          {\n            "abs_max": 1998.0,\n            "dtype": "torch.int64",\n            "sha256": "9d9815f09cae2bb918ac67e66815382f838afe065ac8f4c5079bb061de82ecef",\n            "shape": [\n              1,\n              856\n            ]\n          }\n        ],\n        [\n          {\n            "abs_max": 2034.0,\n            "dtype": "torch.int64",\n            "sha256": "4a5e5e3f43f1449cd85b3cd072bd2c3f0368d5c94c5839c56fa98c0d74c82197",\n            "shape": [\n              1,\n              496\n            ]\n          },\n          {\n            "abs_max": 2047.0,\n            "dtype": "torch.int64",\n            "sha256": "48fdfe7c1b23195c1dda9ef101e67ed95e7b43fb6ba7e3e057d63ea14c26cb8c",\n            "shape": [\n              1,\n              1552\n            ]\n          }\n        ]\n      ],\n      [\n        [\n          {\n            "abs_max": 2047.0,\n            "dtype": "torch.int64",\n            "sha256": "655e0ee9580df61e419642bd65a1561fd0e168f1917909f26650464e7f464a9c",\n            "shape": [\n              1,\n              1032\n            ]\n          },\n          {\n            "abs_max": 1994.0,\n            "dtype": "torch.int64",\n            "sha256": "5287ed4f1e21999433134f322ae0942754d8ebabc19c0a83f1a7d16264377d2b",\n            "shape": [\n              1,\n              1016\n            ]\n          }\n        ],\n        [\n          {\n            "abs_max": 2047.0,\n            "dtype": "torch.int64",\n            "sha256": "a0a97df05a3990345902465dfbfd537c8dd69f9a0637e67292807e33daa6d991",\n            "shape": [\n              1,\n              488\n            ]\n          },\n          {\n            "abs_max": 2028.0,\n            "dtype": "torch.int64",\n            "sha256": "ce1cca6bda9cb847d216552cfd2fb676c95044ccf00a294cd0e0a9af4c2dfe08",\n            "shape": [\n              1,\n              1560\n            ]\n          }\n        ]\n      ],\n      [\n        [\n          {\n            "abs_max": 2047.0,\n            "dtype": "torch.int64",\n            "sha256": "4e6d1dc4357264ad63a47257fd1bd8f88b2a55df6d3ad140bfc51b7629430fa1",\n            "shape": [\n              1,\n              856\n            ]\n          },\n          {\n            "abs_max": 2046.0,\n            "dtype": "torch.int64",\n            "sha256": "ed6549bf5b7c118de2f864a222e934ef0f2590f26e4a8e149c26ba8096dcc21a",\n            "shape": [\n              1,\n              1192\n            ]\n          }\n        ],\n        [\n          {\n            "abs_max": 2047.0,\n            "dtype": "torch.int64",\n            "sha256": "68fcd1f62a76acbfd19c122776a9792093ed71ac3c9cc076d4493bcdefcf3a24",\n            "shape": [\n              1,\n              256\n            ]\n          },\n          {\n            "abs_max": 2046.0,\n            "dtype": "torch.int64",\n            "sha256": "611dc19bb15b80321262874d45f5734bf4d330125c98a4a95580bfbbaf3104d4",\n            "shape": [\n              1,\n              1792\n            ]\n          }\n        ]\n      ],\n      [\n        [\n          {\n            "abs_max": 2047.0,\n            "dtype": "torch.int64",\n            "sha256": "88d30d225eb7134a18f9001ae9023fd238e6a8d17d805d505fa8efe744d4098d",\n            "shape": [\n              1,\n              1000\n            ]\n          },\n          {\n            "abs_max": 2038.0,\n            "dtype": "torch.int64",\n            "sha256": "37faabfc1f4ad32c6d892de2cf4c2fcf53915d2cef60245b9176ae91a5c38318",\n            "shape": [\n              1,\n              1048\n            ]\n          }\n        ],\n        [\n          {\n            "abs_max": 2047.0,\n            "dtype": "torch.int64",\n            "sha256": "a3e8a7e940ec70e62892a7c1636707015fcf3ce7fa09d985c4adf05d90370e9e",\n            "shape": [\n              1,\n              592\n            ]\n          },\n          {\n            "abs_max": 1999.0,\n            "dtype": "torch.int64",\n            "sha256": "21326f574578c254060243fb52460215ca7a0c6e907c7f53353ba178ef4e738a",\n            "shape": [\n              1,\n              1456\n            ]\n          }\n        ]\n      ]\n    ],\n    "mask_bank_sha256": "5c607050294bd78a8a38876798941146c6a055e0c77425a1130b7036fb969d89",\n    "new_checkpoint_or_dependencies_in_payload": false,\n    "predictor_depth": 12,\n    "report": "report/checks/2026-10-30-eot-work-profile-review.json",\n    "report_sha256": "7cdf7f16051bc5fdd12068275983a2505d6bcd43ecc0d42de9bb9376f1060ac4",\n    "target_computed_original_once_per_fresh_worker_before_adapter": true,\n    "target_encoder_checkpointing": false,\n    "weights_sha256": "cfbc0bc74549bfb01c02e99f4f8a998ebe1851cdf1ca7ded46498567dd2b28a2",\n    "worker_sha256": "8e993374ca3195cdca53bb7f25a708ea4fd9868a9b2597c3a82788126302c9b2"\n  },\n  "research_admitted": false,\n  "resource": {\n    "allocator_bytes": 7516192768,\n    "bootstrap_bytes": 2097152,\n    "cache_bytes": 33554432,\n    "capture_bytes": 2415919104,\n    "client_seconds": 3000,\n    "code_bytes": 134217728,\n    "collection_bytes": 3221225472,\n    "collection_seconds": 600,\n    "emergency_bytes": 16777216,\n    "final_reserve_seconds": 120,\n    "initial_VJPs_per_candidate": 192,\n    "input_bytes": 134217728,\n    "max_files": 1024,\n    "max_simultaneous_scientific_processes": 1,\n    "metadata_bytes": 67108864,\n    "native_call_rows_per_objective": 4096,\n    "predictor_calls_per_objective": 384,\n    "proof_bytes": 2684354560,\n    "proof_seconds": 60,\n    "raw_bytes": 1879048192,\n    "setup_bytes": 3221225472,\n    "setup_seconds": 180,\n    "tensor_file_bytes": 16777216,\n    "trace_bytes": 8388608,\n    "trace_file_bytes": 4194304,\n    "work_seconds": 2400,\n    "worker_seconds": 1920,\n    "workers": 1\n  },\n  "worker_roles": [\n    {\n      "index": 0,\n      "mode": "paired-PGD2-R2",\n      "objectives": 8\n    }\n  ],\n  "workload": {\n    "EOT_draws": 8,\n    "candidate_gradients_max": 4,\n    "completed_clean_cohort_repeats": 0,\n    "epsilon_255": 4,\n    "fixed_test_step_255": 1,\n    "fresh_rollout_replay_in_this_attempt": false,\n    "independent_original_no_grad_scoring_forwards": 2,\n    "mask_families": 2,\n    "original_no_grad_attack_forwards": 4,\n    "random_start": true,\n    "restarts": 2,\n    "same_input_original_gradients_max": 4,\n    "scoring_draws": 4,\n    "scoring_selects_candidate": false,\n    "seeds": {\n      "attack": [\n        865311023795641958,\n        7399188755883649554,\n        867422081190991898,\n        3902520512841466707,\n        6032771523782279135,\n        3779409266305138859,\n        169826197422819580,\n        8283481338430586684\n      ],\n      "score": [\n        6982911954465024005,\n        5302220948187786186,\n        5825470569967192652,\n        8563720553849908607\n      ],\n      "start": 4417571294737150623\n    },\n    "steps": 2,\n    "test_step_is_research_calibration": false,\n    "total_new_gradients_max": 8\n  }\n}\n', 'caps': {'cache': 33554432, 'code': 134217728, 'emergency': 16777216, 'input': 134217728, 'metadata': 67108864, 'proof': 2684354560, 'raw': 1879048192, 'trace': 8388608}, 'capsule_manifest_sha256': 'a7e6740c69f5f3bf8c5fc6b71683296f2c9ebc818b644814b2f37eaf983c0e24', 'design_text': '{\n  "actual_scientific_execution_in_this_step": false,\n  "allocator_bytes": 7516192768,\n  "authorized": false,\n  "automatic_retries": 0,\n  "base_contract_sha256": "c7bb7f3af5e6b9e65ac746177b657ae966e2bc50171a6e5b516e30b848252960",\n  "boundary": {\n    "draw": 0,\n    "family": 0,\n    "native_arrays": [\n      "q",\n      "k",\n      "v",\n      "output",\n      "grad_output",\n      "dq",\n      "dk",\n      "dv"\n    ],\n    "native_forward_row_index": 24,\n    "new_boundary_captures": 1,\n    "original_private_alias_hooks_graph_layout_flags_and_restoration_required": true,\n    "owner": "predictor.0",\n    "phase": "EOT8-before",\n    "source": "new original objective zero only; never historical partial/clean/component arrays",\n    "source_objective_index": 0\n  },\n  "claims": {\n    "all_numerical_native_root_checkpoint_restoration_gates_required": true,\n    "component_VJP_accuracy_qualified": false,\n    "fresh_GPU_rollout_replay_qualified": false,\n    "full_EOT8_kernel_counts_qualified": false,\n    "legacy_worker_or_backend_success_compatible": false,\n    "new_diagnostic_identity_required": true,\n    "performance_or_cost_qualified": false,\n    "representative_backend_only": true,\n    "training_or_research_admitted": false\n  },\n  "cleanup": false,\n  "coverage": {\n    "checkpoint_calls_per_owner": 16,\n    "draws": 8,\n    "families": 2,\n    "initial_math_VJPs_per_candidate": 192,\n    "legacy_full_trace_counts_not_claimed_here": {\n      "candidate_efficient_backward": 384,\n      "candidate_inner_math": 192,\n      "efficient_forward": 1152,\n      "original_efficient_backward": 576\n    },\n    "native_rows_per_objective": 1152,\n    "owners": 36,\n    "predictor_forward_rows_per_candidate": 384,\n    "primary_calls_per_owner": 16\n  },\n  "date": "2026-11-15",\n  "failure_policy": "Stop subsequent work; preserve numerical/provisional/trace evidence; no retry or old claim reset.",\n  "financial_state_unchanged": {\n    "compute_cap_USD": 160,\n    "conservative_encumbrance_USD": 9.8116,\n    "full_account_reconciliation_complete": false,\n    "historical_inclusive_hold_USD": 2,\n    "historical_unassigned_gap_USD": 0.0792,\n    "last_user_reported_wallet_USD": 6.56,\n    "new_paid_hold_USD": 0,\n    "noncompute_reserve_USD": 40,\n    "project_cap_USD": 200,\n    "single_diagnostic_cap_USD": 10,\n    "unreserved_diagnostic_USD": 0.1884\n  },\n  "gradient_accounting": {\n    "CPU_gradient_or_FP64_reference_objectives": 0,\n    "completed_cohort_repeats": 0,\n    "extra_component_VJP_evaluations": 2,\n    "extra_component_inner_math_VJPs": 1,\n    "full_model_objectives": 8,\n    "initial_full_model_inner_math_VJPs": 768,\n    "maximum_total_evaluation_units": 10,\n    "warmups_or_untimed_profile_controls": 0\n  },\n  "held_out_pixels": false,\n  "inventory": {\n    "canonical_result_sha256": "9617cdaaecc0a0b3b18e2a93262150ac0b785124b0988cf58094957a581015ed",\n    "maximum_all_files": 1024,\n    "maximum_extra_mutable_metadata_bytes": 524288,\n    "maximum_metadata_partials": 2,\n    "maximum_simultaneous_raw_partials": 1,\n    "metadata_trace_is_shared_not_additive": true,\n    "reserved_other_files": 256,\n    "result_files": 419,\n    "unknown_result_paths": "refuse"\n  },\n  "kind": "pretrained-pgd-split-profile-local-design-v1",\n  "local_only": true,\n  "new_target_execution_permitted": false,\n  "numerical": {\n    "EOT_draws": 8,\n    "candidate": 4,\n    "families": 2,\n    "fresh_rollout_replay": false,\n    "full_model_gradient_objectives": 8,\n    "no_grad_forwards": 6,\n    "old_targets_masks_seed_selection_projection_and_37_returns_unchanged": true,\n    "profiled_full_model_objectives": 0,\n    "restarts": 2,\n    "same_input_original": 4,\n    "steps": 2\n  },\n  "numerical_bounds": {\n    "input": {\n      "atol": 2e-07,\n      "rtol": 0.002\n    },\n    "loss": {\n      "atol": 2e-06,\n      "rtol": 1e-05\n    }\n  },\n  "old_claims_or_clocks_reset": false,\n  "ordering": [\n    "storage proof",\n    "eight numerical/ownership/restoration gates",\n    "native component",\n    "forward-preserving component",\n    "combined review"\n  ],\n  "outstanding_before_target": [\n    "New hybrid native-forward/math-backward raw-correlation reader, including owned cross-thread and inner-task association.",\n    "New component route must retain exact candidate arithmetic and real root/leaf ownership; old all-math component entry is incompatible.",\n    "New numerical worker/receipts and v2 progress lifecycle must separate provisional capture from detailed profiling, including boundary-observer cleanup.",\n    "Implement/test actual typed inventory, serial partials, source/layout lineage, parent wall/RSS monitor and bounded collection.",\n    "Verify complete source package and combined scope independently before requesting one new exact university authorization."\n  ],\n  "paid_compute": false,\n  "phase_seconds": {\n    "candidate-probe": 120,\n    "finalization": 120,\n    "native-probe": 120,\n    "numerical": 1680,\n    "overhead": 120,\n    "proof": 60,\n    "setup": 180\n  },\n  "private_key_upload": false,\n  "probes": [\n    {\n      "FP64_reference_builds": 0,\n      "component_VJP_accuracy_qualified": false,\n      "correlation_rule": "recorded host/GPU links plus CPU ancestry; cross-thread roots need unique recorded forward-thread/sequence/producer binding",\n      "custom_backward": null,\n      "device": "cuda:0",\n      "dtype": "float32",\n      "efficient_backward_in_backward_phase": 1,\n      "efficient_forward_in_forward_phase": 1,\n      "full_EOT8_backend_coverage_qualified": false,\n      "inner_math_root_and_three_leaf_hooks_required": false,\n      "kernel_name_or_duration_is_identity": false,\n      "math_forward_in_backward_phase": 0,\n      "math_forward_in_forward_phase": 0,\n      "performance_qualified": false,\n      "profiled_VJP_evaluations": 1,\n      "profiled_forward_evaluations": 1,\n      "replay_workers": 0,\n      "required_positive_correlated_kernels": [\n        "efficient-forward",\n        "efficient-backward"\n      ],\n      "role": "native",\n      "shape": [\n        1,\n        12,\n        2048,\n        32\n      ],\n      "unprofiled_iterations": 0,\n      "warmups": 0\n    },\n    {\n      "FP64_reference_builds": 0,\n      "component_VJP_accuracy_qualified": false,\n      "correlation_rule": "recorded host/GPU links plus CPU ancestry; cross-thread roots need unique recorded forward-thread/sequence/producer binding",\n      "custom_backward": "_AttentionBackward",\n      "device": "cuda:0",\n      "dtype": "float32",\n      "efficient_backward_in_backward_phase": 0,\n      "efficient_forward_in_forward_phase": 1,\n      "full_EOT8_backend_coverage_qualified": false,\n      "inner_math_root_and_three_leaf_hooks_required": true,\n      "kernel_name_or_duration_is_identity": false,\n      "math_forward_in_backward_phase": 1,\n      "math_forward_in_forward_phase": 0,\n      "performance_qualified": false,\n      "profiled_VJP_evaluations": 1,\n      "profiled_forward_evaluations": 1,\n      "replay_workers": 0,\n      "required_positive_correlated_kernels": [\n        "efficient-forward",\n        "math-GEMM",\n        "math-softmax-forward",\n        "math-softmax-backward"\n      ],\n      "role": "forward-preserving",\n      "shape": [\n        1,\n        12,\n        2048,\n        32\n      ],\n      "unprofiled_iterations": 0,\n      "warmups": 0\n    }\n  ],\n  "profile": {\n    "RSS_monitor_is_not_address_space_or_instantaneous_allocation_guarantee": true,\n    "RSS_supervisor_required": true,\n    "combined_trace_bytes": 8388608,\n    "component_RSS_ceiling_bytes": 4294967296,\n    "component_wall_seconds": 120,\n    "event_limit_is_not_pre_materialization_memory_bound": true,\n    "options": {\n      "profile_memory": false,\n      "record_shapes": false,\n      "with_stack": false\n    },\n    "processed_profile_events_used": false,\n    "raw_API": "torch.profiler.profile.profiler.kineto_results",\n    "raw_event_cap": 20000,\n    "raw_file_bytes": 4194304,\n    "raw_trace_saved_before_semantic_validation": true\n  },\n  "resource": {\n    "allocated_fit_requires_live_observation": true,\n    "capture_allocation_reserve_bytes": 100663296,\n    "capture_category_caps_bytes": 2264924160,\n    "capture_ceiling_bytes": 2415919104,\n    "collection_seconds": 600,\n    "complete_metadata_including_trace_bytes": 60178432,\n    "complete_raw_bytes": 1835864112,\n    "connection_including_login_seconds": 3000,\n    "metadata_plus_mutable_partials_bytes": 60702720,\n    "proof_allocation_reserve_bytes": 33554432,\n    "proof_category_caps_bytes": 3070230528,\n    "proposed_work_seconds": 2400,\n    "raw_cap_bytes": 1879048192,\n    "raw_plus_one_16MiB_partial_bytes": 1852641328,\n    "result_files": 419,\n    "setup_ceiling_bytes": 3221225472,\n    "shared_metadata_cap_bytes": 67108864,\n    "trace_bytes_included_in_metadata": 8388608,\n    "trace_cap_bytes": 8388608,\n    "work_cap_seconds": 2400\n  },\n  "source_sha256": {\n    "configs/post_gradient_progress_v1.json": "ad2ec9c87022737a91f9db9476581f7dd58449186772dcce857bfb3e1e4c5219",\n    "configs/predictor_pgd_target_v1.json": "c7bb7f3af5e6b9e65ac746177b657ae966e2bc50171a6e5b516e30b848252960",\n    "models/post_gradient_progress_v1.py": "fd7458b29118ca6171dee3a3c99eefc0c10dd2ccd238c230ff1a84fa43a4da1a",\n    "models/predictor_forward_preserving_cpu_v1.py": "6ba308e948c0ce346512f17ef4fc11eccb542954048c859ad6c6a751db1fbb66",\n    "models/predictor_forward_preserving_target_v1.py": "000591f52fd66e7b1d4a78ff0c38ed265a5870e4a8463f9d810031d56752710a",\n    "models/predictor_pgd_progress_adapter_v1.py": "6ec07dda3d71b3f6369f6a6f4cc4b52487ddd653185a459c8bf6230b744ed22e",\n    "models/predictor_pgd_split_plan_v1.py": "cdd2729d2f05782d269a45aa4c22412dcff7b9c5d684267550eb16f59c838ba6",\n    "models/predictor_pgd_target_records_v1.py": "d2b98fe16a0d1efd997d74092be68cbbcd3f2e73eef7fb44c5fcb2d0843204e8",\n    "models/predictor_pgd_target_worker_v1.py": "0eb5b56e6e3f74bfdc53e4f1f0f5861e0f066340961fbbfc2b134cc75536f3a0",\n    "models/predictor_sdpa_component_events_v2.py": "ebde9c0f32078979cfa5920503004bb79ca697fdc05d7adacbfc59bde41f515a",\n    "models/predictor_sdpa_launch_evidence_v1.py": "c9749407f3d237050252ec7e9d4831c901df813a35b3b4a38ad8d693e0e1899f",\n    "models/predictor_sdpa_launch_evidence_v2.py": "05f99a96ca34d65c4167cdbd37f97600b14468aa08fbfe5aa9b581b72e5fdddb",\n    "models/predictor_sdpa_observer_v1.py": "5e6e00593f3befb1979b37f0fec155bb35322c9e4c196a7abe968d2f0ff68047",\n    "models/predictor_sdpa_raw_capture_v1.py": "6da98072ac5f38e1939f0d470970b7f68b6d63b7d475c23ac5d4d3e60c6ac9e4",\n    "tests/test_predictor_pgd_split_plan_v1.py": "d1849260298e111cdbe086186ce5300ba35444cefacae8dc14d2b769c2d74c77"\n  },\n  "target_runner_available": false\n}\n', 'fixed_support': {'predictor_pgd_split_watch_v1.py': '07237d77806aba1c129e0e93520a2ceedd3ff2b06001f391d1cdaa46e55fae06', 'scratch_allocation_v2.py': '6b0503b8ab2c88a44a4d95c8fafbad52d1c0414293379406520a22f67700e874'}, 'packet': {'archive': {'bytes': 27279360, 'sha256': '82a30efedfb0e85fd39c634c5bd0a8b82bd839b91bc7259841e2d517d495011e'}, 'files': {'MANIFEST.json': {'bytes': 13480, 'sha256': 'a7e6740c69f5f3bf8c5fc6b71683296f2c9ebc818b644814b2f37eaf983c0e24'}, 'boundary/attention-calls.jsonl': {'bytes': 949546, 'sha256': '6f2ccd095b107eeaf59ba872be04de80abdfe348f4495243f1c8f19da293e126'}, 'boundary/attention-scopes.json': {'bytes': 312, 'sha256': 'fe85b6c1e54097cae93a1891d9ced17f4694f8de4f4754cb398fd1b33ab9d043'}, 'boundary/boundary-complete.json': {'bytes': 273, 'sha256': '8a7233c4138b87eca2b16b3e28c6412d2d664bd221c50cb494d8569e89bf524e'}, 'boundary/boundary.json': {'bytes': 9234, 'sha256': '9e4758d2209b9fd989926ee40d2ad92cc4b99aa98c86e6cec0722eaf6a83308d'}, 'boundary/dk.pt': {'bytes': 3147305, 'sha256': '00288e2c75abd4eef65a1e97b00f4fe25546ab20df640702375a5d7d2d15bf8c'}, 'boundary/dq.pt': {'bytes': 3147305, 'sha256': '6973339c06e07ba92e2c79f04aee4bbb982a3642cc4895d6c974c30840c15b09'}, 'boundary/dv.pt': {'bytes': 3147305, 'sha256': 'deead61cbb8cf5b6f7462219907082b6cc17db99af67734e170d6c3b35d9ffa5'}, 'boundary/grad_output.pt': {'bytes': 3147305, 'sha256': 'a673aabb28ca9e181c2426ee4ce81062b1fcecdc0609d31754996efdf45fcb7c'}, 'boundary/k.pt': {'bytes': 3147305, 'sha256': 'f80ea46c5e6bea6b117763c3551a5320b92aa45f65916d2c451553d2083aad37'}, 'boundary/output.pt': {'bytes': 3147305, 'sha256': '33cf9f63b54b285ee5c6fa1ed3dac252f7edcb40c4f5aa3a9d574ba686c8209e'}, 'boundary/q.pt': {'bytes': 3147305, 'sha256': 'eede3a77598c212faa77ca935e61622aa698fd4ffdad658e717cd80fed968866'}, 'boundary/v.pt': {'bytes': 3147305, 'sha256': '8881fa0dcbc8b653e5547ca2ef5533e3bfb436cfdabfe5025dd437792a7e30da'}, 'provenance/accepted-numerical-review.json': {'bytes': 7914, 'sha256': '13d2a40d2a989d9433f042dcf59187f84850ea8294e4ad8f494accda00074c8b'}, 'provenance/objective-0/VJP-calls.jsonl': {'bytes': 0, 'sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'}, 'provenance/objective-0/native-calls.jsonl': {'bytes': 949546, 'sha256': '6f2ccd095b107eeaf59ba872be04de80abdfe348f4495243f1c8f19da293e126'}, 'provenance/objective-0/numerical-complete.json': {'bytes': 5105, 'sha256': '9c688ffa5a5e34c0c3a2e6d13fb2b74780b443e2d893b35e52db045f9c677cbd'}, 'provenance/objective-0/numerical.json': {'bytes': 9661, 'sha256': '348f5ee5843a42a63c924ce19944228ff1cb9aed4bf288876de6a7298d8921e2'}, 'provenance/objective-0/scopes.json': {'bytes': 328, 'sha256': '539deace994dc1809bd5265c69be4a91d037f58440c73606aaf658e8b18d0b58'}, 'provenance/objective-1/numerical-complete.json': {'bytes': 5111, 'sha256': '22761fc13f36d293356e6662d7157adbb308f50fc1cbc251518e9020041d5e83'}, 'provenance/objective-1/numerical.json': {'bytes': 10130, 'sha256': '8af887389465c399cfb0e1b1bcf5b9e24a4fe2431c2cfad0cccf49838bdea1dc'}, 'provenance/objective-2/numerical-complete.json': {'bytes': 5105, 'sha256': 'b7077961078735f1407b8ce265ae9507ba33719ed186b606e71030431fc46fa9'}, 'provenance/objective-2/numerical.json': {'bytes': 9661, 'sha256': '8f8dbd56c57b7d8ed5b1e8eff6b62a95c880a3151c1b1fa0237d457200fe0a66'}, 'provenance/objective-3/numerical-complete.json': {'bytes': 5111, 'sha256': 'fafb91946d2ca6b12b241bbde81ca9f4089eb8996cd12f1a86661d978e822908'}, 'provenance/objective-3/numerical.json': {'bytes': 10131, 'sha256': '183023510653f86119b51811e1403cca9a7f1aa3db588afd444fe1592367a624'}, 'provenance/objective-4/numerical-complete.json': {'bytes': 5105, 'sha256': '9a5a3fef8ce4ba4bdd961215774ae235e2fc2f09cd7a4272abff5c8e1ee06ef8'}, 'provenance/objective-4/numerical.json': {'bytes': 9661, 'sha256': 'aa9fba339232bab8fac9137bd16c38e5cea4d6a63bef95f8f82a9bff8b59fc5d'}, 'provenance/objective-5/numerical-complete.json': {'bytes': 5111, 'sha256': 'bca25f28404c198eadeaa0471eba65e1499c153269a3936e154f641ec7f7ef5c'}, 'provenance/objective-5/numerical.json': {'bytes': 10130, 'sha256': 'b27fc9fe8d08935d1b954350a39865a0b5250d8f42a022176f848a26c22cb376'}, 'provenance/objective-6/numerical-complete.json': {'bytes': 5105, 'sha256': '277e8292d0d478482ba5385a5fbd722720f663988352624e3fc3686be5779119'}, 'provenance/objective-6/numerical.json': {'bytes': 9661, 'sha256': 'f9f58ec5ed398318f7ae32fb59e465afa288c6c38b9347904e088f81c01df7d3'}, 'provenance/objective-7/numerical-complete.json': {'bytes': 5111, 'sha256': '812ed32dda099acf9b175d6c8e11613e02357dbae93ff16e2905bb5bb7eb465e'}, 'provenance/objective-7/numerical.json': {'bytes': 10131, 'sha256': 'e260795efb3902fce97b860e5f90f58dc8b95a29f2d2f0e71cc51e81adf4be10'}, 'provenance/parent-numerical.json': {'bytes': 481, 'sha256': 'c422f69479a94734b9ec7c3d3065335d0c53bd440907935a79cc51af43856761'}, 'provenance/run-status.json': {'bytes': 13337, 'sha256': '56a6f931c8c9a4eb5367ef85244f497d8584eccd484a004190d1c459c2b5553d'}, 'provenance/session.json': {'bytes': 5177, 'sha256': '6cd4d8b18cd2a542ffb6c7c1285611e2ff7ab2cae3cc2fa7c53c248e1ec68697'}}}, 'result': {'logs/forward-preserving.log': {'category': 'metadata', 'max_bytes': 524288, 'mutable': False, 'partial_allowed': False}, 'results/collection-status.json': {'category': 'metadata', 'max_bytes': 131072, 'mutable': True, 'partial_allowed': True}, 'results/components/forward-preserving/backend.json': {'category': 'metadata', 'max_bytes': 131072, 'mutable': False, 'partial_allowed': True}, 'results/components/forward-preserving/complete.json': {'category': 'metadata', 'max_bytes': 16384, 'mutable': False, 'partial_allowed': True}, 'results/components/forward-preserving/dk.npy': {'category': 'raw', 'max_bytes': 4194304, 'mutable': False, 'partial_allowed': True}, 'results/components/forward-preserving/dq.npy': {'category': 'raw', 'max_bytes': 4194304, 'mutable': False, 'partial_allowed': True}, 'results/components/forward-preserving/dv.npy': {'category': 'raw', 'max_bytes': 4194304, 'mutable': False, 'partial_allowed': True}, 'results/components/forward-preserving/events.json': {'category': 'trace', 'max_bytes': 4194304, 'mutable': False, 'partial_allowed': True}, 'results/components/forward-preserving/failed.json': {'category': 'metadata', 'max_bytes': 16384, 'mutable': False, 'partial_allowed': True}, 'results/components/forward-preserving/output.npy': {'category': 'raw', 'max_bytes': 4194304, 'mutable': False, 'partial_allowed': True}, 'results/components/forward-preserving/ownership.json': {'category': 'metadata', 'max_bytes': 65536, 'mutable': False, 'partial_allowed': True}, 'results/components/forward-preserving/progress.jsonl': {'category': 'metadata', 'max_bytes': 65536, 'mutable': False, 'partial_allowed': False}, 'results/components/forward-preserving/resource.json': {'category': 'metadata', 'max_bytes': 16384, 'mutable': False, 'partial_allowed': True}, 'results/components/forward-preserving/trace-manifest.json': {'category': 'metadata', 'max_bytes': 16384, 'mutable': False, 'partial_allowed': True}, 'results/components/forward-preserving/worker.json': {'category': 'metadata', 'max_bytes': 262144, 'mutable': True, 'partial_allowed': True}, 'results/final-receipt.json': {'category': 'metadata', 'max_bytes': 131072, 'mutable': False, 'partial_allowed': True}, 'results/components/forward-preserving/accuracy.json': {'category': 'metadata', 'max_bytes': 16384, 'mutable': False, 'partial_allowed': True}}, 'root_metadata': {'allocation-events-v2.jsonl': [1048576, True, False], 'allocation-result-v2.json': [16384, False, False], 'authorization.json': [65536, False, True], 'control.json': [262144, False, True], 'gpu-samples.jsonl': [524288, True, False], 'installation.json': [65536, False, True], 'parent-forward-preserving.json': [131072, False, True], 'parent-proof.json': [131072, False, True], 'proof.log': [524288, False, False], 'proposal.json': [262144, False, True], 'runtime-source-manifest.json': [131072, False, True], 'session.json': [131072, True, True], 'storage-before.json': [16384, False, True], 'storage-status.json': [65536, True, True], 'worker-plan-forward-preserving.json': [16384, False, True], 'worker-plan-proof.json': [16384, False, True]}, 'runtime_files': {'attacks/__init__.py': {'bytes': 69, 'sha256': '41a090d550527278b09bfb62fbed810805b379151b7a5ab7cb5469bc12ee01cb'}, 'attacks/pgd.py': {'bytes': 5089, 'sha256': '947c3e4aa275fa98b5f84a2a7a89f74b8161d6ba498b948862f7d488712c2d81'}, 'models/__init__.py': {'bytes': 86, 'sha256': '1d079e9ff2633dfad73abff31961c9e81b7aff65e44ec6bedc60a32189630ff5'}, 'models/_pgd_component_contract_v1.py': {'bytes': 1188, 'sha256': '918854511a73fda012e1aad0703821a9e9793c5af62d3ede8b137e9f9a5f8401'}, 'models/_pgd_component_entry_v1.py': {'bytes': 5914, 'sha256': 'c59538e1114d0ee3337c161d0365571963785ea0c445709853e477cbb028ee9f'}, 'models/_pgd_component_io_v1.py': {'bytes': 32614, 'sha256': '36d70563fb2e14618a5f052e09b21b3aee089f894e17f55efd24c2c6211e4265'}, 'models/_pgd_component_job_v1.py': {'bytes': 27999, 'sha256': 'a8820eacf15c6e64c327ec85c3514bc91aa416f3a69a954645df7f658c50b180'}, 'models/_pgd_component_providers_v1.py': {'bytes': 4036, 'sha256': '57636cb1935b64576eaffc8c2015fd091ebd41b8b4efec2d2a3933ea94de380c'}, 'models/_pgd_component_source_v1.py': {'bytes': 32731, 'sha256': 'ba2c71deae006d6c3d1e9e523c4b6c4ef8f3aa14257fdc27cb88a90b903af940'}, 'models/_pgd_component_store_v1.py': {'bytes': 10349, 'sha256': '838785328996a0377492082aae2eb80a0cffcd89ab5247685cf29f39bc2631bd'}, 'models/_pgd_split_boundary_v1.py': {'bytes': 17495, 'sha256': '95237713919109f3e2c7d1ae4a3acdeb3967b1f2ef703ae675ae1489938d0980'}, 'models/_pgd_split_raw_v1.py': {'bytes': 4174, 'sha256': '78c9fcd2244d756e6c1aa5ad73c6c9f2907e27f6074156a06a5fec2a746c8663'}, 'models/_pgd_split_route_v1.py': {'bytes': 25865, 'sha256': 'fccad384e34147b615ca37a67f5c5ace9424270710f9b65a15977c30e44ca203'}, 'models/_pgd_split_startup_v1.py': {'bytes': 3449, 'sha256': '2deac57c321daeb7fcbc1979322a2f6eb50dbc7c852e19a562afd41937f84b68'}, 'models/checkpointing.py': {'bytes': 6645, 'sha256': 'edc5940bc73db42c89b3494be9b0f2098680762f4fa972305f51abe47c3d8373'}, 'models/diagnostics.py': {'bytes': 5555, 'sha256': '12185169ca1ebe2564bc0b1815ec19b8599c08f625af056bc30c25fca6f390d0'}, 'models/execution.py': {'bytes': 12043, 'sha256': 'e665f7c842c0d26733bd6c4d16887b10e2590b76f15e4fb1981b304ff37903ba'}, 'models/native_patch.py': {'bytes': 3250, 'sha256': '8f5ede9ea8ba61773ca11dffac5a2bdba01a62affc497d8327b3e1ede31fa5e7'}, 'models/patch_embed.py': {'bytes': 1827, 'sha256': '64838d575479e067ffecb8352efa7b51fcfafeabeb234e295605d87c1033b0ec'}, 'models/pilot.py': {'bytes': 2119, 'sha256': 'a8e947c5ce58c6497b653b3bdc9ac1051b045d07fe3f280f9c04daf44858ae82'}, 'models/precision_patch.py': {'bytes': 7991, 'sha256': '30e8a6b28b8f082fd771aaeab5a70a73cf200714ef2b24416a55938f61e9c550'}, 'models/predictor_eot8_adapter_v1.py': {'bytes': 7839, 'sha256': '89b7e0c67e88f2f776bb0bb6ad786a2202115e64ffc6c6f9a962381cc2e9a6b8'}, 'models/predictor_eot8_capture_v1.py': {'bytes': 2194, 'sha256': 'fcf582f0152f9ad1d93d1bf038895ba6c6c84d75a4fdfc02d5d4b2ea96e0d3e4'}, 'models/predictor_eot8_owned_receipt_v1.py': {'bytes': 4599, 'sha256': '55005fd96e994c660e693e0347a337d01826eaf5224ecf6f38329a8905538531'}, 'models/predictor_eot8_record_v2.py': {'bytes': 14128, 'sha256': 'c75455b9a08f1b25bbb7269d4ae3c1fa34826656ba1800c0a45fdf9433765dd9'}, 'models/predictor_pgd_candidate_accuracy_v1.py': {'bytes': 7105, 'sha256': '9e8bac5b01bcb9721218bc28f499ad6484a6e05396ea6ae383b69eb549c1d13e'}, 'models/predictor_pgd_candidate_reference_v1.py': {'bytes': 15140, 'sha256': 'e7933921c0143ff7b7e779c703da9062c1c954ec333a79f426ae6d1ecf6b2c13'}, 'models/predictor_pgd_component_capsule_v1.py': {'bytes': 14106, 'sha256': '453dea53b4e33b2187b8ffc248546aaba9c011e535d11dbc285a3c4fd15dcd37'}, 'models/predictor_pgd_split_backend_v4.py': {'bytes': 24232, 'sha256': '0a5ef4a5dc99b55eb5df6a3af8d11f99dbb18352b4ffab55c310a3f5a0f62622'}, 'models/predictor_pgd_split_lineage_v1.py': {'bytes': 8431, 'sha256': 'a53103bfc665b30930af2b6046a347f81369b0de3a1b72ff2743bc5f3c643647'}, 'models/predictor_pgd_split_plan_v1.py': {'bytes': 15697, 'sha256': 'cdd2729d2f05782d269a45aa4c22412dcff7b9c5d684267550eb16f59c838ba6'}, 'models/predictor_sdpa_component_events_v1.py': {'bytes': 8268, 'sha256': 'c1397e06ca89d1299a7d853e6c88124d1cc7874aae3f5763f018fbc67bd195fc'}, 'models/predictor_sdpa_component_events_v2.py': {'bytes': 15023, 'sha256': 'ebde9c0f32078979cfa5920503004bb79ca697fdc05d7adacbfc59bde41f515a'}, 'models/predictor_sdpa_launch_evidence_v1.py': {'bytes': 17552, 'sha256': 'c9749407f3d237050252ec7e9d4831c901df813a35b3b4a38ad8d693e0e1899f'}, 'models/predictor_sdpa_launch_evidence_v2.py': {'bytes': 19485, 'sha256': '05f99a96ca34d65c4167cdbd37f97600b14468aa08fbfe5aa9b581b72e5fdddb'}, 'models/predictor_sdpa_launch_evidence_v3.py': {'bytes': 19602, 'sha256': 'd8ca7092b346d3e1ccd9c43a99a4379eba21914a08ced173b692148015010675'}, 'models/runtime.py': {'bytes': 4827, 'sha256': '0764f5c63384882e06f1d409535e96b3423861866c16bc8ba5e3fa71955f83de'}, 'models/training.py': {'bytes': 16803, 'sha256': '15e280dfcc409a94f36defa42c59304e96dc198c8e4e063a9da6f4adf4f03679'}, 'models/vjepa2.py': {'bytes': 12926, 'sha256': '5501eb4ce79317d8ebab591d7fe4d8d0b74b1b5b692d2f7513d8c583d188debf'}}, 'scope': {'GPU_bytes': 7516192768, 'capture_bytes': 2415919104, 'collection_seconds': 600, 'completed_objectives_repeated': 0, 'component_RSS_bytes': 4294967296, 'component_seconds': 120, 'components': ['forward-preserving'], 'connection_seconds': 3000, 'design_sha256': 'c5daae30cbe4043cf4d5265de2a2a7420d015ab1e244cc3d74fcc93b5f0186af', 'financial_state_unchanged': True, 'kind': 'candidate-only-source-interface-not-a-grant', 'new_full_model_objectives': 0, 'proof_seconds': 60, 'response_bytes': 3221225472, 'setup_bytes': 3221225472, 'work_seconds': 2400, 'native_component_reruns': 0, 'reference_manifest_sha256': '3dc2484c07c04bb72593c2ee386824edf52b8d80ec0fe3bb803e645714e5d932'}, 'supervisor_names': ['bootstrap.py', 'predictor_pgd_split_scratch_v1.py', 'predictor_pgd_split_coordinator_v1.py', 'predictor_pgd_split_collection_v1.py', 'predictor_pgd_split_watch_v1.py', 'scratch_allocation_v2.py'], 'reference': {'archive': {'bytes': 13281280, 'sha256': 'dbaee50b765273f530976e9b8a8301ee055f4e2dc76b006239b4864b306b7d75'}, 'files': {'native/backend.json': {'bytes': 3268, 'sha256': '4bd060d01524a8381c0dc978fd0ff248d3f1e25742df0c581b40ef47ddaa9bc5'}, 'native/complete.json': {'bytes': 632, 'sha256': '51407b1c1569e6e870695b0f2ddaa6d79bb0a20a69b3d59ee3181b84e91a6b98'}, 'native/dk.npy': {'bytes': 3145856, 'sha256': 'c31f6d5017e237d9488ad8ba96f745ddbcf1e5e21eb5224d618c7998f92fb49b'}, 'native/dq.npy': {'bytes': 3145856, 'sha256': '218dc436f9795ef1014e910c990534eb75a06676b28145595964f0653d864afb'}, 'native/dv.npy': {'bytes': 3145856, 'sha256': '46e728741ea905d43a9ebb3e0ea44f9bd7e0bf7750b6b5641e81b8f6e4d09134'}, 'native/events.json': {'bytes': 569123, 'sha256': '97139d0db676bf14e2e14a0b294611687cec3e0860279e4755067c1c13b2c916'}, 'native/output.npy': {'bytes': 3145856, 'sha256': '47f0ac9fbba30e14e9b82011f062596c32c05c6fa152aa89d5c1168e73b1c618'}, 'native/ownership.json': {'bytes': 2845, 'sha256': '2bae7b2feb0c26f3238db3b07dfbee04114e2f5a77ebc7b191e3aa24f7cf2536'}, 'native/parent.json': {'bytes': 435, 'sha256': 'cc57544291e2896af1996a1c05e20715fa9bde424c8b705c84c1a067198cb12e'}, 'native/progress.jsonl': {'bytes': 29650, 'sha256': 'e13db72771087f38fb6cdbc6e7fa87a70f3b2fb03b80d35739ae0c0ecb82a134'}, 'native/resource.json': {'bytes': 1197, 'sha256': 'aa5917e82ed385bd748fa878cf97e30a3b163f2b6dc7b49c248396bd03e9aa36'}, 'native/worker.json': {'bytes': 6329, 'sha256': '6f1cdea1d114cc605103c645cd379f3c12867ae88855c211e97d0fd26e3fea29'}, 'reviews/component.json': {'bytes': 9305, 'sha256': '45318db20a419f3f2671be590b3b3fe218f35a2a2ca0f935a37de727eceaa3f2'}, 'reviews/metadata.json': {'bytes': 8346, 'sha256': '332667b3c9a6902464f75a920fb8c45bed5c37fc0387e4e9e45554de257dc56b'}, 'reviews/pgd.json': {'bytes': 7914, 'sha256': '13d2a40d2a989d9433f042dcf59187f84850ea8294e4ad8f494accda00074c8b'}, 'reviews/values.json': {'bytes': 7963, 'sha256': '1c632f09f34e05e37516df19c27d855780738c1208c42e3691890665f9622b53'}, 'source/boundary.json': {'bytes': 9234, 'sha256': '9e4758d2209b9fd989926ee40d2ad92cc4b99aa98c86e6cec0722eaf6a83308d'}, 'source/capsule-manifest.json': {'bytes': 13480, 'sha256': 'a7e6740c69f5f3bf8c5fc6b71683296f2c9ebc818b644814b2f37eaf983c0e24'}, 'MANIFEST.json': {'bytes': 3289, 'sha256': '3dc2484c07c04bb72593c2ee386824edf52b8d80ec0fe3bb803e645714e5d932'}}, 'manifest_sha256': '3dc2484c07c04bb72593c2ee386824edf52b8d80ec0fe3bb803e645714e5d932'}, 'source_hashes': {'numerical_sha256': '56a6f931c8c9a4eb5367ef85244f497d8584eccd484a004190d1c459c2b5553d', 'original_sha256': '348f5ee5843a42a63c924ce19944228ff1cb9aed4bf288876de6a7298d8921e2', 'boundary_sha256': '9e4758d2209b9fd989926ee40d2ad92cc4b99aa98c86e6cec0722eaf6a83308d', 'reference_manifest_sha256': '3dc2484c07c04bb72593c2ee386824edf52b8d80ec0fe3bb803e645714e5d932'}, 'reference_manifest_sha256': '3dc2484c07c04bb72593c2ee386824edf52b8d80ec0fe3bb803e645714e5d932'}

def static_contract(supervisors):
    assets = dict(STATIC['assets'])
    for name, d in supervisors.items():
        assets['supervisor/' + name] = dict(d)
    return {'kind': 'candidate-only-whole-scratch-v1', 'assets': assets,
            'runtime_files': STATIC['runtime_files'], 'result': STATIC['result'],
            'root_metadata': STATIC['root_metadata'], 'caps': STATIC['caps'],
            'packet': STATIC['packet'], 'reference': STATIC['reference']}

class Layout(_OriginalLayout):

    def __init__(self, contract_bytes, base_bytes, design_bytes, installation_bytes, installation_sha256):
        c = parse_json(contract_bytes)
        install = parse_json(installation_bytes)
        if base_bytes.decode() != STATIC['base_text'] or design_bytes.decode() != STATIC['design_text'] or encoded(c) != contract_bytes or (install.get('kind') != 'pgd-split-scratch-installation-v1') or (hashlib.sha256(installation_bytes).hexdigest() != installation_sha256):
            raise ValueError('Exact component layout/config/source installation required.')
        files = install.get('files')
        if type(files) is not dict or set(files) != set(STATIC['supervisor_names']) or c != static_contract(files):
            raise ValueError('Complete exact component catalogue and supervisor source set required.')
        self.contract = c
        self.result = c['result']
        self.assets = dict(c['assets'])
        self.installation = install
        self.installation_sha256 = installation_sha256
        self.assets['installation.json'] = {'bytes': len(installation_bytes), 'sha256': installation_sha256}
        self.directories = {'.', 'capture', 'cache', 'cache/cuda', 'tmp'}
        for name in (*self.assets, *STATIC['root_metadata'], *('capture/' + n for n in self.result)):
            self.directories.update((str(p) for p in PurePosixPath(name).parents))

    def classify(self, name):
        safe(name)
        if name.startswith('capture/'):
            d, plain, partial = plan.classify(name[8:], self.result)
            return dict(d, plain='capture/' + plain, partial=partial, space='result')
        if name in ('space-proof-v2.bin.partial', 'emergency-space.bin'):
            proof = name.startswith('space-proof')
            return dict(category='proof' if proof else 'emergency', max_bytes=5 * GiB // 2 if proof else 16 * MiB, plain=name, partial=False, mutable=False, partial_allowed=False, space='proof' if proof else 'emergency')
        if name.startswith(('cache/', 'tmp/')):
            if len(PurePosixPath(name).parts) > 8:
                raise ValueError('Cache directory depth ceiling.')
            return dict(category='cache', max_bytes=32 * MiB, plain=name, partial=False, mutable=True, partial_allowed=False, space='cache')
        partial = name.endswith('.partial')
        plain = name.removesuffix('.partial')
        if plain in STATIC['root_metadata']:
            cap, mutable, allow = STATIC['root_metadata'][plain]
            if partial and (not allow):
                raise ValueError('Disallowed supervisor partial.')
            return dict(category='metadata', max_bytes=cap, plain=plain, partial=partial, mutable=mutable, partial_allowed=allow, space='metadata')
        if name in self.assets or (partial and plain in ('component-inputs.tar', 'native-reference.tar')):
            d = self.assets[plain]
            return dict(category='input' if plain in ('component-inputs.tar', 'native-reference.tar') or plain.startswith(('source/', 'reference/')) else 'code', max_bytes=d['bytes'], plain=plain, partial=partial, mutable=False, partial_allowed=plain in ('component-inputs.tar', 'native-reference.tar'), space='asset')
        raise ValueError('Unknown component scratch path: ' + name)
