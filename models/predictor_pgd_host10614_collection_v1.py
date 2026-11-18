"""Bounded evidence stream definitions; no network, process, science or CLI API.

The future outer driver must enforce blocking-I/O alarms and single-connection
claims. Current checks use tiny ordinary files and inert byte streams only.
"""
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import time
from models import predictor_pgd_host10614_bootstrap_v1 as boot
from models import predictor_pgd_host10614_scratch_v1 as scratch

MiB, GiB = 2**20, 2**30
MAGIC = b'VJEPA-PGD-SPLIT-EVIDENCE-1\n'
END = b'VJEPA-PGD-SPLIT-END-1\n'
HEADER_CAP, TRAILER_CAP, WIRE_CAP = MiB, 16*1024, 3*GiB
RECEIPT = 'capture/results/final-receipt.json'
STATUS = 'capture/results/collection-status.json'
FALSE_FLAGS = {'scientific_qualified': False, 'training_admitted': False}


class Window:
    """One continuing clock; cooperative checks, not a blocking-I/O interrupt."""
    def __init__(self, connection_started, collection_started, *, clock=time.monotonic):
        now = clock()
        if (not all(boot.finite(x) for x in (connection_started, collection_started, now))
                or not connection_started <= collection_started <= now):
            raise ValueError('Original finite connection and collection origins required.')
        self.clock = clock
        self.connection_started, self.started = connection_started, collection_started
        self.deadline = min(connection_started+3000, collection_started+600)
        self.last = now
        self.check()

    def check(self, deadline=None):
        now = self.clock()
        limit = self.deadline if deadline is None else min(self.deadline, deadline)
        if not boot.finite(now) or now < self.last or now >= limit:
            raise TimeoutError('Original evidence collection clock exhausted or changed; preserve partial bytes.')
        self.last = now
        return now


def excluded(name):
    # Contents are never collected. Their exact metadata still enters inventory.
    return name == 'emergency-space.bin' or name.startswith(('cache/', 'tmp/'))


def _row(info):
    return {'identity': (*scratch.directory_id(info), info.st_nlink),
            'bytes': info.st_size, 'allocated_bytes': info.st_blocks*512, 'mtime_ns': info.st_mtime_ns}


class FileReader:
    """Held/no-follow source streaming with the snapshot's exact identities."""
    def __init__(self, guard, name, expected, window):
        scratch.safe(name)
        self.guard, self.name, self.expected, self.window = guard, name, expected, window

    def chunks(self):
        g = self.guard
        self.window.check()
        parts = PurePosixPath(self.name).parts
        g.layout.classify(self.name)
        fd = g.open_root()
        held = []
        filefd = None
        try:
            for part in parts[:-1]:
                before = os.stat(part, dir_fd=fd, follow_symlinks=False)
                if (not stat.S_ISDIR(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o700
                        or before.st_uid != os.getuid() or before.st_dev != g.binding['root_identity'][0]):
                    raise ValueError('Private same-device stream ancestry required.')
                child = os.open(part, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=fd)
                if scratch.directory_id(os.fstat(child)) != scratch.directory_id(before):
                    os.close(child)
                    raise ValueError('Held stream ancestor changed.')
                held.append((fd, part, before))
                fd = child
            before = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode) or _row(before) != self.expected:
                raise ValueError('Only the exact scanned regular file may be streamed.')
            filefd = os.open(parts[-1], os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK, dir_fd=fd)
            if not stat.S_ISREG(os.fstat(filefd).st_mode) or _row(os.fstat(filefd)) != self.expected:
                raise ValueError('Held stream source changed before read.')
            remaining = before.st_size
            while remaining:
                self.window.check()
                data = os.read(filefd, min(MiB, remaining))
                if not data:
                    raise ValueError('Source truncated during stream; preserve partial response.')
                remaining -= len(data)
                yield data
                self.window.check()
            if os.read(filefd, 1) or _row(os.fstat(filefd)) != self.expected:
                raise ValueError('Stream source grew or changed.')
            if _row(os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)) != self.expected:
                raise ValueError('Named stream source changed.')
            for parent, part, before in held:
                if scratch.directory_id(os.stat(part, dir_fd=parent, follow_symlinks=False)) != scratch.directory_id(before):
                    raise ValueError('Named stream ancestry changed.')
            g.check_root()
        finally:
            if filefd is not None:
                os.close(filefd)
            os.close(fd)
            for parent, _, _ in reversed(held):
                os.close(parent)


def publish(guard, name, value):
    """Exclusive bounded metadata only; no replacement, retry or archive path."""
    d = guard.layout.classify(name)
    data = boot.encoded(value)
    if name not in (RECEIPT, STATUS) or d['category'] != 'metadata' or len(data) > d['max_bytes']:
        raise ValueError('Only the two declared bounded collection records may be published.')
    guard.scan(phase='collection')
    parts = PurePosixPath(name).parts
    fd = guard.open_root()
    parents = []
    try:
        for part in parts[:-1]:
            before = os.stat(part, dir_fd=fd, follow_symlinks=False)
            if (not stat.S_ISDIR(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o700
                    or before.st_uid != os.getuid() or before.st_dev != guard.binding['root_identity'][0]):
                raise ValueError('Private publication ancestor required.')
            child = os.open(part, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=fd)
            if scratch.directory_id(os.fstat(child)) != scratch.directory_id(before):
                os.close(child)
                raise ValueError('Publication ancestor changed.')
            parents.append((fd, part, before))
            fd = child
        leaf = parts[-1]
        with os.fdopen(os.open(leaf+'.partial', os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600, dir_fd=fd), 'wb') as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.link(leaf+'.partial', leaf, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
        # Only this successful publication's own temporary name is removed.
        os.unlink(leaf+'.partial', dir_fd=fd)
        os.fsync(fd)
        for parent, part, before in parents:
            if scratch.directory_id(os.stat(part, dir_fd=parent, follow_symlinks=False)) != scratch.directory_id(before):
                raise ValueError('Publication ancestry changed.')
        guard.check_root()
    finally:
        os.close(fd)
        for parent, _, _ in reversed(parents):
            os.close(parent)
    return boot.digest(data)


def work_receipt(session_bytes, control_sha256, authorization_sha256, installation_sha256, window):
    s = boot.parsed(session_bytes)
    if (type(s) is not dict or s.get('kind') != 'pgd-split-parent-session-v1'
            or s.get('control_sha256') != control_sha256 or s.get('parent_pid') != os.getpid()
            or s.get('phase') not in ('await-collection', 'failed') or s.get('complete') is not False
            or s.get('connection_started') != window.connection_started
            or s.get('work_deadline') != window.connection_started+2400
            or s.get('connection_deadline') != window.connection_started+3000
            or any(s.get(k) is not False for k in FALSE_FLAGS)):
        raise ValueError('One owned ended-work session on the original connection clock is required.')
    roles, receipts = s.get('completed'), s.get('receipts')
    if (type(roles) is not list or roles != list(boot.ROLES[:len(roles)]) or len(roles) > 4
            or type(receipts) is not list or len(receipts) != len(roles)):
        raise ValueError('Only the original serial completed prefix may enter final receipts.')
    joined = []
    identities = set()
    for role, r in zip(roles, receipts):
        if (type(r) is not dict or r.get('role') != role or r.get('saved_gates_returned') is not True
                or not boot.positive(r.get('pid')) or not boot.positive(r.get('start_ticks'))
                or any(r.get(k) is not False for k in FALSE_FLAGS) or r['pid'] in identities):
            raise ValueError('Distinct completed owned role receipts required; no promotion of partials.')
        identities.add(r['pid'])
        joined.append({'role': role, 'pid': r['pid'], 'start_ticks': r['start_ticks'], 'receipt_sha256': boot.digest(boot.encoded(r))})
    passed = s['phase'] == 'await-collection'
    if passed and (roles != list(boot.ROLES) or s.get('current') is not None or s.get('collection_pending') is not True):
        raise ValueError('All four saved roles must finish before a complete-work receipt.')
    return {'kind': 'pgd-split-combined-work-receipt-v1', 'control_sha256': control_sha256,
            'authorization_sha256': authorization_sha256, 'installation_sha256': installation_sha256,
            'session_sha256': boot.digest(session_bytes), 'roles': joined, 'work_phase': s['phase'],
            'all_saved_role_checks_returned': passed, 'collection_started': window.started,
            'collection_deadline': window.deadline, 'connection_deadline': window.connection_started+3000,
            'collection_complete': False, 'independent_review_pending': True,
            'metadata_is_not_authentication_of_computation': True, **FALSE_FLAGS}


class Collector:
    """One snapshot and one stream; no retry, new connection or on-disk archive."""
    def __init__(self, guard, control_bytes, authorization_bytes, *, connection_started, clock=time.monotonic):
        self.window = Window(connection_started, clock(), clock=clock)
        c, policy, csha = boot.control(control_bytes)
        boot.authorization(authorization_bytes, csha)
        ib, isha = boot.installation(c, csha)
        if guard.layout.installation_sha256 != isha:
            raise ValueError('The original source/control installation must join collection.')
        for name, value in (('control.json', control_bytes), ('authorization.json', authorization_bytes), ('installation.json', ib)):
            if guard.read(name, len(value)) != value:
                raise ValueError('Saved collection identity differs from parent-reviewed bytes.')
        self.guard, self.control, self.auth = guard, control_bytes, authorization_bytes
        self.csha, self.asha, self.isha = csha, boot.digest(authorization_bytes), isha
        self.window.check()
        self.prepared = self.streamed = False
        self.header = self.inventory = self.directories = None

    def prepare(self):
        if self.prepared:
            raise ValueError('One collection preparation; preserve prior receipts.')
        self.prepared = True  # A refusal also consumes this object; never auto-retry.
        final_deadline = min(self.window.connection_started+2400, self.window.started+120)
        self.window.check(final_deadline)
        g = self.guard
        session = g.read('session.json', 128*1024)
        receipt = work_receipt(session, self.csha, self.asha, self.isha, self.window)
        rsha = publish(g, RECEIPT, receipt)
        publish(g, STATUS, {'kind': 'pgd-split-collection-status-v1', 'control_sha256': self.csha,
            'phase': 'snapshot-preparation', 'final_receipt_sha256': rsha, 'stream_complete': False, **FALSE_FLAGS})
        report, inventory, directories = g.scan(phase='collection')
        files = {}
        for name in sorted(inventory):
            self.window.check(final_deadline)
            if excluded(name):
                continue
            digest = hashlib.sha256()
            for data in FileReader(g, name, inventory[name], self.window).chunks():
                self.window.check(final_deadline)
                digest.update(data)
            row = inventory[name]
            files[name] = {'bytes': row['bytes'], 'sha256': digest.hexdigest(), 'mode': row['identity'][3], 'mtime_ns': row['mtime_ns']}
        after, current, dirs = g.scan(phase='collection')
        if current != inventory or dirs != directories:
            raise ValueError('Scratch changed around snapshot; no retry or partial promotion.')
        header = {'kind': 'pgd-split-evidence-stream-v1', 'control_sha256': self.csha, 'authorization_sha256': self.asha,
            'root_binding': g.binding, 'files': files, 'inventory': inventory, 'directories': directories,
            'storage': after, 'final_receipt_sha256': rsha, 'session_sha256': boot.digest(session),
            'collection_started': self.window.started, 'collection_deadline': self.window.deadline,
            'connection_started': self.window.connection_started, 'connection_deadline': self.window.connection_started+3000,
            'logical_bytes': sum(r['bytes'] for r in files.values()),
            'omitted_content': sorted(n for n in inventory if excluded(n)), **FALSE_FLAGS}
        validate_header(header, self.control, self.auth)
        if len(boot.encoded(header)) > HEADER_CAP:
            raise ValueError('Bounded evidence manifest exceeded.')
        self.window.check(final_deadline)
        self.header, self.inventory, self.directories = header, inventory, directories
        return header

    def stream(self, out):
        if not self.prepared or self.header is None or self.streamed:
            raise ValueError('One prepared collection stream only.')
        self.streamed = True
        self.window.check()
        header = boot.encoded(self.header)
        writer = WireWriter(out, self.window)
        writer.write(MAGIC+struct.pack('!I', len(header))+header)
        for name, item in self.header['files'].items():
            digest = hashlib.sha256()
            for data in FileReader(self.guard, name, self.inventory[name], self.window).chunks():
                digest.update(data)
                writer.write(data)
            if digest.hexdigest() != item['sha256']:
                raise ValueError('Source hash changed since manifest; no terminal success receipt.')
        _, current, dirs = self.guard.scan(phase='collection')
        if current != self.inventory or dirs != self.directories:
            raise ValueError('Scratch changed during collection; no terminal receipt.')
        trailer = {'kind': 'pgd-split-stream-terminal-v1', 'control_sha256': self.csha,
            'authorization_sha256': self.asha, 'header_sha256': boot.digest(header),
            'prefix_sha256': writer.hasher.hexdigest(), 'prefix_bytes': writer.count,
            'files': len(self.header['files']), 'logical_bytes': self.header['logical_bytes'],
            'final_receipt_sha256': self.header['final_receipt_sha256'], 'stream_complete': True, **FALSE_FLAGS}
        data = boot.encoded(trailer)
        writer.write(END+struct.pack('!I', len(data))+data)
        out.flush()
        self.window.check()
        return dict(trailer, wire_bytes=writer.count, wire_sha256=writer.hasher.hexdigest(),
                    receiver_verified=False, independent_review_pending=True)


class WireWriter:
    def __init__(self, out, window):
        self.out, self.window = out, window
        self.count, self.hasher = 0, hashlib.sha256()

    def write(self, data):
        if type(data) is not bytes or self.count+len(data) > WIRE_CAP:
            raise ValueError('Fixed 3 GiB total wire ceiling.')
        offset = 0
        while offset < len(data):
            self.window.check()
            n = self.out.write(data[offset:])
            if type(n) is not int or not 0 < n <= len(data)-offset:
                raise OSError('Incomplete evidence write; preserve partial stream.')
            self.hasher.update(data[offset:offset+n])
            offset += n
            self.count += n
            self.window.check()


def validate_header(h, control_bytes, authorization_bytes):
    c, p, csha = boot.control(control_bytes)
    boot.authorization(authorization_bytes, csha)
    ib, isha = boot.installation(c, csha)
    layout = scratch.Layout(p['scratch_contract_text'].encode(), p['base_text'].encode(), p['design_text'].encode(), ib, isha)
    keys = {'kind','control_sha256','authorization_sha256','root_binding','files','inventory','directories','storage',
        'final_receipt_sha256','session_sha256','collection_started','collection_deadline','connection_started',
        'connection_deadline','logical_bytes','omitted_content',*FALSE_FLAGS}
    if (type(h) is not dict or set(h) != keys or h['kind'] != 'pgd-split-evidence-stream-v1'
            or h['control_sha256'] != csha or h['authorization_sha256'] != boot.digest(authorization_bytes)
            or any(h[k] is not False for k in FALSE_FLAGS)
            or not boot.sha(h['final_receipt_sha256']) or not boot.sha(h['session_sha256'])):
        raise ValueError('Exact bounded collection identity required.')
    origin, started, end = h['connection_started'], h['collection_started'], h['collection_deadline']
    if (not all(boot.finite(x) for x in (origin, started, end, h['connection_deadline'])) or not origin <= started < end
            or end != min(origin+3000, started+600) or h['connection_deadline'] != origin+3000):
        raise ValueError('Original continuing remote collection clocks required.')
    inventory, directories = h['inventory'], h['directories']
    if type(inventory) is not dict or type(directories) is not dict or type(h['storage']) is not dict:
        raise ValueError('Bounded typed inventory required.')
    storage = layout.assess(inventory, directories, h['storage']['capacity'], phase='collection')
    if storage != h['storage']:
        raise ValueError('Inventory/category accounting changed.')
    binding = h['root_binding']
    if (type(binding) is not dict or set(binding) != {'root','root_identity','parent','parent_identity'}
            or type(binding['root']) is not str or type(binding['parent']) is not str
            or not Path(binding['root']).is_absolute() or str(Path(binding['root']).parent) != binding['parent']
            or binding['root_identity'] != list(directories['.']['identity'])
            or type(binding['parent_identity']) is not list or len(binding['parent_identity']) != 4
            or not all(scratch.natural(v) for v in binding['parent_identity'])
            or binding['parent_identity'][0] != binding['root_identity'][0]):
        raise ValueError('Root snapshot identity changed.')
    files = h['files']
    wanted = {name for name in inventory if not excluded(name)}
    if type(files) is not dict or set(files) != wanted or len(files) > 1024:
        raise ValueError('Every declared evidence name must appear exactly once.')
    required = set(layout.assets) | {'control.json','authorization.json','session.json',RECEIPT,STATUS}
    if not required <= set(files):
        raise ValueError('Complete source/control and collection receipts required; no partial installation admission.')
    for name, item in files.items():
        row = inventory[name]
        if (type(item) is not dict or set(item) != {'bytes','sha256','mode','mtime_ns'}
                or type(item['bytes']) is not int or item['bytes'] != row['bytes']
                or type(item['mode']) is not int or item['mode'] != 0o600
                or type(item['mtime_ns']) is not int or item['mtime_ns'] != row['mtime_ns'] or not boot.sha(item['sha256'])):
            raise ValueError('Private exact file manifest required.')
    pins = dict(layout.assets)
    pins.update({n: {'bytes': len(v), 'sha256': boot.digest(v)} for n, v in
        (('control.json', control_bytes), ('authorization.json', authorization_bytes), ('installation.json', ib))})
    for name, item in pins.items():
        if any(files[name][key] != item[key] for key in ('bytes','sha256')):
            raise ValueError('Collected source/control manifest is not the pinned installation.')
    if files[RECEIPT]['sha256'] != h['final_receipt_sha256'] or files['session.json']['sha256'] != h['session_sha256']:
        raise ValueError('Combined receipt/session hash join changed.')
    omitted = sorted(name for name in inventory if excluded(name))
    total = sum(item['bytes'] for item in files.values())
    if (h['omitted_content'] != omitted or type(h['logical_bytes']) is not int or h['logical_bytes'] != total
            or total+HEADER_CAP+TRAILER_CAP+len(MAGIC)+len(END)+8 > WIRE_CAP):
        raise ValueError('Fixed full-response byte bound or omitted-content policy changed.')
    return layout


class WireReader:
    def __init__(self, source, window):
        self.source, self.window = source, window
        self.count, self.hasher = 0, hashlib.sha256()

    def read(self, size):
        if type(size) is not int or not 0 <= size <= MiB or self.count+size > WIRE_CAP:
            raise ValueError('Bounded evidence read required.')
        result = bytearray()
        while len(result) < size:
            self.window.check()
            block = self.source.read(size-len(result))
            if type(block) is not bytes or not 0 < len(block) <= size-len(result):
                raise ValueError('Empty, truncated or invalid evidence stream; preserve partial files.')
            result.extend(block)
            self.count += len(block)
            self.hasher.update(block)
            self.window.check()
        return bytes(result)

    def frame(self, magic, cap):
        if self.read(len(magic)) != magic:
            raise ValueError('Unexpected evidence framing; no JSON guessing or resynchronization.')
        count = struct.unpack('!I', self.read(4))[0]
        if not 0 < count <= cap:
            raise ValueError('Bounded canonical JSON frame required.')
        raw = self.read(count)
        obj = boot.parsed(raw)
        if boot.encoded(obj) != raw:
            raise ValueError('Canonical duplicate-free finite JSON frame required.')
        return raw, obj


def _directory(fd, name, device):
    try:
        os.mkdir(name, 0o700, dir_fd=fd)
    except FileExistsError:
        pass
    child = os.open(name, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=fd)
    s = os.fstat(child)
    if (not stat.S_ISDIR(s.st_mode) or stat.S_IMODE(s.st_mode) != 0o700
            or s.st_uid != os.getuid() or s.st_dev != device):
        os.close(child)
        raise ValueError('Private owned collection destination ancestry required.')
    return child



def verify_local(rootfd, parentfd, rootname, root_id, records, window):
    """Final bounded held/named tree/hash check after terminal EOF, read-only.

    This is a consistency observation in an owned private directory, not an
    atomic snapshot or protection against arbitrary concurrent same-UID code.
    """
    allowed_dirs = {'.'}
    for name in records:
        allowed_dirs.update(str(p) for p in PurePosixPath(name).parents)
    seen = set()
    nodes = 0
    device, _, uid, _ = root_id
    def walk(fd, prefix):
        nonlocal nodes
        window.check()
        before = os.fstat(fd)
        if (not stat.S_ISDIR(before.st_mode) or before.st_uid != uid or before.st_dev != device
                or stat.S_IMODE(before.st_mode) != 0o700):
            raise ValueError('Final private local directory check refused.')
        with os.scandir(fd) as entries:
            for entry in entries:
                window.check()
                nodes += 1
                if nodes > 1280:
                    raise ValueError('Final local collection node ceiling.')
                name = (prefix+'/' if prefix else '')+entry.name
                scratch.safe(name)
                info = os.stat(entry.name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    if name not in allowed_dirs:
                        raise ValueError('Unexpected local collection directory.')
                    child = os.open(entry.name, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        if scratch.directory_id(os.fstat(child)) != scratch.directory_id(info):
                            raise ValueError('Held local directory changed.')
                        walk(child, name)
                    finally:
                        os.close(child)
                    if scratch.directory_id(os.stat(entry.name, dir_fd=fd, follow_symlinks=False)) != scratch.directory_id(info):
                        raise ValueError('Named local directory changed.')
                else:
                    wanted = records.get(name)
                    if (wanted is None or not stat.S_ISREG(info.st_mode) or info.st_uid != uid or info.st_dev != device
                            or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600
                            or info.st_size != wanted['bytes'] or info.st_mtime_ns != wanted['mtime_ns']):
                        raise ValueError('Final local member identity/extent/mode/nsmtime refused.')
                    datafd = os.open(entry.name, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK, dir_fd=fd)
                    try:
                        current = os.fstat(datafd)
                        if not stat.S_ISREG(current.st_mode) or scratch.file_id(current) != scratch.file_id(info):
                            raise ValueError('Held final local member changed.')
                        digest = hashlib.sha256()
                        left = info.st_size
                        while left:
                            window.check()
                            block = os.read(datafd, min(MiB, left))
                            if not block:
                                raise ValueError('Final local member truncated.')
                            digest.update(block)
                            left -= len(block)
                        if (os.read(datafd, 1) or digest.hexdigest() != wanted['sha256']
                                or scratch.file_id(os.fstat(datafd)) != scratch.file_id(info)):
                            raise ValueError('Final local member hash or held identity changed.')
                    finally:
                        os.close(datafd)
                    if scratch.file_id(os.stat(entry.name, dir_fd=fd, follow_symlinks=False)) != scratch.file_id(info):
                        raise ValueError('Named final local member changed.')
                    seen.add(name)
        if scratch.directory_id(os.fstat(fd)) != scratch.directory_id(before):
            raise ValueError('Final local directory identity changed.')
    if scratch.directory_id(os.stat(rootname, dir_fd=parentfd, follow_symlinks=False)) != root_id:
        raise ValueError('Named collection root changed at terminal EOF.')
    walk(rootfd, '')
    if seen != set(records) or scratch.directory_id(os.stat(rootname, dir_fd=parentfd, follow_symlinks=False)) != root_id:
        raise ValueError('Final local collection incomplete or replaced.')
    window.check()


def receive(source, destination, control_bytes, authorization_bytes, *, window):
    """Decode current bounded byte stream into one fresh private local directory.

    All partial files survive every failure. This performs no SSH/retrieval and
    does not reset the caller's original client clock or authorize an experiment.
    """
    reader = WireReader(source, window)
    header_bytes, header = reader.frame(MAGIC, HEADER_CAP)
    validate_header(header, control_bytes, authorization_bytes)
    dest = Path(destination)
    if (not dest.is_absolute() or dest.parent.resolve(strict=True) != dest.parent or os.path.lexists(dest)):
        raise ValueError('Fresh canonical private destination required; preserve old output.')
    parent = os.open(dest.parent, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    rootfd = None
    try:
        ps = os.fstat(parent)
        if (stat.S_IMODE(ps.st_mode) != 0o700 or ps.st_uid != os.getuid()
                or scratch.directory_id(dest.parent.lstat()) != scratch.directory_id(ps)):
            raise ValueError('Private destination parent required.')
        os.mkdir(dest.name, 0o700, dir_fd=parent)
        rootfd = os.open(dest.name, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=parent)
        root_id = scratch.directory_id(os.fstat(rootfd))
        if root_id[0] != ps.st_dev or root_id[2:] != (os.getuid(), 0o700):
            raise ValueError('Private same-device collection root required.')
        records = {}
        for name in sorted(header['files']):
            item = header['files'][name]
            parts = PurePosixPath(name).parts
            fd = os.dup(rootfd)
            try:
                for part in parts[:-1]:
                    nextfd = _directory(fd, part, ps.st_dev)
                    os.close(fd)
                    fd = nextfd
                outfd = os.open(parts[-1], os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600, dir_fd=fd)
                digest = hashlib.sha256()
                try:
                    left = item['bytes']
                    while left:
                        data = reader.read(min(MiB, left))
                        offset = 0
                        while offset < len(data):
                            window.check()
                            wrote = os.write(outfd, data[offset:])
                            if wrote <= 0:
                                raise OSError('Short local evidence write.')
                            offset += wrote
                        digest.update(data)
                        left -= len(data)
                    os.fsync(outfd)
                    os.utime(outfd, ns=(item['mtime_ns'], item['mtime_ns']))
                    s = os.fstat(outfd)
                    if (not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_dev != ps.st_dev
                            or s.st_nlink != 1 or stat.S_IMODE(s.st_mode) != 0o600 or s.st_size != item['bytes']
                            or s.st_mtime_ns != item['mtime_ns'] or digest.hexdigest() != item['sha256']):
                        raise ValueError('Collected member hash/type/extent/mode/nsmtime differs; preserve partials.')
                    records[name] = {'sha256': digest.hexdigest(), 'bytes': s.st_size, 'mode': 0o600, 'mtime_ns': s.st_mtime_ns}
                finally:
                    os.close(outfd)
                os.fsync(fd)
            finally:
                os.close(fd)
            if scratch.directory_id(os.stat(dest.name, dir_fd=parent, follow_symlinks=False)) != root_id:
                raise ValueError('Named local collection root changed.')
        prefix_sha, prefix_bytes = reader.hasher.hexdigest(), reader.count
        _, terminal = reader.frame(END, TRAILER_CAP)
        expected = {'kind': 'pgd-split-stream-terminal-v1', 'control_sha256': header['control_sha256'],
            'authorization_sha256': header['authorization_sha256'], 'header_sha256': boot.digest(header_bytes),
            'prefix_sha256': prefix_sha, 'prefix_bytes': prefix_bytes, 'files': len(records),
            'logical_bytes': header['logical_bytes'], 'final_receipt_sha256': header['final_receipt_sha256'],
            'stream_complete': True, **FALSE_FLAGS}
        if terminal != expected:
            raise ValueError('Missing or mismatched terminal evidence receipt; no completion admission.')
        window.check()
        if source.read(1) != b'':
            raise ValueError('Unexpected trailing stream bytes; preserve all output.')
        window.check()
        if (dest.parent.resolve(strict=True) != dest.parent
                or scratch.directory_id(dest.parent.lstat()) != scratch.directory_id(ps)):
            raise ValueError('Named local collection parent changed at terminal EOF.')
        verify_local(rootfd, parent, dest.name, root_id, records, window)
        if scratch.directory_id(dest.parent.lstat()) != scratch.directory_id(ps):
            raise ValueError('Named local collection parent changed during final verification.')
        os.fsync(rootfd)
        window.check()
        # This receipt describes transport integrity only. Scientific review is separate.
        return {'kind': 'pgd-split-local-collection-receipt-v1', 'control_sha256': header['control_sha256'],
            'authorization_sha256': header['authorization_sha256'], 'header_sha256': boot.digest(header_bytes),
            'final_receipt_sha256': header['final_receipt_sha256'], 'wire_sha256': reader.hasher.hexdigest(),
            'wire_bytes': reader.count, 'files': records, 'transport_verified': True,
            'independent_review_pending': True, **FALSE_FLAGS}
    finally:
        if rootfd is not None:
            os.close(rootfd)
        os.close(parent)
