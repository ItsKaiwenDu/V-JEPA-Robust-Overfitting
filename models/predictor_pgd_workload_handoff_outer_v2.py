"""Unarmed outer transport/setup definitions; no CLI or activity on import.

Real signal/process/proof/GPU operations are future authorized target bindings.
Local tests substitute those bindings; scientific workers never execute locally.
"""
import base64
import ast
import _thread
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import signal
import stat
import struct
import sys
import tarfile
import threading
import time
from types import SimpleNamespace
MiB, GiB = (2 ** 20, 2 ** 30)
FRAME = b'VJEPA-PGD-SPLIT-OUTER-1\n'
POLICY_SHA = '405638030b1e1defdb3e04f47538ef68b87b628896b4fc66724e3a580b7363a9'
BOOT_SHA = '5a5e68be3148ce93acad8f06cf0568e2c8c00eb5b45057dd63d8f755db358ae3'
COLLECTOR_SHA = '085f7ee7dc28e6b93569bfc36d4ed1554f9a8120cbdcf3a362f0bcfe2c6331bc'
HOST = 'linux10614.dc.engr.scu.edu'
FINGERPRINT = 'SHA256:ARgVRWOzRLA1LmAANxD6Ya17EIIo2sa4ZbHvkeqzgKQ'
KNOWN_HOSTS = 'linux10614.dc.engr.scu.edu ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAELc+tu7f+683dQbOPg5lndZ9pGDeODvo8r1Sp8d7ri\n'
PACKET_PATH = 'runs/pretrained/diagnostic/eot-work-profile-university-20261029/packet/vjepa-eot-profile.tar.gz'
LOCAL_SOURCES = ('models/predictor_pgd_workload_handoff_outer_v2.py', 'models/predictor_pgd_workload_handoff_client_v2.py', 'models/predictor_pgd_workload_operator_v1.py', 'configs/predictor_pgd_workload_operator_inputs_v2.json', 'models/predictor_pgd_workload_os_v1.py', 'models/predictor_pgd_workload_bootstrap_v1.py', 'models/predictor_pgd_workload_scratch_v1.py', 'models/predictor_pgd_workload_collection_v1.py', 'models/__init__.py', 'models/predictor_pgd_split_plan_v1.py', 'models/predictor_pgd_workload_handoff_v2.py', 'configs/predictor_pgd_workload_handoff_inputs_v2.json', 'scripts/run_predictor_pgd_workload_handoff_v2.py', 'models/predictor_pgd_workload_claim_repair_v1.py', 'configs/predictor_pgd_workload_claim_repair_v1.json', 'configs/predictor_pgd_workload_claim_binding_v1.json')
DIRECTORY_POLICY_SHA = 'aee644895d35e0a3c7a67b471f4da0f0910a56f654ece38edeb2d3efa4e5a4c0'
REFUSAL_POLICY_SHA = 'c6e045b37602f639051dc8c717af4a583eeffec3d2ea12fb46a737db912b7721'
FROZEN_V3_SHA = '8c21d936e5d311c7ddc0dffa4bc91c01a7f5c1c0e8731efa5fc33c58df57ac46'
SOURCE_CAP = 2 * MiB
FLAGS = {'scientific_qualified': False, 'training_admitted': False}

def encoded(v):
    return (json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def sha(v):
    return type(v) is str and len(v) == 64 and all((c in '0123456789abcdef' for c in v))

def finite(v):
    return type(v) in (int, float) and math.isfinite(v)

def parsed(data):

    def pairs(rows):
        out = {}
        for k, v in rows:
            if k in out:
                raise ValueError('Duplicate outer metadata key.')
            out[k] = v
        return out

    def bad(x):
        raise ValueError('Nonfinite outer metadata: ' + x)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)

def safe(name):
    if type(name) is not str or not name or len(name) > 768 or any((c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/' for c in name)) or PurePosixPath(name).is_absolute() or (str(PurePosixPath(name)) != name) or any((p in ('.', '..') or len(p) > 128 for p in PurePosixPath(name).parts)):
        raise ValueError('Canonical bounded private relative name required.')
    return name

class Deadline(TimeoutError):
    pass

class Stopped(TimeoutError):
    pass

class Alarms:
    """Default kernel hard stop plus a best-effort main-thread soft refusal.

    Python handlers are not an instantaneous kernel/C-extension stop guarantee.
    Tests exercise this class with fake signal/time/exit APIs only.
    """

    def __init__(self, *, clock=time.monotonic):
        if threading.current_thread() is not threading.main_thread():
            raise ValueError('Outer timers require the main thread.')
        self.clock = clock
        self.owner = os.getpid()
        self.phase = None
        self.hard = None
        self.soft_deadline = None
        self.last = clock()
        self.closed = False
        self.waker = None
        self.generation = 0
        self.previous = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGHUP, signal.SIGTERM, signal.SIGUSR1, signal.SIGINT)}
        self.previous_timer = signal.getitimer(signal.ITIMER_REAL)
        if self.previous_timer != (0.0, 0.0):
            raise ValueError('Fresh outer process must not replace an existing timer.')

    def moment(self):
        now = self.clock()
        if self.closed or os.getpid() != self.owner or (not finite(now)) or (now < self.last):
            raise ValueError('Owned continuing outer clock required.')
        self.last = now
        return now

    def check(self):
        now = self.moment()
        if self.soft_deadline is not None and now >= self.soft_deadline:
            self.handler(signal.SIGALRM, None)
        return now

    def cancel_wake(self):
        self.generation += 1
        if self.waker is not None:
            self.waker.cancel()
            self.waker = None

    def handler(self, number, frame):
        now = self.clock()
        if number == signal.SIGUSR1 and now < self.soft_deadline:
            return
        self.cancel_wake()
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        remaining = min(2, self.hard - now)
        if remaining <= 0:
            os._exit(124)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        if number in (signal.SIGALRM, signal.SIGUSR1):
            raise Deadline('Original ' + str(self.phase) + ' deadline reached; no reset.')
        raise Stopped('Attended connection was interrupted; stop owned work and preserve evidence.')

    def arm(self, deadline, *, hard_deadline, phase):
        now = self.moment()
        if not finite(deadline) or not finite(hard_deadline) or (not now < deadline <= hard_deadline):
            raise Deadline('Original outer deadline is already exhausted.')
        self.cancel_wake()
        self.phase = phase
        self.hard = hard_deadline
        self.soft_deadline = deadline
        for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(s, self.handler)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        current = self.moment()
        if current >= deadline:
            raise Deadline('Original phase exhausted while binding its timers.')
        signal.setitimer(signal.ITIMER_REAL, hard_deadline - current)
        token = self.generation

        def wake():
            if not self.closed and token == self.generation:
                _thread.interrupt_main(signal.SIGUSR1)
        self.waker = threading.Timer(deadline - current, wake)
        self.waker.daemon = True
        self.waker.start()

    def close(self):
        if self.closed:
            return
        self.cancel_wake()
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, h in self.previous.items():
            signal.signal(s, h)
        self.closed = True

def read_exact(stream, size, check):
    if type(size) is not int or size < 0 or size > 3 * GiB:
        raise ValueError('Bounded transport extent required.')
    out = bytearray()
    while len(out) < size:
        check()
        block = stream.read(min(MiB, size - len(out)))
        if type(block) is not bytes or not 0 < len(block) <= min(MiB, size - len(out)):
            raise ValueError('Empty/truncated outer transport; preserve all partial output.')
        out.extend(block)
        check()
    return bytes(out)

def write_all(stream, data, check):
    offset = 0
    while offset < len(data):
        check()
        n = stream.write(data[offset:])
        if type(n) is not int or not 0 < n <= len(data) - offset:
            raise OSError('Incomplete outer transport write.')
        offset += n
        check()

def send_frame(stream, value, check, *, cap=64 * 1024):
    data = encoded(value)
    if not 0 < len(data) <= cap:
        raise ValueError('Bounded outer metadata frame required.')
    write_all(stream, FRAME + struct.pack('!I', len(data)) + data, check)
    stream.flush()
    check()

def frame(stream, check, *, cap=64 * 1024):
    if read_exact(stream, len(FRAME), check) != FRAME:
        raise ValueError('Wrong outer transport framing; no implicit retry or resynchronization.')
    count = struct.unpack('!I', read_exact(stream, 4, check))[0]
    if not 0 < count <= cap:
        raise ValueError('Outer metadata frame too large.')
    data = read_exact(stream, count, check)
    value = parsed(data)
    if encoded(value) != data:
        raise ValueError('Canonical finite outer metadata required.')
    return value

def validate_control(raw, code_sha256):
    if type(raw) is not bytes or len(raw) > 256 * 1024:
        raise ValueError('Bounded component outer control required.')
    c = parsed(raw)
    keys = {'kind', 'supervised_control_text', 'supervised_control_sha256', 'outer_source_sha256', 'local_source_sha256', 'runtime_files', 'repository', 'local_handoff', 'packet_path', 'known_hosts_text', 'known_hosts_sha256', 'expected_ssh_fingerprint', 'financial_state_unchanged', 'authorized_at_preparation', 'target_helper_reviewed'}
    if type(c) is not dict or set(c) != keys or c['kind'] != 'whole-pgd-outer-control-v1' or (encoded(c) != raw) or (c['outer_source_sha256'] != code_sha256) or (not sha(code_sha256)) or (c['packet_path'] != PACKET_PATH) or (c['known_hosts_text'] != KNOWN_HOSTS) or (c['known_hosts_sha256'] != digest(KNOWN_HOSTS.encode())) or (c['expected_ssh_fingerprint'] != FINGERPRINT) or (c['financial_state_unchanged'] is not True) or (c['authorized_at_preparation'] is not False) or (c['target_helper_reviewed'] is not False):
        raise ValueError('Exact new component operator/control/host binding required.')
    if type(c['local_source_sha256']) is not dict or set(c['local_source_sha256']) != set(LOCAL_SOURCES) or any((not sha(v) for v in c['local_source_sha256'].values())) or (c['local_source_sha256'][LOCAL_SOURCES[0]] != code_sha256):
        raise ValueError('Complete exact component operator inputs required.')
    for key in ('repository', 'local_handoff'):
        value = c[key]
        if type(value) is not str or not Path(value).is_absolute() or str(Path(value)) != value or ('..' in Path(value).parts):
            raise ValueError('Exact canonical local handoff/repository required.')
    text = c['supervised_control_text']
    if type(text) is not str or len(text.encode()) > 256 * 1024 or digest(text.encode()) != c['supervised_control_sha256']:
        raise ValueError('Component supervised control digest changed.')
    b = parsed(text)
    p = parsed(b['policy_text'])
    if digest(b['policy_text'].encode()) != POLICY_SHA or b['policy_sha256'] != POLICY_SHA or c['runtime_files'] != STATIC_RUNTIME or (len(c['runtime_files']) != 46):
        raise ValueError('Pinned component-only policy/runtime changed.')
    if set(b['supervisor_sources']) != set(SUPERVISOR_NAMES) or b['supervisor_sources']['bootstrap.py']['sha256'] != BOOT_SHA or b['supervisor_sources']['predictor_pgd_split_collection_v1.py']['sha256'] != COLLECTOR_SHA:
        raise ValueError('Pinned component supervisors changed.')
    return (c, b, p, digest(raw))

def authorization(c, csha, basis):
    if type(basis) is not str or not 1 <= len(basis) <= 8192:
        raise ValueError('One explicit exact scoped user basis required.')
    return {'kind': 'whole-pgd-outer-university-exception-v1', 'approved': True, 'outer_control_sha256': csha, 'supervised_control_sha256': c['supervised_control_sha256'], 'maximum_new_attempts': 1, 'work_seconds': 2400, 'collection_seconds': 600, 'connection_including_login_seconds': 3000, 'full_model_gradients': 120, 'extra_component_VJPs': 0, 'automatic_retries': 0, 'paid_compute': False, 'cleanup': False, 'reset_old_claims': False, 'private_key_upload': False, 'held_out_pixels': False, 'user_basis': basis}

def validate_authorization(raw, c, csha):
    if type(raw) is not bytes or len(raw) > 64 * 1024:
        raise ValueError('Bounded new outer authorization required.')
    a = parsed(raw)
    if type(a) is not dict or a != authorization(c, csha, a.get('user_basis')) or encoded(a) != raw:
        raise ValueError('A new exact one-attempt outer grant is required; old grants are consumed.')
    return a

def decode_sources(mapping, c, b):
    if type(mapping) is not dict or len(mapping) != 55:
        raise ValueError('All 55 exact whole-PGD and supervisor source files required.')
    descriptors = {'supervisor/' + n: d for n, d in b['supervisor_sources'].items()}
    descriptors.update({'overlay/' + n: d for n, d in c['runtime_files'].items()})
    if set(mapping) != set(descriptors):
        raise ValueError('Source namespace changed.')
    out = {}
    total = 0
    for name, value in mapping.items():
        safe(name)
        if type(value) is not str or len(value) > 2 * MiB:
            raise ValueError('Bounded source representation required.')
        data = base64.b64decode(value, validate=True)
        want = descriptors[name]
        total += len(data)
        if len(data) != want['bytes'] or digest(data) != want['sha256'] or total > 2 * MiB:
            raise ValueError('Pinned source bytes/extent changed.')
        compile(data, '<CHECKED-UNEXECUTED-' + name + '>', 'exec')
        out[name] = data
    return out

def boot_namespace(sources):
    data = sources['supervisor/bootstrap.py']
    if digest(data) != BOOT_SHA:
        raise ValueError('Original stdlib bootstrap required before imports.')
    namespace = {'__name__': '_pgd_outer_checked_bootstrap', '__file__': '<checked-stdin-bootstrap>'}
    exec(compile(data, namespace['__file__'], 'exec'), namespace)
    return SimpleNamespace(**namespace)

def remote_root(csha):
    if not sha(csha):
        raise ValueError('Exact outer attempt identity required.')
    return Path('/var/tmp') / ('vjepa-pgd-workload-v1-' + str(os.getuid()) + '-' + csha)

def preflight_parent():
    if (platform.node().split('.')[0], platform.system(), platform.python_version()) != ('linux10614', 'Linux', '3.11.17'):
        raise ValueError('The exact university host/runtime is required before any target write.')
    p = Path('/var/tmp')
    info = p.lstat()
    if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_uid, stat.S_IMODE(info.st_mode)) != (66307, 0, 1023) or p.resolve(strict=True) != p or (Path.home().stat().st_dev == info.st_dev):
        raise ValueError('Original separate local scratch device/parent changed.')
    with Path('/proc/self/mountinfo').open('rb') as stream:
        data = stream.read(2 * MiB + 1)
    if len(data) > 2 * MiB:
        raise ValueError('Bounded mount metadata required.')
    matches = []
    for row in data.decode().splitlines():
        left, sep, right = row.partition(' - ')
        a = left.split()
        b = right.split()
        if sep and len(a) >= 6 and (len(b) >= 3):
            point = a[4]
            if str(p) == point or str(p).startswith(point.rstrip('/') + '/'):
                matches.append((point, b[0], a[5].split(',') + b[2].split(',')))
    if not matches:
        raise ValueError('No original scratch mount.')
    point, fs, options = max(matches, key=lambda row: len(row[0]))
    if (point, fs) != ('/', 'xfs') or 'rw' not in options:
        raise ValueError('Original writable local XFS mount required.')

def directory_id(info):
    return (info.st_dev, info.st_ino, info.st_uid, stat.S_IMODE(info.st_mode))

def private_write(root, name, data, check):
    safe(name)
    check()
    if type(data) is not bytes or len(data) > 2 * MiB:
        raise ValueError('Bounded source/metadata publication required.')
    root = Path(root)
    identity = directory_id(root.lstat())
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if identity[2:] != (os.getuid(), 448) or directory_id(os.fstat(fd)) != identity:
            raise ValueError('Private setup root required.')
        parts = PurePosixPath(name).parts
        for part in parts[:-1]:
            try:
                os.mkdir(part, 448, dir_fd=fd)
            except FileExistsError:
                pass
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            s = os.fstat(nxt)
            if (s.st_dev, s.st_uid, stat.S_IMODE(s.st_mode)) != (identity[0], os.getuid(), 448):
                os.close(nxt)
                raise ValueError('Private setup ancestor required.')
            os.close(fd)
            fd = nxt
        with os.fdopen(os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 384, dir_fd=fd), 'wb') as out:
            write_all(out, data, check)
            out.flush()
            os.fsync(out.fileno())
        os.fsync(fd)
        if directory_id(root.lstat()) != identity:
            raise ValueError('Named setup root changed.')
        check()
    finally:
        os.close(fd)

def receive_packet(incoming,root,spec,check):
    if spec['bytes']!=17192095 or spec['sha256']!='2278bce2b519943c0f83a49c556bb73dd928fe880fba0f6c56de2a4c3c2ec0da':
        raise ValueError('The unchanged input packet is required.')
    path=Path(root)/'original-packet.tar.gz.partial';h=hashlib.sha256();left=spec['bytes']
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as out:
        while left:
            data=read_exact(incoming,min(MiB,left),check);write_all(out,data,check);h.update(data);left-=len(data)
        out.flush();os.fsync(out.fileno())
    if h.hexdigest()!=spec['sha256']:raise ValueError('Input packet hash changed; retain the partial.')
    os.link(path,path.with_name('original-packet.tar.gz'));os.unlink(path)
    if incoming.read(1)!=b'':raise ValueError('Unexpected bytes after the single approved packet.')
    check()

def extract_packet(root,files,check):
    if len(files)!=99 or sum(w['bytes'] for w in files.values())!=26446958:raise ValueError('Original packet member extent required.')
    seen=set()
    with tarfile.open(Path(root)/'original-packet.tar.gz',mode='r:gz') as archive:
        for item in archive:
            check();name=safe(item.name)
            if name in seen or name not in files or not item.isfile() or item.size!=files[name]['bytes']:
                raise ValueError('Unexpected, duplicate or nonregular packet member.')
            with archive.extractfile(item) as source:data=read_exact(source,item.size,check)
            if digest(data)!=files[name]['sha256']:raise ValueError('Pinned packet member changed.')
            # The largest original member is below 16 MiB. Chunked input files
            # are installed explicitly; private_write's 2 MiB source cap stays.
            destination=Path(root)/'code'/name;destination.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            q=destination.parent
            while q!=Path(root):
                if q.is_symlink() or stat.S_IMODE(q.stat().st_mode)!=0o700 or q.stat().st_uid!=os.getuid():raise ValueError('Private packet ancestry required.')
                q=q.parent
            with os.fdopen(os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as out:
                write_all(out,data,check);out.flush();os.fsync(out.fileno())
            seen.add(name)
    if seen!=set(files):raise ValueError('Incomplete original packet; no proof or model work.')
    check()
_DIRECTORY_SHELL = {'.', 'capture', 'capture/results', 'capture/logs', 'cache', 'cache/cuda', 'tmp'}
_DIRECTORY_WORKER_CLAIMS = {'forward-preserving': 'capture/results/components/forward-preserving'}

def _setup_directory_plan(layout):
    claims={r:'capture/results/workload/'+r for r in ('original-unit','candidate-unit','candidate-replay')}
    selected={'.','capture','capture/logs','capture/results','capture/results/workload','cache','cache/cuda','tmp'}
    for name in layout.assets:selected.update(str(p) for p in PurePosixPath(name).parents)
    if any(n==p or n.startswith(p+'/') for n in selected for p in claims.values()):
        raise ValueError('Worker directories must not be precreated by setup.')
    for name in selected:layout.directory(name)
    return {'kind':'whole-pgd-minimal-directory-plan-v1',
        'setup_directories':sorted(selected,key=lambda n:(len(PurePosixPath(n).parts),n)),
        'worker_claim_directories':claims,'authorized':False,'scientific_qualified':False,'training_admitted':False}

def prepare_tree(root, sources, c, b, p, auth, outer_raw, outer_auth, boot, incoming, check):
    supervised = c['supervised_control_text'].encode()
    b, policy, bsha = boot.control(supervised)
    derived = boot.encoded({'kind': 'whole-pgd-supervised-university-exception-v1', 'approved': True, 'control_sha256': bsha, 'maximum_new_attempts': 1, 'user_basis': auth['user_basis']})
    boot.authorization(derived, bsha)
    ib, isha = boot.installation(b, bsha)
    for name, data in sorted(sources.items()):
        private_write(root, name, data, check)
    private_write(root, 'control.json', supervised, check)
    private_write(root, 'authorization.json', derived, check)
    private_write(root, 'installation.json', ib, check)
    private_write(root, 'configs/base.json', p['base_text'].encode(), check)
    private_write(root, 'configs/design.json', p['design_text'].encode(), check)
    private_write(root, 'proposal.json', encoded({'kind': 'pgd-split-outer-ATTEMPT-v1', 'consumed': True, 'outer_control': parsed(outer_raw), 'outer_authorization': parsed(outer_auth), 'outer_control_sha256': digest(outer_raw), 'outer_authorization_sha256': digest(outer_auth), **FLAGS}), check)
    binding = {'root': str(root), 'root_identity': list(directory_id(root.lstat())), 'parent': str(root.parent), 'parent_identity': list(directory_id(root.parent.lstat()))}
    finder = boot.PinnedImporter(root, binding, b, p).install()
    from models import predictor_pgd_split_scratch_v1 as scratch
    from models import predictor_pgd_split_coordinator_v1 as coordinator
    layout = scratch.Layout(p['scratch_contract_text'].encode(), p['base_text'].encode(), p['design_text'].encode(), ib, isha)
    directory_plan = _setup_directory_plan(layout)
    for name in directory_plan['setup_directories']:
        if name == '.':
            continue
        q = root / name
        try:
            q.mkdir(mode=448)
        except FileExistsError:
            pass
        info = q.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 448 or info.st_uid != os.getuid() or (info.st_dev != binding['root_identity'][0]):
            raise ValueError('Private declared setup directory required.')
    private_write(root, 'runtime-source-manifest.json', encoded({'kind': 'pgd-split-installed-source-list-v1', 'files': {n: {'bytes': len(data), 'sha256': digest(data)} for n, data in sources.items()}, 'outer_source_sha256': c['outer_source_sha256'], 'setup_directory_plan': directory_plan}), check)
    guard = scratch.Guard(root, binding, layout)
    before, _, _ = guard.scan(phase='setup', initial=True, settled=True)
    private_write(root, 'storage-before.json', encoded(before), check)
    coordinator.target_gate(root, binding)
    check()
    fd = os.open(root / 'emergency-space.bin', os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 384)
    try:
        os.posix_fallocate(fd, 0, 16 * MiB)
        os.fsync(fd)
        info = os.fstat(fd)
        if info.st_size != 16 * MiB or info.st_blocks * 512 < 16 * MiB:
            raise ValueError('Full allocated emergency space required; no sparse fallback.')
    finally:
        os.close(fd)
    contract = parsed(p['scratch_contract_text'])
    receive_packet(incoming, root, contract['packet']['archive'], check)
    extract_packet(root, contract['packet']['files'], check)
    guard.verify_assets(phase='setup')
    guard.proof_readiness()
    check()
    return (coordinator, scratch, guard, binding, derived)

class OutputBudget:

    def __init__(self, out, check):
        self.out = out
        self.count = 0
        self.check = check

    def write(self, data):
        self.check()
        if self.count + len(data) > 3 * GiB:
            raise ValueError('Whole same-SSH response exceeds 3 GiB.')
        n = self.out.write(data)
        if type(n) is not int or not 0 < n <= len(data):
            raise OSError('Incomplete same-SSH response.')
        self.count += n
        self.check()
        return n

    def flush(self):
        self.out.flush()
        self.check()

def remote_session(incoming, outgoing, *, code_sha256):
    """One future SSH connection. No alternate host, source port or retry mode."""
    alarms = Alarms()
    hello = time.monotonic()
    nonce = os.urandom(32).hex()
    origin = None
    owner = None
    root = None
    streaming = False
    alarms.arm(hello + 180, hard_deadline=hello + 180, phase='handshake')
    output = OutputBudget(outgoing, alarms.check)
    try:
        send_frame(output, {'kind': 'pgd-split-hello-v1', 'nonce': nonce, 'hello_monotonic': hello, 'code_sha256': code_sha256}, alarms.check)
        message = frame(incoming, alarms.check, cap=2 * MiB)
        received = time.monotonic()
        if type(message) is not dict or set(message) != {'nonce', 'client_elapsed', 'control_text', 'authorization_text', 'sources'} or message['nonce'] != nonce or (not finite(message['client_elapsed'])) or (not 0 <= message['client_elapsed'] < 180):
            raise ValueError('One fresh handshake with original elapsed client time required.')
        origin = hello - message['client_elapsed']
        alarms.arm(origin + 180, hard_deadline=origin + 180, phase='setup')
        raw = message['control_text'].encode()
        araw = message['authorization_text'].encode()
        c, b, p, csha = validate_control(raw, code_sha256)
        a = validate_authorization(araw, c, csha)
        sources = decode_sources(message['sources'], c, b)
        boot = boot_namespace(sources)
        boot.control(c['supervised_control_text'].encode())
        preflight_parent()
        alarms.check()
        root = remote_root(csha)
        root.mkdir(mode=448)
        send_frame(output, {'kind': 'pgd-split-root-claimed-v1', 'outer_control_sha256': csha, 'scratch': str(root), 'hello_monotonic': hello, 'handshake_received': received, 'connection_started': origin, 'work_deadline': origin + 2400, 'connection_deadline': origin + 3000, **FLAGS}, alarms.check)
        coordinator, scratch, guard, binding, derived = prepare_tree(root, sources, c, b, p, a, raw, araw, boot, incoming, alarms.check)
        alarms.arm(origin + 2280, hard_deadline=origin + 2400, phase='work')
        owner = coordinator.OuterCoordinator(root, binding, c['supervised_control_text'].encode(), derived, connection_started=origin, outer_guard=guard, outer_alarm=alarms)
        work_error = None
        try:
            owner.run_work()
        except BaseException as error:
            owner.driver.assert_idle()
            work_error = error
        owner.driver.assert_idle()
        if not owner.claimed or owner.driver.guard is None:
            if work_error is not None:
                raise work_error
            raise RuntimeError('Parent session was not initialized; no collection permitted.')
        alarms.arm(min(origin + 2400, time.monotonic() + 120), hard_deadline=origin + 2400, phase='finalization')
        if work_error is None:
            try:
                owner.driver.outer_journal.checkpoint('finalization')
            except BaseException as error:
                work_error = error
                owner.state.update(phase='failed', error_type=type(error).__name__, error=str(error)[:1024])
                owner.driver.save(owner.state, replace=True)
        owner.driver.outer_journal.close()
        from models import predictor_pgd_split_collection_v1 as collection
        collector = collection.Collector(owner.driver.guard, c['supervised_control_text'].encode(), derived, connection_started=origin)
        collector.prepare()
        alarms.arm(collector.window.deadline, hard_deadline=collector.window.deadline, phase='collection')
        send_frame(output, {'kind': 'pgd-split-collection-ready-v1', 'outer_control_sha256': csha, 'collection_started': collector.window.started, 'collection_deadline': collector.window.deadline, 'connection_started': origin, 'handshake_received': received, **FLAGS}, alarms.check)
        streaming = True
        collector.stream(output)
        return 0 if work_error is None else 1
    except BaseException as error:
        if owner is not None:
            owner.driver.assert_idle()
            owner.driver.outer_journal.close()
        if not streaming:
            bound = origin + 3000 if origin is not None else hello + 180
            alarms.arm(min(bound, time.monotonic() + 2), hard_deadline=bound, phase='refusal')
            send_frame(output, {'kind': 'pgd-split-outer-refusal-v1', 'error_type': type(error).__name__, 'message': str(error)[:1024], 'scratch': None if root is None else str(root), 'new_work_started_after_error': False, 'last_guard_refusal': None if owner is None else owner.driver.outer_journal.last_refusal, **FLAGS}, alarms.check)
        else:
            sys.stderr.write('Bounded evidence stream interrupted: ' + type(error).__name__ + '; preserve partial local files.\n')
        return 1
    finally:
        alarms.close()
STATIC_RUNTIME = {'attacks/__init__.py': {'bytes': 69, 'sha256': '41a090d550527278b09bfb62fbed810805b379151b7a5ab7cb5469bc12ee01cb'}, 'attacks/pgd.py': {'bytes': 5089, 'sha256': '947c3e4aa275fa98b5f84a2a7a89f74b8161d6ba498b948862f7d488712c2d81'}, 'attacks/pgd_workload_target_v1.py': {'bytes': 5081, 'sha256': 'dfcc35139dd5f51f1c52f1e79e1ecbc8d7f50bb7d0ae3dda9b18963c00d320a1'}, 'datasets/__init__.py': {'bytes': 67, 'sha256': 'd81c61d6627e4e28075736950978c3cca23843d93fb213454458bd32f5357cf1'}, 'datasets/prepared.py': {'bytes': 3784, 'sha256': '531dcb4e4e69e3a59890e020940a6a8d0459d47100154730eabc6d267c5e2bb5'}, 'models/__init__.py': {'bytes': 86, 'sha256': '1d079e9ff2633dfad73abff31961c9e81b7aff65e44ec6bedc60a32189630ff5'}, 'models/_pgd_workload_entry_v1.py': {'bytes': 1822, 'sha256': 'd3dfbe3a5836ed80b9a4cf1cdde115cfca860f19b5539ae3c7d5dada5b0f6b6d'}, 'models/_pgd_workload_inputs_v1.py': {'bytes': 3029, 'sha256': 'b6b2d13cc9899ed77facde2697187807c8906a9486f3daf4a906b52d361b2540'}, 'models/candidate_contract.py': {'bytes': 5094, 'sha256': '13f664f7146eb4816cbaf9fde15d1b6e572db0949e22c319eb2420d23c1e4572'}, 'models/candidate_evaluation.py': {'bytes': 9732, 'sha256': '26510d6bd5557421fc10bcdc28dd4b5c553b69c3a9914a3cd80457d5bc746a30'}, 'models/checkpointing.py': {'bytes': 6645, 'sha256': 'edc5940bc73db42c89b3494be9b0f2098680762f4fa972305f51abe47c3d8373'}, 'models/cuda_convolution.py': {'bytes': 24713, 'sha256': '6254b621ebb8aa28d44616d8527a930f042f83a786eeb468137a58d1b76ff6e3'}, 'models/cuda_profile.py': {'bytes': 23812, 'sha256': 'eac91d88a688ec073a3a965e656d88cc1b28a6b446279bc4b77196ee104990d0'}, 'models/diagnostics.py': {'bytes': 5555, 'sha256': '12185169ca1ebe2564bc0b1815ec19b8599c08f625af056bc30c25fca6f390d0'}, 'models/eot_profile_packet.py': {'bytes': 5740, 'sha256': 'e2ed28d35d428e85a2c9f7b447c66de77b6917c0f0c32cfff4d169dbf142aa17'}, 'models/evaluation.py': {'bytes': 8544, 'sha256': '1431951d158b57b615208006edb5365a6bad910b2a14e0aa782a1c6f82034089'}, 'models/evaluation_store.py': {'bytes': 9252, 'sha256': '9503834d106b788c1638b4a57d886259e2cf6f6be5069b18a1105e8beac48ee1'}, 'models/execution.py': {'bytes': 12043, 'sha256': 'e665f7c842c0d26733bd6c4d16887b10e2590b76f15e4fb1981b304ff37903ba'}, 'models/fixture_state.py': {'bytes': 4418, 'sha256': '4f1d891a26caf35a9c42f802e84f650e2e0c7dc0857369794bfef65e54c487d2'}, 'models/gradient_attribution.py': {'bytes': 21684, 'sha256': 'f8bd9d39dd41c61ffadd8ac882d08a36bb2513a948047632a789c8d0900414d3'}, 'models/native_diagnostic.py': {'bytes': 20448, 'sha256': 'c56b29496397d5516416a884ab591ec6de43e0fc4dca1bc05536234e10ea75d9'}, 'models/native_patch.py': {'bytes': 3250, 'sha256': '8f5ede9ea8ba61773ca11dffac5a2bdba01a62affc497d8327b3e1ede31fa5e7'}, 'models/native_probe.py': {'bytes': 5340, 'sha256': '10dd9afd62615106aa298c1970278dcd9f203979c8006bc150ecbdc0228db762'}, 'models/patch_embed.py': {'bytes': 1827, 'sha256': '64838d575479e067ffecb8352efa7b51fcfafeabeb234e295605d87c1033b0ec'}, 'models/pilot.py': {'bytes': 2119, 'sha256': 'a8e947c5ce58c6497b653b3bdc9ac1051b045d07fe3f280f9c04daf44858ae82'}, 'models/pilot_bundle.py': {'bytes': 9169, 'sha256': 'd25f0e1f085bb7addcfc855986c7bf984765990cb473be9d5315f5399696a9db'}, 'models/precision_diagnostic.py': {'bytes': 25361, 'sha256': 'c096e38bf53c4cdc0273a74f3fd0e4be0c1fc197925a71cfa9d633d192bb7d00'}, 'models/precision_patch.py': {'bytes': 7991, 'sha256': '30e8a6b28b8f082fd771aaeab5a70a73cf200714ef2b24416a55938f61e9c550'}, 'models/predictor_eot8_adapter_v1.py': {'bytes': 7839, 'sha256': '89b7e0c67e88f2f776bb0bb6ad786a2202115e64ffc6c6f9a962381cc2e9a6b8'}, 'models/predictor_eot8_owned_v1.py': {'bytes': 16089, 'sha256': 'ef3675e69ef56b0b4ef4f543cab96b7f2bc757a64202d74500f3802ccd5dfc12'}, 'models/predictor_forward_preserving_cpu_v1.py': {'bytes': 13654, 'sha256': '6ba308e948c0ce346512f17ef4fc11eccb542954048c859ad6c6a751db1fbb66'}, 'models/predictor_forward_preserving_eot8_cpu_v1.py': {'bytes': 10782, 'sha256': '256079682d27aa2be42b0c909298a5adb3632cc385e95aa3590d2724b75a1a9e'}, 'models/predictor_forward_preserving_upstream_v1.py': {'bytes': 10600, 'sha256': 'db71de4b6b8c3ba1532639258f23e7b4b5a850493625bf6ba1a7e2f8c9b13b88'}, 'models/predictor_forward_preserving_workload_v1.py': {'bytes': 13089, 'sha256': '735dd2fd40120637c5b360da9057b707af12d2ecd5acb74773f23f0f52f304d8'}, 'models/predictor_pgd_workload_records_v1.py': {'bytes': 6488, 'sha256': '6aede713697bce853c5d09e92b5e9c031be4c6398828961d1ef362922c708e0e'}, 'models/predictor_pgd_workload_runtime_records_v1.py': {'bytes': 6943, 'sha256': '20a54a1d1c2caa9da967b707c5530710b239654b21d5e1310ff2fb63594c6946'}, 'models/predictor_pgd_workload_target_evaluation_v1.py': {'bytes': 5661, 'sha256': 'e6844c26a6881ccddcab86a8b6316232437460a450bb0ed47fa5b4886b5c1ef8'}, 'models/predictor_pgd_workload_target_v1.py': {'bytes': 25290, 'sha256': 'e3d45b6ca52bbc4a490f963e556fcfede3c8a8f0660dc0629713c4bf28d262a4'}, 'models/projection_reuse.py': {'bytes': 6156, 'sha256': '28bd626d16831f9b30e4c83262f60c2486c9c08ffb8b94c485e0eb6214e36061'}, 'models/reuse_diagnostic.py': {'bytes': 34024, 'sha256': '84cace760315f9aeed601d241314f8da884b4159e33d063463ff165a552ac082'}, 'models/reuse_execution.py': {'bytes': 5944, 'sha256': '7c0014e467277a954e9d739b89397f70d5a9ee4fcd9332c943dfa9aab2aa4c88'}, 'models/reuse_handoff.py': {'bytes': 7284, 'sha256': 'abed51af6f3fb8560b412ec49a2435cc82efd151faad99c0f24a3c7765f7d581'}, 'models/runtime.py': {'bytes': 4827, 'sha256': '0764f5c63384882e06f1d409535e96b3423861866c16bc8ba5e3fa71955f83de'}, 'models/study_artifacts.py': {'bytes': 1011, 'sha256': 'c9ace85300c4efcc62e14bf60a89a4c22415497ea8a5ee5f8e3f12230ac8fcad'}, 'models/training.py': {'bytes': 16803, 'sha256': '15e280dfcc409a94f36defa42c59304e96dc198c8e4e063a9da6f4adf4f03679'}, 'models/vjepa2.py': {'bytes': 12926, 'sha256': '5501eb4ce79317d8ebab591d7fe4d8d0b74b1b5b692d2f7513d8c583d188debf'}}
CAPSULE_FILES = {'boundary/attention-calls.jsonl': {'bytes': 949546, 'sha256': '6f2ccd095b107eeaf59ba872be04de80abdfe348f4495243f1c8f19da293e126'}, 'boundary/attention-scopes.json': {'bytes': 312, 'sha256': 'fe85b6c1e54097cae93a1891d9ced17f4694f8de4f4754cb398fd1b33ab9d043'}, 'boundary/boundary-complete.json': {'bytes': 273, 'sha256': '8a7233c4138b87eca2b16b3e28c6412d2d664bd221c50cb494d8569e89bf524e'}, 'boundary/boundary.json': {'bytes': 9234, 'sha256': '9e4758d2209b9fd989926ee40d2ad92cc4b99aa98c86e6cec0722eaf6a83308d'}, 'boundary/dk.pt': {'bytes': 3147305, 'sha256': '00288e2c75abd4eef65a1e97b00f4fe25546ab20df640702375a5d7d2d15bf8c'}, 'boundary/dq.pt': {'bytes': 3147305, 'sha256': '6973339c06e07ba92e2c79f04aee4bbb982a3642cc4895d6c974c30840c15b09'}, 'boundary/dv.pt': {'bytes': 3147305, 'sha256': 'deead61cbb8cf5b6f7462219907082b6cc17db99af67734e170d6c3b35d9ffa5'}, 'boundary/grad_output.pt': {'bytes': 3147305, 'sha256': 'a673aabb28ca9e181c2426ee4ce81062b1fcecdc0609d31754996efdf45fcb7c'}, 'boundary/k.pt': {'bytes': 3147305, 'sha256': 'f80ea46c5e6bea6b117763c3551a5320b92aa45f65916d2c451553d2083aad37'}, 'boundary/output.pt': {'bytes': 3147305, 'sha256': '33cf9f63b54b285ee5c6fa1ed3dac252f7edcb40c4f5aa3a9d574ba686c8209e'}, 'boundary/q.pt': {'bytes': 3147305, 'sha256': 'eede3a77598c212faa77ca935e61622aa698fd4ffdad658e717cd80fed968866'}, 'boundary/v.pt': {'bytes': 3147305, 'sha256': '8881fa0dcbc8b653e5547ca2ef5533e3bfb436cfdabfe5025dd437792a7e30da'}, 'provenance/accepted-numerical-review.json': {'bytes': 7914, 'sha256': '13d2a40d2a989d9433f042dcf59187f84850ea8294e4ad8f494accda00074c8b'}, 'provenance/objective-0/VJP-calls.jsonl': {'bytes': 0, 'sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'}, 'provenance/objective-0/native-calls.jsonl': {'bytes': 949546, 'sha256': '6f2ccd095b107eeaf59ba872be04de80abdfe348f4495243f1c8f19da293e126'}, 'provenance/objective-0/numerical-complete.json': {'bytes': 5105, 'sha256': '9c688ffa5a5e34c0c3a2e6d13fb2b74780b443e2d893b35e52db045f9c677cbd'}, 'provenance/objective-0/numerical.json': {'bytes': 9661, 'sha256': '348f5ee5843a42a63c924ce19944228ff1cb9aed4bf288876de6a7298d8921e2'}, 'provenance/objective-0/scopes.json': {'bytes': 328, 'sha256': '539deace994dc1809bd5265c69be4a91d037f58440c73606aaf658e8b18d0b58'}, 'provenance/objective-1/numerical-complete.json': {'bytes': 5111, 'sha256': '22761fc13f36d293356e6662d7157adbb308f50fc1cbc251518e9020041d5e83'}, 'provenance/objective-1/numerical.json': {'bytes': 10130, 'sha256': '8af887389465c399cfb0e1b1bcf5b9e24a4fe2431c2cfad0cccf49838bdea1dc'}, 'provenance/objective-2/numerical-complete.json': {'bytes': 5105, 'sha256': 'b7077961078735f1407b8ce265ae9507ba33719ed186b606e71030431fc46fa9'}, 'provenance/objective-2/numerical.json': {'bytes': 9661, 'sha256': '8f8dbd56c57b7d8ed5b1e8eff6b62a95c880a3151c1b1fa0237d457200fe0a66'}, 'provenance/objective-3/numerical-complete.json': {'bytes': 5111, 'sha256': 'fafb91946d2ca6b12b241bbde81ca9f4089eb8996cd12f1a86661d978e822908'}, 'provenance/objective-3/numerical.json': {'bytes': 10131, 'sha256': '183023510653f86119b51811e1403cca9a7f1aa3db588afd444fe1592367a624'}, 'provenance/objective-4/numerical-complete.json': {'bytes': 5105, 'sha256': '9a5a3fef8ce4ba4bdd961215774ae235e2fc2f09cd7a4272abff5c8e1ee06ef8'}, 'provenance/objective-4/numerical.json': {'bytes': 9661, 'sha256': 'aa9fba339232bab8fac9137bd16c38e5cea4d6a63bef95f8f82a9bff8b59fc5d'}, 'provenance/objective-5/numerical-complete.json': {'bytes': 5111, 'sha256': 'bca25f28404c198eadeaa0471eba65e1499c153269a3936e154f641ec7f7ef5c'}, 'provenance/objective-5/numerical.json': {'bytes': 10130, 'sha256': 'b27fc9fe8d08935d1b954350a39865a0b5250d8f42a022176f848a26c22cb376'}, 'provenance/objective-6/numerical-complete.json': {'bytes': 5105, 'sha256': '277e8292d0d478482ba5385a5fbd722720f663988352624e3fc3686be5779119'}, 'provenance/objective-6/numerical.json': {'bytes': 9661, 'sha256': 'f9f58ec5ed398318f7ae32fb59e465afa288c6c38b9347904e088f81c01df7d3'}, 'provenance/objective-7/numerical-complete.json': {'bytes': 5111, 'sha256': '812ed32dda099acf9b175d6c8e11613e02357dbae93ff16e2905bb5bb7eb465e'}, 'provenance/objective-7/numerical.json': {'bytes': 10131, 'sha256': 'e260795efb3902fce97b860e5f90f58dc8b95a29f2d2f0e71cc51e81adf4be10'}, 'provenance/parent-numerical.json': {'bytes': 481, 'sha256': 'c422f69479a94734b9ec7c3d3065335d0c53bd440907935a79cc51af43856761'}, 'provenance/run-status.json': {'bytes': 13337, 'sha256': '56a6f931c8c9a4eb5367ef85244f497d8584eccd484a004190d1c459c2b5553d'}, 'provenance/session.json': {'bytes': 5177, 'sha256': '6cd4d8b18cd2a542ffb6c7c1285611e2ff7ab2cae3cc2fa7c53c248e1ec68697'}, 'MANIFEST.json': {'bytes': 13480, 'sha256': 'a7e6740c69f5f3bf8c5fc6b71683296f2c9ebc818b644814b2f37eaf983c0e24'}}
SUPERVISOR_NAMES = ('predictor_pgd_split_watch_v1.py', 'scratch_allocation_v2.py', 'workload_watch_v1.py', 'workload_os_v1.py', 'workload_plan_v1.py', 'bootstrap.py', 'predictor_pgd_split_scratch_v1.py', 'predictor_pgd_split_coordinator_v1.py', 'predictor_pgd_split_collection_v1.py')

REFERENCE_PATH = 'runs/pretrained/diagnostic/pgd-candidate-completion-v1-20261120/native-reference-v1.tar'
REFERENCE_MANIFEST_SHA = '3dc2484c07c04bb72593c2ee386824edf52b8d80ec0fe3bb803e645714e5d932'
REFERENCE_FILES = {'native/backend.json': {'bytes': 3268, 'sha256': '4bd060d01524a8381c0dc978fd0ff248d3f1e25742df0c581b40ef47ddaa9bc5'}, 'native/complete.json': {'bytes': 632, 'sha256': '51407b1c1569e6e870695b0f2ddaa6d79bb0a20a69b3d59ee3181b84e91a6b98'}, 'native/dk.npy': {'bytes': 3145856, 'sha256': 'c31f6d5017e237d9488ad8ba96f745ddbcf1e5e21eb5224d618c7998f92fb49b'}, 'native/dq.npy': {'bytes': 3145856, 'sha256': '218dc436f9795ef1014e910c990534eb75a06676b28145595964f0653d864afb'}, 'native/dv.npy': {'bytes': 3145856, 'sha256': '46e728741ea905d43a9ebb3e0ea44f9bd7e0bf7750b6b5641e81b8f6e4d09134'}, 'native/events.json': {'bytes': 569123, 'sha256': '97139d0db676bf14e2e14a0b294611687cec3e0860279e4755067c1c13b2c916'}, 'native/output.npy': {'bytes': 3145856, 'sha256': '47f0ac9fbba30e14e9b82011f062596c32c05c6fa152aa89d5c1168e73b1c618'}, 'native/ownership.json': {'bytes': 2845, 'sha256': '2bae7b2feb0c26f3238db3b07dfbee04114e2f5a77ebc7b191e3aa24f7cf2536'}, 'native/parent.json': {'bytes': 435, 'sha256': 'cc57544291e2896af1996a1c05e20715fa9bde424c8b705c84c1a067198cb12e'}, 'native/progress.jsonl': {'bytes': 29650, 'sha256': 'e13db72771087f38fb6cdbc6e7fa87a70f3b2fb03b80d35739ae0c0ecb82a134'}, 'native/resource.json': {'bytes': 1197, 'sha256': 'aa5917e82ed385bd748fa878cf97e30a3b163f2b6dc7b49c248396bd03e9aa36'}, 'native/worker.json': {'bytes': 6329, 'sha256': '6f1cdea1d114cc605103c645cd379f3c12867ae88855c211e97d0fd26e3fea29'}, 'reviews/component.json': {'bytes': 9305, 'sha256': '45318db20a419f3f2671be590b3b3fe218f35a2a2ca0f935a37de727eceaa3f2'}, 'reviews/metadata.json': {'bytes': 8346, 'sha256': '332667b3c9a6902464f75a920fb8c45bed5c37fc0387e4e9e45554de257dc56b'}, 'reviews/pgd.json': {'bytes': 7914, 'sha256': '13d2a40d2a989d9433f042dcf59187f84850ea8294e4ad8f494accda00074c8b'}, 'reviews/values.json': {'bytes': 7963, 'sha256': '1c632f09f34e05e37516df19c27d855780738c1208c42e3691890665f9622b53'}, 'source/boundary.json': {'bytes': 9234, 'sha256': '9e4758d2209b9fd989926ee40d2ad92cc4b99aa98c86e6cec0722eaf6a83308d'}, 'source/capsule-manifest.json': {'bytes': 13480, 'sha256': 'a7e6740c69f5f3bf8c5fc6b71683296f2c9ebc818b644814b2f37eaf983c0e24'}, 'MANIFEST.json': {'bytes': 3289, 'sha256': '3dc2484c07c04bb72593c2ee386824edf52b8d80ec0fe3bb803e645714e5d932'}}

def receive_reference(*a,**kw):
    raise ValueError("No component reference archive belongs to a full-unit workload.")

def extract_reference(*a,**kw):
    raise ValueError("No component reference archive belongs to a full-unit workload.")
