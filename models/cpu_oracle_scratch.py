"""Standard-library supervisor for a new, explicitly authorized CPU scratch run.

The archived numerical probe is not modified. Filesystem polling and per-file
limits are safeguards, not an exclusive disk reservation or a kernel quota.
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import resource
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time

KIND = 'cpu-oracle-scratch-recovery-v1'
MIB = 2**20
GIB = 2**30
POLICY = {
    'max_seconds': 600, 'worker_seconds': 90, 'automatic_retries': 0,
    'write_test_bytes': 5 * GIB // 2, 'write_test_seconds': 60,
    'minimum_free_bytes': 4 * GIB, 'minimum_free_inodes': 2048,
    'runtime_free_floor': GIB, 'capture_ceiling': 9 * GIB // 4,
    'setup_ceiling': 3 * GIB, 'max_input_bytes': 64 * MIB,
    'max_code_bytes': 64 * MIB, 'max_raw_bytes': GIB,
    'max_archive_bytes': GIB + 8 * MIB, 'max_metadata_bytes': 64 * MIB,
    'emergency_bytes': 16 * MIB, 'max_array_bytes': 32 * MIB,
    'max_files': 1024, 'poll_seconds': 0.25,
    'max_collection_bytes': 3 * GIB, 'collection_seconds': 600,
    'max_header_bytes': MIB, 'max_launcher_log_bytes': 16 * MIB,
}
HOSTS = ('linux10603', 'linux10609')
ARCHIVE_SHA = 'd6b02dc362be2a3a5244174bc2d1175f763d1f661d5cc89ab2e18afa17e63f53'
HANDOFF_SHA = '952ee2e1a9e45914b36557d0d3e5700ca2c1c0efb10ded675a2e443723ddb4b5'
TASKS = [[4, 0], [1, 0], [4, 1], [1, 1]]


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def safe_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name or '\\' in name or name == '.':
        raise ValueError('Unsafe evidence member name.')
    return name


def write_json(path, value, *, replace=False):
    path = Path(path)
    data = json_bytes(value)
    if len(data) > POLICY['max_header_bytes']:
        raise ValueError('Operational JSON exceeds bound.')
    part = path.with_name(path.name + '.partial')
    with part.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if replace:
        os.replace(part, path)
    else:
        os.link(part, path)
        part.unlink()


def validate_control(control):
    if (control.get('kind') != KIND or control.get('host') not in HOSTS
            or control.get('policy') != POLICY or control.get('parent') != '/var/tmp'
            or control.get('expected_filesystem') != 'xfs' or control.get('expected_device') != 66307
            or control.get('expected_mount') != '/' or control.get('CPU_only') is not True
            or any(control.get(k) is not False for k in ('GPU_work', 'paid_compute', 'research_admitted'))
            or control.get('archive_sha256') != ARCHIVE_SHA or control.get('handoff_sha256') != HANDOFF_SHA
            or control.get('tasks') != TASKS or not 0 < control.get('archive_bytes', 0) <= POLICY['max_input_bytes']):
        raise ValueError('Changed host, source, task, storage or resource envelope.')
    specification = control['archive_manifest']
    if specification['sha256'] != ARCHIVE_SHA or specification['bytes'] != control['archive_bytes']:
        raise ValueError('Archive specification does not bind the original packet.')
    if len(specification['files']) != 129 or sum(i['bytes'] for i in specification['files'].values()) > POLICY['max_code_bytes']:
        raise ValueError('Unexpected original packet coverage/extent.')
    for name, item in specification['files'].items():
        safe_name(name)
        if type(item['bytes']) is not int or item['bytes'] < 0 or len(item['sha256']) != 64:
            raise ValueError('Invalid original packet member.')


def validate_authorization(authorization, control_sha, host):
    if (authorization.get('kind') != 'cpu-oracle-scratch-resource-exception-v1'
            or authorization.get('approved') is not True or authorization.get('control_sha256') != control_sha
            or authorization.get('host') != host or authorization.get('maximum_new_attempts') != 1
            or authorization.get('max_seconds') != 600 or authorization.get('write_test_bytes') != POLICY['write_test_bytes']
            or authorization.get('CPU_only') is not True or authorization.get('paid_compute') is not False
            or authorization.get('reset_old_claims') is not False or not authorization.get('user_basis')):
        raise ValueError('A separate explicit CPU scratch resource exception is required.')


def mount_for(path, text):
    # The caller verifies the selected path is already resolved. Do not apply
    # the review machine's symlinks when interpreting a captured mount table.
    path = str(Path(path).absolute())
    matches = []
    for line in text.splitlines():
        left, separator, right = line.partition(' - ')
        if not separator:
            continue
        fields, tail = left.split(), right.split()
        if len(fields) < 6 or len(tail) < 3:
            continue
        point = fields[4]
        for escaped, actual in (('\\040', ' '), ('\\011', '\t'), ('\\134', '\\')):
            point = point.replace(escaped, actual)
        if path == point or path.startswith(point.rstrip('/') + '/'):
            matches.append({'mount': point, 'filesystem': tail[0], 'source': tail[1],
                            'options': fields[5].split(',') + tail[2].split(',')})
    if not matches:
        raise ValueError('No mount identity for selected scratch parent.')
    return max(matches, key=lambda item: len(item['mount']))


def readiness(control):
    validate_control(control)
    if platform.node() != control['host'] or sys.platform != 'linux':
        raise ValueError('Unexpected CPU target host or operating system.')
    parent = Path(control['parent'])
    home = Path.home()
    if parent.is_symlink() or parent.resolve() != parent or not parent.is_dir():
        raise ValueError('Scratch parent changed or is a symlink.')
    info = parent.stat()
    mount = mount_for(parent, Path('/proc/self/mountinfo').read_text())
    capacity = os.statvfs(parent)
    result = {'host': platform.node(), 'parent': str(parent), 'device': info.st_dev,
        'inode': info.st_ino, 'home_device': home.stat().st_dev,
        'mode': stat.S_IMODE(info.st_mode), 'owner_uid': info.st_uid, **mount,
        'available_bytes': capacity.f_bavail * capacity.f_frsize,
        'available_inodes': capacity.f_favail,
        'writable_for_current_user': os.access(parent, os.W_OK | os.X_OK),
        'quota_reservation': False}
    validate_readiness(result, control)
    return result


def validate_readiness(result, control):
    if (result['parent'] != control['parent'] or result['host'] != control['host']
            or result['device'] != control['expected_device'] or result['device'] == result['home_device']
            or result['mount'] != control['expected_mount'] or result['filesystem'] != 'xfs'
            or 'ro' in result['options'] or result['owner_uid'] != 0 or result['mode'] != 0o1777
            or not result['writable_for_current_user'] or result['available_bytes'] < POLICY['minimum_free_bytes']
            or result['available_inodes'] < POLICY['minimum_free_inodes']):
        raise ValueError('Scratch mount, ownership, available space or inode gate failed.')


def full_write_test(folder, size, deadline, *, random_bytes=os.urandom):
    """Actual writes and readback; only the successful new test file is removed."""
    path = Path(folder) / 'space-proof.bin.partial'
    expected = hashlib.sha256()
    with path.open('xb') as handle:
        remaining = size
        while remaining:
            if time.monotonic() >= deadline:
                raise TimeoutError('Full-size scratch write test deadline reached.')
            block = random_bytes(min(4 * MIB, remaining))
            if not block or len(block) > remaining:
                raise ValueError('Invalid bounded proof block.')
            handle.write(block)
            expected.update(block)
            remaining -= len(block)
        handle.flush()
        os.fsync(handle.fileno())
        written = os.fstat(handle.fileno())
    actual = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * MIB), b''):
            if time.monotonic() >= deadline:
                raise TimeoutError('Scratch proof readback deadline reached.')
            actual.update(block)
    if (written.st_size != size or written.st_blocks * 512 < size
            or actual.hexdigest() != expected.hexdigest()):
        raise ValueError('Full-size write/readback or allocated-block proof failed.')
    result = {'bytes_written': size, 'bytes_read_back': size, 'sha256': actual.hexdigest(),
              'allocated_bytes': written.st_blocks * 512, 'fsync_succeeded': True,
              'reservation': False, 'test_file_removed': True}
    path.unlink()
    return result


def receive_archive(stream, path, expected_bytes, expected_sha, deadline):
    part = Path(path).with_name(Path(path).name + '.partial')
    hashed = hashlib.sha256()
    with part.open('xb') as handle:
        remaining = expected_bytes
        while remaining:
            if time.monotonic() >= deadline:
                raise TimeoutError('Input transfer deadline reached.')
            block = stream.read(min(MIB, remaining))
            if not block:
                raise ValueError('Input archive stream ended early.')
            handle.write(block)
            hashed.update(block)
            remaining -= len(block)
        handle.flush()
        os.fsync(handle.fileno())
    if hashed.hexdigest() != expected_sha:
        raise ValueError('Original CPU archive checksum changed.')
    os.link(part, path)
    part.unlink()


def extract_packet(archive, specification, destination):
    destination = Path(destination)
    if destination.exists() or digest(archive) != specification['sha256']:
        raise ValueError('Archive changed or scratch code directory is not fresh.')
    with tarfile.open(archive) as source:
        members = source.getmembers()
        names = [safe_name(item.name) for item in members]
        if (len(names) != len(set(names)) or set(names) != set(specification['files'])
                or len(names) > 512 or any(not item.isfile() for item in members)):
            raise ValueError('Unexpected or non-regular CPU archive member.')
        for item in members:
            expected = specification['files'][item.name]
            if item.size != expected['bytes']:
                raise ValueError('CPU packet member extent changed.')
            with source.extractfile(item) as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != expected['sha256']:
                    raise ValueError('CPU packet member checksum changed.')
        destination.mkdir(mode=0o700)
        for item in members:
            path = destination / item.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(item) as incoming, path.open('xb') as outgoing:
                for block in iter(lambda: incoming.read(MIB), b''):
                    outgoing.write(block)
    if digest(destination / 'CPU-ORACLE-HANDOFF.json') != HANDOFF_SHA:
        raise ValueError('CPU handoff identity changed.')


def tree_inventory(folder, *, hashing=False, max_bytes=None, running=False):
    folder = Path(folder)
    root = folder.stat()
    files = {}
    physical = 0
    seen = set()
    for current, dirs, names in os.walk(folder, followlinks=False):
        for name in dirs + names:
            path = Path(current) / name
            try:
                info = path.lstat()
            except FileNotFoundError:
                if running:
                    continue  # Atomic writer replaced a temporary file.
                raise
            if info.st_dev != root.st_dev or stat.S_ISLNK(info.st_mode):
                raise ValueError('Scratch path changed device or introduced a symlink.')
            if stat.S_ISDIR(info.st_mode):
                continue
            paired_array_link = False
            if info.st_nlink == 2 and (name.endswith('.npz') or name.endswith('.npz.partial')):
                partner = path.with_name(name[:-8] if name.endswith('.partial') else name + '.partial')
                try:
                    partner_info = partner.lstat()
                    paired_array_link = (stat.S_ISREG(partner_info.st_mode)
                        and (partner_info.st_dev, partner_info.st_ino) == (info.st_dev, info.st_ino))
                except FileNotFoundError:
                    paired_array_link = running and path.stat().st_nlink == 1
            if not stat.S_ISREG(info.st_mode) or (info.st_nlink != 1 and not paired_array_link):
                # Preserve even a crash between the frozen writer's link and
                # unlink, but never accept an unrelated/outside hardlink.
                raise ValueError('Non-regular or multiply linked scratch file.')
            relative = safe_name(str(path.relative_to(folder)))
            item = {'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns}
            if hashing:
                item['sha256'] = digest(path)
                after = path.stat()
                if (info.st_ino, info.st_size, info.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                    raise ValueError('Evidence changed while hashing.')
            files[relative] = item
            if (info.st_dev, info.st_ino) not in seen:
                physical += info.st_blocks * 512
                seen.add((info.st_dev, info.st_ino))
            if len(files) > POLICY['max_files'] or (max_bytes is not None and sum(i['bytes'] for i in files.values()) > max_bytes):
                raise ValueError('Scratch inventory exceeds file-count or byte bound.')
    return files, physical


def storage_gate(folder, parent_identity, *, parent_path=Path('/var/tmp')):
    folder = Path(folder)
    current = Path(parent_path).stat()
    info = folder.stat()
    if ((current.st_dev, current.st_ino) != (parent_identity['device'], parent_identity['inode'])
            or info.st_dev != current.st_dev or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700 or folder.is_symlink()):
        raise ValueError('Private scratch directory or parent identity changed.')
    capacity = os.statvfs(folder)
    if capacity.f_bavail * capacity.f_frsize < POLICY['runtime_free_floor'] or capacity.f_favail < 1024:
        raise ValueError('Scratch available-space/inode floor reached.')
    files, allocated = tree_inventory(folder, max_bytes=POLICY['capture_ceiling'], running=True)
    categories = {'input': 0, 'code': 0, 'raw': 0, 'archive': 0, 'metadata': 0, 'emergency': 0}
    for name, item in files.items():
        size = item['bytes']
        if name == 'original-packet.tar.gz':
            category = 'input'
        elif name.startswith('code/'):
            category = 'code'
        elif name == 'emergency-space.bin':
            category = 'emergency'
        elif name == 'results/evidence.tar.gz':
            category = 'archive'
        elif name.startswith('results/') and (name.endswith('.npz') or name.endswith('.npz.partial')):
            category = 'raw'
            if size > POLICY['max_array_bytes']:
                raise ValueError('Captured array file exceeds its bound.')
        else:
            category = 'metadata'
        categories[category] += size
    limits = {'input': 'max_input_bytes', 'code': 'max_code_bytes', 'raw': 'max_raw_bytes',
              'archive': 'max_archive_bytes', 'metadata': 'max_metadata_bytes', 'emergency': 'emergency_bytes'}
    if any(categories[name] > POLICY[key] for name, key in limits.items()) or allocated > POLICY['capture_ceiling']:
        raise ValueError('Scratch category or allocated-byte ceiling reached.')
    return {'logical_bytes': sum(i['bytes'] for i in files.values()), 'allocated_bytes': allocated,
            'available_bytes': capacity.f_bavail * capacity.f_frsize, 'files': len(files), 'categories': categories}


def process_identity(pid, proc=Path('/proc')):
    try:
        data = (proc / str(pid) / 'stat').read_text()
        fields = data[data.rfind(')') + 2:].split()
        return int(fields[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def descendants(pid, proc=Path('/proc')):
    result = {}
    queue = [pid]
    while queue:
        parent = queue.pop()
        try:
            values = (proc / str(parent) / 'task' / str(parent) / 'children').read_text().split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        for value in values:
            child = int(value)
            identity = process_identity(child, proc)
            if child != pid and identity is not None and child not in result:
                result[child] = identity
                queue.append(child)
        if len(result) > 64:
            raise ValueError('Unexpected descendant process count.')
    return result


def terminate_owned(child, tracked):
    tracked.update(descendants(child.pid))
    if child.poll() is None:
        child.terminate()  # Original parent handles TERM and kills its worker.
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            child.kill()
    child.wait()
    killed = []
    for pid, identity in tracked.items():
        if process_identity(pid) == identity:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except ProcessLookupError:
                pass
    return killed


def child_limits():
    # Original NPZ writer also refuses arrays over 32 MiB. This kernel cap
    # bounds an individual export/log; aggregate use has separate supervision.
    resource.setrlimit(resource.RLIMIT_FSIZE, (POLICY['max_archive_bytes'], POLICY['max_archive_bytes']))
    os.umask(0o077)


def supervise(command, folder, deadline, gate):
    """Kill only this launch and identities observed as its descendants."""
    folder = Path(folder)
    tracked = {}
    snapshots = []
    child = None
    started = time.monotonic()
    try:
        gate()
        if deadline - time.monotonic() <= 95:
            raise TimeoutError('Insufficient original time remains before any child can start.')
        with (folder / 'launcher.log').open('xb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, preexec_fn=child_limits,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
                     'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
            while child.poll() is None:
                tracked.update(descendants(child.pid))
                if time.monotonic() >= deadline - 5:
                    raise TimeoutError('Original scratch deadline reached; no retry.')
                if os.fstat(log.fileno()).st_size > POLICY['max_launcher_log_bytes']:
                    raise ValueError('Launcher log exceeds its bound.')
                tick = time.monotonic()
                snapshot = gate()
                if len(snapshots) < 2400:
                    snapshots.append(snapshot)
                time.sleep(max(0, tick + POLICY['poll_seconds'] - time.monotonic()))
            code = child.returncode
        if any(process_identity(pid) == identity for pid, identity in tracked.items()):
            raise ValueError('A launch descendant remains after the CPU parent exits.')
        final = gate()
        return {'exit_status': code, 'wall_seconds': time.monotonic() - started,
                'storage_samples': len(snapshots), 'peak_logical_bytes': max([s.get('logical_bytes', 0) for s in snapshots] + [final.get('logical_bytes', 0)]),
                'peak_allocated_bytes': max([s.get('allocated_bytes', 0) for s in snapshots] + [final.get('allocated_bytes', 0)]),
                'final_storage': final, 'polling_is_a_hard_kernel_quota': False,
                'individual_file_kernel_limit_bytes': POLICY['max_archive_bytes']}
    finally:
        if child is not None:
            terminate_owned(child, tracked)


def make_plan(control, handoff):
    return {'kind': 'cpu-oracle-attended-plan-v1', 'expected_host': control['host'],
        'authorized': True, 'attended': True, 'CPU_only': True, 'GPU_work_allowed': False,
        'paid_compute': False, 'research_admitted': False, 'old_refusals_waived': False,
        'prior_consumed_claims': handoff['prior_consumed_claims'], 'handoff_sha256': HANDOFF_SHA,
        'tasks': TASKS, **handoff['limits'], 'outer_scratch_control_sha256': control['_control_sha256'],
        'authorization_basis': 'Separate explicit scratch CPU resource exception; old attempts stay consumed.'}


def result_stream(folder, control_sha, output):
    """Stream existing files; no extra remote archive or result regeneration."""
    folder = Path(folder)
    files, _ = tree_inventory(folder, hashing=True, max_bytes=POLICY['max_collection_bytes'])
    header = {'phase': 'collection', 'kind': KIND, 'control_sha256': control_sha,
              'files': files, 'logical_bytes': sum(i['bytes'] for i in files.values()),
              'scientific_review_complete': False, 'model_restarted_for_collection': False}
    data = json_bytes(header)
    if len(data) > POLICY['max_header_bytes']:
        raise ValueError('Collection inventory exceeds its bound.')
    output.write(data)
    output.flush()
    with tarfile.open(mode='w|', fileobj=output) as archive:
        for name, expected in sorted(files.items()):
            path = folder / name
            if path.is_symlink() or digest(path) != expected['sha256']:
                raise ValueError('Collection member changed before streaming.')
            info = tarfile.TarInfo(name)
            info.size = expected['bytes']
            info.mode = 0o600
            info.mtime = expected['mtime_ns'] // 10**9
            with path.open('rb') as handle:
                archive.addfile(info, handle)
            if digest(path) != expected['sha256']:
                raise ValueError('Collection member changed while streaming.')


def run_remote(control, authorization, incoming, output):
    validate_control(control)
    control_sha = hashlib.sha256(json_bytes(control)).hexdigest()
    validate_authorization(authorization, control_sha, control['host'])
    identity = readiness(control)
    started_unix, started = time.time(), time.monotonic()
    deadline = started + POLICY['max_seconds']
    scratch = Path(tempfile.mkdtemp(prefix='vjepa-cpu-oracle-20261104-' + control['host'] + '-', dir=control['parent']))
    os.chmod(scratch, 0o700)
    output.write(json_bytes({'phase': 'scratch-created', 'kind': KIND, 'host': control['host'],
        'control_sha256': control_sha, 'scratch_directory': str(scratch), 'started_unix': started_unix}))
    output.flush()
    result = {'kind': KIND, 'control_sha256': control_sha, 'host': control['host'],
              'run_phase_finished': False, 'reference_process_started': False,
              'CPU_only': True, 'GPU_work': False, 'paid_compute': False,
              'scratch_directory': str(scratch), 'started_unix': started_unix,
              'deadline_unix': started_unix + 600, 'capture_integrity_verified': False,
              'scientific_review_complete': False, 'old_refusals_waived': False}
    def interrupted(*unused):
        raise RuntimeError('Attended scratch session interrupted; preserve files, no retry.')
    for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, 595)
    try:
        write_json(scratch / 'operator-control.json', control)
        write_json(scratch / 'resource-authorization.json', authorization)
        write_json(scratch / 'storage-before.json', identity)
        with (scratch / 'emergency-space.bin').open('xb') as handle:
            handle.write(os.urandom(POLICY['emergency_bytes']))
            handle.flush()
            os.fsync(handle.fileno())
        print('Checking 2.5 GiB of local scratch writes and readback; no reference call yet.', file=sys.stderr, flush=True)
        proof_deadline = min(deadline - 5, time.monotonic() + 60)
        signal.setitimer(signal.ITIMER_REAL, max(.001, proof_deadline - time.monotonic()))
        try:
            proof = full_write_test(scratch, POLICY['write_test_bytes'], proof_deadline)
        finally:
            signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic() - 5))
        write_json(scratch / 'storage-write-proof.json', proof)
        result['write_test_complete'] = True
        receive_archive(incoming, scratch / 'original-packet.tar.gz', control['archive_bytes'], ARCHIVE_SHA, deadline - 5)
        extract_packet(scratch / 'original-packet.tar.gz', control['archive_manifest'], scratch / 'code')
        if time.time() - started_unix > 110:
            raise TimeoutError('Setup leaves no margin for original clock freshness gate.')
        gate = lambda: storage_gate(scratch, identity)
        write_json(scratch / 'storage-after-setup.json', gate())
        handoff = json.loads((scratch / 'code/CPU-ORACLE-HANDOFF.json').read_text())
        control_with_sha = {**control, '_control_sha256': control_sha}
        plan = make_plan(control_with_sha, handoff)
        write_json(scratch / 'new-attended-plan.json', plan)
        plan_sha = digest(scratch / 'new-attended-plan.json')
        upstream = Path.home() / 'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting/external/vjepa2'
        command = [sys.executable, '-B', '-u', str(scratch / 'code/scripts/check_cpu_oracle.py'),
            '--diagnostic', '--handoff-sha256', HANDOFF_SHA, '--upstream', str(upstream),
            '--plan', str(scratch / 'new-attended-plan.json'), '--plan-sha256', plan_sha,
            '--launcher-started', str(started_unix), '--output', str(scratch / 'results')]
        result['reference_process_started'] = True
        print('Starting the unchanged CPU reference probe in private local scratch.', file=sys.stderr, flush=True)
        execution = supervise(command, scratch, deadline, gate)
        write_json(scratch / 'supervisor-result.json', execution)
        result['reference_exit_status'] = execution['exit_status']
        if execution['exit_status'] not in (0, 1):
            raise RuntimeError('Original probe returned a technical failure; no retry.')
        result['reference_process_finished'] = True
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        print('Scratch attempt stopped: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr, flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        # Free only this new operational reserve so a failure receipt can fit.
        reserve = scratch / 'emergency-space.bin'
        if reserve.exists():
            reserve.unlink()
        result.update(run_phase_finished=True, wall_seconds=time.monotonic() - started)
        write_json(scratch / 'operator-result.json', result)
    if result['wall_seconds'] > 600:
        result['deadline_exceeded'] = True
        write_json(scratch / 'operator-result.json', result, replace=True)
    signal.setitimer(signal.ITIMER_REAL, POLICY['collection_seconds'])
    try:
        result_stream(scratch, control_sha, output)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return result


def collect_remote(control, scratch, output):
    """Copy only a finished private run; never invokes the numerical probe."""
    validate_control(control)
    folder = Path(scratch)
    prefix = 'vjepa-cpu-oracle-20261104-' + control['host'] + '-'
    if (platform.node() != control['host'] or folder.parent != Path('/var/tmp')
            or not folder.name.startswith(prefix) or folder.is_symlink()
            or folder.stat().st_uid != os.getuid() or stat.S_IMODE(folder.stat().st_mode) != 0o700):
        raise ValueError('Unexpected private scratch path or host for download only.')
    sha = hashlib.sha256(json_bytes(control)).hexdigest()
    saved_control = json.loads((folder / 'operator-control.json').read_text())
    state = json.loads((folder / 'operator-result.json').read_text())
    if saved_control != control or state['control_sha256'] != sha or state['run_phase_finished'] is not True:
        raise ValueError('Require the matching finished run; collection cannot restart it.')
    output.write(json_bytes({'phase': 'scratch-created', 'kind': KIND, 'host': control['host'],
        'control_sha256': sha, 'scratch_directory': str(folder), 'download_only': True}))
    result_stream(folder, sha, output)


def remote_main(control_base64, mode='run', scratch=None):
    """Called from hash-bound source embedded in an attended SSH command."""
    os.umask(0o077)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PYTHONNOUSERSITE'] = '1'
    control = json.loads(base64.b64decode(control_base64, validate=True))
    if mode == 'collect':
        collect_remote(control, scratch, sys.stdout.buffer)
    elif mode == 'run':
        header = sys.stdin.buffer.readline(POLICY['max_header_bytes'] + 1)
        if len(header) > POLICY['max_header_bytes'] or not header.endswith(b'\n'):
            raise ValueError('Invalid bounded authorization header.')
        authorization = json.loads(header)
        run_remote(control, authorization, sys.stdin.buffer, sys.stdout.buffer)
    else:
        raise ValueError('Only run and download-only modes exist.')
