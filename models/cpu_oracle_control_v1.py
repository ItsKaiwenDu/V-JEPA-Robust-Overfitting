"""Versioned CPU capture supervisor. Preparation never grants resources.

Only the new authorization kind can start one run. The original numerical
packet and v1 operator are dependencies, never patched or resumed.
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time

from models import cpu_oracle_scratch as old
from models import cpu_oracle_control_prerequisite as prerequisite

KIND = 'cpu-oracle-control-v1'
AUTH_KIND = 'cpu-oracle-control-resource-exception-v1'
GIB = 2**30
MIB = 2**20
POLICY = {**old.POLICY, 'minimum_free_bytes': 8*GIB, 'max_overlay_bytes': 2*MIB}
REMOTE_NAMES = (
    'models/__init__.py', 'models/cpu_oracle_scratch.py',
    'models/cpu_oracle_control_prerequisite.py',
    'models/scratch_allocation_v2.py', 'models/cpu_oracle_reassessment_v2.py',
    'models/cpu_oracle_control_v1.py', 'models/cpu_oracle_control_runtime_v1.py',
    'scripts/capture_cpu_control_v1.py',
)
OVERLAY_INIT = ("from pathlib import Path\n"
                "__path__.append(str(Path(__file__).resolve().parents[2] / 'code/models'))\n").encode()
PREFIX = 'vjepa-cpu-control-v1-20261104-'


def validate_control(control):
    if (control.get('kind') != KIND or control.get('host') != 'linux10609'
            or control.get('role') != 'control' or control.get('policy') != POLICY
            or control.get('authorization_present_at_preparation') is not False
            or control.get('reuse_old_grant') is not False
            or control.get('original_refusals_waived') is not False
            or set(control.get('remote_source_sha256', {})) != set(REMOTE_NAMES)
            or set(control.get('local_source_sha256', {})) != (set(REMOTE_NAMES)-{'models/__init__.py'} | {
                'models/cpu_oracle_control_client_v1.py', 'scripts/run_cpu_oracle_control_v1.py'})
            or control.get('prior_claims_sha256', {}).get(
                'runs/pretrained/diagnostic/cpu-oracle-scratch-recovery-20261104/linux10603/ATTEMPT.json')
                != 'fbafa36e9ab2ef1f1e9966ed737cf68c96911fb4e44d1a951500d51a820dd406'
            or control.get('prior_claims_sha256', {}).get(
                'runs/pretrained/diagnostic/cpu-oracle-capture-v2-20261104/linux10603/ATTEMPT.json')
                != '8a3c97be2f4b64c393cb65ad95387c49e752cde420087a874d3b2ca2831eb05c'):
        raise ValueError('Changed control-host, implementation, policy or history scope.')
    prerequisite.validate_binding(control.get('target_prerequisite'))
    for name, sha in prerequisite.REQUIRED_PINS.items():
        if control['prior_claims_sha256'].get(name) != sha:
            raise ValueError('Missing accepted target review or consumed target claim binding.')
    # Reuse only the frozen packet/domain validator, with its explicit v1 view.
    # No module globals or historical control bytes are changed.
    old.validate_control({**control, 'kind': old.KIND, 'policy': old.POLICY})
    if control['archive_bytes'] != 19042363:
        raise ValueError('Require the unchanged full original CPU packet.')
    for name, value in control['remote_source_sha256'].items():
        old.safe_name(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid control source hash.')


def validate_authorization(auth, control_sha, host):
    if host != 'linux10609':
        raise ValueError('The new control authorization is linux10609 only.')
    expected = {'kind': AUTH_KIND, 'approved': True, 'control_sha256': control_sha,
        'host': host, 'maximum_new_attempts': 1, 'max_seconds': 600,
        'worker_seconds': 90, 'write_test_bytes': POLICY['write_test_bytes'],
        'setup_ceiling': POLICY['setup_ceiling'], 'capture_ceiling': POLICY['capture_ceiling'],
        'collection_seconds': 600, 'CPU_only': True, 'paid_compute': False,
        'reset_old_claims': False, 'automatic_retries': 0}
    if set(auth) != {*expected, 'user_basis'} or any(type(auth[k]) is not type(v) or auth[k] != v for k, v in expected.items()):
        raise ValueError('A separate explicit control-host resource exception is required.')
    if not isinstance(auth['user_basis'], str) or not 1 <= len(auth['user_basis']) <= 8192:
        raise ValueError('Missing explicit user authorization basis.')


def readiness(control):
    validate_control(control)
    if platform.node() != control['host'] or sys.platform != 'linux':
        raise ValueError('Unexpected control host or OS.')
    parent = Path(control['parent'])
    if parent.is_symlink() or parent.resolve() != parent:
        raise ValueError('Scratch parent is aliased.')
    info = parent.stat(); capacity = os.statvfs(parent)
    value = {'host': platform.node(), 'parent': str(parent), 'device': info.st_dev,
        'inode': info.st_ino, 'home_device': Path.home().stat().st_dev,
        'owner_uid': info.st_uid, 'mode': stat.S_IMODE(info.st_mode),
        **old.mount_for(parent, Path('/proc/self/mountinfo').read_text()),
        'available_bytes': capacity.f_bavail*capacity.f_frsize,
        'available_inodes': capacity.f_favail,
        'writable_for_current_user': os.access(parent, os.W_OK | os.X_OK)}
    old.validate_readiness(value, {**control, 'kind': old.KIND, 'policy': old.POLICY})
    if value['available_bytes'] < POLICY['minimum_free_bytes']:
        raise ValueError('Control v1 requires at least 8 GiB available; this is not a reservation.')
    return value


def write_overlay(folder, sources, control):
    if set(sources) != set(REMOTE_NAMES) or sum(len(b) for b in sources.values()) > POLICY['max_overlay_bytes']:
        raise ValueError('Wrong or oversized control source payload.')
    for name, data in sources.items():
        if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != control['remote_source_sha256'][name]:
            raise ValueError('Control v1 payload source changed: '+name)
    if sources['models/__init__.py'] != OVERLAY_INIT:
        raise ValueError('Changed overlay package path.')
    root = Path(folder)/'overlay'; root.mkdir(mode=0o700)
    for name, data in sources.items():
        path = root/old.safe_name(name); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())


def storage_gate(folder, identity, *, phase='capture', parent_path=Path('/var/tmp')):
    folder = Path(folder); parent = Path(parent_path).stat(); info = folder.lstat()
    if (phase not in ('setup', 'capture') or folder.is_symlink()
            or (parent.st_dev, parent.st_ino) != (identity['device'], identity['inode'])
            or not stat.S_ISDIR(info.st_mode) or info.st_dev != parent.st_dev
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError('Scratch phase/path/identity changed.')
    cap = POLICY['setup_ceiling' if phase == 'setup' else 'capture_ceiling']
    files, allocated = old.tree_inventory(folder, max_bytes=cap, running=True)
    # The original helper accounts for file blocks. Include all directories too.
    for current, dirs, _ in os.walk(folder, followlinks=False):
        directory = Path(current).lstat()
        if not stat.S_ISDIR(directory.st_mode) or directory.st_dev != info.st_dev:
            raise ValueError('Directory changed device or type.')
        allocated += directory.st_blocks*512
    logical = sum(item['bytes'] for item in files.values())
    capacity = os.statvfs(folder)
    if (allocated > cap or capacity.f_bavail*capacity.f_frsize < POLICY['runtime_free_floor']
            or capacity.f_favail < 1024):
        raise ValueError('Observed allocated-byte ceiling or capacity floor reached.')
    categories = {'code': 0, 'input': 0, 'raw': 0, 'archive': 0, 'proof': 0, 'metadata': 0, 'emergency': 0}
    for name, item in files.items():
        if name.startswith(('code/', 'overlay/')): category = 'code'
        elif name in ('original-packet.tar.gz', 'original-packet.tar.gz.partial'): category = 'input'
        elif name == 'space-proof-v2.bin.partial': category = 'proof'
        elif name == 'emergency-space.bin': category = 'emergency'
        elif name == 'results/evidence.tar.gz': category = 'archive'
        elif name.startswith('results/') and name.endswith(('.npz', '.npz.partial')):
            category = 'raw'
            if item['bytes'] > POLICY['max_array_bytes']: raise ValueError('Captured array extent changed.')
        else: category = 'metadata'
        categories[category] += item['bytes']
    limits = {'code': POLICY['max_code_bytes'], 'input': POLICY['max_input_bytes'],
        'raw': POLICY['max_raw_bytes'], 'archive': POLICY['max_archive_bytes'],
        'proof': POLICY['write_test_bytes'] if phase == 'setup' else 0,
        'metadata': POLICY['max_metadata_bytes'], 'emergency': POLICY['emergency_bytes']}
    if any(categories[k] > limit for k, limit in limits.items()):
        raise ValueError('Control v1 scratch category ceiling reached.')
    return {'logical_bytes': logical, 'allocated_bytes': allocated,
        'available_bytes': capacity.f_bavail*capacity.f_frsize, 'files': len(files),
        'phase': phase, 'ceiling': cap, 'categories': categories}


def supervise(command, folder, *, role, deadline, gate, file_limit, env=None):
    """Supervise blocking proof/reference calls; never retries or resets a clock."""
    folder = Path(folder); started = time.monotonic(); child = None; tracked = {}
    result = {'role': role, 'exit_status': None, 'complete': False, 'storage_samples': 0,
        'peak_logical_bytes': 0, 'peak_allocated_bytes': 0, 'file_limit_bytes': file_limit,
        'kernel_aggregate_quota': False, 'automatic_retry': False}
    def sample():
        row = gate(); result['storage_samples'] += 1; result['last_storage'] = row
        for metric in ('logical_bytes', 'allocated_bytes'):
            result['peak_'+metric] = max(result['peak_'+metric], row[metric])
    def limits():
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit)); os.umask(0o077)
    try:
        sample()
        if time.monotonic() >= deadline: raise TimeoutError('Original phase deadline expired before launch.')
        with (folder/(role+'.log')).open('xb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, preexec_fn=limits, env={**os.environ,
                    'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
                    'PYTHONNOUSERSITE': '1', **(env or {})})
            while child.poll() is None:
                tracked.update(old.descendants(child.pid))
                if time.monotonic() >= deadline: raise TimeoutError('Original phase deadline reached; no retry.')
                if os.fstat(log.fileno()).st_size > POLICY['max_launcher_log_bytes']:
                    raise ValueError('Phase log exceeds its bound.')
                sample(); time.sleep(min(.1, max(0, deadline-time.monotonic())))
            result['exit_status'] = child.returncode
        if time.monotonic() >= deadline: raise TimeoutError('Phase completed after its deadline.')
        if any(old.process_identity(pid) == ident for pid, ident in tracked.items()):
            raise ValueError('A launched descendant remains after its parent exited.')
        sample(); result['complete'] = True
        return result
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        if child is not None: result['owned_descendants_terminated'] = old.terminate_owned(child, tracked)
        result['wall_seconds'] = time.monotonic()-started
        old.write_json(folder/('supervisor-'+role+'.json'), result)


def validate_proof(folder):
    proof = json.loads((Path(folder)/'allocation-result-v2.json').read_text())
    if (proof.get('kind') != 'scratch-allocation-proof-v2'
            or any(proof.get(k) is not True for k in ('passed', 'complete', 'proof_removed', 'fsync_succeeded'))
            or proof.get('preallocator_injected_for_test') is not False
            or proof.get('requested_bytes') != POLICY['write_test_bytes']
            or proof.get('bytes_written') != POLICY['write_test_bytes']
            or proof.get('bytes_read_back') != POLICY['write_test_bytes']
            or proof.get('setup_ceiling_bytes') != POLICY['setup_ceiling']
            or not 0 < proof.get('peak_observed_allocated_bytes', 0) <= POLICY['setup_ceiling']
            or not 0 < proof.get('wall_seconds', 0) <= POLICY['write_test_seconds']
            or (Path(folder)/'space-proof-v2.bin.partial').exists()):
        raise ValueError('Full fixed-extent allocation proof did not pass the v2 envelope.')
    return proof


def read_runtime_record(path):
    """Read the frozen record format without importing scientific libraries."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > POLICY['max_header_bytes']:
        raise ValueError('Missing, aliased or oversized runtime record.')
    record = json.loads(path.read_text())
    if set(record) != {'payload', 'sha256'}:
        raise ValueError('Runtime record wrapper changed.')
    canonical = json.dumps(record['payload'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != record['sha256']:
        raise ValueError('Runtime record payload checksum differs.')
    return record['payload']


def reference_status(folder, exit_status):
    """Separate startup/technical failures from captured numerical refusals."""
    folder = Path(folder); session_path = folder/'results/session-control-v1.json'
    base = {'reference_run_passed': False, 'runtime_session_present': session_path.is_file(),
        'original_workers_completed': 0}
    if not session_path.is_file():
        return {**base, 'reference_outcome': 'startup_failure',
            'failure_detail': 'Reference parent exited before saving a runtime session; inspect reference.log.'}
    try:
        state = read_runtime_record(session_path)
        if (state.get('kind') != 'cpu-oracle-control-runtime-v1' or state.get('CPU_only') is not True
                or state.get('target_review_sha256') != prerequisite.REVIEW_SHA
                or state.get('handoff_sha256') != old.HANDOFF_SHA
                or any(state.get(k) is not False for k in ('GPU_work', 'local_tiny_rehearsal',
                    'original_parent_invoked', 'original_refusals_waived', 'research_admitted'))):
            raise ValueError('Runtime source/domain/refusal labels changed.')
        rows = state['tasks']
        if (not isinstance(rows, list) or len(rows) > 4
                or [[r['threads'], r['replay']] for r in rows] != old.TASKS[:len(rows)]):
            raise ValueError('Runtime worker coverage/order changed.')
        base['original_workers_completed'] = len(rows)
        base['runtime_session_sha256'] = old.digest(session_path)
        if state.get('complete') is not True:
            return {**base, 'reference_outcome': 'runtime_failure',
                'failure_detail': 'Capture runtime is incomplete; preserve its session and logs.'}
        summary_path = folder/'results/summary-control-v1.json'
        if old.digest(summary_path) != state['summary_sha256']:
            raise ValueError('Runtime summary hash changed.')
        summary = read_runtime_record(summary_path)
        if (summary.get('kind') != 'cpu-oracle-control-runtime-v1' or summary.get('handoff_sha256') != old.HANDOFF_SHA
                or summary.get('host') != 'linux10609'
                or summary.get('target_review_sha256') != prerequisite.REVIEW_SHA
                or summary.get('CPU_only') is not True or summary.get('tiny_fixture') is not False
                or summary.get('capture_integrity_verified') is not True
                or any(summary.get(k) is not False for k in ('GPU_work', 'original_parent_invoked',
                    'original_refusals_waived', 'root_cause_established', 'host_reliability_established',
                    'optimization_adopted', 'research_admitted'))
                or summary.get('bounds') != {'rtol': 1e-12, 'atol': 1e-14}):
            raise ValueError('Runtime summary contract changed.')
        checked = summary['workers']
        if (len(checked) != len(rows) or not checked
                or any((r['threads'], r['replay'], r['worker_sha256']) !=
                    (c['threads'], c['replay'], c['worker_sha256']) for r, c in zip(rows, checked))
                or any(type(c.get('passed')) is not bool for c in checked)):
            raise ValueError('Runtime summary worker identities changed.')
        complete = len(rows) == 4
        passed = complete and all(c['passed'] for c in checked)
        for record in (state, summary):
            if record.get('all_planned_workers_completed') is not complete or record.get('saved_value_checks_passed') is not passed:
                raise ValueError('Runtime completion labels disagree with worker checks.')
        if type(exit_status) is not int or exit_status not in (0, 1) or (exit_status == 0) != passed:
            raise ValueError('Runtime exit disagrees with its saved decisions.')
        if not passed and (any(not c['passed'] for c in checked[:-1]) or checked[-1]['passed']):
            raise ValueError('Runtime continued after refusal or stopped without one.')
        return {**base, 'reference_outcome': 'saved_value_checks_passed' if passed else 'numerical_refusal',
            'reference_run_passed': passed, 'runtime_summary_sha256': state['summary_sha256']}
    except (ValueError, TypeError, KeyError, OSError) as exc:
        return {**base, 'reference_outcome': 'invalid_runtime_receipts', 'failure_detail': str(exc)}


def result_stream(folder, control_sha, output):
    files, _ = old.tree_inventory(folder, hashing=True, max_bytes=POLICY['max_collection_bytes'])
    data = old.json_bytes({'kind': KIND, 'phase': 'collection', 'control_sha256': control_sha,
        'files': files, 'logical_bytes': sum(v['bytes'] for v in files.values()),
        'scientific_review_complete': False, 'model_restarted_for_collection': False})
    if len(data) > POLICY['max_header_bytes']: raise ValueError('Collection header exceeds its bound.')
    output.write(data); output.flush()
    with tarfile.open(mode='w|', fileobj=output) as archive:
        for name, spec in sorted(files.items()):
            path = Path(folder)/name
            if path.is_symlink() or old.digest(path) != spec['sha256']: raise ValueError('Collection source changed.')
            member = tarfile.TarInfo(name); member.size = spec['bytes']; member.mode = 0o600
            member.mtime = spec['mtime_ns']//10**9
            with path.open('rb') as source: archive.addfile(member, source)
            if old.digest(path) != spec['sha256']: raise ValueError('Source changed during collection.')


def run_remote(control, auth, incoming, output, sources):
    validate_control(control); control_sha = hashlib.sha256(old.json_bytes(control)).hexdigest()
    validate_authorization(auth, control_sha, control['host'])
    identity = readiness(control); began = time.monotonic(); unix = time.time(); deadline = began+600
    folder = Path(tempfile.mkdtemp(prefix=PREFIX+control['host']+'-', dir=control['parent']))
    os.chmod(folder, 0o700)
    output.write(old.json_bytes({'kind': KIND, 'phase': 'scratch-created', 'host': control['host'],
        'control_sha256': control_sha, 'scratch_directory': str(folder), 'started_unix': unix})); output.flush()
    result = {'kind': KIND, 'control_sha256': control_sha, 'host': control['host'],
        'run_phase_finished': False, 'reference_process_started': False, 'CPU_only': True,
        'GPU_work': False, 'paid_compute': False, 'scientific_review_complete': False,
        'old_refusals_waived': False, 'target_review_sha256': prerequisite.REVIEW_SHA, 'started_unix': unix, 'deadline_unix': unix+600,
        'reference_outcome': 'not_started', 'reference_run_passed': False}
    def stop(*unused): raise RuntimeError('Attended control attempt interrupted; preserve files, no retry.')
    previous = {s: signal.getsignal(s) for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGALRM)}
    for s in previous: signal.signal(s, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()-5))
    try:
        old.write_json(folder/'operator-control.json', control)
        old.write_json(folder/'resource-authorization.json', auth)
        old.write_json(folder/'storage-before.json', identity)
        write_overlay(folder, sources, control)
        with (folder/'emergency-space.bin').open('xb') as reserve:
            reserve.write(os.urandom(POLICY['emergency_bytes'])); reserve.flush(); os.fsync(reserve.fileno())
        setup_gate = lambda: storage_gate(folder, identity, phase='setup')
        proof_end = min(deadline-5, time.monotonic()+60)
        proof_program = ('import sys;sys.path.insert(0,'+repr(str(folder/'overlay'))+');'
            'from models.scratch_allocation_v2 import bounded_write_proof;'
            'bounded_write_proof('+repr(str(folder))+',deadline='+repr(proof_end)+')')
        print('Control v1: supervising the fixed-extent allocation proof before any reference worker.', file=sys.stderr, flush=True)
        proof_execution = supervise([sys.executable, '-B', '-u', '-c', proof_program], folder,
            role='proof', deadline=proof_end, gate=setup_gate, file_limit=POLICY['setup_ceiling'])
        if proof_execution['exit_status'] != 0: raise RuntimeError('Allocation child refused; no reference starts.')
        validate_proof(folder); setup_gate(); result['write_test_complete'] = True
        old.receive_archive(incoming, folder/'original-packet.tar.gz', control['archive_bytes'], old.ARCHIVE_SHA, deadline-5)
        setup_gate(); old.extract_packet(folder/'original-packet.tar.gz', control['archive_manifest'], folder/'code'); setup_gate()
        if time.time()-unix > 110: raise TimeoutError('Setup exhausted the original freshness margin.')
        handoff = json.loads((folder/'code/CPU-ORACLE-HANDOFF.json').read_text())
        plan = old.make_plan({**control, '_control_sha256': control_sha}, handoff)
        old.write_json(folder/'new-attended-plan.json', plan)
        gate = lambda: storage_gate(folder, identity)
        old.write_json(folder/'storage-after-setup.json', gate())
        command = [sys.executable, '-B', '-u', str(folder/'overlay/scripts/capture_cpu_control_v1.py'),
            '--packet', str(folder/'code'), '--scratch', str(folder), '--control-sha256', control_sha,
            '--launcher-started', str(unix), '--upstream', str(Path.home()/'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting/external/vjepa2')]
        result['reference_process_started'] = True
        execution = supervise(command, folder, role='reference', deadline=deadline-5,
            gate=gate, file_limit=POLICY['max_archive_bytes'],
            env={'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
        result['reference_exit_status'] = execution['exit_status']
        result['reference_process_finished'] = True
        status = reference_status(folder, execution['exit_status']); result.update(status)
        if 'failure_detail' in status: raise RuntimeError(status['failure_detail'])
    except BaseException as exc:
        if result['reference_process_started'] and result['reference_outcome'] == 'not_started':
            result['reference_outcome'] = 'supervision_failure'
        result.update(error_type=type(exc).__name__, error=str(exc))
        print('Control v1 attempt stopped: '+type(exc).__name__+': '+str(exc), file=sys.stderr, flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        reserve = folder/'emergency-space.bin'
        if reserve.exists(): reserve.unlink()  # Only this new operational reserve.
        result.update(run_phase_finished=True, wall_seconds=time.monotonic()-began)
        if result['wall_seconds'] > 600: result['deadline_exceeded'] = True
        old.write_json(folder/'operator-result.json', result)
    signal.setitimer(signal.ITIMER_REAL, POLICY['collection_seconds'])
    try: result_stream(folder, control_sha, output)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, handler in previous.items(): signal.signal(s, handler)
    return result


def collect_remote(control, scratch, output):
    validate_control(control); folder = Path(scratch); info = folder.lstat()
    sha = hashlib.sha256(old.json_bytes(control)).hexdigest()
    if (platform.node() != control['host'] or folder.parent != Path('/var/tmp')
            or not folder.name.startswith(PREFIX+control['host']+'-') or folder.is_symlink()
            or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError('Wrong private control download path or host.')
    state = json.loads((folder/'operator-result.json').read_text())
    if (json.loads((folder/'operator-control.json').read_text()) != control
            or state.get('control_sha256') != sha or state.get('kind') != KIND
            or state.get('run_phase_finished') is not True):
        raise ValueError('Download requires the matching finished control attempt.')
    output.write(old.json_bytes({'kind': KIND, 'phase': 'scratch-created', 'host': control['host'],
        'control_sha256': sha, 'scratch_directory': str(folder), 'download_only': True}))
    def stop(*unused): raise TimeoutError('Download-only collection deadline reached.')
    previous = signal.signal(signal.SIGALRM, stop); signal.setitimer(signal.ITIMER_REAL, 600)
    try: result_stream(folder, sha, output)
    finally: signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)


def remote_main(control_encoded, sources_encoded, mode='run', scratch=None):
    os.umask(0o077); os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    control = json.loads(base64.b64decode(control_encoded, validate=True))
    if mode == 'collect': return collect_remote(control, scratch, sys.stdout.buffer)
    if mode != 'run': raise ValueError('Only control run and collect modes exist.')
    line = sys.stdin.buffer.readline(POLICY['max_header_bytes']+1)
    if len(line) > POLICY['max_header_bytes'] or not line.endswith(b'\n'): raise ValueError('Invalid authorization header.')
    sources = {n: base64.b64decode(b, validate=True) for n, b in sources_encoded.items()}
    return run_remote(control, json.loads(line), sys.stdin.buffer, sys.stdout.buffer, sources)
