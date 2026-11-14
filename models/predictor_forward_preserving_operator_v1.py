"""Cold bounded supervisor for a new, separately authorized one-EOT8 cohort."""
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import tempfile
import time
from types import FunctionType
from models import predictor_eot8_record_operator_v2 as envelope
from models import predictor_forward_preserving_records_v1 as records

base, old = envelope.base, envelope.old
MiB, GiB = 2**20, 2**30
KIND = 'forward-preserving-native-EOT8-operator-v1'
AUTH_KIND = 'forward-preserving-native-EOT8-university-exception-v1'
PREFIX = 'vjepa-forward-preserving-v1-20261114-'
CONTRACT = 'configs/predictor_forward_preserving_handoff_v1.json'
CONTRACT_SHA = 'fdd822dcfa930984d57564ed63ad9b481cce94e0df6d520deffecd89cca538e1'
PACKET_SHA = envelope.PACKET_SHA
INIT = envelope.INIT
FROZEN_NAMES = (*envelope.FROZEN_NAMES, 'models/predictor_eot8_owned_receipt_v1.py',
                'models/predictor_eot8_record_v2.py', 'models/predictor_eot8_events_v1.py',
                'models/predictor_eot8_record_operator_v2.py', 'models/predictor_boundary_contract_v1.py',
                'models/predictor_boundary_record_v1.py')
NEW_NAMES = ('models/predictor_eot8_adapter_v1.py', 'models/predictor_eot8_owned_v1.py',
             'models/predictor_forward_preserving_cpu_v1.py', 'models/predictor_forward_preserving_upstream_v1.py',
             'models/predictor_forward_preserving_target_v1.py', 'models/predictor_forward_preserving_records_v1.py',
             'models/predictor_forward_preserving_worker_v1.py', 'models/predictor_forward_preserving_operator_v1.py',
             'scripts/capture_predictor_forward_preserving_v1.py', CONTRACT)
REMOTE_NAMES = tuple(dict.fromkeys(('models/__init__.py', *FROZEN_NAMES, *NEW_NAMES)))
LOCAL_EXTRA = ('models/predictor_forward_preserving_client_v1.py', 'scripts/prepare_predictor_forward_preserving_v1.py',
               'scripts/run_predictor_forward_preserving_v1.py', 'tests/test_predictor_forward_preserving_handoff_v1.py')


def exact(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def validate_control(c):
    expected = {'kind': KIND, 'host': 'linux10603', 'parent': '/var/tmp', 'expected_device': 66307, 'expected_mount': '/',
                'expected_gpu': 'NVIDIA GeForce RTX 3070', 'expected_gpu_uuid': 'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f',
                'expected_driver': '580.178.04', 'expected_ssh_fingerprint': base.SSH_FINGERPRINT,
                'contract_sha256': CONTRACT_SHA, 'paid_compute': False, 'research_admitted': False,
                'authorization_present_at_preparation': False, 'automatic_retries': 0}
    more = {'contract', 'contract_text', 'resource', 'repository', 'local_handoff_directory', 'archive_path', 'archive_manifest',
            'known_hosts_sha256', 'remote_source_sha256', 'local_source_sha256', 'prior_inputs_manifest',
            'prior_inputs_manifest_sha256', 'host_binding'}
    if set(c) != set(expected) | more or not exact({k: c[k] for k in expected}, expected):
        raise ValueError('Exact new one-EOT8 control/domain required.')
    p = c['contract']
    if (hashlib.sha256(c['contract_text'].encode()).hexdigest() != CONTRACT_SHA or not exact(json.loads(c['contract_text']), p)
            or p['authorized'] is not False or p['new_target_execution_permitted'] is not False
            or not exact(c['resource'], p['resource']) or c['resource']['work_seconds'] != 2400
            or c['resource']['allocator_bytes'] != 7*GiB or p['full_model_EOT8_passes_at_most'] != 4
            or not exact(c['host_binding'], {'prior_consumed_files': p['prior_consumed_files']})):
        raise ValueError('New plan, resource bound or consumed lineage changed.')
    if (set(c['remote_source_sha256']) != set(REMOTE_NAMES)
            or set(c['local_source_sha256']) != set(REMOTE_NAMES)-{'models/__init__.py'} | set(LOCAL_EXTRA)
            or c['remote_source_sha256'][CONTRACT] != CONTRACT_SHA):
        raise ValueError('Exact source coverage required.')
    spec = c['archive_manifest']
    if (spec['sha256'] != PACKET_SHA or spec['bytes'] != 17192095 or len(spec['files']) != 99
            or sum(item['bytes'] for item in spec['files'].values()) != 26446958):
        raise ValueError('Unchanged original one-training-clip packet required.')
    for name in spec['files']:
        old.safe_name(name)


def authorization_template(c, sha, basis):
    return {'kind': AUTH_KIND, 'approved': True, 'control_sha256': sha, 'host': 'linux10603',
            'maximum_new_attempts': 1, 'resource': c['resource'], 'roles': c['contract']['worker_roles'],
            'full_EOT8_passes_at_most': 4, 'automatic_retries': 0, 'paid_compute': False, 'cleanup': False,
            'reset_old_claims': False, 'private_key_upload': False, 'held_out_pixels': False, 'user_basis': basis}


def validate_authorization(a, sha, c=None):
    if c is None:
        # Client and bootstrap supply their bound control via the explicit reader below.
        raise ValueError('Bound control is required for new authorization.')
    basis = a.get('user_basis') if isinstance(a, dict) else None
    if not isinstance(basis, str) or not 1 <= len(basis) <= 8192 or not exact(a, authorization_template(c, sha, basis)):
        raise ValueError('One new exact scoped one-EOT8 authorization required.')


def inventory(folder, *, hashing=False, running=False):
    files, allocated = base.inventory(folder, hashing=hashing, max_bytes=3*GiB, running=running)
    for name, item in files.items():
        leaf = Path(name).name
        plain = leaf.removesuffix('.partial')
        cap = (3*GiB if plain == 'space-proof-v2.bin' else 65536 if plain == 'loss.npy'
               else MiB if plain == 'masks.npy' else 12*MiB if plain == 'targets.npy'
               else 16*MiB if plain.endswith(('.npy', '.log')) else records.LIMITS.get(plain, 64*MiB))
        if item['bytes'] > cap:
            raise ValueError('Explicit typed evidence file cap exceeded: '+name)
    return files, allocated


def storage(folder, identity, *, setup=False, parent=Path('/var/tmp')):
    folder = Path(folder); st = folder.lstat(); pst = Path(parent).stat()
    if (folder.is_symlink() or not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid()
            or stat.S_IMODE(st.st_mode) != 0o700 or st.st_dev != identity['device']
            or (pst.st_dev, pst.st_ino) != (identity['device'], identity['inode'])):
        raise ValueError('Scratch ownership/device/parent changed.')
    files, allocated = inventory(folder, running=True)
    for current, dirs, names in os.walk(folder, followlinks=False):
        allocated += Path(current).stat().st_blocks*512
    limit = 3*GiB if setup else 9*GiB//4
    fs = os.statvfs(folder)
    if allocated > limit or fs.f_bavail*fs.f_frsize < GiB or fs.f_favail < 1024:
        raise ValueError('Aggregate allocation/free-space/inode ceiling.')
    categories = dict.fromkeys(('raw', 'input', 'code', 'metadata', 'cache', 'emergency', 'proof'), 0)
    for name, item in files.items():
        category = ('proof' if name == 'space-proof-v2.bin.partial' else 'input' if name.startswith('original-packet.tar.gz')
                    else 'code' if name.startswith(('code/', 'overlay/')) else 'cache' if name.startswith(('cache/', 'tmp/'))
                    else 'emergency' if name == 'emergency-space.bin' else 'raw' if name.startswith('results/') and '.npy' in name else 'metadata')
        categories[category] += item['bytes']
    caps = {'raw': 1792*MiB, 'input': 128*MiB, 'code': 128*MiB, 'metadata': 64*MiB,
            'cache': 32*MiB, 'emergency': 16*MiB, 'proof': 5*GiB//2 if setup else 0}
    if any(categories[k] > cap for k, cap in caps.items()):
        raise ValueError('Typed storage category ceiling.')
    return {'logical_bytes': sum(f['bytes'] for f in files.values()), 'allocated_bytes': allocated,
            'categories': categories, 'ceiling': limit, 'files': len(files)}


readiness, extract = envelope.readiness, envelope.extract
stream = FunctionType(envelope.stream.__code__, dict(envelope.stream.__globals__, inventory=inventory, KIND=KIND))


def verify_outputs(folder, index, *, control_sha256, live):
    state = records.load(folder, control_sha256=control_sha256, index=index, live=live)
    return {'worker_sha256': old.digest(Path(folder)/'worker.json'), 'completion_sha256': old.digest(Path(folder)/'success.json'),
            'pid': state['pid'], 'index': index, 'numerical_passed': True, 'CUDA_domain': live}


def check_worker(folder, index, *, control_sha256):
    return verify_outputs(folder, index, control_sha256=control_sha256, live=True)


def _execute(c, auth, incoming, output, sources, sha, start, deadline, stop):
    identity = readiness(c); parent = Path(identity['parent']); base.gpu_gate(c)
    folder = Path(tempfile.mkdtemp(prefix=PREFIX, dir=parent)); folder.chmod(0o700)
    output.write(old.json_bytes({'kind': KIND, 'phase': 'scratch-created', 'scratch_directory': str(folder), 'control_sha256': sha})); output.flush()
    state = {'kind': KIND, 'complete': False, 'passed': False, 'tasks': [], 'started_unix': time.time(),
             'deadline_unix': time.time()+max(0, deadline-time.monotonic()), 'work_seconds': 2400, 'research_admitted': False}
    try:
        old.write_json(folder/'control.json', c); old.write_json(folder/'authorization.json', auth); old.write_json(folder/'storage-before.json', identity)
        if set(sources) != set(REMOTE_NAMES) or sum(map(len, sources.values())) > 2*MiB:
            raise ValueError('Exact overlay extent required.')
        for name, data in sources.items():
            if hashlib.sha256(data).hexdigest() != c['remote_source_sha256'][name]:
                raise ValueError('Overlay source changed.')
            p = folder/'overlay'/old.safe_name(name); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data); p.chmod(0o600)
        with (folder/'emergency-space.bin').open('xb') as f:
            f.write(os.urandom(16*MiB)); f.flush(); os.fsync(f.fileno())
        end = min(deadline-5, time.monotonic()+60)
        program = 'import sys;sys.path.insert(0,'+repr(str(folder/'overlay'))+');from models.scratch_allocation_v2 import bounded_write_proof;bounded_write_proof('+repr(str(folder))+',deadline='+repr(end)+')'
        proof = base.primitives.supervise([sys.executable, '-B', '-u', '-c', program], folder, role='proof', deadline=end,
                                           gate=lambda: storage(folder, identity, setup=True), file_limit=3*GiB)
        if proof['exit_status'] != 0:
            raise RuntimeError('Full storage proof failed before archive storage.')
        base.primitives.validate_proof(folder)
        old.receive_archive(incoming, folder/'original-packet.tar.gz', 17192095, PACKET_SHA, deadline-5)
        extract(folder/'original-packet.tar.gz', c['archive_manifest'], folder/'code')
        if time.monotonic()-start > 180:
            raise TimeoutError('Setup time exhausted.')
        for name in ('tmp', 'cache', 'cache/cuda', 'results'):
            (folder/name).mkdir(mode=0o700, exist_ok=True)
        old.write_json(folder/'session.json', state)
        for i in range(4):
            # Only complete passing predecessors unlock the next role, including both replay roles.
            for j, accepted in enumerate(state['tasks']):
                if check_worker(folder/'results'/str(j), j, control_sha256=sha) != accepted:
                    raise ValueError('Earlier passing worker evidence changed.')
            base.gpu_gate(c)
            entry = folder/'overlay/scripts/capture_predictor_forward_preserving_v1.py'
            end = min(deadline-125, time.monotonic()+480)
            if end <= time.monotonic():
                raise TimeoutError('Continuing work allowance exhausted.')
            old.write_json(folder/('worker-plan-'+str(i)+'.json'), {'index': i, 'control_sha256': sha, 'deadline_unix': time.time()+end-time.monotonic()})
            command = [sys.executable, '-I', '-B', '-u', str(entry), '--scratch', str(folder), '--control-sha256', sha, '--index', str(i)]
            last, samples = [0.], []
            def gate():
                info = storage(folder, identity)
                if time.monotonic()-last[0] >= 5:
                    owned = []
                    for pid in old.descendants(os.getpid()):
                        try:
                            argv = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
                        except FileNotFoundError:
                            continue
                        if str(entry).encode() in argv:
                            owned.append(pid)
                    if len(owned) > 1:
                        raise ValueError('Multiple scientific processes refused.')
                    samples.append(base.gpu_gate(c, owned[0] if owned else None)); last[0] = time.monotonic()
                return info
            env = {'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0'), 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
                   'TMPDIR': str(folder/'tmp'), 'XDG_CACHE_HOME': str(folder/'cache'), 'CUDA_CACHE_PATH': str(folder/'cache/cuda'), 'CUDA_CACHE_MAXSIZE': str(32*MiB)}
            if env['CUDA_VISIBLE_DEVICES'] in ('', '-1'):
                raise ValueError('CUDA visibility hidden; refuse.')
            child = base.primitives.supervise(command, folder, role='worker-'+str(i), deadline=end, gate=gate, file_limit=16*MiB, env=env)
            old.write_json(folder/('gpu-samples-'+str(i)+'.json'), samples); base.gpu_gate(c)
            if child['exit_status'] != 0:
                raise RuntimeError('Worker failed; stop before replay/next worker, no retry.')
            state['tasks'].append(check_worker(folder/'results'/str(i), i, control_sha256=sha))
            old.write_json(folder/'session.json', state, replace=True)
        if len({t['pid'] for t in state['tasks']}) != 4:
            raise ValueError('Every role must be a fresh serial process.')
        state.update(complete=True, passed=True)
    except BaseException as error:
        state.update(error_type=type(error).__name__, error=str(error)[:2048])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        state['wall_seconds'] = time.monotonic()-start
        if state['wall_seconds'] > 2400:
            state.update(passed=False, error='Outer work deadline exceeded.')
        old.write_json(folder/'session.json', state, replace=(folder/'session.json').exists())
    signal.signal(signal.SIGALRM, stop); signal.setitimer(signal.ITIMER_REAL, 600)
    try:
        stream(folder, sha, output)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return 0 if state['passed'] else 1


def remote(c, auth, incoming, output, sources):
    validate_control(c); sha = hashlib.sha256(old.json_bytes(c)).hexdigest(); validate_authorization(auth, sha, c)
    # Frozen signal/deadline envelope, new scope and new implementation namespace only.
    runner = FunctionType(envelope.remote.__code__, dict(envelope.remote.__globals__, validate_control=validate_control,
                          validate_authorization=lambda a, h: validate_authorization(a, h, c), _execute=_execute))
    return runner(c, auth, incoming, output, sources)


def main(control64, sources64, mode='run', scratch=None):
    if mode != 'run' or scratch is not None:
        raise ValueError('Only one original attended SSH/collection; new retrieval needs independent review.')
    c = json.loads(base64.b64decode(control64, validate=True))
    sources = {n: base64.b64decode(v, validate=True) for n, v in json.loads(base64.b64decode(sources64, validate=True)).items()}
    validate_control(c)
    auth = json.loads(sys.stdin.buffer.readline(65537))
    return remote(c, auth, sys.stdin.buffer, sys.stdout.buffer, sources)
