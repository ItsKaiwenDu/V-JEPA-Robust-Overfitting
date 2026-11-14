"""Replacement scope for the child-binding correction; frozen v1 stays untouched."""
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import FunctionType
from models import predictor_forward_preserving_operator_v1 as frozen

base, old, envelope, records = frozen.base, frozen.old, frozen.envelope, frozen.records
MiB, GiB, INIT = frozen.MiB, frozen.GiB, frozen.INIT
KIND = 'forward-preserving-native-EOT8-operator-v2'
AUTH_KIND = 'forward-preserving-native-EOT8-university-exception-v2'
PREFIX = 'vjepa-forward-preserving-v2-20261114-'
CONTRACT = 'configs/predictor_forward_preserving_handoff_v2.json'
CONTRACT_SHA = '423bd39fffa62055610f25fa176d6ded99121401fffa4aad1d2e1d2dd4a317a8'
PACKET_SHA = frozen.PACKET_SHA
FROZEN_NAMES = (*frozen.FROZEN_NAMES, 'models/predictor_forward_preserving_records_v1.py',
                'models/predictor_forward_preserving_operator_v1.py')
REMOTE_NAMES = (*frozen.REMOTE_NAMES, 'models/predictor_forward_preserving_operator_v2.py',
                'scripts/capture_predictor_forward_preserving_v2.py', CONTRACT)
LOCAL_EXTRA = ('models/predictor_forward_preserving_client_v2.py', 'scripts/prepare_predictor_forward_preserving_v2.py',
               'scripts/run_predictor_forward_preserving_v2.py', 'tests/test_predictor_forward_preserving_child_v2.py')
exact = frozen.exact
inventory, storage, readiness, extract = frozen.inventory, frozen.storage, frozen.readiness, frozen.extract


def _copy(name):
    original = getattr(frozen, name)
    result = FunctionType(original.__code__, globals(), name, original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    return result


# New private namespace changes scope identifiers, not any frozen module globals.
validate_control = _copy('validate_control')
authorization_template = _copy('authorization_template')
validate_authorization = _copy('validate_authorization')
verify_outputs = frozen.verify_outputs
check_worker = frozen.check_worker
stream = FunctionType(frozen.stream.__code__, dict(frozen.stream.__globals__, KIND=KIND))
remote = _copy('remote')
main = _copy('main')


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
            entry = folder/'overlay/scripts/capture_predictor_forward_preserving_v2.py'
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
