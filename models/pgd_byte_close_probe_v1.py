"""One separately authorized inert byte probe, with one same-SSH response.

Preparation is local and unarmed. This is not the prior model/read grant. The
frozen Store methods are copied unchanged into a small one-file byte namespace.
"""
import ast
from datetime import datetime
import os
from pathlib import Path
import secrets
import shlex
import stat
import time
from models import pgd_byte_close_core_v1 as core
from models import university_gpu_availability_v1 as local
from scripts import university_gpu_availability_remote_v1 as transport

HOST = 'linux10609.dc.engr.scu.edu'
FINGERPRINT = 'SHA256:ARgVRWOzRLA1LmAANxD6Ya17EIIo2sa4ZbHvkeqzgKQ'
KNOWN_HOSTS = 'linux10609.dc.engr.scu.edu ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAELc+tu7f+683dQbOPg5lndZ9pGDeODvo8r1Sp8d7ri\n'
FROZEN = 'models/predictor_pgd_split_store_v1.py'
CANDIDATE = 'runs/pretrained/diagnostic/predictor-pgd-store-close-v2-20261116/UNEXECUTED-store-close-v2.py'
FROZEN_SHA = '2e48a86724bf8d8e49981d7b4dd21875a5f31010e54a558488d16034fb93bb54'
CANDIDATE_SHA = '7bcbd863eab7c23940fa1d32b194448b88353202c2100b66a9a30f59e4ef6022'
SOURCES = ('models/pgd_byte_close_core_v1.py', 'models/pgd_byte_close_probe_v1.py',
           'scripts/run_pgd_byte_close_probe_v1.py', FROZEN, CANDIDATE,
           'models/university_gpu_availability_v1.py', 'scripts/university_gpu_availability_remote_v1.py')
NAMES = {'CONTROL.json', 'PROBE.py', 'known-hosts', 'PACKAGE.json', 'BYTE-AUTHORIZATION.json',
         'BYTE-ATTEMPT.json', 'STDOUT.bin', 'STDERR.bin', 'RESULT.json'}

encoded, digest, read_file, parse = core.encoded, core.digest, local.read_file, local.parse


def private_folder(folder):
    folder = Path(folder)
    s = folder.lstat()
    if (not folder.is_absolute() or folder.resolve(strict=True) != folder or not stat.S_ISDIR(s.st_mode)
            or s.st_uid != os.getuid() or stat.S_IMODE(s.st_mode) != 0o700):
        raise ValueError('Canonical owned private handoff directory required.')
    return folder


def write_new(root, name, data):
    if name not in NAMES or type(data) is not bytes or len(data) > 65536:
        raise ValueError('Only bounded byte-probe metadata/source files may be created.')
    root = private_folder(root)
    with os.fdopen(os.open(root/name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def definition(raw, name):
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
    if len(nodes) != 1:
        raise ValueError('One exact source definition required.')
    n = nodes[0]
    return b''.join(raw.splitlines(keepends=True)[n.lineno-1:n.end_lineno])


def build(source):
    if set(source) != set(SOURCES) or digest(source[FROZEN]) != FROZEN_SHA or digest(source[CANDIDATE]) != CANDIDATE_SHA:
        raise ValueError('Exact previously reviewed writer sources required.')
    parts = [source[SOURCES[0]].rstrip()+b'\n\n']
    for name in ('identity', 'private'):
        if definition(source[FROZEN], name) != definition(source[CANDIDATE], name):
            raise ValueError('Original private/identity helper changed.')
        parts.append(definition(source[FROZEN], name)+b'\n\n')
    for name, label in ((FROZEN, b'FrozenStore'), (CANDIDATE, b'CandidateStore')):
        cls = definition(source[name], 'Store')
        if cls.count(b'class Store:') != 1:
            raise ValueError('Exact original class label required.')
        parts.append(cls.replace(b'class Store:', b'class '+label+b':', 1)+b'\n\n')
    parts.append(b"if __name__ == '__main__':\n    sys.exit(main())\n")
    program = b''.join(parts)
    if len(program) > 65536:
        raise ValueError('Small stdin-free byte-probe command source bound exceeded.')
    compile(program, '<UNEXECUTED-BYTE-CLOSE-PROBE>', 'exec')
    return program


def control(raw):
    c = parse(raw)
    keys = {'kind', 'repository', 'handoff', 'host', 'user', 'fingerprint', 'known_hosts_sha256',
            'program_sha256', 'program_bytes', 'source_pins', 'limits', 'flags', 'data_sha256'}
    if (type(c) is not dict or set(c) != keys or encoded(c) != raw or c['kind'] != 'pgd-byte-close-control-v1'
            or c['host'] != HOST or c['user'] != 'kdu' or c['fingerprint'] != FINGERPRINT
            or c['known_hosts_sha256'] != digest(KNOWN_HOSTS.encode()) or c['data_sha256'] != core.DATA_SHA
            or encoded(c['limits']) != encoded(core.LIMITS) or encoded(c['flags']) != encoded(core.FLAGS)
            or type(c['program_bytes']) is not int or not 0 < c['program_bytes'] <= 65536
            or not core.is_sha(c['program_sha256']) or type(c['source_pins']) is not dict
            or set(c['source_pins']) != set(SOURCES) or any(not core.is_sha(v) for v in c['source_pins'].values())
            or c['source_pins'][FROZEN] != FROZEN_SHA or c['source_pins'][CANDIDATE] != CANDIDATE_SHA):
        raise ValueError('Exact byte-only host/source/limits/control required.')
    for name in ('repository', 'handoff'):
        if type(c[name]) is not str:
            raise ValueError('Bound absolute local path required.')
        p = Path(c[name])
        if not p.is_absolute() or str(p) != c[name] or '..' in p.parts:
            raise ValueError('Bound canonical local path required.')
    return c, digest(raw)


def authorization(csha, basis):
    if not core.is_sha(csha) or type(basis) is not str or not 1 <= len(basis) <= 8192:
        raise ValueError('One exact new byte-probe approval basis required.')
    return {'kind': 'pgd-byte-close-authorization-v1', 'approved': True, 'control_sha256': csha,
            'host': HOST, 'limits': core.LIMITS, 'flags': core.FLAGS, 'basis': basis}


def prepare(repository, destination):
    root, dest = Path(repository), Path(destination)
    if root.resolve(strict=True) != root or not root.is_absolute() or not dest.is_absolute() or os.path.lexists(dest):
        raise ValueError('Fresh absolute bound handoff required.')
    private_folder(dest.parent)
    source = {n: read_file(root/n, 65536) for n in SOURCES}
    program = build(source)
    c = {'kind': 'pgd-byte-close-control-v1', 'repository': str(root), 'handoff': str(dest), 'host': HOST,
         'user': 'kdu', 'fingerprint': FINGERPRINT, 'known_hosts_sha256': digest(KNOWN_HOSTS.encode()),
         'program_sha256': digest(program), 'program_bytes': len(program), 'source_pins': {n: digest(v) for n, v in source.items()},
         'limits': core.LIMITS, 'flags': core.FLAGS, 'data_sha256': core.DATA_SHA}
    raw = encoded(c)
    _, csha = control(raw)
    result = {'kind': 'pgd-byte-close-unarmed-package-v1', 'control_sha256': csha, 'program_sha256': digest(program),
              'program_bytes': len(program), 'authorization_created': False, 'SSH_or_remote_work_started': False, **core.FLAGS}
    dest.mkdir(mode=0o700)
    for n, data in (('CONTROL.json', raw), ('PROBE.py', program), ('known-hosts', KNOWN_HOSTS.encode()), ('PACKAGE.json', encoded(result))):
        write_new(dest, n, data)
    return result


def check_local(folder):
    folder = Path(folder)
    if not (folder/'BYTE-AUTHORIZATION.json').is_file():
        raise ValueError('Unarmed: a new exact byte-probe approval is required before SSH.')
    private_folder(folder)
    c, csha = control(read_file(folder/'CONTROL.json', 16384))
    if str(folder) != c['handoff']:
        raise ValueError('Copied handoff cannot reuse an attempt grant.')
    source = {n: read_file(Path(c['repository'])/n, 65536, sha) for n, sha in c['source_pins'].items()}
    program = read_file(folder/'PROBE.py', 65536, c['program_sha256'])
    if (program != build(source) or len(program) != c['program_bytes']
            or read_file(folder/'known-hosts', 1024, c['known_hosts_sha256']) != KNOWN_HOSTS.encode()):
        raise ValueError('Exact rebuilt program and pinned public host required.')
    raw = read_file(folder/'BYTE-AUTHORIZATION.json', 16384)
    a = parse(raw)
    if type(a) is not dict or encoded(authorization(csha, a.get('basis'))) != raw:
        raise ValueError('Only one new bound byte-probe grant is accepted.')
    if os.path.lexists(folder/'BYTE-ATTEMPT.json') or os.path.lexists(folder/'byte-response'):
        raise ValueError('Attempt consumed; preserve evidence and never repeat.')
    return c, csha, program, digest(raw)


def ssh_command(folder, c, csha, program, nonce):
    if not core.is_sha(nonce) or not core.is_sha(csha) or digest(program) != c['program_sha256']:
        raise ValueError('Pinned public request bindings required.')
    command = ('exec /usr/bin/timeout --signal=TERM --kill-after=2s 65s "$HOME/.venvs/vjepa/bin/python" -I -S -B -u -c '
               +shlex.quote(program.decode())+' '+shlex.quote(nonce)+' '+shlex.quote(csha)+' '+shlex.quote(c['program_sha256']))
    if len(command.encode()) > 100000:
        raise ValueError('Bounded SSH command required.')
    return ['ssh', '-F', '/dev/null', '-T', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'UserKnownHostsFile='+str(Path(folder)/'known-hosts'), '-o', 'GlobalKnownHostsFile=/dev/null',
            '-o', 'ForwardAgent=no', '-o', 'ClearAllForwardings=yes', '-o', 'PermitLocalCommand=no',
            '-o', 'PubkeyAuthentication=no', '-o', 'PreferredAuthentications=keyboard-interactive,password',
            '-o', 'NumberOfPasswordPrompts=1', '-o', 'ConnectionAttempts=1', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2', 'kdu@'+HOST, command]


def valid_error(v):
    if v is None:
        return
    if (type(v) is not dict or set(v) != {'type', 'message', 'notes'}
            or type(v['type']) is not str or not 1 <= len(v['type']) <= 128
            or type(v['message']) is not str or len(v['message']) > 512
            or type(v['notes']) is not list or len(v['notes']) > 2
            or any(type(n) is not str or len(n) > 2200 for n in v['notes'])):
        raise ValueError('Bounded typed failure record required.')


def valid_row(v):
    if v is None:
        return
    if (type(v) is not dict or set(v) != set(core.FIELDS)
            or any(type(v[n]) is not int or v[n] < 0 for n in ('bytes', 'allocated_bytes', 'mtime_ns'))
            or v['bytes'] > core.FILE_SIZE or v['allocated_bytes'] > core.LIMITS['allocated_bytes']
            or type(v['identity']) is not list or len(v['identity']) != 5
            or any(type(n) is not int or n < 0 for n in v['identity'])):
        raise ValueError('Bounded byte metadata row required.')


def validate_response(raw, c, csha, nonce):
    if type(raw) is not bytes or not 0 < len(raw) <= 65536:
        raise ValueError('One bounded complete byte observation required.')
    r = parse(raw)
    keys = {'kind', 'request_nonce', 'control_sha256', 'program_sha256', 'checked_utc', 'preflight', 'root',
            'cases', 'footprint', 'error', 'assessment', 'elapsed_seconds', *core.FLAGS}
    if (type(r) is not dict or set(r) != keys or encoded(r) != raw or r['kind'] != 'pgd-byte-close-observation-v1'
            or r['request_nonce'] != nonce or r['control_sha256'] != csha or r['program_sha256'] != c['program_sha256']
            or any(type(r[k]) is not type(v) or r[k] != v for k, v in core.FLAGS.items())
            or type(r['elapsed_seconds']) not in (int, float) or not 0 <= r['elapsed_seconds'] <= 65):
        raise ValueError('Exact byte-probe response bindings and clock required.')
    stamp = datetime.fromisoformat(r['checked_utc'])
    if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError('UTC observation timestamp required.')
    valid_error(r['error'])
    p = r['preflight']
    if p is not None:
        pkeys = {'host', 'uid', 'device', 'filesystem', 'parent_identity', 'free_bytes_observed', 'free_inodes_observed'}
        if (type(p) is not dict or set(p) != pkeys or p['host'] != core.EXPECTED_HOST or p['filesystem'] != 'xfs'
                or type(p['device']) is not int or p['device'] != 66307 or type(p['uid']) is not int or p['uid'] <= 0
                or type(p['parent_identity']) is not list or len(p['parent_identity']) != 4
                or any(type(n) is not int or n < 0 for n in p['parent_identity'])
                or p['parent_identity'][0] != p['device'] or p['parent_identity'][2:] != [0, 0o1777]
                or type(p['free_bytes_observed']) is not int or p['free_bytes_observed'] < 64*core.MiB
                or type(p['free_inodes_observed']) is not int or p['free_inodes_observed'] < 64):
            raise ValueError('Observed original host/XFS/private scratch parent required.')
    if r['root'] is not None and (p is None or r['root'] != '/var/tmp/vjepa-byte-close-v1-'+str(p['uid'])+'-'+csha):
        raise ValueError('Derived fresh byte-only root required.')
    if type(r['cases']) is not list or len(r['cases']) > 2:
        raise ValueError('At most two serial byte observations required.')
    for index, case in enumerate(r['cases']):
        ckeys = {'kind', 'receipt', 'write_error', 'committed', 'close_observations', 'before_read',
                 'check_before_error', 'readback', 'read_error', 'check_after_error', 'classification'}
        if type(case) is not dict or set(case) != ckeys or case['kind'] != ('frozen', 'candidate')[index]:
            raise ValueError('One original then one candidate only.')
        for k in ('write_error', 'check_before_error', 'read_error', 'check_after_error'):
            valid_error(case[k])
        for k in ('committed', 'before_read'):
            valid_row(case[k])
        receipt = case['receipt']
        if receipt is not None and (type(receipt) is not dict or set(receipt) != {'file', 'bytes', 'sha256'}
                or receipt['file'] != 'pixels.npy' or type(receipt['bytes']) is not int
                or receipt['bytes'] != core.FILE_SIZE or not core.is_sha(receipt['sha256'])):
            raise ValueError('One inert-byte publication receipt required.')
        rb = case['readback']
        if rb is not None:
            if (type(rb) is not dict or set(rb) != {'bytes', 'sha256', 'before', 'after'}
                    or type(rb['bytes']) is not int or rb['bytes'] != core.FILE_SIZE or not core.is_sha(rb['sha256'])):
                raise ValueError('One bounded byte hash record required.')
            valid_row(rb['before']); valid_row(rb['after'])
        if type(case['close_observations']) is not list or len(case['close_observations']) > 1:
            raise ValueError('At most one owned publication close per case.')
        for o in case['close_observations']:
            if type(o) is not dict or set(o) != {'before_held', 'before_named', 'after_named', 'committed_at_close', 'observation_errors', 'close_error'}:
                raise ValueError('Exact before/after-owned-close observation required.')
            for k in ('before_held', 'before_named', 'after_named', 'committed_at_close'):
                valid_row(o[k])
            valid_error(o['close_error'])
            if type(o['observation_errors']) is not list or len(o['observation_errors']) > 2:
                raise ValueError('Bounded observer errors required.')
            for e in o['observation_errors']:
                valid_error(e)
        if case['classification'] != core.classify_case(case):
            raise ValueError('Case classification differs from its saved metadata/byte evidence.')
    if len(r['cases']) > 1 and r['cases'][0]['classification'] not in ('consistent', 'allocation_only_refusal'):
        raise ValueError('Later case ran after an inconclusive original.')
    footprint = r['footprint']
    if footprint is not None:
        if type(footprint) is not dict or set(footprint) != {'logical_bytes', 'allocated_bytes', 'files', 'directories'}:
            raise ValueError('Exact footprint record required.')
        if any(type(v) is not int or not 0 <= v <= core.LIMITS[k] for k, v in footprint.items()):
            raise ValueError('Byte footprint exceeded its scope.')
    assessed = core.assessment(r['cases']) if r['error'] is None else 'inconclusive'
    if r['assessment'] != assessed or (r['assessment'] != 'inconclusive' and (p is None or r['root'] is None or footprint is None)):
        raise ValueError('Assessment differs from complete saved evidence.')
    return r


def run(folder, *, clock=None, capture_fn=None, nonce_fn=None):
    folder = Path(folder)
    if not (folder/'BYTE-AUTHORIZATION.json').is_file():
        raise ValueError('Unarmed: no byte-probe approval; no connection started.')
    clock = time.monotonic if clock is None else clock
    started = clock()
    deadline = started+180
    c, csha, program, asha = check_local(folder)
    nonce = secrets.token_hex(32) if nonce_fn is None else nonce_fn()
    argv = ssh_command(folder, c, csha, program, nonce)
    if clock() >= deadline:
        raise TimeoutError('Original client window consumed before connection.')
    write_new(folder, 'BYTE-ATTEMPT.json', encoded({'kind': 'pgd-byte-close-attempt-v1', 'consumed': True,
              'control_sha256': csha, 'authorization_sha256': asha, 'request_nonce': nonce,
              'started_monotonic': started, 'deadline_monotonic': deadline, **core.FLAGS}))
    session = folder/'byte-response'
    session.mkdir(mode=0o700)
    out = err = b''
    status = observation = failure = None
    try:
        capture_fn = transport.capture if capture_fn is None else capture_fn
        status, out, err = capture_fn(argv, deadline=deadline, stdout_cap=65536, stderr_cap=16384)
        if type(out) is not bytes or type(err) is not bytes or len(out) > 65536 or len(err) > 16384:
            raise ValueError('Bounded SSH response required.')
        if clock() >= deadline:
            raise transport.CaptureError('Original client deadline reached.', out, err)
        observation = validate_response(out, c, csha, nonce)
        if type(status) is not int or status != 0 or observation['assessment'] == 'inconclusive':
            failure = 'Byte comparison incomplete or refused; retain all evidence and do not repeat.'
    except (transport.CaptureError, ValueError, TypeError, KeyError, OSError, KeyboardInterrupt) as error:
        out = bytes(getattr(error, 'stdout', out))[:65536]
        err = bytes(getattr(error, 'stderr', err))[:16384]
        failure = type(error).__name__+': '+str(error)[:512]
    write_new(session, 'STDOUT.bin', out)
    write_new(session, 'STDERR.bin', err)
    record = {'kind': 'pgd-byte-close-client-result-v1', 'control_sha256': csha, 'authorization_sha256': asha,
              'request_nonce': nonce, 'SSH_exit_status': status, 'stdout_bytes': len(out), 'stdout_sha256': digest(out),
              'stderr_bytes': len(err), 'stderr_sha256': digest(err), 'response_validated': observation is not None,
              'assessment': observation['assessment'] if observation is not None else 'inconclusive',
              'error': failure, 'client_elapsed_seconds': clock()-started, **core.FLAGS}
    write_new(session, 'RESULT.json', encoded(record))
    print('Byte-probe evidence saved at '+str(session)+'. No model work ran. Do not repeat this command.')
    return 0 if failure is None else 1
