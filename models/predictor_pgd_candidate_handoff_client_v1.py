"""Unarmed attended client definitions; no connection or CLI activity on import."""
import hashlib
import os
from pathlib import Path
import shlex
import stat
import subprocess
import threading
import time
from models import predictor_pgd_candidate_handoff_outer_v1 as outer
from models import predictor_pgd_candidate_bootstrap_v1 as boot
from models import predictor_pgd_candidate_collection_v1 as collection
INITIAL = "\nimport hashlib,signal,sys\nsignal.signal(signal.SIGALRM,signal.SIG_DFL)\nsignal.setitimer(signal.ITIMER_REAL,180)\ncount=int(sys.argv[1]);want=sys.argv[2]\nif not 0<count<=2*2**20:raise ValueError('Bounded exact outer source required.')\nparts=[];left=count\nwhile left:\n    data=sys.stdin.buffer.read(left)\n    if not data:raise ValueError('Truncated checked outer source.')\n    parts.append(data);left-=len(data)\ndata=b''.join(parts)\nif hashlib.sha256(data).hexdigest()!=want:raise ValueError('Outer source hash changed.')\nsignal.setitimer(signal.ITIMER_REAL,0)\nns={'__name__':'_checked_outer_connection','__file__':'<checked-stdin-outer>'}\nexec(compile(data,ns['__file__'],'exec'),ns)\nsys.exit(ns['remote_session'](sys.stdin.buffer,sys.stdout.buffer,code_sha256=want))\n"

def file_bytes(path, cap, sha256=None):
    p = Path(path)
    before = p.lstat()
    if p.resolve(strict=True) != p or not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or (before.st_nlink != 1) or (stat.S_IMODE(before.st_mode) not in (256, 384, 420)) or (not 0 < before.st_size <= cap):
        raise ValueError('Owned single-link bounded local input required.')
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as source:
        held = os.fstat(source.fileno())
        data = source.read(cap + 1)
        after = os.fstat(source.fileno())

    def sig(s):
        return (s.st_dev, s.st_ino, s.st_uid, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if sig(held) != sig(before) or sig(after) != sig(before) or sig(p.lstat()) != sig(before) or (len(data) != before.st_size) or (sha256 is not None and hashlib.sha256(data).hexdigest() != sha256):
        raise ValueError('Bound source/hash/identity changed.')
    return data

def check_local(folder):
    folder = Path(folder)
    if not (folder / 'AUTHORIZATION.json').is_file():
        raise ValueError('Unarmed: a new exact authorization is required; no SSH or work started.')
    if not folder.is_absolute() or folder.resolve(strict=True) != folder or folder.lstat().st_uid != os.getuid() or (stat.S_IMODE(folder.lstat().st_mode) != 448):
        raise ValueError('One canonical private handoff required.')
    here = Path(outer.__file__).resolve()
    code = file_bytes(here, 2 * outer.MiB)
    raw = file_bytes(folder / 'CONTROL-CANDIDATE.json', 256 * 1024)
    c, b, p, csha = outer.validate_control(raw, outer.digest(code))
    if str(folder) != c['local_handoff']:
        raise ValueError('Copied helpers cannot reset the exact local attempt identity.')
    root = Path(c['repository'])
    if root.resolve(strict=True) != root:
        raise ValueError('Canonical original repository required.')
    for name, want in c['local_source_sha256'].items():
        file_bytes(root / name, 2 * outer.MiB, want)
    if file_bytes(folder / 'known-hosts', 16 * 1024, c['known_hosts_sha256']) != outer.KNOWN_HOSTS.encode():
        raise ValueError('Pinned public host key changed.')
    araw = file_bytes(folder / 'AUTHORIZATION.json', 64 * 1024)
    a = outer.validate_authorization(araw, c, csha)
    payload = file_bytes(folder / 'SOURCE-PAYLOAD.json', 2 * outer.MiB)
    mapping = outer.parsed(payload)
    if outer.encoded(mapping) != payload:
        raise ValueError('Canonical exact source payload required.')
    outer.decode_sources(mapping, c, b)
    contract = outer.parsed(p['scratch_contract_text'])
    packet = root / c['packet_path']
    wanted = contract['packet']['archive']
    file_bytes(packet, wanted['bytes'], wanted['sha256'])
    reference = root / c['reference_path']
    wanted_reference = contract['reference']['archive']
    file_bytes(reference, wanted_reference['bytes'], wanted_reference['sha256'])
    return (c, b, p, csha, raw, araw, a, mapping, code, (packet, reference))

class ReceiverClock:
    """Convert durations with a causal LOWER bound on clock offset, never epochs.

    The client sampled sent_at before the server's handshake_received. Adding
    their difference to a server deadline yields an earlier/equal local bound.
    """

    def __init__(self, origin, sent_at, received_remote, remote_deadline, *, clock=time.monotonic):
        now = clock()
        if not all((outer.finite(v) for v in (origin, sent_at, received_remote, remote_deadline, now))) or not origin <= sent_at <= now or (not 0 < remote_deadline - received_remote <= 3000):
            raise ValueError('Original causal deadline mapping required.')
        self.clock = clock
        self.deadline = min(origin + 3000, sent_at + (remote_deadline - received_remote))
        self.last = now
        self.check()

    def check(self):
        now = self.clock()
        if not outer.finite(now) or now < self.last or now >= self.deadline:
            raise outer.Deadline('Original login-inclusive collection deadline reached; retain partial output.')
        self.last = now
        return now

class InputBudget:

    def __init__(self, source, check):
        self.source = source
        self.check = check
        self.bytes = 0

    def read(self, size):
        self.check()
        if type(size) is not int or not 0 < size <= outer.MiB or self.bytes + size > 3 * outer.GiB:
            raise ValueError('Full response byte ceiling.')
        data = self.source.read(size)
        if type(data) is not bytes or len(data) > size:
            raise ValueError('Invalid response read.')
        self.bytes += len(data)
        self.check()
        return data

def ssh_command(folder, code):
    command = 'exec "$HOME/.venvs/vjepa/bin/python" -I -B -u -c ' + shlex.quote(INITIAL) + ' ' + str(len(code)) + ' ' + outer.digest(code)
    return ['ssh', '-T', '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + str(Path(folder) / 'known-hosts'), '-o', 'GlobalKnownHostsFile=/dev/null', '-o', 'ForwardAgent=no', '-o', 'ClearAllForwardings=yes', '-o', 'PermitLocalCommand=no', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', 'kdu@' + outer.HOST, command]

class PrefixReader:

    def __init__(self, prefix, source):
        self.prefix = prefix
        self.source = source

    def read(self, size):
        if self.prefix:
            result = self.prefix[:size]
            self.prefix = self.prefix[len(result):]
            return result
        return self.source.read(size)

def bound_collection_source(incoming, window, claimed, ready, c, derived):
    decoder = collection.WireReader(incoming, window)
    raw, h = decoder.frame(collection.MAGIC, collection.HEADER_CAP)
    collection.validate_header(h, c['supervised_control_text'].encode(), derived)
    base = h['base']
    if base['root_binding']['root'] != claimed['scratch'] or base['root_binding']['parent'] != '/var/tmp' or base['connection_started'] != claimed['connection_started'] or (base['collection_started'] != ready['collection_started']) or (base['collection_deadline'] != ready['collection_deadline']):
        raise ValueError('Collection header belongs to another root or clock; preserve the original claim.')
    prefix = collection.MAGIC + len(raw).to_bytes(4, 'big') + raw
    return PrefixReader(prefix, incoming)

def run(folder):
    """Future attended one-shot API. No download-only or alternate-host mode."""
    started = time.monotonic()
    folder = Path(folder)
    if not (folder / 'AUTHORIZATION.json').is_file():
        raise ValueError('Unarmed: no new exact authorization; no timer, claim or SSH created.')
    alarms = outer.Alarms()
    proc = None
    sender = None
    errors = []
    stderr_thread = None
    session = None
    alarms.arm(started + 3000, hard_deadline=started + 3000, phase='client-connection')
    try:
        c, b, p, csha, raw, araw, auth, mapping, code, packet = check_local(folder)
        outer.private_write(folder, 'ATTEMPT.json', outer.encoded({'kind': 'pgd-split-local-ATTEMPT-v1', 'consumed': True, 'outer_control_sha256': csha, 'authorization_sha256': outer.digest(araw), 'started_monotonic': started, **outer.FLAGS}), alarms.check)
        session = folder / 'attended-session'
        session.mkdir(mode=448)
        proc = subprocess.Popen(ssh_command(folder, code), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True, shell=False, bufsize=0)

        def drain():
            try:
                count = 0
                with os.fdopen(os.open(session / 'ssh-stderr.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 384), 'wb') as out:
                    while True:
                        data = proc.stderr.read(4096)
                        if not data:
                            break
                        if count + len(data) > outer.MiB:
                            errors.append('SSH stderr exceeded 1 MiB; transport stopped.')
                            proc.terminate()
                            break
                        out.write(data)
                        out.flush()
                        count += len(data)
            except BaseException as error:
                errors.append('stderr: ' + type(error).__name__)
        stderr_thread = threading.Thread(target=drain, daemon=True)
        stderr_thread.start()

        def write_source():
            try:
                left = memoryview(code)
                while left:
                    n = os.write(proc.stdin.fileno(), left)
                    if n <= 0:
                        raise OSError('Incomplete checked bootstrap upload.')
                    left = left[n:]
            except BaseException as error:
                errors.append('bootstrap: ' + type(error).__name__)
                proc.terminate()
        first = threading.Thread(target=write_source, daemon=True)
        first.start()
        incoming = InputBudget(proc.stdout, alarms.check)
        hello = outer.frame(incoming, alarms.check)
        first.join(timeout=0)
        if type(hello) is not dict or set(hello) != {'kind', 'nonce', 'hello_monotonic', 'code_sha256'} or hello['kind'] != 'pgd-split-hello-v1' or (hello['code_sha256'] != c['outer_source_sha256']) or (not outer.sha(hello['nonce'])) or (not outer.finite(hello['hello_monotonic'])):
            raise ValueError('Fresh checked server hello required.')
        sent = time.monotonic()
        elapsed = sent - started
        message = {'nonce': hello['nonce'], 'client_elapsed': elapsed, 'control_text': raw.decode(), 'authorization_text': araw.decode(), 'sources': mapping}
        if not 0 <= elapsed < 180:
            raise outer.Deadline('Login consumed the original setup allowance; no target work authorized by this connection.')

        def upload():
            try:
                outer.send_frame(proc.stdin, message, lambda: None, cap=2 * outer.MiB)
                for archive in packet:
                    with archive.open('rb') as source:
                        while True:
                            block = source.read(outer.MiB)
                            if not block:
                                break
                            outer.write_all(proc.stdin, block, lambda: None)
                proc.stdin.close()
            except BaseException as error:
                errors.append('upload: ' + type(error).__name__)
            finally:
                try:
                    proc.stdin.close()
                except BaseException as error:
                    errors.append('stdin close: ' + type(error).__name__)
        sender = threading.Thread(target=upload, daemon=True)
        sender.start()
        claimed = outer.frame(incoming, alarms.check)
        if claimed.get('kind') == 'pgd-split-outer-refusal-v1':
            outer.private_write(session, 'REFUSAL.json', outer.encoded(claimed), alarms.check)
            raise RuntimeError(claimed['message'])
        scratch = Path(claimed.get('scratch', ''))
        stem = scratch.name.removeprefix('vjepa-pgd-candidate-v1-').removesuffix('-' + csha)
        if claimed.get('kind') != 'pgd-split-root-claimed-v1' or claimed.get('outer_control_sha256') != csha or scratch.parent != Path('/var/tmp') or (scratch.name != 'vjepa-pgd-candidate-v1-' + stem + '-' + csha) or (not stem.isdigit()) or (claimed.get('hello_monotonic') != hello['hello_monotonic']) or (not outer.finite(claimed.get('handshake_received'))) or (claimed['handshake_received'] < hello['hello_monotonic']) or (claimed.get('connection_started') != hello['hello_monotonic'] - elapsed) or (claimed.get('work_deadline') != claimed['connection_started'] + 2400) or (claimed.get('connection_deadline') != claimed['connection_started'] + 3000) or any((claimed.get(k) is not False for k in outer.FLAGS)):
            raise ValueError('Durable remote claim or continuing connection clock changed.')
        outer.private_write(session, 'ROOT-RECEIPT.json', outer.encoded(claimed), alarms.check)
        ready = outer.frame(incoming, alarms.check)
        if ready.get('kind') == 'pgd-split-outer-refusal-v1':
            outer.private_write(session, 'REFUSAL.json', outer.encoded(ready), alarms.check)
            raise RuntimeError(ready['message'])
        start = ready.get('collection_started')
        end = ready.get('collection_deadline')
        if ready.get('kind') != 'pgd-split-collection-ready-v1' or ready.get('outer_control_sha256') != csha or ready.get('connection_started') != claimed['connection_started'] or (ready.get('handshake_received') != claimed['handshake_received']) or (not outer.finite(start)) or (not outer.finite(end)) or (not claimed['handshake_received'] <= start < end) or (end != min(start + 600, claimed['connection_deadline'])) or any((ready.get(k) is not False for k in outer.FLAGS)):
            raise ValueError('Original collection phase/deadline changed.')
        window = ReceiverClock(started, sent, claimed['handshake_received'], end)
        alarms.arm(window.deadline, hard_deadline=started + 3000, phase='client-collection')
        outer.private_write(session, 'COLLECTION-READY.json', outer.encoded(ready), window.check)
        derived = boot.encoded({'kind': 'pgd-candidate-supervised-university-exception-v1', 'approved': True, 'control_sha256': c['supervised_control_sha256'], 'maximum_new_attempts': 1, 'user_basis': auth['user_basis']})
        incoming = bound_collection_source(incoming, window, claimed, ready, c, derived)
        receipt = collection.receive(incoming, session / 'saved', c['supervised_control_text'].encode(), derived, window=window)
        status = proc.wait(timeout=max(0, window.deadline - time.monotonic()))
        sender.join(timeout=0)
        stderr_thread.join(timeout=0)
        if errors:
            raise RuntimeError('Bounded SSH transport failed: ' + errors[0])
        receipt.update(outer_control_sha256=csha, SSH_exit_status=status, independent_review_pending=True)
        outer.private_write(session, 'COLLECTION-RECEIPT.json', outer.encoded(receipt), window.check)
        print('Evidence saved at ' + str(session) + '. Stop for independent review; never repeat this launch.', flush=True)
        return status
    except BaseException as error:
        if session is not None:
            try:
                outer.private_write(session, 'CLIENT-REFUSAL.json', outer.encoded({'kind': 'pgd-split-client-refusal-v1', 'error_type': type(error).__name__, 'message': str(error)[:1024], 'consumed': True, **outer.FLAGS}), alarms.check)
            except BaseException:
                pass
        raise
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=max(0, min(2, started + 3000 - time.monotonic())))
            except subprocess.TimeoutExpired:
                proc.kill()
                if time.monotonic() < started + 3000:
                    proc.wait(timeout=max(0, min(0.1, started + 3000 - time.monotonic())))
        alarms.close()
