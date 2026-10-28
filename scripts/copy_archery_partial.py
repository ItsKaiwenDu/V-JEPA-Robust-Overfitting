#!/usr/bin/env python3
"""Capture only existing failed Archery evidence. No remote project imports or writes."""
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import socket
import subprocess
import sys
import tarfile

KIND = 'read-only-archery-timecarry-partial-v1'
HOST, CASE = 'linux10603', 'clip-06-bank-1'
BASE = 'Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024'
PREFIX, ROOT = 'archery-timecarry-20261027', 'archery-timecarry-execution-20261027'
OPERATION = '17a122f0c6d3c35c00b7a6dcf2037583466bf9498c996941a4f449cd50e470d4'
REFERENCE = 'da5aeb93bd6d3e46040b158cc14f0d6feb1284a83b2b2778a3639b420db0acf9'
LIMIT = 350 * 2**20
LAUNCH = ('started-utc.txt', 'started-epoch.txt', 'ended-utc.txt', 'exit-status.txt',
          'case.log', 'cpu.log', 'cpu-results.xml', 'cpu-summary.log', 'cpu-old.log',
          'cpu-new.log', 'cpu-old.xml', 'cpu-new.xml', 'source-launch.sh', 'source-pipeline.sh')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe(name):
    p = PurePosixPath(name)
    if not name or name == '.' or p.is_absolute() or '..' in p.parts or '\\' in name or str(p) != name:
        raise ValueError('Unsafe incident member.')
    return name


def record(path):
    r = json.loads(Path(path).read_text())
    if set(r) != {'payload', 'sha256'} or canonical(r['payload']) != r['sha256']:
        raise ValueError('Record checksum changed: ' + str(path))
    return r['payload']


def paths(home):
    transfer = (Path(home) / BASE).resolve()
    packet = transfer / PREFIX / 'unpacked' / PREFIX
    root = transfer / ROOT
    return transfer, packet, root, root / 'cases' / CASE / 'attempt-01'


def regular(path, allowed):
    p = Path(path)
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(allowed):
        raise ValueError('Missing or escaped incident file: ' + str(path))
    relative = p.relative_to(allowed)
    if any((allowed / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts))):
        raise ValueError('Redirected incident ancestor.')
    return p


@contextmanager
def read_locks(home):
    transfer, _, root, _ = paths(home)
    # SH on existing rb descriptors is compatible with NFS record-lock access rules.
    # Both locks stay held through streaming; no file or lock is created/truncated.
    with ExitStack() as stack:
        for p in (transfer / 'two-host-panel-20261026/leases' / HOST, root / '.registry.lock'):
            regular(p, transfer)
            handle = stack.enter_context(p.open('rb'))
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield


def inventory(home, spec):
    transfer, packet, root, case = paths(home)
    reference = transfer / 'two-host-execution-20261026/unpacked/two-host-execution-20261026'
    files = {'packet/' + n: packet / n for n in (*spec['sources_sha256'], 'CONTINUATION.json',
             'INPUT.json', 'prior-OPERATION.json', 'prior-INPUT.json', 'prior-receipt.json')}
    files.update({'launcher/' + n: packet / 'launchers' / (HOST + '-' + CASE) / n for n in LAUNCH})
    files.update({'ledger/' + n: root / n for n in ('imports.json', 'resolution.json', 'time-carry.json',
        'genesis.json', 'requests/' + CASE + '.json', 'claims/' + CASE + '/claim.json',
        'outcomes/' + CASE + '.json', 'STOP.json')})
    files.update({'reference/' + n: reference / n for n in ('IMPLEMENTATION.json', 'models/two_host_panel.py')})
    names = ('session.json', 'case-envelope.json', 'matrix.log', 'hardware-matrix-before.json',
             'hardware-matrix-probe.json', 'hardware-matrix-after.json')
    files.update({'case/' + n: case / n for n in names})
    optional = {'case/' + n: case / n for n in ('numeric-evidence.tar.gz', 'numeric-evidence.tar.gz.manifest.json',
        'plan.json', 'execution.json', 'receipt.json', 'replay.log', 'preservation-before-export.json',
        'preservation-after-export.json', 'continuation-evidence.tar.gz', 'continuation-evidence.tar.gz.manifest.json',
        *('hardware-replay-' + s + '.json' for s in ('before', 'probe', 'after')),
        *(role + '/' + n for role in ('matrix', 'replay') for n in ('worker.json', 'observation.json', 'contract.json',
            'inputs.json', 'A.json', 'B.json', 'C.json', 'D.json')))}
    optional.update({'ledger/' + n + '/' + CASE + '.json': root / n / (CASE + '.json')
                     for n in ('exports', 'reviews', 'failures')})
    return files, optional


def collect_locked(home, host):
    if host != HOST:
        raise ValueError('Use the pinned linux10603 workstation.')
    transfer, packet, root, case = paths(home)
    spec = json.loads(regular(packet / 'CONTINUATION.json', transfer).read_text())
    if canonical(spec) != OPERATION or len(spec['sources_sha256']) != 27:
        raise ValueError('Wrong frozen Archery operation.')
    required, optional = inventory(home, spec)
    for n, pin in spec['sources_sha256'].items():
        safe(n)
        if digest(regular(packet / n, transfer)) != pin:
            raise ValueError('Frozen source changed: ' + n)
    if digest(regular(packet / 'INPUT.json', transfer)) != spec['inputs_file_sha256']:
        raise ValueError('Pinned preservation input changed.')
    implementation = json.loads(regular(required['reference/IMPLEMENTATION.json'], transfer).read_text())
    if (canonical(implementation) != REFERENCE or digest(regular(required['reference/models/two_host_panel.py'], transfer))
            != implementation['source_sha256']['models/two_host_panel.py']):
        raise ValueError('Original strict guard implementation changed.')
    q = record(regular(root / 'requests' / (CASE + '.json'), transfer))
    claim = record(regular(root / 'claims' / CASE / 'claim.json', transfer))
    outcome = record(regular(root / 'outcomes' / (CASE + '.json'), transfer))
    stop = record(regular(root / 'STOP.json', transfer))
    control = record(regular(case / 'case-envelope.json', transfer))
    numeric = record(regular(case / 'session.json', transfer))
    after = record(regular(case / 'hardware-matrix-after.json', transfer))
    if (q['operation_sha256'] != OPERATION or q['case_id'] != CASE or q['host'] != HOST
            or q['case_seconds'] != 4757 or q['carried_startup_seconds'] != 43 or q['original_case_seconds'] != 4800
            or claim['operation_sha256'] != OPERATION or claim['request_sha256'] != canonical(q)
            or claim['case_id'] != CASE or claim['host'] != HOST or claim['attempt'] != 'attempt-01'
            or outcome['operation_sha256'] != OPERATION or outcome['complete'] is not False
            or outcome['case_id'] != CASE or outcome['host'] != HOST or stop != outcome
            or control['operation_sha256'] != OPERATION or control['complete'] is not False
            or numeric['complete'] is not False or numeric['passed'] is not False
            or after['stage'] != 'matrix-after' or after['host'] != HOST or after['passed'] is not False
            or 'Another CUDA compute process is active' not in after['error']):
        raise ValueError('Not the pinned failed matrix guard incident; preserve for separate review.')
    if (packet / 'launchers' / (HOST + '-' + CASE) / 'exit-status.txt').read_text().strip() != '1':
        raise ValueError('Launcher has not ended with the reported failure.')
    files = dict(required)
    absent = []
    for n, p in optional.items():
        if p.exists() or p.is_symlink():
            files[n] = p
        else:
            absent.append(n)
    entries = {}
    for n, p in files.items():
        safe(n); regular(p, transfer)
        size = p.stat().st_size
        if size > (256 * 2**20 if n.endswith('.tar.gz') else 4 * 2**20):
            raise ValueError('Oversized incident artifact.')
        entries[n] = {'bytes': size, 'sha256': digest(p)}
    if sum(row['bytes'] for row in entries.values()) > LIMIT:
        raise ValueError('Incident capture exceeds its fixed bound.')
    archive = 'case/numeric-evidence.tar.gz'
    if (archive in files) != (archive + '.manifest.json' in files):
        raise ValueError('Incomplete existing numerical export pair.')
    if archive in files:
        sidecar = json.loads(files[archive + '.manifest.json'].read_text())
        if sidecar['sha256'] != entries[archive]['sha256'] or sidecar['bytes'] != entries[archive]['bytes']:
            raise ValueError('Existing partial export differs from its sidecar.')
    pins = json.loads((packet / 'INPUT.json').read_text())['preserved_files_sha256']
    if len(pins) != 392:
        raise ValueError('Wrong historical preservation scope.')
    observed = {}
    for n, pin in pins.items():
        safe(n); p = transfer / n
        try:
            regular(p, transfer); seen = digest(p)
        except ValueError:
            seen = None
        observed[n] = {'expected_sha256': pin, 'observed_sha256': seen}
    snapshot = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    manifest = {'kind': KIND, 'host': HOST, 'case_id': CASE, 'operation_sha256': OPERATION,
        'snapshot_UTC': datetime.now(timezone.utc).isoformat(), 'files': entries,
        'absent_optional_files': sorted(absent), 'coordinator_file_hashes': observed,
        'coordinator_pins_unchanged': all(v['expected_sha256'] == v['observed_sha256'] for v in observed.values()),
        'current_nvidia_smi': {'exit_status': snapshot.returncode, 'stdout': snapshot.stdout, 'stderr': snapshot.stderr},
        'host_lease_mode': 'shared-read-nonblocking', 'both_read_locks_held_through_stream': True,
        'peer_host_lease_not_acquired': True, 'university_files_changed': False,
        'model_or_test_job_started': False, 'scientific_case_accepted': False}
    return files, manifest


def collect(home, host):
    if host != HOST:
        raise ValueError('Use the pinned linux10603 workstation.')
    with read_locks(home):
        return collect_locked(home, host)


def stream(home, host):
    if host != HOST:
        raise ValueError('Use the pinned linux10603 workstation.')
    with read_locks(home):
        files, manifest = collect_locked(home, host)
        class Writer:
            def __init__(self): self.checksum, self.bytes = hashlib.sha256(), 0
            def write(self, data):
                count = sys.stdout.buffer.write(data)
                if count != len(data): raise OSError('Short incident transport write.')
                self.checksum.update(data); self.bytes += count; return count
            def flush(self): sys.stdout.buffer.flush()
        writer = Writer()
        with tarfile.open(fileobj=writer, mode='w|') as archive:
            for n, p in sorted(files.items()):
                info = tarfile.TarInfo(n); info.size = manifest['files'][n]['bytes']; info.mode = 0o600
                with p.open('rb') as source: archive.addfile(info, source)
                if digest(p) != manifest['files'][n]['sha256']: raise ValueError('Evidence changed during copy.')
            data = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
            info = tarfile.TarInfo('transport-manifest.json'); info.size = len(data); info.mode = 0o600
            archive.addfile(info, io.BytesIO(data))
        writer.flush()
        print('ARCHERY_PARTIAL_BUNDLE ' + json.dumps({'bytes': writer.bytes, 'sha256': writer.checksum.hexdigest()}), file=sys.stderr, flush=True)


def receive(directory, spec):
    directory = Path(directory); path = directory / 'transport.tar.part'
    lines = [s.removeprefix('ARCHERY_PARTIAL_BUNDLE ') for s in (directory / 'transport.stderr').read_text().splitlines() if s.startswith('ARCHERY_PARTIAL_BUNDLE ')]
    if len(lines) != 1: raise ValueError('No unique complete incident checksum; preserve partial copy.')
    transport = json.loads(lines[0])
    if (canonical(spec) != OPERATION or path.is_symlink() or not path.is_file() or path.stat().st_size > LIMIT + 2**20
            or path.stat().st_size != transport['bytes'] or digest(path) != transport['sha256']):
        raise ValueError('Incident transport checksum, size or source pin differs.')
    required, optional = inventory(Path('/unused'), spec)
    with tarfile.open(path) as archive:
        members = archive.getmembers(); names = [safe(m.name) for m in members]
        if (len(names) != len(set(names)) or not all(m.isfile() and m.size >= 0 for m in members)
                or sum(m.size for m in members) > LIMIT or 'transport-manifest.json' not in names):
            raise ValueError('Invalid incident inventory.')
        if archive.getmember('transport-manifest.json').size > 512 * 1024: raise ValueError('Oversized incident manifest.')
        with archive.extractfile('transport-manifest.json') as source: manifest = json.load(source)
        declared = set(manifest['files']); absent = manifest['absent_optional_files']
        if (manifest['kind'] != KIND or manifest['host'] != HOST or manifest['case_id'] != CASE
                or manifest['operation_sha256'] != OPERATION or not set(required) <= declared
                or not declared <= set(required) | set(optional) or sorted(set(optional) - declared) != sorted(absent)
                or len(absent) != len(set(absent)) or set(names) != declared | {'transport-manifest.json'}
                or any(manifest[k] is not False for k in ('university_files_changed', 'model_or_test_job_started', 'scientific_case_accepted'))):
            raise ValueError('Wrong incident-copy scope.')
        for member in members:
            if member.name == 'transport-manifest.json': continue
            entry = manifest['files'][member.name]
            with archive.extractfile(member) as source: checksum = hashlib.file_digest(source, 'sha256').hexdigest()
            if member.size != entry['bytes'] or checksum != entry['sha256']: raise ValueError('Incident member checksum differs.')
        output = directory / 'downloaded'; output.mkdir(exist_ok=False)
        for member in members:
            target = output / member.name; target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('xb') as destination:
                while block := source.read(2**20): destination.write(block)
    result = {'kind': KIND + '-byte-verification', 'verified': True, 'transport': transport,
        'files_verified': len(declared), 'absent_optional_files': absent, 'scientific_case_accepted': False,
        'model_repeated': False, 'independent_incident_review_complete': False}
    (directory / 'download-verification.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    path.rename(directory / 'transport.tar')
    return result


if __name__ == '__main__':
    if len(sys.argv) == 1: stream(Path.home(), socket.gethostname())
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('mode', choices=('receive',)); parser.add_argument('directory', type=Path)
        parser.add_argument('--operation', type=Path, required=True); args = parser.parse_args()
        print(json.dumps(receive(args.directory, json.loads(args.operation.read_text())), sort_keys=True, indent=2))
