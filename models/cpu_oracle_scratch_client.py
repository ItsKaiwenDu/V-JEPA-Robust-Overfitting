"""Attended Mac operator: explicit grant, one SSH stream, verified saved evidence."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

from models.cpu_oracle_scratch import (KIND, POLICY, digest, json_bytes, safe_name,
    validate_authorization, validate_control)

ROOT = Path(__file__).resolve().parents[1]


def verify_local(base, expected_sha, *, download_only=False):
    base = Path(base)
    control_path = base / 'CONTROL.json'
    if control_path.is_symlink() or digest(control_path) != expected_sha:
        raise ValueError('Scratch operator control changed.')
    control = json.loads(control_path.read_text())
    validate_control(control)
    if json_bytes(control) != control_path.read_bytes():
        raise ValueError('Scratch control must retain canonical bytes.')
    for name, expected in control['local_source_sha256'].items():
        safe_name(name)
        path = ROOT / name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError('Local scratch operator source changed: ' + name)
    for name, expected in control['prior_claims_sha256'].items():
        safe_name(name)
        if digest(ROOT / name) != expected:
            raise ValueError('Consumed prior claim changed: ' + name)
    if digest(base / 'known-hosts') != control['known_hosts_sha256']:
        raise ValueError('Pinned university host key changed.')
    if shutil.disk_usage(base).free < 8 * 2**30:
        raise ValueError('Require 8 GiB available on the Mac before a run or copy-only retrieval.')
    if not download_only:
        archive = ROOT / control['original_archive_path']
        if archive.is_symlink() or archive.stat().st_size != control['archive_bytes'] or digest(archive) != control['archive_sha256']:
            raise ValueError('Original CPU packet changed.')
        if not (base / 'AUTHORIZATION.json').is_file():
            raise ValueError('UNARMED: a separate explicit CPU scratch resource exception is required before SSH.')
        authorization = json.loads((base / 'AUTHORIZATION.json').read_text())
        validate_authorization(authorization, expected_sha, control['host'])
        if control['host'] == 'linux10609':
            review = json.loads((base.parent / 'target-independent-review.json').read_text())
            if (review.get('kind') != 'cpu-oracle-scratch-independent-review-v1'
                    or review.get('host') != 'linux10603' or review.get('independent_review_complete') is not True
                    or review.get('control_sha256') != control['target_control_sha256']):
                raise ValueError('The control host requires independent review of the new target run.')
    return control


def command_for(base, control, mode='run', scratch=None):
    base = Path(base)
    program = (ROOT / 'models/cpu_oracle_scratch.py').read_bytes()
    encoded = base64.b64encode(program).decode()
    control_encoded = base64.b64encode(json_bytes(control)).decode()
    bootstrap = ('import base64,hashlib;code=base64.b64decode(' + repr(encoded) + ');'
        'assert hashlib.sha256(code).hexdigest()==' + repr(hashlib.sha256(program).hexdigest()) + ';'
        'ns={"__name__":"scratch_remote_operator"};exec(compile(code,"<cpu_scratch_operator>","exec"),ns);'
        'ns["remote_main"](' + repr(control_encoded) + ',' + repr(mode) + ',' + repr(scratch) + ')')
    remote = '"$HOME/.venvs/vjepa/bin/python" -B -u -c ' + shlex.quote(bootstrap)
    return ['ssh', '-T', '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + str(base / 'known-hosts'),
        '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'ClearAllForwardings=yes',
        '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
        'kdu@' + control['host'] + '.dc.engr.scu.edu', remote]


def first_header(response, control, control_sha):
    with Path(response).open('rb') as handle:
        line = handle.readline(POLICY['max_header_bytes'] + 1)
    if len(line) > POLICY['max_header_bytes'] or not line.endswith(b'\n'):
        raise ValueError('Missing bounded scratch-location prelude; preserve partial response.')
    first = json.loads(line)
    folder = Path(first['scratch_directory'])
    if (first.get('phase') != 'scratch-created' or first.get('kind') != KIND
            or first.get('host') != control['host'] or first.get('control_sha256') != control_sha
            or folder.parent != Path('/var/tmp')
            or not folder.name.startswith('vjepa-cpu-oracle-20261104-' + control['host'] + '-')):
        raise ValueError('Unexpected host, control or scratch location in prelude.')
    return first


def verify_response(response, destination, control, control_sha):
    response, destination = Path(response), Path(destination)
    first = first_header(response, control, control_sha)
    if destination.exists():
        raise FileExistsError('Preserve the existing extraction directory.')
    with response.open('rb') as incoming:
        incoming.readline(POLICY['max_header_bytes'] + 1)
        line = incoming.readline(POLICY['max_header_bytes'] + 1)
        if len(line) > POLICY['max_header_bytes'] or not line.endswith(b'\n'):
            raise ValueError('Missing complete collection manifest; never rerun the reference.')
        header = json.loads(line)
        files = header['files']
        if (header.get('phase') != 'collection' or header.get('kind') != KIND
                or header.get('control_sha256') != control_sha
                or header.get('scientific_review_complete') is not False
                or header.get('model_restarted_for_collection') is not False
                or not 1 <= len(files) <= POLICY['max_files']):
            raise ValueError('Wrong collection domain or declared member count.')
        total = 0
        for name, item in files.items():
            safe_name(name)
            if (type(item['bytes']) is not int or item['bytes'] < 0
                    or type(item['mtime_ns']) is not int or item['mtime_ns'] < 0
                    or len(item['sha256']) != 64):
                raise ValueError('Malformed collection extent or hash.')
            total += item['bytes']
        if total != header['logical_bytes'] or total > POLICY['max_collection_bytes']:
            raise ValueError('Collection exceeds its total byte bound.')
        if shutil.disk_usage(destination.parent).free < total + 2**30:
            raise ValueError('Insufficient Mac space for the verified extraction; retain response for copy-only review.')
        destination.mkdir(mode=0o700)
        seen = set()
        with tarfile.open(mode='r|', fileobj=incoming) as archive:
            for member in archive:
                name = safe_name(member.name)
                if not member.isfile() or name in seen or name not in files or member.size != files[name]['bytes']:
                    raise ValueError('Unexpected, duplicate, linked or changed collection member.')
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                computed = hashlib.sha256()
                with archive.extractfile(member) as source, target.open('xb') as out:
                    for block in iter(lambda: source.read(2**20), b''):
                        computed.update(block)
                        out.write(block)
                    out.flush()
                    os.fsync(out.fileno())
                if computed.hexdigest() != files[name]['sha256']:
                    raise ValueError('Collection member checksum differs: ' + name)
                os.utime(target, ns=(files[name]['mtime_ns'], files[name]['mtime_ns']))
                seen.add(name)
        if seen != set(files):
            raise ValueError('Incomplete collection; retain partial extraction.')
    if json.loads((destination / 'operator-control.json').read_text()) != control:
        raise ValueError('Returned control differs from the reviewed source/scope.')
    state = json.loads((destination / 'operator-result.json').read_text())
    if (state.get('control_sha256') != control_sha or state.get('host') != control['host']
            or state.get('run_phase_finished') is not True or state.get('CPU_only') is not True
            or state.get('GPU_work') is not False or state.get('paid_compute') is not False
            or state.get('scientific_review_complete') is not False):
        raise ValueError('Missing or relabelled finished CPU attempt receipt.')
    for name, item in control['archive_manifest']['files'].items():
        returned = destination / 'code' / name
        if returned.exists() and (returned.stat().st_size != item['bytes'] or digest(returned) != item['sha256']):
            raise ValueError('Returned original packet member changed: ' + name)
    receipt = {'kind': 'cpu-oracle-scratch-transport-review-v1', 'host': control['host'],
        'control_sha256': control_sha, 'scratch_directory': first['scratch_directory'],
        'response_sha256': digest(response), 'manifest_sha256': hashlib.sha256(line).hexdigest(),
        'members_verified': len(seen), 'logical_bytes_verified': total,
        'transport_verified': True, 'reference_process_started': state['reference_process_started'],
        'reference_exit_status': state.get('reference_exit_status'), 'technical_error': state.get('error'),
        'scientific_review_complete': False, 'old_refusals_waived': False,
        'remote_scratch_deleted': False, 'model_restarted_for_collection': False}
    (destination.parent / 'transport-verification.json').write_bytes(json_bytes(receipt))
    return receipt


def run_client(base, control_sha, *, download_only=False, ssh_override=None):
    base = Path(base)
    control = verify_local(base, control_sha, download_only=download_only)
    scratch = None
    if download_only:
        attempt = json.loads((base / 'ATTEMPT.json').read_text())
        original = base / attempt['local_session'] / 'response.bin'
        scratch = first_header(original, control, control_sha)['scratch_directory']
    elif (base / 'ATTEMPT.json').exists():
        raise FileExistsError('This new attempt is already claimed; use download only after review.')
    prefix = 'download-only-' if download_only else 'attended-'
    session = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
    if not download_only:
        # This immutable new claim precedes all network access. A failed login
        # is reviewed before any resource exception can be carried forward.
        with (base / 'ATTEMPT.json').open('xb') as handle:
            handle.write(json_bytes({'kind': KIND, 'control_sha256': control_sha,
                'invoked_unix': time.time(), 'local_session': session.name, 'single_attempt': True}))
    response = session / 'response.bin'
    payload = None
    if not download_only:
        payload = session / 'upload-stream.bin'
        with payload.open('xb') as outgoing:
            authorization = json.loads((base / 'AUTHORIZATION.json').read_text())
            outgoing.write(json_bytes(authorization))
            with (ROOT / control['original_archive_path']).open('rb') as source:
                shutil.copyfileobj(source, outgoing, length=2**20)
    command = (ssh_override if ssh_override is not None else
        command_for(base, control, 'collect' if download_only else 'run', scratch))
    print(('Download only: no reference calculation starts.' if download_only else
        'One newly authorized CPU attempt. Keep this Terminal, SSH and Mac awake through collection.'), flush=True)
    print('Enter the university password promptly at the SSH prompt. Partial files: ' + str(session), flush=True)
    child = None
    try:
        with (payload.open('rb') if payload else open(os.devnull, 'rb')) as source, response.open('xb') as out:
            child = subprocess.Popen(command, stdin=source, stdout=out)
            end = time.monotonic() + (720 if download_only else 1320)
            while child.poll() is None:
                if response.stat().st_size > POLICY['max_collection_bytes'] + 4 * 2**20:
                    raise ValueError('Collection stream exceeded its bound.')
                if time.monotonic() > end:
                    raise TimeoutError('Attended connection deadline reached; preserve scratch and local files.')
                time.sleep(.1)
            if child.returncode:
                raise RuntimeError('SSH returned ' + str(child.returncode) + '; never repeat the launch.')
        receipt = verify_response(response, session / 'saved', control, control_sha)
        print('Evidence copied and byte-verified at ' + str(session) + '.', flush=True)
        if receipt['technical_error']:
            print('Attempt stopped: ' + receipt['technical_error'], flush=True)
        print('Reference exit: ' + str(receipt['reference_exit_status']) + '. Reply Done for independent review; do not relaunch.', flush=True)
        return receipt
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
            child.wait()
        if payload is not None and payload.exists():
            # Delete only this local transfer staging copy, never the packet or
            # captured evidence. The original packet remains byte-verified.
            payload.unlink()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--control-sha256', required=True)
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    try:
        run_client(args.base, args.control_sha256, download_only=args.download_only)
    except (ValueError, OSError, RuntimeError, TimeoutError) as exc:
        print(str(exc) + '\nPreserve all partial files. No automatic retry or next-host launch.', file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
