"""Attended, hash-bound control client. An unarmed packet opens no connection."""
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
import zlib

from models import cpu_oracle_control_v1 as remote
from models import cpu_oracle_control_prerequisite as prerequisite
old = remote.old
ROOT = Path(__file__).resolve().parents[1]


def verify_local(base, expected_sha, *, download_only=False):
    base = Path(base); path = base/'CONTROL.json'
    if path.is_symlink() or old.digest(path) != expected_sha: raise ValueError('Control v1 control changed.')
    control = json.loads(path.read_text()); remote.validate_control(control)
    if path.read_bytes() != old.json_bytes(control): raise ValueError('Noncanonical control bytes.')
    for mapping in ('local_source_sha256', 'prior_claims_sha256'):
        for name, expected in control[mapping].items():
            path = ROOT/old.safe_name(name)
            if path.is_symlink() or old.digest(path) != expected: raise ValueError('Pinned local source/history changed: '+name)
    if old.digest(base/'known-hosts') != control['known_hosts_sha256']: raise ValueError('Pinned public host key changed.')
    if shutil.disk_usage(base).free < 8*2**30: raise ValueError('Require 8 GiB Mac space before run/collection.')
    if not download_only:
        auth_path = base/'AUTHORIZATION.json'
        if not auth_path.is_file() or auth_path.is_symlink(): raise ValueError('UNARMED: the new control-host needs a separate explicit resource exception before SSH.')
        remote.validate_authorization(json.loads(auth_path.read_text()), expected_sha, control['host'])
        if (base/'ATTEMPT.json').exists(): raise FileExistsError('Control v1 claim is consumed; never repeat the launch.')
        packet = ROOT/old.safe_name(control['original_archive_path'])
        if packet.is_symlink() or packet.stat().st_size != control['archive_bytes'] or old.digest(packet) != old.ARCHIVE_SHA:
            raise ValueError('Original packet changed.')
    prerequisite.verify_target(ROOT)
    return control


def sources(control):
    result = {name: (remote.OVERLAY_INIT if name == 'models/__init__.py' else (ROOT/name).read_bytes()) for name in remote.REMOTE_NAMES}
    if any(hashlib.sha256(data).hexdigest() != control['remote_source_sha256'][name] for name, data in result.items()):
        raise ValueError('Bound control upload source differs.')
    return result


def command_for(base, control, *, download_only=False, scratch=None):
    encoded = {n: base64.b64encode(b).decode() for n, b in sources(control).items()}
    control_encoded = base64.b64encode(old.json_bytes(control)).decode()
    compressed = base64.b64encode(zlib.compress(old.json_bytes({'sources': encoded, 'control': control_encoded}))).decode()
    code = ('import base64,zlib,json,types,sys,hashlib;'
        'bundle=json.loads(zlib.decompress(base64.b64decode('+repr(compressed)+')));'
        'blobs=bundle["sources"];control64=bundle["control"];control=json.loads(base64.b64decode(control64));'
        'package=types.ModuleType("models");package.__path__=[];sys.modules["models"]=package;'
        '\nfor name in ("models/cpu_oracle_scratch.py","models/cpu_oracle_control_prerequisite.py","models/cpu_oracle_control_v1.py"):\n'
        ' data=base64.b64decode(blobs[name]);assert hashlib.sha256(data).hexdigest()==control["remote_source_sha256"][name]\n'
        ' module=types.ModuleType(name[:-3].replace("/","."));sys.modules[module.__name__]=module;'
        'setattr(package,module.__name__.split(".")[-1],module);exec(compile(data,name,"exec"),module.__dict__)\n'
        'sys.modules["models.cpu_oracle_control_v1"].remote_main(control64,blobs,'
        +repr('collect' if download_only else 'run')+','+repr(scratch)+')')
    remote_command = '"$HOME/.venvs/vjepa/bin/python" -B -u -c '+shlex.quote(code)
    return ['ssh', '-T', '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile='+str(Path(base)/'known-hosts'),
        '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'ClearAllForwardings=yes',
        '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
        'kdu@'+control['host']+'.dc.engr.scu.edu', remote_command]


def first_header(response, control, sha):
    with Path(response).open('rb') as f: line = f.readline(remote.POLICY['max_header_bytes']+1)
    if len(line) > remote.POLICY['max_header_bytes'] or not line.endswith(b'\n'): raise ValueError('Incomplete control scratch prelude.')
    value = json.loads(line); folder = Path(value['scratch_directory'])
    if (value.get('kind') != remote.KIND or value.get('phase') != 'scratch-created'
            or value.get('host') != control['host'] or value.get('control_sha256') != sha
            or folder.parent != Path('/var/tmp') or not folder.name.startswith(remote.PREFIX+control['host']+'-')):
        raise ValueError('Wrong new scratch host, path or plan identity.')
    return value


def verify_response(response, destination, control, sha):
    response, destination = Path(response), Path(destination); first = first_header(response, control, sha)
    if destination.exists(): raise FileExistsError('Preserve existing control extraction.')
    with response.open('rb') as source:
        source.readline(remote.POLICY['max_header_bytes']+1); line = source.readline(remote.POLICY['max_header_bytes']+1)
        if len(line) > remote.POLICY['max_header_bytes'] or not line.endswith(b'\n'): raise ValueError('Missing control collection manifest.')
        header = json.loads(line); files = header['files']
        if (header.get('kind') != remote.KIND or header.get('phase') != 'collection'
                or header.get('control_sha256') != sha or header.get('scientific_review_complete') is not False
                or header.get('model_restarted_for_collection') is not False or not 1 <= len(files) <= remote.POLICY['max_files']):
            raise ValueError('Wrong collection scope.')
        for name, item in files.items():
            old.safe_name(name)
            if (type(item['bytes']) is not int or item['bytes'] < 0 or type(item['mtime_ns']) is not int
                    or item['mtime_ns'] < 0 or not isinstance(item['sha256'], str) or len(item['sha256']) != 64):
                raise ValueError('Wrong collection member extent/hash.')
        total = sum(x['bytes'] for x in files.values())
        if total != header['logical_bytes'] or total > remote.POLICY['max_collection_bytes']: raise ValueError('Collection byte bound exceeded.')
        if shutil.disk_usage(destination.parent).free < total+2**30: raise ValueError('Mac extraction headroom insufficient.')
        destination.mkdir(mode=0o700); seen = set()
        with tarfile.open(mode='r|', fileobj=source) as archive:
            for member in archive:
                name = old.safe_name(member.name)
                if name in seen or name not in files or not member.isfile() or member.size != files[name]['bytes']:
                    raise ValueError('Unexpected, duplicate or linked collection member.')
                target = destination/name; target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as incoming, target.open('xb') as out:
                    shutil.copyfileobj(incoming, out, length=2**20); out.flush(); os.fsync(out.fileno())
                if old.digest(target) != files[name]['sha256']: raise ValueError('Collection member checksum changed.')
                os.utime(target, ns=(files[name]['mtime_ns'], files[name]['mtime_ns'])); seen.add(name)
        if seen != set(files): raise ValueError('Collection is incomplete.')
    if json.loads((destination/'operator-control.json').read_text()) != control: raise ValueError('Returned control changed.')
    state = json.loads((destination/'operator-result.json').read_text())
    if (state.get('kind') != remote.KIND or state.get('control_sha256') != sha
            or state.get('target_review_sha256') != prerequisite.REVIEW_SHA or state.get('host') != control['host']
            or state.get('run_phase_finished') is not True or state.get('CPU_only') is not True
            or any(state.get(k) is not False for k in ('GPU_work', 'paid_compute', 'scientific_review_complete', 'old_refusals_waived'))):
        raise ValueError('Wrong finished control attempt receipt.')
    allowed_outcomes = ('not_started', 'startup_failure', 'runtime_failure', 'invalid_runtime_receipts',
        'supervision_failure', 'numerical_refusal', 'saved_value_checks_passed')
    if (state.get('reference_outcome') not in allowed_outcomes or type(state.get('reference_run_passed')) is not bool
            or state['reference_run_passed'] != (state['reference_outcome'] == 'saved_value_checks_passed')):
        raise ValueError('Missing or contradictory control reference outcome.')
    if (type(state.get('reference_process_started')) is not bool
            or (state['reference_outcome'] == 'not_started') != (not state['reference_process_started'])
            or (state['reference_run_passed'] and state.get('reference_exit_status') != 0)):
        raise ValueError('Control v1 reference start/exit labels contradict its outcome.')
    for prefix, specs in (('code', control['archive_manifest']['files']),
                           ('overlay', {n: {'sha256': s} for n, s in control['remote_source_sha256'].items()})):
        for name, spec in specs.items():
            path = destination/prefix/name
            if state['reference_process_started'] is True and not path.is_file():
                raise ValueError('Missing source from a started reference attempt.')
            if path.exists() and old.digest(path) != spec['sha256']: raise ValueError('Returned bound source changed.')
    receipt = {'kind': 'cpu-oracle-control-transport-v1', 'transport_verified': True,
        'control_sha256': sha, 'host': control['host'], 'scratch_directory': first['scratch_directory'],
        'target_review_sha256': prerequisite.REVIEW_SHA, 'response_sha256': old.digest(response), 'members_verified': len(seen), 'logical_bytes_verified': total,
        'reference_process_started': state['reference_process_started'], 'reference_exit_status': state.get('reference_exit_status'),
        'reference_outcome': state['reference_outcome'], 'reference_run_passed': state['reference_run_passed'],
        'technical_error': state.get('error'), 'scientific_review_complete': False, 'old_refusals_waived': False,
        'remote_scratch_deleted': False, 'model_restarted_for_collection': False}
    old.write_json(destination.parent/'transport-verification.json', receipt)
    return receipt


def run_client(base, sha, *, download_only=False, ssh_override=None):
    base = Path(base); control = verify_local(base, sha, download_only=download_only); scratch = None
    if download_only:
        attempt = json.loads((base/'ATTEMPT.json').read_text())
        if (attempt.get('kind') != remote.KIND or attempt.get('control_sha256') != sha
                or not isinstance(attempt.get('local_session'), str)
                or Path(attempt['local_session']).name != attempt['local_session']
                or not attempt['local_session'].startswith('attended-')):
            raise ValueError('Wrong immutable control attempt for collection.')
        scratch = first_header(base/old.safe_name(attempt['local_session'])/'response.bin', control, sha)['scratch_directory']
    session = Path(tempfile.mkdtemp(prefix='download-only-' if download_only else 'attended-', dir=base))
    if not download_only:
        old.write_json(base/'ATTEMPT.json', {'kind': remote.KIND, 'control_sha256': sha,
            'local_session': session.name, 'invoked_unix': time.time(), 'single_attempt': True})
    payload = None; child = None; response = session/'response.bin'
    try:
        if not download_only:
            payload = session/'upload-stream.bin'
            with payload.open('xb') as out:
                out.write(old.json_bytes(json.loads((base/'AUTHORIZATION.json').read_text())))
                with (ROOT/control['original_archive_path']).open('rb') as source: shutil.copyfileobj(source, out, length=2**20)
        command = ssh_override if ssh_override is not None else command_for(base, control, download_only=download_only, scratch=scratch)
        print('Download only; no worker starts.' if download_only else 'One control-host CPU attempt. Keep this Terminal and Mac awake through collection.', flush=True)
        print('Enter the university password promptly. Partial files: '+str(session), flush=True)
        with (payload.open('rb') if payload else open(os.devnull, 'rb')) as source, response.open('xb') as out:
            child = subprocess.Popen(command, stdin=source, stdout=out); end = time.monotonic()+(720 if download_only else 1320)
            while child.poll() is None:
                if response.stat().st_size > remote.POLICY['max_collection_bytes']+4*2**20: raise ValueError('Response exceeds bound.')
                if time.monotonic() >= end: raise TimeoutError('Connection deadline reached; preserve files, no automatic retry.')
                time.sleep(.1)
            if child.returncode: raise RuntimeError('SSH returned '+str(child.returncode)+'; never repeat this launch.')
        receipt = verify_response(response, session/'saved', control, sha)
        print('Saved and byte-verified: '+str(session)+'. Reply Done for independent review.', flush=True)
        if receipt['technical_error']: print('Attempt stopped: '+receipt['technical_error'], flush=True)
        print('Reference outcome: '+receipt['reference_outcome'], flush=True)
        print('Reference exit: '+str(receipt['reference_exit_status'])+'. Never repeat this launch.', flush=True)
        return receipt
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill()
            child.wait()
        if payload is not None and payload.exists(): payload.unlink()  # Only this local transfer copy.


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--base', type=Path, required=True)
    p.add_argument('--control-sha256', required=True); p.add_argument('--download-only', action='store_true')
    a = p.parse_args()
    try: run_client(a.base, a.control_sha256, download_only=a.download_only)
    except (ValueError, OSError, RuntimeError, TimeoutError) as exc:
        print(str(exc)+'\nPreserve partial files. No automatic retry or next host.', file=sys.stderr, flush=True); raise SystemExit(1)
