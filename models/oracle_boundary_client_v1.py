"""One attended boundary capture; an unarmed handoff cannot open SSH."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time

from models import oracle_boundary_operator_v1 as remote
old=remote.old
ROOT=Path(__file__).resolve().parents[1]


def verify_prerequisites(control):
    for name,sha in control['prior_bindings_sha256'].items():
        path=ROOT/old.safe_name(name)
        if path.is_symlink() or not path.is_file() or old.digest(path)!=sha:
            raise ValueError('Preserved evidence/source changed: '+name)


def verify_local(base,sha,*,download_only=False):
    base=Path(base);path=base/'CONTROL.json'
    if path.is_symlink() or old.digest(path)!=sha:raise ValueError('New boundary control changed.')
    c=json.loads(path.read_text());remote.validate_control(c)
    if path.read_bytes()!=old.json_bytes(c):raise ValueError('Control bytes are not canonical.')
    for name,want in c['local_source_sha256'].items():
        path=ROOT/old.safe_name(name)
        if path.is_symlink() or old.digest(path)!=want:raise ValueError('Boundary operator changed: '+name)
    key_path=base/'known-hosts'
    if key_path.is_symlink() or old.digest(key_path)!=c['known_hosts_sha256']:raise ValueError('Pinned public host key changed.')
    rows=[line.split() for line in key_path.read_text().splitlines() if line.strip()]
    if len(rows)!=1 or len(rows[0])!=3 or rows[0][:2]!=[c['host']+'.dc.engr.scu.edu','ssh-ed25519']:
        raise ValueError('Pinned key must name this exact target host.')
    fingerprint='SHA256:'+base64.b64encode(hashlib.sha256(base64.b64decode(rows[0][2],validate=True)).digest()).decode().rstrip('=')
    if fingerprint!=c['expected_ssh_fingerprint']:raise ValueError('Public fingerprint differs from the independently supplied target key.')
    if shutil.disk_usage(base).free<8*2**30:raise ValueError('Require 8 GiB local space before connection.')
    if not download_only:
        auth=base/'AUTHORIZATION.json'
        if not auth.is_file() or auth.is_symlink():
            raise ValueError('UNARMED: a new CUDA-setup boundary resource exception is required before any claim or SSH.')
        remote.validate_authorization(json.loads(auth.read_text()),sha)
        if (base/'ATTEMPT.json').exists():raise FileExistsError('This boundary attempt is consumed; never repeat launch.')
        archive=ROOT/old.safe_name(c['original_archive_path'])
        if archive.is_symlink() or archive.stat().st_size!=c['archive_bytes'] or old.digest(archive)!=remote.ARCHIVE_SHA:
            raise ValueError('The unchanged full original packet is required.')
    verify_prerequisites(c)
    return c


def sources(c):
    data={n:(remote.OVERLAY_INIT if n=='models/__init__.py' else (ROOT/n).read_bytes()) for n in remote.REMOTE_NAMES}
    if any(hashlib.sha256(b).hexdigest()!=c['remote_source_sha256'][n] for n,b in data.items()):
        raise ValueError('Upload source hash changed.')
    return data


def bootstrap_payload(c):
    return old.json_bytes({'control':c,'sources':{n:base64.b64encode(b).decode() for n,b in sources(c).items()}})


def command_for(base,c,*,download_only=False,scratch=None):
    # Source/plan travel in a bounded stdin prelude, not one oversized -c arg.
    code=('import base64,json,types,sys,hashlib;'
        'line=sys.stdin.buffer.readline(1048577);assert len(line)<=1048576 and line.endswith(b"\\n");'
        'bundle=json.loads(line);blobs=bundle["sources"];control=bundle["control"];'
        'control64=base64.b64encode((json.dumps(control,sort_keys=True,separators=(",",":"),allow_nan=False)+"\\n").encode()).decode();'
        'package=types.ModuleType("models");package.__path__=[];sys.modules["models"]=package;'
        '\nfor name in ("models/cpu_oracle_scratch.py","models/cpu_oracle_control_prerequisite.py","models/cpu_oracle_control_v1.py","models/oracle_boundary_operator_v1.py"):\n'
        ' data=base64.b64decode(blobs[name]);assert hashlib.sha256(data).hexdigest()==control["remote_source_sha256"][name]\n'
        ' module=types.ModuleType(name[:-3].replace("/","."));sys.modules[module.__name__]=module;'
        'setattr(package,module.__name__.split(".")[-1],module);exec(compile(data,name,"exec"),module.__dict__)\n'
        'sys.modules["models.oracle_boundary_operator_v1"].remote_main(control64,blobs,'
        +repr('collect' if download_only else 'run')+','+repr(scratch)+')')
    command='"$HOME/.venvs/vjepa/bin/python" -B -u -c '+shlex.quote(code)
    return ['ssh','-T','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(Path(base)/'known-hosts'),
        '-o','ControlMaster=no','-o','ControlPath=none','-o','ClearAllForwardings=yes','-o','ConnectTimeout=15',
        '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','kdu@linux10603.dc.engr.scu.edu',command]


def first_header(response,c,sha):
    with Path(response).open('rb') as f:line=f.readline(remote.POLICY['max_header_bytes']+1)
    if len(line)>remote.POLICY['max_header_bytes'] or not line.endswith(b'\n'):raise ValueError('Missing bounded scratch prelude.')
    value=json.loads(line);folder=Path(value['scratch_directory'])
    if (value.get('kind')!=remote.KIND or value.get('phase')!='scratch-created' or value.get('host')!=c['host']
            or value.get('control_sha256')!=sha or folder.parent!=Path('/var/tmp')
            or not folder.name.startswith(remote.PREFIX+c['host']+'-')):
        raise ValueError('Wrong boundary scratch, host or plan identity.')
    return value


def verify_response(response,destination,c,sha):
    """Transport verification only; numerical review is a separate local step."""
    response,destination=Path(response),Path(destination);first=first_header(response,c,sha)
    if destination.exists():raise FileExistsError('Preserve prior extraction.')
    with response.open('rb') as incoming:
        incoming.readline(remote.POLICY['max_header_bytes']+1);line=incoming.readline(remote.POLICY['max_header_bytes']+1)
        if len(line)>remote.POLICY['max_header_bytes'] or not line.endswith(b'\n'):raise ValueError('Missing bounded collection header.')
        h=json.loads(line);files=h['files']
        if (h.get('kind')!=remote.KIND or h.get('phase')!='collection' or h.get('control_sha256')!=sha
                or h.get('scientific_review_complete') is not False or h.get('model_restarted_for_collection') is not False
                or not 1<=len(files)<=remote.POLICY['max_files']):raise ValueError('Wrong collection scope.')
        for name,item in files.items():
            old.safe_name(name)
            if (type(item['bytes']) is not int or item['bytes']<0 or type(item['mtime_ns']) is not int or item['mtime_ns']<0
                    or not isinstance(item['sha256'],str) or len(item['sha256'])!=64):raise ValueError('Invalid collection member extent.')
        total=sum(i['bytes'] for i in files.values())
        if total!=h['logical_bytes'] or total>remote.POLICY['max_collection_bytes']:raise ValueError('Collection exceeds byte limit.')
        if shutil.disk_usage(destination.parent).free<total+2**30:raise ValueError('Insufficient extraction headroom.')
        destination.mkdir(mode=0o700);seen=set()
        with tarfile.open(mode='r|',fileobj=incoming) as archive:
            for m in archive:
                name=old.safe_name(m.name)
                if name in seen or name not in files or not m.isfile() or m.size!=files[name]['bytes']:
                    raise ValueError('Unexpected, linked or duplicate collection member.')
                path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(m) as src,path.open('xb') as out:
                    shutil.copyfileobj(src,out,length=2**20);out.flush();os.fsync(out.fileno())
                if old.digest(path)!=files[name]['sha256']:raise ValueError('Saved file checksum differs.')
                os.utime(path,ns=(files[name]['mtime_ns'],files[name]['mtime_ns']));seen.add(name)
        if seen!=set(files):raise ValueError('Collection is incomplete.')
    if json.loads((destination/'operator-control.json').read_text())!=c:raise ValueError('Returned control differs.')
    state=json.loads((destination/'operator-result.json').read_text())
    if (state.get('kind')!=remote.KIND or state.get('control_sha256')!=sha or state.get('host')!=c['host']
            or state.get('work_finished') is not True or state.get('CUDA_setup_allowed') is not True
            or any(state.get(k) is not False for k in ('GPU_SDPA_allowed','paid_compute','original_full_worker_accepted',
                'old_refusals_waived','scientific_review_complete'))):raise ValueError('Wrong finished boundary receipt.')
    complete=state.get('scope_capture_complete');outcome=state.get('capture_outcome');passed=state.get('original_gate_passed')
    if type(complete) is not bool or outcome not in ('captured-gate-pass','captured-original-refusal','technical-failure','startup-failure'):
        raise ValueError('Missing capture outcome.')
    if complete:
        if type(passed) is not bool or passed!=(outcome=='captured-gate-pass'):raise ValueError('Original gate contradicts outcome.')
        checked=remote.capture_status(destination,state.get('worker_exit_status'))
        if checked['capture_outcome']!=outcome:raise ValueError('Inner capture contradicts outer receipt.')
    elif outcome not in ('technical-failure','startup-failure'):raise ValueError('Incomplete scope cannot pass.')
    for prefix,specs in [('code',c['archive_manifest']['files']),('overlay',{n:{'sha256':s} for n,s in c['remote_source_sha256'].items()})]:
        for name,item in specs.items():
            path=destination/prefix/name
            if state.get('worker_process_started') is True and not path.is_file():raise ValueError('Started worker lacks its bound source.')
            if path.exists() and old.digest(path)!=item['sha256']:raise ValueError('Returned source changed.')
    result={'kind':'oracle-boundary-transport-v1','transport_verified':True,'control_sha256':sha,'host':c['host'],
        'scratch_directory':first['scratch_directory'],'response_sha256':old.digest(response),
        'members_verified':len(seen),'logical_bytes_verified':total,'capture_outcome':outcome,
        'scope_capture_complete':complete,'original_gate_passed':passed,'technical_error':state.get('error'),
        'worker_exit_status':state.get('worker_exit_status'),'original_full_worker_accepted':False,
        'scientific_review_complete':False,'remote_scratch_deleted':False,'model_restarted_for_collection':False}
    old.write_json(destination.parent/'transport-verification.json',result);return result


def run_client(base,sha,*,download_only=False,ssh_override=None):
    base=Path(base);c=verify_local(base,sha,download_only=download_only);scratch=None
    if download_only:
        claim=json.loads((base/'ATTEMPT.json').read_text());session_name=claim.get('local_session','')
        if (claim.get('kind')!=remote.KIND or claim.get('control_sha256')!=sha
                or not isinstance(session_name,str) or Path(session_name).name!=session_name or not session_name.startswith('attended-')):
            raise ValueError('Download requires the same consumed boundary claim.')
        scratch=first_header(base/session_name/'response.bin',c,sha)['scratch_directory']
    session=Path(tempfile.mkdtemp(prefix='download-only-' if download_only else 'attended-',dir=base))
    if not download_only:old.write_json(base/'ATTEMPT.json',{'kind':remote.KIND,'control_sha256':sha,
        'local_session':session.name,'invoked_unix':time.time(),'single_attempt':True})
    payload=None;child=None;response=session/'response.bin'
    try:
        payload=session/'upload-stream.bin'
        with payload.open('xb') as out:
            prelude=bootstrap_payload(c)
            if len(prelude)>remote.POLICY['max_header_bytes']:raise ValueError('Bootstrap prelude exceeds bound.')
            out.write(prelude)
            if not download_only:
                out.write(old.json_bytes(json.loads((base/'AUTHORIZATION.json').read_text())))
                with (ROOT/c['original_archive_path']).open('rb') as src:shutil.copyfileobj(src,out,length=2**20)
        command=ssh_override if ssh_override is not None else command_for(base,c,download_only=download_only,scratch=scratch)
        print('Download only; no worker starts.' if download_only else 'One original CPU-oracle capture after CUDA setup. Keep Terminal and Mac awake.',flush=True)
        print('Enter the university password promptly. Partial files: '+str(session),flush=True)
        with (payload.open('rb') if payload else open(os.devnull,'rb')) as src,response.open('xb') as out,(session/'ssh.stderr').open('xb') as err:
            child=subprocess.Popen(command,stdin=src,stdout=out,stderr=err);end=time.monotonic()+(720 if download_only else 1320)
            while child.poll() is None:
                if response.stat().st_size>remote.POLICY['max_collection_bytes']+4*2**20 or (session/'ssh.stderr').stat().st_size>16*2**20:
                    raise ValueError('Transport output exceeds bound.')
                if time.monotonic()>=end:raise TimeoutError('Connection deadline expired; no automatic retry.')
                time.sleep(.1)
            if child.returncode:raise RuntimeError('SSH returned '+str(child.returncode)+'; preserve stderr, never repeat launch.')
        receipt=verify_response(response,session/'saved',c,sha)
        print('Saved and byte-verified: '+str(session)+'. Reply Done for independent review.',flush=True)
        print('Capture outcome: '+receipt['capture_outcome']+'. Original GPU worker remains incomplete.',flush=True)
        return receipt
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill()
            child.wait()
        if payload is not None and payload.exists():payload.unlink()  # This local transfer copy only.


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--control-sha256',required=True);p.add_argument('--download-only',action='store_true');a=p.parse_args()
    try:run_client(a.base,a.control_sha256,download_only=a.download_only)
    except (ValueError,OSError,RuntimeError,TimeoutError) as exc:
        print(str(exc)+'\nPreserve partial files. No automatic retry or next worker.',file=__import__('sys').stderr,flush=True);raise SystemExit(1)
