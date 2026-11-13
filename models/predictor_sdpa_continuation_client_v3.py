"""Unarmed/copy/reuse refusal, one SSH connection, bounded byte-exact collection."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import threading
import time
from models import predictor_sdpa_continuation_operator_v3 as op
old=op.old


def pending_unarmed(root):
    path=Path(root)/'runs/pretrained/diagnostic/predictor-eot8-v1-20261105/linux10609-final'
    if (path/'AUTHORIZATION.json').exists() or (path/'ATTEMPT.json').exists():raise ValueError('Old alternative host is armed/claimed; no parallel/fallback plan.')


def check_local(folder):
    folder=Path(folder);c=json.loads((folder/'CONTROL.json').read_text());op.validate_control(c)
    if folder.is_symlink() or str(folder.resolve())!=c['local_handoff_directory']:raise ValueError('Exact local handoff directory required; copies cannot reset claims.')
    root=Path(c['repository']);pending_unarmed(root)
    for name,want in c['local_source_sha256'].items():
        if old.digest(root/name)!=want:raise ValueError('Local source changed: '+name)
    for key in ('parent_consumed_claim','preceding_component_claim','origin_native_claim','local_association_review','local_startup_review','local_launch_evidence_review','local_worker_review','preserved_inputs','capsule_manifest','capsule'):
        item=c['contract'][key]
        if old.digest(root/item['path'])!=item['sha256']:raise ValueError('Frozen prior binding changed: '+item['path'])
    pins=json.loads((root/c['contract']['preserved_inputs']['path']).read_text())
    if len(pins)!=c['contract']['preserved_inputs']['files']:raise ValueError('Prior evidence coverage changed.')
    for name,want in pins.items():
        if old.digest(root/name)!=want:raise ValueError('Frozen prior source/evidence changed: '+name)
    public=folder/'known-hosts'
    if public.is_symlink() or old.digest(public)!=c['known_hosts_sha256']:raise ValueError('Pinned public host key changed.')
    fields=public.read_text().split()
    if len(fields)!=3 or fields[:2]!=['linux10603.dc.engr.scu.edu','ssh-ed25519']:raise ValueError('Exact public host key entry required.')
    fp='SHA256:'+base64.b64encode(hashlib.sha256(base64.b64decode(fields[2],validate=True)).digest()).decode().rstrip('=')
    if fp!=c['expected_ssh_fingerprint']:raise ValueError('Public fingerprint changed.')
    if old.digest(root/c['archive_path'])!=op.CAPSULE_SHA:raise ValueError('Original packet changed.')
    return c,hashlib.sha256(old.json_bytes(c)).hexdigest()


def payload(c):
    root=Path(c['repository']);sources={n:op.INIT if n=='models/__init__.py' else (root/n).read_bytes() for n in op.REMOTE_NAMES}
    if sum(map(len,sources.values()))>2*op.MiB or any(hashlib.sha256(data).hexdigest()!=c['remote_source_sha256'][n] for n,data in sources.items()):raise ValueError('Exact bounded overlay required.')
    body="import base64,json,sys,types\nfrom pathlib import Path\npackage=types.ModuleType('models');package.__path__=[];sys.modules['models']=package\n"
    order=(*op.FROZEN_BOOT,'models/predictor_sdpa_continuation_contract_v3.py',
        'models/predictor_sdpa_continuation_storage_v3.py','models/predictor_sdpa_continuation_operator_v3.py')
    for n in order:
        name=n[:-3].replace('/','.');encoded=base64.b64encode(sources[n]).decode()
        body+=f"m=types.ModuleType({name!r});m.__file__={n!r};sys.modules[{name!r}]=m;setattr(package,{name.split('.')[-1]!r},m);exec(compile(base64.b64decode({encoded!r}),{n!r},'exec'),m.__dict__)\n"
    control=base64.b64encode(old.json_bytes(c)).decode()
    data=base64.b64encode(old.json_bytes({n:base64.b64encode(v).decode() for n,v in sources.items()})).decode()
    body+=f"sys.exit(m.main({control!r},{data!r}))\n"
    if len(body.encode())>2*op.MiB:raise ValueError('Bootstrap exceeds reviewed 2 MiB ceiling.')
    return body.encode()


def unpack(response,destination,c,sha,*,collection_only=False):
    destination=Path(destination);destination.mkdir(mode=0o700,exist_ok=False)
    with Path(response).open('rb') as f:
        first=json.loads(f.readline(op.MiB+1))
        if first['kind']!=op.KIND or first['control_sha256']!=sha:raise ValueError('Response identity changed.')
        if collection_only:header=first
        else:
            if first['phase']!='scratch-created':raise ValueError('Missing scratch receipt; preserve partial response.')
            line=f.readline(op.MiB+1)
            if not line.strip():raise ValueError('Collection header missing; preserve saved SSH stderr and response, never repeat the launch.')
            header=json.loads(line)
        if (header['kind']!=op.KIND or header['phase']!='collection' or header['control_sha256']!=sha or header['model_restarted'] is not False):
            raise ValueError('Complete same-session collection required.')
        files=header['files']
        if len(files)>1024 or sum(x['bytes'] for x in files.values())!=header['logical_bytes'] or Path(response).stat().st_size>3*op.GiB:raise ValueError('Collection byte/file ceiling exceeded.')
        extent=sum(512+((i['bytes']+511)//512)*512 for i in files.values())+1024
        if Path(response).stat().st_size-f.tell()!=((extent+10239)//10240)*10240:raise ValueError('Truncated/extra evidence stream.')
        seen=set()
        with tarfile.open(fileobj=f,mode='r|') as tar:
            for member in tar:
                name=old.safe_name(member.name)
                if name not in files or name in seen or not member.isfile() or member.size!=files[name]['bytes']:raise ValueError('Unexpected evidence archive member.')
                item=files[name]
                if (item['mode'] not in (0o400,0o600) or member.mode!=item['mode'] or member.mtime!=item['mtime_ns']//10**9
                        or item['bytes']>op.policy.typed_file_limit(name)):
                    raise ValueError('Private typed evidence metadata required.')
                p=destination/name;p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
                with tar.extractfile(member) as source,p.open('xb') as out:
                    for block in iter(lambda:source.read(op.MiB),b''):out.write(block)
                    out.flush();os.fsync(out.fileno())
                if old.digest(p)!=item['sha256']:raise ValueError('Collected evidence hash changed.')
                p.chmod(item['mode']);os.utime(p,ns=(item['mtime_ns'],item['mtime_ns']));seen.add(name)
        if seen!=set(files):raise ValueError('Missing collected evidence; never restart model.')
    return first


def run(folder,*,download_only=False,popen=subprocess.Popen):
    if download_only:raise ValueError('Separate retrieval/clock reset is outside this single-connection scope.')
    folder=Path(folder);a=folder/'AUTHORIZATION.json'
    if not a.is_file():raise ValueError('Unarmed: no authorization, claim, SSH or model work.')
    if a.is_symlink() or a.stat().st_mode&0o777!=0o600:raise ValueError('Separate private regular authorization file required.')
    c,sha=check_local(folder);auth=json.loads(a.read_text());op.validate_authorization(auth,sha)
    if (folder/'ATTEMPT.json').exists():raise FileExistsError('Consumed attempt; never rerun/reset this helper.')
    bootstrap=payload(c)
    old.write_json(folder/'ATTEMPT.json',{'kind':op.KIND,'control_sha256':sha,'scratch_directory':None,'started_unix':time.time(),'consumed':True})
    session=Path(tempfile.mkdtemp(prefix='attended-',dir=folder));session.chmod(0o700)
    response=session/'response.bin'
    remote='exec "$HOME/.venvs/vjepa/bin/python" -I -B -u -c '+shlex.quote('import sys;exec(compile(sys.stdin.buffer.read('+str(len(bootstrap))+'),'+repr('<reviewed-component-only-bootstrap>')+','+repr('exec')+'))')
    command=['ssh','-T','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(folder/'known-hosts'),
        '-o','GlobalKnownHostsFile=/dev/null','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','kdu@linux10603.dc.engr.scu.edu',remote]
    end=time.monotonic()+1500;errors=[]
    print('Keep this Terminal and Mac lid open through work and collection (up to 25 minutes). Enter the university password promptly.',flush=True)
    with response.open('xb') as out,(session/'ssh.stderr.log').open('xb') as stderr:
        response.chmod(0o600);os.chmod(stderr.name,0o600)
        proc=popen(command,stdin=subprocess.PIPE,stdout=out,stderr=stderr)
        def send():
            try:
                proc.stdin.write(bootstrap);proc.stdin.write(old.json_bytes(auth));proc.stdin.flush()
                with (Path(c['repository'])/c['archive_path']).open('rb') as incoming:
                    for block in iter(lambda:incoming.read(op.MiB),b''):proc.stdin.write(block)
                proc.stdin.close()
            except BaseException as e:errors.append(type(e).__name__)
        thread=threading.Thread(target=send,daemon=True);thread.start()
        try:
            while proc.poll() is None:
                if time.monotonic()>end or os.fstat(out.fileno()).st_size>3*op.GiB:raise TimeoutError('Bounded same-SSH transport ended; preserve partials, no retry.')
                time.sleep(.1)
            thread.join(timeout=1)
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
    old.write_json(session/'transport.json',{'exit_status':proc.returncode,'send_errors':errors,'same_connection':True,'model_restarted':False})
    with response.open('rb') as f:line=f.readline(op.MiB+1)
    if not line.strip():raise ValueError('Missing bound scratch receipt; preserve saved SSH stderr and consumed claim.')
    first=json.loads(line)
    if first['kind']!=op.KIND or first['phase']!='scratch-created' or first['control_sha256']!=sha:raise ValueError('Missing bound scratch receipt; keep partial response.')
    claim=json.loads((folder/'ATTEMPT.json').read_text());claim['scratch_directory']=first['scratch_directory'];old.write_json(folder/'ATTEMPT.json',claim,replace=True)
    unpack(response,session/'saved',c,sha)
    print('Evidence saved at',session,'for independent review. Never repeat this launch.',flush=True)
    return proc.returncode
