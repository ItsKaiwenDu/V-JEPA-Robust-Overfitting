"""Unarmed refuses before claim/SSH; bounded same-session collection."""
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

from models import predictor_boundary_operator_v1 as op
old=op.old


def ensure_previous_unarmed(root):
    pending=Path(root)/'runs/pretrained/diagnostic/predictor-eot8-v1-20261105/linux10609-final'
    if (pending/'AUTHORIZATION.json').exists() or (pending/'ATTEMPT.json').exists():
        raise ValueError('Pending 10609 handoff is armed/claimed; stop for review, never run both hosts.')


def check_local(folder):
    folder=Path(folder);c=json.loads((folder/'CONTROL.json').read_text());op.validate_control(c)
    root=Path(c['repository']);ensure_previous_unarmed(root)
    if (folder/'known-hosts').is_symlink() or old.digest(folder/'known-hosts')!=c['known_hosts_sha256']:raise ValueError('Pinned public host key changed.')
    fields=(folder/'known-hosts').read_text().split()
    if len(fields)!=3 or fields[:2]!=['linux10603.dc.engr.scu.edu','ssh-ed25519']:raise ValueError('Exact public host entry required.')
    fp='SHA256:'+base64.b64encode(hashlib.sha256(base64.b64decode(fields[2],validate=True)).digest()).decode().rstrip('=')
    if fp!=c['expected_ssh_fingerprint']:raise ValueError('Unverified SSH host fingerprint.')
    if folder.is_symlink() or str(folder.resolve())!=c.get('local_handoff_directory'):
        raise ValueError('Exact local handoff directory required; copied helpers cannot reset the claim.')
    for name,sha in c['local_source_sha256'].items():
        if old.digest(root/name)!=sha:raise ValueError('Local operator changed: '+name)
    pins_path=root/c['prior_inputs_manifest']
    if old.digest(pins_path)!=c['prior_inputs_manifest_sha256']:raise ValueError('Prior pin manifest changed.')
    for name,sha in json.loads(pins_path.read_text()).items():
        if old.digest(root/name)!=sha:raise ValueError('Frozen prior evidence changed: '+name)
    for name,want in c['contract']['prior_consumed_files'].items():
        if old.digest(root/name)!=want:raise ValueError('Consumed predecessor source/claim changed.')
    if old.digest(root/c['archive_path'])!=op.PACKET_SHA:raise ValueError('Original input archive changed.')
    if old.digest(root/c['anchors_path'])!=c['anchors_manifest']['sha256']:raise ValueError('Frozen value-anchor capsule changed.')
    return c,hashlib.sha256(old.json_bytes(c)).hexdigest()


def payload(c,*,collect=None):
    if collect is not None:raise ValueError('Separate retrieval requires independent scope review; no model restart.')
    root=Path(c['repository']);sources={n:op.INIT if n=='models/__init__.py' else (root/n).read_bytes() for n in op.REMOTE_NAMES}
    if any(hashlib.sha256(data).hexdigest()!=c['remote_source_sha256'][n] for n,data in sources.items()) or sum(map(len,sources.values()))>2*op.MiB:
        raise ValueError('Exact overlay source or byte bound changed.')
    body='import base64,json,sys,types\nfrom pathlib import Path\n'
    # Bootstrap modules entirely in memory; no target write occurs before authorization/readiness.
    body+="package=types.ModuleType('models');package.__path__=[];sys.modules['models']=package\n"
    order=[*op.FROZEN_NAMES,'models/predictor_eot8_owned_receipt_v1.py','models/predictor_eot8_record_v2.py','models/predictor_boundary_contract_v1.py','models/predictor_boundary_record_v1.py','models/predictor_boundary_operator_v1.py']
    for n in order:
        name=n[:-3].replace('/','.')
        # Imports in primitives require prerequisite/oracle_trace before their importers.
        data=base64.b64encode(sources[n]).decode()
        body+=f"m=types.ModuleType({name!r});m.__file__={n!r};sys.modules[{name!r}]=m;setattr(package,{name.split('.')[-1]!r},m);exec(compile(base64.b64decode({data!r}),{n!r},'exec'),m.__dict__)\n"
    control=base64.b64encode(old.json_bytes(c)).decode();src=base64.b64encode(old.json_bytes({n:base64.b64encode(data).decode() for n,data in sources.items()})).decode()
    body+=f"sys.exit(m.main({control!r},{src!r},mode={'collect' if collect else 'run'!r},scratch={collect!r}))\n"
    if len(body.encode())>2*op.MiB:raise ValueError('Bootstrap exceeds reviewed 2 MiB bound.')
    return body.encode()


def unpack(response,destination,c,sha,*,collect=False):
    destination=Path(destination);destination.mkdir(mode=0o700,exist_ok=False)
    with Path(response).open('rb') as f:
        first=json.loads(f.readline(op.MiB+1))
        if first['kind']!=op.KIND or first['control_sha256']!=sha:raise ValueError('Response binding changed.')
        if not collect:
            if first['phase']!='scratch-created':raise ValueError('Missing scratch receipt.')
            header=json.loads(f.readline(op.MiB+1))
        else:header=first
        if header['kind']!=op.KIND or header['phase']!='collection' or header['control_sha256']!=sha or header['model_restarted'] is not False:
            raise ValueError('Incomplete collection header.')
        files=header['files']
        if len(files)>1024 or sum(x['bytes'] for x in files.values())!=header['logical_bytes'] or header['logical_bytes']>3*op.GiB:raise ValueError('Collection extent exceeded.')
        extent=sum(512+((item['bytes']+511)//512)*512 for item in files.values())+1024
        expected=((extent+10239)//10240)*10240
        if Path(response).stat().st_size-f.tell()!=expected:raise ValueError('Truncated/extra collection stream.')
        seen=set()
        with tarfile.open(fileobj=f,mode='r|') as tar:
            for member in tar:
                name=old.safe_name(member.name)
                if name not in files or name in seen or not member.isfile() or member.size!=files[name]['bytes']:raise ValueError('Untrusted archive member.')
                item=files[name];p=destination/name;p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
                with tar.extractfile(member) as source,p.open('xb') as out:
                    for block in iter(lambda:source.read(op.MiB),b''):out.write(block)
                    out.flush();os.fsync(out.fileno())
                if old.digest(p)!=item['sha256'] or item['mode'] not in (0o400,0o600):raise ValueError('Collected member hash/mode changed.')
                p.chmod(item['mode']);os.utime(p,ns=(item['mtime_ns'],item['mtime_ns']));seen.add(name)
        if seen!=set(files):raise ValueError('Incomplete evidence collection; never restart model.')
    return first


def run(folder,*,download_only=False,popen=subprocess.Popen):
    if download_only:raise ValueError('Separate retrieval requires independent review; no old collection clock reset.')
    folder=Path(folder);auth_path=folder/'AUTHORIZATION.json'
    # This check deliberately precedes even provenance scans or creating a session.
    if not auth_path.is_file():raise ValueError('Unarmed: no authorization; no claim, session or SSH created.')
    c,sha=check_local(folder);auth=json.loads(auth_path.read_text());op.validate_authorization(auth,sha)
    scratch=None
    if download_only:
        claim=json.loads((folder/'ATTEMPT.json').read_text());scratch=claim['scratch_directory']
        if not scratch:raise ValueError('No authenticated scratch receipt; inspect partial response first.')
    elif (folder/'ATTEMPT.json').exists():raise FileExistsError('Attempt consumed; never repeat launch.')
    bootstrap=payload(c,collect=scratch)
    if not download_only:old.write_json(folder/'ATTEMPT.json',{'kind':op.KIND,'control_sha256':sha,'scratch_directory':None,'started_unix':time.time(),'consumed':True})
    session=Path(tempfile.mkdtemp(prefix='download-' if download_only else 'attended-',dir=folder));session.chmod(0o700)
    response=session/'response.bin';remote="exec \"$HOME/.venvs/vjepa/bin/python\" -I -B -u -c "+shlex.quote('import sys;exec(compile(sys.stdin.buffer.read('+str(len(bootstrap))+'),'+repr('<reviewed-bootstrap>')+','+repr('exec')+'))')
    command=['ssh','-T','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(folder/'known-hosts'),'-o','GlobalKnownHostsFile=/dev/null','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','kdu@linux10603.dc.engr.scu.edu',remote]
    end=time.monotonic()+(600 if download_only else 1800);errors=[]
    with response.open('xb') as out:
        proc=popen(command,stdin=subprocess.PIPE,stdout=out)
        def send():
            try:
                proc.stdin.write(bootstrap);proc.stdin.write(old.json_bytes(auth));proc.stdin.flush()
                if not download_only:
                    for field in ('archive_path','anchors_path'):
                        with (Path(c['repository'])/c[field]).open('rb') as incoming:
                            for chunk in iter(lambda:incoming.read(op.MiB),b''):proc.stdin.write(chunk)
                proc.stdin.close()
            except BaseException as exc:errors.append(type(exc).__name__)
        thread=threading.Thread(target=send,daemon=True);thread.start()
        try:
            while proc.poll() is None:
                if time.monotonic()>end or os.fstat(out.fileno()).st_size>3*op.GiB:
                    raise TimeoutError('Bounded transport ended; preserve partial evidence.')
                time.sleep(.1)
            thread.join(timeout=1)
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
    if not download_only:
        with response.open('rb') as f:first=json.loads(f.readline(op.MiB+1))
        if first['kind']!=op.KIND or first['phase']!='scratch-created' or first['control_sha256']!=sha:raise ValueError('Missing authenticated scratch receipt; preserve partial files.')
        claim=json.loads((folder/'ATTEMPT.json').read_text());claim['scratch_directory']=first['scratch_directory'];old.write_json(folder/'ATTEMPT.json',claim,replace=True)
    unpack(response,session/'saved',c,sha,collect=download_only)
    print('Evidence saved at',session,'for independent review. No automatic retry.',flush=True)
    return proc.returncode
