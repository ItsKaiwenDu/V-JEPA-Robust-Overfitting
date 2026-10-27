#!/usr/bin/env python3
"""Read-only capture of a failed recovery startup; never import model code."""
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import socket
import subprocess
import sys
import tarfile

KIND='read-only-recovery-lock-startup-v1'
HOST='linux10603';CASE='clip-04-bank-1'
BASE='Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024'
PREFIX='contended-panel-recovery-20261026'
PROPOSAL='2d841e90f4378507b80e4805e1096e17ed2231c1257c9b57d405b802f32164f4'
AUTH='721bbf9e77804b7acc7bdf542c63e2d29cda3f56b148dec7b24fea810c0f9d2d'
LIMIT=4*2**20
CODE=('scripts/recover_contended_panel_case.py','scripts/prepare_contended_panel_recovery.py',
      'tests/test_contended_panel_recovery.py','tests/fixtures/contended_panel_metadata.json')
LAUNCH=('started-utc.txt','ended-utc.txt','exit-status.txt','case.log','cpu.log','cpu-results.xml','cpu-summary.log',
        'cpu-old.log','cpu-new.log','cpu-old.xml','cpu-new.xml','source-launch.sh','source-pipeline.sh')


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def safe(name):
    p=PurePosixPath(name)
    if not name or name=='.' or p.is_absolute() or '..' in p.parts or '\\' in name or str(p)!=name:raise ValueError('Unsafe startup member.')
    return name


def files_for(home):
    transfer=Path(home)/BASE;packet=transfer/PREFIX/'unpacked'/PREFIX;root=transfer/'two-host-panel-20261026'
    original=transfer/'two-host-execution-20261026/unpacked/two-host-execution-20261026'
    files={'packet/'+name:packet/name for name in (*CODE,'RECOVERY.json','authorization.json','biking-independent-review.json','remote-launch.sh','remote-pipeline.sh')}
    files.update({'launcher/'+n:packet/'launcher'/n for n in LAUNCH})
    for label,relative in {'STOP.json':'STOP.json','genesis.json':'genesis.json','imports.json':'imports.json',
        'failed-claim.json':'claims/'+CASE+'/claim.json','failed-outcome.json':'outcomes/'+CASE+'.json',
        'failed-session.json':'cases/'+CASE+'/attempt-01/session.json','failed-envelope.json':'cases/'+CASE+'/attempt-01/case-envelope.json'}.items():files['ledger/'+label]=root/relative
    files['reference/two_host_panel.py']=original/'models/two_host_panel.py';files['reference/IMPLEMENTATION.json']=original/'IMPLEMENTATION.json'
    return files,packet,root


def collect(home,host):
    if host!=HOST:raise ValueError('Use the pinned linux10603 workstation.')
    files,packet,root=files_for(home);allowed=(Path(home)/BASE).resolve();inventory={}
    for name,path in files.items():
        safe(name)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(allowed):raise ValueError('Missing or escaped startup file: '+name)
        if path.stat().st_size>2**20:raise ValueError('Oversized startup artifact.')
        inventory[name]={'bytes':path.stat().st_size,'sha256':digest(path)}
    if sum(x['bytes'] for x in inventory.values())>LIMIT:raise ValueError('Startup capture exceeds its fixed bound.')
    spec=json.loads((packet/'RECOVERY.json').read_text())
    if canonical(spec)!=PROPOSAL or digest(packet/'authorization.json')!=AUTH:raise ValueError('Wrong failed recovery identity.')
    for name,pin in spec['sources_sha256'].items():
        if digest(packet/name)!=pin:raise ValueError('Deployed recovery source changed.')
    observed={}
    for name,pin in spec['preserved_coordinator_files_sha256'].items():
        safe(name);path=root/name
        observed[name]={'expected_sha256':pin,'observed_sha256':digest(path) if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()) else None}
    locations={}
    for label,relative in {'recovery_grant':'recovery-grants/'+CASE,'attempt_02':'cases/'+CASE+'/attempt-02'}.items():
        path=root/relative;locations[label]={'relative_path':relative,'exists':path.exists(),'is_symlink':path.is_symlink()}
    result=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=10)
    manifest={'kind':KIND,'host':HOST,'case_id':CASE,'proposal_sha256':PROPOSAL,'authorization_file_sha256':AUTH,
        'snapshot_UTC':datetime.now(timezone.utc).isoformat(),'files':inventory,'coordinator_file_hashes':observed,
        'coordinator_pins_unchanged':all(v['expected_sha256']==v['observed_sha256'] for v in observed.values()),
        'recovery_locations':locations,'current_nvidia_smi':{'exit_status':result.returncode,'stdout':result.stdout,'stderr':result.stderr},
        'university_files_changed':False,'model_or_test_job_started':False,'scientific_case_accepted':False}
    return files,manifest


def stream(home,host):
    files,manifest=collect(home,host)
    class Writer:
        def __init__(self):self.checksum=hashlib.sha256();self.bytes=0
        def write(self,data):
            count=sys.stdout.buffer.write(data)
            if count!=len(data):raise OSError('Short startup evidence write.')
            self.checksum.update(data);self.bytes+=count;return count
        def flush(self):sys.stdout.buffer.flush()
    writer=Writer()
    with tarfile.open(fileobj=writer,mode='w|') as archive:
        for name,path in sorted(files.items()):
            info=tarfile.TarInfo(name);info.size=manifest['files'][name]['bytes'];info.mode=0o600
            with path.open('rb') as f:archive.addfile(info,f)
            if digest(path)!=manifest['files'][name]['sha256']:raise ValueError('Startup file changed during copy.')
        data=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode();info=tarfile.TarInfo('transport-manifest.json');info.size=len(data);info.mode=0o600
        archive.addfile(info,io.BytesIO(data))
    writer.flush();print('RECOVERY_STARTUP_BUNDLE '+json.dumps({'bytes':writer.bytes,'sha256':writer.checksum.hexdigest()}),file=sys.stderr,flush=True)


def receive(directory):
    directory=Path(directory);path=directory/'transport.tar.part'
    lines=[s.removeprefix('RECOVERY_STARTUP_BUNDLE ') for s in (directory/'transport.stderr').read_text().splitlines() if s.startswith('RECOVERY_STARTUP_BUNDLE ')]
    if len(lines)!=1:raise ValueError('No unique completed startup checksum; retain partial copy.')
    transport=json.loads(lines[0]);expected=set(files_for(Path('/unused'))[0])
    if path.is_symlink() or not path.is_file() or path.stat().st_size>LIMIT+2**20 or path.stat().st_size!=transport['bytes'] or digest(path)!=transport['sha256']:
        raise ValueError('Startup transport checksum/size mismatch.')
    with tarfile.open(path) as archive:
        members=archive.getmembers();names=[safe(m.name) for m in members]
        if (set(names)!=expected|{'transport-manifest.json'} or len(names)!=len(set(names))
            or not all(m.isfile() and m.size>=0 for m in members) or sum(m.size for m in members)>LIMIT):raise ValueError('Invalid startup inventory.')
        if archive.getmember('transport-manifest.json').size>65536:raise ValueError('Oversized startup manifest.')
        with archive.extractfile('transport-manifest.json') as f:manifest=json.load(f)
        if (manifest['kind']!=KIND or manifest['host']!=HOST or manifest['case_id']!=CASE
            or manifest['proposal_sha256']!=PROPOSAL or manifest['authorization_file_sha256']!=AUTH
            or set(manifest['files'])!=expected or any(manifest[k] is not False for k in ['university_files_changed','model_or_test_job_started','scientific_case_accepted'])):
            raise ValueError('Wrong startup-copy scope.')
        for m in members:
            if m.name=='transport-manifest.json':continue
            row=manifest['files'][m.name]
            with archive.extractfile(m) as f:pin=hashlib.file_digest(f,'sha256').hexdigest()
            if m.size!=row['bytes'] or pin!=row['sha256']:raise ValueError('Startup member checksum mismatch.')
        out=directory/'downloaded';out.mkdir(exist_ok=False)
        for m in members:
            target=out/m.name;target.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(m) as f,target.open('xb') as destination:
                while block:=f.read(2**20):destination.write(block)
    result={'kind':KIND+'-byte-verification','verified':True,'transport':transport,'files':len(expected),
        'independent_startup_review_complete':False,'new_model_job_started':False}
    (directory/'download-verification.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');path.rename(directory/'transport.tar')
    return result


if __name__=='__main__':
    if len(sys.argv)==1:stream(Path.home(),socket.gethostname())
    elif len(sys.argv)==3 and sys.argv[1]=='receive':print(json.dumps(receive(Path(sys.argv[2])),sort_keys=True,indent=2))
    else:raise SystemExit('Use receive DIRECTORY locally, or no arguments to stream fixed existing university files.')
