#!/usr/bin/env python3
"""Read-only capture of the failed 10603 case; never start or accept a model.

No project imports, tests, CUDA, model calls, writes to university files, or
ledger repair. Receiving locally verifies bytes, not numerical acceptance.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import socket
import subprocess
from datetime import datetime, timezone
import sys
import tarfile

KIND='panel-partial-contention-download-v1'
HOST='linux10603'
REVISION='5ca7596fe95ecb8f6b6c6752426e56bb354b1f019d3576c6618f7ede98ae4116'
IMPLEMENTATION='da5aeb93bd6d3e46040b158cc14f0d6feb1284a83b2b2778a3639b420db0acf9'
BASE='Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024'
CASES={'linux10603':'clip-04-bank-1'}
OPERATION='b5501e3fbc4c76b2d66d49ddc063fe64dd70f22885ef3d506ce51c4bcf391a05'
LIMIT=350*2**20


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def inventory(home):
    transfer=Path(home)/BASE;root=transfer/'two-host-panel-20261026'
    packet=transfer/'two-host-continuation-20261026/unpacked/two-host-continuation-20261026'
    original=transfer/'two-host-execution-20261026/unpacked/two-host-execution-20261026'
    directory=root/'cases/clip-04-bank-1/attempt-01'
    required={'ledger/STOP.json':root/'STOP.json','ledger/genesis.json':root/'genesis.json','ledger/imports.json':root/'imports.json',
        'ledger/claim.json':root/'claims/clip-04-bank-1/claim.json','ledger/outcome.json':root/'outcomes/clip-04-bank-1.json',
        'operation/OPERATION.json':packet/'OPERATION.json','operation/resolution.json':root/'stop-resolutions/export-alias-20261026.json',
        'operation/archived-STOP.json':root/'archived-stops/b42a023f4b7e3bf0c9183f7af384d4abe3cccecaf05d727495b9782107c0f892.json',
        'packet/IMPLEMENTATION.json':original/'IMPLEMENTATION.json','packet/revision.json':original/'configs/matched_panel_two_host_v1.json'}
    for name in ('numeric-evidence.tar.gz','numeric-evidence.tar.gz.manifest.json','session.json','case-envelope.json',
                 'case-claim.json','case-request.json','matrix.log'):required['case/'+name]=directory/name
    optional={}
    for name in ('receipt.json','replay.log','revision-evidence.tar.gz','revision-evidence.tar.gz.manifest.json',
                 'matrix/observation.json','matrix/worker.json','replay/observation.json','replay/worker.json'):
        optional['case/'+name]=directory/name
    for name in ('exports','reviews','failures'):optional['ledger/'+name+'.json']=root/name/'clip-04-bank-1.json'
    for name in ('started-utc.txt','ended-utc.txt','exit-status.txt','case.log','cpu.log','cpu-results.xml','cpu-summary.log',
                 'cpu-old.log','cpu-new.log','cpu-old.xml','cpu-new.xml','source-launch.sh','source-pipeline.sh'):
        required['launcher/'+name]=packet/'launchers/linux10603-clip-04-bank-1'/name
    return required,optional


def safe_name(name):
    value=PurePosixPath(name)
    if not name or value.is_absolute() or '..' in value.parts or '\\' in name or str(value)!=name or name=='.':
        raise ValueError('Unsafe incident member name.')
    return name


def collect(home,host):
    if host!=HOST:raise ValueError('Use the explicitly pinned university host.')
    required,optional=inventory(home);files=dict(required);missing=[]
    for name,path in optional.items():
        if path.exists():files[name]=path
        else:missing.append(name)
    allowed=(Path(home)/BASE).resolve();entries={}
    for name,path in files.items():
        safe_name(name)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(allowed):
            raise ValueError('Missing or escaped incident artifact: '+name)
        size=path.stat().st_size
        if size>(256*2**20 if name.endswith('.tar.gz') else 16*2**20):raise ValueError('Oversized incident artifact.')
        entries[name]={'bytes':size,'sha256':digest(path)}
    if sum(e['bytes'] for e in entries.values())>LIMIT:raise ValueError('Incident evidence exceeds its fixed bound.')
    if canonical(json.loads(files['packet/IMPLEMENTATION.json'].read_text()))!=IMPLEMENTATION:
        raise ValueError('Deployed implementation differs from the completed batch.')
    if canonical(json.loads(files['packet/revision.json'].read_text()))!=REVISION:
        raise ValueError('Deployed revision differs from the completed batch.')
    name='case/numeric-evidence.tar.gz';sidecar=json.loads(files[name+'.manifest.json'].read_text())
    if sidecar['sha256']!=entries[name]['sha256'] or sidecar['bytes']!=entries[name]['bytes']:
        raise ValueError('Partial export differs from its sidecar.')
    if canonical(json.loads(files['operation/OPERATION.json'].read_text()))!=OPERATION:raise ValueError('Operational revision changed.')
    snapshot=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=10)
    manifest={'kind':KIND,'host':host,'revision_sha256':REVISION,'implementation_sha256':IMPLEMENTATION,
        'files':entries,'absent_optional_files':sorted(missing),'case_id':'clip-04-bank-1','operation_sha256':OPERATION,
        'snapshot_utc':datetime.now(timezone.utc).isoformat(),'current_nvidia_smi':{'exit_status':snapshot.returncode,'stdout':snapshot.stdout,'stderr':snapshot.stderr},'remote_files_changed':False,
        'GPU_or_test_job_launched':False,'independent_numerical_acceptance':False}
    return files,manifest


def stream(home,host):
    files,manifest=collect(home,host)
    class Writer:
        def __init__(self):self.checksum=hashlib.sha256();self.bytes=0
        def write(self,data):
            count=sys.stdout.buffer.write(data)
            if count!=len(data):raise OSError('Short incident transport write.')
            self.checksum.update(data);self.bytes+=len(data);return count
        def flush(self):sys.stdout.buffer.flush()
    writer=Writer()
    print('Copying only the existing failed 10603 case, logs and current read-only GPU snapshot.',file=sys.stderr,flush=True)
    with tarfile.open(fileobj=writer,mode='w|') as archive:
        for name,path in sorted(files.items()):
            info=tarfile.TarInfo(name);info.size=manifest['files'][name]['bytes'];info.mode=0o600
            with path.open('rb') as source:archive.addfile(info,source)
            if digest(path)!=manifest['files'][name]['sha256']:raise ValueError('Evidence changed while copying.')
        data=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
        info=tarfile.TarInfo('transport-manifest.json');info.size=len(data);info.mode=0o600
        archive.addfile(info,io.BytesIO(data))
    writer.flush()
    print('PANEL_PARTIAL_BUNDLE '+json.dumps({'bytes':writer.bytes,'sha256':writer.checksum.hexdigest()}),file=sys.stderr,flush=True)


def receive(directory):
    directory=Path(directory);path=directory/'transport.tar.part'
    lines=[s.removeprefix('PANEL_PARTIAL_BUNDLE ') for s in (directory/'transport.stderr').read_text().splitlines()
           if s.startswith('PANEL_PARTIAL_BUNDLE ')]
    if len(lines)!=1:raise ValueError('No unique completed transport checksum; preserve the partial copy.')
    transport=json.loads(lines[0])
    if (path.is_symlink() or not path.is_file() or path.stat().st_size>LIMIT+2**20
        or path.stat().st_size!=transport['bytes'] or digest(path)!=transport['sha256']):
        raise ValueError('Incident transport checksum/size mismatch; preserve the partial copy.')
    required,optional=inventory(Path('/unused'))
    with tarfile.open(path) as archive:
        members=archive.getmembers();names=[safe_name(m.name) for m in members]
        if (len(names)!=len(set(names)) or not all(m.isfile() and m.size>=0 for m in members)
            or sum(m.size for m in members)>LIMIT or 'transport-manifest.json' not in names):
            raise ValueError('Invalid incident inventory.')
        entry=archive.getmember('transport-manifest.json')
        if entry.size>65536:raise ValueError('Oversized transport manifest.')
        with archive.extractfile(entry) as source:manifest=json.load(source)
        declared=set(manifest['files']);absent=manifest['absent_optional_files']
        if (manifest['kind']!=KIND or manifest['host']!=HOST or manifest['revision_sha256']!=REVISION
            or manifest['implementation_sha256']!=IMPLEMENTATION or manifest['case_id']!='clip-04-bank-1' or manifest['operation_sha256']!=OPERATION
            or manifest['remote_files_changed'] is not False or manifest['GPU_or_test_job_launched'] is not False
            or manifest['independent_numerical_acceptance'] is not False
            or not set(required)<=declared or not declared<=set(required)|set(optional)
            or sorted(set(optional)-declared)!=sorted(absent) or len(absent)!=len(set(absent))
            or set(names)!=declared|{'transport-manifest.json'}):raise ValueError('Wrong or incomplete incident scope.')
        for member in members:
            if member.name=='transport-manifest.json':continue
            expected=manifest['files'][member.name]
            with archive.extractfile(member) as source:checksum=hashlib.file_digest(source,'sha256').hexdigest()
            if member.size!=expected['bytes'] or checksum!=expected['sha256']:raise ValueError('Incident member checksum mismatch.')
        destination=directory/'downloaded';destination.mkdir(exist_ok=False)
        for member in members:
            target=destination/member.name;target.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as source,target.open('xb') as output:
                while block:=source.read(2**20):output.write(block)
    result={'kind':KIND+'-byte-verification','transport':transport,'files_verified':len(manifest['files']),
        'absent_optional_files':absent,'independent_numerical_acceptance':False,'GPU_job_repeated':False}
    (directory/'download-verification.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    path.rename(directory/'transport.tar')
    return result


if __name__=='__main__':
    if len(sys.argv)==1:stream(Path.home(),socket.gethostname())
    else:
        parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('receive',));parser.add_argument('directory',type=Path)
        args=parser.parse_args();print(json.dumps(receive(args.directory),indent=2,sort_keys=True))
