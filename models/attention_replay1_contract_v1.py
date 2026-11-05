"""Reference-subset provenance and bounds; imports only the standard library."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import tarfile

from models.oracle_boundary_observer_v1 import record,read_record,digest

KIND='attention-replay1-reference-subset-v1'
LIVE_MANIFEST_SHA='806afcf7bd8ee7b6384829607d92b820c9a3cb0ee75e4d53fe7004155780db18'
PAIR_REVIEW_SHA='349e1d1225b92afaf8c75702c0bd03b607e55c026b1618cd9cf9e0fd5f45ecd8'
MIB=2**20
WORKER_FILES=('authorization.json','worker-plan.json','worker.json','capture.json','saved-audit.json','values.pt')
PHASES=('untimed','profiled','warmup','timed-1','timed-2','timed-3')


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def safe(name):
    p=Path(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p)!=name or '\\' in name or name=='.':
        raise ValueError('Unsafe reference path.')
    return name


def imported(local):
    return ('encoder-1-math-0','encoder-1-math-1') if local else ('encoder-1-efficient-0','encoder-1-math-0')


def task(index,local):
    if type(index) is not int or index not in (0,1):raise ValueError('Exactly two new worker slots.')
    if local:
        return {'case':'encoder-1','arm':'math','replay':1,
            'worker':'encoder-1-math-replay1-'+('baseline' if index==0 else 'candidate')+'-analogue'}
    arm=('efficient','math')[index]
    return {'case':'encoder-1','arm':arm,'replay':1,'worker':'encoder-1-'+arm+'-1'}


def expected_files(local):
    names={f'workers/{worker}/{name}' for worker in imported(local) for name in WORKER_FILES}
    provenance=('local-pair.json',) if local else ('accepted-pair-review.json','accepted-pair-receipt.json',
        'accepted-operator-result.json','accepted-transport-verification.json')
    return names|{'provenance/'+n for n in provenance}


def validate_manifest(manifest,*,local):
    if local:
        if platform.node() in ('linux10603','linux10609') or manifest.get('kind')!='local-tiny-CPU-reference-subset-v1':
            raise ValueError('Tiny reference analogues are local only.')
    elif hashlib.sha256(canonical(manifest)).hexdigest()!=LIVE_MANIFEST_SHA:
        raise ValueError('Only the exact accepted 16-file reference subset is allowed.')
    files=manifest['files']
    if set(files)!=expected_files(local):raise ValueError('Reference subset coverage differs.')
    if sum(x['bytes'] for x in files.values())>64*MIB or not local and sum(x['bytes'] for x in files.values())!=59132357:
        raise ValueError('Reference subset extent differs.')
    for name,item in files.items():
        safe(name);safe(item['source'])
        if (type(item['bytes']) is not int or not 0<item['bytes']<=32*MIB or type(item['mtime_ns']) is not int
                or item['mtime_ns']<0 or not isinstance(item['sha256'],str) or len(item['sha256'])!=64):
            raise ValueError('Invalid reference extent/hash/time.')


def create_archive(source_root,manifest,path,*,local):
    validate_manifest(manifest,local=local);source_root=Path(source_root).resolve();path=Path(path)
    if path.exists() or path.with_name(path.name+'.partial').exists():raise FileExistsError('Preserve the existing reference archive.')
    for name,item in manifest['files'].items():
        p=source_root/item['source'];info=p.lstat()
        if (p.is_symlink() or not p.resolve().is_relative_to(source_root) or not stat.S_ISREG(info.st_mode)
                or info.st_nlink!=1 or (info.st_size,info.st_mtime_ns)!=(item['bytes'],item['mtime_ns']) or digest(p)!=item['sha256']):
            raise ValueError('Accepted reference source changed: '+name)
    partial=path.with_name(path.name+'.partial')
    with partial.open('xb') as out:
        with tarfile.open(fileobj=out,mode='w',format=tarfile.USTAR_FORMAT) as archive:
            for name,item in sorted(manifest['files'].items()):
                member=tarfile.TarInfo(name);member.size=item['bytes'];member.mode=0o400;member.mtime=item['mtime_ns']//10**9
                with (source_root/item['source']).open('rb') as src:archive.addfile(member,src)
        out.flush();os.fsync(out.fileno())
    if partial.stat().st_size>64*MIB:raise ValueError('Reference archive exceeds 64 MiB; partial retained.')
    os.link(partial,path);partial.unlink()
    return {'kind':'attention-replay1-reference-archive-v1','sha256':digest(path),'bytes':path.stat().st_size,
        'manifest_sha256':hashlib.sha256(canonical(manifest)).hexdigest(),'local_tiny_analogue':local}


def extract_archive(path,spec,manifest,results,*,local):
    validate_manifest(manifest,local=local);path,results=Path(path),Path(results)
    if (path.is_symlink() or not 0<spec['bytes']<=64*MIB or path.stat().st_size!=spec['bytes'] or digest(path)!=spec['sha256']
            or spec['kind']!='attention-replay1-reference-archive-v1' or spec['local_tiny_analogue'] is not local
            or spec['manifest_sha256']!=hashlib.sha256(canonical(manifest)).hexdigest()):raise ValueError('Reference archive binding differs.')
    if results.is_symlink() or not results.is_dir():raise ValueError('Require the fresh owned results directory.')
    for name in manifest['files']:
        p=results/safe(name)
        if p.exists() or p.is_symlink():raise FileExistsError('Reference import never overwrites: '+name)
        for parent in p.parents:
            if parent==results.parent:break
            if parent.is_symlink():raise ValueError('Reference parent alias rejected.')
    with tarfile.open(path,mode='r:') as archive:
        members=archive.getmembers()
        if (len(members)!=len(manifest['files']) or {m.name for m in members}!=set(manifest['files'])
                or any(not m.isfile() or m.size!=manifest['files'][m.name]['bytes'] for m in members)):
            raise ValueError('Reference archive contains unexpected, duplicate or linked members.')
        for member in members:
            p=results/member.name;item=manifest['files'][member.name];p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            with archive.extractfile(member) as src,p.open('xb') as out:
                shutil.copyfileobj(src,out,2**20);out.flush();os.fsync(out.fileno())
            if digest(p)!=item['sha256']:raise ValueError('Imported reference checksum differs; preserve evidence.')
            os.utime(p,ns=(item['mtime_ns'],item['mtime_ns']));p.chmod(0o400)
    return verify_imports(results,manifest,local=local)


def verify_imports(results,manifest,*,local):
    validate_manifest(manifest,local=local);results=Path(results)
    for name,item in manifest['files'].items():
        p=results/name;info=p.lstat()
        if (p.is_symlink() or not p.resolve().is_relative_to(results.resolve()) or info.st_uid!=os.getuid()
                or not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or stat.S_IMODE(info.st_mode)!=0o400
                or (info.st_size,info.st_mtime_ns)!=(item['bytes'],item['mtime_ns']) or digest(p)!=item['sha256']):
            raise ValueError('Imported immutable reference changed: '+name)
    for name in imported(local):
        folder=results/'workers'/name
        if {p.name for p in folder.iterdir()}!=set(WORKER_FILES):raise ValueError('The old reference is an exact subset, not a new full capture.')
        w=read_record(folder/'worker.json');cap=read_record(folder/'capture.json');plan=read_record(folder/'worker-plan.json')
        if (not w['passed'] or not w['complete'] or not w['within_process_exact'] or w['tiny_fixture'] is not local
                or cap['local_tiny_rehearsal'] is not local or cap['outcome']!='original-worker-passed'
                or cap['capture_coverage_complete'] is not True or cap['saved_value_checks_passed'] is not True
                or digest(folder/'values.pt')!=w['artifact']['sha256'] or digest(folder/'worker.json')!=cap['original_worker_sha256']
                or digest(folder/'saved-audit.json')!=cap['saved_audit_sha256']
                or digest(folder/'authorization.json')!=plan['original_authorization_sha256']):
            raise ValueError('Imported original worker/capture was not accepted.')
    if local:
        prior=read_record(results/'provenance/local-pair.json')
        if prior['local_tiny_rehearsal'] is not True or not prior['pair_checks_passed']:raise ValueError('Require genuine tiny CPU math provenance.')
    else:
        if digest(results/'provenance/accepted-pair-review.json')!=PAIR_REVIEW_SHA:raise ValueError('Accepted review changed.')
        prior=json.loads((results/'provenance/accepted-pair-review.json').read_text())
        if not prior['independent_review_complete'] or not prior['pair_checks_passed'] or not prior['attempt_consumed']:
            raise ValueError('Require the accepted consumed pair review.')
    return {'kind':KIND,'local_tiny_analogue':local,'workers':list(imported(local)),'files_verified':len(manifest['files']),
        'manifest_sha256':hashlib.sha256(canonical(manifest)).hexdigest(),'read_only':True,'imported_reference_subset_only':True,
        'old_NPZ_files_present':False,'old_grants_renewed':False,'new_workers_executed_by_import':0}
