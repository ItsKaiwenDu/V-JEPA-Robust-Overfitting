"""One separately authorized read-only SSH observation; never a model launcher."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import stat
import time
from scripts import university_gpu_availability_remote_v1 as remote

HOST='linux10603.dc.engr.scu.edu'
FINGERPRINT='SHA256:ARgVRWOzRLA1LmAANxD6Ya17EIIo2sa4ZbHvkeqzgKQ'
KNOWN_HOSTS='linux10603.dc.engr.scu.edu ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAELc+tu7f+683dQbOPg5lndZ9pGDeODvo8r1Sp8d7ri\n'
SOURCE_NAMES=('models/university_gpu_availability_v1.py','scripts/university_gpu_availability_remote_v1.py','scripts/run_university_gpu_availability_v1.py')
LIMITS={'client_connection_seconds_including_login':180,'remote_group_seconds':20,'remote_group_kill_grace_seconds':1,'each_query_seconds':8,
        'query_stdout_bytes':8192,'query_stderr_bytes':2048,'response_bytes':65536,'SSH_stderr_bytes':16384,'maximum_connections':1,'automatic_retries':0}
FLAGS={'read_only':True,'model_work_authorized':False,'GPU_reserved':False,'scientific_qualified':False,'training_admitted':False}


def encoded(v):return remote.encoded(v)
def digest(data):return remote.digest(data)
def parse(raw):
    def pairs(rows):
        d={}
        for k,v in rows:
            if k in d:raise ValueError('Duplicate metadata key')
            d[k]=v
        return d
    def bad(v):raise ValueError('Nonfinite metadata')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)


def read_file(path,cap,wanted=None):
    path=Path(path);before=path.lstat()
    if (path.resolve(strict=True)!=path or not stat.S_ISREG(before.st_mode) or before.st_uid!=os.getuid() or before.st_nlink!=1
            or stat.S_IMODE(before.st_mode) not in (0o400,0o600,0o644) or not 0<before.st_size<=cap):raise ValueError('Canonical owned bounded single-link source required.')
    def identity(s):return (s.st_dev,s.st_ino,s.st_uid,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
        held=os.fstat(f.fileno());data=f.read(cap+1);after=os.fstat(f.fileno())
    if identity(before)!=identity(held) or identity(before)!=identity(after) or identity(before)!=identity(path.lstat()) or len(data)!=before.st_size or wanted is not None and digest(data)!=wanted:
        raise ValueError('Source identity/extent/hash changed.')
    return data


def write_new(root,name,data):
    if name not in ('CONTROL.json','QUERY.py','known-hosts','PACKAGE.json','READ-AUTHORIZATION.json','READ-CONNECTION.json','STDOUT.bin','STDERR.bin','RESULT.json'):
        raise ValueError('Only availability metadata files may be created.')
    root=Path(root)
    if root.resolve(strict=True)!=root or stat.S_IMODE(root.stat().st_mode)!=0o700 or root.stat().st_uid!=os.getuid():raise ValueError('Private local directory required.')
    with os.fdopen(os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())


def control(raw):
    c=parse(raw)
    keys={'kind','repository','handoff','host','user','fingerprint','known_hosts_sha256','program_sha256','program_bytes','source_pins','limits','flags'}
    if (type(c) is not dict or set(c)!=keys or encoded(c)!=raw or c['kind']!='university-gpu-availability-control-v1'
            or c['host']!=HOST or c['user']!='kdu' or c['fingerprint']!=FINGERPRINT or c['known_hosts_sha256']!=digest(KNOWN_HOSTS.encode())
            or encoded(c['limits'])!=encoded(LIMITS) or encoded(c['flags'])!=encoded(FLAGS) or type(c['program_bytes']) is not int or not 0<c['program_bytes']<=32768
            or not remote.is_sha(c['program_sha256']) or type(c['source_pins']) is not dict or set(c['source_pins'])!=set(SOURCE_NAMES)
            or any(not remote.is_sha(v) for v in c['source_pins'].values())):raise ValueError('Exact availability-only source/host/limits required.')
    for name in ('repository','handoff'):
        p=Path(c[name])
        if type(c[name]) is not str or not p.is_absolute() or str(p)!=c[name] or '..' in p.parts:raise ValueError('Exact canonical local path required.')
    if c['program_sha256']!=c['source_pins'][SOURCE_NAMES[1]]:raise ValueError('Pinned query program required.')
    return c,digest(raw)


def authorization(csha,basis):
    if not remote.is_sha(csha) or type(basis) is not str or not 1<=len(basis)<=8192:raise ValueError('One exact read-only approval basis required.')
    return {'kind':'university-gpu-availability-read-only-authorization-v1','approved':True,'control_sha256':csha,
            'maximum_connections':1,'host':HOST,'limits':LIMITS,'flags':FLAGS,'basis':basis}


def prepare(repository,destination):
    root=Path(repository);dest=Path(destination)
    if (root.resolve(strict=True)!=root or not root.is_absolute() or not dest.is_absolute() or dest.parent.resolve(strict=True)!=dest.parent
            or os.path.lexists(dest) or stat.S_IMODE(dest.parent.stat().st_mode)!=0o700 or dest.parent.stat().st_uid!=os.getuid()):raise ValueError('Fresh bound private handoff required.')
    data={n:read_file(root/n,65536) for n in SOURCE_NAMES}
    for n,value in data.items():compile(value,'<UNEXECUTED-AVAILABILITY-'+n+'>','exec')
    program=data[SOURCE_NAMES[1]]
    c={'kind':'university-gpu-availability-control-v1','repository':str(root),'handoff':str(dest),'host':HOST,'user':'kdu','fingerprint':FINGERPRINT,
       'known_hosts_sha256':digest(KNOWN_HOSTS.encode()),'program_sha256':digest(program),'program_bytes':len(program),
       'source_pins':{n:digest(v) for n,v in data.items()},'limits':LIMITS,'flags':FLAGS}
    raw=encoded(c);_,sha=control(raw);dest.mkdir(mode=0o700)
    result={'kind':'university-gpu-availability-unarmed-package-v1','control_sha256':sha,'program_sha256':c['program_sha256'],
            'program_bytes':len(program),'authorization_created':False,'SSH_or_query_started':False,**FLAGS}
    for n,value in [('CONTROL.json',raw),('QUERY.py',program),('known-hosts',KNOWN_HOSTS.encode()),('PACKAGE.json',encoded(result))]:write_new(dest,n,value)
    return result


def check_local(folder):
    folder=Path(folder)
    if not (folder/'READ-AUTHORIZATION.json').is_file():raise ValueError('Unarmed: exact read-only approval required; no SSH or model work.')
    if folder.resolve(strict=True)!=folder or not folder.is_absolute() or stat.S_IMODE(folder.stat().st_mode)!=0o700 or folder.stat().st_uid!=os.getuid():raise ValueError('Bound private handoff required.')
    c,csha=control(read_file(folder/'CONTROL.json',16384))
    if str(folder)!=c['handoff']:raise ValueError('Copied handoff cannot reuse a connection grant.')
    for n,w in c['source_pins'].items():read_file(Path(c['repository'])/n,65536,w)
    program=read_file(folder/'QUERY.py',32768,c['program_sha256'])
    if len(program)!=c['program_bytes'] or read_file(folder/'known-hosts',16384,c['known_hosts_sha256'])!=KNOWN_HOSTS.encode():raise ValueError('Query/host binding changed.')
    raw=read_file(folder/'READ-AUTHORIZATION.json',16384);a=parse(raw)
    if type(a) is not dict or encoded(authorization(csha,a.get('basis')))!=raw:raise ValueError('One new exact availability-only grant required.')
    return c,csha,program,digest(raw)


def ssh_command(folder,c,csha,program,nonce):
    if not remote.is_sha(nonce) or digest(program)!=c['program_sha256']:raise ValueError('Bound request nonce/program required.')
    command='exec /usr/bin/timeout --signal=TERM --kill-after=1s 20s "$HOME/.venvs/vjepa/bin/python" -I -S -B -u -c '+shlex.quote(program.decode())+' '+shlex.quote(nonce)+' '+shlex.quote(csha)+' '+shlex.quote(c['program_sha256'])
    return ['ssh','-F','/dev/null','-T','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(Path(folder)/'known-hosts'),
        '-o','GlobalKnownHostsFile=/dev/null','-o','ForwardAgent=no','-o','ClearAllForwardings=yes','-o','PermitLocalCommand=no',
        '-o','PubkeyAuthentication=no','-o','PreferredAuthentications=keyboard-interactive,password','-o','NumberOfPasswordPrompts=1',
        '-o','ConnectionAttempts=1','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','kdu@'+HOST,command]


def validate_response(raw,c,csha,nonce):
    if type(raw) is not bytes or not 0<len(raw)<=65536:raise ValueError('One bounded complete response required.')
    r=parse(raw)
    keys={'kind','request_nonce','control_sha256','program_sha256','host','checked_utc','elapsed_seconds','queries','availability','gpu','compute_processes','errors',*FLAGS}
    if (type(r) is not dict or set(r)!=keys or encoded(r)!=raw or r['kind']!='university-gpu-availability-observation-v1'
            or r['request_nonce']!=nonce or r['control_sha256']!=csha or r['program_sha256']!=c['program_sha256']
            or any(type(r.get(k)) is not type(v) or r[k]!=v for k,v in FLAGS.items())
            or type(r['elapsed_seconds']) not in (int,float) or not 0<=r['elapsed_seconds']<=20):raise ValueError('Bound read-only response identity/limits required.')
    stamp=datetime.fromisoformat(r['checked_utc'])
    if stamp.tzinfo is None or stamp.utcoffset().total_seconds()!=0:raise ValueError('UTC observation timestamp required.')
    evaluated=remote.evaluate(r['host'],r['queries'])
    if any(r[k]!=v for k,v in evaluated.items()):raise ValueError('Availability classification differs from saved query evidence.')
    return r


def run(folder,*,clock=None,capture_fn=None,nonce_fn=None):
    """One read-only connection. No model launch, retries, packet or budget upload."""
    folder=Path(folder)
    if not (folder/'READ-AUTHORIZATION.json').is_file():raise ValueError('Unarmed: no read-only approval; no connection started.')
    clock=time.monotonic if clock is None else clock;started=clock();deadline=started+180
    c,csha,program,asha=check_local(folder)
    if (folder/'READ-CONNECTION.json').exists():raise ValueError('Read connection already consumed; preserve evidence and do not repeat.')
    nonce=(lambda:secrets.token_hex(32))() if nonce_fn is None else nonce_fn()
    if not remote.is_sha(nonce):raise ValueError('Fresh bounded nonce required.')
    if clock()>=deadline:raise TimeoutError('Local preflight consumed the original connection window.')
    write_new(folder,'READ-CONNECTION.json',encoded({'kind':'university-gpu-availability-connection-v1','consumed':True,
        'control_sha256':csha,'authorization_sha256':asha,'request_nonce':nonce,'started_monotonic':started,'deadline_monotonic':deadline,**FLAGS}))
    session=folder/'read-response';session.mkdir(mode=0o700)
    out=err=b'';status=None;observation=None;problem=None
    try:
        capture_fn=remote.capture if capture_fn is None else capture_fn
        status,out,err=capture_fn(ssh_command(folder,c,csha,program,nonce),deadline=deadline,stdout_cap=65536,stderr_cap=16384)
        if clock()>=deadline:raise remote.CaptureError('Original total deadline reached',out,err)
        observation=validate_response(out,c,csha,nonce)
        if status!=0 or observation['availability'] not in ('idle_observed','busy_observed'):problem='Read did not establish availability; retain diagnostics.'
    except (remote.CaptureError,ValueError,TypeError,KeyError,OSError,KeyboardInterrupt) as error:
        out=getattr(error,'stdout',out);err=getattr(error,'stderr',err);problem=type(error).__name__+': '+str(error)[:512]
    finally:
        write_new(session,'STDOUT.bin',out);write_new(session,'STDERR.bin',err)
        result={'kind':'university-gpu-availability-local-result-v1','control_sha256':csha,'request_nonce':nonce,'connection_consumed':True,
            'SSH_exit_status':status,'stdout_sha256':digest(out),'stdout_bytes':len(out),'stderr_sha256':digest(err),'stderr_bytes':len(err),
            'observation':observation,'error':problem,'independent_review_pending':True,'model_attempt_claim_created':False,**FLAGS}
        write_new(session,'RESULT.json',encoded(result))
    print('Availability observation: '+(observation['availability'] if observation is not None else 'unknown')+'. Evidence: '+str(session)+'. No model starts; do not repeat this read helper.',flush=True)
    return 0 if problem is None else 1
