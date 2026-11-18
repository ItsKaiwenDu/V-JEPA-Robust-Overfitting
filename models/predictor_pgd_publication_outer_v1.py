"""Unarmed outer transport/setup definitions; no CLI or activity on import.

Real signal/process/proof/GPU operations are future authorized target bindings.
Local tests substitute those bindings; scientific workers never execute locally.
"""
import base64
import ast
import _thread
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import signal
import stat
import struct
import sys
import tarfile
import threading
import time
from types import SimpleNamespace

MiB,GiB=2**20,2**30
FRAME=b'VJEPA-PGD-SPLIT-OUTER-1\n'
POLICY_SHA='9f1d10b01fc48e09711084be515f1193fa5140244e1d07eb1848b0fde5164920'
BOOT_SHA='763a9d1d0e609583e1356392991ad93a52b50309099baeb417751183d4847239'
COLLECTOR_SHA='50e1deede0e0d9bf9a0f5a7ad9f1879920f4d01b74015666adcc89106b2523a1'
HOST='linux10614.dc.engr.scu.edu'
FINGERPRINT='SHA256:ARgVRWOzRLA1LmAANxD6Ya17EIIo2sa4ZbHvkeqzgKQ'
KNOWN_HOSTS='linux10614.dc.engr.scu.edu ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAELc+tu7f+683dQbOPg5lndZ9pGDeODvo8r1Sp8d7ri\n'
PACKET_PATH='runs/pretrained/diagnostic/eot-work-profile-university-20261029/packet/vjepa-eot-profile.tar.gz'
LOCAL_SOURCES=('models/predictor_pgd_publication_outer_v1.py', 'models/predictor_pgd_publication_client_v1.py', 'scripts/run_predictor_pgd_publication_v1.py', 'models/predictor_pgd_publication_handoff_v1.py', 'models/predictor_pgd_publication_integration_v1.py', 'models/predictor_pgd_publication_scan_v1.py', 'models/predictor_pgd_publication_scratch_v1.py', 'models/predictor_pgd_publication_bootstrap_v1.py', 'models/predictor_pgd_publication_bound_scratch_v1.py', 'models/predictor_pgd_publication_collection_v1.py', 'runs/pretrained/diagnostic/pgd-host10614-v1-20261117/handoff-parent/linux10614-final-unarmed/CONTROL-CANDIDATE.json', 'runs/pretrained/diagnostic/pgd-host10614-v1-20261117/handoff-parent/linux10614-final-unarmed/SOURCE-PAYLOAD.json', 'models/predictor_pgd_host10614_outer_v1.py', 'models/predictor_pgd_host10614_client_v1.py', 'scripts/run_predictor_pgd_host10614_v1.py', 'models/predictor_pgd_host10614_v1.py', 'models/predictor_pgd_host10614_bootstrap_v1.py', 'models/predictor_pgd_host10614_scratch_v1.py', 'models/predictor_pgd_host10614_collection_v1.py', 'configs/university_host10614_identity_v1.json', 'runs/pretrained/diagnostic/pgd-close-handoff-v1-20261117/handoff-parent/linux10609-final-unarmed/CONTROL-CANDIDATE.json', 'runs/pretrained/diagnostic/pgd-close-handoff-v1-20261117/handoff-parent/linux10609-final-unarmed/SOURCE-PAYLOAD.json', 'models/predictor_pgd_close_outer_v1.py', 'models/predictor_pgd_close_client_v1.py', 'models/predictor_pgd_split_outer_sources_v1.py', 'models/predictor_pgd_split_bootstrap_v1.py', 'models/predictor_pgd_split_coordinator_v1.py', 'models/predictor_pgd_split_scratch_v1.py', 'models/predictor_pgd_split_collection_v1.py', 'models/predictor_pgd_split_watch_v1.py', 'models/scratch_allocation_v2.py', 'scripts/run_predictor_pgd_close_v1.py', 'models/predictor_pgd_split_directories_v2.py', 'models/predictor_pgd_split_startup_sources_v3.py', 'models/predictor_pgd_split_preclaim_refusal_v4.py', 'models/predictor_pgd_split_outer_v3.py', 'models/predictor_pgd_host10609_v1.py', 'models/predictor_pgd_close_bootstrap_v1.py', 'models/predictor_pgd_close_collection_v1.py', 'models/predictor_pgd_split_source_bundle_v1.py', 'runs/pretrained/diagnostic/predictor-pgd-outer-v4-20261116/handoff-parent/linux10603-unarmed/CONTROL-CANDIDATE.json', 'runs/pretrained/diagnostic/predictor-pgd-outer-v4-20261116/handoff-parent/linux10603-unarmed/SOURCE-PAYLOAD.json', 'models/predictor_pgd_split_outer_v4.py', 'models/predictor_pgd_split_outer_client_v4.py', 'scripts/run_predictor_pgd_split_outer_v4.py', 'runs/pretrained/diagnostic/eot-work-profile-university-20261029/known-hosts', 'models/predictor_pgd_close_scratch_v1.py', 'models/predictor_pgd_close_handoff_v1.py', 'models/predictor_pgd_close_integration_v1.py', 'models/predictor_pgd_store_close_v2.py', 'models/predictor_pgd_exit_race_v1.py', 'runs/pretrained/diagnostic/predictor-pgd-host10609-v1-20261116/handoff-parent/linux10609-final-unarmed/CONTROL-CANDIDATE.json', 'runs/pretrained/diagnostic/predictor-pgd-host10609-v1-20261116/handoff-parent/linux10609-final-unarmed/SOURCE-PAYLOAD.json', 'models/predictor_pgd_split_outer_10609_v1.py', 'models/predictor_pgd_split_outer_client_10609_v1.py', 'scripts/run_predictor_pgd_split_outer_10609_v1.py')
DIRECTORY_POLICY_SHA='aee644895d35e0a3c7a67b471f4da0f0910a56f654ece38edeb2d3efa4e5a4c0'
REFUSAL_POLICY_SHA='c6e045b37602f639051dc8c717af4a583eeffec3d2ea12fb46a737db912b7721'
FROZEN_V3_SHA='8c21d936e5d311c7ddc0dffa4bc91c01a7f5c1c0e8731efa5fc33c58df57ac46'
SOURCE_CAP=2*MiB
FLAGS={'scientific_qualified':False,'training_admitted':False}


def encoded(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(data):return hashlib.sha256(data).hexdigest()
def sha(v):return type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v)
def finite(v):return type(v) in (int,float) and math.isfinite(v)
def parsed(data):
    def pairs(rows):
        out={}
        for k,v in rows:
            if k in out:raise ValueError('Duplicate outer metadata key.')
            out[k]=v
        return out
    def bad(x):raise ValueError('Nonfinite outer metadata: '+x)
    return json.loads(data,object_pairs_hook=pairs,parse_constant=bad)
def safe(name):
    if (type(name) is not str or not name or len(name)>768 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/' for c in name)
            or PurePosixPath(name).is_absolute() or str(PurePosixPath(name))!=name
            or any(p in ('.','..') or len(p)>128 for p in PurePosixPath(name).parts)):
        raise ValueError('Canonical bounded private relative name required.')
    return name


class Deadline(TimeoutError):pass
class Stopped(TimeoutError):pass


class Alarms:
    """Default kernel hard stop plus a best-effort main-thread soft refusal.

    Python handlers are not an instantaneous kernel/C-extension stop guarantee.
    Tests exercise this class with fake signal/time/exit APIs only.
    """
    def __init__(self,*,clock=time.monotonic):
        if threading.current_thread() is not threading.main_thread():raise ValueError('Outer timers require the main thread.')
        self.clock=clock;self.owner=os.getpid();self.phase=None;self.hard=None;self.soft_deadline=None;self.last=clock();self.closed=False;self.waker=None;self.generation=0
        self.previous={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGHUP,signal.SIGTERM,signal.SIGUSR1,signal.SIGINT)}
        self.previous_timer=signal.getitimer(signal.ITIMER_REAL)
        if self.previous_timer!=(0.0,0.0):raise ValueError('Fresh outer process must not replace an existing timer.')
    def moment(self):
        now=self.clock()
        if self.closed or os.getpid()!=self.owner or not finite(now) or now<self.last:raise ValueError('Owned continuing outer clock required.')
        self.last=now;return now
    def check(self):
        now=self.moment()
        if self.soft_deadline is not None and now>=self.soft_deadline:self.handler(signal.SIGALRM,None)
        return now
    def cancel_wake(self):
        self.generation+=1
        if self.waker is not None:self.waker.cancel();self.waker=None
    def handler(self,number,frame):
        now=self.clock()
        if number==signal.SIGUSR1 and now<self.soft_deadline:return # stale earlier-phase pending notification
        self.cancel_wake()
        signal.signal(signal.SIGALRM,signal.SIG_DFL)
        remaining=min(2,self.hard-now)
        if remaining<=0:os._exit(124)
        signal.setitimer(signal.ITIMER_REAL,remaining)
        if number in (signal.SIGALRM,signal.SIGUSR1):raise Deadline('Original '+str(self.phase)+' deadline reached; no reset.')
        raise Stopped('Attended connection was interrupted; stop owned work and preserve evidence.')
    def arm(self,deadline,*,hard_deadline,phase):
        now=self.moment()
        if not finite(deadline) or not finite(hard_deadline) or not now<deadline<=hard_deadline:
            raise Deadline('Original outer deadline is already exhausted.')
        self.cancel_wake()
        self.phase=phase;self.hard=hard_deadline;self.soft_deadline=deadline
        for s in (signal.SIGHUP,signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(s,self.handler)
        # The hard alarm never depends on a Python callback getting the GIL.
        signal.signal(signal.SIGALRM,signal.SIG_DFL)
        current=self.moment()
        if current>=deadline:raise Deadline('Original phase exhausted while binding its timers.')
        signal.setitimer(signal.ITIMER_REAL,hard_deadline-current)
        token=self.generation
        def wake():
            if not self.closed and token==self.generation:
                _thread.interrupt_main(signal.SIGUSR1) # schedules Python handling; sends no OS/PID signal
        self.waker=threading.Timer(deadline-current,wake);self.waker.daemon=True;self.waker.start()
    def close(self):
        if self.closed:return
        self.cancel_wake()
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in self.previous.items():signal.signal(s,h)
        self.closed=True


def read_exact(stream,size,check):
    if type(size) is not int or size<0 or size>3*GiB:raise ValueError('Bounded transport extent required.')
    out=bytearray()
    while len(out)<size:
        check();block=stream.read(min(MiB,size-len(out)))
        if type(block) is not bytes or not 0<len(block)<=min(MiB,size-len(out)):raise ValueError('Empty/truncated outer transport; preserve all partial output.')
        out.extend(block);check()
    return bytes(out)

def write_all(stream,data,check):
    offset=0
    while offset<len(data):
        check();n=stream.write(data[offset:])
        if type(n) is not int or not 0<n<=len(data)-offset:raise OSError('Incomplete outer transport write.')
        offset+=n;check()

def send_frame(stream,value,check,*,cap=64*1024):
    data=encoded(value)
    if not 0<len(data)<=cap:raise ValueError('Bounded outer metadata frame required.')
    write_all(stream,FRAME+struct.pack('!I',len(data))+data,check);stream.flush();check()

def frame(stream,check,*,cap=64*1024):
    if read_exact(stream,len(FRAME),check)!=FRAME:raise ValueError('Wrong outer transport framing; no implicit retry or resynchronization.')
    count=struct.unpack('!I',read_exact(stream,4,check))[0]
    if not 0<count<=cap:raise ValueError('Outer metadata frame too large.')
    data=read_exact(stream,count,check);value=parsed(data)
    if encoded(value)!=data:raise ValueError('Canonical finite outer metadata required.')
    return value


def validate_control(raw,code_sha256):
    if type(raw) is not bytes or len(raw)>256*1024:raise ValueError('Bounded outer control required.')
    c=parsed(raw)
    keys={'kind','supervised_control_text','supervised_control_sha256','outer_source_sha256','local_source_sha256',
          'runtime_files','repository','local_handoff','packet_path','known_hosts_text','known_hosts_sha256',
          'expected_ssh_fingerprint','financial_state_unchanged','authorized_at_preparation','target_helper_reviewed'}
    if (type(c) is not dict or set(c)!=keys or c['kind']!='pgd-publication-outer-control-v1' or encoded(c)!=raw
            or not sha(code_sha256) or c['outer_source_sha256']!=code_sha256 or c['packet_path']!=PACKET_PATH
            or c['known_hosts_text']!=KNOWN_HOSTS or c['known_hosts_sha256']!=digest(KNOWN_HOSTS.encode())
            or c['expected_ssh_fingerprint']!=FINGERPRINT or c['financial_state_unchanged'] is not True
            or c['authorized_at_preparation'] is not False or c['target_helper_reviewed'] is not False):
        raise ValueError('Exact new outer control/source/host binding required.')
    if (type(c['local_source_sha256']) is not dict or set(c['local_source_sha256'])!=set(LOCAL_SOURCES)
            or any(not sha(v) for v in c['local_source_sha256'].values()) or c['local_source_sha256'][LOCAL_SOURCES[0]]!=code_sha256
            or c['local_source_sha256']['models/predictor_pgd_split_directories_v2.py']!=DIRECTORY_POLICY_SHA
            or c['local_source_sha256']['models/predictor_pgd_split_preclaim_refusal_v4.py']!=REFUSAL_POLICY_SHA
            or c['local_source_sha256']['models/predictor_pgd_split_outer_v3.py']!=FROZEN_V3_SHA):
        raise ValueError('Exact local driver source coverage required.')
    for key in ('repository','local_handoff'):
        value=c[key]
        if type(value) is not str or not value or not Path(value).is_absolute() or str(Path(value))!=value or '..' in Path(value).parts:
            raise ValueError('Exact canonical local handoff/repository identity required.')
    text=c['supervised_control_text']
    if type(text) is not str or len(text.encode())>256*1024 or digest(text.encode())!=c['supervised_control_sha256']:
        raise ValueError('Supervised control identity changed.')
    b=parsed(text);p=parsed(b['policy_text'])
    if digest(b['policy_text'].encode())!=POLICY_SHA or b['policy_sha256']!=POLICY_SHA:raise ValueError('Frozen work/host/resource policy changed.')
    contract=parsed(p['scratch_contract_text'])
    if c['runtime_files']!=contract['runtime_files'] or len(c['runtime_files'])!=63:raise ValueError('Frozen runtime sources changed.')
    if b['supervisor_sources']['bootstrap.py']['sha256']!=BOOT_SHA or b['supervisor_sources']['predictor_pgd_split_collection_v1.py']['sha256']!=COLLECTOR_SHA:
        raise ValueError('Frozen initial bootstrap/collector changed.')
    return c,b,p,digest(raw)


def authorization(c,csha,basis):
    if type(basis) is not str or not 1<=len(basis)<=8192:raise ValueError('One explicit exact scoped user basis required.')
    return {'kind':'pgd-publication-outer-university-exception-v1','approved':True,'outer_control_sha256':csha,
        'supervised_control_sha256':c['supervised_control_sha256'],'maximum_new_attempts':1,
        'work_seconds':2400,'collection_seconds':600,'connection_including_login_seconds':3000,
        'full_model_gradients':8,'extra_component_VJPs':2,'automatic_retries':0,'paid_compute':False,
        'cleanup':False,'reset_old_claims':False,'private_key_upload':False,'held_out_pixels':False,'user_basis':basis}

def validate_authorization(raw,c,csha):
    if type(raw) is not bytes or len(raw)>64*1024:raise ValueError('Bounded new outer authorization required.')
    a=parsed(raw)
    if type(a) is not dict or a!=authorization(c,csha,a.get('user_basis')) or encoded(a)!=raw:
        raise ValueError('A new exact one-attempt outer grant is required; old grants are consumed.')
    return a


def decode_sources(mapping,c,b):
    if type(mapping) is not dict or len(mapping)!=69:raise ValueError('All 69 exact source files required.')
    descriptors={'supervisor/'+n:d for n,d in b['supervisor_sources'].items()}
    descriptors.update({'overlay/'+n:d for n,d in c['runtime_files'].items()})
    if set(mapping)!=set(descriptors):raise ValueError('Source namespace changed.')
    out={};total=0
    for name,value in mapping.items():
        safe(name)
        if type(value) is not str or len(value)>2*MiB:raise ValueError('Bounded source representation required.')
        data=base64.b64decode(value,validate=True);want=descriptors[name];total+=len(data)
        if len(data)!=want['bytes'] or digest(data)!=want['sha256'] or total>2*MiB:raise ValueError('Pinned source bytes/extent changed.')
        compile(data,'<CHECKED-UNEXECUTED-'+name+'>','exec');out[name]=data
    return out


def boot_namespace(sources):
    data=sources['supervisor/bootstrap.py']
    if digest(data)!=BOOT_SHA:raise ValueError('Original stdlib bootstrap required before imports.')
    namespace={'__name__':'_pgd_outer_checked_bootstrap','__file__':'<checked-stdin-bootstrap>'}
    exec(compile(data,namespace['__file__'],'exec'),namespace)
    return SimpleNamespace(**namespace)


def remote_root(csha):
    if not sha(csha):raise ValueError('Exact outer attempt identity required.')
    return Path('/var/tmp')/('vjepa-pgd-split-v1-'+str(os.getuid())+'-'+csha)


def preflight_parent():
    if (platform.node().split('.')[0],platform.system(),platform.python_version())!=('linux10614','Linux','3.11.17'):
        raise ValueError('The exact university host/runtime is required before any target write.')
    p=Path('/var/tmp');info=p.lstat()
    if (not stat.S_ISDIR(info.st_mode) or (info.st_dev,info.st_uid,stat.S_IMODE(info.st_mode))!=(66307,0,0o1777)
            or p.resolve(strict=True)!=p or Path.home().stat().st_dev==info.st_dev):
        raise ValueError('Original separate local scratch device/parent changed.')
    with Path('/proc/self/mountinfo').open('rb') as stream:data=stream.read(2*MiB+1)
    if len(data)>2*MiB:raise ValueError('Bounded mount metadata required.')
    matches=[]
    for row in data.decode().splitlines():
        left,sep,right=row.partition(' - ');a=left.split();b=right.split()
        if sep and len(a)>=6 and len(b)>=3:
            point=a[4]
            if str(p)==point or str(p).startswith(point.rstrip('/')+'/'):matches.append((point,b[0],a[5].split(',')+b[2].split(',')))
    if not matches:raise ValueError('No original scratch mount.')
    point,fs,options=max(matches,key=lambda row:len(row[0]))
    if (point,fs)!=('/','xfs') or 'rw' not in options:raise ValueError('Original writable local XFS mount required.')


def directory_id(info):return (info.st_dev,info.st_ino,info.st_uid,stat.S_IMODE(info.st_mode))
def private_write(root,name,data,check):
    safe(name);check()
    if type(data) is not bytes or len(data)>2*MiB:raise ValueError('Bounded source/metadata publication required.')
    root=Path(root);identity=directory_id(root.lstat());fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        if identity[2:]!=(os.getuid(),0o700) or directory_id(os.fstat(fd))!=identity:raise ValueError('Private setup root required.')
        parts=PurePosixPath(name).parts
        for part in parts[:-1]:
            try:os.mkdir(part,0o700,dir_fd=fd)
            except FileExistsError:pass
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);s=os.fstat(nxt)
            if (s.st_dev,s.st_uid,stat.S_IMODE(s.st_mode))!=(identity[0],os.getuid(),0o700):os.close(nxt);raise ValueError('Private setup ancestor required.')
            os.close(fd);fd=nxt
        with os.fdopen(os.open(parts[-1],os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd),'wb') as out:
            write_all(out,data,check);out.flush();os.fsync(out.fileno())
        os.fsync(fd)
        if directory_id(root.lstat())!=identity:raise ValueError('Named setup root changed.')
        check()
    finally:os.close(fd)


def receive_packet(incoming,root,spec,check):
    if spec['bytes']!=17192095 or spec['sha256']!='2278bce2b519943c0f83a49c556bb73dd928fe880fba0f6c56de2a4c3c2ec0da':
        raise ValueError('The unchanged input packet is required.')
    path=Path(root)/'original-packet.tar.gz.partial';h=hashlib.sha256();left=spec['bytes']
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as out:
        while left:
            data=read_exact(incoming,min(MiB,left),check);write_all(out,data,check);h.update(data);left-=len(data)
        out.flush();os.fsync(out.fileno())
    if h.hexdigest()!=spec['sha256']:raise ValueError('Input packet hash changed; retain the partial.')
    os.link(path,path.with_name('original-packet.tar.gz'));os.unlink(path)
    if incoming.read(1)!=b'':raise ValueError('Unexpected bytes after the single approved packet.')
    check()


def extract_packet(root,files,check):
    if len(files)!=99 or sum(w['bytes'] for w in files.values())!=26446958:raise ValueError('Original packet member extent required.')
    seen=set()
    with tarfile.open(Path(root)/'original-packet.tar.gz',mode='r:gz') as archive:
        for item in archive:
            check();name=safe(item.name)
            if name in seen or name not in files or not item.isfile() or item.size!=files[name]['bytes']:
                raise ValueError('Unexpected, duplicate or nonregular packet member.')
            with archive.extractfile(item) as source:data=read_exact(source,item.size,check)
            if digest(data)!=files[name]['sha256']:raise ValueError('Pinned packet member changed.')
            # The largest original member is below 16 MiB. Chunked input files
            # are installed explicitly; private_write's 2 MiB source cap stays.
            destination=Path(root)/'code'/name;destination.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            q=destination.parent
            while q!=Path(root):
                if q.is_symlink() or stat.S_IMODE(q.stat().st_mode)!=0o700 or q.stat().st_uid!=os.getuid():raise ValueError('Private packet ancestry required.')
                q=q.parent
            with os.fdopen(os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as out:
                write_all(out,data,check);out.flush();os.fsync(out.fileno())
            seen.add(name)
    if seen!=set(files):raise ValueError('Incomplete original packet; no proof or model work.')
    check()


_DIRECTORY_SHELL = {'.', 'capture', 'capture/results', 'capture/logs', 'cache', 'cache/cuda', 'tmp'}
_DIRECTORY_WORKER_CLAIMS = {'numerical': 'capture/results/numerical', 'boundary': 'capture/results/boundary', 'native': 'capture/results/components/native', 'forward-preserving': 'capture/results/components/forward-preserving', **{'objective-' + str(i): 'capture/results/numerical/objective-' + str(i) for i in range(8)}}

def _setup_directory_plan(layout):
    """Select only source ancestors and supervisor-owned shell directories.

    The caller must first construct the original verified Layout and enforce the
    fresh root/source/host/authorization gates. Allowed result names are a
    validation catalogue; they do not transfer directory ownership to setup.
    """
    if type(layout.assets) is not dict or len(layout.assets) != 172 or type(layout.result) is not dict or (len(layout.result) != 419) or (type(layout.directories) is not set):
        raise ValueError('Original full source/result catalogue required.')
    selected = set(_DIRECTORY_SHELL)
    for name in layout.assets:
        path = PurePosixPath(name)
        if not name or path.is_absolute() or str(path) != name or any((p in ('.', '..') for p in path.parts)) or name.startswith('capture/'):
            raise ValueError('Only original source assets may contribute setup ancestors.')
        selected.update((str(parent) for parent in path.parents))
    if any((n.startswith('capture/results/') for n in selected)):
        raise ValueError('Scientific result subtrees belong to their workers, never setup.')
    for name in selected:
        layout.directory(name)
    if not set(_DIRECTORY_WORKER_CLAIMS.values()) <= layout.directories:
        raise ValueError('Original worker-owned namespace changed.')
    return {'kind': 'pgd-split-minimal-setup-directory-plan-v2', 'setup_directories': sorted(selected, key=lambda n: (len(PurePosixPath(n).parts), n)), 'worker_claim_directories': dict(_DIRECTORY_WORKER_CLAIMS), 'allowed_but_not_precreated': sorted(layout.directories - selected), 'authorized': False, 'target_helper_ready': False, 'scientific_qualified': False, 'training_admitted': False}

def prepare_tree(root,sources,c,b,p,auth,outer_raw,outer_auth,boot,incoming,check):
    supervised=c['supervised_control_text'].encode();b,policy,bsha=boot.control(supervised)
    derived=boot.encoded({'kind':'pgd-split-supervised-university-exception-v1','approved':True,'control_sha256':bsha,
        'maximum_new_attempts':1,'user_basis':auth['user_basis']})
    boot.authorization(derived,bsha);ib,isha=boot.installation(b,bsha)
    for name,data in sorted(sources.items()):private_write(root,name,data,check)
    private_write(root,'control.json',supervised,check);private_write(root,'authorization.json',derived,check)
    private_write(root,'installation.json',ib,check)
    private_write(root,'configs/base.json',p['base_text'].encode(),check);private_write(root,'configs/design.json',p['design_text'].encode(),check)
    private_write(root,'proposal.json',encoded({'kind':'pgd-split-outer-ATTEMPT-v1','consumed':True,'outer_control':parsed(outer_raw),
        'outer_authorization':parsed(outer_auth),'outer_control_sha256':digest(outer_raw),'outer_authorization_sha256':digest(outer_auth),**FLAGS}),check)
    binding={'root':str(root),'root_identity':list(directory_id(root.lstat())),
             'parent':str(root.parent),'parent_identity':list(directory_id(root.parent.lstat()))}
    finder=boot.PinnedImporter(root,binding,b,p).install()
    from models import predictor_pgd_split_scratch_v1 as scratch
    from models import predictor_pgd_split_coordinator_v1 as coordinator
    layout=scratch.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),ib,isha)
    # Catalogue membership permits a path; the worker still owns its creation.
    directory_plan=_setup_directory_plan(layout)
    for name in directory_plan['setup_directories']:
        if name=='.':continue
        q=root/name
        try:q.mkdir(mode=0o700)
        except FileExistsError:pass
        info=q.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode)!=0o700 or info.st_uid!=os.getuid() or info.st_dev!=binding['root_identity'][0]:
            raise ValueError('Private declared setup directory required.')
    private_write(root,'runtime-source-manifest.json',encoded({'kind':'pgd-split-installed-source-list-v1',
        'files':{n:{'bytes':len(data),'sha256':digest(data)} for n,data in sources.items()},
        'outer_source_sha256':c['outer_source_sha256'],'setup_directory_plan':directory_plan}),check)
    guard=scratch.Guard(root,binding,layout)
    before,_,_=guard.scan(phase='setup',initial=True,settled=True)
    private_write(root,'storage-before.json',encoded(before),check)
    coordinator.target_gate(root,binding);check()
    # This is only the old 16 MiB emergency extent, never a substitute proof.
    fd=os.open(root/'emergency-space.bin',os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        os.posix_fallocate(fd,0,16*MiB);os.fsync(fd);info=os.fstat(fd)
        if info.st_size!=16*MiB or info.st_blocks*512<16*MiB:raise ValueError('Full allocated emergency space required; no sparse fallback.')
    finally:os.close(fd)
    contract=parsed(p['scratch_contract_text'])
    receive_packet(incoming,root,contract['packet']['archive'],check)
    extract_packet(root,contract['packet']['files'],check)
    guard.verify_assets(phase='setup');guard.proof_readiness();check()
    return coordinator,scratch,guard,binding,derived


class OutputBudget:
    def __init__(self,out,check):self.out=out;self.count=0;self.check=check
    def write(self,data):
        self.check()
        if self.count+len(data)>3*GiB:raise ValueError('Whole same-SSH response exceeds 3 GiB.')
        n=self.out.write(data)
        if type(n) is not int or not 0<n<=len(data):raise OSError('Incomplete same-SSH response.')
        self.count+=n;self.check();return n
    def flush(self):self.out.flush();self.check()


def remote_session(incoming, outgoing, *, code_sha256):
    """One future SSH connection. No alternate host, source port or retry mode."""
    alarms = Alarms()
    hello = time.monotonic()
    nonce = os.urandom(32).hex()
    origin = None
    owner = None
    root = None
    streaming = False
    alarms.arm(hello + 180, hard_deadline=hello + 180, phase='handshake')
    output = OutputBudget(outgoing, alarms.check)
    try:
        send_frame(output, {'kind': 'pgd-split-hello-v1', 'nonce': nonce, 'hello_monotonic': hello, 'code_sha256': code_sha256}, alarms.check)
        message = frame(incoming, alarms.check, cap=2 * MiB)
        received = time.monotonic()
        if type(message) is not dict or set(message) != {'nonce', 'client_elapsed', 'control_text', 'authorization_text', 'sources'} or message['nonce'] != nonce or (not finite(message['client_elapsed'])) or (not 0 <= message['client_elapsed'] < 180):
            raise ValueError('One fresh handshake with original elapsed client time required.')
        origin = hello - message['client_elapsed']
        alarms.arm(origin + 180, hard_deadline=origin + 180, phase='setup')
        raw = message['control_text'].encode()
        araw = message['authorization_text'].encode()
        c, b, p, csha = validate_control(raw, code_sha256)
        a = validate_authorization(araw, c, csha)
        sources = decode_sources(message['sources'], c, b)
        boot = boot_namespace(sources)
        boot.control(c['supervised_control_text'].encode())
        preflight_parent()
        alarms.check()
        root = remote_root(csha)
        root.mkdir(mode=448)
        send_frame(output, {'kind': 'pgd-split-root-claimed-v1', 'outer_control_sha256': csha, 'scratch': str(root), 'hello_monotonic': hello, 'handshake_received': received, 'connection_started': origin, 'work_deadline': origin + 2400, 'connection_deadline': origin + 3000, **FLAGS}, alarms.check)
        coordinator, scratch, guard, binding, derived = prepare_tree(root, sources, c, b, p, a, raw, araw, boot, incoming, alarms.check)
        alarms.arm(origin + 2280, hard_deadline=origin + 2400, phase='work')
        owner = coordinator.OuterCoordinator(root, binding, c['supervised_control_text'].encode(), derived, connection_started=origin, outer_guard=guard, outer_alarm=alarms)
        work_error = None
        try:
            owner.run_work()
        except BaseException as error:
            owner.driver.assert_idle()
            work_error = error
        owner.driver.assert_idle()
        if not owner.claimed or owner.driver.guard is None:
            if work_error is not None:
                raise work_error
            raise RuntimeError('Parent session was not initialized; no collection permitted.')
        alarms.arm(min(origin + 2400, time.monotonic() + 120), hard_deadline=origin + 2400, phase='finalization')
        if work_error is None:
            try:
                owner.driver.outer_journal.checkpoint('finalization')
            except BaseException as error:
                work_error = error
                owner.state.update(phase='failed', error_type=type(error).__name__, error=str(error)[:1024])
                owner.driver.save(owner.state, replace=True)
        owner.driver.outer_journal.close()
        from models import predictor_pgd_split_collection_v1 as collection
        collector = collection.Collector(owner.driver.guard, c['supervised_control_text'].encode(), derived, connection_started=origin)
        collector.prepare()
        alarms.arm(collector.window.deadline, hard_deadline=collector.window.deadline, phase='collection')
        send_frame(output, {'kind': 'pgd-split-collection-ready-v1', 'outer_control_sha256': csha, 'collection_started': collector.window.started, 'collection_deadline': collector.window.deadline, 'connection_started': origin, 'handshake_received': received, **FLAGS}, alarms.check)
        streaming = True
        collector.stream(output)
        return 0 if work_error is None else 1
    except BaseException as error:
        if owner is not None:
            owner.driver.assert_idle()
            owner.driver.outer_journal.close()
        if not streaming:
            bound = origin + 3000 if origin is not None else hello + 180
            alarms.arm(min(bound, time.monotonic() + 2), hard_deadline=bound, phase='refusal')
            send_frame(output, {'kind': 'pgd-split-outer-refusal-v1', 'error_type': type(error).__name__, 'message': str(error)[:1024], 'scratch': None if root is None else str(root), 'new_work_started_after_error': False, 'last_guard_refusal': None if owner is None else owner.driver.outer_journal.last_refusal, **FLAGS}, alarms.check)
        else:
            sys.stderr.write('Bounded evidence stream interrupted: ' + type(error).__name__ + '; preserve partial local files.\n')
        return 1
    finally:
        alarms.close()



def verify_directory_policy_sources(root,source):
    """Cold source check only; the stdin module has no new project import."""
    helper=source.read_source(root/'models/predictor_pgd_split_directories_v2.py',cap=2*MiB,sha256=DIRECTORY_POLICY_SHA)
    current=source.read_source(root/'models/predictor_pgd_publication_outer_v1.py',cap=2*MiB)
    expected=ast.parse(helper);actual=ast.parse(current)
    wanted=[n for n in expected.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef)]
    found=[n for n in actual.body if isinstance(n,ast.FunctionDef) and n.name=='_setup_directory_plan'
        or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('_DIRECTORY_SHELL','_DIRECTORY_WORKER_CLAIMS') for t in n.targets)]
    class Restore(ast.NodeTransformer):
        def visit_Name(self,n):
            n.id={'_DIRECTORY_SHELL':'SHELL','_DIRECTORY_WORKER_CLAIMS':'WORKER_CLAIMS'}.get(n.id,n.id)
            return n
        def visit_FunctionDef(self,n):
            if n.name=='_setup_directory_plan':n.name='plan'
            return self.generic_visit(n)
    restored=Restore().visit(ast.Module(body=found,type_ignores=[]))
    if ast.dump(restored,include_attributes=False)!=ast.dump(ast.Module(body=wanted,type_ignores=[]),include_attributes=False):
        raise ValueError('Inline minimal-directory policy differs from the reviewed pure source.')


def verify_refusal_policy_sources(root,source):
    """Cold whole-function check against the pinned reviewed v3 recipe."""
    from models import predictor_pgd_split_preclaim_refusal_v4 as refusal
    source.read_source(root/'models/predictor_pgd_split_preclaim_refusal_v4.py',cap=2*MiB,sha256=REFUSAL_POLICY_SHA)
    old=source.read_source(root/'models/predictor_pgd_split_outer_v3.py',cap=2*MiB,sha256=FROZEN_V3_SHA)
    expected,metadata=refusal.build(old)
    current=source.read_source(root/'models/predictor_pgd_publication_outer_v1.py',cap=2*MiB)
    actual=next(n for n in ast.parse(current).body if isinstance(n,ast.FunctionDef) and n.name=='remote_session')
    if ast.dump(actual,include_attributes=False)!=ast.dump(ast.parse(expected).body[0],include_attributes=False):
        raise ValueError('Outer refusal route differs from the exact reviewed whole-function correction.')


def prepare_package(repository,destination):
    """Source-only exact 10609 preparation; no auth, packet data or target work."""
    from models import predictor_pgd_publication_handoff_v1 as binding
    return binding.prepare_package(repository,destination)
