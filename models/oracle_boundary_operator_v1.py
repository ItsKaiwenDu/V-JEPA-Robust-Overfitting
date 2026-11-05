"""Standard-library outer bound for one CUDA-setup / CPU-return capture."""
import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time

from models import cpu_oracle_scratch as old
from models import cpu_oracle_control_v1 as primitives

KIND='oracle-boundary-capture-v1'
AUTH_KIND='oracle-boundary-resource-exception-v1'
PREFIX='vjepa-oracle-boundary-v1-20261104-'
HANDOFF_SHA='61158d9b148ad32741d4cfaa7ec2862671b4ee5e534c0b9e1b26f611ec018b37'
ARCHIVE_SHA='51c4e4b71ec292afc65ee3bd53c5556c6b538056d607244aef4945ed5fcded9a'
CONTEXT_SHA='23cfb997d0d3b73ff01f6c2125f2156215ac96e77bff21dfc056fb18917707f0'
CONTROL_REVIEW_SHA='98764bb3fcc4072f07a385cfe21339ec3a12a2b4662aa7f8362b45f829faf420'
SSH_FINGERPRINT='SHA256:ARgVRWOzRLA1LmAANxD6Ya17EIIo2sa4ZbHvkeqzgKQ'
PREREQUISITE={'context_comparison_sha256':CONTEXT_SHA,'accepted_control_review_sha256':CONTROL_REVIEW_SHA,
    'historical_discrepancy_resolved':False,'new_attempt_authorized':False}
GIB=2**30;MIB=2**20
POLICY={**primitives.POLICY,'worker_seconds':120,'max_input_bytes':128*MIB,'max_code_bytes':128*MIB,'max_raw_bytes':256*MIB}
REMOTE_NAMES=('models/__init__.py','models/cpu_oracle_scratch.py','models/cpu_oracle_control_prerequisite.py',
    'models/cpu_oracle_control_v1.py','models/scratch_allocation_v2.py','models/oracle_trace.py',
    'models/oracle_boundary_observer_v1.py','models/oracle_boundary_operator_v1.py','scripts/capture_oracle_boundary_v1.py')
LOCAL_NAMES=set(REMOTE_NAMES)-{'models/__init__.py'}|{'models/oracle_boundary_client_v1.py','scripts/run_oracle_boundary_v1.py'}
OVERLAY_INIT=primitives.OVERLAY_INIT


def validate_control(c):
    required={'kind':KIND,'host':'linux10603','role':'pre-SDPA-original-math-label-0','parent':'/var/tmp',
        'expected_filesystem':'xfs','expected_device':66307,'expected_mount':'/',
        'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f',
        'expected_driver':'580.178.04','expected_ssh_fingerprint':SSH_FINGERPRINT,'archive_sha256':ARCHIVE_SHA,'archive_bytes':85622679,
        'handoff_sha256':HANDOFF_SHA,'policy':POLICY,'prerequisite':PREREQUISITE,
        'CUDA_setup_allowed':True,'GPU_SDPA_allowed':False,'paid_compute':False,'research_admitted':False,
        'original_refusals_waived':False,'reuse_old_grant':False,'authorization_present_at_preparation':False,
        'tasks':[{'case':'encoder-1','arm_label':'math','replay':0,'original_oracle_calls':1}]}
    if any(type(c.get(k)) is not type(v) or c.get(k)!=v for k,v in required.items()):
        raise ValueError('Changed boundary host/domain, source, task or resource bound.')
    if set(c['remote_source_sha256'])!=set(REMOTE_NAMES) or set(c['local_source_sha256'])!=LOCAL_NAMES:
        raise ValueError('Boundary source coverage changed.')
    spec=c['archive_manifest']
    if spec['sha256']!=ARCHIVE_SHA or spec['bytes']!=85622679 or len(spec['files'])!=139:
        raise ValueError('Require the complete unchanged original fixed-input packet.')
    if sum(v['bytes'] for v in spec['files'].values())!=90901652:raise ValueError('Original packet extent changed.')
    for name,item in spec['files'].items():
        old.safe_name(name)
        if type(item['bytes']) is not int or item['bytes']<0 or len(item['sha256'])!=64:raise ValueError('Invalid fixed packet member.')
    pins=c['prior_bindings_sha256']
    if (pins.get('report/checks/2026-11-04-cpu-oracle-context-comparison.json')!=CONTEXT_SHA
            or pins.get('report/checks/2026-11-04-cpu-oracle-control-v1-review.json')!=CONTROL_REVIEW_SHA
            or pins.get('runs/pretrained/diagnostic/cpu-oracle-control-v1-20261104/linux10609/ATTEMPT.json')
            !='6ed5c4941a3eba87cd70a5aa5381234acfc494b77f3044b8d42a866630996b79'):
        raise ValueError('Missing preserved review or consumed control claim.')


def validate_authorization(a,sha):
    expected={'kind':AUTH_KIND,'approved':True,'control_sha256':sha,'host':'linux10603','maximum_new_attempts':1,
        'max_seconds':600,'worker_seconds':120,'original_oracle_calls':1,'CUDA_setup_allowed':True,
        'GPU_SDPA_allowed':False,'write_test_bytes':2684354560,'setup_ceiling':3*GIB,
        'capture_ceiling':9*GIB//4,'collection_seconds':600,'paid_compute':False,
        'reset_old_claims':False,'automatic_retries':0}
    if set(a)!={*expected,'user_basis'} or any(type(a[k]) is not type(v) or a[k]!=v for k,v in expected.items()):
        raise ValueError('A separate explicit CUDA-setup boundary resource exception is required.')
    if not isinstance(a['user_basis'],str) or not 1<=len(a['user_basis'])<=8192:raise ValueError('Missing user authorization basis.')


def readiness(c):
    validate_control(c)
    if platform.node()!=c['host'] or sys.platform!='linux':raise ValueError('Wrong boundary host/OS.')
    parent=Path('/var/tmp');info=parent.stat();cap=os.statvfs(parent)
    value={'host':platform.node(),'parent':str(parent),'device':info.st_dev,'inode':info.st_ino,
        'home_device':Path.home().stat().st_dev,'owner_uid':info.st_uid,'mode':stat.S_IMODE(info.st_mode),
        **old.mount_for(parent,Path('/proc/self/mountinfo').read_text()),
        'available_bytes':cap.f_bavail*cap.f_frsize,'available_inodes':cap.f_favail,
        'writable_for_current_user':os.access(parent,os.W_OK|os.X_OK)}
    if parent.is_symlink() or parent.resolve()!=parent:raise ValueError('Scratch parent is aliased.')
    old.validate_readiness(value,c)
    if value['available_bytes']<POLICY['minimum_free_bytes']:raise ValueError('Require 8 GiB scratch space, not a reservation.')
    if os.environ.get('CUDA_VISIBLE_DEVICES') in ('','-1'):raise ValueError('CUDA is hidden; preserve environment and refuse.')
    return value


def gpu_gate(c,allowed_pid=None):
    row=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,driver_version',
        '--format=csv,noheader,nounits'],text=True,timeout=10).strip().splitlines()
    if len(row)!=1:raise ValueError('Exactly one expected GPU is required.')
    name,uuid,total,used,driver=[x.strip() for x in row[0].split(',')]
    if (name,uuid,driver)!=(c['expected_gpu'],c['expected_gpu_uuid'],c['expected_driver']) or float(total)!=8192 or allowed_pid is None and float(used)>512:
        raise ValueError('GPU/driver/initial-memory gate rejected.')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
    if any(int(line.split(',')[0].strip())!=allowed_pid for line in active if line.strip()):
        raise ValueError('Another CUDA compute process is active; leave it alone and stop.')
    return {'observed_unix':time.time(),'GPU_row':row[0],'allowed_compute_pid':allowed_pid,'compute_process_rows':active,
        'sampling_is_not_continuous_exclusivity':True}


def inventory(folder,*,hashing=False,max_bytes=None,running=False):
    """Account actual files; allow only same-directory atomic writer pairs."""
    folder=Path(folder);root=folder.stat();files={};seen=set();physical=0
    for current,dirs,names in os.walk(folder,followlinks=False):
        for name in dirs+names:
            path=Path(current)/name
            try:info=path.lstat()
            except FileNotFoundError:
                if running:continue
                raise
            if info.st_dev!=root.st_dev or stat.S_ISLNK(info.st_mode):raise ValueError('Scratch symlink/device changed.')
            if stat.S_ISDIR(info.st_mode):continue
            paired=False
            if info.st_nlink==2:
                final=name[:-8] if name.endswith('.partial') else name
                if final.endswith(('.json','.npz','.pt')):
                    partner=path.with_name(final if name.endswith('.partial') else final+'.partial')
                    try:
                        other=partner.lstat()
                        paired=stat.S_ISREG(other.st_mode) and (other.st_dev,other.st_ino)==(info.st_dev,info.st_ino)
                    except FileNotFoundError:paired=running and path.stat().st_nlink==1
            if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 and not paired:
                raise ValueError('Unexpected special file or external hardlink.')
            relative=old.safe_name(str(path.relative_to(folder)));item={'bytes':info.st_size,'mtime_ns':info.st_mtime_ns}
            if hashing:
                item['sha256']=old.digest(path);after=path.stat()
                if (info.st_ino,info.st_size,info.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
                    raise ValueError('Evidence changed during hashing.')
            files[relative]=item
            if (info.st_dev,info.st_ino) not in seen:
                seen.add((info.st_dev,info.st_ino));physical+=info.st_blocks*512
            if len(files)>POLICY['max_files'] or max_bytes is not None and sum(v['bytes'] for v in files.values())>max_bytes:
                raise ValueError('Scratch file/byte bound exceeded.')
    return files,physical


def storage_gate(folder,identity,phase='capture'):
    folder=Path(folder);info=folder.lstat();parent=Path('/var/tmp').stat();cap=POLICY['setup_ceiling' if phase=='setup' else 'capture_ceiling']
    if (folder.is_symlink() or not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o700
            or info.st_dev!=identity['device'] or (parent.st_dev,parent.st_ino)!=(identity['device'],identity['inode'])):
        raise ValueError('Scratch path/device/owner changed.')
    files,allocated=inventory(folder,max_bytes=cap,running=True)
    for current,dirs,names in os.walk(folder,followlinks=False):
        d=Path(current).lstat()
        if not stat.S_ISDIR(d.st_mode) or d.st_dev!=info.st_dev:raise ValueError('Directory type/device changed.')
        allocated+=d.st_blocks*512
    space=os.statvfs(folder)
    if allocated>cap or space.f_bavail*space.f_frsize<GIB or space.f_favail<1024:raise ValueError('Observed footprint/capacity bound reached.')
    category={'code':0,'input':0,'raw':0,'proof':0,'metadata':0,'emergency':0}
    for name,item in files.items():
        if name.startswith(('code/','overlay/')):key='code'
        elif name.startswith('original-packet.tar.gz'):key='input'
        elif name=='space-proof-v2.bin.partial':key='proof'
        elif name=='emergency-space.bin':key='emergency'
        elif name.startswith('results/') and name.endswith(('.npz','.npz.partial','.pt','.pt.partial')):
            key='raw'
            if item['bytes']>POLICY['max_array_bytes']:raise ValueError('Individual capture exceeds 32 MiB.')
        else:key='metadata'
        category[key]+=item['bytes']
    limits={'code':POLICY['max_code_bytes'],'input':POLICY['max_input_bytes'],'raw':POLICY['max_raw_bytes'],
        'proof':POLICY['write_test_bytes'] if phase=='setup' else 0,'metadata':POLICY['max_metadata_bytes'],'emergency':POLICY['emergency_bytes']}
    if any(category[k]>limit for k,limit in limits.items()):raise ValueError('Boundary storage category exceeded.')
    return {'logical_bytes':sum(v['bytes'] for v in files.values()),'allocated_bytes':allocated,
        'available_bytes':space.f_bavail*space.f_frsize,'phase':phase,'categories':category,'ceiling':cap,'files':len(files)}


def extract_packet(archive,specification,destination):
    """Extract the full fixed-input packet, never the older CPU-only format."""
    destination=Path(destination)
    if destination.exists() or old.digest(archive)!=specification['sha256']:
        raise ValueError('Archive changed or destination is not fresh.')
    with tarfile.open(archive) as source:
        members=source.getmembers();names=[old.safe_name(m.name) for m in members]
        if (len(names)!=len(set(names)) or set(names)!=set(specification['files']) or len(names)>512
                or any(not m.isfile() for m in members)):
            raise ValueError('Unexpected fixed-input archive member.')
        for member in members:
            item=specification['files'][member.name]
            if member.size!=item['bytes']:raise ValueError('Fixed-input member extent changed.')
            with source.extractfile(member) as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=item['sha256']:raise ValueError('Fixed-input member hash changed.')
        destination.mkdir(mode=0o700)
        for member in members:
            path=destination/member.name;path.parent.mkdir(parents=True,exist_ok=True)
            with source.extractfile(member) as incoming,path.open('xb') as out:
                for block in iter(lambda:incoming.read(MIB),b''):out.write(block)
    descriptor=destination/'FIXED-ATTENTION-HANDOFF.json'
    if old.digest(descriptor)!=specification['files']['FIXED-ATTENTION-HANDOFF.json']['sha256']:
        raise ValueError('Fixed-input manifest changed.')


def capture_status(folder,exit_code):
    path=Path(folder)/'results/boundary.json'
    if not path.exists():return {'capture_outcome':'startup-failure','scope_capture_complete':False,'original_gate_passed':None}
    state=primitives.read_runtime_record(path)
    if (state.get('kind')!='oracle-boundary-observer-v1' or state.get('local_tiny_rehearsal') is not False
            or state.get('original_full_worker_accepted') is not False or state.get('GPU_attention_executed') is not False
            or state.get('GPU_input_transfer_completed') is not False or state.get('detailed_introspection_before_first_oracle') is not False
            or state.get('old_failure_waived') is not False or state.get('research_admitted') is not False
            or state.get('profile_restored') is not True or state.get('trace_restored') is not True):
        raise ValueError('Boundary receipt domain/restoration changed.')
    complete=state['capture_complete'];passed=state['original_gate_passed'];outcome=state['outcome']
    if complete:
        if state['original_oracle_calls']!=1 or state['original_oracle_returns']!=1 or state['forbidden_attention_calls']!=0:
            raise ValueError('Unexpected oracle or attention coverage.')
        if (passed is True)!=(outcome=='captured-gate-pass') or outcome not in ('captured-gate-pass','captured-original-refusal'):
            raise ValueError('Gate outcome contradicts capture.')
        if exit_code!=(0 if passed else 1):raise ValueError('Exit contradicts original gate.')
        artifact=state['computed_return_artifact'];f=path.parent/old.safe_name(artifact['file'])
        if f.name!='computed-oracle.npz' or f.stat().st_size!=artifact['bytes'] or old.digest(f)!=artifact['sha256']:
            raise ValueError('Actual computed return artifact changed.')
        old_path=path.parent/'original-worker/worker.json'
        if old.digest(old_path)!=state['original_worker_sha256']:raise ValueError('Original worker record changed.')
    elif outcome!='technical-failure' or exit_code==0:raise ValueError('Incomplete capture cannot pass.')
    return {'capture_outcome':outcome,'scope_capture_complete':complete,'original_gate_passed':passed,
        'boundary_receipt_sha256':old.digest(path),'original_full_worker_accepted':False}


def result_stream(folder,sha,output):
    files,_=inventory(folder,hashing=True,max_bytes=POLICY['max_collection_bytes'])
    header=old.json_bytes({'kind':KIND,'phase':'collection','control_sha256':sha,'files':files,
        'logical_bytes':sum(v['bytes'] for v in files.values()),'scientific_review_complete':False,'model_restarted_for_collection':False})
    if len(header)>POLICY['max_header_bytes']:raise ValueError('Collection header exceeds bound.')
    output.write(header);output.flush()
    with tarfile.open(mode='w|',fileobj=output) as archive:
        for name,spec in sorted(files.items()):
            path=Path(folder)/name
            if path.is_symlink() or old.digest(path)!=spec['sha256']:raise ValueError('Collection source changed.')
            member=tarfile.TarInfo(name);member.size=spec['bytes'];member.mode=0o600;member.mtime=spec['mtime_ns']//10**9
            with path.open('rb') as f:archive.addfile(member,f)


def run_remote(c,auth,incoming,output,sources):
    validate_control(c);sha=hashlib.sha256(old.json_bytes(c)).hexdigest();validate_authorization(auth,sha)
    began=time.monotonic();unix=time.time();deadline=began+600
    identity=readiness(c);gpu_gate(c)
    folder=Path(tempfile.mkdtemp(prefix=PREFIX+c['host']+'-',dir='/var/tmp'));folder.chmod(0o700)
    output.write(old.json_bytes({'kind':KIND,'phase':'scratch-created','host':c['host'],'control_sha256':sha,'scratch_directory':str(folder),'started_unix':unix}));output.flush()
    state={'kind':KIND,'host':c['host'],'control_sha256':sha,'started_unix':unix,'deadline_unix':unix+600,
        'work_finished':False,'CUDA_setup_allowed':True,'GPU_SDPA_allowed':False,'paid_compute':False,
        'scope_capture_complete':False,'original_gate_passed':None,'capture_outcome':'not-started',
        'original_full_worker_accepted':False,'old_refusals_waived':False,'scientific_review_complete':False}
    previous={s:signal.getsignal(s) for s in (signal.SIGHUP,signal.SIGTERM,signal.SIGALRM)}
    def stop(*unused):raise RuntimeError('Boundary attempt interrupted; no retry.')
    for s in previous:signal.signal(s,stop)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()-5))
    try:
        old.write_json(folder/'operator-control.json',c);old.write_json(folder/'resource-authorization.json',auth)
        old.write_json(folder/'storage-before.json',identity)
        if set(sources)!=set(REMOTE_NAMES) or sum(len(v) for v in sources.values())>POLICY['max_overlay_bytes']:raise ValueError('Wrong overlay scope.')
        for name,data in sources.items():
            if hashlib.sha256(data).hexdigest()!=c['remote_source_sha256'][name]:raise ValueError('Overlay source changed.')
            path=folder/'overlay'/old.safe_name(name);path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        with (folder/'emergency-space.bin').open('xb') as f:f.write(os.urandom(POLICY['emergency_bytes']));f.flush();os.fsync(f.fileno())
        proof_end=min(deadline-5,time.monotonic()+60)
        program='import sys;sys.path.insert(0,'+repr(str(folder/'overlay'))+');from models.scratch_allocation_v2 import bounded_write_proof;bounded_write_proof('+repr(str(folder))+',deadline='+repr(proof_end)+')'
        proof=primitives.supervise([sys.executable,'-B','-u','-c',program],folder,role='proof',deadline=proof_end,
            gate=lambda:storage_gate(folder,identity,'setup'),file_limit=3*GIB)
        if proof['exit_status']!=0:raise RuntimeError('Proof refused before CUDA worker.')
        primitives.validate_proof(folder);state['write_test_complete']=True
        old.receive_archive(incoming,folder/'original-packet.tar.gz',c['archive_bytes'],ARCHIVE_SHA,deadline-5)
        extract_packet(folder/'original-packet.tar.gz',c['archive_manifest'],folder/'code');storage_gate(folder,identity)
        if time.time()-unix>110:raise TimeoutError('Setup exhausted original freshness margin.')
        entry=folder/'overlay/scripts/capture_oracle_boundary_v1.py'
        command=[sys.executable,'-B','-u',str(entry),'--packet',str(folder/'code'),'--scratch',str(folder),
            '--control-sha256',sha,'--launcher-started',str(unix),'--upstream',str(Path.home()/'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting/external/vjepa2')]
        visibility=os.environ.get('CUDA_VISIBLE_DEVICES');state['worker_CUDA_VISIBLE_DEVICES']=visibility
        # The reused supervisor hides CUDA by default. Restore the caller's
        # unset visibility before Python starts; explicit visibility is passed.
        if visibility is None:command=['/usr/bin/env','-u','CUDA_VISIBLE_DEVICES',*command]
        env={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
        if visibility is not None:env['CUDA_VISIBLE_DEVICES']=visibility
        monitoring=[];last_check=[0.0]
        def combined_gate():
            storage=storage_gate(folder,identity)
            if time.monotonic()-last_check[0]>=5:
                allowed=[]
                children=Path('/proc/self/task')/str(os.getpid())/'children'
                for pid in children.read_text().split():
                    try:argv=(Path('/proc')/pid/'cmdline').read_bytes().split(b'\0')
                    except FileNotFoundError:continue
                    if str(entry).encode() in argv:allowed.append(int(pid))
                if len(allowed)>1:raise ValueError('More than one boundary worker is active.')
                monitoring.append(gpu_gate(c,allowed[0] if allowed else None));last_check[0]=time.monotonic()
            return storage
        state['worker_process_started']=True
        try:
            execution=primitives.supervise(command,folder,role='boundary',deadline=min(deadline-5,time.monotonic()+120),
                gate=combined_gate,file_limit=POLICY['max_archive_bytes'],env=env)
        finally:old.write_json(folder/'GPU-monitor.json',{'observations':monitoring,'continuous_exclusivity_proved':False})
        gpu_gate(c);state['worker_exit_status']=execution['exit_status']
        state.update(capture_status(folder,execution['exit_status']))
    except BaseException as exc:
        state.update(error_type=type(exc).__name__,error=str(exc),capture_outcome='technical-failure',scope_capture_complete=False)
        print('Boundary attempt stopped: '+str(exc),file=sys.stderr,flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if (folder/'emergency-space.bin').exists():(folder/'emergency-space.bin').unlink()
        state.update(work_finished=True,wall_seconds=time.monotonic()-began)
        old.write_json(folder/'operator-result.json',state)
    signal.setitimer(signal.ITIMER_REAL,600)
    try:result_stream(folder,sha,output)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,handler in previous.items():signal.signal(s,handler)
    return state


def collect_remote(c,scratch,output):
    validate_control(c);folder=Path(scratch);info=folder.lstat();sha=hashlib.sha256(old.json_bytes(c)).hexdigest()
    if platform.node()!=c['host'] or folder.parent!=Path('/var/tmp') or not folder.name.startswith(PREFIX+c['host']+'-') or folder.is_symlink() or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o700:
        raise ValueError('Wrong finished boundary scratch path.')
    state=json.loads((folder/'operator-result.json').read_text())
    if state.get('control_sha256')!=sha or state.get('work_finished') is not True:raise ValueError('Collection requires the same finished attempt.')
    output.write(old.json_bytes({'kind':KIND,'phase':'scratch-created','host':c['host'],'control_sha256':sha,'scratch_directory':str(folder),'download_only':True}))
    previous=signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('Copy deadline reached.')))
    signal.setitimer(signal.ITIMER_REAL,600)
    try:result_stream(folder,sha,output)
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def remote_main(control64,sources64,mode='run',scratch=None):
    os.umask(0o077);os.environ['PYTHONDONTWRITEBYTECODE']='1'
    c=json.loads(base64.b64decode(control64,validate=True))
    if mode=='collect':return collect_remote(c,scratch,sys.stdout.buffer)
    if mode!='run':raise ValueError('Only one run or finished-file collection is supported.')
    line=sys.stdin.buffer.readline(POLICY['max_header_bytes']+1)
    if len(line)>POLICY['max_header_bytes'] or not line.endswith(b'\n'):raise ValueError('Missing bounded new authorization.')
    sources={n:base64.b64decode(b,validate=True) for n,b in sources64.items()}
    return run_remote(c,json.loads(line),sys.stdin.buffer,sys.stdout.buffer,sources)
