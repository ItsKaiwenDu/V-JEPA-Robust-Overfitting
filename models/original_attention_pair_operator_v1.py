"""A separately authorized, bounded original efficient/math pair; stdlib only."""
import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import stat
import sys
import tarfile
import tempfile
import time

from models import oracle_boundary_operator_v1 as base
from models import original_attention_pair_runtime_v1 as runtime

KIND='original-attention-pair-capture-v1'
AUTH_KIND='original-attention-pair-resource-exception-v1'
PREFIX='vjepa-original-attention-pair-v1-20261104-'
HANDOFF_SHA=base.HANDOFF_SHA;ARCHIVE_SHA=base.ARCHIVE_SHA
PROPOSAL_SHA='a8be90ea0e3c0ebba5702066dd80f10cc6aab2e2a22ec7a4504544ca2f181d05'
REVIEW_SHA='6b9c357b4b8e60ccd1dbf0b84db5a20474cc690593d899961b00718f1a78282c'
CLAIM_SHA='5e06811a7c9aeb781addd3e6f3f90baa85641fe7502619f1f5de1f3e8465cda3'
GIB=2**30;MIB=2**20
POLICY={**base.POLICY,'max_raw_bytes':320*MIB,'max_CPU_return_bytes':16*MIB,'max_GPU_return_bytes':8*MIB,
    'max_workers':2,'max_CPU_oracle_calls':2,'max_attention_iterations':12}
PREREQUISITE={'proposal_sha256':PROPOSAL_SHA,'accepted_boundary_review_sha256':REVIEW_SHA,
    'consumed_boundary_claim_sha256':CLAIM_SHA,'previous_grants_consumed':True,'historical_discrepancy_resolved':False,
    'GPU_attention_qualified':False,'research_admitted':False,'new_attempt_authorized_at_preparation':False}
REMOTE_NAMES=('models/__init__.py','models/cpu_oracle_scratch.py','models/cpu_oracle_control_prerequisite.py',
    'models/cpu_oracle_control_v1.py','models/scratch_allocation_v2.py','models/oracle_trace.py',
    'models/oracle_boundary_observer_v1.py','models/oracle_boundary_operator_v1.py',
    'models/attention_return_capture_v1.py','models/original_attention_pair_observer_v1.py',
    'models/original_attention_pair_runtime_v1.py','models/original_attention_pair_operator_v1.py',
    'scripts/capture_original_attention_pair_v1.py')
LOCAL_NAMES=set(REMOTE_NAMES)-{'models/__init__.py'}|{'models/original_attention_pair_client_v1.py','scripts/run_original_attention_pair_v1.py'}
OVERLAY_INIT=base.OVERLAY_INIT


def validate_control(c):
    expected={'kind':KIND,'host':'linux10603','role':'encoder-1-efficient-math-replay-0-pair',
        'parent':'/var/tmp','expected_filesystem':'xfs','expected_device':66307,'expected_mount':'/',
        'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f',
        'expected_driver':'580.178.04','expected_ssh_fingerprint':base.SSH_FINGERPRINT,
        'archive_sha256':ARCHIVE_SHA,'archive_bytes':85622679,'handoff_sha256':HANDOFF_SHA,
        'policy':POLICY,'prerequisite':PREREQUISITE,'tasks':[runtime.task(i,False) for i in (0,1)],
        'GPU_input_transfer_allowed':True,'GPU_attention_allowed':True,'paid_compute':False,'research_admitted':False,
        'old_refusals_waived':False,'reuse_old_grant':False,'authorization_present_at_preparation':False,
        'timings_usable_for_speed_or_budget':False,'original_four_worker_session_qualified':False}
    if any(type(c.get(k)) is not type(v) or c.get(k)!=v for k,v in expected.items()):
        raise ValueError('Pair host, task, numerical domain or resource scope changed.')
    if set(c['local_source_sha256'])!=LOCAL_NAMES or set(c['remote_source_sha256'])!=set(REMOTE_NAMES):
        raise ValueError('Pair source coverage differs.')
    spec=c['archive_manifest']
    if spec['sha256']!=ARCHIVE_SHA or spec['bytes']!=85622679 or len(spec['files'])!=139 or sum(x['bytes'] for x in spec['files'].values())!=90901652:
        raise ValueError('Require the complete unchanged full fixed-input packet.')
    for name,item in spec['files'].items():
        base.old.safe_name(name)
        if type(item['bytes']) is not int or item['bytes']<0 or len(item['sha256'])!=64:raise ValueError('Invalid original packet member.')
    required={'configs/original_attention_pair_proposal_v1.json':PROPOSAL_SHA,
        'report/checks/2026-11-04-oracle-boundary-v1-target-review.json':REVIEW_SHA,
        'runs/pretrained/diagnostic/oracle-boundary-v1-20261104/linux10603/ATTEMPT.json':CLAIM_SHA}
    if any(c['prior_bindings_sha256'].get(k)!=v for k,v in required.items()):raise ValueError('Pair prerequisite or consumed claim missing.')


def validate_authorization(a,sha):
    expected={'kind':AUTH_KIND,'approved':True,'control_sha256':sha,'host':'linux10603','maximum_new_attempts':1,
        'max_seconds':600,'worker_seconds':120,'max_fresh_workers':2,'max_CPU_oracle_calls':2,'max_attention_iterations':12,
        'GPU_input_transfer_allowed':True,'GPU_attention_allowed':True,'write_test_bytes':2684354560,
        'setup_ceiling':3*GIB,'capture_ceiling':9*GIB//4,'raw_capture_ceiling':320*MIB,
        'collection_seconds':600,'paid_compute':False,'reset_old_claims':False,'automatic_retries':0}
    if set(a)!={*expected,'user_basis'} or any(type(a[k]) is not type(v) or a[k]!=v for k,v in expected.items()):
        raise ValueError('A separate explicit GPU-execution pair resource exception is required.')
    if not isinstance(a['user_basis'],str) or not 1<=len(a['user_basis'])<=8192:raise ValueError('Missing pair approval basis.')


def readiness(c):
    validate_control(c)
    if platform.node()!=c['host'] or sys.platform!='linux':raise ValueError('Wrong pair target host/OS.')
    parent=Path('/var/tmp');info=parent.stat();space=os.statvfs(parent)
    value={'host':platform.node(),'parent':str(parent),'device':info.st_dev,'inode':info.st_ino,
        'home_device':Path.home().stat().st_dev,'owner_uid':info.st_uid,'mode':stat.S_IMODE(info.st_mode),
        **base.old.mount_for(parent,Path('/proc/self/mountinfo').read_text()),
        'available_bytes':space.f_bavail*space.f_frsize,'available_inodes':space.f_favail,
        'writable_for_current_user':os.access(parent,os.W_OK|os.X_OK)}
    if parent.is_symlink() or parent.resolve()!=parent:raise ValueError('Scratch parent alias rejected.')
    base.old.validate_readiness(value,c)
    if value['available_bytes']<8*GIB:raise ValueError('Require 8 GiB initial space; it is not a reservation.')
    if os.environ.get('CUDA_VISIBLE_DEVICES') in ('','-1'):raise ValueError('CUDA is hidden; refuse rather than change caller visibility.')
    return value


def storage_gate(folder,identity,phase='capture'):
    folder=Path(folder);info=folder.lstat();parent=Path('/var/tmp').stat();limit=POLICY['setup_ceiling' if phase=='setup' else 'capture_ceiling']
    if (folder.is_symlink() or not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o700
            or info.st_dev!=identity['device'] or (parent.st_dev,parent.st_ino)!=(identity['device'],identity['inode'])):
        raise ValueError('Scratch path, device or ownership changed.')
    files,allocated=base.inventory(folder,max_bytes=limit,running=True)
    for current,dirs,names in os.walk(folder,followlinks=False):
        d=Path(current).lstat()
        if not stat.S_ISDIR(d.st_mode) or d.st_dev!=info.st_dev:raise ValueError('Scratch directory changed.')
        allocated+=d.st_blocks*512
    space=os.statvfs(folder)
    if allocated>limit or space.f_bavail*space.f_frsize<GIB or space.f_favail<1024:raise ValueError('Observed scratch capacity exceeded.')
    counts={'code':0,'input':0,'raw':0,'proof':0,'metadata':0,'emergency':0}
    for name,item in files.items():
        if name.startswith(('code/','overlay/')):key='code'
        elif name.startswith('original-packet.tar.gz'):key='input'
        elif name=='space-proof-v2.bin.partial':key='proof'
        elif name=='emergency-space.bin':key='emergency'
        elif name.startswith('results/') and name.endswith(('.pt','.pt.partial','.npz','.npz.partial')):
            key='raw'
            if item['bytes']>32*MIB:raise ValueError('Individual tensor file exceeds 32 MiB.')
        else:key='metadata'
        counts[key]+=item['bytes']
    limits={'code':128*MIB,'input':128*MIB,'raw':320*MIB,'proof':POLICY['write_test_bytes'] if phase=='setup' else 0,
        'metadata':64*MIB,'emergency':16*MIB}
    if any(counts[k]>v for k,v in limits.items()):raise ValueError('Pair storage category bound exceeded.')
    return {'phase':phase,'logical_bytes':sum(x['bytes'] for x in files.values()),'allocated_bytes':allocated,
        'available_bytes':space.f_bavail*space.f_frsize,'ceiling':limit,'files':len(files),'categories':counts}


def result_stream(folder,sha,output):
    files,_=base.inventory(folder,hashing=True,max_bytes=3*GIB)
    header=base.old.json_bytes({'kind':KIND,'phase':'collection','control_sha256':sha,'files':files,
        'logical_bytes':sum(x['bytes'] for x in files.values()),'scientific_review_complete':False,'model_restarted_for_collection':False})
    if len(header)>POLICY['max_header_bytes']:raise ValueError('Collection header exceeds bound.')
    output.write(header);output.flush()
    with tarfile.open(mode='w|',fileobj=output) as archive:
        for name,item in sorted(files.items()):
            p=Path(folder)/name
            if p.is_symlink() or base.old.digest(p)!=item['sha256']:raise ValueError('Evidence changed during collection.')
            member=tarfile.TarInfo(name);member.size=item['bytes'];member.mode=0o600;member.mtime=item['mtime_ns']//10**9
            with p.open('rb') as source:archive.addfile(member,source)


def pair_status(folder):
    folder=Path(folder);path=folder/'results/pair.json'
    if not path.exists():return {'pair_checks_passed':False,'pair_run_present':False,'completed_workers':0}
    state=runtime.read_record(path)
    if state['kind']!=runtime.KIND or state['local_tiny_rehearsal'] is not False or state['run_finished'] is not True:
        raise ValueError('Missing finished live pair receipt.')
    if state['planned']!=[runtime.task(i,False) for i in (0,1)] or any(state.get(k) is not False for k in (
            'original_four_worker_session_qualified','performance_qualified','research_admitted','paid_compute','old_refusals_waived')):
        raise ValueError('Pair scope or qualification labels changed.')
    if state['pair_checks_passed']:
        if len(state['completed'])!=2 or 'error' in state:raise ValueError('A partial pair cannot pass.')
        for i,prior in enumerate(state['completed']):
            actual=runtime.worker_status(folder/'results/workers'/runtime.task(i,False)['worker'],local=False)
            if actual!=prior:raise ValueError('Completed worker evidence changed.')
    return {'pair_checks_passed':state['pair_checks_passed'],'pair_run_present':True,'completed_workers':len(state['completed']),
        'pair_receipt_sha256':base.old.digest(path),'original_four_worker_session_qualified':False}


def run_remote(c,auth,incoming,output,sources):
    validate_control(c);sha=hashlib.sha256(base.old.json_bytes(c)).hexdigest();validate_authorization(auth,sha)
    began=time.monotonic();unix=time.time();deadline=began+600;identity=readiness(c);base.gpu_gate(c)
    folder=Path(tempfile.mkdtemp(prefix=PREFIX+c['host']+'-',dir='/var/tmp'));folder.chmod(0o700)
    output.write(base.old.json_bytes({'kind':KIND,'phase':'scratch-created','host':c['host'],'control_sha256':sha,
        'scratch_directory':str(folder),'started_unix':unix}));output.flush()
    state={'kind':KIND,'host':c['host'],'control_sha256':sha,'started_unix':unix,'deadline_unix':unix+600,
        'work_finished':False,'pair_checks_passed':False,'completed_workers':0,'GPU_attention_allowed':True,
        'paid_compute':False,'scientific_review_complete':False,'old_refusals_waived':False,
        'original_four_worker_session_qualified':False,'performance_qualified':False,'research_admitted':False}
    previous={s:signal.getsignal(s) for s in (signal.SIGHUP,signal.SIGTERM,signal.SIGALRM)}
    def interrupted(*unused):raise RuntimeError('Original pair interrupted; no retry or next worker.')
    for s in previous:signal.signal(s,interrupted)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()-5))
    try:
        base.old.write_json(folder/'operator-control.json',c);base.old.write_json(folder/'resource-authorization.json',auth)
        base.old.write_json(folder/'storage-before.json',identity)
        if set(sources)!=set(REMOTE_NAMES) or sum(map(len,sources.values()))>2*MIB:raise ValueError('Overlay scope exceeded.')
        for name,data in sources.items():
            if hashlib.sha256(data).hexdigest()!=c['remote_source_sha256'][name]:raise ValueError('New overlay hash differs.')
            p=folder/'overlay'/base.old.safe_name(name);p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        with (folder/'emergency-space.bin').open('xb') as f:f.write(os.urandom(16*MIB));f.flush();os.fsync(f.fileno())
        proof_end=min(deadline-5,time.monotonic()+60)
        proof_program='import sys;sys.path.insert(0,'+repr(str(folder/'overlay'))+');from models.scratch_allocation_v2 import bounded_write_proof;bounded_write_proof('+repr(str(folder))+',deadline='+repr(proof_end)+')'
        proof=base.primitives.supervise([sys.executable,'-B','-u','-c',proof_program],folder,role='proof',deadline=proof_end,
            gate=lambda:storage_gate(folder,identity,'setup'),file_limit=3*GIB)
        if proof['exit_status']!=0:raise RuntimeError('Allocation proof rejected before scientific dispatch.')
        base.primitives.validate_proof(folder);state['write_test_complete']=True
        base.old.receive_archive(incoming,folder/'original-packet.tar.gz',85622679,ARCHIVE_SHA,deadline-5)
        base.extract_packet(folder/'original-packet.tar.gz',c['archive_manifest'],folder/'code');storage_gate(folder,identity)
        if time.time()-unix>110:raise TimeoutError('Original setup freshness allowance exhausted.')
        runtime.initialize(folder,folder/'code',local=False,control_sha=sha,started=unix,deadline=unix+600)
        for relative in ('tmp','cache','cache/cuda'):(folder/relative).mkdir(mode=0o700,exist_ok=True)
        for index in (0,1):
            monitoring=[{'stage':'before-launch',**base.gpu_gate(c)}];last=[0.0]
            worker,plan=runtime.issue_task(folder,folder/'code',index,local=False,control_sha=sha)
            entry=folder/'overlay/scripts/capture_original_attention_pair_v1.py'
            command=[sys.executable,'-I','-B','-u',str(entry),'--packet',str(folder/'code'),'--scratch',str(folder),
                '--control-sha256',sha,'--task-index',str(index),'--worker-plan-sha256',base.old.digest(worker/'worker-plan.json'),
                '--upstream',str(Path.home()/'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting/external/vjepa2')]
            visibility=os.environ.get('CUDA_VISIBLE_DEVICES')
            if visibility is None:command=['/usr/bin/env','-u','CUDA_VISIBLE_DEVICES',*command]
            env={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','TMPDIR':str(folder/'tmp'),
                'XDG_CACHE_HOME':str(folder/'cache'),'CUDA_CACHE_PATH':str(folder/'cache/cuda'),'CUDA_CACHE_MAXSIZE':str(32*MIB)}
            if visibility is not None:env['CUDA_VISIBLE_DEVICES']=visibility
            def gate():
                sample=storage_gate(folder,identity)
                if time.monotonic()-last[0]>=5:
                    pids=[]
                    for pid in (Path('/proc/self/task')/str(os.getpid())/'children').read_text().split():
                        try:argv=(Path('/proc')/pid/'cmdline').read_bytes().split(b'\0')
                        except FileNotFoundError:continue
                        if str(entry).encode() in argv:pids.append(int(pid))
                    if len(pids)>1:raise ValueError('More than one scientific child is active.')
                    monitoring.append({'stage':'during',**base.gpu_gate(c,pids[0] if pids else None)});last[0]=time.monotonic()
                return sample
            state['workers_started']=index+1
            try:
                execution=base.primitives.supervise(command,folder,role='worker-'+str(index),
                    deadline=min(deadline-5,time.monotonic()+runtime.read_record(worker/'authorization.json')['deadline_unix']-time.time()),
                    gate=gate,file_limit=POLICY['max_archive_bytes'],env=env)
                monitoring.append({'stage':'after-exit',**base.gpu_gate(c)})
            finally:base.old.write_json(folder/('GPU-monitor-worker-'+str(index)+'.json'),{'observations':monitoring,'continuous_exclusivity_proved':False})
            if execution['exit_status']!=0:raise RuntimeError('Original worker/capture rejected; no next worker or retry.')
            runtime.collect_task(folder,index,local=False);state['completed_workers']=index+1
        runtime.finish(folder);state.update(pair_status(folder))
    except BaseException as exc:
        state.update(error_type=type(exc).__name__,error=str(exc),pair_checks_passed=False)
        path=folder/'results/pair.json'
        if path.exists() and not runtime.read_record(path)['run_finished']:
            runtime.finish(folder,error={'type':type(exc).__name__,'message':str(exc)})
        print('Original pair stopped: '+str(exc),file=sys.stderr,flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if (folder/'emergency-space.bin').exists():(folder/'emergency-space.bin').unlink()
        state.update(work_finished=True,wall_seconds=time.monotonic()-began)
        base.old.write_json(folder/'operator-result.json',state)
    signal.setitimer(signal.ITIMER_REAL,600)
    try:result_stream(folder,sha,output)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,handler in previous.items():signal.signal(s,handler)
    return state


def collect_remote(c,scratch,output):
    validate_control(c);folder=Path(scratch);info=folder.lstat();sha=hashlib.sha256(base.old.json_bytes(c)).hexdigest()
    if (platform.node()!=c['host'] or folder.parent!=Path('/var/tmp') or not folder.name.startswith(PREFIX+c['host']+'-')
            or folder.is_symlink() or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o700):raise ValueError('Wrong finished pair scratch.')
    state=json.loads((folder/'operator-result.json').read_text())
    if state['control_sha256']!=sha or state['work_finished'] is not True:raise ValueError('Only this finished attempt can be collected.')
    output.write(base.old.json_bytes({'kind':KIND,'phase':'scratch-created','host':c['host'],'control_sha256':sha,
        'scratch_directory':str(folder),'download_only':True}));output.flush()
    previous=signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('Collection deadline reached.')))
    signal.setitimer(signal.ITIMER_REAL,600)
    try:result_stream(folder,sha,output)
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def remote_main(control64,sources64,mode='run',scratch=None):
    os.umask(0o077);os.environ['PYTHONDONTWRITEBYTECODE']='1'
    c=json.loads(base64.b64decode(control64,validate=True))
    if mode=='collect':return collect_remote(c,scratch,sys.stdout.buffer)
    if mode!='run':raise ValueError('Only a new pair or finished-file collection is supported.')
    line=sys.stdin.buffer.readline(POLICY['max_header_bytes']+1)
    if len(line)>POLICY['max_header_bytes'] or not line.endswith(b'\n'):raise ValueError('Missing bounded new pair authorization.')
    sources={n:base64.b64decode(b,validate=True) for n,b in sources64.items()}
    return run_remote(c,json.loads(line),sys.stdin.buffer,sys.stdout.buffer,sources)
