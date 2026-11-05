"""Bounded replay-1 upload/import, two CUDA workers and one CPU verifier."""
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
from models import attention_replay1_runtime_v1 as runtime
from models import attention_replay1_contract_v1 as refs

KIND='attention-replay1-capture-v1'
AUTH_KIND='attention-replay1-resource-exception-v1'
PREFIX='vjepa-attention-replay1-v1-20261105-'
HANDOFF_SHA=base.HANDOFF_SHA;ARCHIVE_SHA=base.ARCHIVE_SHA
PROPOSAL_SHA='327ade0b9445ad0c13b9d6e88d4cd6ee45824ce156e40c26f52e9e13a7a80e44'
REVIEW_SHA='2c34e86380b781f3cb66b731dba38825a448fc669c22eaefb699a0e0e27a72f8'
CLAIM_SHA='f7d663471c3e2afd4d22ced57fca02f6c2210cb08d0a7c9e234355d2dd213a4e'
GIB=2**30;MIB=2**20
POLICY={**base.POLICY,'max_input_bytes':256*MIB,'max_raw_bytes':400*MIB,'max_CPU_return_bytes':16*MIB,
    'max_GPU_return_bytes':8*MIB,'max_workers':2,'max_CPU_verifiers':1,'CPU_verifier_seconds':90,
    'max_CPU_oracle_calls':3,'max_Torch_CPU_oracle_calls':2,'max_NumPy_CPU_oracle_calls':1,
    'max_attention_iterations':12,'max_reference_archive_bytes':64*MIB}
PREREQUISITE={'proposal_sha256':PROPOSAL_SHA,'accepted_pair_review_sha256':REVIEW_SHA,
    'consumed_pair_claim_sha256':CLAIM_SHA,'previous_grants_consumed':True,'historical_discrepancy_resolved':False,
    'original_parent_session_qualified':False,'research_admitted':False,'new_attempt_authorized_at_preparation':False}
REMOTE_NAMES=('models/__init__.py','models/cpu_oracle_scratch.py','models/cpu_oracle_control_prerequisite.py',
    'models/cpu_oracle_control_v1.py','models/scratch_allocation_v2.py','models/oracle_trace.py',
    'models/oracle_boundary_observer_v1.py','models/oracle_boundary_operator_v1.py',
    'models/attention_return_capture_v1.py','models/original_attention_pair_observer_v1.py',
    'models/attention_replay1_contract_v1.py','models/attention_replay1_cohort_v1.py',
    'models/attention_replay1_observer_v1.py','models/attention_replay1_runtime_v1.py',
    'models/attention_replay1_operator_v1.py','scripts/capture_attention_replay1_v1.py')
LOCAL_NAMES=set(REMOTE_NAMES)-{'models/__init__.py'}|{'models/attention_replay1_client_v1.py','scripts/run_attention_replay1_v1.py'}
OVERLAY_INIT=base.OVERLAY_INIT


def validate_control(c):
    expected={'kind':KIND,'host':'linux10603','role':'encoder-1-replay1-and-CPU-cohort-verifier',
        'parent':'/var/tmp','expected_filesystem':'xfs','expected_device':66307,'expected_mount':'/',
        'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f',
        'expected_driver':'580.178.04','expected_ssh_fingerprint':base.SSH_FINGERPRINT,
        'archive_sha256':ARCHIVE_SHA,'archive_bytes':85622679,'handoff_sha256':HANDOFF_SHA,
        'policy':POLICY,'prerequisite':PREREQUISITE,'tasks':[refs.task(i,False) for i in (0,1)],
        'GPU_input_transfer_allowed':True,'GPU_attention_allowed':True,'paid_compute':False,'research_admitted':False,
        'old_refusals_waived':False,'reuse_old_grant':False,'authorization_present_at_preparation':False,
        'timings_usable_for_speed_or_budget':False,'original_four_worker_session_qualified':False,'original_parent_session_qualified':False,'CPU_only_verifier_allowed':True}
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
    required={'configs/original_attention_replay1_proposal_v1.json':PROPOSAL_SHA,
        'report/checks/2026-11-04-original-attention-pair-v1-target-review.json':REVIEW_SHA,
        'runs/pretrained/diagnostic/original-attention-pair-v1-20261104/linux10603/ATTEMPT.json':CLAIM_SHA}
    if any(c['prior_bindings_sha256'].get(k)!=v for k,v in required.items()):raise ValueError('Pair prerequisite or consumed claim missing.')
    refs.validate_manifest(c['reference_manifest'],local=False)
    spec=c['reference_spec']
    if (spec['kind']!='attention-replay1-reference-archive-v1' or spec['local_tiny_analogue'] is not False
            or spec['manifest_sha256']!=refs.LIVE_MANIFEST_SHA or not 0<spec['bytes']<=64*MIB
            or not isinstance(spec['sha256'],str) or len(spec['sha256'])!=64):raise ValueError('Exact bounded reference archive required.')



def validate_authorization(a,sha):
    expected={'kind':AUTH_KIND,'approved':True,'control_sha256':sha,'host':'linux10603','maximum_new_attempts':1,
        'max_seconds':600,'worker_seconds':120,'max_fresh_workers':2,'max_CPU_only_verifiers':1,'CPU_verifier_seconds':90,'max_CPU_oracle_calls':3,
        'max_Torch_CPU_oracle_calls':2,'max_NumPy_CPU_oracle_calls':1,'max_attention_iterations':12,
        'GPU_input_transfer_allowed':True,'GPU_attention_allowed':True,'write_test_bytes':2684354560,
        'setup_ceiling':3*GIB,'capture_ceiling':9*GIB//4,'raw_capture_ceiling':400*MIB,'input_ceiling':256*MIB,'reference_archive_ceiling':64*MIB,
        'collection_seconds':600,'CPU_only_verifier_allowed':True,'paid_compute':False,'reset_old_claims':False,'automatic_retries':0}
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
        elif name.startswith(('original-packet.tar.gz','reference-subset.tar')):key='input'
        elif name=='space-proof-v2.bin.partial':key='proof'
        elif name=='emergency-space.bin':key='emergency'
        elif name.startswith('results/') and name.endswith(('.pt','.pt.partial','.npz','.npz.partial')):
            key='raw'
            if item['bytes']>32*MIB:raise ValueError('Individual tensor file exceeds 32 MiB.')
        else:key='metadata'
        counts[key]+=item['bytes']
    limits={'code':128*MIB,'input':256*MIB,'raw':400*MIB,'proof':POLICY['write_test_bytes'] if phase=='setup' else 0,
        'metadata':64*MIB,'emergency':16*MIB}
    if any(counts[k]>v for k,v in limits.items()):raise ValueError('Pair storage category bound exceeded.')
    return {'phase':phase,'logical_bytes':sum(x['bytes'] for x in files.values()),'allocated_bytes':allocated,
        'available_bytes':space.f_bavail*space.f_frsize,'ceiling':limit,'files':len(files),'categories':counts}


def result_stream(folder,sha,output):
    files,_=base.inventory(folder,hashing=True,max_bytes=3*GIB)
    for name,item in files.items():
        item['mode']=stat.S_IMODE((Path(folder)/name).stat().st_mode)
        if item['mode'] not in (0o400,0o600):raise ValueError('Unexpected evidence file mode.')
    header=base.old.json_bytes({'kind':KIND,'phase':'collection','control_sha256':sha,'files':files,
        'logical_bytes':sum(x['bytes'] for x in files.values()),'scientific_review_complete':False,'model_restarted_for_collection':False})
    if len(header)>POLICY['max_header_bytes']:raise ValueError('Collection header exceeds bound.')
    output.write(header);output.flush()
    with tarfile.open(mode='w|',fileobj=output) as archive:
        for name,item in sorted(files.items()):
            p=Path(folder)/name
            if p.is_symlink() or base.old.digest(p)!=item['sha256']:raise ValueError('Evidence changed during collection.')
            member=tarfile.TarInfo(name);member.size=item['bytes'];member.mode=item['mode'];member.mtime=item['mtime_ns']//10**9
            with p.open('rb') as source:archive.addfile(member,source)


def cohort_status(folder):
    folder=Path(folder);path=folder/'results/replay1.json'
    if not path.exists():return {'cross_attempt_component_cohort_checks_passed':False,'cohort_run_present':False,'new_workers_completed':0,'verifier_completed':False}
    state=runtime.read_record(path)
    if state['kind']!=runtime.KIND or state['local_tiny_analogue'] is not False or state['run_finished'] is not True:
        raise ValueError('Missing finished live replay-1 receipt.')
    if (state['new_tasks']!=[refs.task(i,False) for i in (0,1)] or state['imported_workers']!=list(refs.imported(False))
            or any(state.get(k) is not v for k,v in runtime.FLAGS.items())):raise ValueError('Cohort execution/import scope changed.')
    if state['cross_attempt_component_cohort_checks_passed']:
        if len(state['new_workers_completed'])!=2 or not state['verifier_completed'] or 'error' in state:raise ValueError('Partial cohort cannot pass.')
        for i,prior in enumerate(state['new_workers_completed']):
            if runtime.worker_status(folder/'results/workers'/refs.task(i,False)['worker'],local=False)!=prior:raise ValueError('Completed worker evidence changed.')
        if runtime.verifier_status(folder,local=False)!=state['verifier']:raise ValueError('CPU verifier evidence changed.')
    return {'cross_attempt_component_cohort_checks_passed':state['cross_attempt_component_cohort_checks_passed'],
        'cohort_run_present':True,'new_workers_completed':len(state['new_workers_completed']),'verifier_completed':state['verifier_completed'],
        'cohort_receipt_sha256':base.old.digest(path),'original_parent_session_qualified':False}


def run_remote(c,auth,incoming,output,sources):
    validate_control(c);sha=hashlib.sha256(base.old.json_bytes(c)).hexdigest();validate_authorization(auth,sha)
    began=time.monotonic();unix=time.time();deadline=began+600;identity=readiness(c);base.gpu_gate(c)
    folder=Path(tempfile.mkdtemp(prefix=PREFIX+c['host']+'-',dir='/var/tmp'));folder.chmod(0o700)
    output.write(base.old.json_bytes({'kind':KIND,'phase':'scratch-created','host':c['host'],'control_sha256':sha,
        'scratch_directory':str(folder),'started_unix':unix}));output.flush()
    state={'kind':KIND,'host':c['host'],'control_sha256':sha,'started_unix':unix,'deadline_unix':unix+600,
        'work_finished':False,'cross_attempt_component_cohort_checks_passed':False,'new_workers_completed':0,'verifier_completed':False,'GPU_attention_allowed':True,
        'paid_compute':False,'scientific_review_complete':False,'old_refusals_waived':False,
        'original_four_worker_session_qualified':False,'original_parent_session_qualified':False,'performance_qualified':False,'research_admitted':False}
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
        runtime.initialize(folder,folder/'code',c['reference_manifest'],local=False,control_sha=sha,started=unix)
        base.old.receive_archive(incoming,folder/'reference-subset.tar',c['reference_spec']['bytes'],c['reference_spec']['sha256'],deadline-5)
        receipt=refs.extract_archive(folder/'reference-subset.tar',c['reference_spec'],c['reference_manifest'],folder/'results',local=False)
        runtime.record(folder/'results/reference-import.json',receipt);state['references_imported']=True
        storage_gate(folder,identity)
        if time.time()-unix>110:raise TimeoutError('Original setup freshness allowance exhausted.')
        for relative in ('tmp','cache','cache/cuda'):(folder/relative).mkdir(mode=0o700,exist_ok=True)
        for phase,index in (('worker',0),('worker',1),('cohort',None)):
            monitoring=[{'stage':'before-launch',**base.gpu_gate(c)}];last=[0.0]
            worker,plan,plan_path=runtime.issue(folder,folder/'code',c['reference_manifest'],phase,local=False,control_sha=sha,index=index)
            entry=folder/'overlay/scripts/capture_attention_replay1_v1.py'
            command=[sys.executable,'-I','-B','-u',str(entry),'--packet',str(folder/'code'),'--scratch',str(folder),
                '--control-sha256',sha,'--phase',phase,'--worker-plan-sha256',base.old.digest(plan_path),
                '--upstream',str(Path.home()/'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting/external/vjepa2')]
            if index is not None:command+=['--task-index',str(index)]
            visibility=os.environ.get('CUDA_VISIBLE_DEVICES')
            if phase=='worker' and visibility is None:command=['/usr/bin/env','-u','CUDA_VISIBLE_DEVICES',*command]
            env={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','TMPDIR':str(folder/'tmp'),
                'XDG_CACHE_HOME':str(folder/'cache'),'CUDA_CACHE_PATH':str(folder/'cache/cuda'),'CUDA_CACHE_MAXSIZE':str(32*MIB)}
            if phase=='worker' and visibility is not None:env['CUDA_VISIBLE_DEVICES']=visibility
            if phase=='cohort':env['CUDA_VISIBLE_DEVICES']=''
            def gate():
                sample=storage_gate(folder,identity)
                if time.monotonic()-last[0]>=5:
                    pids=[]
                    for pid in (Path('/proc/self/task')/str(os.getpid())/'children').read_text().split():
                        try:argv=(Path('/proc')/pid/'cmdline').read_bytes().split(b'\0')
                        except FileNotFoundError:continue
                        if str(entry).encode() in argv:pids.append(int(pid))
                    if len(pids)>1:raise ValueError('More than one scientific child is active.')
                    allowed=pids[0] if phase=='worker' and pids else None
                    monitoring.append({'stage':'during',**base.gpu_gate(c,allowed)});last[0]=time.monotonic()
                return sample
            role='worker-'+str(index) if phase=='worker' else 'cohort'
            if phase=='worker':state['workers_started']=index+1
            else:state['CPU_verifiers_started']=1
            try:
                execution=base.primitives.supervise(command,folder,role=role,
                    deadline=min(deadline-5,time.monotonic()+plan['deadline_unix']-time.time()),
                    gate=gate,file_limit=32*MIB,env=env)
                monitoring.append({'stage':'after-exit',**base.gpu_gate(c)})
            finally:base.old.write_json(folder/('GPU-monitor-'+role+'.json'),{'observations':monitoring,'continuous_exclusivity_proved':False})
            if execution['exit_status']!=0:raise RuntimeError('Original worker/capture/cohort refused; no next phase or retry.')
            refs.verify_imports(folder/'results',c['reference_manifest'],local=False)
            if phase=='worker':runtime.collect_worker(folder,index,local=False);state['new_workers_completed']=index+1
        runtime.finish(folder,local=False);state.update(cohort_status(folder))
    except BaseException as exc:
        state.update(error_type=type(exc).__name__,error=str(exc),cross_attempt_component_cohort_checks_passed=False)
        path=folder/'results/replay1.json'
        if path.exists() and not runtime.read_record(path)['run_finished']:
            runtime.finish(folder,local=False,error={'type':type(exc).__name__,'message':str(exc)})
        print('Replay-1 stopped: '+str(exc),file=sys.stderr,flush=True)
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
