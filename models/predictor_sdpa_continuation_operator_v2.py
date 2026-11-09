"""Four component workers on frozen native inputs; no full-model recapture."""
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import tarfile
import tempfile
import time
from models import predictor_sdpa_handoff_operator_v2 as prior
from models import predictor_sdpa_continuation_storage_v2 as policy
from models.predictor_sdpa_continuation_contract_v2 import (KIND,AUTH_KIND,PREFIX,CONTRACT_SHA,CAPSULE_SHA,
    MANIFEST_SHA,PARENT_CONTROL_SHA,PRIOR_CLAIM_SHA,NATIVE_WORKER_SHA,BOUNDARY_SHA,
    MiB,GiB,RESOURCE,ROLES,FROZEN_BOOT,COMPONENT_SOURCES,REMOTE_NAMES,LOCAL_EXTRA)
base=prior.base;old=prior.old;INIT=prior.INIT
inventory=policy.inventory;storage=policy.storage;stream=policy.stream
readiness=prior.readiness
activate_sources=prior.activate_sources
exact=prior.exact


def validate_control(c):
    expected={'kind':KIND,'host':'linux10603','parent':'/var/tmp','expected_device':66307,'expected_mount':'/',
        'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f','expected_driver':'580.178.04',
        'expected_ssh_fingerprint':base.SSH_FINGERPRINT,'contract_sha256':CONTRACT_SHA,'prior_consumed_claim_sha256':PRIOR_CLAIM_SHA,
        'paid_compute':False,'research_admitted':False,'authorization_present_at_preparation':False,'automatic_retries':0,'resource':RESOURCE}
    other={'contract_text','contract','repository','local_handoff_directory','archive_path','archive_manifest_text','archive_manifest',
        'known_hosts_sha256','local_source_sha256','remote_source_sha256'}
    if set(c)!=set(expected)|other or not exact({k:c[k] for k in expected},expected):raise ValueError('Exact component-only control required.')
    if (hashlib.sha256(c['contract_text'].encode()).hexdigest()!=CONTRACT_SHA or not exact(json.loads(c['contract_text']),c['contract'])
            or hashlib.sha256(c['archive_manifest_text'].encode()).hexdigest()!=MANIFEST_SHA
            or not exact(json.loads(c['archive_manifest_text']),c['archive_manifest'])):raise ValueError('Frozen continuation contract/capsule manifest changed.')
    contract=c['contract'];spec=c['archive_manifest']
    if (contract['authorized'] is not False or contract['new_target_execution_permitted'] is not False
            or not exact(contract['resource'],RESOURCE) or contract['parent_control_sha256']!=PARENT_CONTROL_SHA
            or contract['parent_consumed_claim']['sha256']!=PRIOR_CLAIM_SHA or contract['full_model_backwards']!=0
            or contract['new_native_captures']!=0 or contract['native_worker_sha256']!=NATIVE_WORKER_SHA
            or contract['boundary_sha256']!=BOUNDARY_SHA or not contract['candidate_remains_rejected']):
        raise ValueError('Exact frozen-input lineage and no-recapture scope required.')
    if (spec['sha256']!=CAPSULE_SHA or spec['bytes']!=36042589 or spec['logical_bytes']!=48627833
            or len(spec['files'])!=118 or spec['new_target_execution_permitted'] is not False
            or spec['capture_control_sha256']!=PARENT_CONTROL_SHA):raise ValueError('Reviewed native capsule changed.')
    for name,item in spec['files'].items():
        old.safe_name(name)
        if not name.startswith(('code/','results/capture/','input/')) or item['mode']!=0o600:raise ValueError('Private frozen input coverage changed.')
    if (set(c['remote_source_sha256'])!=set(REMOTE_NAMES)
            or set(c['local_source_sha256'])!=(set(REMOTE_NAMES)-{'models/__init__.py'}|set(LOCAL_EXTRA))
            or c['remote_source_sha256']['configs/predictor_sdpa_continuation_v2.json']!=CONTRACT_SHA):
        raise ValueError('Exact continuation source coverage required.')


def authorization_template(sha,user_basis):
    return {'kind':AUTH_KIND,'approved':True,'control_sha256':sha,'host':'linux10603','maximum_new_attempts':1,
        'resource':RESOURCE,'scientific_workers':list(ROLES),'full_model_backwards':0,'new_native_captures':0,'FP64_reference_builds':1,
        'component_untimed_iterations':4,'component_profiled_iterations':4,'component_warmup_or_timed_iterations':0,
        'bootstrap_bytes':2*MiB,'automatic_retries':0,'paid_compute':False,'cleanup':False,'reset_old_claims':False,
        'prior_consumed_claim_sha256':PRIOR_CLAIM_SHA,'parent_control_sha256':PARENT_CONTROL_SHA,'user_basis':user_basis}


def validate_authorization(a,sha):
    if (not isinstance(a,dict) or not isinstance(a.get('user_basis'),str) or not 1<=len(a['user_basis'])<=8192
            or not exact(a,authorization_template(sha,a['user_basis']))):
        raise ValueError('One new exact component-only university authorization required.')


def extract(archive,spec,folder):
    """Verify the complete capsule before exclusively creating its new paths."""
    archive=Path(archive);folder=Path(folder)
    if archive.stat().st_size!=spec['bytes'] or old.digest(archive)!=spec['sha256']:raise ValueError('Frozen capsule archive changed.')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[old.safe_name(m.name) for m in members]
        if len(names)!=len(set(names)) or set(names)!=set(spec['files']):raise ValueError('Exact capsule coverage required.')
        for m in members:
            item=spec['files'][m.name];path=folder/m.name
            if (not m.isfile() or (m.size,m.mode,m.mtime)!=(item['bytes'],item['mode'],item['mtime_ns']//10**9)
                    or path.exists() or path.is_symlink()):raise ValueError('Capsule metadata/collision refused.')
            for parent in path.parents:
                if parent==folder:break
                if parent.is_symlink() or parent.exists() and not parent.is_dir():raise ValueError('Unsafe capsule parent.')
            with tar.extractfile(m) as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=item['sha256']:raise ValueError('Frozen capsule member changed.')
        for m in members:
            path=folder/m.name;path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            with tar.extractfile(m) as source,os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as out:
                for block in iter(lambda:source.read(MiB),b''):out.write(block)
                out.flush();os.fsync(out.fileno())
            ns=spec['files'][m.name]['mtime_ns'];os.utime(path,ns=(ns,ns))


def verify_inputs(folder,c):
    for name,item in c['archive_manifest']['files'].items():
        p=Path(folder)/old.safe_name(name);s=p.lstat()
        if (p.is_symlink() or not stat.S_ISREG(s.st_mode) or s.st_nlink!=1
                or (s.st_size,stat.S_IMODE(s.st_mode),s.st_mtime_ns,old.digest(p))!=(item['bytes'],0o600,item['mtime_ns'],item['sha256'])):
            raise ValueError('Frozen input changed: '+name)


def verify_frozen_capture(folder,c):
    verify_inputs(folder,c);folder=Path(folder)
    parent=folder/'input/parent-control.json';capture=folder/'results/capture'
    if old.digest(parent)!=PARENT_CONTROL_SHA or old.digest(capture/'worker.json')!=NATIVE_WORKER_SHA:
        raise ValueError('Frozen native parent identity changed.')
    receipt=prior.verify_capture(capture,PARENT_CONTROL_SHA,json.loads(parent.read_text()))
    if receipt['boundary_sha256']!=BOUNDARY_SHA:raise ValueError('Frozen boundary changed.')
    return receipt


def component_plan(overlay,capture,components,*,local_tiny=False,control_sha256=PARENT_CONTROL_SHA):
    from models import predictor_sdpa_component_v3 as core
    from models.predictor_sdpa_component_records_v1 import sha
    overlay=Path(overlay);capture=Path(capture);w=json.loads((capture/'worker.json').read_text())
    p={'kind':core.PLAN_KIND,'local_tiny':local_tiny,'repository':str(overlay.resolve()),'capture_root':str(capture.resolve()),
        'native_worker_sha256':sha(capture/'worker.json'),'boundary_sha256':w['boundary_header_sha256'],'capture_control_sha256':control_sha256,
        'full_native_anchor_logical_sha256':({n:w['artifacts'][n]['source']['sha256'] for n in ('loss','input')} if local_tiny else w['full_native_anchor_logical_sha256']),'reference_dir':str((Path(components)/'reference').resolve()),
        'cells':[list(x) for x in core.CELLS],'bounds':core.BOUNDS,'bookkeeping':core.BOOK,'reference_builder_calls':1,
        'worker_seconds':120,'untimed_iterations':1,'profiled_iterations':1,'research_admitted':False,'performance_qualified':False,
        'full_model_candidate_adopted':False,'pure_backward_arithmetic_isolated':False,'paid_compute':False,'automatic_retry':False,'private_backward_API':False,
        'source_sha256':{n:sha(overlay/n) for n in core.SOURCE_NAMES}}
    core.validate_plan(p);core.capture(p);return p


def verify_task(folder,role,sha,c,*,tiny=False,deep=True):
    folder=Path(folder);index=ROLES.index(role);components=folder/'results/components';path=components/str(index)
    if not tiny:verify_frozen_capture(folder,c)
    plan_path=components/'component-plan.json';plan=json.loads(plan_path.read_text());ps=old.digest(plan_path)
    if plan['local_tiny'] is not tiny or not tiny and plan['capture_control_sha256']!=PARENT_CONTROL_SHA:
        raise ValueError('Component/frozen capture domain changed.')
    if deep:
        from models import predictor_sdpa_component_v3 as core
        core.validate_plan(plan);_,_,_,identity=core.capture(plan);w=core.load_worker(path,plan,ps,identity,index)
    else:
        from models import predictor_sdpa_component_records_v1 as rec
        w=rec.load(path,kind=rec.WORKER_KIND,plan_sha256=ps)
        for phase in ('untimed','profiled'):
            for item in w['outputs'][phase].values():rec.verified(path,item)
        rec.verified(path,w['trace'])
    resource=path/'component-resource.json'
    if not tiny:
        r=json.loads(resource.read_text())
        if (r['kind']!='native-fixed-SDPA-component-resource-v3' or r['pid']!=w['pid'] or r['index']!=index
                or r['plan_sha256']!=ps or r['worker_sha256']!=old.digest(path/'worker.json')
                or not 0<r['peak_allocator_bytes']<=7*GiB or not exact(r['startup'],w['startup'])):
            raise ValueError('Actual startup/allocator receipt missing/changed.')
    return {'role':role,'pid':w['pid'],'worker_sha256':old.digest(path/'worker.json'),
        'marker_sha256':old.digest(path/'complete.json'),'plan_sha256':ps,'resource_sha256':old.digest(resource) if not tiny else None}


def remote(c,auth,incoming,output,sources):
    validate_control(c);sha=hashlib.sha256(old.json_bytes(c)).hexdigest();validate_authorization(auth,sha)
    start=time.monotonic();deadline=start+RESOURCE['work_seconds'];handlers={s:signal.getsignal(s) for s in (signal.SIGHUP,signal.SIGTERM,signal.SIGALRM)}
    def stop(*unused):raise RuntimeError('Attended component-only attempt interrupted; preserve evidence, no retry.')
    mask=os.umask(0o077)
    for s in handlers:signal.signal(s,stop)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()-5))
    try:return _execute(c,auth,incoming,output,sources,sha,start,deadline,stop)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)
        os.umask(mask)


def _execute(c,auth,incoming,output,sources,sha,start,deadline,stop):
    identity=readiness(c);base.gpu_gate(c);folder=Path(tempfile.mkdtemp(prefix=PREFIX,dir=identity['parent']));folder.chmod(0o700)
    output.write(old.json_bytes({'kind':KIND,'phase':'scratch-created','scratch_directory':str(folder),'control_sha256':sha}));output.flush()
    state={'kind':KIND,'complete':False,'component_comparison_complete':False,'control_sha256':sha,'tasks':[],
        'started_unix':time.time(),'deadline_unix':time.time()+max(0,deadline-time.monotonic()),'work_seconds':900,
        'full_model_backwards':0,'new_native_captures':0,'candidate_native_numerical_passed':False,'performance_qualified':False,'research_admitted':False}
    try:
        old.write_json(folder/'control.json',c);old.write_json(folder/'authorization.json',auth);old.write_json(folder/'storage-before.json',identity)
        if set(sources)!=set(REMOTE_NAMES) or sum(map(len,sources.values()))>2*MiB:raise ValueError('Exact overlay coverage/extent required.')
        for n,data in sources.items():
            if hashlib.sha256(data).hexdigest()!=c['remote_source_sha256'][n]:raise ValueError('Overlay source changed.')
            path=folder/'overlay'/old.safe_name(n);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);path.chmod(0o600)
        with (folder/'emergency-space.bin').open('xb') as f:f.write(os.urandom(16*MiB));f.flush();os.fsync(f.fileno())
        end=min(deadline-5,time.monotonic()+60)
        program='import sys;sys.path.insert(0,'+repr(str(folder/'overlay'))+');from models.scratch_allocation_v2 import bounded_write_proof;bounded_write_proof('+repr(str(folder))+',deadline='+repr(end)+')'
        proof=base.primitives.supervise([sys.executable,'-B','-u','-c',program],folder,role='proof',deadline=end,
            gate=lambda:storage(folder,identity,setup=True),file_limit=3*GiB)
        if proof['exit_status']!=0:raise RuntimeError('Full storage proof failed before capsule upload.')
        base.primitives.validate_proof(folder)
        setup_end=min(deadline-5,start+RESOURCE['setup_seconds'])
        old.receive_archive(incoming,folder/'native-input-capsule.tar.gz',c['archive_manifest']['bytes'],CAPSULE_SHA,setup_end)
        extract(folder/'native-input-capsule.tar.gz',c['archive_manifest'],folder)
        if time.monotonic()>setup_end:raise TimeoutError('Setup allowance exhausted.')
        for name in ('tmp','cache','cache/cuda','results/components'):(folder/name).mkdir(mode=0o700,parents=True,exist_ok=True)
        activate_sources(folder);state['frozen_capture']=verify_frozen_capture(folder,c)
        p=component_plan(folder/'overlay',folder/'results/capture',folder/'results/components')
        old.write_json(folder/'results/components/component-plan.json',p);old.write_json(folder/'session.json',state)
        entry=folder/'overlay/scripts/capture_predictor_sdpa_continuation_v2.py'
        for role in ROLES:
            if verify_frozen_capture(folder,c)!=state['frozen_capture']:raise ValueError('Frozen native receipt changed.')
            for task in state['tasks']:
                if verify_task(folder,task['role'],sha,c)!=task:raise ValueError('Earlier component evidence changed.')
            base.gpu_gate(c);end=min(deadline-125,time.monotonic()+120)
            if end<=time.monotonic():raise TimeoutError('Continuing allowance exhausted before next component.')
            old.write_json(folder/('worker-plan-'+role+'.json'),{'role':role,'control_sha256':sha,'deadline_unix':time.time()+end-time.monotonic()})
            command=[sys.executable,'-I','-B','-u',str(entry),'--scratch',str(folder),'--control-sha256',sha,'--role',role]
            last=[0.];samples=[]
            def gate():
                row=storage(folder,identity)
                if time.monotonic()-last[0]>=5:
                    owned=[]
                    for pid in old.descendants(os.getpid()):
                        try:argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
                        except FileNotFoundError:continue
                        if str(entry).encode() in argv:owned.append(pid)
                    if len(owned)>1:raise ValueError('Multiple component child processes.')
                    samples.append(base.gpu_gate(c,owned[0] if owned else None));last[0]=time.monotonic()
                return row
            env={'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES','0'),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
                'TMPDIR':str(folder/'tmp'),'XDG_CACHE_HOME':str(folder/'cache'),'CUDA_CACHE_PATH':str(folder/'cache/cuda'),'CUDA_CACHE_MAXSIZE':str(32*MiB)}
            if env['CUDA_VISIBLE_DEVICES'] in ('','-1'):raise ValueError('CUDA visibility hidden; refuse.')
            result=base.primitives.supervise(command,folder,role=role,deadline=end,gate=gate,file_limit=16*MiB,env=env)
            old.write_json(folder/('gpu-samples-'+role+'.json'),samples);base.gpu_gate(c)
            if result['exit_status']!=0:raise RuntimeError('Component worker failed; no retry or next worker.')
            state['tasks'].append(verify_task(folder,role,sha,c));old.write_json(folder/'session.json',state,replace=True)
        if len({t['pid'] for t in state['tasks']})!=4:raise ValueError('Four fresh component processes required.')
        state.update(complete=True,component_comparison_complete=True)
    except BaseException as e:state.update(error_type=type(e).__name__,error=str(e))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);state['wall_seconds']=time.monotonic()-start
        if state['wall_seconds']>900:state.update(complete=False,component_comparison_complete=False,error='Continuing work deadline exceeded.')
        old.write_json(folder/'session.json',state,replace=(folder/'session.json').exists())
    signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,600)
    try:stream(folder,sha,output)
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    return 0 if state['component_comparison_complete'] else 1


def child(scratch,sha,role,overlay):
    scratch=Path(scratch);overlay=Path(overlay)
    if (scratch.is_symlink() or overlay!=scratch.resolve()/'overlay' or scratch.parent!=Path('/var/tmp')
            or not scratch.name.startswith(PREFIX) or role not in ROLES):raise ValueError('Exact scratch/child role required.')
    if old.digest(scratch/'control.json')!=sha:raise ValueError('Bound control changed.')
    c=json.loads((scratch/'control.json').read_text());validate_control(c);validate_authorization(json.loads((scratch/'authorization.json').read_text()),sha)
    for name,want in c['remote_source_sha256'].items():
        p=overlay/old.safe_name(name)
        if p.is_symlink() or old.digest(p)!=want:raise ValueError('Overlay source changed before scientific imports.')
    verify_frozen_capture(scratch,c)
    state=json.loads((scratch/'session.json').read_text());plan=json.loads((scratch/('worker-plan-'+role+'.json')).read_text());index=ROLES.index(role)
    if (state['complete'] or len(state['tasks'])!=index or plan['role']!=role or plan['control_sha256']!=sha
            or state['full_model_backwards']!=0 or state['new_native_captures']!=0):raise ValueError('Frozen-input component order changed.')
    remaining=plan['deadline_unix']-time.time()
    if not 0<remaining<=120 or plan['deadline_unix']>state['deadline_unix']-120:raise ValueError('Continuing child clock changed.')
    signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,remaining)
    for task in state['tasks']:
        if verify_task(scratch,task['role'],sha,c,deep=False)!=task:raise ValueError('Prior component receipt changed before scientific imports.')
    base.gpu_gate(c);sys.path.insert(1,str(scratch/'code'))
    from types import SimpleNamespace
    from models import predictor_sdpa_component_target_v3 as target
    path=scratch/'results/components/component-plan.json'
    result=target.run_component(SimpleNamespace(plan=path,plan_sha256=old.digest(path),output=path.parent/str(index),index=index,device='cuda',deadline_unix=plan['deadline_unix']))
    return 0 if result['complete'] else 1


def main(control64,sources64):
    c=json.loads(base64.b64decode(control64,validate=True));sources={n:base64.b64decode(v,validate=True) for n,v in json.loads(base64.b64decode(sources64,validate=True)).items()}
    validate_control(c);auth=json.loads(sys.stdin.buffer.readline(65537))
    return remote(c,auth,sys.stdin.buffer,sys.stdout.buffer,sources)
