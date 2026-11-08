"""One source-bound native capture/four components, continuing bounds and collection."""
import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import stat
import struct
import sys
import tarfile
import tempfile
import time
from models import oracle_boundary_operator_v1 as base
from models.predictor_sdpa_handoff_contract_v2 import (KIND,AUTH_KIND,PREFIX,CONTRACT_SHA,PACKET_SHA,PRIOR_CLAIM_SHA,
    MiB,GiB,RESOURCE,ROLES,FROZEN_BOOT,COMPONENT_SOURCES,REMOTE_NAMES,LOCAL_EXTRA,
    PRIOR_METADATA_CLAIM_SHA,SCIENTIFIC_PRIOR_CLAIM_SHA)
old=base.old
INIT=base.OVERLAY_INIT
from models import predictor_sdpa_storage_v2 as policy
inventory=policy.inventory
storage=policy.storage
stream=policy.stream


def exact(actual,expected):return old.json_bytes(actual)==old.json_bytes(expected)


def validate_control(c):
    expected={'kind':KIND,'host':'linux10603','parent':'/var/tmp','expected_device':66307,'expected_mount':'/',
        'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f','expected_driver':'580.178.04',
        'expected_ssh_fingerprint':base.SSH_FINGERPRINT,'contract_sha256':CONTRACT_SHA,'prior_consumed_claim_sha256':PRIOR_CLAIM_SHA,
        'paid_compute':False,'research_admitted':False,'authorization_present_at_preparation':False,'automatic_retries':0,'resource':RESOURCE}
    other={'contract_text','contract','proposal_text','proposal','repository','local_handoff_directory','archive_path','archive_manifest',
        'known_hosts_sha256','local_source_sha256','remote_source_sha256'}
    if set(c)!=set(expected)|other or not exact({k:c[k] for k in expected},expected):raise ValueError('Exact new fixed-input attention control required.')
    if (hashlib.sha256(c['contract_text'].encode()).hexdigest()!=CONTRACT_SHA or json.loads(c['contract_text'])!=c['contract']
            or hashlib.sha256(c['proposal_text'].encode()).hexdigest()!=c['contract']['proposal_sha256'] or json.loads(c['proposal_text'])!=c['proposal']):
        raise ValueError('Fixed native scope/proposal changed.')
    contract=c['contract'];p=c['proposal']
    if (contract['kind']!='native-fixed-SDPA-target-handoff-contract-v2' or contract['authorized'] is not False
            or not exact(contract['resource'],RESOURCE) or p['authorized'] is not False or p['new_target_execution_permitted'] is not False
            or not exact(p['resource_proposal'],{k:v for k,v in RESOURCE.items() if k!='mask_indices_file_bytes'}) or p['prior_consumed_claim_sha256']!=SCIENTIFIC_PRIOR_CLAIM_SHA
            or p['selection']!=contract['selection'] or p['financial_state_unchanged']!=contract['financial_state_unchanged']):
        raise ValueError('Inert proposal and immutable scientific/resources contract required.')
    if set(c['remote_source_sha256'])!=set(REMOTE_NAMES) or set(c['local_source_sha256'])!=(set(REMOTE_NAMES)-{'models/__init__.py'}|set(LOCAL_EXTRA)):
        raise ValueError('Exact source coverage required.')
    if c['remote_source_sha256']['configs/predictor_sdpa_handoff_v2.json']!=CONTRACT_SHA:raise ValueError('Overlay scope changed.')
    if (policy.COLLECTION_KIND!=KIND or policy.MASK_FILE_BYTES!=RESOURCE['mask_indices_file_bytes']
            or policy.RESOURCE!={k:v for k,v in RESOURCE.items() if k!='mask_indices_file_bytes'}
            or contract['mask_file_policy']!={'paths':[policy.MASK_PATH,policy.MASK_PATH+'.partial'],'bytes':policy.MASK_FILE_BYTES}
            or contract['prior_metadata_probe_claim_sha256']!=PRIOR_METADATA_CLAIM_SHA
            or contract['repair_of_control_sha256']!='becbdc2607a9cc5281830d20bb5f2f63b6ecab88aa1f9421eee52009bbbdc124'):
        raise ValueError('Exact incident-bound mask-file policy required.')
    spec=c['archive_manifest']
    if spec['sha256']!=PACKET_SHA or spec['bytes']!=17192095 or len(spec['files'])!=99 or sum(i['bytes'] for i in spec['files'].values())!=26446958:
        raise ValueError('Unchanged original one-clip packet required.')
    for n,item in spec['files'].items():
        old.safe_name(n)
        if type(item['bytes']) is not int or not 0<=item['bytes']<=32*MiB or len(item['sha256'])!=64:raise ValueError('Packet member bound changed.')


def authorization_template(sha,user_basis):
    return {'kind':AUTH_KIND,'approved':True,'control_sha256':sha,'host':'linux10603','maximum_new_attempts':1,
        'resource':RESOURCE,'scientific_workers':list(ROLES),'full_model_backwards':1,'FP64_reference_builds':1,
        'component_untimed_iterations':4,'component_profiled_iterations':4,'component_warmup_or_timed_iterations':0,
        'bootstrap_bytes':2*MiB,'automatic_retries':0,'paid_compute':False,'cleanup':False,'reset_old_claims':False,
        'prior_consumed_claim_sha256':PRIOR_CLAIM_SHA,'prior_metadata_probe_claim_sha256':PRIOR_METADATA_CLAIM_SHA,'user_basis':user_basis}


def validate_authorization(a,sha):
    if not isinstance(a,dict) or not isinstance(a.get('user_basis'),str) or not 1<=len(a['user_basis'])<=8192:
        raise ValueError('One new exact fixed-input university authorization required.')
    if not exact(a,authorization_template(sha,a['user_basis'])):raise ValueError('One new exact fixed-input university authorization required.')






def extract(archive,spec,destination):
    destination=Path(destination)
    if destination.exists() or old.digest(archive)!=spec['sha256']:raise ValueError('Packet changed or destination reused.')
    with tarfile.open(archive) as tar:
        members=tar.getmembers();names=[old.safe_name(m.name) for m in members]
        if len(names)!=len(set(names)) or set(names)!=set(spec['files']) or any(not m.isfile() for m in members):raise ValueError('Exact regular packet members required.')
        for m in members:
            if m.size!=spec['files'][m.name]['bytes']:raise ValueError('Packet extent changed.')
            with tar.extractfile(m) as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=spec['files'][m.name]['sha256']:raise ValueError('Packet member hash changed.')
        destination.mkdir(mode=0o700)
        for m in members:
            p=destination/m.name;p.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as f,p.open('xb') as out:
                for block in iter(lambda:f.read(MiB),b''):out.write(block)


def verify_capture(folder,sha,c,*,tiny=False):
    from models import predictor_sdpa_capture_record_v1 as native
    folder=Path(folder);w=json.loads((folder/'worker.json').read_text())
    cap=native.load(folder/'boundary',control_sha256=sha)
    want='original-SDPA-observer-local-CPU-rehearsal-v1' if tiny else 'original-SDPA-native-target-capture-v1'
    if (w['kind']!=want or w['local_tiny'] is not tiny or w['control_sha256']!=sha or w['complete'] is not True
            or any(w.get(k) is not True for k in ('original_value_replay_exact','model_unchanged','route_restored','observer_restored'))
            or any(w.get(k) is not False for k in ('research_admitted','performance_qualified','native_candidate_adopted','CUDA_kernel_qualified'))
            or not 0<w['wall_seconds']<=420 or w['baseline']!=c['contract']['approved_baseline']
            or w['full_native_anchor_logical_sha256']!=c['contract']['full_native_anchor_logical_sha256']
            or w['boundary_header_sha256']!=old.digest(folder/'boundary/boundary.json') or cap['local_tiny'] is not tiny):
        raise ValueError('Passing exact original capture required before components.')
    marker=json.loads((folder/'capture-complete.json').read_text())
    if (marker!={'kind':want,'control_sha256':sha,'header':{'file':'worker.json','bytes':(folder/'worker.json').stat().st_size,'sha256':old.digest(folder/'worker.json')},'complete':True}
            or (folder/'capture-failure.json').exists()):raise ValueError('Capture final marker changed.')
    if not tiny and (w['execution_host']!='linux10603' or not 0<w.get('peak_allocator_bytes',0)<=7*GiB):raise ValueError('Target capture host/allocator changed.')
    for n in ('loss','input','targets'):
        item=w['artifacts'][n];path=folder/(n+'.pt')
        if path.is_symlink() or item['file']!=path.name or item['bytes']!=path.stat().st_size or old.digest(path)!=item['sha256']:raise ValueError('Native actual raw value changed.')
        desired=w['baseline']['targets']['sha256'] if n=='targets' else c['contract']['full_native_anchor_logical_sha256'][n]
        if item['source']['sha256']!=desired:raise ValueError('Native original loss/input/target value anchor changed.')
    mp=folder/'mask-indices.json';item=w['mask_indices']
    if item['file']!=mp.name or mp.stat().st_size!=item['bytes'] or old.digest(mp)!=item['sha256']:raise ValueError('Actual mask bytes changed.')
    indices=json.loads(mp.read_text())
    if len(indices)!=8 or any(len(d)!=2 for d in indices):raise ValueError('All EOT8 masks required.')
    for draw,actual in zip(w['baseline']['masks'],indices):
        for signatures,pair in zip(draw,actual):
            for signature,name in zip(signatures,('context','target')):
                ids=pair[name]
                if (not isinstance(ids,list) or len(ids)!=1 or not ids[0] or any(type(v) is not int for v in ids[0])
                        or hashlib.sha256(struct.pack('<'+'q'*len(ids[0]),*ids[0])).hexdigest()!=signature['sha256']):raise ValueError('Typed original mask indices changed.')
    return {'role':'capture','pid':w['pid'],'worker_sha256':old.digest(folder/'worker.json'),
        'marker_sha256':old.digest(folder/'capture-complete.json'),'boundary_sha256':w['boundary_header_sha256']}


def verify_task(folder,role,sha,c,*,tiny=False,deep=True):
    folder=Path(folder)
    if role=='capture':return verify_capture(folder/'results/capture',sha,c,tiny=tiny)
    index=int(role.rsplit('-',1)[1]);components=folder/'results/components';plan_path=components/'component-plan.json'
    plan=json.loads(plan_path.read_text());ps=old.digest(plan_path)
    if plan['capture_control_sha256']!=sha or plan['local_tiny'] is not tiny:raise ValueError('Component/capture domain changed.')
    verify_capture(folder/'results/capture',sha,c,tiny=tiny)
    path=components/str(index)
    if deep:
        from models import predictor_sdpa_component_v1 as core
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
        if (r['kind']!='native-fixed-SDPA-component-resource-v1' or r['pid']!=w['pid'] or r['index']!=index
                or r['plan_sha256']!=ps or r['worker_sha256']!=old.digest(path/'worker.json') or not 0<r['peak_allocator_bytes']<=7*GiB):
            raise ValueError('Actual component allocator receipt missing/changed.')
    return {'role':role,'pid':w['pid'],'worker_sha256':old.digest(path/'worker.json'),
        'marker_sha256':old.digest(path/'complete.json'),'plan_sha256':ps,
        'resource_sha256':old.digest(resource) if not tiny else None}




def readiness(c):
    if platform.node()!='linux10603' or sys.platform!='linux':raise ValueError('Wrong target host/OS.')
    parent=Path('/var/tmp');s=parent.stat();fs=os.statvfs(parent)
    value={'host':platform.node(),'parent':str(parent),'device':s.st_dev,'inode':s.st_ino,'home_device':Path.home().stat().st_dev,
        'owner_uid':s.st_uid,'mode':stat.S_IMODE(s.st_mode),**old.mount_for(parent,Path('/proc/self/mountinfo').read_text()),
        'available_bytes':fs.f_bavail*fs.f_frsize,'available_inodes':fs.f_favail,'writable_for_current_user':os.access(parent,os.W_OK|os.X_OK)}
    if parent.is_symlink() or value['available_bytes']<8*GiB:raise ValueError('Separate local scratch unavailable.')
    old.validate_readiness(value,c);return value


def remote(c,auth,incoming,output,sources):
    validate_control(c);sha=hashlib.sha256(old.json_bytes(c)).hexdigest();validate_authorization(auth,sha)
    start=time.monotonic();deadline=start+RESOURCE['work_seconds'];handlers={s:signal.getsignal(s) for s in (signal.SIGHUP,signal.SIGTERM,signal.SIGALRM)}
    def stop(*unused):raise RuntimeError('Attended fixed-input attempt interrupted; preserve evidence, no retry.')
    mask=os.umask(0o077)
    for s in handlers:signal.signal(s,stop)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()-5))
    try:return _execute(c,auth,incoming,output,sources,sha,start,deadline,stop)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)
        os.umask(mask)


def activate_sources(folder):
    import models
    models.__path__=[str(folder/'overlay/models'),str(folder/'code/models')]


def _execute(c,auth,incoming,output,sources,sha,start,deadline,stop):
    identity=readiness(c);base.gpu_gate(c);folder=Path(tempfile.mkdtemp(prefix=PREFIX,dir=identity['parent']));folder.chmod(0o700)
    output.write(old.json_bytes({'kind':KIND,'phase':'scratch-created','scratch_directory':str(folder),'control_sha256':sha}));output.flush()
    state={'kind':KIND,'complete':False,'scientific_passed':False,'component_comparison_complete':False,'control_sha256':sha,
        'tasks':[],'started_unix':time.time(),'deadline_unix':time.time()+max(0,deadline-time.monotonic()),'work_seconds':1500,
        'candidate_native_numerical_passed':False,'performance_qualified':False,'research_admitted':False}
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
        if proof['exit_status']!=0:raise RuntimeError('Full storage proof failed before packet upload.')
        base.primitives.validate_proof(folder)
        setup_end=min(deadline-5,start+180)
        old.receive_archive(incoming,folder/'original-packet.tar.gz',17192095,PACKET_SHA,setup_end)
        extract(folder/'original-packet.tar.gz',c['archive_manifest'],folder/'code')
        if time.monotonic()>setup_end:raise TimeoutError('Original setup allowance exhausted.')
        for name in ('tmp','cache','cache/cuda','results','results/components'):(folder/name).mkdir(mode=0o700,exist_ok=True)
        # Cold bootstrap never imports science. Real overlay paths become visible
        # only after authorization, readiness, proof and source/packet verification.
        activate_sources(folder)
        old.write_json(folder/'session.json',state)
        entry=folder/'overlay/scripts/capture_predictor_sdpa_handoff_v2.py'
        for role in ROLES:
            for task in state['tasks']:
                if verify_task(folder,task['role'],sha,c)!=task:raise ValueError('Earlier native/component evidence changed.')
            base.gpu_gate(c);duration=420 if role=='capture' else 120;end=min(deadline-125,time.monotonic()+duration)
            if end<=time.monotonic():raise TimeoutError('Continuing allowance exhausted before next worker.')
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
                    if len(owned)>1:raise ValueError('Multiple scientific child processes.')
                    samples.append(base.gpu_gate(c,owned[0] if owned else None));last[0]=time.monotonic()
                return row
            env={'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES','0'),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
                'TMPDIR':str(folder/'tmp'),'XDG_CACHE_HOME':str(folder/'cache'),'CUDA_CACHE_PATH':str(folder/'cache/cuda'),'CUDA_CACHE_MAXSIZE':str(32*MiB)}
            if env['CUDA_VISIBLE_DEVICES'] in ('','-1'):raise ValueError('CUDA visibility hidden; refuse.')
            result=base.primitives.supervise(command,folder,role=role,deadline=end,gate=gate,file_limit=16*MiB,env=env)
            old.write_json(folder/('gpu-samples-'+role+'.json'),samples);base.gpu_gate(c)
            if result['exit_status']!=0:raise RuntimeError('Native/component worker failed; no retry or next worker.')
            state['tasks'].append(verify_task(folder,role,sha,c));old.write_json(folder/'session.json',state,replace=True)
        if len({t['pid'] for t in state['tasks']})!=5:raise ValueError('Five fresh processes required.')
        state.update(complete=True,component_comparison_complete=True)
    except BaseException as e:state.update(error_type=type(e).__name__,error=str(e))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);state['wall_seconds']=time.monotonic()-start
        if state['wall_seconds']>1500:state.update(complete=False,component_comparison_complete=False,error='Original continuing work deadline exceeded.')
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
    for name,item in c['archive_manifest']['files'].items():
        p=scratch/'code'/old.safe_name(name)
        if p.is_symlink() or p.stat().st_size!=item['bytes'] or old.digest(p)!=item['sha256']:raise ValueError('Frozen source/input changed.')
    state=json.loads((scratch/'session.json').read_text());plan=json.loads((scratch/('worker-plan-'+role+'.json')).read_text())
    count=ROLES.index(role);limit=420 if count==0 else 120
    if state['complete'] or len(state['tasks'])!=count or plan['role']!=role or plan['control_sha256']!=sha:raise ValueError('Capture/component order changed.')
    remaining=plan['deadline_unix']-time.time()
    if not 0<remaining<=limit or plan['deadline_unix']>state['deadline_unix']-120:raise ValueError('Continuing child clock changed.')
    signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,remaining)
    for task in state['tasks']:
        if verify_task(scratch,task['role'],sha,c,deep=False)!=task:raise ValueError('Prior verified task changed before scientific imports.')
    base.gpu_gate(c)
    sys.path.insert(1,str(scratch/'code'))
    from types import SimpleNamespace
    from models import predictor_sdpa_target_v1 as target
    existing=Path.home()/'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting'
    if role=='capture':
        result=target.run_capture(SimpleNamespace(output=scratch/'results/capture',control_sha256=sha,tiny=False,device='cuda',
            packet=scratch/'code',contract=c['contract'],upstream=existing/'external/vjepa2',checkpoint=existing/'checkpoints/vitl.pt'))
        components=scratch/'results/components';p=target.component_plan(overlay,scratch/'results/capture',components,control_sha256=sha,local_tiny=False)
        old.write_json(components/'component-plan.json',p)
    else:
        index=count-1;path=scratch/'results/components/component-plan.json'
        result=target.run_component(SimpleNamespace(plan=path,plan_sha256=old.digest(path),output=path.parent/str(index),index=index,device='cuda',deadline_unix=plan['deadline_unix']))
    return 0 if result['complete'] else 1


def main(control64,sources64):
    c=json.loads(base64.b64decode(control64,validate=True));sources={n:base64.b64decode(v,validate=True) for n,v in json.loads(base64.b64decode(sources64,validate=True)).items()}
    validate_control(c);auth=json.loads(sys.stdin.buffer.readline(65537))
    return remote(c,auth,sys.stdin.buffer,sys.stdout.buffer,sources)
