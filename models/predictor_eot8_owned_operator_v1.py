"""New bounded operator. Old resource grants never authorize this scope."""
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
from models.predictor_eot8_owned_receipt_v1 import verify_ownership
old=base.old
MiB=2**20;GiB=2**30
KIND='predictor-clean-eot8-owned-operator-v1'
PREFIX='vjepa-predictor-eot8-owned-v1-20261106-'
PACKET_SHA='2278bce2b519943c0f83a49c556bb73dd928fe880fba0f6c56de2a4c3c2ec0da'
PROPOSAL_SHA='9a44e4b57a3ac36e0a5fb5092a089a6971da1ad0425714b06ae98793e4c7660d'
HOST_BINDING_SHA='83cc0a8b13594a19f5b4c0fb51b0137b6749bac2c9d38352e1463ac8d002f67b'
PRIOR_CONSUMED_CONTROL='29b781e522f16396940cf4e014ff5aa6fe4699bfa2d6f6c49da4385650c8fa56'
PRIOR_CONSUMED_CLAIM='9b80b1cf479f442919f097704ac95bdb8304338eadc2053c0efec13f64cf6494'
RAW_LIMIT=1792*MiB
FROZEN_NAMES=('models/cpu_oracle_scratch.py','models/cpu_oracle_control_prerequisite.py',
 'models/cpu_oracle_control_v1.py','models/scratch_allocation_v2.py','models/oracle_boundary_operator_v1.py')
NEW_NAMES=('models/predictor_eot8_adapter_v1.py','models/predictor_eot8_owned_v1.py','models/predictor_eot8_owned_receipt_v1.py','models/predictor_eot8_capture_v1.py','models/predictor_eot8_events_v1.py','models/predictor_eot8_owned_worker_v1.py','models/predictor_eot8_owned_operator_v1.py','scripts/capture_predictor_eot8_owned_v1.py')
REMOTE_NAMES=('models/__init__.py',*FROZEN_NAMES,*NEW_NAMES,'configs/predictor_attention_eot8_proposal_v1.json','configs/predictor_eot8_owned_target_v1.json')
INIT=base.OVERLAY_INIT


def validate_control(c):
    expected={'kind':KIND,'host':'linux10603','parent':'/var/tmp','expected_device':66307,'expected_mount':'/',
        'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f',
        'expected_driver':'580.178.04','proposal_sha256':PROPOSAL_SHA,'paid_compute':False,'research_admitted':False,
        'old_grants_consumed':True,'authorization_present_at_preparation':False,'automatic_retries':0,'host_binding_sha256':HOST_BINDING_SHA,'prior_consumed_control_sha256':PRIOR_CONSUMED_CONTROL,'prior_consumed_claim_sha256':PRIOR_CONSUMED_CLAIM}
    if any(type(c.get(k)) is not type(v) or c.get(k)!=v for k,v in expected.items()):raise ValueError('Changed operator scope.')
    if hashlib.sha256(c['host_binding_text'].encode()).hexdigest()!=HOST_BINDING_SHA or json.loads(c['host_binding_text'])!=c['host_binding']:raise ValueError('Host binding changed.')
    if c['remote_source_sha256'].get('configs/predictor_eot8_owned_target_v1.json')!=HOST_BINDING_SHA:raise ValueError('Host binding overlay changed.')
    p=c['proposal'];r=p['proposed_resource_envelope']
    if hashlib.sha256(c['proposal_text'].encode()).hexdigest()!=PROPOSAL_SHA or json.loads(c['proposal_text'])!=p:
        raise ValueError('Proposal binding changed.')
    if not p['proposal_only'] or p['authorized'] or r['outer_work_seconds']!=2400 or r['torch_allocator_bytes']!=7*GiB:
        raise ValueError('Require separately reviewed proposal, not an old execution grant.')
    if set(c['remote_source_sha256'])!=set(REMOTE_NAMES):raise ValueError('Missing exact overlay source coverage.')
    a=c['archive_manifest']
    if a['sha256']!=PACKET_SHA or a['bytes']!=17192095 or len(a['files'])!=99 or sum(v['bytes'] for v in a['files'].values())!=26446958:
        raise ValueError('Require unchanged original profile packet.')
    for name,item in a['files'].items():
        old.safe_name(name)
        if type(item['bytes']) is not int or item['bytes']<0:raise ValueError('Invalid packet extent.')
    if c['remote_source_sha256']['configs/predictor_attention_eot8_proposal_v1.json']!=PROPOSAL_SHA:raise ValueError('Proposal overlay changed.')


def validate_authorization(a,sha):
    expected={'kind':'predictor-clean-eot8-owned-university-exception-v1','approved':True,'control_sha256':sha,'host':'linux10603',
        'maximum_new_attempts':1,'work_seconds':2400,'worker_seconds':480,'workers':4,'collection_seconds':600,
        'collection_bytes':3*GiB,'write_proof_bytes':5*GiB//2,'setup_bytes':3*GiB,'capture_bytes':9*GiB//4,
        'raw_bytes':1792*MiB,'allocator_bytes':7*GiB,'bootstrap_bytes':2*MiB,'paid_compute':False,
        'reset_old_claims':False,'automatic_retries':0,'prior_consumed_control_sha256':PRIOR_CONSUMED_CONTROL,'prior_consumed_claim_sha256':PRIOR_CONSUMED_CLAIM}
    if set(a)!={*expected,'user_basis'} or any(type(a[k]) is not type(v) or a[k]!=v for k,v in expected.items()):
        raise ValueError('New exact university resource authorization required.')
    if not isinstance(a['user_basis'],str) or not 1<=len(a['user_basis'])<=8192:raise ValueError('Approval basis missing.')


def inventory(folder,*,hashing=False,running=False):
    # Frozen primitive permits regular files and same-directory JSON atomic pairs.
    files,allocated=base.inventory(folder,hashing=hashing,max_bytes=3*GiB,running=running)
    for name,item in files.items():
        limit=65536 if name.endswith(('-loss.pt','-loss.pt.partial')) else 16*MiB if name.endswith(('-input.pt','-input.pt.partial')) else 12*MiB if name.endswith(('targets.pt','targets.pt.partial')) else MiB if name.endswith('mask-signatures.json') else 16*MiB if name.endswith('.log') else (256*MiB if name.endswith(('trace.jsonl','trace.jsonl.partial')) else 3*GiB if name=='space-proof-v2.bin.partial' else 32*MiB if name.endswith(('.pt','.pt.partial')) else 64*MiB)
        if item['bytes']>limit:raise ValueError('Typed file limit exceeded.')
    return files,allocated


def storage(folder,identity,*,setup=False,parent=Path('/var/tmp')):
    folder=Path(folder);st=folder.lstat();pst=Path(parent).stat()
    if (folder.is_symlink() or not stat.S_ISDIR(st.st_mode) or st.st_uid!=os.getuid() or stat.S_IMODE(st.st_mode)!=0o700
            or st.st_dev!=identity['device'] or (pst.st_dev,pst.st_ino)!=(identity['device'],identity['inode'])):raise ValueError('Scratch ownership/device/parent changed.')
    files,allocated=inventory(folder,running=True)
    for current,dirs,names in os.walk(folder,followlinks=False):allocated+=Path(current).stat().st_blocks*512
    limit=3*GiB if setup else 9*GiB//4
    fs=os.statvfs(folder)
    if allocated>limit or fs.f_bavail*fs.f_frsize<GiB or fs.f_favail<1024:raise ValueError('Allocated/free/inode bound.')
    categories=dict.fromkeys(('raw','input','code','metadata','cache','emergency','proof'),0)
    for name,item in files.items():
        if name=='space-proof-v2.bin.partial':category='proof'
        elif name.startswith('original-packet.tar.gz'):category='input'
        elif name.startswith(('code/','overlay/')):category='code'
        elif name.startswith(('cache/','tmp/')):category='cache'
        elif name=='emergency-space.bin':category='emergency'
        elif name.startswith('results/') and ('.pt' in name or 'trace.jsonl' in name):category='raw'
        else:category='metadata'
        categories[category]+=item['bytes']
    caps={'raw':1792*MiB,'input':128*MiB,'code':128*MiB,'metadata':64*MiB,'cache':32*MiB,'emergency':16*MiB,'proof':5*GiB//2 if setup else 0}
    if any(categories[k]>v for k,v in caps.items()):raise ValueError('Storage category exceeded.')
    return {'logical_bytes':sum(x['bytes'] for x in files.values()),'allocated_bytes':allocated,'categories':categories,'ceiling':limit,'files':len(files)}


def extract(archive,spec,destination):
    destination=Path(destination)
    if destination.exists() or old.digest(archive)!=spec['sha256']:raise ValueError('Packet changed or destination reused.')
    with tarfile.open(archive) as tar:
        members=tar.getmembers();names=[old.safe_name(m.name) for m in members]
        if len(names)!=len(set(names)) or set(names)!=set(spec['files']) or any(not m.isfile() for m in members):raise ValueError('Nonregular or unexpected packet member.')
        for m in members:
            item=spec['files'][m.name]
            if m.size!=item['bytes']:raise ValueError('Packet size changed.')
            with tar.extractfile(m) as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=item['sha256']:raise ValueError('Packet member changed.')
        destination.mkdir(mode=0o700)
        for m in members:
            p=destination/m.name;p.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as f,p.open('xb') as out:
                for chunk in iter(lambda:f.read(MiB),b''):out.write(chunk)


def check_worker(folder,index):
    folder=Path(folder);w=json.loads((folder/'worker.json').read_text())
    if not w['complete'] or not w['passed'] or w['local_tiny_analogue'] or w['index']!=index or w.get('kind')!='predictor-clean-eot8-owned-worker-v1' or w.get('execution_host')!='linux10603' or w['phases']!=['EOT8-before','draw-before','draw-profiled','draw-after','EOT8-after'] or not w['adapter_restored']:
        raise ValueError('Incomplete live worker; stop without retry.')
    if verify_ownership(w)!=w.get('ownership_check'):raise ValueError('Graph-owned backward receipt rejected.')
    if len(w['artifacts'])!=10 or not w.get('backend',{}).get('CUDA_qualified'):raise ValueError('Missing complete captures/owned CUDA proof.')
    for item in w['artifacts'].values():
        p=folder/old.safe_name(item['file'])
        if p.stat().st_size!=item['bytes'] or old.digest(p)!=item['sha256']:raise ValueError('Completed capture changed.')
    if not 0<w['wall_seconds']<=480 or not w['model_and_targets_unchanged']:raise ValueError('Worker time/state invalid.')
    return {'worker_sha256':old.digest(folder/'worker.json'),'pid':w['pid'],'index':index}


def stream(folder,sha,output):
    files,_=inventory(folder,hashing=True)
    for name,item in files.items():item['mode']=stat.S_IMODE((Path(folder)/name).stat().st_mode)
    header=old.json_bytes({'kind':KIND,'phase':'collection','control_sha256':sha,'files':files,'logical_bytes':sum(x['bytes'] for x in files.values()),'model_restarted':False})
    if len(header)>MiB:raise ValueError('Collection header cap.')
    output.write(header);output.flush()
    with tarfile.open(fileobj=output,mode='w|',format=tarfile.USTAR_FORMAT) as tar:
        for name,item in sorted(files.items()):
            p=Path(folder)/name
            if old.digest(p)!=item['sha256']:raise ValueError('Collection source changed.')
            m=tarfile.TarInfo(name);m.size=item['bytes'];m.mode=item['mode'];m.mtime=item['mtime_ns']//10**9
            with p.open('rb') as f:tar.addfile(m,f)


def readiness(c):
    if platform.node()!='linux10603' or sys.platform!='linux':raise ValueError('Wrong target host/OS.')
    parent=Path('/var/tmp');st=parent.stat();fs=os.statvfs(parent)
    identity={'host':platform.node(),'parent':str(parent),'device':st.st_dev,'inode':st.st_ino,'home_device':Path.home().stat().st_dev,
        'owner_uid':st.st_uid,'mode':stat.S_IMODE(st.st_mode),**old.mount_for(parent,Path('/proc/self/mountinfo').read_text()),
        'available_bytes':fs.f_bavail*fs.f_frsize,'available_inodes':fs.f_favail,'writable_for_current_user':os.access(parent,os.W_OK|os.X_OK)}
    if parent.is_symlink() or identity['available_bytes']<8*GiB:raise ValueError('Local scratch unavailable.')
    old.validate_readiness(identity,c)
    return identity

def remote(c,auth,incoming,output,sources):
    validate_control(c);sha=hashlib.sha256(old.json_bytes(c)).hexdigest();validate_authorization(auth,sha)
    start=time.monotonic();deadline=start+2400
    handlers={s:signal.getsignal(s) for s in (signal.SIGHUP,signal.SIGTERM,signal.SIGALRM)}
    def stop(*unused):raise RuntimeError('Attended work interrupted; preserve evidence, no retry.')
    previous_umask=os.umask(0o077)
    for s in handlers:signal.signal(s,stop)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()-5))
    try:return _execute(c,auth,incoming,output,sources,sha,start,deadline,stop)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)
        os.umask(previous_umask)


def _execute(c,auth,incoming,output,sources,sha,start,deadline,stop):
    identity=readiness(c);parent=Path(identity['parent']);base.gpu_gate(c)
    folder=Path(tempfile.mkdtemp(prefix=PREFIX,dir=parent));folder.chmod(0o700)
    output.write(old.json_bytes({'kind':KIND,'phase':'scratch-created','scratch_directory':str(folder),'control_sha256':sha}));output.flush()
    state={'kind':KIND,'complete':False,'passed':False,'tasks':[],'started_unix':time.time(),'deadline_unix':time.time()+max(0,deadline-time.monotonic()),'work_seconds':2400,'performance_qualified':False,'research_admitted':False}
    try:
        old.write_json(folder/'control.json',c);old.write_json(folder/'authorization.json',auth);old.write_json(folder/'storage-before.json',identity)
        if set(sources)!=set(REMOTE_NAMES) or sum(map(len,sources.values()))>2*MiB:raise ValueError('Overlay extent changed.')
        for name,data in sources.items():
            if hashlib.sha256(data).hexdigest()!=c['remote_source_sha256'][name]:raise ValueError('Overlay source changed.')
            p=folder/'overlay'/old.safe_name(name);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(0o600)
        with (folder/'emergency-space.bin').open('xb') as f:f.write(os.urandom(16*MiB));f.flush();os.fsync(f.fileno())
        end=min(deadline-5,time.monotonic()+60)
        program='import sys;sys.path.insert(0,'+repr(str(folder/'overlay'))+');from models.scratch_allocation_v2 import bounded_write_proof;bounded_write_proof('+repr(str(folder))+',deadline='+repr(end)+')'
        proof=base.primitives.supervise([sys.executable,'-B','-u','-c',program],folder,role='proof',deadline=end,gate=lambda:storage(folder,identity,setup=True),file_limit=3*GiB)
        if proof['exit_status']!=0:raise RuntimeError('Full storage proof failed before input archive.')
        base.primitives.validate_proof(folder)
        old.receive_archive(incoming,folder/'original-packet.tar.gz',17192095,PACKET_SHA,deadline-5)
        extract(folder/'original-packet.tar.gz',c['archive_manifest'],folder/'code')
        if time.monotonic()-start>180:raise TimeoutError('Setup time exhausted.')
        for name in ('tmp','cache','cache/cuda','results'):(folder/name).mkdir(mode=0o700,exist_ok=True)
        old.write_json(folder/'session.json',state)
        for i in range(4):
            for j,task in enumerate(state['tasks']):
                if check_worker(folder/'results'/str(j),j)!=task:raise ValueError('Earlier worker evidence changed.')
            base.gpu_gate(c);entry=folder/'overlay/scripts/capture_predictor_eot8_owned_v1.py'
            end=min(deadline-125,time.monotonic()+480)
            if end<=time.monotonic():raise TimeoutError('Continuing work allowance exhausted.')
            old.write_json(folder/('worker-plan-'+str(i)+'.json'),{'index':i,'control_sha256':sha,'deadline_unix':time.time()+end-time.monotonic()})
            command=[sys.executable,'-I','-B','-u',str(entry),'--scratch',str(folder),'--control-sha256',sha,'--index',str(i)]
            last=[0.0];samples=[]
            def gate():
                row=storage(folder,identity)
                if time.monotonic()-last[0]>=5:
                    children=old.descendants(os.getpid());owned=[]
                    for pid in children:
                        try:argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
                        except FileNotFoundError:continue
                        if str(entry).encode() in argv:owned.append(pid)
                    if len(owned)>1:raise ValueError('Multiple scientific processes.')
                    samples.append(base.gpu_gate(c,owned[0] if owned else None));last[0]=time.monotonic()
                return row
            env={'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES','0'),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
                 'TMPDIR':str(folder/'tmp'),'XDG_CACHE_HOME':str(folder/'cache'),'CUDA_CACHE_PATH':str(folder/'cache/cuda'),'CUDA_CACHE_MAXSIZE':str(32*MiB)}
            if env['CUDA_VISIBLE_DEVICES'] in ('','-1'):raise ValueError('CUDA visibility hidden; refuse.')
            child=base.primitives.supervise(command,folder,role='worker-'+str(i),deadline=end,gate=gate,file_limit=256*MiB,env=env)
            old.write_json(folder/('gpu-samples-'+str(i)+'.json'),samples);base.gpu_gate(c)
            if child['exit_status']!=0:raise RuntimeError('Worker refused; no retry or next worker.')
            state['tasks'].append(check_worker(folder/'results'/str(i),i));old.write_json(folder/'session.json',state,replace=(folder/'session.json').exists())
        if len({x['pid'] for x in state['tasks']})!=4:raise ValueError('Workers were not fresh processes.')
        state.update(complete=True,passed=True)
    except BaseException as exc:state.update(error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        state['wall_seconds']=time.monotonic()-start
        old.write_json(folder/'session.json',state,replace=(folder/'session.json').exists())
        if state['wall_seconds']>2400:
            state.update(passed=False,error='Outer work deadline exceeded.')
            old.write_json(folder/'session.json',state,replace=True)
    # Same SSH collection always follows work, including refusals; no new model starts.
    signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,600)
    try:stream(folder,sha,output)
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    return 0 if state['passed'] else 1


def main(control64,sources64,mode='run',scratch=None):
    c=json.loads(base64.b64decode(control64,validate=True));sources={n:base64.b64decode(v,validate=True) for n,v in json.loads(base64.b64decode(sources64,validate=True)).items()}
    validate_control(c)
    # Authorization is a separate first input line, never inferred from a proposal.
    auth=json.loads(sys.stdin.buffer.readline(65537))
    if mode!='run' or scratch is not None:raise ValueError('Only original same-SSH work/collection is covered; separate retrieval needs review.')
    return remote(c,auth,sys.stdin.buffer,sys.stdout.buffer,sources)
