"""Separate one-attempt native boundary operator; no prior grant authorizes it."""
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
from models import predictor_eot8_record_v2 as records
from models import predictor_boundary_record_v1 as boundary
from models.predictor_boundary_contract_v1 import KIND,PREFIX,CONTRACT_SHA,PROPOSAL_SHA,PRIOR_CLAIM_SHA,PACKET_SHA,MiB,GiB
old=base.old
FROZEN_NAMES=('models/cpu_oracle_scratch.py','models/cpu_oracle_control_prerequisite.py','models/cpu_oracle_control_v1.py','models/scratch_allocation_v2.py','models/oracle_boundary_operator_v1.py')
REMOTE_NAMES=('models/__init__.py',*FROZEN_NAMES,'models/predictor_eot8_owned_receipt_v1.py','models/predictor_eot8_record_v2.py',
    'models/predictor_eot8_adapter_v1.py','models/predictor_eot8_owned_v1.py','models/predictor_eot8_capture_v1.py',
    'models/predictor_boundary_contract_v1.py','models/predictor_boundary_record_v1.py','models/predictor_boundary_observer_v1.py','models/predictor_boundary_worker_v1.py',
    'models/predictor_boundary_operator_v1.py','scripts/capture_predictor_boundary_v1.py','configs/predictor_attention_eot8_proposal_v1.json',
    'configs/predictor_gradient_attribution_v1.json','configs/predictor_boundary_target_v1.json')
INIT=base.OVERLAY_INIT


def anchor_metadata(contract,*,tiny=False,baseline=None,anchors=None):
    return {'kind':'predictor-boundary-value-anchors-v1','local_tiny':tiny,'contract_sha256':CONTRACT_SHA,
            'anchors':contract['anchors'] if anchors is None else anchors,
            'baseline':contract['approved_target_source_baseline'] if baseline is None else baseline}


def validate_control(c):
    expected={'kind':KIND,'host':'linux10603','parent':'/var/tmp','expected_device':66307,'expected_mount':'/',
        'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f','expected_driver':'580.178.04',
        'contract_sha256':CONTRACT_SHA,'proposal_sha256':PROPOSAL_SHA,'prior_consumed_claim_sha256':PRIOR_CLAIM_SHA,
        'paid_compute':False,'research_admitted':False,'automatic_retries':0,'old_grants_consumed':True,'authorization_present_at_preparation':False}
    if any(type(c.get(k)) is not type(v) or c.get(k)!=v for k,v in expected.items()):raise ValueError('Exact separate boundary scope required.')
    for key,want in [('contract',CONTRACT_SHA),('proposal',PROPOSAL_SHA)]:
        text=c[key+'_text']
        if hashlib.sha256(text.encode()).hexdigest()!=want or json.loads(text)!=c[key]:raise ValueError('Reviewed boundary proposal/contract changed.')
    p=c['proposal'];r=p['resource_proposal']
    if not p['proposal_only'] or p['authorized'] or r['whole_work_seconds']!=1200 or r['worker_seconds']!=420 or r['raw_bytes']!=512*MiB or r['workers']!=2:raise ValueError('Inert reviewed resource proposal required.')
    if set(c['remote_source_sha256'])!=set(REMOTE_NAMES):raise ValueError('Exact overlay coverage required.')
    if c['remote_source_sha256']['configs/predictor_boundary_target_v1.json']!=CONTRACT_SHA or c['remote_source_sha256']['configs/predictor_gradient_attribution_v1.json']!=PROPOSAL_SHA:raise ValueError('Source-bound plan changed.')
    a=c['archive_manifest']
    if a['sha256']!=PACKET_SHA or a['bytes']!=17192095 or len(a['files'])!=99 or sum(i['bytes'] for i in a['files'].values())!=26446958:raise ValueError('Require unchanged original packet.')
    expected_files={}
    for i in (0,1):
        anchor=c['contract']['anchors'][str(i)]
        if anchor['source_worker_passed'] is not (i==0) or anchor['source_worker_complete'] is not (i==0) or not anchor['value_only_not_executable_reference']:raise ValueError('Known failed values cannot be a passing reference.')
        for name,item in anchor['artifacts'].items():expected_files[item['file']]={'bytes':item['bytes'],'sha256':item['sha256']}
    data=old.json_bytes(anchor_metadata(c['contract']))
    expected_files['ANCHORS.json']={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    s=c['anchors_manifest']
    if s['files']!=expected_files or type(s['bytes']) is not int or not 0<s['bytes']<64*MiB or len(s['sha256'])!=64:raise ValueError('Exact four value anchors and metadata required.')
    for spec in (a,s):
        for n,item in spec['files'].items():
            old.safe_name(n)
            if type(item['bytes']) is not int or item['bytes']<0:raise ValueError('Unsafe input extent.')


def validate_authorization(a,sha):
    expected={'kind':'predictor-boundary-university-exception-v1','approved':True,'control_sha256':sha,'host':'linux10603',
        'maximum_new_attempts':1,'work_seconds':1200,'workers':2,'worker_seconds':420,'collection_seconds':600,'collection_bytes':3*GiB,
        'write_proof_bytes':5*GiB//2,'setup_bytes':3*GiB,'capture_bytes':9*GiB//4,'raw_bytes':512*MiB,'metadata_bytes':64*MiB,
        'allocator_bytes':7*GiB,'bootstrap_bytes':2*MiB,'automatic_retries':0,'paid_compute':False,'reset_old_claims':False,
        'prior_consumed_claim_sha256':PRIOR_CLAIM_SHA}
    if set(a)!={*expected,'user_basis'} or any(type(a[k]) is not type(v) or a[k]!=v for k,v in expected.items()):raise ValueError('New exact boundary university authorization required.')
    if not isinstance(a['user_basis'],str) or not 1<=len(a['user_basis'])<=8192:raise ValueError('Explicit approval basis required.')


def inventory(folder,*,hashing=False,running=False):
    files,allocated=base.inventory(folder,hashing=hashing,max_bytes=3*GiB,running=running)
    for name,item in files.items():
        leaf=Path(name).name;stem=leaf[:-8] if leaf.endswith('.partial') else leaf
        limit=128*MiB if name.endswith('.tar.gz') else 16*MiB if stem.endswith('.pt') else 16*MiB if stem.endswith('.log') else MiB if stem.endswith('.json') else 64*MiB
        if name=='space-proof-v2.bin.partial':limit=3*GiB
        if name.startswith('results/') and stem in boundary.LIMITS:limit=min(limit,boundary.LIMITS[stem])
        if item['bytes']>limit:raise ValueError('Boundary typed-file ceiling exceeded.')
    return files,allocated


def storage(folder,identity,*,setup=False,parent=Path('/var/tmp')):
    folder=Path(folder);st=folder.lstat();pst=Path(parent).stat()
    if (folder.is_symlink() or not stat.S_ISDIR(st.st_mode) or st.st_uid!=os.getuid() or stat.S_IMODE(st.st_mode)!=0o700
            or st.st_dev!=identity['device'] or (pst.st_dev,pst.st_ino)!=(identity['device'],identity['inode'])):raise ValueError('Private boundary scratch changed.')
    files,allocated=inventory(folder,running=True)
    for current,dirs,names in os.walk(folder,followlinks=False):allocated+=Path(current).stat().st_blocks*512
    ceiling=3*GiB if setup else 9*GiB//4;fs=os.statvfs(folder)
    if allocated>ceiling or fs.f_bavail*fs.f_frsize<GiB or fs.f_favail<1024:raise ValueError('Boundary allocated/free/inode limit.')
    categories=dict.fromkeys(('raw','input','code','metadata','cache','emergency','proof'),0)
    for n,item in files.items():
        if n=='space-proof-v2.bin.partial':category='proof'
        elif n.startswith(('original-packet.tar.gz','anchors.tar.gz','anchors/')):category='input'
        elif n.startswith(('code/','overlay/')):category='code'
        elif n.startswith(('tmp/','cache/')):category='cache'
        elif n=='emergency-space.bin':category='emergency'
        elif n.startswith('results/') and '.pt' in n:category='raw'
        else:category='metadata'
        categories[category]+=item['bytes']
    caps={'raw':512*MiB,'input':128*MiB,'code':128*MiB,'metadata':64*MiB,'cache':32*MiB,'emergency':16*MiB,'proof':5*GiB//2 if setup else 0}
    if any(categories[k]>v for k,v in caps.items()):raise ValueError('Boundary category ceiling exceeded.')
    return {'logical_bytes':sum(i['bytes'] for i in files.values()),'allocated_bytes':allocated,'categories':categories,'ceiling':ceiling,'files':len(files)}


def verify_capture(folder,index,*,control_sha256,live,require_passed=True):
    if type(live) is not bool or (not live and platform.node() in ('linux10603','linux10609')):raise ValueError('Exact local/live capture domain required.')
    folder=Path(folder);w=boundary.load(folder,control_sha256=control_sha256,require_passed=require_passed)
    if (w['index']!=index or w['local_tiny'] is not (not live) or live and w['execution_host']!='linux10603'
            or not w.get('capture_complete') or not w.get('reproduces_anchor') or not w.get('observer_restored')
            or not w.get('adapter_restored') or not w.get('model_unchanged') or not 0<w.get('wall_seconds',0)<=420
            or w.get('backend_kernel_qualified') is not False or w.get('full_five_phase_qualified') is not False):raise ValueError('Incomplete, wrong-domain or mislabelled boundary capture.')
    if w['passed'] and w.get('numerical_passed') is not True:raise ValueError('Passing capture lacks original numerical gates.')
    if not w['passed'] and (index!=1 or w.get('numerical_passed') is not False or w.get('error_type')!='ValueError' or w.get('error')!='Original numerical gate rejected; no tolerance change.'):
        raise ValueError('Only a fully captured canonical numerical refusal can be attribution evidence.')
    mask=w.get('mask_indices',{});mp=folder/'mask-indices.json'
    if mask.get('file')!='mask-indices.json' or mp.is_symlink() or mp.stat().st_size!=mask.get('bytes') or old.digest(mp)!=mask.get('sha256') or mp.stat().st_size>MiB:raise ValueError('Mask indices changed.')
    indices=json.loads(mp.read_text());tokens,features=(2048,1024) if live else (32,128)
    if len(indices)!=8 or any(len(draw)!=2 for draw in indices):raise ValueError('Complete original mask bank required.')
    for d,draw in enumerate(indices):
        for f,pair in enumerate(draw):
            if set(pair)!={'context','target'}:raise ValueError('Mask fields changed.')
            for name,j in [('context',0),('target',1)]:
                data=pair[name]
                if len(data)!=1 or not data[0] or any(type(x) is not int or not 0<=x<tokens for x in data[0]) or len(set(data[0]))!=len(data[0]):raise ValueError('Invalid actual mask indices.')
                info=w['baseline']['masks'][d][f][j]
                packed=struct.pack('<'+'q'*len(data[0]),*data[0])
                if info['shape']!=[1,len(data[0])] or info['dtype']!='torch.int64' or hashlib.sha256(packed).hexdigest()!=info['sha256']:raise ValueError('Mask byte identities changed.')
            if set(pair['context'][0])&set(pair['target'][0]):raise ValueError('Original mask partition changed.')
    def typed(item,filename,shape):
        p=folder/filename;source=item.get('source',{})
        if (item.get('file')!=filename or p.is_symlink() or p.stat().st_size!=item.get('bytes') or not 0<p.stat().st_size<=16*MiB or old.digest(p)!=item.get('sha256')
                or source.get('shape')!=shape or source.get('dtype')!='torch.float32' or source.get('device')!=('cuda:0' if live else 'cpu') or item.get('saved_before_numerical_gate') is not True):raise ValueError('Typed actual boundary capture changed.')
    if set(w['artifacts'])!={'targets','loss','input'}:raise ValueError('All native returns and target required.')
    typed(w['artifacts']['targets'],'targets.pt',[1,tokens,features]);typed(w['artifacts']['loss'],'loss.pt',[])
    typed(w['artifacts']['input'],'input.pt',[1,3,16,256,256] if live else [1,3,4,16,16])
    observations=w.get('observations',[])
    if len(observations)!=16:raise ValueError('All sixteen native observation pairs required.')
    for i,row in enumerate(observations):
        d,f=divmod(i,2)
        if (row.get('index')!=i or row.get('draw')!=d or row.get('family')!=f or row.get('pid')!=w['pid'] or not row.get('hook_removed')
                or type(row.get('forward_thread')) is not int or type(row.get('cotangent_thread')) is not int):raise ValueError('Observer identity/cleanup incomplete.')
        for kind in ('prediction','cotangent'):typed(row[kind],f'draw-{d:02d}-family-{f}-{kind}.pt',[1,len(indices[d][f]['target'][0]),features])
    return {'worker_sha256':old.digest(folder/records.HEADER),'marker_sha256':old.digest(folder/(records.COMPLETE if w['passed'] else records.FAILURE)),
            'calls_sha256':old.digest(folder/records.CALLS),'scopes_sha256':old.digest(folder/records.SCOPES),
            'mask_sha256':old.digest(mp),'index':index,'pid':w['pid'],'numerical_passed':w['numerical_passed'],
            'capture_complete':True,'attribution_only':True,'CUDA_backend_qualified':False,'research_admitted':False}


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
    start=time.monotonic();deadline=start+1200
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
    state={'kind':KIND,'complete':False,'passed':False,'tasks':[],'started_unix':time.time(),'deadline_unix':time.time()+max(0,deadline-time.monotonic()),'work_seconds':1200,'performance_qualified':False,'research_admitted':False,'capture_complete':False}
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
        spec=c['anchors_manifest']
        old.receive_archive(incoming,folder/'anchors.tar.gz',spec['bytes'],spec['sha256'],deadline-5)
        extract(folder/'anchors.tar.gz',spec,folder/'anchors')
        if time.monotonic()-start>180:raise TimeoutError('Setup time exhausted.')
        for name in ('tmp','cache','cache/cuda','results'):(folder/name).mkdir(mode=0o700,exist_ok=True)
        old.write_json(folder/'session.json',state)
        for i in range(2):
            for j,task in enumerate(state['tasks']):
                if verify_capture(folder/'results'/str(j),j,control_sha256=sha,live=True)!=task:raise ValueError('Earlier worker evidence changed.')
            base.gpu_gate(c);entry=folder/'overlay/scripts/capture_predictor_boundary_v1.py'
            end=min(deadline-125,time.monotonic()+420)
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
            if child['exit_status']!=0:
                if i==1:
                    task=verify_capture(folder/'results'/str(i),i,control_sha256=sha,live=True,require_passed=False)
                    if task['numerical_passed']:raise ValueError('Failed process may not claim numerical success.')
                    state['tasks'].append(task);state['capture_complete']=len(state['tasks'])==2
                raise RuntimeError('Worker refused; no retry or next worker.')
            state['tasks'].append(verify_capture(folder/'results'/str(i),i,control_sha256=sha,live=True));old.write_json(folder/'session.json',state,replace=(folder/'session.json').exists())
        if len({x['pid'] for x in state['tasks']})!=2:raise ValueError('Workers were not fresh processes.')
        state.update(complete=True,passed=True,capture_complete=True)
    except BaseException as exc:state.update(error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        state['wall_seconds']=time.monotonic()-start
        old.write_json(folder/'session.json',state,replace=(folder/'session.json').exists())
        if state['wall_seconds']>1200:
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
