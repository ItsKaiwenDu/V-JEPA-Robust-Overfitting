"""Cold bounded supervisor for one new eight-gradient pretrained PGD diagnostic."""
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import tempfile
import time
from types import FunctionType
from models import predictor_forward_preserving_operator_v1 as frozen
from models import predictor_pgd_target_records_v1 as records
from models import scratch_inventory_observer_v1 as observed

base,old,envelope=frozen.base,frozen.old,frozen.envelope
MiB,GiB,INIT=frozen.MiB,frozen.GiB,frozen.INIT
ATTACK_INIT=b"from pathlib import Path\n__path__.append(str(Path(__file__).resolve().parents[2] / 'code/attacks'))\n"
VIRTUAL={'models/__init__.py':INIT,'attacks/__init__.py':ATTACK_INIT}
KIND='pretrained-pgd2-r2-iterated-operator-v1'
AUTH_KIND='pretrained-pgd2-r2-iterated-university-exception-v1'
PREFIX='vjepa-pretrained-pgd-v1-20261114-'
CONTRACT='configs/predictor_pgd_target_v1.json'
CONTRACT_SHA='c7bb7f3af5e6b9e65ac746177b657ae966e2bc50171a6e5b516e30b848252960'
PACKET_SHA=frozen.PACKET_SHA
BOOT_NAMES=(*frozen.FROZEN_NAMES,'models/predictor_forward_preserving_records_v1.py',
            'models/predictor_forward_preserving_operator_v1.py','models/scratch_inventory_observer_v1.py',
            'models/predictor_pgd_target_records_v1.py','models/predictor_pgd_target_operator_v1.py')
REMOTE_NAMES=tuple(dict.fromkeys(('models/__init__.py','attacks/__init__.py',*BOOT_NAMES,
    'models/predictor_eot8_adapter_v1.py','models/predictor_eot8_owned_v1.py',
    'models/predictor_forward_preserving_cpu_v1.py','models/predictor_forward_preserving_upstream_v1.py',
    'models/predictor_forward_preserving_target_v1.py','models/predictor_pgd_target_worker_v1.py',
    'attacks/pgd_pretrained_v1.py','scripts/capture_predictor_pgd_target_v1.py',CONTRACT)))
LOCAL_EXTRA=('models/predictor_pgd_target_client_v1.py','models/private_collection_v1.py',
             'models/predictor_eot8_record_client_v2.py','scripts/prepare_predictor_pgd_target_v1.py',
             'scripts/run_predictor_pgd_target_v1.py','tests/test_predictor_pgd_target_v1.py')


def exact(a,b):return old.json_bytes(a)==old.json_bytes(b)


def validate_control(c):
    expected={'kind':KIND,'host':'linux10603','parent':'/var/tmp','expected_device':66307,'expected_mount':'/',
              'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f',
              'expected_driver':'580.178.04','expected_ssh_fingerprint':base.SSH_FINGERPRINT,
              'contract_sha256':CONTRACT_SHA,'paid_compute':False,'research_admitted':False,
              'authorization_present_at_preparation':False,'automatic_retries':0}
    more={'contract','contract_text','resource','repository','local_handoff_directory','archive_path','archive_manifest',
          'known_hosts_sha256','remote_source_sha256','local_source_sha256','prior_inputs_manifest','prior_inputs_manifest_sha256','host_binding'}
    if set(c)!=set(expected)|more or not exact({k:c[k] for k in expected},expected):raise ValueError('Exact new iterated target control required.')
    p=c['contract'];r=c['resource'];w=p['workload']
    if (hashlib.sha256(c['contract_text'].encode()).hexdigest()!=CONTRACT_SHA or not exact(json.loads(c['contract_text']),p)
            or p['authorized'] is not False or p['new_target_execution_permitted'] is not False
            or not exact(r,p['resource']) or r['work_seconds']!=2400 or r['worker_seconds']!=1920 or r['workers']!=1
            or r['allocator_bytes']!=7*GiB or w['total_new_gradients_max']!=8 or w['candidate_gradients_max']!=4
            or w['same_input_original_gradients_max']!=4 or w['completed_clean_cohort_repeats']!=0
            or w['fresh_rollout_replay_in_this_attempt'] is not False
            or not exact(c['host_binding'],{'prior_consumed_files':p['prior_consumed_files']})):
        raise ValueError('Frozen plan/resources/consumed lineage changed.')
    if (set(c['remote_source_sha256'])!=set(REMOTE_NAMES)
            or set(c['local_source_sha256'])!=(set(REMOTE_NAMES)-set(VIRTUAL))|set(LOCAL_EXTRA)
            or c['remote_source_sha256'][CONTRACT]!=CONTRACT_SHA):raise ValueError('Exact overlay/local source coverage required.')
    spec=c['archive_manifest']
    if spec['sha256']!=PACKET_SHA or spec['bytes']!=17192095 or len(spec['files'])!=99 or sum(v['bytes'] for v in spec['files'].values())!=26446958:
        raise ValueError('Unchanged one-training-clip packet required.')
    for name in spec['files']:old.safe_name(name)


def authorization_template(c,sha,basis):
    return {'kind':AUTH_KIND,'approved':True,'control_sha256':sha,'host':'linux10603','maximum_new_attempts':1,
            'resource':c['resource'],'roles':c['contract']['worker_roles'],'maximum_new_gradient_objectives':8,
            'candidate_gradients':4,'same_input_original_gradients':4,'profiled_existing_objective_indices':[0,1],
            'additional_profile_objectives':0,'automatic_retries':0,'paid_compute':False,'cleanup':False,
            'reset_old_claims':False,'private_key_upload':False,'held_out_pixels':False,'user_basis':basis}


def validate_authorization(a,sha,c=None):
    basis=a.get('user_basis') if isinstance(a,dict) else None
    if c is None or not isinstance(basis,str) or not 1<=len(basis)<=8192 or not exact(a,authorization_template(c,sha,basis)):
        raise ValueError('One new exact scoped iterated university authorization required.')


def category(name):
    plain=name.removesuffix('.partial')
    if plain=='space-proof-v2.bin':return 'proof'
    if plain=='original-packet.tar.gz':return 'input'
    if name.startswith(('code/','overlay/')):return 'code'
    if name.startswith(('cache/','tmp/')):return 'cache'
    if name=='emergency-space.bin':return 'emergency'
    if name.startswith('results/') and not name.startswith('results/0/'):raise ValueError('Only results/0 is allowed.')
    if plain.startswith('results/0/') and plain.endswith('.npy'):return 'raw'
    return 'metadata'


def inventory(folder,*,hashing=False,running=False):
    files,allocated=observed.inventory(folder,hashing=hashing,max_bytes=3*GiB,running=running)
    traces=0
    for name,item in files.items():
        category(name);plain=name.removesuffix('.partial');leaf=Path(plain).name
        cap=(5*GiB//2 if plain=='space-proof-v2.bin' else 17192095 if plain=='original-packet.tar.gz'
             else 65536 if leaf=='loss.npy' else MiB if leaf in ('masks.npy','scoring-masks.npy')
             else 12*MiB if leaf=='targets.npy' else 16*MiB if leaf.endswith(('.npy','.log'))
             else 4*MiB if leaf=='backend.json' else records.LIMITS.get(leaf,64*MiB))
        if leaf=='backend.json':traces+=item['bytes']
        if item['bytes']>cap:raise ValueError('Typed iterated evidence file cap exceeded: '+name)
    if traces>8*MiB:raise ValueError('Combined attention backend aggregate cap exceeded.')
    return files,allocated


def storage(folder, identity, *, setup=False, parent=Path('/var/tmp')):
    folder=Path(folder);st=folder.lstat();pst=Path(parent).stat()
    if (folder.is_symlink() or not stat.S_ISDIR(st.st_mode) or st.st_uid!=os.getuid()
            or stat.S_IMODE(st.st_mode)!=0o700 or st.st_dev!=identity['device']
            or (pst.st_dev,pst.st_ino)!=(identity['device'],identity['inode'])):
        raise ValueError('Scratch ownership/device/parent changed.')
    files,allocated=inventory(folder,running=True)
    for current,dirs,names in os.walk(folder,followlinks=False):
        info=Path(current).lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_dev!=st.st_dev:raise ValueError('Directory type/device changed.')
        allocated+=info.st_blocks*512
    limit=3*GiB if setup else 9*GiB//4;fs=os.statvfs(folder)
    logical=sum(i['bytes'] for i in files.values())
    if logical>limit or allocated>limit or fs.f_bavail*fs.f_frsize<GiB or fs.f_favail<1024:
        raise ValueError('Aggregate logical/allocated/free-space/inode ceiling.')
    categories=dict.fromkeys(('raw','input','code','metadata','cache','emergency','proof'),0)
    for name,item in files.items():categories[category(name)]+=item['bytes']
    caps={'raw':1792*MiB,'input':128*MiB,'code':128*MiB,'metadata':64*MiB,'cache':32*MiB,
          'emergency':16*MiB,'proof':5*GiB//2 if setup else 0}
    if any(categories[k]>v for k,v in caps.items()):raise ValueError('Shared replay storage category exceeded.')
    return {'logical_bytes':logical,'allocated_bytes':allocated,'categories':categories,'ceiling':limit,'files':len(files)}


readiness,extract=frozen.readiness,frozen.extract
stream=FunctionType(envelope.stream.__code__,dict(envelope.stream.__globals__,inventory=inventory,KIND=KIND))


def check_worker(folder,*,control_sha256):return records.verify_worker(folder,control_sha256=control_sha256)


def _execute(c,auth,incoming,output,sources,sha,start,deadline,stop):
    identity=readiness(c);base.gpu_gate(c);folder=Path(tempfile.mkdtemp(prefix=PREFIX,dir=identity['parent']));folder.chmod(0o700)
    output.write(old.json_bytes({'kind':KIND,'phase':'scratch-created','scratch_directory':str(folder),'control_sha256':sha}));output.flush()
    state={'kind':KIND,'complete':False,'passed':False,'tasks':[],'work_seconds':2400,'started_unix':time.time(),
           'deadline_unix':time.time()+max(0,deadline-time.monotonic()),'research_admitted':False,'performance_qualified':False}
    try:
        old.write_json(folder/'control.json',c);old.write_json(folder/'authorization.json',auth);old.write_json(folder/'storage-before.json',identity)
        if set(sources)!=set(REMOTE_NAMES) or sum(map(len,sources.values()))>2*MiB:raise ValueError('Exact overlay extent required.')
        for name,data in sources.items():
            if hashlib.sha256(data).hexdigest()!=c['remote_source_sha256'][name]:raise ValueError('Overlay hash changed.')
            path=folder/'overlay'/old.safe_name(name);path.parent.mkdir(mode=0o700,parents=True,exist_ok=True);path.write_bytes(data);path.chmod(0o600)
        for name in BOOT_NAMES:
            module=sys.modules.get(name[:-3].replace('/','.'))
            if module is not None and module.__file__==name:module.__file__=str(folder/'overlay'/name)
        with (folder/'emergency-space.bin').open('xb') as f:f.write(os.urandom(16*MiB));f.flush();os.fsync(f.fileno())
        end=min(deadline-5,time.monotonic()+60)
        program='import sys;sys.path.insert(0,'+repr(str(folder/'overlay'))+');from models.scratch_allocation_v2 import bounded_write_proof;bounded_write_proof('+repr(str(folder))+',deadline='+repr(end)+')'
        proof=base.primitives.supervise([sys.executable,'-B','-u','-c',program],folder,role='proof',deadline=end,gate=lambda:storage(folder,identity,setup=True),file_limit=3*GiB)
        if proof['exit_status']!=0:raise RuntimeError('Full storage proof failed before scientific work.')
        base.primitives.validate_proof(folder)
        old.receive_archive(incoming,folder/'original-packet.tar.gz',17192095,PACKET_SHA,deadline-5)
        extract(folder/'original-packet.tar.gz',c['archive_manifest'],folder/'code')
        if time.monotonic()-start>180:raise TimeoutError('Setup allowance exhausted.')
        for name in ('tmp','cache','cache/cuda','results'):(folder/name).mkdir(mode=0o700,exist_ok=True)
        old.write_json(folder/'session.json',state)
        base.gpu_gate(c);entry=folder/'overlay/scripts/capture_predictor_pgd_target_v1.py'
        end=min(deadline-125,time.monotonic()+1920)
        if end<=time.monotonic():raise TimeoutError('Continuing work clock exhausted.')
        old.write_json(folder/'worker-plan-0.json',{'index':0,'control_sha256':sha,'deadline_unix':time.time()+end-time.monotonic()})
        command=[sys.executable,'-I','-B','-u',str(entry),'--scratch',str(folder),'--control-sha256',sha,'--index','0']
        last,samples=[0.],[]
        def gate():
            info=storage(folder,identity)
            if time.monotonic()-last[0]>=5:
                owned=[]
                for pid in old.descendants(os.getpid()):
                    try:argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
                    except FileNotFoundError:continue
                    if str(entry).encode() in argv:owned.append(pid)
                if len(owned)>1:raise ValueError('Concurrent scientific workers refused.')
                samples.append(base.gpu_gate(c,owned[0] if owned else None));last[0]=time.monotonic()
            return info
        env={'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES','0'),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
             'TMPDIR':str(folder/'tmp'),'XDG_CACHE_HOME':str(folder/'cache'),'CUDA_CACHE_PATH':str(folder/'cache/cuda'),'CUDA_CACHE_MAXSIZE':str(32*MiB)}
        if env['CUDA_VISIBLE_DEVICES'] in ('','-1'):raise ValueError('CUDA hidden; refuse.')
        child=base.primitives.supervise(command,folder,role='worker-0',deadline=end,gate=gate,file_limit=16*MiB,env=env)
        old.write_json(folder/'gpu-samples-0.json',samples);base.gpu_gate(c)
        if child['exit_status']!=0:raise RuntimeError('Iterated worker failed; no retry or next worker.')
        state['tasks']=[check_worker(folder/'results/0',control_sha256=sha)];state.update(complete=True,passed=True)
    except BaseException as error:
        state.update(complete=False,passed=False,error_type=type(error).__name__,error=str(error)[:2048])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);state['wall_seconds']=time.monotonic()-start
        if state['wall_seconds']>2400:state.update(complete=False,passed=False,error='Outer work deadline exceeded.')
        old.write_json(folder/'session.json',state,replace=(folder/'session.json').exists())
    signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,600)
    try:stream(folder,sha,output)
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    return 0 if state['passed'] else 1


def remote(c,auth,incoming,output,sources):
    validate_control(c);sha=hashlib.sha256(old.json_bytes(c)).hexdigest();validate_authorization(auth,sha,c)
    runner=FunctionType(envelope.remote.__code__,dict(envelope.remote.__globals__,validate_control=validate_control,
                        validate_authorization=lambda a,h:validate_authorization(a,h,c),_execute=_execute))
    return runner(c,auth,incoming,output,sources)


def main(control64,sources64,mode='run',scratch=None):
    if mode!='run' or scratch is not None:raise ValueError('Only the original attended SSH with same-session collection; no separate retrieval.')
    c=json.loads(base64.b64decode(control64,validate=True));sources={n:base64.b64decode(v,validate=True) for n,v in json.loads(base64.b64decode(sources64,validate=True)).items()}
    validate_control(c);auth=json.loads(sys.stdin.buffer.readline(65537))
    return remote(c,auth,sys.stdin.buffer,sys.stdout.buffer,sources)
