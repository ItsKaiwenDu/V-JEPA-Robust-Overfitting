"""One separately authorized candidate replay; no old-cohort scheduling."""
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
from types import FunctionType
from models import predictor_forward_preserving_operator_v1 as frozen
from models import predictor_replay_capsule_v1 as capsule
from models import predictor_missing_replay_records_v1 as records
from models import scratch_inventory_observer_v1 as observed

base, old, envelope = frozen.base, frozen.old, frozen.envelope
MiB, GiB, INIT = frozen.MiB, frozen.GiB, frozen.INIT
KIND = 'forward-preserving-missing-candidate-replay-operator-v1'
AUTH_KIND = 'forward-preserving-missing-candidate-replay-exception-v1'
PREFIX = 'vjepa-missing-candidate-replay-v1-20261114-'
CONTRACT = 'configs/predictor_missing_replay_handoff_v1.json'
CONTRACT_SHA = '5e7988071b80fb1b9f0c4cdbc810d503e17165487a173938c6c12825e1982315'
PACKET_SHA = frozen.PACKET_SHA
REFERENCE_SHA = 'd3853c4d57679d76b6f452ca0ef0ccd7125cd3c9e639f60e81b32a810e1c91aa'
REFERENCE_BYTES = 116903177
REFERENCE_EXTENTS = {'arrays/0a19dd216f4cf2d7512eafa67b691f6b34c90125660c2340b4d09009b021960a.npy': 12583040, 'arrays/0b4fd5cfcb74a6d95e4c0144208e8ec2e3cff0eea9232bfaf0016fc16080a183.npy': 4882560, 'arrays/11242756eb598a4c5859e07ad7fbbfb433de586d3c0a1c2099f477ba3a54dea4.npy': 4161664, 'arrays/12f10cde53e5c82ca3084eba23255ec160718c7bbf20bb3cb48a06de3bf9aa04.npy': 6750336, 'arrays/1a16f664d38213a6cfa89893cc78b904862f695c5073a62a47745c4cd903e42c.npy': 7209088, 'arrays/2a97040450b20ba4dee9f127128321abecd87a172e9d4512aa3e6f06cffa1a85.npy': 5963904, 'arrays/2b022eb86bd65360479bffba3c3c4882163ed8b349bf8384875f9dcae8fef71a.npy': 262272, 'arrays/2ca8223c745b8e12c69360552f63c814b0e613117f519806b7f9737e166d1e4c.npy': 4784256, 'arrays/2f1675b804ccd84990c03485ee4abb582263a838231f8c86560dc54f244a8d26.npy': 7340160, 'arrays/3477d962c8caf5e0284c4f817329880803966b8168881b6c070aec2038c7ce19.npy': 6389888, 'arrays/3e879eac78957909036c633570eec83bb610e83576dcfb88f1720490bc10b52a.npy': 6291584, 'arrays/3e9c93c27a560f7326c9d63499314bc382d4107de049e7d725b224fc228fdd31.npy': 8388736, 'arrays/442a35fc756758e98fab7d293ef2b2fdcb42ffa2814659b98035029b4344291d.npy': 3506304, 'arrays/44935510f93ae742a311ec79edc6d90b3d8f350dea2f5841380a735c5a4ebe9d.npy': 6357120, 'arrays/48e8f9af2174976c16d862b2c938ec3514617d3e3e336df3f4556fcecc1e4bfe.npy': 6291584, 'arrays/51b1932b3e51b5a5ed7d76059a8829c66aa919b49df28a911c026da9ed96e828.npy': 5046400, 'arrays/57981323ec765907eed28d96c2608899a6c196c7b2a7a316eb99992ed26989bf.npy': 7209088, 'arrays/61228b43ea72f2938b1ba6b1d4565692c3441d3afc4cd1ceb3831b61f7cc260d.npy': 6357120, 'arrays/6288dd365d6498df444a2f3737303e0d44ebf87052baac96c12b958c2a357d72.npy': 5046400, 'arrays/6de0631a487bd4b9ad10c1d4452612bff0efb8547bebc7b56d94e6d661e872f4.npy': 132, 'arrays/6f3fa80260fda831d7437fdf4bacb81ccea59c9d09a30cf5ce2de3cff6f9b5ac.npy': 4161664, 'arrays/7051e71301537a57b6535f355e52f6c5670e6e985ab06a621945a3c6b4d55f62.npy': 12583040, 'arrays/7cdae7c5c86aa4f2432ae5c5f38ee56f5e25cf7659a2ddc6d0680df58484a718.npy': 7340160, 'arrays/7f43f6235068a07ed704904677bcfe0dc302679ad2af807baaf0bacf2ca9a4f8.npy': 4292736, 'arrays/86c5d50a2ab7ebefc2159dc48db667a668a39e6522627e3a55e994c80dd72cfb.npy': 4522112, 'arrays/a017c023c038281ca7bbbc3317a709d8ae43badb9bf103ce31fcb48945926a0d.npy': 5963904, 'arrays/a76ec32a62922b59b843642529a15031ba33c31e57ccc2a201ed5eef43ef7fad.npy': 7209088, 'arrays/aa773a4a82f96c586c86629a5c70ad77d3fc054bcbae871e4d312a0472067550.npy': 5701760, 'arrays/b069c8b1a92fce0a21194c5d83657a9766f5075471ac9526fed27e541853cef5.npy': 6389888, 'arrays/b5a1bcc8ec320fbe994e1072b20da11ae3a91ae482ef5bf81ca5edb5a874af62.npy': 3506304, 'arrays/c89ea2f32005b250fc9a573b8f589bf61dd810cad4c98d2ae622f968641f92b9.npy': 4882560, 'arrays/c9d9b9943715d0ba039cc8ecb2bc7df17675b166da0e8b3a4be550d7fe9671b0.npy': 4784256, 'arrays/cdd93d5a3f51c3661845fde750935a60cc3be7e29d7d6129b149ffeb57c3f14b.npy': 4522112, 'arrays/d0998a5eadf1ac9ca4ed4fb788740b18ed31e4daaf5d6493de0d03f445ac964e.npy': 7209088, 'arrays/d57c6b33f74c5e18cb7b17b372ec8a0f7bb6903b3acf4db5339442c5d04917e6.npy': 4292736, 'arrays/edad95d6cdfbbfac3a7415b1e7146bbac6a7fe36712fec209c537cae63b550d6.npy': 6750336, 'arrays/f0c16b9ed98e2ca83b1cb51f34fec03facfa1048cd892c9991d9163d89cd053f.npy': 5701760, 'lineage/parent-session.json': 1037, 'lineage/plan.json': 40126, 'lineage/source-authorization.json': 2308, 'lineage/source-claim.json': 264, 'lineage/source-control.json': 47902, 'lineage/target-pins.json': 133748, 'lineage/target-review.json': 11099, 'manifest.json': 74784, 'references/fresh_process_candidate_reference/VJP-calls.jsonl': 725282, 'references/fresh_process_candidate_reference/native-calls.jsonl': 948394, 'references/fresh_process_candidate_reference/scopes.json': 577, 'references/fresh_process_candidate_reference/success.json': 5002, 'references/fresh_process_candidate_reference/worker.json': 17221, 'references/same_replay_original/VJP-calls.jsonl': 0, 'references/same_replay_original/native-calls.jsonl': 948394, 'references/same_replay_original/scopes.json': 327, 'references/same_replay_original/success.json': 4996, 'references/same_replay_original/worker.json': 16743}
FROZEN_NAMES = (*frozen.FROZEN_NAMES, 'models/predictor_forward_preserving_records_v1.py',
                'models/predictor_forward_preserving_operator_v1.py')
REMOTE_NAMES = tuple(dict.fromkeys((*frozen.REMOTE_NAMES,
    'models/predictor_forward_preserving_operator_v2.py', 'scripts/capture_predictor_forward_preserving_v2.py',
    'configs/predictor_forward_preserving_handoff_v2.json', 'models/predictor_replay_capsule_v1.py',
    'models/scratch_inventory_observer_v1.py', 'models/predictor_missing_replay_records_v1.py',
    'models/predictor_missing_replay_worker_v1.py', 'models/predictor_missing_replay_operator_v1.py',
    'scripts/capture_predictor_missing_replay_v1.py', CONTRACT)))
LOCAL_EXTRA = ('models/predictor_missing_replay_client_v1.py', 'scripts/prepare_predictor_missing_replay_v1.py',
               'scripts/run_predictor_missing_replay_v1.py', 'tests/test_predictor_missing_replay_handoff_v1.py')
BOOT_NAMES = (*FROZEN_NAMES, 'models/predictor_replay_capsule_v1.py', 'models/predictor_missing_replay_records_v1.py',
              'models/scratch_inventory_observer_v1.py', 'models/predictor_missing_replay_operator_v1.py')
exact = frozen.exact
readiness, extract = frozen.readiness, frozen.extract


def validate_control(c):
    expected = {'kind': KIND, 'host': 'linux10603', 'parent': '/var/tmp', 'expected_device': 66307, 'expected_mount': '/',
                'expected_gpu': 'NVIDIA GeForce RTX 3070', 'expected_gpu_uuid': 'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f',
                'expected_driver': '580.178.04', 'expected_ssh_fingerprint': base.SSH_FINGERPRINT,
                'contract_sha256': CONTRACT_SHA, 'paid_compute': False, 'research_admitted': False,
                'authorization_present_at_preparation': False, 'automatic_retries': 0}
    more = {'contract','contract_text','resource','repository','local_handoff_directory','archive_path','archive_manifest',
            'known_hosts_sha256','remote_source_sha256','local_source_sha256','prior_inputs_manifest',
            'prior_inputs_manifest_sha256','host_binding','reference_archive_path','reference_manifest'}
    if set(c) != set(expected)|more or not exact({k:c[k] for k in expected}, expected):
        raise ValueError('Exact new one-replay control required.')
    p = c['contract']
    if (hashlib.sha256(c['contract_text'].encode()).hexdigest() != CONTRACT_SHA
            or not exact(json.loads(c['contract_text']), p) or p['authorized'] is not False
            or p['new_target_execution_permitted'] is not False or not exact(c['resource'], p['resource'])
            or p['full_model_EOT8_passes_at_most'] != 1 or p['worker_roles'] != [{'index':3,'mode':'candidate','replay':1}]
            or p['resource']['work_seconds'] != 900 or p['resource']['workers'] != 1
            or p['resource']['worker_seconds'] != 480 or p['resource']['allocator_bytes'] != 7*GiB
            or not exact(c['host_binding'], {'prior_consumed_files':p['prior_consumed_files']})):
        raise ValueError('Single-replay resources/lineage/contract changed.')
    if (set(c['remote_source_sha256']) != set(REMOTE_NAMES)
            or set(c['local_source_sha256']) != set(REMOTE_NAMES)-{'models/__init__.py'}|set(LOCAL_EXTRA)
            or c['remote_source_sha256'][CONTRACT] != CONTRACT_SHA):
        raise ValueError('Exact single-replay source coverage required.')
    spec = c['archive_manifest']
    if (hashlib.sha256(old.json_bytes(spec)).hexdigest() != p['packet_manifest_sha256']
            or spec['bytes'] != 17192095 or spec['sha256'] != PACKET_SHA
            or c['archive_path'] != p['reference']['existing_packet_archive']
            or c['reference_archive_path'] != p['reference_archive']['path']
            or p['reference_archive']['bytes'] != REFERENCE_BYTES or p['reference_archive']['sha256'] != REFERENCE_SHA
            or hashlib.sha256(capsule.encode(c['reference_manifest'])).hexdigest() != records.MANIFEST_SHA
            or spec['bytes']+REFERENCE_BYTES > 128*MiB):
        raise ValueError('Exact original packet and immutable reference capsule required.')
    if c['prior_inputs_manifest'] != p['prior_inputs_manifest']:
        raise ValueError('Preservation manifest changed.')


def authorization_template(c, sha, basis):
    return {'kind':AUTH_KIND,'approved':True,'control_sha256':sha,'host':'linux10603','maximum_new_attempts':1,
            'resource':c['resource'],'roles':[{'index':3,'mode':'candidate','replay':1}], 'full_EOT8_passes_at_most':1,
            'automatic_retries':0,'paid_compute':False,'cleanup':False,'reset_old_claims':False,
            'private_key_upload':False,'held_out_pixels':False,'user_basis':basis}


def validate_authorization(a, sha, c=None):
    if c is None:
        raise ValueError('Bound new control required for replay authorization.')
    records.binding(sha)
    basis = a.get('user_basis') if isinstance(a,dict) else None
    if not isinstance(basis,str) or not 1 <= len(basis) <= 8192 or not exact(a,authorization_template(c,sha,basis)):
        raise ValueError('One new exact scoped candidate-replay authorization required.')


def category(name):
    plain = name.removesuffix('.partial')
    if plain == 'space-proof-v2.bin':return 'proof'
    if plain in ('original-packet.tar.gz','reference-capsule.tar.gz'):return 'input'
    if name.startswith(('code/','overlay/')):return 'code'
    if name.startswith(('cache/','tmp/')):return 'cache'
    if name == 'emergency-space.bin':return 'emergency'
    if plain.startswith('reference/arrays/'):
        if plain.removeprefix('reference/') not in REFERENCE_EXTENTS:raise ValueError('Unknown reference array path.')
        return 'raw'
    if plain.startswith('results/3/') and plain.endswith('.npy'):return 'raw'
    if name.startswith('results/') and not name.startswith('results/3/'):
        raise ValueError('No other worker result directory is allowed.')
    return 'metadata'


def inventory(folder, *, hashing=False, running=False):
    # One call to the actual frozen inventory; only its post-refusal metadata is added.
    files, allocated = observed.inventory(folder,hashing=hashing,max_bytes=3*GiB,running=running)
    for name,item in files.items():
        plain = name.removesuffix('.partial');leaf = Path(plain).name
        if plain in ('original-packet.tar.gz','reference-capsule.tar.gz'):
            cap = 17192095 if plain == 'original-packet.tar.gz' else REFERENCE_BYTES
        elif plain.startswith('reference/'):
            relative = plain.removeprefix('reference/')
            if relative not in REFERENCE_EXTENTS:raise ValueError('Unexpected reference member.')
            cap = REFERENCE_EXTENTS[relative]
        else:
            cap = (5*GiB//2 if plain == 'space-proof-v2.bin' else 65536 if leaf == 'loss.npy'
                   else MiB if leaf == 'masks.npy' else 12*MiB if leaf == 'targets.npy'
                   else 16*MiB if leaf.endswith(('.npy','.log')) else records.LIMITS.get(leaf,64*MiB))
        category(name)
        if item['bytes'] > cap:raise ValueError('Typed replay file cap exceeded: '+name)
    return files, allocated


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


def extract_reference(archive, manifest, destination):
    destination=Path(destination);raw=capsule.encode(manifest)
    if destination.exists() or old.digest(archive)!=REFERENCE_SHA or hashlib.sha256(raw).hexdigest()!=records.MANIFEST_SHA:
        raise ValueError('Fresh reference destination and exact archive/manifest required.')
    expected=dict(manifest['files']);expected['manifest.json']={'bytes':len(raw),'sha256':records.MANIFEST_SHA}
    destination.mkdir(mode=0o700);seen=set()
    with tarfile.open(archive,mode='r|gz') as tar:
        for member in tar:
            name=old.safe_name(member.name)
            if name not in expected or name in seen or member.type!=tarfile.REGTYPE or member.pax_headers:
                raise ValueError('Only exact unique regular reference members allowed.')
            item=expected[name]
            if member.size!=item['bytes'] or member.mode!=0o600:raise ValueError('Reference member extent/mode changed.')
            path=destination/name;parent=destination
            # mkdir(parents=True) does not apply mode to intermediate parents.
            # Construct each fresh capsule component explicitly, independent of umask.
            for component in Path(name).parts[:-1]:
                parent=parent/component;parent.mkdir(mode=0o700,exist_ok=True)
                info=parent.lstat()
                if parent.is_symlink() or not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode)!=0o700 or info.st_uid!=os.getuid():
                    raise ValueError('Private reference extraction parent required.')
            with tar.extractfile(member) as source,path.open('xb') as out:
                for chunk in iter(lambda:source.read(MiB),b''):out.write(chunk)
                out.flush();os.fsync(out.fileno())
            path.chmod(0o600)
            if old.digest(path)!=item['sha256']:raise ValueError('Reference member hash changed.')
            ns=item.get('mtime_ns',member.mtime*10**9);os.utime(path,ns=(ns,ns));seen.add(name)
    if seen!=set(expected):raise ValueError('Incomplete reference capsule.')
    with capsule.Capsule(destination,manifest_sha256=records.MANIFEST_SHA):pass


stream=FunctionType(frozen.stream.__code__,dict(frozen.stream.__globals__,KIND=KIND,inventory=inventory))


def check_worker(folder, *, control_sha256):
    state=records.load(folder,control_sha256=control_sha256)
    return {'worker_sha256':old.digest(Path(folder)/'worker.json'),
            'completion_sha256':old.digest(Path(folder)/'success.json'),'pid':state['pid'],
            'index':3,'numerical_passed':True,'CUDA_domain':True}


def _execute(c,auth,incoming,output,sources,sha,start,deadline,stop):
    identity=readiness(c);parent=Path(identity['parent']);base.gpu_gate(c)
    folder=Path(tempfile.mkdtemp(prefix=PREFIX,dir=parent));folder.chmod(0o700)
    output.write(old.json_bytes({'kind':KIND,'phase':'scratch-created','scratch_directory':str(folder),'control_sha256':sha}));output.flush()
    state={'kind':KIND,'complete':False,'passed':False,'tasks':[],'started_unix':time.time(),
           'deadline_unix':time.time()+max(0,deadline-time.monotonic()),'work_seconds':900,'research_admitted':False,
           'reference_binding':records.binding(sha),'source_parent_failed':True,'old_work_seconds_not_reset':477.3622766463086}
    try:
        old.write_json(folder/'control.json',c);old.write_json(folder/'authorization.json',auth);old.write_json(folder/'storage-before.json',identity)
        if set(sources)!=set(REMOTE_NAMES) or sum(map(len,sources.values()))>2*MiB:raise ValueError('Overlay extent changed.')
        for name,data in sources.items():
            if hashlib.sha256(data).hexdigest()!=c['remote_source_sha256'][name]:raise ValueError('Overlay source changed.')
            path=folder/'overlay'/old.safe_name(name);path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            path.write_bytes(data);path.chmod(0o600)
        # Bootstrap modules were compiled in memory. Point source checks at their
        # identical, now-verified regular files; keep code objects/globals intact.
        for name in BOOT_NAMES:
            module=sys.modules.get(name[:-3].replace('/','.'))
            if module is not None and module.__file__==name:module.__file__=str(folder/'overlay'/name)
        with (folder/'emergency-space.bin').open('xb') as f:f.write(os.urandom(16*MiB));f.flush();os.fsync(f.fileno())
        end=min(deadline-5,time.monotonic()+60)
        program='import sys;sys.path.insert(0,'+repr(str(folder/'overlay'))+');from models.scratch_allocation_v2 import bounded_write_proof;bounded_write_proof('+repr(str(folder))+',deadline='+repr(end)+')'
        proof=base.primitives.supervise([sys.executable,'-B','-u','-c',program],folder,role='proof',deadline=end,
                                      gate=lambda:storage(folder,identity,setup=True),file_limit=3*GiB)
        if proof['exit_status']!=0:raise RuntimeError('Full storage proof failed; no scientific work.')
        base.primitives.validate_proof(folder)
        old.receive_archive(incoming,folder/'original-packet.tar.gz',17192095,PACKET_SHA,deadline-5)
        old.receive_archive(incoming,folder/'reference-capsule.tar.gz',REFERENCE_BYTES,REFERENCE_SHA,deadline-5)
        extract(folder/'original-packet.tar.gz',c['archive_manifest'],folder/'code')
        extract_reference(folder/'reference-capsule.tar.gz',c['reference_manifest'],folder/'reference')
        with capsule.Capsule(folder/'reference',manifest_sha256=records.MANIFEST_SHA) as reader:
            if reader.execution_binding(sha)!=state['reference_binding']:raise ValueError('Reference binding changed.')
        if time.monotonic()-start>180:raise TimeoutError('Setup time exhausted.')
        for name in ('tmp','cache','cache/cuda','results'):(folder/name).mkdir(mode=0o700,exist_ok=True)
        old.write_json(folder/'session.json',state)
        base.gpu_gate(c);entry=folder/'overlay/scripts/capture_predictor_missing_replay_v1.py'
        end=min(deadline-125,time.monotonic()+480)
        if end<=time.monotonic():raise TimeoutError('Continuing work allowance exhausted.')
        old.write_json(folder/'worker-plan-3.json',{'index':3,'control_sha256':sha,'deadline_unix':time.time()+end-time.monotonic()})
        command=[sys.executable,'-I','-B','-u',str(entry),'--scratch',str(folder),'--control-sha256',sha,'--index','3']
        last,samples=[0.],[]
        def gate():
            row=storage(folder,identity)
            if time.monotonic()-last[0]>=5:
                owned=[]
                for pid in old.descendants(os.getpid()):
                    try:argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
                    except FileNotFoundError:continue
                    if str(entry).encode() in argv:owned.append(pid)
                if len(owned)>1:raise ValueError('Multiple scientific processes refused.')
                samples.append(base.gpu_gate(c,owned[0] if owned else None));last[0]=time.monotonic()
            return row
        env={'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES','0'),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
             'TMPDIR':str(folder/'tmp'),'XDG_CACHE_HOME':str(folder/'cache'),'CUDA_CACHE_PATH':str(folder/'cache/cuda'),'CUDA_CACHE_MAXSIZE':str(32*MiB)}
        if env['CUDA_VISIBLE_DEVICES'] in ('','-1'):raise ValueError('CUDA visibility hidden; refuse.')
        child=base.primitives.supervise(command,folder,role='worker-3',deadline=end,gate=gate,file_limit=16*MiB,env=env)
        old.write_json(folder/'gpu-samples-3.json',samples);base.gpu_gate(c)
        if child['exit_status']!=0:raise RuntimeError('Only candidate replay failed; no retry or next worker.')
        task=check_worker(folder/'results/3',control_sha256=sha)
        if task['pid'] in (41586,48079,54863,61329):raise ValueError('New scientific process identity required.')
        state['tasks']=[task];state.update(complete=True,passed=True)
    except BaseException as error:
        state.update(complete=False,passed=False,error_type=type(error).__name__,error=str(error)[:2048])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);state['wall_seconds']=time.monotonic()-start
        if state['wall_seconds']>900:state.update(complete=False,passed=False,error='Outer work deadline exceeded.')
        old.write_json(folder/'session.json',state,replace=(folder/'session.json').exists())
    signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,600)
    try:stream(folder,sha,output)
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    return 0 if state['passed'] else 1


def remote(c,auth,incoming,output,sources):
    validate_control(c);sha=hashlib.sha256(old.json_bytes(c)).hexdigest();validate_authorization(auth,sha,c)
    start=time.monotonic();deadline=start+900
    handlers={s:signal.getsignal(s) for s in (signal.SIGHUP,signal.SIGTERM,signal.SIGALRM)}
    def stop(*unused):raise RuntimeError('Attended replay interrupted; preserve evidence, no retry.')
    previous_umask=os.umask(0o077)
    for s in handlers:signal.signal(s,stop)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()-5))
    try:return _execute(c,auth,incoming,output,sources,sha,start,deadline,stop)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)
        os.umask(previous_umask)


def main(control64,sources64,mode='run',scratch=None):
    if mode!='run' or scratch is not None:raise ValueError('Only original same-SSH work/collection; no separate retrieval.')
    c=json.loads(base64.b64decode(control64,validate=True));sources={n:base64.b64decode(v,validate=True) for n,v in json.loads(base64.b64decode(sources64,validate=True)).items()}
    validate_control(c);auth=json.loads(sys.stdin.buffer.readline(65537))
    return remote(c,auth,sys.stdin.buffer,sys.stdout.buffer,sources)
