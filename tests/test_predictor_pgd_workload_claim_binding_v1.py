"""New bound-client integration; explicit fake OS ports, real codec/FS byte guards."""
import ast
import base64
import builtins
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
from types import ModuleType,SimpleNamespace as NS
import pytest
from models import predictor_pgd_workload_handoff_v2 as h
from models import predictor_pgd_workload_handoff_outer_v2 as outer
from models import predictor_pgd_workload_handoff_client_v2 as client
from models import predictor_pgd_workload_bootstrap_v1 as boot
from models import predictor_pgd_split_plan_v1 as plan

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'runs/pretrained/diagnostic/pgd-workload-claim-binding-v1-20261123'
EVIDENCE=Path(os.environ['VJEPA_CLAIM_BINDING_EVIDENCE']).resolve()
ORIGIN='EXPLICIT-FAKE-CLAIM-BINDING; NO USER GRANT OR TARGET EXECUTION'


def put(path,value):
    missing=[];p=path.parent
    while not p.exists():missing.append(p);p=p.parent
    for p in reversed(missing):p.mkdir(mode=0o700)
    data=value if type(value) is bytes else h.enc(value)
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as out:out.write(data)


@pytest.fixture(scope='module',autouse=True)
def forbid_work():
    patch=pytest.MonkeyPatch()
    def forbidden(*a,**k):pytest.fail('Real process/signal/timer/network/scientific work forbidden.')
    for obj,name in [(subprocess,'Popen'),(subprocess,'run'),(subprocess,'check_output'),(os,'system'),(os,'fork'),
        (os,'posix_spawn'),(os,'kill'),(os,'killpg'),(os,'pidfd_open'),(os,'pipe2'),(os,'posix_fallocate'),
        (signal,'signal'),(signal,'setitimer'),(signal,'alarm'),(signal,'pidfd_send_signal'),(socket,'socket'),(threading.Thread,'start')]:
        if hasattr(obj,name):patch.setattr(obj,name,forbidden)
    original=builtins.__import__
    def load(name,*a,**k):
        if name.split('.')[0] in ('torch','numpy'):forbidden()
        return original(name,*a,**k)
    patch.setattr(builtins,'__import__',load)
    yield
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    patch.undo()


@pytest.fixture
def area(request):
    p=EVIDENCE/request.node.name;p.mkdir(mode=0o700)
    put(p/'FAKE-ORIGIN.json',{'origin':ORIGIN,'actual_user_grant':False,'actual_target_work':False})
    return p


def package(parent):
    p=parent/'fake-package';h.prepare_unarmed(ROOT,p)
    c=json.loads((p/'CONTROL-WORKLOAD.json').read_bytes())
    put(p/'AUTHORIZATION.json',outer.authorization(c,h.digest((p/'CONTROL-WORKLOAD.json').read_bytes()),ORIGIN))
    return p


def private_module(path,imports):
    m=ModuleType('_EXPLICIT_FAKE_'+path.stem);m.__file__=str(path)
    original=builtins.__import__
    def load(name,*a,**k):return imports[name] if name in imports else original(name,*a,**k)
    m.__dict__['__builtins__']={**vars(builtins),'__import__':load}
    exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__)
    return m


class FakeClock:
    def __init__(self,values=None,now=1002.):self.values=list(values or []);self.now=now
    def __call__(self):
        if self.values:self.now=self.values.pop(0)
        return self.now


class FakeAlarms:
    def __init__(self,events):self.events=events
    def arm(self,*a,**kw):self.events.append(['FAKE-ALARM',list(a),kw])
    def check(self):pass
    def close(self):self.events.append(['FAKE-ALARM-CLOSE'])


class AliasRoot:
    """Fake OS path port: source-generated remote name, real private local backing."""
    def __init__(self,virtual,physical):self.virtual=virtual;self.physical=physical
    def __str__(self):return str(self.virtual)
    def mkdir(self,mode):self.physical.mkdir(mode=mode)


class GuardView:
    """No device remap. Only the new FAKE wire root strings use a virtual path."""
    def __init__(self,guard,virtual):
        self.guard=guard;self.binding={**guard.binding,'root':str(virtual),'parent':'/var/tmp'}
    def __getattr__(self,name):return getattr(self.guard,name)


class FakeStopBeforeProof(RuntimeError):pass


class ServerFixture:
    def __init__(self,area,checked,elapsed):
        self.area=area;self.checked=checked;self.events=[];self.clock=FakeClock([1000.,1001.]);self.pid=900001
        self.os=NS(**{**vars(os),'getpid':lambda:self.pid,
            'fstatvfs':lambda fd:NS(f_bavail=20*2**30,f_frsize=1,f_favail=100000),
            'urandom':lambda n:b'F'*n})
        self.project=NS(predictor_pgd_split_plan_v1=plan,predictor_pgd_workload_bootstrap_v1=boot)
        imports={'os':self.os,'time':NS(monotonic=self.clock),'threading':NS(get_ident=lambda:700001),'models':self.project}
        self.scratch=private_module(ROOT/'models/predictor_pgd_workload_scratch_v1.py',imports)
        self.project.predictor_pgd_workload_scratch_v1=self.scratch
        self.codec=private_module(ROOT/'models/predictor_pgd_workload_collection_v1.py',imports)
        self.project.predictor_pgd_split_collection_v1=self.codec
        self.server=private_module(ROOT/h.OUTER,{**imports,'threading':NS(),
            '_thread':NS(interrupt_main=lambda *a:pytest.fail('No interrupt port expected.'))})
        self.server.Alarms=lambda:FakeAlarms(self.events)
        original_root=self.server.remote_root
        def root(csha):
            virtual=original_root(csha);self.alias=AliasRoot(virtual,area/'server-files')
            put(area/'FAKE-ROOT-PROJECTION.json',{'origin':ORIGIN,'virtual_root':str(virtual),
                'private_local_backing':str(self.alias.physical),'local_device':ROOT.lstat().st_dev,
                'device_or_old_records_remapped':False,'target_filesystem_created':False})
            return self.alias
        self.server.remote_root=root
        self.server.preflight_parent=lambda:self.events.append(['FAKE-HOST-GPU-STORAGE-GATE'])
        self.server.prepare_tree=self.setup
        c,b,p,csha,raw,araw,a,mapping,code,packets=checked
        message={'nonce':(b'F'*32).hex(),'client_elapsed':elapsed,'control_text':raw.decode(),
            'authorization_text':araw.decode(),'sources':mapping}
        incoming=io.BytesIO();outer.send_frame(incoming,message,lambda:None,cap=2*2**20)
        for path in packets:incoming.write(path.read_bytes())
        self.input=incoming.getvalue();source=io.BytesIO(self.input);out=io.BytesIO()
        self.status=self.server.remote_session(source,out,code_sha256=h.digest(code))
        self.output=out.getvalue();assert source.read()==b''
        put(area/'SERVER-INPUT.bin',self.input);put(area/'SERVER-OUTPUT.bin',self.output)
        put(area/'SERVER-EVENTS.json',{'origin':ORIGIN,'events':self.events,'fake_status':self.status})

    def setup(self,alias,sources,c,b,p,auth,raw,araw,bootstrap,incoming,check):
        actual=alias.physical;supervised=c['supervised_control_text'].encode()
        b,policy,bsha=bootstrap.control(supervised)
        # Execute the exact server's derived-authorization assignment, not a copied kind literal.
        fn=next(n for n in ast.parse((ROOT/h.OUTER).read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='prepare_tree')
        assignment=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='derived' for t in n.targets))
        ns={'boot':bootstrap,'bsha':bsha,'auth':auth}
        exec(compile(ast.Module(body=[assignment],type_ignores=[]),'<SOURCE-SERVER-AUTH>','exec'),ns)
        derived=ns['derived'];bootstrap.authorization(derived,bsha);ib,isha=bootstrap.installation(b,bsha)
        layout=self.scratch.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),ib,isha)
        for name in self.server._setup_directory_plan(layout)['setup_directories']:
            if name!='.':(actual/name).mkdir(mode=0o700,exist_ok=True)
        for name,data in sources.items():put(actual/name,data)
        for name,data in [('control.json',supervised),('authorization.json',derived),('installation.json',ib),
            ('configs/base.json',p['base_text'].encode()),('configs/design.json',p['design_text'].encode()),
            ('proposal.json',{'origin':ORIGIN,'approved':False}),
            ('runtime-source-manifest.json',{'origin':ORIGIN,'source_files':55})]:put(actual/name,data)
        self.server.receive_packet(incoming,actual,json.loads(p['scratch_contract_text'])['packet']['archive'],check)
        self.server.extract_packet(actual,json.loads(p['scratch_contract_text'])['packet']['files'],check)
        put(actual/'emergency-space.bin',b'R'*(16*2**20))  # Ordinary allocated fixture reserve, never proof.
        binding={'root':str(actual),'parent':str(actual.parent),'root_identity':list(self.scratch.directory_id(actual.lstat())),
            'parent_identity':list(self.scratch.directory_id(actual.parent.lstat()))}
        physical=self.scratch.Guard(actual,binding,layout);physical.verify_assets(phase='setup')
        self.guard=GuardView(physical,alias.virtual);self.clock.now=1002.
        journal=NS(close=lambda:self.events.append(['FAKE-JOURNAL-CLOSE']),last_refusal=None)
        driver=NS(guard=self.guard,outer_journal=journal,assert_idle=lambda:self.events.append(['FAKE-IDLE-NO-CHILDREN']))
        parent=self
        class Owner:
            def __init__(self,root,binding,control,authorization,*,connection_started,**kw):
                self.claimed=True;self.driver=driver;self.state={};self.origin=connection_started
            def run_work(self):
                state={'kind':'whole-pgd-parent-session-v1','control_sha256':bsha,'parent_pid':parent.pid,
                    'phase':'failed','complete':False,'connection_started':self.origin,'work_deadline':self.origin+2400,
                    'connection_deadline':self.origin+3000,'completed':[],'receipts':[],
                    'origin':ORIGIN,'error':'EXPLICIT FAKE STOP BEFORE PROOF','scientific_qualified':False,'training_admitted':False}
                put(actual/'session.json',state);self.state=state
                parent.events.append(['FAKE-STOP-BEFORE-PROOF','NO WORKER OR COMPUTE'])
                raise FakeStopBeforeProof(state['error'])
        return NS(OuterCoordinator=Owner),self.scratch,self.guard,self.guard.binding,derived


class FakeThread:
    def __init__(self,target,daemon=True):self.target=target
    def start(self):self.target()
    def join(self,timeout=0):assert timeout==0


class FakeSink:
    def __init__(self):self.data=bytearray();self.closed=False
    def write(self,data):assert not self.closed;self.data.extend(data);return len(data)
    def flush(self):pass
    def fileno(self):return 800001
    def close(self):self.closed=True


class FakeProc:
    def __init__(self,output,status):
        self.stdin=FakeSink();self.stdout=io.BytesIO(output);self.stderr=io.BytesIO();self.status=status;self.alive=True;self.terminated=False
    def wait(self,timeout):assert timeout>=0;self.alive=False;return self.status
    def poll(self):return None if self.alive else self.status
    def terminate(self):self.terminated=True;self.alive=False
    def kill(self):pytest.fail('No fake forced kill expected.')


class ClientFixture:
    def __init__(self,area,folder=None,*,golden=None,response=None):
        self.area=area;self.events=[];self.clock=FakeClock([100.,102.],now=103.);self.clock.now=103.
        # After the two source clock samples, all subsequent fake observations stay103.
        def clock():
            if self.clock.values:return self.clock.values.pop(0)
            return 103.
        self.folder=folder or area/'fake-package';self.golden=golden
        if golden:
            self.checked=golden.checked;put(self.folder/'AUTHORIZATION.json',{'origin':ORIGIN,'actual_user_approval':False})
        self.project=NS(predictor_pgd_workload_handoff_outer_v2=outer,predictor_pgd_workload_bootstrap_v1=boot)
        self.codec=golden.server.codec if golden else None
        self.proc=None
        def popen(*args,**kw):
            assert args[0][0]=='ssh' and kw['shell'] is False
            self.events.append(['FAKE-Popen-NO-CONNECTION'])
            if golden:output=response;status=golden.server.status
            else:
                self.server=ServerFixture(area,self.checked,2.);output=self.server.output;status=self.server.status
                self.codec=self.server.codec;self.project.predictor_pgd_workload_collection_v1=self.codec
            self.proc=FakeProc(output,status);return self.proc
        def write(fd,data):assert fd==800001;return self.proc.stdin.write(data)
        local_outer=NS(**{**vars(outer),'Alarms':lambda:FakeAlarms(self.events)})
        self.project.predictor_pgd_workload_handoff_outer_v2=local_outer
        # Codec isn't used until after Popen's fake server has produced its transcript.
        class CodecPort:
            def __getattr__(_,name):return getattr(self.codec,name)
        self.project.predictor_pgd_workload_collection_v1=CodecPort()
        # Keep the proxy stable when the fake server is attached.
        self.codec_proxy=self.project.predictor_pgd_workload_collection_v1
        imports={'models':self.project,'os':NS(**{**vars(os),'write':write}),
            'time':NS(monotonic=clock),'threading':NS(Thread=FakeThread),
            'subprocess':NS(Popen=popen,PIPE=-1,TimeoutExpired=subprocess.TimeoutExpired)}
        self.client=private_module(ROOT/h.CLIENT,imports)
        if golden:
            self.client.check_local=lambda _:self.checked  # Explicit forked fixture identity port for saved-wire negatives.
        else:
            real_check=self.client.check_local
            def check(folder):self.checked=real_check(folder);return self.checked
            self.client.check_local=check
        self.project.predictor_pgd_workload_collection_v1=self.codec_proxy

    def run(self):
        try:return self.client.run(self.folder)
        finally:
            if self.proc:
                put(self.area/'CLIENT-OUTBOUND.bin',bytes(self.proc.stdin.data))
                put(self.area/'CLIENT-EVENTS.json',{'origin':ORIGIN,'events':self.events,'terminated':self.proc.terminated})


def test_complete_source_inverses_and_frozen_runtime():
    generated,d,c,f=h.assemble(ROOT)
    for name,edits in d['edits'].items():
        raw=generated[name]
        for e in reversed(edits):
            assert h.digest(raw)==e['derived_sha256'];start=e['start'];after=e['after'].encode()
            assert raw[start:start+len(after)]==after
            raw=raw[:start]+e['before'].encode()+raw[start+len(after):]
            assert h.digest(raw)==e['original_sha256']
        expected=f[c['pins'][{h.OUTER:'outer',h.CLIENT:'repaired_client',h.CLI:'cli_scaffold'}[name]]]
        assert raw==expected and generated[name]==(ROOT/name).read_bytes()
    assert len(outer.LOCAL_SOURCES)==16 and len(set(outer.LOCAL_SOURCES))==16
    assert h.SELF in outer.LOCAL_SOURCES and h.INPUTS in outer.LOCAL_SOURCES and h.CLI in outer.LOCAL_SOURCES
    assert d['scratch_prefix']=='vjepa-pgd-workload-v1-' and not d['uploaded_sources_changed']
    payload=json.loads(f[c['pins']['source_payload']]);manifest=json.loads(f[c['pins']['runtime_manifest']])
    assert len(payload)==55 and sum(len(base64.b64decode(v)) for v in payload.values())==895657
    for name,want in manifest['files'].items():
        raw=base64.b64decode(payload[name if name.startswith('supervisor/') else 'overlay/'+name])
        assert want=={'bytes':len(raw),'sha256':h.digest(raw)}


def test_full_new_preflight_pins_and_protocol_identity(area):
    folder=package(area);checked=client.check_local(folder)
    assert len(checked[0]['local_source_sha256'])==16 and len(checked[7])==55
    assert checked[6]['full_model_gradients']==120 and checked[6]['user_basis']==ORIGIN
    assert checked[-1]==(ROOT/outer.PACKET_PATH,)
    pkg=json.loads((folder/'PACKAGE.json').read_bytes())
    assert pkg['scratch_prefix']=='vjepa-pgd-workload-v1-' and pkg['supervised_authorization_kind']=='whole-pgd-supervised-university-exception-v1'
    assert not (folder/'ATTEMPT.json').exists()


def test_consumed_v1_package_cannot_enter_new_preflight():
    old=ROOT/'runs/pretrained/diagnostic/pgd-workload-handoff-v1-20261123/handoff-parent/linux10614-primary-final-unarmed'
    with pytest.raises(ValueError):client.check_local(old)
    assert json.loads((old/'ATTEMPT.json').read_bytes())['consumed']


@pytest.mark.parametrize('armed',[False,True])
def test_new_isolated_CLI_dispatches_only_new_client(armed,area):
    folder=package(area) if armed else area/'missing'
    events=[];fake_sys=NS(path=[])
    original=builtins.__import__
    def imports(name,*a,**kw):
        if name=='sys':return fake_sys
        if name=='models.predictor_pgd_workload_handoff_client_v2':
            assert fake_sys.path==[str(ROOT)];events.append('NEW-CLIENT')
            return NS(run=lambda p:client.check_local(p) and 0)
        assert not name.startswith('models.')
        return original(name,*a,**kw)
    ns={'__name__':'_FAKE_CLI','__file__':str(ROOT/h.CLI),'__builtins__':{**vars(builtins),'__import__':imports}}
    exec(compile((ROOT/h.CLI).read_bytes(),str(ROOT/h.CLI),'exec'),ns)
    if armed:assert ns['main']([str(folder)])==0 and events==['NEW-CLIENT']
    else:
        with pytest.raises(SystemExit,match='Unarmed'):ns['main']([str(folder)])
        assert not events and fake_sys.path==[]


@pytest.fixture(scope='module')
def preflight_baseline():
    parent=EVIDENCE/'preflight-baseline';parent.mkdir(mode=0o700);folder=package(parent)
    project=parent/'project';project.mkdir(mode=0o700)
    for name in [*outer.LOCAL_SOURCES,outer.PACKET_PATH]:put(project/name,(ROOT/name).read_bytes())
    return parent


@pytest.mark.parametrize('fault',['client','outer','design','grant-kind','payload','archive'])
def test_new_preflight_refuses_drift_before_attempt(fault,preflight_baseline,area):
    copied=area/'copied';shutil.copytree(preflight_baseline,copied);folder=copied/'fake-package';project=copied/'project'
    c=json.loads((folder/'CONTROL-WORKLOAD.json').read_bytes());c.update(repository=str(project),local_handoff=str(folder))
    raw=h.enc(c);(folder/'CONTROL-WORKLOAD.json').write_bytes(raw)
    a=outer.authorization(c,h.digest(raw),ORIGIN);(folder/'AUTHORIZATION.json').write_bytes(h.enc(a))
    if fault in ('client','outer','design'):
        name={'client':h.CLIENT,'outer':h.OUTER,'design':'configs/predictor_pgd_workload_claim_binding_v1.json'}[fault]
        (project/name).write_bytes(b'EXPLICIT FAKE DRIFT\n')
    elif fault=='grant-kind':
        a['kind']='pgd-candidate-outer-university-exception-v1';(folder/'AUTHORIZATION.json').write_bytes(h.enc(a))
    elif fault=='payload':(folder/'SOURCE-PAYLOAD.json').write_bytes(b'{}\n')
    else:(project/outer.PACKET_PATH).write_bytes(b'EXPLICIT FAKE ARCHIVE DRIFT')
    with pytest.raises(ValueError):client.check_local(folder)
    assert not (folder/'ATTEMPT.json').exists() and not (folder/'attended-session').exists()


def test_stale_collection_grant_kind_refused_by_real_server_validator():
    csha='f'*64
    a={'kind':'pgd-candidate-supervised-university-exception-v1','approved':True,'control_sha256':csha,'maximum_new_attempts':1,'user_basis':ORIGIN}
    with pytest.raises(ValueError):boot.authorization(h.enc(a),csha)
    a['kind']='whole-pgd-supervised-university-exception-v1'
    assert boot.authorization(h.enc(a),csha)==a
    assert b"'pgd-candidate-supervised-university-exception-v1'" not in (ROOT/h.CLIENT).read_bytes()


@pytest.fixture(scope='module')
def transcript():
    area=EVIDENCE/'integrated-transcript';area.mkdir(mode=0o700)
    put(area/'FAKE-ORIGIN.json',{'origin':ORIGIN,'actual_user_grant':False,'actual_target_work':False})
    folder=package(area);f=ClientFixture(area,folder)
    before=({k:v for k,v in sys.modules.items() if k.split('.')[0] in ('models','attacks','datasets')},list(sys.meta_path))
    f.result=f.run()
    after=({k:v for k,v in sys.modules.items() if k.split('.')[0] in ('models','attacks','datasets')},list(sys.meta_path))
    assert before==after
    return f


def test_real_client_server_and_priority_collection_transcript(transcript):
    f=transcript;assert f.result==1 and f.server.status==1 and not f.proc.terminated
    # Exact source upload + client handshake + ONE original archive match the same fake server input.
    assert bytes(f.proc.stdin.data)==f.checked[8]+f.server.input
    session=f.folder/'attended-session';observed=json.loads((session/'CLAIM-OBSERVED.json').read_bytes())
    assert observed==json.loads((session/'ROOT-RECEIPT.json').read_bytes())
    assert Path(observed['scratch']).parent==Path('/var/tmp') and Path(observed['scratch']).name.startswith('vjepa-pgd-workload-v1-')
    receipt=json.loads((session/'COLLECTION-RECEIPT.json').read_bytes());assert receipt['transport_verified'] and receipt['SSH_exit_status']==1
    assert not receipt['scientific_qualified'] and not receipt['training_admitted']
    saved=session/'saved';assert json.loads((saved/'session.json').read_bytes())['phase']=='failed'
    work=json.loads((saved/'capture/results/final-receipt.json').read_bytes());assert work['roles']==[] and not work['collection_complete'] and not work['all_saved_role_checks_returned']
    partial=json.loads((session/'saved-transport/PARTIAL-RECEIPT.json').read_bytes());assert not partial['transport_verified'] and not partial['stream_complete']
    put(f.area/'TRANSCRIPT-RESULT.json',{'result':'PASS','origin':ORIGIN,'files':len(receipt['files']),
        'wire_sha256':receipt['wire_sha256'],'wire_bytes':receipt['wire_bytes'],'transport_verified':True,
        'actual_proof_or_worker_or_collection_OS_enforcement_qualified':False,'fake_work_stopped_before_proof':True,
        'exact_client_outbound_matches_server_input':True,'private_global_project_namespaces_unchanged':True})


def changed_response(f,kind):
    source=io.BytesIO(f.server.output);frames=[outer.frame(source,lambda:None) for _ in range(3)]
    codec=f.server.codec;raw,header=codec.WireReader(source,NS(check=lambda:None)).frame(codec.MAGIC,codec.HEADER_CAP)
    tail=source.read()
    if kind=='prefix':frames[1]['scratch']=frames[1]['scratch'].replace('workload-v1-','candidate-v1-')
    elif kind=='work-clock':frames[1]['work_deadline']+=1
    elif kind=='header-root':header['base']['root_binding']['root']+='-other'
    elif kind=='header-origin':
        b=header['base'];b['connection_started']+=1;b['connection_deadline']+=1
    elif kind=='partial':
        stream=io.BytesIO(tail)
        for _ in range(3):
            _,record=codec.WireReader(stream,NS(check=lambda:None)).frame(codec.RECORD,codec.RECORD_CAP)
            if record['kind']=='data':stream.read(header['base']['files'][record['name']]['bytes'])
        tail=tail[:stream.tell()+20]
    elif kind=='trailing':tail+=b'EXPLICIT-FAKE-TRAILING-BYTES'
    else:raise AssertionError(kind)
    out=io.BytesIO()
    for frame in frames:outer.send_frame(out,frame,lambda:None)
    data=h.enc(header);out.write(codec.MAGIC+len(data).to_bytes(4,'big')+data+tail)
    return out.getvalue()


@pytest.mark.parametrize('fault',['prefix','work-clock','header-root','header-origin','partial','trailing'])
def test_saved_transcript_mutations_stop_and_preserve_evidence(fault,transcript,area):
    response=changed_response(transcript,fault);put(area/'MUTATED-SERVER-OUTPUT.bin',response)
    f=ClientFixture(area,golden=transcript,response=response)
    with pytest.raises((ValueError,EOFError)):f.run()
    session=f.folder/'attended-session'
    assert (session/'CLAIM-OBSERVED.json').exists() and (session/'CLIENT-REFUSAL.json').exists()
    assert not (session/'COLLECTION-RECEIPT.json').exists() and f.proc.terminated
    if fault in ('prefix','work-clock'):assert not (session/'ROOT-RECEIPT.json').exists()
    else:assert (session/'ROOT-RECEIPT.json').exists()
    if fault in ('partial','trailing'):
        partial=json.loads((session/'saved-transport/PARTIAL-RECEIPT.json').read_bytes());assert not partial['transport_verified'] and not partial['stream_complete']
    put(area/'RESULT.json',{'result':'PASS','origin':ORIGIN,'fault':fault,'actual_target_work':False,'rejected_response_preserved':True})
