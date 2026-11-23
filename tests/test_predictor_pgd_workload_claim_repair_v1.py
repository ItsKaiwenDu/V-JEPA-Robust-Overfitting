"""New failure-specific controls; every process/clock/collection port is FAKE."""
import ast
import base64
import builtins
import copy
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_workload_claim_repair_v1 as repair

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'runs/pretrained/diagnostic/pgd-workload-handoff-v1-20261123/target-review-20261123'
EVIDENCE=Path(os.environ['VJEPA_WORKLOAD_CLAIM_EVIDENCE']).resolve()
ORIGIN='EXPLICIT-FAKE-CLAIM-REPAIR; NO REAL GRANT PROCESS CLOCK COLLECTION OR SCIENCE'
CONFIG=json.loads((ROOT/repair.INPUTS).read_bytes())
OLD=(ROOT/CONFIG['client_source']).read_bytes()
NEW=(STAGE/'client-prefix-v2/UNBOUND-CLIENT.py').read_bytes()
SERVER=(ROOT/CONFIG['server_source']).read_bytes()
CSHA='a'*64


@pytest.fixture(scope='module',autouse=True)
def forbid_execution():
    patch=pytest.MonkeyPatch()
    def forbidden(*a,**k):pytest.fail('Real process/timer/thread/proof/GPU/network/science forbidden.')
    for obj,name in [(subprocess,'Popen'),(subprocess,'run'),(os,'system'),(os,'fork'),(os,'posix_spawn'),
        (os,'kill'),(os,'killpg'),(os,'pidfd_open'),(os,'pipe2'),(os,'posix_fallocate'),(signal,'alarm'),
        (signal,'signal'),(signal,'setitimer'),(signal,'pidfd_send_signal'),(socket,'socket'),(threading.Thread,'start')]:
        if hasattr(obj,name):patch.setattr(obj,name,forbidden)
    original=builtins.__import__
    def load(name,*a,**k):
        if name.split('.')[0] in ('torch','numpy'):forbidden()
        return original(name,*a,**k)
    patch.setattr(builtins,'__import__',load)
    yield
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    patch.undo()


def put(path,value):
    raw=value if type(value) is bytes else (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as out:out.write(raw)


@pytest.fixture
def area(request):
    path=EVIDENCE/request.node.name;path.mkdir(mode=0o700)
    put(path/'FAKE-ORIGIN.json',{'origin':ORIGIN,'actual_user_authorization':False,'actual_target_work':False})
    return path


def definitions():
    ns={'__name__':'_UNINVOKED_SEALED_OUTER_DEFINITIONS'}
    exec(compile(SERVER,'<SEALED-OUTER-DEFINITIONS>','exec'),ns)
    return NS(**ns)


def server_claim(outer):
    """Actual pinned root function + actual claim emit statement, with fake UID/time."""
    tree=ast.parse(SERVER);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='remote_root')
    ns={'Path':Path,'os':NS(getuid=lambda:1001),'sha':outer.sha}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<SOURCE-ROOT-FAKE-UID>','exec'),ns)
    root=ns['remote_root'](CSHA)
    assert root==Path('/var/tmp')/('vjepa-pgd-workload-v1-1001-'+CSHA)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='remote_session')
    calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='send_frame'
        and len(n.args)>1 and isinstance(n.args[1],ast.Dict)
        and any(isinstance(x,ast.Constant) and x.value=='pgd-split-root-claimed-v1' for x in n.args[1].values)]
    assert len(calls)==1
    output=io.BytesIO();ns={'send_frame':outer.send_frame,'output':output,'csha':CSHA,'root':root,
        'hello':100.,'received':101.,'origin':98.,'FLAGS':outer.FLAGS,'alarms':NS(check=lambda:None)}
    statement=ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=calls[0])],type_ignores=[]))
    exec(compile(statement,'<SOURCE-CLAIM-FAKE-CLOCK>','exec'),ns)
    return outer.frame(io.BytesIO(output.getvalue()),lambda:None)


class FakeThread:
    def __init__(self,target,daemon=True):self.target=target
    def start(self):self.target()
    def join(self,timeout=0):assert timeout==0


class FakeSink:
    def __init__(self):self.data=bytearray();self.closed=False
    def write(self,data):assert not self.closed;self.data.extend(data);return len(data)
    def flush(self):pass
    def fileno(self):return 900001  # FAKE integer, never sent to a real fd API.
    def close(self):self.closed=True


class FakeProc:
    def __init__(self,wire):
        self.stdin=FakeSink();self.stdout=io.BytesIO(wire);self.stderr=io.BytesIO();self.alive=True;self.terminated=False
    def wait(self,timeout):assert timeout>=0;self.alive=False;return 0
    def poll(self):return None if self.alive else 0
    def terminate(self):self.terminated=True;self.alive=False
    def kill(self):pytest.fail('Fake kill path was not expected.')


def client_ports(area,source,mutation=None):
    outer=definitions();claim=server_claim(outer);put(area/'SERVER-CLAIM.json',claim)
    claimed=copy.deepcopy(claim)
    if mutation:
        if mutation=='candidate-prefix':claimed['scratch']=claimed['scratch'].replace('workload-v1-','candidate-v1-')
        elif mutation=='prefix-spoof':claimed['scratch']=claimed['scratch'].replace('workload-v1-','workload-v1-extra-')
        elif mutation=='parent':claimed['scratch']=claimed['scratch'].replace('/var/tmp/','/tmp/')
        elif mutation=='uid':claimed['scratch']=claimed['scratch'].replace('-1001-','-not-a-uid-')
        elif mutation=='hash-path':claimed['scratch']=claimed['scratch'].replace(CSHA,'b'*64)
        elif mutation=='hash-field':claimed['outer_control_sha256']='b'*64
        elif mutation=='kind':claimed['kind']='other-claimed-v1'
        elif mutation=='hello':claimed['hello_monotonic']=100.5
        elif mutation=='received':claimed['handshake_received']=99.
        elif mutation=='origin':claimed['connection_started']=98.25
        elif mutation=='work':claimed['work_deadline']+=1
        elif mutation=='connection':claimed['connection_deadline']+=1
        elif mutation=='training-flag':claimed['training_admitted']=True
        else:raise AssertionError(mutation)
    put(area/'WIRE-CLAIM.json',claimed)
    hello={'kind':'pgd-split-hello-v1','nonce':'c'*64,'hello_monotonic':100.,'code_sha256':'d'*64}
    ready={'kind':'pgd-split-collection-ready-v1','outer_control_sha256':CSHA,'collection_started':150.,
        'collection_deadline':750.,'connection_started':98.,'handshake_received':101.,**outer.FLAGS}
    wire=io.BytesIO()
    for value in [hello,claimed,ready]:outer.send_frame(wire,value,lambda:None)
    proc=FakeProc(wire.getvalue());events=[]
    class FakeAlarms:
        def arm(self,*a,**k):events.append('fake-alarm-arm')
        def check(self):pass
        def close(self):events.append('fake-alarm-close')
    def private_write(folder,name,raw,check):check();put(Path(folder)/name,raw)
    ports=NS(**{**vars(outer),'Alarms':FakeAlarms,'private_write':private_write})
    calls=[10.,12.]
    def clock():return calls.pop(0) if calls else 13.
    def popen(*a,**k):events.append('fake-Popen');return proc
    def write(fd,data):assert fd==900001;return proc.stdin.write(data)
    fake_os=NS(**{**vars(os),'write':write})
    c={'outer_source_sha256':'d'*64,'supervised_control_sha256':'e'*64,'supervised_control_text':'FAKE-SUPERVISED-CONTROL'}
    header={'base':{'root_binding':{'root':claim['scratch'],'parent':'/var/tmp'},'connection_started':98.,
        'collection_started':150.,'collection_deadline':750.}}
    class FakeWireReader:
        def __init__(self,*a):pass
        def frame(self,*a):return b'EXPLICIT-FAKE-HEADER',header
    def receive(*a,**k):events.append('fake-collection');return {'fake_collection':True,'science_qualified':False}
    collection=NS(WireReader=FakeWireReader,MAGIC=b'FAKE-CODEC\n',HEADER_CAP=65536,
        validate_header=lambda *a:events.append('fake-header-check'),receive=receive)
    project=NS(predictor_pgd_workload_handoff_outer_v1=ports,
        predictor_pgd_workload_bootstrap_v1=NS(encoded=outer.encoded),predictor_pgd_workload_collection_v1=collection)
    original=builtins.__import__
    imports={'os':fake_os,'subprocess':NS(Popen=popen,PIPE=-1,TimeoutExpired=subprocess.TimeoutExpired),
        'threading':NS(Thread=FakeThread),'time':NS(monotonic=clock),'models':project}
    def load(name,*a,**k):return imports[name] if name in imports else original(name,*a,**k)
    ns={'__name__':'_UNBOUND_EXPLICIT_FAKE_CLIENT','__file__':'<UNBOUND-FAKE-CLIENT>',
        '__builtins__':{**vars(builtins),'__import__':load}}
    exec(compile(source,'<UNBOUND-FAKE-CLIENT>','exec'),ns)
    folder=area/'fake-package';folder.mkdir(mode=0o700)
    put(folder/'AUTHORIZATION.json',{'origin':ORIGIN,'actual_user_approval':False})
    packet=area/'FAKE-PACKET.bin';put(packet,b'EXPLICIT FAKE BYTES; NO PIXELS OR TENSORS\n')
    auth={'user_basis':ORIGIN}
    ns['check_local']=lambda _: (c,{}, {},CSHA,b'FAKE-CONTROL',b'FAKE-AUTH',auth,{},b'FAKE-OUTER-SOURCE',(packet,))
    return ns,folder,proc,events,claimed


def test_byte_inverses_preserve_every_other_client_and_server_byte():
    raw,record=repair.derive(ROOT);assert raw==NEW and len(record['edits'])==3
    for edit in reversed(record['edits']):
        assert repair.digest(raw)==edit['derived_sha256'];start=edit['start'];after=edit['after'].encode()
        assert raw[start:start+len(after)]==after
        raw=raw[:start]+edit['before'].encode()+raw[start+len(after):]
        assert repair.digest(raw)==edit['original_sha256']
    assert raw==OLD and record['unbound'] and not record['old_sources_changed']
    assert not record['server_sources_changed'] and not record['scientific_or_resource_clock_gates_changed']


def test_old_client_rejects_source_exact_server_claim_with_reported_error(area):
    ns,folder,proc,events,claim=client_ports(area,OLD)
    with pytest.raises(ValueError,match='Durable remote claim or continuing connection clock changed'):ns['run'](folder)
    assert (folder/'ATTEMPT.json').exists() and (folder/'attended-session/CLIENT-REFUSAL.json').exists()
    assert not (folder/'attended-session/CLAIM-OBSERVED.json').exists()
    assert not (folder/'attended-session/ROOT-RECEIPT.json').exists() and 'fake-collection' not in events
    assert proc.terminated


def test_repaired_unbound_client_accepts_source_exact_claim_with_fake_ports(area):
    ns,folder,proc,events,claim=client_ports(area,NEW)
    assert ns['run'](folder)==0 and 'fake-collection' in events and not proc.terminated
    session=folder/'attended-session'
    assert json.loads((session/'CLAIM-OBSERVED.json').read_bytes())==claim
    assert json.loads((session/'ROOT-RECEIPT.json').read_bytes())==claim
    assert json.loads((session/'COLLECTION-RECEIPT.json').read_bytes())['fake_collection']
    assert not (session/'CLIENT-REFUSAL.json').exists()
    put(area/'RESULT.json',{'result':'PASS','origin':ORIGIN,'fake_client_lifecycle_complete':True,
        'actual_collection_or_science_qualified':False,'real_target_grant_or_attempt':False})


@pytest.mark.parametrize('fault',['candidate-prefix','prefix-spoof','parent','uid','hash-path','hash-field',
    'kind','hello','received','origin','work','connection','training-flag'])
def test_repaired_client_preserves_rejected_claim_and_retains_guards(fault,area):
    ns,folder,proc,events,claim=client_ports(area,NEW,fault)
    with pytest.raises(ValueError,match='Durable remote claim or continuing connection clock changed'):ns['run'](folder)
    session=folder/'attended-session'
    assert json.loads((session/'CLAIM-OBSERVED.json').read_bytes())==claim
    assert (session/'CLIENT-REFUSAL.json').exists() and not (session/'ROOT-RECEIPT.json').exists()
    assert not (session/'COLLECTION-READY.json').exists() and 'fake-collection' not in events and proc.terminated
    put(area/'RESULT.json',{'result':'PASS','origin':ORIGIN,'fault':fault,'rejected_claim_preserved':True,
        'actual_collection_or_science_qualified':False,'real_target_grant_or_attempt':False})
