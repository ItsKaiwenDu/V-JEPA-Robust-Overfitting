"""Read-only operator tests: all process/query/selector bindings are inert."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from models import university_gpu_availability_v1 as op
from scripts import university_gpu_availability_remote_v1 as remote

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'runs/pretrained/diagnostic/predictor-pgd-outer-v4-20261116/handoff-parent/linux10603-unarmed'
SAVED=json.loads((OLD/'attended-session/REFUSAL.json').read_bytes())['last_guard_refusal']['queries']
GPU_TEXT=SAVED[0]['text'];NX_TEXT=SAVED[1]['text'];IDLE_TEXT=GPU_TEXT.replace(', 293,',', 94,')
NONCE='c'*64;CSHA='a'*64;PSHA='b'*64


def query_records(gpu=IDLE_TEXT,compute=''):
    return [{'argv':list(argv),'exit_status':0,'stdout':text,'stderr':'','capture_error':None} for argv,text in zip(remote.COMMANDS,(gpu,compute))]


@pytest.mark.parametrize('case,expected',[
 ('idle','idle_observed'),('NX','busy_observed'),('memory','busy_observed'),('UUID','identity_refused'),
 ('badPID','unknown'),('duplicatePID','unknown'),('unknownMemory','unknown'),('queryFailed','unknown'),('tooMany','unknown')])
def test_classification_joins_actual_saved_query_shape_and_refuses_bad_or_busy_states(case,expected):
    q=query_records()
    if case=='NX':q=query_records(GPU_TEXT,NX_TEXT)
    elif case=='memory':q[0]['stdout']=IDLE_TEXT.replace(', 94,',', 600,')
    elif case=='UUID':q[0]['stdout']=IDLE_TEXT.replace(remote.GPU['uuid'],'GPU-OTHER')
    elif case=='badPID':q[1]['stdout']='no-pid, /unknown, 1\n'
    elif case=='duplicatePID':q[1]['stdout']=NX_TEXT+NX_TEXT
    elif case=='unknownMemory':q[1]['stdout']='1, /unknown, N/A\n'
    elif case=='queryFailed':q[1]['exit_status']=1
    elif case=='tooMany':q[1]['stdout']=''.join(str(i)+', /inert, 1\n' for i in range(1,66))
    result=remote.evaluate(dict(remote.HOST),q)
    assert result['availability']==expected
    if case=='NX':assert result['compute_processes']==[{'pid':3418810,'process_name':'/usr/NX/bin/nxnode.bin','used_memory_MiB':191.0}]


def test_wrong_host_never_runs_a_query_or_grants_model_work():
    result=remote.observe(NONCE,CSHA,PSHA,identity=lambda:{'hostname':'linux10609','system':'Linux','python':'3.11.17'},
        capture_fn=lambda *a,**kw:pytest.fail('Wrong host must not query'),clock=lambda:10.0,utc=lambda:'2026-11-16T23:00:00+00:00')
    assert result['availability']=='identity_refused' and result['queries']==[]
    assert result['read_only'] and not result['GPU_reserved'] and not result['model_work_authorized'] and not result['training_admitted']


def test_observer_uses_only_two_exact_bounded_commands_and_UTC_output():
    calls=[];values=[GPU_TEXT.encode(),NX_TEXT.encode()]
    def fake(argv,**kw):calls.append((argv,kw));return 0,values.pop(0),b''
    result=remote.observe(NONCE,CSHA,PSHA,capture_fn=fake,identity=lambda:dict(remote.HOST),clock=lambda:10.0,utc=lambda:'2026-11-16T23:00:00+00:00')
    assert [q[0] for q in calls]==list(remote.COMMANDS)
    assert all(q[1]=={'deadline':18.0,'stdout_cap':8192,'stderr_cap':2048} for q in calls)
    assert result['availability']=='busy_observed' and len(remote.encoded(result))<=65536
    assert result['request_nonce']==NONCE and result['control_sha256']==CSHA and result['program_sha256']==PSHA


def test_query_timeout_retains_partial_diagnostics_and_stops_later_query():
    calls=[]
    def fake(*a,**kw):calls.append(a);raise remote.CaptureError('deadline',b'partial',b'inert error')
    result=remote.observe(NONCE,CSHA,PSHA,capture_fn=fake,identity=lambda:dict(remote.HOST),clock=lambda:10.0,utc=lambda:'2026-11-16T23:00:00+00:00')
    assert len(calls)==1 and result['availability']=='unknown'
    assert result['queries'][0]['stdout']=='partial' and result['queries'][0]['stderr']=='inert error'


class Pipe:
    def __init__(self,fd):self.fd=fd;self.closed=False
    def fileno(self):return self.fd
    def close(self):self.closed=True
class Child:
    def __init__(self):self.stdout=Pipe(101);self.stderr=Pipe(102);self.status=None;self.killed=False;self.waits=[]
    def poll(self):return self.status
    def kill(self):self.killed=True
    def wait(self,timeout):self.waits.append(timeout);self.status=0;return 0
class Selector:
    def __init__(self):self.mapping={};self.closed=False
    def register(self,fd,event,data):self.mapping[fd]=NS(fd=fd,data=data)
    def get_map(self):return self.mapping
    def select(self,timeout):assert 0<timeout<=.25;return [(k,1) for k in list(self.mapping.values())]
    def unregister(self,fd):self.mapping.pop(fd)
    def close(self):self.closed=True


def fake_capture(chunks,*,clock=lambda:10.0,deadline=20.0,outcap=32):
    child=Child();selector=Selector();calls=[];queues={k:list(v) for k,v in chunks.items()}
    def popen(argv,**kw):
        assert kw['shell'] is False and kw['close_fds'] is True;calls.append(argv);return child
    def read(fd,n):
        assert n>0;value=queues[fd][0]
        if len(value)>n:queues[fd][0]=value[n:];return value[:n]
        queues[fd].pop(0);return value
    try:result=remote.capture(['EXPLICIT-INERT-OWNED-CHILD'],deadline=deadline,stdout_cap=outcap,stderr_cap=32,
        clock=clock,popen=popen,selector_factory=lambda:selector,set_blocking=lambda fd,mode:None,read=read)
    except remote.CaptureError as error:result=error
    return NS(result=result,child=child,selector=selector,calls=calls)


def test_capture_bounds_nonblocking_pipes_and_wait_without_real_subprocess():
    x=fake_capture({101:[b'OK',b''],102:[b'warning',b'']})
    assert x.result==(0,b'OK',b'warning') and not x.child.killed and len(x.calls)==1
    assert x.child.stdout.closed and x.child.stderr.closed and x.selector.closed
    assert x.child.waits==[10.0]


def test_capture_output_bound_preserves_prefix_and_kills_only_its_owned_fake_child():
    x=fake_capture({101:[b'OK',b'X'*10,b''],102:[b'W',b'']},outcap=3)
    assert isinstance(x.result,remote.CaptureError) and x.result.stdout==b'OK' and x.result.stderr==b'W'
    assert x.child.killed and x.child.stdout.closed and x.selector.closed and len(x.calls)==1
    assert x.child.waits==[.25]


def test_capture_expired_deadline_never_spawns_or_retries():
    with pytest.raises(remote.CaptureError):remote.capture(['INERT'],deadline=10.0,stdout_cap=1,stderr_cap=1,
        clock=lambda:10.0,popen=lambda *a,**kw:pytest.fail('Expired deadline spawned'))


@pytest.fixture
def package(tmp_path):
    parent=tmp_path/'private-parent';parent.mkdir(mode=0o700);folder=parent/'EXPLICIT-UNARMED-READ'
    result=op.prepare(ROOT,folder);c,csha=op.control((folder/'CONTROL.json').read_bytes())
    return NS(folder=folder,c=c,csha=csha,program=(folder/'QUERY.py').read_bytes(),result=result,tmp=tmp_path)


def arm_inert(x):
    auth=op.authorization(x.csha,'EXPLICIT INERT READ-ONLY TEST GRANT; NOT A LIVE USER APPROVAL')
    op.write_new(x.folder,'READ-AUTHORIZATION.json',op.encoded(auth))


def test_package_and_entry_remain_unarmed_without_model_claim_or_timer(package):
    x=package
    assert set(p.name for p in x.folder.iterdir())=={'CONTROL.json','QUERY.py','known-hosts','PACKAGE.json'}
    assert x.result['program_bytes']==len(x.program) and not x.result['authorization_created']
    spec=importlib.util.spec_from_file_location('EXPLICIT_UNARMED_READ_ENTRY',ROOT/'scripts/run_university_gpu_availability_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(SystemExit,match='No SSH, query or model'):module.main([str(x.folder)])
    with pytest.raises(ValueError,match='Unarmed'):op.run(x.folder,clock=lambda:pytest.fail('Unarmed touched clock'))
    assert not (x.folder/'READ-CONNECTION.json').exists() and not (x.folder/'ATTEMPT.json').exists()


@pytest.mark.parametrize('fault',['model-kind','extra-connection','model-flag','float-limit'])
def test_read_only_grant_cannot_use_model_authority_or_broaden_scope(package,fault):
    x=package;a=copy.deepcopy(op.authorization(x.csha,'EXPLICIT INERT TEST'))
    if fault=='model-kind':a['kind']='pgd-split-outer-university-exception-v4'
    elif fault=='extra-connection':a['maximum_connections']=2
    elif fault=='model-flag':a['flags']['model_work_authorized']=True
    else:a['limits']['client_connection_seconds_including_login']=180.0
    op.write_new(x.folder,'READ-AUTHORIZATION.json',op.encoded(a))
    with pytest.raises(ValueError):op.check_local(x.folder)
    assert not (x.folder/'READ-CONNECTION.json').exists()


def test_bound_query_and_copied_handoff_cannot_reuse_a_read_grant(package):
    x=package;arm_inert(x);other=x.tmp/'COPY';other.mkdir(mode=0o700)
    for p in x.folder.iterdir():q=other/p.name;q.write_bytes(p.read_bytes());q.chmod(0o600)
    with pytest.raises(ValueError,match='Copied'):op.check_local(other)
    (x.folder/'QUERY.py').write_bytes(b'CHANGED QUERY')
    with pytest.raises(ValueError):op.check_local(x.folder)


def response(x,nonce=NONCE,busy=True):
    outputs=[GPU_TEXT.encode() if busy else IDLE_TEXT.encode(),NX_TEXT.encode() if busy else b'']
    return remote.observe(nonce,x.csha,x.c['program_sha256'],capture_fn=lambda *a,**kw:(0,outputs.pop(0),b''),
        clock=lambda:10.0,identity=lambda:dict(remote.HOST),utc=lambda:'2026-11-16T23:00:00+00:00')


def test_command_pins_host_and_bounds_only_its_own_read_group(package):
    x=package;argv=op.ssh_command(x.folder,x.c,x.csha,x.program,NONCE)
    assert argv[0]=='ssh' and 'kdu@linux10603.dc.engr.scu.edu' in argv
    for option in ('StrictHostKeyChecking=yes','ForwardAgent=no','ClearAllForwardings=yes','PubkeyAuthentication=no','NumberOfPasswordPrompts=1','ConnectionAttempts=1'):
        assert option in argv
    assert '--kill-after=1s 20s' in argv[-1] and '-I -S -B -u -c' in argv[-1]
    assert 'GPU' not in argv[-2] and 'scp' not in argv and 'sftp' not in argv
    assert 'check_local' not in x.program.decode() and 'pgd' not in x.program.decode().lower()


@pytest.mark.parametrize('busy',[True,False])
def test_one_inert_SSH_response_is_saved_but_never_launches_or_authorizes_a_model(package,busy):
    x=package;arm_inert(x);calls=[]
    def capture(argv,**kw):
        calls.append((argv,kw));assert (x.folder/'READ-CONNECTION.json').is_file()
        return 0,op.encoded(response(x,busy=busy)),b''
    assert op.run(x.folder,capture_fn=capture,clock=lambda:10.0,nonce_fn=lambda:NONCE)==0
    assert len(calls)==1 and calls[0][1]=={'deadline':190.0,'stdout_cap':65536,'stderr_cap':16384}
    result=json.loads((x.folder/'read-response/RESULT.json').read_bytes())
    assert result['observation']['availability']==('busy_observed' if busy else 'idle_observed')
    assert not result['model_work_authorized'] and not result['model_attempt_claim_created'] and not result['GPU_reserved']
    assert not (x.folder/'ATTEMPT.json').exists()
    with pytest.raises(ValueError,match='already consumed'):op.run(x.folder,capture_fn=lambda *a,**kw:pytest.fail('Retry'),clock=lambda:10.0)
    (x.tmp/'INERT-READ-SESSION.json').write_bytes(op.encoded({'actual_SSH_or_GPU_queries':False,'result':result,'one_connection':len(calls)}))


@pytest.mark.parametrize('fault',['wrong-nonce','forged-idle','malformed','timeout'])
def test_bad_or_partial_responses_remain_evidence_and_never_advance_work(package,fault):
    x=package;arm_inert(x);r=response(x)
    if fault=='wrong-nonce':r['request_nonce']='e'*64
    elif fault=='forged-idle':r['availability']='idle_observed'
    data=b'INCOMPLETE' if fault=='malformed' else op.encoded(r)
    def capture(*a,**kw):
        if fault=='timeout':raise remote.CaptureError('deadline',data[:12],b'INERT STDERR')
        return 0,data,b''
    assert op.run(x.folder,capture_fn=capture,clock=lambda:10.0,nonce_fn=lambda:NONCE)==1
    result=json.loads((x.folder/'read-response/RESULT.json').read_bytes())
    assert result['error'] and result['observation'] is None and not result['model_work_authorized']
    assert (x.folder/'read-response/STDOUT.bin').read_bytes()==(data[:12] if fault=='timeout' else data)
    assert not (x.folder/'ATTEMPT.json').exists()


def test_capture_deadline_during_open_pipes_retains_prefix_and_stops_owned_child():
    ticks=iter([10.0,10.0,21.0,21.0])
    x=fake_capture({101:[b'partial',b''],102:[b'warning',b'']},clock=lambda:next(ticks),deadline=20.0)
    assert isinstance(x.result,remote.CaptureError) and 'deadline' in str(x.result)
    assert x.result.stdout==b'partial' and x.result.stderr==b'warning'
    assert x.child.killed and x.child.stdout.closed and x.child.stderr.closed and x.selector.closed
    assert x.child.waits==[0] and len(x.calls)==1
