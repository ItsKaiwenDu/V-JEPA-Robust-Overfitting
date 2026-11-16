"""New bounded outer driver checks with tiny bytes and fake OS/worker APIs.

No real process, pipe, timer, signal, /proc, proof allocation, GPU, SSH, array,
gradient, profiler or scientific worker is used. Only supervisor definitions run.
"""
import ast
import base64
import copy
import io
import json
import os
from pathlib import Path
import signal
import stat
import sys
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_split_outer_v1 as o
from models import predictor_pgd_split_outer_client_v1 as client
from models import predictor_pgd_split_outer_sources_v1 as adapter
from models import predictor_pgd_split_bootstrap_v1 as boot
from models import predictor_pgd_split_coordinator_v1 as frozen
from models import predictor_pgd_split_scratch_v1 as scratch

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'runs/pretrained/diagnostic/predictor-pgd-split-outer-v1-20261115'
BASIS='EXPLICIT INERT UNIT TEST ONLY; NOT A LIVE USER GRANT'


def put(root,name,data):
    p=root/name;p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    q=p.parent
    while q!=root:q.chmod(0o700);q=q.parent
    p.write_bytes(data);p.chmod(0o600);return p

@pytest.fixture(scope='module')
def package(tmp_path_factory):
    parent=tmp_path_factory.mktemp('UNARMED-SOURCE-PREPARATION');parent.chmod(0o700)
    folder=parent/'candidate'
    result=o.prepare_package(ROOT,folder)
    raw=(folder/'CONTROL-CANDIDATE.json').read_bytes();c,b,p,sha=o.validate_control(raw,o.digest((ROOT/o.LOCAL_SOURCES[0]).read_bytes()))
    mapping=o.parsed((folder/'SOURCE-PAYLOAD.json').read_bytes());sources=o.decode_sources(mapping,c,b)
    # Deliberately keep the fake grant IN MEMORY, never an actual armed handoff.
    auth=o.encoded(o.authorization(c,sha,BASIS))
    return NS(folder=folder,result=result,raw=raw,c=c,b=b,p=p,sha=sha,mapping=mapping,sources=sources,auth=auth)


def test_concrete_unarmed_package_has_all_69_sources_original_workload_and_financial_snapshot(package):
    x=package
    assert x.result['source_files']==69 and x.result['source_bytes']<2*o.MiB
    assert x.result['source_envelope_bytes']+len(x.raw)+64*1024<2*o.MiB
    assert not (x.folder/'AUTHORIZATION.json').exists() and not (x.folder/'ATTEMPT.json').exists()
    assert x.p==json.loads((ROOT/'configs/predictor_pgd_split_launch_v1.json').read_bytes())
    assert x.p['scope']['gradient_accounting']['full_model_objectives']==8
    assert x.p['scope']['gradient_accounting']['extra_component_VJP_evaluations']==2
    assert x.result['adapter_edits']=={'prepare_gpu':1,'guard_rebind':1,'launch_gate':1,'child':1,'lease':1}
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


def dumps(n):return ast.dump(n,include_attributes=False)
def methods(t,name):return {n.name:n for n in next(q for q in t.body if isinstance(q,ast.ClassDef) and q.name==name).body if isinstance(n,ast.FunctionDef)}

def test_exact_inverse_preserves_entire_original_launch_cleanup_and_prepare_bodies(package,tmp_path):
    old=(ROOT/'models/predictor_pgd_split_coordinator_v1.py').read_bytes()
    new=package.sources['supervisor/predictor_pgd_split_coordinator_v1.py']
    assert new.startswith(old+b'\n')
    original=ast.parse(old);adapted=ast.parse(new)
    assert [dumps(n) for n in adapted.body[:len(original.body)]]==[dumps(n) for n in original.body]
    parent=methods(original,'SystemDriver');derived=methods(adapted,'OuterDriver')
    prepare=copy.deepcopy(derived['prepare']);launch=copy.deepcopy(derived['_outer_owned_launch']);launch.name='launch'
    changed={'gpu':0,'rebind':0,'gate':0,'child':0,'lease':0}
    class Inverse(ast.NodeTransformer):
        def visit_Expr(self,n):
            if ast.unparse(n)=="self.outer_journal.checkpoint('setup')":changed['gpu']+=1;return ast.parse('gpu_gate()').body[0]
            return self.generic_visit(n)
        def visit_Assign(self,n):
            text=ast.unparse(n)
            keys={'self.outer_journal.guard = self.guard':'rebind','self.outer_child = child':'child','self.outer_lease = lease':'lease'}
            if text in keys:changed[keys[text]]+=1;return None
            return self.generic_visit(n)
        def visit_FunctionDef(self,n):
            if n.name=='gate':changed['gate']+=1;n.body=copy.deepcopy(next(q for q in ast.walk(parent['launch']) if isinstance(q,ast.FunctionDef) and q.name=='gate').body);return n
            return self.generic_visit(n)
    prepare=Inverse().visit(prepare);launch=Inverse().visit(launch)
    assert dumps(prepare)==dumps(parent['prepare']) and dumps(launch)==dumps(parent['launch'])
    assert changed=={'gpu':1,'rebind':1,'gate':1,'child':1,'lease':1}
    put(tmp_path,'UNEXECUTED-derived-coordinator.py.txt',new)
    put(tmp_path,'inverse-result.json',o.encoded({'passed':True,'five_inverse_changes':changed,'stop_and_finally_exact':True,'scientific_execution':False}))

@pytest.mark.parametrize('fault',['old-outer','code','known-host','fingerprint','old-supervisor','runtime','local-coverage','scope-policy','extra-key','noncanonical','duplicate'])
def test_outer_scope_source_host_or_policy_drift_refuses_before_any_target_write(package,fault):
    c=copy.deepcopy(package.c)
    if fault=='old-outer':c['kind']='OLD-CONSUMED'
    elif fault=='code':c['outer_source_sha256']='f'*64
    elif fault=='known-host':c['known_hosts_text']+='CHANGED'
    elif fault=='fingerprint':c['expected_ssh_fingerprint']='OLD'
    elif fault=='runtime':c['runtime_files'].pop(next(iter(c['runtime_files'])))
    elif fault=='local-coverage':c['local_source_sha256'].pop(next(iter(c['local_source_sha256'])))
    elif fault=='extra-key':c['port']='UNREVIEWED'
    else:
        b=o.parsed(c['supervised_control_text'])
        if fault=='old-supervisor':b['supervisor_sources']['bootstrap.py']['sha256']='e'*64
        elif fault=='scope-policy':b['policy_text']+='\n'
        c['supervised_control_text']=o.encoded(b).decode();c['supervised_control_sha256']=o.digest(c['supervised_control_text'].encode())
    raw=o.encoded(c)
    if fault=='noncanonical':raw+=b'\n'
    elif fault=='duplicate':raw=raw.rstrip()[:-1]+b',"kind":"pgd-split-outer-control-v1"}\n'
    with pytest.raises((ValueError,KeyError)):o.validate_control(raw,package.c['outer_source_sha256'])

@pytest.mark.parametrize('fault',['old-kind','wrong-control','wrong-base','repeat','paid','gradients','cleanup','basis','extra'])
def test_old_or_broadened_authorization_cannot_start_an_outer_attempt(package,fault):
    a=o.parsed(package.auth)
    if fault=='old-kind':a['kind']='OLD-CONSUMED'
    elif fault=='wrong-control':a['outer_control_sha256']='f'*64
    elif fault=='wrong-base':a['supervised_control_sha256']='f'*64
    elif fault=='repeat':a['maximum_new_attempts']=2
    elif fault=='paid':a['paid_compute']=True
    elif fault=='gradients':a['full_model_gradients']=9
    elif fault=='cleanup':a['cleanup']=True
    elif fault=='basis':a['user_basis']=''
    else:a['retry']=True
    with pytest.raises(ValueError):o.validate_authorization(o.encoded(a),package.c,package.sha)

@pytest.mark.parametrize('fault',['missing','extra','hash','extent','bad-base64'])
def test_exact_source_envelope_refuses_changed_names_or_bytes(package,fault):
    data=dict(package.mapping);name=next(iter(data))
    if fault=='missing':data.pop(name)
    elif fault=='extra':data['UNKNOWN.py']='aQ=='
    elif fault=='bad-base64':data[name]='%%INVALID%%'
    else:
        value=base64.b64decode(data[name]);value=(b'!'+value[1:]) if fault=='hash' else value+b'\n';data[name]=base64.b64encode(value).decode()
    with pytest.raises((ValueError,TypeError)):o.decode_sources(data,package.c,package.b)

@pytest.mark.parametrize('fault',['empty','magic','length','duplicate','noncanonical','truncated'])
def test_metadata_framing_is_bounded_canonical_and_never_resynchronizes(fault):
    data=o.encoded({'kind':'EXPLICIT-INERT'})
    wire=o.FRAME+len(data).to_bytes(4,'big')+data
    if fault=='empty':wire=b''
    elif fault=='magic':wire=b'X'+wire[1:]
    elif fault=='length':wire=o.FRAME+(65537).to_bytes(4,'big')
    elif fault=='duplicate':
        data=b'{"kind":"one","kind":"two"}\n';wire=o.FRAME+len(data).to_bytes(4,'big')+data
    elif fault=='noncanonical':
        data+=b'\n';wire=o.FRAME+len(data).to_bytes(4,'big')+data
    else:wire=wire[:-1]
    with pytest.raises(ValueError):o.frame(io.BytesIO(wire),lambda:None)


def test_unarmed_client_refuses_before_timer_claim_session_or_SSH(package,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('UNARMED EFFECT')
    monkeypatch.setattr(client.subprocess,'Popen',forbidden)
    monkeypatch.setattr(o,'Alarms',forbidden)
    with pytest.raises(ValueError,match='Unarmed'):client.run(package.folder)
    assert not (package.folder/'ATTEMPT.json').exists()


def test_ssh_is_one_pinned_direct_connection_with_no_agent_forwarding(tmp_path):
    cmd=client.ssh_command(tmp_path,b'EXPLICIT INERT SOURCE')
    assert cmd[0]=='ssh' and cmd[-2]=='kdu@linux10603.dc.engr.scu.edu'
    assert 'ForwardAgent=no' in cmd and 'StrictHostKeyChecking=yes' in cmd and 'ClearAllForwardings=yes' in cmd
    assert '-I -B -u -c' in cmd[-1] and 'download-only' not in cmd[-1]
    assert not any('id_ed25519' in v for v in cmd)

@pytest.mark.parametrize('origin,sent,received,end,now,expected',[(100,115,100000,100600,120,715),(100,100,30,3030,100,3100),(100,150,9000,9010,151,160)])
def test_remote_clock_conversion_uses_causal_duration_bound_not_absolute_epochs(origin,sent,received,end,now,expected):
    clock=[now];w=client.ReceiverClock(origin,sent,received,end,clock=lambda:clock[0]);assert w.deadline==expected
    clock[0]=expected
    with pytest.raises(o.Deadline):w.check()

@pytest.mark.parametrize('fault',['nan','reverse','future-sent','extended','already-expired','rollback'])
def test_remote_clock_mapping_refuses_reset_or_invalid_deadlines(fault):
    args=[100.,110.,900.,1000.];clock=[120.]
    if fault=='nan':args[2]=float('nan')
    elif fault=='reverse':args[3]=800.
    elif fault=='future-sent':args[1]=121.
    elif fault=='extended':args[3]=4001.
    elif fault=='already-expired':args[3]=901.
    if fault=='rollback':
        w=client.ReceiverClock(*args,clock=lambda:clock[0]);clock[0]=119.
        with pytest.raises(o.Deadline):w.check()
    else:
        with pytest.raises((ValueError,o.Deadline)):client.ReceiverClock(*args,clock=lambda:clock[0])


@pytest.fixture
def alarm(monkeypatch):
    events=[];clock=[100.]
    api=NS(SIGALRM=signal.SIGALRM,SIGINT=signal.SIGINT,SIGTERM=signal.SIGTERM,SIGHUP=signal.SIGHUP,
        SIGUSR1=signal.SIGUSR1,SIG_DFL=signal.SIG_DFL,ITIMER_REAL=signal.ITIMER_REAL,getsignal=lambda n:('PREVIOUS',n),getitimer=lambda n:(0.,0.),
        signal=lambda *a:events.append(('handler',a)),setitimer=lambda *a:events.append(('timer',a)))
    timers=[];interrupts=[];main=object()
    class Timer:
        def __init__(self,seconds,callback):self.seconds=seconds;self.callback=callback;self.cancelled=False;timers.append(self)
        def start(self):pass # NO real thread or timer
        def cancel(self):self.cancelled=True
    monkeypatch.setattr(o,'signal',api)
    monkeypatch.setattr(o,'threading',NS(current_thread=lambda:main,main_thread=lambda:main,Timer=Timer))
    monkeypatch.setattr(o,'_thread',NS(interrupt_main=lambda number:interrupts.append(number)))
    return NS(events=events,clock=clock,api=api,timers=timers,interrupts=interrupts)


def test_parent_alarm_keeps_original_hard_bound_and_restores_handlers(alarm):
    a=o.Alarms(clock=lambda:alarm.clock[0]);a.arm(110.,hard_deadline=120.,phase='native')
    assert alarm.events[-1]==('timer',(signal.ITIMER_REAL,20.))
    alarm.clock[0]=110.
    with pytest.raises(o.Deadline):a.handler(signal.SIGALRM,None)
    assert alarm.events[-1]==('timer',(signal.ITIMER_REAL,2))
    a.close();assert ('timer',(signal.ITIMER_REAL,0)) in alarm.events
    assert alarm.events[-1]==('handler',(signal.SIGINT,('PREVIOUS',signal.SIGINT)))


def test_signal_interruption_uses_same_owned_cleanup_allowance_without_a_new_clock(alarm):
    a=o.Alarms(clock=lambda:alarm.clock[0]);a.arm(115.,hard_deadline=116.,phase='numerical')
    alarm.clock[0]=115.
    with pytest.raises(o.Stopped):a.handler(signal.SIGHUP,None)
    assert alarm.events[-1]==('timer',(signal.ITIMER_REAL,1.))
    a.close()

@pytest.mark.parametrize('fault',['nan','past','hard-before-soft','rollback','inherited-timer'])
def test_alarm_refuses_invalid_or_replaced_timer_state(alarm,fault):
    if fault=='inherited-timer':
        alarm.api.getitimer=lambda n:(3.,0.)
        with pytest.raises(ValueError):o.Alarms(clock=lambda:100.)
        return
    a=o.Alarms(clock=lambda:alarm.clock[0]);deadline,hard=110.,120.
    if fault=='nan':deadline=float('nan')
    elif fault=='past':deadline=99.
    elif fault=='hard-before-soft':hard=109.
    else:alarm.clock[0]=99.
    with pytest.raises((ValueError,o.Deadline)):a.arm(deadline,hard_deadline=hard,phase='setup')
    a.close()


def supervisor_namespace(package):
    # Execute standard-library SUPERVISOR DEFINITIONS only; no worker, import
    # finder installation, OS launch or scientific module is executed.
    old=ast.parse((ROOT/'models/predictor_pgd_split_coordinator_v1.py').read_bytes())
    full=ast.parse(package.sources['supervisor/predictor_pgd_split_coordinator_v1.py'])
    tail=ast.fix_missing_locations(ast.Module(body=full.body[len(old.body):],type_ignores=[]))
    namespace=dict(frozen.__dict__);namespace['__name__']='EXPLICIT_FAKE_SUPERVISOR_DEFINITIONS'
    exec(compile(tail,'<FAKE-OS-BOUND-SUPERVISOR-DEFINITIONS>','exec'),namespace)
    return namespace

@pytest.fixture
def journal(tmp_path,package):
    root=tmp_path/'tiny-metadata-only';root.mkdir(mode=0o700)
    b,p,sha=boot.control(package.c['supervised_control_text'].encode());ib,isha=boot.installation(b,sha)
    layout=scratch.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),ib,isha)
    guard=scratch.Guard(root,scratch.identity_binding(root),layout)
    ns=supervisor_namespace(package);queries=[];active=[''];clock=[100.]
    def query(command,**kw):
        queries.append(command)
        if '--query-gpu=name,uuid,memory.total,memory.used,driver_version' in command:
            return 'NVIDIA GeForce RTX 3070, GPU-e54fae90-feeb-8106-d8c0-1974fe48555f, 8192, 94, 580.178.04\n'
        return active[0]
    ns['subprocess']=NS(check_output=query,PIPE='EXPLICIT-FAKE-PIPE')
    j=ns['OuterJournal'](guard,lambda:clock[0])
    yield NS(j=j,ns=ns,root=root,queries=queries,active=active,clock=clock,guard=guard)
    j.close()


def test_every_gate_checks_GPU_but_only_bounded_intervals_persist_samples(journal):
    x=journal
    for tick in range(30):x.clock[0]=100.+tick;x.j.checkpoint('numerical')
    assert x.j.calls==30 and len(x.queries)==60 and x.j.rows==3
    lines=(x.root/'gpu-samples.jsonl').read_bytes().splitlines()
    assert len(lines)==3 and [json.loads(q)['guard_calls'] for q in lines]==[1,11,21]
    status=json.loads((x.root/'storage-status.json').read_bytes())
    assert status['guard_calls']==21 and not status['scientific_qualified']
    assert all(not json.loads(q)['sampling_is_continuous_exclusivity'] for q in lines)
    x.j.checkpoint('finalization');assert x.j.rows==4


def test_other_GPU_process_is_recorded_and_refused_without_signaling_it(journal):
    x=journal;x.active[0]='2522822, /usr/NX/bin/nxnode.bin, 183\n'
    with pytest.raises(ValueError,match='Another CUDA'):x.j.checkpoint('native')
    assert x.j.refusals==1 and x.j.last_refusal['storage_observation_available']
    record=json.loads((x.root/'gpu-samples.jsonl').read_bytes())
    assert '/usr/NX/bin/nxnode.bin' in json.dumps(record) and record['refusal']['error_type']=='ValueError'
    assert all(q[0]=='nvidia-smi' for q in x.queries)


def test_storage_refusal_does_not_add_bytes_to_an_invalid_tree(journal):
    x=journal;put(x.root,'UNDECLARED-PRESERVED.bin',b'TINY-NOT-A-PROOF')
    with pytest.raises(ValueError):x.j.checkpoint('numerical')
    assert x.j.refusals==1 and not x.j.last_refusal['storage_observation_available']
    assert not (x.root/'gpu-samples.jsonl').exists() and not x.queries
    assert (x.root/'UNDECLARED-PRESERVED.bin').read_bytes()==b'TINY-NOT-A-PROOF'

@pytest.mark.parametrize('fault',['calls','bytes','rows','clock','status-changed'])
def test_guard_journal_refuses_bounds_or_changed_owned_metadata_without_reset(journal,fault):
    x=journal
    if fault=='calls':x.j.calls=25000
    elif fault=='bytes':x.j.bytes=512*1024-8192 # synthetic boundary, not a real large file
    elif fault=='rows':x.j.rows=384
    elif fault=='clock':x.j.checkpoint('native');x.clock[0]=99.
    else:
        x.j.checkpoint('native');p=x.root/'storage-status.json';p.write_bytes(p.read_bytes()+b'\n');x.clock[0]=111.
    with pytest.raises(ValueError):x.j.checkpoint('native')


def test_outer_driver_will_not_handoff_clocks_until_owned_child_and_handle_are_settled(package):
    ns=supervisor_namespace(package);driver=object.__new__(ns['OuterDriver'])
    driver.outer_child=NS(poll=lambda:None);driver.outer_lease=NS(closed=False)
    with pytest.raises(RuntimeError,match='exit is unconfirmed'):driver.assert_idle()
    driver.outer_child=NS(poll=lambda:0)
    with pytest.raises(RuntimeError,match='handle cleanup'):driver.assert_idle()
    driver.outer_lease.closed=True;driver.assert_idle()


def test_target_preflight_refuses_local_machine_before_proc_or_GPU(monkeypatch):
    monkeypatch.setattr(o,'platform',NS(node=lambda:'EXPLICIT-MAC-TEST',system=lambda:'Darwin',python_version=lambda:'3.11.16'))
    with pytest.raises(ValueError,match='before any target write'):o.preflight_parent()


def test_deterministic_remote_root_is_a_per_grant_claim_not_a_random_retry_directory(package,tmp_path):
    a=o.remote_root(package.sha);b=o.remote_root(package.sha)
    assert a==b and a.parent==Path('/var/tmp') and a.name.endswith(package.sha)
    root=tmp_path/a.name;root.mkdir(mode=0o700);put(root,'proposal.json',b'{"EXPLICIT":"CONSUMED TEST CLAIM"}\n')
    with pytest.raises(FileExistsError):root.mkdir(mode=0o700)
    assert json.loads((root/'proposal.json').read_bytes())['EXPLICIT']=='CONSUMED TEST CLAIM'


def fake_alarm(events):
    class Fake:
        def __init__(self):events.append(('ALARM-CREATED',))
        def arm(self,deadline,**kw):events.append(('ARM',deadline,kw));return None
        def check(self):return 100.
        def close(self):events.append(('ALARM-CLOSED',))
    return Fake


@pytest.mark.parametrize('failure',[None,'work','setup','idle'])
def test_remote_one_connection_handoff_uses_fake_workers_and_keeps_failed_work_failed(package,tmp_path,monkeypatch,failure):
    events=[];now=[500.];root=tmp_path/'EXPLICIT-FAKE-REMOTE-CLAIM'
    monkeypatch.setattr(o,'Alarms',fake_alarm(events));monkeypatch.setattr(o.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(o.os,'urandom',lambda n:b'x'*n)
    monkeypatch.setattr(o,'preflight_parent',lambda:events.append(('PREFLIGHT-FAKE',)))
    monkeypatch.setattr(o,'remote_root',lambda csha:root)
    monkeypatch.setattr(o,'boot_namespace',lambda sources:NS(control=boot.control))
    class Journal:
        last_refusal=None
        def checkpoint(self,role):events.append(('GPU-STORAGE-FAKE',role))
        def close(self):events.append(('JOURNAL-CLOSED',))
    class Driver:
        outer_journal=Journal();guard=NS(EXPLICIT='INERT GUARD')
        def assert_idle(self):
            events.append(('OWNED-IDLE-FAKE',))
            if failure=='idle':raise RuntimeError('EXPLICIT UNREAPED FAKE CHILD')
        def save(self,*a,**kw):events.append(('FAILED-SESSION-SAVED',))
    class Coordinator:
        driver=Driver();state={}
        def __init__(self,*a,**kw):events.append(('COORDINATOR-FAKE',kw['connection_started']))
        def run_work(self):
            events.append(('SERIAL-WORK-FAKE-NO-COMPUTATION',))
            if failure=='work':raise ValueError('EXPLICIT FAKE WORK REFUSAL')
    def setup(*a):
        events.append(('SETUP-FAKE-NO-PROOF-NO-PIXELS',))
        if failure=='setup':raise ValueError('EXPLICIT FAKE SETUP REFUSAL')
        return NS(OuterCoordinator=Coordinator),NS(),NS(),{},b'EXPLICIT-INERT-DERIVED-AUTH'
    monkeypatch.setattr(o,'prepare_tree',setup)
    class Collector:
        def __init__(self,*a,**kw):self.window=NS(started=500.,deadline=1100.);events.append(('COLLECTOR-FAKE',))
        def prepare(self):events.append(('COLLECTION-PREPARED-FAKE',))
        def stream(self,out):out.write(b'EXPLICIT-INERT-BODY-NO-SCIENCE');events.append(('STREAM-FAKE',))
    monkeypatch.setattr(client.collection,'Collector',Collector)
    message={'nonce':(b'x'*32).hex(),'client_elapsed':10.,'control_text':package.raw.decode(),'authorization_text':package.auth.decode(),'sources':package.mapping}
    incoming=io.BytesIO();o.send_frame(incoming,message,lambda:None,cap=2*o.MiB);incoming.seek(0);out=io.BytesIO()
    if failure=='idle':
        with pytest.raises(RuntimeError,match='UNREAPED'):o.remote_session(incoming,out,code_sha256=package.c['outer_source_sha256'])
    else:
        status=o.remote_session(incoming,out,code_sha256=package.c['outer_source_sha256'])
        assert status==(1 if failure else 0)
    output=io.BytesIO(out.getvalue());hello=o.frame(output,lambda:None);claimed=o.frame(output,lambda:None)
    assert hello['kind']=='pgd-split-hello-v1' and claimed['connection_started']==490.
    assert claimed['work_deadline']==2890. and claimed['connection_deadline']==3490.
    assert root.is_dir()
    if failure not in ('setup','idle'):
        ready=o.frame(output,lambda:None);assert ready['kind']=='pgd-split-collection-ready-v1'
        assert output.read()==b'EXPLICIT-INERT-BODY-NO-SCIENCE'
        assert ('STREAM-FAKE',) in events
    else:assert ('STREAM-FAKE',) not in events
    if failure=='setup':assert ('SERIAL-WORK-FAKE-NO-COMPUTATION',) not in events
    put(tmp_path,'FAKE-REMOTE-EVENTS.json',o.encoded({'origin':'EXPLICIT INERT ENDPOINT, NOT TARGET EVIDENCE','failure':failure,'events':events,**o.FLAGS}))


@pytest.mark.parametrize('fault',[None,'root','clock','start','end'])
def test_collection_header_is_bound_to_same_claim_and_original_phase_before_reconstruction(package,monkeypatch,fault):
    h={'root_binding':{'root':'/var/tmp/EXPLICIT-ROOT','parent':'/var/tmp'},'connection_started':100.,'collection_started':200.,'collection_deadline':800.}
    claimed={'scratch':'/var/tmp/EXPLICIT-ROOT','connection_started':100.}
    ready={'collection_started':200.,'collection_deadline':800.}
    if fault=='root':h['root_binding']['root']='/var/tmp/WRONG-ROOT'
    elif fault=='clock':h['connection_started']=101.
    elif fault=='start':h['collection_started']=201.
    elif fault=='end':h['collection_deadline']=801.
    raw=o.encoded(h)
    class Reader:
        def __init__(self,*a):pass
        def frame(self,*a):return raw,h
    monkeypatch.setattr(client.collection,'WireReader',Reader)
    monkeypatch.setattr(client.collection,'validate_header',lambda *a:None) # header decoder already checked separately; this tests only the new join
    if fault:
        with pytest.raises(ValueError,match='another root or clock'):client.bound_collection_source(io.BytesIO(),NS(),claimed,ready,package.c,b'INERT')
    else:
        joined=client.bound_collection_source(io.BytesIO(b'BODY'),NS(),claimed,ready,package.c,b'INERT')
        assert joined.read(4096)==client.collection.MAGIC+len(raw).to_bytes(4,'big')+raw
        assert joined.read(4096)==b'BODY'


@pytest.mark.parametrize('failure',[None,'server-refusal','client-error','transport-exit'])
def test_attended_client_claim_is_durable_single_SSH_no_retry_using_fake_endpoint(package,tmp_path,monkeypatch,failure):
    folder=tmp_path/'EXPLICIT-FAKE-CLIENT';folder.mkdir(mode=0o700)
    # Not a valid authorization file. Only the injected fake check below supplies
    # in-memory metadata, so these saved fixtures cannot arm a real connection.
    put(folder,'AUTHORIZATION.json',b'{"kind":"EXPLICIT-INERT-NOT-A-GRANT"}\n')
    packet=put(folder,'INERT-NOT-A-TRAINING-PACKET.bin',b'INERT-NO-PIXELS')
    c=copy.deepcopy(package.c);c['local_handoff']=str(folder);csha=o.digest(o.encoded(c));a=o.authorization(c,csha,BASIS)
    checked=(c,package.b,package.p,csha,o.encoded(c),o.encoded(a),a,package.mapping,b'EXPLICIT-INERT-BOOTSTRAP',packet)
    events=[];monkeypatch.setattr(o,'Alarms',fake_alarm(events));monkeypatch.setattr(client.time,'monotonic',lambda:100.)
    real_clock=client.ReceiverClock
    monkeypatch.setattr(client,'ReceiverClock',lambda *a:real_clock(*a,clock=lambda:100.))
    monkeypatch.setattr(client,'check_local',lambda path:checked)
    replies=io.BytesIO()
    o.send_frame(replies,{'kind':'pgd-split-hello-v1','nonce':'a'*64,'hello_monotonic':500.,'code_sha256':c['outer_source_sha256']},lambda:None)
    claim={'kind':'pgd-split-root-claimed-v1','outer_control_sha256':csha,'scratch':'/var/tmp/vjepa-pgd-split-v1-1000-'+csha,
        'hello_monotonic':500.,'handshake_received':510.,'connection_started':500.,'work_deadline':2900.,'connection_deadline':3500.,**o.FLAGS}
    if failure=='server-refusal':o.send_frame(replies,{'kind':'pgd-split-outer-refusal-v1','message':'EXPLICIT FAKE REFUSAL',**o.FLAGS},lambda:None)
    else:
        o.send_frame(replies,claim,lambda:None)
        o.send_frame(replies,{'kind':'pgd-split-collection-ready-v1','outer_control_sha256':csha,'connection_started':500.,
            'collection_started':600.,'collection_deadline':1200.,'handshake_received':510.,**o.FLAGS},lambda:None)
    replies.seek(0)
    class Stdin(io.BytesIO):
        def fileno(self):return 777777 # FAKE; no pipe or actual descriptor is opened
        def close(self):events.append(('INPUT-CLOSED-FAKE',))
    class Process:
        pid=777777
        def __init__(self,*args,**kw):
            events.append(('SSH-POPEN-FAKE',));self.stdin=Stdin();self.stdout=replies;self.stderr=io.BytesIO();self.done=False
        def poll(self):return (7 if failure=='transport-exit' else 0) if self.done else None
        def wait(self,timeout):self.done=True;return self.poll()
        def terminate(self):self.done=True;events.append(('SSH-TERMINATED-FAKE',))
        def kill(self):self.done=True;events.append(('SSH-KILLED-FAKE',))
    monkeypatch.setattr(client,'subprocess',NS(Popen=Process,PIPE='FAKE-PIPE',TimeoutExpired=TimeoutError))
    fakeos=NS(**{name:getattr(os,name) for name in dir(os) if not name.startswith('__')});fakeos.write=lambda fd,data:len(data)
    monkeypatch.setattr(client,'os',fakeos)
    class Thread:
        def __init__(self,target,**kw):self.target=target
        def start(self):self.target()
        def join(self,**kw):pass
    monkeypatch.setattr(client,'threading',NS(Thread=Thread))
    monkeypatch.setattr(client,'bound_collection_source',lambda source,*a:source)
    def receive(source,destination,*a,**kw):
        events.append(('COLLECT-FAKE',kw['window'].deadline))
        if failure=='client-error':raise ValueError('EXPLICIT FAKE INCOMPLETE COLLECTION')
        destination.mkdir(mode=0o700)
        return {'kind':'EXPLICIT-FAKE-COLLECTION-NO-SCIENTIFIC-DATA','transport_verified':True,'files':{},**o.FLAGS}
    monkeypatch.setattr(client.collection,'receive',receive)
    if failure in ('server-refusal','client-error'):
        with pytest.raises((ValueError,RuntimeError)):client.run(folder)
    else:assert client.run(folder)==(7 if failure=='transport-exit' else 0)
    claimbytes=(folder/'ATTEMPT.json').read_bytes()
    assert json.loads(claimbytes)['consumed'] and len([e for e in events if e[0]=='SSH-POPEN-FAKE'])==1
    with pytest.raises(FileExistsError):client.run(folder)
    assert (folder/'ATTEMPT.json').read_bytes()==claimbytes and len([e for e in events if e[0]=='SSH-POPEN-FAKE'])==1
    if failure!='server-refusal':assert ('COLLECT-FAKE',790.) in events
    put(folder,'FAKE-CLIENT-EVENTS.json',o.encoded({'origin':'EXPLICIT FAKE ENDPOINT; NEVER A REAL GRANT','failure':failure,'events':events,**o.FLAGS}))


def test_cooperative_parent_check_catches_elapsed_phase_even_before_python_signal_delivery(alarm):
    a=o.Alarms(clock=lambda:alarm.clock[0]);a.arm(110.,hard_deadline=120.,phase='native')
    alarm.clock[0]=110.
    with pytest.raises(o.Deadline):a.check()
    a.close()


def test_expired_phase_can_only_be_replaced_by_explicit_existing_bound_handoff(alarm):
    a=o.Alarms(clock=lambda:alarm.clock[0]);a.arm(110.,hard_deadline=120.,phase='native')
    alarm.clock[0]=111.
    with pytest.raises(o.Deadline):a.check()
    # This explicit arm models the adapter's AFTER assert_idle handoff. It must
    # stay inside the old hard budget, rather than inventing a fresh work budget.
    a.arm(119.,hard_deadline=120.,phase='finalization');assert a.check()==111.
    a.close()


@pytest.mark.parametrize('fault',[None,'sparse','allocator-error'])
def test_setup_orders_sources_scope_namespace_and_one_reserve_before_fake_packet_verification(package,tmp_path,monkeypatch,fault):
    root=tmp_path/'EXPLICIT-INERT-SETUP';root.mkdir(mode=0o700)
    events=[]
    class Installer:
        def __init__(self,*a):events.append(('FINDER-CONSTRUCTED-NO-EXEC',))
        def install(self):
            assert (root/'proposal.json').is_file() and (root/'control.json').is_file()
            assert len(list((root/'supervisor').glob('*.py')))==6
            events.append(('FINDER-INSTALL-FAKE',));return self
    b=NS(control=boot.control,encoded=boot.encoded,authorization=boot.authorization,installation=boot.installation,PinnedImporter=Installer)
    class Layout:
        directories={'.','capture','capture/results','capture/logs','cache','cache/cuda','tmp','code','code/input'}
        def __init__(self,*a):events.append(('LAYOUT-FAKE',))
    class Guard:
        def __init__(self,*a):events.append(('GUARD-FAKE',))
        def scan(self,**kw):events.append(('STORAGE-OBSERVATION-FAKE',));return {'origin':'EXPLICIT SYNTHETIC CAPACITY; NOT A STORAGE PROOF'},{},{}
        def verify_assets(self,**kw):events.append(('ASSET-VERIFY-FAKE',))
        def proof_readiness(self):events.append(('PROOF-READINESS-FAKE-NOT-EXECUTION',))
    monkeypatch.setattr(scratch,'Layout',Layout);monkeypatch.setattr(scratch,'Guard',Guard)
    monkeypatch.setattr(frozen,'target_gate',lambda *a:events.append(('HOST-GATE-FAKE',)))
    monkeypatch.setattr(o,'receive_packet',lambda *a:events.append(('PACKET-RECEIVE-FAKE-NO-PIXELS',)))
    monkeypatch.setattr(o,'extract_packet',lambda *a:events.append(('PACKET-VERIFY-FAKE-NO-PIXELS',)))
    realos=os
    def allocate(fd,start,size):
        events.append(('RESERVE-FAKE-NO-ALLOCATION',start,size))
        if fault=='allocator-error':raise OSError('EXPLICIT unsupported preallocation')
    def fstat(fd):
        info=realos.fstat(fd)
        reserve=root/'emergency-space.bin'
        if reserve.exists() and info.st_ino==reserve.stat().st_ino:
            return NS(st_size=16*o.MiB,st_blocks=0 if fault=='sparse' else 16*o.MiB//512)
        return info
    fakeos=NS(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')});fakeos.posix_fallocate=allocate;fakeos.fstat=fstat
    monkeypatch.setattr(o,'os',fakeos)
    args=(root,package.sources,package.c,package.b,package.p,o.parsed(package.auth),package.raw,package.auth,b,io.BytesIO(),lambda:None)
    if fault:
        with pytest.raises((ValueError,OSError)):o.prepare_tree(*args)
        assert ('PACKET-RECEIVE-FAKE-NO-PIXELS',) not in events
    else:
        coord,_,_,binding,derived=o.prepare_tree(*args)
        assert binding['root']==str(root) and boot.authorization(derived,package.c['supervised_control_sha256'])['user_basis']==BASIS
        assert events.index(('FINDER-INSTALL-FAKE',))<events.index(('HOST-GATE-FAKE',))<events.index(('PACKET-RECEIVE-FAKE-NO-PIXELS',))
        assert events[-1]==('PROOF-READINESS-FAKE-NOT-EXECUTION',)
    assert (root/'emergency-space.bin').stat().st_size==0 # physical fixture is tiny, NEVER a 16 MiB/2.5 GiB proof
    assert ('RESERVE-FAKE-NO-ALLOCATION',0,16*o.MiB) in events
    put(root,'SETUP-TEST-EVENTS.json',o.encoded({'origin':'EXPLICIT SUBSTITUTED SETUP; NO ACTUAL INSTALLATION/ALLOCATION/PIXELS','events':events,'failure':fault,**o.FLAGS}))


@pytest.mark.parametrize('fault',['repeat','escape','symlink-parent','public-parent','oversized'])
def test_setup_metadata_writer_refuses_nonprivate_escaped_or_replaced_source_paths(tmp_path,fault):
    root=tmp_path/'private';root.mkdir(mode=0o700);name='source/data.json';data=b'INERT'
    if fault=='repeat':o.private_write(root,name,data,lambda:None)
    elif fault=='escape':name='../ESCAPE'
    elif fault=='symlink-parent':
        other=tmp_path/'PRESERVED';other.mkdir(mode=0o700);(root/'source').symlink_to(other,target_is_directory=True)
    elif fault=='public-parent':(root/'source').mkdir(mode=0o755)
    else:data=b'x'*(2*o.MiB+1) # source-byte bound only; no scientific array or large proof
    with pytest.raises((ValueError,FileExistsError,OSError)):o.private_write(root,name,data,lambda:None)
    if fault=='repeat':assert (root/name).read_bytes()==b'INERT'
    assert not (tmp_path/'ESCAPE').exists()


def test_truncated_input_preserves_tiny_partial_and_never_installs_archive(package,tmp_path):
    spec=o.parsed(package.p['scratch_contract_text'])['packet']['archive']
    root=tmp_path/'tiny-truncated-packet';root.mkdir(mode=0o700)
    with pytest.raises(ValueError,match='truncated'):o.receive_packet(io.BytesIO(b'INERT'),root,spec,lambda:None)
    assert (root/'original-packet.tar.gz.partial').is_file() and not (root/'original-packet.tar.gz').exists()
    assert (root/'original-packet.tar.gz.partial').stat().st_size==0 # incomplete chunk is never published/written


@pytest.mark.parametrize('fault',['size','hash'])
def test_changed_original_packet_descriptor_refuses_before_any_input_file_creation(package,tmp_path,fault):
    spec=copy.deepcopy(o.parsed(package.p['scratch_contract_text'])['packet']['archive'])
    if fault=='size':spec['bytes']-=1
    else:spec['sha256']='0'*64
    root=tmp_path/'descriptor-refusal';root.mkdir(mode=0o700)
    with pytest.raises(ValueError):o.receive_packet(io.BytesIO(),root,spec,lambda:None)
    assert list(root.iterdir())==[]


def test_CLI_refuses_unarmed_before_importing_the_client_or_creating_a_claim(tmp_path):
    import importlib.util
    path=ROOT/'scripts/run_predictor_pgd_split_outer_v1.py'
    spec=importlib.util.spec_from_file_location('EXPLICIT_UNARMED_CLI_TEST',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    empty=tmp_path/'UNARMED';empty.mkdir(mode=0o700)
    with pytest.raises(SystemExit,match='No timer, claim, SSH or model work'):module.main([str(empty)])
    assert list(empty.iterdir())==[]


def test_full_stdout_response_budget_is_shared_with_all_outer_frames_without_large_allocation():
    out=io.BytesIO();budget=o.OutputBudget(out,lambda:None)
    budget.count=3*o.GiB-2 # synthetic numeric boundary only
    assert budget.write(b'OK')==2 and budget.count==3*o.GiB
    with pytest.raises(ValueError):budget.write(b'X')
    assert out.getvalue()==b'OK'


def test_local_bound_input_checks_are_inside_the_original_client_timer(tmp_path,monkeypatch):
    folder=tmp_path/'EXPLICIT-INERT';folder.mkdir(mode=0o700)
    put(folder,'AUTHORIZATION.json',b'{"kind":"EXPLICIT-INERT-NOT-A-GRANT"}')
    events=[];monkeypatch.setattr(o,'Alarms',fake_alarm(events));monkeypatch.setattr(client.time,'monotonic',lambda:100.)
    def check(path):events.append(('BOUND-INPUT-CHECK-FAKE',));raise ValueError('EXPLICIT INPUT REFUSAL')
    monkeypatch.setattr(client,'check_local',check)
    with pytest.raises(ValueError,match='INPUT REFUSAL'):client.run(folder)
    assert events[1][0]=='ARM' and events[2][0]=='BOUND-INPUT-CHECK-FAKE'
    assert not (folder/'ATTEMPT.json').exists()


def test_kernel_hard_alarm_is_default_from_start_without_waiting_for_a_python_handler(alarm):
    a=o.Alarms(clock=lambda:alarm.clock[0]);a.arm(110.,hard_deadline=120.,phase='native')
    handlers=[x for x in alarm.events if x[0]=='handler' and x[1][0]==signal.SIGALRM]
    assert handlers[-1]==('handler',(signal.SIGALRM,signal.SIG_DFL))
    assert alarm.events[-1]==('timer',(signal.ITIMER_REAL,20.))
    a.close()


def test_cancelled_soft_notifications_cannot_signal_a_later_phase_or_closed_alarm(alarm):
    a=o.Alarms(clock=lambda:alarm.clock[0]);a.arm(110.,hard_deadline=140.,phase='native')
    first=alarm.timers[-1];a.arm(120.,hard_deadline=140.,phase='finalization')
    assert first.cancelled;first.callback();assert alarm.interrupts==[]
    current=alarm.timers[-1];current.callback();assert alarm.interrupts==[signal.SIGUSR1]
    a.close();current.callback();assert alarm.interrupts==[signal.SIGUSR1]


def test_stale_pending_soft_notification_before_current_deadline_cannot_extend_or_stop_new_phase(alarm):
    a=o.Alarms(clock=lambda:alarm.clock[0]);a.arm(120.,hard_deadline=140.,phase='finalization')
    before=list(alarm.events);a.handler(signal.SIGUSR1,None)
    assert alarm.events==before and a.soft_deadline==120. and a.hard==140.
    alarm.clock[0]=120.
    with pytest.raises(o.Deadline):a.handler(signal.SIGUSR1,None)
    assert alarm.events[-1]==('timer',(signal.ITIMER_REAL,2))
    a.close()
