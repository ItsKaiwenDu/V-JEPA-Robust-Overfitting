"""Cold source and fake OS/compute controls. No scientific modules are imported."""
import ast
import builtins
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_workload_runtime_v1 as cold
from models import predictor_pgd_workload_session_v1 as session
from models import predictor_pgd_workload_runtime_records_v1 as records

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_science_or_live_OS(monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kw):
        if name.split('.')[0] in ('torch','numpy','socket','subprocess'):
            raise AssertionError('No scientific/process/network import: '+name)
        return original(name,*args,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    def refused(*a,**kw):raise AssertionError('No process, signal, timer or sleep.')
    import signal,time
    for module,name in ((os,'kill'),(os,'killpg'),(os,'fork'),(os,'system'),(os,'posix_spawn'),
                        (signal,'setitimer'),(signal,'signal'),(time,'sleep')):
        if hasattr(module,name):monkeypatch.setattr(module,name,refused)
    monkeypatch.setattr(os,'fstatvfs',lambda fd:NS(f_bavail=20*2**30,f_frsize=1,f_favail=100000))


@pytest.fixture
def case(request,tmp_path):
    base=os.environ.get('VJEPA_WHOLE_PGD_RUNTIME_EVIDENCE')
    folder=Path(base)/request.node.name if base else tmp_path/'case'
    folder.mkdir(mode=0o700)
    return folder


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with path.open('xb') as out:out.write(data)
    path.chmod(0o600)


class Clock:
    def __init__(self):self.now=100.
    def __call__(self):return self.now


class Child:
    def __init__(self,pid):self.pid=pid;self.exited=False;self.status=0
    def poll(self):return self.status if self.exited else None
    def wait(self,timeout=None):return self.status


class Lease:
    def __init__(self,child,events):
        self.child=child;self.pid=child.pid;self.birth=child.pid+500;self.events=events;self.closed=False
    def dead(self):return self.child.exited
    def send(self,number):
        self.events.append(['FAKE-OWNED-SIGNAL',self.pid,int(number)])
        self.child.exited=True;self.child.status=-int(number)
    def close(self):
        self.closed=True;self.events.append(['FAKE-HANDLE-CLOSED',self.pid])


def fake_work(plan,pid,*,variant=None):
    role=plan['role'];n=40 if plan['method']=='primary' else 100
    baseline={'weights_sha256':cold.digest(b'FAKE-WEIGHTS'),'clean_sha256':cold.digest(b'FAKE-CLEAN'),
        'video_id':'fake-training-row-for-metadata-only','checkpoint_sha256':cold.digest(b'FAKE-CHECKPOINT')}
    h=baseline['clean_sha256'];selected=cold.digest(b'FAKE-SELECTED')
    engine='full-eot8-original-owned-v1' if role=='original-unit' else 'full-eot8-forward-preserving-owned-v1'
    proof={'kind':engine,'native_root_hook_removed':True,'native_scopes':1,'native_calls':1152,'native_refusals':0}
    if role!='original-unit':proof.update(forward_calls=384,initial_predictor_VJPs=192)
    steps=[{'index':i,'pixels':cold.digest(('FAKE-PIXEL-'+str(i)).encode()),
        'gradient':cold.digest(('FAKE-GRAD-'+str(i)).encode()),'losses':[1.],
        'consider_before_backward':True,'route':dict(proof),'context_seconds_including_observations':.001} for i in range(n)]
    ids=[1,0,0,0,0,1]+[i for i in range(2,2+plan['sensitivity_banks']) for _ in range(2)]
    scores=[{'bank':bank,'draws':8 if bank==0 else 4,'pixels':selected if i==5 or (i>=6 and i%2) else h,
        'losses':[1.],'grad_enabled':False,'route_active':False} for i,bank in enumerate(ids)]
    settings={'method':plan['method'],'steps':n//2,'restarts':2,'attack_masks':8,'scoring_masks':4,
        'epsilon_255':4.,'step_size_255':plan['step_size_255'],'profile':'research-primary-eot8-v1'}
    metrics={'method':plan['method'],'video_id':baseline['video_id'],'settings':settings,
        'seeds':records._seeds(baseline['video_id'],plan['sensitivity_banks']),
        'scoring_audits':[{'clean_loss':1.,'adversarial_loss':1.} for _ in range(plan['sensitivity_banks'])],
        'clean_loss':1.,'attack_clean_loss':1.,'adversarial_loss':1.,'attack_loss':1.,'max_pixel_delta':.01,'seconds':.01}
    unit={'kind':'target-whole-pgd-input-gradient-workload-v1','status':'complete','route':engine,'metrics':metrics,
        'steps':steps,'scores':scores,'adversarial':selected,'restored':True,'execution_binding':None,
        'research_admitted':False,'GPU_performance_qualified':False,'overhead_subtracted':False,
        'full_unit_seconds_including_guards_journal_and_observations':.02}
    row=records.wrap_work(unit,role=role,plan=plan,baseline=baseline,pid=pid,wall_seconds=.03,peak_allocator_bytes=1024)
    row['evidence_origin']=session.ORIGIN
    if variant=='replay':row['unit']['steps'][0]['gradient']=cold.digest(b'CHANGED-FAKE-REPLAY')
    return row


class Harness:
    def __init__(self,folder,*,method='primary',fault=None):
        self.folder,self.fault=folder,fault;self.clock=Clock();self.events=[];self.current=None;self.children={};self.rows={}
        self.blueprint=b=session.Blueprint(ROOT,method=method,sensitivity_banks=2 if method=='convergence' else 0)
        self.root=folder/'scratch';self.root.mkdir(mode=0o700)
        layout=b.Layout()
        claimed={'capture/results/workload/'+role for role in records.ROLES}
        for name in sorted(layout.directories,key=lambda n:(len(Path(n).parts),n)):
            if name=='.' or any(name==p or name.startswith(p+'/') for p in claimed):continue
            (self.root/name).mkdir(mode=0o700,exist_ok=True)
        for n,data in b.payload.items():write(self.root/n,data)
        write(self.root/'installation.json',b.installation_bytes)
        write(self.root/'emergency-space.bin',b'R'*(16*2**20))  # Ordinary reserve, not proof.
        binding=session.scratch.identity_binding(self.root)
        self.guard=session.scratch.Guard(self.root,binding,layout)
        self.cache=b.Cache(self.root,binding,b)
        self.session=session.LocalSession(b,self.guard,clock=self.clock,connection_started=100.)

    def launch(self,role,plan,owner):
        self.events.append(['FAKE-LAUNCH',role]);self.current=role
        child=Child(800000+len(self.children));self.children[child.pid]=child;self.current_child=child
        lease=Lease(child,self.events)
        if role=='proof':
            session.publish(self.guard,'allocation-result-v2.json',{'kind':'scratch-allocation-proof-v2',
                'passed':False,'complete':False,'preallocator_injected_for_test':True,'bytes_written':0,
                'bytes_read_back':0,'evidence_origin':session.ORIGIN})
        else:
            p=self.root/'capture/results/workload'/role;p.mkdir(mode=0o700);(p/'unit').mkdir(mode=0o700)
            row=fake_work(plan,child.pid,variant='replay' if self.fault=='replay' and role=='candidate-replay' else None)
            if self.fault=='owner' and role=='candidate-unit':row['unit']['steps'][0]['route']['native_root_hook_removed']=False
            if self.fault=='allocator' and role=='candidate-unit':row['peak_allocator_bytes']=7*2**30+1
            if self.fault=='source' and role=='candidate-unit':row['source_manifest_sha256']='0'*64
            if self.fault=='admission' and role=='candidate-unit':row['training_admitted']=True
            self.rows[role]=row
            session.publish(self.guard,'capture/results/workload/'+role+'/unit-work.json',row)
            session.publish(self.guard,'capture/results/workload/'+role+'/unit/unit-started.json',{'kind':session.ORIGIN,'actual_objectives':0})
            for step in row['unit']['steps']:
                prefix='capture/results/workload/'+role+f"/unit/step-{step['index']:03}"
                session.publish(self.guard,prefix+'-started.json',{'index':step['index'],'evidence_origin':session.ORIGIN})
                session.publish(self.guard,prefix+'-complete.json',step)
            session.publish(self.guard,'capture/results/workload/'+role+'/unit/unit-complete.json',row['unit'])
        return child,lease

    def sample(self,pid,birth):
        rss=4*2**30+1 if self.fault=='rss' and self.current=='candidate-unit' else 1024*1024
        return {'processes':{pid:{'pid':pid,'start_ticks':birth,'parent_pid':os.getpid(),'session_pid':pid,'rss_bytes':rss}},'rss_bytes':rss}

    def pause(self,seconds):
        self.clock.now+=seconds
        if self.fault=='clock' and self.current=='candidate-unit':self.clock.now+=800
        else:self.current_child.exited=True

    def verify(self,role,plan,observed,owner):
        row={'role':role,'pid':observed['pid'],'start_ticks':observed['start_ticks'],
            'saved_gates_returned':True,'evidence_origin':session.ORIGIN,'scientific_qualified':False,'training_admitted':False}
        if role=='proof':row.update(proof_is_simulated=True,actual_bytes_written=0)
        else:row['work']=self.rows[role]
        if self.fault=='identity' and role=='candidate-unit':row['start_ticks']+=1
        return row

    def timer(self,deadline,**kw):
        h=self
        class Scope:
            def __enter__(self):h.events.append(['FAKE-HARD-TIMER',deadline,kw]);return self
            def __exit__(self,*error):
                h.events.append(['FAKE-TIMER-RESTORED',deadline])
                if h.clock()>=deadline:raise TimeoutError('Explicit fake hard clock expired.')
                return False
        return Scope()

    def ports(self):
        def no_descendants(*a):raise AssertionError('No fake descendant is present.')
        return {'origin':session.ORIGIN,'launch':self.launch,'sample':self.sample,'lease':no_descendants,
            'verify':self.verify,'pause':self.pause,'hard_timer':self.timer,
            'gpu':lambda:{'evidence_origin':session.ORIGIN,'empty_for_simulated_role':True}}


def test_cold_derivation_source_inverses_and_no_old_dispatch(case):
    files,m=cold.assemble(ROOT)
    assert len(files)==47 and sum(map(len,files.values()))<2*2**20
    for name,change in m['changes'].items():
        original=cold.inverse(files[name],change['edits'])
        assert cold.digest(original)==change['original_sha256']
        assert original==(ROOT/change['original_path']).read_bytes()
    for n in ('models/predictor_eot8_owned_v1.py','models/predictor_eot8_adapter_v1.py',
              'models/predictor_forward_preserving_cpu_v1.py','attacks/pgd.py'):
        assert files[n]==(ROOT/n).read_bytes()
    assert b'OBJECTIVE_TIMER = None' in files[cold.UNIT]
    assert b'+120)' in files[cold.UNIT] and b'384' in files[cold.UNIT] and b'4096' in files[cold.UNIT]
    write(case/'manifest.json',cold.encoded(m))


def test_unbound_entry_refuses_before_scientific_import(case):
    files,_=cold.assemble(ROOT);ns=session.definitions(files[cold.ENTRY],{})
    with pytest.raises(ValueError,match='unbound'):ns['entry'](NS())
    with pytest.raises(ValueError,match='consumed'):ns['entry'](NS())


@pytest.mark.parametrize('method',['primary','convergence'])
def test_complete_fake_owned_lifecycle_and_priority_collection(case,method):
    h=Harness(case,method=method);b=h.blueprint
    before=(dict(sys.modules),list(sys.meta_path));entry=h.cache.load_metadata('models._pgd_workload_entry_v1')
    with pytest.raises(ValueError,match='unbound'):entry.entry(NS())
    assert before==(dict(sys.modules),list(sys.meta_path))
    state=h.session.run(h.ports());assert state['completed']==list(session.ROLES)
    assert state['complete'] is False and state['completion_is_simulated'] is True
    assert len([x for x in h.events if x[0]=='FAKE-HANDLE-CLOSED'])==4
    assert not [x for x in h.events if x[0]=='FAKE-OWNED-SIGNAL']
    with pytest.raises(ValueError):h.guard.verify_proof(cold.digest(h.guard.read('allocation-result-v2.json',16384)))
    wire=io.BytesIO();terminal=h.session.collect(wire);data=wire.getvalue();write(case/'wire.bin',data)
    received=b.codec.receive(io.BytesIO(data),case/'received',b.control_bytes,b.local_binding_bytes,
        window=b.codec.Window(100.,h.clock(),clock=h.clock))
    assert received['transport_verified'] is True and received['scientific_qualified'] is received['training_admitted'] is False
    write(case/'transport.json',cold.encoded(received));write(case/'events.json',cold.encoded(h.events))
    assert terminal['stream_complete'] is True
    assert json.loads((h.root/'session.json').read_text())['complete'] is False
    with pytest.raises(ValueError):h.session.run(h.ports())
    with pytest.raises(ValueError):h.session.collect(io.BytesIO())


@pytest.mark.parametrize('fault',['rss','clock','owner','allocator','source','admission','identity','replay'])
def test_fake_failure_stops_later_roles_and_retains_failed_session(case,fault):
    h=Harness(case,fault=fault)
    with pytest.raises((ValueError,MemoryError,TimeoutError)):h.session.run(h.ports())
    assert h.session.state['phase']=='failed' and h.session.state['complete'] is False
    if fault!='replay':assert 'candidate-replay' not in [e[1] for e in h.events if e[0]=='FAKE-LAUNCH']
    assert not (h.root/'capture/results/replay.json').exists()
    assert len([e for e in h.events if e[0]=='FAKE-HANDLE-CLOSED'])>=3
    write(case/'events.json',cold.encoded(h.events))


@pytest.mark.parametrize('mutation',['route','count','scoring','bound','seed','gradient_time','unit_time','nan'])
def test_metadata_reader_refuses_modified_fake_units(case,mutation):
    plan={'role':'candidate-unit','method':'primary','step_size_255':1.,'sensitivity_banks':0,
        'control_sha256':'1'*64,'source_manifest_sha256':'2'*64,'plan_sha256':'3'*64}
    row=fake_work(plan,800001)
    if mutation=='route':row['unit']['steps'][0]['route']['initial_predictor_VJPs']=191
    elif mutation=='count':row['unit']['steps'].pop()
    elif mutation=='scoring':row['unit']['scores'][0]['route_active']=True
    elif mutation=='bound':row['unit']['metrics']['max_pixel_delta']=.1
    elif mutation=='seed':row['unit']['metrics']['seeds']['start']+=1
    elif mutation=='gradient_time':row['unit']['steps'][0]['context_seconds_including_observations']=121
    elif mutation=='unit_time':row['wall_seconds']=721
    elif mutation=='nan':row['unit']['metrics']['attack_loss']=float('nan')
    with pytest.raises(ValueError):records.validate_work(row,plan)


def test_cache_source_and_loader_mutation_refuse(case):
    h=Harness(case);entry=h.cache.load_metadata('models._pgd_workload_entry_v1')
    with pytest.raises(ValueError):h.cache.load_metadata('models.predictor_pgd_workload_target_v1')
    with pytest.raises(ValueError):h.cache.install_live()
    with pytest.raises(ValueError):h.blueprint.live_run()
    entry.__loader__=None
    with pytest.raises(ValueError):h.cache.check_module(entry.__name__)


def test_scratch_catalogue_keeps_all_old_shared_caps_and_rejects_unknown_paths(case):
    h=Harness(case);layout=h.guard.layout
    assert layout.contract['category_caps']==session.scratch.CAPS
    assert len(layout.result)==265 and len(h.blueprint.runtime)==46
    with pytest.raises(ValueError):layout.classify('capture/results/components/native/complete.json')
    with pytest.raises(ValueError):layout.classify('capture/results/workload/candidate-unit/unit/step-040-complete.json')
    with pytest.raises(ValueError):layout.classify('../outside.json')
    with pytest.raises(ValueError):session.publish(h.guard,'session.json',{},replace=True)


def test_partial_priority_collection_cannot_promote(case):
    h=Harness(case);h.session.run(h.ports());wire=io.BytesIO();h.session.collect(wire);data=wire.getvalue()
    write(case/'truncated-wire.bin',data[:len(data)//2])
    with pytest.raises((ValueError,EOFError)):
        h.blueprint.codec.receive(io.BytesIO(data[:len(data)//2]),case/'partial',
            h.blueprint.control_bytes,h.blueprint.local_binding_bytes,
            window=h.blueprint.codec.Window(100.,h.clock(),clock=h.clock))
    assert json.loads((case/'partial-transport/PARTIAL-RECEIPT.json').read_text())['transport_verified'] is False


def test_backward_or_expired_continuing_clock_refuses(case):
    h=Harness(case);h.clock.now=99.
    with pytest.raises(ValueError):h.session.check()
    h.clock.now=2500.
    with pytest.raises(TimeoutError):h.session.check()


@pytest.mark.parametrize('fail',[False,True])
def test_emitted_owned_step_fake_timer_and_failure_cleanup(case,fail):
    from contextlib import AbstractContextManager
    files,_=cold.assemble(ROOT);tree=ast.parse(files[cold.UNIT])
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OwnedStep')
    raw=ast.get_source_segment(files[cold.UNIT].decode(),node).encode()
    clock=Clock();events=[]
    class Value:
        def __init__(self,label,shape=()):self.label=label;self.shape=shape;self._version=0;self.requires_grad=True
        def view(self,*shape):return Value(self.label+'-view',shape)
        def detach(self):return self
        def tolist(self):return [1.]
    value=Value('fake-input')
    class Route:
        def __enter__(self):events.append('FAKE-ROUTE-ENTER');factory.active=True;return self
        def loss(self,v,t,m):self.tokens=(Value('fake-root'),v);return self.tokens[0]
        def gradient(self,r,v):
            assert (r,v)==self.tokens;events.append('FAKE-OWNED-GRADIENT')
            if fail:raise RuntimeError('Explicit fake gradient interruption')
            return Value('fake-gradient')
        def __exit__(self,*error):factory.active=False;events.append('FAKE-ROUTE-EXIT')
    factory=NS(active=False,create=lambda m:Route(),observe=lambda r:{'evidence_origin':session.ORIGIN})
    owner=NS(active=None,poisoned=False,factory=factory,model=object(),targets=object(),attack_bank=[],steps=[])
    def idle():assert owner.active is None and not factory.active
    def timer(deadline,**kw):
        class Scope:
            def __enter__(self):events.append('FAKE-TIMER-ENTER');assert deadline==kw['hard_deadline']==200.
            def __exit__(self,*error):events.append('FAKE-TIMER-EXIT')
        return Scope()
    owner.idle=idle;owner.time_gate=lambda:None;owner.snapshot=NS(check=lambda **kw:None)
    owner.journal=NS(put=lambda name,row:write(case/(name+'.json'),cold.encoded(row)))
    step_type=session.definitions(raw,{'AbstractContextManager':AbstractContextManager,
        'torch':NS(isfinite=lambda t:True),'time':NS(monotonic=clock,perf_counter=clock),
        'tensor_hash':lambda t:cold.digest(t.label.encode()),'checked_gradient':lambda g,v:g,
        'EXECUTION_BINDING':{'deadline':200.},'OBJECTIVE_TIMER':timer})['OwnedStep']
    step=step_type(owner,value,0)
    def run():
        with step:
            step.note_consider(value,step.losses);step.gradient()
    if fail:
        with pytest.raises(RuntimeError):run()
    else:run()
    assert events==['FAKE-TIMER-ENTER','FAKE-ROUTE-ENTER','FAKE-OWNED-GRADIENT','FAKE-ROUTE-EXIT','FAKE-TIMER-EXIT']
    assert owner.active is None and owner.poisoned is fail and len(owner.steps)==(0 if fail else 1)
    assert step.root is step.losses is step.value is None

