"""Component operator integration; private opaque bytes and explicit fake OS ports.

No child, proc query, signal, timer, network, GPU, proof or scientific work runs.
"""
import ast
import base64
import builtins
import copy
import hashlib
import importlib.util
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
from types import ModuleType,SimpleNamespace as NS
import pytest
from models import predictor_pgd_component_operator_v1 as build
from models import predictor_pgd_component_client_v1 as client
from models import predictor_pgd_component_outer_v1 as outer
ROOT=Path(__file__).resolve().parents[1]
ORIGIN='EXPLICIT-FAKE-COMPONENT-OPERATOR-LOCAL-CHECK; NOT A USER GRANT'

@pytest.fixture(scope='module',autouse=True)
def forbid_live_APIs():
    patch=pytest.MonkeyPatch()
    def forbidden(*a,**k):pytest.fail('No real process/query/signal/timer/network/proof allowed.')
    for module,name in [(subprocess,'Popen'),(subprocess,'check_output'),(os,'system'),(os,'fork'),(os,'posix_spawn'),(os,'kill'),(os,'killpg'),(os,'pidfd_open'),(os,'posix_fallocate'),(os,'pipe2'),(signal,'signal'),(signal,'setitimer'),(signal,'pidfd_send_signal'),(socket,'socket')]:
        if hasattr(module,name):patch.setattr(module,name,forbidden)
    yield
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    patch.undo()

@pytest.fixture(scope='module')
def bundle():return build.assemble(ROOT)

def module(raw,name,imports=None):
    m=ModuleType(name);m.__file__='<EXPLICIT-LOCAL-OPERATOR-'+name+'>'
    if imports:
        def load(name,globals=None,locals=None,fromlist=(),level=0):
            if name=='models':return imports
            return builtins.__import__(name,globals,locals,fromlist,level)
        m.__dict__['__builtins__']={**vars(builtins),'__import__':load}
    exec(compile(raw,m.__file__,'exec'),m.__dict__);return m

@pytest.fixture(scope='module')
def support(bundle):
    boot=module(bundle['sources']['bootstrap.py'],'fake_component_boot')
    scratch=module(bundle['sources']['predictor_pgd_split_scratch_v1.py'],'fake_component_scratch')
    scratch.os=NS(**vars(os));scratch.os.fstatvfs=lambda fd:NS(f_bavail=32*2**30//4096,f_frsize=4096,f_favail=10000)
    watch=module(bundle['sources']['predictor_pgd_split_watch_v1.py'],'fake_component_watch')
    watch.__file__=str(ROOT/build.local.load_inputs(ROOT)[1]['files']['monitor']['path'])
    bindings=NS(predictor_pgd_split_bootstrap_v1=boot,predictor_pgd_split_scratch_v1=scratch,predictor_pgd_split_watch_v1=watch)
    coord=module(bundle['sources']['predictor_pgd_split_coordinator_v1.py'],'fake_component_coordinator',bindings)
    return NS(boot=boot,scratch=scratch,watch=watch,coord=coord)

def put(p,data):
    missing=[];parent=p.parent
    while not parent.exists():missing.append(parent);parent=parent.parent
    for q in reversed(missing):q.mkdir(mode=0o700)
    data=data if type(data) is bytes else build.enc(data)
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(data)

def fake_authorization(boot,raw):
    return boot.encoded({'kind':'pgd-component-supervised-university-exception-v1','approved':True,'control_sha256':boot.digest(raw),'maximum_new_attempts':1,'user_basis':ORIGIN})

@pytest.fixture(scope='module')
def baseline(bundle,support,tmp_path_factory):
    root=tmp_path_factory.mktemp('private-operator-baseline');root.chmod(0o700)
    bp=build.local.Blueprint(ROOT);payload=dict(bp.payload);payload.pop('installation.json')
    payload.update(bundle['uploads']);control=bundle['control'];b,p,csha=support.boot.control(control);installation,isha=support.boot.installation(b,csha)
    payload['installation.json']=installation
    for name,data in payload.items():put(root/name,data)
    for name,data in [('control.json',control),('authorization.json',fake_authorization(support.boot,control)),('proposal.json',{'origin':ORIGIN,'actual_approval':False}),('runtime-source-manifest.json',{'files':list(bundle['uploads']),'origin':ORIGIN})]:put(root/name,data)
    layout=support.scratch.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),installation,isha)
    for name in outer._setup_directory_plan(layout)['setup_directories']:
        if name!='.':(root/name).mkdir(mode=0o700,exist_ok=True)
    # Only an ordinary16MiB reserve fixture; no2.5GiB preallocation/proof.
    put(root/'emergency-space.bin',b'\xa7'*(16*2**20))
    return root

def guard_for(support,root):
    b,p,csha=support.boot.control((root/'control.json').read_bytes());installation,isha=support.boot.installation(b,csha)
    binding=support.scratch.identity_binding(root)
    layout=support.scratch.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),installation,isha)
    return support.scratch.Guard(root,binding,layout),binding,b,p,csha


def test_unarmed_client_refuses_before_timer_process_or_claim(tmp_path):
    with pytest.raises(ValueError,match='Unarmed'):client.run(tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_complete_source_and_scientific_isolation(bundle):
    assert len(bundle['uploads'])==43 and len(bundle['catalogue']['assets'])==82
    assert set(bundle['sources'])==set(build.NAMES)
    for name,d in bundle['static']['runtime_files'].items():
        data=bundle['uploads']['overlay/'+name];assert hashlib.sha256(data).hexdigest()==d['sha256']
    assert not any('_pgd_split_'+n+'_v1.py' in '\n'.join(bundle['uploads']) for n in ['entry','numerical','records','observer','progress','facade','providers','io'])
    t=ast.parse(bundle['sources']['predictor_pgd_split_coordinator_v1.py'])
    dispatch=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='dispatch_child')
    assert 'numerical_entry' not in ast.unparse(dispatch) and 'component_entry' in ast.unparse(dispatch)
    assert bundle['derivation']['execution_authorized'] is False


def test_actual_catalogue_and_proof_log_isolation(bundle,support,baseline,tmp_path):
    root=tmp_path/'scratch';shutil.copytree(baseline,root);guard,binding,c,p,csha=guard_for(support,root)
    guard.verify_assets(phase='setup');guard.proof_readiness();put(root/'proof.log',b'EXPLICIT-FAKE LOG; no child\n')
    guard.scan(phase='capture',settled=True)
    plan=NS(predictor_pgd_split_plan_v1=support.scratch.plan)
    store=module(bundle['uploads']['overlay/models/_pgd_component_store_v1.py'],'fake_component_store',plan)
    inv=store.Inventory(root/'capture',base_bytes=p['base_text'].encode(),design_bytes=p['design_text'].encode());inv.scan()
    put(root/'capture/logs/proof.log',b'EXPLICIT-FAKE MISPLACED LOG\n')
    with pytest.raises(ValueError,match='Unknown'):inv.scan()


def test_exact_supervised_grant_and_policy_reject_old_cohort(bundle,support):
    c,p,csha=support.boot.control(bundle['control']);a=fake_authorization(support.boot,bundle['control']);support.boot.authorization(a,csha)
    changed=json.loads(a);changed['kind']='pgd-split-supervised-university-exception-v1'
    with pytest.raises(ValueError):support.boot.authorization(build.enc(changed),csha)
    wrong=copy.deepcopy(c);policy=json.loads(wrong['policy_text']);policy['scope']['new_full_model_objectives']=8;wrong['policy_text']=build.enc(policy).decode();wrong['policy_sha256']=build.digest(wrong['policy_text'].encode())
    with pytest.raises(ValueError):support.boot.control(build.enc(wrong))
    a=outer.authorization({'supervised_control_sha256':csha},'f'*64,ORIGIN)
    assert a['full_model_gradients']==0 and a['extra_component_VJPs']==2 and a['automatic_retries']==0


def cache_root(baseline,tmp_path,support):
    root=tmp_path/'scratch';shutil.copytree(baseline,root);g,binding,c,p,csha=guard_for(support,root)
    finder=support.boot.PinnedImporter(root,binding,c,p)
    return root,finder,csha


def test_fresh_pinned_cache_and_parent_search_path(support,baseline,tmp_path,monkeypatch):
    root,finder,csha=cache_root(baseline,tmp_path,support)
    fake=NS(modules={},meta_path=[]);monkeypatch.setattr(support.boot,'sys',fake);finder.install()
    for name in ['models','models.predictor_pgd_split_lineage_v1']:
        path=None if name=='models' else fake.modules['models'].__path__;spec=finder.find_spec(name,path);m=importlib.util.module_from_spec(spec);fake.modules[name]=m;finder.exec_module(m)
        if '.' in name:setattr(fake.modules['models'],name.split('.')[1],m)
    assert support.boot.require_namespace(root,finder.binding,csha) is finder
    with pytest.raises(ImportError):finder.find_spec('models.predictor_sdpa_launch_evidence_v3',['/ambient/models'])
    with pytest.raises(ModuleNotFoundError):finder.find_spec('models._pgd_split_numerical_v1',fake.modules['models'].__path__)
    with pytest.raises(ValueError):finder.install()

@pytest.mark.parametrize('fault',['source','origin','parent'])
def test_cache_identity_drift_refuses(fault,support,baseline,tmp_path,monkeypatch):
    root,finder,csha=cache_root(baseline,tmp_path,support);fake=NS(modules={},meta_path=[]);monkeypatch.setattr(support.boot,'sys',fake);finder.install()
    spec=finder.find_spec('models',None);m=importlib.util.module_from_spec(spec);fake.modules['models']=m;finder.exec_module(m)
    if fault=='source':(root/'overlay/models/__init__.py').write_bytes(b'changed\n')
    elif fault=='origin':m.__spec__.origin='/foreign/models/__init__.py'
    else:m.__path__=['/foreign/models']
    with pytest.raises((ValueError,ImportError)):support.boot.require_namespace(root,finder.binding,csha)


class Clock:
    def __init__(self):self.value=100.
    def __call__(self):self.value+=.0001;return self.value
    def pause(self,n):self.value+=n

class Child:
    def __init__(self,pid):self.pid=pid;self.exited=False;self.status=0
    def poll(self):return self.status if self.exited else None
    def wait(self,timeout=None):return self.status

class Lease:
    def __init__(self,child,birth,events):self.child=child;self.pid=child.pid;self.birth=birth;self.events=events;self.closed=False
    def dead(self):return self.child.exited
    def send(self,n):self.events.append(('OWNED-FAKE-SIGNAL',int(n)));self.child.exited=True;self.child.status=-int(n)
    def close(self):self.closed=True;self.events.append(('lease-closed',self.pid))


def launcher(support,bundle,root,role,monkeypatch,fault=None):
    coord=support.coord;clock=Clock();events=[];holder={}
    class Reader:
        def identity(self,pid):
            events.append(('identity',pid));return {'pid':pid,'parent_pid':os.getpid(),'session_pid':pid,'group_pid':pid,'start_ticks':717171,'rss_bytes':128*2**20,'children':[]}
        def sample(self,pid,birth):
            row=self.identity(pid);rss=5*2**30 if fault=='rss' else 128*2**20;row['rss_bytes']=rss
            if fault=='deadline':clock.value=210.
            if fault not in ('rss','deadline'):holder['child'].exited=True
            return {'processes':{pid:row},'rss_bytes':rss}
    def lease(pid,birth,parent,session,reader):
        events.append(('lease-bound',pid))
        if fault=='lease':raise ValueError('EXPLICIT-FAKE handle binding refusal.')
        value=Lease(holder['child'],birth,events);holder['lease']=value;return value
    def popen(command,**kwargs):
        assert kwargs['shell'] is False and kwargs['start_new_session'] is True and kwargs['pass_fds']==(9001,) and kwargs['umask']==0o077
        assert command[1:5]==['-I','-B','-u','-c'];assert command[5]==coord.CHILD_DRIVER
        active_role=json.loads(base64.b64decode(command[-2]))['role'];assert role is None or active_role==role
        events.append(('Popen-FAKE',active_role));child=Child(818180+sum(e[0]=='Popen-FAKE' for e in events));holder['child']=child;os.write(kwargs['stdout'],b'EXPLICIT-FAKE: no child was started.\n');return child
    def close(fd):
        if fd in (9001,9002):
            events.append(('pipe-closed',fd))
            if fd==9002 and not any(e[0]=='release' for e in events):holder.get('child',NS()).__dict__.update(exited=True)
        else:os.close(fd)
    def write(fd,data):
        assert fd==9002 and data==b'G';events.append(('release',True));return 1
    proxy=NS(**vars(os));proxy.pipe2=lambda flags:(9001,9002);proxy.close=close;proxy.write=write
    monkeypatch.setattr(coord,'os',proxy);monkeypatch.setattr(coord,'subprocess',NS(Popen=popen,DEVNULL=-3,TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(support.watch,'ProcReader',Reader);monkeypatch.setattr(support.watch,'PidfdLease',lease)
    monkeypatch.setattr(support.watch,'time',NS(monotonic=clock,sleep=clock.pause))
    tree=ast.parse(bundle['sources']['predictor_pgd_split_watch_v1.py']);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='watch');exec(compile(ast.Module(body=[fn],type_ignores=[]),'<EXPLICIT-FAKE-CLOCK-UNCHANGED-WATCH>','exec'),support.watch.__dict__)
    owner=NS(root=root,clock=clock,work_deadline=2500.,c=json.loads(bundle['control']),outer_guard=None,outer_alarm=NS(arm=lambda deadline,**kw:events.append(('alarm',deadline,kw))))
    driver=coord.OuterDriver(owner)
    def gate(role,pid=None):
        events.append(('live-guard-FAKE',pid))
        if fault=='foreign' and pid is not None:raise ValueError('EXPLICIT-FAKE foreign CUDA process.')
    driver.outer_journal=NS(checkpoint=gate)
    return driver,clock,events,holder

@pytest.mark.parametrize('role',['proof','native','forward-preserving'])
def test_actual_launcher_release_after_owned_handle_only(role,bundle,support,baseline,tmp_path,monkeypatch):
    root=tmp_path/'scratch';shutil.copytree(baseline,root);driver,clock,events,holder=launcher(support,bundle,root,role,monkeypatch)
    result=driver.launch(role,{'role':role,'launch_started':100.,'deadline':150. if role=='proof' else 210.})
    assert result['parent_observation_passed'] and result['actual_CUDA_execution_admitted'] is False
    names=[e[0] for e in events];assert names.index('lease-bound')<names.index('live-guard-FAKE')<names.index('release')
    assert holder['lease'].closed and not any(n=='OWNED-FAKE-SIGNAL' for n in names)
    driver.assert_idle();assert (root/'proof.log' if role=='proof' else root/'capture/logs'/(role+'.log')).is_file()

@pytest.mark.parametrize('fault',['lease','foreign','rss','deadline'])
def test_actual_launcher_failure_closes_only_owned_handles(fault,bundle,support,baseline,tmp_path,monkeypatch):
    root=tmp_path/'scratch';shutil.copytree(baseline,root);driver,clock,events,holder=launcher(support,bundle,root,'native',monkeypatch,fault)
    with pytest.raises((ValueError,MemoryError,TimeoutError)):driver.launch('native',{'role':'native','launch_started':100.,'deadline':210.})
    names=[e[0] for e in events]
    if fault in ('lease','foreign'):assert 'release' not in names
    if fault=='lease':assert 'OWNED-FAKE-SIGNAL' not in names
    else:assert holder['lease'].closed
    assert driver.current is None;driver.assert_idle()
    put(tmp_path/'FAILURE-OBSERVATION.json',{'origin':ORIGIN,'fault':fault,'events':events,'actual_process_or_signals':False,'scientific_qualified':False})


def make_plan(support,bundle,root,role):
    guard,binding,c,p,csha=guard_for(support,root);index=support.boot.ROLES.index(role)
    sources={} if role=='proof' else bundle['source_hashes']
    current={'role':role,'nonce':'1'*64,'launch_started':100.,'deadline':150. if role=='proof' else 210.,'source_hashes':sources}
    state={'kind':'pgd-component-parent-session-v1','control_sha256':csha,'parent_pid':os.getpid(),'connection_started':100.,'work_deadline':2500.,'connection_deadline':3100.,'current':current,'completed':list(support.boot.ROLES[:index]),'phase':'running','complete':False}
    put(root/'session.json',state)
    return dict(current,kind='pgd-component-child-launch-v1',control_sha256=csha,authorization_sha256=build.digest((root/'authorization.json').read_bytes()),parent_pid=os.getpid(),root_binding=binding,work_deadline=2500.,connection_deadline=3100.,session_sha256=build.digest(build.enc(state))),binding

@pytest.mark.parametrize('role',['native','forward-preserving'])
def test_dispatch_binds_only_component_entry(role,bundle,support,baseline,tmp_path,monkeypatch):
    root=tmp_path/'scratch';shutil.copytree(baseline,root);plan,binding=make_plan(support,bundle,root,role);events=[]
    runtime=NS(EXECUTION_BINDING=None,component_entry=lambda args:events.append(('component',vars(args))))
    original=support.coord.__dict__['__builtins__']['__import__']
    def load(name,globals=None,locals=None,fromlist=(),level=0):
        if name=='resource':return NS(RLIMIT_FSIZE=1,setrlimit=lambda resource,limits:events.append(('rlimit',limits)))
        if name=='models' and '_pgd_component_entry_v1' in fromlist:return NS(_pgd_component_entry_v1=runtime)
        return original(name,globals,locals,fromlist,level)
    monkeypatch.setitem(support.coord.__dict__,'__builtins__',{**vars(builtins),'__import__':load})
    # Functions keep their existing builtins mapping; patch that exact private mapping.
    monkeypatch.setitem(support.coord.dispatch_child.__builtins__,'__import__',load)
    monkeypatch.setattr(support.coord,'target_gate',lambda *a:events.append(('target-FAKE',True)))
    monkeypatch.setattr(support.coord,'gpu_gate',lambda *a:events.append(('GPU-FAKE',True)))
    monkeypatch.setattr(support.coord,'time',NS(monotonic=lambda:101.))
    assert support.coord.dispatch_child(root,binding,bundle['control'],(root/'authorization.json').read_bytes(),plan)==0
    assert runtime.EXECUTION_BINDING['host']=='linux10614' and runtime.EXECUTION_BINDING['capsule_root']==str(root/'source')
    assert events[-1][0]=='component' and events[-1][1]['role']==role and events[-1][1]['deadline']==plan['deadline']
    assert ('rlimit',(16*2**20,16*2**20)) in events
    with pytest.raises(ValueError,match='fresh null'):support.coord.dispatch_child(root,binding,bundle['control'],(root/'authorization.json').read_bytes(),plan)


def test_no_numerical_role_and_fake_proof_is_refused(bundle,support,baseline,tmp_path):
    root=tmp_path/'scratch';shutil.copytree(baseline,root);plan,binding=make_plan(support,bundle,root,'proof')
    bad=dict(plan,role='numerical')
    with pytest.raises(ValueError):support.boot.validate_plan(bad,build.digest(bundle['control']),json.loads((root/'session.json').read_bytes()),101.)
    guard,*_=guard_for(support,root);put(root/'allocation-result-v2.json',{'kind':'EXPLICIT-FAKE-PROOF','passed':False,'actual_bytes_written':0,'actual_bytes_read_back':0})
    driver=support.coord.SystemDriver(NS(root=root));driver.guard=guard
    with pytest.raises(ValueError,match='proof'):driver.verify('proof',plan,{'pid':818181,'exit_status':0,'parent_observation_passed':True})
    assert driver.sources=={}

@pytest.mark.parametrize('fault',['ok','EOF','owner','deadline'])
def test_child_release_hard_timer_precedes_any_dispatch(fault,support,monkeypatch):
    events=[];clock=101. if fault!='deadline' else 151.
    osproxy=NS(**vars(os));osproxy.getppid=lambda:999 if fault=='owner' else 111;osproxy.fstat=lambda fd:NS(st_mode=stat.S_IFIFO|0o600);osproxy.read=lambda fd,count:b'' if fault=='EOF' else b'G';osproxy.close=lambda fd:events.append(('close',fd))
    sig=NS(**vars(signal));sig.signal=lambda *a:events.append(('signal-FAKE',a));sig.setitimer=lambda *a:events.append(('timer-FAKE',a))
    monkeypatch.setattr(support.boot,'os',osproxy);monkeypatch.setattr(support.boot,'time',NS(monotonic=lambda:clock));monkeypatch.setattr(support.boot,'signal',sig);monkeypatch.setattr(support.boot,'select',NS(select=lambda *a:([13],[],[])))
    if fault=='ok':support.boot.wait_release(13,111,150.);assert events[0][0]=='signal-FAKE' and events[1][0]=='timer-FAKE' and events[-1]==('close',13)
    else:
        with pytest.raises((ValueError,TimeoutError)):support.boot.wait_release(13,111,150.)
    assert not any(e[0]=='dispatch' for e in events)


def test_alarm_uses_default_hard_stop_and_continuing_deadline(monkeypatch):
    events=[];clock=Clock();main=object()
    class Timer:
        def __init__(self,delay,callback):self.delay=delay;self.callback=callback;self.daemon=False
        def start(self):events.append(('timer-thread-FAKE',self.delay))
        def cancel(self):events.append(('cancel',True))
    sig=NS(**vars(signal));sig.getsignal=lambda n:signal.SIG_DFL;sig.getitimer=lambda kind:(0.,0.);sig.signal=lambda *a:events.append(('signal-FAKE',a));sig.setitimer=lambda *a:events.append(('kernel-timer-FAKE',a))
    monkeypatch.setattr(outer,'signal',sig);monkeypatch.setattr(outer,'threading',NS(current_thread=lambda:main,main_thread=lambda:main,Timer=Timer))
    alarms=outer.Alarms(clock=clock);alarms.arm(210.,hard_deadline=2500.,phase='native')
    assert any(e[0]=='signal-FAKE' and e[1]==(signal.SIGALRM,signal.SIG_DFL) for e in events)
    assert alarms.hard==2500. and alarms.soft_deadline==210.
    clock.value=210.
    with pytest.raises(outer.Deadline):alarms.check()
    assert alarms.hard==2500. and any(e[0]=='kernel-timer-FAKE' and 0<e[1][1]<=2 for e in events)
    alarms.close();assert alarms.closed


def test_unarmed_package_is_bound_without_approval(bundle,tmp_path):
    tmp_path.chmod(0o700);folder=tmp_path/'unarmed';package=build.prepare_unarmed(ROOT,folder)
    assert not (folder/'AUTHORIZATION.json').exists() and package['authorization_created'] is False
    assert package['source_files']==43 and package['completed_full_model_gradients_repeated']==0
    with pytest.raises(ValueError,match='Unarmed'):client.check_local(folder)
    c=json.loads((folder/'CONTROL-CANDIDATE.json').read_bytes());assert c['runtime_files']==bundle['static']['runtime_files'] and c['packet_path'].endswith('component-inputs.tar')
    assert len(c['local_source_sha256'])==12


def test_proof_failure_diagnostic_uses_supervisor_log(bundle,support,baseline,tmp_path,monkeypatch):
    root=tmp_path/'scratch';shutil.copytree(baseline,root);driver,clock,events,holder=launcher(support,bundle,root,'proof',monkeypatch,'lease')
    with pytest.raises(ValueError) as caught:driver.launch('proof',{'role':'proof','launch_started':100.,'deadline':150.})
    assert 'retained at proof.log' in '\n'.join(caught.value.__notes__)
    assert 'capture/logs/proof.log' not in '\n'.join(caught.value.__notes__)


def test_three_role_parent_to_real_client_priority_codec(bundle,support,baseline,tmp_path,monkeypatch):
    """Whole new source/receipt chain, with fake children/proof/verification only."""
    from types import MethodType
    root=tmp_path/'scratch';shutil.copytree(baseline,root);guard,binding,c,p,csha=guard_for(support,root)
    driver,clock,events,holder=launcher(support,bundle,root,None,monkeypatch)
    alarms=NS(arm=lambda deadline,**kw:events.append(('alarm',deadline,kw)))
    owner=support.coord.OuterCoordinator(root,binding,bundle['control'],(root/'authorization.json').read_bytes(),connection_started=100.,clock=clock,outer_guard=guard,outer_alarm=alarms)
    driver.owner=owner;owner.driver=driver
    def prepare():
        guard.verify_assets(phase='setup');guard.proof_readiness();driver.guard=guard
        return {'origin':ORIGIN,'actual_namespace_process_GPU_or_proof_authenticated':False}
    def verify(self,role,plan,observed):
        assert observed['parent_observation_passed'] and self.outer_lease.closed
        if role=='proof':
            put(root/'allocation-result-v2.json',{'kind':'EXPLICIT-FAKE-PROOF','passed':False,'complete':False,'actual_bytes_written':0,'actual_bytes_read_back':0,'origin':ORIGIN})
            self.sources=dict(bundle['source_hashes'])
        support.coord._metadata_write(guard,'parent-'+role+'.json',observed,replace=False)
        return {'role':role,'pid':observed['pid'],'start_ticks':observed['start_ticks'],'saved_gates_returned':True,'receipt':{'origin':ORIGIN,'actual_verification_and_proof_authenticated':False},'source_hashes':dict(self.sources),'scientific_qualified':False,'training_admitted':False}
    monkeypatch.setattr(driver,'prepare',prepare);monkeypatch.setattr(driver,'verify',MethodType(verify,driver))
    state=owner.run_work();driver.assert_idle()
    assert state['completed']==['proof','native','forward-preserving'] and state['complete'] is False
    assert len({r['pid'] for r in state['receipts']})==3
    with pytest.raises(ValueError,match='no retries'):owner.run_work()
    imports=NS(predictor_pgd_split_bootstrap_v1=support.boot,predictor_pgd_split_scratch_v1=support.scratch)
    codec=module(bundle['sources']['predictor_pgd_split_collection_v1.py'],'fake_component_codec',imports)
    collector=codec.Collector(guard,bundle['control'],(root/'authorization.json').read_bytes(),connection_started=100.,clock=clock)
    header=collector.prepare();out=io.BytesIO();terminal=collector.stream(out);wire=out.getvalue()
    claimed={'scratch':str(root),'connection_started':100.};ready={'collection_started':collector.window.started,'collection_deadline':collector.window.deadline}
    # Keep the actual university-only parent gate; a Mac fixture cannot pass it.
    with pytest.raises(ValueError,match='another root or clock'):
        client.bound_collection_source(io.BytesIO(wire),collector.window,claimed,ready,{'supervised_control_text':bundle['control'].decode()},(root/'authorization.json').read_bytes())
    receipt=codec.receive(io.BytesIO(wire),tmp_path/'received',bundle['control'],(root/'authorization.json').read_bytes(),window=collector.window)
    assert receipt['transport_verified'] is True and receipt['scientific_qualified'] is False and receipt['training_admitted'] is False
    assert (tmp_path/'received/proof.log').read_bytes()==(root/'proof.log').read_bytes()
    assert not (tmp_path/'received/capture/logs/proof.log').exists()
    final=json.loads((tmp_path/'received/capture/results/final-receipt.json').read_bytes());assert final['kind']=='pgd-component-work-receipt-v1' and final['all_saved_role_checks_returned'] is True and final['collection_complete'] is False
    for name,d in receipt['files'].items():
        target=tmp_path/'received'/name;assert hashlib.sha256(target.read_bytes()).hexdigest()==d['sha256']
        assert stat.S_IMODE(target.stat().st_mode)==0o600 and target.stat().st_nlink==1
    for name,value in [('HEADER.json',header),('TERMINAL.json',terminal),('RECEIPT.json',receipt),('FLOW.json',{'origin':ORIGIN,'completed':state['completed'],'wire_bytes':len(wire),'wire_sha256':hashlib.sha256(wire).hexdigest(),'actual_process_signals_timers_GPU_proof_verification_or_scientific_work':False,'collection_transport_verified_only':True,'real_client_refuses_nonuniversity_fixture_parent':True})]:put(tmp_path/name,value)
    put(tmp_path/'WIRE.bin',wire)


@pytest.mark.parametrize('fault',[None,'truncated','hash'])
def test_real_setup_upload_and_capsule_extraction(bundle,support,tmp_path,monkeypatch,fault):
    """Actual file/setup/decoder path; installer/target/fallocate are fake ports."""
    boot=support.boot;events=[]
    imported=NS(predictor_pgd_split_scratch_v1=support.scratch,predictor_pgd_split_coordinator_v1=support.coord)
    adapter=module(bundle['generated'][build.OUTER],'fake_component_setup_outer',imported)
    def no_install(self):events.append(('installer-FAKE',len(self.files)));return self
    monkeypatch.setattr(boot.PinnedImporter,'install',no_install)
    monkeypatch.setattr(support.coord,'target_gate',lambda *a:events.append(('target-FAKE',True)))
    proxy=NS(**vars(os))
    def reserve(fd,offset,length):
        assert offset==0 and length==16*2**20;events.append(('preallocator-FAKE-ordinary-reserve-bytes',length))
        data=b'\xa7'*(2**20)
        for _ in range(16):assert os.write(fd,data)==len(data)
    proxy.posix_fallocate=reserve;adapter.os=proxy
    root=tmp_path/'scratch';root.mkdir(mode=0o700)
    c={'supervised_control_text':bundle['control'].decode(),'outer_source_sha256':build.digest(bundle['generated'][build.OUTER])}
    b,p,csha=boot.control(bundle['control']);packet=build.local.Blueprint(ROOT).inputs['capsule_archive']
    if fault=='truncated':packet=packet[:-1]
    if fault=='hash':packet=bytes([packet[0]^1])+packet[1:]
    clock=Clock()
    def prepare():return adapter.prepare_tree(root,bundle['uploads'],c,b,p,{'user_basis':ORIGIN},build.enc(c),build.enc({'origin':ORIGIN,'actual_user_approval':False}),boot,io.BytesIO(packet),clock)
    if fault:
        with pytest.raises((ValueError,EOFError)):prepare()
        assert (root/'component-inputs.tar.partial').exists() and not (root/'component-inputs.tar').exists()
        assert not (root/'source/MANIFEST.json').exists()
    else:
        coord,scratch,guard,binding,derived=prepare();guard.verify_assets(phase='setup');guard.proof_readiness()
        assert coord is support.coord and scratch is support.scratch and len(guard.layout.assets)==83
        assert support.boot.control((root/'control.json').read_bytes())[2]==csha
        assert support.boot.authorization(derived,csha)['user_basis']==ORIGIN
        for name,d in bundle['static']['packet']['files'].items():assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==d['sha256']
        assert not (root/'capture/results/components/native').exists() and not (root/'capture/results/components/forward-preserving').exists()
    put(tmp_path/'SETUP-OBSERVATION.json',{'origin':ORIGIN,'fault':fault,'events':events,'actual_namespace_install_target_query_proof_or_preallocation':False,'ordinary_reserve_fixture_bytes':16*2**20,'science_or_target_work':False})


@pytest.mark.parametrize('armed_fixture',[False,True])
def test_isolated_cli_checks_approval_before_project_import(bundle,tmp_path,armed_fixture):
    folder=tmp_path/'handoff';folder.mkdir(mode=0o700)
    if armed_fixture:put(folder/'AUTHORIZATION.json',{'origin':ORIGIN,'actual_user_approval':False})
    events=[];fake=NS(path=[],exit=lambda status:events.append(('exit',status)))
    def load(name,globals=None,locals=None,fromlist=(),level=0):
        if name=='sys':return fake
        if name=='models.predictor_pgd_component_client_v1':
            events.append(('project-import',name));assert fake.path==[str(ROOT)]
            return NS(run=lambda path:events.append(('run-FAKE',str(path))) or 0)
        assert not name.startswith(('models','torch','numpy'))
        return builtins.__import__(name,globals,locals,fromlist,level)
    scope={'__name__':'EXPLICIT-ISOLATED-CLI-DEFINITION','__file__':str(ROOT/build.CLI),'__builtins__':{**vars(builtins),'__import__':load}}
    exec(compile(bundle['generated'][build.CLI],str(ROOT/build.CLI),'exec'),scope)
    if armed_fixture:assert scope['main']([str(folder)])==0;assert events==[('project-import','models.predictor_pgd_component_client_v1'),('run-FAKE',str(folder))]
    else:
        with pytest.raises(SystemExit,match='Unarmed'):scope['main']([str(folder)])
        assert events==[] and fake.path==[]
