"""New cold source/metadata and explicit fake launch/import/pipe checks only.

No actual process, pipe, timer, PID API, signal, proc tree, proof, Torch/NumPy,
model, gradient, profiler, GPU, SSH or paid work is used.
"""
import ast
import base64
import builtins
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import stat
import sys
from types import ModuleType,SimpleNamespace as NS
import pytest
from models import predictor_pgd_split_bootstrap_v1 as b
from models import predictor_pgd_split_coordinator_v1 as c
from models import predictor_pgd_split_watch_v1 as oldwatch

ROOT=Path(__file__).resolve().parents[1]
POLICY=(ROOT/'configs/predictor_pgd_split_launch_v1.json').read_bytes()

@pytest.fixture(scope='module')
def concrete():
    source={}
    for name in b.SUPERVISORS.values():
        path=ROOT/'models'/('predictor_pgd_split_bootstrap_v1.py' if name=='bootstrap.py' else name)
        source[name]=path.read_bytes() if path.exists() else b'"""EXPLICIT INERT TEST COLLECTOR PLACEHOLDER; NOT A RUNNER."""\n'
    raw=c.make_control(POLICY,source);control,policy,sha=b.control(raw)
    auth=b.encoded({'kind':'pgd-split-supervised-university-exception-v1','approved':True,'control_sha256':sha,'maximum_new_attempts':1,
        'user_basis':'EXPLICIT LOCAL METADATA SIMULATION, NOT A LIVE USER GRANT'})
    return NS(sources=source,raw=raw,control=control,policy=policy,sha=sha,auth=auth)


def test_control_installer_and_frozen_workload_have_no_hash_cycle_or_scope_change(concrete):
    x=concrete;assert len(x.raw)<256*1024 and hashlib.sha256(POLICY).hexdigest()==b.POLICY_SHA
    ib,isha=b.installation(x.control,x.sha);i=b.parsed(ib)
    assert i['control_sha256']==x.sha and i['files']==x.control['supervisor_sources'] and isha==b.digest(ib)
    assert 'installation_sha256' not in x.control
    d=json.loads(x.policy['design_text']);assert x.policy['scope']['gradient_accounting']==d['gradient_accounting']
    assert x.policy['scope']['phase_seconds']==d['phase_seconds'] and sum(d['phase_seconds'].values())==2400
    assert b.SECONDS=={'proof':60,'numerical':1680,'native':120,'forward-preserving':120}
    assert x.policy['scope']['host']==c.HOST and not x.policy['authorized'] and not x.policy['target_helper_ready']
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules

@pytest.mark.parametrize('fault',['policy','missing-source','support','extra-control','old-kind','noncanonical','digest','duplicate-json','old-auth','auth-count','auth-control'])
def test_scope_drift_cannot_reuse_old_authorization_or_unpinned_source(concrete,fault):
    x=concrete
    if fault in ('policy','missing-source','support'):
        sources=dict(x.sources);policy=POLICY
        if fault=='policy':policy+=b'\n'
        elif fault=='missing-source':sources.pop('bootstrap.py')
        else:sources['scratch_allocation_v2.py']+=b'\n'
        with pytest.raises(ValueError):c.make_control(policy,sources)
        return
    if fault.startswith('auth') or fault=='old-auth':
        a=b.parsed(x.auth)
        if fault=='old-auth':a['kind']='OLD-CONSUMED-GRANT'
        elif fault=='auth-count':a['maximum_new_attempts']=2
        else:a['control_sha256']='e'*64
        with pytest.raises(ValueError):b.authorization(b.encoded(a),x.sha)
        return
    d=copy.deepcopy(x.control)
    if fault=='extra-control':d['ports']={'compute':'UNTRUSTED'}
    elif fault=='old-kind':d['kind']='OLD-CONTROL'
    elif fault=='digest':d['policy_sha256']='f'*64
    raw=b.encoded(d)
    if fault=='noncanonical':raw+=b'\n'
    elif fault=='duplicate-json':raw=raw.rstrip()[:-1]+b',"kind":"pgd-split-supervised-control-v1"}\n'
    with pytest.raises(ValueError):b.control(raw)


def write(root,name,data):
    p=root/name;p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    q=p.parent
    while q!=root:q.chmod(0o700);q=q.parent
    p.write_bytes(data);p.chmod(0o600);return p

@pytest.fixture
def source_tree(tmp_path,concrete):
    root=tmp_path/'scratch';root.mkdir(mode=0o700)
    previous=ROOT/'runs/pretrained/diagnostic/predictor-pgd-split-runtime-v1-20261115/UNEXECUTED-assembly-02'
    for path in json.loads(concrete.policy['scratch_contract_text'])['runtime_files']:
        write(root,'overlay/'+path,(previous/path).read_bytes())
    for path,data in concrete.sources.items():write(root,'supervisor/'+path,data)
    from models.predictor_pgd_split_scratch_v1 import identity_binding
    return root,identity_binding(root)


def test_pinned_loader_caches_verified_bytes_executes_only_inert_package_and_refuses_reload(source_tree,concrete):
    root,binding=source_tree;f=b.PinnedImporter(root,binding,concrete.control,concrete.policy)
    assert len(f.files)==69 and f.control_sha256==concrete.sha
    spec=f.find_spec('models');module=ModuleType('models');module.__spec__=spec
    f.exec_module(module) # ONLY the pinned one-line inert package docstring.
    assert module.__file__==str(root/'overlay/models/__init__.py') and module.__path__==[str(root/'overlay/models')]
    assert module.__cached__ is None and f.loaded=={'models'}
    with pytest.raises(ImportError,match='reload'):f.find_spec('models')
    with pytest.raises(ImportError):f.exec_module(module)
    assert f.find_spec('json') is None
    with pytest.raises(ModuleNotFoundError,match='Unpinned'):f.find_spec('models.UNREVIEWED')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules

@pytest.mark.parametrize('fault',['extent','same-size-hash','symlink','hardlink','mode'])
def test_loader_refuses_changed_source_before_any_module_execution(source_tree,concrete,fault):
    root,binding=source_tree;p=root/'overlay/models/__init__.py'
    if fault=='extent':p.write_bytes(p.read_bytes()+b'\n')
    elif fault=='same-size-hash':data=p.read_bytes();p.write_bytes(b'#'+data[1:])
    elif fault=='symlink':p.rename(p.with_name('PRESERVED'));p.symlink_to('PRESERVED')
    elif fault=='hardlink':os.link(p,p.with_name('EXTRA-LINK'))
    else:p.chmod(0o644)
    with pytest.raises(ValueError):b.PinnedImporter(root,binding,concrete.control,concrete.policy)
    assert 'torch' not in sys.modules

@pytest.mark.parametrize('fault',[None,'ambient','foreign-loader','foreign-file','no-finder'])
def test_import_installation_and_loaded_namespace_must_match_one_exact_finder(source_tree,concrete,monkeypatch,fault):
    root,binding=source_tree;f=b.PinnedImporter(root,binding,concrete.control,concrete.policy)
    fake_sys=NS(meta_path=[],modules={});monkeypatch.setattr(b,'sys',fake_sys)
    if fault=='ambient':
        fake_sys.modules['models']=ModuleType('models')
        with pytest.raises(ValueError,match='Fresh project'):f.install()
        return
    f.install();module=ModuleType('models');module.__spec__=f.find_spec('models');f.exec_module(module);fake_sys.modules['models']=module
    if fault=='foreign-loader':module.__spec__.loader=object()
    elif fault=='foreign-file':module.__file__='/UNTRUSTED/models/__init__.py'
    elif fault=='no-finder':fake_sys.meta_path.clear()
    if fault is None:
        assert b.require_namespace(root,binding,concrete.sha) is f
        with pytest.raises(ValueError):f.install()
    else:
        with pytest.raises(ValueError):b.require_namespace(root,binding,concrete.sha)


def plan_and_session(concrete,role='proof'):
    start=10.;deadline=start+b.SECONDS[role];source={} if role in ('proof','numerical') else {'numerical_sha256':'1'*64,'original_sha256':'2'*64,'boundary_sha256':'3'*64}
    current={'role':role,'nonce':'4'*64,'launch_started':start,'deadline':deadline,'source_hashes':source}
    session={'kind':'pgd-split-parent-session-v1','control_sha256':concrete.sha,'parent_pid':77,'connection_started':0.,'work_deadline':2400.,'connection_deadline':3000.,
        'current':current,'completed':list(b.ROLES[:b.ROLES.index(role)]),'phase':'running','complete':False}
    plan=dict(current,kind='pgd-split-child-launch-v1',control_sha256=concrete.sha,authorization_sha256=b.digest(concrete.auth),parent_pid=77,
        root_binding={'opaque':'EXPLICIT METADATA ROOT STUB'},work_deadline=2400.,connection_deadline=3000.,session_sha256=b.digest(b.encoded(session)))
    return plan,session

@pytest.mark.parametrize('role',b.ROLES)
def test_child_plan_binds_original_parent_clock_and_exact_serial_prefix(concrete,role):
    plan,session=plan_and_session(concrete,role)
    assert b.validate_plan(plan,concrete.sha,session,10.) is plan

@pytest.mark.parametrize('fault',['cap','clock-reset','clock-nan','role-order','nonce','parent','complete','sources','extra-arg','old-control','session-deadline'])
def test_child_plan_refuses_changed_scope_session_or_clock(concrete,fault):
    p,s=plan_and_session(concrete)
    if fault=='cap':p['deadline']+=.001
    elif fault=='clock-reset':p['launch_started']=11.
    elif fault=='clock-nan':p['deadline']=float('nan')
    elif fault=='role-order':s['completed']=['proof']
    elif fault=='nonce':s['current']=dict(s['current'],nonce='f'*64)
    elif fault=='parent':s['parent_pid']=999
    elif fault=='complete':s['complete']=True
    elif fault=='sources':p['source_hashes']={'numerical_sha256':'e'*64}
    elif fault=='extra-arg':p['callback']='UNTRUSTED'
    elif fault=='old-control':p['control_sha256']='a'*64
    else:s['work_deadline']=2401.
    with pytest.raises(ValueError):b.validate_plan(p,concrete.sha,s,10.)

@pytest.mark.parametrize('fault',[None,'EOF','extra','wrong','timeout','parent','not-pipe','deadline','late'])
def test_release_gate_uses_only_owned_fake_pipe_and_original_default_alarm(monkeypatch,fault):
    events=[];now=[10.];parents=[77]
    def parent():return parents[0]
    def read(fd,size):
        events.append(('read',fd,size))
        if fault=='late':now[0]=71.
        return {'EOF':b'','extra':b'GG','wrong':b'X'}.get(fault,b'G')
    fake_os=NS(getppid=parent,fstat=lambda fd:NS(st_mode=(stat.S_IFREG if fault=='not-pipe' else stat.S_IFIFO)|0o600),read=read,close=lambda fd:events.append(('close',fd)))
    fake_signal=NS(SIGALRM=signal.SIGALRM,SIG_DFL=signal.SIG_DFL,ITIMER_REAL=signal.ITIMER_REAL,
        signal=lambda *a:events.append(('signal',a)),setitimer=lambda *a:events.append(('timer',a)))
    monkeypatch.setattr(b,'os',fake_os);monkeypatch.setattr(b,'signal',fake_signal);monkeypatch.setattr(b,'time',NS(monotonic=lambda:now[0]))
    monkeypatch.setattr(b,'select',NS(select=lambda *a:([] if fault=='timeout' else [20],[],[])))
    if fault=='parent':parents[0]=78
    deadline=9. if fault=='deadline' else 70.
    if fault is None:
        b.wait_release(20,77,deadline)
        assert events==[('signal',(signal.SIGALRM,signal.SIG_DFL)),('timer',(signal.ITIMER_REAL,60.)),('read',20,2),('close',20)]
    else:
        with pytest.raises((ValueError,TimeoutError)):b.wait_release(20,77,deadline)
        if fault not in ('parent','not-pipe','deadline'):assert events[-1]==('close',20)


def test_mount_and_host_preflight_refuse_local_analogue_before_proc_or_GPU(monkeypatch):
    monkeypatch.setattr(c,'platform',NS(node=lambda:'LOCAL-FAKE-MAC',system=lambda:'Darwin',python_version=lambda:'3.11.16'))
    class Forbidden:
        def __init__(self,*a):raise AssertionError('No path/proc query after foreign-host refusal')
    monkeypatch.setattr(c,'Path',Forbidden)
    with pytest.raises(ValueError,match='Exact authorized'):c.target_gate('/FAKE',{})
    text='10 1 1:1 / / rw - xfs /dev/root rw\n11 10 0:55 / /home rw - nfs server:/home rw\n'
    assert c.mount_row('/var/tmp/test',text)=={'mount':'/','filesystem':'xfs','options':['rw','rw']}
    assert c.mount_row('/home/user',text)['filesystem']=='nfs'

@pytest.mark.parametrize('fault',[None,'nxnode','foreign','GPU','driver','used','columns','count','nan','oversized'])
def test_GPU_gate_queries_are_fake_and_another_compute_process_stays_refused(monkeypatch,fault):
    gpu='NVIDIA GeForce RTX 3070, '+c.HOST['gpu_uuid']+', 8192, 94, 580.178.04\n';active='';calls=[]
    if fault=='nxnode':active='2522822, /usr/NX/bin/nxnode.bin, 183\n'
    elif fault=='foreign':active='999, other-task, 10\n'
    elif fault=='GPU':gpu=gpu.replace('RTX 3070','RTX 3090')
    elif fault=='driver':gpu=gpu.replace('580.178.04','OTHER')
    elif fault=='used':gpu=gpu.replace('8192, 94','8192, 513')
    elif fault=='columns':gpu='WRONG\n'
    elif fault=='count':gpu*=2
    elif fault=='nan':gpu=gpu.replace('8192, 94','8192, nan')
    elif fault=='oversized':gpu='X'*16385
    def output(cmd,**kw):calls.append(cmd);return active if '--query-compute-apps=pid,process_name,used_memory' in cmd else gpu
    monkeypatch.setattr(c,'subprocess',NS(check_output=output,PIPE='EXPLICIT-FAKE-PIPE-SENTINEL'))
    if fault is None:
        row=c.gpu_gate();assert row['allowed_pid'] is None and not row['sampling_is_continuous_exclusivity'] and len(calls)==2
    else:
        with pytest.raises(ValueError):c.gpu_gate()
    assert all(cmd[0]=='nvidia-smi' for cmd in calls)


class Clock:
    def __init__(self):self.now=10.
    def __call__(self):return self.now
    def pause(self,n):assert 0<=n<=.1;self.now+=n

class FakeDriver:
    def __init__(self,owner):self.owner=owner;self.sources={};self.events=[];self.fault=FAKE_FAULT;self.saved=[];self.plans=[]
    def prepare(self):
        self.events.append('prepare')
        if self.fault=='prepare':raise ValueError('SIMULATED preflight refusal')
        if self.fault=='setup-time':self.owner.clock.now=181.
    def save(self,state,*,replace):
        self.events.append(('save',replace,state['phase']))
        if self.fault=='claim-exists' and not replace:raise FileExistsError('SIMULATED prior session')
        if self.fault=='save-running' and state['phase']=='running':raise OSError('SIMULATED session publication')
        if self.fault=='save-final' and state['phase']=='await-collection':raise OSError('SIMULATED final publication')
        self.saved.append(copy.deepcopy(state));return b.digest(b.encoded(state))
    def launch(self,role,plan):
        self.events.append(('launch',role));self.plans.append(copy.deepcopy(plan))
        b.validate_plan(plan,self.owner.csha,self.owner.state,self.owner.clock())
        if self.fault==role+'-launch':raise RuntimeError('SIMULATED '+role+' launch refusal')
        if self.fault=='late-launch':self.owner.clock.now=plan['deadline']
        elif self.fault=='backward':self.owner.clock.now-=1
        else:self.owner.clock.now+=.1
        index=b.ROLES.index(role);pid=900+index
        if self.fault=='reuse' and role=='numerical':pid=900
        return {'kind':'EXPLICIT-FAKE-CHILD-OBSERVATION','pid':pid,'start_ticks':100+index,'exit_status':0,'parent_observation_passed':True}
    def verify(self,role,plan,observed):
        self.events.append(('verify',role))
        if self.fault==role+'-verify':raise ValueError('SIMULATED '+role+' saved gate rejection')
        if role=='numerical':self.sources={'numerical_sha256':'1'*64,'original_sha256':'2'*64,'boundary_sha256':'3'*64}
        if self.fault=='late-verify':self.owner.clock.now=plan['deadline']
        result={'role':role,'pid':observed['pid'],'start_ticks':observed['start_ticks'],'saved_gates_returned':True,
            'scientific_qualified':False,'training_admitted':False,'evidence_origin':'EXPLICIT FAKE CHILD/RECEIPT ONLY'}
        if self.fault=='receipt':result['pid']+=1000
        return result
FAKE_FAULT=None


def coordinator(concrete,tmp_path,monkeypatch,fault=None):
    monkeypatch.setattr(c,'SystemDriver',FakeDriver);monkeypatch.setattr(sys.modules[__name__],'FAKE_FAULT',fault)
    return c.Coordinator(tmp_path,{'opaque':'FAKE-ROOT-BINDING'},concrete.raw,concrete.auth,connection_started=0.,clock=Clock())


def test_coordinator_orders_all_four_new_roles_with_no_clock_reset_or_completion_claim(concrete,tmp_path,monkeypatch):
    q=coordinator(concrete,tmp_path,monkeypatch);state=q.run_work()
    assert state['phase']=='await-collection' and state['completed']==list(b.ROLES) and not state['complete'] and state['collection_pending']
    assert [x[1] for x in q.driver.events if isinstance(x,tuple) and x[0]=='launch']==list(b.ROLES)
    for role,plan in zip(b.ROLES,q.driver.plans):
        assert plan['deadline']<=plan['launch_started']+b.SECONDS[role] and plan['work_deadline']==2400. and plan['connection_deadline']==3000.
        assert set(plan['source_hashes'])==({'numerical_sha256','original_sha256','boundary_sha256'} if role in ('native','forward-preserving') else set())
    assert q.claimed and len(q.receipts)==4 and not state['scientific_qualified']
    with pytest.raises(ValueError,match='One coordinator'):q.run_work()
    (tmp_path/'FAKE-COORDINATOR-SESSION.json').write_bytes(b.encoded({'origin':'EXPLICIT-FAKE-PROCESSES-NO-WORK','session':state,'plans':q.driver.plans}))

@pytest.mark.parametrize('fault',['prepare','claim-exists','setup-time','save-running','save-final','reuse','late-launch','late-verify','backward','receipt',
    'proof-launch','numerical-launch','native-launch','forward-preserving-launch','proof-verify','numerical-verify','native-verify','forward-preserving-verify'])
def test_coordinator_refusal_stops_later_work_and_cannot_overwrite_an_unclaimed_session(concrete,tmp_path,monkeypatch,fault):
    q=coordinator(concrete,tmp_path,monkeypatch,fault)
    with pytest.raises((ValueError,RuntimeError,OSError,TimeoutError)):q.run_work()
    launches=[v[1] for v in q.driver.events if isinstance(v,tuple) and v[0]=='launch']
    assert launches==list(b.ROLES[:len(launches)]) and len(set(launches))==len(launches)
    if fault in ('prepare','claim-exists','setup-time'):
        assert not launches and not any(isinstance(v,tuple) and v[0]=='save' and v[1] is True for v in q.driver.events)
    if fault.startswith('native-'):assert 'forward-preserving' not in launches
    assert q.state['phase']=='failed' and not q.state['complete']
    with pytest.raises(ValueError,match='One coordinator'):q.run_work()


class LaunchWorld:
    def __init__(self,owner,fault=None):self.owner=owner;self.fault=fault;self.events=[];self.closed=[];self.lease=None
    def pipe(self,flags):
        self.events.append('pipe')
        if self.fault=='pipe':raise OSError('SIMULATED pipe')
        return 20,21
    def open(self,path,flags,mode):self.events.append(('log',str(path)));return 22
    def close(self,fd):self.closed.append(fd)
    def write(self,fd,data):
        self.events.append(('release',fd,data))
        if self.fault=='release':raise BrokenPipeError('SIMULATED release')
        return 1
    def popen(self,command,**kw):
        self.events.append(('Popen',command,kw))
        if self.fault=='Popen':raise OSError('SIMULATED Popen')
        self.child=NS(pid=901,wait=lambda timeout:self.events.append(('wait',timeout)))
        return self.child
    def row(self,pid):return {'pid':pid,'start_ticks':71,'parent_pid':77,'session_pid':999 if self.fault=='identity' else pid,'group_pid':pid}
    def lease_factory(self,pid,birth,parent,session,reader):
        self.events.append('lease')
        if self.fault=='lease':raise ValueError('SIMULATED PID binding')
        world=self
        class Lease:
            def __init__(self):self.closed=False
            def send(self,sig):world.events.append(('held-handle-signal',sig))
            def close(self):self.closed=True;world.events.append('lease-close')
        self.lease=Lease()
        if self.fault=='late':self.owner.clock.now=self.owner.plan['deadline']
        return self.lease
    def gate(self,*a):
        self.events.append('GPU-gate')
        if self.fault=='gate':raise ValueError('SIMULATED live gate')
    def monitor(self,child,lease,**kw):
        self.events.append(('monitor',kw))
        if self.fault=='handoff':raise TimeoutError('SIMULATED alarm before monitor finally takes ownership')
        if self.fault=='unclosed-return':return {'pid':901,'start_ticks':71,'parent_observation_passed':True}
        try:
            if self.fault=='monitor':raise RuntimeError('SIMULATED monitor refusal')
            return {'pid':901,'start_ticks':71,'parent_observation_passed':True}
        finally:lease.close()

@pytest.mark.parametrize('role',b.ROLES)
def test_concrete_launch_recipe_binds_handle_before_fake_release_and_closes_every_fd(concrete,tmp_path,monkeypatch,role):
    owner=NS(root=tmp_path,c=concrete.control,clock=Clock(),work_deadline=2400.)
    plan,_=plan_and_session(concrete,role);owner.plan=plan;w=LaunchWorld(owner)
    fake_os=NS(O_CLOEXEC=os.O_CLOEXEC,O_WRONLY=os.O_WRONLY,O_CREAT=os.O_CREAT,O_EXCL=os.O_EXCL,O_NOFOLLOW=os.O_NOFOLLOW,
        pipe2=w.pipe,open=w.open,close=w.close,write=w.write,getpid=lambda:77,environ={'CUDA_VISIBLE_DEVICES':'0','HOME':'EXPLICIT-FAKE-HOME','UNWANTED_SECRET':'MUST-NOT-INHERIT'})
    monkeypatch.setattr(c,'os',fake_os);monkeypatch.setattr(c,'gpu_gate',w.gate);monkeypatch.setattr(c,'monitor_for',lambda role:w.monitor)
    monkeypatch.setattr(c,'subprocess',NS(Popen=w.popen,DEVNULL='FAKE-DEVNULL',TimeoutExpired=TimeoutError))
    monkeypatch.setattr(c,'frozen_watch',NS(ProcReader=lambda:NS(identity=w.row,sample=lambda *a:None),PidfdLease=w.lease_factory))
    driver=c.SystemDriver(owner);driver.guard=NS(scan=lambda **kw:w.events.append(('storage',kw)))
    out=driver.launch(role,plan)
    assert out['pid']==901 and w.events.index('lease')<next(i for i,e in enumerate(w.events) if isinstance(e,tuple) and e[0]=='release')
    assert sorted(w.closed)==[20,21,22] and w.events.count('lease-close')==1
    args=next(e for e in w.events if isinstance(e,tuple) and e[0]=='Popen');command,kw=args[1:]
    assert command[1:5]==['-I','-B','-u','-c'] and kw['pass_fds']==(20,) and kw['close_fds'] and kw['start_new_session'] and not kw['shell'] and kw['umask']==0o077
    assert kw['stdout']==kw['stderr']==22 and 'UNWANTED_SECRET' not in kw['env']
    monitor=next(e[1] for e in w.events if isinstance(e,tuple) and e[0]=='monitor')
    assert monitor['launch_started']==plan['launch_started'] and monitor['phase_deadline']==plan['deadline'] and monitor['work_deadline']==2400.

@pytest.mark.parametrize('fault',['pipe','Popen','identity','lease','gate','release','late','monitor','handoff','unclosed-return'])
def test_fake_launch_failure_never_releases_unbound_work_or_uses_numeric_signals(concrete,tmp_path,monkeypatch,fault):
    owner=NS(root=tmp_path,c=concrete.control,clock=Clock(),work_deadline=2400.)
    plan,_=plan_and_session(concrete);owner.plan=plan;w=LaunchWorld(owner,fault)
    fake_os=NS(O_CLOEXEC=os.O_CLOEXEC,O_WRONLY=os.O_WRONLY,O_CREAT=os.O_CREAT,O_EXCL=os.O_EXCL,O_NOFOLLOW=os.O_NOFOLLOW,
        pipe2=w.pipe,open=w.open,close=w.close,write=w.write,getpid=lambda:77,environ={})
    monkeypatch.setattr(c,'os',fake_os);monkeypatch.setattr(c,'gpu_gate',w.gate);monkeypatch.setattr(c,'monitor_for',lambda role:w.monitor)
    monkeypatch.setattr(c,'subprocess',NS(Popen=w.popen,DEVNULL='FAKE-DEVNULL',TimeoutExpired=TimeoutError))
    monkeypatch.setattr(c,'frozen_watch',NS(ProcReader=lambda:NS(identity=w.row,sample=lambda *a:None),PidfdLease=w.lease_factory))
    driver=c.SystemDriver(owner);driver.guard=NS(scan=lambda **kw:None)
    with pytest.raises((OSError,ValueError,RuntimeError,TimeoutError)):driver.launch('proof',plan)
    if fault in ('pipe','Popen','identity','lease','gate','late'):assert not any(isinstance(e,tuple) and e[0]=='release' for e in w.events)
    if fault not in ('pipe',):assert sorted(w.closed)==[20,21,22]
    if fault in ('identity','lease'):assert not any(isinstance(e,tuple) and e[0]=='held-handle-signal' for e in w.events)
    if w.lease is not None:assert w.events.count('lease-close')==1
    assert len(w.closed)==len(set(w.closed))


def test_monitor_preparation_failure_precedes_every_OS_launch(concrete,tmp_path,monkeypatch):
    owner=NS(root=tmp_path,c=concrete.control,clock=Clock(),work_deadline=2400.);driver=c.SystemDriver(owner)
    def refuse(role):raise ValueError('SIMULATED source verification/compile refusal')
    monkeypatch.setattr(c,'monitor_for',refuse)
    monkeypatch.setattr(c,'os',NS(pipe2=lambda *a:pytest.fail('No pipe before monitor preparation')))
    with pytest.raises(ValueError,match='SIMULATED'):driver.launch('proof',{'role':'proof'})


def test_session_writer_never_replaces_foreign_or_unclaimed_state(concrete,tmp_path,monkeypatch):
    owner=NS();driver=c.SystemDriver(owner);data=b'EXPLICIT EXISTING SESSION';calls=[]
    driver.guard=NS(read=lambda *a:data)
    monkeypatch.setattr(c,'_metadata_write',lambda *a,**kw:calls.append(kw) or 'a'*64)
    with pytest.raises(ValueError,match='Only this parent'):driver.save({'current':None},replace=True)
    assert not calls
    driver.session_sha256=b.digest(data)
    assert driver.save({'current':{'role':'proof'}},replace=True)=='a'*64 and calls[-1]['phase']=='proof'
    driver.guard.read=lambda *a:b'FOREIGN CHANGE'
    with pytest.raises(ValueError,match='Only this parent'):driver.save({'current':None},replace=True)


class WatchWorld:
    def __init__(self,role,fault=None):self.role=role;self.fault=fault;self.clock=Clock();self.pid=901;self.samples=0;self.closed=0;self.sent=[];self.alive=True
    @property
    def birth(self):return 71
    def dead(self):return not self.alive
    def close(self):self.closed+=1
    def send(self,sig):self.sent.append(sig);self.alive=False
    def poll(self):
        if self.samples>=2:self.alive=False;return 1 if self.fault=='exit' else 0
        return None
    def wait(self,timeout):return 0
    def sample(self,pid,birth):
        self.samples+=1
        if self.fault=='long':self.clock.now+=130.
        if self.fault=='timeout':self.clock.now+=b.SECONDS[self.role]
        return {'processes':{901:{'pid':901,'start_ticks':71,'parent_pid':77,'session_pid':901,'rss_bytes':16*2**30}},'rss_bytes':16*2**30}

@pytest.mark.parametrize('role',['proof','numerical'])
def test_noncomponent_monitor_adapts_only_explicit_role_budget_and_preserves_stop_finally(role):
    data=(ROOT/'models/predictor_pgd_split_watch_v1.py').read_bytes();tree,audit=c.watch_source(data)
    assert audit=={'time':1,'samples':1,'rss':1,'kind':1}
    ns=dict(oldwatch.__dict__);exec(compile(tree,'<EXPLICIT-FAKE-PARENT-MONITOR>','exec'),ns)
    w=WatchWorld(role)
    receipt=ns['watch_stage'](w,w,role=role,sample=w.sample,lease_factory=lambda *a:pytest.fail('No fake descendant'),gate=lambda:None,
        phase_deadline=10.+b.SECONDS[role],work_deadline=2400.,launch_started=10.,clock=w.clock,pause=w.clock.pause)
    assert receipt['role']==role and not receipt['RSS_ceiling_enforced'] and receipt['peak_observed_RSS_bytes']==16*2**30
    assert w.closed==1 and not w.sent and not receipt['scientific_qualified']

@pytest.mark.parametrize('role,fault',[('proof','timeout'),('numerical','timeout'),('proof','long'),('numerical','exit')])
def test_role_monitor_keeps_original_deadline_and_failure_stop_paths(role,fault):
    tree,_=c.watch_source((ROOT/'models/predictor_pgd_split_watch_v1.py').read_bytes());ns=dict(oldwatch.__dict__);exec(compile(tree,'<FAKE-MONITOR-ONLY>','exec'),ns)
    w=WatchWorld(role,fault)
    with pytest.raises((TimeoutError,RuntimeError)):
        ns['watch_stage'](w,w,role=role,sample=w.sample,lease_factory=lambda *a:None,gate=lambda:None,
            phase_deadline=10.+b.SECONDS[role],work_deadline=2400.,launch_started=10.,clock=w.clock,pause=w.clock.pause)
    assert w.closed==1


def test_numerical_role_can_use_its_own_existing_phase_without_component_deadline():
    tree,_=c.watch_source((ROOT/'models/predictor_pgd_split_watch_v1.py').read_bytes());ns=dict(oldwatch.__dict__);exec(compile(tree,'<FAKE-NUMERICAL-CLOCK-ONLY>','exec'),ns)
    w=WatchWorld('numerical','long')
    out=ns['watch_stage'](w,w,role='numerical',sample=w.sample,lease_factory=lambda *a:None,gate=lambda:None,
        phase_deadline=1690.,work_deadline=2400.,launch_started=10.,clock=w.clock,pause=w.clock.pause)
    assert 120<out['wall_seconds']<1680 and not out['RSS_ceiling_enforced'] and w.closed==1


def test_child_driver_hashes_exact_inert_bytes_and_propagates_return_code():
    # Execute ONLY a two-function inert test bootstrap, never the real bootstrap.
    body=b"TRACE=[]\ndef wait_release(fd,parent,deadline): TRACE.append(('release-stub',fd,parent,deadline))\ndef child_entry(raw): TRACE.append(('entry-stub',raw));return 7\n"
    p={'parent_pid':77,'deadline':70.};encoded=base64.b64encode(b.encoded(p)).decode()
    row=NS(st_mode=stat.S_IFREG|0o600,st_nlink=1,st_uid=os.getuid(),st_size=len(body),st_dev=55,st_ino=99,st_mtime_ns=1)
    class Stream(io.BytesIO):
        def fileno(self):return 42
    fake_os=NS(lstat=lambda p:row,getuid=os.getuid,open=lambda *a:42,fdopen=lambda *a:Stream(body),fstat=lambda fd:row,
        O_RDONLY=os.O_RDONLY,O_NOFOLLOW=os.O_NOFOLLOW)
    fake_sys=NS(argv=['-c','/EXPLICIT-INERT-STUB.py',b.digest(body),str(len(body)),encoded,'20'],exit=lambda code:(_ for _ in ()).throw(SystemExit(code)))
    imports={'os':fake_os,'sys':fake_sys,'stat':stat,'hashlib':hashlib,'json':json,'base64':base64}
    def importer(name,*a,**kw):
        if name not in imports:raise AssertionError('Unreviewed inert-driver import')
        return imports[name]
    ns={'__name__':'EXPLICIT-INERT-DRIVER-SIMULATION','__builtins__':dict(vars(builtins),__import__=importer)}
    with pytest.raises(SystemExit) as out:exec(compile(c.CHILD_DRIVER,'<INERT-BOOTSTRAP-ONLY>','exec'),ns)
    assert out.value.code==7 and ns['ns']['TRACE']==[('release-stub',20,77,70.),('entry-stub',b.encoded(p))]
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


def test_child_argument_dispatch_source_retains_original_math_packet_paths_and_limits():
    tree=ast.parse((ROOT/'models/predictor_pgd_split_coordinator_v1.py').read_bytes())
    dispatch=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dispatch_child');text=ast.unparse(dispatch)
    assert "'configs/base.json'" in text and "'external/vjepa2'" in text and "'checkpoints/vitl.pt'" in text
    assert 'resource.RLIMIT_FSIZE' in text and '16 * 2 ** 20' in text and '3 * 2 ** 30' in text
    assert text.index('target_gate')<text.index('runtime.numerical_entry') and text.count('runtime.numerical_entry')==1 and text.count('runtime.component_entry')==1
    # No scientific runtime is imported by these cold inspections.
    assert not any(isinstance(n,ast.Import) and any(a.name in ('torch','numpy') for a in n.names) for n in tree.body)


@pytest.mark.parametrize('fault',[None,'old-auth','session-bytes','parent','session-id','importer','late-install','dispatch'])
def test_child_entry_joins_real_control_checks_to_fake_source_installation_before_dispatch(concrete,monkeypatch,fault):
    p,s=plan_and_session(concrete,'native');events=[];now=[10.]
    p['parent_pid']=77;s['parent_pid']=77;p['root_binding']={'root':'/EXPLICIT-FAKE-SCRATCH'}
    sbytes=b.encoded(s);p['session_sha256']=b.digest(sbytes)
    abytes=concrete.auth
    if fault=='old-auth':
        a=b.parsed(abytes);a['kind']='OLD-CONSUMED';abytes=b.encoded(a);p['authorization_sha256']=b.digest(abytes)
    def read(root,binding,name,cap):
        events.append(('read',name))
        return {'control.json':concrete.raw,'authorization.json':abytes,'session.json':sbytes+(b' ' if fault=='session-bytes' else b'')}[name]
    class Importer:
        def __init__(self,*a):events.append('source-constructor')
        def install(self):
            events.append('install')
            if fault=='importer':raise ValueError('SIMULATED source installation refusal')
            if fault=='late-install':now[0]=p['deadline']
    def dispatch(*a):
        events.append('dispatch')
        if fault=='dispatch':raise ValueError('SIMULATED dispatch refusal')
        return 0
    def importer(name,*a,**kw):
        assert name=='models';return NS(predictor_pgd_split_coordinator_v1=NS(dispatch_child=dispatch))
    tree=ast.parse((ROOT/'models/predictor_pgd_split_bootstrap_v1.py').read_bytes());entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='child_entry')
    ns=dict(b.__dict__);ns.update(_CHILD_USED=False,private_read=read,PinnedImporter=Importer,time=NS(monotonic=lambda:now[0]),
        os=NS(getppid=lambda:78 if fault=='parent' else 77,getsid=lambda pid:999 if fault=='session-id' else 901,getpid=lambda:901,getpgrp=lambda:901),
        __builtins__=dict(vars(builtins),__import__=importer))
    exec(compile(ast.Module(body=[entry],type_ignores=[]),'<FAKE-CHILD-ENTRY-ONLY>','exec'),ns)
    raw=b.encoded(p)
    if fault is None:
        assert ns['child_entry'](raw)==0 and events[-3:]==['source-constructor','install','dispatch']
    else:
        with pytest.raises((ValueError,TimeoutError)):ns['child_entry'](raw)
        if fault!='dispatch':assert 'dispatch' not in events
    with pytest.raises(ValueError,match='One child entry'):ns['child_entry'](raw)
    assert 'torch' not in sys.modules


@pytest.mark.parametrize('role',b.ROLES)
def test_concrete_parent_saved_receipt_binding_uses_expected_child_and_source_hashes(concrete,tmp_path,role):
    events=[];parent=NS(root=tmp_path,csha=concrete.sha,policy=concrete.policy)
    binding={'design_sha256':concrete.policy['scope']['design_sha256'],'control_sha256':concrete.sha,'boundary_sha256':'3'*64,'probe_index':int(role=='forward-preserving')}
    numerical=b.encoded({'pid':901});original=b'ORIGINAL-METADATA-STUB';boundary=b'BOUNDARY-METADATA-STUB'
    marker=b.encoded({'opaque':'COMPLETE-STUB'});worker=b.encoded({'opaque':'WORKER-STUB'});resource=b.encoded({'opaque':'RESOURCE-STUB'})
    class Guard:
        def read(self,name,cap):
            events.append(('read',name))
            if name=='allocation-result-v2.json':return b'EXPLICIT-PROOF-RECEIPT-STUB'
            if name=='capture/results/run-status.json':return numerical
            if name.endswith('objective-0/numerical.json'):return original
            if name=='capture/results/boundary/boundary.json':return boundary
            if name.endswith('/complete.json'):return marker
            if name.endswith('/progress.jsonl'):return b'PROGRESS-STUB'
            if name.endswith('/worker.json'):return worker
            if name.endswith('/resource.json'):return resource
            raise AssertionError('Unreviewed receipt path: '+name)
        def verify_proof(self,sha):events.append(('proof',sha));return {'proof_execution_authenticated':False}
        def scan(self,**kw):events.append(('scan',kw))
    store=NS(_STORE=None)
    def bind(path):events.append(('bind',path));store._STORE=object()
    store.bind=bind
    records=NS(verify_worker=lambda *a,**kw:dict(worker_sha256=b.digest(numerical),numerical_scoped=True))
    def complete(*a,**kw):
        assert kw['marker_sha256']==b.digest(marker) and kw['progress_sha256']==b.digest(b'PROGRESS-STUB')
        events.append('component-completion-read');return {'metadata_complete':True}
    def joined(marker_value,worker_value,resource_value,observed,**kw):
        assert observed['pid']==901 and kw['binding']==binding
        events.append('parent-component-join');return {'metadata_bindings_consistent':True}
    job=NS(read_completion=complete,bind_parent_metadata=joined)
    def importer(name,*a,**kw):
        assert name=='models';return NS(_pgd_split_io_v1=store,_pgd_split_records_v1=records,predictor_pgd_split_probe_job_v1=job)
    tree=ast.parse((ROOT/'models/predictor_pgd_split_coordinator_v1.py').read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SystemDriver')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='verify')
    ns=dict(c.__dict__);ns.update(__builtins__=dict(vars(builtins),__import__=importer),_metadata_write=lambda *a,**kw:events.append(('write-parent',a[1])))
    exec(compile(ast.Module(body=[method],type_ignores=[]),'<FAKE-RECEIPT-IMPORTS-ONLY>','exec'),ns)
    fake_self=NS(owner=parent,guard=Guard(),sources={} if role in ('proof','numerical') else {'numerical_sha256':'1'*64,'original_sha256':'2'*64,'boundary_sha256':'3'*64})
    observed={'pid':901,'start_ticks':71,'exit_status':0,'parent_observation_passed':True}
    result=ns['verify'](fake_self,role,{},observed)
    assert result['role']==role and result['saved_gates_returned'] and not result['scientific_qualified']
    assert events[-1]==('scan',{'phase':'capture','settled':True})
    if role=='numerical':assert fake_self.sources=={'numerical_sha256':b.digest(numerical),'original_sha256':b.digest(original),'boundary_sha256':b.digest(boundary)}
    if role in ('native','forward-preserving'):assert 'component-completion-read' in events and 'parent-component-join' in events
    if role!='proof':assert ('write-parent','parent-'+role+'.json') in events
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
