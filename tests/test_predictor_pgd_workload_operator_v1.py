"""Concrete adapter controls with private fake OS/compute namespaces only."""
import ast
import base64
import builtins
import copy
import io
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
from types import ModuleType,SimpleNamespace as NS
import pytest
from models import predictor_pgd_workload_operator_v1 as op

ROOT=Path(__file__).resolve().parents[1]
ORIGIN='EXPLICIT-FAKE-WHOLE-PGD-OPERATOR; NOT A GRANT OR EXECUTION CLAIM'


def private_parents(parent):
    missing=[]
    while not parent.exists():missing.append(parent);parent=parent.parent
    for p in reversed(missing):p.mkdir(mode=0o700)


def put(p,data):
    private_parents(p.parent)
    data=data if type(data) is bytes else op.encoded(data)
    with p.open('xb') as out:out.write(data)
    p.chmod(0o600)


@pytest.fixture
def case(request,tmp_path):
    base=os.environ.get('VJEPA_WHOLE_PGD_OPERATOR_EVIDENCE')
    p=Path(base)/request.node.name if base else tmp_path/'case'
    p.mkdir(mode=0o700);return p


@pytest.fixture(autouse=True)
def no_actual_work(monkeypatch):
    def refuse(*a,**kw):raise AssertionError('Actual process/signal/timer/network work forbidden.')
    for m,n in ((subprocess,'Popen'),(subprocess,'check_output'),(signal,'signal'),(signal,'setitimer'),
                (threading.Thread,'start'),(os,'kill'),(os,'killpg'),(os,'system'),(os,'fork'),(os,'posix_spawn')):
        if hasattr(m,n):monkeypatch.setattr(m,n,refuse)
    original=builtins.__import__
    def imports(name,*a,**kw):
        if name.split('.')[0] in ('torch','numpy'):raise AssertionError('No scientific import in new adapter checks.')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',imports)
    monkeypatch.setattr(os,'fstatvfs',lambda fd:NS(f_bavail=20*2**30,f_frsize=1,f_favail=100000))


class Clock:
    def __init__(self):self.now=100.
    def __call__(self):self.now+=.00001;return self.now


class FakeSignal:
    SIGALRM=signal.SIGALRM;SIG_DFL=signal.SIG_DFL;SIGCHLD=signal.SIGCHLD
    SIGHUP=signal.SIGHUP;SIGTERM=signal.SIGTERM;SIGUSR1=signal.SIGUSR1
    SIGINT=signal.SIGINT;SIGKILL=signal.SIGKILL;ITIMER_REAL=signal.ITIMER_REAL
    def __init__(self,e):self.e=e;self.handlers={};self.end=None
    def getsignal(self,s):return self.handlers.get(s,self.SIG_DFL)
    def signal(self,s,h):self.handlers[s]=h;self.e.events.append(['FAKE-SIGNAL-HANDLER',int(s)])
    def getitimer(self,s):return (0.,0.) if self.end is None else (max(0.,self.end-self.e.clock.now),0.)
    def setitimer(self,s,seconds):
        self.end=self.e.clock.now+seconds if seconds else None
        self.e.events.append(['FAKE-HARD-ALARM',self.e.pid,seconds,self.end])


class FakeTimer:
    def __init__(self,e,seconds,callback):self.e=e;self.seconds=seconds;self.callback=callback;self.daemon=False
    def start(self):self.e.events.append(['FAKE-SOFT-WAKE',self.e.pid,self.seconds])
    def cancel(self):self.e.events.append(['FAKE-WAKE-CANCEL',self.e.pid])


class Child:
    def __init__(self,e,pid,plan):self.e=e;self.pid=pid;self.plan=plan;self.exited=False;self.status=0;self.released=False
    def poll(self):return self.status if self.exited else None
    def wait(self,timeout=None):return self.status


def fake_unit(plan,scope,pid):
    from models import predictor_pgd_workload_runtime_records_v1 as records
    n=40 if scope['method']=='primary' else 100
    clean=op.digest(b'FAKE-CLEAN');selected=op.digest(b'FAKE-SELECTED')
    role=plan['role'];engine='full-eot8-original-owned-v1' if role=='original-unit' else 'full-eot8-forward-preserving-owned-v1'
    proof={'kind':engine,'native_root_hook_removed':True,'native_scopes':1,'native_calls':1152,'native_refusals':0}
    if role!='original-unit':proof.update(initial_predictor_VJPs=192,forward_calls=384)
    steps=[{'index':i,'pixels':op.digest(('FAKE-POINT-'+str(i)).encode()),'gradient':op.digest(('FAKE-GRAD-'+str(i)).encode()),
        'losses':[1.],'consider_before_backward':True,'route':dict(proof),'context_seconds_including_observations':.001} for i in range(n)]
    ids=[1,0,0,0,0,1]+[j for j in range(2,2+scope['sensitivity_banks']) for _ in range(2)]
    scores=[{'bank':b,'draws':8 if b==0 else 4,'pixels':selected if i==5 or (i>=6 and i%2) else clean,
        'losses':[1.],'grad_enabled':False,'route_active':False} for i,b in enumerate(ids)]
    settings={'method':scope['method'],'steps':n//2,'restarts':2,'attack_masks':8,'scoring_masks':4,
        'epsilon_255':4.,'step_size_255':scope['step_size_255'],'profile':'research-primary-eot8-v1'}
    video='explicit-fake-training-row'
    u={'kind':'target-whole-pgd-input-gradient-workload-v1','status':'complete','route':engine,
        'metrics':{'settings':settings,'method':scope['method'],'video_id':video,'seeds':records._seeds(video,scope['sensitivity_banks']),
            'clean_loss':1.,'attack_clean_loss':1.,'attack_loss':1.,'adversarial_loss':1.,'max_pixel_delta':.01,'seconds':.001,
            'scoring_audits':[{'clean_loss':1.,'adversarial_loss':1.} for _ in range(scope['sensitivity_banks'])]},
        'steps':steps,'scores':scores,'adversarial':selected,'restored':True,'execution_binding':None,'research_admitted':False,
        'GPU_performance_qualified':False,'overhead_subtracted':False,'full_unit_seconds_including_guards_journal_and_observations':.002}
    baseline={'weights_sha256':op.digest(b'FAKE-WEIGHTS'),'clean_sha256':clean,'video_id':video,'checkpoint_sha256':op.digest(b'FAKE-CHECKPOINT')}
    w={'role':role,'method':scope['method'],'step_size_255':scope['step_size_255'],'sensitivity_banks':scope['sensitivity_banks'],
        'control_sha256':plan['control_sha256'],'source_manifest_sha256':scope['runtime_manifest_sha256'],'plan_sha256':op.digest(op.encoded(plan))}
    row=records.wrap_work(u,role=role,plan=w,baseline=baseline,pid=pid,wall_seconds=.003,peak_allocator_bytes=1024)
    row['evidence_origin']=ORIGIN;return row


class Virtual:
    """Private source-cache namespace; it never changes real sys.modules/meta_path."""
    def __init__(self,e):
        self.e=e;self.finder=None;self.sys=NS(**vars(sys));self.sys.modules={};self.sys.meta_path=[]
        self.signal=FakeSignal(e)
        self.thread=NS(get_ident=threading.get_ident,current_thread=threading.current_thread,
            main_thread=threading.main_thread,Timer=lambda s,fn:FakeTimer(e,s,fn))
    def imports(self,name,globals=None,locals=None,fromlist=(),level=0):
        e=self.e
        substitutes={'os':e.os,'sys':self.sys,'signal':self.signal,'time':e.time,'threading':self.thread,
            '_thread':NS(interrupt_main=lambda s:e.events.append(['FAKE-INTERRUPT',int(s)])),
            'platform':NS(node=lambda:'linux10614',system=lambda:'Linux',python_version=lambda:'3.11.17'),
            'subprocess':e.subprocess,'select':NS(select=lambda f,w,x,t:(f,[],[])),
            'resource':NS(RLIMIT_FSIZE=1,setrlimit=lambda *a:e.events.append(['FAKE-RLIMIT',list(a)]))}
        if name in substitutes:return substitutes[name]
        if name.split('.')[0] in ('torch','numpy'):raise AssertionError('Scientific cache entry forbidden in adapter checks.')
        if name.split('.')[0] not in ('models','attacks','datasets'):
            return builtins.__import__(name,globals,locals,fromlist,level)
        if self.finder is None:
            matches=[f for f in self.sys.meta_path if getattr(f,'kind',None)=='pgd-split-pinned-importer-v1']
            if len(matches)!=1:raise ImportError('Fake private cache is not installed.')
            self.finder=matches[0]
        m=self.load(name)
        for child in fromlist:
            if name+'.'+child in self.finder.files:setattr(m,child,self.load(name+'.'+child))
        return m if fromlist else self.load(name.split('.')[0])
    def module(self,data,name,path):
        m=ModuleType(name);m.__file__=str(path)
        m.__dict__['__builtins__']={**vars(builtins),'__import__':self.imports}
        exec(compile(data,m.__file__,'exec'),m.__dict__);return m
    def load(self,name):
        if name in self.sys.modules:return self.sys.modules[name]
        allow={'models','attacks','datasets','models.predictor_pgd_split_bootstrap_v1','models.predictor_pgd_split_coordinator_v1',
            'models.predictor_pgd_split_scratch_v1','models.predictor_pgd_split_collection_v1','models.predictor_pgd_split_watch_v1',
            'models.predictor_pgd_workload_watch_v1','models.predictor_pgd_workload_os_v1','models.scratch_allocation_v2',
            'models.predictor_pgd_split_plan_v1','models.predictor_pgd_workload_runtime_records_v1',
            'models.predictor_pgd_workload_records_v1','models._pgd_workload_entry_v1'}
        if name not in allow:raise ImportError('Only explicit stdlib/cache metadata may execute locally: '+name)
        if '.' in name:self.load(name.rsplit('.',1)[0])
        search=None if '.' not in name else self.sys.modules[name.rsplit('.',1)[0]].__path__
        spec=self.finder.find_spec(name,search)
        m=ModuleType(name);m.__spec__=spec;m.__loader__=self.finder
        m.__package__=name if spec.submodule_search_locations is not None else name.rsplit('.',1)[0]
        m.__dict__['__builtins__']={**vars(builtins),'__import__':self.imports}
        self.sys.modules[name]=m
        self.finder.exec_module(m)
        if '.' in name:
            parent,child=name.rsplit('.',1);setattr(self.sys.modules[parent],child,m)
        if name=='models.predictor_pgd_split_coordinator_v1':m.target_gate=self.e.fake_target
        if name=='models.predictor_pgd_split_scratch_v1':
            self.original_proof=m.Guard.verify_proof
            m.Guard.verify_proof=lambda guard,want:self.e.fake_proof_reader(guard,want)
        if name=='models.scratch_allocation_v2':m.bounded_write_proof=self.e.fake_proof
        if name=='models.predictor_pgd_workload_os_v1':m._compute=self.e.fake_compute
        if name in ('models.predictor_pgd_split_watch_v1','models.predictor_pgd_workload_watch_v1'):
            e=self.e
            class Reader:
                def identity(self,pid):return e.identity(pid)
                def sample(self,pid,birth):return e.sample(pid,birth)
            m.ProcReader=Reader
            m.system_pidfd_api=lambda:NS(open=lambda pid:e.handle_open(pid),dead=e.handle_dead,
                send=e.handle_send,close=lambda fd:e.events.append(['FAKE-HANDLE-CLOSE',fd]))
        return m


class Engine:
    def __init__(self,folder,method='primary',fault=None):
        self.folder=folder;self.method=method;self.fault=fault;self.clock=Clock();self.events=[]
        self.parent=os.getpid();self.pid=self.parent;self.children={};self.current=None;self.pipes={};self.nextfd=900000
        self.built=op.assemble(ROOT,method=method,sensitivity_banks=2 if method=='convergence' else 0)
        self.os=NS(**vars(os));self.time=NS(monotonic=self.clock,sleep=self.pause)
        self.os.getpid=lambda:self.pid;self.os.getppid=lambda:self.parent if self.pid!=self.parent else os.getppid()
        self.os.getsid=lambda x:self.pid;self.os.getpgrp=lambda:self.pid
        self.os.pipe2=self.pipe;self.os.close=self.close;self.os.write=self.write;self.os.read=self.read
        self.os.fstat=lambda fd:NS(st_mode=stat.S_IFIFO) if fd in self.pipes else os.fstat(fd)
        self.os.urandom=lambda n:bytes([len(self.children)+1])*n
        self.os._exit=lambda code:(_ for _ in ()).throw(SystemExit(code))
        self.subprocess=NS(PIPE=-1,DEVNULL=-3,TimeoutExpired=subprocess.TimeoutExpired,
            check_output=self.query,Popen=self.popen)
        self.v=Virtual(self)
        self.root=folder/('vjepa-pgd-workload-v1-'+str(os.getuid())+'-FAKE');self.root.mkdir(mode=0o700)
        self.bootstrap=self.v.module(self.built['sources']['bootstrap.py'],'_checked_bootstrap',self.root/'supervisor/bootstrap.py')
        b=self.built;self.csha=op.digest(b['control']);c=json.loads(b['control'])
        self.auth=op.encoded({'kind':'whole-pgd-supervised-university-exception-v1','approved':True,
            'control_sha256':self.csha,'maximum_new_attempts':1,'user_basis':ORIGIN})
        ib,isha=self.bootstrap.installation(c,self.csha)
        # Scratch imports only through the private finder after source installation.
        self.setup(c,ib,isha)

    def setup(self,c,ib,isha):
        b=self.built
        for n,data in b['uploads'].items():put(self.root/n,data)
        for n,data in (('control.json',b['control']),('authorization.json',self.auth),('installation.json',ib),
            ('configs/base.json',b['policy']['base_text'].encode()),('configs/design.json',b['policy']['design_text'].encode()),
            ('proposal.json',{'kind':ORIGIN,'approved':False}),('runtime-source-manifest.json',{'kind':ORIGIN})):
            put(self.root/n,data)
        self.binding={'root':str(self.root),'root_identity':list(self.bootstrap.dir_id(self.root.lstat())),
            'parent':str(self.root.parent),'parent_identity':list(self.bootstrap.dir_id(self.root.parent.lstat()))}
        self.finder=self.bootstrap.PinnedImporter(self.root,self.binding,c,b['policy']).install();self.v.finder=self.finder
        self.scratch=self.v.load('models.predictor_pgd_split_scratch_v1')
        self.coordinator=self.v.load('models.predictor_pgd_split_coordinator_v1')
        layout=self.scratch.Layout(b['policy']['scratch_contract_text'].encode(),b['policy']['base_text'].encode(),
            b['policy']['design_text'].encode(),ib,isha)
        # Exact generated directory ownership plan; scientific directories remain absent.
        out=self.v.module(b['generated'][op.OUTER],'_cold_outer',self.root/'_source-only-outer')
        for n in out._setup_directory_plan(layout)['setup_directories']:
            if n!='.':(self.root/n).mkdir(mode=0o700,exist_ok=True)
        import tarfile
        cfg=json.loads((ROOT/op.INPUTS).read_bytes());archive=(ROOT/cfg['packet_path']).read_bytes()
        put(self.root/'original-packet.tar.gz',archive)
        with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as a:
            for item in a:put(self.root/'code'/item.name,a.extractfile(item).read())
        put(self.root/'emergency-space.bin',b'R'*(16*2**20))
        self.guard=self.scratch.Guard(self.root,self.binding,layout)
        self.alarm=out.Alarms(clock=self.clock)
        self.owner=self.coordinator.OuterCoordinator(self.root,self.binding,b['control'],self.auth,
            connection_started=100.,clock=self.clock,outer_guard=self.guard,outer_alarm=self.alarm)

    def fake_target(self,root,binding):
        assert str(root)==self.binding['root'] and list(self.bootstrap.dir_id(self.root.lstat()))==binding['root_identity']
        self.events.append(['FAKE-HOST-DEVICE-MOUNT-GATE']);return {'origin':ORIGIN}
    def query(self,cmd,**kw):
        h=self.coordinator.HOST
        if '--query-gpu=' in ' '.join(cmd):return ', '.join([h['gpu'],h['gpu_uuid'],'8192','0',h['driver']])+'\n'
        return ''  # Explicit fake GPU process list, not a hardware observation.
    def pipe(self,flags):
        r,w=self.nextfd,self.nextfd+1;self.nextfd+=2
        self.pipes[r]={'data':b'','peer':w};self.pipes[w]={'data':b'','peer':r};return r,w
    def close(self,fd):
        if fd in self.pipes:self.events.append(['FAKE-PIPE-CLOSE',fd])
        else:os.close(fd)
    def write(self,fd,data):
        if fd in self.pipes:
            self.events.append(['FAKE-RELEASE',self.current.pid]);self.current.released=True
            self.pipes[self.pipes[fd]['peer']]['data']=data;return len(data)
        return os.write(fd,data)
    def read(self,fd,count):
        if fd in self.pipes:return self.pipes[fd]['data']
        return os.read(fd,count)
    def popen(self,command,**kw):
        plan=json.loads(base64.b64decode(command[-2]));pid=950000+len(self.children)
        child=Child(self,pid,plan);self.children[pid]=child;self.current=child
        child.fd=int(command[-1]);self.events.append(['FAKE-Popen',pid,plan['role']])
        assert kw['start_new_session'] and kw['shell'] is False and kw['close_fds'] and kw['pass_fds']==(child.fd,)
        return child
    def identity(self,pid):
        p=self.children[pid]
        return {'pid':pid,'start_ticks':pid+500,'parent_pid':self.parent,'group_pid':pid,'session_pid':pid,'state':'S'}
    def handle_open(self,pid):
        self.events.append(['FAKE-HANDLE-BOUND',pid]);return pid+1000000
    def handle_dead(self,fd):return self.children[fd-1000000].exited
    def handle_send(self,fd,sig):
        child=self.children[fd-1000000];self.events.append(['FAKE-OWNED-SIGNAL',child.pid,int(sig)])
        child.exited=True;child.status=-int(sig)
    def pause(self,seconds):self.clock.now+=seconds
    def sample(self,pid,birth):
        child=self.children[pid]
        if not child.exited:self.execute(child)
        rss=4*2**30+1 if self.fault=='rss' and child.plan['role']=='candidate-unit' else 1024*1024
        row=dict(self.identity(pid),rss_bytes=rss)
        return {'processes':{pid:row},'rss_bytes':rss}
    def execute(self,child):
        assert child.released and ['FAKE-HANDLE-BOUND',child.pid] in self.events
        previous=self.pid;self.pid=child.pid
        cv=Virtual(self)
        bootstrap=cv.module(self.built['sources']['bootstrap.py'],'_child_bootstrap',self.root/'supervisor/bootstrap.py')
        self.active_v=cv
        try:
            bootstrap.wait_release(child.fd,self.parent,child.plan['deadline'])
            child.status=bootstrap.child_entry(op.encoded(child.plan))
        except BaseException as error:
            child.status=124 if isinstance(error,(SystemExit,TimeoutError)) else 1
            put(self.folder/('child-error-'+str(child.pid)+'.json'),{'origin':ORIGIN,'type':type(error).__name__,'error':str(error)})
        finally:
            self.pid=previous;self.active_v=None;child.exited=True
    def fake_proof(self,root,deadline):
        assert self.current.plan['role']=='proof'
        put(Path(root)/'allocation-result-v2.json',{'kind':'scratch-allocation-proof-v2','passed':False,
            'complete':False,'bytes_written':0,'bytes_read_back':0,'preallocator_injected_for_test':True,'origin':ORIGIN})
        return {'origin':ORIGIN}
    def fake_proof_reader(self,guard,want):
        raw=guard.read('allocation-result-v2.json',16384);row=json.loads(raw)
        assert op.digest(raw)==want and row['origin']==ORIGIN and row['passed'] is False and row['bytes_written']==0
        return {'proof_is_simulated':True,'origin':ORIGIN,'actual_bytes_written':0}
    def fake_compute(self,root,plan,scope,timers):
        row=fake_unit(plan,scope,self.pid);start=self.clock.now
        path=Path(root)/'capture/results/workload'/plan['role']/'unit';path.mkdir(mode=0o700)
        put(path/'unit-started.json',{'pid':self.pid,'gradient_budget':len(row['unit']['steps']),
            'settings':row['unit']['metrics']['settings'],'video_id':row['baseline']['video_id'],'origin':ORIGIN})
        for step in row['unit']['steps']:
            with timers.objective(min(plan['deadline'],self.clock()+120),hard_deadline=plan['deadline']):
                if self.fault=='objective' and plan['role']=='candidate-unit':self.clock.now+=121
                else:self.clock.now+=.002
                put(path/f"step-{step['index']:03}-started.json",{'index':step['index'],'pid':self.pid,
                    'pixels':step['pixels'],'factory':row['unit']['route'],'origin':ORIGIN})
                put(path/f"step-{step['index']:03}-complete.json",step)
        row['wall_seconds']=self.clock.now-start
        if self.fault=='replay' and plan['role']=='candidate-replay':
            row['unit']['steps'][0]['gradient']='0'*64
            # Preserve immutable work files and let the final replay join reject the changed return.
            f=path/'step-000-complete.json';f.unlink();put(f,row['unit']['steps'][0])
        put(path/'unit-complete.json',row['unit'])
        if self.fault=='source' and plan['role']=='candidate-unit':row['source_manifest_sha256']='0'*64
        return row


def test_cold_graph_and_scientific_bytes_unchanged(case):
    b=op.assemble(ROOT,method='primary')
    assert len(b['uploads'])==55 and len(b['sources'])==9 and len(b['workers'])==46
    assert sum(map(len,b['uploads'].values()))<=2*2**20
    m=json.loads((ROOT/'runs/pretrained/diagnostic/pgd-workload-runtime-v1-20261123/SOURCE-MANIFEST-v2.json').read_text())
    for n,d in m['files'].items():
        raw=b['sources']['workload_watch_v1.py'] if n.startswith('supervisor/') else b['workers'][n]
        assert op.digest(raw)==d['sha256'] and len(raw)==d['bytes']
    put(case/'derivation.json',b['derivation'])


@pytest.mark.parametrize('method',['primary','convergence'])
def test_concrete_driver_child_timers_completion_and_collection(case,method):
    before=(dict(sys.modules),list(sys.meta_path));e=Engine(case,method)
    result=e.owner.run_work();assert result['completed']==list(op.ROLES) and result['complete'] is False
    e.owner.driver.assert_idle();e.owner.driver.outer_journal.close()
    codec=e.v.load('models.predictor_pgd_split_collection_v1')
    collector=codec.Collector(e.guard,e.built['control'],e.auth,connection_started=100.,clock=e.clock)
    collector.prepare();wire=io.BytesIO();terminal=collector.stream(wire);data=wire.getvalue()
    put(case/'wire.bin',data)
    received=codec.receive(io.BytesIO(data),case/'received',e.built['control'],e.auth,
        window=codec.Window(100.,e.clock(),clock=e.clock))
    assert received['transport_verified'] and not received['scientific_qualified'] and not received['training_admitted']
    put(case/'transport.json',received);put(case/'events.json',e.events)
    project=lambda values:{k:v for k,v in values.items() if k.split('.')[0] in ('models','attacks','datasets')}
    assert project(before[0])==project(sys.modules) and before[1]==list(sys.meta_path)
    additions=set(sys.modules)-set(before[0])
    assert all(n.split('.')[0] in sys.stdlib_module_names for n in additions)
    put(case/'namespace-observation.json',{'added_standard_library_modules':sorted(additions),
        'project_module_identities_unchanged':True,'global_meta_path_unchanged':True,'origin':ORIGIN})
    assert len([x for x in e.events if x[0]=='FAKE-RELEASE'])==4
    assert len([x for x in e.events if x[0]=='FAKE-HANDLE-CLOSE'])==4
    # Exercise the unchanged real proof reader against the saved false/zero-byte fixture.
    with pytest.raises(ValueError):e.v.original_proof(e.guard,op.digest(e.guard.read('allocation-result-v2.json',16384)))
    assert terminal['stream_complete'] is True and json.loads((e.root/'session.json').read_bytes())['complete'] is False


@pytest.mark.parametrize('fault',['rss','objective','source','replay'])
def test_concrete_failure_stops_roles_and_preserves_original_results(case,fault):
    e=Engine(case,fault=fault)
    with pytest.raises((ValueError,RuntimeError,TimeoutError,MemoryError)):e.owner.run_work()
    e.owner.driver.assert_idle();e.owner.driver.outer_journal.close()
    assert e.owner.state['phase']=='failed'
    if fault!='replay':assert 'candidate-replay' not in [x[2] for x in e.events if x[0]=='FAKE-Popen']
    put(case/'events.json',e.events)


def test_unarmed_client_and_CLI_refuse_before_timer_or_project_dispatch(case):
    from models import predictor_pgd_workload_client_v1 as client
    with pytest.raises(ValueError,match='Unarmed'):client.run(case/'missing')
    raw=(ROOT/'scripts/run_predictor_pgd_workload_v1.py').read_bytes()
    ns={'__name__':'_isolated_CLI_test','__file__':str(ROOT/'scripts/run_predictor_pgd_workload_v1.py')}
    exec(compile(raw,ns['__file__'],'exec'),ns)
    with pytest.raises(SystemExit,match='Unarmed'):ns['main']([str(case/'missing')])


def test_historical_component_grant_and_scope_are_refused(case):
    from models import predictor_pgd_workload_bootstrap_v1 as boot
    b=op.assemble(ROOT,method='primary');_,_,sha=boot.control(b['control'])
    with pytest.raises(ValueError):
        boot.authorization(op.encoded({'kind':'pgd-candidate-supervised-university-exception-v1',
            'approved':True,'control_sha256':sha,'maximum_new_attempts':1,'user_basis':ORIGIN}),sha)


def test_runtime_compute_argument_binding_without_scientific_import(case):
    e=Engine(case);m=e.v.load('models.predictor_pgd_workload_os_v1')
    calls=[]
    entry=NS(EXECUTION_BINDING=None,entry=lambda args,**kw:(calls.append((args,kw)) or ({'fake':True},None)))
    package=e.v.sys.modules['models'];package._pgd_workload_entry_v1=entry
    e.v.sys.modules['models._pgd_workload_entry_v1']=entry
    scope=e.built['static']['scope']
    plan={'role':'candidate-unit','control_sha256':e.csha,'deadline':e.clock()+700,'source_hashes':e.built['static']['source_hashes']}
    timers=NS(objective=lambda *a,**kw:None)
    # Call the source's original _compute, not the explicit fake numerical port.
    tree=ast.parse(e.built['sources']['workload_os_v1.py']);node=next(n for n in tree.body if getattr(n,'name',None)=='_compute')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<CHECKED-ARGUMENT-BRIDGE>','exec'),m.__dict__)
    assert m._compute(e.root,plan,scope,timers)=={'fake':True}
    args,kw=calls[0]
    assert args.packet==e.root/'code' and args.output.endswith('/candidate-unit/unit')
    assert args.plan_sha256==op.digest(op.encoded(plan)) and entry.EXECUTION_BINDING['role']=='candidate-unit'
    assert kw['objective_timer'] is timers.objective


@pytest.mark.parametrize('remaining,interval,handler',[(0.,0.,None),(10.,1.,None),(1000.,0.,None),(10.,0.,'foreign')])
def test_worker_timer_refuses_foreign_or_missing_release_clock(case,remaining,interval,handler):
    e=Engine(case);m=e.v.load('models.predictor_pgd_workload_os_v1');e.pid=950001
    e.v.signal.end=None if not remaining else e.clock.now+remaining
    if interval:m.signal.getitimer=lambda s:(remaining,interval)
    if handler:m.signal.handlers[signal.SIGALRM]=handler
    with pytest.raises(ValueError):m.UnitTimers(e.clock()+100)


def test_absolute_timer_restore_and_expired_objective_cannot_extend(case):
    e=Engine(case);m=e.v.load('models.predictor_pgd_workload_os_v1');e.pid=950001
    end=e.clock()+700;e.v.signal.setitimer(signal.ITIMER_REAL,end-e.clock())
    with pytest.raises(m.Deadline):
        with m.UnitTimers(end) as timers:
            first=e.clock()+120
            with timers.objective(first,hard_deadline=end):e.clock.now+=1
            assert abs(e.v.signal.end-end)<.01
            with pytest.raises(m.Deadline):
                with timers.objective(e.clock()+1,hard_deadline=end):e.clock.now+=2
            assert e.v.signal.end<end
            with pytest.raises(ValueError):timers.objective(e.clock()+1,hard_deadline=end).__enter__()
    put(case/'events.json',e.events)


def test_private_cache_rejects_science_reload_and_wrong_source(case):
    e=Engine(case)
    with pytest.raises(ImportError):e.v.load('models.predictor_pgd_workload_target_v1')
    with pytest.raises(ImportError):e.finder.find_spec('models.predictor_pgd_split_coordinator_v1',target=object())
    assert 'datasets' in e.finder.files and len(e.finder.files)==55
    module=e.v.sys.modules['models.predictor_pgd_split_coordinator_v1'];module.__loader__=None
    with pytest.raises(ValueError):e.bootstrap.require_namespace(e.root,e.binding,e.csha)


def test_client_causal_clock_and_original_host_guard_still_refuse(case):
    from models import predictor_pgd_workload_client_v1 as client
    from models import predictor_pgd_workload_outer_v1 as outer
    with pytest.raises(ValueError):outer.preflight_parent()  # Local Mac refuses before /proc/GPU/write.
    clock=Clock();window=client.ReceiverClock(100.,101.,1000.,2000.,clock=lambda:102.)
    assert window.deadline==1101.
    with pytest.raises(ValueError):client.ReceiverClock(100.,99.,1000.,2000.,clock=lambda:102.)


def fake_package(folder,built):
    from models import predictor_pgd_workload_outer_v1 as outer
    p=folder/'FAKE-package';p.mkdir(mode=0o700)
    c={'kind':'whole-pgd-outer-control-v1','supervised_control_text':built['control'].decode(),
        'supervised_control_sha256':op.digest(built['control']),'outer_source_sha256':op.digest(built['generated'][op.OUTER]),
        'local_source_sha256':{n:op.digest((ROOT/n).read_bytes()) for n in outer.LOCAL_SOURCES},
        'runtime_files':built['static']['runtime_files'],'repository':str(ROOT),'local_handoff':str(p),
        'packet_path':outer.PACKET_PATH,'known_hosts_text':outer.KNOWN_HOSTS,
        'known_hosts_sha256':op.digest(outer.KNOWN_HOSTS.encode()),'expected_ssh_fingerprint':outer.FINGERPRINT,
        'financial_state_unchanged':True,'authorized_at_preparation':False,'target_helper_reviewed':False}
    raw=op.encoded(c);auth=outer.authorization(c,op.digest(raw),ORIGIN)
    put(p/'CONTROL-WORKLOAD.json',raw);put(p/'AUTHORIZATION.json',auth);put(p/'known-hosts',outer.KNOWN_HOSTS.encode())
    put(p/'SOURCE-PAYLOAD.json',{n:base64.b64encode(v).decode() for n,v in built['uploads'].items()})
    return p


def test_concrete_client_preflight_fake_grant_has_one_original_archive(case):
    from models import predictor_pgd_workload_client_v1 as client
    b=op.assemble(ROOT,method='primary');p=fake_package(case,b)
    result=client.check_local(p)
    assert len(result[-1])==1 and result[-1][0]==ROOT/json.loads((ROOT/op.INPUTS).read_bytes())['packet_path']
    assert len(result[7])==55 and result[6]['full_model_gradients']==120
    assert not (p/'ATTEMPT.json').exists() and not (p/'attended-session').exists()


def test_source_graph_is_closed_in_the_executable_cache(case):
    from models.predictor_pgd_workload_runtime_v1 import imports
    b=op.assemble(ROOT,method='convergence')
    mapping={n[8:-3].replace('/','.').removesuffix('.__init__') for n in b['uploads'] if n.startswith('overlay/')}
    slots=op.assigned(b['sources']['bootstrap.py'],'SUPERVISORS');mapping.update(slots)
    for name,raw in b['uploads'].items():
        deps,_=imports(raw)
        assert deps<=mapping,(name,deps-mapping)
    assert len(mapping)==55 and 'models.predictor_pgd_split_plan_v1' in mapping


def test_original_packet_receive_extract_and_trailing_refusal(case):
    from models import predictor_pgd_workload_outer_v1 as outer
    cfg=json.loads((ROOT/op.INPUTS).read_bytes());archive=(ROOT/cfg['packet_path']).read_bytes()
    root=case/'packet';root.mkdir(mode=0o700)
    outer.receive_packet(io.BytesIO(archive),root,cfg['packet']['archive'],lambda:None)
    for name in cfg['packet']['files']:private_parents((root/'code'/name).parent)
    outer.extract_packet(root,cfg['packet']['files'],lambda:None)
    assert len(list((root/'code').rglob('*')))>=99
    for n,d in cfg['packet']['files'].items():assert op.digest((root/'code'/n).read_bytes())==d['sha256']
    bad=case/'trailing';bad.mkdir(mode=0o700)
    with pytest.raises(ValueError,match='Unexpected bytes'):
        outer.receive_packet(io.BytesIO(archive+b'X'),bad,cfg['packet']['archive'],lambda:None)


def test_worker_hard_alarm_uses_objective_expiry_and_preserves_frozen_Alarms_AST(case):
    e=Engine(case);m=e.v.load('models.predictor_pgd_workload_os_v1');e.pid=950001
    end=e.clock()+700;e.v.signal.setitimer(signal.ITIMER_REAL,end-e.clock())
    with m.UnitTimers(end) as timer:
        objective=e.clock()+100
        with timer.objective(objective,hard_deadline=end):
            assert abs(e.v.signal.end-objective)<.01
        assert abs(e.v.signal.end-end)<.01
    old=ast.parse((ROOT/'models/predictor_pgd_candidate_outer_v1.py').read_bytes())
    new=ast.parse(e.built['sources']['workload_os_v1.py'])
    a=next(n for n in old.body if getattr(n,'name',None)=='Alarms')
    b=next(n for n in new.body if getattr(n,'name',None)=='Alarms')
    assert ast.dump(a)==ast.dump(b)


def test_saved_stream_partial_journal_and_client_root_binding_refuse(case):
    import struct
    from models import predictor_pgd_workload_client_v1 as client
    from models import predictor_pgd_workload_collection_v1 as collection
    source=ROOT/'runs/pretrained/diagnostic/pgd-workload-operator-v1-20261123/test-evidence-v4/test_concrete_driver_child_timers_completion_and_collection[primary]'
    raw=(source/'wire.bin').read_bytes();scratch=next(p for p in source.iterdir() if p.name.startswith('vjepa-pgd-workload-v1-'))
    control=(scratch/'control.json').read_bytes();authorization=(scratch/'authorization.json').read_bytes()
    start=len(collection.MAGIC);size=struct.unpack('!I',raw[start:start+4])[0]
    base=json.loads(raw[start+4:start+4+size])['base']
    moment=base['collection_started']+.001
    window=lambda:collection.Window(100.,base['collection_started'],clock=lambda:moment)
    # A valid local fixture stream still cannot impersonate the university's /var/tmp binding.
    with pytest.raises(ValueError,match='another root or clock'):
        client.bound_collection_source(io.BytesIO(raw),window(),
            {'scratch':str(scratch),'connection_started':100.},
            {'collection_started':base['collection_started'],'collection_deadline':base['collection_deadline']},
            {'supervised_control_text':control.decode()},authorization)
    cut=raw[:len(raw)//2];put(case/'truncated-wire.bin',cut)
    with pytest.raises((ValueError,EOFError)):
        collection.receive(io.BytesIO(cut),case/'partial',control,authorization,window=window())
    receipt=json.loads((case/'partial-transport/PARTIAL-RECEIPT.json').read_bytes())
    assert receipt['transport_verified'] is False and receipt['stream_complete'] is False



