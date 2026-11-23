"""New candidate operator integration, with explicit fake OS/compute/proof ports.

Local files contain real pinned opaque source/reference bytes. No values decode;
no subprocess, proc/PIDFD, signal, timer, GPU, proof or network API may run.
"""
import ast
import base64
import builtins
import collections
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import threading
from argparse import Namespace
from types import ModuleType, SimpleNamespace as NS
import pytest
from models import predictor_pgd_candidate_operator_v1 as build
from models import predictor_pgd_candidate_outer_v1 as outer
from models import predictor_pgd_candidate_client_v1 as client
from models import predictor_pgd_candidate_completion_v1 as adapter
from models import predictor_pgd_component_capsule_v1 as capsule
from models import predictor_pgd_split_backend_v4 as backend
from models import predictor_pgd_split_lineage_v1 as lineage
ROOT = Path(__file__).resolve().parents[1]
ORIGIN = 'EXPLICIT-FAKE-CANDIDATE-OPERATOR; NO GRANT OR EXECUTION ADMISSION'
PREFIX = 'models/_pgd_component_'


@pytest.fixture(scope='module', autouse=True)
def forbid_live_APIs():
    patch = pytest.MonkeyPatch()
    def forbidden(*args, **kwargs):
        pytest.fail('No actual process/proof/GPU/timer/signal/network/compute port allowed.')
    for owner, name in [(subprocess, 'Popen'), (subprocess, 'check_output'), (os, 'system'),
                        (os, 'fork'), (os, 'posix_spawn'), (os, 'kill'), (os, 'killpg'),
                        (os, 'pidfd_open'), (os, 'pipe2'), (os, 'posix_fallocate'),
                        (signal, 'signal'), (signal, 'alarm'), (signal, 'setitimer'),
                        (signal, 'pidfd_send_signal'), (socket, 'socket'), (threading.Thread, 'start')]:
        if hasattr(owner, name):
            patch.setattr(owner, name, forbidden)
    original = builtins.__import__
    def import_guard(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'numpy'):
            forbidden()
        return original(name, *args, **kwargs)
    patch.setattr(builtins, '__import__', import_guard)
    yield
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    patch.undo()


@pytest.fixture(scope='module')
def bundle():
    return build.assemble(ROOT)


@pytest.fixture(scope='module')
def support(bundle, tmp_path_factory):
    boot = module(bundle['sources'][build.NAMES[0]], 'fake_candidate_boot')
    scratch = module(bundle['sources'][build.NAMES[1]], 'fake_candidate_scratch')
    scratch.os = NS(**vars(os))
    scratch.os.fstatvfs = lambda fd: NS(f_bavail=32*2**30//4096, f_frsize=4096, f_favail=10000)
    watch = module(bundle['sources'][build.NAMES[4]], 'fake_candidate_watch')
    path = tmp_path_factory.mktemp('unchanged-monitor-source') / 'watch.py'
    put(path, bundle['sources'][build.NAMES[4]]); watch.__file__ = str(path)
    imports = NS(predictor_pgd_split_bootstrap_v1=boot, predictor_pgd_split_scratch_v1=scratch,
                 predictor_pgd_split_watch_v1=watch)
    coord = module(bundle['sources'][build.NAMES[2]], 'fake_candidate_coordinator', imports)
    return NS(boot=boot, scratch=scratch, watch=watch, coord=coord)


def fake_authorization(raw):
    return build.enc({'kind':'pgd-candidate-supervised-university-exception-v1', 'approved':True,
                      'control_sha256':build.digest(raw), 'maximum_new_attempts':1, 'user_basis':ORIGIN})


@pytest.fixture(scope='module')
def baseline(bundle, support, tmp_path_factory):
    root = tmp_path_factory.mktemp('candidate-byte-baseline'); root.chmod(0o700)
    contract, files = build.inputs(ROOT)
    for name, path in contract['source'].items():
        put(root/'source'/name, files[path])
    spec = json.loads(files[contract['pins']['reference_manifest']])
    for name, path in contract['reference'].items():
        put(root/'reference'/name, build.enc(spec) if name=='MANIFEST.json' else files[path])
    for name, data in bundle['uploads'].items():
        put(root/name, data)
    for name, alias in [('component-inputs.tar','capsule_archive'), ('native-reference.tar','reference_archive')]:
        put(root/name, files[contract['pins'][alias]])
    for name, data in [('configs/base.json',bundle['static']['base_text'].encode()),
                       ('configs/design.json',bundle['static']['design_text'].encode()),
                       ('control.json',bundle['control']), ('authorization.json',fake_authorization(bundle['control'])),
                       ('proposal.json',{'origin':ORIGIN,'actual_approval':False}),
                       ('runtime-source-manifest.json',{'files':list(bundle['uploads']),'origin':ORIGIN})]:
        put(root/name, data)
    b,p,csha = support.boot.control(bundle['control']); installation,isha = support.boot.installation(b,csha)
    put(root/'installation.json',installation)
    layout = support.scratch.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),installation,isha)
    for name in outer._setup_directory_plan(layout)['setup_directories']:
        if name!='.': (root/name).mkdir(mode=0o700,exist_ok=True)
    put(root/'emergency-space.bin',b'\xa7'*(16*2**20))
    return root

def module(raw,name,imports=None):
    m=ModuleType(name);m.__file__='<EXPLICIT-LOCAL-OPERATOR-'+name+'>'
    if imports:
        def load(name,globals=None,locals=None,fromlist=(),level=0):
            if name=='models':return imports
            return builtins.__import__(name,globals,locals,fromlist,level)
        m.__dict__['__builtins__']={**vars(builtins),'__import__':load}
    exec(compile(raw,m.__file__,'exec'),m.__dict__);return m

def put(p,data):
    missing=[];parent=p.parent
    while not parent.exists():missing.append(parent);parent=parent.parent
    for q in reversed(missing):q.mkdir(mode=0o700)
    data=data if type(data) is bytes else build.enc(data)
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(data)

def guard_for(support,root):
    b,p,csha=support.boot.control((root/'control.json').read_bytes());installation,isha=support.boot.installation(b,csha)
    binding=support.scratch.identity_binding(root)
    layout=support.scratch.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),installation,isha)
    return support.scratch.Guard(root,binding,layout),binding,b,p,csha

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

def namespace(data,provided):
    tree=ast.parse(data);tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    ns={'__name__':'EXPLICIT_FAKE_CANDIDATE_COMPLETION',**provided}
    exec(compile(ast.fix_missing_locations(tree),'<EXPLICIT-FAKE-PORTS-AND-OPAQUE-BYTES>','exec'),ns)
    return ns

class Value:
    def __init__(self,label):self.label=label
    def detach(self):return self

class Route:
    def __init__(self,h,role,values,incoming,binding):self.h=h;self.role=role;self.values=values;self.incoming=incoming;self.binding=binding;self.closed=False
    def __enter__(self):self.h.calls['route_enter']+=1;return self
    def __exit__(self,*exc):self.h.calls['route_exit']+=1;self.closed=True;return False
    def attention(self):self.h.calls['forward']+=1;return Value('output')
    def gradient(self,*a,**k):
        assert k=={'create_graph':False,'retain_graph':False};self.h.calls['outer_VJP']+=1;self.h.calls['inner_math_VJP']+=1
        return tuple(Value(n) for n in ['dq','dk','dv'])
    def snapshot(self):return {'route_complete':self.closed,'role':self.role,'binding':self.binding,'scientific_qualified':False,'origin':ORIGIN}

class Profile:
    def __init__(self,h):self.h=h;self.stopped=False
    def __enter__(self):self.h.calls['profile_start']+=1;return self
    def __exit__(self,*exc):self.h.calls['profile_stop']+=1;self.stopped=True;return False

def fake_materialized_trace():
    rows=[];origin=1_700_000_000_000_000_000
    def event(name,start,end,thread=7,scope=0,seq=-1,fwd=0,annotation=False):
        i=len(rows);row={'index':i,'name':name,'device':'cpu','device_index':17,'resource_id':17,'thread':thread,'end_thread':thread,'fwd_thread':fwd,'sequence_nr':seq,'scope':scope,'start_ns':origin+start*1000,'end_ns':origin+end*1000,'correlation_id':i+100,'linked_correlation_id':0,'user_annotation':annotation,'is_async':False,'legacy_cuda_elapsed_us':-1};rows.append(row);return row
    def kernel(name,host):
        row=copy.deepcopy(host);row.update(index=len(rows),name=name,device='cuda',device_index=0,resource_id=5,thread=1,end_thread=1,fwd_thread=0,sequence_nr=-1,scope=0,correlation_id=len(rows)+1000,linked_correlation_id=host['correlation_id'],start_ns=origin+2000*1000,end_ns=origin+2010*1000);rows.append(row)
    def node(name,seq,start,end,thread,fwd,copied):
        event(backend.ENGINE+name,start,end,thread=thread,seq=seq if copied else -1,fwd=fwd if copied else 0)
        event(name,start+10,end-10,thread=thread,scope=1,seq=seq,fwd=fwd)
    event(backend.FORWARD,0,200,scope=7,annotation=True);event(backend.CUSTOM,10,190,seq=100)
    f=event(backend.EFFICIENT,20,180,seq=101);kernel('fmha_cutlassF_f32_EXPLICIT_FAKE',f)
    event(backend.BACKWARD,400,1800,scope=7,annotation=True)
    node(backend.CUSTOM_BACKWARD,100,410,1790,11,7,False);event(backend.MATH,600,1000,thread=11,seq=200)
    for seq,start,end in [(201,650,740),(203,850,980)]:
        event('aten::matmul',start,end,thread=11,seq=seq);b=event('aten::bmm',start+10,end-10,thread=11,seq=seq);kernel('sgemm_EXPLICIT_FAKE',b)
    event('aten::softmax',750,840,thread=11,seq=202);b=event('aten::_softmax',760,830,thread=11,seq=202);kernel('softmax_warp_forward<EXPLICIT_FAKE>',b)
    for name,seq,start,end,op,kn in [('BmmBackward0',203,1100,1200,'aten::bmm','sgemm_EXPLICIT_FAKE'),('SoftmaxBackward0',202,1230,1340,'aten::_softmax_backward_data','softmax_warp_backward<EXPLICIT_FAKE>'),('BmmBackward0',201,1400,1590,'aten::bmm','sgemm_EXPLICIT_FAKE')]:
        node(name,seq,start,end,12,11,True);b=event(op,start+20,end-20,thread=12);kernel(kn,b)
    return {'kind':backend.frozen.KIND,'source_api':backend.frozen.API,'torch_version':backend.RUNTIME,'trace_start_ns':origin,'records':rows}

class Harness:
    def __init__(self,built,references,result_root,fault=None,pid=None,clock=None):
        files,_,_=built;self.files=files;self.calls=collections.Counter();self.clock=clock or Clock();self.fault=fault;self.result_root=result_root;self.result_root.mkdir(mode=0o700,exist_ok=True)
        source_root,reference_root=references;namespace_os=NS(**vars(os));namespace_os.getpid=lambda:os.getpid() if pid is None else pid;common={'hashlib':hashlib,'json':json,'math':math,'os':namespace_os,'Path':Path,'stat':stat,'threading':threading}
        plan=NS(**namespace(files['models/predictor_pgd_split_plan_v1.py'],{**common,'PurePosixPath':PurePosixPath}))
        store=NS(**namespace(files[adapter.STORE],{**common,'plan':plan}))
        helpers=ast.parse(files['models/predictor_eot8_record_v2.py']);helpers.body=[n for n in helpers.body if isinstance(n,ast.FunctionDef) and n.name in ('_json','_parse')];codec=NS(**namespace(ast.unparse(helpers),{'json':json}))
        self.io=namespace(files[PREFIX+'io_v1.py'],{**common,'original_json':codec,'Inventory':store.Inventory,'Store':store.Store,'identity':store.identity})
        boundary=NS(**namespace(files['models/_pgd_split_boundary_v1.py'],{**common,'Counter':collections.Counter}))
        self.source=namespace(files[PREFIX+'source_v1.py'],{**common,'copy':copy,'capsule':capsule,'boundary':boundary})
        self.reference=namespace(files[adapter.REF],{**common,'ast':ast,'io':__import__('io'),'tarfile':tarfile,'capsule':capsule})
        self.accuracy=namespace(files[adapter.ACCURACY],{**common,'ast':ast,'capsule':capsule,'reference':NS(**self.reference)})
        if fault=='reference':
            base=self.reference['Reference']
            class BadReference(base):
                def describe(self,binding):
                    value=super().describe(binding);value['native_worker_sha256']='0'*64;return value
            self.reference['Reference']=BadReference
        def fake_accuracy(binding,outputs,folder,source):
            self.calls['fake_accuracy']+=1
            description=self.reference['expected_description'](binding)
            checks={n:{'passed':True,'failed_elements':0,'shape':[1,12,2048,32],'max_absolute_error':0.,'max_tolerance_ratio':0.,'relative_l2_error':0.,**self.accuracy['BOUNDS'][n]} for n in ['output','dq','dk','dv']}
            value={'kind':self.accuracy['KIND'],'binding':dict(binding),'reference':description,'outputs':outputs,'checks':checks,'bounds':self.accuracy['BOUNDS'],'forward_logical_bytes_exact':True,'canonical_FP64_accuracy_qualified':False,'worker_completion_or_scientific_admission':False}
            if fault=='accuracy':value['checks']['dq'].update(passed=False,failed_elements=1,max_tolerance_ratio=2.)
            if fault=='forward':value['forward_logical_bytes_exact']=False
            if fault=='deadline':self.clock.value=1001.
            return value
        self.accuracy['check_values']=fake_accuracy
        self.job=namespace(files[adapter.JOB],{**common,'ExitStack':contextlib.ExitStack,'time':NS(monotonic=self.clock),'backend':backend,'lineage':lineage,'continuation':NS(**self.source),'reference':NS(**self.reference),'accuracy':NS(**self.accuracy)})
        cap=json.loads((source_root/'boundary/boundary.json').read_bytes());self.flags=cap['selected_call']['actual_call']['entry_flags']
        expected=json.loads((source_root/'provenance/run-status.json').read_bytes())['baseline']['runtime'];expected={k:v for k,v in expected.items() if k!='model_execution'}
        def initialize(device):
            self.calls['startup']+=1
            if fault=='startup':raise ValueError('EXPLICIT-FAKE startup failure')
            return {'kind':'native-SDPA-component-startup-v2','pid':namespace_os.getpid(),'configure_calls':1,'cuda_initialized_before':False,'cuda_initialized_after':True,'allocator_cap_bytes':7*2**30}
        def operands(canonical,cap):
            self.calls['fake_operands']+=1;canonical.check();layout=lineage.operands(cap);metadata={}
            for n,w in layout['operands'].items():metadata[n]={'shape':w['shape'],'stride':w['stride'],'sha256':w['logical_sha256'],'dtype':'torch.float32','device':'cuda:0','storage_offset_elements':w['storage_offset_elements'],'storage_bytes':w['storage_bytes'],'alias_group':w['alias_group'],'requires_grad':w['requires_grad'],'unknown_padding_filled_zero':True,'native_physical_alignment_identity_claimed':False}
            return tuple(Value(n) for n in ['q','k','v']),Value('grad_output'),metadata
        provider=ast.parse(files[PREFIX+'providers_v1.py']);provider.body=[n for n in provider.body if isinstance(n,ast.FunctionDef) and n.name=='compute_once']
        self.torch=NS(set_num_threads=lambda n:self.calls.update(['threads']),get_num_threads=lambda:4,cuda=NS(synchronize=lambda:self.calls.update(['cuda_sync']),max_memory_allocated=lambda:7*2**30+1 if fault=='allocator' else 128*2**20),profiler=NS(record_function=lambda n:contextlib.nullcontext()))
        compute=namespace(ast.unparse(provider),{'torch':self.torch})['compute_once']
        def raw(profile):
            assert profile.stopped;self.calls['fake_raw']+=1;trace=fake_materialized_trace()
            if fault=='backend':next(r for r in trace['records'] if r['device']=='cuda')['linked_correlation_id']=0
            return adapter.encoded(trace)
        def job(*a,**k):k['evidence_origin']=ORIGIN;return self.job['ComponentJob'](*a,**k)
        self.entry=namespace(files[adapter.ENTRY],{**common,'Namespace':Namespace,'platform':NS(node=lambda:'EXPLICIT-FAKE-HOST'),'SimpleNamespace':NS,'torch':self.torch,'io':NS(**self.io),'source':NS(**self.source),'reference':NS(**self.reference),'accuracy':NS(**self.accuracy),'provider':NS(prepare_operands=operands,make_profile=lambda:Profile(self),compute_once=compute,raw_bytes=raw,serialize_return=lambda value:('EXPLICIT-OPAQUE-FAKE-'+value.label).encode(),flags=lambda:{} if fault=='restoration' and self.calls['forward'] else self.flags),'contract':NS(runtime_contract=lambda device:copy.deepcopy(expected)),'startup':NS(initialize=initialize,validate=lambda *a,**k:self.calls.update(['startup_validate'])),'routes':NS(ComponentRoute=lambda role,values,incoming,*,binding:Route(self,role,values,incoming,binding)),'ComponentJob':job})
        self.entry['EXECUTION_BINDING']={'kind':'candidate-only-reviewed-execution-binding-v1','control_sha256':'e'*64,'capsule_root':str(source_root),'result_root':str(result_root),'reference_root':str(reference_root),'host':'EXPLICIT-FAKE-HOST','capsule_manifest_sha256':self.source['MANIFEST_SHA'],'reference_manifest_sha256':self.reference['SPEC_SHA']}
        self.args=Namespace(control_sha256='e'*64,capsule_root=str(source_root),result_root=str(result_root),reference_root=str(reference_root),role='forward-preserving',deadline=1000.)
    def run(self):return self.entry['component_entry'](self.args)


def launcher(support, bundle, root, monkeypatch, *, fault=None, work=None):
    coord=support.coord; clock=Clock(); events=[]; holder={}
    class Reader:
        def identity(self,pid):
            events.append(('identity',pid))
            return {'pid':pid,'parent_pid':os.getpid(),'session_pid':pid,'group_pid':pid,
                    'start_ticks':717171,'rss_bytes':128*2**20,'children':[]}
        def sample(self,pid,birth):
            row=self.identity(pid); row['rss_bytes']=5*2**30 if fault=='rss' else 128*2**20
            if fault=='deadline': clock.value=221.
            if fault not in ('rss','deadline'): holder['child'].exited=True
            return {'processes':{pid:row},'rss_bytes':row['rss_bytes']}
    def lease(pid,birth,parent,session,reader):
        events.append(('lease-bound',pid))
        if fault=='lease': raise ValueError('EXPLICIT-FAKE owned handle refusal.')
        value=Lease(holder['child'],birth,events); holder['lease']=value; return value
    def popen(command,**kwargs):
        assert kwargs['shell'] is False and kwargs['start_new_session'] is True
        assert kwargs['pass_fds']==(9001,) and kwargs['umask']==0o077
        assert command[1:5]==['-I','-B','-u','-c'] and command[5]==coord.CHILD_DRIVER
        plan=json.loads(base64.b64decode(command[-2])); assert plan['role'] in build.ROLES
        events.append(('Popen-FAKE',plan['role'])); holder['plan']=plan
        child=Child(818180+sum(e[0]=='Popen-FAKE' for e in events)); holder['child']=child
        os.write(kwargs['stdout'],b'EXPLICIT-FAKE: no process was started.\n'); return child
    def close(fd):
        if fd in (9001,9002):
            events.append(('pipe-closed',fd))
            if fd==9002 and not holder.get('released') and 'child' in holder: holder['child'].exited=True
        else: os.close(fd)
    def release(fd,data):
        assert fd==9002 and data==b'G'; events.append(('release',holder['plan']['role']))
        holder['released']=True
        if work is not None: work(holder['plan'],holder['child'],clock)
        return 1
    proxy=NS(**vars(os)); proxy.pipe2=lambda flags:(9001,9002); proxy.close=close; proxy.write=release
    monkeypatch.setattr(coord,'os',proxy)
    monkeypatch.setattr(coord,'subprocess',NS(Popen=popen,DEVNULL=-3,TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(support.watch,'ProcReader',Reader); monkeypatch.setattr(support.watch,'PidfdLease',lease)
    monkeypatch.setattr(support.watch,'time',NS(monotonic=clock,sleep=clock.pause))
    fn=next(n for n in ast.parse(bundle['sources'][build.NAMES[4]]).body if isinstance(n,ast.FunctionDef) and n.name=='watch')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<UNCHANGED-WATCH-WITH-EXPLICIT-FAKE-CLOCK>', 'exec'),support.watch.__dict__)
    owner=NS(root=root,clock=clock,work_deadline=2500.,c=json.loads(bundle['control']),outer_guard=None,
             outer_alarm=NS(arm=lambda deadline,**kw:events.append(('alarm',deadline,kw))))
    driver=coord.OuterDriver(owner)
    def gate(role,pid=None):
        events.append(('guard-FAKE',pid))
        if fault=='foreign' and pid is not None: raise ValueError('EXPLICIT-FAKE foreign GPU process.')
    driver.outer_journal=NS(checkpoint=gate)
    return driver,clock,events,holder


def make_plan(support,bundle,root,role='forward-preserving'):
    guard,binding,c,p,csha=guard_for(support,root)
    current={'role':role,'nonce':'1'*64,'launch_started':100.,'deadline':150. if role=='proof' else 220.,
             'source_hashes':{} if role=='proof' else dict(bundle['source_hashes'])}
    state={'kind':'pgd-candidate-parent-session-v1','control_sha256':csha,'parent_pid':os.getpid(),
           'connection_started':100.,'work_deadline':2500.,'connection_deadline':3100.,'current':current,
           'completed':list(build.ROLES[:build.ROLES.index(role)]),'phase':'running','complete':False}
    put(root/'session.json',state)
    return dict(current,kind='pgd-candidate-child-launch-v1',control_sha256=csha,
                authorization_sha256=build.digest((root/'authorization.json').read_bytes()),parent_pid=os.getpid(),
                root_binding=binding,work_deadline=2500.,connection_deadline=3100.,
                session_sha256=build.digest(build.enc(state))),binding


def test_unbound_client_and_no_handoff_or_arming_API(tmp_path):
    assert build.LIVE_EXECUTION_BINDING is None
    assert not hasattr(build,'prepare_unarmed') and not hasattr(build,'authorization')
    with pytest.raises(ValueError,match='Unarmed'): client.run(tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_cold_derivation_all_inverses_and_unchanged_scientific_graph(bundle):
    c,f=build.inputs(ROOT); old_payload=json.loads(f[c['pins']['old_source_payload']])
    d=bundle['derivation']; assert len(bundle['uploads'])==45 and d['runtime_bytes']==438445
    assert d['roles']==['proof','forward-preserving'] and d['execution_binding'] is None
    for name,edits in d['edits'].items():
        if name in build.NAMES:
            original=base64.b64decode(old_payload['supervisor/'+name]); derived=bundle['sources'][name]
        elif name in ('outer','client'):
            original=f[c['pins']['old_'+name]]; derived=bundle['generated'][build.OUTER if name=='outer' else build.CLIENT]
        else:
            original=bundle['sources'][build.NAMES[3]]; derived=bundle['generated'][build.READERS[build.NAMES[3]]]
        assert build.inverse(derived,edits['edits'])==original
    assert len(bundle['workers'])==39 and not set(bundle['workers']) & adapter.FORBIDDEN
    assert adapter.READER in bundle['workers'] and 'models/predictor_pgd_split_backend_v3.py' not in bundle['workers']
    for name,value in bundle['workers'].items(): assert value==f[c['runtime'][name]]
    for name in build.NAMES[4:]: assert bundle['sources'][name]==base64.b64decode(old_payload['supervisor/'+name])
    assert not bundle['derivation']['new_scope_handoff_authorization_or_target_work']


def test_catalogue_reference_accounting_and_candidate_namespace(bundle,support,baseline,tmp_path):
    root=tmp_path/'scratch'; shutil.copytree(baseline,root)
    guard,binding,c,p,csha=guard_for(support,root); guard.verify_assets(phase='setup')
    assert len(guard.layout.assets)==105 and len(guard.layout.result)==17
    guard.proof_readiness(); ref=bundle['static']['reference']
    assert len(ref['files'])==19 and ref['manifest_sha256']==bundle['source_hashes']['reference_manifest_sha256']
    for name in ref['files']:
        d=guard.layout.classify('reference/'+name); assert d['category']=='input' and d['mutable'] is False
    accuracy=guard.layout.classify('capture/results/components/forward-preserving/accuracy.json')
    assert accuracy['category']=='metadata' and accuracy['max_bytes']==16384
    assert guard.layout.contract['caps']['metadata']==64*2**20
    assert not any('native/' in n or 'numerical/' in n for n in guard.layout.result)
    for name in ('capture/results/components/native/output.npy','worker-plan-native.json','parent-native.json'):
        with pytest.raises(ValueError): guard.layout.classify(name)
    (root/'reference/native/dq.npy').write_bytes(b'EXPLICIT-FAKE REFERENCE DRIFT')
    with pytest.raises(ValueError): guard.verify_assets(phase='setup')


@pytest.mark.parametrize('fault',['old-control','native-role','reference-hash','clock-reset'])
def test_new_control_plan_refuses_cross_scope_reuse(fault,bundle,support,baseline,tmp_path):
    root=tmp_path/'scratch'; shutil.copytree(baseline,root); plan,binding=make_plan(support,bundle,root)
    state=json.loads((root/'session.json').read_bytes()); csha=build.digest(bundle['control'])
    if fault=='old-control':
        changed=json.loads(bundle['control']); changed['kind']='pgd-component-supervised-control-v1'
        with pytest.raises(ValueError): support.boot.control(build.enc(changed))
        return
    if fault=='native-role': plan['role']='native'
    elif fault=='reference-hash': plan['source_hashes']['reference_manifest_sha256']='0'*64
    else: state['connection_started']=101.
    with pytest.raises(ValueError): support.boot.validate_plan(plan,csha,state,101.)


@pytest.mark.parametrize('fault',[None,'reference-truncated','reference-hash','trailing','source-truncated'])
def test_two_archive_setup_same_budget_and_atomic_failure(fault,bundle,support,tmp_path,monkeypatch):
    boot=support.boot; events=[]
    imports=NS(predictor_pgd_split_scratch_v1=support.scratch,predictor_pgd_split_coordinator_v1=support.coord)
    setup=module(bundle['generated'][build.OUTER],'fake_candidate_setup',imports)
    monkeypatch.setattr(boot.PinnedImporter,'install',lambda self: events.append(('installer-FAKE',len(self.files))) or self)
    monkeypatch.setattr(support.coord,'target_gate',lambda *a:events.append(('target-FAKE',True)))
    proxy=NS(**vars(os))
    def reserve(fd,offset,length):
        assert offset==0 and length==16*2**20
        events.append(('ordinary-reserve-FAKE-PREALLOCATION',length))
        for _ in range(16): assert os.write(fd,b'\xa7'*2**20)==2**20
    proxy.posix_fallocate=reserve; setup.os=proxy
    root=tmp_path/'scratch'; root.mkdir(mode=0o700)
    c,f=build.inputs(ROOT); packet=f[c['pins']['capsule_archive']]; reference=f[c['pins']['reference_archive']]
    if fault=='reference-truncated': reference=reference[:-1]
    elif fault=='reference-hash': reference=bytes([reference[0]^1])+reference[1:]
    elif fault=='trailing': reference+=b'EXPLICIT-FAKE TRAILING BYTES'
    elif fault=='source-truncated': packet=packet[:-1]; reference=b''
    b,p,csha=boot.control(bundle['control']); outer_control={'supervised_control_text':bundle['control'].decode(),
        'outer_source_sha256':build.digest(bundle['generated'][build.OUTER])}
    def run():
        return setup.prepare_tree(root,bundle['uploads'],outer_control,b,p,{'user_basis':ORIGIN},
            build.enc(outer_control),build.enc({'origin':ORIGIN,'actual_approval':False}),boot,io.BytesIO(packet+reference),Clock())
    if fault:
        with pytest.raises((ValueError,EOFError)): run()
        assert not (root/'reference/MANIFEST.json').exists()
        assert not (root/'session.json').exists()
    else:
        _,_,guard,binding,derived=run(); guard.verify_assets(phase='setup'); guard.proof_readiness()
        assert len(guard.layout.assets)==105 and ('installer-FAKE',45) in events
        for name,d in bundle['static']['reference']['files'].items():
            assert build.digest((root/'reference'/name).read_bytes())==d['sha256']
        assert not (root/'capture/results/components/forward-preserving').exists()
        assert not (root/'capture/results/components/native').exists()
    put(tmp_path/'SETUP.json',{'origin':ORIGIN,'fault':fault,'events':events,
        'actual_install_process_GPU_proof_preallocation_or_science':False})


def test_private_cache_resolves_v4_reference_accuracy_and_refuses_drift(bundle,support,baseline,tmp_path,monkeypatch):
    root=tmp_path/'scratch'; shutil.copytree(baseline,root); guard,binding,c,p,csha=guard_for(support,root)
    finder=support.boot.PinnedImporter(root,binding,c,p); fake=NS(modules={},meta_path=[])
    monkeypatch.setattr(support.boot,'sys',fake); finder.install()
    assert len(finder.files)==45 and 'models.predictor_pgd_split_backend_v3' not in finder.files
    original=builtins.__import__
    def load_module(name):
        if name in fake.modules: return fake.modules[name]
        if '.' in name: load_module(name.rsplit('.',1)[0])
        path=None if '.' not in name else fake.modules[name.rsplit('.',1)[0]].__path__
        spec=finder.find_spec(name,path); value=importlib.util.module_from_spec(spec)
        value.__dict__['__builtins__']={**vars(builtins),'__import__':load}
        fake.modules[name]=value; finder.exec_module(value)
        if '.' in name: setattr(fake.modules[name.rsplit('.',1)[0]],name.rsplit('.',1)[1],value)
        return value
    def load(name,globals=None,locals=None,fromlist=(),level=0):
        if name.split('.')[0] not in ('models','attacks'): return original(name,globals,locals,fromlist,level)
        assert level==0; value=load_module(name)
        for child in fromlist:
            if name+'.'+child in finder.files: load_module(name+'.'+child)
        return value if fromlist else load_module(name.split('.')[0])
    for name in ('models.predictor_pgd_split_backend_v4','models.predictor_pgd_candidate_reference_v1',
                 'models.predictor_pgd_candidate_accuracy_v1'): load_module(name)
    assert support.boot.require_namespace(root,binding,csha) is finder
    with pytest.raises(ImportError): finder.find_spec('models.predictor_pgd_candidate_accuracy_v1',fake.modules['models'].__path__,target=object())
    with pytest.raises(ModuleNotFoundError): finder.find_spec('models._pgd_split_numerical_v1',fake.modules['models'].__path__)
    (root/'overlay/models/predictor_pgd_candidate_reference_v1.py').write_bytes(b'EXPLICIT-FAKE DRIFT\n')
    with pytest.raises(ValueError): support.boot.require_namespace(root,binding,csha)


@pytest.mark.parametrize('role',['proof','forward-preserving'])
def test_candidate_role_release_requires_owned_handle(role,bundle,support,baseline,tmp_path,monkeypatch):
    root=tmp_path/'scratch'; shutil.copytree(baseline,root)
    driver,clock,events,holder=launcher(support,bundle,root,monkeypatch)
    observed=driver.launch(role,{'role':role,'launch_started':100.,'deadline':150. if role=='proof' else 220.})
    names=[e[0] for e in events]
    assert names.index('lease-bound')<names.index('guard-FAKE')<names.index('release')
    assert holder['lease'].closed and observed['parent_observation_passed']
    assert not observed['actual_CUDA_execution_admitted']; driver.assert_idle()


@pytest.mark.parametrize('fault',['lease','foreign','rss','deadline'])
def test_candidate_launch_failure_stops_release_and_closes_owned_handles(fault,bundle,support,baseline,tmp_path,monkeypatch):
    root=tmp_path/'scratch'; shutil.copytree(baseline,root)
    driver,clock,events,holder=launcher(support,bundle,root,monkeypatch,fault=fault)
    with pytest.raises((ValueError,MemoryError,TimeoutError)):
        driver.launch('forward-preserving',{'role':'forward-preserving','launch_started':100.,'deadline':220.})
    if fault in ('lease','foreign'): assert not any(e[0]=='release' for e in events)
    if fault=='lease': assert not any(e[0]=='OWNED-FAKE-SIGNAL' for e in events)
    else: assert holder['lease'].closed
    driver.assert_idle()
    put(tmp_path/'LAUNCH-FAILURE.json',{'origin':ORIGIN,'fault':fault,'events':events,'actual_process_or_signals':False})


def test_dispatch_supplies_distinct_reference_root_and_manifest(bundle,support,baseline,tmp_path,monkeypatch):
    root=tmp_path/'scratch'; shutil.copytree(baseline,root); plan,binding=make_plan(support,bundle,root)
    events=[]; runtime=NS(EXECUTION_BINDING=None,component_entry=lambda args:events.append(vars(args)))
    original=support.coord.dispatch_child.__builtins__['__import__']
    def load(name,globals=None,locals=None,fromlist=(),level=0):
        if name=='resource': return NS(RLIMIT_FSIZE=1,setrlimit=lambda *a:None)
        if name=='models' and '_pgd_component_entry_v1' in fromlist: return NS(_pgd_component_entry_v1=runtime)
        return original(name,globals,locals,fromlist,level)
    monkeypatch.setitem(support.coord.dispatch_child.__builtins__,'__import__',load)
    monkeypatch.setattr(support.coord,'target_gate',lambda *a:None); monkeypatch.setattr(support.coord,'gpu_gate',lambda *a:None)
    monkeypatch.setattr(support.coord,'time',NS(monotonic=lambda:101.))
    support.coord.dispatch_child(root,binding,bundle['control'],fake_authorization(bundle['control']),plan)
    assert events[0]['reference_root']==str(root/'reference') and events[0]['role']=='forward-preserving'
    assert runtime.EXECUTION_BINDING['reference_manifest_sha256']==bundle['source_hashes']['reference_manifest_sha256']
    assert len({runtime.EXECUTION_BINDING[k] for k in ('capsule_root','reference_root','result_root')})==3
    assert runtime.EXECUTION_BINDING['kind']=='candidate-only-reviewed-execution-binding-v1'
    with pytest.raises(ValueError,match='fresh null'):
        support.coord.dispatch_child(root,binding,bundle['control'],fake_authorization(bundle['control']),plan)


def test_real_proof_reader_rejects_fake_proof_before_candidate(bundle,support,baseline,tmp_path):
    root=tmp_path/'scratch'; shutil.copytree(baseline,root); guard,*_=guard_for(support,root)
    put(root/'allocation-result-v2.json',{'kind':'EXPLICIT-FAKE-PROOF','passed':False,'complete':False,
        'actual_bytes_written':0,'actual_bytes_read_back':0,'origin':ORIGIN})
    driver=support.coord.SystemDriver(NS(root=root)); driver.guard=guard
    with pytest.raises(ValueError): driver.verify('proof',{}, {'pid':818181,'exit_status':0,'parent_observation_passed':True})
    assert driver.sources=={} and not (root/'capture/results/components/forward-preserving').exists()


def test_hard_and_soft_clock_ports_keep_original_connection_origin(monkeypatch):
    clock=Clock(); events=[]; main=object()
    class Timer:
        def __init__(self,delay,callback): self.delay=delay; self.daemon=False
        def start(self): events.append(('soft-wake-FAKE',self.delay))
        def cancel(self): events.append(('cancel-FAKE',True))
    sig=NS(**vars(signal)); sig.getsignal=lambda n:signal.SIG_DFL; sig.getitimer=lambda n:(0.,0.)
    sig.signal=lambda *a:events.append(('signal-FAKE',a)); sig.setitimer=lambda *a:events.append(('kernel-timer-FAKE',a))
    monkeypatch.setattr(outer,'signal',sig)
    monkeypatch.setattr(outer,'threading',NS(current_thread=lambda:main,main_thread=lambda:main,Timer=Timer))
    alarms=outer.Alarms(clock=clock); alarms.arm(220.,hard_deadline=2500.,phase='forward-preserving')
    assert alarms.hard==2500. and any(e==('signal-FAKE',(signal.SIGALRM,signal.SIG_DFL)) for e in events)
    clock.value=220.
    with pytest.raises(outer.Deadline): alarms.check()
    assert alarms.hard==2500.; alarms.close()
    receiver=client.ReceiverClock(100.,101.,1001.,1200.,clock=lambda:102.)
    assert receiver.deadline==300.
    receiver.clock=lambda:301.
    with pytest.raises(outer.Deadline): receiver.check()


def test_outer_source_and_reference_control_are_bound_without_package(bundle,tmp_path):
    c,f=build.inputs(ROOT)
    pins={n:build.digest((ROOT/n).read_bytes()) for n in outer.LOCAL_SOURCES}
    value={'kind':'pgd-candidate-outer-control-v1','supervised_control_text':bundle['control'].decode(),
        'supervised_control_sha256':build.digest(bundle['control']),'outer_source_sha256':pins[build.OUTER],
        'local_source_sha256':pins,'runtime_files':bundle['static']['runtime_files'], 'repository':str(ROOT),
        'local_handoff':str(tmp_path/'EXPLICIT-FAKE-NONEXISTENT'),'packet_path':outer.PACKET_PATH,
        'reference_path':outer.REFERENCE_PATH,'reference_manifest_sha256':outer.REFERENCE_MANIFEST_SHA,
        'known_hosts_text':outer.KNOWN_HOSTS,'known_hosts_sha256':build.digest(outer.KNOWN_HOSTS.encode()),
        'expected_ssh_fingerprint':outer.FINGERPRINT,'financial_state_unchanged':True,
        'authorized_at_preparation':False,'target_helper_reviewed':False}
    raw=build.enc(value); outer.validate_control(raw,pins[build.OUTER])
    mapping={n:base64.b64encode(v).decode() for n,v in bundle['uploads'].items()}
    assert outer.decode_sources(mapping,value,json.loads(bundle['control']))==bundle['uploads']
    a=outer.authorization(value,build.digest(raw),ORIGIN)
    assert a['full_model_gradients']==0 and a['extra_component_VJPs']==1 and a['automatic_retries']==0
    changed=dict(value,reference_manifest_sha256='0'*64)
    with pytest.raises(ValueError): outer.validate_control(build.enc(changed),pins[build.OUTER])
    changed=dict(value,runtime_files={n:d for n,d in value['runtime_files'].items() if 'candidate_accuracy' not in n})
    with pytest.raises(ValueError): outer.validate_control(build.enc(changed),pins[build.OUTER])
    assert list(tmp_path.iterdir())==[]


@pytest.fixture(scope='module')
def completed_flow(bundle,support,baseline,tmp_path_factory):
    """One fake lifecycle, real candidate completion/parent readers and byte codec."""
    folder=tmp_path_factory.mktemp('candidate-integrated-flow'); folder.chmod(0o700)
    root=folder/'scratch'; shutil.copytree(baseline,root)
    guard,binding,c,p,csha=guard_for(support,root); patch=pytest.MonkeyPatch(); state={}
    try:
        def work(plan,child,clock):
            if plan['role']=='proof':
                put(root/'allocation-result-v2.json',{'kind':'EXPLICIT-FAKE-PROOF','passed':False,'complete':False,
                    'actual_bytes_written':0,'actual_bytes_read_back':0,'origin':ORIGIN})
                return
            spec=json.loads(build.inputs(ROOT)[1][build.inputs(ROOT)[0]['pins']['reference_manifest']])
            h=Harness((bundle['workers'],None,spec),(root/'source',root/'reference'),root/'capture',pid=child.pid,clock=clock)
            state['harness']=h
            runtime=NS(EXECUTION_BINDING=None)
            def entry(args):
                h.entry['EXECUTION_BINDING']=runtime.EXECUTION_BINDING; h.entry['platform']=NS(node=lambda:'linux10614')
                h.args=args; return h.run()
            runtime.component_entry=entry; state['runtime']=runtime
            support.coord.dispatch_child(root,binding,bundle['control'],fake_authorization(bundle['control']),plan)
        driver,clock,events,holder=launcher(support,bundle,root,patch,work=work)
        original=support.coord.dispatch_child.__builtins__['__import__']
        def load(name,globals=None,locals=None,fromlist=(),level=0):
            if name=='resource': return NS(RLIMIT_FSIZE=1,setrlimit=lambda *a:events.append(('RLIMIT-FAKE',a)))
            if name=='models' and '_pgd_component_entry_v1' in fromlist: return NS(_pgd_component_entry_v1=state['runtime'])
            if name=='models' and '_pgd_component_io_v1' in fromlist: return NS(_pgd_component_io_v1=NS(**state['harness'].io))
            if name=='models' and '_pgd_component_job_v1' in fromlist: return NS(_pgd_component_job_v1=NS(**state['harness'].job))
            return original(name,globals,locals,fromlist,level)
        patch.setitem(support.coord.dispatch_child.__builtins__,'__import__',load)
        patch.setattr(support.coord,'target_gate',lambda *a:events.append(('target-FAKE',True)))
        patch.setattr(support.coord,'gpu_gate',lambda *a:events.append(('GPU-FAKE',True)))
        patch.setattr(support.coord,'time',NS(monotonic=clock))
        owner=support.coord.OuterCoordinator(root,binding,bundle['control'],fake_authorization(bundle['control']),
            connection_started=100.,clock=clock,outer_guard=guard,
            outer_alarm=NS(arm=lambda deadline,**kw:events.append(('alarm-FAKE',deadline,kw))))
        driver.owner=owner; owner.driver=driver
        def prepare():
            guard.verify_assets(phase='setup'); guard.proof_readiness(); driver.guard=guard
            return {'origin':ORIGIN,'actual_source_install_process_GPU_or_proof_authenticated':False}
        patch.setattr(driver,'prepare',prepare)
        def fake_proof(expected_sha):
            marker=guard.read('allocation-result-v2.json',16*1024)
            assert build.digest(marker)==expected_sha and json.loads(marker)['passed'] is False
            return {'origin':ORIGIN,'actual_proof_execution_authenticated':False,'passed':False,'bytes_written':0}
        patch.setattr(guard,'verify_proof',fake_proof)
        saved=owner.run_work(); driver.assert_idle()
        with pytest.raises(ValueError,match='no retries'): owner.run_work()
        imports=NS(predictor_pgd_split_bootstrap_v1=support.boot,predictor_pgd_split_scratch_v1=support.scratch)
        codec=module(bundle['sources'][build.NAMES[3]],'fake_candidate_flow_codec',imports)
        collector=codec.Collector(guard,bundle['control'],fake_authorization(bundle['control']),connection_started=100.,clock=clock)
        header=collector.prepare(); wire=io.BytesIO(); terminal=collector.stream(wire); data=wire.getvalue()
        receipt=codec.receive(io.BytesIO(data),folder/'received',bundle['control'],fake_authorization(bundle['control']),window=collector.window)
        flow={'origin':ORIGIN,'completed':saved['completed'],'calls':dict(state['harness'].calls),'events':events,
            'actual_process_proof_timer_signal_GPU_array_model_gradient_profile_or_network':False,
            'candidate_metadata_and_transport_readers_are_real':True,'wire_bytes':len(data),'wire_sha256':build.digest(data)}
        for name,value in [('FLOW.json',flow),('HEADER.json',header),('TERMINAL.json',terminal),('RECEIPT.json',receipt),('WIRE.bin',data)]:
            put(folder/name,value)
        return NS(folder=folder,root=root,state=saved,h=state['harness'],codec=codec,wire=data,
                  receipt=receipt,header=header,terminal=terminal,collector=collector,clock=clock)
    finally:
        patch.undo()


def test_proof_candidate_completion_accuracy_parent_and_same_stream_join(completed_flow,bundle):
    f=completed_flow; assert f.state['completed']==['proof','forward-preserving']
    assert len({r['pid'] for r in f.state['receipts']})==2 and f.state['complete'] is False
    assert all(f.h.calls[n]==1 for n in ('forward','outer_VJP','inner_math_VJP','fake_accuracy','profile_start','profile_stop','route_exit'))
    assert f.receipt['transport_verified'] and not f.receipt['scientific_qualified'] and not f.receipt['training_admitted']
    received=f.folder/'received'; folder=received/'capture/results/components/forward-preserving'
    worker=json.loads((folder/'worker.json').read_bytes()); accuracy=json.loads((folder/'accuracy.json').read_bytes())
    resource=json.loads((folder/'resource.json').read_bytes()); parent=json.loads((received/'parent-forward-preserving.json').read_bytes())
    assert accuracy['binding']==worker['binding'] and accuracy['outputs']==worker['outputs']
    assert worker['binding']['control_sha256']==build.digest(bundle['control'])
    assert accuracy['reference']['reference_manifest_sha256']==bundle['source_hashes']['reference_manifest_sha256']
    assert resource['pid']==parent['pid']==worker['pid'] and resource['wall_seconds']<=parent['wall_seconds']<=120
    assert worker['saved_native_relative_values_passed'] and worker['evidence_origin']==ORIGIN
    assert f.state['receipts'][1]['receipt']['metadata_bindings_consistent']
    assert not f.state['receipts'][1]['receipt']['actual_CUDA_execution_admitted']
    progress=(folder/'progress.jsonl').read_bytes().splitlines(); assert len(progress)==47
    assert len({json.loads(row)['phase'] for row in progress if json.loads(row)['event']=='begin'})==23
    final=json.loads((received/'capture/results/final-receipt.json').read_bytes())
    assert final['kind']=='pgd-candidate-work-receipt-v1' and final['all_saved_role_checks_returned']
    assert final['collection_complete'] is False
    assert not (received/'parent-native.json').exists() and not (received/'capture/results/components/native').exists()
    names=[row['name'] for row in f.header['records']]
    assert names.index('capture/results/components/forward-preserving/accuracy.json')<names.index('reference/native/dq.npy')
    for name,d in f.receipt['files'].items():
        path=received/name; assert build.digest(path.read_bytes())==d['sha256']
        assert path.lstat().st_nlink==1 and stat.S_IMODE(path.lstat().st_mode)==0o600


@pytest.mark.parametrize('fault',['truncated','terminal-hash','trailing'])
def test_candidate_stream_failure_preserves_incomplete_journal(fault,completed_flow,bundle,tmp_path):
    f=completed_flow; data=f.wire
    if fault=='truncated': data=data[:-17]
    elif fault=='trailing': data+=b'EXPLICIT-FAKE TRAILING BYTES'
    else:
        last=data.rindex(f.codec.END); start=last+len(f.codec.END)+4
        terminal=json.loads(data[start:]); terminal['prefix_sha256']='0'*64
        encoded=build.enc(terminal); data=data[:last]+f.codec.END+len(encoded).to_bytes(4,'big')+encoded
    with pytest.raises((ValueError,EOFError)):
        f.codec.receive(io.BytesIO(data),tmp_path/'partial',bundle['control'],fake_authorization(bundle['control']),window=f.collector.window)
    journal=json.loads((tmp_path/'partial-transport/PARTIAL-RECEIPT.json').read_bytes())
    assert journal['transport_verified'] is False and journal['stream_complete'] is False
    assert (tmp_path/'partial-transport/VERIFIED-FILES.jsonl').stat().st_size>0


def test_client_keeps_university_root_and_continuing_collection_clocks(completed_flow,bundle):
    f=completed_flow; claimed={'scratch':str(f.root),'connection_started':100.}
    ready={'collection_started':f.collector.window.started,'collection_deadline':f.collector.window.deadline}
    with pytest.raises(ValueError,match='another root or clock'):
        client.bound_collection_source(io.BytesIO(f.wire),f.collector.window,claimed,ready,
            {'supervised_control_text':bundle['control'].decode()},fake_authorization(bundle['control']))
