"""Cold source assembly and ordinary metadata/fake binding checks only.

No real arrays, model, gradient, profiler, process, proc, PID API or remote work.
"""
import ast
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import symtable
import sys
import threading
from argparse import Namespace
from types import CodeType,SimpleNamespace as NS
import pytest
from models import predictor_pgd_split_runtime_v1 as assembly

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=(ROOT/'configs/predictor_pgd_split_runtime_v1.json').read_bytes()
PINNED=json.loads(CONTRACT)['source_sha256']
INPUTS={p:(ROOT/p).read_bytes() for p in PINNED}

@pytest.fixture(scope='module')
def built():return assembly.build(INPUTS,CONTRACT)

def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
def code(c):
    return (c.co_code,c.co_argcount,c.co_posonlyargcount,c.co_kwonlyargcount,c.co_flags,c.co_names,c.co_varnames,
        c.co_freevars,c.co_cellvars,c.co_stacksize,c.co_exceptiontable,tuple(code(x) if isinstance(x,CodeType) else x for x in c.co_consts))
def function_code(tree,name):
    def search(c):
        for k in c.co_consts:
            if isinstance(k,CodeType):
                if k.co_name==name:return k
                result=search(k)
                if result:return result
    return search(compile(tree,'<SOURCE-ONLY-NO-EXEC>','exec'))

def test_closed_source_graph_is_deterministic_and_never_imports_runtime(built):
    files,m=built
    assert assembly.build(INPUTS,CONTRACT)==built
    assert len(m['generated_names'])==12
    for p,b in files.items():
        compile(b,p,'exec')
        assert m['files'][p]=={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
        assert all(q in files for q in m['imports'][p])
    assert not m['actual_runtime_loaded_or_executed'] and not m['target_runner_available'] and not m['new_authorization_ready']
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules

@pytest.mark.parametrize('changed',[
 'models/predictor_pgd_target_worker_v1.py','models/predictor_pgd_target_records_v1.py',
 'models/predictor_sdpa_observer_v1.py','models/predictor_sdpa_component_v7.py',
 'models/training.py','models/execution.py','models/predictor_pgd_split_store_v1.py',
 'models/predictor_pgd_split_probe_job_v1.py','models/__init__.py','configs/predictor_pgd_split_profile_v1.json'])
def test_dependency_drift_refuses_without_any_module_load(changed):
    bad=dict(INPUTS);bad[changed]+=b'\n'
    with pytest.raises(ValueError,match='Pinned runtime source changed'):assembly.build(bad,CONTRACT)
    assert 'torch' not in sys.modules

@pytest.mark.parametrize('fault',['contract','missing','extra'])
def test_no_caller_defined_contract_or_extra_module_can_enter(fault):
    inputs=dict(INPUTS);contract=CONTRACT
    if fault=='contract':contract+=b'\n'
    elif fault=='missing':inputs.pop('models/training.py')
    else:inputs['models/evil.py']=b'raise RuntimeError("MUST NOT EXECUTE")'
    with pytest.raises(ValueError):assembly.build(inputs,contract)


def test_compiled_math_reconstruction_and_saved_value_gates_survive_real_import_binding(built):
    files,_=built;trees={p:ast.parse(b) for p,b in files.items()}
    route=trees[assembly.generated('route')]
    cpu=ast.parse(INPUTS['models/predictor_forward_preserving_cpu_v1.py'])
    target=ast.parse(INPUTS['models/predictor_forward_preserving_target_v1.py'])
    for old,new,method in [(cpu,route,'backward'),(target,route,'forward')]:
        assert code(function_code(old,method))==code(function_code(new,method))
    # Select the bridge by its declaring class; both classes define backward.
    def bridge(t):
        n=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='_Attention')
        return ast.fix_missing_locations(ast.Module(body=[n],type_ignores=[]))
    # Module compiler import context is tested on the complete generated source.
    def named_class_code(t,cls,method):
        top=compile(t,'<COLD-CLASS>','exec');cc=next(c for c in top.co_consts if isinstance(c,CodeType) and c.co_name==cls)
        return next(c for c in cc.co_consts if isinstance(c,CodeType) and c.co_name==method)
    assert code(named_class_code(target,'_Attention','backward'))==code(named_class_code(route,'_Attention','backward'))
    operands=trees[assembly.generated('providers')];old=ast.parse(INPUTS['models/predictor_sdpa_component_v7.py'])
    assert dump(fn(old,'_operands'))==dump(fn(operands,'_operands'))
    assert code(function_code(old,'_operands'))==code(function_code(operands,'_operands'))
    old=ast.parse(INPUTS['models/predictor_pgd_target_records_v1.py']);records=trees[assembly.generated('records')]
    for name in ('identity','check_scope','arrays','compare'):
        assert dump(fn(old,name))==dump(fn(records,name))
        assert code(function_code(old,name))==code(function_code(records,name))
    assert trees[assembly.generated('startup')] and dump(trees[assembly.generated('startup')])==dump(ast.parse(INPUTS['models/predictor_sdpa_component_startup_v2.py']))


def test_all_generated_names_have_concrete_lexical_providers(built):
    files,_=built
    for path in (p for p in files if p.startswith('models/_pgd_split_')):
        table=symtable.symtable(files[path].decode(),path,'exec')
        bound={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
        unknown=set()
        def visit(t):
            for s in t.get_symbols():
                if s.is_referenced() and s.is_global() and s.get_name() not in bound and not hasattr(builtins,s.get_name()):unknown.add(s.get_name())
            for child in t.get_children():visit(child)
        visit(table)
        assert not unknown,(path,unknown)


def io_namespace(built):
    ns={'__name__':'EXPLICIT_LOCAL_BYTE_IO_ONLY'}
    exec(compile(built[0][assembly.generated('io')],'<BYTE-IO-NO-SCIENCE>','exec'),ns)
    return NS(**ns)


def test_serial_thread_publications_retain_unchanged_writer_metadata(built,tmp_path):
    tmp_path.chmod(0o700);io=io_namespace(built);store=io.bind(tmp_path)
    errors=[]
    def write(i):
        try:io._atomic(tmp_path/'results/numerical',f'restart-noise-{i}.npy',[b'OPAQUE BYTES: NO ARRAY'],16*2**20)
        except BaseException as e:errors.append(e)
    threads=[threading.Thread(target=write,args=(i,)) for i in (0,1)]
    for t in threads:t.start()
    for t in threads:t.join()
    assert not errors
    assert len(store.written)==2
    assert io.read(tmp_path/'results/numerical','restart-noise-0.npy',16*2**20)==b'OPAQUE BYTES: NO ARRAY'
    assert io.sha_file(tmp_path/'results/numerical/restart-noise-1.npy')==hashlib.sha256(b'OPAQUE BYTES: NO ARRAY').hexdigest()
    p=tmp_path/'results/numerical/restart-noise-0.npy';p.write_bytes(b'CHANGED')
    with pytest.raises(ValueError,match='committed result metadata changed'):store.check()

@pytest.mark.parametrize('fault',['rebind','foreign-pid','unknown','escape','symlink','hardlink','mode','read-limit','write-limit','wrong-chunk','foreign-mutable','poison','recursive'])
def test_byte_namespace_refusals_preserve_prior_or_partial_evidence(built,tmp_path,fault):
    tmp_path.chmod(0o700);io=io_namespace(built);store=io.bind(tmp_path)
    folder=tmp_path/'results';folder.mkdir(mode=0o700)
    def publication():return io._atomic(folder,'run-status.json',[b'{}\n'],256*1024)
    if fault=='rebind':
        with pytest.raises(ValueError,match='One result namespace'):io.bind(tmp_path)
    elif fault=='foreign-pid':
        store.pid=-1
        with pytest.raises(ValueError,match='serial process writer'):publication()
    elif fault in ('unknown','escape'):
        with pytest.raises(ValueError):io._atomic(folder,'unknown.json' if fault=='unknown' else '../run-status.json',[b'x'],4)
    elif fault in ('symlink','hardlink','mode','read-limit'):
        publication();p=folder/'run-status.json'
        if fault=='symlink':
            p.rename(folder/'supervisor.json');p.symlink_to('supervisor.json')
        elif fault=='hardlink':os.link(p,folder/'supervisor.json')
        elif fault=='mode':p.chmod(0o644)
        with pytest.raises(ValueError):io.read(folder,'run-status.json',1 if fault=='read-limit' else 256*1024)
    elif fault in ('write-limit','wrong-chunk'):
        with pytest.raises(ValueError,match='whole byte publication cap'):io._atomic(folder,'run-status.json',[b'too long' if fault=='write-limit' else 'TEXT NOT BYTES'],2)
        assert not (folder/'run-status.json').exists()
    elif fault=='foreign-mutable':
        p=folder/'run-status.json';p.write_bytes(b'FOREIGN');p.chmod(0o600)
        with pytest.raises(FileExistsError,match='foreign result'):publication()
        assert p.read_bytes()==b'FOREIGN'
    elif fault=='poison':
        publication()
        with pytest.raises(FileExistsError):io._atomic(folder/'numerical','restart-noise-0.npy',[b'x'],1);io._atomic(folder/'numerical','restart-noise-0.npy',[b'x'],1)
        with pytest.raises(ValueError,match='unpoisoned'):publication()
    elif fault=='recursive':
        store.busy=True
        with pytest.raises(ValueError,match='Recursive'):publication()
        assert store.poisoned


class BindingHarness:
    """Opaque ports replace imports only in this explicit simulation namespace."""
    def __init__(self,built,root,fault=None):
        self.events=[];self.fault=fault;self.threads=99;self.flags={'flash':True};self.actual={'threads':4,'precision':'CAPTURED-STUB'}
        self.namespace={};self.job=None;self.fail=None
        cap={'selected_call':{'actual_call':{'entry_flags':dict(self.flags)}}}
        self.store=NS(root=root,check=lambda:self.events.append('store-check'))
        def initialize(device):self.events.append(('startup',device));return {'opaque':'STARTUP-STUB'}
        def source(args):
            self.events.append('source')
            if fault=='source':raise ValueError('SIMULATED source refusal')
            return cap,{'opaque':'JOIN-STUB'}
        def operands(args,cap):self.events.append('operands');return ('OPAQUE-Q','OPAQUE-K','OPAQUE-V'),'OPAQUE-COT',{}
        def compute(route,record,observe):
            self.events.append(('compute',route,record,observe))
            if fault=='restore-flags':self.flags['flash']=False
            if fault=='restore-runtime':self.actual['precision']='CHANGED-STUB'
            return {'opaque':'RETURNS-STUB'}
        self.provider=NS(load_new_source=source,prepare_operands=operands,flags=lambda:dict(self.flags),
            make_profile=lambda:'PROFILE-STUB',compute_once=compute,raw_bytes=lambda p:b'RAW-STUB',serialize_return=lambda t:b'OPAQUE-RETURN')
        def runtime(device):
            self.events.append(('runtime',device))
            value=dict(self.actual)
            if fault=='runtime-before' and 'operands' not in self.events:value['precision']='CHANGED-STUB'
            return value
        def set_threads(n):
            self.events.append(('threads',n));self.threads=3 if fault=='threads' else n
        torch=NS(set_num_threads=set_threads,get_num_threads=lambda:self.threads,
            profiler=NS(record_function=object()),cuda=NS(max_memory_allocated=lambda:42))
        def bound(root):self.events.append(('bind',root));return self.store
        self.worker_bytes=json.dumps({'baseline':{'runtime':{'threads':4,'precision':'CAPTURED-STUB','model_execution':'EXCLUDED-STUB'}}}).encode()
        io=NS(bind=bound,_parse=json.loads,read=lambda *a:self.worker_bytes+(b' ' if fault=='runtime-source' else b''))
        def numerical_worker(args):self.events.append('numerical');return {'opaque':'EIGHT-OBJECTIVES-STUB'}
        def verify(*a,**kw):self.events.append('verify');return {'opaque':'VERIFIED-STUB'}
        numerical=NS(verify_worker=verify)
        self.routes=NS(ComponentRoute=lambda *a,**kw:('ROUTE-STUB',a,kw))
        harness=self
        class Job:
            def __init__(self,role,binding,store,ports,**kw):
                harness.events.append(('Job',role,binding,kw));harness.job=self
                self.ports=ports;self.state={'pid':os.getpid()}
                assert set(ports)=={'startup','source','operands','route','profile','compute','raw','serialize','restore','allocator'}
            def execute(self):
                p=self.ports;p['startup']();cap,joined=p['source']();values,incoming,meta=p['operands'](cap)
                route=p['route']('SIMULATED-ROLE',values,incoming,{'opaque':'BIND-STUB'});profile=p['profile']()
                returns=p['compute'](route,'OBSERVE-STUB');assert p['raw'](profile)==b'RAW-STUB'
                assert p['serialize'](returns)==b'OPAQUE-RETURN' and p['allocator']()==42
                p['restore']()
                return self.state
        module_models=NS(_pgd_split_io_v1=io,_pgd_split_numerical_v1=NS(unbound_numerical_worker=numerical_worker),
            _pgd_split_records_v1=numerical,_pgd_split_startup_v1=NS(initialize=initialize,validate=lambda *a,**k:self.events.append('startup-validate')),
            _pgd_split_providers_v1=self.provider,_pgd_split_route_v1=self.routes)
        from models.predictor_pgd_split_lineage_v1 import DESIGN_SHA,sha
        imports={'torch':torch,'models':module_models,'models.training':NS(runtime_contract=runtime),
            'models.predictor_pgd_split_probe_job_v1':NS(ComponentJob=Job),
            'models.predictor_pgd_split_lineage_v1':NS(DESIGN_SHA=DESIGN_SHA,sha=sha)}
        real_import=builtins.__import__
        def fake_import(name,globals=None,locals=None,fromlist=(),level=0):
            if name in imports:return imports[name]
            if name in ('argparse','hashlib','math','pathlib'):return real_import(name,globals,locals,fromlist,level)
            raise AssertionError('Unreviewed import in explicit runtime-binding simulation: '+name)
        self.namespace={'__name__':'EXPLICIT_OPAQUE_BINDING_SIMULATION','__builtins__':dict(vars(builtins),__import__=fake_import)}
        exec(compile(built[0][assembly.generated('entry')],'<NO-SCIENCE-FAKE-IMPORTS-ONLY>','exec'),self.namespace)
    def component_args(self,root,role='native'):
        return Namespace(result_root=str(root),control_sha256='2'*64,numerical_sha256=hashlib.sha256(self.worker_bytes).hexdigest(),original_sha256='4'*64,boundary_sha256='5'*64,role=role,deadline=12345.)

@pytest.mark.parametrize('role',['native','forward-preserving'])
def test_concrete_component_bindings_use_captured_runtime_and_four_threads(built,tmp_path,role):
    h=BindingHarness(built,tmp_path);args=h.component_args(tmp_path,role)
    out=h.namespace['component_entry'](args)
    assert out['runtime']=={'threads':4,'precision':'CAPTURED-STUB'}
    assert out['runtime_source_numerical_sha256']==hashlib.sha256(h.worker_bytes).hexdigest()
    assert h.events.index(('threads',4))<h.events.index('source')<h.events.index('operands')
    assert h.events.count('source')==1 and sum(type(x) is tuple and x[0]=='compute' for x in h.events)==1
    assert h.events[-1]=='store-check'
    with pytest.raises(ValueError,match='One entry'):h.namespace['component_entry'](args)
    assert 'torch' not in sys.modules

@pytest.mark.parametrize('fault',['threads','source','runtime-before','restore-runtime','restore-flags','runtime-source'])
def test_binding_failure_stops_later_work_without_retry(built,tmp_path,fault):
    h=BindingHarness(built,tmp_path,fault);a=h.component_args(tmp_path)
    with pytest.raises(ValueError):h.namespace['component_entry'](a)
    if fault in ('threads','source','runtime-before','runtime-source'):assert 'operands' not in h.events
    with pytest.raises(ValueError,match='One entry'):h.namespace['component_entry'](a)

@pytest.mark.parametrize('fault',['ports','digest','role','deadline','relative-root','tiny','output'])
def test_entry_arguments_cannot_inject_work_or_reuse_old_numerical_domain(built,tmp_path,fault):
    h=BindingHarness(built,tmp_path);a=h.component_args(tmp_path)
    name='component_entry'
    if fault=='ports':a.ports={'compute':lambda:None}
    elif fault=='digest':a.numerical_sha256='WRONG'
    elif fault=='role':a.role='old-math'
    elif fault=='deadline':a.deadline=float('nan')
    elif fault=='relative-root':a.result_root='relative'
    else:
        name='numerical_entry';a=Namespace(result_root=str(tmp_path),control_sha256='2'*64,packet='PACKET',upstream='UPSTREAM',checkpoint='CHECKPOINT',proposal='PROPOSAL',
            output=str(tmp_path/'results/numerical'),device='cuda',tiny=False)
        if fault=='tiny':a.tiny=True
        else:a.output=str(tmp_path/'OLD-OUTPUT')
    with pytest.raises(ValueError):h.namespace[name](a)
    assert 'numerical' not in h.events and not h.job
    with pytest.raises(ValueError,match='One entry'):h.namespace[name](a)


def test_numerical_entry_uses_new_reader_after_one_unmodified_worker_call(built,tmp_path):
    h=BindingHarness(built,tmp_path)
    a=Namespace(result_root=str(tmp_path),control_sha256='2'*64,packet='PACKET',upstream='UPSTREAM',checkpoint='CHECKPOINT',proposal='PROPOSAL',
        output=str(tmp_path/'results/numerical'),device='cuda',tiny=False)
    out,receipt=h.namespace['numerical_entry'](a)
    assert h.events[-3:]==['numerical','verify','store-check']
    assert out=={'opaque':'EIGHT-OBJECTIVES-STUB'} and receipt=={'opaque':'VERIFIED-STUB'}


@pytest.mark.parametrize('fault',[None,'nonfinite','extent','existing'])
def test_numerical_tensor_transport_keeps_transform_order_and_publishes_before_finite_gate(built,tmp_path,fault):
    # The array and serializer are opaque label stubs; no NumPy/Torch is loaded.
    tmp_path.chmod(0o700);io=io_namespace(built);io.bind(tmp_path)
    calls=[]
    class Label:
        nbytes=8 if fault!='extent' else 65536
        shape=()
        dtype='float32'
        device='cuda:0'
        def detach(self):calls.append('detach');return self
        def cpu(self):calls.append('cpu');return self
        def contiguous(self):calls.append('contiguous');return self
        def numpy(self):calls.append('numpy-STUB');return self
        def copy(self):calls.append('copy-STUB');return self
    token=Label()
    def save(stream,a,**kw):
        assert a is token and kw=={'allow_pickle':False}
        calls.append('save-STUB');stream.write(b'OPAQUE SERIALIZER LABEL; NOT AN ARRAY')
    np=NS(save=save,isfinite=lambda a:NS(all=lambda:fault!='nonfinite'))
    original_import=builtins.__import__
    def importer(name,*args,**kwargs):
        if name=='numpy':return np
        if name=='io':return original_import(name,*args,**kwargs)
        raise AssertionError('Unexpected import: '+name)
    tree=ast.parse(built[0][assembly.generated('records')]);function=fn(tree,'save_tensor')
    ns={'__name__':'EXPLICIT_OPAQUE_SERIALIZER_SIMULATION','__builtins__':dict(vars(builtins),__import__=importer),'Path':Path,
        'frozen':NS(atomic=io,sha_file=io.sha_file)}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<OPAQUE-SERIALIZER-STUB-ONLY>','exec'),ns)
    folder=tmp_path/'results/numerical/objective-0';state={'artifacts':{}}
    if fault=='existing':
        io._atomic(folder,'loss.npy',[b'PRESERVED ORIGINAL LABEL'],65536)
    if fault is None:
        assert ns['save_tensor'](folder,state,'loss',token) is token
        assert state['artifacts']['loss']['saved_before_gate'] is True
        assert calls==['detach','cpu','contiguous','numpy-STUB','copy-STUB','save-STUB']
    else:
        with pytest.raises(ValueError):ns['save_tensor'](folder,state,'loss',token)
        if fault=='extent':assert not (folder/'loss.npy').exists() and 'save-STUB' not in calls
        elif fault=='existing':assert (folder/'loss.npy').read_bytes()==b'PRESERVED ORIGINAL LABEL'
        else:
            assert (folder/'loss.npy').read_bytes().startswith(b'OPAQUE SERIALIZER LABEL')
            assert state['artifacts']['loss']['saved_before_gate'] is True
    assert 'numpy' not in sys.modules and 'torch' not in sys.modules
