"""New source and integrated fake-engine/metadata checks. No actual tensors/CUDA.

Frozen helpers are loaded as class/function definitions ONLY. No historical test
function, scientific cohort, Torch/NumPy import, model or derivative is executed.
"""
import ast
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from types import CodeType, SimpleNamespace as NS
import pytest
from models import predictor_pgd_split_component_v1 as component
from models import predictor_pgd_split_backend_v1 as backend
from models import predictor_pgd_split_lineage_v1 as lineage
from models import predictor_pgd_split_probe_sources_v1 as sources
from models import predictor_pgd_split_probe_job_v1 as job
from models import predictor_pgd_split_store_v1 as typed

ROOT=Path(__file__).resolve().parents[1]
FILES={'observer':'predictor_sdpa_observer_v1','boundary':'predictor_sdpa_capture_record_v1','core':'predictor_sdpa_component_v7',
 'startup':'predictor_sdpa_component_startup_v2','raw':'predictor_sdpa_raw_capture_v1','cpu':'predictor_forward_preserving_cpu_v1',
 'target':'predictor_forward_preserving_target_v1','route_builder':'predictor_pgd_split_component_v1'}
INPUTS={k:(ROOT/'models'/(n+'.py')).read_bytes() for k,n in FILES.items()}
BASE=(ROOT/'configs/predictor_pgd_target_v1.json').read_bytes();DESIGN=(ROOT/'configs/predictor_pgd_split_profile_v1.json').read_bytes()
CTRL='2'*64;BOUNDARY='3'*64;PIXEL='6'*64


@pytest.fixture(scope='module')
def built():return sources.build(INPUTS)


def helpers():
    p=ROOT/'tests/test_predictor_pgd_split_component_v1.py';raw=p.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='eb7ade7a4e9f02a7c6180eb365f78ce6a6d275006f2cb8bd14d24b5a9d1b8ffd'
    t=ast.parse(raw)
    names={'Device','Storage','Handle','FakeTensor','Node','RNG','Runtime','namespace'}
    selected=[n for n in t.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    ns={'__name__':'EXPLICIT_REUSED_FAKE_ENGINE_DEFINITIONS','NS':NS,'contextmanager':contextmanager,'copy':copy,'hashlib':hashlib,
        'json':json,'os':os,'threading':threading,'builder':component,'CPU':INPUTS['cpu'],'TARGET':INPUTS['target'],'SHAPE':(1,12,2048,32)}
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<ONLY-FAKE-HELPER-DEFINITIONS-NO-OLD-TESTS>','exec'),ns)
    p=ROOT/'tests/test_predictor_pgd_split_backend_v1.py';raw=p.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='8aafe32c6dc74e952ba77e722cf00f42a2d24a4fa7426b7a5de7f69b166c5e46'
    f=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='fixture')
    fn={'r':backend,'ORIGIN':1_700_000_000_000_000_000}
    exec(compile(ast.Module(body=[f],type_ignores=[]),'<HANDWRITTEN-RAW-METADATA-HELPER-ONLY>','exec'),fn)
    return ns,fn['fixture']


@pytest.fixture(scope='module')
def fake_helpers():return helpers()


def signature(c):
    return (c.co_code,c.co_argcount,c.co_posonlyargcount,c.co_kwonlyargcount,c.co_flags,c.co_names,c.co_varnames,c.co_freevars,
        c.co_cellvars,c.co_stacksize,c.co_exceptiontable,tuple(signature(x) if isinstance(x,CodeType) else x for x in c.co_consts))


def test_unbound_providers_preserve_frozen_observer_startup_reconstruction_and_route_source(built):
    modules,audit=built
    def functions(t):return {n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
    def dump(t):return ast.dump(t,include_attributes=False)
    assert audit['observer_reversible_edits']==1 and audit['boundary_reversible_edits']==5 and audit['raw_reversible_edits']==1
    original=ast.parse(INPUTS['observer']);new=modules['observer']
    old_class=next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name=='NativeSDPAObserver')
    new_class=next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name=='NativeSDPAObserver')
    assert dump(old_class)==dump(new_class)
    old=functions(ast.parse(INPUTS['boundary']));current=functions(modules['boundary'])
    for n in ('scope_check','load_tensor','rebuild_cpu_inputs'):assert dump(old[n])==dump(current[n])
    assert dump(modules['startup'])==dump(ast.parse(INPUTS['startup']))
    old=functions(ast.parse(INPUTS['core']))['_operands'];current=functions(modules['providers'])['_operands']
    assert dump(old)==dump(current)
    def function_code(t,name):
        c=compile(t,'<COLD-ONLY>','exec')
        return next(x for x in c.co_consts if isinstance(x,CodeType) and x.co_name==name)
    assert signature(function_code(ast.parse(INPUTS['core']),'_operands'))==signature(function_code(modules['providers'],'_operands'))
    for k,t in modules.items():compile(t,'<NO-RUNTIME-BINDING-'+k+'>','exec')
    compute=functions(modules['providers'])['compute_once']
    assert sum(isinstance(n,ast.Attribute) and n.attr=='gradient' for n in ast.walk(compute))==1
    assert sum(isinstance(n,ast.Attribute) and n.attr=='attention' for n in ast.walk(compute))==1
    assert not audit['scientific_qualified'] and not audit['target_runner_available']
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


@pytest.mark.parametrize('key',list(FILES))
def test_source_drift_refuses_before_building_any_runtime(key):
    data=dict(INPUTS);data[key]+=b'\n'
    with pytest.raises(ValueError,match='Frozen provider changed'):sources.build(data)


def fixture_metadata(rt):
    rows=[{'origin':'HANDWRITTEN-SOURCE-METADATA','index':i} for i in range(1152)]
    rows[24].update(owner='predictor.0',phase='EOT8-before',during_backward=False)
    scope={'input_id':1234,'state':'complete','hook_removed':True,'pid':os.getpid()}
    proof={'calls':rows,'scopes':[scope],'refusals':[]}
    worker={'kind':lineage.NUMERICAL,'control_sha256':CTRL,'execution_host':'linux10603','complete':True,'passed':True,
        'gradient_objectives_started':8,'candidate_points':4,'model_unchanged':True,'global_rng_unchanged':True,
        'all_methods_hooks_flags_restored':True,'research_admitted':False,'performance_qualified':False,'pid':os.getpid(),'baseline_sha256':'5'*64}
    original={'kind':lineage.OBJECTIVE,'index':0,'step':0,'mode':'original','local_tiny':False,'execution_host':'linux10603',
        'control_sha256':CTRL,'baseline_sha256':'5'*64,'pid':os.getpid(),'complete':True,'passed':True,'model_unchanged':True,
        'adapter_restored':True,'parameter_gradients_absent':True,'input_dot_grad_absent':True,'research_admitted':False}
    cap={'kind':lineage.BOUNDARY_KIND,'control_sha256':CTRL,'execution_host':'linux10603','local_tiny':False,'pid':os.getpid(),
        'mode':'original','capture_complete':True,'observer_restored':True,'alias_restored':True,'tensor_versions_unchanged':True,
        **dict.fromkeys(('native_forward_modified','native_cotangent_replaced','native_full_model_qualified','CUDA_kernel_qualified','research_admitted','performance_qualified'),False),
        'split_lineage':{'kind':lineage.LINEAGE_KIND,'source_objective_index':0,'source_value_id':1234,'source_value_version':0,
            'source_input_signature':{'shape':[1,3,16,256,256],'dtype':'torch.float32','sha256':PIXEL},'frozen_observer_sha256':lineage.OBSERVER_SHA},
        'call_rows':copy.deepcopy(rows),'backward_scopes':copy.deepcopy(proof['scopes']),'ownership_refusals':[],
        'selected':{'owner':'predictor.0','draw':0,'family':0,'forward_row_index':24,'phase':'EOT8-before'},
        'selected_call':{'actual_call':copy.deepcopy(rows[24])},'artifacts':{},'source_storage':{}}
    tensors=dict(zip(('q','k','v'),rt.values),grad_output=rt.incoming)
    for name,t in tensors.items():
        digest=hashlib.sha256(t.data).hexdigest();group=0 if name!='grad_output' else 2
        span=4*(1+sum((n-1)*s for n,s in zip(t.shape,t.stride())))
        cap['artifacts'][name]={'file':name+'.pt','bytes':32,'sha256':'7'*64,'saved_before_numerical_gate':True,
            'stored_device':'cpu','logical_copy_is_contiguous':True,'native_storage_alias_preserved_in_file':False,
            'source':{'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),'device':str(t.device),'sha256':digest}}
        cap['source_storage'][name]={'alias_group':group,'storage_bytes':t.untyped_storage().nbytes(),
            'storage_offset_elements':t.storage_offset(),'version':t._version,'source_logical_hash':digest,
            'requires_grad':name!='grad_output','positive_stride_span_bytes':span}
    return worker,original,cap,proof


def joined(data):return lineage.join(*data,control_sha256=CTRL,boundary_sha256=BOUNDARY,pixel_sha256=PIXEL)


@pytest.mark.parametrize('fault',['old-worker','old-original','old-boundary','worker-incomplete','baseline','pid','bool-pid','wrong-control',
 'candidate-source','value-id','version','pixel-hash','old-observer','call-row','scope','selected',
 'dtype','shape-bool','stride','offset','backing','alias-extent','logical-hash','file','file-size','requires-grad'])
def test_new_input_join_refuses_historical_or_changed_source_before_tensor_allocation(fake_helpers,fault):
    ns,_=fake_helpers;rt=ns['Runtime']();data=fixture_metadata(rt);w,o,c,p=data
    if fault=='old-worker':w['kind']='pretrained-pgd2-r2-iterated-worker-v1'
    elif fault=='old-original':o['kind']='pretrained-pgd2-r2-owned-objective-v1'
    elif fault=='old-boundary':c['kind']='original-SDPA-one-boundary-observation-v1'
    elif fault=='worker-incomplete':w['passed']=False
    elif fault=='baseline':o['baseline_sha256']='9'*64
    elif fault=='pid':c['pid']+=1
    elif fault=='bool-pid':c['pid']=True
    elif fault=='wrong-control':o['control_sha256']='9'*64
    elif fault=='candidate-source':c['split_lineage']['source_objective_index']=1
    elif fault=='value-id':c['split_lineage']['source_value_id']+=1
    elif fault=='version':c['split_lineage']['source_value_version']=-1
    elif fault=='pixel-hash':c['split_lineage']['source_input_signature']['sha256']='9'*64
    elif fault=='old-observer':c['split_lineage']['frozen_observer_sha256']='9'*64
    elif fault=='call-row':c['call_rows'][-1]['origin']='different'
    elif fault=='scope':c['backward_scopes'][0]['state']='failed'
    elif fault=='selected':c['selected']['draw']=1
    else:
        d=c['artifacts']['q'];s=d['source'];g=c['source_storage']['q']
        if fault=='dtype':s['dtype']='torch.float64'
        elif fault=='shape-bool':s['shape'][0]=True
        elif fault=='stride':s['stride'][1]=0
        elif fault=='offset':g['storage_offset_elements']=2**24
        elif fault=='backing':g['storage_bytes']=16*2**20+4
        elif fault=='alias-extent':g['storage_bytes']+=4
        elif fault=='logical-hash':g['source_logical_hash']='8'*64
        elif fault=='file':d['file']='../historical/q.pt'
        elif fault=='file-size':d['bytes']=4*2**20+1
        else:g['requires_grad']=False
    with pytest.raises(ValueError):joined(data)
    assert not rt.events and not rt.grad_entries


class Clock:
    def __init__(self):self.value=10.
    def __call__(self):self.value+=.001;return self.value


class ProbeFixture:
    def __init__(self,built,helpers,tmp,role,*,fault=None):
        ns,trace_fixture=helpers;self.rt=ns['Runtime'](outer_thread=True,inner_thread=True,fault='engine-error' if fault=='backward' else None)
        self.ns,_=ns['namespace'](self.rt);self.clock=Clock();self.fault=fault;self.calls=[];self.closed_profile=False
        self.rt.torch.cuda.synchronize=lambda:self.calls.append('cuda-sync-stub')
        tmp.chmod(0o700);self.store=typed.Store(typed.Inventory(tmp,base_bytes=BASE,design_bytes=DESIGN))
        self.role=role;self.binding={'design_sha256':lineage.DESIGN_SHA,'control_sha256':CTRL,'boundary_sha256':BOUNDARY,'probe_index':0 if role=='native' else 1}
        self.data=fixture_metadata(self.rt);self.cap=self.data[2];self.join=joined(self.data)
        self.trace=trace_fixture(role)
        if fault=='semantic':next(r for r in self.trace['records'] if r['device']=='cuda')['linked_correlation_id']=0
        engine=self
        class Profile:
            def __enter__(self):
                engine.calls.append('profiler_start')
                if fault=='profiler_start':raise RuntimeError('SIMULATED profiler_start')
                return self
            def __exit__(self,*exc):
                engine.calls.append('profiler_stop');engine.closed_profile=True
                if fault=='profiler_stop':raise RuntimeError('SIMULATED profiler_stop')
                return False
        @contextmanager
        def record(label):self.calls.append(label);yield
        compute=next(n for n in built[0]['providers'].body if isinstance(n,ast.FunctionDef) and n.name=='compute_once')
        self.compute_ns={'torch':self.rt.torch}
        exec(compile(ast.Module(body=[compute],type_ignores=[]),'<COMPUTE-ON-FAKE-ENGINE-ONLY>','exec'),self.compute_ns)
        raw_ns={'__name__':'EXPLICIT_FAKE_KINETO_METADATA_EXTRACTOR'}
        exec(compile(built[0]['raw'],'<NEW-RAW-METADATA-EXTRACTOR-NO-PROFILER>','exec'),raw_ns)
        def startup():
            self.calls.append('startup')
            if fault=='startup':raise RuntimeError('SIMULATED startup')
            return {'kind':'native-SDPA-component-startup-v2','pid':os.getpid(),'configure_calls':1,'cuda_initialized_before':False,
                'cuda_initialized_after':True,'allocator_cap_bytes':7*2**30}
        def source():
            self.calls.append('source_verify')
            if fault=='source':raise RuntimeError('SIMULATED source')
            return self.cap,self.join
        def inputs(cap):
            self.calls.append('operands');metadata={}
            for name,w in self.join['layout']['operands'].items():
                metadata[name]={'shape':w['shape'],'stride':w['stride'],'sha256':w['logical_sha256'],'dtype':'torch.float32','device':'cuda:0',
                    **{k:w[k] for k in ('storage_offset_elements','storage_bytes','alias_group','requires_grad')},
                    'unknown_padding_filled_zero':True,'native_physical_alignment_identity_claimed':False}
            if fault=='layout':metadata['k']['storage_offset_elements']+=1
            return self.rt.values,self.rt.incoming,metadata
        def compute(route,observe):
            if fault=='forward':
                def refused():raise RuntimeError('SIMULATED forward')
                return observe('forward',refused)
            return self.compute_ns['compute_once'](route,record,observe)
        class Event:
            def __init__(self,row):self.row=row
            def __getattr__(self,method):
                if method=='device_type':return lambda:'DeviceType.'+('CPU' if self.row['device']=='cpu' else 'CUDA')
                field=next(k for k,v in raw_ns['frozen'].METHODS.items() if v==method)
                return lambda:self.row[field]
        def raw(profile):
            assert self.closed_profile,'Raw materialization must follow actual fake profiler exit.'
            self.calls.append('raw_extract')
            if fault=='raw':raise RuntimeError('SIMULATED raw extraction')
            if fault=='deadline_raw':self.clock.value+=121
            if fault=='raw_size':return b'{'*(4*2**20+1)
            result=NS(events=lambda:[Event(r) for r in self.trace['records']],trace_start_ns=lambda:self.trace['trace_start_ns'])
            p=NS(profiler=NS(kineto_results=result))
            return backend.encoded(raw_ns['capture'](p,torch_version=backend.RUNTIME))
        def restore():
            self.calls.append('restore')
            if fault=='restore':raise RuntimeError('SIMULATED restoration')
            assert not self.rt.live and self.rt.flags==self.rt.initial
        self.ports={'startup':startup,'source':source,'operands':inputs,
            'route':lambda role,values,incoming,binding:self.ns['ComponentRoute'](role,values,incoming,binding=binding),
            'profile':Profile,'compute':compute,'raw':raw,'serialize':lambda t:b'OPAQUE COMPONENT RETURN: '+t.data,
            'restore':restore,'allocator':lambda:7*2**30+1 if fault=='allocator' else 512*2**20}
        self.job=job.ComponentJob(role,self.binding,self.store,self.ports,deadline=self.clock.value+120,clock=self.clock,
            evidence_origin='OPAQUE-COMPONENT-INTEGRATION-SIMULATION')
    def validate(self):
        folder=self.job.folder
        return job.read_completion(folder,role=self.role,binding=self.binding,
            marker_sha256=hashlib.sha256((folder/'complete.json').read_bytes()).hexdigest(),
            progress_sha256=hashlib.sha256((folder/'progress.jsonl').read_bytes()).hexdigest())


@pytest.mark.parametrize('role',['native','forward-preserving'])
def test_complete_component_lifecycle_joins_frozen_route_raw_reader_store_and_progress(built,fake_helpers,tmp_path,role):
    p=ProbeFixture(built,fake_helpers,tmp_path,role);marker=p.job.execute()
    assert marker['complete'] and not marker['scientific_qualified']
    assert p.validate()=={'metadata_complete':True,'scientific_qualified':False,'actual_CUDA_execution_admitted':False}
    assert len(p.rt.events)==(1 if role=='native' else 2) and len(p.rt.grad_entries)==(1 if role=='native' else 2)
    assert p.calls.count('startup')==1 and p.calls.count('cuda-sync-stub')==1 and p.closed_profile and not p.rt.live
    assert p.calls.index('profiler_stop')<p.calls.index('raw_extract')
    ownership=json.loads((p.job.folder/'ownership.json').read_bytes())
    assert ownership['route_complete'] and ownership['outer']['root_calls']==1 and len(ownership['outer']['leaf_hooks'])==3
    if role!='native':assert ownership['calls'][0]['inner_graph_id']!=ownership['outer']['graph_id']
    assert p.job.log.closed and p.store.check()['scientific_qualified'] is False
    with pytest.raises(ValueError,match='no retry'):p.job.execute()
    (tmp_path.parent/(tmp_path.name+'-SIMULATION-REVIEW.json')).write_text(json.dumps({'origin':'OPAQUE-COMPONENT-INTEGRATION-SIMULATION','role':role,
        'forward_stubs':len(p.rt.events),'VJP_stubs':len(p.rt.grad_entries),'metadata_complete':True,'scientific_qualified':False})+'\n')


@pytest.mark.parametrize('fault',['startup','source','layout','profiler_start','forward','backward','profiler_stop','raw',
 'raw_size','semantic','restore','allocator','deadline_raw','journal_before_backward','journal_after_marker'])
def test_component_failures_stop_publication_and_never_repeat_a_compute(built,fake_helpers,tmp_path,monkeypatch,fault):
    p=ProbeFixture(built,fake_helpers,tmp_path,'forward-preserving',fault=fault)
    if fault=='journal_before_backward':
        original=job.Stages.write
        def fail(log,phase,event,error=None):
            if phase=='backward' and event=='begin':raise OSError('SIMULATED journal before backward')
            return original(log,phase,event,error)
        monkeypatch.setattr(job.Stages,'write',fail)
    if fault=='journal_after_marker':
        original=job.Stages.end
        def fail(log,phase,error=None):
            if phase=='complete_publish' and error is None:raise OSError('SIMULATED journal after marker')
            return original(log,phase,error)
        monkeypatch.setattr(job.Stages,'end',fail)
    with pytest.raises((ValueError,RuntimeError,OSError,TimeoutError)) as caught:p.job.execute()
    messages=[];error=caught.value;seen=set()
    while error is not None and id(error) not in seen:
        seen.add(id(error));messages.append(str(error));error=error.__context__
    needles={'startup':'SIMULATED startup','source':'SIMULATED source','layout':'Canonical reconstructed','profiler_start':'SIMULATED profiler_start',
        'forward':'SIMULATED forward','backward':'SIMULATED engine error','profiler_stop':'SIMULATED profiler_stop','raw':'SIMULATED raw extraction',
        'raw_size':'Whole 4 MiB','restore':'SIMULATED restoration','allocator':'Component allocator cap','deadline_raw':'deadline exhausted',
        'journal_before_backward':'SIMULATED journal before backward','journal_after_marker':'SIMULATED journal after marker'}
    if fault!='semantic':assert needles[fault] in '\n'.join(messages),messages
    else:assert 'backend_validate' in p.job.log.raised and (p.job.folder/'events.json').exists() and not (p.job.folder/'backend.json').exists()
    assert len(p.rt.events)<=2 and len(p.rt.grad_entries)<=2 and p.job.used
    assert p.job.log is None or p.job.log.closed
    assert (p.job.folder/'failed.json').exists()
    if fault=='journal_after_marker':assert (p.job.folder/'complete.json').exists()
    else:assert not (p.job.folder/'complete.json').exists()
    if fault in ('forward','backward','journal_before_backward'):assert p.closed_profile
    if fault in ('raw','raw_size','semantic','restore','allocator','deadline_raw','journal_after_marker'):assert p.closed_profile and not p.rt.live
    if fault in ('startup','source','layout','profiler_start','forward'):assert not p.rt.grad_entries
    with pytest.raises(ValueError,match='no retry'):p.job.execute()


@pytest.mark.parametrize('change',['missing-journal-return','wrong-marker-hash','changed-array','descriptor-traversal'])
def test_terminal_marker_does_not_bypass_progress_or_bound_artifact_receipts(built,fake_helpers,tmp_path,change):
    p=ProbeFixture(built,fake_helpers,tmp_path,'native');p.job.execute();folder=p.job.folder
    if change=='missing-journal-return':
        f=folder/'progress.jsonl';f.write_bytes(b'\n'.join(f.read_bytes().splitlines()[:-1])+b'\n')
    elif change=='changed-array':(folder/'dq.npy').write_bytes(b'CHANGED OPAQUE SIMULATION')
    elif change=='descriptor-traversal':
        f=folder/'complete.json';d=json.loads(f.read_bytes());d['worker']['file']='../historical/worker.json';f.write_bytes(job.encoded(d))
    if change=='wrong-marker-hash':
        with pytest.raises(ValueError):job.read_completion(folder,role=p.role,binding=p.binding,marker_sha256='0'*64,
            progress_sha256=hashlib.sha256((folder/'progress.jsonl').read_bytes()).hexdigest())
    else:
        with pytest.raises(ValueError):p.validate()


@pytest.mark.parametrize('fault',['none','pid','RSS','wall','sampling-claim'])
def test_parent_metadata_must_join_same_child_resource_and_unextended_wall(built,fake_helpers,tmp_path,fault):
    p=ProbeFixture(built,fake_helpers,tmp_path,'native');marker=p.job.execute();folder=p.job.folder
    w=json.loads((folder/'worker.json').read_bytes());r=json.loads((folder/'resource.json').read_bytes())
    observed={'kind':'pgd-split-component-parent-observation-v1','pid':w['pid'],'start_ticks':71,'exit_status':0,
        'parent_observation_passed':True,'RSS_samples':3,'peak_observed_RSS_bytes':2**30,'observed_process_identities':1,
        'wall_seconds':r['wall_seconds']+.1,'sampling_is_instantaneous_kernel_or_address_space_quota':False,
        'proc_child_enumeration_is_exhaustive':False,'scientific_qualified':False,'actual_CUDA_execution_admitted':False}
    if fault=='pid':observed['pid']+=1
    elif fault=='RSS':observed['peak_observed_RSS_bytes']=4*2**30+1
    elif fault=='wall':observed['wall_seconds']=121
    elif fault=='sampling-claim':observed['proc_child_enumeration_is_exhaustive']=True
    if fault!='none':
        with pytest.raises(ValueError):job.bind_parent_metadata(marker,w,r,observed,role=p.role,binding=p.binding)
    else:
        result=job.bind_parent_metadata(marker,w,r,observed,role=p.role,binding=p.binding)
        assert result['metadata_bindings_consistent'] and not result['actual_CUDA_execution_admitted'] and not result['scientific_qualified']
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


@pytest.mark.parametrize('change',['same','different','missing-key'])
def test_native_attention_flag_identity_is_checked_before_any_input_reconstruction(change,tmp_path):
    t=ast.parse(sources.PROBE_SOURCE)
    function=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_operands')
    calls=[];original={'flash':True,'mem_efficient':True,'math':True,'cudnn':True}
    actual=dict(original)
    if change=='different':actual['mem_efficient']=False
    elif change=='missing-key':actual.pop('cudnn')
    q,k,v,cot=object(),object(),object(),object()
    def rebuild(parent,cap,device):
        calls.append((parent,device));return {'q':q,'k':k,'v':v,'grad_output':cot},{'origin':'OPAQUE-METADATA-STUB'}
    ns={'flags':lambda:actual,'_operands':rebuild,'Path':Path}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<NATIVE-FLAG-GATE-NO-TENSOR-RECONSTRUCTION>','exec'),ns)
    cap={'selected_call':{'actual_call':{'entry_flags':original}}}
    if change=='same':
        values,incoming,metadata=ns['prepare_operands'](NS(result_root=str(tmp_path)),cap)
        assert values==(q,k,v) and incoming is cot and calls==[(tmp_path/'results','cuda')]
    else:
        with pytest.raises(ValueError,match='flags changed'):ns['prepare_operands'](NS(result_root=str(tmp_path)),cap)
        assert not calls
