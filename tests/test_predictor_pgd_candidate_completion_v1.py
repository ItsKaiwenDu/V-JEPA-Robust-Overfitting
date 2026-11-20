"""New cold/byte/metadata and explicit fake-port cases, never tensor execution."""
import ast
import collections
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path,PurePosixPath
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import threading
from argparse import Namespace
from types import SimpleNamespace as NS

import pytest
from models import predictor_pgd_candidate_completion_v1 as adapter
from models import predictor_pgd_candidate_reference_v1 as reference
from models import predictor_pgd_component_capsule_v1 as capsule
from models import predictor_pgd_split_backend_v4 as backend
from models import predictor_pgd_split_lineage_v1 as lineage

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'runs/pretrained/diagnostic/pgd-candidate-completion-v1-20261120'
ORIGIN='EXPLICIT-FAKE-CANDIDATE-COMPLETION; NO EXECUTION ADMISSION'
PREFIX='models/_pgd_component_'


@pytest.fixture(autouse=True)
def no_scientific_or_target_work(monkeypatch):
    def forbidden(*a,**k):pytest.fail('Adapter checks cannot run process/GPU/signal/timer/network work.')
    for mod,name in [(subprocess,'Popen'),(subprocess,'check_output'),(os,'system'),(os,'fork'),(os,'posix_spawn'),(os,'kill'),(os,'killpg'),(os,'pidfd_open'),(signal,'alarm'),(signal,'setitimer'),(signal,'signal'),(socket,'socket')]:
        if hasattr(mod,name):monkeypatch.setattr(mod,name,forbidden)
    yield
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


def save(folder,name,value):
    raw=value if type(value) is bytes else adapter.encoded(value)
    with os.fdopen(os.open(folder/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(raw)


def namespace(data,provided):
    tree=ast.parse(data);tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    ns={'__name__':'EXPLICIT_FAKE_CANDIDATE_COMPLETION',**provided}
    exec(compile(ast.fix_missing_locations(tree),'<EXPLICIT-FAKE-PORTS-AND-OPAQUE-BYTES>','exec'),ns)
    return ns


@pytest.fixture(scope='module')
def built():
    m=json.loads((STAGE/'SOURCE-MANIFEST-v1.json').read_bytes());files={n:(STAGE/'cold-source-v1'/n).read_bytes() for n in m['files']}
    assert all(adapter.digest(raw)==m['files'][n]['sha256'] for n,raw in files.items())
    return files,m,json.loads((STAGE/'NATIVE-REFERENCE-v1.json').read_bytes())


def unpack(raw,folder):
    folder.mkdir(mode=0o700)
    import io
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tf:
        for item in tf:
            assert item.isfile() and not item.name.startswith('/') and '..' not in PurePosixPath(item.name).parts
            p=folder/item.name;p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(tf.extractfile(item).read())
    return folder


@pytest.fixture(scope='module')
def references(tmp_path_factory):
    base=tmp_path_factory.mktemp('read-only-source-and-native-reference');base.chmod(0o700)
    c=json.loads((ROOT/adapter.INPUTS).read_bytes())
    return unpack((ROOT/c['unchanged_capsule_archive']['path']).read_bytes(),base/'source'),unpack((STAGE/'native-reference-v1.tar').read_bytes(),base/'native-reference')


def test_cold_graph_source_inverse_and_exact_scientific_preservation(built,tmp_path):
    files,m,spec=built;c=json.loads((ROOT/adapter.INPUTS).read_bytes())
    assert len(files)==39 and len(spec['files'])==18 and m['execution_binding'] is None
    assert not set(files)&adapter.FORBIDDEN and adapter.READER in files and 'models/predictor_pgd_split_backend_v3.py' not in files
    for n,e in m['reversible_source_edits'].items():
        old=adapter.inverse(files[n],e['edits'])
        expected=(ROOT/n).read_bytes() if n==adapter.REF else (ROOT/c['runtime_files'][n]['path']).read_bytes()
        assert old==expected and adapter.digest(old)==e['original_sha256']
    for n,d in c['runtime_files'].items():
        if n in files and n not in (adapter.JOB,adapter.ENTRY,adapter.STORE):assert files[n]==(ROOT/d['path']).read_bytes()
    for n,v in files.items():
        compile(v,'<UNEXECUTED-SOURCE>','exec');assert all(dep in files for dep in m['imports'][n])
    save(tmp_path,'COLD-SOURCE-PROOF.json',{'files':m['files'],'changed':list(m['reversible_source_edits']),'all_byte_inverses':True,'scientific_execution':False})


def test_original_numerical_decoder_and_bounds_AST(built):
    files,_,_=built;old=ast.parse((ROOT/'models/predictor_pgd_component_returns_v1.py').read_bytes());new=ast.parse(files[adapter.ACCURACY])
    for name in ['decode_npy','agreement']:
        fn=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(fn(old),include_attributes=False)==ast.dump(fn(new),include_attributes=False)
    select=lambda t:next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='BOUNDS' for x in n.targets))
    assert ast.dump(select(old),include_attributes=False)==ast.dump(select(new),include_attributes=False)


def test_reference_archive_preserves_actual_bytes_without_value_decode(built,tmp_path):
    _,_,spec=built;raw=(STAGE/'native-reference-v1.tar').read_bytes();members=reference.inspect_archive(raw,spec)
    assert len(members)==18 and all(reference.digest(v)==spec['files'][n]['sha256'] for n,v in members.items())
    assert reference.archive(members,spec)==raw
    save(tmp_path,'REFERENCE-BYTES.json',{'archive_bytes':len(raw),'archive_sha256':reference.digest(raw),'members':len(members),'actual_array_or_tensor_values_decoded':False})


@pytest.mark.parametrize('case',['missing-return','parent-failure','changed-native-control','changed-boundary'])
def test_reference_lineage_refuses(built,tmp_path,case):
    _,_,spec=built;members=reference.inspect_archive((STAGE/'native-reference-v1.tar').read_bytes(),spec);descriptors=copy.deepcopy(spec['files'])
    if case=='missing-return':del members['native/dq.npy']
    else:
        name={'parent-failure':'native/parent.json','changed-native-control':'native/worker.json','changed-boundary':'source/capsule-manifest.json'}[case]
        value=json.loads(members[name])
        if case=='parent-failure':value['exit_status']=1
        elif case=='changed-native-control':value['binding']['control_sha256']='0'*64
        else:value['boundary_sha256']='0'*64
        members[name]=reference.encoded(value);descriptors[name]={'bytes':len(members[name]),'sha256':reference.digest(members[name])}
    with pytest.raises(ValueError) as error:reference.describe_members(members,descriptors)
    save(tmp_path,'REFERENCE-REFUSAL.json',{'origin':ORIGIN,'case':case,'error':str(error.value),'numerical_values_decoded':False})


class Clock:
    def __init__(self):self.value=10.
    def __call__(self):self.value+=.001;return self.value

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
    def __init__(self,built,references,result_root,fault=None):
        files,_,_=built;self.files=files;self.calls=collections.Counter();self.clock=Clock();self.fault=fault;self.result_root=result_root;self.result_root.mkdir(mode=0o700)
        source_root,reference_root=references;common={'hashlib':hashlib,'json':json,'math':math,'os':os,'Path':Path,'stat':stat,'threading':threading}
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
            return {'kind':'native-SDPA-component-startup-v2','pid':os.getpid(),'configure_calls':1,'cuda_initialized_before':False,'cuda_initialized_after':True,'allocator_cap_bytes':7*2**30}
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


def test_candidate_lifecycle_and_new_parent_join_with_explicit_fake_ports(built,references,tmp_path):
    h=Harness(built,references,tmp_path/'results');marker=h.run();folder=h.result_root/'results/components/forward-preserving'
    assert marker['complete'] and marker['kind']=='pgd-candidate-component-complete-v1'
    assert h.calls['forward']==h.calls['outer_VJP']==h.calls['inner_math_VJP']==h.calls['fake_accuracy']==1
    assert h.calls['profile_start']==h.calls['profile_stop']==h.calls['route_exit']==1
    worker=json.loads((folder/'worker.json').read_bytes());resource=json.loads((folder/'resource.json').read_bytes());binding=worker['binding']
    check=h.job['read_completion'](folder,role='forward-preserving',binding=binding,marker_sha256=adapter.digest((folder/'complete.json').read_bytes()),progress_sha256=adapter.digest((folder/'progress.jsonl').read_bytes()))
    assert check['metadata_complete'] and not check['actual_CUDA_execution_admitted']
    observed={'kind':'pgd-split-component-parent-observation-v1','pid':worker['pid'],'start_ticks':12345,'exit_status':0,'parent_observation_passed':True,'RSS_samples':3,'peak_observed_RSS_bytes':512*2**20,'observed_process_identities':1,'wall_seconds':resource['wall_seconds']+1.,'sampling_is_instantaneous_kernel_or_address_space_quota':False,'proc_child_enumeration_is_exhaustive':False,'scientific_qualified':False,'actual_CUDA_execution_admitted':False}
    parent=h.job['bind_parent_metadata'](marker,worker,resource,observed,role='forward-preserving',binding=binding)
    assert parent['metadata_bindings_consistent'] and not parent['actual_CUDA_execution_admitted'] and not parent['training_admitted']
    for name in ['scientific_qualified','component_accuracy_qualified','full_EOT8_backend_qualified','training_admitted']:assert worker[name] is False
    assert worker['saved_native_relative_values_passed'] and worker['evidence_origin']==ORIGIN
    assert not (h.result_root/'results/components/native').exists() and not (h.result_root/'results/numerical').exists()
    save(tmp_path,'SIMULATED-SUCCESS.json',{'origin':ORIGIN,'counts':dict(h.calls),'worker_metadata_complete':True,'completion_reader':check,'fake_parent':observed,'parent_join':parent,'actual_tensor_array_GPU_profile_gradient_or_parent_observation':False})


@pytest.mark.parametrize('fault',['reference','startup','backend','accuracy','forward','restoration','allocator','deadline'])
def test_failure_preserves_evidence_and_stops_without_retry(built,references,tmp_path,fault):
    h=Harness(built,references,tmp_path/'results',fault)
    with pytest.raises((ValueError,TimeoutError)) as error:h.run()
    folder=h.result_root/'results/components/forward-preserving';assert (folder/'failed.json').exists() and not (folder/'complete.json').exists()
    assert h.calls['forward']<=1 and h.calls['outer_VJP']<=1 and h.calls['inner_math_VJP']<=1
    if fault in ['reference','startup']:assert h.calls['forward']==h.calls['fake_accuracy']==0
    else:assert (folder/'events.json').exists() and h.calls['profile_stop']==h.calls['route_exit']==1
    if fault=='backend':assert h.calls['fake_accuracy']==0
    with pytest.raises(ValueError,match='One component entry'):h.run()
    save(tmp_path,'SIMULATED-FAILURE.json',{'origin':ORIGIN,'fault':fault,'error':str(error.value),'counts':dict(h.calls),'retry_refused':True,'actual_scientific_or_target_work':False})


@pytest.mark.parametrize('fault',['unbound','native-role','historical-control','extra-numerical-argument','wrong-host','overlap-reference','wrong-reference-manifest'])
def test_entry_refuses_unbound_old_native_or_foreign_identity_before_ports(built,references,tmp_path,fault):
    h=Harness(built,references,tmp_path/'results')
    if fault=='unbound':h.entry['EXECUTION_BINDING']=None
    elif fault=='native-role':h.args.role='native'
    elif fault=='historical-control':h.args.control_sha256=built[2]['native_control_sha256'];h.entry['EXECUTION_BINDING']['control_sha256']=h.args.control_sha256
    elif fault=='extra-numerical-argument':h.args.numerical_sha256='f'*64
    elif fault=='wrong-host':h.entry['EXECUTION_BINDING']['host']='OTHER-FAKE-HOST'
    elif fault=='overlap-reference':h.args.reference_root=h.args.result_root;h.entry['EXECUTION_BINDING']['reference_root']=h.args.reference_root
    else:h.entry['EXECUTION_BINDING']['reference_manifest_sha256']='f'*64
    with pytest.raises(ValueError) as error:h.run()
    assert h.calls['startup']==h.calls['forward']==h.calls['fake_accuracy']==0
    assert list(h.result_root.iterdir())==[]
    save(tmp_path,'ENTRY-REFUSAL.json',{'origin':ORIGIN,'fault':fault,'error':str(error.value),'ports_started':0})


def test_inventory_is_candidate_only_and_accuracy_bound_is_metadata(built,references,tmp_path):
    h=Harness(built,references,tmp_path/'results');store=h.io['bind'](h.result_root);spec=store.inventory.spec
    assert len(spec)==15 and set(spec)=={n for n in spec if n.startswith('results/components/forward-preserving/') or n=='logs/forward-preserving.log'}
    assert spec['results/components/forward-preserving/accuracy.json']=={'category':'metadata','max_bytes':16*1024,'mutable':False,'partial_allowed':True}
    for name in ['results/components/native/output.npy','results/numerical/objective-0/loss.npy','results/boundary/q.pt']:
        with pytest.raises(ValueError):store.write_bytes(name,b'EXPLICIT-FAKE')
    save(tmp_path,'CANDIDATE-INVENTORY.json',{'paths':spec,'full_objectives_and_native_reruns':0,'origin':ORIGIN})


def test_parent_join_refuses_native_and_over_limit_observations(built,references,tmp_path):
    h=Harness(built,references,tmp_path/'results');marker=h.run();folder=h.result_root/'results/components/forward-preserving';w=json.loads((folder/'worker.json').read_bytes());res=json.loads((folder/'resource.json').read_bytes())
    o={'kind':'pgd-split-component-parent-observation-v1','pid':w['pid'],'start_ticks':123,'exit_status':0,'parent_observation_passed':True,'RSS_samples':1,'peak_observed_RSS_bytes':4*2**30+1,'observed_process_identities':1,'wall_seconds':res['wall_seconds']+1.,'sampling_is_instantaneous_kernel_or_address_space_quota':False,'proc_child_enumeration_is_exhaustive':False,'scientific_qualified':False,'actual_CUDA_execution_admitted':False}
    with pytest.raises(ValueError):h.job['bind_parent_metadata'](marker,w,res,o,role='forward-preserving',binding=w['binding'])
    with pytest.raises(ValueError):h.job['bind_parent_metadata'](marker,w,res,o,role='native',binding=w['binding'])
    save(tmp_path,'PARENT-REFUSAL.json',{'origin':ORIGIN,'native_role_and_RSS_overrun_refused':True,'actual_parent_observation':False})
