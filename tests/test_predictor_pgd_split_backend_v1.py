"""Handwritten raw metadata only; no tensor/model/autograd/profiler/GPU work."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from models import predictor_pgd_split_backend_v1 as r
from scripts import check_predictor_pgd_split_backend_v1 as cli

ROOT=Path(__file__).resolve().parents[1]
ORIGIN=1_700_000_000_000_000_000


def binding(role):
    return {'design_sha256':r.DESIGN_SHA,'control_sha256':'2'*64,'boundary_sha256':'3'*64,
            'probe_index':0 if role=='native' else 1}


def assess(trace,role='forward-preserving'):
    return r.assess_bytes(r.encoded(trace),role=role,binding=binding(role))


def fixture(role='forward-preserving',*,outer=8,inner=9):
    """Times, IDs, threads and kernel names are synthetic, never CUDA evidence."""
    rows=[]
    def event(name,start,end,*,thread=7,sequence=-1,fwd=0,scope=0,annotation=False,device='cpu',link=0):
        i=len(rows)
        row={'index':i,'name':name,'device':device,'device_index':17 if device=='cpu' else 0,
             'resource_id':thread if device=='cpu' else 5,'thread':thread,'end_thread':thread,
             'fwd_thread':fwd,'sequence_nr':sequence,'scope':scope,
             'start_ns':ORIGIN+start*1000,'end_ns':ORIGIN+end*1000,
             'correlation_id':100+i,'linked_correlation_id':link,'is_async':False,
             'user_annotation':annotation,'legacy_cuda_elapsed_us':-1}
        rows.append(row);return row
    def kernel(name,owner,duration=3):
        return event(name,1100+len(rows)*10,1100+len(rows)*10+duration,device='cuda',thread=1,link=owner['correlation_id'])
    event(r.FORWARD,0,100,scope=7,annotation=True)
    if role=='forward-preserving':event(r.CUSTOM,5,95,sequence=10)
    sequence=11 if role=='forward-preserving' else 23
    event('aten::scaled_dot_product_attention',15,90,sequence=sequence)
    f=event(r.EFFICIENT,20,80,sequence=sequence);kernel('fmha_cutlassF_f32_SYNTHETIC',f)
    event(r.BACKWARD,200,950,scope=7,annotation=True)
    if role=='native':
        name='ScaledDotProductEfficientAttentionBackward0'
        event(r.ENGINE+name,210,930,thread=outer,sequence=23,fwd=7)
        event(name,220,920,thread=outer,sequence=23,fwd=7,scope=1)
        b=event(r.EFFICIENT_BACKWARD,230,900,thread=outer);kernel('fmha_cutlassB_f32_SYNTHETIC',b)
    else:
        event(r.ENGINE+r.CUSTOM_BACKWARD,210,930,thread=outer,sequence=10,fwd=7)
        event(r.CUSTOM_BACKWARD,220,920,thread=outer,sequence=10,fwd=7,scope=1)
        event(r.MATH,300,500,thread=outer,sequence=30)
        event('aten::matmul',310,350,thread=outer,sequence=31)
        a=event('aten::bmm',315,345,thread=outer,sequence=31);kernel('sgemm_SYNTHETIC',a)
        event('aten::softmax',360,410,thread=outer,sequence=32)
        a=event('aten::_softmax',365,405,thread=outer,sequence=32);kernel('softmax_warp_forward<SYNTHETIC>',a)
        event('aten::matmul',420,490,thread=outer,sequence=33)
        a=event('aten::bmm',425,485,thread=outer,sequence=33);kernel('sgemm_SYNTHETIC',a)
        for name,seq,start,end,op,kn in [
                ('BmmBackward0',33,520,620,'aten::bmm','sgemm_SYNTHETIC'),
                ('SoftmaxBackward0',32,640,740,'aten::_softmax_backward_data','softmax_warp_backward<SYNTHETIC>'),
                ('BmmBackward0',31,760,880,'aten::bmm','sgemm_SYNTHETIC')]:
            event(r.ENGINE+name,start,end,thread=inner,sequence=seq,fwd=outer)
            event(name,start+10,end-10,thread=inner,sequence=seq,fwd=outer,scope=1)
            a=event(op,start+20,end-20,thread=inner);kernel(kn,a)
    return {'kind':r.frozen.KIND,'source_api':r.frozen.API,'torch_version':r.RUNTIME,
            'trace_start_ns':ORIGIN,'records':rows}


def at(trace,name):return next(row for row in trace['records'] if row['name']==name)
def all_named(trace,name):return [row for row in trace['records'] if row['name']==name]
def append(trace,row):
    row=copy.deepcopy(row);row['index']=len(trace['records']);row['correlation_id']=10000+row['index'];trace['records'].append(row);return row


def saved(folder,trace):
    folder.mkdir(mode=0o700);p=folder/'events.json';data=r.encoded(trace);p.write_bytes(data);p.chmod(0o600)
    return p,hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize('outer,inner',[(7,7),(8,8),(8,9),(8,7),(7,9)])
def test_hybrid_separately_joins_outer_custom_and_all_inner_producers_without_rewriting_raw(outer,inner):
    trace=fixture(outer=outer,inner=inner);before=r.encoded(trace);proof=assess(trace)
    assert proof['operator_counts']=={'efficient_forward':1,'efficient_backward':0,'math_forward_in_backward':1}
    assert len(proof['producer_anchors'])==4
    outer_anchor=next(a for a in proof['producer_anchors'] if a['stage']=='custom-outer')
    assert (outer_anchor['forward_thread'],outer_anchor['backward_thread'],outer_anchor['sequence_nr'])==(7,outer,10)
    inside=[a for a in proof['producer_anchors'] if a['stage'].startswith('inner-')]
    assert {(a['forward_thread'],a['backward_thread']) for a in inside}=={(outer,inner)}
    assert {a['sequence_nr'] for a in inside}=={31,32,33} and all(a['sequence_nr']!=30 for a in inside)
    assert all(a['cross_thread']==(a['forward_thread']!=a['backward_thread']) for a in proof['producer_anchors'])
    assert proof['raw_records']==proof['projected_records']==len(trace['records'])
    assert r.encoded(trace)==before
    assert not proof['actual_CUDA_execution_admitted'] and not proof['component_root_leaf_or_GraphTask_ownership_verified']


@pytest.mark.parametrize('outer',[7,8])
def test_native_probe_has_one_producer_bound_efficient_backward(outer):
    trace=fixture('native',outer=outer);proof=assess(trace,'native')
    assert proof['operator_counts']=={'efficient_forward':1,'efficient_backward':1,'math_forward_in_backward':0}
    assert len(proof['producer_anchors'])==1
    assert proof['producer_anchors'][0]['producer_name']==r.EFFICIENT
    assert {g for w in proof['kernel_witnesses'] for g in w['groups']}=={'efficient-forward','efficient-backward'}


def test_equal_kernel_names_durations_and_GPU_correlation_ids_do_not_merge_launches(tmp_path):
    trace=fixture();devices=[v for v in trace['records'] if v['device']=='cuda']
    for row in devices:row['correlation_id']=777
    proof=assess(trace)
    gems=[w for w in proof['kernel_witnesses'] if w['name']=='sgemm_SYNTHETIC']
    assert len(gems)==4 and len({w['CUDA_event_index'] for w in gems})==4
    assert len({w['host_correlation_id'] for w in gems})==4
    assert {w['GPU_correlation_id'] for w in gems}=={777}
    assert {w['end_ns']-w['start_ns'] for w in gems}=={3000}
    for w in proof['kernel_witnesses']:
        assert trace['records'][w['CUDA_event_index']]['linked_correlation_id']==trace['records'][w['host_event_index']]['correlation_id']
    (tmp_path/'SYNTHETIC-trace.json').write_bytes(r.encoded(trace))
    (tmp_path/'SYNTHETIC-assessment.json').write_bytes(r.encoded(proof))


def test_enumeration_order_and_asynchronous_GPU_intervals_are_not_identity():
    trace=fixture();trace['records'].reverse()
    for i,row in enumerate(trace['records']):
        row['index']=i
        if row['device']=='cuda':row['start_ns']=ORIGIN+10_000_000+i*100;row['end_ns']=row['start_ns']+19
    proof=assess(trace)
    assert len(proof['producer_anchors'])==4
    assert all(w['start_ns']>at(trace,r.BACKWARD)['end_ns'] for w in proof['kernel_witnesses'])
    assert not proof['kernel_name_or_duration_used_as_identity']


def test_equal_sequence_number_on_different_source_threads_is_valid():
    trace=fixture(outer=8,inner=9)
    # The inner producer's 31 is replaced by outer's 10. Its source THREAD is 8,
    # while the actual custom forward's identical 10 belongs to thread 7.
    for row in trace['records']:
        if row['sequence_nr']==31:row['sequence_nr']=10
    proof=assess(trace)
    assert sum(a['sequence_nr']==10 for a in proof['producer_anchors'])==2
    assert {a['forward_thread'] for a in proof['producer_anchors'] if a['sequence_nr']==10}=={7,8}


@pytest.mark.parametrize('stage',['outer','inner'])
@pytest.mark.parametrize('fault',['sequence','forward_thread','engine_sequence','engine_thread','engine_name',
 'node_name','node_scope','node_annotation','missing_engine','duplicate_node','duplicate_engine',
 'duplicate_producer','wrong_producer','outside_container','wrong_source_pool'])
def test_missing_foreign_or_ambiguous_outer_and_inner_bindings_refuse(stage,fault):
    trace=fixture();node=at(trace,r.CUSTOM_BACKWARD if stage=='outer' else 'BmmBackward0')
    engine=at(trace,r.ENGINE+node['name']);producer=at(trace,r.CUSTOM) if stage=='outer' else all_named(trace,'aten::bmm')[1]
    if fault=='sequence':node['sequence_nr']+=100
    elif fault=='forward_thread':node['fwd_thread']+=100
    elif fault=='engine_sequence':engine['sequence_nr']+=100
    elif fault=='engine_thread':engine['thread']=engine['end_thread']=99
    elif fault=='engine_name':engine['name']='unrelated engine'
    elif fault=='node_name':node['name']='UnrelatedBackward0'
    elif fault=='node_scope':node['scope']=0
    elif fault=='node_annotation':node['user_annotation']=True
    elif fault=='missing_engine':engine['start_ns']=engine['end_ns']+1
    elif fault=='duplicate_node':
        row=append(trace,node);row['start_ns']+=1;row['end_ns']-=1
    elif fault=='duplicate_engine':
        row=append(trace,engine);row['start_ns']+=1;row['end_ns']-=1
    elif fault=='duplicate_producer':
        row=append(trace,producer);row['start_ns']=ORIGIN+101000;row['end_ns']=ORIGIN+102000
    elif fault=='wrong_producer':producer['name']='aten::unrelated'
    elif fault=='outside_container':engine['end_ns']=at(trace,r.BACKWARD if stage=='outer' else r.CUSTOM_BACKWARD)['end_ns']+100
    else:producer['sequence_nr']+=100
    before=r.encoded(trace)
    with pytest.raises(ValueError):assess(trace)
    assert r.encoded(trace)==before


@pytest.mark.parametrize('fault',['math_outside_custom','matrix_backward_in_caller_only','matrix_producer_outside_math',
 'extra_math','extra_efficient','efficient_backward_in_hybrid','all_math_forward','foreign_autograd_parent',
 'extra_matrix','extra_softmax','missing_matrix_backward','missing_softmax_backward','math_forward_only',
 'missing_custom_forward','missing_custom_backward','wrong_label_thread','overlapping_phases'])
def test_hybrid_cannot_use_broad_backward_time_or_wrong_recipe_to_qualify(fault):
    trace=fixture();math=at(trace,r.MATH)
    if fault=='math_outside_custom':
        # Different source thread gives no CPU ancestor relationship to custom.
        math['thread']=math['end_thread']=99
    elif fault=='matrix_backward_in_caller_only':
        all_named(trace,'aten::bmm')[-1]['thread']=all_named(trace,'aten::bmm')[-1]['end_thread']=7
    elif fault=='matrix_producer_outside_math':math['start_ns']=all_named(trace,'aten::bmm')[0]['end_ns']+1
    elif fault=='extra_math':row=append(trace,math);row['start_ns']+=1;row['end_ns']-=1
    elif fault=='extra_efficient':row=append(trace,at(trace,r.EFFICIENT));row['start_ns']+=1;row['end_ns']-=1
    elif fault=='efficient_backward_in_hybrid':all_named(trace,'aten::bmm')[-1]['name']=r.EFFICIENT_BACKWARD
    elif fault=='all_math_forward':at(trace,r.EFFICIENT)['name']=r.MATH
    elif fault=='foreign_autograd_parent':
        engine=at(trace,r.ENGINE+'BmmBackward0');row=append(trace,engine)
        row.update(name='ForeignBackward0',scope=1,start_ns=engine['start_ns']-1,end_ns=engine['end_ns']+1)
    elif fault=='extra_matrix':row=append(trace,all_named(trace,'aten::bmm')[0]);row.update(start_ns=ORIGIN+960000,end_ns=ORIGIN+970000,sequence_nr=-1)
    elif fault=='extra_softmax':row=append(trace,at(trace,'aten::_softmax'));row.update(start_ns=ORIGIN+960000,end_ns=ORIGIN+970000,sequence_nr=-1)
    elif fault=='missing_matrix_backward':all_named(trace,'aten::bmm')[-1]['name']='aten::unrelated'
    elif fault=='missing_softmax_backward':at(trace,'aten::_softmax_backward_data')['name']='aten::unrelated'
    elif fault=='math_forward_only':at(trace,'SoftmaxBackward0')['name']='UnrelatedBackward0'
    elif fault=='missing_custom_forward':at(trace,r.CUSTOM)['name']='unrelated forward'
    elif fault=='missing_custom_backward':at(trace,r.CUSTOM_BACKWARD)['name']='ForeignBackward0'
    elif fault=='wrong_label_thread':at(trace,r.BACKWARD)['thread']=at(trace,r.BACKWARD)['end_thread']=99
    else:at(trace,r.BACKWARD)['start_ns']=at(trace,r.FORWARD)['end_ns']-1
    with pytest.raises(ValueError):assess(trace)


@pytest.mark.parametrize('fault',['missing_link','wrong_owner','phase_owner','duplicate_CPU_ID','duplicate_GPU_row','foreign_device',
 'zero_kernel','missing_GEMM','missing_forward_softmax','missing_backward_softmax','GPU_annotation',
 'legacy_fallback','async_operator','unstable_operator_thread','operator_scope','operator_annotation',
 'unrelated_same_name_kernel','unknown_backend','CPU_ancestry_tie'])
def test_kernel_names_or_timing_cannot_repair_missing_links_or_corrupt_raw_schema(fault):
    trace=fixture();gpu=next(row for row in trace['records'] if row['device']=='cuda' and row['name']=='sgemm_SYNTHETIC')
    op=all_named(trace,'aten::bmm')[0]
    if fault=='missing_link':gpu['linked_correlation_id']=0
    elif fault=='wrong_owner':gpu['linked_correlation_id']=999999
    elif fault=='phase_owner':gpu['linked_correlation_id']=at(trace,r.BACKWARD)['correlation_id']
    elif fault=='duplicate_CPU_ID':at(trace,r.FORWARD)['correlation_id']=op['correlation_id']
    elif fault=='duplicate_GPU_row':
        row=copy.deepcopy(gpu);row['index']=len(trace['records']);trace['records'].append(row)
    elif fault=='foreign_device':gpu['device_index']=1
    elif fault=='zero_kernel':gpu['end_ns']=gpu['start_ns']
    elif fault=='missing_GEMM':gpu['name']='unrelated kernel'
    elif fault=='missing_forward_softmax':next(row for row in trace['records'] if row['name']=='softmax_warp_forward<SYNTHETIC>')['name']='unrelated kernel'
    elif fault=='missing_backward_softmax':next(row for row in trace['records'] if row['name']=='softmax_warp_backward<SYNTHETIC>')['name']='unrelated kernel'
    elif fault=='GPU_annotation':gpu['user_annotation']=True
    elif fault=='legacy_fallback':op['legacy_cuda_elapsed_us']=1
    elif fault=='async_operator':op['is_async']=True
    elif fault=='unstable_operator_thread':op['end_thread']=999
    elif fault=='operator_scope':op['scope']=1
    elif fault=='operator_annotation':op['user_annotation']=True
    elif fault=='unrelated_same_name_kernel':gpu['linked_correlation_id']=at(trace,r.FORWARD)['correlation_id']
    elif fault=='unknown_backend':at(trace,r.EFFICIENT)['name']='aten::_scaled_dot_product_cudnn_attention'
    else:row=append(trace,op);row['name']='unrelated same interval'
    with pytest.raises(ValueError):assess(trace)


def overhead(trace,**changes):
    row=copy.deepcopy(trace['records'][0]);row.update(index=len(trace['records']),name=r.frozen.OVERHEAD_NAME,
        device_index=-1,resource_id=0,thread=1,end_thread=1,fwd_thread=0,sequence_nr=-1,scope=0,
        start_ns=ORIGIN,end_ns=ORIGIN+2_000_000,correlation_id=99999,user_annotation=False)
    row.update(changes);trace['records'].append(row);return row


def gpu_annotation(trace):
    phase=at(trace,r.FORWARD);row=copy.deepcopy(next(v for v in trace['records'] if v['device']=='cuda'))
    row.update(index=len(trace['records']),name=r.FORWARD,thread=1,end_thread=1,fwd_thread=0,sequence_nr=-1,
        scope=0,correlation_id=phase['correlation_id'],linked_correlation_id=0,user_annotation=True)
    trace['records'].append(row);return row


def test_strict_overhead_and_GPU_annotations_stay_present_and_never_own_kernel_witnesses():
    trace=fixture();h=overhead(trace);g=gpu_annotation(trace);before=r.encoded(trace);proof=assess(trace)
    assert proof['retained_overhead_indices']==[h['index']] and proof['retained_GPU_annotation_indices']==[g['index']]
    assert all(w['CUDA_event_index']!=g['index'] and w['host_event_index']!=h['index'] for w in proof['kernel_witnesses'])
    assert not proof['overhead_or_annotation_qualifies_as_kernel'] and r.encoded(trace)==before


@pytest.mark.parametrize('fault',['unknown_overhead','overhead_owns_GPU','malformed_GPU_annotation','overhead_shadows_owner'])
def test_overhead_and_annotation_signature_cannot_supply_or_relabel_missing_proof(fault):
    trace=fixture();h=overhead(trace)
    if fault=='unknown_overhead':h['name']='unknown overhead'
    elif fault=='overhead_owns_GPU':next(v for v in trace['records'] if v['device']=='cuda')['linked_correlation_id']=h['correlation_id']
    elif fault=='malformed_GPU_annotation':gpu_annotation(trace)['name']='unknown GPU annotation'
    else:
        # An inherited ID may not override a genuine source, but cannot supply
        # an absent one either. Remove the real owner identity.
        f=at(trace,r.EFFICIENT);h['correlation_id']=f['correlation_id'];f['correlation_id']=888888
    with pytest.raises(ValueError):assess(trace)


@pytest.mark.parametrize('field,value',[('index',True),('correlation_id',False),('linked_correlation_id',-1),
 ('start_ns',ORIGIN-1),('end_ns',2**63),('sequence_nr',-2),('device_index',0.0),('scope','0'),('is_async',0)])
def test_inherited_strict_field_types_refuse(field,value):
    trace=fixture();trace['records'][0][field]=value
    with pytest.raises(ValueError):assess(trace)


@pytest.mark.parametrize('fault',['duplicate_key','NaN','trailing','old_processed_schema','runtime','count','byte_cap'])
def test_bounded_complete_JSON_and_pinned_runtime_are_required(fault,monkeypatch):
    trace=fixture();data=r.encoded(trace)
    if fault=='duplicate_key':data=data.replace(b'"kind":',b'"kind":"duplicate","kind":',1)
    elif fault=='NaN':data=data.replace(b'"trace_start_ns":',b'"extra":NaN,"trace_start_ns":',1)
    elif fault=='trailing':data+=b'{}'
    elif fault=='old_processed_schema':data=r.encoded([{'name':'invented'}])
    elif fault=='runtime':trace['torch_version']='2.14.1';data=r.encoded(trace)
    elif fault=='count':monkeypatch.setattr(r,'MAX_EVENTS',1)
    else:monkeypatch.setattr(r,'MAX_BYTES',len(data)-1)
    with pytest.raises((ValueError,TypeError)):r.assess_bytes(data,role='forward-preserving',binding=binding('forward-preserving'))


@pytest.mark.parametrize('field,value',[('design_sha256','0'*64),('control_sha256','Z'*64),('boundary_sha256','3'*63),('probe_index',True),('probe_index',0)])
def test_external_binding_must_match_declared_design_and_selection(field,value):
    b=binding('forward-preserving');b[field]=value
    with pytest.raises(ValueError):r.assess_bytes(r.encoded(fixture()),role='forward-preserving',binding=b)


def test_frozen_projection_binding_and_new_work_caps_are_enforced(monkeypatch):
    trace=fixture();monkeypatch.setattr(r.frozen,'project',lambda _:[])
    with pytest.raises(ValueError,match='binding'):assess(trace)


def test_bounded_anchor_proof_and_ancestry_limits_refuse_instead_of_truncating(monkeypatch):
    trace=fixture()
    monkeypatch.setattr(r,'MAX_ANCHORS',3)
    with pytest.raises(ValueError):assess(trace)
    monkeypatch.setattr(r,'MAX_ANCHORS',16);monkeypatch.setattr(r,'MAX_DEPTH',2)
    with pytest.raises(ValueError):assess(trace)
    monkeypatch.setattr(r,'MAX_DEPTH',128);monkeypatch.setattr(r,'MAX_PROOF_BYTES',1)
    with pytest.raises(ValueError,match='no witness truncation'):assess(trace)


def test_readonly_private_file_and_cli_preserve_input_and_never_infer_worker_completion(tmp_path):
    source,digest=saved(tmp_path/'input',fixture());before=source.stat()
    data=r.read_private_trace(source,digest);assert data==source.read_bytes()
    out=tmp_path/'assessment'
    proof=cli.assess(source,out,expected_sha256=digest,role='forward-preserving',control_sha256='2'*64,boundary_sha256='3'*64)
    assert proof['metadata_matches_recipe'] and not proof['actual_CUDA_execution_admitted'] and not proof['worker_completion_inferred']
    assert source.stat().st_mtime_ns==before.st_mtime_ns and hashlib.sha256(source.read_bytes()).hexdigest()==digest
    assert out.stat().st_mode&0o777==0o700 and (out/'assessment.json').stat().st_mode&0o777==0o600
    with pytest.raises(FileExistsError):cli.assess(source,out,expected_sha256=digest,role='native',control_sha256='2'*64,boundary_sha256='3'*64)


@pytest.mark.parametrize('fault',['file_mode','parent_mode','hardlink','symlink','hash','name'])
def test_source_reader_rejects_permission_link_digest_or_nonmetadata_file_drift(tmp_path,fault):
    source,digest=saved(tmp_path/'input',fixture());data=source.read_bytes()
    if fault=='file_mode':source.chmod(0o644)
    elif fault=='parent_mode':source.parent.chmod(0o755)
    elif fault=='hardlink':os.link(source,source.with_name('other'))
    elif fault=='symlink':source.rename(source.with_name('original'));source.symlink_to(source.with_name('original'))
    elif fault=='hash':digest='0'*64
    else:source.rename(source.with_name('tensor.pt'));source=source.with_name('tensor.pt')
    with pytest.raises((ValueError,OSError)):r.read_private_trace(source,digest)
    assert source.read_bytes()==data


def test_invalid_json_creates_only_an_explicit_local_refusal_record(tmp_path):
    folder=tmp_path/'input';folder.mkdir(mode=0o700);source=folder/'events.json';source.write_bytes(b'not JSON');source.chmod(0o600)
    out=tmp_path/'refused';proof=cli.assess(source,out,expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),role='native',control_sha256='2'*64,boundary_sha256='3'*64)
    assert not proof['metadata_matches_recipe'] and proof['error_type']=='JSONDecodeError'
    assert not proof['actual_CUDA_execution_admitted'] and source.read_bytes()==b'not JSON'


def test_isolated_cli_runs_with_science_process_network_and_profiler_imports_blocked(tmp_path):
    source,digest=saved(tmp_path/'input',fixture());out=tmp_path/'cold'
    code='''import importlib.abc,sys,subprocess,socket,os,runpy
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','numpy','timm'):raise RuntimeError('Scientific import forbidden')
sys.meta_path.insert(0,Deny())
def no(*a,**k):raise RuntimeError('Process/network forbidden')
subprocess.Popen=subprocess.run=socket.socket=socket.create_connection=os.system=os.fork=no
sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')'''
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(ROOT/'scripts/check_predictor_pgd_split_backend_v1.py'),
        '--input',str(source),'--sha256',digest,'--output',str(out),'--role','forward-preserving',
        '--control-sha256','2'*64,'--boundary-sha256','3'*64],cwd=ROOT,capture_output=True,timeout=15)
    assert result.returncode==0,result.stderr.decode()
    value=json.loads((out/'assessment.json').read_bytes())
    assert value['metadata_matches_recipe'] and not value['new_model_gradient_profiler_GPU_SSH_proof_retrieval_or_paid_work']


def test_complete_metadata_proof_keeps_all_scientific_and_lineage_admissions_false():
    proof=assess(fixture())
    for field in ('binding_lineage_authenticated','tensor_geometry_or_accuracy_verified','component_root_leaf_or_GraphTask_ownership_verified',
                  'actual_CUDA_execution_admitted','worker_completion_inferred','full_EOT8_backend_qualified','fresh_replay_performance_cost_or_training_qualified'):
        assert proof[field] is False
    assert len(r.encoded(proof))<=128*1024
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    tree=ast.parse((ROOT/'models/predictor_pgd_split_backend_v1.py').read_bytes())
    assert not any(isinstance(n,ast.FunctionDef) and n.name in ('run','worker','main','capture') for n in tree.body)
    for module in ('models/predictor_sdpa_launch_evidence_v2.py','models/predictor_sdpa_launch_evidence_v1.py','models/predictor_sdpa_component_events_v2.py','models/predictor_sdpa_component_events_v1.py'):
        assert hashlib.sha256((ROOT/module).read_bytes()).hexdigest()==r.PINNED[Path(module).name]


@pytest.mark.parametrize('fault',['mode','link','replace','rewrite','parent'])
def test_held_private_source_changes_during_read_refuse(tmp_path,monkeypatch,fault):
    source,digest=saved(tmp_path/'input',fixture());before=source.stat();real=r.os.fdopen
    def opened(*args,**kwargs):
        stream=real(*args,**kwargs)
        class Held:
            def __enter__(self):return self
            def __exit__(self,*exc):return stream.__exit__(*exc)
            def fileno(self):return stream.fileno()
            def read(self,size):
                if fault=='rewrite':
                    data=source.read_bytes();source.write_bytes(b'['+data[1:]);os.utime(source,ns=(before.st_atime_ns,before.st_mtime_ns))
                data=stream.read(size)
                if fault=='mode':source.chmod(0o644)
                elif fault=='link':os.link(source,source.with_name('other'))
                elif fault=='replace':
                    source.rename(source.with_name('preserved'));source.write_bytes(data);source.chmod(0o600)
                elif fault=='parent':
                    source.parent.rename(source.parent.with_name('preserved-parent'));source.parent.mkdir(mode=0o700)
                    source.write_bytes(data);source.chmod(0o600)
                return data
        return Held()
    monkeypatch.setattr(r.os,'fdopen',opened)
    with pytest.raises((ValueError,OSError)):r.read_private_trace(source,digest)


def test_linked_runtime_thread_does_not_replace_its_recorded_frontend_host():
    trace=fixture();host=all_named(trace,'aten::bmm')[0]
    row=append(trace,host);row.update(name='cudaLaunchKernel',thread=99,end_thread=99,resource_id=99,
        start_ns=host['start_ns']+1,end_ns=host['start_ns']+2,sequence_nr=-1,linked_correlation_id=host['correlation_id'])
    proof=assess(trace)
    assert any(w['host_event_index']==host['index'] for w in proof['kernel_witnesses'])
    assert all(w['host_event_index']!=row['index'] for w in proof['kernel_witnesses'])
    assert row['thread']==99 and row['linked_correlation_id']==host['correlation_id']


def test_nested_duplicate_operator_keeps_deepest_producer_and_exact_launch_owner():
    trace=fixture();outer=all_named(trace,'aten::bmm')[0]
    inner=append(trace,outer);inner['start_ns']+=1000;inner['end_ns']-=1000
    gpu=next(row for row in trace['records'] if row['device']=='cuda' and row['linked_correlation_id']==outer['correlation_id'])
    gpu['linked_correlation_id']=inner['correlation_id'];proof=assess(trace)
    source=next(a for a in proof['producer_anchors'] if a['sequence_nr']==31)
    assert source['producer_index']==inner['index'] and {inner['index'],outer['index']}<=set(source['producer_chain'])
    assert any(w['host_event_index']==inner['index'] for w in proof['kernel_witnesses'])
    assert all(w['host_event_index']!=outer['index'] for w in proof['kernel_witnesses'])


def test_source_supported_overhead_may_inherit_ID_without_shadowing_real_owner():
    trace=fixture();host=at(trace,r.EFFICIENT);h=overhead(trace,correlation_id=host['correlation_id'])
    proof=assess(trace)
    assert proof['retained_overhead_indices']==[h['index']]
    assert any(w['host_event_index']==host['index'] for w in proof['kernel_witnesses'])
    assert all(w['host_event_index']!=h['index'] for w in proof['kernel_witnesses'])


def test_legacy_same_family_reader_stays_incompatible_with_new_hybrid_metadata():
    trace=fixture();before=r.encoded(trace)
    with pytest.raises(ValueError):r.frozen.verify(trace,arm='original',device='cuda')
    assert assess(trace)['metadata_matches_recipe'] and r.encoded(trace)==before
