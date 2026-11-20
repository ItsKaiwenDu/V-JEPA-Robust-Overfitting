"""New saved/modified/handwritten metadata cases; never scientific execution."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import pytest
from models import predictor_pgd_engine_materialization_v1 as recipe
from models import predictor_pgd_split_backend_v3 as old
from models import predictor_pgd_split_backend_v4 as reader

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'runs/pretrained/diagnostic/pgd-engine-materialization-v1-20261119'
CAPTURE=ROOT/'runs/pretrained/diagnostic/pgd-component-operator-v1-20261119/handoff-parent/linux10614-final-unarmed/attended-session/saved/capture/results/components'
DIGESTS={'native':'97139d0db676bf14e2e14a0b294611687cec3e0860279e4755067c1c13b2c916','forward-preserving':'1e1a2687392e37941ba1ba6d306637f39c5474850602e3d777ff5f36e073d73f'}
FAKE='EXPLICIT-MODIFIED-OR-HANDWRITTEN-METADATA; NO EXECUTION ADMISSION'

@pytest.fixture(scope='module',autouse=True)
def no_live_operations():
    patch=pytest.MonkeyPatch()
    def forbidden(*a,**k):pytest.fail('Metadata cases cannot start scientific/process/signal/timer/network work.')
    for mod,name in [(subprocess,'Popen'),(subprocess,'check_output'),(os,'system'),(os,'fork'),(os,'posix_spawn'),(os,'kill'),(os,'killpg'),(os,'pidfd_open'),(signal,'setitimer'),(signal,'alarm'),(signal,'signal'),(socket,'socket')]:
        if hasattr(mod,name):patch.setattr(mod,name,forbidden)
    yield
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    patch.undo()


def save(folder,name,value):
    data=value if type(value) is bytes else reader.encoded(value)
    with os.fdopen(os.open(folder/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(data)

@pytest.fixture(scope='module')
def captures():
    result={}
    for role in DIGESTS:
        path=CAPTURE/role;data=reader.read_private_trace(path/'events.json',DIGESTS[role]);worker=(path/'worker.json').read_bytes();w=json.loads(worker)
        result[role]=(data,json.loads(data),w['binding'],worker)
    return result


def test_cold_inverse_and_unchanged_all_other_code(tmp_path):
    candidate,meta=recipe.build(ROOT);assert candidate==(ROOT/recipe.NEW).read_bytes()
    e=meta['method_edit'];restored=candidate[:e['start']]+e['before'].encode()+candidate[e['new_end']:]
    restored=restored.replace(meta['policy_edit']['after'].encode(),meta['policy_edit']['before'].encode(),1)
    assert restored==(ROOT/recipe.OLD).read_bytes() and recipe.digest(restored)==recipe.FROZEN
    for name in ['MAX_BYTES','MAX_EVENTS','MAX_DEPTH','MAX_ANCHORS','MAX_PROOF_BYTES','PINNED','MATRIX','SOFTMAX','RUNTIME']:
        assert getattr(reader,name)==getattr(old,name)
    save(tmp_path,'SOURCE-INVERSE.json',meta)


def test_primary_source_collation_and_kineto_accessors(tmp_path):
    paths=['collection-full.json','profiler-kineto-full.json'];sources={};out={}
    for name in paths:
        p=STAGE/'upstream-primary-snapshots'/name;v=json.loads(p.read_bytes());r=v['request'];s=v['response']['structuredContent'];data=s['content'].encode()
        assert r['repository_full_name']=='pytorch/pytorch' and r['ref']=='v2.8.0'
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==s['sha']
        sources[name]=s['content'];out[name]={'request':r,'Git_blob_sha':s['sha'],'content_sha256':hashlib.sha256(data).hexdigest()}
    collection=sources['collection-full.json'];part=collection[collection.index('void ThreadLocalSubqueue::TorchOpStorage::materialize('):collection.index('// `AccumulateGrad`')]
    for text in ['auto& first = it->basic_fields_;','auto& second = (++it)->basic_fields_;','first.scope_ == at::RecordScope::FUNCTION','second.scope_ == at::RecordScope::BACKWARD_FUNCTION','first.name_.rfind("autograd::engine::evaluate_function: ", 0) == 0','first.sequence_number_ = second.sequence_number_;','first.forward_tid_ = second.forward_tid_;']:assert text in part
    kineto=sources['profiler-kineto-full.json'];assert 'TYPED_ATTR_WITH_DEFAULT(TorchOp, sequenceNr, e.sequence_number_, -1)' in kineto and 'TYPED_ATTR(TorchOp, fwdThreadId, e.sequence_number_ >= 0 ? e.forward_tid_ : 0)' in kineto
    save(tmp_path,'PRIMARY-SEMANTICS.json',{'sources':out,'installed_wheel_commit_or_native_storage_authenticated':False})


def test_saved_candidate_matches_new_metadata_recipe_without_worker_promotion(captures,tmp_path):
    data,t,b,worker=captures['forward-preserving'];before=reader.encoded(t)
    p=reader.assess_bytes(data,role='forward-preserving',binding=b)
    assert p['metadata_matches_recipe'] and p['raw_records']==p['projected_records']==2236
    assert p['operator_counts']=={'efficient_forward':1,'efficient_backward':0,'math_forward_in_backward':1}
    assert [(a['engine_index'],a['node_index']) for a in p['producer_anchors']]==[(968,990),(1468,1469),(1528,1529),(1511,1512)]
    outer,*inner=p['producer_anchors'];assert outer['engine_signature_policy']=='pytorch-2.8.0-generic-function-wrapper-defaults-v1'
    assert all(a['engine_signature_policy']=='pytorch-2.8.0-adjacent-node-tag-materialization-v1' and a['recorded_next_same_thread_CPU_index']==a['node_index'] and a['native_thread_local_storage_order_observed'] is False for a in inner)
    assert {(a['engine_forward_thread_recorded'],a['engine_sequence_recorded']) for a in inner}=={(2,14),(2,7),(2,9)}
    assert {g for w in p['kernel_witnesses'] for g in w['groups']}=={'efficient-forward','math-GEMM-forward','math-GEMM-backward','math-softmax-forward','math-softmax-backward'}
    assert reader.encoded(t)==before and all(not a['raw_tags_copied_or_filled'] and not a['native_parent_edge_observed'] for a in p['producer_anchors'])
    assert all(p[k] is False for k in ['actual_CUDA_execution_admitted','worker_completion_inferred','component_root_leaf_or_GraphTask_ownership_verified','full_EOT8_backend_qualified','fresh_replay_performance_cost_or_training_qualified'])
    assert (CAPTURE/'forward-preserving/worker.json').read_bytes()==worker and not json.loads(worker)['complete'] and (CAPTURE/'forward-preserving/failed.json').exists() and not (CAPTURE/'forward-preserving/complete.json').exists()
    save(tmp_path,'SAVED-CANDIDATE-METADATA-ASSESSMENT.json',{'proof':p,'rows':reader.frozen.project(t),'old_failed_worker_still_failed':True,'new_scientific_or_target_work':False})


def test_saved_native_retains_previous_metadata_result(captures,tmp_path):
    data,t,b,worker=captures['native'];p=reader.assess_bytes(data,role='native',binding=b);expected=json.loads((CAPTURE/'native/backend.json').read_bytes());normalized=copy.deepcopy(p);normalized['policy']=expected['policy']
    assert normalized==expected and len(p['producer_anchors'])==1 and p['producer_anchors'][0]['engine_signature_policy']=='pytorch-2.8.0-generic-function-wrapper-defaults-v1'
    assert p['raw_records']==p['projected_records']==1535 and (CAPTURE/'native/worker.json').read_bytes()==worker
    save(tmp_path,'SAVED-NATIVE-METADATA-ASSESSMENT.json',p)


def test_frozen_v3_refusal_is_retained(captures,tmp_path):
    with pytest.raises(ValueError,match='Unique source-shaped engine wrapper') as e:old.assess_bytes(captures['forward-preserving'][0],role='forward-preserving',binding=captures['forward-preserving'][2])
    save(tmp_path,'FROZEN-REFUSAL.json',{'error':str(e.value),'raw_sha256':DIGESTS['forward-preserving'],'new_target_or_scientific_work':False})


def insert(t,at,row):
    t['records'].insert(at,row)
    for i,r in enumerate(t['records']):r['index']=i


def filler(t,thread,overhead=False):
    e=t['records'][1468];r=copy.deepcopy(e);r.update(name='Activity Buffer Request' if overhead else 'synthetic::intervening-CPU',start_ns=e['start_ns']+1,end_ns=e['start_ns']+2,thread=thread,end_thread=thread,fwd_thread=0,sequence_nr=-1,scope=0,correlation_id=max(x['correlation_id'] for x in t['records'])+100,linked_correlation_id=0)
    if overhead:r.update(device_index=-1,resource_id=0,thread=1,end_thread=1)
    return r

@pytest.mark.parametrize('kind',['other-thread','overhead'])
def test_other_thread_or_overhead_interleaving_is_not_a_thread_local_op(captures,tmp_path,kind):
    t=copy.deepcopy(captures['forward-preserving'][1]);insert(t,1469,filler(t,1,kind=='overhead'));raw=reader.encoded(t)
    p=reader.assess_bytes(raw,role='forward-preserving',binding=captures['forward-preserving'][2]);a=next(a for a in p['producer_anchors'] if a['stage']=='inner-matrix')
    assert p['raw_records']==p['projected_records']==2237 and a['node_index']==1470 and a['recorded_next_same_thread_CPU_index']==1470
    assert reader.encoded(t)==raw
    save(tmp_path,'MODIFIED-METADATA.json',raw);save(tmp_path,'ASSESSMENT.json',{'origin':FAKE,'proof':p})


CASES=['partial-forward-thread','partial-sequence','wrong-engine-key','wrong-typed-producer-thread','same-thread-gap','raw-engine-node-reorder','duplicate-default-wrapper','duplicate-copied-wrapper','second-direct-node','untyped-producer','typed-producer-outside-pool','producer-after-engine','reused-node-key','missing-math-kernel-links','wrong-CUDA-device','engine-not-FUNCTION','annotated-engine','asynchronous-engine','foreign-engine-process','wrong-runtime']

def mutate(t,case):
    rows=t['records'];e=rows[1468];n=rows[1469]
    if case=='partial-forward-thread':e['fwd_thread']=0
    elif case=='partial-sequence':e['sequence_nr']=-1
    elif case=='wrong-engine-key':e['sequence_nr']=n['sequence_nr']+100
    elif case=='wrong-typed-producer-thread':e['fwd_thread']=n['fwd_thread']=999
    elif case=='same-thread-gap':insert(t,1469,filler(t,2))
    elif case=='raw-engine-node-reorder':rows[1468],rows[1469]=rows[1469],rows[1468]
    elif case in ('duplicate-default-wrapper','duplicate-copied-wrapper'):
        v=copy.deepcopy(e);v['correlation_id']=max(x['correlation_id'] for x in rows)+100
        if case=='duplicate-default-wrapper':v.update(fwd_thread=0,sequence_nr=-1)
        rows.append(v)
    elif case=='second-direct-node':
        v=copy.deepcopy(n);v.update(sequence_nr=777,correlation_id=max(x['correlation_id'] for x in rows)+100,start_ns=e['start_ns']+1,end_ns=e['start_ns']+2);insert(t,1469,v)
    elif case in ('untyped-producer','typed-producer-outside-pool','producer-after-engine'):
        producer=next(x for x in rows if x['name']=='aten::bmm' and x['thread']==2 and x['sequence_nr']==14)
        if case=='untyped-producer':producer['name']='synthetic::not-a-typed-producer'
        elif case=='typed-producer-outside-pool':
            v=copy.deepcopy(producer);v.update(start_ns=t['trace_start_ns']+1,end_ns=t['trace_start_ns']+2,correlation_id=max(x['correlation_id'] for x in rows)+100);rows.append(v)
        else:producer.update(start_ns=e['end_ns']+1,end_ns=e['end_ns']+2)
    elif case=='reused-node-key':rows[1528]['sequence_nr']=rows[1529]['sequence_nr']=14
    elif case=='missing-math-kernel-links':
        for r in rows:
            if r['device']=='cuda' and 'gemm' in r['name'].lower():r['linked_correlation_id']=0
    elif case=='wrong-CUDA-device':next(r for r in rows if r['device']=='cuda')['device_index']=1
    elif case=='engine-not-FUNCTION':e['scope']=1
    elif case=='annotated-engine':e['user_annotation']=True
    elif case=='asynchronous-engine':e['is_async']=True
    elif case=='foreign-engine-process':e['device_index']+=1;e['resource_id']+=1
    elif case=='wrong-runtime':t['torch_version']='2.8.1+cu128'
    else:raise AssertionError(case)
    for i,r in enumerate(rows):r['index']=i

@pytest.mark.parametrize('case',CASES)
def test_ambiguous_mismatched_or_unbound_metadata_refuses(captures,tmp_path,case):
    t=copy.deepcopy(captures['forward-preserving'][1]);mutate(t,case);raw=reader.encoded(t)
    save(tmp_path,'MODIFIED-METADATA.json',raw)
    with pytest.raises(ValueError) as e:reader.assess_bytes(raw,role='forward-preserving',binding=captures['forward-preserving'][2])
    assert reader.encoded(t)==raw
    save(tmp_path,'REFUSAL.json',{'origin':FAKE,'case':case,'error':str(e.value),'raw_rows':len(t['records']),'raw_unchanged':True,'scientific_or_target_work':False})


def native_fixture(backward_thread):
    rows=[];origin=1_700_000_000_000_000_000
    def event(name,lo,hi,thread=7,seq=-1,fwd=0,scope=0,annotation=False,device='cpu',linked=0):
        i=len(rows);r={'index':i,'name':name,'device':device,'device_index':17 if device=='cpu' else 0,'resource_id':17 if device=='cpu' else 5,'thread':thread,'end_thread':thread,'fwd_thread':fwd,'sequence_nr':seq,'scope':scope,'start_ns':origin+lo*1000,'end_ns':origin+hi*1000,'correlation_id':i+100,'linked_correlation_id':linked,'user_annotation':annotation,'is_async':False,'legacy_cuda_elapsed_us':-1};rows.append(r);return r
    event(reader.FORWARD,0,100,scope=7,annotation=True);f=event(reader.EFFICIENT,10,90,seq=23)
    event('fmha_cutlassF_f32_SYNTHETIC',1000,1010,device='cuda',linked=f['correlation_id'])
    event(reader.BACKWARD,200,950,scope=7,annotation=True)
    event(reader.ENGINE+'ScaledDotProductEfficientAttentionBackward0',210,930,thread=backward_thread,seq=23,fwd=7)
    event('ScaledDotProductEfficientAttentionBackward0',220,920,thread=backward_thread,seq=23,fwd=7,scope=1)
    b=event(reader.EFFICIENT_BACKWARD,230,900,thread=backward_thread)
    event('fmha_cutlassB_f32_SYNTHETIC',1020,1030,device='cuda',linked=b['correlation_id'])
    return {'kind':reader.frozen.KIND,'source_api':reader.frozen.API,'torch_version':reader.RUNTIME,'trace_start_ns':origin,'records':rows}

@pytest.mark.parametrize('backward_thread',[7,9])
def test_handwritten_native_copied_tags_on_same_or_distinct_thread(tmp_path,backward_thread):
    t=native_fixture(backward_thread);raw=reader.encoded(t);b={'design_sha256':reader.DESIGN_SHA,'control_sha256':'2'*64,'boundary_sha256':'3'*64,'probe_index':0};p=reader.assess_bytes(raw,role='native',binding=b);a,=p['producer_anchors']
    assert a['engine_signature_policy']=='pytorch-2.8.0-adjacent-node-tag-materialization-v1' and a['recorded_next_same_thread_CPU_index']==5 and not a['native_thread_local_storage_order_observed']
    assert reader.encoded(t)==raw and p['actual_CUDA_execution_admitted'] is False
    save(tmp_path,'HANDWRITTEN-METADATA.json',raw);save(tmp_path,'ASSESSMENT.json',{'origin':FAKE,'proof':p})
