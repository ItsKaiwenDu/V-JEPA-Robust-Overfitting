"""New source-shaped association checks; saved/handwritten metadata only."""
import ast, copy, hashlib, json, os, signal, socket, subprocess, sys
from pathlib import Path
import pytest
from models import predictor_pgd_engine_association_v1 as recipe
from models import predictor_pgd_split_backend_v2 as frozen
from models import predictor_pgd_split_backend_v3 as reader

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'runs/pretrained/diagnostic/pgd-engine-association-v1-20261118'
TRACE=ROOT/'runs/pretrained/diagnostic/pgd-joint-handoff-v1-20261118/handoff-parent/linux10614-final-unarmed/attended-session/saved/capture/results/components/native/events.json'
TRACE_SHA='3db71b394c0bf0f251040445bca28f3f9b7c9a2bdeb265bb373b4a302054965d'
ORIGIN=1_700_000_000_000_000_000
NODE='ScaledDotProductEfficientAttentionBackward0'

@pytest.fixture(autouse=True)
def no_scientific_or_target_work(monkeypatch):
 def forbidden(*args,**kwargs):pytest.fail('Metadata-only test cannot start a process, signal, timer or network connection.')
 for module,name in ((subprocess,'Popen'),(os,'system'),(os,'fork'),(os,'posix_spawn'),(os,'kill'),(signal,'setitimer'),(signal,'alarm'),(socket,'socket')):
  if hasattr(module,name):monkeypatch.setattr(module,name,forbidden)
 yield
 assert not any(name in sys.modules for name in ('torch','numpy'))

@pytest.fixture(scope='module')
def captured():
 raw=TRACE.read_bytes();assert len(raw)==566539 and hashlib.sha256(raw).hexdigest()==TRACE_SHA
 return raw,json.loads(raw),json.loads((TRACE.parent/'failed.json').read_bytes())['binding']

def save(folder,name,value):
 raw=value if type(value) is bytes else reader.encoded(value)
 p=folder/name
 with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(raw)
 return p

def binding(role):
 return {'design_sha256':reader.DESIGN_SHA,'control_sha256':'2'*64,'boundary_sha256':'3'*64,'probe_index':0 if role=='native' else 1}

def assess(trace,role='native'):
 return reader.assess_bytes(reader.encoded(trace),role=role,binding=binding(role))

def fixture(role='native',outer=8,inner=9):
 """All values are invented metadata, never captured CUDA execution."""
 records=[]
 def event(name,start,end,thread=7,seq=-1,fwd=0,scope=0,annotation=False,device='cpu',link=0):
  i=len(records);row={'index':i,'name':name,'device':device,'device_index':17 if device=='cpu' else 0,
   'resource_id':thread if device=='cpu' else 5,'thread':thread,'end_thread':thread,'fwd_thread':fwd,'sequence_nr':seq,
   'scope':scope,'start_ns':ORIGIN+start*1000,'end_ns':ORIGIN+end*1000,'correlation_id':100+i,
   'linked_correlation_id':link,'is_async':False,'user_annotation':annotation,'legacy_cuda_elapsed_us':-1}
  records.append(row);return row
 def kernel(name,host):return event(name,1100+len(records)*10,1103+len(records)*10,thread=1,device='cuda',link=host['correlation_id'])
 def backward(name,seq,start,end,thread,forward_thread):
  event(reader.ENGINE+name,start,end,thread=thread)
  return event(name,start+10,end-10,thread=thread,seq=seq,fwd=forward_thread,scope=1)
 event(reader.FORWARD,0,100,scope=7,annotation=True)
 if role!='native':event(reader.CUSTOM,5,95,seq=10)
 seq=23 if role=='native' else 11
 event('aten::detach',7,9,seq=seq)
 event('aten::scaled_dot_product_attention',15,90,seq=seq)
 f=event(reader.EFFICIENT,20,80,seq=seq);kernel('fmha_cutlassF_f32_SYNTHETIC',f)
 event(reader.BACKWARD,200,950,scope=7,annotation=True)
 if role=='native':
  backward(NODE,23,210,930,outer,7)
  b=event(reader.EFFICIENT_BACKWARD,230,900,thread=outer);kernel('fmha_cutlassB_f32_SYNTHETIC',b)
 else:
  backward(reader.CUSTOM_BACKWARD,10,210,930,outer,7)
  event(reader.MATH,300,500,thread=outer,seq=30)
  for seq,start,end in ((31,310,350),(33,420,490)):
   event('aten::matmul',start,end,thread=outer,seq=seq)
   b=event('aten::bmm',start+5,end-5,thread=outer,seq=seq);kernel('sgemm_SYNTHETIC',b)
  event('aten::softmax',360,410,thread=outer,seq=32)
  s=event('aten::_softmax',365,405,thread=outer,seq=32);kernel('softmax_warp_forward<SYNTHETIC>',s)
  for name,seq,start,end,op,kn in ((
   'BmmBackward0',33,520,620,'aten::bmm','sgemm_SYNTHETIC'),(
   'SoftmaxBackward0',32,640,740,'aten::_softmax_backward_data','softmax_warp_backward<SYNTHETIC>'),(
   'BmmBackward0',31,760,880,'aten::bmm','sgemm_SYNTHETIC')):
   backward(name,seq,start,end,inner,outer)
   b=event(op,start+20,end-20,thread=inner);kernel(kn,b)
 return {'kind':reader.frozen.KIND,'source_api':reader.frozen.API,'torch_version':reader.RUNTIME,'trace_start_ns':ORIGIN,'records':records}

def at(t,name):return next(row for row in t['records'] if row['name']==name)
def append(t,row):
 row=copy.deepcopy(row);row['index']=len(t['records']);row['correlation_id']=10000+row['index'];t['records'].append(row);return row

def refuse(folder,t,role='native',call=None):
 raw=reader.encoded(t);save(folder,'MUTATED-METADATA.json',raw)
 with pytest.raises(ValueError) as e:(call or (lambda:assess(t,role)))()
 assert reader.encoded(t)==raw
 save(folder,'REFUSAL.json',{'error':str(e.value),'raw_rows':len(t['records']),'raw_unchanged':True,'target_or_scientific_work':False})


def test_complete_byte_inverse_changes_only_policy_and_one_association_method(tmp_path):
 candidate,meta=recipe.build(ROOT);assert candidate==(ROOT/recipe.NEW).read_bytes()
 edit=meta['method_edit'];original=candidate[:edit['start']]+edit['old'].encode()+candidate[edit['new_end']:]
 original=original.replace(meta['policy_edit']['new'].encode(),meta['policy_edit']['old'].encode(),1)
 assert original==(ROOT/recipe.OLD).read_bytes() and hashlib.sha256(original).hexdigest()==recipe.FROZEN
 assert meta['all_other_AST_exact'] and meta['complete_byte_inverse']
 assert reader.PINNED==frozen.PINNED and reader.RUNTIME==frozen.RUNTIME
 for name in ('MAX_BYTES','MAX_EVENTS','MAX_DEPTH','MAX_ANCHORS','MAX_PROOF_BYTES','MATRIX','SOFTMAX'):
  assert getattr(reader,name)==getattr(frozen,name)
 save(tmp_path,'SOURCE-INVERSE.json',meta)


def test_saved_capture_associates_recorded_defaults_without_promoting_failed_worker(captured,tmp_path):
 raw,trace,b=captured;before=reader.encoded(trace)
 proof=reader.assess_bytes(reader.read_private_trace(TRACE,TRACE_SHA),role='native',binding=b)
 rows=reader.frozen.project(trace);anchor,=proof['producer_anchors']
 assert (anchor['engine_index'],anchor['node_index'],anchor['producer_index'])==(965,987,273)
 assert (anchor['engine_forward_thread_recorded'],anchor['engine_sequence_recorded'])==(0,-1)
 assert (anchor['forward_thread'],anchor['sequence_nr'])==(1,0)
 assert proof['raw_records']==proof['projected_records']==1535 and len(proof['retained_overhead_indices'])==12
 assert anchor['same_counter_nonproducer_indices'] and all(i!=273 for i in anchor['same_counter_nonproducer_indices'])
 assert all(trace['records'][i]['sequence_nr']==0 and trace['records'][i]['thread']==1 for i in anchor['same_counter_nonproducer_indices'])
 assert proof['operator_counts']=={'efficient_forward':1,'efficient_backward':1,'math_forward_in_backward':0}
 assert {g for w in proof['kernel_witnesses'] for g in w['groups']}=={'efficient-forward','efficient-backward'}
 assert all(not proof[k] for k in ('actual_CUDA_execution_admitted','worker_completion_inferred','component_root_leaf_or_GraphTask_ownership_verified','full_EOT8_backend_qualified','fresh_replay_performance_cost_or_training_qualified'))
 assert not anchor['raw_tags_copied_or_filled'] and not anchor['native_parent_edge_observed']
 assert not json.loads((TRACE.parent/'worker.json').read_bytes())['complete'] and not (TRACE.parent/'complete.json').exists()
 assert reader.encoded(trace)==before and TRACE.read_bytes()==raw
 save(tmp_path,'SAVED-NATIVE-METADATA-ASSESSMENT.json',{'proof':proof,'rows':rows,'historical_worker_still_failed':True,'new_scientific_or_target_work':False})


def test_frozen_v2_still_rejects_default_wrapper_tags(captured,tmp_path):
 with pytest.raises(ValueError,match='Exact engine/node identity') as e:frozen.assess_bytes(captured[0],role='native',binding=captured[2])
 save(tmp_path,'FROZEN-REFUSAL.json',{'error':str(e.value),'original_trace_sha256':TRACE_SHA,'old_gate_unchanged':True})


@pytest.mark.parametrize('outer,inner',((8,9),(7,7),(8,8)))
def test_handwritten_hybrid_supports_distinct_default_wrappers_and_inner_node_tags(tmp_path,outer,inner):
 t=fixture('forward-preserving',outer,inner);raw=reader.encoded(t);p=assess(t,'forward-preserving')
 assert len(p['producer_anchors'])==4 and p['operator_counts']=={'efficient_forward':1,'efficient_backward':0,'math_forward_in_backward':1}
 assert len({a['engine_index'] for a in p['producer_anchors']})==4
 for a in p['producer_anchors']:
  engine=t['records'][a['engine_index']];node=t['records'][a['node_index']]
  assert (engine['fwd_thread'],engine['sequence_nr'])==(0,-1) and (node['fwd_thread'],node['sequence_nr'])==(a['forward_thread'],a['sequence_nr'])
 assert {a['sequence_nr'] for a in p['producer_anchors']}=={10,31,32,33}
 assert not p['actual_CUDA_execution_admitted'] and reader.encoded(t)==raw
 save(tmp_path,'EXPLICIT-SYNTHETIC-BASELINE.json',{'trace':t,'proof':p,'is_captured_trace':False,'worker_completion_inferred':False})


def test_same_peek_counter_on_unrelated_ops_is_retained_and_cannot_own(tmp_path):
 t=fixture();row=append(t,at(t,'aten::detach'));row.update(name='aten::isfinite',start_ns=ORIGIN+1000,end_ns=ORIGIN+2000)
 p=assess(t);a,=p['producer_anchors']
 assert set(a['same_counter_nonproducer_indices'])=={1,2,row['index']} and a['producer_chain']==[3]
 assert p['raw_records']==len(t['records']) and not p['raw_rows_removed_or_rewritten']
 assert all(w['host_event_index'] not in a['same_counter_nonproducer_indices'] for w in p['kernel_witnesses'])
 save(tmp_path,'EXPLICIT-SYNTHETIC-BASELINE.json',{'trace':t,'proof':p,'is_captured_trace':False,'worker_completion_inferred':False})


FAULTS=(
 ('engine','fwd_thread',7),('engine','sequence_nr',23),('engine','thread',99),('engine','end_thread',99),
 ('engine','device_index',99),('engine','name',reader.ENGINE+'OtherBackward'),('engine','scope',1),
 ('engine','is_async',True),('engine','user_annotation',True),
 ('node','fwd_thread',0),('node','fwd_thread',99),('node','sequence_nr',-1),('node','sequence_nr',999),
 ('node','sequence_nr',True),('node','thread',99),('node','end_thread',99),('node','scope',0),
 ('node','name','OtherBackward'),('node','is_async',True),('node','user_annotation',True),
 ('producer','sequence_nr',999),('producer','thread',99),('producer','device_index',99),
 ('producer','name','aten::_scaled_dot_product_flash_attention'),('producer','scope',1),
 ('kernel','linked_correlation_id',0),('kernel','name','unrecognized-kernel'),
 ('kernel','user_annotation',True),('kernel','device_index',1),('kernel','legacy_cuda_elapsed_us',0))
@pytest.mark.parametrize('target,field,value',FAULTS)
def test_names_and_intervals_cannot_bypass_recorded_identity_or_kernel_gates(tmp_path,target,field,value):
 t=fixture();name={'engine':reader.ENGINE+NODE,'node':NODE,'producer':reader.EFFICIENT,'kernel':'fmha_cutlassB_f32_SYNTHETIC'}[target]
 at(t,name)[field]=value
 if field=='legacy_cuda_elapsed_us':
  # Legacy duration is retained metadata, not an identity gate. It cannot
  # replace the unchanged positive linked-kernel requirement.
  p=assess(t);assert p['metadata_only'] and not p['actual_CUDA_execution_admitted']
  save(tmp_path,'EXPLICIT-SYNTHETIC-LEGACY-METADATA.json',{'trace':copy.deepcopy(t),'proof':p,'is_captured_trace':False,'legacy_duration_qualifies_as_identity':False})
  at(t,name)['linked_correlation_id']=0
 refuse(tmp_path,t)


@pytest.mark.parametrize('fault',('tagged-wrapper','nested-wrapper','second-direct-node','producer-order','wrapper-outside-phase','typed-producer-outside-scope','disjoint-typed-producers','zero-kernel-duration','missing-node-tags'))
def test_ambiguous_or_incomplete_metadata_refuses(tmp_path,fault):
 t=fixture();engine=at(t,reader.ENGINE+NODE);node=at(t,NODE);producer=at(t,reader.EFFICIENT)
 if fault=='tagged-wrapper':engine.update(fwd_thread=7,sequence_nr=23)
 elif fault=='nested-wrapper':
  other=append(t,engine);other.update(start_ns=engine['start_ns']-1000,end_ns=engine['end_ns']+1000)
 elif fault=='second-direct-node':
  other=append(t,node);other.update(name='OtherBackward',sequence_nr=777,start_ns=ORIGIN+921000,end_ns=ORIGIN+925000)
 elif fault=='producer-order':
  producer.update(start_ns=ORIGIN+941000,end_ns=ORIGIN+949000)
 elif fault=='wrapper-outside-phase':engine['start_ns']=ORIGIN+199000
 elif fault=='typed-producer-outside-scope':
  other=append(t,producer);other.update(start_ns=ORIGIN+101000,end_ns=ORIGIN+102000)
 elif fault=='disjoint-typed-producers':
  other=append(t,producer);other.update(start_ns=ORIGIN+10000,end_ns=ORIGIN+11000)
 elif fault=='zero-kernel-duration':
  k=at(t,'fmha_cutlassB_f32_SYNTHETIC');k['end_ns']=k['start_ns']
 else:del node['fwd_thread']
 refuse(tmp_path,t)


@pytest.mark.parametrize('fault',('sequence','producer-type','nested-default-wrapper','same-key-other-node'))
def test_hybrid_inner_binding_cannot_be_rescued_by_default_wrapper_names(tmp_path,fault):
 t=fixture('forward-preserving');node=at(t,'SoftmaxBackward0');engine=at(t,reader.ENGINE+'SoftmaxBackward0')
 if fault=='sequence':node['sequence_nr']=31
 elif fault=='producer-type':at(t,'aten::_softmax')['sequence_nr']=999;at(t,'aten::softmax')['sequence_nr']=999
 elif fault=='nested-default-wrapper':
  other=append(t,engine);other.update(start_ns=engine['start_ns']-1000,end_ns=engine['end_ns']+1000)
 else:other=append(t,node);other.update(name='OtherBackward',start_ns=ORIGIN+741000,end_ns=ORIGIN+745000)
 refuse(tmp_path,t,'forward-preserving')


@pytest.mark.parametrize('fault',('unsupported-runtime','wrong-design','wrong-role','counter-cap','raw-cap'))
def test_runtime_binding_and_resource_bounds_stay_strict(tmp_path,fault,monkeypatch):
 t=fixture();b=binding('native');role='native'
 if fault=='unsupported-runtime':t['torch_version']='2.9.0+cu128'
 elif fault=='wrong-design':b['design_sha256']='0'*64
 elif fault=='wrong-role':role='forward-preserving'
 elif fault=='counter-cap':monkeypatch.setattr(reader,'MAX_DEPTH',1)
 else:monkeypatch.setattr(reader,'MAX_BYTES',8)
 refuse(tmp_path,t,call=lambda:reader.assess_bytes(reader.encoded(t),role=role,binding=b))


def test_source_gate_and_private_hash_still_refuse_drift(captured,monkeypatch):
 with pytest.raises(ValueError,match='bytes changed'):reader.read_private_trace(TRACE,'0'*64)
 monkeypatch.setattr(reader.frozen,'POLICY','unreviewed-policy')
 with pytest.raises(ValueError,match='binding changed'):reader.assess_bytes(captured[0],role='native',binding=captured[2])


def test_pinned_pytorch_source_distinguishes_wrapper_node_and_peek_counter(tmp_path):
 def content(name):return json.loads((STAGE/'upstream-primary-snapshots'/(name+'.json')).read_bytes())['response']['structuredContent']['content']
 engine=content('engine_full');node=content('node');record=content('record_full');dispatcher=content('dispatcher_impl')
 assert 'RECORD_FUNCTION(' in engine and 'autograd::engine::evaluate_function: ' in engine
 assert 'guard.setForwardThreadId(thread_id_)' in node and 'guard.before(name(), static_cast<int64_t>(sequence_nr()))' in node and 'RecordScope::BACKWARD_FUNCTION' in node
 assert 'sequence_nr_ = -1' in record and 'fwd_thread_id_ = 0' in record
 assert 'at::sequence_number::peek()' in dispatcher and 'might collect the same sequence' in dispatcher
 save(tmp_path,'UPSTREAM-SEMANTICS.json',{'pytorch_ref':'v2.8.0','generic_engine_tags':[0,-1],'node_tags_explicitly_recorded':True,'dispatcher_peek_not_unique':True,'installed_wheel_or_native_parent_authenticated':False})
