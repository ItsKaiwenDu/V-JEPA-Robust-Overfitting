"""New metadata-only checks for observed loading overhead; no scientific execution."""
import ast,collections,copy,hashlib,json,os,signal,socket,subprocess,sys
from pathlib import Path
import pytest
from models import predictor_pgd_profiler_loading_v1 as recipe
from models import predictor_sdpa_launch_evidence_v2 as old_projection
from models import predictor_sdpa_launch_evidence_v3 as projection
from models import predictor_pgd_split_backend_v1 as old_reader
from models import predictor_pgd_split_backend_v2 as reader
ROOT=Path(__file__).resolve().parents[1]
SAVED=ROOT/'runs/pretrained/diagnostic/pgd-joint-handoff-v1-20261118/handoff-parent/linux10614-final-unarmed/attended-session/saved'
TRACE=SAVED/'capture/results/components/native/events.json'
SHA='3db71b394c0bf0f251040445bca28f3f9b7c9a2bdeb265bb373b4a302054965d'
NEW=('Runtime Triggered Module Loading','Lazy Function Loading')


def save(folder,name,value):
 raw=value if type(value) is bytes else reader.encoded(value)
 path=folder/name
 with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(raw)
 return path

@pytest.fixture(scope='module')
def captured():
 raw=TRACE.read_bytes();assert len(raw)==566539 and hashlib.sha256(raw).hexdigest()==SHA
 trace=json.loads(raw);binding=json.loads((TRACE.parent/'failed.json').read_bytes())['binding']
 return raw,trace,binding

@pytest.fixture(autouse=True)
def forbid_science_or_remote(monkeypatch):
 def forbidden(*args,**kwargs):pytest.fail('Metadata test cannot start a process, timer, signal or network connection.')
 for module,name in ((subprocess,'Popen'),(os,'system'),(os,'fork'),(os,'posix_spawn'),(os,'kill'),(signal,'setitimer'),(signal,'alarm'),(socket,'socket')):
  if hasattr(module,name):monkeypatch.setattr(module,name,forbidden)
 yield
 assert not any(n in sys.modules for n in ('torch','numpy'))


def altered(captured,name=NEW[0]):
 t=copy.deepcopy(captured[1]);row=next(r for r in t['records'] if r['device_index']==-1 and r['name']==name)
 return t,row

def refusal(tmp_path,trace,call):
 before=reader.encoded(trace);save(tmp_path,'MUTATED-RAW-TRACE.json',before)
 with pytest.raises(ValueError) as error:call()
 assert reader.encoded(trace)==before
 save(tmp_path,'REFUSAL.json',{'error':str(error.value),'raw_rows':len(trace['records']),'raw_sha256':hashlib.sha256(before).hexdigest(),'raw_unchanged':True,'scientific_or_target_work':False})


def test_only_reviewed_name_predicate_and_local_reader_bindings_change():
 files,meta=recipe.build(ROOT)
 for name,data in files.items():assert (ROOT/name).read_bytes()==data
 assert meta['all_other_functions_AST_exact'] and meta['complete_byte_inverses']
 for name,edits in meta['edits'].items():
  raw=files[name]
  for e in reversed(edits):raw=raw.replace(e['new'].encode(),e['old'].encode(),1)
  old=recipe.OLD_PROJECTION if name==recipe.PROJECTION else recipe.OLD_READER
  assert raw==(ROOT/old).read_bytes() and recipe.digest(raw)==recipe.FROZEN[old]
 assert projection.MAX_EVENTS==old_projection.MAX_EVENTS==20000 and projection.MAX_BYTES==old_projection.MAX_BYTES==8*2**20
 assert reader.MAX_BYTES==old_reader.MAX_BYTES==4*2**20 and reader.MAX_PROOF_BYTES==old_reader.MAX_PROOF_BYTES==128*1024
 assert projection.OVERHEAD_NAMES==frozenset(recipe.NAMES) and projection.OVERHEAD_RUNTIME=='2.8.0+cu128'
 assert old_projection.OVERHEAD_NAME=='Activity Buffer Request' and not hasattr(old_projection,'OVERHEAD_NAMES')


def test_original_saved_failure_reproduces_and_frozen_reader_remains_strict(captured,tmp_path):
 raw,trace,binding=captured
 for module,call in [('projection',lambda:old_projection.project(trace)),('reader',lambda:old_reader.assess_bytes(raw,role='native',binding=binding))]:
  with pytest.raises(ValueError,match='Unknown or malformed negative-device'):call()
 save(tmp_path,'FROZEN-REFUSAL-REPRODUCED.json',{'original_trace_sha256':SHA,'first_unknown_index':295,'trace_rows':1535,'old_refusal_unchanged':True,'scientific_or_target_work':False})
 assert TRACE.read_bytes()==raw


def test_all1535_raw_rows_are_retained_with_twelve_non_owning_overhead_rows(captured,tmp_path):
 raw,trace,binding=captured;before=reader.encoded(trace);rows=projection.project(trace)
 overhead=[r['index'] for r in trace['records'] if r['device_index']==-1]
 assert len(rows)==len(trace['records'])==1535 and len(overhead)==12
 assert collections.Counter(trace['records'][i]['name'] for i in overhead)=={'Activity Buffer Request':1,NEW[0]:6,NEW[1]:5}
 for i,(actual,projected) in enumerate(zip(trace['records'],rows)):
  assert projected['index']==i and all(projected[k]==actual[k] for k in ('name','device','device_index','correlation_id','linked_correlation_id','start_ns','end_ns','resource_id'))
  assert projected['raw_thread']==actual['thread']
  if i in overhead:
   assert projected['record_role']=='profiler-overhead-signature' and projected['parent'] is None and not projected['linked_kernels']
 assert not any(r['parent'] in overhead or r['annotation_host_event_index'] in overhead for r in rows)
 assert reader.encoded(trace)==before and TRACE.read_bytes()==raw
 save(tmp_path,'PROJECTED-RAW-METADATA.json',{'raw_sha256':SHA,'rows':rows,'raw_removed_or_changed':False})


def test_new_saved_metadata_assessment_preserves_worker_failure_and_admission_gates(captured,tmp_path):
 raw,trace,binding=captured;checked=reader.read_private_trace(TRACE,SHA);assert checked==raw
 rows=projection.project(trace);node=next(r for r in rows if r['name']=='ScaledDotProductEfficientAttentionBackward0' and r['scope']==1);engine=rows[node['parent']]
 assert (node['index'],engine['index'])==(987,965) and (node['fwd_thread'],node['sequence_nr'])==(1,0) and (engine['fwd_thread'],engine['sequence_nr'])==(0,-1)
 with pytest.raises(ValueError,match='Exact engine/node identity and enclosing CPU interval required') as error:reader.assess_bytes(checked,role='native',binding=binding)
 worker=json.loads((TRACE.parent/'worker.json').read_bytes());assert worker['complete'] is False and not (TRACE.parent/'complete.json').exists() and TRACE.read_bytes()==raw
 keys=('index','name','device_index','thread','fwd_thread','sequence_nr','parent','start_ns','end_ns','correlation_id')
 save(tmp_path,'SAVED-NATIVE-METADATA-REFUSAL.json',{'raw_trace_sha256':SHA,'raw_records':1535,'projected_records':len(rows),'retained_overhead_indices':[r['index'] for r in rows if r['record_role']=='profiler-overhead-signature'],
  'next_refusal':str(error.value),'node':{k:node[k] for k in keys},'engine':{k:engine[k] for k in keys},'historical_worker_still_failed':True,'component_or_kernel_admission':False,'array_or_new_scientific_work':False})


def synthetic_native_loading():
 """New handwritten metadata with strict valid anchors, not a captured GPU run."""
 origin=1_700_000_000_000_000_000;records=[]
 def event(name,start,end,thread=7,seq=-1,fwd=0,scope=0,annotation=False,device='cpu',link=0):
  i=len(records);r={'index':i,'name':name,'device':device,'device_index':17 if device=='cpu' else 0,'resource_id':thread if device=='cpu' else 5,'thread':thread,'end_thread':thread,'fwd_thread':fwd,'sequence_nr':seq,'scope':scope,'start_ns':origin+start*1000,'end_ns':origin+end*1000,'correlation_id':100+i,'linked_correlation_id':link,'is_async':False,'user_annotation':annotation,'legacy_cuda_elapsed_us':-1};records.append(r);return r
 event(reader.FORWARD,0,100,scope=7,annotation=True)
 event('aten::scaled_dot_product_attention',15,90,seq=23);forward=event(reader.EFFICIENT,20,80,seq=23)
 event('fmha_cutlassF_f32_SYNTHETIC',1100,1103,thread=1,device='cuda',link=forward['correlation_id'])
 event(reader.BACKWARD,200,950,scope=7,annotation=True)
 name='ScaledDotProductEfficientAttentionBackward0';event(reader.ENGINE+name,210,930,thread=8,seq=23,fwd=7)
 event(name,220,920,thread=8,seq=23,fwd=7,scope=1);backward=event(reader.EFFICIENT_BACKWARD,230,900,thread=8)
 event('fmha_cutlassB_f32_SYNTHETIC',1200,1203,thread=1,device='cuda',link=backward['correlation_id'])
 for i,name in enumerate(NEW):
  r=event(name,40+i*4,41+i*4,thread=1);r.update(device_index=-1,resource_id=0,correlation_id=forward['correlation_id'])
 return {'kind':projection.KIND,'source_api':projection.API,'torch_version':'2.8.0+cu128','trace_start_ns':origin,'records':records}


FAULTS=[('device','cuda'),('resource_id',9),('thread',2),('end_thread',2),('fwd_thread',1),('sequence_nr',0),('scope',7),('correlation_id',0),('linked_correlation_id',1),('is_async',True),('user_annotation',True),('legacy_cuda_elapsed_us',0),('device_index',-2),('name','Command Buffer Full'),('start_ns',-1),('device_index',True)]
@pytest.mark.parametrize('name',NEW)
@pytest.mark.parametrize('field,value',FAULTS)
def test_loading_name_alone_never_bypasses_the_exact_signature(captured,tmp_path,name,field,value):
 trace,row=altered(captured,name);row[field]=value
 refusal(tmp_path,trace,lambda:projection.project(trace))

@pytest.mark.parametrize('name',('Driver Compiler','Unrecognized','Resource'))
def test_unobserved_overhead_variants_stay_rejected(captured,tmp_path,name):
 trace,row=altered(captured);row['name']=name
 refusal(tmp_path,trace,lambda:projection.project(trace))

@pytest.mark.parametrize('version',('2.8.0','2.8.0+cu129','2.9.0+cu128'))
def test_loading_policy_rejects_unsupported_runtime(captured,tmp_path,version):
 trace,row=altered(captured);trace['torch_version']=version
 refusal(tmp_path,trace,lambda:projection.project(trace))

@pytest.mark.parametrize('name',NEW)
def test_positive_process_row_with_loading_name_is_not_filtered(captured,tmp_path,name):
 trace=copy.deepcopy(captured[1]);row=next(r for r in trace['records'] if r['device']=='cpu' and r['device_index']>0 and r['name']=='aten::detach' and r['linked_correlation_id']==0)
 row['name']=name;before=reader.encoded(trace);result=projection.project(trace)[row['index']]
 assert result['record_role']=='frontend-CPU' and result['device_index']>0 and result['name']==name and reader.encoded(trace)==before
 save(tmp_path,'POSITIVE-PROCESS-NAME-RETAINED.json',{'row':row,'projection':result,'scientific_or_target_work':False})


def test_recognized_overhead_cannot_become_a_kernel_owner(captured,tmp_path):
 trace,row=altered(captured);unique=1+max(r['correlation_id'] for r in trace['records']);row['correlation_id']=unique
 gpu=next(r for r in trace['records'] if r['device']=='cuda' and not r['user_annotation'] and r['end_ns']>r['start_ns']);gpu['linked_correlation_id']=unique
 refusal(tmp_path,trace,lambda:projection.project(trace))

@pytest.mark.parametrize('group',('efficient-forward','efficient-backward'))
def test_recognized_loading_cannot_replace_required_positive_kernel_evidence(captured,tmp_path,group):
 trace=synthetic_native_loading();binding=captured[2];proof=reader.assess_bytes(reader.encoded(trace),role='native',binding=binding)
 indices={w['CUDA_event_index'] for w in proof['kernel_witnesses'] if group in w['groups']};assert indices
 save(tmp_path,'EXPLICIT-SYNTHETIC-BASELINE.json',{'trace':trace,'proof':proof,'is_captured_trace':False,'native_execution_or_worker_success_inferred':False})
 template=next(r for r in trace['records'] if r['device_index']==-1 and r['name']==NEW[0])
 for i in indices:
  r=copy.deepcopy(template);r.update(index=i,start_ns=trace['records'][i]['start_ns'],end_ns=trace['records'][i]['end_ns']);trace['records'][i]=r
 refusal(tmp_path,trace,lambda:reader.assess_bytes(reader.encoded(trace),role='native',binding=binding))


def test_loading_records_do_not_bypass_recorded_autograd_producer_identity(captured,tmp_path):
 trace=copy.deepcopy(captured[1]);node=next(r for r in trace['records'] if r['scope']==1 and r['name']=='ScaledDotProductEfficientAttentionBackward0');node['fwd_thread']=0
 refusal(tmp_path,trace,lambda:reader.assess_bytes(reader.encoded(trace),role='native',binding=captured[2]))


def test_native_trace_cannot_be_relabelled_as_candidate_component(captured,tmp_path):
 binding=dict(captured[2],probe_index=1);trace=copy.deepcopy(captured[1])
 refusal(tmp_path,trace,lambda:reader.assess_bytes(reader.encoded(trace),role='forward-preserving',binding=binding))

@pytest.mark.parametrize('fault',('names','policy','projection-identity','cap'))
def test_local_reader_rejects_runtime_policy_and_dependency_drift(captured,monkeypatch,fault):
 if fault=='names':monkeypatch.setattr(projection,'OVERHEAD_NAMES',projection.OVERHEAD_NAMES|{'Command Buffer Full'})
 elif fault=='policy':monkeypatch.setattr(projection,'POLICY','unreviewed-policy')
 elif fault=='projection-identity':monkeypatch.setattr(projection,'project',lambda t:[])
 else:monkeypatch.setattr(projection,'MAX_EVENTS',20001)
 with pytest.raises(ValueError,match='binding changed'):reader.assess_bytes(captured[0],role='native',binding=captured[2])


def test_new_projection_source_pin_is_mandatory_before_assessment(captured,monkeypatch):
 real=reader.Path
 class Bound:
  def __init__(self,p):self.p=real(p)
  def with_name(self,name):
   p=self.p.with_name(name)
   if name=='predictor_sdpa_launch_evidence_v3.py':return type('Drift',(),{'read_bytes':lambda self:p.read_bytes()+b'# DRIFT'})()
   return p
 monkeypatch.setattr(reader,'Path',Bound)
 with pytest.raises(ValueError,match='source changed'):reader.assess_bytes(captured[0],role='native',binding=captured[2])


def test_private_trace_hash_gate_stays_strict(captured):
 with pytest.raises(ValueError,match='bytes changed'):reader.read_private_trace(TRACE,'0'*64)
 assert reader.read_private_trace(TRACE,SHA)==captured[0]


def test_synthetic_loading_fixture_has_strict_positive_metadata_result_only(captured,tmp_path):
 trace=synthetic_native_loading();raw=reader.encoded(trace);proof=reader.assess_bytes(raw,role='native',binding=captured[2])
 assert proof['metadata_matches_recipe'] and proof['metadata_only'] and proof['operator_counts']=={'efficient_forward':1,'efficient_backward':1,'math_forward_in_backward':0}
 assert len(proof['retained_overhead_indices'])==2 and len(proof['kernel_witnesses'])==2 and len(proof['producer_anchors'])==1
 assert not proof['worker_completion_inferred'] and not proof['actual_CUDA_execution_admitted'] and not proof['fresh_replay_performance_cost_or_training_qualified']
 save(tmp_path,'EXPLICIT-SYNTHETIC-BASELINE.json',{'trace':trace,'proof':proof,'is_captured_trace':False,'native_execution_or_worker_success_inferred':False})


def test_synthetic_missing_engine_tags_still_refuse_after_loading_is_recognized(captured,tmp_path):
 trace=synthetic_native_loading();engine=next(r for r in trace['records'] if r['name'].startswith(reader.ENGINE));engine.update(fwd_thread=0,sequence_nr=-1)
 refusal(tmp_path,trace,lambda:reader.assess_bytes(reader.encoded(trace),role='native',binding=captured[2]))


def test_pinned_upstream_maps_new_names_to_overhead_and_negative_device_wrapper():
 d=ROOT/'runs/pretrained/diagnostic/pgd-profiler-loading-v1-20261118/upstream-primary-snapshots'
 def content(n):return json.loads((d/(n+'.json')).read_bytes())['structuredContent']['content']
 mapping=content('cupti_strings');wrapper=content('overhead_wrapper');gitlink=json.loads(content('kineto_gitlink'))
 assert gitlink['sha']==projection.KINETO_PIN=='5e7501833f1021ce6f618572d3baf657b6319658'
 assert '#if CUDART_VERSION >= 12040' in mapping
 for name in NEW:assert 'return "'+name+'"' in mapping
 assert 'CUPTI_ACTIVITY_OVERHEAD_RUNTIME_TRIGGERED_MODULE_LOADING' in mapping and 'CUPTI_ACTIVITY_OVERHEAD_LAZY_FUNCTION_LOADING' in mapping
 assert 'return ActivityType::OVERHEAD;' in wrapper and 'return -1;' in wrapper and 'overheadKindString(activity_.overheadKind)' in wrapper
