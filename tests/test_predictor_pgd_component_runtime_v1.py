"""New cold-source/opaque-byte/fake-port integration. Never scientific execution."""
import ast,collections,contextlib,copy,hashlib,json,math,os,platform,shutil,signal,socket,subprocess,sys,symtable,tarfile,threading
from argparse import Namespace
from pathlib import Path,PurePosixPath
from types import CodeType,SimpleNamespace as NS
import pytest
from models import predictor_pgd_component_runtime_v1 as assembly
from models import predictor_pgd_component_capsule_v1 as capsule
from models import predictor_pgd_split_backend_v3 as backend
from models import predictor_pgd_split_lineage_v1 as lineage
ROOT=Path(__file__).resolve().parents[1]
PREFIX=assembly.PREFIX

@pytest.fixture(autouse=True)
def no_scientific_or_target_work(monkeypatch):
 def forbidden(*args,**kwargs):pytest.fail('Source integration cannot run process/signal/timer/network work.')
 for mod,name in ((subprocess,'Popen'),(os,'system'),(os,'fork'),(os,'posix_spawn'),(os,'kill'),(signal,'alarm'),(signal,'setitimer'),(socket,'socket')):
  if hasattr(mod,name):monkeypatch.setattr(mod,name,forbidden)
 yield
 assert 'torch' not in sys.modules and 'numpy' not in sys.modules

@pytest.fixture(scope='module')
def built():
 inputs,cb=assembly.load(ROOT);files,m=assembly.assemble(inputs,cb);return inputs,cb,files,m

@pytest.fixture(scope='module')
def source_folder(tmp_path_factory):
 folder=tmp_path_factory.mktemp('shared-opaque-capsule');folder.chmod(0o700)
 archive=ROOT/'runs/pretrained/diagnostic/pgd-component-capsule-v1-20261119/component-inputs.tar'
 assert capsule.digest(archive.read_bytes())=='82a30efedfb0e85fd39c634c5bd0a8b82bd839b91bc7259841e2d517d495011e'
 with tarfile.open(archive,mode='r:') as tf:
  for entry in tf:
   assert entry.isfile() and not entry.name.startswith('/') and '..' not in PurePosixPath(entry.name).parts
   p=folder/entry.name;p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
   with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(tf.extractfile(entry).read())
 return folder


def save(folder,name,value):
 raw=value if type(value) is bytes else capsule.encoded(value);p=folder/name
 with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(raw)
 return p

def dump(n):return ast.dump(n,include_attributes=False)
def fn(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
def code(c):
 return (c.co_code,c.co_argcount,c.co_posonlyargcount,c.co_kwonlyargcount,c.co_flags,c.co_names,c.co_varnames,c.co_freevars,c.co_cellvars,c.co_stacksize,c.co_exceptiontable,tuple(code(x) if isinstance(x,CodeType) else x for x in c.co_consts))
def find_code(tree,name):
 def search(c):
  for k in c.co_consts:
   if isinstance(k,CodeType):
    if k.co_name==name:return k
    result=search(k)
    if result:return result
 return search(compile(tree,'<COLD-CODE-ONLY>','exec'))

def namespace(data,provided):
 """Evaluate only selected metadata definitions with explicit fake dependencies."""
 t=ast.parse(data);t.body=[n for n in t.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
 ns={'__name__':'EXPLICIT_FAKE_COMPONENT_INTEGRATION',**provided}
 exec(compile(ast.fix_missing_locations(t),'<EXPLICIT-FAKE-AND-BYTE-IO-ONLY>','exec'),ns)
 return ns


def test_source_graph_has_seven_concrete_modules_and_no_full_numeric_dispatch(built,tmp_path):
 inputs,cb,files,m=built
 assert len(files)==37 and sum(map(len,files.values()))<2*2**20 and len(m['generated_names'])==7
 assert not set(files)&assembly.FORBIDDEN and m['execution_binding'] is None and m['new_full_model_objectives']==0
 for name,raw in files.items():
  assert m['files'][name]=={'bytes':len(raw),'sha256':capsule.digest(raw)}
  assert all(dep in files for dep in m['imports'][name]);compile(raw,'<UNEXECUTED-SOURCE>','exec')
 for name in (PREFIX+'entry_v1.py',PREFIX+'providers_v1.py'):
  assert 'numerical_entry' not in files[name].decode() and 'load_new_source' not in files[name].decode()
 assert m['full_numerical_dispatch_absent'] and not m['actual_runtime_loaded_or_executed'] and not m['target_supervisor_inventory_collection_integrated'] and not m['new_target_scope_handoff_or_authorization']
 save(tmp_path,'COLD-GRAPH.json',{'files':m['files'],'imports':m['imports'],'generated_runtime_executed':False,'scientific_admission':False})


def test_reversible_transport_edits_retain_arithmetic_reconstruction_and_readers(built,tmp_path):
 inputs,_,files,m=built
 originals={PREFIX+'io_v1.py':'models/_pgd_split_io_v1.py',PREFIX+'store_v1.py':'models/predictor_pgd_split_store_v1.py',PREFIX+'providers_v1.py':'models/_pgd_split_providers_v1.py',PREFIX+'job_v1.py':'models/predictor_pgd_split_probe_job_v1.py'}
 for name,source in originals.items():assert assembly.inverse(files[name],m['reversible_source_edits'][name])==inputs[source]
 for label in ('route','raw','startup','boundary'):
  name='models/_pgd_split_'+label+'_v1.py';assert files[name]==inputs[name]
 old=ast.parse(inputs['models/_pgd_split_providers_v1.py']);new=ast.parse(files[PREFIX+'providers_v1.py'])
 for name in ('_operands','compute_once','make_profile','raw_bytes','serialize_return'):
  assert dump(fn(old,name))==dump(fn(new,name)) and code(find_code(old,name))==code(find_code(new,name))
 old=ast.parse(inputs['models/training.py']);new=ast.parse(files[PREFIX+'contract_v1.py'])
 assert dump(fn(old,'runtime_contract'))==dump(fn(new,'runtime_contract')) and code(find_code(old,'runtime_contract'))==code(find_code(new,'runtime_contract'))
 old=ast.parse(inputs['models/predictor_pgd_split_store_v1.py']);new=ast.parse(files[PREFIX+'store_v1.py'])
 assert dump(next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name=='Store'))==dump(next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name=='Store'))
 assert files['models/predictor_pgd_split_backend_v3.py']==inputs['models/predictor_pgd_split_backend_v3.py'] and files['models/predictor_sdpa_launch_evidence_v3.py']==inputs['models/predictor_sdpa_launch_evidence_v3.py']
 save(tmp_path,'SOURCE-PRESERVATION.json',{'byte_inverses':list(originals),'frozen_scientific_functions_unchanged':True,'actual_runtime_executed':False})


@pytest.mark.parametrize('changed',('models/_pgd_split_route_v1.py','models/_pgd_split_providers_v1.py','models/predictor_pgd_split_probe_job_v1.py','models/predictor_pgd_split_store_v1.py','models/predictor_pgd_split_backend_v3.py','models/predictor_sdpa_launch_evidence_v3.py','models/predictor_pgd_component_capsule_v1.py','CAPSULE_MANIFEST'))
def test_pinned_source_or_capsule_binding_drift_refuses_before_source_execution(built,changed):
 inputs,cb,_,_=built;bad=dict(inputs);bad[changed]+=b'\n'
 with pytest.raises(ValueError,match='Pinned cold source changed'):assembly.assemble(bad,cb)

@pytest.mark.parametrize('fault',('contract','missing','extra'))
def test_no_caller_selected_source_contract_or_namespace(built,fault):
 inputs,cb,_,_=built;bad=dict(inputs)
 if fault=='contract':cb+=b'\n'
 elif fault=='missing':del bad['models/_pgd_split_route_v1.py']
 else:bad['models/other.py']=b'raise RuntimeError("DO NOT EXECUTE")'
 with pytest.raises(ValueError):assembly.assemble(bad,cb)


def test_unbound_entry_refuses_before_any_source_or_runtime_port(built,tmp_path):
 calls=[]
 class Trap:
  def __getattr__(self,name):calls.append(name);raise AssertionError('No entry dependency may run.')
 entry=namespace(built[2][PREFIX+'entry_v1.py'],{'Namespace':Namespace,'source':Trap(),'io':Trap(),'torch':Trap()})
 with pytest.raises(ValueError,match='unbound'):entry['component_entry'](Namespace())
 with pytest.raises(ValueError,match='One component entry'):entry['component_entry'](Namespace())
 assert not calls
 save(tmp_path,'UNBOUND-REFUSAL.json',{'entry_calls':2,'dependent_ports_called':calls,'actual_execution_authorized':False})


def trace(role):
 """New invented raw records used only by the fake integration profiler."""
 records=[];origin=1_800_000_000_000_000_000
 def event(name,start,end,thread=7,seq=-1,fwd=0,scope=0,annotation=False,device='cpu',link=0):
  i=len(records);row={'index':i,'name':name,'device':device,'device_index':os.getpid() if device=='cpu' else 0,'resource_id':thread if device=='cpu' else 17,'thread':thread,'end_thread':thread,'fwd_thread':fwd,'sequence_nr':seq,'scope':scope,'start_ns':origin+start*1000,'end_ns':origin+end*1000,'correlation_id':1000+i,'linked_correlation_id':link,'is_async':False,'user_annotation':annotation,'legacy_cuda_elapsed_us':-1};records.append(row);return row
 def kernel(name,owner):event(name,2000+len(records)*10,2007+len(records)*10,thread=1,device='cuda',link=owner['correlation_id'])
 def node(name,seq,start,end,thread,fwd):
  event(backend.ENGINE+name,start,end,thread=thread);event(name,start+10,end-10,thread=thread,seq=seq,fwd=fwd,scope=1)
 event(backend.FORWARD,40,190,scope=7,annotation=True)
 if role=='forward-preserving':event(backend.CUSTOM,50,185,seq=100)
 event('aten::detach',55,57,seq=101);event('aten::scaled_dot_product_attention',65,175,seq=101)
 f=event(backend.EFFICIENT,70,150,seq=101);kernel('fmha_cutlassF_f32_EXPLICIT_FAKE',f)
 event(backend.BACKWARD,400,1700,scope=7,annotation=True)
 if role=='native':
  node('ScaledDotProductEfficientAttentionBackward0',101,410,1690,11,7)
  b=event(backend.EFFICIENT_BACKWARD,500,1660,thread=11);kernel('fmha_cutlassB_f32_EXPLICIT_FAKE',b)
 else:
  node(backend.CUSTOM_BACKWARD,100,410,1690,11,7);event(backend.MATH,600,1000,thread=11,seq=200)
  for seq,start,end in ((201,650,740),(203,850,980)):
   event('aten::matmul',start,end,thread=11,seq=seq);b=event('aten::bmm',start+10,end-10,thread=11,seq=seq);kernel('sgemm_EXPLICIT_FAKE',b)
  event('aten::softmax',750,840,thread=11,seq=202);b=event('aten::_softmax',760,830,thread=11,seq=202);kernel('softmax_warp_forward<EXPLICIT_FAKE>',b)
  for name,seq,start,end,op,kn in (('BmmBackward0',203,1100,1200,'aten::bmm','sgemm_EXPLICIT_FAKE'),('SoftmaxBackward0',202,1230,1340,'aten::_softmax_backward_data','softmax_warp_backward<EXPLICIT_FAKE>'),('BmmBackward0',201,1400,1590,'aten::bmm','sgemm_EXPLICIT_FAKE')):
   node(name,seq,start,end,12,11);b=event(op,start+20,end-20,thread=12);kernel(kn,b)
 return {'kind':backend.frozen.KIND,'source_api':backend.frozen.API,'torch_version':backend.RUNTIME,'trace_start_ns':origin,'records':records}


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
 def gradient(self,*args,**kwargs):
  assert kwargs=={'create_graph':False,'retain_graph':False};self.h.calls['outer_VJP']+=1
  if self.role=='forward-preserving':self.h.calls['inner_math_VJP']+=1
  return tuple(Value(n) for n in ('dq','dk','dv'))
 def snapshot(self):return {'route_complete':self.closed,'role':self.role,'binding':self.binding,'scientific_qualified':False,'origin':'EXPLICIT-FAKE-ENGINE-SIMULATION'}

class Profile:
 def __init__(self,h):self.h=h;self.stopped=False
 def __enter__(self):self.h.calls['profile_start']+=1;return self
 def __exit__(self,*exc):self.h.calls['profile_stop']+=1;self.stopped=True;return False

class Harness:
 def __init__(self,built,source_root,result_root,role='native',fault=None):
  self.files=built[2];self.calls=collections.Counter();self.clock=Clock();self.role=role;self.fault=fault;self.source_root=source_root;self.result_root=result_root
  self.result_root.mkdir(mode=0o700);files=self.files
  plan=NS(**namespace(files['models/predictor_pgd_split_plan_v1.py'],{'hashlib':hashlib,'json':json,'PurePosixPath':PurePosixPath}))
  store=NS(**namespace(files[PREFIX+'store_v1.py'],{'hashlib':hashlib,'os':os,'Path':Path,'stat':__import__('stat'),'threading':threading,'plan':plan}))
  helpers=ast.parse(files['models/predictor_eot8_record_v2.py']);helpers.body=[n for n in helpers.body if isinstance(n,ast.FunctionDef) and n.name in ('_json','_parse')]
  codec=namespace(ast.unparse(helpers),{'json':json})
  self.io=namespace(files[PREFIX+'io_v1.py'],{'hashlib':hashlib,'os':os,'Path':Path,'stat':__import__('stat'),'threading':threading,'original_json':NS(**codec),'Inventory':store.Inventory,'Store':store.Store,'identity':store.identity})
  def load_boundary(folder,control_sha256):
   self.calls['fake_boundary_load']+=1;cap=json.loads((folder/'boundary.json').read_bytes());assert cap['control_sha256']==control_sha256
   cap['call_rows']=[json.loads(v) for v in (folder/'attention-calls.jsonl').read_bytes().splitlines()]
   s=json.loads((folder/'attention-scopes.json').read_bytes());cap['backward_scopes']=s['scopes'];cap['ownership_refusals']=s['refusals'];return cap
  self.source=namespace(files[PREFIX+'source_v1.py'],{'copy':copy,'hashlib':hashlib,'os':os,'Path':Path,'stat':__import__('stat'),'capsule':capsule,'boundary':NS(load=load_boundary)})
  if fault=='source-binding':
   base=self.source['Source']
   class BadSource(base):
    def join(self,control):
     cap,joined=super().join(control);joined['source_control_sha256']='0'*64;return cap,joined
   self.source['Source']=BadSource
  self.job=namespace(files[PREFIX+'job_v1.py'],{'ExitStack':contextlib.ExitStack,'hashlib':hashlib,'json':json,'math':math,'os':os,'Path':Path,'stat':__import__('stat'),'threading':threading,'time':NS(monotonic=self.clock),'backend':backend,'lineage':lineage,'continuation':NS(**self.source)})
  self.expected=json.loads((source_root/'provenance/run-status.json').read_bytes())['baseline']['runtime'];self.expected={k:v for k,v in self.expected.items() if k!='model_execution'}
  self.cap=json.loads((source_root/'boundary/boundary.json').read_bytes());self.flags=self.cap['selected_call']['actual_call']['entry_flags']
  def initialize(device):
   self.calls['startup']+=1
   if fault=='startup':raise ValueError('EXPLICIT-FAKE startup failure')
   return {'kind':'native-SDPA-component-startup-v2','pid':os.getpid(),'configure_calls':1,'cuda_initialized_before':False,'cuda_initialized_after':True,'allocator_cap_bytes':7*2**30}
  def validate(*args,**kwargs):self.calls['startup_validate']+=1
  def operands(canonical,cap):
   self.calls['fake_operands']+=1;canonical.check();layout=lineage.operands(cap);metadata={}
   for n,w in layout['operands'].items():metadata[n]={'shape':w['shape'],'stride':w['stride'],'sha256':w['logical_sha256'],'dtype':'torch.float32','device':'cuda:0','storage_offset_elements':w['storage_offset_elements'],'storage_bytes':w['storage_bytes'],'alias_group':w['alias_group'],'requires_grad':w['requires_grad'],'unknown_padding_filled_zero':True,'native_physical_alignment_identity_claimed':False}
   return tuple(Value(n) for n in ('q','k','v')),Value('grad_output'),metadata
  provider_tree=ast.parse(files[PREFIX+'providers_v1.py']);provider_tree.body=[n for n in provider_tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_once']
  def synchronize():self.calls['cuda_sync']+=1
  self.torch=NS(set_num_threads=lambda n:self.calls.update(['set_threads']),get_num_threads=lambda:4,cuda=NS(synchronize=synchronize,max_memory_allocated=lambda:7*2**30+1 if fault=='allocator' else 128*2**20),profiler=NS(record_function=lambda name:contextlib.nullcontext()))
  compute=namespace(ast.unparse(provider_tree),{'torch':self.torch})['compute_once']
  def raw(profile):
   self.calls['fake_raw']+=1;assert profile.stopped;t=trace(role)
   if fault=='backend':next(r for r in t['records'] if r['device']=='cuda')['linked_correlation_id']=0
   if fault=='deadline':self.clock.value=1001.
   return capsule.encoded(t)
  def serialize(value):self.calls['fake_serialize']+=1;return ('EXPLICIT-OPAQUE-FAKE-'+value.label).encode()
  def runtime(device):
   self.calls['fake_runtime']+=1;v=copy.deepcopy(self.expected)
   if fault=='runtime':v['threads']=99
   return v
  def flags():return {} if fault=='restoration' and self.calls['outer_VJP'] else self.flags
  def job(*args,**kwargs):kwargs['evidence_origin']='EXPLICIT-FAKE-COMPONENT-ONLY-INTEGRATION';return self.job['ComponentJob'](*args,**kwargs)
  self.entry=namespace(files[PREFIX+'entry_v1.py'],{'Namespace':Namespace,'math':math,'os':os,'Path':Path,'platform':NS(node=lambda:'EXPLICIT-FAKE-HOST'),'SimpleNamespace':NS,'torch':self.torch,'io':NS(**self.io),'source':NS(**self.source),'provider':NS(prepare_operands=operands,make_profile=lambda:Profile(self),compute_once=compute,raw_bytes=raw,serialize_return=serialize,flags=flags),'contract':NS(runtime_contract=runtime),'startup':NS(initialize=initialize,validate=validate),'routes':NS(ComponentRoute=lambda role,values,incoming,*,binding:Route(self,role,values,incoming,binding)),'ComponentJob':job})
  self.entry['EXECUTION_BINDING']={'kind':'component-only-reviewed-execution-binding-v1','control_sha256':'e'*64,'capsule_root':str(source_root),'result_root':str(result_root),'host':'EXPLICIT-FAKE-HOST','capsule_manifest_sha256':self.source['MANIFEST_SHA']}
  self.args=Namespace(control_sha256='e'*64,capsule_root=str(source_root),result_root=str(result_root),role=role,deadline=1000.)
 def run(self):return self.entry['component_entry'](self.args)


@pytest.mark.parametrize('role',('native','forward-preserving'))
def test_new_entry_uses_real_capsule_io_progress_and_reader_with_explicit_fake_compute(built,source_folder,tmp_path,role):
 h=Harness(built,source_folder,tmp_path/'new-results',role);marker=h.run();folder=h.result_root/'results/components'/role
 assert marker['complete'] and marker['kind']=='pgd-capsule-component-complete-v1' and h.calls['forward']==h.calls['outer_VJP']==1 and h.calls['inner_math_VJP']==(role=='forward-preserving')
 assert h.calls['profile_start']==h.calls['profile_stop']==h.calls['route_exit']==h.calls['startup']==1 and h.calls['fake_serialize']==4
 worker=json.loads((folder/'worker.json').read_bytes());assert worker['source']['source_control_sha256']==h.source['MANIFEST']['source_control_sha256'] and worker['source']['control_sha256']=='e'*64
 assert worker['source']['control_sha256']!=worker['source']['source_control_sha256'] and worker['source']['capsule_manifest_sha256']==h.source['MANIFEST_SHA']
 assert worker['evidence_origin']=='EXPLICIT-FAKE-COMPONENT-ONLY-INTEGRATION' and not worker['scientific_qualified'] and not worker['training_admitted'] and not worker['actual_CUDA_execution_admitted']
 binding=worker['binding'];receipt=h.job['read_completion'](folder,role=role,binding=binding,marker_sha256=capsule.digest((folder/'complete.json').read_bytes()),progress_sha256=capsule.digest((folder/'progress.jsonl').read_bytes()))
 assert receipt['metadata_complete'] and not receipt['actual_CUDA_execution_admitted']
 assert not (h.result_root/'results/numerical').exists()
 save(tmp_path,'EXPLICIT-SIMULATION.json',{'origin':'EXPLICIT-FAKE-COMPONENT-ONLY-INTEGRATION','role':role,'counts':dict(h.calls),'source_control':worker['source']['source_control_sha256'],'new_control':binding['control_sha256'],'real_tensor_gradient_GPU_or_profiler_execution':False,'complete_metadata_receipt':receipt})


@pytest.mark.parametrize('fault',('startup','source-binding','runtime','backend','restoration','allocator','deadline'))
def test_incomplete_component_stays_failed_and_never_restarts_full_cohort(built,source_folder,tmp_path,fault):
 h=Harness(built,source_folder,tmp_path/'new-results',fault=fault)
 with pytest.raises((ValueError,TimeoutError)) as error:h.run()
 folder=h.result_root/'results/components/native';assert (folder/'failed.json').exists() and not (folder/'complete.json').exists() and not (h.result_root/'results/numerical').exists()
 assert h.calls['forward']<=1 and h.calls['outer_VJP']<=1 and h.calls['inner_math_VJP']==0
 if fault in ('startup','source-binding','runtime'):assert h.calls['forward']==h.calls['fake_operands']==0
 if fault in ('backend','restoration','allocator'):assert (folder/'events.json').exists() and (folder/'trace-manifest.json').exists() and h.calls['profile_stop']==h.calls['route_exit']==1
 if fault=='deadline':assert not (folder/'events.json').exists() and not (folder/'trace-manifest.json').exists() and h.calls['profile_stop']==h.calls['route_exit']==1
 with pytest.raises(ValueError,match='One component entry'):h.run()
 save(tmp_path,'EXPLICIT-FAILURE-SIMULATION.json',{'origin':'EXPLICIT-FAKE-COMPONENT-ONLY-INTEGRATION','fault':fault,'error':str(error.value),'counts':dict(h.calls),'full_model_objectives':0,'retry_blocked':True,'real_tensor_gradient_GPU_or_profiler_execution':False})


@pytest.mark.parametrize('fault',('historical-control','extra-numerical-argument','wrong-host','overlapping-roots'))
def test_entry_arguments_cannot_rebind_old_work_or_reach_a_numeric_route(built,source_folder,tmp_path,fault):
 h=Harness(built,source_folder,tmp_path/'new-results')
 if fault=='historical-control':h.args.control_sha256=h.source['MANIFEST']['source_control_sha256'];h.entry['EXECUTION_BINDING']['control_sha256']=h.args.control_sha256
 elif fault=='extra-numerical-argument':h.args.numerical_sha256='a'*64
 elif fault=='wrong-host':h.entry['EXECUTION_BINDING']['host']='OTHER-FAKE-HOST'
 else:h.args.result_root=str(source_folder/'child');h.entry['EXECUTION_BINDING']['result_root']=h.args.result_root
 with pytest.raises(ValueError) as e:h.run()
 assert not h.calls['startup'] and not h.calls['forward']
 save(tmp_path,'ENTRY-REFUSAL.json',{'fault':fault,'error':str(e.value),'port_counts':dict(h.calls),'real_scientific_work':False})


def test_component_inventory_rejects_full_numerical_and_boundary_publications(built,source_folder,tmp_path):
 h=Harness(built,source_folder,tmp_path/'new-results');store=h.io['bind'](h.result_root)
 assert len(store.inventory.spec)==28 and all(n.startswith(('results/components/','logs/')) for n in store.inventory.spec)
 for name in ('results/numerical/objective-0/loss.npy','results/run-status.json','results/boundary/q.pt'):
  with pytest.raises(ValueError):store.write_bytes(name,b'OPAQUE-FAKE')
 assert list(h.result_root.iterdir())==[]
 save(tmp_path,'COMPONENT-ONLY-INVENTORY.json',{'paths':sorted(store.inventory.spec),'full_numerical_and_source_write_paths_refused':True,'scientific_work':False})


def test_capsule_byte_mutation_is_detected_before_fake_startup(built,source_folder,tmp_path):
 changed=tmp_path/'altered-source';shutil.copytree(source_folder,changed);p=changed/'boundary/q.pt'
 with p.open('r+b') as f:first=f.read(1);f.seek(0);f.write(bytes([first[0]^1]))
 h=Harness(built,changed,tmp_path/'new-results')
 with pytest.raises(ValueError,match='Pinned source bytes changed'):h.run()
 assert not h.calls['startup'] and not list(h.result_root.iterdir())
 save(tmp_path,'SOURCE-BYTE-REFUSAL.json',{'changed':'boundary/q.pt','startup_calls':h.calls['startup'],'no_tensor_decode_or_scientific_work':True})


def test_serialized_group_keys_restore_only_canonical_runtime_ids_without_changing_manifest(built,source_folder,tmp_path):
 h=Harness(built,source_folder,tmp_path/'new-results');before=capsule.encoded(h.source['MANIFEST'])
 typed=h.source['runtime_layout']();cap=json.loads((source_folder/'boundary/boundary.json').read_bytes())
 assert typed==lineage.operands(cap) and all(type(k) is int for k in typed['backing_groups'])
 assert all(type(k) is str for k in h.source['MANIFEST']['layout']['backing_groups'])
 assert capsule.encoded(h.source['MANIFEST'])==before and capsule.encoded(typed)==capsule.encoded(h.source['MANIFEST']['layout'])
 save(tmp_path,'TYPED-LAYOUT-BOUNDARY.json',{'serialized_groups':h.source['MANIFEST']['layout']['backing_groups'],'runtime_group_IDs':list(typed['backing_groups']),'manifest_unchanged':True,'scientific_work':False})


def test_backend_source_pin_gate_checks_generated_cache_including_nonimport_dependencies(built,tmp_path):
 files=built[2];m=built[3];cache=tmp_path/'cache';cache.mkdir(mode=0o700)
 for name,raw in files.items():
  p=cache/name;p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
  with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(raw)
 reader='models/predictor_pgd_split_backend_v3.py';pin='models/predictor_sdpa_launch_evidence_v2.py'
 assert pin in files and m['nonimport_source_pin_dependencies'][reader][pin]==capsule.digest(files[pin])
 def gate(root):
  ns=namespace(files[reader],{'__file__':str(root/reader),'defaultdict':collections.defaultdict,'hashlib':hashlib,'json':json,'os':os,'Path':Path,'stat':__import__('stat'),'frozen':backend.frozen})
  return ns['source_gate']()
 assert gate(cache) is None
 bad=tmp_path/'drifted-cache';shutil.copytree(cache,bad)
 with (bad/pin).open('ab') as f:f.write(b'\n# EXPLICIT SOURCE DRIFT')
 with pytest.raises(ValueError,match='source changed'):gate(bad)
 save(tmp_path,'CACHE-SOURCE-PINS.json',{'pin_dependencies':m['nonimport_source_pin_dependencies'],'all_present_and_hash_verified':True,'drifted_pin_refused':pin,'generated_scientific_code_executed':False})
