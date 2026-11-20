"""Local-only component session integration; no live launcher or authorization.

Composes the frozen scanner, monitor and collection codec with a smaller catalogue.
The only executable orchestration API requires explicit simulation ports. A live
binding remains null, and no model, process, proof or network operation is started.
"""
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path,PurePosixPath
import stat
from types import ModuleType,SimpleNamespace as NS
from models import predictor_pgd_joint_scratch_v1 as frozen_scratch
from models import predictor_pgd_joint_bootstrap_v1 as frozen_boot

INPUTS='configs/predictor_pgd_component_session_inputs_v1.json'
INPUTS_SHA='218f2cf26bece4af11dfb50353a7ed0e09a43715ec3538a227b94a5d1e8ad88d'
ORIGIN='EXPLICIT-LOCAL-COMPONENT-SESSION-SIMULATION'
LIVE_EXECUTION_BINDING=None
ROLES=('proof','native','forward-preserving')
SECONDS={'proof':60,'native':120,'forward-preserving':120}
MiB,GiB=2**20,2**30


def encoded(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(v):return hashlib.sha256(v).hexdigest()
def require(value,message):
 if not value:raise ValueError(message)

def load_inputs(repository):
 root=Path(repository);data=(root/INPUTS).read_bytes();require(digest(data)==INPUTS_SHA,'Exact local session source contract required.')
 c=json.loads(data);files={}
 for name,d in c['files'].items():
  raw=(root/d['path']).read_bytes();require(type(raw) is bytes and len(raw)==d['bytes'] and digest(raw)==d['sha256'],'Pinned local session input changed: '+name);files[name]=raw
 require(digest(Path(frozen_scratch.__file__).read_bytes())==c['files']['scratch']['sha256'] and digest(Path(frozen_boot.__file__).read_bytes())==c['files']['bootstrap']['sha256'],'Frozen scan/read source changed.')
 return files,c


def codec_source(original):
 """Retain the codec; replace only its two imports and local metadata kinds."""
 t=ast.parse(original);factory=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_frozen_legacy_scope')
 removed=[n for n in factory.body if isinstance(n,ast.ImportFrom) and n.module=='models']
 require(len(removed)==2,'Two exact codec dependency bridges required.')
 factory.body=[n for n in factory.body if n not in removed];factory.args.args=[ast.arg(arg='boot'),ast.arg(arg='scratch')]
 calls=0
 for n in t.body:
  if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='legacy' for x in n.targets):
   require(isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='_frozen_legacy_scope','One exact codec factory call required.')
   n.value.args=[ast.Name('_BOOT',ast.Load()),ast.Name('_SCRATCH',ast.Load())];calls+=1
 require(calls==1,'One codec factory binding required.')
 names={'pgd-split-parent-session-v1':'pgd-component-local-session-v1','pgd-split-combined-work-receipt-v1':'pgd-component-local-work-receipt-v1','pgd-split-evidence-stream-v1':'pgd-component-local-evidence-stream-v1','authorization.json':'local-binding.json','All four saved roles must finish before a complete-work receipt.':'All three local callbacks must finish before a complete-work receipt.'}
 changes={k:0 for k in names}
 class Rename(ast.NodeTransformer):
  def visit_Constant(self,n):
   if type(n.value) is str and n.value in names:changes[n.value]+=1;return ast.copy_location(ast.Constant(names[n.value]),n)
   return n
 t=Rename().visit(t)
 require(all(changes.values()),'Exact local-only codec metadata substitutions required.')
 return (ast.unparse(ast.fix_missing_locations(t))+'\n').encode(),{'removed_dependency_imports':[ast.unparse(n) for n in removed],'local_metadata_literals':names,'literal_counts':changes,'all_other_codec_AST_retained':True}


class Blueprint:
 def __init__(self,repository):
  self.inputs,self.input_contract=load_inputs(repository);i=self.inputs
  runtime=json.loads(i['runtime_manifest']);cap=json.loads(i['capsule_manifest'])
  self.runtime=runtime['files'];self.capsule=cap;self.assets={};self.payload={}
  def add(name,raw):self.payload[name]=raw;self.assets[name]={'bytes':len(raw),'sha256':digest(raw)}
  for name in self.runtime:add('overlay/'+name,i['runtime/'+name])
  for name in cap['files']:add('source/'+name,i['source/'+name])
  add('source/MANIFEST.json',i['capsule_manifest']);add('component-inputs.tar',i['capsule_archive'])
  io=ast.parse(i['runtime/models/_pgd_component_io_v1.py']);constants={n.targets[0].id:ast.literal_eval(n.value) for n in io.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
  self.base,self.design=constants['BASE_BYTES'],constants['DESIGN_BYTES'];add('configs/base.json',self.base);add('configs/design.json',self.design)
  all_results=frozen_scratch.plan.result_specs(json.loads(self.base))
  self.result={n:d for n,d in all_results.items() if n.startswith(('results/components/native/','results/components/forward-preserving/')) or n in ('logs/native.log','logs/forward-preserving.log','logs/proof.log','results/final-receipt.json','results/collection-status.json')}
  require(len(self.result)==31,'Exact component plus collection result set required.')
  self.root_meta={n:v for n,v in frozen_scratch.ROOT_META.items() if n not in ('authorization.json','worker-plan-numerical.json','parent-numerical.json')}
  self.root_meta.update({'local-binding.json':(64*1024,False,True),'worker-plan-proof.json':(16*1024,False,True),'parent-proof.json':(128*1024,False,True)})
  catalogue={'kind':'pgd-component-local-catalogue-v1','assets':self.assets,'runtime':self.runtime,'result':self.result,'root_metadata':self.root_meta,'category_caps':frozen_scratch.CAPS,'live_binding':None}
  self.catalogue_bytes=encoded(catalogue)
  self.control_bytes=encoded({'kind':'pgd-component-local-control-v1','catalogue_sha256':digest(self.catalogue_bytes),'runtime_manifest_sha256':digest(i['runtime_manifest']),'execution_binding':None,'scientific_qualified':False,'training_admitted':False})
  self.control_sha=digest(self.control_bytes)
  # This is expressly a rejection of authorization, used only by byte simulations.
  self.local_binding_bytes=encoded({'kind':ORIGIN,'control_sha256':self.control_sha,'approved':False,'maximum_new_attempts':0,'live_execution_binding':None})
  self.installation_bytes=encoded({'kind':'pgd-component-local-installation-v1','control_sha256':self.control_sha,'files':self.runtime})
  self.assets=dict(self.assets);self.assets['installation.json']={'bytes':len(self.installation_bytes),'sha256':digest(self.installation_bytes)}
  self.payload['installation.json']=self.installation_bytes
  self.policy={'scratch_contract_text':self.catalogue_bytes.decode(),'base_text':self.base.decode(),'design_text':self.design.decode()}
  self.encoded_codec,self.codec_derivation=codec_source(i['codec'])
  boot=NS(encoded=encoded,digest=digest,parsed=frozen_scratch.parse_json,sha=frozen_scratch.sha,finite=lambda v:type(v) in (int,float) and math.isfinite(v),positive=lambda v:type(v) is int and v>0,ROLES=ROLES,SECONDS=SECONDS,control=self.control,authorization=self.local_binding,installation=self.installation)
  outer=self
  class Layout(frozen_scratch.Layout):
   def __init__(self,contract,base,design,installation,installation_sha):
    require(contract==outer.catalogue_bytes and base==outer.base and design==outer.design and installation==outer.installation_bytes and installation_sha==digest(installation),'Fixed local component catalogue/source installation required.')
    self.contract=json.loads(contract);self.result=copy.deepcopy(outer.result);self.assets=copy.deepcopy(outer.assets);self.installation=json.loads(installation);self.installation_sha256=installation_sha
    self.directories={'.','capture','cache','cache/cuda','tmp'}
    for name in (*self.assets,*outer.root_meta,*(('capture/'+n) for n in self.result)):self.directories.update(str(p) for p in PurePosixPath(name).parents)
   def classify(self,name):
    frozen_scratch.safe(name)
    if name.startswith('capture/'):
     d,plain,partial=frozen_scratch.plan.classify(name[8:],self.result);return dict(d,plain='capture/'+plain,partial=partial,space='result')
    if name in ('space-proof-v2.bin.partial','emergency-space.bin'):
     proof=name.startswith('space-proof');return dict(category='proof' if proof else 'emergency',max_bytes=5*GiB//2 if proof else 16*MiB,plain=name,partial=False,mutable=False,partial_allowed=False,space='proof' if proof else 'emergency')
    if name.startswith(('cache/','tmp/')):
     require(len(PurePosixPath(name).parts)<=8,'Cache directory depth ceiling.');return dict(category='cache',max_bytes=32*MiB,plain=name,partial=False,mutable=True,partial_allowed=False,space='cache')
    partial=name.endswith('.partial');plain=name.removesuffix('.partial')
    if plain in outer.root_meta:
     cap,mutable,allow=outer.root_meta[plain];require(not partial or allow,'Disallowed supervisor partial.');return dict(category='metadata',max_bytes=cap,plain=plain,partial=partial,mutable=mutable,partial_allowed=allow,space='metadata')
    if name in self.assets or partial and plain=='component-inputs.tar':
     d=self.assets[plain];return dict(category='input' if plain=='component-inputs.tar' or plain.startswith('source/') else 'code',max_bytes=d['bytes'],plain=plain,partial=partial,mutable=False,partial_allowed=plain=='component-inputs.tar',space='asset')
    raise ValueError('Unknown component scratch path: '+name)
  self.Layout=Layout
  scratch=NS(**{k:v for k,v in frozen_scratch.__dict__.items() if not k.startswith('__')});scratch.Layout=Layout
  ns={'__name__':'_LOCAL_COMPONENT_BYTE_CODEC_','_BOOT':boot,'_SCRATCH':scratch}
  exec(compile(self.encoded_codec,'<LOCAL-ONLY-COMPONENT-CODEC>','exec'),ns);self.codec=NS(**ns)
  # Exact repaired owned-handle monitor; no proc/PID API is called here.
  monitor={'__name__':'_LOCAL_FROZEN_MONITOR_'};exec(compile(i['monitor'],'<UNINVOKED-FROZEN-MONITOR>','exec'),monitor);self.monitor=NS(**monitor)
  tree=ast.parse(i['coordinator']);watch_source=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='watch_source');watch_sha=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WATCH_SHA' for t in n.targets))
  cold={'ast':ast,'copy':copy,'boot':boot,'WATCH_SHA':watch_sha};exec(compile(ast.Module(body=[watch_source],type_ignores=[]),'<COLD-PROOF-MONITOR-BUILDER>','exec'),cold)
  proof_tree,edits=cold['watch_source'](i['monitor']);proof=dict(monitor);exec(compile(proof_tree,'<UNINVOKED-FROZEN-PROOF-MONITOR>','exec'),proof)
  self.proof_monitor=proof['watch_stage'];self.monitor_derivation=edits
 def control(self,data):
  require(data==self.control_bytes,'Exact local-only control required; live controls are not accepted.');return json.loads(data),self.policy,self.control_sha
 def local_binding(self,data,csha):
  require(data==self.local_binding_bytes and csha==self.control_sha,'Exact rejected local binding required; this is not an execution grant.');return json.loads(data)
 def installation(self,c,csha):
  require(csha==self.control_sha and encoded(c)==self.control_bytes,'Local installation identity changed.');return self.installation_bytes,digest(self.installation_bytes)
 def layout(self):return self.Layout(self.catalogue_bytes,self.base,self.design,self.installation_bytes,digest(self.installation_bytes))
 def live_run(self,*args,**kwargs):raise ValueError('Live execution binding is null; no target, proof or process work is authorized.')


PURE=frozenset(('models','attacks','models.predictor_sdpa_component_events_v1','models.predictor_sdpa_component_events_v2','models.predictor_sdpa_launch_evidence_v1','models.predictor_sdpa_launch_evidence_v2','models.predictor_sdpa_launch_evidence_v3','models.predictor_pgd_split_backend_v3','models.predictor_pgd_split_lineage_v1','models.predictor_pgd_component_capsule_v1','models._pgd_component_source_v1','models._pgd_split_boundary_v1'))

class PinnedCache:
 """Private cache verifier and local metadata-only loader; no live install API."""
 def __init__(self,root,binding,blueprint):
  self.root=Path(root);self.binding=binding;self.files={};self.modules={};self.loading=set()
  for name,d in blueprint.runtime.items():
   raw=frozen_boot.private_read(root,binding,'overlay/'+name,d['bytes']);require(len(raw)==d['bytes'] and digest(raw)==d['sha256'],'Pinned component cache source changed.')
   module=name[:-3].replace('/','.');package=module.endswith('.__init__');module=module.removesuffix('.__init__')
   require(module not in self.files,'Duplicate cache module.');self.files[module]=('overlay/'+name,raw,package)
  require(len(self.files)==37,'Exact reviewed source-cache size required.')
 def install_live(self):raise ValueError('Live source-cache installation is unbound.')
 def check_module(self,name):
  m=self.modules[name];p,data,package=self.files[name];expected=str(self.root/p)
  require(m.__loader__ is self and m.__file__==expected and m.__spec__.loader is self and m.__spec__.origin==expected and (not package or m.__path__==[str((self.root/p).parent)]),'Cache module loader/origin/path changed.')
  require(frozen_boot.private_read(self.root,self.binding,p,len(data))==data,'Loaded cache source bytes changed.')
  if '.' in name:
   parent,child=name.rsplit('.',1);require(parent in self.modules and getattr(self.modules[parent],child,None) is m,'Cached parent-child identity changed.')
  return m
 def _import(self,name,globals=None,locals=None,fromlist=(),level=0):
  if name.split('.')[0] not in ('models','attacks'):return builtins.__import__(name,globals,locals,fromlist,level)
  require(level==0,'Relative project imports are not admitted.')
  module=self.load_metadata(name)
  for child in fromlist:
   require(child!='*','Wildcard cache import refused.')
   if name+'.'+child in self.files:setattr(module,child,self.load_metadata(name+'.'+child))
   elif not hasattr(module,child):raise ImportError('Missing pinned metadata export: '+child)
  return module if fromlist else self.load_metadata(name.split('.')[0])
 def load_metadata(self,name):
  require(name in PURE and name in self.files,'Only explicit metadata modules may execute locally; scientific entry stays disabled.')
  if name in self.modules:return self.check_module(name)
  require(name not in self.loading,'Unresolved metadata import cycle.')
  path,data,package=self.files[name]
  if '.' in name:
   parent=name.rsplit('.',1)[0];self.load_metadata(parent);self.check_module(parent)
  current=frozen_boot.private_read(self.root,self.binding,path,len(data));require(current==data,'Source changed after cache resolution.')
  self.loading.add(name)
  m=ModuleType(name);m.__file__=str(self.root/path);m.__loader__=self;m.__cached__=None;m.__package__=name if package else name.rsplit('.',1)[0];m.__spec__=importlib.util.spec_from_loader(name,self,origin=m.__file__,is_package=package)
  if package:m.__path__=[str((self.root/path).parent)]
  m.__dict__['__builtins__']={**vars(builtins),'__import__':self._import}
  try:exec(compile(data,m.__file__,'exec'),m.__dict__)
  finally:self.loading.remove(name)
  require(frozen_boot.private_read(self.root,self.binding,path,len(data))==data,'Cache source changed around metadata definition load.')
  self.modules[name]=m
  if '.' in name:setattr(self.modules[name.rsplit('.',1)[0]],name.rsplit('.',1)[1],m)
  return self.check_module(name)


class LocalSession:
 """Integration harness for explicit simulation callbacks, never a live driver."""
 def __init__(self,blueprint,root,binding,*,connection_started,clock):
  require(LIVE_EXECUTION_BINDING is None,'This interface is local-only, not a live executor.')
  self.blueprint=blueprint;self.root=Path(root);self.binding=binding;self.clock=clock;self.origin=connection_started;self.last=connection_started
  require(type(connection_started) in (int,float) and math.isfinite(connection_started),'Finite continuing origin required.')
  self.work_deadline=self.origin+2400;self.connection_deadline=self.origin+3000;self.used=self.claimed=self.collected=False;self.session_sha=None
  self.guard=frozen_scratch.Guard(root,binding,blueprint.layout());self.cache=PinnedCache(root,binding,blueprint)
 def check(self,deadline=None):
  now=self.clock();limit=min(self.work_deadline,self.connection_deadline,deadline if deadline is not None else self.work_deadline)
  if type(now) not in (int,float) or not math.isfinite(now) or now<self.last or now>=limit:raise TimeoutError('Original continuing session deadline reached; no reset.')
  self.last=now;return now
 def write(self,name,value,replace=False):
  self.guard.scan(phase='proof' if (self.state.get('current') or {}).get('role')=='proof' else 'capture')
  require(name in ({'session.json'}|{'worker-plan-'+r+'.json' for r in ROLES}|{'parent-'+r+'.json' for r in ROLES}),'Parent may publish only its owned session/plan/observation records.')
  d=self.guard.layout.classify(name);raw=encoded(value);require(d['category']=='metadata' and len(raw)<=d['max_bytes'] and (not replace or d['mutable']),'Bounded owned metadata publication required.')
  if name=='session.json' and replace:require(self.session_sha is not None and digest(self.guard.read(name,128*1024))==self.session_sha,'Only the current owned session may be replaced.')
  path=self.root/name;part=Path(str(path)+'.partial');path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
  with os.fdopen(os.open(part,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  if replace:os.replace(part,path)
  else:os.link(part,path);os.unlink(part)
  fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
  try:os.fsync(fd)
  finally:os.close(fd)
  if name=='session.json':self.session_sha=digest(raw)
  return digest(raw)
 def run_local(self,ports):
  require(type(ports) is dict and set(ports)=={'origin','gpu','launch','verify','pause'} and ports['origin']==ORIGIN,'Explicit local simulation ports required; no system driver.')
  require(not self.used,'One local session; no retry.');self.used=True
  self.state={'kind':'pgd-component-local-session-v1','control_sha256':self.blueprint.control_sha,'parent_pid':os.getpid(),'connection_started':self.origin,'work_deadline':self.work_deadline,'connection_deadline':self.connection_deadline,'phase':'preparing','completed':[],'receipts':[],'current':None,'complete':False,'scientific_qualified':False,'training_admitted':False,'evidence_origin':ORIGIN}
  try:
   self.check(self.origin+180);ports['gpu'](None);self.guard.verify_assets(phase='setup');self.guard.proof_readiness();self.check(self.origin+180)
   self.write('session.json',self.state);self.claimed=True
   for index,role in enumerate(ROLES):
    start=self.check();later=sum(SECONDS[v] for v in ROLES[index+1:]);deadline=min(start+SECONDS[role],self.work_deadline-240-later);self.check(deadline)
    plan={'role':role,'launch_started':start,'deadline':deadline,'work_deadline':self.work_deadline,'connection_deadline':self.connection_deadline,'evidence_origin':ORIGIN}
    self.state.update(phase='running',current=dict(plan));self.write('session.json',self.state,True)
    self.write('worker-plan-'+role+'.json',plan)
    child,lease,sample,lease_factory=ports['launch'](role,plan,self)
    def gate():self.check(deadline);self.guard.scan(phase='proof' if role=='proof' else 'capture');ports['gpu'](child.pid)
    kwargs={'sample':sample,'lease_factory':lease_factory,'gate':gate,'phase_deadline':deadline,'work_deadline':self.work_deadline,'launch_started':start,'clock':self.clock,'pause':ports['pause']}
    observed=self.blueprint.proof_monitor(child,lease,role='proof',**kwargs) if role=='proof' else self.blueprint.monitor.watch(child,lease,**kwargs)
    self.check(deadline);require(lease.closed,'Owned handle cleanup must complete.')
    require(observed['pid'] not in {r['pid'] for r in self.state['receipts']},'Distinct new child identities required.')
    receipt=ports['verify'](role,plan,observed,self)
    require(receipt.get('role')==role and receipt.get('pid')==observed['pid'] and receipt.get('start_ticks')==observed['start_ticks'] and receipt.get('saved_gates_returned') is True and receipt.get('evidence_origin')==ORIGIN and receipt.get('scientific_qualified') is False and receipt.get('training_admitted') is False,'Same explicit local fake-role verification receipt required.')
    self.write('parent-'+role+'.json',observed);self.state['receipts'].append(receipt);self.state['completed'].append(role);self.state.update(current=None,phase='between-roles');self.write('session.json',self.state,True);self.check(deadline)
   self.state.update(phase='await-collection',collection_pending=True);self.write('session.json',self.state,True);self.check();return copy.deepcopy(self.state)
  except BaseException as error:
   self.state.update(phase='failed',complete=False,error_type=type(error).__name__,error=str(error)[:1024])
   if self.claimed:
    try:self.write('session.json',self.state,True)
    except BaseException as extra:error.add_note('Local failure publication also failed: '+type(extra).__name__)
   raise
 def collect_local(self,out):
  require(self.claimed and not self.collected and self.state['phase'] in ('failed','await-collection'),'One ended local work session required before collection.')
  self.collected=True
  codec=self.blueprint.codec;c=codec.Collector(self.guard,self.blueprint.control_bytes,self.blueprint.local_binding_bytes,connection_started=self.origin,clock=self.clock)
  header=c.prepare();terminal=c.stream(out);return header,terminal
