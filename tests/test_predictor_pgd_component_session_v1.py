"""New local session integration: real private bytes and explicit fake processes.
No model/array decoding, real process, proc query, GPU, storage-proof or network.
"""
import ast,copy,hashlib,io,json,os,shutil,signal,socket,subprocess,sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_component_session_v1 as session
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module',autouse=True)
def no_scientific_or_target_operations():
 mp=pytest.MonkeyPatch()
 def forbidden(*a,**k):pytest.fail('Local integration cannot use process/signal/timer/network APIs.')
 for mod,name in ((subprocess,'Popen'),(os,'system'),(os,'fork'),(os,'posix_spawn'),(os,'kill'),(os,'pidfd_open'),(signal,'alarm'),(signal,'setitimer'),(signal,'pidfd_send_signal'),(socket,'socket')):
  if hasattr(mod,name):mp.setattr(mod,name,forbidden)
 proxy=NS(**{n:getattr(os,n) for n in dir(os)})
 proxy.fstatvfs=lambda fd:NS(f_bavail=32*2**30//4096,f_frsize=4096,f_favail=10000)
 mp.setattr(session.frozen_scratch,'os',proxy)
 yield
 assert 'torch' not in sys.modules and 'numpy' not in sys.modules
 mp.undo()

@pytest.fixture(scope='module')
def blueprint():return session.Blueprint(ROOT)

def write(path,raw):
 pending=[];parent=path.parent
 while not parent.exists():pending.append(parent);parent=parent.parent
 for folder in reversed(pending):folder.mkdir(mode=0o700)
 with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(raw)

def save(folder,name,value):
 raw=value if type(value) is bytes else session.encoded(value);p=folder/name;write(p,raw);return p

@pytest.fixture(scope='module')
def base_root(blueprint,tmp_path_factory):
 root=tmp_path_factory.mktemp('private-byte-baseline');root.chmod(0o700)
 for name,raw in blueprint.payload.items():write(root/name,raw)
 write(root/'control.json',blueprint.control_bytes);write(root/'local-binding.json',blueprint.local_binding_bytes)
 write(root/'runtime-source-manifest.json',blueprint.inputs['runtime_manifest']);write(root/'proposal.json',session.encoded({'kind':session.ORIGIN,'live_binding':None,'approved':False}))
 # Ordinary 16 MiB reserve fixture: no 2.5 GiB proof worker/preallocator is run.
 with os.fdopen(os.open(root/'emergency-space.bin',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f:
  chunk=b'\xa7'*(2**20)
  for _ in range(16):f.write(chunk)
  f.flush();os.fsync(f.fileno())
 return root

class Clock:
 def __init__(self):self.value=100.
 def __call__(self):self.value+=.0001;return self.value
 def pause(self,seconds):self.value+=seconds

class Child:
 def __init__(self,pid):self.pid=pid;self.exited=False;self.status=0
 def poll(self):return self.status if self.exited else None
 def wait(self,timeout=None):return self.status

class Lease:
 def __init__(self,child,birth):self.child=child;self.pid=child.pid;self.birth=birth;self.closed=False;self.signals=[]
 def dead(self):return self.child.exited
 def send(self,sig):self.signals.append(int(sig));self.child.exited=True;self.child.status=-int(sig)
 def close(self):self.closed=True

class Ports:
 def __init__(self,clock,fault=None):self.clock=clock;self.fault=fault;self.launched=[];self.leases=[];self.verified=[]
 def gpu(self,pid):
  if self.fault=='busy':raise ValueError('EXPLICIT-FAKE foreign CUDA process; leave it alone.')
  return {'kind':session.ORIGIN,'allowed_pid':pid,'query_was_simulated':True}
 def launch(self,role,plan,owner):
  assert role in session.ROLES and role!='numerical';self.launched.append(role)
  child=Child(720000+100*len(self.launched));lease=Lease(child,810000+len(self.launched));self.leases.append(lease)
  if role=='proof':
   write(owner.root/'allocation-result-v2.json',session.encoded({'kind':'EXPLICIT-INERT-FAKE-PROOF','evidence_origin':session.ORIGIN,'passed':False,'complete':False,'actual_bytes_written':0,'actual_bytes_read_back':0,'requested_recipe_bytes':5*2**30//2}))
  else:
   write(owner.root/'capture/results/components'/role/'worker.json',session.encoded({'kind':'EXPLICIT-FAKE-CALLBACK-RETURN','role':role,'pid':child.pid,'origin':session.ORIGIN,'actual_CUDA_execution_admitted':False,'training_admitted':False}))
  def sample(pid,birth):
   assert (pid,birth)==(child.pid,lease.birth)
   rss=5*2**30 if self.fault=='rss' and role=='native' else 128*2**20
   if not(self.fault=='rss' and role=='native'):child.exited=True
   row={'pid':pid,'start_ticks':birth,'parent_pid':os.getpid(),'session_pid':pid,'group_pid':pid,'rss_bytes':rss,'children':[]}
   if self.fault=='deadline' and role=='native':self.clock.value=plan['deadline']
   return {'processes':{pid:row},'rss_bytes':rss}
  def unexpected(*args):raise AssertionError('This fake has no descendants.')
  return child,lease,sample,unexpected
 def verify(self,role,plan,observed,owner):
  self.verified.append(role);assert observed['exit_status']==0 and observed['parent_observation_passed']
  if self.fault=='verification' and role=='native':raise ValueError('EXPLICIT-FAKE saved-worker gate failed.')
  source=owner.cache.load_metadata('models._pgd_component_source_v1')
  bound=source.binding(owner.blueprint.control_sha)
  assert bound['source_control_sha256']!=bound['control_sha256']
  return {'role':role,'pid':observed['pid'],'start_ticks':observed['start_ticks'],'saved_gates_returned':True,'receipt':{'kind':'EXPLICIT-FAKE-VERIFIER','proof_execution_authenticated':False,'source_binding':bound},'source_hashes':{},'evidence_origin':session.ORIGIN,'scientific_qualified':False,'training_admitted':False}
 def mapping(self):return {'origin':session.ORIGIN,'gpu':self.gpu,'launch':self.launch,'verify':self.verify,'pause':self.clock.pause}


def make(blueprint,base_root,folder,fault=None):
 root=folder/'scratch';shutil.copytree(base_root,root);clock=Clock();binding=session.frozen_scratch.identity_binding(root)
 owner=session.LocalSession(blueprint,root,binding,connection_started=clock.value,clock=clock);ports=Ports(clock,fault);return owner,ports

@pytest.fixture(scope='module')
def full_flow(blueprint,base_root,tmp_path_factory):
 folder=tmp_path_factory.mktemp('complete-local-flow');owner,ports=make(blueprint,base_root,folder)
 state=owner.run_local(ports.mapping());stream=io.BytesIO();header,terminal=owner.collect_local(stream);wire=stream.getvalue();base=header['base']
 window=blueprint.codec.Window(base['connection_started'],base['collection_started'],clock=ports.clock)
 receipt=blueprint.codec.receive(io.BytesIO(wire),folder/'received',blueprint.control_bytes,blueprint.local_binding_bytes,window=window)
 save(folder,'LOCAL-PRIORITY-WIRE.bin',wire);save(folder,'HEADER.json',header);save(folder,'TERMINAL.json',terminal);save(folder,'RECEIPT.json',receipt)
 save(folder,'LOCAL-FLOW.json',{'origin':session.ORIGIN,'completed_roles':state['completed'],'launched_roles':ports.launched,'source_files':len(header['base']['files']),'wire_sha256':session.digest(wire),'wire_bytes':len(wire),'receiver_transport_verified':receipt['transport_verified'],'real_process_GPU_model_gradient_or_proof_work':False,'reserve_fixture_bytes':16*2**20,'capacity_observation_simulated':True,'proof_callback_is_simulated':True,'live_binding':None})
 return folder,owner,ports,header,terminal,receipt,wire


def test_complete_local_storage_monitor_and_collection_join(full_flow):
 folder,owner,ports,header,terminal,receipt,wire=full_flow
 assert owner.state['completed']==list(session.ROLES)==ports.launched==ports.verified and all(l.closed for l in ports.leases)
 assert receipt['transport_verified'] and receipt['wire_sha256']==session.digest(wire) and receipt['wire_bytes']==len(wire) and not receipt['scientific_qualified'] and not receipt['training_admitted']
 assert all(not l.signals for l in ports.leases)
 assert not (owner.root/'capture/results/numerical').exists() and owner.state['complete'] is False and session.LIVE_EXECUTION_BINDING is None
 final=json.loads((folder/'received/capture/results/final-receipt.json').read_bytes())
 assert final['kind']=='pgd-component-local-work-receipt-v1' and final['all_saved_role_checks_returned'] and not final['collection_complete']
 binding=json.loads((folder/'received/local-binding.json').read_bytes());assert binding['approved'] is False and binding['maximum_new_attempts']==0
 with pytest.raises(ValueError):owner.run_local(ports.mapping())
 with pytest.raises(ValueError):owner.collect_local(io.BytesIO())


def test_transferred_members_match_bytes_modes_times_and_private_inodes(full_flow):
 folder,owner,_,header,_,receipt,_=full_flow
 assert receipt['deduplicated_bytes']>0 and receipt['expanded_bytes']==header['expanded_bytes'] and len(receipt['files'])==len(header['records'])
 for name,d in receipt['files'].items():
  src=owner.root/name;dst=folder/'received'/name;a=src.stat();b=dst.stat()
  assert session.digest(dst.read_bytes())==d['sha256']==session.digest(src.read_bytes()) and b.st_size==a.st_size==d['bytes']
  assert b.st_mtime_ns==a.st_mtime_ns==d['mtime_ns'] and (b.st_mode&0o777)==0o600 and b.st_nlink==1 and (a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino)
 assert not (folder/'received/emergency-space.bin').exists()


def test_metadata_cache_uses_private_registered_sources_and_refuses_live_entry(full_flow):
 _,owner,_,_,_,_,_=full_flow;cache=owner.cache
 module=cache.load_metadata('models.predictor_pgd_split_backend_v3');module.source_gate()
 assert module.__loader__ is cache and module.__file__==str(owner.root/'overlay/models/predictor_pgd_split_backend_v3.py')
 source=cache.load_metadata('models._pgd_component_source_v1');canonical=source.Source(owner.root/'source');cap,joined=canonical.join(owner.blueprint.control_sha)
 assert joined['source_control_sha256']==owner.blueprint.capsule['source_control_sha256'] and cap['pid']==919926
 with pytest.raises(ValueError):cache.load_metadata('models._pgd_component_entry_v1')
 with pytest.raises(ValueError):cache.install_live()
 with pytest.raises(ValueError):owner.blueprint.live_run()
 assert 'torch' not in sys.modules and 'numpy' not in sys.modules


@pytest.mark.parametrize('name',('authorization.json','worker-plan-numerical.json','parent-numerical.json','capture/results/numerical/objective-0/input.npy','source/boundary/q.pt.partial','overlay/models/ambient.py'))
def test_whole_catalogue_has_no_old_grant_numeric_or_ambient_path(blueprint,name):
 with pytest.raises(ValueError):blueprint.layout().classify(name)


def observations(blueprint,path,size,allocated=None):
 row={'identity':(1,101,2170,0o600,1),'bytes':size,'allocated_bytes':size if allocated is None else allocated,'mtime_ns':123}
 dirs={'.':{'identity':(1,1,2170,0o700),'allocated_bytes':4096}}
 for i,p in enumerate(reversed(Path(path).parents)):
  if str(p)!='.':dirs[str(p)]={'identity':(1,10+i,2170,0o700),'allocated_bytes':4096}
 return {path:row},dirs,{'available_bytes':16*2**30,'available_inodes':5000}

@pytest.mark.parametrize('fault',('capture-allocation','cache-cap','proof-in-capture','capture-in-proof','low-bytes','low-inodes','settled-partial','initial-capacity'))
def test_original_whole_scratch_caps_and_phase_gates_are_strict(blueprint,fault):
 layout=blueprint.layout();phase='capture';initial=False;settled=False
 path='cache/opaque.bin';size=1;allocated=None
 if fault=='capture-allocation':allocated=9*2**30//4+1
 elif fault=='cache-cap':size=32*2**20+1
 elif fault=='proof-in-capture':path='space-proof-v2.bin.partial';size=5*2**30//2
 elif fault=='capture-in-proof':phase='proof';path='capture/results/components/native/dq.npy'
 elif fault=='settled-partial':path='session.json.partial';settled=True
 elif fault=='initial-capacity':phase='setup';initial=True
 files,dirs,capacity=observations(blueprint,path,size,allocated)
 if fault=='low-bytes':capacity['available_bytes']=2**30-1
 elif fault=='low-inodes':capacity['available_inodes']=1023
 elif fault=='initial-capacity':capacity['available_bytes']=8*2**30-1
 with pytest.raises(ValueError):layout.assess(files,dirs,capacity,phase=phase,initial=initial,settled=settled)


@pytest.mark.parametrize('fault',('rss','deadline','verification','busy'))
def test_local_failure_stops_next_role_and_preserves_evidence(blueprint,base_root,tmp_path,fault):
 owner,ports=make(blueprint,base_root,tmp_path,fault)
 with pytest.raises((ValueError,MemoryError,TimeoutError)):owner.run_local(ports.mapping())
 assert not ports.launched if fault=='busy' else ports.launched==['proof','native']
 assert 'forward-preserving' not in ports.launched and all(l.closed for l in ports.leases)
 assert owner.state['phase']=='failed' and owner.state['complete'] is False
 if fault=='rss':assert ports.leases[-1].signals
 if fault!='busy':
  raw=(owner.root/'session.json').read_bytes();assert json.loads(raw)['phase']=='failed'
  if fault=='verification':
   out=io.BytesIO();header,_=owner.collect_local(out);assert header['base']['session_sha256']==session.digest(raw)
 save(tmp_path,'EXPLICIT-FAILURE.json',{'origin':session.ORIGIN,'fault':fault,'launched':ports.launched,'completed':owner.state['completed'],'handle_signals':[l.signals for l in ports.leases],'actual_signals_or_process_work':False})


def test_fake_proof_can_never_pass_the_real_fixed_extent_proof_verifier(full_flow):
 _,owner,_,_,_,_,_=full_flow
 raw=(owner.root/'allocation-result-v2.json').read_bytes()
 assert json.loads(raw)['actual_bytes_written']==0
 with pytest.raises(ValueError,match='Original full fixed-extent'):owner.guard.verify_proof(session.digest(raw))


@pytest.mark.parametrize('fault',('approved','attempt-count','control'))
def test_local_binding_never_accepts_an_execution_grant(blueprint,fault):
 value=json.loads(blueprint.local_binding_bytes)
 if fault=='approved':value['approved']=True
 elif fault=='attempt-count':value['maximum_new_attempts']=1
 else:value['control_sha256']='0'*64
 with pytest.raises(ValueError):blueprint.local_binding(session.encoded(value),blueprint.control_sha)


@pytest.mark.parametrize('fault',('truncated','trailing','terminal-corrupt'))
def test_partial_or_altered_collection_is_not_promoted(full_flow,blueprint,tmp_path,fault):
 _,_,ports,header,_,_,wire=full_flow
 if fault=='truncated':bad=wire[:len(wire)//2]
 elif fault=='trailing':bad=wire+b'EXTRA'
 else:bad=wire[:-20]+bytes([wire[-20]^1])+wire[-19:]
 base=header['base'];window=blueprint.codec.Window(base['connection_started'],base['collection_started'],clock=ports.clock)
 with pytest.raises((ValueError,EOFError,json.JSONDecodeError)):
  blueprint.codec.receive(io.BytesIO(bad),tmp_path/'partial',blueprint.control_bytes,blueprint.local_binding_bytes,window=window)
 assert (tmp_path/'partial').exists()
 save(tmp_path,'INCOMPLETE-TRANSPORT.json',{'fault':fault,'transport_verified':False,'preserve_partial_files':True,'scientific_or_target_work':False})


def test_cache_detects_source_byte_and_parent_path_drift(blueprint,base_root,tmp_path):
 root=tmp_path/'cache-root';shutil.copytree(base_root,root);binding=session.frozen_scratch.identity_binding(root);cache=session.PinnedCache(root,binding,blueprint)
 cache.load_metadata('models')
 cache.modules['models'].__path__=['/EXPLICIT-FAKE-FOREIGN']
 with pytest.raises(ValueError,match='loader/origin/path'):cache.load_metadata('models.predictor_pgd_split_lineage_v1')
 cache.modules['models'].__path__=[str(root/'overlay/models')]
 p=root/'overlay/models/predictor_sdpa_launch_evidence_v2.py'
 with p.open('ab') as f:f.write(b'\n# EXPLICIT-PIN-DRIFT')
 with pytest.raises(ValueError):session.PinnedCache(root,binding,blueprint)


def test_parent_publisher_cannot_replace_child_records(full_flow):
 _,owner,_,_,_,_,_=full_flow
 with pytest.raises(ValueError,match='Parent may publish only'):owner.write('capture/results/components/native/worker.json',{'FOREIGN':True},True)


def test_codec_retains_primitive_bodies_except_declared_local_metadata_literals(blueprint):
 original=ast.parse(blueprint.inputs['codec']);changed=ast.parse(blueprint.encoded_codec)
 old=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='_frozen_legacy_scope');new=next(n for n in changed.body if isinstance(n,ast.FunctionDef) and n.name=='_frozen_legacy_scope')
 inverse={v:k for k,v in blueprint.codec_derivation['local_metadata_literals'].items()}
 class Restore(ast.NodeTransformer):
  def visit_Constant(self,n):
   if type(n.value) is str and n.value in inverse:n.value=inverse[n.value]
   return n
 changed=Restore().visit(changed);new=next(n for n in changed.body if isinstance(n,ast.FunctionDef) and n.name=='_frozen_legacy_scope')
 for name in ('Window','FileReader','WireWriter','WireReader','verify_local'):
  a=next(n for n in old.body if getattr(n,'name',None)==name);b=next(n for n in new.body if getattr(n,'name',None)==name);assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
 for name in ('receive','Journal','_prior_chunks','_disk_sample'):
  a=next(n for n in original.body if getattr(n,'name',None)==name);b=next(n for n in changed.body if getattr(n,'name',None)==name);assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
