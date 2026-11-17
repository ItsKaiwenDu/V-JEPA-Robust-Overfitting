"""Local opaque-byte publication checks; no arrays, scientific/runtime or target work."""
import ast
import builtins
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_store_close_v2 as recipe

ROOT=Path(__file__).resolve().parents[1]
SAVED=ROOT/'runs/pretrained/diagnostic/predictor-pgd-host10609-v1-20261116/handoff-parent/linux10609-final-unarmed/attended-session/saved'
FROZEN=(ROOT/'models/predictor_pgd_split_store_v1.py').read_bytes()
IO_SOURCE=(SAVED/'overlay/models/_pgd_split_io_v1.py').read_bytes()
IO_SHA='2bb6c3c7ecfbbb386a30bff8254b7174a9ff6c1f012006925f54d68d1265e808'
DATA=b'OPAQUE LOCAL RESULT BYTES; NO NUMPY ARRAY OR GRADIENT\n'*41
KEY='results/numerical/objective-0/pixels.npy'


@pytest.fixture(scope='module')
def candidate():
 assert hashlib.sha256(IO_SOURCE).hexdigest()==IO_SHA
 assert (SAVED/'overlay/models/predictor_pgd_split_store_v1.py').read_bytes()==FROZEN
 return recipe.build(FROZEN)


def harness(tmp_path,source,*,transition=False):
 """Real source/files/fsync; explicit injected metadata at the owned-close edge.

 No actual filesystem is claimed to change allocation on close. The injected
 report delta is 4096 bytes, keeping real identity, extent and mtime intact.
 """
 root=tmp_path/'capture';root.mkdir(mode=0o700)
 state=NS(fds={},live=set(),closed=[],events=[],before={},after={},transition=transition,
          foreign_extra=0,close_hook=None,scan_hook=None,close_error=False)
 namespace={'__name__':'EXPLICIT_LOCAL_FROZEN_OR_CANDIDATE_STORE','__file__':'<LOCAL-OPAQUE-BYTE-STORE>'}
 exec(compile(source,namespace['__file__'],'exec'),namespace)
 osproxy=NS(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')})
 def opened(*args,**kwargs):
  fd=os.open(*args,**kwargs);s=os.fstat(fd)
  if stat.S_ISREG(s.st_mode) and args[1] & (os.O_WRONLY|os.O_RDWR):
   identity=(s.st_dev,s.st_ino);state.fds[fd]=identity;state.live.add(identity);state.events.append(('publication-fd-open',identity))
  return fd
 def closed(fd):
  identity=state.fds.pop(fd,None)
  os.close(fd)
  if identity is not None:
   assert identity in state.live;state.live.remove(identity);state.closed.append(identity)
   state.events.append(('publication-fd-closed',identity))
   if state.close_hook is not None:state.close_hook()
   if state.close_error:raise OSError('EXPLICIT INJECTED OWN CLOSE ERROR AFTER REAL CLOSE')
 osproxy.open=opened;osproxy.close=closed;namespace['os']=osproxy
 actual_inventory=namespace['Inventory']
 def inventory(*args,**kwargs):
  inv=actual_inventory(*args,**kwargs);scan=inv.scan
  def observed():
   if state.scan_hook is not None:state.scan_hook()
   report=scan()
   for name,row in report['files'].items():
    identity=tuple(row['identity'][:2]);extra=4096 if transition and identity in state.live else 0
    if name==KEY:extra+=state.foreign_extra
    row['allocated_bytes']+=extra;report['allocated_bytes']+=extra
    if name==KEY:
     target=state.before if identity in state.live else state.after;target[name]=dict(row)
   return report
  inv.scan=observed
  return inv
 real_import=builtins.__import__
 def imports(name,globals=None,locals=None,fromlist=(),level=0):
  if name=='models.predictor_pgd_split_store_v1':return NS(Inventory=inventory,Store=namespace['Store'],identity=namespace['identity'])
  return real_import(name,globals,locals,fromlist,level)
 io_ns={'__name__':'EXPLICIT_LOCAL_SAVED_IO_FACADE','__file__':'<FROZEN-OPAQUE-BYTE-IO>','__builtins__':dict(vars(builtins),__import__=imports)}
 exec(compile(IO_SOURCE,io_ns['__file__'],'exec'),io_ns)
 io=NS(**io_ns);store=io.bind(root)
 return NS(root=root,folder=root/'results/numerical/objective-0',path=root/KEY,io=io,store=store,state=state,namespace=namespace,osproxy=osproxy)


def publish(h,data=DATA,name='pixels.npy',folder=None):
 return h.io._atomic(h.folder if folder is None else folder,name,[data],16*2**20)


def test_recipe_is_cold_and_restores_the_entire_frozen_module(candidate):
 data,meta=candidate
 assert recipe.build(FROZEN)==candidate and meta['whole_module_exact_inverse_verified']
 restored=data.decode().replace(recipe.NEW_COMMIT,recipe.OLD_COMMIT,1).replace(recipe.NEW_CHECK,recipe.OLD_CHECK,1).replace('import hashlib\nimport json\n','import hashlib\n',1).encode()
 assert restored==FROZEN and meta['allocation_guard_removed'] is False
 assert meta['changed_methods']==['Store.check','Store.write_bytes'] and not meta['target_handoff_ready']
 assert 'torch' not in sys.modules and 'numpy' not in sys.modules
 with pytest.raises(ValueError,match='Exact consumed'):recipe.build(FROZEN+b'\n')


def test_actual_local_filesystem_frozen_control_publishes_and_hashes_opaque_input(tmp_path):
 h=harness(tmp_path,FROZEN)
 result=publish(h);assert result['sha256']==hashlib.sha256(DATA).hexdigest()
 assert h.io.sha_file(h.path)==result['sha256'] and h.io.read(h.folder,'pixels.npy',16*2**20)==DATA
 assert not h.state.live and not h.state.fds
 (tmp_path/'LOCAL-FILESYSTEM-CONTROL.json').write_text(json.dumps({'native_local_filesystem_control':'PASS','close_transition_injected':False,'no_arrays_or_science':True},sort_keys=True))


def test_controlled_owned_close_allocation_transition_reproduces_the_saved_primary_error(tmp_path):
 h=harness(tmp_path,FROZEN,transition=True);result=publish(h)
 assert h.path.read_bytes()==DATA and result['sha256']==hashlib.sha256(DATA).hexdigest()
 with pytest.raises(ValueError,match='Previously committed result metadata changed: '+KEY):h.io.sha_file(h.path)
 before=h.store.written[KEY];after=h.store.inventory.scan()['files'][KEY]
 fields=sorted(k for k in before if before[k]!=after[k]);assert fields==['allocated_bytes']
 assert before['allocated_bytes']-after['allocated_bytes']==4096 and before==h.state.before[KEY]
 assert not h.state.live and len(h.state.closed)==1
 (tmp_path/'CONTROLLED-CLOSE-TRANSITION-REPRO.json').write_text(json.dumps({'frozen_sha256':recipe.FROZEN_SHA,
  'only_injected_difference':fields,'expected':before,'observed':after,'failure_message_matches_saved_live_trace':True,
  'live_XFS_cause_established':False,'actual_file_bytes_unchanged':True,'array_or_model_executed':False},sort_keys=True))


def test_candidate_records_after_owned_close_and_real_facade_readback_remains_strict(candidate,tmp_path):
 h=harness(tmp_path,candidate[0],transition=True);result=publish(h)
 assert h.io.sha_file(h.path)==hashlib.sha256(DATA).hexdigest() and h.io.read(h.folder,'pixels.npy',16*2**20)==DATA
 assert h.store.written[KEY]==h.store.inventory.scan()['files'][KEY]==h.state.after[KEY]
 assert KEY not in h.state.before  # The candidate never pins the published name while its descriptor is open.
 assert h.store.written[KEY]['allocated_bytes']==h.path.stat().st_blocks*512
 assert len(h.state.closed)==1 and not h.store.poisoned and not h.state.live
 h.state.foreign_extra=4096
 with pytest.raises(ValueError,match='Previously committed') as caught:h.io.sha_file(h.path)
 notes=caught.value.__notes__;assert len(notes)==1 and len(notes[0].split(': ',1)[1])<=2048
 detail=json.loads(notes[0].split(': ',1)[1]);assert detail['fields']==['allocated_bytes'] and detail['expected']!=detail['observed']
 (tmp_path/'CANDIDATE-CLOSE-ORDER-AND-DIAGNOSTIC.json').write_text(json.dumps({'candidate_sha256':candidate[1]['candidate_sha256'],
  'owned_close_transition_passes':True,'later_foreign_allocation_change_refused':True,'diagnostic':detail,'no_arrays_or_science':True},sort_keys=True))


@pytest.mark.parametrize('fault',['mode','mtime','size','inode','hardlink'])
def test_foreign_metadata_changes_after_commit_still_refuse(candidate,tmp_path,fault):
 h=harness(tmp_path,candidate[0],transition=True);publish(h)
 if fault=='mode':h.path.chmod(0o644)
 elif fault=='mtime':
  s=h.path.stat();os.utime(h.path,ns=(s.st_atime_ns,s.st_mtime_ns+101))
 elif fault=='size':
  with h.path.open('ab') as f:f.write(b'X')
 elif fault=='inode':
  h.path.rename(tmp_path/'RETAINED-ORIGINAL-PIXELS');h.path.write_bytes(DATA);h.path.chmod(0o600)
 else:os.link(h.path,tmp_path/'RETAINED-HARDLINK')
 with pytest.raises(ValueError):h.io.sha_file(h.path)


@pytest.mark.parametrize('fault',['mode','mtime','size','inode'])
def test_mutation_at_owned_close_is_not_accepted_as_a_new_baseline(candidate,tmp_path,fault):
 h=harness(tmp_path,candidate[0],transition=True)
 def mutate():
  if fault=='mode':h.path.chmod(0o644)
  elif fault=='mtime':
   s=h.path.stat();os.utime(h.path,ns=(s.st_atime_ns,s.st_mtime_ns+19))
  elif fault=='size':
   with h.path.open('ab') as f:f.write(b'X')
  else:h.path.rename(tmp_path/'RETAINED-CLOSE-ORIGINAL');h.path.write_bytes(DATA);h.path.chmod(0o600)
 h.state.close_hook=mutate
 with pytest.raises(ValueError,match='after owned close'):publish(h)
 assert h.store.poisoned and KEY not in h.store.written and not h.state.live
 with pytest.raises(ValueError,match='unpoisoned'):h.io.sha_file(h.path)


def test_changed_previous_member_during_new_publication_close_poisoned(candidate,tmp_path):
 h=harness(tmp_path,candidate[0],transition=True);publish(h)
 old=dict(h.store.written[KEY]);h.state.close_hook=lambda:h.path.write_bytes(DATA+b'CHANGED')
 with pytest.raises(ValueError,match='Other committed result changed'):publish(h,b'NEXT','targets.npy')
 assert h.store.poisoned and h.store.written[KEY]==old
 assert 'results/numerical/objective-0/targets.npy' not in h.store.written


def test_mutation_between_closed_name_check_and_scan_refuses(candidate,tmp_path):
 h=harness(tmp_path,candidate[0],transition=True);mutated=[]
 def mutate():
  if h.state.closed and not mutated:
   h.path.write_bytes(DATA+b'CHANGED');mutated.append(True)
 h.state.scan_hook=mutate
 with pytest.raises(ValueError,match='during closed-file scan'):publish(h)
 assert mutated and h.store.poisoned and KEY not in h.store.written


def test_own_publication_close_error_is_not_retried_or_snapshotted(candidate,tmp_path):
 h=harness(tmp_path,candidate[0]);h.state.close_error=True
 with pytest.raises(OSError,match='EXPLICIT INJECTED OWN CLOSE ERROR'):publish(h)
 assert len(h.state.closed)==1 and not h.state.live and not h.state.fds
 assert h.store.poisoned and KEY not in h.store.written


def test_mutable_replacement_keeps_other_committed_rows_and_full_metadata(candidate,tmp_path):
 h=harness(tmp_path,candidate[0],transition=True);publish(h);prior=dict(h.store.written[KEY])
 folder=h.root/'results'
 publish(h,b'FIRST','run-status.json',folder);publish(h,b'SECOND','run-status.json',folder)
 assert h.io.read(folder,'run-status.json',256*1024)==b'SECOND' and h.store.written[KEY]==prior
 assert h.io.sha_file(h.path)==hashlib.sha256(DATA).hexdigest()


@pytest.mark.parametrize('fault',['immutable-repeat','foreign-mutable','symlink','hardlink','limit','owner','recursive','unknown'])
def test_original_publication_refusals_and_evidence_preservation(candidate,tmp_path,fault):
 h=harness(tmp_path,candidate[0])
 if fault=='immutable-repeat':
  publish(h)
  with pytest.raises(FileExistsError):publish(h)
  assert h.path.read_bytes()==DATA
 elif fault=='foreign-mutable':
  folder=h.root/'results';folder.mkdir(mode=0o700);p=folder/'run-status.json';p.write_bytes(b'FOREIGN');p.chmod(0o600)
  with pytest.raises(FileExistsError):publish(h,b'NEW','run-status.json',folder)
  assert p.read_bytes()==b'FOREIGN'
 elif fault in ('symlink','hardlink'):
  h.folder.mkdir(mode=0o700,parents=True);h.folder.parent.chmod(0o700);h.folder.parent.parent.chmod(0o700)
  foreign=tmp_path/'FOREIGN';foreign.write_bytes(DATA);foreign.chmod(0o600)
  if fault=='symlink':h.path.symlink_to(foreign)
  else:os.link(foreign,h.path)
  with pytest.raises(ValueError):publish(h)
  assert foreign.read_bytes()==DATA
 elif fault=='limit':
  with pytest.raises(ValueError,match='whole byte publication cap'):h.io._atomic(h.folder,'pixels.npy',[DATA],1)
  assert not h.path.exists()
 elif fault=='owner':
  h.store.pid=-1
  with pytest.raises(ValueError,match='serial process writer'):publish(h)
 elif fault=='recursive':
  h.store.busy=True
  with pytest.raises(ValueError,match='Recursive'):publish(h)
  assert h.store.poisoned
 else:
  with pytest.raises(ValueError,match='Unknown split result path'):h.io._atomic(h.folder,'not-in-plan.npy',[DATA],16*2**20)


def test_actual_inventory_allocation_ceiling_remains_enforced(candidate,tmp_path):
 h=harness(tmp_path,candidate[0]);publish(h);real=h.namespace['private']
 class LargeAllocation:
  def __init__(self,s):self.s=s;self.st_blocks=(9*2**30//4)//512+1
  def __getattr__(self,n):return getattr(self.s,n)
 def private(path,**kwargs):
  s=real(path,**kwargs)
  return LargeAllocation(s) if path==h.path else s
 h.namespace['private']=private
 with pytest.raises(ValueError,match='allocated-byte ceiling'):h.io.sha_file(h.path)


def test_standalone_store_owner_thread_and_poison_guards_unchanged(candidate,tmp_path):
 h=harness(tmp_path,candidate[0]);writer=h.store.writer();writer.owner=(os.getpid(),-1)
 with pytest.raises(ValueError,match='Owned unpoisoned serial result writer'):writer.check()
 writer.owner=(os.getpid(),__import__('threading').get_ident());writer.poisoned=True
 with pytest.raises(ValueError,match='Owned unpoisoned serial result writer'):writer.check()


@pytest.mark.parametrize('fixed',[False,True])
def test_original_save_tensor_callsite_reproduces_or_passes_with_inert_serializer(candidate,tmp_path,fixed):
 import io
 data=(SAVED/'overlay/models/_pgd_split_records_v1.py').read_bytes()
 control=json.loads((SAVED/'control.json').read_bytes());policy=json.loads(control['policy_text']);contract=json.loads(policy['scratch_contract_text'])
 assert hashlib.sha256(data).hexdigest()==contract['runtime_files']['models/_pgd_split_records_v1.py']['sha256']
 fn=next(n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name=='save_tensor')
 h=harness(tmp_path,candidate[0] if fixed else FROZEN,transition=True);h.folder.mkdir(mode=0o700,parents=True)
 h.folder.parent.chmod(0o700);h.folder.parent.parent.chmod(0o700)
 class OpaqueArray:
  nbytes=len(DATA);shape=(len(DATA),);dtype='EXPLICIT-OPAQUE-STUB'
  def copy(self):return self
 class InertTensor:
  device='EXPLICIT-INERT-DEVICE'
  def detach(self):return self
  def cpu(self):return self
  def contiguous(self):return self
  def numpy(self):return OpaqueArray()
 calls=[]
 def save(stream,value,*,allow_pickle):
  assert type(value) is OpaqueArray and allow_pickle is False;calls.append('INERT-NUMPY-SERIALIZER');stream.write(DATA)
 numpy=NS(save=save,isfinite=lambda a:NS(all=lambda:True))
 def importing(name,*args,**kwargs):
  if name=='numpy':return numpy
  if name=='io':return io
  raise AssertionError('Unreviewed import in inert saved serializer: '+name)
 g={'Path':Path,'frozen':NS(atomic=h.io,sha_file=h.io.sha_file),'__builtins__':dict(vars(builtins),__import__=importing)}
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'<SAVED-SAVE-TENSOR-INERT-SERIALIZER>','exec'),g)
 state={'artifacts':{}}
 if fixed:
  result=g['save_tensor'](h.folder,state,'pixels',InertTensor())
  assert type(result) is OpaqueArray and state['artifacts']['pixels']['sha256']==hashlib.sha256(DATA).hexdigest()
  assert state['artifacts']['pixels']['bytes']==len(DATA) and h.io.sha_file(h.path)==hashlib.sha256(DATA).hexdigest()
 else:
  with pytest.raises(ValueError,match='Previously committed result metadata changed: '+KEY):g['save_tensor'](h.folder,state,'pixels',InertTensor())
  assert state['artifacts']=={}
 assert calls==['INERT-NUMPY-SERIALIZER'] and h.path.read_bytes()==DATA
 assert 'numpy' not in sys.modules and 'torch' not in sys.modules
 (tmp_path/'ORIGINAL-SAVE-TENSOR-IO-PATH.json').write_text(json.dumps({'candidate':fixed,'actual_saved_function_sha256':hashlib.sha256(data).hexdigest(),
  'serializer_tensor_and_array_are_stubs':True,'real_opaque_bytes_facade_store_and_fsync':True,'old_failure_reproduced_or_candidate_passed':True,
  'actual_numpy_torch_array_or_model_executed':False,'live_XFS_cause_established':False},sort_keys=True))


def test_parent_replacement_at_owned_close_preserves_failed_namespace(candidate,tmp_path):
 h=harness(tmp_path,candidate[0],transition=True)
 retained=tmp_path/'RETAINED-PUBLISHED-PARENT'
 def replace_parent():
  h.folder.rename(retained);h.folder.mkdir(mode=0o700)
 h.state.close_hook=replace_parent
 with pytest.raises(ValueError,match='Result parent changed after owned publication close'):publish(h)
 assert h.store.poisoned and KEY not in h.store.written and not h.state.live
 assert (retained/'pixels.npy').read_bytes()==DATA
