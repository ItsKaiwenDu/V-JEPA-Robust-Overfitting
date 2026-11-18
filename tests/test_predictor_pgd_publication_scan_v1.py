"""Controlled inert JSON publication schedules; no children, arrays, GPU or proof."""
import ast
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_publication_scan_v1 as recipe
from models import predictor_pgd_host10614_bootstrap_v1 as boot
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'runs/pretrained/diagnostic/pgd-host10614-v1-20261117/handoff-parent/linux10614-final-unarmed'
FROZEN=(ROOT/recipe.SOURCE).read_bytes()
CANDIDATE=(ROOT/recipe.CANDIDATE).read_bytes()
PLAIN='capture/results/run-status.json'
DATA=b'{"inert":"new owned metadata; no model or arrays"}\n'

@pytest.fixture(scope='module')
def policy():
 c=json.loads((FOLDER/'CONTROL-CANDIDATE.json').read_bytes());control=c['supervised_control_text'].encode()
 bc,p,sha=boot.control(control);ib,isha=boot.installation(bc,sha)
 payload=json.loads((FOLDER/'SOURCE-PAYLOAD.json').read_bytes());store=base64.b64decode(payload['overlay/models/predictor_pgd_split_store_v1.py'])
 assert hashlib.sha256(store).hexdigest()=='7bcbd863eab7c23940fa1d32b194448b88353202c2100b66a9a30f59e4ef6022'
 return NS(c=c,p=p,ib=ib,isha=isha,store=store)

def private_write(path,data):
 path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 q=path.parent
 while 'pytest-fixtures' not in q.name and q!=q.parent:q.chmod(0o700);q=q.parent
 with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 return path

def namespace(source):
 ns={'__name__':'LOCAL_INERT_JSON_SCHEDULE','__file__':'<FROZEN-OR-CANDIDATE-SCANNER>'}
 exec(compile(source,ns['__file__'],'exec'),ns);return ns

def case(tmp_path,policy,*,source=CANDIDATE,name=PLAIN,order='final-first',timing='missing',fault=None,phase='capture',settled=False,pins=False):
 tmp_path.chmod(0o700);root=tmp_path/'scratch';root.mkdir(mode=0o700)
 final=root/name;final.parent.mkdir(mode=0o700,parents=True);q=final.parent
 while q!=root:q.chmod(0o700);q=q.parent
 parent_id=(final.parent.stat().st_dev,final.parent.stat().st_ino)
 ns=namespace(source);m=NS(**ns);layout=m.Layout(policy.p['scratch_contract_text'].encode(),policy.p['base_text'].encode(),policy.p['design_text'].encode(),policy.ib,policy.isha)
 assert len(layout.result)==419 and len(layout.assets)==172
 g=m.Guard(root,m.identity_binding(root),layout)
 state=NS(root=root,final=final,name=name,events=[],tracked=set(),snapshots=0,rename_done=False,used=False,result=None,
          error=None,original_error=None,final_stats=0,spoof_allocation=0)
 real_replace=os.replace
 proxy=NS(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')})
 def opened(*a,**kw):
  fd=os.open(*a,**kw);state.tracked.add(fd)
  if str(a[0])==final.name and state.rename_done and fault=='partial-reappears':
   private_write(final.with_name(final.name+'.partial'),b'new partial after the observed gap')
  return fd
 def dup(fd):
  child=os.dup(fd);state.tracked.add(child);return child
 def closed(fd):state.tracked.remove(fd);os.close(fd)
 proxy.open=opened;proxy.dup=dup;proxy.close=closed
 def mutate():
  if fault=='delete-final':final.unlink()
  elif fault in ('wrong-inode','replace-after-resolution'):
   p=private_write(root/'tmp/replacement.json',DATA);os.replace(p,final)
  elif fault=='symlink':
   final.unlink();os.symlink(root/'tmp/OUTSIDE-NOT-READ',final)
  elif fault=='mode':final.chmod(0o644)
  elif fault=='hardlink':
   (root/'tmp').mkdir(mode=0o700,exist_ok=True);os.link(final,root/'tmp/alias')
  elif fault=='oversize':
   with final.open('ab') as f:f.write(b'x'*(256*1024+1))
  elif fault in ('partial-reappears','reappears-after-resolution'):private_write(final.with_name(final.name+'.partial'),b'next publication')
  elif fault=='mtime':os.utime(final,ns=(1234,5678))
  elif fault=='moved-parent':
   # Move a nested result directory under another name after binding; no follow.
   final.parent.rename(final.parent.with_name('moved-unbound'))
 def publish():
  assert not state.rename_done;real_replace(final.name+'.partial',final.name,src_dir_fd=state.writer_parent,dst_dir_fd=state.writer_parent)
  state.rename_done=True;state.events.append('REAL-ATOMIC-REPLACE')
  if fault not in ('partial-reappears','replace-after-resolution','reappears-after-resolution','mtime','moved-parent'):mutate()
 def stat_call(path,*a,**kw):
  fd=kw.get('dir_fd');target=fd is not None and (os.fstat(fd).st_dev,os.fstat(fd).st_ino)==parent_id
  leaf=str(path)
  if target and not state.rename_done:
   trigger=leaf==final.name+'.partial' if timing=='missing' else leaf==final.name
   if trigger:publish()
  if target and state.rename_done and leaf==final.name:
   state.final_stats+=1
   if state.final_stats==2 and fault in ('replace-after-resolution','reappears-after-resolution','mtime','moved-parent'):mutate()
  s=os.stat(path,*a,**kw)
  return s
 proxy.stat=stat_call
 class Entry:
  def __init__(self,e):self.name=e.name;self.number=e.inode()
  def inode(self):return 0 if fault=='invalid-inode' and self.name==final.name+'.partial' else self.number
 class Entries:
  def __init__(self,rows):self.rows=rows
  def __enter__(self):return iter(self.rows)
  def __exit__(self,*a):return False
 def scandir(fd):
  with os.scandir(fd) as stream:rows=[Entry(e) for e in stream]
  state.snapshots+=1
  if (os.fstat(fd).st_dev,os.fstat(fd).st_ino)==parent_id:
   # Freeze the real readdir names/inodes before triggering the stat schedule.
   rows.sort(key=lambda e:(e.name.endswith('.partial')) if order=='final-first' else (not e.name.endswith('.partial')))
  return Entries(rows)
 proxy.scandir=scandir
 def fstat(fd):
  s=os.fstat(fd)
  if fault in ('device','uid') and state.rename_done and stat.S_ISREG(s.st_mode) and (s.st_dev,s.st_ino)==(final.lstat().st_dev,final.lstat().st_ino):
   attrs={n:getattr(s,n) for n in ('st_mode','st_dev','st_ino','st_uid','st_nlink','st_size','st_blocks','st_mtime_ns')}
   attrs['st_dev' if fault=='device' else 'st_uid']+=1;return NS(**attrs)
  return s
 proxy.fstat=fstat;ns['os']=proxy
 writer_ns={'__name__':'ACTUAL_PINNED_REPAIRED_STORE','__file__':'<INERT-JSON-WRITER>'}
 exec(compile(policy.store,writer_ns['__file__'],'exec'),writer_ns)
 writer_proxy=NS(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')})
 def replacement(src,dst,**kw):
  assert str(src)==final.name+'.partial' and str(dst)==final.name
  state.writer_parent=kw['src_dir_fd'];assert not state.used;state.used=True
  try:state.result=g.scan(phase=phase,settled=settled)
  except BaseException as error:state.error=error;raise
  assert state.rename_done
 writer_proxy.replace=replacement
 inventory=writer_ns['Inventory'](root/'capture',base_bytes=policy.p['base_text'].encode(),design_bytes=policy.p['design_text'].encode())
 writer=writer_ns['Store'](inventory)
 writer.write_bytes(Path(name).relative_to('capture').as_posix(),b'{"inert":"old metadata"}\n')
 writer_ns['os']=writer_proxy
 if pins:
  s=final.stat();g.pins[name]={'identity':(*m.directory_id(s),s.st_nlink),'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'mtime_ns':s.st_mtime_ns}
 def run():
  try:return writer.write_bytes(Path(name).relative_to('capture').as_posix(),DATA)
  finally:assert not state.tracked,'Guard descriptor leak: '+str(state.tracked)
 return NS(state=state,run=run,guard=g,layout=layout,ns=ns,writer=writer,root=root,final=final)


def test_frozen_control_reproduces_missing_partial_during_actual_atomic_store_publication(policy,tmp_path):
 h=case(tmp_path,policy,source=FROZEN)
 with pytest.raises(FileNotFoundError,match='run-status.json.partial'):h.run()
 assert h.state.rename_done and h.state.error is not None and h.final.read_bytes()==DATA
 (tmp_path/'FORCED-FROZEN-SCAN-FAILURE.json').write_text(json.dumps({'forced_schedule':'list partial, atomic replace, stat missing partial','native_same_thread_inert_bytes_only':True,'matches_saved_error':True,'live_interleaving_proven':False}))

@pytest.mark.parametrize('order,timing',[('final-first','missing'),('partial-first','missing'),('partial-first','alias')])
def test_candidate_resolves_only_pinned_publication_and_charges_existing_budget(policy,tmp_path,order,timing):
 h=case(tmp_path,policy,order=order,timing=timing);result=h.run();report,files,dirs=h.state.result
 assert result['sha256']==hashlib.sha256(DATA).hexdigest() and h.final.read_bytes()==DATA
 assert set(files)=={PLAIN} and files[PLAIN]['identity'][1]==h.final.stat().st_ino
 account=report['publication_accounting'];assert account['logical_reserve_bytes']==256*1024 and account['metadata_partial_slots_reserved']==1
 assert account['allocated_reserve_bytes']>=256*1024 and report['logical_bytes']==len(DATA)+256*1024
 assert report['allocated_bytes']==account['observed_allocated_bytes']+account['allocated_reserve_bytes']
 assert account['observations'][0]['listed_inode']==h.final.stat().st_ino and not account['atomic_snapshot_or_continuous_allocation_claimed']
 assert h.state.events==['REAL-ATOMIC-REPLACE'] and h.state.snapshots==len(dirs)
 (tmp_path/'PINNED-PUBLICATION-RESOLUTION.json').write_text(json.dumps({'order':order,'timing':timing,'report':report,'file_bytes_are_inert_metadata':True,'actual_pinned_writer_passed':True},sort_keys=True))

@pytest.mark.parametrize('fault',['delete-final','wrong-inode','symlink','mode','hardlink','oversize','partial-reappears','invalid-inode','device','uid','replace-after-resolution','reappears-after-resolution','mtime','moved-parent'])
def test_ambiguous_or_changed_publication_refuses_without_retry_or_descriptor_leak(policy,tmp_path,fault):
 h=case(tmp_path,policy,fault=fault)
 with pytest.raises((ValueError,OSError)):h.run()
 assert h.state.error is not None and h.state.rename_done and h.state.events==['REAL-ATOMIC-REPLACE']

@pytest.mark.parametrize('phase,settled',[('capture',True),('setup',False),('proof',False),('collection',False),('collection',True)])
def test_settled_and_non_capture_phases_keep_original_missing_name_refusal(policy,tmp_path,phase,settled):
 h=case(tmp_path,policy,phase=phase,settled=settled)
 with pytest.raises(FileNotFoundError):h.run()
 assert h.state.events==['REAL-ATOMIC-REPLACE']


def test_pinned_mutable_name_cannot_be_resolved(policy,tmp_path):
 h=case(tmp_path,policy,pins=True)
 with pytest.raises(ValueError,match='unpinned JSON'):h.run()


def test_source_recipe_changes_only_scan_and_has_exact_whole_module_inverse():
 candidate,meta=recipe.build(FROZEN);assert candidate==CANDIDATE
 a=ast.parse(FROZEN);b=ast.parse(candidate)
 def stripped(tree):
  guard=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Guard');guard.body=[n for n in guard.body if not isinstance(n,ast.FunctionDef) or n.name!='scan'];return ast.dump(tree,include_attributes=False)
 assert stripped(a)==stripped(b)
 change=meta['replacement'];assert candidate[:change['begin']]+change['old_method'].encode()+candidate[change['new_end']:]==FROZEN
 assert meta['maximum_resolutions_per_scan']==2 and not meta['target_handoff_ready'] and not meta['authorized']
 assert not any(n in sys.modules for n in ('torch','numpy'))
 with pytest.raises(ValueError):recipe.build(FROZEN+b'\n')


# Additional pure-scanner boundary schedules use only tiny owned JSON fixtures.
# Synthetic allocation appears only in explicit fake stat fields, never in disk.
def staged_observation(tmp_path,policy,*,count=1,live=0,allocated_edge=False,old_link=False,missing=None):
 tmp_path.chmod(0o700);root=tmp_path/'scratch';root.mkdir(mode=0o700)
 paths=[PLAIN,'capture/results/numerical/objective-0/numerical.json','capture/results/numerical/objective-1/numerical.json'][:count]
 if missing is not None:paths=[missing.removesuffix('.partial')]
 for name in paths:
  private_write(root/name,b'old inert metadata');private_write(root/(name+'.partial'),b'new inert metadata')
 for i in range(live):private_write(root/('capture/results/numerical/objective-'+str(i+3)+'/numerical.json.partial'),b'live inert partial')
 if old_link:
  (root/'tmp').mkdir(mode=0o700);os.link(root/paths[0],root/'tmp/old-alias')
 ns=namespace(CANDIDATE);m=NS(**ns);layout=m.Layout(policy.p['scratch_contract_text'].encode(),policy.p['base_text'].encode(),policy.p['design_text'].encode(),policy.ib,policy.isha)
 g=m.Guard(root,m.identity_binding(root),layout);state=NS(renamed=set(),tracked=set(),assessments=[],walks=0)
 original_assess=layout.assess
 def assess(*a,**kw):
  result=original_assess(*a,**kw);state.assessments.append(result);return result
 layout.assess=assess
 proxy=NS(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')})
 def opened(*a,**kw):fd=os.open(*a,**kw);state.tracked.add(fd);return fd
 def dup(fd):v=os.dup(fd);state.tracked.add(v);return v
 def closed(fd):state.tracked.remove(fd);os.close(fd)
 proxy.open=opened;proxy.dup=dup;proxy.close=closed
 parents={(root/name).parent.stat().st_ino:(root/name).parent for name in paths}
 pending={str((root/(name+'.partial')).absolute()) for name in paths}
 def status(path,*a,**kw):
  fd=kw.get('dir_fd');parent=parents.get(os.fstat(fd).st_ino) if fd is not None else None
  key=str(parent/str(path)) if parent is not None else ''
  target=key if key in pending else str(parent/(str(path)+'.partial')) if missing and not missing.endswith('.partial') and parent is not None and key==str(root/missing) else ''
  if target in pending and target not in state.renamed:
   state.renamed.add(target)
   if missing is not None:
    (root/missing).unlink()
   else:os.replace(target,target.removesuffix('.partial'))
  return os.stat(path,*a,**kw)
 proxy.stat=status
 class Entry:
  def __init__(self,e):self.name=e.name;self.number=e.inode()
  def inode(self):return self.number
 class Entries:
  def __init__(self,entries):self.entries=entries
  def __enter__(self):return iter(self.entries)
  def __exit__(self,*a):return False
 def scan(fd):
  state.walks+=1
  with os.scandir(fd) as entries:rows=[Entry(e) for e in entries]
  rows.sort(key=lambda e:e.name.endswith('.partial'));return Entries(rows)
 proxy.scandir=scan
 root_inode=root.stat().st_ino
 if allocated_edge:
  actual=sum(q.lstat().st_blocks*512 for q in [root,*root.rglob('*')] if not str(q).endswith('.partial'))
  adjustment=9*m.GiB//4-128*1024-actual;assert adjustment>0 and adjustment%512==0
  def fstat(fd):
   s=os.fstat(fd)
   if s.st_ino!=root_inode:return s
   attrs={k:getattr(s,k) for k in ('st_mode','st_dev','st_ino','st_uid','st_nlink','st_size','st_blocks','st_mtime_ns')};attrs['st_blocks']+=adjustment//512
   return NS(**attrs)
  proxy.fstat=fstat
 ns['os']=proxy
 def run():
  try:return g.scan(phase='capture')
  finally:assert not state.tracked
 return NS(run=run,root=root,state=state,layout=layout)

@pytest.mark.parametrize('count,live,allowed',[(2,0,True),(3,0,False),(1,1,True),(2,1,False)])
def test_additional_publication_slots_and_metadata_reserves_share_original_cap(policy,tmp_path,count,live,allowed):
 h=staged_observation(tmp_path,policy,count=count,live=live)
 if not allowed:
  with pytest.raises(ValueError,match='Bounded distinct|original shared limits'):h.run()
 else:
  report,files,dirs=h.run();a=report['publication_accounting']
  assert a['metadata_partial_slots_reserved']==count and report['partials']['metadata']==live
  assert a['logical_reserve_bytes']==count*256*1024 and a['remaining_metadata_partial_slots']==2-count-live
  assert a['logical_reserve_bytes']+report['extra_metadata_bytes']<=512*1024
  assert h.state.walks==len(dirs)
  (tmp_path/'SHARED-PUBLICATION-BUDGET.json').write_text(json.dumps(report,sort_keys=True))


def test_additional_observed_allocation_fits_but_publication_reserve_still_refuses(policy,tmp_path):
 h=staged_observation(tmp_path,policy,allocated_edge=True)
 with pytest.raises(ValueError,match='reserve exceeds original shared limits'):h.run()
 assert len(h.state.assessments)==1
 observed=h.state.assessments[0];assert observed['allocated_bytes']<observed['ceiling_bytes']
 assert observed['allocated_bytes']+256*1024>observed['ceiling_bytes']
 (tmp_path/'SYNTHETIC-ALLOCATION-RESERVE-REFUSAL.json').write_text(json.dumps({'fake_allocation_fields_only':True,'observed':observed,'proof_or_preallocation':False},sort_keys=True))

@pytest.mark.parametrize('name',['tmp/free.partial','capture/results/plan.json.partial','capture/results/numerical/objective-0/input.npy.partial','original-packet.tar.gz.partial','capture/results/UNKNOWN.json.partial','capture/results/run-status.json'])
def test_additional_missing_cache_raw_immutable_asset_unknown_or_final_names_never_resolve(policy,tmp_path,name):
 h=staged_observation(tmp_path,policy,missing=name)
 with pytest.raises((ValueError,OSError)):h.run()
 assert len(h.state.renamed)==(0 if name=='capture/results/UNKNOWN.json.partial' else 1)
 # An unknown final name is rejected even before the scheduled partial stat.


def test_additional_recovery_cannot_hide_a_previously_observed_old_final_hardlink(policy,tmp_path):
 h=staged_observation(tmp_path,policy,old_link=True)
 with pytest.raises(ValueError,match='non-single-link final'):h.run()


def test_additional_frozen_scanner_rejects_stale_one_link_alias_schedule(policy,tmp_path):
 h=case(tmp_path,policy,source=FROZEN,order='partial-first',timing='alias')
 with pytest.raises(ValueError,match='Repeated single-link'):h.run()
 assert h.state.rename_done

@pytest.mark.parametrize('phase,settled',[('capture',False),('collection',True)])
def test_additional_quiet_scan_retains_original_file_and_budget_observation(policy,tmp_path,phase,settled):
 root=tmp_path/'scratch';tmp_path.chmod(0o700);root.mkdir(mode=0o700);private_write(root/PLAIN,b'quiet inert metadata')
 observations=[]
 for source in (FROZEN,CANDIDATE):
  ns=namespace(source);m=NS(**ns);layout=m.Layout(policy.p['scratch_contract_text'].encode(),policy.p['base_text'].encode(),policy.p['design_text'].encode(),policy.ib,policy.isha)
  g=m.Guard(root,m.identity_binding(root),layout);observations.append(g.scan(phase=phase,settled=settled))
 assert observations[0]==observations[1] and 'publication_accounting' not in observations[1][0]
