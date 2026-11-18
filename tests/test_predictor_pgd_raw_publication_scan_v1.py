"""New raw-publication schedules with inert bytes; no arrays, child, GPU or proof."""
import ast,base64,hashlib,json,os,stat
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_raw_publication_scan_v1 as recipe
from models import predictor_pgd_publication_bootstrap_v1 as boot
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'runs/pretrained/diagnostic/pgd-publication-handoff-v1-20261117/handoff-parent/linux10614-final-unarmed'
FROZEN=(ROOT/recipe.SOURCE).read_bytes();CANDIDATE=(ROOT/recipe.CANDIDATE).read_bytes()
NAME='capture/results/numerical/objective-5/cotangent-14.npy'
DATA=b'INERT OWNED BYTE FIXTURE; this is not an array or experimental result.\n'

@pytest.fixture(scope='module')
def policy():
 c=json.loads((FOLDER/'CONTROL-CANDIDATE.json').read_bytes());bc,p,csha=boot.control(c['supervised_control_text'].encode());ib,isha=boot.installation(bc,csha)
 store=base64.b64decode(json.loads((FOLDER/'SOURCE-PAYLOAD.json').read_bytes())['overlay/models/predictor_pgd_split_store_v1.py'],validate=True)
 assert hashlib.sha256(store).hexdigest()=='7bcbd863eab7c23940fa1d32b194448b88353202c2100b66a9a30f59e4ef6022'
 return NS(c=c,p=p,ib=ib,isha=isha,store=store)

def write_private(path,data):
 path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 return path

def namespace(source):
 n={'__name__':'INERT_RAW_PUBLICATION_LOCAL_ONLY','__file__':'<FROZEN-OR-CANDIDATE-SCANNER>'};exec(compile(source,n['__file__'],'exec'),n);return n

def harness(tmp_path,policy,*,source=CANDIDATE,schedule='missing',fault=None,name=NAME,phase='capture',settled=False,metadata=0,edge=None):
 tmp_path.chmod(0o700);root=tmp_path/'scratch';root.mkdir(mode=0o700);(root/'capture').mkdir(mode=0o700)
 ns=namespace(source);m=NS(**ns);layout=m.Layout(policy.p['scratch_contract_text'].encode(),policy.p['base_text'].encode(),policy.p['design_text'].encode(),policy.ib,policy.isha)
 assert len(layout.result)==419 and len(layout.assets)==172
 guard=m.Guard(root,m.identity_binding(root),layout);final=root/name
 state=NS(events=[],tracked=set(),linked=False,unlinked=False,started=False,result=None,error=None,walks=0,parent_id=None,
  verified_final_stats=0,mutated=False,raw_reads=0,assessments=[],linked_partial_stats=0,raw_fstats=0)
 proxy=NS(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')})
 real_link,real_unlink=os.link,os.unlink
 def mutate():
  if state.mutated:return
  state.mutated=True
  if fault=='delete-final':final.unlink()
  elif fault in ('wrong-inode','replace-after-resolution'):
   p=write_private(root/'tmp/replacement',DATA);os.replace(p,final)
  elif fault=='symlink':final.unlink();final.symlink_to(root/'tmp/NO-CONTENT-READ')
  elif fault=='mode':final.chmod(0o644)
  elif fault=='hardlink':
   (root/'tmp').mkdir(mode=0o700,exist_ok=True);os.link(final,root/'tmp/alias')
  elif fault=='oversize':
   with final.open('ab') as f:f.write(b'x'*(layout.classify(name)['max_bytes']+1))
  elif fault=='partial-reappears':write_private(final.with_name(final.name+'.partial'),b'foreign next partial')
  elif fault=='mtime':os.utime(final,ns=(1234,5678))
  elif fault=='moved-parent':final.parent.rename(final.parent.with_name('moved-owned-directory'))
 def link_now():
  assert not state.linked;real_link(final.name+'.partial',final.name,src_dir_fd=state.writer_fd,dst_dir_fd=state.writer_fd,follow_symlinks=False)
  state.linked=True;state.events.append('REAL-OWNED-LINK')
  if fault=='linked-third':
   (root/'tmp').mkdir(mode=0o700,exist_ok=True);os.link(final,root/'tmp/third-name')
 def unlink_now():
  assert state.linked and not state.unlinked;real_unlink(final.name+'.partial',dir_fd=state.writer_fd)
  state.unlinked=True;state.events.append('REAL-OWNED-UNLINK')
  if fault in ('delete-final','wrong-inode','symlink','mode','hardlink','oversize'):mutate()
 def opened(*a,**kw):
  fd=os.open(*a,**kw);state.tracked.add(fd);return fd
 def dup(fd):fd=os.dup(fd);state.tracked.add(fd);return fd
 def close(fd):state.tracked.remove(fd);os.close(fd)
 proxy.open=opened;proxy.dup=dup;proxy.close=close
 proxy.read=lambda *a:pytest.fail('Scanner must never read file contents.')
 def matches(fd):return fd is not None and (os.fstat(fd).st_dev,os.fstat(fd).st_ino)==state.parent_id
 def stats(path,*a,**kw):
  fd=kw.get('dir_fd');leaf=str(path);target=matches(fd)
  if target and not state.unlinked:
   if schedule=='one-to-two-verify' and state.linked and leaf==final.name+'.partial':
    state.linked_partial_stats+=1
    if state.linked_partial_stats==2:unlink_now()
   if schedule=='missing' and leaf==final.name+'.partial':link_now();unlink_now()
   elif schedule=='two-to-missing' and leaf==final.name+'.partial':unlink_now()
   elif schedule=='two-to-one' and leaf==final.name:unlink_now()
  if target and state.unlinked and leaf==final.name:
   state.verified_final_stats+=1
   if state.verified_final_stats==2 and fault in ('replace-after-resolution','partial-reappears','mtime','moved-parent'):mutate()
  s=os.stat(path,*a,**kw)
  if target and leaf==final.name+'.partial' and schedule in ('one-to-two','one-to-two-verify') and not state.linked and fault in ('old-links','growing-partial'):
   attrs={k:getattr(s,k) for k in ('st_mode','st_dev','st_ino','st_uid','st_nlink','st_size','st_blocks','st_mtime_ns')}
   if fault=='old-links':attrs['st_nlink']=3
   else:attrs['st_size']=len(DATA)//2;attrs['st_mtime_ns']-=100
   s=NS(**attrs)
  return s
 proxy.stat=stats
 def fstats(fd):
  s=os.fstat(fd)
  if state.linked and stat.S_ISREG(s.st_mode) and fault=='allocation-drop':
   state.raw_fstats+=1
   if state.raw_fstats==1:
    attrs={k:getattr(s,k) for k in ('st_mode','st_dev','st_ino','st_uid','st_nlink','st_size','st_blocks','st_mtime_ns')}
    attrs['st_blocks']=(layout.classify(name)['max_bytes']+511)//512+16;return NS(**attrs)
  if state.linked and stat.S_ISREG(s.st_mode) and fault in ('device','uid'):
   attrs={k:getattr(s,k) for k in ('st_mode','st_dev','st_ino','st_uid','st_nlink','st_size','st_blocks','st_mtime_ns')};attrs['st_dev' if fault=='device' else 'st_uid']+=1;return NS(**attrs)
  return s
 proxy.fstat=fstats
 class Entry:
  def __init__(self,e):self.name=e.name;self.value=e.inode()
  def inode(self):return 0 if fault=='invalid-inode' and self.name==final.name+'.partial' else self.value
 class Entries:
  def __init__(self,rows,dynamic=False):self.rows=rows;self.dynamic=dynamic
  def __enter__(self):
   def generate():
    yield from self.rows
    if self.dynamic:
     link_now()
     with os.scandir(state.writer_fd) as items:extra=[Entry(e) for e in items if e.name==final.name]
     assert len(extra)==1;yield extra[0]
   return generate()
  def __exit__(self,*a):return False
 def scandir(fd):
  state.walks+=1
  with os.scandir(fd) as items:rows=[Entry(e) for e in items]
  target=matches(fd)
  if target:rows.sort(key=lambda e:e.name.endswith('.partial') if schedule=='two-to-missing' else not e.name.endswith('.partial'))
  return Entries(rows,dynamic=target and schedule in ('one-to-two','one-to-two-verify'))
 proxy.scandir=scandir;ns['os']=proxy
 real_assess=layout.assess
 def assessed(*a,**kw):
  report=real_assess(*a,**kw);state.assessments.append(json.loads(json.dumps(report)))
  if edge is not None:
   report=json.loads(json.dumps(report));reserve=layout.classify(name)['max_bytes']
   if edge=='allocated':report['allocated_bytes']=report['ceiling_bytes']-reserve//2
   elif edge=='logical':report['logical_bytes']=report['ceiling_bytes']-reserve//2
   elif edge=='raw-category':report['categories']['raw']=m.CAPS['raw']-reserve//2
   elif edge=='files':report['files']=1024
   else:raise AssertionError(edge)
  return report
 layout.assess=assessed
 writer_ns={'__name__':'ACTUAL_PINNED_INERT_BYTE_STORE','__file__':'<PINNED-REPAIRED-STORE>'};exec(compile(policy.store,writer_ns['__file__'],'exec'),writer_ns)
 writer_proxy=NS(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')})
 def observe():
  assert not state.started;state.started=True
  if metadata:
   for rel in ('session.json','storage-status.json')[:metadata]:
    write_private(root/rel,b'{"old":"owned metadata"}\n');write_private(root/(rel+'.partial'),b'{"new":"owned metadata"}\n')
   old_stat=proxy.stat
   def with_metadata(path,*a,**kw):
    if kw.get('dir_fd') is not None and (os.fstat(kw['dir_fd']).st_dev,os.fstat(kw['dir_fd']).st_ino)==(root.stat().st_dev,root.stat().st_ino) and str(path).endswith('.partial'):
     os.replace(path,str(path).removesuffix('.partial'),src_dir_fd=kw['dir_fd'],dst_dir_fd=kw['dir_fd']);state.events.append('REAL-OWNED-JSON-REPLACE')
    return old_stat(path,*a,**kw)
   proxy.stat=with_metadata
  if fault=='other-raw-partial':write_private(root/'capture/results/numerical/objective-6/loss.npy.partial',b'inert other pending raw')
  if fault=='pinned-name':
   guard.pins[name]={'test_only_pin':True}
  try:state.result=guard.scan(phase=phase,settled=settled)
  except BaseException as error:state.error=error;raise
 def linked(src,dst,**kw):
  state.writer_fd=kw['src_dir_fd'];s=os.fstat(state.writer_fd);state.parent_id=(s.st_dev,s.st_ino)
  if schedule in ('missing','one-to-two','one-to-two-verify'):observe()
  else:link_now()
 def unlinked(path,**kw):
  if schedule not in ('missing','one-to-two','one-to-two-verify'):observe()
  if not state.unlinked:unlink_now()
 writer_proxy.link=linked;writer_proxy.unlink=unlinked;writer_ns['os']=writer_proxy
 inventory=writer_ns['Inventory'](root/'capture',base_bytes=policy.p['base_text'].encode(),design_bytes=policy.p['design_text'].encode());writer=writer_ns['Store'](inventory)
 def run():
  try:return writer.write_bytes(Path(name).relative_to('capture').as_posix(),DATA)
  finally:
   assert not state.tracked,'Guard descriptor leak: '+str(state.tracked)
   record={'kind':'INERT-RAW-PUBLICATION-CASE','schedule':schedule,'fault':fault,'phase':phase,'settled':settled,'events':state.events,'metadata_resolutions_requested':metadata,'edge':edge,
    'pinned_writer_sha256':hashlib.sha256(policy.store).hexdigest(),'same_thread_controlled_schedule':True,'experimental_or_array_contents':False,'model_GPU_proof_or_child_work':False,'live_interleaving_proven':False,'result':state.result[0] if state.result else None,
    'error_type':type(state.error).__name__ if state.error else None,'error':str(state.error)[:512] if state.error else None,'guard_owned_FDs_remaining':len(state.tracked),'stat_identity_or_allocation_fields_explicitly_fake':fault in ('device','uid','old-links','growing-partial','allocation-drop') or edge is not None}
   (tmp_path/'INERT-RAW-PUBLICATION-CASE.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
 return NS(run=run,state=state,guard=guard,layout=layout,root=root,final=final,writer=writer,namespace=ns)

@pytest.mark.parametrize('schedule',['missing','two-to-missing','two-to-one','one-to-two'])
def test_frozen_approved_scanner_reproduces_raw_publication_refusal(policy,tmp_path,schedule):
 h=harness(tmp_path,policy,source=FROZEN,schedule=schedule)
 with pytest.raises((FileNotFoundError,ValueError)):h.run()
 assert h.state.started and h.state.error is not None
 if schedule in ('missing','two-to-missing'):assert isinstance(h.state.error,FileNotFoundError) and 'cotangent-14.npy.partial' in str(h.state.error)

@pytest.mark.parametrize('schedule',['missing','two-to-missing','two-to-one','one-to-two'])
def test_candidate_accepts_exact_raw_inode_transition_and_reserves_full_typed_cap(policy,tmp_path,schedule):
 h=harness(tmp_path,policy,schedule=schedule);value=h.run();report,files,dirs=h.state.result;a=report['publication_accounting'];cap=h.layout.classify(NAME)['max_bytes']
 assert value['sha256']==hashlib.sha256(DATA).hexdigest() and h.final.read_bytes()==DATA
 assert a['raw_observation']['listed_inode']==h.final.stat().st_ino and a['raw_partial_slots_reserved']==1 and a['raw_reserve_bytes']==cap
 assert a['logical_reserve_bytes']==cap and a['allocated_reserve_bytes']>=cap and not a['raw_observation']['completion_admitted']
 assert report['logical_bytes']==a['observed_logical_bytes']+cap and report['categories']['raw']==a['observed_categories']['raw']+cap
 assert report['allocated_bytes']==a['observed_allocated_bytes']+a['allocated_reserve_bytes'] and report['partials']['raw']==1
 assert report['result_categories']['raw']==len(DATA)+cap and not report['scientific_qualified']
 assert a['raw_observation']['state']==('two-owned-names' if schedule in ('one-to-two','one-to-two-verify') else 'single-final')
 assert a['raw_virtual_partial_slots_charged']==(0 if schedule in ('one-to-two','one-to-two-verify') else 1)
 assert h.state.events==['REAL-OWNED-LINK','REAL-OWNED-UNLINK'] and h.state.walks==len(dirs)

@pytest.mark.parametrize('fault',['delete-final','wrong-inode','symlink','mode','hardlink','oversize','partial-reappears','invalid-inode','device','uid','replace-after-resolution','mtime','moved-parent','pinned-name'])
def test_changed_or_ambiguous_raw_publication_refuses_and_closes_only_owned_FDs(policy,tmp_path,fault):
 h=harness(tmp_path,policy,fault=fault)
 with pytest.raises((ValueError,OSError)):h.run()
 assert h.state.started and h.state.error is not None

@pytest.mark.parametrize('fault',['old-links','growing-partial'])
def test_prepublication_raw_observation_cannot_erase_bad_links_but_can_bound_growth(policy,tmp_path,fault):
 h=harness(tmp_path,policy,schedule='one-to-two',fault=fault)
 if fault=='old-links':
  with pytest.raises(ValueError,match='Previously observed raw identity'):h.run()
 else:
  h.run();assert h.state.result[0]['publication_accounting']['raw_reserve_bytes']==h.layout.classify(NAME)['max_bytes']

@pytest.mark.parametrize('phase,settled',[('setup',False),('proof',False),('collection',False),('capture',True),('collection',True)])
def test_non_capture_and_settled_scans_do_not_resolve_disappearing_raw_names(policy,tmp_path,phase,settled):
 h=harness(tmp_path,policy,phase=phase,settled=settled)
 with pytest.raises((FileNotFoundError,ValueError)):h.run()
 assert h.state.started

@pytest.mark.parametrize('edge',['allocated','logical','raw-category','files'])
def test_fake_near_cap_observation_still_refuses_full_raw_publication_reserve(policy,tmp_path,edge):
 h=harness(tmp_path,policy,edge=edge)
 with pytest.raises(ValueError,match='reserves exceed original shared limits'):h.run()
 assert len(h.state.assessments)==1


def test_another_live_raw_partial_consumes_the_same_single_slot(policy,tmp_path):
 h=harness(tmp_path,policy,fault='other-raw-partial')
 with pytest.raises(ValueError,match='reserves exceed original shared limits'):h.run()
 assert h.state.error is not None

@pytest.mark.parametrize('count',[1,2])
def test_raw_and_existing_JSON_resolutions_share_original_category_and_slot_caps(policy,tmp_path,count):
 h=harness(tmp_path,policy,metadata=count);h.run();report,files,dirs=h.state.result;a=report['publication_accounting']
 assert a['metadata_partial_slots_reserved']==count and report['partials']['metadata']==count and report['partials']['raw']==1
 assert a['metadata_reserve_bytes']==sum(h.layout.classify(n)['max_bytes'] for n in ('session.json','storage-status.json')[:count])
 assert report['extra_metadata_bytes']==a['metadata_reserve_bytes']<=512*1024 and report['categories']['metadata']+report['categories']['trace']<=64*2**20
 assert h.state.walks==len(dirs)


def test_stable_two_link_capture_keeps_existing_pending_pair_accounting(policy,tmp_path):
 h=harness(tmp_path,policy,schedule='stable-two');h.run();report,files,dirs=h.state.result
 assert report['publication_pairs']==[[NAME,NAME+'.partial']] and report['partials']['raw']==1
 assert files[NAME]==files[NAME+'.partial'] and files[NAME]['identity'][4]==2
 assert 'publication_accounting' not in report and report['logical_bytes']==len(DATA)

@pytest.mark.parametrize('phase,settled',[('capture',False),('collection',True)])
def test_quiet_complete_raw_and_metadata_observation_is_exact_to_frozen_source(policy,tmp_path,phase,settled):
 root=tmp_path/'quiet';tmp_path.chmod(0o700);root.mkdir(mode=0o700)
 write_private(root/NAME,DATA);write_private(root/'session.json',b'{"inert":"quiet"}\n')
 for current,dirs,files in os.walk(root):Path(current).chmod(0o700) # Only this newly owned quiet fixture.
 values=[]
 for source in (FROZEN,CANDIDATE):
  ns=namespace(source);m=NS(**ns);layout=m.Layout(policy.p['scratch_contract_text'].encode(),policy.p['base_text'].encode(),policy.p['design_text'].encode(),policy.ib,policy.isha)
  g=m.Guard(root,m.identity_binding(root),layout);values.append(g.scan(phase=phase,settled=settled))
 assert values[0]==values[1] and 'publication_accounting' not in values[1][0]


def test_recipe_whole_source_inverse_and_other_AST_definitions_are_exact():
 data,record=recipe.build(FROZEN);assert data==CANDIDATE
 change=record['replacement'];assert data[:change['begin']]+change['old_method'].encode()+data[change['new_end']:]==FROZEN
 a,b=ast.parse(FROZEN),ast.parse(data)
 for tree in (a,b):
  guard=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Guard');guard.body=[n for n in guard.body if not isinstance(n,ast.FunctionDef) or n.name!='scan']
 assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
 assert record['local_source_only_not_execution_binding'] and not record['target_handoff_ready'] and not record['authorized']
 with pytest.raises(ValueError):recipe.build(FROZEN+b'# source drift')


def test_additional_partial_unlink_during_final_verification_is_bounded_and_single_link(policy,tmp_path):
 h=harness(tmp_path,policy,schedule='one-to-two-verify');h.run();report,files,dirs=h.state.result;a=report['publication_accounting']
 assert h.state.linked_partial_stats==2 and a['raw_observation']['state']=='single-final' and a['raw_observation']['absence_seen']
 assert a['raw_virtual_partial_slots_charged']==1 and files[NAME]['identity'][4]==1 and NAME+'.partial' not in files
 assert h.state.walks==len(dirs) and h.state.events==['REAL-OWNED-LINK','REAL-OWNED-UNLINK']


def test_additional_third_link_during_linked_state_refuses(policy,tmp_path):
 h=harness(tmp_path,policy,schedule='one-to-two',fault='linked-third')
 with pytest.raises(ValueError,match='bounded final'):h.run()
 assert h.state.started and h.final.stat().st_nlink==3


def test_additional_observed_allocation_high_water_is_retained_after_fake_drop(policy,tmp_path):
 h=harness(tmp_path,policy,fault='allocation-drop');h.run();report,files,dirs=h.state.result;a=report['publication_accounting'];cap=h.layout.classify(NAME)['max_bytes']
 high=((cap+511)//512+16)*512
 assert a['raw_observation']['allocated_reserve_bytes']==a['allocated_reserve_bytes']==high>cap
 assert report['allocated_bytes']==a['observed_allocated_bytes']+high


def test_additional_catalogued_immutable_boundary_bytes_use_same_raw_bounds_without_decode(policy,tmp_path):
 name='capture/results/boundary/q.pt';h=harness(tmp_path,policy,name=name);h.run();report,files,dirs=h.state.result
 assert h.final.read_bytes()==DATA and report['publication_accounting']['raw_observation']['final']==name
 assert report['publication_accounting']['raw_reserve_bytes']==4*2**20


def test_additional_second_raw_resolution_is_refused_without_forgetting_the_first(policy,tmp_path):
 root=tmp_path/'scratch';tmp_path.chmod(0o700);root.mkdir(mode=0o700)
 names=[NAME,'capture/results/numerical/objective-6/loss.npy'];ns=namespace(CANDIDATE);m=NS(**ns)
 for name in names:write_private(root/(name+'.partial'),DATA)
 for current,dirs,files in os.walk(root):Path(current).chmod(0o700)
 layout=m.Layout(policy.p['scratch_contract_text'].encode(),policy.p['base_text'].encode(),policy.p['design_text'].encode(),policy.ib,policy.isha);g=m.Guard(root,m.identity_binding(root),layout)
 proxy=NS(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')});events=[]
 def scheduled(path,*a,**kw):
  if kw.get('dir_fd') is not None and str(path).endswith('.partial'):
   leaf=str(path);os.link(leaf,leaf.removesuffix('.partial'),src_dir_fd=kw['dir_fd'],dst_dir_fd=kw['dir_fd'],follow_symlinks=False);os.unlink(leaf,dir_fd=kw['dir_fd']);events.append(leaf)
  return os.stat(path,*a,**kw)
 proxy.stat=scheduled;ns['os']=proxy
 with pytest.raises(ValueError,match='One distinct unpinned'):g.scan(phase='capture')
 assert len(events)==2
 (tmp_path/'SECOND-RAW-RESOLUTION-REFUSAL.json').write_text(json.dumps({'controlled_inert_events':events,'no_array_or_model_work':True,'maximum_one_raw_resolution_retained':True}))


@pytest.mark.parametrize('name',['tmp/untracked.npy.partial','original-packet.tar.gz.partial','capture/results/plan.json.partial','capture/results/numerical/objective-99/unknown.npy.partial'])
def test_additional_missing_cache_asset_immutable_metadata_or_unknown_name_cannot_use_raw_resolution(policy,tmp_path,name):
 root=tmp_path/'scratch';tmp_path.chmod(0o700);root.mkdir(mode=0o700);write_private(root/name,DATA)
 for current,dirs,files in os.walk(root):Path(current).chmod(0o700)
 ns=namespace(CANDIDATE);m=NS(**ns);layout=m.Layout(policy.p['scratch_contract_text'].encode(),policy.p['base_text'].encode(),policy.p['design_text'].encode(),policy.ib,policy.isha);g=m.Guard(root,m.identity_binding(root),layout)
 proxy=NS(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')});events=[]
 def vanished(path,*a,**kw):
  if kw.get('dir_fd') is not None and str(path)==Path(name).name:
   os.unlink(path,dir_fd=kw['dir_fd']);events.append(str(path))
  return os.stat(path,*a,**kw)
 proxy.stat=vanished;ns['os']=proxy
 with pytest.raises((FileNotFoundError,ValueError)):g.scan(phase='capture')
 assert len(events)<=1 and not (root/name.removesuffix('.partial')).exists()
 (tmp_path/'NON_RAW_OR_UNKNOWN_DISAPPEARANCE-REFUSAL.json').write_text(json.dumps({'name':name,'controlled_inert_deletion':events,'raw_resolution_not_allowed':True,'no_array_or_model_work':True}))
