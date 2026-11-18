"""Integrated source/cache/journal/header checks; target and GPU are always inert."""
import ast
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_publication_integration_v1 as integration
from models import predictor_pgd_publication_bootstrap_v1 as boot
from models import predictor_pgd_publication_bound_scratch_v1 as scratch
from models import predictor_pgd_publication_collection_v1 as collection
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def package():
 control,sources,readers,manifest,metadata=integration.assemble(ROOT)
 c,p,sha=boot.control(control);ib,isha=boot.installation(c,sha)
 layout=scratch.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),ib,isha)
 return NS(control=control,sources=sources,readers=readers,manifest=manifest,metadata=metadata,c=c,p=p,sha=sha,ib=ib,isha=isha,layout=layout)


def test_full_payload_integrates_only_scanner_and_bootstrap_and_preserves_every_scope(package):
 oldc,oldb,oldp,old,files=integration.load_frozen(ROOT)
 assert len(package.sources)==69 and sum(map(len,package.sources.values()))==888096 and package.metadata['unchanged_uploaded_sources']==67
 changed={n for n in old if old[n]!=package.sources[n]};assert changed=={integration.SCRATCH,integration.BOOT}
 for key in ('scope','base_text','design_text','host_change','host10614_binding','runtime_repair'):
  assert package.p[key]==oldp[key]
 assert package.p['scope']['host']['hostname']=='linux10614'
 assert all(package.sources[n]==old[n] for n in old if n not in changed)
 contract=boot.parsed(package.p['scratch_contract_text']);oldcontract=boot.parsed(oldp['scratch_contract_text'])
 assert contract.pop('publication_observation')==integration.RULE and contract==oldcontract
 assert not package.metadata['runnable_outer_client_or_target_handoff_prepared'] and not package.metadata['authorized']
 assert not any(n in sys.modules for n in ('torch','numpy'))


def test_dependent_control_contract_installation_and_reader_pins_join_without_science(package):
 x=package;ct=boot.parsed(x.p['scratch_contract_text']);meta=x.metadata
 assert boot.POLICY_SHA==integration.digest(x.c['policy_text'].encode())==meta['policy_sha256']
 assert scratch.CONTRACT_SHA==x.p['scratch_contract_sha256']==integration.digest(x.p['scratch_contract_text'].encode())
 assert x.p['fixed_support']['predictor_pgd_split_scratch_v1.py']==integration.digest(x.sources[integration.SCRATCH])
 assert x.p['publication_scan_repair']['rules']==ct['publication_observation']==integration.RULE
 for n,d in x.c['supervisor_sources'].items():assert d=={'bytes':len(x.sources['supervisor/'+n]),'sha256':integration.digest(x.sources['supervisor/'+n])}
 assert x.readers[integration.READERS['scratch']]==x.sources[integration.SCRATCH] and x.readers[integration.READERS['bootstrap']]==x.sources[integration.BOOT]
 result=integration.verify_readers(ROOT,x.readers);assert result['source_identity_verified']
 assert collection.boot is boot and collection.scratch is scratch and len(x.layout.assets)==172 and len(x.layout.result)==419
 template=integration.literals.read(ROOT/integration.TEMPLATE)
 def reverse(raw,trace):
  parts=[];cursor=0
  for e in trace:parts.extend((raw[cursor:e['new_begin']],e['old_literal'].encode()));cursor=e['new_end']
  return b''.join(parts+[raw[cursor:]])
 assert reverse(x.sources[integration.SCRATCH],meta['source_literal_edits'][integration.SCRATCH])==template
 old=integration.load_frozen(ROOT)[3]
 assert reverse(x.sources[integration.BOOT],meta['source_literal_edits'][integration.BOOT])==old[integration.BOOT]
 restored=x.readers[integration.READERS['collection']]
 for one,two in ((b'predictor_pgd_publication_bootstrap_v1 as boot',b'predictor_pgd_split_bootstrap_v1 as boot'),(b'predictor_pgd_publication_bound_scratch_v1 as scratch',b'predictor_pgd_split_scratch_v1 as scratch')):restored=restored.replace(one,two,1)
 assert restored==x.sources[integration.COLLECTION]

@pytest.mark.parametrize('fault',['reader','old-contract','old-control'])
def test_stale_scanner_readers_contract_and_control_are_rejected(package,tmp_path,fault):
 if fault=='reader':
  for n,data in package.readers.items():
   path=tmp_path/n;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data+(b'\n' if n==integration.READERS['scratch'] else b''))
  with pytest.raises(ValueError):integration.verify_readers(tmp_path,package.readers)
 elif fault=='old-contract':
  oldp=integration.load_frozen(ROOT)[2]
  with pytest.raises(ValueError,match='Exact source/storage'):scratch.Layout(oldp['scratch_contract_text'].encode(),package.p['base_text'].encode(),package.p['design_text'].encode(),package.ib,package.isha)
 else:
  oldb=integration.load_frozen(ROOT)[1]
  with pytest.raises(ValueError,match='Exact new supervised'):boot.control(integration.encoded(oldb))

@pytest.mark.parametrize('fault',['old-scanner','missing-scanner'])
def test_bootstrap_requires_new_fixed_scanner_support_descriptor(package,fault):
 c=copy.deepcopy(package.c)
 if fault=='old-scanner':c['supervisor_sources']['predictor_pgd_split_scratch_v1.py']['sha256']=integration.scanner.FROZEN_SHA
 else:c['supervisor_sources'].pop('predictor_pgd_split_scratch_v1.py')
 with pytest.raises(ValueError):boot.control(boot.encoded(c))


def write_private(root,name,data):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True,mode=0o700);q=p.parent
 while q!=root:q.chmod(0o700);q=q.parent
 with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(data)
 return p

@pytest.mark.parametrize('fault',['none','scanner','bootstrap'])
def test_actual_pinned_source_cache_binds_all69_new_sources_without_installing(package,tmp_path,fault):
 root=tmp_path/'source-only';root.mkdir(mode=0o700)
 for name,data in package.sources.items():write_private(root,name,data)
 if fault!='none':
  name=integration.SCRATCH if fault=='scanner' else integration.BOOT
  old=integration.load_frozen(ROOT)[3][name];path=root/name;path.write_bytes(old)
  with pytest.raises(ValueError):boot.PinnedImporter(root,scratch.identity_binding(root),package.c,package.p)
 else:
  finder=boot.PinnedImporter(root,scratch.identity_binding(root),package.c,package.p)
  assert len(finder.files)==69 and len(finder.packet_imports)==2 and not finder.loaded and not finder.installed
  assert finder.files['models.predictor_pgd_split_scratch_v1'][1]==package.sources[integration.SCRATCH]
  assert finder.find_spec('models.predictor_pgd_split_scratch_v1').loader is finder
  assert not any(n in sys.modules for n in ('torch','numpy'))
  (tmp_path/'INTEGRATED-PINNED-CACHE.json').write_text(json.dumps({'source_files':69,'installed':False,'loaded_project_modules':0,'bound_scanner_sha256':integration.digest(package.sources[integration.SCRATCH])},sort_keys=True))



# Pure full-manifest fixture rebound to the new policy; all sizes are explicit fakes.
def header_fixture(x):
    auth=boot.encoded({'kind':'pgd-split-supervised-university-exception-v1','approved':True,'control_sha256':x.sha,
                       'maximum_new_attempts':1,'user_basis':'EXPLICIT INERT HEADER FIXTURE, NOT A TARGET GRANT'})
    session={'kind':'pgd-split-parent-session-v1','control_sha256':x.sha,'parent_pid':os.getpid(),
             'connection_started':0.,'work_deadline':2400.,'connection_deadline':3000.,'phase':'failed',
             'complete':False,'completed':[],'receipts':[],'scientific_qualified':False,'training_admitted':False}
    win=collection.Window(0.,100.,clock=lambda:100.)
    receipt=collection.work_receipt(boot.encoded(session),x.sha,boot.digest(auth),x.isha,win)
    extras={'control.json':x.control,'authorization.json':auth,'session.json':boot.encoded(session),
            collection.RECEIPT:boot.encoded(receipt),collection.STATUS:boot.encoded({'EXPLICIT':'INERT SYNTHETIC METADATA'})}
    pins=dict(x.layout.assets);pins.update({n:{'bytes':len(v),'sha256':boot.digest(v)} for n,v in extras.items()})
    uid=os.getuid();files={n:dict(d,mode=0o600,mtime_ns=1000000000+i) for i,(n,d) in enumerate(sorted(pins.items()))}
    inventory={n:{'identity':[500,1000+i,uid,0o600,1],'bytes':v['bytes'],'allocated_bytes':((v['bytes']+4095)//4096)*4096,
                  'mtime_ns':v['mtime_ns']} for i,(n,v) in enumerate(files.items())}
    inventory['emergency-space.bin']={'identity':[500,900,uid,0o600,1],'bytes':16*scratch.MiB,'allocated_bytes':16*scratch.MiB,'mtime_ns':100}
    dirs={'.'}
    for name in inventory:dirs.update(str(p) for p in Path(name).parents)
    directories={n:{'identity':[500,2000+i,uid,0o700],'allocated_bytes':4096} for i,n in enumerate(sorted(dirs))}
    storage=x.layout.assess(inventory,directories,{'available_bytes':8*scratch.GiB,'available_inodes':2048},phase='collection')
    h={'kind':'pgd-split-evidence-stream-v1','control_sha256':x.sha,'authorization_sha256':boot.digest(auth),
       'root_binding':{'root':'/EXPLICIT-INERT/scratch','root_identity':list(directories['.']['identity']),
                       'parent':'/EXPLICIT-INERT','parent_identity':[500,1,uid,0o700]},
       'files':files,'inventory':inventory,'directories':directories,'storage':storage,
       'final_receipt_sha256':boot.digest(extras[collection.RECEIPT]),'session_sha256':boot.digest(extras['session.json']),
       'collection_started':100.,'collection_deadline':700.,'connection_started':0.,'connection_deadline':3000.,
       'logical_bytes':sum(v['bytes'] for v in files.values()),'omitted_content':['emergency-space.bin'],
       'scientific_qualified':False,'training_admitted':False}
    return h,auth


@pytest.mark.parametrize('fault',['none','old-scanner','old-bootstrap','old-control','live-reserve-shape','partial','clock','admission'])
def test_full_collection_header_keeps_new_pins_and_original_strict_settled_accounting(package,tmp_path,fault):
 h,auth=header_fixture(package)
 if fault=='old-scanner':h['files'][integration.SCRATCH]['sha256']=integration.scanner.FROZEN_SHA
 elif fault=='old-bootstrap':h['files'][integration.BOOT]['sha256']=integration.digest(integration.load_frozen(ROOT)[3][integration.BOOT])
 elif fault=='old-control':h['control_sha256']=integration.load_frozen(ROOT)[0]['supervised_control_sha256']
 elif fault=='live-reserve-shape':h['storage']['publication_accounting']={'EXPLICIT':'capture-only metadata cannot enter collection assessment'}
 elif fault=='partial':h['inventory']['capture/results/run-status.json.partial']=dict(h['inventory']['session.json'])
 elif fault=='clock':h['collection_deadline']+=1
 elif fault=='admission':h['scientific_qualified']=True
 if fault!='none':
  with pytest.raises(ValueError):collection.validate_header(h,package.control,auth)
 else:
  layout=collection.validate_header(h,package.control,auth);assert len(layout.assets)==172 and len(layout.result)==419
  assert 'publication_accounting' not in h['storage']
  wire=io.BytesIO();window=collection.Window(0.,100.,clock=lambda:100.)
  writer=collection.WireWriter(wire,window);raw=boot.encoded(h);writer.write(collection.MAGIC+len(raw).to_bytes(4,'big')+raw)
  decoder=collection.WireReader(io.BytesIO(wire.getvalue()),window);encoded,decoded=decoder.frame(collection.MAGIC,collection.HEADER_CAP)
  assert encoded==raw and decoded==h and decoder.hasher.hexdigest()==writer.hasher.hexdigest()
  collection.validate_header(decoded,package.control,auth)
  (tmp_path/'INTEGRATED-STRICT-COLLECTION-HEADER.json').write_bytes(raw)


def journal_case(package,tmp_path):
 root=tmp_path/'journal';tmp_path.chmod(0o700);root.mkdir(mode=0o700)
 final=write_private(root,'capture/results/run-status.json',b'{"EXPLICIT":"old inert metadata"}\n')
 write_private(root,'capture/results/run-status.json.partial',b'{"EXPLICIT":"new inert metadata"}\n')
 parent_id=(final.parent.stat().st_dev,final.parent.stat().st_ino);state=NS(fired=False,queries=[],now=1000.,reports=[])
 ns={'__name__':'EXPLICIT-INTEGRATED-SCANNER','__file__':'<LOCAL-INTEGRATED-SCANNER>'}
 exec(compile(package.sources[integration.SCRATCH],ns['__file__'],'exec'),ns)
 proxy=NS(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')})
 class Entry:
  def __init__(self,e):self.name=e.name;self.number=e.inode()
  def inode(self):return self.number
 class Entries:
  def __init__(self,rows):self.rows=rows
  def __enter__(self):return iter(self.rows)
  def __exit__(self,*a):return False
 def scan(fd):
  with os.scandir(fd) as rows:data=[Entry(e) for e in rows]
  data.sort(key=lambda e:e.name.endswith('.partial'));return Entries(data)
 def named(path,*a,**kw):
  fd=kw.get('dir_fd')
  if fd is not None and (os.fstat(fd).st_dev,os.fstat(fd).st_ino)==parent_id and str(path)=='run-status.json.partial' and not state.fired:
   state.fired=True;os.replace(final.with_name(final.name+'.partial'),final)
  return os.stat(path,*a,**kw)
 proxy.scandir=scan;proxy.stat=named;ns['os']=proxy
 guard=ns['Guard'](root,ns['identity_binding'](root),package.layout)
 scan_original=guard.scan
 def observed(**kw):
  value=scan_original(**kw);state.reports.append(value[0]);return value
 guard.scan=observed
 coord={'__name__':'EXPLICIT-INTEGRATED-JOURNAL','__file__':'<LOCAL-INTEGRATED-JOURNAL>'}
 exec(compile(package.sources[integration.COORDINATOR],coord['__file__'],'exec'),coord)
 coord.update(boot=boot,scratch=scratch)
 def fake_query(args,**kw):
  state.queries.append(args);assert args[0]=='nvidia-smi'
  if '--query-compute-apps=' in args[1]:return ''
  return 'NVIDIA GeForce RTX 3070, GPU-c7a30a6f-08fb-371f-41b3-86a13a1a632d, 8192, 43, 580.178.04\n'
 coord['subprocess']=NS(check_output=fake_query,PIPE=object())
 journal=coord['OuterJournal'](guard,lambda:state.now)
 return NS(root=root,journal=journal,state=state,guard=guard)

@pytest.mark.parametrize('fault',['none','foreign-status'])
def test_real_journal_persists_integrated_capture_reserves_and_retains_status_ownership(package,tmp_path,fault):
 x=journal_case(package,tmp_path)
 try:
  x.journal.checkpoint('numerical')
  status=json.loads((x.root/'storage-status.json').read_bytes());account=status['last_storage']['publication_accounting']
  assert x.state.fired and x.journal.calls==1 and len(x.state.queries)==2 and x.journal.refusals==0
  assert account['logical_reserve_bytes']==256*1024 and status['last_storage']['logical_bytes']==account['observed_logical_bytes']+account['logical_reserve_bytes']
  assert (x.root/'storage-status.json').stat().st_size<scratch.ROOT_META['storage-status.json'][0]
  assert account['metadata_partial_slots_reserved']==1 and not status['scientific_qualified'] and not status['training_admitted']
  assert sum('publication_accounting' in v for v in x.state.reports)==1
  if fault=='foreign-status':
   with (x.root/'storage-status.json').open('ab') as f:f.write(b' ')
   x.state.now=1011.
   with pytest.raises(ValueError,match='Only this journal'):x.journal.checkpoint('numerical')
  else:
   (tmp_path/'INTEGRATED-JOURNAL-RESERVE.json').write_bytes((x.root/'storage-status.json').read_bytes())
 finally:x.journal.close()
 assert x.journal.closed and x.journal.fd is None


def test_source_bundle_caps_and_scanner_rule_are_explicit_without_new_execution_scope(package):
 x=package
 assert len(x.control)<256*1024 and len(integration.encoded({n:base64.b64encode(v).decode() for n,v in x.sources.items()}))+len(x.control)+65536<2*2**20
 rule=x.p['publication_scan_repair']['rules'];assert rule['maximum_resolutions_per_scan']==2 and rule['metadata_partial_slots_shared']==2
 assert rule['extra_metadata_bytes_shared']==512*1024 and rule['metadata_plus_trace_bytes_shared']==64*scratch.MiB
 assert rule['collection_setup_proof_and_settled_scans_strict'] and not rule['atomic_snapshot_kernel_quota_continuous_bound_or_computation_authentication']
 assert x.p['scope']['resource']['work_cap_seconds']==2400 and x.p['scope']['resource']['collection_seconds']==600
 assert x.p['scope']['resource']['connection_including_login_seconds']==3000
 assert x.p['scope']['automatic_retries']==0 and x.p['scope']['paid_compute'] is False
 assert not x.metadata['authorized'] and not x.metadata['runnable_outer_client_or_target_handoff_prepared']
