"""New local byte/provenance/archive checks; no tensor or scientific execution."""
import ast,copy,hashlib,io,json,os,signal,socket,subprocess,sys,tarfile
from pathlib import Path
import pytest
from models import predictor_pgd_component_capsule_v1 as capsule
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def no_scientific_or_remote_work(monkeypatch):
 def forbidden(*args,**kwargs):pytest.fail('Capsule checks cannot start processes, signals, timers or network.')
 for module,name in ((subprocess,'Popen'),(os,'system'),(os,'fork'),(os,'posix_spawn'),(os,'kill'),(signal,'setitimer'),(signal,'alarm'),(socket,'socket')):
  if hasattr(module,name):monkeypatch.setattr(module,name,forbidden)
 yield
 assert not any(name in sys.modules for name in ('torch','numpy'))

@pytest.fixture(scope='module')
def prepared():
 cb=(ROOT/capsule.CONTRACT).read_bytes();raw,manifest=capsule.build(ROOT)
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tf:members={m.name:tf.extractfile(m).read() for m in tf if m.name!='MANIFEST.json'}
 return cb,raw,manifest,members

def save(folder,name,value):
 raw=value if type(value) is bytes else capsule.encoded(value)
 p=folder/name
 with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(raw)
 return p


def test_real_saved_inputs_form_one_closed_opaque_capsule(prepared,tmp_path):
 cb,raw,manifest,members=prepared;receipt=capsule.verify_archive(raw,cb)
 assert receipt['source_files']==35 and receipt['tensor_files']==8 and receipt['member_bytes']==27234618
 assert receipt['archive_bytes']<32*2**20 and receipt['source_bytes_and_metadata_verified']
 assert not receipt['tensor_or_array_contents_deserialized'] and not receipt['complete_numerical_array_evidence_reverified']
 assert receipt['new_full_model_objectives']==receipt['new_component_VJPs']==0 and not receipt['target_runtime_integrated'] and not receipt['target_execution_authorized'] and not receipt['training_admitted']
 assert not any(n.endswith('.npy') or 'checkpoint' in n or 'private_key' in n or 'components/' in n for n in members)
 for name in capsule.TENSORS:
  d=manifest['files']['boundary/'+name+'.pt'];assert len(members['boundary/'+name+'.pt'])==d['bytes']==3147305
  assert hashlib.sha256(members['boundary/'+name+'.pt']).hexdigest()==d['sha256']
 save(tmp_path,'VERIFIED-COMPONENT-CAPSULE.tar',raw);save(tmp_path,'VERIFIED-CAPSULE.json',receipt);save(tmp_path,'MANIFEST.json',manifest)


def test_manifest_preserves_layout_reuse_limits_and_unbound_execution_identity(prepared,tmp_path):
 _,_,m,_=prepared
 assert m['reuse']=={'accepted_full_model_objectives':8,'accepted_same_input_pairs':4,'new_full_model_objectives':0,'old_session_stays_failed':True}
 assert m['execution_control_sha256'] is None and m['execution_host'] is None and m['source_control_sha256']=='fbf7a4806e751b4d4dbaadd953a734444986306eb2304ba39bac1bf1570b8b5b'
 v=m['layout']['operands']['v'];assert v['storage_bytes']==9437184 and v['storage_offset_elements']==768 and v['stride']==[2359296,32,1152,1]
 assert m['layout']['unknown_padding_filled_zero'] and not m['layout']['native_physical_alignment_identity_claimed']
 assert m['references']==['output','dq','dk','dv']
 assert m['remaining_component_recipe']=={'roles':['native','forward-preserving'],'forwards_per_role':1,'outer_VJPs_per_role':1,'candidate_inner_math_VJPs':1,'warmups':0,'extra_full_model_capture':0}
 assert not m['component_accuracy_qualified'] and not m['full_EOT8_backend_qualified'] and not m['performance_replay_cost_or_training_admitted']
 save(tmp_path,'UNBOUND-CONTINUATION-CONTRACT.json',m)


@pytest.mark.parametrize('name',('boundary/q.pt','boundary/grad_output.pt','boundary/dq.pt','boundary/boundary.json','provenance/run-status.json','provenance/objective-7/numerical-complete.json','provenance/accepted-numerical-review.json'))
def test_every_critical_input_or_review_stays_pinned(prepared,tmp_path,name):
 cb,_,_,members=prepared;changed=dict(members);changed[name]=changed[name][:-1]+bytes([changed[name][-1]^1])
 with pytest.raises(ValueError,match='Capsule source bytes changed') as e:capsule.validate_members(changed,cb)
 save(tmp_path,'REFUSAL.json',{'mutation':name,'error':str(e.value),'scientific_work':False})


@pytest.mark.parametrize('fault',('missing','unexpected','partial','failed-component-substitution'))
def test_closed_input_inventory_cannot_include_failed_component_or_partial_data(prepared,tmp_path,fault):
 cb,_,_,members=prepared;changed=dict(members)
 if fault=='missing':del changed['boundary/v.pt']
 elif fault=='unexpected':changed['checkpoint.pt']=b'OPAQUE-FAKE'
 elif fault=='partial':changed['boundary/q.pt.partial']=changed.pop('boundary/q.pt')
 else:changed['components/native/output.npy']=changed.pop('boundary/output.pt')
 with pytest.raises(ValueError,match='Exact capsule source names') as e:capsule.validate_members(changed,cb)
 save(tmp_path,'REFUSAL.json',{'mutation':fault,'error':str(e.value),'scientific_work':False})


@pytest.mark.parametrize('fault',('new-control','new-host','false-acceptance','lower-cap'))
def test_contract_cannot_rebind_old_numerical_work_or_create_authorization(prepared,tmp_path,fault):
 cb,_,_,members=prepared;c=json.loads(cb)
 if fault=='new-control':c['source_control_sha256']='1'*64
 elif fault=='new-host':c['historical_source_host']='linux99999'
 elif fault=='false-acceptance':c['new_execution_authorized']=True
 else:c['caps']['archive_bytes']-=1
 with pytest.raises(ValueError,match='Exact frozen capsule contract') as e:capsule.validate_members(members,capsule.encoded(c))
 save(tmp_path,'REFUSAL.json',{'mutation':fault,'error':str(e.value),'scientific_work':False})


def hostile_archive(fault):
 out=io.BytesIO()
 with tarfile.open(fileobj=out,mode='w:',format=tarfile.USTAR_FORMAT) as tf:
  t=tarfile.TarInfo('boundary/q.pt');t.size=3147305;t.mode=0o600
  if fault=='traversal':t.name='../q.pt'
  elif fault=='absolute':t.name='/q.pt'
  elif fault=='symlink':t.type=tarfile.SYMTYPE;t.linkname='q.pt';t.size=0
  elif fault=='hardlink':t.type=tarfile.LNKTYPE;t.linkname='q.pt';t.size=0
  elif fault=='directory':t.type=tarfile.DIRTYPE;t.size=0
  elif fault=='oversized':t.size=4*2**20+1
  elif fault=='mode':t.mode=0o644
  elif fault=='duplicate':
   t.name='MANIFEST.json';t.size=2;tf.addfile(t,io.BytesIO(b'{}'));tf.addfile(t,io.BytesIO(b'{}'));return out.getvalue()
  # Deliberately omit content: header rejection must happen before any read.
  tf.addfile(t)
 return out.getvalue()

@pytest.mark.parametrize('fault',('traversal','absolute','symlink','hardlink','directory','oversized','mode','duplicate'))
def test_archive_headers_cannot_escape_or_widen_the_fixed_inventory(prepared,tmp_path,fault):
 cb,_,_,_=prepared;raw=hostile_archive(fault)
 with pytest.raises(ValueError) as e:capsule.verify_archive(raw,cb)
 save(tmp_path,'HOSTILE-HEADER.tar',raw);save(tmp_path,'REFUSAL.json',{'mutation':fault,'error':str(e.value),'scientific_work':False})


@pytest.mark.parametrize('fault',('trailing','truncated','manifest-spoof'))
def test_whole_archive_and_manifest_are_canonical(prepared,tmp_path,fault):
 cb,raw,manifest,members=prepared
 if fault=='trailing':changed=raw+b'UNTRUSTED-TRAILER'
 elif fault=='truncated':changed=raw[:-1]
 else:
  m=copy.deepcopy(manifest);m['target_execution_authorized']=True;changed=capsule._archive(members,m)
 with pytest.raises(ValueError,match='Canonical manifest') as e:capsule.verify_archive(changed,cb)
 save(tmp_path,'REFUSAL.json',{'mutation':fault,'error':str(e.value),'scientific_work':False})


@pytest.mark.parametrize('fault',('symlink','hardlink','hash-drift'))
def test_local_source_read_never_follows_aliases_or_changed_bytes(tmp_path,fault):
 p=save(tmp_path,'opaque.bin',b'OPAQUE-NONSCIENTIFIC-BYTES');want=capsule.digest(p.read_bytes())
 if fault=='symlink':q=tmp_path/'alias';q.symlink_to(p);p=q
 elif fault=='hardlink':os.link(p,tmp_path/'second-name')
 else:want='0'*64
 with pytest.raises(ValueError) as e:capsule._read(p,1024,want)
 save(tmp_path,'REFUSAL.json',{'mutation':fault,'error':str(e.value),'scientific_work':False})


def test_operand_metadata_planner_source_and_function_identity_remain_pinned(prepared,monkeypatch):
 cb=prepared[0];monkeypatch.setattr(capsule.lineage,'operands',lambda cap:{})
 with pytest.raises(ValueError,match='planner changed'):capsule.contract(cb)


def test_builder_has_no_scientific_or_remote_execution_entry():
 tree=ast.parse((ROOT/'models/predictor_pgd_component_capsule_v1.py').read_bytes())
 imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
 assert all((n.module or '').split('.')[0] not in ('torch','numpy','subprocess','socket','signal') if isinstance(n,ast.ImportFrom) else all(a.name.split('.')[0] not in ('torch','numpy','subprocess','socket','signal') for a in n.names) for n in imports)
 assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('eval','exec','compile') for n in ast.walk(tree))
 assert {n.name for n in tree.body if isinstance(n,ast.FunctionDef)} >= {'build','verify_archive','validate_members'}
