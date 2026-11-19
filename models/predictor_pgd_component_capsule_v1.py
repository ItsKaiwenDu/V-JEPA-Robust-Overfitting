"""Local byte-only component input capsule. No target or scientific entry exists.

Serialized tensors stay opaque. This subset reuses a pinned accepted numerical
review; it cannot replace the complete numerical audit or complete an old worker.
"""
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import stat
import tarfile
from models import predictor_pgd_split_lineage_v1 as lineage

CONTRACT='configs/predictor_pgd_component_capsule_v1.json'
CONTRACT_SHA='082ced8b88be471d6f443bec186ccfb4b0e93a7433f52313b71dba00b5fd10ca'
LINEAGE_SHA='a53103bfc665b30930af2b6046a347f81369b0de3a1b72ff2743bc5f3c643647'
_OPERANDS=lineage.operands
MAX_ARCHIVE=32*2**20
MAX_MEMBER=4*2**20
MAX_MANIFEST=128*1024
TENSORS=('q','k','v','grad_output','output','dq','dk','dv')


def digest(data):return hashlib.sha256(data).hexdigest()
def encoded(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def _pairs(rows):
 out={}
 for k,v in rows:
  if k in out:raise ValueError('Duplicate metadata key refused.')
  out[k]=v
 return out

def parsed(data):
 def bad(value):raise ValueError('Nonfinite metadata refused: '+value)
 return json.loads(data,object_pairs_hook=_pairs,parse_constant=bad)

def _require(value,message):
 if not value:raise ValueError(message)

def _name(value):
 _require(type(value) is str and value and '\\' not in value,'Canonical relative capsule name required.')
 p=PurePosixPath(value)
 _require(not p.is_absolute() and '..' not in p.parts and str(p)==value,'Canonical relative capsule name required.')
 return value

def contract(data):
 _require(type(data) is bytes and digest(data)==CONTRACT_SHA,'Exact frozen capsule contract required.')
 c=parsed(data)
 _require(c['caps']=={'archive_bytes':MAX_ARCHIVE,'member_bytes':MAX_MEMBER,'manifest_bytes':MAX_MANIFEST,'members':36},'Original capsule caps required.')
 _require(len(c['source_files'])==35,'Exact source member inventory required.')
 for name,d in c['source_files'].items():
  _name(name);_name(d['repository_path'])
  _require(type(d['bytes']) is int and 0<=d['bytes']<=MAX_MEMBER,'Bounded immutable source extent required.')
 _require(lineage.operands is _OPERANDS and digest(Path(lineage.__file__).read_bytes())==LINEAGE_SHA,'Frozen operand metadata planner changed.')
 return c

def _identity(s):return (s.st_dev,s.st_ino,s.st_uid,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def _read(path,cap,want):
 path=Path(path)
 _require(path.is_absolute() and path.resolve(strict=True)==path,'Canonical local input path required.')
 a=path.lstat();parent=path.parent.stat()
 _require(stat.S_ISREG(a.st_mode) and a.st_uid==os.getuid() and a.st_nlink==1 and 0<=a.st_size<=cap,'Single-link bounded ordinary source required; never open special files.')
 with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
  b=os.fstat(f.fileno());data=f.read(cap+1);after=os.fstat(f.fileno())
 _require(_identity(a)==_identity(b)==_identity(after)==_identity(path.lstat()) and _identity(parent)==_identity(path.parent.stat()),'Held local source or parent identity changed.')
 _require(len(data)==a.st_size and digest(data)==want,'Pinned source bytes changed.')
 return data

def validate_members(members,contract_bytes):
 """Compare opaque bytes and join saved metadata; never deserialize tensor data."""
 c=contract(contract_bytes);spec=c['source_files']
 _require(type(members) is dict and set(members)==set(spec),'Exact capsule source names required; no extras or partial files.')
 for name,d in spec.items():
  raw=members[name]
  _require(type(raw) is bytes and len(raw)==d['bytes'] and digest(raw)==d['sha256'],'Capsule source bytes changed: '+name)
 j=lambda n:parsed(members[n])
 review=j('provenance/accepted-numerical-review.json');run=j('provenance/run-status.json');session=j('provenance/session.json')
 control=c['source_control_sha256'];pid=c['historical_source_pid'];host=c['historical_source_host']
 _require(review['audit_result']=='PASS' and review['numerical_scoped_saved_passed'] is True and review['gradient_objectives_completed']==8 and review['same_input_pairs_passed']==4,'Accepted saved numerical review required.')
 _require(run['control_sha256']==control and run['pid']==pid and run['execution_host']==host and run['complete'] is True and run['passed'] is True and run['gradient_objectives_started']==8 and run['candidate_points']==4,'Completed historical numerical identity required.')
 _require(all(run[k] is True for k in ('all_methods_hooks_flags_restored','model_unchanged','global_rng_unchanged')),'Saved numerical restoration gates required.')
 _require(all(run[k] is False for k in ('fresh_rollout_replay_qualified','performance_qualified','research_admitted')),'Saved numerical limits must remain explicit.')
 _require(session['complete'] is False and session['phase']=='failed' and session['completed']==['proof','numerical'] and session['control_sha256']==control,'Historical failed session must stay failed.')
 receipts=[r for r in session['receipts'] if r['role']=='numerical'];_require(len(receipts)==1,'One accepted numerical receipt required.')
 receipt=receipts[0];r=receipt['receipt'];spec=c['source_files']
 _require(receipt['pid']==pid and receipt['saved_gates_returned'] is True and r['numerical_scoped'] is True and r['gradient_objectives']==8 and r['worker_sha256']==spec['provenance/run-status.json']['sha256'],'Saved numerical receipt identity changed.')
 _require(r['same_input_checks']==review['worker_recorded_comparisons'],'Saved comparison receipt changed.')
 original=None
 for i in range(8):
  prefix='provenance/objective-'+str(i)+'/'
  obj=j(prefix+'numerical.json');done=j(prefix+'numerical-complete.json')
  identity={k:obj[k] for k in ('kind','index','step','mode','execution_host','local_tiny','pid','control_sha256')}
  expected={'kind':'pretrained-pgd-split-numerical-objective-v1','index':i,'step':i//2,'mode':'candidate' if i%2 else 'original','execution_host':host,'local_tiny':False,'pid':pid,'control_sha256':control}
  _require(identity==expected and all(obj[k] is True for k in ('complete','passed','adapter_restored','model_unchanged','parameter_gradients_absent','input_dot_grad_absent')),'Restored accepted objective metadata required.')
  _require(obj['baseline_sha256']==run['baseline_sha256'] and obj['numerical_bounds']==run['numerical_bounds'] and obj['research_admitted'] is False,'Original numerical baseline/bounds required.')
  _require(done['identity']==identity and done['passed'] is True and done['worker_sha256']==spec[prefix+'numerical.json']['sha256'] and done['proof_files']==obj['proof_files'],'Objective completion join changed.')
  _require(len(obj['artifacts'])==37 and done['artifacts']=={n:{'file':d['file'],'sha256':d['sha256']} for n,d in obj['artifacts'].items()},'Saved 37-artifact descriptors required; arrays are not recertified here.')
  if i%2:
   _require(obj['comparison']==r['same_input_checks'][i//2] and all(obj['artifacts'][n]['sha256']==original['artifacts'][n]['sha256'] for n in original['artifacts'] if n!='input'),'Saved same-input comparison lineage changed.')
  else:original=obj
 parent=j('provenance/parent-numerical.json')
 _require(parent['pid']==pid and parent['exit_status']==0 and parent['parent_observation_passed'] is True,'Saved numerical supervisor record required.')
 cap=j('boundary/boundary.json');done=j('boundary/boundary-complete.json')
 _require(cap['control_sha256']==control and cap['pid']==pid and cap['execution_host']==host and cap['capture_complete'] is True and cap['mode']=='original' and cap['local_tiny'] is False,'Original boundary provenance required.')
 _require(cap['selected']==c['source_selection'] and cap['split_lineage']['source_objective_index']==0,'Only original objective-zero selected attention is allowed.')
 _require(all(cap[k] is True for k in ('observer_restored','alias_restored','tensor_versions_unchanged')) and all(cap[k] is False for k in ('native_forward_modified','native_cotangent_replaced','native_full_model_qualified','CUDA_kernel_qualified','research_admitted','performance_qualified')),'Original unchanged boundary gates required.')
 _require(done['capture_complete'] is True and done['control_sha256']==control and done['header']=={'file':'boundary.json','bytes':len(members['boundary/boundary.json']),'sha256':c['boundary_sha256']},'Boundary completion identity changed.')
 hashes=receipt['source_hashes']
 _require(hashes=={'numerical_sha256':spec['provenance/run-status.json']['sha256'],'original_sha256':spec['provenance/objective-0/numerical.json']['sha256'],'boundary_sha256':c['boundary_sha256']},'Historical source identity must remain separate from a future execution control.')
 for name in TENSORS:
  d=cap['artifacts'][name];raw=members['boundary/'+name+'.pt']
  _require(d['file']==name+'.pt' and d['bytes']==len(raw) and d['sha256']==digest(raw) and d['saved_before_numerical_gate'] is True,'Canonical serialized operand/reference changed.')
 layout=_OPERANDS(cap)
 _require(members['boundary/attention-calls.jsonl']==members['provenance/objective-0/native-calls.jsonl'],'Boundary and original native call inventories differ.')
 calls=[parsed(line) for line in members['boundary/attention-calls.jsonl'].splitlines()]
 scopes=j('boundary/attention-scopes.json');original_scopes=j('provenance/objective-0/scopes.json')
 _require(len(calls)==1152 and cap['selected_call']['actual_call']==calls[24] and scopes['scopes']==original_scopes['scopes'] and scopes['refusals']==original_scopes['refusals']==[] and len(scopes['scopes'])==1,'Selected native call/root metadata changed.')
 scope=scopes['scopes'][0]
 _require(scope['state']=='complete' and scope['hook_removed'] is True and scope['pid']==pid and scope['input_id']==cap['split_lineage']['source_value_id'],'Original full-input root/restoration identity changed.')
 return {'kind':'pgd-component-capsule-manifest-v1','contract_sha256':CONTRACT_SHA,'files':spec,
  'source_control_sha256':control,'boundary_sha256':c['boundary_sha256'],'accepted_numerical_review_sha256':c['accepted_numerical_review_sha256'],
  'selected':c['source_selection'],'layout':layout,'references':[n for n in TENSORS if n not in lineage.NAMES],
  'reuse':{'accepted_full_model_objectives':8,'accepted_same_input_pairs':4,'new_full_model_objectives':0,'old_session_stays_failed':True},
  'remaining_component_recipe':{'roles':['native','forward-preserving'],'forwards_per_role':1,'outer_VJPs_per_role':1,'candidate_inner_math_VJPs':1,'warmups':0,'extra_full_model_capture':0},
  'execution_control_sha256':None,'execution_host':None,'target_runtime_integrated':False,'target_execution_authorized':False,
  'opaque_tensor_bytes_only':True,'complete_numerical_array_evidence_included':False,'saved_metrics_recomputed':False,
  'actual_CUDA_execution_admitted':False,'component_accuracy_qualified':False,'full_EOT8_backend_qualified':False,
  'performance_replay_cost_or_training_admitted':False}

def _archive(members,manifest):
 payload={'MANIFEST.json':encoded(manifest),**members}
 _require(len(payload['MANIFEST.json'])<=MAX_MANIFEST and sum(map(len,payload.values()))+len(payload)*1024+10240<=MAX_ARCHIVE,'Bounded deterministic archive required.')
 out=io.BytesIO()
 with tarfile.open(fileobj=out,mode='w:',format=tarfile.USTAR_FORMAT) as tf:
  for name,data in sorted(payload.items()):
   info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600;info.uid=info.gid=info.mtime=0
   tf.addfile(info,io.BytesIO(data))
 raw=out.getvalue();_require(len(raw)<=MAX_ARCHIVE,'Archive bound exceeded.')
 return raw

def build(repository):
 """Return an unbound local capsule; no CLI, upload or execution path."""
 root=Path(repository).absolute();_require(root.resolve(strict=True)==root,'Canonical repository required.')
 cb=_read(root/CONTRACT,MAX_MANIFEST,CONTRACT_SHA);c=contract(cb)
 members={n:_read(root/d['repository_path'],MAX_MEMBER,d['sha256']) for n,d in c['source_files'].items()}
 manifest=validate_members(members,cb);raw=_archive(members,manifest)
 return raw,manifest

def verify_archive(raw,contract_bytes):
 """Read fixed ordinary archive members into bounded bytes; never extract/load."""
 c=contract(contract_bytes);_require(type(raw) is bytes and 0<len(raw)<=MAX_ARCHIVE,'Bounded complete capsule bytes required.')
 allowed={'MANIFEST.json':{'bytes':None},**c['source_files']};members={}
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tf:
  for member in tf:
   name=member.name;_name(name)
   _require(name in allowed and name not in members and len(members)<c['caps']['members'],'Unknown or duplicate capsule member.')
   cap=MAX_MANIFEST if name=='MANIFEST.json' else allowed[name]['bytes']
   _require(member.type==tarfile.REGTYPE and not member.pax_headers and member.mode==0o600 and member.uid==member.gid==member.mtime==0 and member.uname==member.gname=='' and 0<=member.size<=cap,'Only fixed bounded ordinary canonical members are allowed.')
   if name!='MANIFEST.json':_require(member.size==cap,'Exact source member extent required.')
   with tf.extractfile(member) as f:data=f.read(cap+1)
   _require(len(data)==member.size,'Whole capsule member required.')
   members[name]=data
 _require(set(members)==set(allowed),'Complete closed capsule inventory required.')
 manifest_bytes=members.pop('MANIFEST.json');manifest=validate_members(members,contract_bytes)
 _require(manifest_bytes==encoded(manifest) and raw==_archive(members,manifest),'Canonical manifest/headers/padding required; no concatenated or trailing data.')
 return {'kind':'pgd-component-capsule-local-verification-v1','archive_bytes':len(raw),'archive_sha256':digest(raw),'contract_sha256':CONTRACT_SHA,
  'source_files':len(members),'tensor_files':8,'member_bytes':sum(map(len,members.values())),
  'source_bytes_and_metadata_verified':True,'tensor_or_array_contents_deserialized':False,
  'complete_numerical_array_evidence_reverified':False,'historical_session_stays_failed':True,
  'new_full_model_objectives':0,'new_component_VJPs':0,'target_runtime_integrated':False,'target_execution_authorized':False,'training_admitted':False}
