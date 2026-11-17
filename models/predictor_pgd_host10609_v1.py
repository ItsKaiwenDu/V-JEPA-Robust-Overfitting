"""Cold host-specific source assembly; never runs a target, model or proof.

Historical base/design/financial bytes remain the scientific reference plan.
The new supervisor policy exclusively authorizes the proposed execution identity.
"""
import ast
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import stat

OLD_HOST='linux10603'
HOST='linux10609'
OLD_UUID='GPU-e54fae90-feeb-8106-d8c0-1974fe48555f'
UUID='GPU-4e2a7c77-19bd-689f-6c08-60eb774c0d52'
ROOT_PACKAGE='runs/pretrained/diagnostic/predictor-pgd-outer-v4-20261116/handoff-parent/linux10603-unarmed'
FROZEN={
 ROOT_PACKAGE+'/CONTROL-CANDIDATE.json':'9e7d66ba7364ef2014fe61c9af40b4ea77f8b05fc62c91b6940e9eb7af0ef4e4',
 ROOT_PACKAGE+'/SOURCE-PAYLOAD.json':'837bf160bf5e5bb9238c18f07f6ba545403b11bc4001eca2cf57a4f6d1b6b310',
 'models/predictor_pgd_split_outer_v4.py':'32cd4c9d00943eefec2e53d2f80dfb8832820b18d6ccee759d6919195a94fcb2',
 'models/predictor_pgd_split_outer_client_v4.py':'6594dbaa9a9e147c10eedcf6ff4803dbdcaaf12a18e0deb1efe42d2010f39090',
 'scripts/run_predictor_pgd_split_outer_v4.py':'24c304613878aaa774ca08c2cd3eb2714263518544d6412d9675c43fe6c82f21',
 'runs/pretrained/diagnostic/eot-work-profile-university-20261029/known-hosts':'44efad85bc728f0784978efe43b1fa5824856a24e98bb69646a7cbd06222b32d'}
EXPECTED_RUNTIME={
 'overlay/attacks/pgd_pretrained_v1.py','overlay/models/_pgd_split_facade_v1.py',
 'overlay/models/_pgd_split_numerical_v1.py','overlay/models/_pgd_split_observer_v1.py',
 'overlay/models/_pgd_split_records_v1.py','overlay/models/_pgd_split_route_v1.py',
 'overlay/models/predictor_forward_preserving_target_v1.py','overlay/models/predictor_pgd_split_lineage_v1.py'}
NAMES={'outer':'models/predictor_pgd_split_outer_10609_v1.py',
 'client':'models/predictor_pgd_split_outer_client_10609_v1.py',
 'cli':'scripts/run_predictor_pgd_split_outer_10609_v1.py',
 'boot':'models/predictor_pgd_split_bootstrap_10609_v1.py',
 'collection':'models/predictor_pgd_split_collection_10609_v1.py',
 'builder':'models/predictor_pgd_host10609_v1.py','scratch':'models/predictor_pgd_split_scratch_10609_v1.py'}


def encoded(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(b):return hashlib.sha256(b).hexdigest()
def parsed(b):
 def pairs(rows):
  d={}
  for k,v in rows:
   if k in d:raise ValueError('Duplicate host-change metadata key.')
   d[k]=v
  return d
 def bad(v):raise ValueError('Nonfinite host-change metadata: '+v)
 return json.loads(b,object_pairs_hook=pairs,parse_constant=bad)


def read(path,cap=2*2**20,wanted=None):
 p=Path(path);s=p.lstat()
 def ident(v):return (v.st_dev,v.st_ino,v.st_uid,v.st_mode,v.st_nlink,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
 if (not p.is_absolute() or p.resolve(strict=True)!=p or not stat.S_ISREG(s.st_mode) or s.st_nlink!=1
     or s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode) not in (0o400,0o600,0o644) or not 0<s.st_size<=cap):
  raise ValueError('Canonical private/owned bounded source required.')
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  h=os.fstat(f.fileno());data=f.read(cap+1);after=os.fstat(f.fileno())
 if (ident(s)!=ident(h) or ident(s)!=ident(after) or ident(s)!=ident(p.lstat()) or len(data)!=s.st_size
     or wanted is not None and digest(data)!=wanted):raise ValueError('Frozen source bytes/identity changed: '+str(p))
 return data


def constants(data,mapping,*,host=False):
 """Replace reviewed literal values only; preserve both-host negative guards.

 Bytes constants (including BASE_BYTES/DESIGN_BYTES) and ordinary source spelling
 outside changed literal spans are untouched. Nested source strings are parsed.
 The saved edit list makes an exact byte inverse possible.
 """
 tree=ast.parse(data);parents={id(c):n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
 lines=data.splitlines(keepends=True);offset=[0]
 for line in lines:offset.append(offset[-1]+len(line))
 edits=[]
 for node in ast.walk(tree):
  if not isinstance(node,ast.Constant) or type(node.value) is not str:continue
  value=node.value;new=value
  parent=parents.get(id(node))
  exclusion=(host and isinstance(parent,(ast.Tuple,ast.List)) and {OLD_HOST,HOST}<=set(c.value for c in parent.elts if isinstance(c,ast.Constant) and type(c.value) is str))
  if exclusion:continue
  if value in mapping:new=mapping[value]
  elif host and OLD_HOST in value:
   if '\n' in value and ('class ' in value or 'def ' in value):
    nested,_=constants(value.encode(),mapping,host=True);new=nested.decode()
   else:new=value.replace(OLD_HOST,HOST)
  if new==value:continue
  begin=offset[node.lineno-1]+node.col_offset;end=offset[node.end_lineno-1]+node.end_col_offset
  edits.append((begin,end,data[begin:end],repr(new).encode()))
 edits.sort();pieces=[];cursor=0;newpos=0;record=[]
 for begin,end,old,new in edits:
  if begin<cursor:raise ValueError('Overlapping literal changes.')
  prefix=data[cursor:begin];pieces.extend((prefix,new));newpos+=len(prefix)
  record.append({'new_begin':newpos,'new_end':newpos+len(new),'old_literal':old.decode(),'new_literal':new.decode()})
  newpos+=len(new);cursor=end
 pieces.append(data[cursor:]);result=b''.join(pieces);compile(result,'<COLD-HOST-BOUND-SOURCE>','exec')
 reverse=[];cursor=0
 for item in record:
  reverse.extend((result[cursor:item['new_begin']],item['old_literal'].encode()));cursor=item['new_end']
 reverse.append(result[cursor:])
 if b''.join(reverse)!=data:raise ValueError('Exact host literal inverse failed.')
 return result,record


def assemble(repository):
 """Read source/metadata only. No packet, checkpoint, tensor, auth or target IO."""
 root=Path(repository)
 frozen={n:read(root/n,wanted=w) for n,w in FROZEN.items()}
 c=parsed(frozen[ROOT_PACKAGE+'/CONTROL-CANDIDATE.json']);b=parsed(c['supervised_control_text']);p=parsed(b['policy_text'])
 if b['policy_sha256']!='0d1c5d92a2d3fdb41907ba8b6d7db0028d39ba1b9ecdb8eec0482a384b2e8621' or digest(b['policy_text'].encode())!=b['policy_sha256']:
  raise ValueError('Original exact v4 policy required.')
 mapping=parsed(frozen[ROOT_PACKAGE+'/SOURCE-PAYLOAD.json'])
 descriptors={'supervisor/'+n:d for n,d in b['supervisor_sources'].items()}
 descriptors.update({'overlay/'+n:d for n,d in c['runtime_files'].items()})
 if set(mapping)!=set(descriptors) or len(mapping)!=69:raise ValueError('Exact original 69 sources required.')
 original={}
 for n,text in mapping.items():
  data=base64.b64decode(text,validate=True);d=descriptors[n]
  if len(data)!=d['bytes'] or digest(data)!=d['sha256']:raise ValueError('Frozen payload changed: '+n)
  original[n]=data
 sources=dict(original);edits={}
 for n,data in original.items():
  if n.startswith('overlay/'):
   changed,trace=constants(data,{OLD_HOST:HOST,OLD_UUID:UUID},host=True)
   if trace:sources[n]=changed;edits[n]=trace
 if set(edits)!=EXPECTED_RUNTIME:raise ValueError('Unexpected execution-host source surface: '+str(sorted(edits)))
 original_contract=parsed(p['scratch_contract_text']);contract=copy.deepcopy(original_contract)
 runtime={n:{'bytes':len(sources['overlay/'+n]),'sha256':digest(sources['overlay/'+n])} for n in c['runtime_files']}
 manifest={'kind':'pgd-split-host10609-runtime-manifest-v1','original_runtime_manifest_sha256':contract['runtime_manifest_sha256'],'runtime_files':runtime}
 contract['runtime_files']=runtime;contract['runtime_manifest_sha256']=digest(encoded(manifest))
 policy=copy.deepcopy(p);policy['scope']['host']['hostname']=HOST;policy['scope']['host']['gpu_uuid']=UUID
 policy['scratch_contract_text']=encoded(contract).decode();policy['scratch_contract_sha256']=digest(encoded(contract))
 policy['host_change']={'kind':'pgd-split-host10609-binding-v1','original_policy_sha256':b['policy_sha256'],'original_host':OLD_HOST,'execution_host':HOST,
   'historical_base_design_bytes_preserved':True,'base_plan_host_is_reference_provenance_not_current_execution_authority':True,
   'runtime_manifest_text':encoded(manifest).decode(),'no_old_claims_or_results_relabelled':True}
 sources['supervisor/predictor_pgd_split_scratch_v1.py'],edits['supervisor/predictor_pgd_split_scratch_v1.py']=constants(original['supervisor/predictor_pgd_split_scratch_v1.py'],{p['scratch_contract_sha256']:digest(encoded(contract))})
 policy_bytes=encoded(policy);psha=digest(policy_bytes)
 sources['supervisor/bootstrap.py'],edits['supervisor/bootstrap.py']=constants(original['supervisor/bootstrap.py'],{b['policy_sha256']:psha})
 sources['supervisor/predictor_pgd_split_coordinator_v1.py'],edits['supervisor/predictor_pgd_split_coordinator_v1.py']=constants(original['supervisor/predictor_pgd_split_coordinator_v1.py'],{OLD_HOST:HOST,OLD_UUID:UUID},host=True)
 for n,data in sources.items():compile(data,'<UNEXECUTED-10609-'+n+'>','exec')
 sourcepins={n.removeprefix('supervisor/'):{'bytes':len(v),'sha256':digest(v)} for n,v in sources.items() if n.startswith('supervisor/')}
 control=encoded({'kind':b['kind'],'policy_text':policy_bytes.decode(),'policy_sha256':psha,'supervisor_sources':sourcepins})
 if policy['base_text']!=p['base_text'] or policy['design_text']!=p['design_text'] or policy['scope']['resource']!=p['scope']['resource']:
  raise ValueError('Original scientific/financial/reference plan must remain exact.')
 for n in original:
  if n not in edits and sources[n]!=original[n]:raise ValueError('Undeclared source change.')
 result={'kind':'pgd-split-host10609-source-delta-v1','host':HOST,'gpu_uuid':UUID,'original_outer_control_sha256':digest(frozen[ROOT_PACKAGE+'/CONTROL-CANDIDATE.json']),
  'original_policy_sha256':b['policy_sha256'],'policy_sha256':psha,'bootstrap_sha256':digest(sources['supervisor/bootstrap.py']),
  'supervised_control_sha256':digest(control),'runtime_manifest':manifest,'source_edits':edits,
  'changed_source_files':len(edits),'unchanged_source_files':69-len(edits),'total_source_files':69,'total_source_bytes':sum(map(len,sources.values())),
  'historical_base_design_financial_scientific_bytes_unchanged':True,'authorization_created':False,'target_work_or_SSH_executed':False,
  'scientific_qualified':False,'training_admitted':False}
 return control,sources,result,frozen


def prepare_package(repository,destination):
 from models import predictor_pgd_split_outer_10609_v1 as outer
 root,dest=Path(repository),Path(destination)
 if root.resolve(strict=True)!=root or not dest.is_absolute() or dest.parent.resolve(strict=True)!=dest.parent or os.path.lexists(dest) or stat.S_IMODE(dest.parent.stat().st_mode)!=0o700:
  raise ValueError('Fresh private exact handoff required.')
 supervised,sources,metadata,_=assemble(root)
 if metadata['policy_sha256']!=outer.POLICY_SHA or metadata['bootstrap_sha256']!=outer.BOOT_SHA:raise ValueError('Generated host/policy bindings changed.')
 outer.verify_directory_policy_sources(root,__import__('models.predictor_pgd_split_source_bundle_v1',fromlist=['read_source']))
 outer.verify_refusal_policy_sources(root,__import__('models.predictor_pgd_split_source_bundle_v1',fromlist=['read_source']))
 b=parsed(supervised);p=parsed(b['policy_text']);runtime=parsed(p['scratch_contract_text'])['runtime_files']
 pins={n:digest(read(root/n)) for n in outer.LOCAL_SOURCES}
 c={'kind':'pgd-split-outer-control-host10609-v1','supervised_control_text':supervised.decode(),'supervised_control_sha256':digest(supervised),
    'outer_source_sha256':pins[outer.LOCAL_SOURCES[0]],'local_source_sha256':pins,'runtime_files':runtime,'repository':str(root),'local_handoff':str(dest),
    'packet_path':outer.PACKET_PATH,'known_hosts_text':outer.KNOWN_HOSTS,'known_hosts_sha256':digest(outer.KNOWN_HOSTS.encode()),'expected_ssh_fingerprint':outer.FINGERPRINT,
    'financial_state_unchanged':True,'authorized_at_preparation':False,'target_helper_reviewed':False}
 raw=encoded(c);outer.validate_control(raw,pins[outer.LOCAL_SOURCES[0]])
 payload=encoded({n:base64.b64encode(v).decode() for n,v in sources.items()})
 if len(payload)+len(raw)+64*1024>2*2**20:raise ValueError('Original source envelope cap exceeded.')
 dest.mkdir(mode=0o700)
 for name,data in [('CONTROL-CANDIDATE.json',raw),('SOURCE-PAYLOAD.json',payload),('known-hosts',outer.KNOWN_HOSTS.encode())]:outer.private_write(dest,name,data,lambda:None)
 result={'kind':'pgd-split-outer-host10609-unarmed-package-v1','outer_control_sha256':digest(raw),'supervised_control_sha256':digest(supervised),
  'source_files':69,'source_bytes':sum(map(len,sources.values())),'source_envelope_bytes':len(payload),'host_change_metadata':metadata,
  'authorization_created':False,'target_work_or_SSH_executed':False,'scientific_qualified':False,'training_admitted':False}
 outer.private_write(dest,'PACKAGE.json',encoded(result),lambda:None)
 return result
