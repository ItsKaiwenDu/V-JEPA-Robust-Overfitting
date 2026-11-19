"""Cold, byte-invertible metadata-parser correction; no target or science API."""
import ast,hashlib
from pathlib import Path

OLD_PROJECTION='models/predictor_sdpa_launch_evidence_v2.py'
OLD_READER='models/predictor_pgd_split_backend_v1.py'
PROJECTION='models/predictor_sdpa_launch_evidence_v3.py'
READER='models/predictor_pgd_split_backend_v2.py'
FROZEN={OLD_PROJECTION:'05f99a96ca34d65c4167cdbd37f97600b14468aa08fbfe5aa9b581b72e5fdddb',
 OLD_READER:'f994c5037b4fbf507eeeb71f3df41eac26ec5f21b052d8faa89052b08fb31e2f'}
NAMES=('Activity Buffer Request','Runtime Triggered Module Loading','Lazy Function Loading')
POLICY='fixed-SDPA-profiler-loading-overhead-signatures-v2'
READER_POLICY='pgd-native-forward-math-backward-loading-metadata-v2'


def digest(data):return hashlib.sha256(data).hexdigest()

def apply(raw,edits):
 original=raw;trace=[]
 for old,new in edits:
  if raw.count(old)!=1:raise ValueError('One exact reviewed byte edit required: '+repr(old))
  raw=raw.replace(old,new,1);trace.append({'old':old.decode(),'new':new.decode()})
 inverse=raw
 for old,new in reversed(edits):
  if inverse.count(new)!=1:raise ValueError('One exact inverse edit required.')
  inverse=inverse.replace(new,old,1)
 if inverse!=original:raise ValueError('Complete byte inverse failed.')
 return raw,trace


def build(repository):
 root=Path(repository);old={}
 for name,want in FROZEN.items():
  raw=(root/name).read_bytes()
  if digest(raw)!=want:raise ValueError('Frozen source changed: '+name)
  old[name]=raw
 projection,pe=apply(old[OLD_PROJECTION],[
  (b"POLICY='fixed-SDPA-profiler-overhead-signature-v1'",('POLICY='+repr(POLICY)).encode()),
  (b"OVERHEAD_NAME='Activity Buffer Request'",b"OVERHEAD_NAME='Activity Buffer Request'\nOVERHEAD_NAMES=frozenset((OVERHEAD_NAME,'Runtime Triggered Module Loading','Lazy Function Loading'))"),
  (b"row['name']!=OVERHEAD_NAME",b"row['name'] not in OVERHEAD_NAMES")])
 ph=digest(projection)
 reader,re=apply(old[OLD_READER],[
  (b'from models import predictor_sdpa_launch_evidence_v2 as frozen',b'from models import predictor_sdpa_launch_evidence_v3 as frozen'),
  (b"POLICY='pgd-native-forward-math-backward-metadata-v1'",('POLICY='+repr(READER_POLICY)).encode()),
  (b'PINNED={',('PINNED={\n '+repr('predictor_sdpa_launch_evidence_v3.py')+':'+repr(ph)+',').encode()),
  (b'or frozen.BACKWARD!=BACKWARD or frozen.OVERHEAD_RUNTIME!=RUNTIME):',
   ("or frozen.BACKWARD!=BACKWARD or frozen.OVERHEAD_RUNTIME!=RUNTIME\n            or frozen.OVERHEAD_NAMES!=frozenset("+repr(NAMES)+") or frozen.POLICY!="+repr(POLICY)+"):").encode())])
 result={PROJECTION:projection,READER:reader}
 def functions(raw):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 for before,after,changed in [(old[OLD_PROJECTION],projection,{'record_role'}),(old[OLD_READER],reader,{'source_gate'})]:
  a,b=functions(before),functions(after)
  if set(a)!=set(b) or any(a[n]!=b[n] for n in a.keys()-changed):raise ValueError('Unchanged schema/association/backend/IO predicate differs.')
 for name,raw in result.items():compile(raw,'<UNEXECUTED-PROFILER-LOADING-'+name+'>','exec')
 return result,{'kind':'pgd-profiler-loading-metadata-v1-derivation','frozen_source_pins':FROZEN,'edits':{PROJECTION:pe,READER:re},
  'source_pins':{n:digest(v) for n,v in result.items()},'source_bytes':{n:len(v) for n,v in result.items()},
  'overhead_names':list(NAMES),'runtime':'2.8.0+cu128','Kineto_pin':'5e7501833f1021ce6f618572d3baf657b6319658',
  'all_other_functions_AST_exact':True,'complete_byte_inverses':True,'new_target_binding':False,'new_scientific_or_target_work':False,'training_admitted':False}
