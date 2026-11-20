"""Local comparison of pinned saved component returns; never regenerates a model.

Only the eight already-collected NPY returns are decoded. Historical serialized
Torch operands remain opaque. This produces a new, separately scoped analysis;
it cannot complete an old worker or qualify canonical FP64, replay or training.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat

import numpy as np

INPUTS='configs/predictor_pgd_component_returns_inputs_v1.json'
INPUTS_SHA='cfe0bade7d7f9a5815cb1326be4fb4793b5bfc6b10e1a2adebf3b5908bdc773e'
RETURNS=('output','dq','dk','dv')
ROLES=('native','forward-preserving')
SHAPE=(1,12,2048,32)
MAX_FILE=4*2**20
MAX_TOTAL=32*2**20
BOUNDS = {'output': {'rtol': 1e-4, 'atol': 2e-5},
          **{name: {'rtol': .002, 'atol': 2e-7} for name in ('dq', 'dk', 'dv')}}
META=frozenset(('target_review','metadata_review','metadata_config','boundary','manifest',
    'accepted_numerical_review','native_worker','native_ownership','forward_preserving_worker',
    'forward_preserving_ownership','candidate_failed','candidate_assessment','native_assessment'))
AGREEMENT_SOURCE='models/attention_saved_evidence.py'
BOUNDS_SOURCE='models/predictor_sdpa_numerical_receipts_v1.py'
POLICY='pgd-component-saved-return-accuracy-v1'


def digest(data):return hashlib.sha256(data).hexdigest()
def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def require(value,message):
    if not value:raise ValueError(message)

def parsed(raw):
    def pairs(items):
        result={}
        for k,v in items:
            require(k not in result,'Duplicate JSON field refused.');result[k]=v
        return result
    def invalid(value):raise ValueError('Nonfinite JSON refused: '+value)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)

def identity(s):return (s.st_dev,s.st_ino,s.st_uid,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def read_bound(path,descriptor,*,cap=MAX_FILE):
    p=Path(path)
    require(p.is_absolute() and p.resolve(strict=True)==p,'Canonical absolute local file required.')
    before=p.lstat();parent=p.parent.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and before.st_nlink==1
        and 0<before.st_size<=cap,'Bounded owned single-link ordinary file required; never open special files.')
    require(before.st_size==descriptor['bytes'],'Pinned file size differs.')
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        held=os.fstat(f.fileno());raw=f.read(cap+1);after=os.fstat(f.fileno())
    require(identity(before)==identity(held)==identity(after)==identity(p.lstat())
        and identity(parent)==identity(p.parent.stat()),'Held input or parent changed during read.')
    require(len(raw)==before.st_size and digest(raw)==descriptor['sha256'],'Pinned file bytes differ.')
    return raw


def decode_npy(raw,*,shape,payload_sha256):
    """Exact NPY v1 little-endian contiguous FP32; no pickle loader or mmap."""
    require(type(raw) is bytes and 10<len(raw)<=MAX_FILE and raw[:8]==b'\x93NUMPY\x01\x00','Exact bounded NPY v1 envelope required.')
    length=int.from_bytes(raw[8:10],'little');end=10+length
    require(0<length<=4096 and end<=len(raw) and end%16==0 and raw[end-1:end]==b'\n','Bounded complete aligned NPY header required.')
    tree=ast.parse(raw[10:end].decode('ascii').strip(),mode='eval')
    require(isinstance(tree.body,ast.Dict) and len(tree.body.keys)==3,'Exact three-field NPY header required.')
    keys=[ast.literal_eval(k) for k in tree.body.keys]
    require(len(set(keys))==3 and set(keys)=={'descr','fortran_order','shape'},'Unique NPY schema fields required.')
    header=ast.literal_eval(tree)
    require(header['descr']=='<f4' and header['fortran_order'] is False
        and type(header['shape']) is tuple and header['shape']==tuple(shape)
        and len(shape)==4 and all(type(n) is int and n>0 for n in shape)
        and all(type(n) is int for n in header['shape']),'Exact FP32 C-order four-dimensional geometry required.')
    payload=raw[end:];require(len(payload)==math.prod(shape)*4 and digest(payload)==payload_sha256,'Complete pinned logical FP32 payload required.')
    value=np.frombuffer(payload,dtype='<f4').reshape(tuple(shape))
    require(np.isfinite(value).all() and np.any(value),'Finite nonzero saved component return required.')
    require(not value.flags.writeable,'Saved bytes must stay read-only.')
    return value,{'header_bytes':end,'payload_sha256':payload_sha256,'shape':list(shape),'dtype':'<f4','finite_nonzero':True,'read_only':True}


def validate_context(c,raws):
    """Join immutable metadata already read and hash-verified by analyze()."""
    j=lambda n:parsed(raws[c['metadata'][n]])
    target,review,meta,boundary,manifest,accepted=(j(n) for n in ('target_review','metadata_review','metadata_config','boundary','manifest','accepted_numerical_review'))
    require(target['audit_result']=='PASS' and target['transport_verified'] and target['candidate_returns_publish_and_raw_publish_returned']
        and target['completed_eight_gradient_cohort_preserved_without_rerun'],'Reviewed complete local collection required.')
    require(review['result']=='PASS' and review['candidate_saved_metadata_matches_recipe'] and review['old_candidate_worker_remains_failed']
        and meta['candidate_saved_metadata_matches_recipe'],'Reviewed source-supported saved metadata policy required.')
    require(accepted['audit_result']=='PASS' and accepted['numerical_scoped_saved_passed'] and accepted['gradient_objectives_completed']==8
        and accepted['same_input_pairs_passed']==4,'Accepted prior numerical provenance required, without rerunning it.')
    boundary_sha=c['files'][c['metadata']['boundary']]['sha256'];manifest_sha=c['files'][c['metadata']['manifest']]['sha256']
    require(boundary['kind']=='pgd-split-original-boundary-v1' and boundary['capture_complete'] and not boundary['local_tiny']
        and boundary['mode']=='original' and boundary['selected']=={'owner':'predictor.0','draw':0,'family':0,'forward_row_index':24,'phase':'EOT8-before'},'Original selected native boundary required.')
    require(manifest['boundary_sha256']==boundary_sha and manifest['source_control_sha256']==boundary['control_sha256']
        and manifest['accepted_numerical_review_sha256']==c['files'][c['metadata']['accepted_numerical_review']]['sha256'],'Historical source manifest join changed.')
    native=j('native_worker');candidate=j('forward_preserving_worker');failed=j('candidate_failed')
    require(native['complete'] is True and candidate['complete'] is False and failed['complete'] is False
        and failed['error_type']=='ValueError' and failed['role']=='forward-preserving' and failed['binding']==candidate['binding'],'Historical native completion and candidate failure must stay unchanged.')
    controls=[];identities={};payloads={}
    for index,role in enumerate(ROLES):
        key=role.replace('-','_');w=j(key+'_worker');o=j(key+'_ownership');binding=w['binding']
        require(w['role']==o['role']==role and w['pid']==o['pid'] and w['pid']>0 and o['binding']==binding
            and type(binding['probe_index']) is int and binding['probe_index']==index and binding['boundary_sha256']==boundary_sha
            and binding['control_sha256']==target['supervised_control_sha256'] and binding['design_sha256']==native['binding']['design_sha256'],'Exact saved component role/PID/binding required.')
        require(o['route_complete'] and o['phase']=='closed' and len(o['calls'])==1,'One closed saved component route required.')
        call=o['calls'][0];outer=o['outer'];require(call['forward_complete'] and call['backward_complete'] and call['original_forward_grad_enabled']
            and outer['state']=='complete' and outer['root_calls']==1 and outer['root_hook_removed'] and outer['leaf_hooks_removed']
            and sorted(h['index'] for h in outer['leaf_hooks'])==[0,1,2],'Saved original root/leaf completion and removal required.')
        require(call['entry_flags']==call['exit_flags'] and call['aliases']==[0,1,2],'Saved flags or alias groups changed.')
        for name,fp in zip(('q','k','v'),call['inputs']):
            plan=manifest['layout']['operands'][name]
            require(fp['sha256']==plan['logical_sha256'] and fp['shape']==list(SHAPE) and fp['stride']==plan['stride']
                and fp['offset']==plan['storage_offset_elements'] and fp['storage_bytes']==plan['storage_bytes'],'Same canonical saved input values/layout required.')
        require(len(call['inputs'])==3 and len(call['returned_component'])==3,'All saved operands and derivative fingerprints required.')
        payloads[role]={'output':call['output']['sha256'],**{n:f['sha256'] for n,f in zip(('dq','dk','dv'),call['returned_component'])}}
        require(all(f['shape']==list(SHAPE) and f['dtype']=='torch.float32' for f in [call['output'],*call['returned_component']]),'Saved return fingerprints need original geometry/dtype.')
        for name in RETURNS:
            d=target['eight_component_return_files_preserved'][role][name];p=c['returns'][role][name]
            require(d['file_sha256']==c['files'][p]['sha256'] and d['bytes']==c['files'][p]['bytes'] and d['shape']==list(SHAPE)
                and d['dtype']=='<f4' and d['payload_sha256']==payloads[role][name],'Return file, logical bytes and route fingerprint disagree.')
        controls.append(binding['control_sha256']);identities[role]={'pid':w['pid'],'binding':binding,'route_complete':True,'historical_worker_complete':w['complete']}
    require(len(set(controls))==1 and controls[0]!=boundary['control_sha256'] and len({native['pid'],candidate['pid'],boundary['pid']})==3,'Historical and new component identities must remain separate.')
    require(native['source']['source_control_sha256']==boundary['control_sha256'] and native['source']['capsule_manifest_sha256']==manifest_sha
        and native['source']['boundary_sha256']==boundary_sha,'Saved native reconstruction must join its historical source.')
    require({n:v['sha256'] for n,v in native['reconstructed_inputs'].items()}=={n:v['logical_sha256'] for n,v in manifest['layout']['operands'].items()},'Saved native operand hashes differ from canonical source.')
    cc=j('forward_preserving_ownership')['calls'][0]
    require(cc['incoming']['sha256']==manifest['layout']['operands']['grad_output']['logical_sha256'] and cc['inner_root_calls']==1
        and cc['inner_graph_id']!=cc['outer_graph_id'] and cc['hooks_removed'],'One separate original incoming-cotangent inner route required.')
    for name in RETURNS:require(payloads['native'][name]==boundary['artifacts'][name]['source']['sha256'],'Native logical return differs from the historical boundary.')
    ca=j('candidate_assessment');na=j('native_assessment')
    require(ca['old_failed_worker_still_failed'] and ca['proof']['metadata_matches_recipe'] and na['metadata_matches_recipe']
        and ca['proof']['binding']==candidate['binding'] and na['binding']==native['binding']
        and ca['proof']['policy']==na['policy']=='pgd-pytorch28-materialized-wrapper-typed-producer-metadata-v4','Saved metadata assessments must join these exact components.')
    require(not ca['proof']['worker_completion_inferred'] and not ca['proof']['actual_CUDA_execution_admitted'],'A metadata match cannot promote the failed worker.')
    return {'identities':identities,'payload_sha256':payloads,'boundary_sha256':boundary_sha,'manifest_sha256':manifest_sha,
        'source_control_sha256':boundary['control_sha256'],'source_pid':boundary['pid'],'source_selection':boundary['selected'],
        'native_exact_historical_logical_returns':True,'candidate_metadata_recipe_matches':True,'old_failed_worker_still_failed':True}


def analyze(repository):
    require(np.__version__=='2.4.6','Pinned local NumPy runtime required; no installation/fallback.')
    root=Path(repository).resolve(strict=True);path=root/INPUTS
    cb=read_bound(path,{'bytes':path.lstat().st_size,'sha256':INPUTS_SHA},cap=128*1024);c=parsed(cb)
    require(c['kind']=='pgd-component-saved-return-inputs-v1' and c['shape']==list(SHAPE) and c['dtype']=='<f4'
        and c['bounds']==BOUNDS and set(c['metadata'])==META and set(c['returns'])==set(ROLES),'Exact saved-return analysis contract required.')
    require(all(set(c['returns'][r])==set(RETURNS) for r in ROLES),'Exactly four saved returns per role required.')
    expected=set(c['metadata'].values())|{p for r in ROLES for p in c['returns'][r].values()}|{AGREEMENT_SOURCE,BOUNDS_SOURCE}
    require(set(c['files'])==expected and len(expected)==23 and sum(d['bytes'] for d in c['files'].values())<=MAX_TOTAL,'Closed bounded source/metadata/return inventory required.')
    raws={}
    for name,d in c['files'].items():
        p=PurePosixPath(name);require(not p.is_absolute() and '..' not in p.parts and str(p)==name,'Canonical repository-relative input required.')
        raws[name]=read_bound(root/name,d)
    own=ast.parse(Path(__file__).read_bytes());gold=ast.parse(raws[AGREEMENT_SOURCE]);bounds=ast.parse(raws[BOUNDS_SOURCE])
    find=lambda tree,name:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    require(ast.dump(find(own,'agreement'),include_attributes=False)==ast.dump(find(gold,'agreement'),include_attributes=False),'Original agreement arithmetic must remain exact.')
    pick=lambda tree:next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='BOUNDS' for x in n.targets))
    require(ast.dump(pick(own),include_attributes=False)==ast.dump(pick(bounds),include_attributes=False),'Original component tolerance source must remain exact.')
    context=validate_context(c,raws);arrays={};descriptors={}
    for role in ROLES:
        arrays[role]={};descriptors[role]={}
        for name in RETURNS:
            p=c['returns'][role][name];a,desc=decode_npy(raws[p],shape=SHAPE,payload_sha256=context['payload_sha256'][role][name])
            arrays[role][name]=a;descriptors[role][name]={**desc,'path':p,**c['files'][p]}
    checks={n:agreement(arrays['forward-preserving'][n],arrays['native'][n],BOUNDS[n]) for n in RETURNS}
    forward_exact=context['payload_sha256']['native']['output']==context['payload_sha256']['forward-preserving']['output']
    passed=forward_exact and all(v['passed'] and v['failed_elements']==0 for v in checks.values())
    return {'kind':POLICY,'analysis_complete':True,'saved_pair_original_bounds_passed':passed,'checks':checks,'bounds':BOUNDS,
        'input_contract_sha256':digest(cb),'input_files':c['files'],'context':context,'arrays':descriptors,'forward_logical_bytes_exact':forward_exact,
        'comparison_context':'passing-original; native logical returns exactly match the accepted historical boundary',
        'new_saved_array_comparisons':4,'arrays_decoded':8,'numpy_version':np.__version__,
        'new_model_gradient_profiler_GPU_reference_oracle_or_paid_work':False,'canonical_FP64_accuracy_qualified':False,
        'old_failed_worker_promoted':False,'worker_completion_inferred':False,'full_EOT8_backend_qualified':False,
        'fresh_replay_performance_cost_or_training_admitted':False,'financial_caps_holds_and_billing_unchanged':True}


def agreement(actual,expected,tolerance):
    actual=np.asarray(actual,dtype=np.float64);expected=np.asarray(expected,dtype=np.float64)
    if (actual.shape!=expected.shape or actual.size==0 or not np.isfinite(actual).all() or not np.isfinite(expected).all()
            or set(tolerance)!={'rtol','atol'} or any(not math.isfinite(v) or v<0 for v in tolerance.values())
            or tolerance['atol']<=0):raise ValueError('Invalid saved-tensor comparison.')
    error=np.abs(actual-expected);allowed=tolerance['atol']+tolerance['rtol']*np.abs(expected)
    return {'passed':bool(np.all(error<=allowed)),'failed_elements':int(np.count_nonzero(error>allowed)),
            'shape':list(actual.shape),'max_absolute_error':float(error.max()),
            'max_tolerance_ratio':float((error/allowed).max()),
            'relative_l2_error':float(np.linalg.norm(error.ravel())/max(float(np.linalg.norm(expected.ravel())),1e-300)),
            **tolerance}
