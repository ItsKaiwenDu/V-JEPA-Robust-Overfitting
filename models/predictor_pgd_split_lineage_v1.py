"""Strict metadata join for NEW split inputs; never a numerical/CUDA admission.

Callers must first run the pinned numerical, objective, boundary and proof readers.
These checks cannot authenticate a caller-supplied dict or replace those readers.
"""
import hashlib
import json

DESIGN_SHA='c5daae30cbe4043cf4d5265de2a2a7420d015ab1e244cc3d74fcc93b5f0186af'
BOUNDARY_KIND='pgd-split-original-boundary-v1'
LINEAGE_KIND='pgd-split-original-objective-zero-source-v1'
OBSERVER_SHA='5e6e00593f3befb1979b37f0fec155bb35322c9e4c196a7abe968d2f0ff68047'
NUMERICAL='pretrained-pgd-split-numerical-worker-v1'
OBJECTIVE='pretrained-pgd-split-numerical-objective-v1'
NAMES=('q','k','v','grad_output')
SHAPE=[1,12,2048,32]


def encoded(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(v):return type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v)
def integer(v):return type(v) is int


def operands(cap):
    """Bounded allocation/layout plan only; no tensor/storage is constructed."""
    result={};groups={}
    for name in NAMES:
        d=cap['artifacts'][name];s=d['source'];g=cap['source_storage'][name]
        if (d.get('file')!=name+'.pt' or not integer(d.get('bytes')) or not 0<d['bytes']<=4*2**20
                or not sha(d.get('sha256')) or d.get('saved_before_numerical_gate') is not True
                or d.get('stored_device')!='cpu' or d.get('logical_copy_is_contiguous') is not True
                or d.get('native_storage_alias_preserved_in_file') is not False
                or s.get('shape')!=SHAPE or any(not integer(n) for n in s['shape'])
                or s.get('dtype')!='torch.float32' or s.get('device')!='cuda:0' or not sha(s.get('sha256'))
                or type(s.get('stride')) is not list or len(s['stride'])!=4 or any(not integer(n) or n<=0 for n in s['stride'])
                or not integer(g.get('alias_group')) or not 0<=g['alias_group']<8
                or not integer(g.get('storage_bytes')) or not 0<g['storage_bytes']<=16*2**20 or g['storage_bytes']%4
                or not integer(g.get('storage_offset_elements')) or g['storage_offset_elements']<0
                or not integer(g.get('version')) or g['version']<0 or g.get('source_logical_hash')!=s['sha256']
                or g.get('requires_grad') is not (name!='grad_output')):
            raise ValueError('Exact bounded canonical source operand metadata required: '+name)
        span=4*(1+sum((a-1)*b for a,b in zip(SHAPE,s['stride'])))
        if span!=g.get('positive_stride_span_bytes') or span+4*g['storage_offset_elements']>g['storage_bytes']:
            raise ValueError('Canonical positive-stride source extent changed.')
        group=g['alias_group']
        if group in groups and groups[group]!=g['storage_bytes']:raise ValueError('Shared backing extent changed.')
        groups[group]=g['storage_bytes']
        result[name]={'shape':list(SHAPE),'stride':list(s['stride']),'logical_sha256':s['sha256'],
            'source_version':g['version'],'storage_offset_elements':g['storage_offset_elements'],
            'storage_bytes':g['storage_bytes'],'alias_group':group,'requires_grad':name!='grad_output'}
    return {'operands':result,'backing_groups':groups,'maximum_backing_bytes':sum(groups.values()),
        'unknown_padding_filled_zero':True,'native_physical_alignment_identity_claimed':False,
        'reconstructed_version_is_separate_from_source_version':True,'scientific_qualified':False}


def join(worker,original,cap,proof,*,control_sha256,boundary_sha256,pixel_sha256):
    """Cross-check verified sources. Output remains a metadata-only descriptor."""
    if not all(sha(v) for v in (control_sha256,boundary_sha256,pixel_sha256)):raise ValueError('Bound source digests required.')
    if (worker.get('kind')!=NUMERICAL or worker.get('control_sha256')!=control_sha256
            or worker.get('execution_host')!='linux10603' or worker.get('complete') is not True or worker.get('passed') is not True
            or worker.get('gradient_objectives_started')!=8 or worker.get('candidate_points')!=4
            or any(worker.get(k) is not True for k in ('model_unchanged','global_rng_unchanged','all_methods_hooks_flags_restored'))
            or worker.get('research_admitted') is not False or worker.get('performance_qualified') is not False
            or not integer(worker.get('pid')) or worker['pid']<=0 or not sha(worker.get('baseline_sha256'))):
        raise ValueError('Verified new eight-objective numerical worker required.')
    if (original.get('kind')!=OBJECTIVE or original.get('index')!=0 or type(original['index']) is not int
            or type(original.get('step')) is not int or original.get('step')!=0 or original.get('mode')!='original' or original.get('local_tiny') is not False
            or original.get('execution_host')!='linux10603' or original.get('control_sha256')!=control_sha256
            or original.get('baseline_sha256')!=worker['baseline_sha256'] or type(original.get('pid')) is not int or original.get('pid')!=worker['pid']
            or any(original.get(k) is not True for k in ('complete','passed','model_unchanged','adapter_restored','parameter_gradients_absent','input_dot_grad_absent'))
            or original.get('research_admitted') is not False):
        raise ValueError('Same new original objective zero and restored model required.')
    if (cap.get('kind')!=BOUNDARY_KIND or cap.get('control_sha256')!=control_sha256 or cap.get('local_tiny') is not False
            or cap.get('execution_host')!='linux10603' or type(cap.get('pid')) is not int or cap.get('pid')!=worker['pid'] or cap.get('mode')!='original'
            or any(cap.get(k) is not True for k in ('capture_complete','observer_restored','alias_restored','tensor_versions_unchanged'))
            or any(cap.get(k) is not False for k in ('native_forward_modified','native_cotangent_replaced','native_full_model_qualified','CUDA_kernel_qualified','research_admitted','performance_qualified'))):
        raise ValueError('Complete new unchanged observational boundary required.')
    lineage=cap.get('split_lineage')
    keys={'kind','source_objective_index','source_value_id','source_value_version','source_input_signature','frozen_observer_sha256'}
    if (type(lineage) is not dict or set(lineage)!=keys or lineage['kind']!=LINEAGE_KIND
            or type(lineage['source_objective_index']) is not int or lineage['source_objective_index']!=0
            or not integer(lineage['source_value_id']) or lineage['source_value_id']<=0
            or not integer(lineage['source_value_version']) or lineage['source_value_version']<0
            or lineage['frozen_observer_sha256']!=OBSERVER_SHA):
        raise ValueError('New source-bound original objective provenance required; no historical capture.')
    sig=lineage['source_input_signature']
    if (type(sig) is not dict or sig.get('shape')!=[1,3,16,256,256] or sig.get('dtype')!='torch.float32'
            or sig.get('sha256')!=pixel_sha256):raise ValueError('Actual perturbed original pixel bytes changed.')
    if (cap.get('call_rows')!=proof.get('calls') or cap.get('backward_scopes')!=proof.get('scopes')
            or cap.get('ownership_refusals')!=[] or proof.get('refusals')!=[] or len(proof.get('scopes',[]))!=1):
        raise ValueError('Boundary and post-restoration original native proofs differ.')
    scope=proof['scopes'][0];rows=proof['calls']
    selected={'owner':'predictor.0','draw':0,'family':0,'forward_row_index':24,'phase':'EOT8-before'}
    if (len(rows)!=1152 or cap.get('selected')!=selected or cap['selected_call'].get('actual_call')!=rows[24]
            or scope.get('input_id')!=lineage['source_value_id'] or scope.get('pid')!=worker['pid']
            or scope.get('state')!='complete' or scope.get('hook_removed') is not True):
        raise ValueError('Exact selected call/full-input identity and completed native root required.')
    layout=operands(cap)
    return {'kind':'pgd-split-input-lineage-metadata-v1','control_sha256':control_sha256,
        'design_sha256':DESIGN_SHA,'boundary_sha256':boundary_sha256,'source_pid':worker['pid'],
        'baseline_sha256':worker['baseline_sha256'],'source_objective_index':0,'pixel_sha256':pixel_sha256,
        'layout':layout,'lineage_metadata_consistent':True,'canonical_lineage_authenticated':False,
        'actual_CUDA_execution_admitted':False,'scientific_qualified':False}
