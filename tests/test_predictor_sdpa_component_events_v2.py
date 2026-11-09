"""Saved-profile and synthetic metadata tests only; never new scientific work."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from models import predictor_sdpa_component_events_v1 as old
from models import predictor_sdpa_component_events_v2 as new
from scripts import check_predictor_sdpa_association_v2 as audit

ROOT=Path(__file__).resolve().parents[1]
TRACE=ROOT/'runs/pretrained/diagnostic/predictor-sdpa-continuation-v1-20261108/linux10603-final/attended-fz_fon25/saved/results/components/0/events.jsonl'
TRACE_SHA='d03b3841518bb0a2380c43e4ed7abef5b36aba7626fe98de85322739a6faa466'
CPU=ROOT/audit.CPU


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def actual():
    assert hashlib.sha256(TRACE.read_bytes()).hexdigest()==TRACE_SHA
    return [json.loads(x) for x in TRACE.read_text().splitlines()]
def at(rows,name):return next(r for r in rows if r['name']==name)
def append(rows,row):
    row=copy.deepcopy(row);row['index']=len(rows);rows.append(row);return row


def fixture(family='efficient',*,cross=False,device='cuda'):
    """Handwritten metadata, not measured CUDA evidence."""
    rows=[];gpu=[]
    def row(name,start,end,parent=None,thread=7,seq=-1,fwd=0,scope=0,annotation=False,kernel=None,duration=2.,device_type='cpu'):
        value={'index':len(rows),'name':name,'device':device_type,'device_index':0,'thread':thread,
               'fwd_thread':fwd,'sequence_nr':seq,'scope':scope,'parent':parent,'start_us':float(start),'end_us':float(end),
               'legacy':False,'user_annotation':annotation,'linked_kernels':[]}
        if kernel and device=='cuda':
            value['linked_kernels']=[{'name':kernel,'device':0,'duration_us':duration}];gpu.append((kernel,duration))
        rows.append(value);return value['index']
    f=row(new.FORWARD,0,40,scope=7,annotation=True)
    fop=row(new.FAMILIES[family][0],1,35,f,seq=1 if family=='math' else 23)
    if family=='math':
        row('aten::bmm',2,10,fop,seq=10,kernel='sgemm_synthetic_forward',duration=3.)
        row('aten::_softmax',12,25,fop,seq=11,kernel='softmax_warp_forward<synthetic>',duration=4.)
    elif family=='efficient':
        rows[fop]['linked_kernels']=[{'name':'fmha_cutlassF_f32_synthetic','device':0,'duration_us':2.}]
        gpu.append(('fmha_cutlassF_f32_synthetic',2.))
    b=row(new.BACKWARD,50,99,scope=7,annotation=True)
    def backward(name,node_name,start,end,seq,kernel,duration):
        parent=b;thread=7
        if cross:
            thread=8;engine=row(new.ENGINE+node_name,start,end,thread=thread,seq=seq,fwd=7)
            parent=row(node_name,start+1,end-1,engine,thread=thread,seq=seq,fwd=7,scope=1)
        row(name,start+2,end-2,parent,thread=thread,kernel=kernel,duration=duration)
    if family=='math':
        backward('aten::bmm','BmmBackward0',51,72,10,'sgemm_synthetic_backward',5.)
        backward(new.FAMILIES[family][1],'SoftmaxBackward0',74,97,11,'softmax_warp_backward<synthetic>',6.)
    elif family=='efficient':backward(new.FAMILIES[family][1],'ScaledDotProductEfficientAttentionBackward0',51,97,23,'fmha_cutlassB_f32_synthetic',7.)
    else:backward(new.FAMILIES[family][1],'ScaledDotProductFlashAttentionForCpuBackward0',51,97,23,None,2.)
    if device=='cuda':
        for i,(name,duration) in enumerate(gpu):row(name,100+i*10,100+i*10+duration,device_type='cuda')
    return rows


def test_real_saved_trace_has_explicit_cross_thread_proof_without_relabeling():
    rows=actual();before=canonical(rows);legacy=copy.deepcopy(old.FAMILIES)
    with pytest.raises(ValueError,match='missing actual component backward backend'):old.verify(rows,arm='original',device='cuda')
    proof=new.verify(rows,arm='original',device='cuda')
    assert proof==new.verify(rows,arm='original',device='cuda')
    assert canonical(rows)==before and old.FAMILIES==legacy and hashlib.sha256(TRACE.read_bytes()).hexdigest()==TRACE_SHA
    assert proof['observed_family']=='efficient' and proof['association_mode']=='CPU-parent-and-autograd-identity'
    assert len(proof['cross_thread_anchors'])==1 and len(proof['kernel_witnesses'])==12
    anchor=proof['cross_thread_anchors'][0]
    assert anchor['forward_thread']==1 and anchor['sequence_nr']==1 and anchor['backward_thread']==2
    assert anchor['forward_chain']==[1,2] and anchor['engine_index']==30 and anchor['node_index']==31
    assert not proof['worker_completion_inferred'] and not proof['timings_usable_for_speed_or_cost']
    assert not proof['unique_launch_correlation_IDs_available'] and not proof['full_model_candidate_adopted']
    with pytest.raises(ValueError):old.verify(rows,arm='original',device='cuda')


@pytest.mark.parametrize('index',range(4))
def test_actual_cached_CPU_profiles_keep_existing_parent_evidence(index):
    rows=audit.read_trace(CPU/str(index)/'events.jsonl');arm='math' if index%2 else 'original'
    prior=old.verify(rows,arm=arm,device='cpu');current=new.verify(rows,arm=arm,device='cpu')
    assert {k:v for k,v in current.items() if k in prior and k!='kind'}=={k:v for k,v in prior.items() if k!='kind'}
    assert current['association_mode']=='CPU-parent' and not current['kernel_witnesses']
    assert not current['CUDA_forward_and_backward_qualified']


@pytest.mark.parametrize('family',('efficient','math'))
@pytest.mark.parametrize('cross',(False,True))
def test_synthetic_CUDA_cases_require_phase_operator_and_device_evidence(family,cross):
    rows=fixture(family,cross=cross);before=canonical(rows)
    proof=new.verify(rows,arm='math' if family=='math' else 'original',device='cuda')
    assert proof['observed_family']==family and proof['CUDA_forward_and_backward_qualified']
    assert bool(proof['cross_thread_anchors']) is cross and proof['phase_correlated_CUDA_kernel_names'][new.BACKWARD]
    assert canonical(rows)==before


def test_cross_thread_cached_math_uses_inner_producer_keys_not_the_outer_math_key():
    rows=audit.read_trace(CPU/'1/events.jsonl');b=at(rows,new.BACKWARD)['index'];source=canonical(rows)
    def ancestors(i):
        values=[]
        while i is not None:values.append(i);i=rows[i]['parent']
        return values
    engines=[r['index'] for r in rows if r['name'].startswith(new.ENGINE) and r['parent']==b]
    descendants={i:[r['index'] for r in rows if i in ancestors(r['index'])] for i in engines}
    for i,indices in descendants.items():
        rows[i]['parent']=None
        for j in indices:rows[j]['thread']=2
    proof=new.verify(rows,arm='math',device='cpu')
    assert len(proof['cross_thread_anchors'])==3 and {a['sequence_nr'] for a in proof['cross_thread_anchors']}=={23,25,30}
    assert at(rows,new.FAMILIES['math'][0])['sequence_nr']==16
    assert canonical(audit.read_trace(CPU/'1/events.jsonl'))==source
    assert not proof['CUDA_forward_and_backward_qualified']


@pytest.mark.parametrize('fault',('node-sequence','node-fwd-thread','engine-sequence','engine-fwd-thread','engine-name',
 'engine-thread','engine-process','engine-annotation','node-annotation','node-scope','node-name','missing-node-parent',
 'missing-engine-root','same-thread-detached','outside-backward-time','repeated-engine','repeated-node',
 'repeated-forward','wrong-producer','extra-forward','foreign-backward','root-before-phase','mismatched-label-thread'))
def test_real_cross_thread_identity_ambiguity_or_foreign_subtrees_refuse(fault):
    rows=actual();node=at(rows,'ScaledDotProductEfficientAttentionBackward0');engine=at(rows,new.ENGINE+node['name'])
    fop=at(rows,new.FAMILIES['efficient'][0]);bop=at(rows,new.FAMILIES['efficient'][1]);label=at(rows,new.BACKWARD)
    if fault=='node-sequence':node['sequence_nr']+=1
    elif fault=='node-fwd-thread':node['fwd_thread']+=3
    elif fault=='engine-sequence':engine['sequence_nr']+=1
    elif fault=='engine-fwd-thread':engine['fwd_thread']+=3
    elif fault=='engine-name':engine['name']=new.ENGINE+'UnknownBackward0'
    elif fault=='engine-thread':engine['thread']+=1
    elif fault=='engine-process':engine['device_index']+=1
    elif fault=='engine-annotation':engine['user_annotation']=True
    elif fault=='node-annotation':node['user_annotation']=True
    elif fault=='node-scope':node['scope']=0
    elif fault=='node-name':node['name']='UnrelatedBackward0'
    elif fault=='missing-node-parent':node['parent']=None
    elif fault=='missing-engine-root':engine['parent']=label['index']
    elif fault=='same-thread-detached':
        for row in rows:
            if row['device']=='cpu' and row['thread']==2:row['thread']=1
    elif fault=='outside-backward-time':label['end_us']=engine['end_us']-1
    elif fault=='repeated-engine':append(rows,engine)
    elif fault=='repeated-node':append(rows,node)
    elif fault=='repeated-forward':
        value=append(rows,rows[1]);value['name']='aten::unrelated';value['parent']=0
    elif fault=='wrong-producer':fop['sequence_nr']+=7
    elif fault=='extra-forward':append(rows,fop)
    elif fault=='foreign-backward':
        value=append(rows,bop);value['parent']=None;value['thread']=3
    elif fault=='root-before-phase':label['start_us']=engine['start_us']+1
    else:label['thread']+=7
    with pytest.raises(ValueError):new.verify(rows,arm='original',device='cuda')


@pytest.mark.parametrize('fault',('no-link','zero-link','wrong-link-device','wrong-GPU-device','annotation-only',
 'name-only','foreign-name','foreign-duration','ambiguous-GPU','unrelated-host-claim','foreign-unlinked-only','unsupported-backend','candidate-efficient'))
def test_device_kernel_names_or_unrelated_activity_cannot_repair_missing_proof(fault):
    rows=actual();gpu=rows[86];host=rows[43]
    if fault=='no-link':
        for r in rows:
            if r['device']=='cpu' and r['thread']==2:r['linked_kernels']=[]
    elif fault=='zero-link':
        for r in rows:
            if r['device']=='cpu' and r['thread']==2:
                for k in r['linked_kernels']:k['duration_us']=0.
    elif fault=='wrong-link-device':host['linked_kernels'][0]['device']=1
    elif fault=='wrong-GPU-device':gpu['device_index']=1
    elif fault=='annotation-only':gpu['user_annotation']=True
    elif fault=='name-only':gpu['end_us']=gpu['start_us']
    elif fault=='foreign-name':gpu['name']='fmha_cutlassB_f32_foreign'
    elif fault=='foreign-duration':gpu['end_us']+=100.
    elif fault=='ambiguous-GPU':append(rows,gpu)
    elif fault=='unrelated-host-claim':
        value=append(rows,host);value['parent']=32;value['start_us']=host['start_us'];value['end_us']=host['end_us']
    elif fault=='foreign-unlinked-only':
        value=append(rows,host);value['parent']=31
        for r in rows[:-1]:r['linked_kernels']=[k for k in r['linked_kernels'] if not k['name'].startswith('fmha_cutlassB_f32_')]
    elif fault=='unsupported-backend':at(rows,new.FAMILIES['efficient'][0])['name']='aten::_scaled_dot_product_cudnn_attention'
    with pytest.raises(ValueError):new.verify(rows,arm='math' if fault=='candidate-efficient' else 'original',device='cuda')


def test_cuda_intervals_need_not_fit_inside_the_CPU_phase():
    rows=actual();gpu=rows[86];b=at(rows,new.BACKWARD)
    assert gpu['end_us']>b['end_us']
    assert new.verify(rows,arm='original',device='cuda')['cross_thread_anchors']


@pytest.mark.parametrize('fault',('parent-cycle','parent-range','parent-type','parent-containment','wrong-index','bool-index',
 'unsupported-device','legacy','parent-object','parent-missing','kernel-overflow','NaN-time','infinite-time','negative-extent','kernel-type','link-list-type','event-type','overlap-phases','duplicate-label'))
def test_malformed_or_ambiguous_event_graphs_refuse(fault):
    rows=actual()
    if fault=='parent-cycle':rows[1]['parent']=1
    elif fault=='parent-range':rows[1]['parent']=len(rows)+10
    elif fault=='parent-type':rows[1]['parent']=False
    elif fault=='parent-containment':rows[2]['start_us']=0.
    elif fault=='wrong-index':rows[2]['index']=99
    elif fault=='bool-index':rows[1]['index']=True
    elif fault=='unsupported-device':rows[86]['device']='mps'
    elif fault=='legacy':rows[86]['legacy']=True
    elif fault=='parent-object':rows[1]['parent']=43;rows[43]=None
    elif fault=='parent-missing':del rows[43]['parent']
    elif fault=='kernel-overflow':rows[43]['linked_kernels'][0]['duration_us']=1e308
    elif fault=='NaN-time':rows[86]['start_us']=float('nan')
    elif fault=='infinite-time':rows[86]['end_us']=float('inf')
    elif fault=='negative-extent':rows[86]['end_us']=0.
    elif fault=='kernel-type':rows[43]['linked_kernels']=[None]
    elif fault=='link-list-type':rows[43]['linked_kernels']={}
    elif fault=='event-type':rows[43]=None
    elif fault=='overlap-phases':at(rows,new.BACKWARD)['start_us']=0.
    else:append(rows,at(rows,new.BACKWARD))
    with pytest.raises(ValueError):new.verify(rows,arm='original',device='cuda')


@pytest.mark.parametrize('rows',(None,[],[{}]*(old.MAX_EVENTS+1)))
def test_invalid_trace_extents_refuse_before_processing(rows):
    with pytest.raises(ValueError):new.verify(rows,arm='original',device='cuda')


@pytest.mark.parametrize('domain',('wrong-domain','empty-backend'))
def test_device_domain_and_no_fallback_rules_remain(domain):
    rows=fixture('cpu-flash',device='cpu')
    if domain=='empty-backend':rows[1]['name']='aten::unknown_attention'
    with pytest.raises(ValueError):new.verify(rows,arm='original',device='cuda')


def test_pure_stdlib_audit_runs_with_science_process_and_network_blocked(tmp_path):
    output=tmp_path/'assessment'
    program='''import importlib.abc,os,socket,subprocess,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','numpy','timm'):raise AssertionError('No scientific imports in metadata audit')
sys.meta_path.insert(0,Deny())
def forbidden(*a,**k):raise AssertionError('No model, subprocess, network or SSH work')
subprocess.Popen=subprocess.run=os.system=os.fork=socket.socket=socket.create_connection=forbidden
from scripts.check_predictor_sdpa_association_v2 import audit
audit(Path(sys.argv[2]))'''
    result=subprocess.run([sys.executable,'-I','-B','-c',program,str(ROOT),str(output)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
    (tmp_path/'audit.log').write_bytes(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr.decode()[-5000:]
    state=json.loads((output/'assessment.json').read_text())
    assert state['assessment_only'] and state['revised_trace_association_passed'] and not state['original_worker_complete']
    assert state['preserved_files_reverified']==41489 and len(state['cached_CPU_traces'])==4
    assert not state['new_model_autograd_SDPA_profiler_CUDA_or_reference_work']
    assert not state['worker_or_full_model_admitted'] and not state['new_target_authorization_or_attempt']
    with pytest.raises(FileExistsError):audit.audit(output)
