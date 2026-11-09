"""Offline v2 association proof; original records and v1 verifier stay untouched.

Match cross-thread first-order autograd nodes by forward thread AND sequence,
recorded engine/node parents and producer ancestry. Kernel names alone never
qualify a phase. Device witnesses may outlive their CPU labels asynchronously.
No model, profiler, CUDA, process or network execution is performed here.
"""
import math
from models import predictor_sdpa_component_events_v1 as frozen
FORWARD=frozen.FORWARD
BACKWARD=frozen.BACKWARD
MAX_EVENTS=frozen.MAX_EVENTS
FAMILIES=dict(frozen.FAMILIES)

# Only recorded first-order autograd producers needed by the supported SDPA
# families. Unknown/missing producers refuse; no name-only or time-only bridge.
PRODUCERS={
 'efficient':{'ScaledDotProductEfficientAttentionBackward0':{'aten::_scaled_dot_product_efficient_attention'}},
 'cpu-flash':{'ScaledDotProductFlashAttentionForCpuBackward0':{'aten::_scaled_dot_product_flash_attention_for_cpu'}},
 'math':{'BmmBackward0':{'aten::bmm'},'MmBackward0':{'aten::mm'},
         'SoftmaxBackward0':{'aten::_softmax','aten::softmax'},
         'SafeSoftmaxBackward0':{'aten::_safe_softmax'}}}
ENGINE='autograd::engine::evaluate_function: '


def associated_regions(rows,ancestry,labels):
    f,b=(rows[labels[n]] for n in (FORWARD,BACKWARD))
    if (f['thread']!=b['thread'] or f['device_index']!=b['device_index'] or f['thread']<=0
            or not f['start_us']<f['end_us']<=b['start_us']<b['end_us']):
        raise ValueError('Distinct ordered phases on one calling thread/process required.')
    regions={name:{i for i,r in enumerate(rows) if r['device']=='cpu' and labels[name] in ancestry[i]} for name in labels}
    forward=[i for i in regions[FORWARD] if rows[i]['scope']==0 and not rows[i]['user_annotation']
             and rows[i]['name'] in {x[0] for x in FAMILIES.values()}]
    if len(forward)!=1:raise ValueError('Exactly one actual forward SDPA invocation required.')
    selected=forward[0];family=next(k for k,v in FAMILIES.items() if v[0]==rows[selected]['name'])
    anchors=[];used_keys=set();base_backward=set(regions[BACKWARD])
    for i,node in enumerate(rows):
        if (node['device']!='cpu' or node['scope']!=1 or node['user_annotation']
                or i in base_backward or node['name'] not in PRODUCERS[family]):continue
        key=(node['fwd_thread'],node['sequence_nr'])
        if key[0]!=f['thread'] or key[1]<0 or node['thread']==b['thread']:
            raise ValueError('Cross-thread backward needs its exact recorded forward identity.')
        candidates=[j for j,r in enumerate(rows) if r['device']=='cpu' and r['scope']==1 and not r['user_annotation']
                    and (r['fwd_thread'],r['sequence_nr'])==key]
        if len(candidates)!=1 or key in used_keys:raise ValueError('Ambiguous autograd node for forward thread/sequence.')
        engine_index=node['parent']
        if engine_index is None:raise ValueError('Missing recorded autograd engine parent.')
        engine=rows[engine_index]
        engines=[j for j,r in enumerate(rows) if r['device']=='cpu' and r['scope']==0 and not r['user_annotation']
                 and r['name']==ENGINE+node['name'] and (r['fwd_thread'],r['sequence_nr'])==key]
        if engines!=[engine_index]:raise ValueError('Ambiguous autograd engine for forward identity.')
        if (engine['device']!='cpu' or engine['scope']!=0 or engine['user_annotation']
                or engine['name']!=ENGINE+node['name'] or engine['parent'] is not None
                or (engine['thread'],engine['fwd_thread'],engine['sequence_nr'])!=(node['thread'],*key)
                or engine['device_index']!=f['device_index'] or node['device_index']!=f['device_index']
                or not b['start_us']<=engine['start_us']<engine['end_us']<=b['end_us']):
            raise ValueError('Exact engine/node identity and backward host interval required.')
        matches=[j for j in regions[FORWARD] if rows[j]['scope']==0 and not rows[j]['user_annotation']
                 and (rows[j]['thread'],rows[j]['sequence_nr'])==key]
        # Dispatch wrappers may share a key, but must form one ancestor chain.
        deepest=[j for j in matches if all(k in ancestry[j] for k in matches)]
        if len(deepest)!=1:raise ValueError('Missing/ambiguous forward producer chain.')
        producer=deepest[0]
        if selected not in ancestry[producer] or rows[producer]['name'] not in PRODUCERS[family][node['name']]:
            raise ValueError('Autograd producer does not belong to selected SDPA.')
        outside=[j for j,r in enumerate(rows) if r['device']=='cpu' and r['scope']==0 and not r['user_annotation']
                 and (r['thread'],r['sequence_nr'])==key and r['end_us']<=b['start_us'] and j not in matches]
        if outside:raise ValueError('Forward identity is repeated outside the selected invocation.')
        descendants={j for j,r in enumerate(rows) if r['device']=='cpu' and i in ancestry[j]}
        regions[BACKWARD].update(descendants);used_keys.add(key)
        anchors.append({'forward_thread':key[0],'sequence_nr':key[1],'forward_chain':sorted(matches),
            'producer_index':producer,'producer_name':rows[producer]['name'],'engine_index':engine_index,
            'node_index':i,'node_name':node['name'],'backward_thread':node['thread'],
            'added_CPU_indices':sorted(descendants),'host_interval_checked':True})
    fwd_names={x[0] for x in FAMILIES.values()};bwd_names={x[1] for x in FAMILIES.values()}
    for i,r in enumerate(rows):
        if r['device']!='cpu' or r['scope']!=0 or r['user_annotation']:continue
        if r['name'] in fwd_names and i not in regions[FORWARD] or r['name'] in bwd_names and i not in regions[BACKWARD]:
            raise ValueError('Unassociated or extra SDPA operator outside qualified regions.')
        if family=='math' and r['name'] in ('aten::mm','aten::bmm') and i not in regions[FORWARD]|regions[BACKWARD]:
            raise ValueError('Unassociated math matrix operator.')
        if '_cudnn_attention' in r['name'] or '_flash_attention' in r['name'] and '_for_cpu' not in r['name']:
            raise ValueError('Unrecognized backend; no fallback.')
    return {name:sorted(value) for name,value in regions.items()},anchors


def duration_ns(value):
    if not math.isfinite(value) or value<=0 or not math.isfinite(value*1000):
        raise ValueError('Finite positive recorded kernel duration required.')
    value=round(value*1000)
    if value<=0:raise ValueError('Positive nanosecond kernel extent required.')
    return value


def verify(rows,*,arm,device):
    if arm not in ('original','math') or device not in ('cpu','cuda') or not isinstance(rows,list) or not 0<len(rows)<=MAX_EVENTS:
        raise ValueError('Exact component trace/domain required.')
    def chain(i):
        seen=set()
        while i is not None:
            if type(i) is not int or not 0<=i<len(rows) or i in seen:raise ValueError('Invalid component parent graph.')
            if not isinstance(rows[i],dict) or 'parent' not in rows[i]:raise ValueError('Invalid component parent record.')
            seen.add(i);yield i;i=rows[i]['parent']
    ancestry={};labels={}
    for i,r in enumerate(rows):
        if not isinstance(r,dict):raise ValueError('Typed event object required.')
        if (type(r.get('index')) is not int or r.get('index')!=i or r.get('device') not in ('cpu','cuda') or not isinstance(r.get('name'),str)
                or any(type(r.get(k)) is not int for k in ('device_index','thread','scope','fwd_thread','sequence_nr')) or r.get('legacy') is not False
                or type(r.get('user_annotation')) is not bool or any(type(r.get(k)) not in (int,float) or not math.isfinite(r[k]) for k in ('start_us','end_us'))
                or r['end_us']<r['start_us']):raise ValueError('Invalid device-typed component event.')
        if not isinstance(r.get('linked_kernels'),list):raise ValueError('Actual linked-kernel list required.')
        c=list(chain(i));ancestry[i]=c
        if r['parent'] is not None:
            parent=rows[r['parent']]
            if r['device']!='cpu' or parent['device']!='cpu' or parent['thread']!=r['thread'] or parent['start_us']>r['start_us'] or parent['end_us']<r['end_us']:
                raise ValueError('Broken CPU event containment.')
        if r['device']=='cpu' and r['name'] in (FORWARD,BACKWARD):
            if not r['user_annotation'] or r['name'] in labels:raise ValueError('Unique actual component phase labels required.')
            labels[r['name']]=i
        for k in r.get('linked_kernels',[]):
            if not isinstance(k,dict):raise ValueError('Typed linked kernel required.')
            if not isinstance(k.get('name'),str) or type(k.get('duration_us')) not in (int,float) or not math.isfinite(k['duration_us']) or k['duration_us']<0 or type(k.get('device')) is not int:
                raise ValueError('Malformed actual linked kernel.')
    if set(labels)!={FORWARD,BACKWARD}:raise ValueError('Missing distinct actual forward/backward phases.')
    regions,anchors=associated_regions(rows,ancestry,labels)
    if set(regions[FORWARD])&set(regions[BACKWARD]):raise ValueError('Ambiguous overlapping component phases.')
    def operators(indices):return {rows[i]['name'] for i in indices if rows[i]['scope']==0 and not rows[i]['user_annotation']}
    fops=operators(regions[FORWARD]);bops=operators(regions[BACKWARD]);allops=fops|bops
    detected=[family for family,(f,b) in FAMILIES.items() if f in fops]
    if len(detected)!=1:raise ValueError('Exactly one observed component forward backend required.')
    family=detected[0];fop,bop=FAMILIES[family]
    if bop not in bops or any('_cudnn_attention' in n or '_flash_attention' in n and '_for_cpu' not in n for n in allops):
        raise ValueError('Unrecognized/mixed or missing actual component backward backend; no fallback.')
    if any(b in bops for other,(f,b) in FAMILIES.items() if other!=family):raise ValueError('Mixed component backward families observed.')
    if arm=='math' and family!='math':raise ValueError('Candidate math selection was not observed.')
    if device=='cpu' and family=='efficient' or device=='cuda' and family=='cpu-flash':raise ValueError('Observed backend belongs to another device domain.')
    if family=='math' and 'aten::bmm' not in bops and 'aten::mm' not in bops:raise ValueError('Math VJP matrix operations not observed.')
    kernel_witnesses={}
    kernels={r['name'] for r in rows if r['device']=='cuda' and r['end_us']>r['start_us'] and not r['user_annotation']}
    if device=='cpu':
        if kernels or any(r['device']=='cuda' or r.get('linked_kernels') for r in rows):raise ValueError('CPU proof cannot carry CUDA activity.')
        phase_kernels={FORWARD:[],BACKWARD:[]}
    else:
        if not kernels:raise ValueError('Missing actual non-annotation CUDA activity.')
        if any(r['device']=='cuda' and r['device_index']!=0 for r in rows):raise ValueError('Unexpected CUDA device index.')
        def correlated(op,region):
            nodes=[i for i in regions[region] if rows[i]['name']==op and not rows[i]['user_annotation']]
            found=set()
            for i in regions[region]:
                if not any(n in ancestry[i] for n in nodes):continue
                for link_index,k in enumerate(rows[i]['linked_kernels']):
                    if k['device']!=0:raise ValueError('Linked kernel belongs to another CUDA device.')
                    if k['duration_us']<=0:continue
                    extent_ns=duration_ns(k['duration_us'])
                    matches=[j for j,r in enumerate(rows) if r['device']=='cuda' and not r['user_annotation']
                        and r['device_index']==k['device'] and r['name']==k['name'] and r['end_us']>r['start_us']
                        and duration_ns(r['end_us']-r['start_us'])==extent_ns]
                    if len(matches)!=1:
                        raise ValueError('Missing/ambiguous positive CUDA kernel witness.')
                    gpu=matches[0];previous=kernel_witnesses.get(gpu)
                    if previous and i not in ancestry[previous['host_event_index']] and previous['host_event_index'] not in ancestry[i]:
                        raise ValueError('One kernel witness claimed by unrelated host events.')
                    if not previous or previous['host_event_index'] in ancestry[i]:
                        kernel_witnesses[gpu]={'host_event_index':i,'host_link_index':link_index,'CUDA_event_index':gpu,
                            'name':k['name'],'device':0,'duration_ns':extent_ns}
                    found.add(k['name'])
            if not found:raise ValueError('Required actual operator has no correlated positive CUDA kernel.')
            return sorted(found)
        phase_kernels={FORWARD:correlated(fop,FORWARD),BACKWARD:correlated(bop,BACKWARD)}
        if family=='efficient':
            if (not any(n.startswith('fmha_cutlassF_f32_') or 'AttentionKernel<' in n for n in phase_kernels[FORWARD])
                    or not any(n.startswith('fmha_cutlassB_f32_') or 'AttentionBackwardKernel<' in n for n in phase_kernels[BACKWARD])):
                raise ValueError('Associated efficient forward/backward kernel family was not recognized.')
        else:
            matrix={}
            for region,ops in ((FORWARD,fops),(BACKWARD,bops)):
                matmul='aten::bmm' if 'aten::bmm' in ops else 'aten::mm'
                matrix[region]=correlated(matmul,region)
                if not any('gemm' in n.lower() for n in matrix[region]):raise ValueError('Associated math matrix CUDA kernel was not recognized.')
                phase_kernels[region]=sorted(set(phase_kernels[region])|set(matrix[region]))
            softmax=correlated('aten::_softmax',FORWARD)
            if (not any('softmax' in n.lower() for n in softmax)
                    or not any('SoftMaxBackward' in n or 'softmax_warp_backward<' in n for n in phase_kernels[BACKWARD])):
                raise ValueError('Associated math forward/backward softmax CUDA kernels were not recognized.')
            phase_kernels[FORWARD]=sorted(set(phase_kernels[FORWARD])|set(softmax))
    return {'kind':'one-fixed-SDPA-associated-backend-v2','observed_family':family,'selection':arm,'device':device,
        'CPU_only':device=='cpu','CUDA_forward_and_backward_qualified':device=='cuda',
        'phase_correlated_CUDA_kernel_names':phase_kernels,'timings_usable_for_speed_or_cost':False,
        'association_mode':'CPU-parent-and-autograd-identity' if anchors else 'CPU-parent',
        'cross_thread_anchors':anchors,'kernel_witnesses':[kernel_witnesses[k] for k in sorted(kernel_witnesses)],
        'kernel_match_fields':['name','device','rounded_nanosecond_duration'],
        'unique_launch_correlation_IDs_available':False,'worker_completion_inferred':False,'full_model_candidate_adopted':False,
        'default_family_was_assumed':False,'unsupported_backend_action':'refuse and preserve; never force a fallback'}
