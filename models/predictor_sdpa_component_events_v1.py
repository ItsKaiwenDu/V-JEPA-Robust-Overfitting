"""Single-call device/phase-associated backend evidence; no flags-only proof."""
import math

FORWARD='fixed-SDPA/forward'
BACKWARD='fixed-SDPA/backward'
MAX_EVENTS=20000
FAMILIES={'math':('aten::_scaled_dot_product_attention_math','aten::_softmax_backward_data'),
          'efficient':('aten::_scaled_dot_product_efficient_attention','aten::_scaled_dot_product_efficient_attention_backward'),
          'cpu-flash':('aten::_scaled_dot_product_flash_attention_for_cpu','aten::_scaled_dot_product_flash_attention_for_cpu_backward')}

def capture(profile):
    import torch
    values=profile.events()
    if not 0<len(values)<=MAX_EVENTS:raise ValueError('Bounded nonempty component profile required; no truncation.')
    ids={id(e):i for i,e in enumerate(values)};rows=[]
    for i,e in enumerate(values):
        if any(not hasattr(e,k) for k in ('sequence_nr','fwd_thread','scope','is_user_annotation')):
            raise ValueError('Component profile lacks actual association fields.')
        rows.append({'index':i,'name':e.name,'device':'cpu' if e.device_type==torch.autograd.DeviceType.CPU else
                'cuda' if e.device_type==torch.autograd.DeviceType.CUDA else 'unsupported',
            'device_index':e.device_index,'thread':e.thread,'fwd_thread':e.fwd_thread,'sequence_nr':e.sequence_nr,'scope':e.scope,
            'user_annotation':bool(e.is_user_annotation),'parent':None if e.cpu_parent is None else ids[id(e.cpu_parent)],
            'start_us':float(e.time_range.start),'end_us':float(e.time_range.end),'legacy':bool(e.is_legacy),
            'linked_kernels':[{'name':k.name,'device':k.device,'duration_us':float(k.duration)} for k in e.kernels] if e.device_type==torch.autograd.DeviceType.CPU else []})
    return rows

def verify(rows,*,arm,device):
    if arm not in ('original','math') or device not in ('cpu','cuda') or not isinstance(rows,list) or not 0<len(rows)<=MAX_EVENTS:
        raise ValueError('Exact component trace/domain required.')
    def chain(i):
        seen=set()
        while i is not None:
            if type(i) is not int or not 0<=i<len(rows) or i in seen:raise ValueError('Invalid component parent graph.')
            seen.add(i);yield i;i=rows[i]['parent']
    ancestry={};labels={}
    for i,r in enumerate(rows):
        if (r.get('index')!=i or r.get('device') not in ('cpu','cuda') or not isinstance(r.get('name'),str)
                or any(type(r.get(k)) is not int for k in ('device_index','thread','scope','fwd_thread','sequence_nr')) or r.get('legacy') is not False
                or type(r.get('user_annotation')) is not bool or any(type(r.get(k)) not in (int,float) or not math.isfinite(r[k]) for k in ('start_us','end_us'))
                or r['end_us']<r['start_us']):raise ValueError('Invalid device-typed component event.')
        c=list(chain(i));ancestry[i]=c
        if r['parent'] is not None:
            parent=rows[r['parent']]
            if r['device']!='cpu' or parent['device']!='cpu' or parent['thread']!=r['thread'] or parent['start_us']>r['start_us'] or parent['end_us']<r['end_us']:
                raise ValueError('Broken CPU event containment.')
        if r['device']=='cpu' and r['name'] in (FORWARD,BACKWARD):
            if not r['user_annotation'] or r['name'] in labels:raise ValueError('Unique actual component phase labels required.')
            labels[r['name']]=i
        for k in r.get('linked_kernels',[]):
            if not isinstance(k.get('name'),str) or type(k.get('duration_us')) not in (int,float) or not math.isfinite(k['duration_us']) or k['duration_us']<0 or type(k.get('device')) is not int:
                raise ValueError('Malformed actual linked kernel.')
    if set(labels)!={FORWARD,BACKWARD}:raise ValueError('Missing distinct actual forward/backward phases.')
    regions={name:[i for i,r in enumerate(rows) if r['device']=='cpu' and labels[name] in ancestry[i]] for name in labels}
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
                if any(n in ancestry[i] for n in nodes):
                    for k in rows[i].get('linked_kernels',[]):
                        if k['device']!=0:raise ValueError('Linked kernel belongs to another CUDA device.')
                        if k['duration_us']>0 and k['name'] in kernels:found.add(k['name'])
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
    return {'kind':'one-fixed-SDPA-associated-backend-v1','observed_family':family,'selection':arm,'device':device,
        'CPU_only':device=='cpu','CUDA_forward_and_backward_qualified':device=='cuda',
        'phase_correlated_CUDA_kernel_names':phase_kernels,'timings_usable_for_speed_or_cost':False,
        'default_family_was_assumed':False,'unsupported_backend_action':'refuse and preserve; never force a fallback'}
