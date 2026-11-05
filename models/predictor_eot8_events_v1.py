"""Retained actual event ownership; missing/ambiguous associations refuse."""
import json
import math
import os
from pathlib import Path

PREFIX = 'predictor-eot8/call/'
MAX_BYTES = 256 * 2**20
MAX_EVENTS = 800000


def save(profile, path):
    import torch
    events = profile.events()
    if not events or len(events) > MAX_EVENTS:raise ValueError('Missing/over-cap events; no truncation.')
    indices = {id(e):i for i,e in enumerate(events)}
    partial = Path(str(path)+'.partial'); size = 0
    if Path(path).exists():raise FileExistsError(path)
    with partial.open('x') as f:
        for i,e in enumerate(events):
            for attr in ('sequence_nr','fwd_thread','scope','is_user_annotation'):
                if not hasattr(e,attr):raise ValueError('Profiler lacks association field: '+attr)
            device = 'cpu' if e.device_type == torch.autograd.DeviceType.CPU else 'cuda' if e.device_type == torch.autograd.DeviceType.CUDA else 'unsupported'
            row = {'index':i,'name':e.name,'device':device,'device_index':e.device_index,
                   'thread':e.thread,'fwd_thread':e.fwd_thread,'sequence_nr':e.sequence_nr,'scope':e.scope,
                   'user_annotation':bool(e.is_user_annotation),'parent':None if e.cpu_parent is None else indices[id(e.cpu_parent)],
                   'start_us':float(e.time_range.start),'end_us':float(e.time_range.end),'legacy':bool(e.is_legacy),
                   'linked_kernels':[{'name':k.name,'device':k.device,'duration_us':float(k.duration)} for k in e.kernels] if device=='cpu' else []}
            line=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n';size+=len(line.encode())
            if size>MAX_BYTES:raise ValueError('Trace byte bound; retain partial file.')
            f.write(line)
        f.flush();os.fsync(f.fileno())
    os.rename(partial,path)
    return {'events':len(events),'bytes':size,'association_fields_retained':True}


def read(path):
    path=Path(path)
    if path.stat().st_size>MAX_BYTES:raise ValueError('Trace byte bound.')
    rows=[]
    with path.open() as f:
        for line in f:
            if len(rows)>=MAX_EVENTS:raise ValueError('Trace event bound.')
            rows.append(json.loads(line))
    return rows


def verify(rows, calls, *, cuda, mode, owners):
    if not rows or not calls or mode not in ('original','predictor-math'):raise ValueError('Missing trace/calls/domain.')
    def ancestors(i):
        seen=set()
        while i is not None:
            if type(i) is not int or not 0<=i<len(rows) or i in seen:raise ValueError('Invalid event parent graph.')
            seen.add(i);yield rows[i];i=rows[i]['parent']
    labels={PREFIX+str(c['index'])+'/'+c['owner']:c for c in calls if c['phase']=='draw-profiled'}
    if len(labels)!=len([c for c in calls if c['phase']=='draw-profiled']):raise ValueError('Duplicate call labels.')
    observed={};sequence={}; forward={};backward={}
    for i,r in enumerate(rows):
        if (r['index']!=i or r['device'] not in ('cpu','cuda') or not isinstance(r['name'],str)
                or any(type(r[k]) is not int for k in ('sequence_nr','scope','thread'))
                or any(not math.isfinite(r[k]) for k in ('start_us','end_us')) or r['end_us']<r['start_us'] or r['legacy']):
            raise ValueError('Invalid/legacy raw event.')
        chain=list(ancestors(i))
        if r['parent'] is not None:
            p=rows[r['parent']]
            if r['device']!='cpu' or p['device']!='cpu' or p['thread']!=r['thread'] or p['start_us']>r['start_us'] or p['end_us']<r['end_us']:
                raise ValueError('Broken CPU containment.')
        for k in r['linked_kernels']:
            if not isinstance(k['name'],str) or not math.isfinite(k['duration_us']) or k['duration_us']<0:raise ValueError('Invalid CUDA link.')
        own=[p['name'] for p in chain if p['name'].startswith(PREFIX)]
        if len(own)>1:raise ValueError('Ambiguous nested SDPA ownership.')
        if r['name'].startswith(PREFIX):
            if r['device']=='cuda':continue  # GPU annotation copies are not kernels or owners.
            if r['name'] not in labels or r['name'] in observed:raise ValueError('Unknown/duplicate owned event.')
            observed[r['name']]=i
        if own and r['device']=='cpu':
            label=own[0]
            if label not in labels:raise ValueError('Unknown call in retained event tree.')
            forward.setdefault(label,[]).append(r)
            if r['scope']==0 and r['sequence_nr']>=0:
                key=(r['thread'],r['sequence_nr']);sequence.setdefault(key,set()).add(label)
    if set(observed)!=set(labels):raise ValueError('Missing actual owned SDPA events.')
    if any(len(v)!=1 for v in sequence.values()):raise ValueError('Ambiguous forward/backward sequence ownership.')
    matched_nodes=0
    for i,r in enumerate(rows):
        if r['device']!='cpu':continue
        chain=list(ancestors(i));nodes=[p for p in chain if p['scope']==1 and p['sequence_nr']>=0]
        keys={(p['fwd_thread'],p['sequence_nr']) for p in nodes}
        assigned=set().union(*(sequence.get(k,set()) for k in keys)) if keys else set()
        if len(assigned)>1:raise ValueError('Backward event has ambiguous ownership.')
        if assigned:
            label=next(iter(assigned));backward.setdefault(label,[]).append(r)
            if r['scope']==1:matched_nodes+=1
    summary={}
    for owner in owners:
        selected=[label for label,c in labels.items() if c['owner']==owner]
        returned=[label for label in selected if labels[label]['outcome']=='returned' and not labels[label].get('during_backward',False)]
        if len(returned)!=2:raise ValueError('Both mask families need completed owned SDPA calls.')
        fs=[r for label in selected for r in forward.get(label,[])]
        bs=[r for label in selected for r in backward.get(label,[])]
        if not bs:raise ValueError('Missing actual backward association for '+owner)
        want_math=mode=='predictor-math' and owner.startswith('predictor.')
        if any(not backward.get(label) for label in returned):raise ValueError('A completed forward call lacks backward association.')
        if cuda:
            for label in returned:
                required='aten::_scaled_dot_product_attention_math' if want_math else 'aten::_scaled_dot_product_efficient_attention'
                if required not in {r['name'] for r in forward.get(label,[])}:raise ValueError('A mask-family call lacks actual backend evidence.')
            names={r['name'] for r in fs};bnames={r['name'] for r in bs}
            if want_math:
                if 'aten::_scaled_dot_product_attention_math' not in names or not any('BmmBackward' in n for n in bnames) or not any('SoftmaxBackward' in n for n in bnames):
                    raise ValueError('Actual predictor math forward/backward not observed.')
                if any('_efficient_attention' in n or '_flash_attention' in n or '_cudnn_attention' in n for n in names|bnames):raise ValueError('Unexpected candidate backend.')
            elif 'aten::_scaled_dot_product_efficient_attention' not in names or 'aten::_scaled_dot_product_efficient_attention_backward' not in bnames:
                raise ValueError('Original efficient forward/backward path not observed.')
            if not any(r['linked_kernels'] for r in fs) or not any(r['linked_kernels'] for r in bs):raise ValueError('Missing owned CUDA kernel links.')
        summary[owner]={'call_labels':len(selected),'completed_calls':len(returned),'backward_events':len(bs),
                        'forward_linked_kernel_count':sum(len(r['linked_kernels']) for r in fs),
                        'backward_linked_kernel_count':sum(len(r['linked_kernels']) for r in bs),
                        'actual_CUDA_backend_proved':cuda,'expected_math':want_math if cuda else None}
    if set(c['owner'] for c in labels.values())!=set(owners):raise ValueError('Owner coverage changed.')
    if cuda and not any(r['device']=='cuda' and not r['user_annotation'] and not r['name'].startswith(PREFIX) for r in rows):raise ValueError('Missing raw CUDA activity.')
    return {'kind':'predictor-eot8-owned-event-check-v1','owners':summary,'matched_backward_nodes':matched_nodes,
            'CUDA_qualified':cuda,'CUDA_proof_scope':'first-draw-two-families-only' if cuda else None,'CPU_analogue_only':not cuda,'timings_usable_for_speed_or_cost':False,
            'unassigned_events_retained':True,'conserved_timing_partition_claimed':False}
