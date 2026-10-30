"""Bounded, independently reaggregatable EOT trace records; no model changes."""
from collections import defaultdict
from contextlib import contextmanager
import functools
import json
import math

import torch

MAX_EVENTS=800000
MAX_EVENT_BYTES=256*2**20
PREFIX='eot-profile/'


@contextmanager
def annotate(model):
    """Wrap only method entry/exit; keep wrappers through checkpoint recomputation."""
    changes=[]
    def install(owner,name,label):
        existed=name in owner.__dict__;old=owner.__dict__.get(name);original=getattr(owner,name)
        @functools.wraps(original)
        def observed(*args,**kwargs):
            with torch.profiler.record_function(PREFIX+label):return original(*args,**kwargs)
        setattr(owner,name,observed);changes.append((owner,name,existed,old))
    try:
        install(model,'normalize','normalize_and_pixel_validation')
        install(model,'_validate_masks','mask_validation')
        install(model,'loss_per_example','draw_loss')
        install(model.encoder,'forward','context_encoder')
        install(model.encoder.patch_embed,'forward','patch_projection')
        install(model.predictor,'forward','predictor')
        for block in model.encoder.blocks:install(block,'forward','encoder_block_forward_or_recompute')
        for block in model.predictor.predictor_blocks:install(block,'forward','predictor_block_forward_or_recompute')
        yield
    finally:
        for owner,name,existed,old in reversed(changes):
            if existed:setattr(owner,name,old)
            else:delattr(owner,name)


def save_events(profile,path):
    events=profile.events()
    if len(events)>MAX_EVENTS:raise ValueError('Profile event cap exceeded; do not truncate or claim success.')
    indices={id(e):i for i,e in enumerate(events)};size=0
    with path.open('x') as handle:
        for i,e in enumerate(events):
            cpu=e.device_type==torch.autograd.DeviceType.CPU
            row={'index':i,'name':e.name,'device':'cpu' if cpu else 'cuda' if e.device_type==torch.autograd.DeviceType.CUDA else str(e.device_type),
                 'device_index':e.device_index,'thread':e.thread,
                 'start_us':float(e.time_range.start),'end_us':float(e.time_range.end),
                 'parent':None if e.cpu_parent is None else indices[id(e.cpu_parent)],
                 'self_cpu_us':float(e.self_cpu_time_total) if cpu else 0.,
                 'legacy':bool(e.is_legacy),'async':bool(e.is_async),
                 'linked_kernels':[{'name':k.name,'device':k.device,'duration_us':float(k.duration)} for k in e.kernels] if cpu else []}
            line=json.dumps(row,separators=(',',':'),allow_nan=False)+'\n';size+=len(line.encode())
            if size>MAX_EVENT_BYTES:raise ValueError('Trace byte cap exceeded; retain partial evidence.')
            handle.write(line)
    return aggregate(path)


def union_us(intervals):
    total=0.;end=None
    for start,stop in sorted(intervals):
        if end is None or start>=end:total+=stop-start;end=stop
        elif stop>end:total+=stop-end;end=stop
    return total


def aggregate(path):
    """Exclusive CPU/event-linked kernel accounting; never sum inclusive parents."""
    if path.stat().st_size>MAX_EVENT_BYTES:raise ValueError('Trace too large.')
    rows=[]
    with path.open() as f:
        for line in f:
            if len(rows)>=MAX_EVENTS:raise ValueError('Too many profile rows.')
            rows.append(json.loads(line))
    if not rows:raise ValueError('Missing trace.')
    children_duration=[0.]*len(rows)
    for i,r in enumerate(rows):
        if (r.get('index')!=i or r.get('device') not in ('cpu','cuda')
                or any(not isinstance(r.get(k),(int,float)) or not math.isfinite(r[k]) for k in ('start_us','end_us','self_cpu_us'))
                or r['end_us']<r['start_us']):raise ValueError('Invalid event record.')
        parent=r.get('parent')
        if parent is not None:
            if type(parent) is not int or not 0<=parent<len(rows):raise ValueError('Invalid CPU parent.')
            children_duration[parent]+=r['end_us']-r['start_us']
    labels={};categories=defaultdict(lambda:{'CPU_self_us':0.,'linked_kernel_us':0.,'CPU_events':0})
    kernels=defaultdict(lambda:{'duration_us':0.,'count':0});intervals=defaultdict(list)
    for i,r in enumerate(rows):
        if (r['index']!=i or not isinstance(r['name'],str) or r['device'] not in ('cpu','cuda')
                or any(not math.isfinite(r[k]) for k in ('start_us','end_us','self_cpu_us'))
                or r['end_us']<r['start_us'] or r['parent'] is not None and (type(r['parent']) is not int or not 0<=r['parent']<len(rows))):
            raise ValueError('Invalid event record.')
        if r['device']=='cuda':
            if r['parent'] is not None or r['linked_kernels'] or r['self_cpu_us']!=0:raise ValueError('CUDA event unexpectedly has CPU nesting.')
            duration=r['end_us']-r['start_us'];kernels[r['name']]['duration_us']+=duration;kernels[r['name']]['count']+=1
            intervals[r['device_index']].append((r['start_us'],r['end_us']))
            continue
        expected_self=0. if r['async'] else r['end_us']-r['start_us']-children_duration[i]
        if not math.isclose(expected_self,r['self_cpu_us'],rel_tol=1e-12,abs_tol=1e-7):raise ValueError('CPU self time disagrees with retained containment intervals.')
        parent=r['parent']
        if parent is not None:
            p=rows[parent]
            if p['device']!='cpu' or p['thread']!=r['thread'] or p['start_us']>r['start_us'] or p['end_us']<r['end_us']:
                raise ValueError('Invalid CPU containment tree.')
        category=r['name'][len(PREFIX):] if r['name'].startswith(PREFIX) else None
        visited={i}
        while parent is not None:
            if parent in visited:raise ValueError('Cyclic CPU containment.')
            visited.add(parent);p=rows[parent]
            if category is None and p['name'].startswith(PREFIX):category=p['name'][len(PREFIX):]
            parent=p['parent']
        category=category or 'unattributed'
        row=categories[category];row['CPU_self_us']+=r['self_cpu_us'];row['CPU_events']+=1
        for k in r['linked_kernels']:
            if not isinstance(k['name'],str) or not math.isfinite(k['duration_us']) or k['duration_us']<0:
                raise ValueError('Invalid event-linked kernel observation.')
            row['linked_kernel_us']+=k['duration_us']
        if r['name'].startswith(PREFIX):
            label=r['name'][len(PREFIX):];v=labels.setdefault(label,{'count':0,'CPU_inclusive_us':0.})
            v['count']+=1;v['CPU_inclusive_us']+=r['end_us']-r['start_us']
    kernel_sum=sum(v['duration_us'] for v in kernels.values());linked_sum=sum(v['linked_kernel_us'] for v in categories.values())
    return {'events':len(rows),'CPU_events':sum(r['device']=='cpu' for r in rows),
            'CUDA_events':sum(r['device']=='cuda' for r in rows),'legacy_events':sum(r['legacy'] for r in rows),
            'annotation_ranges':labels,'exclusive_CPU_and_linked_kernel_categories':dict(sorted(categories.items())),
            'CUDA_kernel_duration_sum_us':kernel_sum,'CPU_event_linked_kernel_sum_us':linked_sum,
            'linked_to_raw_kernel_duration_ratio':linked_sum/kernel_sum if kernel_sum else None,
            'CUDA_busy_interval_union_us_by_device':{str(k):union_us(v) for k,v in intervals.items()},
            'top_CUDA_kernels':sorted([{'name':k,**v} for k,v in kernels.items()],key=lambda r:(-r['duration_us'],r['name']))[:40],
            'native_backward_observed':any('at::native::vol2im_kernel<' in k for k in kernels),
            'cudnn_convolution_kernel_observed':any('cudnn' in k.lower() and 'conv' in k.lower() for k in kernels),
            'accounting':'CPU self time and each CPU event-linked kernel counted once; inclusive annotation totals are separate and must not be added. Kernel sums are not elapsed wall time.',
            'profiled_timings_are_throughput_or_cost_units':False}
