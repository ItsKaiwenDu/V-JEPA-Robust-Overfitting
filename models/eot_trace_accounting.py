"""Offline correction of device attribution; never changes captured summaries.

Kineto can emit CUDA-side record_function ranges alongside real device
activities. Our own eot-profile/* ranges are known annotations, not kernels.
Keep those separate and use only direct device events for activity shares.
CPU-linked durations are independent observations, not a second total to add.
"""
from collections import defaultdict
import math

ANNOTATION_PREFIX='eot-profile/'


def interval_union(intervals):
    result=0.;end=None
    for start,stop in sorted(intervals):
        if end is None or start>=end:result+=stop-start;end=stop
        elif stop>end:result+=stop-end;end=stop
    return result


def device_activity(rows):
    names=defaultdict(lambda:{'count':0,'duration_us':0.})
    annotations=defaultdict(lambda:{'count':0,'inclusive_duration_us':0.})
    linked=defaultdict(lambda:{'count':0,'duration_us':0.})
    intervals=defaultdict(list)
    for r in rows:
        if (r.get('device') not in ('cpu','cuda') or not isinstance(r.get('name'),str)
                or any(type(r.get(k)) not in (int,float) or not math.isfinite(r[k]) for k in ('start_us','end_us'))
                or r['end_us']<r['start_us']):raise ValueError('Invalid retained device event.')
        if r['device']=='cpu':
            for k in r['linked_kernels']:
                if (not isinstance(k['name'],str) or type(k['duration_us']) not in (int,float)
                        or not math.isfinite(k['duration_us']) or k['duration_us']<0):raise ValueError('Invalid linked activity.')
                if k['name'].startswith(ANNOTATION_PREFIX):raise ValueError('User annotation unexpectedly linked as a kernel.')
                linked[k['name']]['count']+=1;linked[k['name']]['duration_us']+=k['duration_us']
            continue
        if type(r.get('device_index')) is not int or r['device_index']<0:raise ValueError('Invalid CUDA device index.')
        duration=r['end_us']-r['start_us']
        if r['name'].startswith(ANNOTATION_PREFIX):
            v=annotations[r['name']];v['count']+=1;v['inclusive_duration_us']+=duration
        else:
            v=names[r['name']];v['count']+=1;v['duration_us']+=duration
            intervals[r['device_index']].append((r['start_us'],r['end_us']))
    total=sum(v['duration_us'] for v in names.values());linked_total=sum(v['duration_us'] for v in linked.values())
    groups={label:{'count':0,'duration_us':0.} for label in ('attention_backward','attention_forward','other_device_activity')}
    for name,v in names.items():
        label=('attention_backward' if 'AttentionBackwardKernel<' in name or name.startswith('fmha_cutlassB_') else
               'attention_forward' if 'AttentionKernel<' in name or name.startswith('fmha_cutlassF_') else 'other_device_activity')
        groups[label]['count']+=v['count'];groups[label]['duration_us']+=v['duration_us']
    for v in groups.values():v['share_of_direct_device_duration']=v['duration_us']/total if total else None
    discrepancy={name:{'raw':names.get(name,{'count':0,'duration_us':0.}),
                       'linked':linked.get(name,{'count':0,'duration_us':0.})}
                 for name in sorted(set(names)|set(linked))
                 if names.get(name,{}).get('count',0)!=linked.get(name,{}).get('count',0)
                 or not math.isclose(names.get(name,{}).get('duration_us',0.),linked.get(name,{}).get('duration_us',0.),rel_tol=1e-12,abs_tol=1e-6)}
    return {'schema_version':1,'known_CUDA_annotation_prefix':ANNOTATION_PREFIX,
            'direct_device_activity_events':sum(v['count'] for v in names.values()),
            'excluded_CUDA_annotation_events':sum(v['count'] for v in annotations.values()),
            'excluded_CUDA_annotation_inclusive_duration_us':sum(v['inclusive_duration_us'] for v in annotations.values()),
            'direct_device_activity_duration_us':total,
            'direct_device_activity_union_us_by_device':{str(k):interval_union(v) for k,v in sorted(intervals.items())},
            'groups_by_explicit_kernel_name':groups,
            'top_direct_device_activity':sorted([{'name':name,**v,'share_of_direct_device_duration':v['duration_us']/total if total else None}
                                                for name,v in names.items()],key=lambda r:(-r['duration_us'],r['name']))[:40],
            'excluded_CUDA_annotation_ranges':dict(sorted(annotations.items())),
            'CPU_linked_activity_duration_us':linked_total,
            'CPU_linked_minus_direct_duration_us':linked_total-total,
            'CPU_linked_to_direct_ratio':linked_total/total if total else None,
            'CPU_linked_discrepancies_by_name':discrepancy,
            'CPU_linked_module_buckets_are_a_conserved_partition':False,
            'device_duration_shares_are_wall_time_shares':False,
            'captured_scope_represents_all_EOT_draws_or_full_PGD':False,
            'scope':'Detailed first-draw device activity, including transfers if present. Known nested GPU annotations excluded; CPU-linked observations retained separately.'}
