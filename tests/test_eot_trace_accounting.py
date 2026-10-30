"""Meaningful regressions for GPU annotation duplication and linkage mismatch."""
import copy
import pytest
from models.eot_trace_accounting import device_activity


def event(name,start,end,device='cuda',linked=()):
    return {'name':name,'device':device,'device_index':0,'start_us':start,'end_us':end,'linked_kernels':list(linked)}


def test_nested_CUDA_annotations_never_inflate_kernel_time_or_busy_union():
    rows=[event('eot-profile/input_backward',0,100),event('eot-profile/encoder_block_forward_or_recompute',10,90),
          event('fmha_cutlassB_kernel',20,50),event('ampere_sgemm',40,60)]
    v=device_activity(rows)
    assert v['excluded_CUDA_annotation_events']==2 and v['excluded_CUDA_annotation_inclusive_duration_us']==180
    assert v['direct_device_activity_events']==2 and v['direct_device_activity_duration_us']==50
    assert v['direct_device_activity_union_us_by_device']=={'0':40}
    assert v['groups_by_explicit_kernel_name']['attention_backward']['duration_us']==30
    assert v['groups_by_explicit_kernel_name']['attention_backward']['share_of_direct_device_duration']==.6
    assert not v['device_duration_shares_are_wall_time_shares']


def test_CPU_linked_duplicates_are_reported_without_replacing_direct_device_time():
    k={'name':'actual_kernel','device':0,'duration_us':10.}
    rows=[event('actual_kernel',0,10),event('aten::operation',0,12,'cpu',[k]),event('lifted-parent',0,13,'cpu',[k])]
    v=device_activity(rows)
    assert v['direct_device_activity_duration_us']==10 and v['CPU_linked_activity_duration_us']==20
    assert v['CPU_linked_to_direct_ratio']==2 and v['CPU_linked_minus_direct_duration_us']==10
    assert v['CPU_linked_discrepancies_by_name']['actual_kernel']['raw']['count']==1
    assert not v['CPU_linked_module_buckets_are_a_conserved_partition']


def test_cpu_only_trace_cannot_claim_CUDA_or_attention_share():
    v=device_activity([event('aten::mm',0,1,'cpu')])
    assert v['direct_device_activity_events']==0 and v['direct_device_activity_duration_us']==0
    assert v['groups_by_explicit_kernel_name']['attention_backward']['share_of_direct_device_duration'] is None


@pytest.mark.parametrize('defect',['nonfinite','negative_interval','bad_device','bad_index','bad_link','annotation_link'])
def test_invalid_observations_reject(defect):
    r=event('kernel',0,1)
    if defect=='nonfinite':r['end_us']=float('nan')
    elif defect=='negative_interval':r['end_us']=-1
    elif defect=='bad_device':r['device']='cuda-fixture'
    elif defect=='bad_index':r['device_index']=-1
    else:
        r['device']='cpu';r['linked_kernels']=[{'name':'eot-profile/x' if defect=='annotation_link' else 'kernel','duration_us':1 if defect=='annotation_link' else -1}]
    with pytest.raises(ValueError):device_activity([r])
