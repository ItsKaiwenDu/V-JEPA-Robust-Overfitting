"""Scientific controls for a local observation prototype and inert proposal."""
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from models import predictor_sdpa_fixed_input_v1 as m

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'runs/pretrained/diagnostic/common-cotangent-handoff-v1-20261106'
SAVED=RAW/'linux10603-final/attended-q8txgu3g/saved'


@pytest.fixture(scope='module')
def evidence():
    review=json.loads((ROOT/'report/checks/2026-11-06-common-cotangent-v1-target-review.json').read_text())
    w=json.loads((SAVED/'results/0/worker.json').read_text())
    rows=[json.loads(line) for line in (SAVED/'results/0/attention-calls.jsonl').read_text().splitlines()]
    return review,w,rows


def test_plan_uses_first_recorded_boundary_and_keeps_missing_values_explicit(evidence):
    plan=m.proposal(*evidence)
    assert plan['selection']['forward_row_index']==24 and plan['selection']['layout']['v']['stride']==[2359296,32,1152,1]
    assert plan['raw_projection']['capture_eight_FP32_arrays_bytes']==25165824
    assert plan['raw_projection']['input_layout_storage_span_bytes']['v']==9434112
    assert plan['missing_actual_target_values']==['q','k','v','output','grad_output','dq','dk','dv']
    assert not plan['authorized'] and not plan['pure_backward_arithmetic_isolated']


@pytest.mark.parametrize('which',('rejected-control','missing-forward-call','shuffled-owners','wrong-selected-pid','changed-target-shape','non-gradient-input'))
def test_changed_saved_case_is_not_used_for_a_proposal(evidence,which):
    r,w,rows=copy.deepcopy(evidence)
    if which=='rejected-control':w['native_numerical_passed']=False
    elif which=='missing-forward-call':rows.pop(24)
    elif which=='shuffled-owners':rows[23]['owner']='predictor.0'
    elif which=='wrong-selected-pid':rows[24]['pid']+=1
    elif which=='changed-target-shape':rows[24]['q']['shape'][-1]=64
    else:rows[24]['q']['requires_grad']=False
    with pytest.raises(ValueError):m.proposal(r,w,rows)


@pytest.mark.parametrize('key',('authorized','new_target_execution_permitted','operator_present','candidate_native_numerical_passed',
    'pure_backward_arithmetic_isolated','same_QKV_and_grad_output_imply_same_backend_internal_saved_state'))
def test_proposal_cannot_be_promoted_to_execution_or_backward_only_proof(evidence,key):
    plan=m.proposal(*evidence);plan[key]=True
    with pytest.raises(ValueError):m.validate(plan)


def test_adaptive_selection_and_expanded_resources_are_not_the_reviewed_plan(evidence):
    plan=m.proposal(*evidence);plan['selection']['block']=11
    with pytest.raises(ValueError):m.validate(plan)
    plan=m.proposal(*evidence);plan['resource_proposal']['work_seconds']+=1
    with pytest.raises(ValueError):m.validate(plan)


def toy(observer=None,*,recompute=False):
    gen=torch.Generator(device='cpu').manual_seed(20261106)
    x=torch.randn(1,6,8,generator=gen,requires_grad=True)
    weight=torch.randn(8,24,generator=gen)*.125
    target=torch.randn(1,2,6,4,generator=gen)
    def forward(t):
        q,k,v=(t@weight).view(1,6,3,2,4).permute(2,0,3,1,4).unbind(0)
        q,k=q.contiguous(),k.contiguous()
        if observer is not None:return observer(q,k,v)
        return torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False,attn_mask=None)
    with sdpa_kernel(SDPBackend.MATH):
        out=checkpoint(forward,x,use_reentrant=False) if recompute else forward(x)
        loss=(out-target).square().mean()
        loss.backward()
    assert weight.grad is None
    return {k:t.detach().clone() for k,t in {'loss':loss,'output':out,'input':x.grad}.items()}


@pytest.mark.parametrize('recompute',(False,True))
def test_nonintrusive_native_capture_and_same_input_vjp_replay(recompute):
    before=toy(recompute=recompute)
    observer=m.LocalSDPABoundary()
    with observer:
        during=toy(observer,recompute=recompute)
    after=toy(recompute=recompute)
    for key in before:
        assert torch.equal(before[key],during[key]) and torch.equal(before[key],after[key])
    assert observer.handles==[] and not observer.active
    assert observer.calls==(2 if recompute else 1)
    proof=m.tiny_same_input_comparison(observer.values)
    assert all(proof['native_replay_exact'].values()) and proof['non_dense_V_stride_preserved']
    assert not proof['CUDA_qualified'] and not proof['pretrained_target_values_captured']
    folder=os.environ.get('VJEPA_FIXED_PLAN_EVIDENCE')
    if folder:
        p=Path(folder)/('tiny-checkpointed.json' if recompute else 'tiny-direct.json')
        item={**proof,'checkpointed':recompute,'capture_total_calls':observer.calls,
            'before_observed_after_full_toy_loss_output_input_bitwise_equal':True,
            'native_value_hashes':{k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in before.items()},
            'observed':{k:{**{n:v for n,v in item.items() if n!='array'},'sha256':hashlib.sha256(item['array'].tobytes()).hexdigest()} for k,item in observer.values.items()}}
        with p.open('x') as f:f.write(json.dumps(item,indent=2,sort_keys=True)+'\n')


def test_primary_exception_removes_hooks_and_keeps_partial_capture():
    observer=m.LocalSDPABoundary()
    with pytest.raises(RuntimeError,match='primary'):
        with observer:
            q=torch.ones(1,2,6,4,requires_grad=True)
            observer(q,q.clone(),q.clone())
            raise RuntimeError('primary')
    assert not observer.active and observer.handles==[]
    assert set(observer.values)=={'q','k','v','output'}
    with pytest.raises(ValueError):m.tiny_same_input_comparison(observer.values)


def test_forward_without_backward_is_not_a_complete_capture():
    with pytest.raises(ValueError,match='Incomplete'):
        with m.LocalSDPABoundary() as observer:
            x=torch.ones(1,1,4,4,requires_grad=True);observer(x,x.clone(),x.clone())
    assert observer.handles==[] and not observer.active


@pytest.mark.parametrize('kind',('FP64','full-size','no-gradient','nonfinite'))
def test_local_prototype_refuses_target_or_invalid_domain(kind):
    shape=(1,12,2048,32) if kind=='full-size' else (1,1,4,4)
    q=torch.ones(shape,dtype=torch.float64 if kind=='FP64' else torch.float32,requires_grad=kind!='no-gradient')
    if kind=='nonfinite':
        with torch.no_grad():q[0,0,0,0]=float('nan')
    with pytest.raises(ValueError):
        with m.LocalSDPABoundary() as observer:observer(q,q.clone(),q.clone())


def test_prototype_refuses_university_host(monkeypatch):
    monkeypatch.setattr(m.platform,'node',lambda:'linux10603')
    with pytest.raises(ValueError,match='Local tiny'):m.LocalSDPABoundary()


def test_changed_cotangent_does_not_reproduce_native_boundary_values():
    with m.LocalSDPABoundary() as observer:toy(observer)
    broken=copy.deepcopy(observer.values)
    broken['grad_output']['array'][0,0,0,0]+=np.float32(.25)
    with pytest.raises(ValueError,match='native replay'):m.tiny_same_input_comparison(broken)


def test_same_input_math_VJP_still_checks_actual_incoming_gradient_not_predictor_output_gradient():
    with m.LocalSDPABoundary() as observer:toy(observer)
    broken=copy.deepcopy(observer.values)
    broken['grad_output']['array']=np.ones((1,6,4),dtype=np.float32)
    with pytest.raises(ValueError):m.tiny_same_input_comparison(broken)
