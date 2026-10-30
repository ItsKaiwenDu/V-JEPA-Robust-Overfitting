"""Independent derivatives, upstream operand capture and backend evidence gates."""
import importlib
import math

import pytest
import torch
from torch.nn import functional as F

from models.attention_component import (CASES,backend_evidence,clone_layout,compare,force_backend,
    fp64_oracle,iteration,operands,sdp_flags)
from models.execution import configure_execution
from test_projection_reuse import restore_process_settings


@pytest.mark.parametrize('chunk',[1,3,7])
def test_chunked_analytical_oracle_matches_dense_autograd_and_directional_derivative(chunk):
    gen=torch.Generator().manual_seed(81)
    xs=[torch.randn(2,2,7,5,generator=gen,dtype=torch.float64).requires_grad_() for _ in range(3)]
    g=torch.randn(xs[0].shape,generator=gen,dtype=torch.float64)
    output=((xs[0]@xs[1].transpose(-2,-1))/math.sqrt(5)).softmax(-1)@xs[2]
    gradients=torch.autograd.grad(output,xs,g)
    actual=fp64_oracle(*xs,g,chunk=chunk)
    for name,expected in zip(('output','dq','dk','dv'),(output,*gradients)):
        torch.testing.assert_close(actual[name],expected,rtol=2e-13,atol=2e-14)
    directions=[torch.randn(x.shape,generator=gen,dtype=torch.float64) for x in xs]
    def loss(step):
        q,k,v=[x.detach()+step*d for x,d in zip(xs,directions)]
        return ((((q@k.transpose(-2,-1))/math.sqrt(5)).softmax(-1)@v)*g).sum()
    numerical=(loss(1e-6)-loss(-1e-6))/2e-6
    analytical=sum((actual[n]*d).sum() for n,d in zip(('dq','dk','dv'),directions))
    torch.testing.assert_close(numerical,analytical,rtol=1e-7,atol=1e-8)


@pytest.mark.parametrize('case',CASES)
def test_upstream_post_rotary_capture_preserves_layout_and_math_vjp(case,upstream):
    configure_execution('shared-fp64-native',device='cpu');torch.set_num_threads(4)
    original=F.scaled_dot_product_attention;flags=sdp_flags();rng=torch.get_rng_state().clone()
    (q,k,v,g),shape=operands(upstream,case,{},True)
    assert F.scaled_dot_product_attention is original and flags==sdp_flags() and torch.equal(rng,torch.get_rng_state())
    assert v.stride()[-2]>q.stride()[-2]
    leaves=[clone_layout(t,'cpu',requires_grad=True) for t in (q,k,v)]
    assert [t.stride() for t in leaves]==[q.stride(),k.stride(),v.stride()]
    assert all(t.is_leaf for t in leaves)
    actual,_=iteration(leaves,g,'math','cpu');expected=fp64_oracle(q,k,v,g)
    assert all(item['passed'] for item in compare(actual,expected).values())
    again,shape_again=operands(upstream,case,{},True)
    assert all(torch.equal(a,b) for a,b in zip((q,k,v,g),again))
    assert torch.equal(shape.pop('positions'),shape_again.pop('positions')) and shape==shape_again
    assert flags==sdp_flags()


def test_capture_exception_restores_global_function_and_flags(upstream,monkeypatch):
    from models.vjepa2 import use_upstream
    use_upstream(upstream);cls=importlib.import_module('src.models.utils.modules').RoPEAttention
    saved=F.scaled_dot_product_attention;flags=sdp_flags()
    def fail(*args,**kwargs):raise RuntimeError('injected upstream failure')
    monkeypatch.setattr(cls,'forward',fail)
    with pytest.raises(RuntimeError,match='injected'):operands(upstream,'predictor',{},True)
    assert F.scaled_dot_product_attention is saved and sdp_flags()==flags


@pytest.mark.parametrize('arm',['math','efficient'])
def test_exclusive_backend_flags_restore_even_when_body_raises(arm):
    before=sdp_flags()
    with pytest.raises(RuntimeError,match='deliberate'):
        with force_backend(arm) as inside:
            assert sum(inside.values())==1 and inside[arm]
            raise RuntimeError('deliberate')
    assert sdp_flags()==before


def rows(arm,device='cuda'):
    ops=['aten::_scaled_dot_product_attention_math'] if arm=='math' else ['aten::_scaled_dot_product_efficient_attention','aten::_scaled_dot_product_efficient_attention_backward']
    kernels=[] if device=='cpu' else ['gemm_kernel'] if arm=='math' else ['fmha_cutlassF_f32_kernel','fmha_cutlassB_f32_kernel']
    return [{'name':n,'device_type':'cpu','duration_us':1.} for n in ops]+[{'name':n,'device_type':'cuda','duration_us':2.} for n in kernels]


@pytest.mark.parametrize('arm',['math','efficient'])
def test_backend_evidence_requires_device_typed_forward_and_backward(arm):
    observed=rows(arm);assert backend_evidence(observed,arm,'cuda')['actual_CUDA_backend_observed']
    for row in observed:row['device_type']='cpu'
    assert not backend_evidence(observed,arm,'cuda')['passed']
    observed=rows(arm);observed.append({'name':'aten::_scaled_dot_product_flash_attention','device_type':'cpu','duration_us':1})
    assert not backend_evidence(observed,arm,'cuda')['passed']
    assert backend_evidence(rows('math','cpu'),'math','cpu')['passed']
    assert not backend_evidence(rows('efficient','cpu'),'efficient','cpu')['passed']


@pytest.mark.parametrize('bad',[float('nan'),-1,'1'])
def test_malformed_profiler_duration_rejects(bad):
    observed=rows('math');observed[0]['duration_us']=bad
    with pytest.raises(ValueError):backend_evidence(observed,'math','cuda')


def test_invalid_operand_or_oracle_bounds_reject_before_comparison():
    x=torch.ones(1,1,3,4)
    with pytest.raises(ValueError):clone_layout(x.double(),'cpu')
    with pytest.raises(ValueError):fp64_oracle(x,x,x,x,chunk=129)
    with pytest.raises(ValueError):fp64_oracle(x,x[...,:2],x,x)
    with pytest.raises(ValueError):fp64_oracle(x*float('nan'),x,x,x)
    with pytest.raises(ValueError):
        with force_backend('default'):pass
