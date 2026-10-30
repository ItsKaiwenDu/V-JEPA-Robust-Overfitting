"""Independent analytical checks and refusal tests for the offline NumPy audit."""
import copy

import numpy as np
import pytest
import torch

from models.attention_saved_evidence import agreement,backend,differences,oracle


@pytest.mark.parametrize('chunk',[1,3,7])
def test_numpy_oracle_matches_dense_torch_derivatives_and_finite_direction(chunk):
    rng=np.random.default_rng(41);xs=[rng.normal(size=(2,2,7,5)) for _ in range(3)];g=rng.normal(size=xs[0].shape)
    q,k,v=[torch.tensor(x,dtype=torch.float64,requires_grad=True) for x in xs]
    result=((q@k.transpose(-2,-1))/(5**.5)).softmax(-1)@v
    gradients=torch.autograd.grad(result,(q,k,v),torch.tensor(g))
    observed=oracle(*xs,g,chunk=chunk)
    for name,tensor in zip(('output','dq','dk','dv'),(result,*gradients)):
        np.testing.assert_allclose(observed[name],tensor.detach().numpy(),rtol=2e-13,atol=2e-14)
    directions=[rng.normal(size=x.shape) for x in xs]
    def loss(step):return np.sum(oracle(*[x+step*d for x,d in zip(xs,directions)],g,chunk=chunk)['output']*g)
    finite=(loss(1e-6)-loss(-1e-6))/2e-6
    analytic=sum(np.sum(observed[n]*d) for n,d in zip(('dq','dk','dv'),directions))
    assert finite==pytest.approx(analytic,rel=1e-7,abs=1e-8)


def test_saved_comparison_keeps_elementwise_absolute_and_relative_gate():
    expected=np.array([0.,1.]);bounds={'atol':2e-7,'rtol':.002}
    assert agreement([1e-7,1.001],expected,bounds)['passed']
    failed=agreement([3e-7,1.001],expected,bounds)
    assert not failed['passed'] and failed['failed_elements']==1 and failed['max_tolerance_ratio']==pytest.approx(1.5)


@pytest.mark.parametrize('defect',['nan','shape','chunk','negative_bound','zero_bound'])
def test_invalid_oracle_or_saved_comparison_refuses(defect):
    x=np.ones((1,1,3,4));bounds={'atol':2e-7,'rtol':.002}
    with pytest.raises(ValueError):
        if defect=='nan':oracle(x*np.nan,x,x,x)
        elif defect=='shape':oracle(x,x[...,:2],x,x)
        elif defect=='chunk':oracle(x,x,x,x,chunk=129)
        elif defect=='negative_bound':agreement(x,x,{'atol':2e-7,'rtol':-1})
        else:agreement(x,x,{'atol':0,'rtol':.002})


def observations(arm):
    ops=['aten::_scaled_dot_product_efficient_attention','aten::_scaled_dot_product_efficient_attention_backward'] if arm=='efficient' else ['aten::_scaled_dot_product_attention_math','aten::_softmax_backward_data']
    kernels=['fmha_cutlassF_f32_test','fmha_cutlassB_f32_test'] if arm=='efficient' else ['ampere_sgemm_test','cunn_SoftMaxBackward_test']
    return [{'name':n,'device_type':d,'duration_us':1.} for d,names in [('cpu',ops),('cuda',kernels)] for n in names]


@pytest.mark.parametrize('arm',['efficient','math'])
def test_real_device_types_and_backward_evidence_required(arm):
    rows=observations(arm);assert backend(rows,arm)['passed']
    no_backward=copy.deepcopy(rows[:-1]);assert not backend(no_backward,arm)['passed']
    for row in rows:row['device_type']='cpu'
    assert not backend(rows,arm)['passed']


def test_a_single_input_hash_change_is_preserved_as_a_difference():
    left={'operand_signatures':{'q':{'sha256':'old'},'k':{'sha256':'same'}}}
    right=copy.deepcopy(left);right['operand_signatures']['q']['sha256']='new'
    assert differences(left,right)==[{'path':'/operand_signatures/q/sha256','left':'old','right':'new'}]


@pytest.mark.parametrize('kernel',['void cunn_SoftMaxBackwardSmem<4, float>','void softmax_warp_backward<float, float, float, 10, false, false>'])
def test_math_backward_accepts_observed_large_and_small_row_cuda_variants(kernel):
    rows=observations('math');rows[-1]['name']=kernel
    assert backend(rows,'math')['passed']
    rows[-1]['device_type']='cpu'
    assert not backend(rows,'math')['passed']
