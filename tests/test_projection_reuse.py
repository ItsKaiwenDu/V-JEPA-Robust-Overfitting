"""Independent CPU equivalence/drift checks; no CUDA or released weights."""
import copy
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from attacks.pgd import native_objective, pgd_linf
from models.execution import apply_model_policy, configure_execution, model_execution_contract, precision_contract
from models.native_diagnostic import tiny_fixture
from models.precision_patch import PrecisionPatchEmbed3D, _PrecisionProjection
from models.projection_reuse import BankProjectionReuse


@pytest.fixture(autouse=True)
def restore_process_settings():
    """Do not leak a strict diagnostic policy into unrelated legacy fixtures."""
    before=precision_contract();rng=torch.get_rng_state().clone();threads=torch.get_num_threads()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(before['deterministic_algorithms'],warn_only=before['deterministic_warn_only'])
        torch.set_float32_matmul_precision(before['matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32=before['matmul_tf32']
        for name,key in [('enabled','cudnn_enabled'),('allow_tf32','cudnn_tf32'),('benchmark','cudnn_benchmark'),('deterministic','cudnn_deterministic')]:
            setattr(torch.backends.cudnn,name,before[key])
        if before['cublas_workspace_config'] is None:os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
        else:os.environ['CUBLAS_WORKSPACE_CONFIG']=before['cublas_workspace_config']
        torch.set_rng_state(rng);torch.set_num_threads(threads)
        assert precision_contract()==before


def layer(backend='native', bias=True):
    torch.manual_seed(52)
    original = SimpleNamespace(proj=nn.Conv3d(3, 5, (2, 2, 2), stride=(2, 2, 2), bias=bias),
                               patch_size=2, tubelet_size=2, training=False)
    result = PrecisionPatchEmbed3D(original, backend)
    result.eval()
    return result


@pytest.mark.parametrize('backend', ['native', 'cudnn'])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('draws', [4, 8])
def test_exact_tokens_strides_loss_and_all_vjps_with_separate_normalization(backend, bias, draws, monkeypatch):
    original, repeated = layer(backend, bias), layer(backend, bias)
    calls = []
    operation = _PrecisionProjection.forward
    def observed(*args):
        calls.append(1)
        return operation(*args)
    monkeypatch.setattr(_PrecisionProjection, 'forward', observed)
    pixels = torch.rand(1, 3, 4, 4, 4)
    weights = [torch.randn(1, 8, 5) for _ in range(draws)]
    mean, std = .37, .21
    def evaluate(patch, reuse):
        x = pixels.clone().requires_grad_()
        values, outputs = [], []
        def draw(fn, i):
            z = fn((x-mean)/std)
            outputs.append(z.detach())
            return (z.sin()*weights[i]).mean()
        if reuse:
            with BankProjectionReuse(patch, expected_calls=draws) as bank:
                values = [draw(bank, i) for i in range(draws)]
            assert bank.forward_computations == 1 and bank.requests == draws
            assert bank._tokens is bank._input is None
        else:
            values = [draw(patch, i) for i in range(draws)]
        loss = torch.stack(values).mean()
        gradients = torch.autograd.grad(loss, [x, *patch.parameters()])
        return outputs, loss, gradients
    baseline = evaluate(original, False);assert len(calls) == draws
    candidate = evaluate(repeated, True);assert len(calls) == draws+1
    for a,b in zip(baseline[0],candidate[0]):
        assert torch.equal(a,b) and a.stride() == b.stride()
    assert torch.equal(baseline[1],candidate[1])
    assert all(torch.equal(a,b) for a,b in zip(baseline[2],candidate[2]))
    assert all(p.grad is None for p in repeated.parameters())


@pytest.mark.parametrize('defect', ['input','signed_zero','weight','weight_data','bias','geometry','mode','autocast','global_precision','threads'])
def test_bank_rejects_changed_operands_or_precision_and_discards_cache(defect):
    patch=layer();x=torch.zeros(1,3,4,4,4)
    bank=BankProjectionReuse(patch,expected_calls=2)
    with pytest.raises(ValueError):
        with bank:
            bank(x)
            if defect=='input':x[0,0,0,0,0]=1
            elif defect=='signed_zero':x[0,0,0,0,0]=-0.0
            elif defect=='weight':
                with torch.no_grad():patch.proj.weight.add_(.1)
            elif defect=='weight_data':patch.proj.weight.data.add_(.1)
            elif defect=='bias':
                with torch.no_grad():patch.proj.bias.add_(.1)
            elif defect=='geometry':patch.proj.stride=(1,1,1)
            elif defect=='mode':patch.train()
            elif defect=='global_precision':torch.set_float32_matmul_precision('medium')
            elif defect=='threads':torch.set_num_threads(torch.get_num_threads()+1)
            elif defect=='autocast':
                with torch.autocast('cpu',dtype=torch.bfloat16):bank(x)
            bank(x)
    assert bank._tokens is bank._input is None


@pytest.mark.parametrize('count',[0,-1,True,1.0])
def test_reject_invalid_count(count):
    with pytest.raises(ValueError):BankProjectionReuse(layer(),expected_calls=count)


def test_bank_lifetime_call_counts_failure_cleanup_and_second_derivative_boundary():
    x=torch.randn(1,3,4,4,4,requires_grad=True);bank=BankProjectionReuse(layer(),expected_calls=1)
    with pytest.raises(ValueError):bank(x)
    with bank:y=bank(x).square().sum()
    with pytest.raises(ValueError):bank(x)
    with pytest.raises(ValueError):
        with bank:pass
    g,=torch.autograd.grad(y,x,create_graph=True)
    with pytest.raises(RuntimeError):torch.autograd.grad(g.sum(),x)
    with pytest.raises(ValueError):
        with BankProjectionReuse(layer(),expected_calls=2) as partial:partial(x)
    with pytest.raises(ValueError):
        with BankProjectionReuse(layer(),expected_calls=1) as extra:
            extra(x);extra(x)
    with pytest.raises(RuntimeError,match='original error'):
        with BankProjectionReuse(layer(),expected_calls=4) as failed:
            failed(x);raise RuntimeError('original error')
    assert failed._tokens is failed._input is None


class TinyOnlyAdapter(nn.Module):
    """Explicit test adapter: existing provenance must reject this class."""
    def __init__(self,original,reuse):
        super().__init__()
        self.proj=original.proj
        self.patch_size=original.patch_size;self.tubelet_size=original.tubelet_size
        self.reuse=reuse
    def forward(self,x,**kwargs):return self.reuse(x)


def tiny_objective(model,targets,banks,*,reuse,receipts):
    objective=native_objective(model,targets,banks)
    original=model.encoder.patch_embed
    if not reuse:return objective
    def call(x):
        with BankProjectionReuse(original,expected_calls=sum(map(len,banks))) as bank:
            model.encoder.patch_embed=TinyOnlyAdapter(original,bank)
            try:return objective(x)
            finally:
                model.encoder.patch_embed=original
                receipts.append(bank)
    return call


def test_actual_tiny_upstream_pgd20_r2_eot8_has_exact_adversaries_and_unmodified_rng(upstream):
    configure_execution('shared-fp64-native',device='cpu')
    torch.manual_seed(20261029)
    baseline,clean,masks_for=tiny_fixture(upstream)
    apply_model_policy(baseline,'shared-fp64-native');baseline.eval()
    candidate=copy.deepcopy(baseline)
    targets=baseline.clean_targets(clean)
    banks=[masks_for(seed) for seed in range(8)]
    before=torch.get_rng_state().clone();outputs=[];receipts=[]
    for model,reuse in [(baseline,False),(candidate,True)]:
        objective=tiny_objective(model,targets,banks,reuse=reuse,receipts=receipts)
        outputs.append(pgd_linf(objective,clean,epsilon=4/255,step_size=2/255,steps=20,restarts=2,seed=402))
        assert torch.equal(torch.get_rng_state(),before)
        assert all(p.grad is None for p in model.parameters())
        assert not targets.requires_grad
    assert torch.equal(outputs[0].adversarial,outputs[1].adversarial)
    assert torch.equal(outputs[0].losses,outputs[1].losses)
    assert len(receipts)==43 and all(v.requests==16 and v.forward_computations==1 for v in receipts)
    assert all(v._tokens is v._input is None for v in receipts)
    for name,value in baseline.state_dict().items():assert torch.equal(value,candidate.state_dict()[name])


def test_test_adapter_is_rejected_by_existing_model_provenance(upstream):
    configure_execution('shared-fp64-native',device='cpu')
    model,clean,_=tiny_fixture(upstream);apply_model_policy(model,'shared-fp64-native');model.eval()
    original=model.encoder.patch_embed
    with BankProjectionReuse(original,expected_calls=1) as bank:
        model.encoder.patch_embed=TinyOnlyAdapter(original,bank)
        try:
            with pytest.raises(ValueError,match='Actual shared-forward module'):
                model_execution_contract(model)
            bank(model.normalize(clean))
        finally:model.encoder.patch_embed=original
    assert model_execution_contract(model)['policy']=='shared-fp64-native'


@pytest.mark.parametrize('backend',['native','cudnn'])
def test_actual_tiny_upstream_all_input_and_parameter_derivatives_are_exact(upstream,backend):
    policy='shared-fp64-'+backend
    configure_execution(policy,device='cpu');torch.manual_seed(619)
    original,clean,masks_for=tiny_fixture(upstream)
    apply_model_policy(original,policy);original.eval();candidate=copy.deepcopy(original)
    targets=original.clean_targets(clean);banks=[masks_for(81+i) for i in range(8)]
    results=[]
    for model,reuse in [(original,False),(candidate,True)]:
        x=clean.clone().requires_grad_()
        fn=tiny_objective(model,targets,banks,reuse=reuse,receipts=[])
        loss=fn(x).sum()
        named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
        derivatives=torch.autograd.grad(loss,[x,*[p for _,p in named]],allow_unused=True)
        results.append((loss.detach(),dict(zip(['raw-pixels',*[name for name,_ in named]],derivatives))))
    assert torch.equal(results[0][0],results[1][0])
    assert results[0][1].keys()==results[1][1].keys()
    for name,expected in results[0][1].items():
        actual=results[1][1][name]
        if expected is None:assert actual is None
        else:assert torch.equal(actual,expected),name
    assert not targets.requires_grad and all(p.grad is None for p in original.target_encoder.parameters())


def test_two_update_synthetic_adamw_and_split_recovery_remain_exact():
    initial=layer();pixels=torch.rand(1,3,4,4,4)
    banks=[torch.rand(1,8,5) for _ in range(8)]
    saved_rng=torch.get_rng_state().clone()
    def optimizer(model):return torch.optim.AdamW(model.parameters(),lr=.002)
    def update(model,opt,reuse):
        opt.zero_grad(set_to_none=True)
        def loss(project):
            return torch.stack([(project((pixels-.4)/.2)-target).abs().mean() for target in banks]).mean()
        if reuse:
            with BankProjectionReuse(model,expected_calls=8) as bank:objective=loss(bank)
        else:objective=loss(model)
        objective.backward();opt.step();opt.zero_grad(set_to_none=True)
    reference=copy.deepcopy(initial);ref_opt=optimizer(reference)
    for _ in range(2):update(reference,ref_opt,False)
    continuous=copy.deepcopy(initial);con_opt=optimizer(continuous)
    update(continuous,con_opt,True)
    checkpoint=copy.deepcopy((continuous.state_dict(),con_opt.state_dict()))
    update(continuous,con_opt,True)
    resumed=copy.deepcopy(initial);res_opt=optimizer(resumed)
    resumed.load_state_dict(checkpoint[0]);res_opt.load_state_dict(checkpoint[1])
    update(resumed,res_opt,True)
    def equal(a,b):
        assert type(a) is type(b)
        if isinstance(a,torch.Tensor):assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for key in a:equal(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for left,right in zip(a,b):equal(left,right)
        else:assert a==b
    for model,opt in [(continuous,con_opt),(resumed,res_opt)]:
        equal(reference.state_dict(),model.state_dict());equal(ref_opt.state_dict(),opt.state_dict())
    assert torch.equal(saved_rng,torch.get_rng_state())


def test_failed_entry_releases_snapshots_and_never_opens_bank():
    bank=BankProjectionReuse(layer(),expected_calls=1)
    with pytest.raises(ValueError):
        with torch.autocast('cpu',dtype=torch.bfloat16):bank.__enter__()
    assert not bank._active and bank._parameter_copies is None


@pytest.mark.parametrize('defect',['nan','infinity','float64'])
def test_invalid_first_input_is_rejected_and_never_retained(defect):
    x=torch.zeros(1,3,4,4,4)
    if defect=='float64':x=x.double()
    else:x[0,0,0,0,0]=float('nan') if defect=='nan' else float('inf')
    bank=BankProjectionReuse(layer(),expected_calls=1)
    with pytest.raises(ValueError):
        with bank:bank(x)
    assert bank._input is bank._tokens is None and not bank._active
