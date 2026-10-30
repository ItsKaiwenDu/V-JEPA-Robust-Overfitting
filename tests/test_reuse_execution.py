"""Diagnostic identity, restoration and derivative checks on real tiny CPU models."""
import copy
import pytest
import torch

from models.execution import apply_model_policy,configure_execution,model_execution_contract
from models.native_diagnostic import tiny_fixture
from models.reuse_execution import KIND,MODES,DiagnosticProjectionBank,diagnostic_bank,diagnostic_objective
from test_projection_reuse import restore_process_settings


@pytest.fixture
def case(upstream):
    configure_execution('shared-fp64-native',device='cpu');torch.manual_seed(620)
    model,pixels,masks_for=tiny_fixture(upstream)
    apply_model_policy(model,'shared-fp64-native');model.eval()
    return model,pixels,model.clean_targets(pixels),[masks_for(17+i) for i in range(8)]


@pytest.mark.parametrize('mode',MODES)
def test_separate_adapter_rejects_legacy_provenance_then_restores_identity(case,mode):
    model,pixels,targets,banks=case
    patch=model.encoder.patch_embed;before=model_execution_contract(model)
    with diagnostic_bank(model,mode,2,'cpu') as bank:
        with pytest.raises(ValueError,match='Unknown model execution policy'):model_execution_contract(model)
        x=model.normalize(pixels)
        a,b=model.encoder.patch_embed(x),model.encoder.patch_embed(x)
        assert torch.equal(a,b)
    assert bank.requests==2 and bank.forward_computations==(1 if mode=='reuse' else 2)
    assert model.encoder.patch_embed is patch and model_execution_contract(model)==before


def test_all_tiny_input_parameter_gradients_and_draw_counts_match(case):
    model,pixels,targets,banks=case;reference=copy.deepcopy(model);results=[]
    for current,mode in [(reference,'recompute'),(model,'reuse')]:
        receipts=[];objective=diagnostic_objective(current,targets,banks,mode=mode,receipts=receipts)
        x=pixels.clone().requires_grad_();loss=objective(x).sum()
        named=[(n,p) for n,p in current.named_parameters() if p.requires_grad]
        values=torch.autograd.grad(loss,[x,*[p for _,p in named]],allow_unused=True)
        results.append((loss,values));assert receipts==[{'requests':16,'forward_computations':1 if mode=='reuse' else 16}]
    assert torch.equal(results[0][0],results[1][0])
    for a,b in zip(results[0][1],results[1][1]):
        if a is None:assert b is None
        else:assert torch.equal(a,b)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('mode',MODES)
def test_failure_restores_module_policy_and_keeps_original_exception(case,mode):
    model,_,_,_=case;patch=model.encoder.patch_embed;before=model_execution_contract(model)
    with pytest.raises(RuntimeError,match='deliberate'):
        with diagnostic_bank(model,mode,16,'cpu'):raise RuntimeError('deliberate')
    assert model.encoder.patch_embed is patch and model_execution_contract(model)==before


@pytest.mark.parametrize('defect',['kind','device','count','mode'])
def test_explicit_scope_device_and_count_required(case,defect):
    model,pixels,_,_=case
    kwargs={'expected_calls':2,'device':'cpu','diagnostic_kind':KIND}
    if defect=='kind':kwargs['diagnostic_kind']='study'
    elif defect=='device':kwargs['device']='cuda'
    elif defect=='count':kwargs['expected_calls']=True
    else:model.encoder.patch_embed.backward_backend='cudnn'
    with pytest.raises(ValueError):DiagnosticProjectionBank(model.encoder.patch_embed,**kwargs)
