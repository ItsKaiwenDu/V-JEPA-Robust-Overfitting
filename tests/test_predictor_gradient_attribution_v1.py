"""Whole-tensor saved analysis and non-replacing native observer proof."""
from pathlib import Path
from types import SimpleNamespace
import copy
from contextlib import contextmanager
import os
import json
import torch
import pytest

from models.predictor_gradient_attribution_v1 import (
    compare_saved, gradient_distribution, l1_cotangent, vjp_decomposition,
    LocalPredictionObserver, INPUT_BOUNDS, LOSS_BOUNDS)

ROOT = Path(__file__).resolve().parents[1]


def test_original_bounds_are_not_aggregate_acceptance_or_a_reverse_comparison():
    assert INPUT_BOUNDS == {'rtol': .002, 'atol': 2e-7}
    assert LOSS_BOUNDS == {'rtol': 1e-5, 'atol': 2e-6}
    ref = torch.ones((), dtype=torch.float32)
    actual = torch.tensor(1.002004)
    assert not compare_saved(actual, ref, kind='input')['passed']
    assert compare_saved(ref, actual, kind='input')['passed']


def test_near_absolute_threshold_uses_original_fp64_rule_on_stored_fp32_values():
    ref = torch.zeros((), dtype=torch.float32)
    above = torch.tensor(2e-7)
    below = torch.nextafter(above, ref)
    assert compare_saved(below, ref, kind='input')['passed']
    assert not compare_saved(above, ref, kind='input')['passed']


@pytest.mark.parametrize('fault', ['fp64','grad','nonfinite','empty','shape','loss_rank','kind'])
def test_invalid_or_computed_or_wrong_domain_values_refuse(fault):
    a = torch.ones(()); b = torch.ones(())
    kind = 'loss'
    if fault == 'fp64': a = a.double()
    elif fault == 'grad': a.requires_grad_()
    elif fault == 'nonfinite': a = torch.tensor(float('nan'))
    elif fault == 'empty': a = b = torch.empty(0); kind='input'
    elif fault == 'shape': a = torch.ones(2); kind='input'
    elif fault == 'loss_rank': a = b = torch.ones(1)
    else: kind='candidate'
    with pytest.raises(ValueError): compare_saved(a,b,kind=kind)


def test_whole_tensor_partitions_include_bin_edges_zero_and_all_frames_channels():
    base = torch.tensor([0.,1e-7,-1e-6,1e-5,-1e-4,1e-3,1e-2,-1e-1,1.])
    ref = base.repeat(2).reshape(1,3,2,1,3)
    actual = ref.clone(); actual[0,0,0,0,0]=1e-8; actual[0,0,1,0,0]*=-1
    r = gradient_distribution(actual,ref)
    assert len(r['per_frame_channel']) == 6
    assert sum(x['elements'] for x in r['reference_magnitude_strata']) == ref.numel()
    assert sum(x['violations'] for x in r['reference_magnitude_strata']) == r['gate']['violations']
    assert sum(x['sign_disagreements'] for x in r['per_frame_channel']) == 2
    assert r['opposite_nonzero_signs']==1 and r['one_zero_sign_changes']==1
    assert r['sign_disagreements_outside_original_bound']==1
    assert not r['root_cause_established'] and r['sign_comparison_is_not_attack_validation']


@pytest.mark.parametrize('shape', [(3,2,2),(2,3,2,2,2),(1,2,2,2,2)])
def test_invalid_gradient_geometry_refuses(shape):
    t=torch.zeros(shape)
    with pytest.raises(ValueError):gradient_distribution(t,t)


def test_equal_fp32_l1_loss_can_hide_changed_output_cotangents():
    # Explicit synthetic counterexample only; not evidence of the target cause.
    eps = 1e-8
    original=torch.tensor([[[eps,1.]]]); candidate=torch.tensor([[[-eps,1.]]]); target=torch.zeros_like(original)
    assert torch.equal((original-target).abs().mean(), (candidate-target).abs().mean())
    co=l1_cotangent(original,target); cm=l1_cotangent(candidate,target)
    assert co[0,0,0] == -cm[0,0,0] and co[0,0,1] == cm[0,0,1]


def test_equal_predictions_and_cotangents_do_not_imply_equal_jacobian_vjp():
    x=torch.zeros(2,requires_grad=True)
    y0=x+1; y1=2*x+1
    c0=l1_cotangent(y0.detach().reshape(1,1,2),torch.zeros(1,1,2)).reshape(2)
    c1=l1_cotangent(y1.detach().reshape(1,1,2),torch.zeros(1,1,2)).reshape(2)
    assert torch.equal(y0,y1) and torch.equal(c0,c1)
    g0,=torch.autograd.grad(y0,x,grad_outputs=c0)
    g1,=torch.autograd.grad(y1,x,grad_outputs=c1)
    assert torch.equal(g1,2*g0) and not torch.equal(g0,g1)


@pytest.mark.parametrize('fault',['shape','draws','families','batch'])
def test_cotangent_reconstruction_refuses_changed_native_domain(fault):
    p=torch.zeros(1,2,3); target=p.clone(); kwargs={}
    if fault=='shape':target=torch.zeros(1,3,3)
    elif fault=='draws':kwargs['draws']=1
    elif fault=='families':kwargs['families']=1
    else:p=target=torch.zeros(2,2,3)
    with pytest.raises(ValueError):l1_cotangent(p,target,**kwargs)


def test_zero_l1_residual_has_zero_cotangent():
    t=torch.zeros(1,3,4)
    assert torch.equal(l1_cotangent(t,t),t)


def test_four_saved_vjp_cells_reconstruct_both_orders_and_retain_interaction():
    x=torch.tensor([.5,-.3],requires_grad=True)
    c0=torch.tensor([1.,-2.]); c1=torch.tensor([-3.,4.])
    matrices=(torch.tensor([[1.,2.],[3.,-1.]]),torch.tensor([[2.,1.],[-1.,4.]]))
    values={}
    for i,m in enumerate(matrices):
        for j,c in enumerate((c0,c1)):
            g,=torch.autograd.grad(m@x,x,grad_outputs=c);values[i,j]=g.detach()
    r=vjp_decomposition(values[0,0],values[1,0],values[0,1],values[1,1])
    assert r['path_first_reconstruction_error']==r['cotangent_first_reconstruction_error']==0
    assert torch.count_nonzero(r['interaction']) and not r['pure_backward_cause_established']
    assert torch.equal(r['total_delta'],(values[1,1]-values[0,0]).double())


def test_vjp_cells_refuse_shape_or_autograd_recomputation_inputs():
    t=torch.ones(2)
    with pytest.raises(ValueError):vjp_decomposition(t,t,t,t[:1])
    with pytest.raises(ValueError):vjp_decomposition(t.requires_grad_(),t,t,t)


@contextmanager
def local_settings():
    # Test-scoped process settings only; no target execution policy is changed.
    from models.execution import precision_contract
    before=precision_contract();rng=torch.get_rng_state().clone();threads=torch.get_num_threads()
    try:yield
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


@pytest.fixture(scope='module')
def native_control(tmp_path_factory):
    from attacks.pgd import native_objective
    from models.candidate_evaluation import EvaluationSettings,seeds_for
    from models.native_diagnostic import tiny_fixture
    from models.execution import configure_execution, apply_model_policy
    from models.fixture_state import load_fixture_state
    from models.diagnostics import state_digest
    from models.predictor_eot8_owned_v1 import PredictorRoute
    with local_settings():
        configure_execution('shared-fp64-native',device='cpu'); torch.set_num_threads(4)
        torch.manual_seed(20261029)
        model,clean,masks_for=tiny_fixture(ROOT/'external/vjepa2')
        old=ROOT/'runs/pretrained/diagnostic/predictor-eot8-10603-v1-20261105/cpu-four-worker'
        load_fixture_state(model,old/'initial.pt','989665a5a4f8ea90600214be5536b4600c37fc448108a5bd177b3416e5ae0338')
        apply_model_policy(model,'shared-fp64-native'); model.eval()
        target=model.clean_targets(clean)
        seeds=seeds_for(EvaluationSettings.for_method('primary',2.),'tiny-predictor-eot8')['attack']
        banks=[masks_for(seed%(2**31-1)) for seed in seeds]
        objective=native_objective(model,target,banks); state=state_digest(model)
        versions=tuple((id(t),t._version) for t in (*model.parameters(),*model.buffers()))
        hooks=len(model.predictor._forward_hooks)
        captures=[]; values=[]; scopes=[]
        for observed in (False,True,False):
            x=clean.clone().requires_grad_(True)
            with PredictorRoute(model,'original',tiny=True) as route:
                route.phase='EOT8-before'
                if observed:
                    with LocalPredictionObserver(model,target,banks) as capture:
                        loss=objective(x).sum(); gradient,=route.input_gradient(loss,x)
                    captures=capture.records
                else:loss=objective(x).sum(); gradient,=route.input_gradient(loss,x)
                values.append({'loss':loss.detach().clone(),'input':gradient.detach().clone()})
                scopes.append(copy.deepcopy(route.scopes))
            assert state_digest(model)==state
            assert tuple((id(t),t._version) for t in (*model.parameters(),*model.buffers()))==versions
            assert not any(p.grad is not None for p in model.parameters())
            assert len(model.predictor._forward_hooks)==hooks
        proof=tmp_path_factory.mktemp('native-observer-proof')
        torch.save({'values':values,'captures':captures,'scopes':scopes,'targets':target.detach().clone(),
            'banks':[[{'context':pair.context.clone(),'target':pair.target.clone()} for pair in draw] for draw in banks]},proof/'values.pt')
        (proof/'proof.json').write_text(json.dumps({'kind':'local-tiny-native-observer-proof-v1',
            'CPU_only':True,'pretrained_or_GPU_qualified':False,'weights_before_and_after':state,
            'model_tensor_versions_unchanged':True,'parameter_gradients_absent':True,
            'module_hooks_restored':True,'native_groups':16,'saved_returns':6,
            'source_fixture_sha256':'989665a5a4f8ea90600214be5536b4600c37fc448108a5bd177b3416e5ae0338'},sort_keys=True))
        yield SimpleNamespace(values=values,captures=captures,scopes=scopes,model=model,targets=target,banks=banks,old=old,hooks=hooks)


def test_actual_native_observer_preserves_before_during_after_loss_and_input(native_control):
    n=native_control
    for kind in ('loss','input'):
        assert torch.equal(n.values[0][kind],n.values[1][kind]) and torch.equal(n.values[1][kind],n.values[2][kind])
        old_worker=json.loads((n.old/'0/worker.json').read_text())
        old=torch.load(n.old/'0'/old_worker['artifacts']['EOT8-before-'+kind]['file'],weights_only=True,map_location='cpu')
        assert torch.equal(n.values[1][kind],old)
    assert all(len(s)==1 and s[0]['state']=='complete' and s[0]['hook_removed'] for s in n.scopes)


def test_actual_native_all_sixteen_cotangents_match_original_l1_weighting(native_control):
    rows=native_control.captures
    assert len(rows)==16 and {(r['draw'],r['family']) for r in rows}=={(d,f) for d in range(8) for f in range(2)}
    for r in rows:
        assert torch.equal(r['cotangent'],l1_cotangent(r['prediction'],r['selected_targets']))
        assert not r['prediction'].requires_grad and not r['cotangent'].requires_grad


def test_observer_failure_preserves_primary_error_and_removes_hooks(native_control):
    n=native_control
    with pytest.raises(RuntimeError,match='original-test-failure'):
        with LocalPredictionObserver(n.model,n.targets,n.banks) as capture:
            raise RuntimeError('original-test-failure')
    assert not capture.active and capture.closed and len(n.model.predictor._forward_hooks)==n.hooks
    with pytest.raises(ValueError,match='Single-use'):capture.__enter__()


def test_incomplete_capture_refuses_after_hook_cleanup(native_control):
    n=native_control
    with pytest.raises(ValueError,match='All sixteen'):
        with LocalPredictionObserver(n.model,n.targets,n.banks):pass
    assert len(n.model.predictor._forward_hooks)==n.hooks


def test_university_and_non_tiny_observer_domains_refuse(native_control,monkeypatch):
    n=native_control
    with pytest.raises(ValueError):LocalPredictionObserver(n.model,torch.zeros(1,2048,1024),n.banks)
    monkeypatch.setattr('models.predictor_gradient_attribution_v1.platform.node',lambda:'linux10603')
    with pytest.raises(ValueError,match='no university'):LocalPredictionObserver(n.model,n.targets,n.banks)


def test_saved_analysis_cannot_dispatch_model_backward_sdpa_or_cuda(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('Saved-only analysis dispatched new work.')
    for obj,name in ((torch.nn.Module,'__call__'),(torch.autograd,'grad'),(torch.nn.functional,'scaled_dot_product_attention'),(torch.cuda,'init')):
        monkeypatch.setattr(obj,name,forbidden)
    r=gradient_distribution(torch.zeros(1,3,2,2,2),torch.zeros(1,3,2,2,2))
    assert r['gate']['passed'] and r['gate']['exact'] and not r['root_cause_established']


def test_native_captured_targets_and_masks_match_the_frozen_cpu_reference(native_control):
    from models.cuda_profile import tensor_signature
    n=native_control
    previous=json.loads((n.old/'0/worker.json').read_text())['baseline']
    assert tensor_signature(n.targets)==previous['targets']
    for row in n.captures:
        pair=n.banks[row['draw']][row['family']]
        expected=previous['masks'][row['draw']][row['family']]
        assert [tensor_signature(pair.context),tensor_signature(pair.target)]==expected
        selected=n.targets.gather(1,pair.target.unsqueeze(-1).expand(-1,-1,128))
        assert torch.equal(selected,row['selected_targets'])
