"""Actual local backward interventions, native preservation and sparse integrity."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import os
import numpy as np
import pytest
import torch
from models.predictor_cotangent_attribution_v1 import (
    build_bundle, validate_bundle, replay_pair, digest_array, LocalCotangentReplay)
from models.predictor_gradient_attribution_v1 import LocalPredictionObserver, vjp_decomposition

ROOT = Path(__file__).resolve().parents[1]


def masks():
    from models.vjepa2 import MaskPair
    pair = MaskPair(torch.arange(16).reshape(1,16), torch.arange(16,32).reshape(1,16))
    return [[pair,pair] for _ in range(8)]


def mask_values(banks):
    return [[{'context': p.context.tolist(), 'target': p.target.tolist()} for p in draw] for draw in banks]


class ToyPredictor(torch.nn.Module):
    def __init__(self, path):
        super().__init__(); self.path = path
    def forward(self, features, context, target, *, mask_index):
        offset = torch.ones((1,target.shape[1],128), dtype=torch.float32)
        offset[0,0,0] = 1e-8 if self.path == 0 else -1e-8
        return features[:,target[0],:]*(self.path+1)+offset


class ToyModel(torch.nn.Module):
    def __init__(self, path):
        super().__init__(); self.predictor = ToyPredictor(path)
    def loss_per_example(self, x, targets, bank):
        losses=[]
        for pair in bank:
            p=self.predictor(x,pair.context,pair.target,mask_index=0)
            losses.append((p-targets[:,pair.target[0],:]).abs().mean(dim=(1,2)))
        return torch.stack(losses).mean(0)


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    from attacks.pgd import native_objective
    target=torch.zeros(1,32,128); banks=masks(); captures=[]; native=[]
    for path in (0,1):
        model=ToyModel(path);x=torch.zeros_like(target,requires_grad=True)
        with LocalPredictionObserver(model,target,banks) as capture:
            loss=native_objective(model,target,banks)(x).sum()
            g,=torch.autograd.grad(loss,x)
        captures.append([{'prediction':r['prediction'].numpy(),'cotangent':r['cotangent'].numpy()} for r in capture.records])
        native.append(g.detach())
    bundle=build_bundle(*captures,target.numpy(),mask_values(banks),baseline_sha256='a'*64,domain='local-tiny-CPU')
    directory=tmp_path_factory.mktemp('cotangent-toy-proof')
    (directory/'bundle.json').write_text(json.dumps(bundle,sort_keys=True))
    cells={}; proofs=[]
    for path,cotangent in ((0,0),(1,1),(0,1),(1,0)):
        model=ToyModel(path);x=torch.zeros_like(target,requires_grad=True)
        replay=LocalCotangentReplay(model,target,banks,bundle,path=path,cotangent=cotangent)
        with replay:
            loss=native_objective(model,target,banks)(x).sum()
            g,=torch.autograd.grad(loss,x)
        cells[path,cotangent]=g.detach()
        proofs.append({'path':path,'cotangent':cotangent,'restored':replay.restored,'rows':replay.rows,
            'loss':loss.detach(),'gradient':g.detach()})
        assert not model.predictor._forward_hooks
    torch.save({'cells':cells,'native':native,'proofs':proofs},directory/'actual.pt')
    return SimpleNamespace(bundle=bundle,captures=captures,target=target,banks=banks,cells=cells,native=native,proofs=proofs,directory=directory)


def test_actual_four_cells_match_explicit_jacobian_and_saved_diagonals(toy):
    for path in (0,1):
        assert torch.equal(toy.cells[path,path],toy.native[path])
        for cotangent in (0,1):
            expected=torch.zeros(1,32,128,dtype=torch.float32)
            for draw in toy.banks:
                for pair in draw:
                    c=torch.ones(1,16,128)/32768
                    c[0,0,0] *= 1 if cotangent==0 else -1
                    expected[:,pair.target[0],:] += (path+1)*c
            assert torch.equal(toy.cells[path,cotangent],expected)
    for proof in toy.proofs:
        assert proof['restored'] and len(proof['rows'])==16
        assert all(r['hook_removed'] for r in proof['rows'])
        for k,row in enumerate(proof['rows']):
            assert digest_array(row['native'].numpy())==toy.bundle['pairs'][k]['arms'][proof['path']]['cotangent_sha256']
            assert digest_array(row['applied'].numpy())==toy.bundle['pairs'][k]['arms'][proof['cotangent']]['cotangent_sha256']


def test_actual_common_cotangent_decomposition_retains_both_orders_and_interaction(toy):
    c=toy.cells;r=vjp_decomposition(c[0,0],c[1,0],c[0,1],c[1,1])
    assert r['path_first_reconstruction_error']==r['cotangent_first_reconstruction_error']==0
    assert torch.count_nonzero(r['interaction']) and not r['pure_backward_cause_established']
    assert torch.equal(toy.proofs[0]['loss'],toy.proofs[1]['loss'])
    assert not torch.equal(c[0,0],c[0,1]) and not torch.equal(c[0,0],c[1,0])


def test_sparse_patch_preserves_incoming_bytes_and_roundtrips_exactly(toy):
    for k,row in enumerate(toy.bundle['pairs']):
        native=toy.captures[0][k]['cotangent'].copy();before=native.tobytes()
        patched=replay_pair(native,row,0,1)
        assert native.tobytes()==before and not np.shares_memory(native,patched)
        assert digest_array(patched)==row['arms'][1]['cotangent_sha256']
        assert np.array_equal(replay_pair(patched,row,1,0).view(np.uint32),native.view(np.uint32))
    assert toy.bundle['changed_entries']==16 and not toy.bundle['authorized']


@pytest.mark.parametrize('fault',['entry_value','unlisted_value','source_hash','destination_hash','missing_edit'])
def test_changed_incoming_tensor_or_incomplete_patch_refuses(toy,fault):
    row=copy.deepcopy(toy.bundle['pairs'][0]);a=toy.captures[0][0]['cotangent'].copy()
    if fault=='entry_value':a[0,0,0]=0
    elif fault=='unlisted_value':a[0,1,1]=0
    elif fault=='source_hash':row['arms'][0]['cotangent_sha256']='f'*64
    elif fault=='destination_hash':row['arms'][1]['cotangent_sha256']='f'*64
    else:row['edits']=[]
    with pytest.raises(ValueError):replay_pair(a,row,0,1)


@pytest.mark.parametrize('fault',['authorized','target_permission','research','missing_pair','order','shape','dtype_geometry',
    'duplicate','coordinate','equal_bits','nonfinite_bits','bad_hash','count','extra_field'])
def test_inert_bundle_rejects_expansion_relabelling_or_incomplete_integrity(toy,fault):
    b=copy.deepcopy(toy.bundle);row=b['pairs'][0]
    if fault=='authorized':b['authorized']=True
    elif fault=='target_permission':b['new_target_execution_permitted']=True
    elif fault=='research':b['research_admitted']=True
    elif fault=='missing_pair':b['pairs'].pop()
    elif fault=='order':row['draw']=1
    elif fault=='shape':row['shape'][0]=2
    elif fault=='dtype_geometry':row['shape'][2]=1024
    elif fault=='duplicate':row['edits'].append(copy.deepcopy(row['edits'][0]));b['changed_entries']+=1
    elif fault=='coordinate':row['edits'][0]['coordinate'][1]=32
    elif fault=='equal_bits':row['edits'][0]['bits'][1]=row['edits'][0]['bits'][0]
    elif fault=='nonfinite_bits':row['edits'][0]['bits'][1]='7f800000'
    elif fault=='bad_hash':row['arms'][0]['cotangent_sha256']='unknown'
    elif fault=='count':b['changed_entries']+=1
    else:b['attempts']=2
    with pytest.raises(ValueError):validate_bundle(b)


@pytest.mark.parametrize('fault',['cotangent','prediction_shape','target_dtype','mask_duplicate','mask_overlap','baseline_hash','domain'])
def test_bundle_requires_actual_l1_tail_and_original_geometry(toy,fault):
    a=copy.deepcopy(toy.captures[0]);b=copy.deepcopy(toy.captures[1]);t=toy.target.numpy().copy();m=mask_values(toy.banks)
    baseline='a'*64;domain='local-tiny-CPU'
    if fault=='cotangent':b[0]['cotangent'][0,0,0]=0
    elif fault=='prediction_shape':a[0]['prediction']=a[0]['prediction'][:,:1]
    elif fault=='target_dtype':t=t.astype(np.float64)
    elif fault=='mask_duplicate':m[0][0]['target'][0][1]=m[0][0]['target'][0][0]
    elif fault=='mask_overlap':m[0][0]['context'][0][0]=16
    elif fault=='baseline_hash':baseline='bad'
    else:domain='university-live'
    with pytest.raises(ValueError):build_bundle(a,b,t,m,baseline_sha256=baseline,domain=domain)


@pytest.mark.parametrize('fault',['university','target_domain','mask','target','path_boolean','cotangent_boolean','existing_hook'])
def test_prototype_refuses_target_or_changed_local_domain(toy,monkeypatch,fault):
    from models import predictor_cotangent_attribution_v1 as module
    model=ToyModel(0);b=copy.deepcopy(toy.bundle);banks=copy.deepcopy(toy.banks);target=toy.target.clone();path=cot=0;handle=None
    if fault=='university':monkeypatch.setattr(module.platform,'node',lambda:'linux10603')
    elif fault=='target_domain':b['domain']='target-saved-only'
    elif fault=='mask':banks[0][0].target[0,0]=31
    elif fault=='target':target[0,0,0]=1
    elif fault=='path_boolean':path=True
    elif fault=='cotangent_boolean':cot=True
    else:handle=model.predictor.register_forward_hook(lambda *a:None)
    try:
        with pytest.raises(ValueError):LocalCotangentReplay(model,target,banks,b,path=path,cotangent=cot)
    finally:
        if handle:handle.remove()


def test_primary_backward_failure_removes_all_handles_and_preserves_error(toy):
    from attacks.pgd import native_objective
    model=ToyModel(0);x=torch.zeros_like(toy.target,requires_grad=True)
    replay=LocalCotangentReplay(model,toy.target,toy.banks,toy.bundle,path=0,cotangent=1)
    with pytest.raises(RuntimeError,match='primary-local-error'):
        with replay:
            native_objective(model,toy.target,toy.banks)(x)
            raise RuntimeError('primary-local-error')
    assert replay.restored and not model.predictor._forward_hooks
    assert all(row['hook_removed'] and row['native'] is None for row in replay.rows)
    with pytest.raises(ValueError):replay.__enter__()


def test_forward_identity_failure_cleans_up_and_incomplete_use_refuses(toy):
    from attacks.pgd import native_objective
    model=ToyModel(1);replay=LocalCotangentReplay(model,toy.target,toy.banks,toy.bundle,path=0,cotangent=1)
    with pytest.raises(ValueError,match='prediction'):
        with replay:native_objective(model,toy.target,toy.banks)(torch.zeros_like(toy.target,requires_grad=True))
    assert replay.restored and not model.predictor._forward_hooks
    replay=LocalCotangentReplay(ToyModel(0),toy.target,toy.banks,toy.bundle,path=0,cotangent=1)
    with pytest.raises(ValueError,match='Incomplete'):
        with replay:pass
    assert replay.restored


@contextmanager
def settings():
    from models.execution import precision_contract
    p=precision_contract();rng=torch.get_rng_state().clone();threads=torch.get_num_threads()
    try:yield
    finally:
        torch.use_deterministic_algorithms(p['deterministic_algorithms'],warn_only=p['deterministic_warn_only'])
        torch.set_float32_matmul_precision(p['matmul_precision']);torch.backends.cuda.matmul.allow_tf32=p['matmul_tf32']
        for name,key in [('enabled','cudnn_enabled'),('allow_tf32','cudnn_tf32'),('benchmark','cudnn_benchmark'),('deterministic','cudnn_deterministic')]:setattr(torch.backends.cudnn,name,p[key])
        if p['cublas_workspace_config'] is None:os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
        else:os.environ['CUBLAS_WORKSPACE_CONFIG']=p['cublas_workspace_config']
        torch.set_rng_state(rng);torch.set_num_threads(threads)
        assert precision_contract()==p


def test_real_native_tiny_model_keeps_diagonal_and_cross_matches_independent_vjp(tmp_path):
    from attacks.pgd import native_objective
    from models.native_diagnostic import tiny_fixture
    from models.fixture_state import load_fixture_state
    from models.execution import configure_execution,apply_model_policy
    from models.candidate_evaluation import EvaluationSettings,seeds_for
    from models.diagnostics import state_digest
    from models.predictor_eot8_owned_v1 import PredictorRoute
    from models.predictor_gradient_attribution_v1 import compare_saved
    with settings():
        configure_execution('shared-fp64-native',device='cpu');torch.set_num_threads(4);torch.manual_seed(20261029)
        model,clean,masks_for=tiny_fixture(ROOT/'external/vjepa2')
        old=ROOT/'runs/pretrained/diagnostic/predictor-eot8-10603-v1-20261105/cpu-four-worker'
        load_fixture_state(model,old/'initial.pt','989665a5a4f8ea90600214be5536b4600c37fc448108a5bd177b3416e5ae0338')
        apply_model_policy(model,'shared-fp64-native');model.eval();target=model.clean_targets(clean)
        banks=[masks_for(seed%(2**31-1)) for seed in seeds_for(EvaluationSettings.for_method('primary',2.),'tiny-predictor-eot8')['attack']]
        objective=native_objective(model,target,banks);state=state_digest(model)
        versions=tuple((id(t),t._version) for t in (*model.parameters(),*model.buffers()))
        def native():
            x=clean.clone().requires_grad_(True)
            with PredictorRoute(model,'original',tiny=True) as route:
                route.phase='EOT8-before'
                with LocalPredictionObserver(model,target,banks) as capture:
                    loss=objective(x).sum();g,=route.input_gradient(loss,x)
            return loss.detach(),g.detach(),capture
        loss0,g0,capture=native()
        a=[{'prediction':r['prediction'].numpy(),'cotangent':r['cotangent'].numpy()} for r in capture.records]
        hypothetical=copy.deepcopy(a);ids=banks[0][0].target[0]
        selected=target[:,ids,:].numpy();p=hypothetical[0]['prediction']
        p[0,0,0]=selected[0,0,0]-(a[0]['prediction'][0,0,0]-selected[0,0,0])
        scale=np.float32(1)/np.float32(8)/np.float32(2)/np.float32(p.size)
        hypothetical[0]['cotangent']=np.sign(p-selected)*scale
        bundle=build_bundle(a,hypothetical,target.numpy(),mask_values(banks),baseline_sha256=state,domain='local-tiny-CPU')
        assert bundle['changed_entries']==1
        actual=[];receipts=[]
        for cot in (0,1):
            x=clean.clone().requires_grad_(True)
            with PredictorRoute(model,'original',tiny=True) as route:
                route.phase='EOT8-before'
                with LocalCotangentReplay(model,target,banks,bundle,path=0,cotangent=cot) as replay:
                    loss=objective(x).sum();g,=route.input_gradient(loss,x)
            assert replay.restored and route.scopes[0]['state']=='complete' and route.scopes[0]['hook_removed']
            assert torch.equal(loss.detach(),loss0)
            actual.append(g.detach());receipts.append(replay.rows)
        assert torch.equal(actual[0],g0) and not torch.equal(actual[1],g0)
        outputs=[]
        hook=model.predictor.register_forward_hook(lambda mod,args,out:outputs.append(out))
        try:
            x=clean.clone().requires_grad_(True);objective(x)
            reference,=torch.autograd.grad(tuple(outputs),x,grad_outputs=tuple(torch.from_numpy(r['cotangent']) for r in hypothetical))
        finally:hook.remove()
        check=compare_saved(actual[1],reference.detach(),kind='input');assert check['passed']
        loss2,g2,_=native();assert torch.equal(loss0,loss2) and torch.equal(g0,g2)
        assert state_digest(model)==state and tuple((id(t),t._version) for t in (*model.parameters(),*model.buffers()))==versions
        assert all(p.grad is None for p in model.parameters()) and not model.predictor._forward_hooks
        frozen=json.loads((old/'0/worker.json').read_text())
        assert torch.equal(g0,torch.load(old/'0'/frozen['artifacts']['EOT8-before-input']['file'],weights_only=True,map_location='cpu'))
        torch.save({'native_before':g0,'native_diagonal':actual[0],'synthetic_cotangent_cross':actual[1],
            'direct_multi_output_vjp':reference.detach(),'native_after':g2,'native_loss_before':loss0,'native_loss_after':loss2,
            'actual_received_and_applied':receipts},tmp_path/'actual-native.pt')
        (tmp_path/'proof.json').write_text(json.dumps({'local_CPU_only':True,'pretrained_or_GPU_qualified':False,
            'other_cotangent_is_a_deliberate_synthetic_hypothesis':True,'changed_entries':bundle['changed_entries'],
            'native_diagonal_and_after_bitwise_exact':True,'cross_vs_direct_vjp':check,
            'model_versions_and_values_unchanged':True,'parameter_gradients_absent':True,'hooks_restored':True},sort_keys=True))
