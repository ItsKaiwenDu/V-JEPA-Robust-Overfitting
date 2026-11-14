"""Actual small upstream transformer integration, without target or CUDA work."""
import copy
import hashlib
import json
import os
from pathlib import Path
import threading

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from models import predictor_forward_preserving_upstream_v1 as integration
from models import predictor_eot8_owned_v1 as frozen
from models import predictor_eot8_adapter_v1 as adapter
from models.native_diagnostic import tiny_fixture
from models.execution import configure_execution, apply_model_policy
from models.diagnostics import state_digest
from models.attention_saved_evidence import agreement

ROOT=Path(__file__).resolve().parents[1]
BOUND={'rtol':.002,'atol':2e-7}


@pytest.fixture
def fixture():
    configure_execution('shared-fp64-native',device='cpu')
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261114)
        model,pixels,masks=tiny_fixture(ROOT/'external/vjepa2')
    apply_model_policy(model,'shared-fp64-native');model.eval()
    return model,pixels,masks


def other_thread(task):
    result=[];errors=[]
    def run():
        try:result.append(task.run())
        except BaseException as e:errors.append(e)
    worker=threading.Thread(target=run);worker.start();worker.join(20)
    assert not worker.is_alive(),'Local native thread timed out.'
    if errors:raise errors[0]
    return result[0]


def restored(model):
    assert not hasattr(model,integration.MARKER)
    assert all('forward' not in b.attn.__dict__ for blocks in (model.encoder.blocks,model.predictor.predictor_blocks) for b in blocks)
    assert not adapter._ACTIVE
    assert not integration.cpu._ACTIVE
    assert model._diagnostic_execution_policy=='shared-fp64-native'


def calculate(fixture,seed,*,candidate=False,threaded=False):
    model,pixels,masks_for=fixture;targets=model.clean_targets(pixels);masks=masks_for(seed)
    value=pixels.detach().clone().requires_grad_(True)
    initial=state_digest(model);versions=[p._version for p in model.parameters()]
    input_version=value._version
    arrays={'targets':targets.detach().clone(),'pixels':pixels.clone()};handles=[];predictions=[]
    def observe(module,args,output):
        index=len(predictions);predictions.append(index)
        arrays['prediction-'+str(index)]=output.detach().clone()
        selected=targets.gather(1,masks[index].target.unsqueeze(-1).expand(-1,-1,128))
        arrays['selected-target-'+str(index)]=selected.detach().clone()
        def incoming(g):
            arrays['cotangent-'+str(index)]=g.detach().clone()
            return None
        handles.append(output.register_hook(incoming))
        return None
    handles.append(model.predictor.register_forward_hook(observe))
    route=integration.LocalUpstreamRoute(model) if candidate else frozen.PredictorRoute(model,'original',tiny=True)
    try:
        with route:
            if not candidate:route.phase='draw-before'
            if candidate and threaded:route._run_native=other_thread
            loss=model.loss_per_example(value,targets,masks).sum()
            arrays['loss']=loss.detach().clone()
            if candidate:
                grad=route.gradient(loss,value)
                native=route.native
                proto=route.prototype
                proof={'identity':integration.IDENTITY,'native_scopes':copy.deepcopy(native.scopes),
                       'native_calls':copy.deepcopy(native.rows),'native_refusals':copy.deepcopy(native.refusals),
                       'prototype_calls':copy.deepcopy(proto.calls),'borrowed_scope':copy.deepcopy(proto.scope)}
            else:
                task=route.prepare_backward(loss,value)
                grad,=(other_thread(task) if threaded else task.run())
                proof={'identity':'passing-original-local-small-upstream','native_scopes':copy.deepcopy(route.scopes),
                       'native_calls':copy.deepcopy(route.rows),'native_refusals':copy.deepcopy(route.refusals)}
            arrays.update(loss=loss.detach().clone(),input=grad.detach().clone())
            assert state_digest(model)==initial
            assert [p._version for p in model.parameters()]==versions
            assert value._version==input_version and value.grad is None and all(p.grad is None for p in model.parameters())
    except BaseException as error:
        native=route.native if candidate else route
        try:
            save_case(f'partial-seed-{seed}-thread-{int(threaded)}-candidate-{int(candidate)}', {'partial':arrays},
                      {'complete':False,'error_type':type(error).__name__,'error':str(error),
                       'native_calls':copy.deepcopy(native.rows),'native_scopes':copy.deepcopy(native.scopes),
                       'native_refusals':copy.deepcopy(native.refusals),
                       'prototype_calls':copy.deepcopy(route.prototype.calls) if candidate else []})
        except BaseException as publication:error.add_note('Partial publication also failed: '+repr(publication))
        raise
    finally:
        for handle in handles:handle.remove()
    restored(model)
    assert not model.predictor._forward_hooks
    assert len(predictions)==2
    assert len(proof['native_scopes'])==1
    scope=proof['native_scopes'][0]
    assert scope['state']=='complete' and scope['hook_removed'] and scope['graph_id']>=0
    primary=[r for r in proof['native_calls'] if not r['during_backward']]
    recomputations=[r for r in proof['native_calls'] if r['during_backward']]
    assert len(primary)==8 and recomputations
    assert set(scope['call_threads'])=={r['thread'] for r in recomputations}
    assert all(r['graph_id']==scope['graph_id'] and r['backward_scope']==0 for r in recomputations)
    assert proof['native_refusals']==[]
    return arrays,{**proof,'state_sha256':initial,'parameter_versions_unchanged':True,'parameter_gradients_absent':True,
                   'all_methods_hooks_and_identity_restored':True,'threaded':threaded,'mask_seed':seed,
                   'masks':[{'context':m.context.tolist(),'target':m.target.tolist()} for m in masks]}


def save_case(name,stages,receipt):
    base=os.environ.get('VJEPA_UPSTREAM_VJP_EVIDENCE')
    if not base:return
    folder=Path(base)/name;folder.mkdir(mode=0o700);files={}
    for stage,arrays in stages.items():
        for key,tensor in arrays.items():
            path=folder/(stage+'-'+key+'.npy');a=tensor.numpy()
            with path.open('xb') as f:np.save(f,a,allow_pickle=False)
            files[stage+'-'+key]={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'shape':list(a.shape),'dtype':str(a.dtype)}
    with (folder/'receipt.json').open('x') as f:json.dump({'CPU_only':True,'pretrained':False,'EOT8_qualified':False,'CUDA_qualified':False,'files':files,**receipt},f,sort_keys=True,indent=2);f.write('\n')


@pytest.mark.parametrize('seed',(19,23))
@pytest.mark.parametrize('threaded',(False,True))
def test_small_upstream_predictor_preserves_forward_and_native_L1_under_frozen_guard(fixture,seed,threaded):
    before,bproof=calculate(fixture,seed,threaded=threaded)
    actual,proof=calculate(fixture,seed,candidate=True,threaded=threaded)
    after,aproof=calculate(fixture,seed,threaded=threaded)
    check=agreement(actual['input'].numpy(),before['input'].numpy(),BOUND)
    # Save actual returns before a numerical refusal; never regenerate to hide it.
    save_case(f'seed-{seed}-thread-{int(threaded)}',{'before':before,'actual':actual,'after':after},
              {'before':bproof,'actual':proof,'after':aproof,'input_check':check})
    assert check['passed'],check
    for name in before:
        assert torch.equal(before[name],after[name])
        if name!='input':assert torch.equal(before[name],actual[name]),name
    assert bproof['state_sha256']==proof['state_sha256']==aproof['state_sha256']
    calls=proof['prototype_calls'];initial=[c for c in calls if c['forward_graph_id']==-1]
    assert len(initial)==4 and all(c['backward_complete'] and c['hooks_removed'] for c in initial)
    assert len({c['inner_graph_id'] for c in initial})==4
    native_scope=proof['native_scopes'][0]
    assert all(c['outer_graph_id']==native_scope['graph_id'] for c in initial)
    assert proof['borrowed_scope']['graph_id']==native_scope['graph_id'] and not proof['borrowed_scope']['new_outer_hook_added']
    assert native_scope['root_hook_thread'] in native_scope['call_threads']
    if threaded:assert native_scope['runner_thread']!=native_scope['creator_thread']


@pytest.mark.parametrize('host',('linux10603','linux10609'))
def test_university_host_refused_before_model_or_method_access(monkeypatch,host):
    monkeypatch.setattr(integration.platform,'node',lambda:host)
    with pytest.raises(ValueError,match='university refused'):integration.LocalUpstreamRoute(None)


def test_missing_checkpoint_recomputation_still_refused_by_unchanged_native_guard(fixture):
    model,pixels,masks=fixture
    model.encoder.use_activation_checkpointing=False;model.predictor.use_activation_checkpointing=False
    targets=model.clean_targets(pixels);value=pixels.clone().requires_grad_(True)
    with pytest.raises(ValueError,match='backward-recomputation-unobserved'):
        with integration.LocalUpstreamRoute(model) as route:
            loss=model.loss_per_example(value,targets,masks(19)).sum()
            route.gradient(loss,value)
    assert route.native.refusals[-1]['predicate']=='backward-recomputation-unobserved'
    assert route.native.scopes[0]['call_threads']==[] and route.native.scopes[0]['hook_removed']
    restored(model)


def test_foreign_graph_inside_math_VJP_still_refused_by_native_guard(fixture,monkeypatch):
    model,pixels,masks=fixture;targets=model.clean_targets(pixels);value=pixels.clone().requires_grad_(True);handles=[]
    route=integration.LocalUpstreamRoute(model);actual=route.prototype._math_call
    def observe_math(*values):
        out=actual(*values)
        def foreign(g):
            # Isolate foreign-graph refusal from the separate native flag guard.
            choices={'flash':SDPBackend.FLASH_ATTENTION,'mem_efficient':SDPBackend.EFFICIENT_ATTENTION,
                     'math':SDPBackend.MATH,'cudnn':SDPBackend.CUDNN_ATTENTION}
            with sdpa_kernel([choice for name,choice in choices.items() if route.native.original_flags[name]]):
                model.encoder.blocks[0].attn(torch.zeros(1,8,128))
            return None
        handles.append(out.register_hook(foreign));return out
    monkeypatch.setattr(route.prototype,'_math_call',observe_math)
    try:
        with pytest.raises(ValueError,match='foreign-autograd-graph'):
            with route:
                loss=model.loss_per_example(value,targets,masks(19)).sum();route.gradient(loss,value)
    finally:
        for handle in handles:handle.remove()
    assert route.native.refusals[-1]['predicate']=='foreign-autograd-graph'
    assert route.native.refusals[-1]['current_graph_id']!=route.native.refusals[-1]['expected_graph_id']
    assert route.native.scopes[0]['hook_removed'] and all(c['hooks_removed'] for c in route.prototype.calls if 'hooks_removed' in c)
    restored(model)
    destination=os.environ.get('VJEPA_UPSTREAM_VJP_EVIDENCE')
    if destination:
        with (Path(destination)/'foreign-graph-refusal.json').open('x') as f:
            json.dump({'CPU_only':True,'expected_rejection':True,'native_refusals':route.native.refusals,
                       'native_scopes':route.native.scopes,'prototype_calls':route.prototype.calls,
                       'all_methods_and_hooks_restored':True,'original_flags_restored':integration.cpu.flags()==route.native.original_flags},f,sort_keys=True,indent=2);f.write('\n')


def test_original_family_coverage_guard_is_not_relaxed(fixture):
    model,pixels,masks=fixture;targets=model.clean_targets(pixels);value=pixels.clone().requires_grad_(True)
    with pytest.raises(ValueError,match='forward-coverage-before-backward'):
        with integration.LocalUpstreamRoute(model) as route:
            loss=model.loss_per_example(value,targets,masks(19)[:1]).sum();route.gradient(loss,value)
    assert route.native.refusals[-1]['predicate']=='forward-coverage-before-backward'
    assert not any(c['backward_complete'] for c in route.prototype.calls)
    restored(model)


def test_primary_exception_restores_private_views_methods_and_identity(fixture):
    model,_,_=fixture
    with pytest.raises(RuntimeError,match='primary'):
        with integration.LocalUpstreamRoute(model):raise RuntimeError('primary')
    restored(model)


def test_changed_private_binding_refuses_and_restores(fixture):
    model,pixels,masks=fixture;targets=model.clean_targets(pixels);value=pixels.clone().requires_grad_(True)
    with pytest.raises(ValueError,match='binding changed'):
        with integration.LocalUpstreamRoute(model) as route:
            function,old,new=route.views[0];function.__globals__['F']=old
            loss=model.loss_per_example(value,targets,masks(19)).sum()
    restored(model)


def test_reused_route_refuses(fixture):
    model,_,_=fixture;route=integration.LocalUpstreamRoute(model)
    with pytest.raises(RuntimeError):
        with route:raise RuntimeError('close without derivative')
    with pytest.raises(ValueError,match='single use'):
        with route:pass
    restored(model)


def test_frozen_source_binding_refuses_changed_guard(fixture,monkeypatch):
    monkeypatch.setattr(integration,'NATIVE_SHA','0'*64)
    with pytest.raises(ValueError,match='source changed'):integration.LocalUpstreamRoute(fixture[0])
    restored(fixture[0])
