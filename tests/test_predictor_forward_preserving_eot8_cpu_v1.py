"""Real local eight-draw upstream objective, original native guard and saved evidence."""
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

from attacks.pgd import native_objective
from models import predictor_forward_preserving_eot8_cpu_v1 as integration
from models import predictor_eot8_owned_v1 as frozen
from models import predictor_eot8_adapter_v1 as adapter
from models.native_diagnostic import tiny_fixture
from models.execution import configure_execution, apply_model_policy
from models.diagnostics import state_digest
from models.attention_saved_evidence import agreement

ROOT = Path(__file__).resolve().parents[1]
BOUND = {'rtol': .002, 'atol': 2e-7}


@pytest.fixture
def fixture():
    configure_execution('shared-fp64-native', device='cpu')
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261114)
        model, pixels, masks = tiny_fixture(ROOT/'external/vjepa2')
    apply_model_policy(model, 'shared-fp64-native')
    model.eval()
    return model, pixels, masks


def bank(fixture, seed=19):
    return [fixture[2](seed+i) for i in range(8)]


def other_thread(task):
    result, errors = [], []
    def run():
        try:
            result.append(task.run())
        except BaseException as e:
            errors.append(e)
    worker = threading.Thread(target=run)
    worker.start()
    worker.join(30)
    assert not worker.is_alive(), 'Local EOT8 native backward thread timed out.'
    if errors:
        raise errors[0]
    return result[0]


def restored(model):
    assert not hasattr(model, integration.MARKER)
    assert not hasattr(model, integration.small.MARKER)
    assert all('forward' not in b.attn.__dict__ for blocks in
               (model.encoder.blocks, model.predictor.predictor_blocks) for b in blocks)
    assert not adapter._ACTIVE and not integration.cpu._ACTIVE
    assert model._diagnostic_execution_policy == 'shared-fp64-native'


def proof_for(route, candidate):
    native = route.native if candidate else route
    proof = {'native_scopes': copy.deepcopy(native.scopes), 'native_calls': copy.deepcopy(native.rows),
             'native_refusals': copy.deepcopy(native.refusals)}
    if candidate:
        proof.update(identity=integration.IDENTITY, prototype_calls=copy.deepcopy(route.prototype.calls),
                     borrowed_scope=copy.deepcopy(route.prototype.scope))
    return proof


def save_case(name, stages, receipt):
    base = os.environ.get('VJEPA_EOT8_CPU_VJP_EVIDENCE')
    if not base:
        return
    folder = Path(base)/name
    folder.mkdir(mode=0o700)
    files = {}
    for stage, arrays in stages.items():
        for key, tensor in arrays.items():
            path = folder/(stage+'-'+key+'.npy')
            a = tensor.numpy()
            with path.open('xb') as f:
                np.save(f, a, allow_pickle=False)
            files[stage+'-'+key] = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                   'shape': list(a.shape), 'dtype': str(a.dtype)}
    with (folder/'receipt.json').open('x') as f:
        json.dump({'CPU_only': True, 'pretrained': False, 'CUDA_qualified': False,
                   'research_or_training_qualified': False, 'files': files, **receipt}, f, sort_keys=True, indent=2)
        f.write('\n')


def calculate(fixture, seed, *, candidate=False, threaded=False):
    model, pixels, _ = fixture
    targets = model.clean_targets(pixels)
    masks = bank(fixture, seed)
    value = pixels.detach().clone().requires_grad_(True)
    initial = state_digest(model)
    versions = [p._version for p in model.parameters()]
    input_version, target_version = value._version, targets._version
    arrays = {'targets': targets.detach().clone(), 'pixels': pixels.clone()}
    handles, predictions = [], []
    def observe(module, args, output):
        index = len(predictions)
        predictions.append(index)
        arrays['prediction-'+str(index)] = output.detach().clone()
        pair = masks[index//2][index % 2]
        selected = targets.gather(1, pair.target.unsqueeze(-1).expand(-1, -1, 128))
        arrays['selected-target-'+str(index)] = selected.detach().clone()
        def incoming(g):
            arrays['cotangent-'+str(index)] = g.detach().clone()
            return None
        handles.append(output.register_hook(incoming))
        return None
    handles.append(model.predictor.register_forward_hook(observe))
    route = integration.LocalEOT8Route(model) if candidate else frozen.PredictorRoute(model, 'original', tiny=True, max_calls=128)
    try:
        with route:
            if candidate:
                if threaded:
                    route._run_native = other_thread
                loss = route.loss(value, targets, masks)
            else:
                route.phase = 'EOT8-before'
                loss = native_objective(model, targets, masks)(value).sum()
            arrays['loss'] = loss.detach().clone()
            if candidate:
                grad = route.gradient(loss, value)
            else:
                task = route.prepare_backward(loss, value)
                grad, = other_thread(task) if threaded else task.run()
            proof = proof_for(route, candidate)
            arrays['input'] = grad.detach().clone()
            assert state_digest(model) == initial
            assert [p._version for p in model.parameters()] == versions
            assert value._version == input_version and targets._version == target_version
            assert torch.equal(targets, arrays['targets'])
            assert value.grad is None and all(p.grad is None for p in model.parameters())
            if candidate:
                with pytest.raises(ValueError, match='exact fresh native EOT8'):
                    route.gradient(loss, value)
    except BaseException as error:
        try:
            save_case(f'partial-seed-{seed}-thread-{int(threaded)}-candidate-{int(candidate)}', {'partial': arrays},
                      {'complete': False, 'error_type': type(error).__name__, 'error': str(error),
                       **proof_for(route, candidate)})
        except BaseException as publication:
            error.add_note('Partial publication also failed: '+repr(publication))
        raise
    finally:
        for handle in handles:
            handle.remove()
    restored(model)
    assert not model.predictor._forward_hooks and len(predictions) == 16
    assert len(proof['native_scopes']) == 1 and proof['native_refusals'] == []
    scope = proof['native_scopes'][0]
    assert scope['phase'] == 'EOT8-before' and scope['state'] == 'complete' and scope['hook_removed']
    primary = [r for r in proof['native_calls'] if not r['during_backward']]
    recompute = [r for r in proof['native_calls'] if r['during_backward']]
    assert len(primary) == len(recompute) == 64
    for owner in ('encoder.0', 'encoder.1', 'predictor.0', 'predictor.1'):
        assert sum(r['owner'] == owner for r in primary) == 16
        assert sum(r['owner'] == owner for r in recompute) == 16
    assert set(scope['call_threads']) == {r['thread'] for r in recompute}
    assert all(r['graph_id'] == scope['graph_id'] and r['backward_scope'] == 0 for r in recompute)
    return arrays, {**proof, 'state_sha256': initial, 'parameter_versions_unchanged': True,
                    'parameter_gradients_absent': True, 'input_and_target_versions_unchanged': True,
                    'all_methods_hooks_and_identity_restored': True, 'threaded': threaded, 'seed': seed,
                    'masks': [[{'context': m.context.tolist(), 'target': m.target.tolist()} for m in draw] for draw in masks]}


@pytest.mark.parametrize('seed', (19, 37))
@pytest.mark.parametrize('threaded', (False, True))
def test_actual_eight_draw_loss_preserves_native_values_and_full_input_bound(fixture, seed, threaded):
    before, bproof = calculate(fixture, seed, threaded=threaded)
    actual, proof = calculate(fixture, seed, candidate=True, threaded=threaded)
    after, aproof = calculate(fixture, seed, threaded=threaded)
    check = agreement(actual['input'].numpy(), before['input'].numpy(), BOUND)
    save_case(f'seed-{seed}-thread-{int(threaded)}', {'before': before, 'actual': actual, 'after': after},
              {'before': bproof, 'actual': proof, 'after': aproof, 'input_check': check, 'complete': True})
    assert check['passed'], check
    for name in before:
        assert torch.equal(before[name], after[name]), name
        if name != 'input':
            assert torch.equal(before[name], actual[name]), name
    assert bproof['state_sha256'] == proof['state_sha256'] == aproof['state_sha256']
    calls = proof['prototype_calls']
    initial = [c for c in calls if c['forward_graph_id'] == -1]
    assert len(calls) == 64 and len(initial) == 32
    assert len({c['inner_graph_id'] for c in initial}) == 32
    scope = proof['native_scopes'][0]
    for c in initial:
        assert c['backward_complete'] and c['hooks_removed']
        assert c['inner_root_calls'] == 1 and c['outer_graph_id'] == scope['graph_id']
        assert sorted(h['index'] for h in c['leaf_hooks']) == [0, 1, 2]
        assert all(h['graph_id'] == c['inner_graph_id'] for h in c['leaf_hooks'])
    assert proof['borrowed_scope']['graph_id'] == scope['graph_id']
    assert not proof['borrowed_scope']['new_outer_hook_added']
    assert scope['root_hook_thread'] in scope['call_threads']
    if threaded:
        assert scope['runner_thread'] != scope['creator_thread']


@pytest.mark.parametrize('host', ('linux10603', 'linux10609'))
def test_local_entry_refuses_university_before_model_access(monkeypatch, host):
    monkeypatch.setattr(integration.platform, 'node', lambda: host)
    with pytest.raises(ValueError, match='university refused'):
        integration.LocalEOT8Route(None)


@pytest.mark.parametrize('fault', ('seven-draws', 'nine-draws', 'one-family', 'swapped-families', 'overlap'))
def test_incomplete_or_changed_eight_draw_families_refused_before_forward(fixture, fault):
    model, pixels, _ = fixture
    targets = model.clean_targets(pixels)
    value = pixels.clone().requires_grad_(True)
    masks = bank(fixture)
    if fault == 'seven-draws':
        masks.pop()
    elif fault == 'nine-draws':
        masks.append(fixture[2](27))
    elif fault == 'one-family':
        masks[0] = masks[0][:1]
    elif fault == 'swapped-families':
        masks[0] = masks[0][::-1]
    else:
        masks[0][0].target[0, 0] = masks[0][0].context[0, 0]
    with pytest.raises(ValueError, match='families required'):
        with integration.LocalEOT8Route(model) as route:
            route.loss(value, targets, masks)
    assert route.native.rows == route.prototype.calls == []
    restored(model)


def test_native_eot8_owner_coverage_is_not_replaced_by_entry_validation(fixture):
    model, pixels, _ = fixture
    targets = model.clean_targets(pixels)
    value = pixels.clone().requires_grad_(True)
    with pytest.raises(ValueError, match='forward-coverage-before-backward'):
        with integration.LocalEOT8Route(model) as route:
            loss = native_objective(model, targets, bank(fixture)[:7])(value).sum()
            route.native.prepare_backward(loss, value)
    assert route.native.refusals[-1]['predicate'] == 'forward-coverage-before-backward'
    assert not any(c['backward_complete'] for c in route.prototype.calls)
    restored(model)
    save_case('native-coverage-refusal', {}, {'expected_rejection': True, **proof_for(route, True), 'restored': True})


def test_missing_checkpoint_still_refused_by_native_guard(fixture):
    model, pixels, _ = fixture
    model.encoder.use_activation_checkpointing = False
    model.predictor.use_activation_checkpointing = False
    targets = model.clean_targets(pixels)
    value = pixels.clone().requires_grad_(True)
    with pytest.raises(ValueError, match='backward-recomputation-unobserved'):
        with integration.LocalEOT8Route(model) as route:
            loss = route.loss(value, targets, bank(fixture))
            route.gradient(loss, value)
    assert route.native.refusals[-1]['predicate'] == 'backward-recomputation-unobserved'
    assert route.native.scopes[0]['call_threads'] == [] and route.native.scopes[0]['hook_removed']
    restored(model)
    save_case('native-missing-checkpoint-refusal', {}, {'expected_rejection': True, **proof_for(route, True), 'restored': True})


def test_foreign_graph_inside_math_still_refused_by_native_guard(fixture, monkeypatch):
    model, pixels, _ = fixture
    targets = model.clean_targets(pixels)
    value = pixels.clone().requires_grad_(True)
    route = integration.LocalEOT8Route(model)
    actual = route.prototype._math_call
    handles = []
    def observe_math(*values):
        out = actual(*values)
        def foreign(g):
            choices = {'flash': SDPBackend.FLASH_ATTENTION, 'mem_efficient': SDPBackend.EFFICIENT_ATTENTION,
                       'math': SDPBackend.MATH, 'cudnn': SDPBackend.CUDNN_ATTENTION}
            with sdpa_kernel([choice for name, choice in choices.items() if route.native.original_flags[name]]):
                model.encoder.blocks[0].attn(torch.zeros(1, 8, 128))
            return None
        handles.append(out.register_hook(foreign))
        return out
    monkeypatch.setattr(route.prototype, '_math_call', observe_math)
    try:
        with pytest.raises(ValueError, match='foreign-autograd-graph'):
            with route:
                loss = route.loss(value, targets, bank(fixture))
                route.gradient(loss, value)
    finally:
        for handle in handles:
            handle.remove()
    assert route.native.refusals[-1]['predicate'] == 'foreign-autograd-graph'
    assert route.native.scopes[0]['hook_removed']
    assert all(c['hooks_removed'] for c in route.prototype.calls if 'hooks_removed' in c)
    restored(model)
    save_case('native-foreign-graph-refusal', {}, {'expected_rejection': True, **proof_for(route, True), 'restored': True,
                                                'flags_restored': integration.cpu.flags() == route.native.original_flags})


def test_exact_objective_tensor_identity_required(fixture):
    model, pixels, _ = fixture
    targets = model.clean_targets(pixels)
    value = pixels.clone().requires_grad_(True)
    with pytest.raises(ValueError, match='exact fresh native EOT8'):
        with integration.LocalEOT8Route(model) as route:
            loss = route.loss(value, targets, bank(fixture))
            route.gradient(loss+0, value)
    assert route.native.scopes == []
    restored(model)


def test_changed_phase_refused_and_restored(fixture):
    with pytest.raises(ValueError, match='binding or phase changed'):
        with integration.LocalEOT8Route(fixture[0]) as route:
            route.native.phase = 'draw-before'
            route.check_binding()
    restored(fixture[0])


def test_new_state_has_no_unbounded_standalone_attention_entry(fixture):
    with pytest.raises(ValueError, match='bounded EOT8 model entry'):
        with integration.LocalEOT8Route(fixture[0]) as route:
            route.prototype.attention(None, None, None)
    restored(fixture[0])
    with pytest.raises(ValueError, match='eight tiny local'):
        integration.cpu.LocalForwardPreservingAttention(max_calls=64)


def test_primary_failure_restores_and_route_remains_single_use(fixture):
    route = integration.LocalEOT8Route(fixture[0])
    with pytest.raises(RuntimeError, match='primary'):
        with route:
            raise RuntimeError('primary')
    restored(fixture[0])
    with pytest.raises(ValueError, match='single use'):
        with route:
            pass


def test_changed_frozen_integration_source_pin_refused(fixture, monkeypatch):
    monkeypatch.setattr(integration, 'SMALL_SHA', '0'*64)
    with pytest.raises(ValueError, match='source changed'):
        integration.LocalEOT8Route(fixture[0])
    restored(fixture[0])
