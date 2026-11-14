"""New local operator checks; no university/historical cohort is re-executed."""
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
from torch.utils.checkpoint import checkpoint

from models import predictor_forward_preserving_cpu_v1 as prototype
from models import attention_saved_evidence as arithmetic

BOUNDS = {'output': {'rtol': 1e-4, 'atol': 2e-5},
          **{n: {'rtol': .002, 'atol': 2e-7} for n in ('dq', 'dk', 'dv')}}


@pytest.fixture(scope='module', autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def data():
    gen = torch.Generator(device='cpu').manual_seed(517)
    x = torch.randn(1, 6, 8, generator=gen)
    weight = torch.randn(8, 24, generator=gen) * .125
    return x, weight


def operands(x, weight):
    q, k, v = (x @ weight).view(1, 6, 3, 2, 4).permute(2, 0, 3, 1, 4).unbind(0)
    return q.contiguous(), k.contiguous(), v


def original_prediction():
    value, weight = data()
    q, k, v = operands(value.requires_grad_(True), weight)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    with sdpa_kernel(SDPBackend.MATH):
        other = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    changed = (out != other).nonzero()
    assert len(changed), 'Actual local original/math rounding example is unavailable.'
    coordinate = tuple(changed[0].tolist())
    return out.detach(), other.detach(), coordinate


def toy(target, *, custom=False, recompute=False, math_route=False):
    value, weight = data()
    x = value.requires_grad_(True)
    weight_before, x_before = weight.clone(), x.detach().clone()
    route = prototype.LocalForwardPreservingAttention() if custom else None
    saved, hooks = {}, []
    def forward(t):
        values = operands(t, weight)
        if not saved:
            for name, v in zip(('q', 'k', 'v'), values):
                saved[name] = v.detach()  # Retain the exact source layout/storage, never mutate it.
                def observed(g, name=name):
                    saved['d'+name] = g.detach().clone()
                    return None
                hooks.append(v.register_hook(observed))
        return route.attention(*values) if custom else torch.nn.functional.scaled_dot_product_attention(*values)
    def evaluate():
        out = checkpoint(forward, x, use_reentrant=False) if recompute else forward(x)
        hooks.append(out.register_hook(lambda g: saved.update(cotangent=g.detach().clone())))
        loss = (out-target).abs().mean()
        grad = route.gradient(loss, x) if custom else torch.autograd.grad(loss, x)[0]
        return out, loss, grad
    try:
        if custom:
            with route:
                out, loss, grad = evaluate()
        elif math_route:
            with sdpa_kernel(SDPBackend.MATH):
                out, loss, grad = evaluate()
        else:
            out, loss, grad = evaluate()
    finally:
        for hook in hooks:
            hook.remove()
    assert torch.equal(weight, weight_before) and torch.equal(x.detach(), x_before)
    assert weight.grad is None and x.grad is None
    saved.update(output=out.detach().clone(), loss=loss.detach().clone(), input=grad.detach().clone(), target=target.clone())
    return saved, route


def direct_math(values):
    qkv = tuple(values[n].detach().requires_grad_(True) for n in ('q', 'k', 'v'))
    with sdpa_kernel(SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(*qkv)
        gradients = torch.autograd.grad(out, qkv, values['cotangent'])
    return dict(zip(('output', 'dq', 'dk', 'dv'), (out.detach(), *gradients)))


def save_case(name, evidence, metadata):
    base = os.environ.get('VJEPA_FORWARD_VJP_EVIDENCE')
    if not base:
        return
    folder = Path(base)/name
    folder.mkdir(mode=0o700)
    files = {}
    for key, tensor in evidence.items():
        arr = tensor.detach().numpy()
        path = folder/(key+'.npy')
        with path.open('xb') as f:
            np.save(f, arr, allow_pickle=False)
        files[key] = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'shape': list(arr.shape), 'dtype': str(arr.dtype)}
    with (folder/'receipt.json').open('x') as f:
        json.dump({'local_CPU_only': True, 'CUDA_qualified': False, 'full_model_qualified': False,
                   'files': files, **metadata}, f, indent=2, sort_keys=True)
        f.write('\n')


@pytest.mark.parametrize('recompute', (False, True))
@pytest.mark.parametrize('near', (False, True))
def test_native_L1_forward_and_cotangent_preserved_with_actual_math_VJP(recompute, near):
    native, math, coordinate = original_prediction()
    target = native + .125
    target[coordinate] = torch.nextafter(native[coordinate], math[coordinate]) if near else native[coordinate]
    before, _ = toy(target, recompute=recompute)
    actual, route = toy(target, custom=True, recompute=recompute)
    after, _ = toy(target, recompute=recompute)
    rejected_forward, _ = toy(target, math_route=True, recompute=recompute)
    for key in ('output', 'loss', 'cotangent', 'target'):
        assert torch.equal(before[key], actual[key]) and torch.equal(actual[key], after[key])
    assert before['cotangent'][coordinate] != rejected_forward['cotangent'][coordinate]
    for key in ('q', 'k', 'v', 'input', 'dq', 'dk', 'dv'):
        assert torch.equal(before[key], after[key])
    input_check = arithmetic.agreement(actual['input'].numpy(), before['input'].numpy(), BOUNDS['dq'])
    assert input_check['passed'], input_check
    assert [prototype.fingerprint(actual[n]) for n in ('q', 'k', 'v')] == route.calls[0]['inputs']
    gold = arithmetic.oracle(*(actual[n].numpy() for n in ('q', 'k', 'v', 'cotangent')))
    direct = direct_math(actual)
    checks = {}
    for name in BOUNDS:
        checks[name] = arithmetic.agreement(actual[name].numpy(), gold[name], BOUNDS[name])
        assert checks[name]['passed'], checks[name]
        if name != 'output':
            assert torch.equal(actual[name], direct[name])
    assert len(route.calls) == (2 if recompute else 1)
    assert route.scope['state'] == 'complete' and route.scope['root_calls'] == 1 and route.scope['root_hook_removed']
    assert not route.active and not prototype._ACTIVE
    initial = route.calls[0]
    assert initial['backward_complete'] and initial['hooks_removed'] and initial['inner_root_calls'] == 1
    assert initial['inner_graph_id'] != initial['outer_graph_id'] == route.scope['graph_id']
    assert [item['sha256'] for item in initial['returned']] == [prototype.fingerprint(actual[n])['sha256'] for n in ('dq', 'dk', 'dv')]
    assert initial['incoming']['sha256'] == hashlib.sha256(actual['cotangent'].numpy().tobytes()).hexdigest()
    assert initial['entry_flags'] == initial['exit_flags'] == route.original_flags
    assert all(c['original_forward_grad_enabled'] and c['forward_complete'] for c in route.calls)
    if recompute:
        assert route.calls[1]['forward_graph_id'] == route.scope['graph_id']
    evidence = {**actual, **{'before-'+k: before[k] for k in ('output', 'loss', 'cotangent', 'input')},
                **{'after-'+k: after[k] for k in ('output', 'loss', 'cotangent', 'input')},
                **{'direct-'+k: v for k, v in direct.items()},
                **{'old-math-'+k: rejected_forward[k] for k in ('output', 'loss', 'cotangent', 'input')}}
    save_case(f'native-L1-checkpoint-{int(recompute)}-near-{int(near)}', evidence,
              {'coordinate': list(coordinate), 'checkpointed': recompute, 'near_zero': near,
               'checks': checks, 'input_check': input_check, 'calls': route.calls, 'outer_scope': route.scope})


def test_alias_groups_and_nondense_layouts_preserved():
    gen = torch.Generator().manual_seed(81)
    x = torch.randn(1, 2, 6, 4, generator=gen, requires_grad=True)
    values = (x.view_as(x), x.view_as(x), x.view_as(x))
    original = torch.nn.functional.scaled_dot_product_attention(*values)
    baseline = torch.autograd.grad(original.square().mean(), x)[0]
    with prototype.LocalForwardPreservingAttention() as route:
        out = route.attention(*values)
        actual = route.gradient(out.square().mean(), x)
    assert route.calls[0]['aliases'] == [0, 0, 0] and torch.equal(out, original)
    assert arithmetic.agreement(actual.numpy(), baseline.numpy(), BOUNDS['dq'])['passed']


@pytest.mark.parametrize('host', ('linux10603', 'linux10609', 'linux10603.dc.engr.scu.edu'))
def test_university_host_refused_before_operator(monkeypatch, host):
    monkeypatch.setattr(prototype.platform, 'node', lambda: host)
    with pytest.raises(ValueError, match='university execution refused'):
        prototype.LocalForwardPreservingAttention()


@pytest.mark.parametrize('kind', ('FP64', 'meta', 'full-size', 'no-gradient', 'nonfinite', 'same-object'))
def test_invalid_or_target_domains_refuse(kind):
    shape = (1, 12, 2048, 32) if kind == 'full-size' else (1, 2, 6, 4)
    device = 'meta' if kind in ('meta', 'full-size') else 'cpu'
    q = torch.ones(shape, device=device, dtype=torch.float64 if kind == 'FP64' else torch.float32,
                   requires_grad=kind != 'no-gradient')
    if kind == 'nonfinite':
        with torch.no_grad():q[0, 0, 0, 0] = float('nan')
    values = (q, q, q) if kind == 'same-object' else (q, q.clone(), q.clone())
    with pytest.raises(ValueError):
        with prototype.LocalForwardPreservingAttention() as route:
            route.attention(*values)
    assert not route.active and not prototype._ACTIVE


@pytest.mark.parametrize('context', ('no-grad', 'inference', 'autocast'))
def test_unsupported_context_refuses(context):
    x, weight = data();values = operands(x.requires_grad_(True), weight)
    manager = {'no-grad': torch.no_grad, 'inference': torch.inference_mode,
               'autocast': lambda: torch.autocast('cpu')}[context]
    with pytest.raises(ValueError):
        with prototype.LocalForwardPreservingAttention() as route:
            with manager():route.attention(*values)
    assert not route.active


def test_external_backward_is_not_owned():
    x, weight = data();x.requires_grad_(True)
    with pytest.raises(ValueError, match='owned outer GraphTask'):
        with prototype.LocalForwardPreservingAttention() as route:
            out = route.attention(*operands(x, weight))
            torch.autograd.grad(out.square().mean(), x)
    assert not route.active and not prototype._ACTIVE


def test_create_graph_refuses_without_running_backward():
    x, weight = data();x.requires_grad_(True)
    with pytest.raises(ValueError, match='first-order'):
        with prototype.LocalForwardPreservingAttention() as route:
            out = route.attention(*operands(x, weight))
            route.gradient(out.square().mean(), x, create_graph=True)
    assert route.scope is None and not route.active


def test_changed_saved_input_refuses_and_removes_outer_hook():
    x, weight = data();x.requires_grad_(True)
    with pytest.raises(RuntimeError):
        with prototype.LocalForwardPreservingAttention() as route:
            out = route.attention(*operands(x, weight))
            # Mutate a saved operand, not a leaf used by earlier projection math.
            with torch.no_grad():out.grad_fn.saved_tensors[0].add_(.25)
            route.gradient(out.square().mean(), x)
    assert route.scope['state'] == 'failed' and route.scope['root_hook_removed'] and not route.active


def test_math_exception_restores_flags_and_preserves_primary_error(monkeypatch):
    x, weight = data();x.requires_grad_(True)
    def fail(*unused):raise RuntimeError('injected-public-math-failure')
    with pytest.raises(RuntimeError, match='injected-public-math-failure'):
        with prototype.LocalForwardPreservingAttention() as route:
            out = route.attention(*operands(x, weight))
            monkeypatch.setattr(route, '_math_call', fail)
            route.gradient(out.square().mean(), x)
    assert prototype.flags() == route.original_flags
    assert route.calls[0]['hooks_removed'] and route.scope['root_hook_removed'] and not route.active


def test_foreign_forward_thread_refuses():
    x, weight = data();x.requires_grad_(True);values = operands(x, weight);errors = []
    def other():
        try:route.attention(*values)
        except ValueError as error:errors.append(str(error))
    with pytest.raises(ValueError, match='Incomplete'):
        with prototype.LocalForwardPreservingAttention() as route:
            thread = threading.Thread(target=other);thread.start();thread.join()
    assert len(errors) == 1 and 'thread' in errors[0] and not route.calls and not route.active


def test_reentrant_checkpoint_refuses_its_no_grad_forward():
    x, weight = data();x.requires_grad_(True)
    with pytest.raises(ValueError, match='Grad-enabled'):
        with prototype.LocalForwardPreservingAttention() as route:
            checkpoint(lambda t: route.attention(*operands(t, weight)), x, use_reentrant=True)


def test_scope_reuse_overlap_and_incomplete_work_refuse():
    first = prototype.LocalForwardPreservingAttention()
    with pytest.raises(ValueError, match='Incomplete'):
        with first:
            with pytest.raises(ValueError, match='fresh serial'):
                with prototype.LocalForwardPreservingAttention():pass
    with pytest.raises(ValueError, match='fresh serial'):
        with first:pass
    assert not prototype._ACTIVE


def test_injected_foreign_graph_id_is_not_accepted(monkeypatch):
    x, weight = data();x.requires_grad_(True)
    with pytest.raises(ValueError, match='owned outer GraphTask'):
        with prototype.LocalForwardPreservingAttention() as route:
            out = route.attention(*operands(x, weight))
            # Alter only the identity query once a real outer GraphTask starts.
            actual = route._graph_api
            monkeypatch.setattr(route, '_graph_api', lambda: 999999 if actual() >= 0 else -1)
            # The native loss root and custom backward would both see this fake
            # value, so separately corrupt the captured expected root identity.
            hook = out.register_hook(lambda g: route.scope.update(graph_id=-2))
            try:route.gradient(out.square().mean(), x)
            finally:hook.remove()
    assert route.scope['root_hook_removed'] and not route.active


@pytest.mark.parametrize('recompute', (False, True))
def test_two_chained_attention_calls_have_distinct_owned_inner_graphs(recompute):
    value, weight = data()
    def run(custom):
        x = value.clone().requires_grad_(True)
        route = prototype.LocalForwardPreservingAttention() if custom else None
        def forward(t):
            values = operands(t, weight)
            y = route.attention(*values) if custom else torch.nn.functional.scaled_dot_product_attention(*values)
            second = (y.view_as(y), y.clone(), y.flip(-1))
            return route.attention(*second) if custom else torch.nn.functional.scaled_dot_product_attention(*second)
        def evaluate():
            out = checkpoint(forward, x, use_reentrant=False) if recompute else forward(x)
            loss = out.square().mean()
            grad = route.gradient(loss, x) if custom else torch.autograd.grad(loss, x)[0]
            return out.detach(), grad, route
        if custom:
            with route:return evaluate()
        return evaluate()
    before, expected, _ = run(False)
    actual, grad, route = run(True)
    after, repeated, _ = run(False)
    assert torch.equal(before, actual) and torch.equal(after, actual) and torch.equal(expected, repeated)
    check = arithmetic.agreement(grad.numpy(), expected.numpy(), BOUNDS['dq']);assert check['passed']
    initial = [c for c in route.calls if c['forward_graph_id'] == -1]
    assert len(initial) == 2 and len({c['inner_graph_id'] for c in initial}) == 2
    assert all(c['outer_graph_id'] == route.scope['graph_id'] and c['hooks_removed'] for c in initial)
    assert all(c['forward_graph_id'] in (-1, route.scope['graph_id']) for c in route.calls)
    save_case(f'chained-checkpoint-{int(recompute)}', {'before': before, 'actual': actual, 'after': after,
              'expected-input': expected, 'actual-input': grad, 'repeated-input': repeated},
              {'checkpointed': recompute, 'calls': route.calls, 'outer_scope': route.scope, 'input_check': check})


def test_input_mutation_in_leaf_hook_rejected_before_gradient_return():
    value, weight = data();value.requires_grad_(True)
    def mutate(grad):
        with torch.no_grad():value.add_(.25)
        return None
    hook = value.register_hook(mutate)
    try:
        with pytest.raises(ValueError, match='input version'):
            with prototype.LocalForwardPreservingAttention() as route:
                out = route.attention(*operands(value, weight))
                route.gradient(out.square().mean(), value)
    finally:hook.remove()
    assert route.scope['state'] == 'failed' and route.scope['root_hook_removed'] and not route.active


def test_new_forward_after_complete_scope_is_refused():
    x, weight = data();x.requires_grad_(True)
    with pytest.raises(ValueError, match='consumed gradient scope'):
        with prototype.LocalForwardPreservingAttention() as route:
            out = route.attention(*operands(x, weight))
            route.gradient(out.square().mean(), x)
            route.attention(*operands(x, weight))
    assert route.scope['state'] == 'complete' and len(route.calls) == 1 and not route.active
