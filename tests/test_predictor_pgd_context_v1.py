"""New cold PGD/control checks. All gradient returns here are analytic simulations.

No autograd.grad/backward, native model objective, CUDA, network or old suite runs.
The separate bounded cohort supplies the actual numerical/ownership evidence.
"""
import ast
from contextlib import ExitStack
import copy
import importlib.util
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import torch

from attacks import pgd_context_v1 as attack
from models.predictor_pgd_cpu_v1 import Capture, IteratedBinding, ObjectiveLedger, write_json
from models import predictor_pgd_records_v1 as saved

ROOT = Path(__file__).resolve().parents[1]


def clean():
    return torch.full((1, 3, 4, 16, 16), .5)


def kwargs():
    return dict(epsilon=4/255, step_size=1/255, steps=2, restarts=2, seed=73, random_start=True)


class AnalyticSimulation:
    def __init__(self, constant=False, fail=None):
        self.points, self.contexts, self.events = [], [], []
        self.constant, self.fail = constant, fail
        self.active = False

    def loss(self, pixels):
        # Analytic parabola deliberately makes intermediate points competitive.
        return torch.ones(1) if self.constant else -(pixels[:, 0, 0, 0, 0]-.506).square()

    def objective(self, pixels):
        assert not torch.is_grad_enabled() and not self.active
        values = self.loss(pixels)
        self.points.append((pixels.detach().clone(), values.detach().clone(), 'no-grad'))
        return values

    def context(self, pixels):
        owner = self
        class Scope:
            def __enter__(self):
                assert not owner.active and torch.is_grad_enabled()
                owner.active = True
                self.value = pixels
                self.losses = owner.loss(pixels)
                owner.events.append('forward')
                return self

            def note_consider(self, value, losses):
                assert value is self.value and losses is self.losses
                owner.points.append((value.detach().clone(), losses.detach().clone(), 'consider'))
                owner.events.append('consider')

            def gradient(self):
                assert owner.events[-1] == 'consider'
                owner.events.append('backward')
                if owner.fail == 'throw':
                    raise RuntimeError('simulated failure')
                result = torch.zeros_like(pixels)
                if not owner.constant:
                    result[:, 0, 0, 0, 0] = -2*(pixels[:, 0, 0, 0, 0].detach()-.506)
                if owner.fail == 'nan':
                    result[0, 0, 0, 0, 0] = float('nan')
                if owner.fail == 'shape':
                    return result[..., :1]
                return result

            def __exit__(self, *unused):
                owner.active = False
                owner.events.append('exit')
        scope = Scope()
        self.contexts.append(scope)
        return scope


def test_only_gradient_bridge_and_tiny_guard_differ_from_frozen_PGD_AST():
    old = ast.parse((ROOT/'attacks/pgd.py').read_text())
    original = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'pgd_linf')
    new = ast.parse((ROOT/'attacks/pgd_context_v1.py').read_text())
    candidate = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'pgd_linf_context_v1')
    candidate.name = original.name
    candidate.args.kwonlyargs.pop()
    candidate.args.kw_defaults.pop()
    candidate.body = [n for n in candidate.body if not (isinstance(n, ast.Expr)
                      and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                      and n.value.func.id == 'tiny_domain')]
    class UndoBridge(ast.NodeTransformer):
        def visit_With(self, node):
            if len(node.items) == 1 and ast.unparse(node.items[0].context_expr) == 'gradient_context(current)':
                assert [ast.unparse(n) for n in node.body] == [
                    'losses = _losses(lambda _: step.losses, current)', 'consider(current, losses)',
                    'step.note_consider(current, losses)', 'gradient = checked_gradient(step.gradient(), current)']
                return ast.parse('losses = _losses(objective, current)\nconsider(current, losses)\ngradient, = torch.autograd.grad(losses.sum(), current)').body
            return self.generic_visit(node)
    candidate = UndoBridge().visit(candidate)
    assert ast.dump(candidate, include_attributes=False) == ast.dump(original, include_attributes=False)
    assert saved.digest(ROOT/'attacks/pgd.py') == attack.FROZEN_PGD_SHA256


@pytest.mark.parametrize('constant', (False, True))
def test_simulated_selection_projection_random_starts_and_consider_order(constant):
    c, sim = clean(), AnalyticSimulation(constant=constant)
    before = torch.get_rng_state().clone()
    result = attack.pgd_linf_context_v1(sim.objective, c, **kwargs(), gradient_context=sim.context)
    assert torch.equal(torch.get_rng_state(), before)
    assert len(sim.contexts) == len({id(s) for s in sim.contexts}) == 4
    assert sim.events == ['forward', 'consider', 'backward', 'exit']*4
    assert len(sim.points) == 7
    best, loss, _ = sim.points[0]
    for value, losses, _ in sim.points[1:]:
        if bool(losses > loss):
            best, loss = value, losses
        assert torch.all((value >= c-4/255) & (value <= c+4/255))
    assert torch.equal(result.adversarial, best) and torch.equal(result.losses, loss)
    assert not result.adversarial.requires_grad and not result.losses.requires_grad
    if constant:
        assert torch.equal(result.adversarial, c)  # Strict > keeps clean on every tie.
    generator = torch.Generator().manual_seed(73)
    expected_starts = [(c+(2*torch.rand(c.shape, generator=generator)-1)*(4/255)).clamp(c-4/255, c+4/255) for _ in range(2)]
    assert torch.equal(sim.points[1][0], expected_starts[0])
    assert torch.equal(sim.points[4][0], expected_starts[1])


@pytest.mark.parametrize('failure', ('throw', 'nan', 'shape'))
def test_simulated_failure_exits_context_and_stops_later_work(failure):
    sim = AnalyticSimulation(fail=failure)
    with pytest.raises((ValueError, RuntimeError)):
        attack.pgd_linf_context_v1(sim.objective, clean(), **kwargs(), gradient_context=sim.context)
    assert not sim.active and len(sim.contexts) == 1
    assert sim.events == ['forward', 'consider', 'backward', 'exit']
    assert len(sim.points) == 2


@pytest.mark.parametrize('fault', ('steps', 'restarts', 'epsilon', 'step_size', 'random_start', 'seed', 'shape', 'dtype', 'meta'))
def test_unsupported_domain_or_counts_refused_before_callback(fault):
    c, options = clean(), kwargs()
    if fault in ('steps', 'restarts'):
        options[fault] = 3
    elif fault == 'epsilon':
        options[fault] = 8/255
    elif fault == 'step_size':
        options[fault] = 2/255
    elif fault == 'random_start':
        options[fault] = False
    elif fault == 'seed':
        options[fault] = '73'
    elif fault == 'shape':
        c = c[..., :8]
    elif fault == 'dtype':
        c = c.double()
    else:
        c = torch.empty(c.shape, device='meta')
    def poison(*unused):
        pytest.fail('Unsupported input accessed a callback.')
    with pytest.raises(ValueError, match='Only tiny CPU'):
        attack.pgd_linf_context_v1(poison, c, **options, gradient_context=poison)


@pytest.mark.parametrize('host', ('linux10603', 'linux10609.dc.engr.scu.edu'))
def test_university_host_refused_before_tensor_access(monkeypatch, host):
    monkeypatch.setattr(attack.platform, 'node', lambda: host)
    with pytest.raises(ValueError, match='university'):
        attack.tiny_domain(None, None, None, None, None, None, None)
    with pytest.raises(ValueError, match='university'):
        IteratedBinding(None, None, None, None, None, None)


@pytest.mark.parametrize('mode', ('inference', 'autocast'))
def test_unsupported_mode_refused_before_callbacks(mode):
    context = torch.inference_mode() if mode == 'inference' else torch.autocast('cpu')
    with context, pytest.raises(ValueError, match='First-order'):
        attack.tiny_domain(clean(), **kwargs())


def fake_capture():
    value = clean().requires_grad_(True)
    root = torch.tensor(.2, requires_grad=True)
    losses = root.view(1)
    b = SimpleNamespace(common=lambda: None)
    capture = Capture(b, value, 0, True)
    capture.entered = True
    capture.root, capture.losses = root, losses
    capture.input_version = value._version
    from models.predictor_pgd_cpu_v1 import tensor_hash
    capture.input_hash = tensor_hash(value)
    capture._token = (root, value, losses, root._version)
    b.active = capture
    return capture


@pytest.mark.parametrize('fault', ('used', 'closed', 'wrong-input', 'wrong-root', 'wrong-loss-view', 'mutated-input', 'inactive'))
def test_stale_or_foreign_exact_tokens_refused_before_native_route(fault):
    c = fake_capture()
    if fault == 'used':
        c.used = True
    elif fault == 'closed':
        c.closed = True
    elif fault == 'wrong-input':
        c.value = c.value.detach().clone().requires_grad_(True)
    elif fault == 'wrong-root':
        c.root = c.root+0
    elif fault == 'wrong-loss-view':
        c.losses = c.losses.view(1)
    elif fault == 'mutated-input':
        with torch.no_grad():
            c.value.add_(.001)
    else:
        c.binding.active = None
    with pytest.raises(ValueError, match='exact loss/input'):
        c.gradient()


@pytest.mark.parametrize('fault', ('before-consider', 'higher-order', 'wrong-consider-input', 'wrong-consider-loss', 'consider-twice'))
def test_first_order_consider_before_backward_contract(fault):
    c = fake_capture()
    if fault == 'higher-order':
        c.note_consider(c.value, c.losses)
        action = lambda: c.gradient(create_graph=True)
    elif fault == 'wrong-consider-input':
        action = lambda: c.note_consider(c.value.clone(), c.losses)
    elif fault == 'wrong-consider-loss':
        action = lambda: c.note_consider(c.value, c.losses.clone())
    elif fault == 'consider-twice':
        c.note_consider(c.value, c.losses)
        action = lambda: c.note_consider(c.value, c.losses)
    else:
        action = c.gradient
    with pytest.raises(ValueError):
        action()
    assert c.used is False


def test_foreign_binding_thread_refused_before_model_access():
    b = SimpleNamespace(poisoned=False, pid=os.getpid(), thread=threading.get_ident()+1)
    with pytest.raises(ValueError, match='Foreign'):
        IteratedBinding.common(b)


def test_no_grad_scoring_cannot_run_inside_a_gradient_context():
    b = SimpleNamespace(check_idle=lambda: None)
    with pytest.raises(ValueError, match='no-grad'):
        IteratedBinding.no_grad(b, None, None, 'score')


def test_four_step_limit_refuses_before_reference_or_route():
    b = SimpleNamespace(check_idle=lambda: None, step=4)
    with pytest.raises(ValueError, match='Four fresh'):
        IteratedBinding.context(b, None)


def test_persistent_ledger_records_consumption_and_refuses_retries(tmp_path):
    ledger = ObjectiveLedger(tmp_path, 'paired')
    with pytest.raises(ValueError, match='budget/order'):
        ledger.begin('candidate', 0)
    for i in range(4):
        ledger.begin('original', i)
        ledger.begin('candidate', i)
    assert len(list(tmp_path.glob('started-*.json'))) == 8
    with pytest.raises(ValueError, match='exhausted'):
        ledger.begin('original', 4)
    second = ObjectiveLedger(tmp_path, 'paired')
    with pytest.raises(FileExistsError):
        second.begin('original', 0)


def test_claim_and_receipt_writes_are_private_and_exclusive(tmp_path):
    previous = os.umask(0)
    try:
        p = tmp_path/'new.json'
        write_json(p, {'simulation': True})
        assert p.stat().st_mode & 0o777 == 0o600
        with pytest.raises(FileExistsError):
            write_json(p, {'overwrite': True})
    finally:
        os.umask(previous)


def test_failed_scope_unwinds_and_publishes_partial_without_retry(tmp_path):
    # A simulated post-forward failure exercises actual production cleanup.
    c = fake_capture()
    c.folder, c.stack, c.arrays = tmp_path, ExitStack(), {}
    restored = []
    c.stack.callback(lambda: restored.append('private-alias-restored'))
    c.route = SimpleNamespace(scopes=[], rows=[], refusals=[])
    c.candidate = False
    c.binding.parameters, c.binding.state = [], 'simulation'
    c.binding.check_idle = lambda: None
    c.__exit__(RuntimeError, RuntimeError('simulated after-forward failure'), None)
    assert restored == ['private-alias-restored'] and c.closed and c.binding.poisoned
    r = json.loads((tmp_path/'receipt.json').read_bytes())
    assert not r['complete'] and not r['restoration_passed'] and not r['CUDA_qualified']


def test_entrypoint_passes_required_arguments_without_model_execution(monkeypatch, tmp_path):
    path = ROOT/'scripts/check_predictor_pgd_cpu_v1.py'
    spec = importlib.util.spec_from_file_location('new_pgd_entry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    got = []
    monkeypatch.setattr(module, 'run', lambda *args: got.append(args) or 0)
    assert module.main(['--stage', str(tmp_path), '--mode', 'paired', '--snapshot', str(tmp_path/'s.json')]) == 0
    assert got == [(tmp_path, 'paired', tmp_path/'s.json')]


def test_independent_reader_has_no_model_Torch_process_or_network_imports():
    tree = ast.parse((ROOT/'models/predictor_pgd_records_v1.py').read_text())
    imports = {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imports |= {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imports <= {'collections', 'hashlib', 'json', 'pathlib', 'numpy'}
    assert saved.agreement([1.0], [1.0], saved.INPUT_BOUND)['passed']
    assert not saved.agreement([1.1], [1.0], saved.INPUT_BOUND)['passed']


def test_saved_reader_rejects_incomplete_rollout_before_arrays(tmp_path):
    write_json(tmp_path/'receipt.json', {'complete': False})
    with pytest.raises(ValueError, match='complete local-only'):
        saved.review(tmp_path)
