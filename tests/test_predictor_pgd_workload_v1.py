"""New local full-unit controls. Old suites/cohorts are never called here."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from attacks import pgd_workload_v1 as attack
from models import candidate_evaluation as original
from models import predictor_pgd_workload_v1 as workload
from models.predictor_pgd_workload_controls_v1 import AnalyticFactory, AnalyticRoute, analytic_fixture
from models import predictor_pgd_workload_records_v1 as records

ROOT = Path(__file__).resolve().parents[1]
VIDEO = 'local-workload-control/primary-convergence-v1'


@pytest.fixture
def case(request, tmp_path):
    base = os.environ.get('VJEPA_PGD_WORKLOAD_EVIDENCE')
    folder = (Path(base)/request.node.name if base else tmp_path/'case')
    folder.mkdir(mode=0o700)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield folder
    finally:
        torch.set_num_threads(old)


def named(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_frozen_PGD_body_inverse():
    new = named(ROOT/'attacks/pgd_workload_v1.py', 'pgd_linf_workload_v1')
    old = named(ROOT/'attacks/pgd.py', 'pgd_linf')
    new.name = old.name; new.args.kwonlyargs.pop(); new.args.kw_defaults.pop()
    new.body = [n for n in new.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Name) and n.value.func.id == 'local_domain')]
    class Undo(ast.NodeTransformer):
        def visit_With(self, node):
            if ast.unparse(node.items[0].context_expr) == 'gradient_context(current)':
                assert [ast.unparse(n) for n in node.body] == [
                    'losses = _losses(lambda _: step.losses, current)', 'consider(current, losses)',
                    'step.note_consider(current, losses)', 'gradient = checked_gradient(step.gradient(), current)']
                return ast.parse('losses = _losses(objective, current)\nconsider(current, losses)\ngradient, = torch.autograd.grad(losses.sum(), current)').body
            return self.generic_visit(node)
    assert ast.dump(Undo().visit(new)) == ast.dump(old)
    assert workload.file_hash(ROOT/'attacks/pgd.py') == attack.FROZEN_PGD_SHA256


def test_frozen_evaluator_flow_inverse():
    new = named(ROOT/'models/predictor_pgd_workload_evaluation_v1.py', 'evaluate_local_unit')
    old = named(ROOT/'models/candidate_evaluation.py', 'evaluate_clip')
    new.name = old.name; new.args.kwonlyargs.pop(); new.args.kw_defaults.pop()
    class Undo(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'binding.prepare':
                assert ast.unparse(node.value) == 'binding.prepare(model, clean, targets)'
                return None
            return self.generic_visit(node)
        def visit_Call(self, node):
            if ast.unparse(node.func) == 'binding.objective':
                node.func = ast.Name(id='native_objective', ctx=ast.Load())
            elif ast.unparse(node.func) == 'pgd_linf_workload_v1':
                assert node.keywords[-1].arg == 'gradient_context'
                assert ast.unparse(node.keywords[-1].value) == 'binding.context'
                node.keywords.pop(); node.func = ast.Name(id='pgd_linf', ctx=ast.Load())
            return self.generic_visit(node)
    assert ast.dump(Undo().visit(new)) == ast.dump(old)


@pytest.mark.parametrize('method,audits', [('primary', 0), ('convergence', 2)])
def test_complete_units_match_original_selection_scoring_restoration(case, method, audits):
    model, clean, masks = analytic_fixture()
    # Mixed caller modes must return exactly, even though execution uses eval mode.
    model.encoder.train(); model.predictor.eval()
    modes = [(m, m.training) for m in model.modules()]
    initial_rng = torch.get_rng_state().clone()
    settings = original.EvaluationSettings.for_method(method)
    original_metrics, expected = original.evaluate_clip(model, clean, masks, settings,
                            video_id=VIDEO, sensitivity_banks=audits)
    assert torch.equal(torch.get_rng_state(), initial_rng)
    unit = workload.LocalUnit(factory=AnalyticFactory())
    row, actual = unit.run(model, clean, masks, settings, video_id=VIDEO,
                          sensitivity_banks=audits, journal=case/'unit')
    assert torch.equal(actual, expected)
    assert {k: v for k, v in row['metrics'].items() if k != 'seconds'} == {
        k: v for k, v in original_metrics.items() if k != 'seconds'}
    assert len(row['steps']) == settings.steps*2
    assert len({r['pixels'] for r in row['steps']}) > 2
    assert all(m.training == mode for m, mode in modes)
    assert torch.equal(torch.get_rng_state(), initial_rng)
    assert all(p.grad is None for p in model.parameters())
    assert row['overhead_subtracted'] is False
    assert records.validate_receipt(row)['complete_gradients'] == settings.steps*2
    workload.write_json(case/'matched-original.json', {'metrics': original_metrics,
                         'adversarial': workload.tensor_hash(expected), 'route': 'original analytic model',
                         'timing_is_CPU_control_only': True})
    with pytest.raises(ValueError, match='consumed'):
        unit.run(model, clean, masks, settings, video_id=VIDEO, journal=case/'retry')


@pytest.mark.parametrize('change', ['routine', 'wrong_video', 'audits', 'big', 'nan', 'inference'])
def test_domain_refuses_before_any_unit_start(case, change):
    model, clean, masks = analytic_fixture()
    settings = original.EvaluationSettings.for_method('routine' if change == 'routine' else 'primary')
    video = 'research-clip' if change == 'wrong_video' else VIDEO
    if change == 'big':
        clean = torch.full((1, 4096), .5)
    if change == 'nan':
        clean[0, 0, 0, 0] = float('nan')
    guard = torch.inference_mode() if change == 'inference' else torch.enable_grad()
    with guard, pytest.raises(ValueError):
        workload.LocalUnit(factory=AnalyticFactory()).run(model, clean, masks, settings,
                video_id=video, sensitivity_banks=1 if change == 'audits' else 0, journal=case/'unit')
    assert not (case/'unit').exists()


@pytest.mark.parametrize('mode', ['throw', 'nan', 'shape', 'parameters', 'mask'])
def test_gradient_failure_consumes_start_and_stops_unit(case, monkeypatch, mode):
    model, clean, masks = analytic_fixture()
    settings = original.EvaluationSettings.for_method('primary')
    real = AnalyticRoute.gradient
    def fail(route, root, value):
        if mode == 'throw':
            raise RuntimeError('new analytic control failure')
        if mode == 'parameters':
            with torch.no_grad():
                model.encoder.weight.add_(.01)
        if mode == 'mask':
            with torch.no_grad():
                unit.binding.attack_bank[0][0].context.add_(1)
        gradient = real(route, root, value)
        if mode == 'nan':
            gradient.fill_(float('nan'))
        return gradient.flatten() if mode == 'shape' else gradient
    monkeypatch.setattr(AnalyticRoute, 'gradient', fail)
    unit = workload.LocalUnit(factory=AnalyticFactory())
    rng = torch.get_rng_state().clone(); modes = [(m, m.training) for m in model.modules()]
    with pytest.raises((ValueError, RuntimeError)):
        unit.run(model, clean, masks, settings, video_id=VIDEO, journal=case/'unit')
    assert unit.state == 'failed' and unit.binding.poisoned
    assert unit.binding.started == 1 and not unit.binding.steps and unit.binding.active is None
    assert all(m.training == mode for m, mode in modes) and torch.equal(torch.get_rng_state(), rng)
    assert not (case/'unit/unit-complete.json').exists()
    failure = json.loads((case/'unit/unit-failed.json').read_text())
    assert failure['gradient_starts'] == 1 and failure['consumed']
    assert failure['restored'] is (mode not in ('parameters', 'mask'))
    with pytest.raises(ValueError, match='consumed'):
        unit.run(model, clean, masks, settings, video_id=VIDEO, journal=case/'retry')


@pytest.mark.parametrize('misuse', ['before_consider', 'foreign_input', 'foreign_loss', 'duplicate'])
def test_exact_context_tokens_and_consider_order(case, misuse):
    model, clean, masks = analytic_fixture()
    settings = original.EvaluationSettings.for_method('primary')
    journal = workload.Journal(case/'unit')
    binding = workload.Binding(model, clean, settings, AnalyticFactory(), journal)
    binding.prepare(model, clean, model.clean_targets(clean))
    binding.objective(model, binding.targets, [masks(i) for i in range(8)])
    value = clean.clone().requires_grad_(True)
    with pytest.raises(ValueError):
        with binding.context(value) as step:
            if misuse == 'before_consider':
                step.gradient()
            elif misuse == 'foreign_input':
                step.note_consider(value.clone(), step.losses)
            elif misuse == 'foreign_loss':
                step.note_consider(value, step.losses.clone())
            else:
                step.note_consider(value, step.losses)
                step.note_consider(value, step.losses)
    assert binding.poisoned and binding.active is None and binding.started == 1 and not binding.steps


def test_no_grad_scoring_refuses_inside_context(case):
    model, clean, masks = analytic_fixture()
    settings = original.EvaluationSettings.for_method('primary')
    binding = workload.Binding(model, clean, settings, AnalyticFactory(), workload.Journal(case/'unit'))
    binding.prepare(model, clean, model.clean_targets(clean))
    score = binding.objective(model, binding.targets, [masks(i) for i in range(8)])
    with pytest.raises(ValueError, match='no-grad'):
        score(clean)
    with pytest.raises(ValueError, match='active'):
        with binding.context(clean.clone().requires_grad_(True)):
            with torch.no_grad():
                score(clean)
    assert binding.poisoned


def test_fresh_process_full_primary_replay(case):
    rows, pids = [], []
    for index in range(2):
        folder = case/f'worker-{index}'
        command = [sys.executable, '-I', '-B', str(ROOT/'scripts/check_predictor_pgd_workload_v1.py'),
                   '--method', 'primary', '--journal', str(folder)]
        with (case/f'worker-{index}.log').open('xb') as log:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=45, check=False,
                                   env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        assert child.returncode == 0, (case/f'worker-{index}.log').read_text()
        rows.append(json.loads((folder/'unit-complete.json').read_text()))
        pids.append(json.loads((folder/'unit-started.json').read_text())['pid'])
    assert len(set(pids)) == 2 and os.getpid() not in pids
    result = records.compare_replay(*rows)
    workload.write_json(case/'replay.json', {**result, 'actual_fresh_process_pids': pids,
                                           'parent_pid': os.getpid(), 'child_exits': [0, 0]})
    # Reader mutations use copied saved records; no additional model execution.
    for mutate in ('gradient', 'score', 'strength', 'admission', 'native_claim', 'time', 'start'):
        broken = copy.deepcopy(rows[1])
        if mutate == 'gradient': broken['steps'][1]['gradient'] = '0'*64
        elif mutate == 'score': broken['scores'][0]['route_active'] = True
        elif mutate == 'strength': broken['metrics']['settings']['steps'] = 19
        elif mutate == 'admission': broken['research_admitted'] = True
        elif mutate == 'native_claim': broken['steps'][0]['route']['native_ownership_proven'] = True
        elif mutate == 'time': broken['metrics']['seconds'] = float('nan')
        elif mutate == 'start': broken['metrics']['seeds']['start'] += 1
        with pytest.raises(ValueError):
            records.compare_replay(rows[0], broken)


def test_one_new_frozen_CPU_bridge_objective(case):
    # New weights, pixels and banks, ONE objective only; no closed PGD2/R2 rerun.
    from models.execution import configure_execution, apply_model_policy
    from models.native_diagnostic import tiny_fixture
    configure_execution('shared-fp64-native', device='cpu')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026112302)
        model, _, masks = tiny_fixture(ROOT/'external/vjepa2')
    apply_model_policy(model, 'shared-fp64-native'); model.eval()
    pixels = .2+.6*torch.rand((1, 3, 4, 16, 16), generator=torch.Generator().manual_seed(2026112303))
    targets = model.clean_targets(pixels)
    settings = original.EvaluationSettings.for_method('primary')
    binding = workload.Binding(model, pixels, settings, workload.FrozenCPUFactory(), workload.Journal(case/'one-objective'))
    binding.prepare(model, pixels, targets)
    binding.objective(model, targets, [masks(2026112304+i) for i in range(8)])
    value = pixels.clone().requires_grad_(True)
    with binding.context(value) as step:
        step.note_consider(value, step.losses)
        gradient = step.gradient()
        assert gradient.shape == pixels.shape and gradient.abs().max() > 0
    binding.idle()
    assert binding.started == len(binding.steps) == 1
    workload.write_json(case/'control.json', {'kind': 'one-new-local-frozen-CPU-bridge-control-v1',
        'seed_weights': 2026112302, 'seed_pixels': 2026112303, 'seed_masks': 2026112304,
        'steps': binding.steps, 'actual_owned_objectives': 1, 'full_rollout_executed': False,
        'pretrained_or_GPU_or_research_qualified': False, 'torch_version': str(torch.__version__)})
