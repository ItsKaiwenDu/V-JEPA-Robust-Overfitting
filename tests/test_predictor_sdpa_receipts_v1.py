"""Saved-data regression tests; never run model, CUDA, oracle or a subprocess."""
import ast
import builtins
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess

import numpy as np
import pytest

from models import attention_saved_evidence as arithmetic
from models import predictor_sdpa_component_v6 as frozen
from models import predictor_sdpa_component_records_v1 as rec
from models import predictor_sdpa_numerical_receipts_v1 as policy
from models import predictor_sdpa_saved_reader_v1 as reader

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'runs/pretrained/diagnostic/predictor-sdpa-continuation-v5-20261113'
TARGET = RAW/'linux10603-final/attended-vlbwjaqq/saved'
CPU = RAW/'pytest-first/components0/cohort'
COHORTS = (CPU, RAW/'pytest-first/test_actual_payload_import_pat0/packaged-CPU',
           RAW/'pytest-packaged-correction/test_actual_payload_import_pat0/packaged-CPU')


@pytest.fixture(autouse=True)
def no_new_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Saved-data tests forbid model, oracle, profile, CUDA, process and network work.')
    import torch
    from models import predictor_sdpa_raw_capture_v1 as raw
    for name in ('run', '_execute', '_operands', '_iteration', 'publish_reference'):
        monkeypatch.setattr(frozen, name, forbidden)
    monkeypatch.setattr(arithmetic, 'oracle', forbidden)
    monkeypatch.setattr(raw, 'capture', forbidden)
    monkeypatch.setattr(raw, 'capture_results', forbidden)
    monkeypatch.setattr(torch.nn.Module, '__call__', forbidden)
    monkeypatch.setattr(torch.autograd, 'grad', forbidden)
    monkeypatch.setattr(torch.autograd, 'backward', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.profiler, 'profile', forbidden)
    monkeypatch.setattr(torch.nn.functional, 'scaled_dot_product_attention', forbidden)
    for name in ('_lazy_init', 'init', 'is_available', 'synchronize', 'get_device_properties'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    for name in ('Popen', 'run', 'call', 'check_output'):
        monkeypatch.setattr(subprocess, name, forbidden)
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(os, 'system', forbidden)
    monkeypatch.setattr(os, 'fork', forbidden)


def sample_checks():
    gold = np.arange(4, dtype=np.float64).reshape(1, 1, 2, 2) + 2
    actual = gold + 1e-7
    return {name: arithmetic.agreement(actual, gold, policy.BOUNDS[name]) for name in policy.RETURNS}


def resign(folder, worker):
    data = rec.encoded(worker)
    (folder/'worker.json').write_bytes(data)
    (folder/'worker.json').chmod(0o600)
    (folder/'complete.json').write_bytes(rec.encoded({'kind': worker['kind'], 'plan_sha256': worker['plan_sha256'],
        'complete': True, 'header': {'file': 'worker.json', 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}))
    (folder/'complete.json').chmod(0o600)


def test_policy_is_cold_stdlib_only_and_uses_original_constants():
    source = (ROOT/'models/predictor_sdpa_numerical_receipts_v1.py').read_text()
    imports = [n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 1 and isinstance(imports[0], ast.Import) and imports[0].names[0].name == 'math'
    original_import = builtins.__import__
    def cold_import(name, *args, **kwargs):
        assert name == 'math'
        return original_import(name, *args, **kwargs)
    scope = {'__builtins__': {**vars(builtins), '__import__': cold_import}}
    exec(compile(source, '<cold-stdlib-receipt-policy>', 'exec'), scope)
    checks = sample_checks()
    assert scope['compare_checks'](checks, copy.deepcopy(checks), context='canonical-FP64')['comparison_passed']
    assert policy.BOOK == frozen.BOOK and policy.BOUNDS == frozen.BOUNDS


@pytest.mark.parametrize('context', ('canonical-FP64', 'passing-original'))
@pytest.mark.parametrize('direction', (-math.inf, math.inf))
def test_last_bit_diagnostic_changes_pass_without_mutating_or_weakening_acceptance(context, direction):
    actual = sample_checks(); recorded = copy.deepcopy(actual)
    for name in policy.RETURNS:
        recorded[name]['relative_l2_error'] = math.nextafter(actual[name]['relative_l2_error'], direction)
    before = copy.deepcopy((actual, recorded))
    result = policy.compare_checks(actual, recorded, context=context)
    assert result['all_acceptance_fields_exact'] and result['original_elementwise_bounds_passed']
    assert all(not item['exact_summary_match'] for item in result['relative_l2'].values())
    assert (actual, recorded) == before
    assert not result['producer_records_rewritten'] and not result['worker_completion_inferred']


@pytest.mark.parametrize('context', ('canonical-FP64', 'passing-original'))
@pytest.mark.parametrize('factor,passes', ((.99, True), (1.01, False)))
def test_existing_bookkeeping_boundary_is_enforced(context, factor, passes):
    actual = sample_checks(); recorded = copy.deepcopy(actual)
    for name in policy.RETURNS:
        value = actual[name]['relative_l2_error']
        recorded[name]['relative_l2_error'] += factor * (policy.BOOK['atol'] + policy.BOOK['rtol'] * abs(value))
    if passes:
        assert policy.compare_checks(actual, recorded, context=context)['comparison_passed']
    else:
        with pytest.raises(ValueError, match='bookkeeping bound'):
            policy.compare_checks(actual, recorded, context=context)


@pytest.mark.parametrize('side', ('actual', 'recorded'))
@pytest.mark.parametrize('field,value', (('passed', False), ('passed', 1), ('failed_elements', 1),
    ('failed_elements', False), ('shape', [1, 1, 4, 1]), ('shape', [True, 1, 2, 2]),
    ('shape', [1, 1, 0, 4]), ('rtol', 0.001), ('atol', 3e-5),
    ('max_absolute_error', 2e-7), ('max_tolerance_ratio', 0.12345), ('max_tolerance_ratio', 1.01)))
def test_acceptance_fields_cannot_use_approximate_summary_rule(side, field, value):
    actual = sample_checks(); recorded = copy.deepcopy(actual)
    (actual if side == 'actual' else recorded)['output'][field] = value
    with pytest.raises(ValueError):
        policy.compare_checks(actual, recorded, context='canonical-FP64')


@pytest.mark.parametrize('field', ('relative_l2_error', 'max_absolute_error', 'max_tolerance_ratio', 'rtol', 'atol'))
@pytest.mark.parametrize('value', (float('nan'), float('inf'), -1.0, True, '0.0', None))
def test_malformed_or_nonfinite_metrics_refuse(field, value):
    actual = sample_checks(); recorded = copy.deepcopy(actual)
    recorded['output'][field] = value
    with pytest.raises(ValueError):
        policy.compare_checks(actual, recorded, context='canonical-FP64')


@pytest.mark.parametrize('fault', ('missing-array', 'extra-array', 'missing-field', 'extra-field', 'bad-context', 'bad-container'))
def test_receipt_schema_is_exact(fault):
    actual = sample_checks(); recorded = copy.deepcopy(actual); context = 'canonical-FP64'
    if fault == 'missing-array':del recorded['dv']
    elif fault == 'extra-array':recorded['extra'] = copy.deepcopy(recorded['dv'])
    elif fault == 'missing-field':del recorded['dv']['relative_l2_error']
    elif fault == 'extra-field':recorded['dv']['ignored'] = 0
    elif fault == 'bad-container':recorded = list(recorded.values())
    else:context = 'unbound'
    with pytest.raises(ValueError):policy.compare_checks(actual, recorded, context=context)


def test_zero_error_must_keep_exact_zero_metrics():
    gold = np.ones((1, 1, 2, 2))
    checks = {n: arithmetic.agreement(gold, gold, policy.BOUNDS[n]) for n in policy.RETURNS}
    assert policy.compare_checks(checks, copy.deepcopy(checks), context='canonical-FP64')['comparison_passed']
    recorded = copy.deepcopy(checks);recorded['output']['relative_l2_error'] = 1e-16
    with pytest.raises(ValueError, match='exact zero'):
        policy.compare_checks(checks, recorded, context='canonical-FP64')


def test_failed_actual_arrays_cannot_be_admitted_by_forged_passing_summaries():
    gold = np.ones((1, 1, 2, 2))
    bad = {n: arithmetic.agreement(gold + 1, gold, policy.BOUNDS[n]) for n in policy.RETURNS}
    with pytest.raises(ValueError, match='elementwise acceptance'):
        policy.compare_checks(bad, sample_checks(), context='canonical-FP64')


def test_deep_reader_AST_changes_only_two_summary_comparisons_and_audit_plumbing():
    source = ast.parse((ROOT/'models/predictor_sdpa_component_v6.py').read_text())
    old = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'load_worker')
    gates = {}
    for n in ast.walk(old):
        if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp):
            s = ast.unparse(n.test)
            if s.startswith("checks != w.get('oracle_checks')"):
                gates['checks'] = n
            if s.startswith("actual_checks != w.get('passing_original_checks')"):
                gates['actual_checks'] = n
    assert len(gates) == 2
    new = next(n for n in ast.parse((ROOT/'models/predictor_sdpa_saved_reader_v1.py').read_text()).body
               if isinstance(n, ast.FunctionDef) and n.name == '_load_worker')
    class Normalize(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.name = 'load_worker';node.args.kwonlyargs = [];node.args.kw_defaults = []
            return self.generic_visit(node)
        def visit_Expr(self, node):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == 'reports' and call.func.attr == 'append'):
                name = call.args[0].args[0].id
                return copy.deepcopy(gates[name])
            return self.generic_visit(node)
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == '_load_worker':
                node.func.id = 'load_worker';node.keywords = [x for x in node.keywords if x.arg != 'reports']
            return node
    assert ast.dump(Normalize().visit(new), include_attributes=False) == ast.dump(old, include_attributes=False)


@pytest.mark.parametrize('cohort', COHORTS, ids=('plain', 'first-packaged', 'corrected-packaged'))
@pytest.mark.parametrize('index', range(4))
def test_complete_saved_CPU_cohorts_keep_math_serial_and_fresh_replay_checks(cohort, index):
    result = reader.assess_saved_worker(cohort/str(index), cohort/'component-plan.json', index)
    assert result['assessment_passed'] and result['local_CPU_fixture']
    assert result['arm'] == frozen.CELLS[index][0] and result['replay'] == frozen.CELLS[index][1]
    assert result['resource_receipts'] == []
    assert not result['new_worker_or_cohort_admitted'] and not result['fresh_execution_or_reference_build']
    assert all(x['all_acceptance_fields_exact'] for x in result['numerical_receipt_comparisons'])
    if index in (1, 3):
        assert any(x['context'] == 'passing-original' for x in result['numerical_receipt_comparisons'])


def test_actual_saved_target_satisfies_new_retrospective_reader_without_changing_old_rejection():
    pins = json.loads((RAW/'target-review/target-input-pins.json').read_text())
    result = reader.assess_saved_worker(TARGET/'results/components/0', TARGET/'results/components/component-plan.json',
                                       0, capture_root=TARGET/'results/capture')
    assert result['assessment_passed'] and not result['local_CPU_fixture']
    assert result['resource_receipts'][0]['peak_allocator_bytes'] == 50534400
    assert result['backend_evidence']['observed_family'] == 'efficient'
    assert result['backend_evidence']['metadata_only'] and not result['backend_evidence']['actual_CUDA_execution_admitted']
    assert any(not x['exact_summary_match'] for x in result['numerical_receipt_comparisons'][0]['relative_l2'].values())
    assert not result['historical_claim_or_parent_status_changed'] and not result['new_worker_or_cohort_admitted']
    plan = frozen.read(TARGET/'results/components/component-plan.json')
    ps = rec.sha(TARGET/'results/components/component-plan.json')
    plan.update(repository=str(ROOT), reference_dir=str(TARGET/'results/components/reference'), capture_root=str(TARGET/'results/capture'))
    identity = frozen.capture(frozen.validate_plan(plan))[3]
    with pytest.raises(ValueError, match='Actual saved component/reference numerical bound rejected'):
        frozen.load_worker(TARGET/'results/components/0', plan, ps, identity, 0)
    for name, item in pins.items():
        p = ROOT/name;s = p.stat()
        assert (s.st_size, s.st_mode & 0o777, s.st_mtime_ns, rec.sha(p)) == (item['bytes'], item['mode'], item['mtime_ns'], item['sha256'])


@pytest.mark.parametrize('fault', ('runtime', 'input-layout', 'profiled-hash', 'trace-policy', 'backend-proof',
                                  'flags', 'native-baseline', 'capture-identity', 'predecessors', 'startup'))
def test_rehashed_saved_completion_cannot_bypass_other_deep_gates(tmp_path, fault):
    cohort = tmp_path/'copy';shutil.copytree(CPU, cohort)
    path = cohort/'0';w = frozen.read(path/'worker.json')
    if fault == 'runtime':w['runtime']['threads'] += 1
    elif fault == 'input-layout':w['actual_inputs']['q']['stride'][0] += 1
    elif fault == 'profiled-hash':w['outputs']['profiled']['output']['logical_sha256'] = 'a'*64
    elif fault == 'trace-policy':w['trace']['metadata_policy'] = 'foreign'
    elif fault == 'backend-proof':w['backend_evidence']['observed_family'] = 'foreign'
    elif fault == 'flags':w['flags_restored'] = False
    elif fault == 'native-baseline':w['native_baseline_exact'] = False
    elif fault == 'capture-identity':w['capture_identity']['boundary_sha256'] = 'b'*64
    elif fault == 'predecessors':w['previous_receipts'] = [{'fake': True}]
    else:w['startup']['configure_calls'] = 2
    resign(path, w)
    with pytest.raises(ValueError):reader.assess_saved_worker(path, cohort/'component-plan.json', 0)


@pytest.mark.parametrize('field,index', (('oracle_checks', 0), ('passing_original_checks', 1)))
def test_deep_reader_checks_diagnostic_bound_at_both_comparison_sites(tmp_path, field, index):
    cohort = tmp_path/'copy';shutil.copytree(CPU, cohort);path = cohort/str(index)
    w = frozen.read(path/'worker.json');w[field]['output']['relative_l2_error'] += 1e-5;resign(path, w)
    with pytest.raises(ValueError, match='bookkeeping bound'):
        reader.assess_saved_worker(path, cohort/'component-plan.json', index)


def test_rehashed_actual_array_violation_still_fails_original_elementwise_gate(tmp_path):
    cohort = tmp_path/'copy';shutil.copytree(CPU, cohort);path = cohort/'0';w = frozen.read(path/'worker.json')
    for phase in ('untimed', 'profiled'):
        item = w['outputs'][phase]['dq'];p = path/item['file'];a = np.load(p, allow_pickle=False)
        a.flat[0] += 1.0
        with p.open('wb') as f:np.save(f, a, allow_pickle=False)
        item.update(bytes=p.stat().st_size, sha256=rec.sha(p), logical_sha256=hashlib.sha256(a.tobytes()).hexdigest())
    resign(path, w)
    with pytest.raises(ValueError, match='elementwise acceptance'):
        reader.assess_saved_worker(path, cohort/'component-plan.json', 0)


@pytest.mark.parametrize('fault', ('fresh-reference', 'duplicate-pid', 'partial-marker', 'foreign-source'))
def test_fresh_replay_publication_and_source_guards_remain(tmp_path, fault):
    cohort = tmp_path/'copy';shutil.copytree(CPU, cohort);path = cohort/'2';w = frozen.read(path/'worker.json')
    if fault == 'fresh-reference':w['fresh_replay_reference_sha256'] = 'f'*64;resign(path, w)
    elif fault == 'duplicate-pid':
        w['pid'] = frozen.read(cohort/'0/worker.json')['pid'];w['startup']['pid'] = w['pid'];resign(path, w)
    elif fault == 'partial-marker':(path/'events.jsonl.partial').write_bytes(b'partial')
    else:
        p = frozen.read(cohort/'component-plan.json');p['source_sha256']['models/attention_saved_evidence.py'] = 'f'*64
        (cohort/'component-plan.json').write_bytes(rec.encoded(p))
    with pytest.raises(ValueError):reader.assess_saved_worker(path, cohort/'component-plan.json', 2)
