"""CPU metadata/failure guards using genuine retained reference metadata.

These tests execute no model or CUDA and cannot qualify a candidate GPU.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get('SHADOW_TEST_PACKAGE', str(ROOT)))
spec = importlib.util.spec_from_file_location('shadow_metadata_tests', ROOT / 'models/cross_host_replay.py')
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)
FIXTURE = json.loads((ROOT / 'tests/fixtures/cross_host_reference_records.json').read_text())


def plan():
    return {'kind': ops.PLAN_KIND, 'candidate_host': ops.HOST, 'candidate_GPU_identity': ops.GPU,
        'reference_host': ops.REFERENCE_HOST, 'reference_GPU_identity': ops.REFERENCE_GPU,
        'reference_plan_sha256': ops.REFERENCE_PLAN, 'handoff_manifest_sha256': ops.MANIFEST,
        'reference_case_id': ops.REFERENCE_CASE, 'arm': 'A', 'fresh_workers': 2,
        'whole_attempt_seconds': 1200, 'worker_seconds': 510, 'kill_grace_seconds': 15,
        'max_attempts': 1, 'automatic_GPU_retries': 0, 'panel_admitted': False,
        'remaining_queue': False, 'research_admitted': False, 'paid_compute': False,
        'reference_execution': copy.deepcopy(FIXTURE['reference_execution']),
        'reference_case_directory': 'runs/panel.qIXeDn/cases/clip-02-bank-1/attempt-01',
        'shadow_sources_sha256': {name: '1' * 64 for name in ops.SOURCES},
        'reference_worker_files_sha256': {name: '2' * 64 for name in ops.WORKER_FILES}}


def records():
    original = copy.deepcopy(FIXTURE['contract'])
    execution = ops.candidate_execution(plan())
    candidate = copy.deepcopy(original)
    candidate['execution_sha256'] = ops.canonical(execution)
    row = copy.deepcopy(FIXTURE['A'])
    observed = copy.deepcopy(row)
    observed['contract_sha256'] = ops.canonical(candidate)
    return original, candidate, row, observed, execution


def test_declared_shadow_request_is_metadata_only_and_bounded():
    asked = plan()
    assert ops.validate_plan(asked, ops.canonical(asked)) == asked
    assert asked['panel_admitted'] is False and asked['fresh_workers'] == 2


@pytest.mark.parametrize('field,value', [
    ('candidate_host', 'linux10609'), ('candidate_GPU_identity', ops.REFERENCE_GPU),
    ('reference_host', ops.HOST), ('reference_case_id', 'clip-00-bank-0'),
    ('arm', 'D'), ('fresh_workers', 3), ('fresh_workers', True),
    ('whole_attempt_seconds', 1201), ('worker_seconds', 511), ('kill_grace_seconds', 16),
    ('max_attempts', 2), ('automatic_GPU_retries', 1), ('panel_admitted', True),
    ('remaining_queue', True), ('research_admitted', True), ('paid_compute', True),
    ('reference_case_directory', '../panel'), ('reference_plan_sha256', '0' * 64)])
def test_request_cannot_extend_time_switch_case_or_admit_panel(field, value):
    asked = plan(); asked[field] = value
    with pytest.raises(ValueError): ops.validate_plan(asked, ops.canonical(asked))


def test_self_consistent_request_change_still_needs_the_external_pin():
    asked = plan(); pin = ops.canonical(asked); asked['note'] = 'changed'
    with pytest.raises(ValueError, match='externally pinned'): ops.validate_plan(asked, pin)


@pytest.mark.parametrize('key', ['shadow_sources_sha256', 'reference_worker_files_sha256'])
def test_missing_source_or_reference_inventory_rejects(key):
    asked = plan(); asked[key].pop(next(iter(asked[key])))
    with pytest.raises(ValueError): ops.validate_plan(asked, ops.canonical(asked))


def test_candidate_identity_records_actual_host_without_mutating_reference():
    asked = plan(); before = copy.deepcopy(asked['reference_execution'])
    candidate = ops.candidate_execution(asked)
    assert asked['reference_execution'] == before
    assert candidate['expected_host'] == ops.HOST and candidate['expected_GPU_identity'] == ops.GPU
    assert candidate['kind'] == 'cross-host-shadow-execution-v1' and candidate['panel_admitted'] is False
    assert ops.canonical(candidate) != ops.canonical(before)


def test_only_the_candidate_and_an_idle_gpu_pass_metadata_gate():
    ops.validate_hardware(ops.HOST, ops.GPU, '')
    ops.validate_hardware(ops.HOST, ops.GPU, '123\n', allowed_pid=123)


@pytest.mark.parametrize('host,identity,active', [
    ('linux10609', ops.GPU, ''), (ops.HOST, ops.REFERENCE_GPU, ''),
    (ops.HOST, ops.GPU.replace('580.178.04', 'other'), ''),
    (ops.HOST, ops.GPU, '456\n'), (ops.HOST, ops.GPU, '123\n456\n')])
def test_wrong_host_gpu_driver_or_foreign_compute_pid_rejects(host, identity, active):
    with pytest.raises(ValueError): ops.validate_hardware(host, identity, active, allowed_pid=123)


def test_comparator_permits_only_declared_execution_and_instrumentation_changes():
    a, b, row, observed, execution = records()
    observed['instrumented_attack_seconds'] += 1
    observed['artifact']['sha256'] = 'different serialization; logical tensors retained'
    assert ops.compare_records(a, b, row, observed, execution)


@pytest.mark.parametrize('change', ['clean', 'target', 'starts', 'mask', 'runtime', 'settings',
                                   'execution', 'trace-loss', 'gradient', 'score', 'adversary', 'tensor'])
def test_numerical_contract_trace_gradients_and_scoring_cannot_drift(change):
    a, b, row, observed, execution = records()
    if change == 'clean': b['clean_sha256'] = '0' * 64
    elif change == 'target': b['targets_sha256'] = '0' * 64
    elif change == 'starts': b['starts_sha256'][0] = '0' * 64
    elif change == 'mask': b['mask_tensors_sha256'][next(iter(b['mask_tensors_sha256']))] = '0' * 64
    elif change == 'runtime': b['runtime']['threads'] += 1
    elif change == 'settings': b['settings']['epsilon'] = 0
    elif change == 'execution': b['execution_sha256'] = a['execution_sha256']
    elif change == 'trace-loss': observed['trace'][0]['loss'] += .001
    elif change == 'gradient': observed['trace'][1]['gradient_sha256'] = '0' * 64
    elif change == 'score': observed['scores']['main']['clean'] += .001
    elif change == 'adversary': observed['adversarial_sha256'] = '0' * 64
    elif change == 'tensor': observed['artifact']['tensors']['adversarial'] = '0' * 64
    with pytest.raises(ValueError): ops.compare_records(a, b, row, observed, execution)


def test_shadow_attempt_is_exclusive_and_cannot_reset_itself(tmp_path):
    output = ops.reserve_output(tmp_path)
    ops.write_record(output / 'session.json', {'elapsed_seconds': 0, 'complete': False})
    with pytest.raises(FileExistsError): ops.reserve_output(tmp_path)
    assert ops.read_record(output / 'session.json')['elapsed_seconds'] == 0


def test_copied_packet_root_cannot_reset_the_fixed_candidate_operation(tmp_path):
    ops.validate_locations(tmp_path / ops.PACKAGE, tmp_path / ops.PACKET, tmp_path)
    with pytest.raises(ValueError, match='copied-root'):
        ops.validate_locations(tmp_path / ops.PACKAGE, tmp_path / 'other-packet', tmp_path)


def test_duplicate_worker_is_claimed_before_any_cuda_access(tmp_path):
    output = tmp_path / 'worker-0'
    ops.reserve_worker(output)
    assert not output.exists()
    with pytest.raises(FileExistsError): ops.reserve_worker(output)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a/../b', 'a\\b', './name', '.'])
def test_escaped_paths_reject(tmp_path, name):
    with pytest.raises(ValueError): ops.bound(tmp_path, name)


def test_symlink_evidence_rejects(tmp_path):
    (tmp_path / 'link').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError): ops.bound(tmp_path, 'link/file')


def test_record_update_is_checked_and_never_overwrites_without_replace(tmp_path):
    path = tmp_path / 'record.json'
    ops.write_record(path, {'complete': False})
    with pytest.raises(FileExistsError): ops.write_record(path, {'complete': True})
    ops.write_record(path, {'complete': False, 'elapsed_seconds': 2}, replace=True)
    assert ops.read_record(path)['elapsed_seconds'] == 2
    wrapper = json.loads(path.read_text()); wrapper['payload']['complete'] = True
    path.write_text(json.dumps(wrapper))
    with pytest.raises(ValueError): ops.read_record(path)


def test_remaining_allowance_accounts_for_elapsed_time():
    assert ops.remaining_worker_seconds(10, 20) == 510
    assert ops.remaining_worker_seconds(10, 1000) == 150
    with pytest.raises(TimeoutError): ops.remaining_worker_seconds(10, 1200)
    with pytest.raises(TimeoutError): ops.remaining_worker_seconds(10, float('nan'))


@pytest.mark.parametrize('change', ['incomplete', 'failed', 'timeout', 'missing-worker', 'mixed-plan',
                                   'active', 'zero-time', 'extended-time', 'nan-time', 'panel', 'research'])
def test_shadow_completion_refuses_partial_or_mixed_metadata(change):
    # In-memory metadata mutations only; no generated neural/CUDA receipt is saved.
    pin = ops.canonical(plan())
    s = {'kind': ops.KIND + '-session', 'shadow_plan_sha256': pin, 'complete': True, 'passed': True,
         'active_worker': None, 'tasks': {'0': {'complete': True, 'exit_status': 0, 'timed_out': False},
                                        '1': {'complete': True, 'exit_status': 0, 'timed_out': False}},
         'panel_admitted': False, 'research_admitted': False, 'whole_attempt_seconds': 1200, 'elapsed_seconds': 600}
    if change == 'incomplete': s['complete'] = False
    elif change == 'failed': s['tasks']['0']['exit_status'] = 1
    elif change == 'timeout': s['tasks']['1']['timed_out'] = True
    elif change == 'missing-worker': s['tasks'].pop('1')
    elif change == 'mixed-plan': s['shadow_plan_sha256'] = '0' * 64
    elif change == 'active': s['active_worker'] = 1
    elif change == 'zero-time': s['elapsed_seconds'] = 0
    elif change == 'extended-time': s['elapsed_seconds'] = 1201
    elif change == 'nan-time': s['elapsed_seconds'] = float('nan')
    elif change == 'panel': s['panel_admitted'] = True
    elif change == 'research': s['research_admitted'] = True
    with pytest.raises(ValueError): ops.validate_session(s, pin)


def test_wrong_cli_plan_pin_stops_before_output_or_model(tmp_path):
    fake = tmp_path / 'packet'; (fake / 'scripts').mkdir(parents=True)
    (fake / 'shadow-plan.json').write_text(json.dumps(plan()))
    (fake / 'scripts/check_cross_host_replay.py').write_text((ROOT / 'scripts/check_cross_host_replay.py').read_text())
    result = subprocess.run([sys.executable, str(fake / 'scripts/check_cross_host_replay.py'), 'run',
        '--package', str(tmp_path), '--plan', str(fake / 'shadow-plan.json'), '--expected-plan-sha256', '0' * 64],
        capture_output=True, text=True)
    assert result.returncode == 2 and 'external pin' in result.stderr
    assert not (fake / 'runs').exists()


def test_failed_first_child_never_dispatches_second_or_marks_pass(tmp_path, monkeypatch):
    import models.matched_diagnostic as primitive
    home = tmp_path / 'home'; package = home / ops.PACKAGE; package.mkdir(parents=True)
    packet = home / ops.PACKET; packet.mkdir(parents=True)
    asked = plan(); calls = []
    monkeypatch.setattr(ops, 'check_sources', lambda *args: ({}, ops.candidate_execution(asked)))
    monkeypatch.setattr(ops, 'reference_gate', lambda *args: None)
    monkeypatch.setattr(ops, 'hardware_gate', lambda **kwargs: ops.GPU)
    monkeypatch.setattr(ops, 'export_evidence', lambda *args: None)
    monkeypatch.setattr(ops.platform, 'node', lambda: ops.HOST)
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: home))
    def failed(command, log, seconds):
        calls.append(command)
        return {'complete': False, 'timed_out': False, 'exit_status': 7}
    monkeypatch.setattr(primitive, 'bounded_child', failed)
    result = ops.run(package, packet, asked, ops.canonical(asked))
    assert len(calls) == 1 and result['complete'] is False and result['passed'] is False
    assert set(result['tasks']) == {'0'}
    assert not (packet / 'runs/attempt-01/comparison.json').exists()
