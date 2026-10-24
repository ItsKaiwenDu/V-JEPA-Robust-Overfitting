"""Operational accounting/composition guards and actual small-model controls.

Generated CPU fixtures below are not A40 timing or a research result. Ordinary
tests need no private runs, released weights, SSH, or cloud account.
"""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest

import models.hardware_controls as controls
from models.study_artifacts import read_record
from models.training import runtime_contract
from models.workflow_pilot import run_phase
from test_workflow_preflight import context
from test_native_execution import restore_flags
from test_vjepa2 import model

ROOT = Path(__file__).resolve().parents[1]


def ledger():
    return dict(conservative_cumulative_credit_usage_usd='5.47', displayed_total_debits_usd='5.47',
        shared_diagnostic_allowance_usd='10.00', conservative_remaining_diagnostic_allowance_usd='4.53',
        remaining_unsettled_internal_reservation_usd='0.00', last_wallet_screenshot_available_credit_usd='9.53',
        initial_account_credit_usd='15.00', project_cap_usd='200.00')


def plan():
    return dict(profile=controls.PROFILE, ledger_sha256='current-ledger', base_manifest_sha256=controls.BASE_MANIFEST,
        quoted_all_in_hourly_usd='.61', new_session_hold_usd='2.00', deployment_epoch=1000, deadline_epoch=8200,
        fresh_quote_confirmed=True, manual_deployment_authorized=True, host_fingerprint_verified=True,
        expected_gpu='NVIDIA A40', container_gb=30, volume_gb=80, volume_mount='/workspace')


def test_settled_holds_are_not_reused_and_new_hold_fits_real_remaining_allowance():
    assert controls.review_plan(plan(), ledger(), 'current-ledger', 1100) == 7100
    assert controls.review_plan(plan(), ledger(), 'current-ledger', 6000, launching=False) == 2200
    stale = ledger(); stale['conservative_remaining_diagnostic_allowance_usd'] = '2.805'
    with pytest.raises(ValueError, match='Ledger'):
        controls.review_plan(plan(), stale, 'current-ledger', 1100)


@pytest.mark.parametrize('key,value', [
    ('ledger_sha256', 'old'), ('base_manifest_sha256', 'old'), ('deployment_epoch', True),
    ('deployment_epoch', 1200), ('deadline_epoch', 8201), ('deadline_epoch', 8200.0),
    ('quoted_all_in_hourly_usd', 0), ('quoted_all_in_hourly_usd', True),
    ('quoted_all_in_hourly_usd', float('nan')), ('quoted_all_in_hourly_usd', float('inf')),
    ('quoted_all_in_hourly_usd', '.71'), ('new_session_hold_usd', '2.80'),
    ('fresh_quote_confirmed', False), ('manual_deployment_authorized', False),
    ('host_fingerprint_verified', False), ('expected_gpu', 'NVIDIA RTX A6000'),
    ('container_gb', 80), ('volume_gb', 30), ('volume_mount', '/tmp')])
def test_stale_unreviewed_oversized_or_extended_launch_rejects(key, value):
    value_plan = plan(); value_plan[key] = value
    with pytest.raises(ValueError):
        controls.review_plan(value_plan, ledger(), 'current-ledger', 1100)


@pytest.mark.parametrize('now', [999, 4601, 7900, 8200])
def test_idle_and_setup_count_from_deployment_and_export_reserve_is_kept(now):
    with pytest.raises(ValueError, match='deployment-based'):
        controls.review_plan(plan(), ledger(), 'current-ledger', now)


def test_wallet_and_diagnostic_allowance_are_separate_and_unsettled_holds_reject():
    for key, value in (('shared_diagnostic_allowance_usd', '6.00'),
                       ('last_wallet_screenshot_available_credit_usd', '1.00'),
                       ('remaining_unsettled_internal_reservation_usd', '2.80')):
        changed = ledger(); changed[key] = value
        with pytest.raises(ValueError):
            controls.review_plan(plan(), changed, 'current-ledger', 1100)


@pytest.mark.parametrize('defect', ['skipped', 'failures', 'errors', 'count'])
def test_fresh_suite_requires_all509_cases_and_zero_skips(tmp_path, defect):
    attributes = dict(tests='509', failures='0', errors='0', skipped='0')
    attributes['tests' if defect == 'count' else defect] = '508' if defect == 'count' else '1'
    path = tmp_path / 'junit.xml'
    path.write_text('<testsuites><testsuite ' + ' '.join(f'{k}="{v}"' for k, v in attributes.items()) + '/></testsuites>')
    with pytest.raises(ValueError, match='zero skips'):
        controls.suite_gate(path)


def test_input_paths_and_source_pins_reject_aliases_and_drift(tmp_path, monkeypatch):
    base = tmp_path / 'base'; base.mkdir()
    source = base / 'source.txt'; source.write_text('frozen\n')
    index = dict(sha256=controls.sha(source), bytes=source.stat().st_size)
    manifest = base / 'handoff-manifest.json'
    manifest.write_text(json.dumps({'archives': {'source': {'files': {'source.txt': index}}}}))
    monkeypatch.setattr(controls, 'BASE_MANIFEST', controls.sha(manifest))
    assert controls.source_gate(base) == 1
    source.write_text('changed\n')
    with pytest.raises(ValueError, match='checksum'):
        controls.source_gate(base)
    source.unlink(); source.symlink_to(manifest)
    with pytest.raises(ValueError, match='regular'):
        controls.source_gate(base)
    for name in ('../source.txt', '/source.txt', './source.txt', 'a/../source.txt'):
        with pytest.raises(ValueError, match='Unsafe'):
            controls.safe_file(base, name)


def cpu_fixture(context, tmp_path, monkeypatch):
    monkeypatch.setattr('models.workflow_pilot.prepare_context', lambda root, args: copy.deepcopy(context))
    past, current = tmp_path / 'past', tmp_path / 'current'
    args = SimpleNamespace(device='cpu', output=past, hardware_screen_only=True, feature_anchor_variant='reference')
    phases = []
    for phase in controls.OLD_PHASES:
        args.phase = phase
        before = time.monotonic()
        run_phase(ROOT, args)
        phases.append(dict(phase=phase, process_wall_seconds=time.monotonic() - before,
                           result_sha256=controls.sha(past / 'phase-results' / (phase + '.json'))))
    # Retain a generated partial session and an unfinished journal, exactly the
    # two distinct states whose composition the operational wrapper handles.
    session = dict(complete=False, error_type='TimeoutExpired', phases=phases, execution_policy='shared-fp64-native',
                   runtime=runtime_contract('cpu'), device={'device': 'cpu', 'fixture': 'generated-upstream'})
    controls.write_wrapped(past / 'session.json', session)
    shutil.copytree(past / 'calibration', current / 'calibration')
    args.output, args.phase = current, 'controls'
    run_phase(ROOT, args)
    unfinished = read_record(current / 'controls/convergence/evaluation.json')
    unfinished['complete'], unfinished['records'] = False, []
    controls.write_wrapped(past / 'controls/convergence/evaluation.json', unfinished)
    controls.write_wrapped(current / 'session.json', dict(profile=controls.PROFILE, required_phases=['controls'],
        quoted_hourly_usd=.61,
        execution_policy=session['execution_policy'], runtime=session['runtime'], device=session['device'],
        complete=False, phases=[dict(phase='controls', result_sha256=controls.sha(current / 'phase-results/controls.json'))]))
    reference = json.loads((ROOT / 'report/checks/2026-10-22-measured-budget-review.json').read_text())
    return past, current, reference


def test_actual_cpu_controls_combine_unique_observations_preserve_failure_and_never_admit(context, tmp_path, monkeypatch):
    past, current, reference = cpu_fixture(context, tmp_path, monkeypatch)
    old_bytes = {str(p.relative_to(past)): p.read_bytes() for p in past.rglob('*.json')}
    result = controls.combine(past, current, reference, .61)
    assert result['combined_economic_measurements_complete'] and result['historical_timeout_preserved']
    assert result['fresh_observation_counts']['primary'] == 6
    assert result['fresh_observation_counts']['routine'] == 5
    assert all(result['fresh_observation_counts'][m] == 1 for m in controls.CONTROLS)
    assert not any(result[k] for k in ('research_admitted', 'budget_approved', 'complete_cost_estimate',
                                      'exact_optimizer_recovery_checked', 'historical_session_complete'))
    assert old_bytes == {str(p.relative_to(past)): p.read_bytes() for p in past.rglob('*.json')}
    assert not read_record(current / 'session.json')['complete']  # combiner itself does not commit success

    # Each independent mutation remains a checksummed record, so these reject
    # semantic drift rather than merely broken JSON hashes.
    target = current / 'controls/convergence/evaluation.json'
    good = target.read_bytes()
    for defect in ('partial', 'seed', 'audit', 'checkpoint', 'data', 'runtime', 'source', 'step', 'zero', 'nan'):
        value = read_record(target)
        if defect == 'partial': value['complete'] = False
        elif defect == 'seed': value['records'][0]['seeds']['audit_0'][0] = value['records'][0]['seeds']['score'][0]
        elif defect == 'audit': value['records'][0]['scoring_audits'] = [{}, {}]
        elif defect == 'checkpoint': value['contract']['weights_sha256']['predictor'] = 'other'
        elif defect == 'data': value['contract']['panel_identity'] = {'other': 'cache'}
        elif defect == 'runtime': value['contract']['runtime']['torch'] = 'other'
        elif defect == 'source': value['contract']['provenance']['pilot_sources'] = {}
        elif defect == 'step': value['contract']['settings']['step_size_255'] = .5 if value['contract']['settings']['step_size_255'] != .5 else 1.
        elif defect == 'zero': value['records'][0]['seconds'] = 0
        else: value['records'][0]['seconds'] = float('nan')
        if defect == 'nan':
            with pytest.raises(ValueError): controls.write_wrapped(target, value, replace=True)
        else:
            controls.write_wrapped(target, value, replace=True)
            with pytest.raises(ValueError): controls.combine(past, current, reference, .61)
        target.write_bytes(good)
    # Different GPU/runtime, missing entire controls, historical relabeling,
    # and duplicate file copies must not enter an optimistic projection.
    session_path = current / 'session.json'; good_session = session_path.read_bytes()
    value = read_record(session_path); value['runtime']['gpu'] = 'different GPU'
    controls.write_wrapped(session_path, value, replace=True)
    with pytest.raises(ValueError, match='runtime changed'): controls.combine(past, current, reference, .61)
    session_path.write_bytes(good_session)
    target.rename(target.with_name('missing.json'))
    with pytest.raises(ValueError, match='Missing'): controls.combine(past, current, reference, .61)
    target.with_name('missing.json').rename(target)
    duplicate = current / 'controls/extra/evaluation.json'; duplicate.parent.mkdir()
    duplicate.write_bytes(good)
    with pytest.raises(ValueError, match='duplicate'): controls.combine(past, current, reference, .61)
    duplicate.unlink()
    old_session = past / 'session.json'; value = read_record(old_session); value['complete'] = True
    controls.write_wrapped(old_session, value, replace=True)
    with pytest.raises(ValueError, match='relabeled'): controls.combine(past, current, reference, .61)


def minimal_package(tmp_path):
    package = tmp_path / 'package'; package.mkdir()
    for name in ('models/hardware_controls.py', 'scripts/run_hardware_controls.py'):
        target = package / 'operations' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    controls.save(package / 'budget-ledger.json', ledger())
    files = {str(p.relative_to(package)): dict(bytes=p.stat().st_size, sha256=controls.sha(p))
             for p in package.rglob('*') if p.is_file()}
    controls.save(package / 'controls-manifest.json', dict(profile=controls.PROFILE,
        base_manifest_sha256=controls.BASE_MANIFEST, prior_archive_sha256=controls.PRIOR_ARCHIVE,
        paid_compute_authorized=False, files=files))
    return package


def test_setup_failure_exports_partial_evidence_without_virtualenv_or_scientific_imports(tmp_path, monkeypatch):
    package = minimal_package(tmp_path)
    base = tmp_path / 'base'; (base / 'models').mkdir(parents=True)
    shutil.copyfile(ROOT / 'models/pilot_bundle.py', base / 'models/pilot_bundle.py')
    run = tmp_path / 'run'; (run / 'pilot').mkdir(parents=True)
    from scripts.run_workflow_pilot import run_child
    def fail_setup_then_export(command, log, seconds):
        if command[0] == 'bash':
            raise RuntimeError('generated setup rejection')
        # A stdlib-only interpreter cannot import the scientific environment.
        command[0] = shutil.which('python3')
        return run_child(command, log, seconds)
    monkeypatch.setattr('scripts.run_workflow_pilot.run_child', fail_setup_then_export)
    assert controls.worker(base, package, run, dict(deadline_epoch=int(time.time()) + 600,
                                                  quoted_all_in_hourly_usd='.61')) == 1
    assert not (base / '.venv').exists()
    with tarfile.open(run / 'evidence.tar.gz') as archive:
        session = json.load(archive.extractfile('run/pilot/session.json'))['payload']
        assert session['complete'] is False and session['failed_stage'] == 'setup'
        assert session['phases'] == [] and session['worker_exit_status'] == 1
        assert not any(name.endswith('.pt') for name in archive.getnames())


def test_unreviewed_template_cannot_create_a_claim_or_background_job(tmp_path):
    package = minimal_package(tmp_path)
    bad = plan(); bad.update(manual_deployment_authorized=False, ledger_sha256=controls.sha(package / 'budget-ledger.json'))
    controls.save(tmp_path / 'unreviewed.json', bad)
    base = tmp_path / 'base'; base.mkdir()
    result = subprocess.run([sys.executable, str(package / 'operations/scripts/run_hardware_controls.py'),
        '--mode', 'launch', '--base', str(base), '--package', str(package), '--plan', str(tmp_path / 'unreviewed.json')],
        text=True, capture_output=True, timeout=10)
    assert result.returncode != 0 and not (base / 'runs').exists()
