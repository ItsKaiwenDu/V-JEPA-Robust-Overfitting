"""Evidence scope, no double counting and independent complete-schedule arithmetic."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from models.candidate_budget import build_report, collect_units, project, statistic

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def report():
    return build_report(ROOT)


@pytest.fixture
def inputs(report):
    return (yaml.safe_load((ROOT / 'configs/budget_study_plan.yaml').read_text()),
            json.loads((ROOT / 'configs/research_attack_candidate_v1.json').read_text()),
            deepcopy(report['measurement_units']))


@pytest.fixture
def raw():
    review = json.loads((ROOT / 'report/checks/2026-10-28-candidate-benchmark-verification.json').read_text())
    b = (ROOT / review['evidence_archive']['path']).parent / 'extracted'
    paths = sorted((b / 'run/evaluations').glob('*/evaluation.json')) + sorted((b / 'run/calibration-diagnostic').glob('*/evaluation.json'))
    evaluations = {str(p.relative_to(b)): json.loads(p.read_text())['payload'] for p in paths}
    timings = {str(p.relative_to(b)): [json.loads(line) for line in p.read_text().splitlines()]
               for p in sorted((b / 'run/timings').glob('*.jsonl'))}
    return evaluations, timings


def test_all_eight_runs_and_full_evaluation_schedule_are_retained(report):
    w = report['economics']['workload']
    assert (w['classes'], w['training_clips'], w['updates_per_run'], w['runs']) == (20, 100, 390, 8)
    assert w['phases']['exploration']['primary_train_validation_clips'] == 520
    assert w['phases']['confirmation']['primary_train_validation_clips'] == 1460
    assert w['phases']['confirmation']['primary_test_clips'] == 600
    assert sum(p['routine_clips'] for p in w['phases'].values()) == 9660
    for label in ('median', 'maximum'):
        phases = report['economics']['scenarios'][label]['phases']
        totals = {k: sum(p['components'][k]['count'] for p in phases.values()) for k in phases['exploration']['components']}
        assert totals['clean_updates'] == totals['adversarial_updates'] == 1560
        assert totals['checkpoint_state_and_save_boundaries'] == 248
        assert totals['trained_selected_anchor_loads'] == 12
        assert totals['convergence_clips_including_two_audits'] == totals['fgsm_clips'] == totals['random_clips'] == 140
        assert totals['calibration_three_candidate_clip_bundles'] == 10
        assert sum(v for k, v in totals.items() if k.endswith('_panel_overhead')) == 634


@pytest.mark.parametrize('label', ['median', 'maximum'])
def test_component_hours_caps_and_break_even_match_independent_fraction_oracle(report, label):
    units = report['measurement_units']
    def select(name):
        vals = sorted(Fraction.from_float(x['seconds']) for x in units[name])
        if label == 'maximum': return vals[-1]
        n = len(vals)
        return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    # Explicit independent counts; do not call the production workload or projection.
    seconds = (1560 * select('clean_updates') + 1560 * select('adversarial_updates')
        + 9660 * select('routine') + 2580 * select('primary') + 140 * select('convergence')
        + 140 * select('fgsm') + 140 * select('random')
        + 10 * select('calibration_three_candidate_clip_bundles')
        + 248 * (select('engine.state') + select('checkpoint.save'))
        + 12 * select('trained_selected_anchor_loads')
        + 483 * select('routine_panel_overhead') + 109 * select('primary_panel_overhead')
        + 14 * select('convergence_panel_overhead') + 14 * select('fgsm_panel_overhead')
        + 14 * select('random_panel_overhead'))
    hours = seconds / 3600
    wall = (hours + 8) * Fraction(5, 4)
    study = wall * Fraction(37, 50)
    scenario = report['economics']['scenarios'][label]
    assert scenario['timed_unit_extrapolation_hours'] == pytest.approx(float(hours), abs=1e-11)
    assert scenario['total_with_historical_allowance_usd'] == pytest.approx(float(study + 50), abs=1e-10)
    assert scenario['compute_including_full_pilot_reserve_usd'] == pytest.approx(float(study + 10), abs=1e-10)
    assert scenario['total_without_historical_allowance_usd'] == pytest.approx(float(hours * Fraction(5, 4) * Fraction(37, 50) + 50), abs=1e-10)
    assert scenario['break_even_all_in_rate_at_same_sample_throughput_usd'] == pytest.approx(float(150 / wall), abs=1e-12)
    assert not scenario['arithmetic_within_caps'] and scenario['total_without_historical_allowance_usd'] > 200


def test_primary_uses_eight_zero_audit_samples_across_all_three_calibration_steps(report):
    samples = report['measurement_units']['primary']
    assert len(samples) == 8 and {s['step_size_255'] for s in samples} == {.5, 1, 2}
    assert not any('controls-primary' in s['path'] for s in samples)
    assert sum('calibration-diagnostic/' in s['path'] for s in samples) == 3
    assert report['excluded_observation']['unit'] == 'primary with two scoring audits'


def test_native_scores_audits_and_calibration_are_not_charged_twice(raw, report):
    evaluations, timings = raw
    baseline = collect_units(evaluations, timings)
    # Change unassigned audited-primary and nested totals, not the workload's measured units.
    evaluations['run/evaluations/controls-primary/evaluation.json']['records'][0]['seconds'] *= 100
    for rows in timings.values():
        for row in rows:
            if row['name'] in ('prepare_context', 'exact_fresh_process_recovery_comparison', 'fit_training_only_probe'):
                row['seconds'] *= 100
            if row.get('unit') == 'controls-primary': row['seconds'] *= 100
    assert collect_units(evaluations, timings) == baseline == report['measurement_units']


def test_wrapper_overhead_is_disjoint_from_native_clip_time(raw, report):
    evaluations, timings = raw
    rows = timings['run/timings/primary-pretrained.jsonl']
    span = next(r for r in rows if r.get('unit') == 'primary-pretrained')
    records = evaluations['run/evaluations/primary-pretrained/evaluation.json']['records']
    residual = next(s for s in report['measurement_units']['primary_panel_overhead'] if s['sample_panel_clips'] == 3)
    assert sum(r['seconds'] for r in records) + residual['seconds'] == pytest.approx(span['seconds'], abs=1e-10)


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf'), True])
def test_missing_nonfinite_and_invalid_measurements_never_become_free(inputs, bad):
    plan, candidate, units = inputs
    units['primary'][0]['seconds'] = bad
    with pytest.raises(ValueError): project(plan, candidate, units, .74)


def test_missing_and_duplicated_unit_observations_are_rejected(inputs):
    plan, candidate, units = inputs
    units['convergence'] = []
    with pytest.raises(ValueError): project(plan, candidate, units, .74)
    units['convergence'] = deepcopy(units['random']) * 2
    with pytest.raises(ValueError): project(plan, candidate, units, .74)


@pytest.mark.parametrize('defect', ['partial', 'old-four-draw', 'missing-audit', 'wrong-panel', 'failed-span', 'negative-wrapper'])
def test_mismatched_evaluation_or_timing_evidence_is_refused(raw, defect):
    e, t = raw
    key = 'run/evaluations/controls-convergence/evaluation.json'
    if defect == 'partial': e[key]['complete'] = False
    elif defect == 'old-four-draw': e[key]['contract']['settings']['attack_masks'] = 4
    elif defect == 'missing-audit': e[key]['records'][0]['scoring_audits'] = []
    elif defect == 'wrong-panel': e[key]['contract']['panel'] = 'test'
    elif defect == 'failed-span': t['run/timings/controls.jsonl'][0]['status'] = 'error'
    else: next(r for r in t['run/timings/controls.jsonl'] if r.get('unit') == 'controls-convergence')['seconds'] = 1
    with pytest.raises(ValueError): collect_units(e, t)


@pytest.mark.parametrize('defect', ['classes', 'seeds', 'schedule', 'cap', 'headroom', 'training-attack', 'candidate-step', 'candidate-masks'])
def test_frozen_design_cannot_be_silently_reduced_or_recalibrated(inputs, defect):
    plan, candidate, units = inputs
    if defect == 'classes': candidate['preserved_study']['classes'] = 10
    elif defect == 'seeds': plan['training']['confirmation_seeds'] = []
    elif defect == 'schedule': plan['evaluation']['primary_epochs'] = [30]
    elif defect == 'cap': plan['budget']['total_usd'] = 300
    elif defect == 'headroom': plan['budget']['runtime_headroom_factor'] = 1
    elif defect == 'training-attack': plan['training']['training_attack']['steps'] = 1
    elif defect == 'candidate-step': candidate['calibration']['selected_step_size_255'] = 2
    else: candidate['primary_candidate']['attack_masks'] = 4
    with pytest.raises(ValueError): project(plan, candidate, units, .74)


@pytest.mark.parametrize('rate', [.01, .29, .74])
def test_favorable_partial_arithmetic_never_grants_budget_or_research_admission(inputs, rate):
    plan, candidate, units = inputs
    value = project(plan, candidate, units, rate)
    assert not value['budget_approved'] and not value['research_admitted'] and not value['complete_cost_estimate']
    assert value['research_step_size_255'] is None and value['full_study_cost_bound_usd'] is None
    assert value['unknown_costs'] and all(v['hours'] is None and v['usd'] is None for v in value['unknown_costs'])
    for s in value['scenarios'].values():
        if rate < .3: assert s['arithmetic_within_caps']
        else: assert not s['arithmetic_within_caps']


def test_reservations_wallet_and_no_work_flags_remain_conservative(report):
    financial = report['financial_status']
    assert financial['billing_reconciliation_pending'] and financial['final_session_charge_usd'] is None
    assert financial['existing_conservative_pilot_encumbrance_usd'] == 9.22
    assert financial['unreserved_diagnostic_envelope_usd'] == .78
    assert financial['pilot_reserve_added_to_study_once_usd'] == 10
    assert not financial['new_hold_created'] and not financial['existing_hold_released']
    assert not report['tensor_deserialization_performed'] and not report['cloud_job_started']
    assert not report['new_paid_compute_authorized'] and not report['protocol_or_YAML_changed']


def test_hardware_price_threshold_and_primary_bottleneck_are_explicit(report):
    scenarios = report['economics']['scenarios']
    assert scenarios['maximum']['total_with_historical_allowance_usd'] >= scenarios['median']['total_with_historical_allowance_usd']
    for s in scenarios.values():
        assert .5 < s['break_even_all_in_rate_at_same_sample_throughput_usd'] < .6
        assert s['primary_native_fraction_of_timed_work'] > .7
        assert s['uniform_timed_work_speedup_needed_at_recorded_rate'] > 1
        assert s['primary_only_speedup_needed_at_recorded_rate'] > s['uniform_timed_work_speedup_needed_at_recorded_rate']


def test_builder_is_deterministic_and_never_repins_changed_inputs(report, tmp_path):
    assert build_report(ROOT) == report
    name = 'configs/budget_study_plan.yaml'
    path = tmp_path / name; path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / name).read_bytes() + b'\n# drift\n')
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='Pinned input changed: configs/budget_study_plan.yaml'):
        build_report(tmp_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
