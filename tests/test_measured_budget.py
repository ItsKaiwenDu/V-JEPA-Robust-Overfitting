"""Cost decisions must retain strong evaluation, unknown costs and provenance."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from models.measured_budget import project
from scripts.estimate_measured_study import build_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    return (yaml.safe_load((ROOT/'configs/budget_study_plan.yaml').read_text()),
            json.loads((ROOT/'report/checks/2026-10-22-shared-forward-cuda-workflow.json').read_text()))


def test_workload_matches_independent_schedule_counts(inputs):
    result = project(*inputs, classes=20, train_per_class=5)
    w = result['workload']
    assert w['updates_per_run'] == 390 and w['runs'] == 8
    a, b = w['phases']['exploration'], w['phases']['confirmation']
    assert (a['routine_clips'], b['routine_clips']) == (2440, 7220)
    assert (a['primary_train_validation_clips'], b['primary_train_validation_clips']) == (520, 1460)
    assert (a['primary_test_clips'], b['primary_test_clips']) == (0, 600)
    assert (a['convergence_clips_including_two_audits'], b['convergence_clips_including_two_audits']) == (40, 100)
    assert sum(p['checkpoint_state_and_save_boundaries'] for p in (a, b)) == 248
    assert w['selected_representation_upper_bound'] == 13
    assert w['selected_checkpoint_validation_panel_pairs'] == 14


def test_scores_audits_and_nested_spans_are_not_added_twice(inputs):
    plan, evidence = inputs
    result = project(plan, evidence, classes=20, train_per_class=5)
    rows = result['scenarios']['median']['phases']['exploration']['components']
    assert rows['convergence_clips_including_two_audits']['seconds_per_unit'] == evidence['native_evaluation_seconds']['convergence']['median_seconds']
    assert rows['calibration_three_candidate_clip_bundles']['count'] == 10
    assert rows['calibration_three_candidate_clip_bundles']['seconds_per_unit'] == evidence['successful_span_timings']['calibration_three_primary_candidates']['median_seconds']
    evidence['successful_span_timings']['scheduled_evaluation']['median_seconds'] *= 100
    evidence['successful_span_timings']['control_evaluation']['median_seconds'] *= 100
    assert project(plan, evidence, classes=20, train_per_class=5) == result


def test_pilot_spend_stays_inside_reserve_and_unknowns_stay_unknown(inputs):
    plan, evidence = inputs
    result = project(plan, evidence, classes=20, train_per_class=5)
    evidence['compute']['derived_cumulative_spending_usd'] = 3.1
    assert project(plan, evidence, classes=20, train_per_class=5) == result
    for scenario in result['scenarios'].values():
        assert scenario['total_with_historical_untimed_allowance_usd'] == pytest.approx(
            scenario['with_headroom_and_allowance_hours'] * 1.21 + 10 + 40)
        for phase in scenario['phases'].values():
            assert phase['unknown_costs'] and all(r['hours'] is None for r in phase['unknown_costs'])
    assert not result['complete_cost_estimate'] and not result['budget_approved']


def test_all_prespecified_variants_exceed_caps_with_new_measurements(inputs):
    plan, evidence = inputs
    for counts in plan['data']['variants'].values():
        value = project(plan, evidence, **counts)
        for scenario in value['scenarios'].values():
            assert not scenario['arithmetic_within_caps']
            assert scenario['total_without_untimed_allowance_usd'] > 200
    value = project(plan, evidence, classes=20, train_per_class=5)
    assert value['scenarios']['maximum']['total_with_historical_untimed_allowance_usd'] >= value['scenarios']['median']['total_with_historical_untimed_allowance_usd']


def test_economics_threshold_preserves_unknown_allowance(inputs):
    plan, evidence = inputs
    s = project(plan, evidence, classes=20, train_per_class=5)['scenarios']['median']
    hours = s['timed_unit_extrapolation_hours']
    rate = s['break_even_all_in_hourly_price_at_same_throughput_usd']
    assert (hours + 8) * 1.25 * rate + 50 == pytest.approx(200)
    speedup = s['required_uniform_timed_work_speedup_at_recorded_quote']
    assert (hours / speedup + 8) * 1.25 * 1.21 + 50 == pytest.approx(200)
    faster = copy.deepcopy(evidence)
    faster['compute']['quoted_hourly_usd'] = rate / 2
    result = project(plan, faster, classes=20, train_per_class=5)
    assert result['scenarios']['median']['arithmetic_within_caps']
    assert not result['budget_approved']


@pytest.mark.parametrize('change', ['policy', 'recovery', 'attack', 'schedule', 'audit', 'timing', 'cap'])
def test_mismatched_or_incomplete_evidence_rejects(inputs, change):
    plan, evidence = inputs
    if change == 'policy': evidence['session']['execution_policy'] = 'original'
    if change == 'recovery': evidence['comparisons']['clean-finetune']['exact_match'] = False
    if change == 'attack': plan['evaluation']['primary_attack']['restarts'] = 1
    if change == 'schedule': plan['evaluation']['primary_epochs'] = [30]
    if change == 'audit': evidence['control_details']['convergence']['records'][0]['scoring_audits'] = []
    if change == 'timing': evidence['native_evaluation_seconds']['primary']['maximum_seconds'] = float('nan')
    if change == 'cap': plan['budget']['total_usd'] = 300
    with pytest.raises(ValueError):
        project(plan, evidence, classes=20, train_per_class=5)


def test_report_is_reproducible_read_only_and_inputs_are_pinned(inputs, tmp_path, monkeypatch):
    # A positive builder fixture has its own declared source tree. Never re-pin
    # the historical on-disk plan/evidence just because working code changed.
    import hashlib
    plan, evidence = inputs
    pins = {**plan['source_sha256'], **evidence['provenance']['unchanged_core_and_configuration_sha256']}
    names = set(pins) | {'models/measured_budget.py', 'scripts/estimate_measured_study.py',
                        'models/evaluation.py', 'models/schedule.py', 'models/checkpointing.py'}
    original_hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    for name in names:
        path = tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    plan['source_sha256'] = {name:original_hashes[name] for name in plan['source_sha256']}
    evidence['provenance']['unchanged_core_and_configuration_sha256'] = {
        name:original_hashes[name] for name in evidence['provenance']['unchanged_core_and_configuration_sha256']}
    for name,value,encoder in (
            ('configs/budget_study_plan.yaml',plan,yaml.safe_dump),
            ('report/checks/2026-10-22-shared-forward-cuda-workflow.json',evidence,json.dumps)):
        path = tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        if name.endswith('.json'):
            # The evidence also pins the proposal YAML; bind the isolated
            # rewritten YAML, not its historical original byte representation.
            plan_name = 'configs/budget_study_plan.yaml'
            evidence['provenance']['unchanged_core_and_configuration_sha256'][plan_name] = (
                hashlib.sha256((tmp_path/plan_name).read_bytes()).hexdigest())
        path.write_text(encoder(value))
    a = build_report(tmp_path)
    assert a == build_report(tmp_path)
    assert not a['budget_approved'] and not a['cloud_job_started'] and not a['pixels_opened']
    assert 'configs/study.yaml' in a['source_sha256']
    assert 'models/evaluation.py' in a['source_sha256']
    historical = json.loads((ROOT/'report/checks/2026-10-22-measured-budget-review.json').read_text())
    # Python 3.12 changed float sum(). Keep native-runtime repeatability exact,
    # and replay the historical sequential-addition algorithm explicitly when
    # comparing with the Python 3.11 report. No tolerance or source pin changes.
    def historical_sum(values, start=0):
        total = start
        for value in values:
            total += value
        return total
    with monkeypatch.context() as historical_runtime:
        historical_runtime.setattr('models.measured_budget.sum', historical_sum, raising=False)
        replay = build_report(tmp_path)
    assert replay['variants'] == historical['variants']
    assert a == build_report(tmp_path)  # Native summation restored unchanged.
    for name, value in a['variants'].items():
        assert value['workload'] == historical['variants'][name]['workload']
        for stat, scenario in value['scenarios'].items():
            assert scenario['arithmetic_within_caps'] == historical['variants'][name]['scenarios'][stat]['arithmetic_within_caps']
    assert original_hashes == {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def test_historical_report_refuses_changed_fixture_sources_without_relaxing_pins():
    with pytest.raises(ValueError,match='Pinned source changed: models/native_diagnostic.py'):
        build_report(ROOT)


def test_source_drift_rejects_before_projection(tmp_path):
    for name in ('configs/budget_study_plan.yaml', 'report/checks/2026-10-22-shared-forward-cuda-workflow.json'):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    path = tmp_path/'configs/study.yaml'
    path.write_text('changed: true\n')
    with pytest.raises(ValueError, match='Pinned source changed: configs/study.yaml'):
        build_report(tmp_path)
