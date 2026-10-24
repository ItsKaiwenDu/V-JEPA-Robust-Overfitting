"""Offline attribution/interpretation tests; no private artifacts or GPU work."""

import copy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from models.feasibility_review import COMPONENT_UNITS, attack_pair_review, cost_review
from models.hardware_benchmark import economic_screen
from scripts.review_a40_feasibility import sha, verify_pins

ROOT = Path(__file__).resolve().parents[1]


def toy_inputs():
    # Analytic fixture: ten one-hour components and one two-hour state/save
    # boundary. These are synthetic accounting units, not GPU observations.
    workload = dict(classes=20, training_clips=100, runs=8, updates_per_run=390,
        phases={'exploration': {k: 1 for k in COMPONENT_UNITS},
                'confirmation': {k: 0 for k in COMPONENT_UNITS}})
    names = {v for v in COMPONENT_UNITS.values() if v} | {'engine.state', 'checkpoint.save'}
    return workload, {name: [3600.] for name in names}


def test_analytical_fixture_charges_each_disjoint_unit_and_reserve_once():
    workload, observations = toy_inputs()
    result = cost_review(workload, observations, .61)
    for row in result['scenarios'].values():
        assert row['timed_unit_hours'] == 12.
        assert row['total_usd'] == 65.25
        assert row['native_evaluation_fraction'] == .5
        assert sum(r['fraction_of_timed_work'] for r in row['component_contributions'].values()) == pytest.approx(1.)
        assert sum(row['phase_timed_hours'].values()) == 12.
        assert not row['budget_approved'] and not row['research_admitted']
    old = economic_screen({'variants': {'focused_20x5': {'workload': workload}}}, observations, .61)
    assert old['candidate_economic_screen']['median']['with_historical_allowances_and_reserves_usd'] == 65.25
    assert result['pilot_charges_not_added_again'] and not result['complete_cost_estimate']


def test_recorded_median_and_maximum_units_match_the_existing_public_screen():
    evidence = json.loads((ROOT / 'report/checks/2026-10-23-a40-controls-completion.json').read_text())
    ref = json.loads((ROOT / 'report/checks/2026-10-22-measured-budget-review.json').read_text())
    # Summary-only fixture preserves the reducers without claiming new samples.
    observations = {name: [row['minimum_seconds'], row['median_seconds'], row['maximum_seconds']]
                    for name, row in evidence['timing_summaries'].items()}
    result = cost_review(ref['variants']['focused_20x5']['workload'], observations, .61)
    for name, row in result['scenarios'].items():
        recorded = evidence['candidate_economic_screen'][name]
        assert math.isclose(row['timed_unit_hours'], recorded['timed_unit_projection_hours'], rel_tol=1e-14)
        assert math.isclose(row['total_usd'], recorded['with_historical_allowances_and_reserves_usd'], rel_tol=1e-14)
        assert .84 < row['native_evaluation_fraction'] < .86
        assert .13 < row['uniform_native_time_reduction_needed'] < .15
        assert row['uniform_native_throughput_speedup_needed'] > 1.14
        assert row['binding_hourly_rate'] == row['project_cap_hourly_rate']
        assert row['binding_hourly_rate'] < row['planned_compute_cap_hourly_rate']
    maximum = result['scenarios']['maximum']
    prices = {r['hypothetical_hourly_usd']: r for r in maximum['price_sensitivity']}
    assert prices[.52]['partial_arithmetic_within_200'] and not prices[.53]['partial_arithmetic_within_200']
    assert prices[.29]['maximum_uniform_native_time_multiplier'] > 1.8
    for price in prices.values():
        assert not price['another_gpu_performance_predicted']
    times = {r['uniform_native_time_multiplier']: r['total_usd'] for r in maximum['timing_sensitivity']}
    assert times[.85] < 200 < times[.90]


@pytest.mark.parametrize('defect', ['missing_unit', 'zero', 'nan', 'infinite', 'negative',
                                  'missing_count', 'removed_component', 'unknown_count', 'boolean_count',
                                  'class_count', 'confirmation', 'zero_rate', 'boolean_rate'])
def test_missing_work_nonfinite_units_or_unreviewed_scope_cannot_improve_the_screen(defect):
    workload, observations = toy_inputs(); rate = .61
    if defect == 'missing_unit': observations.pop('primary')
    elif defect in ('zero', 'nan', 'infinite', 'negative'):
        observations['primary'] = [{'zero': 0., 'nan': float('nan'), 'infinite': float('inf'), 'negative': -1.}[defect]]
    elif defect == 'missing_count': workload['phases']['exploration'].pop('primary_test_clips')
    elif defect == 'removed_component': workload['phases']['exploration']['primary_test_clips'] = 0
    elif defect == 'unknown_count': workload['phases']['exploration']['extra'] = 1
    elif defect == 'boolean_count': workload['phases']['confirmation']['routine_clips'] = True
    elif defect == 'class_count': workload['classes'] = 10
    elif defect == 'confirmation': workload['phases'].pop('confirmation')
    elif defect == 'zero_rate': rate = 0
    else: rate = True
    with pytest.raises(ValueError): cost_review(workload, observations, rate)


def pair():
    common = dict(panel='train_monitor', video_ids=['training-video'], checkpoint={'sha256': 'original'},
                  weights={'target': 'fixed'}, source={'sha256': 'source'}, runtime={'device': 'generated-fixture'})
    result = []
    for method, steps, masks, loss in (('primary', 20, 4, .66), ('convergence', 50, 8, .71)):
        settings = dict(method=method, steps=steps, restarts=2, attack_masks=masks, scoring_masks=4,
                        epsilon_255=4., step_size_255=2.)
        row = dict(video_id='training-video', method=method, settings=settings, clean_loss=.57,
                   adversarial_loss=loss, seeds={'score': [100, 101, 102, 103],
                       'attack': list(range(1 if method == 'primary' else 10, 1 + masks if method == 'primary' else 10 + masks)),
                       'start': 1 if method == 'primary' else 2}, scoring_audits=[])
        if method == 'convergence':
            row['scoring_audits'] = [dict(clean_loss=.55, adversarial_loss=.80), dict(clean_loss=.54, adversarial_loss=.70)]
        result.append(dict(complete=True, contract={**copy.deepcopy(common), 'settings': settings,
                           'sensitivity_banks': 0 if method == 'primary' else 2}, records=[row]))
    return result


def test_larger_combined_setting_loss_cannot_be_labeled_an_iteration_effect_or_study_convergence():
    result = attack_pair_review(*pair())
    assert result['combined_configuration_loss_difference'] == pytest.approx(.05)
    assert result['same_main_scoring_bank']
    assert result['changed_factors']['steps'] == [20, 50]
    assert result['changed_factors']['attack_mask_count'] == [4, 8]
    assert not result['changed_factors']['attack_mask_prefix_shared']
    assert not result['changed_factors']['random_start_seed_shared']
    assert not result['iteration_effect_identified'] and not result['study_attack_convergence_validated']
    assert not result['cause_of_scoring_bank_variation_identified'] and not result['research_admitted']
    assert [r['adversarial_minus_clean_loss'] for r in result['scoring_banks']] == pytest.approx([.14, .25, .16])
    assert result['next_diagnostic']['status'] == 'designed_not_implemented_or_run'
    assert len(result['next_diagnostic']['arms']) == 4


@pytest.mark.parametrize('defect', ['partial', 'checkpoint', 'runtime', 'source', 'score', 'clean', 'audit', 'radius'])
def test_unpaired_or_incomplete_attack_evidence_is_rejected(defect):
    p, c = pair()
    if defect == 'partial': c['complete'] = False
    elif defect in ('checkpoint', 'runtime', 'source'): c['contract'][defect] = {'changed': True}
    elif defect == 'score': c['records'][0]['seeds']['score'][0] = 999
    elif defect == 'clean': c['records'][0]['clean_loss'] = .58
    elif defect == 'audit': c['records'][0]['scoring_audits'].pop()
    else: c['contract']['settings']['epsilon_255'] = 2.
    with pytest.raises(ValueError): attack_pair_review(p, c)


def test_changed_or_aliased_pinned_inputs_reject_before_review(tmp_path):
    path = tmp_path / 'evidence.json'; path.write_text('accepted')
    pins = {'evidence.json': sha(path)}; verify_pins(tmp_path, pins)
    path.write_text('changed')
    with pytest.raises(ValueError, match='Pinned'): verify_pins(tmp_path, pins)
    path.unlink(); other = tmp_path / 'other.json'; other.write_text('accepted'); path.symlink_to(other)
    with pytest.raises(ValueError, match='Pinned'): verify_pins(tmp_path, pins)


def test_cli_refuses_to_overwrite_before_opening_any_run_or_pixel_data(tmp_path):
    path = tmp_path / 'review.json'; path.write_text('previous evidence')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/review_a40_feasibility.py'),
                             '--output', str(path)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and path.read_text() == 'previous evidence'
    assert 'do not overwrite' in result.stderr
