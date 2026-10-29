"""Proposal scope, seed isolation, conditional budget ceilings and no-admission checks."""
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path

import pytest

from models.research_attack_candidate import (calibration_spec, canonical, conditional_budget,
    draw, proposal, seed, validate_proposal)

ROOT = Path(__file__).resolve().parents[1]


def plan():
    return json.loads((ROOT / 'configs/matched_panel_diagnostic_v1.json').read_text())


def review():
    return json.loads((ROOT / 'report/checks/2026-10-23-offline-a40-review.json').read_text())


def test_primary_candidate_preserves_training_scope_and_stronger_control():
    p = plan(); x = proposal(p['ordered_clips'], p['cases'])
    assert x['primary_candidate'] == dict(steps=20, restarts=2, attack_masks=8, scoring_masks=4, epsilon_255=4, step_size_255=None)
    assert x['stronger_control']['steps'] == 50 and x['stronger_control']['attack_masks'] == 8
    assert (x['preserved_study']['classes'], x['preserved_study']['training_clips'], x['preserved_study']['runs']) == (20, 100, 8)
    assert x['preserved_study']['training_attack']['steps'] == 5 and x['preserved_study']['training_attack']['step_size_255'] == 1
    assert not any(x[k] for k in ('launch_authorized', 'new_GPU_job_started', 'paid_compute_authorized', 'research_admitted', 'attack_convergence_pass'))
    assert x['budget']['complete_candidate_cost_usd'] is None and x['budget']['current_live_quote_usd_per_hour'] is None


def test_calibration_seeds_match_existing_serialization_and_do_not_reuse_panel():
    from models.evaluation import evaluation_seed
    p = plan(); x = calibration_spec(p['ordered_clips'], p['cases'])
    for row in x['rows']:
        vid = row['video_id']
        assert row['start_seed'] == evaluation_seed(20261028, 'research_candidate_eot8_v1_start', vid)
        assert row['attack_seeds'] == [evaluation_seed(20261028, 'research_candidate_eot8_v1_attack', vid, draw=i) for i in range(8)]
    assert x['evaluations_required'] == 30 and x['candidate_step_sizes_255'] == [.5, 1., 2.]
    assert x['selected_step_size_255'] is None and x['audit_banks_used_for_selection'] is False
    assert x['exact_chosen_research_runtime_required'] and not x['another_GPU_diagnostic_calibration_portable_to_research']
    assert not x['original_four_draw_calibration_reusable']


def test_candidate_collision_with_previous_mask_bank_rejects():
    p = plan(); old = deepcopy(p['cases'])
    old[0]['attack_seeds'][0] = draw('research_candidate_eot8_v1_attack', p['ordered_clips'][0]['video_id'], 0)
    with pytest.raises(ValueError): calibration_spec(p['ordered_clips'], old)


@pytest.mark.parametrize('defect', ['missing-row', 'duplicate-row'])
def test_calibration_keeps_exactly_ten_unique_monitor_rows(defect):
    p = plan(); rows = p['ordered_clips']
    if defect == 'missing-row': rows.pop()
    else: rows[-1] = deepcopy(rows[0])
    with pytest.raises(ValueError): calibration_spec(rows, p['cases'])


@pytest.mark.parametrize('defect', ['admit', 'steps', 'masks', 'training', 'classes', 'budget', 'chosen-step', 'audit-selection', 'portable'])
def test_proposal_drift_requires_explicit_new_review(defect):
    p = plan(); expected = proposal(p['ordered_clips'], p['cases']); changed = deepcopy(expected)
    if defect == 'admit': changed['research_admitted'] = True
    elif defect == 'steps': changed['primary_candidate']['steps'] = 5
    elif defect == 'masks': changed['primary_candidate']['attack_masks'] = 4
    elif defect == 'training': changed['preserved_study']['training_attack']['steps'] = 1
    elif defect == 'classes': changed['preserved_study']['classes'] = 10
    elif defect == 'budget': changed['budget']['total_cap_usd'] = 300
    elif defect == 'chosen-step': changed['calibration']['selected_step_size_255'] = 2
    elif defect == 'audit-selection': changed['calibration']['audit_banks_used_for_selection'] = True
    else: changed['calibration']['another_GPU_diagnostic_calibration_portable_to_research'] = True
    with pytest.raises(ValueError): validate_proposal(changed, expected)


@pytest.mark.parametrize('label', ['median', 'maximum'])
def test_historical_screen_and_primary_ceiling_match_fraction_arithmetic(label):
    r = review(); x = conditional_budget(r)['scenarios'][label]
    rows = r['economics']['scenarios'][label]['component_contributions']
    variable = {'primary_train_validation_clips', 'primary_test_clips', 'calibration_three_candidate_clip_bundles'}
    fixed = sum(Fraction(row['count']) * Fraction.from_float(row['unit_seconds']) / 3600 for k, row in rows.items() if k not in variable)
    assert x['historical_other_component_hours'] == pytest.approx(float(fixed), abs=1e-12)
    for s in x['conditional_price_scenarios']:
        rate = Fraction(str(s['hypothetical_all_in_hourly_usd']))
        available = Fraction(150) / (Fraction(5, 4) * rate) - 8 - fixed
        assert s['optimistic_primary_clip_seconds_ceiling_if_calibration_free'] == pytest.approx(float(max(0, available * 3600 / 2580)), abs=1e-10)
        assert not s['budget_approved'] and not s['another_GPU_speed_assumed_or_predicted'] and not s['current_quote']
        assert s['zero_calibration_is_only_a_necessary_condition_relaxation'] is True


def test_unknown_candidate_units_and_total_are_never_filled_from_old_units():
    x = conditional_budget(review())
    assert x['complete_candidate_cost_usd'] is None and not x['budget_approved']
    for s in x['scenarios'].values():
        assert s['candidate_total_usd'] is None and len(s['missing_required_units']) == 2
        assert not s['complete_cost_estimate']


@pytest.mark.parametrize('defect', ['scope', 'test-count', 'nan', 'zero', 'drop-control'])
def test_invalid_or_reduced_historical_workload_is_rejected(defect):
    r = review(); e = r['economics']; rows = e['scenarios']['maximum']['component_contributions']
    if defect == 'scope': e['retained_design']['runs'] = 2
    elif defect == 'test-count': rows['primary_test_clips']['count'] = 0
    elif defect == 'nan': rows['routine_clips']['unit_seconds'] = float('nan')
    elif defect == 'zero': rows['primary_train_validation_clips']['unit_seconds'] = 0
    else: rows.pop('convergence_clips_including_two_audits')
    with pytest.raises(ValueError): conditional_budget(r)


def test_original_four_draw_evaluator_guard_remains_intact():
    from models.evaluation import EvaluationSettings
    assert EvaluationSettings.for_method('primary').attack_masks == 4
    with pytest.raises(ValueError): EvaluationSettings('primary', 20, 2, 8, 4)
