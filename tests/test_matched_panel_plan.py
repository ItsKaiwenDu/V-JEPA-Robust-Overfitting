"""Metadata-only scientific design checks; no device, pixel or weight loading."""

import copy
import json
from pathlib import Path

import pytest

from models.evaluation import evaluation_seed
from models.matched_panel_plan import (ARMS, BASE_SEED, MODULUS, clip_seeds, compile_plan,
                                      freeze_plan, plan_digest, validate_plan, validate_seeds)


@pytest.fixture
def inputs():
    training = [{'video_id': f'video-{c}-{i}', 'class_id': str(c), 'source_group_id': f'group-{c}-{i}',
                 'relative_path': f'class-{c}/video-{c}-{i}.avi'} for c in range(20) for i in range(5)]
    monitor = [training[c * 5] for c in range(20)]
    entries = [{**row, 'file': row['video_id'] + '.pt', 'pixel_sha256': f'{i:064x}'}
               for i, row in enumerate(training)]
    # Avoid asserting the hash constants against a second copy of the algorithm.
    reference = {'session_seconds': 2457.3780396250077,
                 'seeds': {'attack': [MODULUS - 1], 'score': [MODULUS - 2], 'start': 2**63 - 1}}
    return training, monitor, entries, {'fixture': 'metadata-only'}, reference


def compile_inputs(inputs):
    train, monitor, entries, identity, reference = inputs
    return compile_plan(train, monitor, entries, identity=identity, reference=reference)


def test_original_order_real_training_members_and_class_design_are_preserved(inputs):
    plan = compile_inputs(inputs)
    assert [r['video_id'] for r in plan['ordered_clips']] == [f'video-{i}-0' for i in range(10)]
    assert len(plan['cases']) == 20 and [c['case_id'] for c in plan['cases'][:4]] == [
        'clip-00-bank-0', 'clip-00-bank-1', 'clip-01-bank-0', 'clip-01-bank-1']
    assert plan['study_training_classes'] == 20 and plan['diagnostic_classes'] == 10
    assert plan['counts'] == {'clips': 10, 'attack_banks_per_clip': 2, 'matrix_cases': 20,
        'matrix_arms': 80, 'fresh_A_replays': 20, 'total_PGD_arms': 100,
        'PGD_objective_calls_excluding_posthoc_scoring': 6700, 'input_gradient_calls': 6400}


def test_mask_replicates_change_only_attack_draws_not_starts_or_scores(inputs):
    plan = compile_inputs(inputs)
    for a, b in zip(plan['cases'][::2], plan['cases'][1::2]):
        assert a['video_id'] == b['video_id']
        assert a['start_seed'] == b['start_seed'] and a['scoring_seeds'] == b['scoring_seeds']
        assert set(a['attack_seeds']).isdisjoint(b['attack_seeds'])
        assert len(a['attack_seeds']) == 8 and a['arm_order'] == b['arm_order'] == ['A','B','C','D']
    assert plan['settings']['arms'] == ARMS
    assert plan['settings']['four_draw_bank'] == 'first-four-of-the-same-eight-draw-bank'


def test_280_draw_seeds_are_distinct_in_actual_upstream_space(inputs):
    plan = compile_inputs(inputs)
    draws = []
    for a, b in zip(plan['cases'][::2], plan['cases'][1::2]):
        draws += a['attack_seeds'] + b['attack_seeds']
        draws += [s for group in a['scoring_seeds'].values() for s in group]
    assert len(draws) == len(set(draws)) == len({s % MODULUS for s in draws}) == 280


def test_namespaced_seed_derivation_uses_the_existing_evaluation_algorithm():
    seed = clip_seeds('video')
    assert seed['attack_banks']['1'][7] == evaluation_seed(BASE_SEED, 'matched_panel_v1_attack_bank_1', 'video', draw=7)
    assert seed['scoring_banks']['main'][2] == evaluation_seed(BASE_SEED, 'matched_panel_v1_main', 'video', draw=2)
    assert seed['start'] == evaluation_seed(BASE_SEED, 'matched_panel_v1_start', 'video')


def test_seed_collision_after_modulo_is_not_missed(monkeypatch):
    counter = iter(range(1000))
    monkeypatch.setattr('models.matched_panel_plan.evaluation_seed', lambda *_a, **_k: next(counter) * MODULUS + 3)
    specs = [clip_seeds(str(i)) for i in range(10)]
    assert len({s for spec in specs for g in ('attack_banks','scoring_banks') for group in spec[g].values() for s in group}) == 280
    with pytest.raises(ValueError, match='upstream seed space'):
        validate_seeds(specs, {})


def test_historical_mask_draws_are_not_silently_reused(inputs):
    reference = inputs[-1]
    reference['seeds']['attack'] = [clip_seeds('video-0-0')['attack_banks']['0'][0]]
    with pytest.raises(ValueError, match='historical'):
        compile_inputs(inputs)


def test_historical_start_is_not_silently_reused(inputs):
    inputs[-1]['seeds']['start'] = clip_seeds('video-0-0')['start']
    with pytest.raises(ValueError, match='start'):
        compile_inputs(inputs)


@pytest.mark.parametrize('change', ['missing_train', 'duplicate_train', 'foreign_monitor', 'duplicate_monitor',
    'changed_group', 'duplicate_class', 'duplicate_source_group', 'missing_index', 'duplicate_index',
    'changed_selected_pixels', 'escaped_file'])
def test_unsafe_or_incomplete_panel_metadata_rejects(inputs, change):
    training, monitor, entries, _, _ = inputs
    if change == 'missing_train': training.pop()
    if change == 'duplicate_train': training[1] = dict(training[0])
    if change == 'foreign_monitor': monitor[0] = {**monitor[0], 'video_id': 'held-out'}
    if change == 'duplicate_monitor': monitor[1] = dict(monitor[0])
    if change == 'changed_group': monitor[0] = {**monitor[0], 'source_group_id': 'other-group'}
    if change == 'duplicate_class': monitor[1] = {**monitor[1], 'class_id': monitor[0]['class_id']}
    if change == 'duplicate_source_group':
        training[5] = {**training[5], 'source_group_id': training[0]['source_group_id']}
        monitor[1] = training[5]
    if change == 'missing_index': entries.pop()
    if change == 'duplicate_index': entries[1] = dict(entries[0])
    if change == 'changed_selected_pixels': entries[0]['pixel_sha256'] = 'not-a-hash'
    if change == 'escaped_file': entries[0]['file'] = '../held-out.pt'
    with pytest.raises(ValueError):
        compile_inputs(inputs)


def test_output_mutation_does_not_rewrite_module_constants_or_input_identity(inputs):
    original = compile_inputs(inputs)
    altered = compile_inputs(inputs)
    altered['settings']['arms']['A']['steps'] = 1
    altered['identity']['fixture'] = 'changed'
    assert compile_inputs(inputs) == original and ARMS['A']['steps'] == 20


@pytest.mark.parametrize('field', ['case', 'seed', 'limit', 'settings', 'admission', 'partial', 'aggregation'])
def test_changed_predeclaration_cannot_validate_as_the_frozen_plan(inputs, monkeypatch, field):
    expected = compile_inputs(inputs)
    plan = copy.deepcopy(expected)
    monkeypatch.setattr('models.matched_panel_plan.build_plan', lambda _root: expected)
    if field == 'case': plan['cases'].pop()
    if field == 'seed': plan['cases'][0]['attack_seeds'][0] += 1
    if field == 'limit': plan['execution_requirements']['outer_seconds'] += 1
    if field == 'settings': plan['settings']['step_size_255'] = 1
    if field == 'admission': plan['launch_authorized_by_this_file'] = True
    if field == 'partial': plan['analysis']['partial_panel_scientific_aggregate_allowed'] = True
    if field == 'aggregation': plan['analysis']['score_banks_pooled_or_selected'] = True
    with pytest.raises(ValueError, match='Frozen panel plan'):
        validate_plan(Path('metadata-only'), plan)


def test_scheduling_reference_is_a_scenario_not_a_performance_or_cost_bound(inputs):
    plan = compile_inputs(inputs)
    scenario = plan['scheduling_scenarios']
    assert scenario['repeat_reference_for_20_cases_hours'] == pytest.approx(13.65210022013893)
    assert scenario['repeat_reference_with_25pct_scheduling_margin_hours'] == pytest.approx(17.06512527517366)
    assert scenario['configured_case_limits_sum_hours'] == pytest.approx(26 + 2/3)
    assert scenario['including_all_outer_kill_graces_hours'] == 26.75
    assert not scenario['runtime_or_memory_bound_inferred_for_other_clips']
    assert not scenario['A40_cost_pool_eligible'] and not scenario['study_cost_revised']
    assert plan['execution_requirements']['minimum_free_disk_bytes_before_panel'] == 12 * 2**30


@pytest.mark.parametrize('seconds', [0, -1, float('nan'), float('inf'), True])
def test_invalid_scheduling_reference_rejects(inputs, seconds):
    inputs[-1]['session_seconds'] = seconds
    with pytest.raises(ValueError, match='finite positive'):
        compile_inputs(inputs)


def test_frozen_output_is_reproducible_and_cannot_be_overwritten(inputs, tmp_path, monkeypatch):
    expected = compile_inputs(inputs)
    monkeypatch.setattr('models.matched_panel_plan.build_plan', lambda _root: copy.deepcopy(expected))
    a, b = tmp_path/'a.json', tmp_path/'b.json'
    first = freeze_plan(tmp_path, a)
    second = freeze_plan(tmp_path, b)
    assert first == second and a.read_bytes() == b.read_bytes()
    assert first['plan_sha256'] == plan_digest(json.loads(a.read_text()))
    assert not first['launch_authorized']
    with pytest.raises(FileExistsError):
        freeze_plan(tmp_path, a)


@pytest.mark.parametrize('field', ['bool-as-int', 'int-as-float'])
def test_scalar_type_edits_cannot_launder_a_different_plan_identity(inputs, monkeypatch, field):
    expected = compile_inputs(inputs)
    plan = copy.deepcopy(expected)
    if field == 'bool-as-int': plan['launch_authorized_by_this_file'] = 0
    if field == 'int-as-float': plan['settings']['epsilon_255'] = 4.0
    assert plan == expected  # Demonstrates why plain Python equality is insufficient.
    monkeypatch.setattr('models.matched_panel_plan.build_plan', lambda _root: expected)
    with pytest.raises(ValueError, match='Frozen panel plan'):
        validate_plan(Path('metadata-only'), plan)


def test_plan_neither_selects_attacks_nor_admits_execution(inputs):
    plan = compile_inputs(inputs)
    assert not plan['launch_authorized_by_this_file'] and not plan['paid_compute_allowed']
    assert not plan['research_admitted'] and not plan['analysis']['automatic_primary_or_arm_choice']
    assert not plan['analysis']['automatic_attack_convergence_pass']
    assert not plan['analysis']['numeric_strength_margin_invented']
    assert not plan['analysis']['historical_one_clip_result_included']
    assert plan['preserved']['held_out_pixels_closed'] and plan['preserved']['canonical_evaluation_unchanged']
