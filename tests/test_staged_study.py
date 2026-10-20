"""Cost accounting and held-out-data safeguards for the proposed staged study."""

import copy
import csv
import json
from pathlib import Path

import pytest
import yaml

from datasets.staged_subset import propose_subset
from scripts.estimate_staged_study import estimate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    plan = yaml.safe_load((ROOT / 'configs/budget_study_plan.yaml').read_text())
    return plan, json.loads((ROOT / plan['pilot']).read_text()), json.loads((ROOT / plan['attack_check']).read_text())


@pytest.fixture
def parent():
    result = {}
    for split in ('train', 'validation', 'test'):
        with (ROOT / 'data/manifests/ucf101-split1-v1' / f'{split}.csv').open() as handle:
            result[split] = list(csv.DictReader(handle))
    return result


def test_phase_counts_share_only_the_same_initialization_panel(inputs):
    result = estimate(*inputs, classes=20, train_per_class=5)
    rows = {(r['phase'], r['component']): r for r in result['components']}
    assert result['updates_per_run'] == 390  # Include partial batches each epoch.
    assert rows['exploration', 'primary_train_validation']['count'] == 520
    assert rows['confirmation', 'primary_train_validation']['count'] == 1460
    assert rows['exploration', 'routine_pgd5']['count'] == 2440
    assert rows['confirmation', 'routine_pgd5']['count'] == 7220
    assert rows['exploration', 'primary_final_test']['count'] == 0
    assert rows['confirmation', 'primary_final_test']['count'] == 600
    assert rows['exploration', 'convergence_pgd50']['count'] == 40
    assert rows['confirmation', 'convergence_pgd50']['count'] == 100


def test_unknown_costs_and_proxies_never_approve_a_run(inputs):
    result = estimate(*inputs, classes=20, train_per_class=5)
    assert result['ledger']['arithmetic_within_cap']
    assert not result['budget_approved'] and not result['complete_cost_estimate']
    for phase in ('exploration', 'confirmation'):
        assert 'probe_and_feature_diagnostics' in result['stages'][phase]['unmeasured_components']
    unknown = [r for r in result['components'] if r['timing_basis'] == 'unmeasured']
    assert unknown and all(r['projected_hours'] is None for r in unknown)
    convergence = [r for r in result['components'] if r['component'] == 'convergence_pgd50']
    assert all(r['timing_basis'] == 'gradient_work_ratio_proxy' for r in convergence)


def test_actual_pilot_spending_is_inside_reserve_not_charged_twice(inputs):
    a = estimate(*inputs, classes=20, train_per_class=5)['ledger']
    plan, pilot, attacks = copy.deepcopy(inputs)
    plan['budget']['spent_pilot_usd'] = 1.0
    b = estimate(plan, pilot, attacks, classes=20, train_per_class=5)['ledger']
    assert a['pilot_spent_user_reported_usd'] == 0.592
    assert a['pilot_unspent_reserved_usd'] == pytest.approx(9.408)
    assert a['total_planning_envelope_usd'] == b['total_planning_envelope_usd']
    plan['budget']['spent_pilot_usd'] = 11
    with pytest.raises(ValueError, match='Budget'):
        estimate(plan, pilot, attacks, classes=20, train_per_class=5)


@pytest.mark.parametrize('mutation', ['weaker_attack', 'mixed_precision', 'reused_seed', 'reduced_pilot'])
def test_changed_workload_cannot_inherit_timings_or_confirmation_status(inputs, mutation):
    plan, pilot, attacks = copy.deepcopy(inputs)
    if mutation == 'weaker_attack':
        plan['evaluation']['primary_attack']['attack_masks'] = 1
    elif mutation == 'mixed_precision':
        plan['training']['precision'] = 'bfloat16'
    elif mutation == 'reused_seed':
        plan['training']['confirmation_seeds'][0] = 0
    else:
        pilot['primary_evaluation_measured'] = False
    with pytest.raises(ValueError):
        estimate(plan, pilot, attacks, classes=20, train_per_class=5)


def test_subset_separates_groups_and_uses_only_metadata(parent, inputs):
    plan = inputs[0]
    args = dict(classes=20, train_per_class=5, seed=20261020,
                excluded_groups=plan['data']['exclude_from_exploration_holdout_groups'])
    result = propose_subset(parent, **args)
    reversed_parent = {s: list(reversed(rows)) for s, rows in parent.items()}
    assert propose_subset(reversed_parent, **args) == result
    assert result['counts'] == {'train': 100, 'exploration_validation': 20,
                                'confirmation_validation': 20, 'test': 60}
    assert set(result['train_monitor_video_ids']) <= {r['video_id'] for r in result['rows']['train']}
    held_out = set(result['exploration_groups_excluded_entirely_from_actual_training'].values())
    assert not held_out & {r['source_group_id'] for r in result['rows']['train']}
    assert not held_out & set(args['excluded_groups'])
    assert set(result['source_groups_per_class']['train'].values()) == {5}
    assert {r['video_id'] for r in result['rows']['test']} <= {r['video_id'] for r in parent['test']}
    assert not result['video_content_checked'] and not result['test_pixels_or_outcomes_accessed']


def test_parent_group_leakage_is_rejected_even_for_different_videos(parent):
    parent['validation'][0]['source_group_id'] = parent['train'][0]['source_group_id']
    with pytest.raises(ValueError, match='Parent partitions overlap'):
        propose_subset(parent, classes=20, train_per_class=5, seed=20261020)


def test_alternatives_preserve_confirmation_and_are_costed_separately(inputs):
    a = estimate(*inputs, classes=20, train_per_class=5)
    b = estimate(*inputs, classes=20, train_per_class=10)
    assert a['training_runs_if_both_stages'] == b['training_runs_if_both_stages'] == 8
    assert not b['ledger']['arithmetic_within_cap']
    for phase in ('exploration', 'confirmation'):
        a_eval = [r for r in a['components'] if r['phase'] == phase and 'finetune' not in r['component']]
        b_eval = [r for r in b['components'] if r['phase'] == phase and 'finetune' not in r['component']]
        assert a_eval == b_eval
