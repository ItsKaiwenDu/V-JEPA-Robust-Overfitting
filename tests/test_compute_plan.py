"""Budget safeguards and scientific workload invariants, without a GPU."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts.estimate_compute import estimate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    plan = yaml.safe_load((ROOT / 'configs/compute_plan.yaml').read_text())
    return (plan, yaml.safe_load((ROOT / plan['study']).read_text()),
            json.loads((ROOT / plan['pilot']).read_text()),
            json.loads((ROOT / plan['attack_check']).read_text()))


def indexed(result):
    return {row['component']: row for row in result['components']}


def test_unknown_costs_cannot_be_treated_as_free_or_approved(inputs):
    result = estimate(*inputs, train_per_class=20)
    assert not result['complete_cost_estimate'] and not result['budget_approved']
    assert 'probe_fit' in result['unmeasured_components']
    assert indexed(result)['probe_fit']['projected_hours'] is None
    assert indexed(result)['pgd50_eot8_validation']['timing_basis'] != 'measured_unit'
    assert all(not quote['is_observed_gpu_speedup'] for quote in result['cloud_break_even_conditions'])


def test_pretrained_is_shared_and_best_final_use_conservative_upper_bound(inputs):
    result = estimate(*inputs, train_per_class=20)
    rows = indexed(result)
    assert result['training_runs'] == 6
    assert result['selected_checkpoint_count_upper_bound'] == 13
    assert rows['primary_train_validation']['count'] == 11110
    assert rows['primary_final_test']['count'] == 6565
    assert rows['epsilon_sweep_extra']['count'] == 3939  # Reuse clean and epsilon 4.


def test_training_fallback_preserves_seeds_validation_and_final_testing(inputs):
    full = estimate(*inputs, train_per_class=20)
    reduced = estimate(*inputs, train_per_class=10)
    a, b = indexed(full), indexed(reduced)
    assert full['training_runs'] == reduced['training_runs']
    assert b['clean-finetune']['count'] == 11430  # Round batches within each epoch.
    for name in ('primary_train_validation', 'primary_final_test', 'pgd50_eot8_validation',
                 'epsilon_sweep_extra', 'direct_classifier_pgd20'):
        assert a[name]['count'] == b[name]['count']


def test_reduced_pilot_cannot_be_used_as_full_workload_measurement(inputs):
    plan, study, pilot, attacks = copy.deepcopy(inputs)
    pilot['primary_evaluation_measured'] = False
    with pytest.raises(ValueError, match='reduced diagnostic'):
        estimate(plan, study, pilot, attacks, train_per_class=20)


def test_dense_validation_adds_primary_measurements_without_cheap_val_duplicates(inputs):
    sparse = indexed(estimate(*inputs, train_per_class=20))
    dense = indexed(estimate(*inputs, train_per_class=20, primary_every_epoch=True))
    assert dense['routine_pgd5']['count'] * 2 == sparse['routine_pgd5']['count']
    assert dense['primary_train_validation']['count'] > sparse['primary_train_validation']['count']
    assert dense['primary_final_test']['count'] == sparse['primary_final_test']['count']
