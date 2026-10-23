"""Offline cost projection for the explicitly validated shared-forward candidate.

Short pilot observations are extrapolations, not full-study bounds. This module
never changes a protocol, selects a dataset, or authorizes paid execution.
"""

import math


UNKNOWN_COSTS = (
    'Full-panel data loading, integrity review, decoding and target caching',
    'Uninstrumented per-panel identity/hash/journal work and GPU transfers',
    'Full-training-set features, ridge fit and held-out probe scoring',
    'Checkpoint retention, temporary disk peaks, export and verified backup',
    'Fresh billed setup, transfers, idle time, restarts and failed work',
)


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be positive and finite.')
    return value


def _summary(value, statistic):
    if (type(value['count']) is not int or value['count'] < 1
            or not 0 < value['minimum_seconds'] <= value['median_seconds'] <= value['maximum_seconds']):
        raise ValueError('Incomplete or inconsistent timing summary.')
    for key in ('minimum_seconds', 'median_seconds', 'maximum_seconds'):
        _positive(value[key], key)
    return value[statistic + '_seconds']


def validate_inputs(plan, evidence):
    tr, ev, data, budget = (plan[k] for k in ('training', 'evaluation', 'data', 'budget'))
    if (tr['conditions'] != ['clean-finetune', 'adversarial-finetune']
            or tr['exploration_seeds'] != [0] or tr['confirmation_seeds'] != [11, 22, 33]
            or tr['epochs'] != 30 or tr['effective_batch_size'] != 8 or tr['micro_batch_size'] != 1
            or tr['precision'] != 'float32' or tr['context_patch_embedding'] != 'conv3d'
            or tr['activation_checkpointing'] is not True or tr['cache_fixed_clean_targets'] is not True
            or tr['final_partial_batch'] != 'retain_and_normalize_by_actual_example_count'
            or tr['training_attack'] != dict(steps=5, restarts=1, attack_masks=1, epsilon_255=4, step_size_255=1)):
        raise ValueError('Training design changed; revise the measured projection.')
    if (ev['primary_epochs'] != [1, 15, 16, 23, 24, 30]
            or ev['routine_epochs'] != 'every_epoch_plus_initialization'
            or ev['primary_attack'] != dict(steps=20, restarts=2, attack_masks=4, scoring_masks=4,
                                            epsilon_255=4, step_size_255=1)
            or ev['routine_attack'] != dict(steps=5, restarts=1, attack_masks=1, scoring_masks=1,
                                            epsilon_255=4, step_size_255=1)
            or ev['convergence_attack'] != dict(steps=50, restarts=2, attack_masks=8, scoring_masks=4)
            or ev['diagnostic_clips'] != 10 or ev['calibration_step_sizes_255'] != [.5, 1, 2]
            or not ev['shared_train_initialization'] or not ev['separate_validation_initializations']
            or not ev['fresh_adversaries_per_checkpoint'] or not ev['reuse_primary_scoring_bank_for_convergence']
            or ev['step_size_status'] != 'provisional_select_on_training_calibration_before_trajectories'
            or (data['train_monitor_per_class'], data['validation_per_class_per_phase'], data['test_per_class']) != (1, 1, 3)):
        raise ValueError('Evaluation design changed; revise the measured projection.')
    if (budget['total_usd'], budget['compute_usd'], budget['noncompute_reserve_usd'],
            budget['pilot_reserve_total_usd'], budget['runtime_headroom_factor']) != (200, 160, 40, 10, 1.25):
        raise ValueError('Preserve the reviewed caps, pilot reserve and headroom.')
    if (evidence['passed'] is not True or evidence['diagnostic_only'] is not True
            or evidence['session']['complete'] is not True
            or evidence['session']['execution_policy'] != 'shared-fp64-native'
            or evidence['session']['device']['gpu'] != 'NVIDIA GeForce RTX 5090'
            or evidence['calibration_is_study_calibration'] is not False
            or evidence['calibration_chosen_step_size_255'] != 2):
        raise ValueError('Require the verified bounded shared-forward RTX 5090 observation.')
    for condition in tr['conditions']:
        result = evidence['comparisons'][condition]
        if result['exact_match'] is not True or result['expected_step'] != 2:
            raise ValueError('Bounded exact recovery evidence is incomplete.')
    for method, settings in (
            ('convergence', dict(steps=50, restarts=2, attack_masks=8, scoring_masks=4)),
            ('fgsm', dict(steps=1, restarts=1, attack_masks=4, scoring_masks=4)),
            ('random', dict(steps=1, restarts=1, attack_masks=4, scoring_masks=4))):
        actual = evidence['control_details'][method]
        if actual['settings'] != dict(settings, method=method, epsilon_255=4, step_size_255=2):
            raise ValueError('Control timing strength differs from the declared workload.')
        if method == 'convergence' and (len(actual['records']) != 1
                                       or len(actual['records'][0]['scoring_audits']) != 2):
            raise ValueError('Convergence timing must already include both scoring-bank audits.')
    _positive(evidence['compute']['quoted_hourly_usd'], 'Recorded quote')


def workload(plan, *, classes, train_per_class):
    """Count only the unchanged staged schedule; no video pixels are needed."""
    if any(type(v) is not int or v < 1 for v in (classes, train_per_class)):
        raise ValueError('Counts must be positive integers.')
    epochs = plan['training']['epochs']
    anchors = len(plan['evaluation']['primary_epochs'])
    updates = math.ceil(classes * train_per_class / 8) * epochs
    diagnostic = min(classes, 10)
    phases = {}
    for phase, pairs in (('exploration', 1), ('confirmation', 3)):
        runs = 2 * pairs
        initial_panels = 2 if phase == 'exploration' else 1
        selected = 1 + 3 * pairs
        phases[phase] = {
            'clean_updates': pairs * updates,
            'adversarial_updates': pairs * updates,
            'routine_clips': classes * (initial_panels + 2 * runs * epochs),
            'primary_train_validation_clips': classes * (initial_panels + 2 * runs * anchors),
            'primary_test_clips': selected * classes * 3 if phase == 'confirmation' else 0,
            'convergence_clips_including_two_audits': selected * diagnostic,
            'fgsm_clips': selected * diagnostic,
            'random_clips': selected * diagnostic,
            'calibration_three_candidate_clip_bundles': diagnostic if phase == 'exploration' else 0,
            'checkpoint_state_and_save_boundaries': runs * (epochs + 1),
            'trained_selected_anchor_loads': 3 * pairs,
        }
    return {'classes': classes, 'training_clips': classes * train_per_class,
            'updates_per_run': updates, 'runs': 8, 'phases': phases,
            'selected_representation_upper_bound': 13,
            'selected_checkpoint_validation_panel_pairs': 14,
            'final_test_checkpoint_upper_bound': 10}


def project(plan, evidence, *, classes, train_per_class):
    validate_inputs(plan, evidence)
    counts = workload(plan, classes=classes, train_per_class=train_per_class)
    budget = plan['budget']
    price = evidence['compute']['quoted_hourly_usd']
    headroom = budget['runtime_headroom_factor']
    study_limit = budget['compute_usd'] - budget['pilot_reserve_total_usd']
    evals, spans = evidence['native_evaluation_seconds'], evidence['successful_span_timings']
    scenarios = {}
    for statistic in ('median', 'maximum'):
        unit = {method + '_clips': _summary(evals[method], statistic)
                for method in ('routine', 'fgsm', 'random')}
        primary = _summary(evals['primary'], statistic)
        unit.update({
            'primary_train_validation_clips': primary,
            'primary_test_clips': primary,
            'convergence_clips_including_two_audits': _summary(evals['convergence'], statistic),
            'calibration_three_candidate_clip_bundles': _summary(spans['calibration_three_primary_candidates'], statistic),
            'checkpoint_state_and_save_boundaries': _summary(spans['engine.state'], statistic) + _summary(spans['checkpoint.save'], statistic),
            'trained_selected_anchor_loads': _summary(spans['load_anchor'], statistic),
            'clean_updates': _summary(evidence['engine_update_seconds_by_condition']['clean-finetune'], statistic),
            'adversarial_updates': _summary(evidence['engine_update_seconds_by_condition']['adversarial-finetune'], statistic),
        })
        phases = {}
        for phase, numbers in counts['phases'].items():
            rows = {name: {'count': count, 'seconds_per_unit': unit[name],
                           'projected_hours': count * unit[name] / 3600}
                    for name, count in numbers.items()}
            hours = sum(row['projected_hours'] for row in rows.values())
            allowance = budget['untimed_allowance_hours'][phase]
            if isinstance(allowance, bool) or not isinstance(allowance, (float, int)) or not math.isfinite(allowance) or allowance < 0:
                raise ValueError('Untimed allowance must be nonnegative and finite.')
            phases[phase] = {'components': rows, 'timed_unit_extrapolation_hours': hours,
                'unknown_costs': [{'component': s, 'hours': None} for s in UNKNOWN_COSTS],
                'historical_untimed_allowance_hours_not_a_bound': allowance,
                'study_compute_with_headroom_without_allowance_usd': hours * headroom * price,
                'study_compute_with_headroom_and_allowance_usd': (hours + allowance) * headroom * price}
        hours = sum(p['timed_unit_extrapolation_hours'] for p in phases.values())
        allowance = sum(p['historical_untimed_allowance_hours_not_a_bound'] for p in phases.values())
        compute = (hours + allowance) * headroom * price
        reserve = budget['pilot_reserve_total_usd'] + budget['noncompute_reserve_usd']
        available_hours = study_limit / (price * headroom) - allowance
        primary_hours = sum(sum(row['projected_hours'] for name, row in p['components'].items()
                                if name in ('primary_train_validation_clips', 'primary_test_clips')) for p in phases.values())
        primary_room = available_hours - (hours - primary_hours)
        scenarios[statistic] = {
            'phases': phases,
            'timed_unit_extrapolation_hours': hours,
            'with_headroom_and_allowance_hours': (hours + allowance) * headroom,
            'total_without_untimed_allowance_usd': hours * headroom * price + reserve,
            'total_with_historical_untimed_allowance_usd': compute + reserve,
            'total_compute_including_full_pilot_reserve_usd': compute + budget['pilot_reserve_total_usd'],
            'excess_over_total_cap_usd': max(0, compute + reserve - budget['total_usd']),
            'arithmetic_within_caps': compute + reserve <= budget['total_usd'] and compute + budget['pilot_reserve_total_usd'] <= budget['compute_usd'],
            'break_even_all_in_hourly_price_at_same_throughput_usd': study_limit / ((hours + allowance) * headroom),
            'required_uniform_timed_work_speedup_at_recorded_quote': hours / available_hours if available_hours > 0 else None,
            'required_primary_only_speedup_at_recorded_quote': primary_hours / primary_room if primary_room > 0 else None,
            'primary_fraction_of_timed_work': primary_hours / hours,
        }
    return {'workload': counts, 'scenarios': scenarios,
            'complete_cost_estimate': False, 'budget_approved': False,
            'research_policy_adopted': False}
