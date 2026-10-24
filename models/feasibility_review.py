"""Offline cost attribution and limits of the saved attack comparison.

No model execution, cloud access, configuration changes, or study admission.
Price and timing scenarios are hypothetical at the measured A40 workload.
"""

import math
import statistics

from models.hardware_benchmark import summarize

COMPONENT_UNITS = {
    'routine_clips': 'routine',
    'primary_train_validation_clips': 'primary',
    'primary_test_clips': 'primary',
    'convergence_clips_including_two_audits': 'convergence',
    'fgsm_clips': 'fgsm',
    'random_clips': 'random',
    'calibration_three_candidate_clip_bundles': 'calibration_three_primary_candidates',
    'clean_updates': 'clean-finetune',
    'adversarial_updates': 'adversarial-finetune',
    'trained_selected_anchor_loads': 'load_anchor',
    'checkpoint_state_and_save_boundaries': None,
}
EVALUATION_COMPONENTS = frozenset((
    'routine_clips', 'primary_train_validation_clips', 'primary_test_clips',
    'convergence_clips_including_two_audits', 'fgsm_clips', 'random_clips'))
PRICE_GRID = (.29, .40, .50, .52, .53, .61)


def positive(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError('Require a positive finite ' + label + '.')
    return value


def screen_at(hours, rate):
    """Retain the historical eight hours,25% headroom,$10pilot+$40 reserves."""
    return (hours + 8) * 1.25 * positive(rate, 'hourly rate') + 50


def cost_review(workload, observations, rate, *, price_grid=PRICE_GRID):
    positive(rate, 'hourly rate')
    if (workload.get('classes') != 20 or workload.get('training_clips') != 100
            or workload.get('runs') != 8 or workload.get('updates_per_run') != 390):
        raise ValueError('This review retains the20-class/100-clip/eight-run design.')
    phases = workload['phases']
    if set(phases) != {'exploration', 'confirmation'}:
        raise ValueError('Keep both exploration and confirmation workloads.')
    for counts in phases.values():
        if set(counts) != set(COMPONENT_UNITS):
            raise ValueError('Missing or unknown study workload component.')
        if any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError('Study workload counts must be nonnegative integers.')
    counts = {k: sum(p[k] for p in phases.values()) for k in COMPONENT_UNITS}
    if any(n == 0 for n in counts.values()):
        raise ValueError('A required whole-study component was removed.')
    needed = {v for v in COMPONENT_UNITS.values() if v} | {'engine.state', 'checkpoint.save'}
    if set(observations) != needed:
        raise ValueError('Missing or unexpected native timing units.')
    summaries = {k: summarize(v) for k, v in observations.items()}
    if not price_grid or len(set(price_grid)) != len(price_grid):
        raise ValueError('Use distinct positive hypothetical rates.')
    for price in price_grid:
        positive(price, 'hypothetical hourly rate')
    scenarios = {}
    for label in ('median', 'maximum'):
        key = label + '_seconds'
        units = {k: (summaries['engine.state'][key] + summaries['checkpoint.save'][key]
                     if unit is None else summaries[unit][key]) for k, unit in COMPONENT_UNITS.items()}
        seconds = {k: counts[k] * units[k] for k in counts}
        total = math.fsum(seconds.values())
        hours = total / 3600
        rows = {k: dict(count=counts[k], unit_seconds=units[k], hours=seconds[k] / 3600,
                       fraction_of_timed_work=seconds[k] / total) for k in counts}
        cap_hours = (200 - 50) / (1.25 * rate) - 8
        evaluation_seconds = math.fsum(seconds[k] for k in EVALUATION_COMPONENTS)
        # The project cap dominates here; retain the separate planned compute
        # cap as well, rather than assuming they are interchangeable.
        total_rate_cap = (200 - 50) / ((hours + 8) * 1.25)
        compute_rate_cap = 160 / ((hours + 8) * 1.25)
        price_rows = []
        for price in price_grid:
            available_hours = (200 - 50) / (1.25 * price) - 8
            price_rows.append(dict(hypothetical_hourly_usd=price,
                total_at_unchanged_a40_units_usd=screen_at(hours, price),
                partial_arithmetic_within_200=screen_at(hours, price) <= 200,
                maximum_uniform_native_time_multiplier=(available_hours / hours if available_hours > 0 else None),
                another_gpu_performance_predicted=False))
        scenarios[label] = dict(timed_unit_hours=hours, total_usd=screen_at(hours, rate),
            component_contributions=rows,
            phase_timed_hours={name: math.fsum(n * units[k] for k, n in phase.items()) / 3600
                               for name, phase in phases.items()},
            native_evaluation_fraction=evaluation_seconds / total,
            calibration_fraction=seconds['calibration_three_candidate_clip_bundles'] / total,
            project_cap_hourly_rate=total_rate_cap, planned_compute_cap_hourly_rate=compute_rate_cap,
            binding_hourly_rate=min(total_rate_cap, compute_rate_cap),
            maximum_native_hours_at_current_rate=(cap_hours if cap_hours > 0 else None),
            uniform_native_time_reduction_needed=(max(0., 1 - cap_hours / hours) if cap_hours > 0 else None),
            uniform_native_throughput_speedup_needed=(max(1., hours / cap_hours) if cap_hours > 0 else None),
            price_sensitivity=price_rows,
            timing_sensitivity=[dict(uniform_native_time_multiplier=factor,
                                     total_usd=screen_at(hours * factor, rate)) for factor in (1., .90, .85)],
            complete_cost_estimate=False, research_admitted=False, budget_approved=False)
    return dict(kind='offline_a40_cost_attribution', scenarios=scenarios,
        retained_design={k: workload[k] for k in ('classes', 'training_clips', 'runs', 'updates_per_run')},
        quote_basis='Historical user-reported0.61/hour for the completed A40; no new quote or availability check.',
        allowance_hours=8, runtime_headroom_factor=1.25, pilot_reserve_usd=10,
        noncompute_reserve_usd=40, project_cap_usd=200, planned_compute_cap_usd=160,
        pilot_charges_not_added_again=True, price_grid_is_hypothetical=True,
        uniform_timing_scaling_is_hypothetical=True,
        complete_cost_estimate=False, research_admitted=False, budget_approved=False)


def attack_pair_review(primary, convergence):
    """Describe the saved combined-setting contrast, without a causal claim."""
    if not primary['complete'] or not convergence['complete']:
        raise ValueError('Only complete saved evaluations may be compared.')
    pc, cc = primary['contract'], convergence['contract']
    common = lambda c: {k: v for k, v in c.items() if k not in ('settings', 'sensitivity_banks')}
    if (common(pc) != common(cc) or pc['panel'] != 'train_monitor'
            or len(pc['video_ids']) != 1 or len(primary['records']) != 1 or len(convergence['records']) != 1
            or pc['sensitivity_banks'] != 0 or cc['sensitivity_banks'] != 2):
        raise ValueError('Keep the identical one-clip training/checkpoint/data/source/runtime contract.')
    p, c = primary['records'][0], convergence['records'][0]
    for payload, method, shape in ((primary, 'primary', (20, 2, 4, 4)),
                                   (convergence, 'convergence', (50, 2, 8, 4))):
        contract, row = payload['contract'], payload['records'][0]
        settings = contract['settings']
        dimensions = tuple(settings[k] for k in ('steps', 'restarts', 'attack_masks', 'scoring_masks'))
        if (settings['method'] != method or dimensions != shape or row['settings'] != settings
                or settings['epsilon_255'] != 4. or settings['step_size_255'] != 2.
                or row['method'] != method or row['video_id'] != contract['video_ids'][0]):
            raise ValueError('Saved methods/settings differ from the measured pair.')
        for name in ('clean_loss', 'adversarial_loss'):
            positive(row[name], 'native loss')
    if p['seeds']['score'] != c['seeds']['score'] or p['clean_loss'] != c['clean_loss']:
        raise ValueError('A paired comparison needs the exact same scoring bank and clean reference.')
    if len(c['scoring_audits']) != 2:
        raise ValueError('Keep both declared independent scoring audits.')
    banks = [dict(bank='main', clean_loss=c['clean_loss'], adversarial_loss=c['adversarial_loss'])]
    for index, audit in enumerate(c['scoring_audits']):
        if set(audit) != {'clean_loss', 'adversarial_loss'}:
            raise ValueError('Missing scoring-bank diagnostics.')
        for number in audit.values():
            positive(number, 'audit loss')
        banks.append(dict(bank='audit_' + str(index), **audit))
    for bank in banks:
        bank['adversarial_minus_clean_loss'] = bank['adversarial_loss'] - bank['clean_loss']
    difference = c['adversarial_loss'] - p['adversarial_loss']
    return dict(kind='saved_training_clip_attack_comparison', video_id=pc['video_ids'][0],
        primary_adversarial_loss=p['adversarial_loss'], convergence_adversarial_loss=c['adversarial_loss'],
        combined_configuration_loss_difference=difference,
        combined_configuration_relative_loss_difference=difference / p['adversarial_loss'],
        same_main_scoring_bank=True, same_clean_reference=True,
        changed_factors=dict(steps=[20, 50], attack_mask_count=[4, 8],
            attack_mask_prefix_shared=p['seeds']['attack'] == c['seeds']['attack'][:4],
            random_start_seed_shared=p['seeds']['start'] == c['seeds']['start']),
        scoring_banks=banks,
        raw_adversarial_loss_range=max(b['adversarial_loss'] for b in banks) - min(b['adversarial_loss'] for b in banks),
        paired_loss_increase_range=max(b['adversarial_minus_clean_loss'] for b in banks) - min(b['adversarial_minus_clean_loss'] for b in banks),
        iteration_effect_identified=False, cause_of_scoring_bank_variation_identified=False,
        study_attack_convergence_validated=False, held_out_result=False, research_admitted=False,
        next_diagnostic=dict(status='designed_not_implemented_or_run',
            purpose='Separate attack-step and EOT-mask effects with matched randomized inputs.',
            arms=[dict(steps=steps, attack_masks=masks) for masks in (4, 8) for steps in (20, 50)],
            restarts=2, epsilon_255=4., step_size_255=2.,
            attack_masks='One eight-draw bank; four-mask arms use its first four draws.',
            starts='Identical pre-generated start for each restart across all four arms.',
            scoring='Same independent four-mask main bank and two fixed four-mask audit banks in every arm.',
            candidate_selection='Attack-bank loss only; no selection using scoring/audit banks.',
            scope='Fixed pretrained checkpoint and training-only clips; no trajectory or held-out pixels.',
            artifact_namespace='Separate diagnostic contract; canonical study methods remain unchanged.',
            preferred_execution='Prepare a bounded university GPU handoff if available; no rental authorized.'))
