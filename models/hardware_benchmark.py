"""Explicit hardware-screen contracts; no cloud provisioning or study admission."""

import math
import statistics

GPU_MIN_GIB = {
    'NVIDIA RTX A5000': 22,
    'NVIDIA RTX A6000': 44,
    'NVIDIA A40': 44,
    'NVIDIA GeForce RTX 3090': 22,
    'NVIDIA GeForce RTX 4090': 22,
    'NVIDIA GeForce RTX 5090': 30,
}
SCREEN_PHASES = ('preflight', 'calibrate', 'features-initial',
                 'clean-finetune/reference', 'adversarial-finetune/reference',
                 'features-trained', 'controls')

# October 23 conservative ledger: both earlier $3 holds remain unresolved.
# This is a proposed next-session hold, applied only on a reviewed deployment.
UNRESERVED_PILOT_USD = 2.805
SESSION_HOLD_USD = 2.80


def validate_gpu(expected, actual, vram_bytes, visible_devices=1):
    if expected not in GPU_MIN_GIB:
        raise ValueError('Choose an explicitly supported GPU name.')
    if actual != expected or visible_devices != 1 or vram_bytes < GPU_MIN_GIB[expected] * 2**30:
        raise ValueError('Actual GPU, visible device count or VRAM differs from the reviewed contract.')


def validate_launch(expected, rate, now, deadline):
    if expected not in GPU_MIN_GIB:
        raise ValueError('Choose an explicitly supported GPU name.')
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 < rate <= .70:
        raise ValueError('This lower-cost screen requires a reviewed all-in running quote at or below $0.70/hour.')
    if type(now) is not int or type(deadline) is not int or not 1200 <= deadline - now <= 7200:
        raise ValueError('Use a deployment-based deadline with 20 minutes to two hours remaining.')
    if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
                for v in (UNRESERVED_PILOT_USD, SESSION_HOLD_USD))
            or SESSION_HOLD_USD > min(3, UNRESERVED_PILOT_USD)
            or rate * 2 > SESSION_HOLD_USD):
        raise ValueError('The proposed session hold exceeds the conservative pilot balance or cannot cover the running allowance.')
    return {'expected_gpu': expected, 'quoted_hourly_usd': rate,
            'deadline_epoch': deadline, 'maximum_billed_hours': 2,
            'maximum_inclusive_session_usd': SESSION_HOLD_USD,
            'new_reservation_exceeds_available_pilot_usd': SESSION_HOLD_USD > UNRESERVED_PILOT_USD,
            'prior_unreserved_pilot_usd': UNRESERVED_PILOT_USD, 'proposed_session_hold_usd': SESSION_HOLD_USD,
            'ledger_date': '2026-10-23', 'earlier_session_holds_pending_reconciliation_usd': [3, 3],
            'cloud_billing_stopped_by_runner': False, 'research_admitted': False}


def summarize(values):
    if not values or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('Missing, nonfinite or nonpositive fresh timing observations.')
    return {'count': len(values), 'minimum_seconds': min(values),
            'median_seconds': statistics.median(values), 'maximum_seconds': max(values)}


def economic_screen(reference, observations, rate):
    """Same counts/reserves as the review; fresh units only, never a GPU name guess."""
    if not math.isfinite(rate) or not 0 < rate <= .70:
        raise ValueError('Require the reviewed lower-cost quote.')
    all_counts = reference['variants']['focused_20x5']['workload']['phases']
    results = {}
    for stat in ('median', 'maximum'):
        timings = {}
        for component in next(iter(all_counts.values())):
            if component == 'checkpoint_state_and_save_boundaries':
                timings[component] = summarize(observations['engine.state'])[stat+'_seconds'] + summarize(observations['checkpoint.save'])[stat+'_seconds']
            else:
                method = {'routine_clips': 'routine', 'primary_train_validation_clips': 'primary',
                          'primary_test_clips': 'primary', 'convergence_clips_including_two_audits': 'convergence',
                          'fgsm_clips': 'fgsm', 'random_clips': 'random',
                          'calibration_three_candidate_clip_bundles': 'calibration_three_primary_candidates',
                          'clean_updates': 'clean-finetune', 'adversarial_updates': 'adversarial-finetune',
                          'trained_selected_anchor_loads': 'load_anchor'}[component]
                timings[component] = summarize(observations[method])[stat+'_seconds']
        hours = sum(count * timings[name] / 3600 for counts in all_counts.values() for name, count in counts.items())
        total = (hours + 8) * 1.25 * rate + 50
        results[stat] = {'timed_unit_projection_hours': hours,
                        'with_historical_allowances_and_reserves_usd': total,
                        'arithmetic_within_caps': total <= 200,
                        'full_study_cost_bound': False}
    return {'candidate_economic_screen': results, 'fresh_observation_counts': {k:len(v) for k,v in observations.items()},
            'exact_optimizer_recovery_checked': False, 'complete_cost_estimate': False,
            'research_admitted': False, 'budget_approved': False}
