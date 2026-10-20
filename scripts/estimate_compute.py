#!/usr/bin/env python3
"""Reproduce a partial workload projection; unknown costs never become zero.

Reads the separate planning proposal and existing Mac measurements. No GPU,
network, data decoding, cloud account access, or job launch is performed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(plan, study, pilot, attacks, *, train_per_class, primary_every_epoch=False):
    ev = plan['evaluation']
    epochs, seeds = study['training']['epochs'], len(study['seeds'])
    conditions = len([c for c in study['conditions'] if c != 'pretrained'])
    if conditions != 2 or len(set(study['seeds'])) != seeds:
        raise ValueError('This proposal requires two conditions and distinct seeds.')
    batch = study['training']['effective_batch_size']
    if pilot['effective_batch'] != batch or not pilot['primary_evaluation_measured']:
        raise ValueError('Use a full batch-eight, primary-evaluation pilot, not a reduced diagnostic.')
    primary = ev['primary_attack']
    if (ev['monitoring_splits'] != ['train', 'validation'] or ev['primary_splits'] != ['train', 'validation']
            or not ev['primary_include_shared_initialization']
            or ev['monitoring_epochs'] != 'every_epoch_plus_shared_initialization'):
        raise ValueError('Changed split/reference scheduling requires a revised workload calculation.')
    expected = {'steps': 20, 'restarts': 2, 'attack_masks': 4, 'scoring_masks': 4,
                'epsilon_255': 4, 'step_size_255': 1}
    monitor = {'steps': 5, 'restarts': 1, 'attack_masks': 1, 'scoring_masks': 1,
               'epsilon_255': 4, 'step_size_255': 1}
    if primary != expected or ev['monitoring_attack'] != monitor:
        raise ValueError('Changed attack settings need corresponding timing evidence.')
    anchors = ev['primary_epochs']
    if (sorted(set(anchors)) != anchors or any(e < 1 or e > epochs for e in anchors)
            or epochs not in anchors):
        raise ValueError('Use unique sorted primary epochs including the final epoch.')
    if train_per_class not in (study['data']['videos_per_class']['train'],
                               study['data']['budget_fallback_train_per_class']):
        raise ValueError('Only the draft training subset or declared fallback may be projected.')
    if plan['budget']['compute_usd'] + plan['budget']['noncompute_reserve_usd'] != plan['budget']['total_usd']:
        raise ValueError('Budget envelopes do not add up.')

    runs = seeds * conditions
    classes = ev['classes']
    train = classes * train_per_class
    validation = classes * study['data']['videos_per_class']['validation']
    test = classes * ev['final_test_clips_per_class']
    subset = classes * ev['monitoring_clips_per_class_per_split']
    diagnostic = classes * ev['diagnostic_clips_per_class']
    updates = math.ceil(train / batch) * epochs * seeds  # Per condition.
    # One common initialization, not six duplicate pretrained evaluations.
    primary_checkpoints = 1 + runs * len(anchors)
    selected_checkpoints = 1 + 2 * runs  # Upper bound; best may equal final.
    routine_count = subset * (1 + runs * epochs) * 2
    primary_count = subset * primary_checkpoints * 2
    if primary_every_epoch:
        primary_count = subset * ((1 + runs * epochs) + primary_checkpoints)
        routine_count = subset * (1 + runs * epochs)  # Cheap train monitoring only.
        # Add train-side primary evaluation for best epochs outside anchors.
        primary_count += subset * runs
    primary_seconds = pilot['evaluation_seconds_per_clip']
    rows = []

    def row(name, count, unit, seconds=None, evidence='unmeasured', note=''):
        if count < 0 or (seconds is not None and (not math.isfinite(seconds) or seconds <= 0)):
            raise ValueError('Work counts and timing observations must be valid.')
        rows.append({'component': name, 'count': count, 'unit': unit,
                     'seconds_per_unit': seconds, 'timing_basis': evidence,
                     'projected_hours': None if seconds is None else count * seconds / 3600,
                     'note': note})

    for condition in ('clean-finetune', 'adversarial-finetune'):
        row(condition, updates, 'batch-eight update', pilot['conditions'][condition]['mean_seconds'],
            'measured_unit', 'Five cached-clip updates; aggregate is an extrapolation.')
    row('routine_pgd5', routine_count, 'clip', attacks['attacks']['pgd5']['seconds'],
        'measured_unit', 'One-clip released-model timing; includes one independent scoring draw.')
    row('primary_train_validation', primary_count, 'clip', primary_seconds, 'measured_unit')
    row('primary_final_test', selected_checkpoints * test, 'clip', primary_seconds, 'measured_unit',
        'Upper bound of 13 distinct checkpoints; reuse perturbations for native-attack transfer.')
    row('attack_step_calibration', diagnostic * len(ev['calibration_step_sizes_255']), 'training clip',
        primary_seconds, 'same_workload_proxy', 'Proposed three step sizes on pretrained training clips only.')
    convergence = ev['convergence_attack']
    ratio = (convergence['steps'] * convergence['restarts'] * convergence['attack_masks'] /
             (primary['steps'] * primary['restarts'] * primary['attack_masks']))
    row('pgd50_eot8_validation', selected_checkpoints * diagnostic, 'clip', primary_seconds * ratio,
        'operation_count_proxy', 'Ratio of gradient work only; must measure actual stronger-attack runtime.')
    extra_eps = [e for e in ev['epsilon_sweep_255'] if e not in (0, primary['epsilon_255'])]
    row('epsilon_sweep_extra', selected_checkpoints * diagnostic * len(extra_eps), 'clip',
        primary_seconds, 'same_workload_proxy', 'Reuse epsilon 4 primary output and epsilon 0 clean score.')

    row('clean_monitor_scoring', routine_count, 'one-mask clean clip')
    row('clean_primary_scoring', primary_count + selected_checkpoints * test, 'four-mask clean clip')
    for baseline in ('random_noise', 'fgsm_eot4', 'embedding_pgd20'):
        row(baseline, selected_checkpoints * diagnostic, 'validation clip',
            note='Include four independent native scoring masks; baseline definitions need implementation.')
    row('decode_and_integrity', train + validation + test, 'clip',
        note='Test preprocessing occurs only after choices are locked; source integrity also needs auditing.')
    row('fixed_target_cache', train + validation + test, 'clip')
    row('probe_clean_feature_extraction', selected_checkpoints * (train + validation + test), 'clip')
    row('probe_fit', selected_checkpoints * ev['probe_candidate_fits_per_checkpoint'], 'candidate fit',
        note='Train-only fitting, validation-only hyperparameter selection; four candidates are a planning reserve.')
    row('direct_classifier_pgd20', selected_checkpoints * test, 'test clip',
        note='Two restarts at 4/255; require convergence checks before final use.')
    row('direct_classifier_convergence', selected_checkpoints * diagnostic * len(ev['calibration_step_sizes_255']),
        'validation clip and step-size candidate',
        note='PGD-50, two restarts, three step sizes; calibrate before opening the test set.')
    row('native_attack_transfer_scoring', selected_checkpoints * test, 'test clip',
        note='Score the already-generated native adversary with the matched fitted probe.')
    row('epoch_checkpoint_bundle_write', runs * epochs, 'epoch weight and resume bundle')
    row('checkpoint_loading', 1 + runs + selected_checkpoints, 'load',
        note='Online anchor evaluation, then selected-checkpoint evaluation; interruptions need extra loads.')
    row('setup_transfer_export', 1, 'billed preparation/export session')

    measured = sum(r['projected_hours'] for r in rows if r['timing_basis'] == 'measured_unit')
    proxies = sum(r['projected_hours'] for r in rows if r['projected_hours'] is not None and r['timing_basis'] != 'measured_unit')
    unknown = [r['component'] for r in rows if r['projected_hours'] is None]
    budget = plan['budget']
    remaining = budget['compute_usd'] - budget['spent_compute_usd'] - budget['pilot_reserve_usd']
    if remaining <= 0:
        raise ValueError('No compute envelope remains after reserves.')
    thresholds = []
    candidates = [dict(c, provider='Lambda', quote_verified=False) for c in plan['lambda']['candidates']]
    alt = plan['alternative_quote_to_check']
    candidates.append({'id': 'runpod_a6000_advertised', 'name': alt['name'], 'provider': alt['provider'],
                       'hourly_usd': alt['advertised_pod_hourly_usd'], 'quote_verified': False})
    for c in candidates:
        rate = c['hourly_usd']
        thresholds.append({**c, 'remaining_billed_hours': remaining / rate,
                           'maximum_base_hours_with_headroom': remaining / rate / budget['runtime_headroom_factor'],
                           'required_uniform_speedup_for_timed_components': measured * budget['runtime_headroom_factor'] * rate / remaining,
                           'required_uniform_speedup_including_proxies': (measured + proxies) * budget['runtime_headroom_factor'] * rate / remaining,
                           'is_observed_gpu_speedup': False})
    components = pilot['checkpoint']['components']
    trainable = components['encoder']['elements'] + components['predictor']['elements']
    weight_bytes = 4 * trainable
    storage = {'fp32_epoch_weights_gib': runs * epochs * weight_bytes / 2**30,
               'two_rolling_resume_slots_gib': 2 * 3 * weight_bytes / 2**30,
               'processed_fp32_clips_gib': (train + validation + test) * 3 * 16 * 256 * 256 * 4 / 2**30,
               'fixed_target_cache_fp32_gib': (train + validation + test) * 2048 * 1024 * 4 / 2**30,
               'basis': 'Tensor-only estimates; excludes raw video, official checkpoint, framework overhead and temporary writes.'}
    return {'training_clips': train, 'training_runs': runs,
            'selection_schedule': 'primary_validation_every_epoch' if primary_every_epoch else 'primary_validation_at_declared_anchors',
            'selected_checkpoint_count_upper_bound': selected_checkpoints,
            'components': rows, 'measured_component_projection_hours': measured,
            'proxy_component_projection_hours': proxies,
            'partial_projection_hours': measured + proxies,
            'unmeasured_components': unknown, 'complete_cost_estimate': False,
            'budget_approved': False, 'cloud_break_even_conditions': thresholds, 'storage': storage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/compute_plan.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output file; existing estimates are never overwritten.')
    plan = yaml.safe_load(args.plan.read_text())
    study_path = ROOT / plan['study']
    if sha256(study_path) != plan['study_sha256']:
        parser.error('The study changed; revise the proposal before estimating it.')
    study = yaml.safe_load(study_path.read_text())
    pilot = json.loads((ROOT / plan['pilot']).read_text())
    attacks = json.loads((ROOT / plan['attack_check']).read_text())
    variants = {name: estimate(plan, study, pilot, attacks, **kw) for name, kw in (
        ('proposed_anchors', {'train_per_class': 20}),
        ('declared_training_fallback', {'train_per_class': 10}),
        ('primary_validation_every_epoch', {'train_per_class': 20, 'primary_every_epoch': True}))}
    report = {'kind': 'planning_projection_not_a_benchmark', 'as_of': plan['as_of'],
              'complete_cost_estimate': False, 'budget_approved': False,
              'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                                (args.plan.resolve(), study_path, ROOT / plan['pilot'], ROOT / plan['attack_check'], Path(__file__).resolve())},
              'variants': variants,
              'limitations': ['All aggregate runtimes are extrapolations from short Mac samples.',
                              'No CUDA speedup or mixed-precision improvement has been measured.',
                              'Proxy rows require measurements; missing rows are null, not zero.',
                              'Uniform speedup thresholds are algebraic requirements, not hardware predictions.',
                              'Price listings do not verify live availability, cloud tier, taxes, or storage cost.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    for name, result in variants.items():
        print(f"{name}: {result['measured_component_projection_hours']:.2f} h from timed units + "
              f"{result['proxy_component_projection_hours']:.2f} h proxies; "
              f"{len(result['unmeasured_components'])} unmeasured components. NOT ADMITTED.")


if __name__ == '__main__':
    main()
