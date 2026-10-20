#!/usr/bin/env python3
"""Project the staged proposal from pinned evidence; never admits or launches work."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.staged_subset import propose_subset


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(plan, pilot, attacks, *, classes, train_per_class):
    tr, ev, budget = plan['training'], plan['evaluation'], plan['budget']
    if (tr['conditions'] != ['clean-finetune', 'adversarial-finetune']
            or len(tr['exploration_seeds']) != 1 or len(tr['confirmation_seeds']) != 3):
        raise ValueError('This calculator requires one exploratory and three confirmation pairs.')
    seeds = tr['exploration_seeds'] + tr['confirmation_seeds']
    if len(set(seeds)) != 4:
        raise ValueError('Exploration and confirmation seeds must be distinct.')
    primary = {'steps': 20, 'restarts': 2, 'attack_masks': 4, 'scoring_masks': 4,
               'epsilon_255': 4, 'step_size_255': 1}
    routine = {**primary, 'steps': 5, 'restarts': 1, 'attack_masks': 1, 'scoring_masks': 1}
    training_attack = {k: v for k, v in routine.items() if k != 'scoring_masks'}
    measured = pilot['evaluation_attack']
    if (ev['primary_attack'] != primary or ev['routine_attack'] != routine
            or tr['training_attack'] != training_attack
            or not pilot['primary_evaluation_measured']
            or measured != {'steps': 20, 'restarts': 2, 'eot_masks': 4, 'scoring_masks': 4,
                            'epsilon': 4 / 255, 'step_size': 1 / 255}
            or attacks['epsilon'] != 4 / 255):
        raise ValueError('Changed attack settings require corresponding timing evidence.')
    if (tr['precision'] != pilot['precision'] or tr['precision'] != 'float32'
            or tr['context_patch_embedding'] != pilot['context_patch_embedding']
            or tr['effective_batch_size'] != pilot['effective_batch']
            or tr['effective_batch_size'] != 8 or tr['micro_batch_size'] != pilot['microbatch']):
        raise ValueError('Changed precision, batch, or projection needs a matching pilot.')
    if (not ev['shared_train_initialization'] or not ev['separate_validation_initializations']
            or ev['routine_epochs'] != 'every_epoch_plus_initialization'
            or plan['data']['train_monitor_per_class'] != 1
            or plan['data']['validation_per_class_per_phase'] != 1
            or plan['data']['test_per_class'] != 3):
        raise ValueError('Changed panel/reference schedule requires a revised calculator.')
    epochs, anchors = tr['epochs'], ev['primary_epochs']
    if sorted(set(anchors)) != anchors or not anchors or anchors[-1] != epochs or anchors[0] < 1:
        raise ValueError('Primary anchors must be unique, sorted, valid, and include the final epoch.')
    if any(type(n) is not int or n < 1 for n in (classes, train_per_class, epochs)):
        raise ValueError('Counts must be positive integers.')
    if (budget['compute_usd'] + budget['noncompute_reserve_usd'] != budget['total_usd']
            or not 0 <= budget['spent_pilot_usd'] <= budget['pilot_reserve_total_usd']
            or budget['runtime_headroom_factor'] < 1):
        raise ValueError('Budget envelopes, recorded spend, or headroom are inconsistent.')
    seconds = pilot['evaluation_seconds_per_clip']
    conv = ev['convergence_attack']
    if conv != {'steps': 50, 'restarts': 2, 'attack_masks': 8, 'scoring_masks': 4}:
        raise ValueError('Changed convergence workload needs a revised projection.')
    diagnostic = min(classes, ev['diagnostic_clips'])
    updates = math.ceil(classes * train_per_class / 8) * epochs
    rows = []

    def add(phase, component, count, unit, sec=None, basis='unmeasured'):
        if count < 0 or (sec is not None and (not math.isfinite(sec) or sec <= 0)):
            raise ValueError('Invalid count or timing observation.')
        rows.append(dict(phase=phase, component=component, count=count, unit=unit,
                         seconds_per_unit=sec, timing_basis=basis,
                         projected_hours=None if sec is None else count * sec / 3600))

    for phase, pairs in (('exploration', 1), ('confirmation', 3)):
        runs = pairs * 2
        # Training initialization is common. Validation panels differ by phase.
        initial_panels = 2 if phase == 'exploration' else 1
        routine_n = classes * (initial_panels + 2 * runs * epochs)
        primary_n = classes * (initial_panels + 2 * runs * len(anchors))
        selected = 1 + 3 * pairs  # Initial, adv best/final, clean final.
        for condition in tr['conditions']:
            add(phase, condition, pairs * updates, 'effective-batch update',
                pilot['conditions'][condition]['mean_seconds'], 'measured_unit_extrapolation')
        add(phase, 'routine_pgd5', routine_n, 'clip-checkpoint',
            attacks['attacks']['pgd5']['seconds'], 'measured_unit_extrapolation')
        add(phase, 'primary_train_validation', primary_n, 'clip-checkpoint',
            seconds, 'measured_unit_extrapolation')
        add(phase, 'convergence_pgd50', selected * diagnostic, 'clip-checkpoint',
            seconds * 5, 'gradient_work_ratio_proxy')
        if phase == 'exploration':
            add(phase, 'training_attack_step_calibration', diagnostic * len(ev['calibration_step_sizes_255']),
                'training clip and candidate', seconds, 'same_workload_proxy')
        test_n = selected * classes * 3 if phase == 'confirmation' else 0
        add(phase, 'primary_final_test', test_n, 'clip-checkpoint', seconds, 'measured_unit_extrapolation')
        # Essential unmeasured work remains null, even with a spending allowance.
        add(phase, 'clean_native_scoring', routine_n + primary_n + test_n, 'clip-checkpoint')
        add(phase, 'random_and_fgsm_baselines', selected * diagnostic * 2, 'validation clip-baseline')
        add(phase, 'independent_scoring_bank_audit', selected * diagnostic * 2, 'extra scoring bank')
        add(phase, 'probe_and_feature_diagnostics', selected, 'checkpoint bundle')
        add(phase, 'data_integrity_decode_targets', 1, 'phase preparation')
        add(phase, 'checkpoint_write_resume_load_export', runs, 'full trajectory')
        add(phase, 'setup_transfer_and_idle', 1, 'phase billed overhead')

    price, headroom = budget['price_usd_per_hour'], budget['runtime_headroom_factor']
    if not math.isfinite(price) or price <= 0:
        raise ValueError('Hourly price must be positive and finite.')
    stages = {}
    for phase in ('exploration', 'confirmation'):
        phase_rows = [r for r in rows if r['phase'] == phase]
        timed = sum(r['projected_hours'] for r in phase_rows if r['timing_basis'] == 'measured_unit_extrapolation')
        proxy = sum(r['projected_hours'] for r in phase_rows
                    if r['projected_hours'] is not None and r['timing_basis'] != 'measured_unit_extrapolation')
        allowance = budget['untimed_allowance_hours'][phase]
        if not math.isfinite(allowance) or allowance < 0:
            raise ValueError('Untimed allowance must be nonnegative and finite.')
        stages[phase] = {
            'measured_unit_projection_hours': timed, 'proxy_projection_hours': proxy,
            'untimed_allowance_hours_not_measurement': allowance,
            'planning_compute_usd_with_headroom': (timed + proxy + allowance) * headroom * price,
            'unmeasured_components': [r['component'] for r in phase_rows if r['projected_hours'] is None],
        }
    study_compute = sum(s['planning_compute_usd_with_headroom'] for s in stages.values())
    # Past pilot spending is INSIDE the $10 envelope, not added to it again.
    total = study_compute + budget['pilot_reserve_total_usd'] + budget['noncompute_reserve_usd']
    return {
        'classes': classes, 'training_clips': classes * train_per_class,
        'updates_per_run': updates, 'training_runs_if_both_stages': 8,
        'final_test_checkpoint_upper_bound': 10,
        'convergence_checkpoint_panel_pairs': 14,
        'stages': stages, 'components': rows,
        'ledger': {
            'pilot_spent_user_reported_usd': budget['spent_pilot_usd'],
            'pilot_unspent_reserved_usd': budget['pilot_reserve_total_usd'] - budget['spent_pilot_usd'],
            'study_compute_planning_usd': study_compute,
            'total_compute_including_pilot_reserve_usd': study_compute + budget['pilot_reserve_total_usd'],
            'noncompute_reserved_usd': budget['noncompute_reserve_usd'],
            'total_planning_envelope_usd': total,
            'nominal_margin_to_total_cap_usd': budget['total_usd'] - total,
            'arithmetic_within_cap': total <= budget['total_usd'],
        },
        'complete_cost_estimate': False, 'budget_approved': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/budget_study_plan.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path; existing reports are never overwritten.')
    plan = yaml.safe_load(args.plan.read_text())
    for name, expected in plan['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            parser.error(f'Source changed; revise proposal before estimating: {name}')
    pilot = json.loads((ROOT / plan['pilot']).read_text())
    attacks = json.loads((ROOT / plan['attack_check']).read_text())
    parent = {}
    for split in ('train', 'validation', 'test'):
        with (ROOT / plan['data']['parent_manifests'] / f'{split}.csv').open() as handle:
            parent[split] = list(csv.DictReader(handle))
    variants = {}
    subsets = {}
    for name, counts in plan['data']['variants'].items():
        variants[name] = estimate(plan, pilot, attacks, **counts)
        subsets[name] = propose_subset(parent, **counts, seed=plan['data']['subset_seed'],
            excluded_groups=plan['data']['exclude_from_exploration_holdout_groups'])
        variants[name]['metadata_counts_verified'] = subsets[name]['counts']
    paths = [args.plan.resolve(), Path(__file__).resolve(), ROOT / 'datasets/staged_subset.py']
    report = {
        'kind': 'staged_planning_projection_not_a_benchmark', 'as_of': plan['as_of'],
        'complete_cost_estimate': False, 'budget_approved': False,
        'source_sha256': {**plan['source_sha256'], **{str(p.relative_to(ROOT)): sha256(p) for p in paths}},
        'recommended_variant': plan['data']['recommended_variant'],
        'variants': variants,
        'proposed_subset': subsets[plan['data']['recommended_variant']],
        'limitations': [
            'Aggregate runtimes extrapolate five updates and one-clip attacks on RTX 5090.',
            'PGD50 and calibration estimates are proxies; essential missing costs are null.',
            'Eight untimed GPU hours are an allowance, not evidence those tasks fit.',
            'Historical $1.20/hour is an assumption, not a live quote or tax-inclusive guarantee.',
            '$0.592 is user-reported session cost; the full $10 pilot reserve includes it.',
            'Proposal changes data scope, update count and evaluation; original study is preserved.',
            'Metadata separation is verified; video corruption/content duplication remains unchecked.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    for name, result in variants.items():
        print(f"{name}: ${result['ledger']['total_planning_envelope_usd']:.2f} planning envelope; "
              f"{result['updates_per_run']} updates/run. NOT ADMITTED.")


if __name__ == '__main__':
    main()
