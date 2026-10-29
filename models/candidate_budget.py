"""Offline EOT8 unit-cost screen; no tensors, cloud access or research admission."""

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics

from models.measured_budget import workload

PLAN_CANONICAL = '4be35ffe85d1e09a96d6d6c82ee64e25f9444a65681506a02837d9d4e2cab8cc'
CANDIDATE_CANONICAL = 'dc0dba86c92f027296678918b3b8fe1ed6795bace094acde0f0a22c027bf9304'
INPUT_PINS = {
    'configs/budget_study_plan.yaml': '70de4b44a71efe8e7ee45094f525992c9ea8f8ef45ec0de999ea6047a5f888fa',
    'configs/research_attack_candidate_v1.json': '1d4901b2893b6cbf35e11e3ca12b59927c8b15a1662b85abedca439520009051',
    'models/measured_budget.py': 'd1e683a37ab70a5a35631d4fb96f76ea764bdea49053b53abd00472d427e158f',
    'report/checks/2026-10-28-candidate-benchmark-verification.json': 'cf4494c16578832a009782b51bae3fd4b4a257ec3bfd995d6e7ffb6b106db604',
}
DIMENSIONS = {'routine': (5, 1, 1, 1), 'primary': (20, 2, 8, 4),
              'convergence': (50, 2, 8, 4), 'fgsm': (1, 1, 4, 4), 'random': (1, 1, 4, 4)}
UNKNOWN_COSTS = (
    'Full-panel loading, source/pixel integrity, decoding and fixed-target caching',
    'Full-size panel identity/journal/transfer overhead beyond the measured small-panel wrapper samples',
    'Full 100-training-clip features, ridge fit and held-out feature/probe scoring',
    'Full-run checkpoint retention, temporary disk peaks, export and verified backup',
    'Future billed setup, transfers, idle time, restarts and failed work',
    'Stopped storage, taxes and other charges outside the recorded running quote',
    'Any additional primary-with-audit control replays required by the eventual frozen research protocol',
)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Require a positive finite measured time or rate.')
    return value


def statistic(samples, which):
    if not isinstance(samples, list) or not samples:
        raise ValueError('A missing measurement is unknown, never zero.')
    values = [positive(s['seconds']) for s in samples]
    origins = [(s['path'], s['field']) for s in samples]
    if len(set(origins)) != len(origins):
        raise ValueError('Duplicated measurement provenance.')
    if which not in ('median', 'maximum'):
        raise ValueError('Use median or maximum-observed samples, not a bound.')
    return statistics.median(values) if which == 'median' else max(values)


def collect_units(evaluations, timings):
    """Native clip time and disjoint wrapper time; calibration counted as a bundle."""
    units = {name: [] for name in DIMENSIONS}
    units.update({name + '_panel_overhead': [] for name in DIMENSIONS})
    units.update({name: [] for name in ('clean_updates', 'adversarial_updates', 'engine.state',
        'checkpoint.save', 'trained_selected_anchor_loads', 'calibration_three_candidate_clip_bundles')})
    for path, value in evaluations.items():
        c, rows = value['contract'], value['records']
        settings, method = c['settings'], c['settings']['method']
        if (not value['complete'] or c['panel'] not in ('train', 'train_monitor') or not rows
                or [r['video_id'] for r in rows] != c['video_ids']
                or method not in DIMENSIONS or settings['profile'] != 'research-primary-eot8-v1'
                or tuple(settings[k] for k in ('steps', 'restarts', 'attack_masks', 'scoring_masks')) != DIMENSIONS[method]
                or settings['epsilon_255'] != 4 or settings['step_size_255'] not in (.5, 1, 2)
                or (method == 'routine' and settings['step_size_255'] != 1)
                or type(c['sensitivity_banks']) is not int
                or (c['sensitivity_banks'] != (2 if method == 'convergence' else 0)
                    and not (method == 'primary' and c['sensitivity_banks'] == 2))):
            raise ValueError('Incomplete, nontraining or mismatched candidate evaluation.')
        for i, row in enumerate(rows):
            if (row['settings'] != settings or row['method'] != method
                    or len(row['scoring_audits']) != c['sensitivity_banks']):
                raise ValueError('Clip settings/audits differ from its contract.')
            positive(row['seconds'])
            # Audited-primary is an extra control unit, not an ordinary primary sample.
            if method == 'primary' and c['sensitivity_banks']:
                continue
            units[method].append({'seconds': row['seconds'], 'path': path,
                'field': f'/payload/records/{i}/seconds', 'step_size_255': settings['step_size_255']})
    for path, rows in timings.items():
        for i, row in enumerate(rows):
            if row['status'] != 'ok':
                raise ValueError('Failed or incomplete timing span.')
            positive(row['seconds'])
            source = {'seconds': row['seconds'], 'path': path, 'field': f'/lines/{i}/seconds'}
            name = row['name']
            if name == 'engine.update':
                group = 'adversarial_updates' if Path(path).stem.startswith('adversarial-finetune-') else 'clean_updates'
                if not Path(path).stem.startswith(('clean-finetune-', 'adversarial-finetune-')):
                    raise ValueError('Unrecognized training update provenance.')
                units[group].append(source)
            elif name in ('engine.state', 'checkpoint.save'):
                units[name].append(source)
            elif name == 'candidate_anchor_load':
                units['trained_selected_anchor_loads'].append(source)
            elif name == 'candidate_three_step_bundle':
                units['calibration_three_candidate_clip_bundles'].append(source)
            elif name == 'candidate_evaluation':
                key = 'run/evaluations/' + row['unit'] + '/evaluation.json'
                value = evaluations[key]
                c = value['contract']; method = c['settings']['method']
                if method != row['method']:
                    raise ValueError('Wrapper and native evaluation method differ.')
                if method == 'primary' and c['sensitivity_banks']:
                    continue
                residual = row['seconds'] - math.fsum(r['seconds'] for r in value['records'])
                positive(residual)
                units[method + '_panel_overhead'].append({**source, 'seconds': residual,
                    'field': f'/lines/{i}/seconds minus native records', 'native_evaluation_path': key,
                    'sample_panel_clips': len(value['records'])})
    expected = {'primary': 8, 'routine': 1, 'convergence': 1, 'fgsm': 1, 'random': 1,
        'primary_panel_overhead': 3, 'routine_panel_overhead': 1, 'convergence_panel_overhead': 1,
        'fgsm_panel_overhead': 1, 'random_panel_overhead': 1, 'clean_updates': 4,
        'adversarial_updates': 4, 'engine.state': 8, 'checkpoint.save': 8,
        'trained_selected_anchor_loads': 3, 'calibration_three_candidate_clip_bundles': 1}
    if {k: len(v) for k, v in units.items()} != expected:
        raise ValueError('Expected all benchmark units; absent units cannot be silently omitted.')
    return units


def project(plan, candidate, units, rate):
    if canonical(plan) != PLAN_CANONICAL or canonical(candidate) != CANDIDATE_CANONICAL:
        raise ValueError('Keep the exact reviewed study, candidate, schedule and caps.')
    rate = Fraction(str(positive(rate)))
    budget = plan['budget']; headroom = Fraction(str(budget['runtime_headroom_factor']))
    pilot, noncompute = budget['pilot_reserve_total_usd'], budget['noncompute_reserve_usd']
    study_cap = budget['compute_usd'] - pilot
    counts = workload(plan, classes=20, train_per_class=5)
    scenarios = {}
    for which in ('median', 'maximum'):
        observed = {k: Fraction.from_float(statistic(v, which)) for k, v in units.items()}
        unit = {method + '_clips': observed[method] for method in ('routine', 'fgsm', 'random')}
        unit.update(primary_train_validation_clips=observed['primary'], primary_test_clips=observed['primary'],
            convergence_clips_including_two_audits=observed['convergence'],
            calibration_three_candidate_clip_bundles=observed['calibration_three_candidate_clip_bundles'],
            checkpoint_state_and_save_boundaries=observed['engine.state'] + observed['checkpoint.save'],
            trained_selected_anchor_loads=observed['trained_selected_anchor_loads'],
            clean_updates=observed['clean_updates'], adversarial_updates=observed['adversarial_updates'])
        phases = {}; all_rows = []
        for phase, numbers in counts['phases'].items():
            rows = {k: (n, unit[k]) for k, n in numbers.items()}
            for category, method, clips_per_panel in (
                    ('routine', 'routine', 20), ('primary_train_validation', 'primary', 20),
                    ('primary_test', 'primary', 60), ('convergence', 'convergence', 10),
                    ('fgsm', 'fgsm', 10), ('random', 'random', 10)):
                key = 'convergence_clips_including_two_audits' if category == 'convergence' else category + '_clips'
                n = numbers[key]
                if n % clips_per_panel:
                    raise ValueError('Cannot derive integral evaluation panel counts.')
                rows[category + '_panel_overhead'] = (n // clips_per_panel, observed[method + '_panel_overhead'])
            rows = {k: {'count': n, 'seconds_per_unit': float(v), 'projected_hours': float(n * v / 3600)}
                    for k, (n, v) in rows.items()}
            all_rows.extend(rows.items())
            hours = sum(Fraction(r['count']) * Fraction.from_float(r['seconds_per_unit']) / 3600 for r in rows.values())
            allowance = Fraction(budget['untimed_allowance_hours'][phase])
            phases[phase] = {'components': rows, 'timed_unit_extrapolation_hours': float(hours),
                'historical_untimed_allowance_hours_not_a_bound': float(allowance),
                'compute_with_headroom_and_historical_allowance_usd': float((hours + allowance) * headroom * rate)}
        hours = sum(Fraction(r['count']) * Fraction.from_float(r['seconds_per_unit']) / 3600 for _, r in all_rows)
        allowance = sum(Fraction(v) for v in budget['untimed_allowance_hours'].values())
        primary = sum(Fraction(r['count']) * Fraction.from_float(r['seconds_per_unit']) / 3600
                      for k, r in all_rows if k in ('primary_train_validation_clips', 'primary_test_clips'))
        wrapper = sum(Fraction(r['count']) * Fraction.from_float(r['seconds_per_unit']) / 3600
                      for k, r in all_rows if k.endswith('_panel_overhead'))
        compute = (hours + allowance) * headroom * rate
        available = study_cap / (headroom * rate) - allowance
        primary_room = available - (hours - primary)
        scenarios[which] = {'phases': phases, 'timed_unit_extrapolation_hours': float(hours),
            'observed_panel_wrapper_extrapolation_hours': float(wrapper),
            'primary_native_fraction_of_timed_work': float(primary / hours),
            'with_headroom_and_historical_allowance_hours': float((hours + allowance) * headroom),
            'total_without_historical_allowance_usd': float(hours * headroom * rate + pilot + noncompute),
            'total_with_historical_allowance_usd': float(compute + pilot + noncompute),
            'compute_including_full_pilot_reserve_usd': float(compute + pilot),
            'excess_over_total_cap_usd': float(max(Fraction(0), compute + pilot + noncompute - budget['total_usd'])),
            'arithmetic_within_caps': compute + pilot <= budget['compute_usd'] and compute + pilot + noncompute <= budget['total_usd'],
            'break_even_all_in_rate_at_same_sample_throughput_usd': float(study_cap / ((hours + allowance) * headroom)),
            'uniform_timed_work_speedup_needed_at_recorded_rate': float(hours / available) if available > 0 else None,
            'primary_only_speedup_needed_at_recorded_rate': float(primary / primary_room) if primary_room > 0 else None}
    return {'workload': counts, 'scenarios': scenarios, 'quote_usd_per_hour': float(rate),
        'unknown_costs': [{'component': name, 'hours': None, 'usd': None} for name in UNKNOWN_COSTS],
        'budget_approved': False, 'research_admitted': False, 'complete_cost_estimate': False,
        'full_study_cost_bound_usd': None, 'research_step_size_255': None}


def build_report(root):
    """Verify the reviewed copy and exact source tree before using any measurements."""
    import yaml
    root = Path(root).resolve(); verified = {}
    def checked(name, expected=None):
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError('Unsafe/missing reviewed input: ' + name)
        value = sha(path)
        if expected is not None and value != expected:
            raise ValueError('Pinned input changed: ' + name)
        verified[name] = value
        return path
    for name, expected in INPUT_PINS.items():
        checked(name, expected)
    review = json.loads((root / 'report/checks/2026-10-28-candidate-benchmark-verification.json').read_text())
    for name, expected in review['input_sha256'].items():
        checked(name, expected)
    archive = review['evidence_archive']; checked(archive['path'], archive['sha256'])
    directory = (root / archive['path']).parent
    specification = json.loads(checked(str((directory / 'evidence-source-overlay.tar.gz.manifest.json').relative_to(root))).read_text())
    if specification['sha256'] != archive['sha256'] or len(specification['files']) != 169:
        raise ValueError('Evidence archive identity changed.')
    extracted = directory / 'extracted'
    for name, item in specification['files'].items():
        path = checked(str((extracted / name).relative_to(root)), item['sha256'])
        if path.stat().st_size != item['bytes']:
            raise ValueError('Evidence member size changed.')
    def record(path):
        value = json.loads(path.read_text())
        if set(value) != {'payload', 'sha256'} or canonical(value['payload']) != value['sha256']:
            raise ValueError('Invalid checksummed evidence record.')
        return value['payload']
    plan = yaml.safe_load((root / 'configs/budget_study_plan.yaml').read_text())
    candidate = json.loads((root / 'configs/research_attack_candidate_v1.json').read_text())
    launch = json.loads((extracted / 'setup/launch.json').read_text())
    for name, expected in launch['candidate_binding']['source_sha256'].items():
        checked(name, expected)
    session = record(extracted / 'run/session.json')
    if (not review['all_required_phases_complete'] or review['exit_status'] != 0 or not session['complete']
            or [v['phase'] for v in session['phases']] != launch['required_phases']):
        raise ValueError('Require the complete independently reviewed diagnostic benchmark.')
    for condition in ('clean-finetune', 'adversarial-finetune'):
        recovery = record(extracted / ('run/recovery-' + condition + '.json'))
        if (not recovery['exact_match'] or recovery['compared_leaves'] != 3899
                or recovery['logical_signatures'][0] != recovery['logical_signatures'][1]):
            raise ValueError('Require both exact recovery comparisons.')
    evaluations = {str(p.relative_to(extracted)): record(p)
        for p in sorted((extracted / 'run').glob('evaluations/*/evaluation.json')) +
                 sorted((extracted / 'run').glob('calibration-diagnostic/*/evaluation.json'))}
    timings = {str(p.relative_to(extracted)): [json.loads(line) for line in p.read_text().splitlines()]
               for p in sorted((extracted / 'run/timings').glob('*.jsonl'))}
    units = collect_units(evaluations, timings)
    result = project(plan, candidate, units, float(launch['total_hourly_usd']))
    ledger = json.loads((extracted / 'setup/budget-envelope.json').read_text())
    reserve = Fraction(launch['inclusive_session_budget_usd']) + Fraction(str(ledger['pilot_spent_and_other_holds_usd']))
    for name in ('models/candidate_budget.py', 'scripts/estimate_candidate_study.py', 'tests/test_candidate_budget.py'):
        checked(name)
    return {'kind': 'candidate-eot8-measured-unit-budget-screen-v1', 'date_Pacific': '2026-10-28',
        'source_sha256': verified, 'measurement_units': units, 'economics': result,
        'quote': {'usd_per_hour': launch['total_hourly_usd'], 'as_of': 'user-approved October 28 session',
                  'gpu': launch['expected_gpu'], 'running_disks_included': True, 'new_live_quote_checked': False},
        'financial_status': {**review['financial_status'], 'existing_conservative_pilot_encumbrance_usd': float(reserve),
            'unreserved_diagnostic_envelope_usd': float(10 - reserve), 'pilot_reserve_added_to_study_once_usd': 10,
            'new_hold_created': False, 'existing_hold_released': False},
        'excluded_observation': {'unit': 'primary with two scoring audits',
            'seconds': evaluations['run/evaluations/controls-primary/evaluation.json']['records'][0]['seconds'],
            'reason': 'Not relabeled as an ordinary zero-audit primary clip or assigned an invented additional study count.'},
        'native_sample_note': 'Eight primary records: three pretrained, two two-update trained anchors and three one-clip calibration candidates. Repeated clips/steps are technical samples, not eight independent videos.',
        'accounting_note': 'Native times include clean/main scoring; convergence includes both audits. Wrapper residuals subtract those native times. Ten three-step bundles already contain thirty calibration evaluations and wrappers. State and save spans are disjoint. Nested phase totals and diagnostic recovery comparisons are not charged again.',
        'feature_scope': 'Two eight-training-clip feature/probe phases passed; full 100-clip/held-out feature/probe costs remain unknown and are not scaled linearly from eight examples.',
        'limitations': ['Median and maximum-observed samples are short-run extrapolations, not bounds or confidence intervals.',
            'The historical two-plus-six-hour allowance and 25% headroom do not measure or bound missing work.',
            'The one-clip diagnostic step is not ten-clip research calibration; research step remains null.',
            'Break-even rate/speed conditions hold sample throughput fixed; they are not predictions for other GPUs or current offers.',
            'Full-panel hashing/journaling can grow with panel size; measured one/three-clip wrapper residuals do not establish full-panel cost.',
            'No confirmation/test pixel tensors are opened. Their retained evaluation counts are planning metadata only.'],
        'tensor_deserialization_performed': False, 'cloud_job_started': False, 'new_paid_compute_authorized': False,
        'protocol_or_YAML_changed': False, 'budget_approved': False, 'research_admitted': False}
