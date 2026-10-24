#!/usr/bin/env python3
"""Audit the completed A40 evidence and write an offline feasibility review."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PINS = {
    'report/checks/2026-10-22-measured-budget-review.json': '427c0e9f4e05e1c3eb0f441d74f43630c6e627fadac1437843410aad471f7128',
    'runs/pretrained/diagnostic/a40-controls-session-20261023/final-verification.json': '7a3182a4a32fbe9d5aac7c2aa161330f5cd4e0fb799f322f8177b86c2d4a7d5e',
    'runs/pretrained/diagnostic/a40-controls-session-20261023/downloaded/evidence.tar.gz': '0d7e6b1c81d266b2470da0b01c7fbe6157bced5455311f3ae70f3f6160305c38',
}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_pins(root, pins):
    for name, expected in pins.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError('Pinned review input changed: ' + name)


def build_report(root):
    from models.feasibility_review import attack_pair_review, cost_review
    from models.hardware_controls import check_evaluation, combine, package_gate, verify_files
    from models.pilot_bundle import extract_verified
    from models.study_artifacts import read_record
    root = Path(root).resolve()
    verify_pins(root, PINS)
    raw = root / 'runs/pretrained/diagnostic/a40-controls-session-20261023'
    final = json.loads((raw / 'final-verification.json').read_text())
    if (not final['passed'] or not final['session_complete'] or final['worker_exit_status'] != 0
            or final['research_admitted'] or not final['old_session_timeout_preserved']):
        raise ValueError('Use the accepted controls diagnostic and preserve its limits.')
    extracted = raw / 'extracted'
    manifest = json.loads((raw / 'downloaded/evidence.tar.gz.manifest.json').read_text())
    if manifest['sha256'] != final['archive_sha256']:
        raise ValueError('Archive identity differs from the verified evidence.')
    # Revalidate the member manifest against the pinned tar itself, then check
    # the retained extraction. An edited sidecar cannot redefine the evidence.
    with tempfile.TemporaryDirectory(prefix='vjepa-offline-review.') as temporary:
        extract_verified(raw / 'downloaded/evidence.tar.gz', manifest, Path(temporary) / 'evidence')
    members = verify_files(extracted, manifest['files'])
    package_gate(extracted / 'package')
    frozen = json.loads((extracted / 'package/prior/extracted/handoff-manifest.json').read_text())
    science_pins = {n: r['sha256'] for n, r in frozen['archives']['source']['files'].items() if not n.endswith('.md')}
    verify_pins(root, science_pins)
    # The old combiner itself is an immutable operational input to this review.
    verify_pins(root, {'models/hardware_controls.py': sha(extracted / 'package/operations/models/hardware_controls.py')})
    reference = json.loads((root / 'report/checks/2026-10-22-measured-budget-review.json').read_text())
    past = extracted / 'package/prior/extracted/pilot'
    current = extracted / 'run/pilot'
    combined = read_record(current / 'combined-economic-screen.json')
    quote = final['session']['quoted_hourly_usd']
    rebuilt = combine(past, current, reference, quote)
    for key in rebuilt:
        if key == 'candidate_economic_screen':
            for stat, row in rebuilt[key].items():
                for field, expected in row.items():
                    actual = combined[key][stat][field]
                    if field in ('timed_unit_projection_hours', 'with_historical_allowances_and_reserves_usd'):
                        if not math.isclose(expected, actual, rel_tol=1e-14):
                            raise ValueError('Derived historical arithmetic differs.')
                    elif expected != actual:
                        raise ValueError('Historical economic flag differs.')
        elif rebuilt[key] != combined[key]:
            raise ValueError('Combined evidence differs: ' + key)
    costs = cost_review(reference['variants']['focused_20x5']['workload'], combined['observations'], quote)
    for stat, row in costs['scenarios'].items():
        if not math.isclose(row['total_usd'], combined['candidate_economic_screen'][stat]['with_historical_allowances_and_reserves_usd'], rel_tol=1e-14):
            raise ValueError('Cost attribution did not reproduce the recorded screen.')
    calibration = read_record(past / 'calibration/calibration.json')
    primary = read_record(past / 'calibration/step-2p0/evaluation.json')
    convergence = read_record(current / 'controls/convergence/evaluation.json')
    for payload, method in ((primary, 'primary'), (convergence, 'convergence')):
        check_evaluation(payload, calibration['common_contract'], method, 2., pretrained=True)
    attack = attack_pair_review(primary, convergence)
    return dict(kind='offline_a40_price_performance_protocol_review', date='2026-10-23',
        source_sha256={**PINS, **science_pins, **{n: sha(root / n) for n in
            ('models/feasibility_review.py', 'scripts/review_a40_feasibility.py', 'models/hardware_controls.py')}},
        archive_members_verified=members, existing_non_markdown_source_pins_verified=len(science_pins),
        economics=costs, attack_comparison=attack,
        quote_is_historical=True, current_hardware_availability_checked=False,
        new_cloud_job_started=False, new_cuda_measurement=False, pixels_opened=False,
        study_yaml_or_protocol_changed=False, scientific_thresholds_changed=False,
        complete_cost_estimate=False, budget_approved=False, research_admitted=False,
        next_step='Prepare the separate matched training-only convergence diagnostic before any further paid benchmark.',
        limitations=[
            'All measurements are bounded diagnostics, not full-study throughput bounds.',
            'Hypothetical prices assume the measured A40 units; another GPU needs new measurements.',
            'Uniform timing multipliers are affordability targets, not performance predictions.',
            'The existing50reserve includes the10diagnostic envelope once; prior charges are not added again.',
            'One training clip cannot identify a step effect or establish study-wide attack convergence.',
            'No primary attack, data split, seed, epoch, precision choice or YAML was changed.',
            'The matched diagnostic is specified here but has not been implemented or executed.',
            'Final billing and the0.0792wallet/Details difference remain unresolved separately.',
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output; do not overwrite prior evidence.')
    report = build_report(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print('Offline review complete; no cloud access or study admission.')
    for name, value in report['economics']['scenarios'].items():
        print(f"{name}: ${value['total_usd']:.2f}; evaluation {100 * value['native_evaluation_fraction']:.2f}%; "
              f"uniform runtime reduction {100 * value['uniform_native_time_reduction_needed']:.2f}%.")


if __name__ == '__main__':
    main()
