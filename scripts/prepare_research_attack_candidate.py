#!/usr/bin/env python3
"""Prepare an offline research attack proposal; no model, data pixels or cloud access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PINS = {
    'report/checks/2026-10-28-complete-panel-analysis.json': '420c414bcbb7eb6392ea83f6cd1bbeb3ea0c3569a738db64d380f3a4a2a2d130',
    'report/checks/2026-10-23-offline-a40-review.json': '9e3e1a5928bb33ccb39a1bad2d5ff64ce0cadc3ec48796c36f5108ceec5b2794',
    'report/checks/2026-10-23-a40-controls-completion.json': '7d29a35a6f4d3bcda67f87a27642ebbe6cc2e6c2d29ac31b348e94441f391897',
    'report/checks/2026-10-22-shared-forward-cuda-workflow.json': 'b8b7b9ad70b1d74abf8deecf4d1385f9ce1c5c23d9557afd016214b9bff4f027',
    'configs/study.yaml': '4a16565ead0e3929a0a707fd71cfff95c0c99da93bef7510a3b0f385c4efe3b1',
    'configs/budget_study_plan.yaml': '70de4b44a71efe8e7ee45094f525992c9ea8f8ef45ec0de999ea6047a5f888fa',
    'models/evaluation.py': '1431951d158b57b615208006edb5365a6bad910b2a14e0aa782a1c6f82034089',
    'models/calibration.py': 'bd3ec55ca45f9788f08fb1f5342454adf86aedb20db3ba6169dfe02ecf6e92f6',
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(root):
    from models.research_attack_candidate import canonical, conditional_budget, proposal, validate_proposal
    root = Path(root).resolve()
    for name, expected in PINS.items():
        path = root / name
        if path.is_symlink() or sha(path) != expected:
            raise ValueError('Reviewed proposal input changed: ' + name)
    analysis = json.loads((root / 'report/checks/2026-10-28-complete-panel-analysis.json').read_text())
    if (not analysis['complete'] or not analysis['independent_arithmetic_verified']
            or analysis['independently_reviewed_and_registered_cases'] != 20 or analysis['distinct_clips'] != 10
            or analysis['research_admitted'] or analysis['automatic_primary_choice']):
        raise ValueError('Use the complete independently verified diagnostic panel without automatic selection.')
    plan_path = root / 'configs/matched_panel_diagnostic_v1.json'
    plan = json.loads(plan_path.read_text())
    if canonical(plan) != analysis['plan_sha256']:
        raise ValueError('Original panel plan changed.')
    monitor_path = root / 'data/manifests/ucf101-staged-20x5-v1/train_monitor.csv'
    rows = list(csv.DictReader(monitor_path.open()))[:10]
    clips = plan['ordered_clips']
    if len(rows) != 10 or any(any(row[k] != clip[k] for k in ('video_id', 'source_group_id', 'class_id', 'relative_path')) for row, clip in zip(rows, clips)):
        raise ValueError('Preserve the original declared ten training-monitor rows.')
    candidate = proposal(clips, plan['cases'])
    validate_proposal(candidate, proposal(clips, plan['cases']))
    old_budget = json.loads((root / 'report/checks/2026-10-23-offline-a40-review.json').read_text())
    economics = conditional_budget(old_budget)
    recovery = json.loads((root / 'report/checks/2026-10-22-shared-forward-cuda-workflow.json').read_text())
    a40 = json.loads((root / 'report/checks/2026-10-23-a40-controls-completion.json').read_text())
    if (not recovery['passed'] or recovery['session']['device']['gpu'] != 'NVIDIA GeForce RTX 5090'
            or recovery['session']['execution_policy'] != 'shared-fp64-native'
            or any(not recovery['comparisons'][c]['exact_match'] or recovery['comparisons'][c]['expected_step'] != 2
                   for c in ('clean-finetune', 'adversarial-finetune'))
            or a40['exact_a40_optimizer_recovery_checked'] is not False):
        raise ValueError('Keep recovery evidence hardware-specific and bounded.')
    provenance = dict(PINS)
    for name in ('configs/matched_panel_diagnostic_v1.json', 'data/manifests/ucf101-staged-20x5-v1/train_monitor.csv',
                 'models/research_attack_candidate.py', 'scripts/prepare_research_attack_candidate.py', 'tests/test_research_attack_candidate.py'):
        provenance[name] = sha(root / name)
    report = dict(kind='research-eot8-candidate-offline-review-v1', date_Pacific='2026-10-28',
        candidate=candidate, candidate_sha256=canonical(candidate), conditional_affordability=economics,
        source_sha256=provenance,
        recovery_qualification=dict(RTX5090=dict(status='passed-bounded-native-two-update-recovery',
            conditions=['clean-finetune', 'adversarial-finetune'], exact_comparisons=recovery['comparisons'],
            source_report='report/checks/2026-10-22-shared-forward-cuda-workflow.json', full_trajectory_qualified=False),
            A40=dict(status='bounded-updates-and-input-checks-passed-exact-resume-not-tested', exact_optimizer_recovery_qualified=False,
                     source_report='report/checks/2026-10-23-a40-controls-completion.json'),
            university_RTX3070=dict(status='twenty-input-attack-cases-qualified-prior-eight-GB-optimizer-OOM-unresolved',
                input_gradient_panel_qualified=True, full_optimizer_training_qualified=False,
                timing_credited_to_A40_or_new_GPU=False)),
        code_implementation_required=['opt-in-versioned-EOT8-evaluation-settings-and-matched-primary-control-seeds',
            'calibration-and-all-artifact-consumers-bind-proposed-strength-source-runtime-and-selected-step',
            'reject-all-old-four-draw-calibrations-and-mismatched-caches-anchors-receipts',
            'tested-bounded-hardware-benchmark-before-quote-deploy-and-full-calibration'],
        original_study_YAML_changed=False, existing_evaluation_or_calibration_code_changed=False,
        new_model_or_GPU_execution=False, new_pixel_or_checkpoint_read=False, held_out_pixels_opened=False,
        current_quote_checked=False, new_paid_compute_authorized=False, budget_approved=False,
        research_policy_adopted=False, research_admitted=False,
        next_step='Implement and test the opt-in candidate evaluation/artifact path and prepare a bounded whole-workload hardware benchmark. Only after actual hardware/economics qualification, run the declared thirty calibration evaluations on the chosen research runtime. No launch is authorized by this proposal.')
    return candidate, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.candidate.exists() or args.report.exists():
        parser.error('Keep previous proposals and reports; choose new paths.')
    candidate, report = build(ROOT)
    for path, value in ((args.candidate, candidate), (args.report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(candidate_sha256=report['candidate_sha256'], proposed_primary=candidate['primary_candidate'],
        calibration_evaluations=30, research_policy_adopted=False, budget_approved=False,
        current_quote_checked=False), sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
