#!/usr/bin/env python3
"""Reprice the unchanged staged workloads from the verified shared-forward pilot."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.measured_budget import project


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(root):
    root = Path(root)
    plan_path = root / 'configs/budget_study_plan.yaml'
    evidence_path = root / 'report/checks/2026-10-22-shared-forward-cuda-workflow.json'
    plan = yaml.safe_load(plan_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    pins = {**plan['source_sha256'], **evidence['provenance']['unchanged_core_and_configuration_sha256']}
    for name, expected in pins.items():
        if sha256(root/name) != expected:
            raise ValueError(f'Pinned source changed: {name}')
    paths = (plan_path, evidence_path, root/'models/measured_budget.py', root/'scripts/estimate_measured_study.py',
             root/'models/evaluation.py', root/'models/schedule.py', root/'models/checkpointing.py')
    variants = {name: project(plan, evidence, **counts) for name, counts in plan['data']['variants'].items()}
    return {
        'kind': 'offline_measured_unit_budget_review', 'date': '2026-10-22',
        'source_sha256': {**pins, **{str(p.relative_to(root)): sha256(p) for p in paths}},
        'quote': {'usd_per_hour': evidence['compute']['quoted_hourly_usd'],
                  'basis': 'User-reported October 22 RTX 5090 quote including selected running disks; not a new availability or price check.'},
        'candidate_execution_policy': evidence['session']['execution_policy'],
        'precision_note': 'FP64 context-patch accumulation with FP32 learned parameters/tokens/tail; this differs from the historical proposal and is not adopted for research.',
        'measurement_sample_sizes': {
            'updates': {k: v['count'] for k, v in evidence['engine_update_seconds_by_condition'].items()},
            'evaluation': {k: v['count'] for k, v in evidence['native_evaluation_seconds'].items()}},
        'billing_snapshot': evidence['compute'],
        'reserve_accounting': 'The entire $10 pilot envelope already includes prior charges and the pending $3 reservation. Add it once, plus $40 noncompute reserve; do not add $2.280 again.',
        'variants': variants,
        'research_calibration_complete': False, 'protocol_or_yaml_changed': False,
        'pixels_opened': False, 'cloud_job_started': False,
        'complete_cost_estimate': False, 'budget_approved': False,
        'limitations': [
            'Both median and maximum-observed summaries extrapolate a small training-only sample; neither is a full-study timing bound or confidence interval.',
            'Primary timings pool seven 2/255 records plus one 0.5/255 and one 1/255 calibration record at identical attack strength. The full calibration span is separately timed; ten-clip research calibration has not frozen a step size.',
            'Clean native scores are already in evaluate_clip timing. Two extra scoring banks are already in convergence timing. Neither is added again.',
            'Checkpoint state and save spans are disjoint; nested workflow/scheduled/control totals are excluded to prevent double counting.',
            'Full feature/probe, panel/I/O, export/storage, setup/idle and repair costs remain null. The historical eight-hour allowance is not a measurement or a bound.',
            'A hypothetical price threshold assumes unchanged throughput and validated numerics. It does not predict another GPU performance or availability.',
            'The provisional aggregate-derived $2.280 spend is not a reconciled Pod invoice. The full pilot reserve is retained.',
            'Only existing class-count variants are compared. No manifests, attacks, seeds, epochs, precision settings or study limits are changed.',
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output; existing evidence is never overwritten.')
    report = build_report(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    for name, value in report['variants'].items():
        median = value['scenarios']['median']['total_with_historical_untimed_allowance_usd']
        maximum = value['scenarios']['maximum']['total_with_historical_untimed_allowance_usd']
        print(f'{name}: ${median:.2f} median / ${maximum:.2f} maximum-observed unit extrapolation. NOT ADMITTED.')


if __name__ == '__main__':
    main()
