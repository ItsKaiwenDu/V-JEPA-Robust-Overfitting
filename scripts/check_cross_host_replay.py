#!/usr/bin/env python3
"""One separately identified candidate-host replay; never dispatch panel cases."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'worker', 'review'))
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--worker-index', type=int, choices=(0, 1))
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--expected-archive-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    packet = Path(__file__).resolve().parents[1]
    if args.plan.resolve() != packet / 'shadow-plan.json':
        parser.error('Use the plan next to this separately identified shadow packet.')
    plan = json.loads(args.plan.read_text())
    checksum = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if checksum != args.expected_plan_sha256:
        parser.error('Shadow plan changed from its external pin.')
    source = packet / 'models/cross_host_replay.py'
    with source.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != plan['shadow_sources_sha256']['models/cross_host_replay.py']:
            parser.error('Shadow operator source changed.')
    spec = importlib.util.spec_from_file_location('cross_host_shadow', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_plan(plan, args.expected_plan_sha256)
    review_args = (args.archive, args.manifest, args.expected_archive_sha256, args.output)
    if args.mode != 'review' and any(v is not None for v in review_args):
        parser.error('Export review arguments cannot launch a worker.')
    if args.mode == 'worker':
        if args.worker_index is None: parser.error('Fresh worker index is required.')
        module.worker(args.package, packet, plan, args.expected_plan_sha256, args.worker_index)
    elif args.mode == 'review':
        if args.worker_index is not None or any(v is None for v in review_args) or args.output.exists():
            parser.error('Review needs an externally pinned archive and fresh output, with no worker override.')
        result = module.review_export(args.package, packet, plan, args.expected_plan_sha256,
            args.archive, json.loads(args.manifest.read_text()), args.expected_archive_sha256, args.output)
        print('Independent shadow export verification PASS; panel admission remains false.')
    else:
        if args.worker_index is not None: parser.error('A supervisor never accepts a worker override.')
        result = module.run(args.package, packet, plan, args.expected_plan_sha256)
        if not result['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
