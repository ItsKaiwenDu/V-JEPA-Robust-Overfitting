#!/usr/bin/env python3
"""Separate matched-FP64-forward diagnostic; does not start research training."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic', action='store_true')
    p.add_argument('--tiny', action='store_true')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--upstream', type=Path, default=ROOT / 'external/vjepa2')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--clips', type=Path)
    p.add_argument('--manifest', type=Path, default=ROOT / 'data/manifests/ucf101-split1-v1/train.csv')
    p.add_argument('--observed', type=Path, default=ROOT / 'OBSERVED.json')
    p.add_argument('--reference-run', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--max-seconds', type=int, default=840)
    p.add_argument('--worker-seconds', type=int, default=180)
    p.add_argument('--initial-fixture', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--initial-fixture-sha256', help=argparse.SUPPRESS)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--backward', choices=('cudnn', 'native'), help=argparse.SUPPRESS)
    p.add_argument('--result', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if (not args.diagnostic or args.tiny != (args.device == 'cpu')
            or not 1 <= args.max_seconds <= 840 or not 1 <= args.worker_seconds <= 180):
        p.error('Use explicit --diagnostic; CUDA or --tiny --device cpu; at most 180s/worker and 840s total.')
    if not args.tiny and any(v is None or not v.exists() for v in (args.checkpoint, args.clips, args.manifest, args.observed, args.reference_run)):
        p.error('Reuse existing checkpoint, training data, original observed spec and failed reference run.')
    if (not args.tiny or not args.worker) and (args.initial_fixture is not None or args.initial_fixture_sha256 is not None):
        p.error('Initial fixture pins belong only to generated CPU workers.')
    if args.tiny and args.worker and (args.initial_fixture is None or args.initial_fixture_sha256 is None):
        p.error('Generated CPU worker requires its parent initial fixture and SHA-256.')
    if args.worker:
        if args.backward is None or args.result is None or args.result.exists():
            p.error('Worker needs explicit backward and fresh result path.')
    elif args.output is None or args.output.exists():
        p.error('Use a fresh output directory.')
    from models.precision_diagnostic import run, worker
    if args.worker:
        worker(args, ROOT)
    else:
        result = run(args, ROOT)
        print('Shared-forward diagnostic finished; passed:', result['passed'])
        print('Separate precision/reference only. Historical FP32 failures and research gates remain.')


if __name__ == '__main__':
    main()
