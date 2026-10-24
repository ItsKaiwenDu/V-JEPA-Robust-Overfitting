#!/usr/bin/env python3
"""Bounded full V-JEPA input-gradient/attack diagnostic; never starts training."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--upstream', type=Path, default=ROOT / 'external/vjepa2')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--clips', type=Path)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/manifests/ucf101-split1-v1/train.csv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--tiny', action='store_true', help='Generated CPU fixture, never pretrained/CUDA evidence.')
    parser.add_argument('--max-seconds', type=int, default=840)
    parser.add_argument('--worker-seconds', type=int, default=180)
    parser.add_argument('--max-memory-gib', type=float, default=7)
    parser.add_argument('--initial-fixture', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--initial-fixture-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--execution-policy', choices=('cudnn-fp32', 'native-fp32'), help=argparse.SUPPRESS)
    parser.add_argument('--result', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.diagnostic or not 1 <= args.worker_seconds <= 180 or not 1 <= args.max_seconds <= 840:
        parser.error('Use --diagnostic with at most 180 seconds/worker and 840 seconds overall.')
    if not 0 < args.max_memory_gib <= 7 or (args.tiny != (args.device == 'cpu')):
        parser.error('Use CUDA for full-model checks or --tiny --device cpu; allocator cap is at most 7 GiB.')
    if not args.tiny and any(v is None or not v.exists() for v in (args.checkpoint, args.clips, args.manifest)):
        parser.error('The full-model diagnostic requires the existing checkpoint, clips and training manifest.')
    if (not args.tiny or not args.worker) and (args.initial_fixture is not None or args.initial_fixture_sha256 is not None):
        parser.error('Initial fixture pins belong only to generated CPU workers.')
    if args.tiny and args.worker and (args.initial_fixture is None or args.initial_fixture_sha256 is None):
        parser.error('Generated CPU worker requires its parent initial fixture and SHA-256.')
    if args.worker:
        if args.execution_policy is None or args.result is None or args.result.exists() or args.result.with_suffix('.pt').exists():
            parser.error('Worker requires explicit policy and fresh result/tensor paths.')
    elif args.output is None or args.output.exists():
        parser.error('Use a fresh output directory.')
    from models.native_diagnostic import run, worker
    if args.worker:
        worker(args, ROOT)
    else:
        session = run(args, ROOT)
        print('Diagnostic finished; passed:', session['passed'])


if __name__ == '__main__':
    main()
