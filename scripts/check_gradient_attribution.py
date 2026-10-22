#!/usr/bin/env python3
"""Bounded diagnostic attribution of the rejected native patch input gradient."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--upstream', type=Path, default=ROOT / 'external/vjepa2')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--clips', type=Path)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/manifests/ucf101-split1-v1/train.csv')
    parser.add_argument('--observed', type=Path, default=ROOT / 'OBSERVED.json')
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--result', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.diagnostic or args.tiny != (args.device == 'cpu') or not 1 <= args.max_seconds <= 300:
        parser.error('Use --diagnostic, at most 300 seconds, and CUDA or an explicit --tiny --device cpu fixture.')
    if not args.tiny and any(p is None or not p.exists() for p in
                            (args.checkpoint, args.clips, args.manifest, args.observed, args.reference_run)):
        parser.error('Reuse existing checkpoint, training clips/manifest, OBSERVED.json and original failed run.')
    if args.worker:
        if args.result is None or args.result.exists():
            parser.error('Worker needs a fresh --result path.')
    elif args.output is None or args.output.exists():
        parser.error('Use a fresh --output directory.')
    from models.gradient_attribution import run, worker
    if args.worker:
        worker(args, ROOT)
    else:
        session = run(args, ROOT)
        print('Derivative control complete:', session['complete'], '; control checks:', session['control_checks_passed'])
        print('No backend adopted; previous numerical rejection and research gates remain unchanged.')


if __name__ == '__main__':
    main()
