#!/usr/bin/env python3
"""Separate paired PGD training diagnostic; no canonical evaluation or optimizer."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=ROOT / 'external/vjepa2')
    parser.add_argument('--manifests', type=Path, default=ROOT / 'data/manifests/ucf101-staged-20x5-v1')
    parser.add_argument('--clips', type=Path, default=ROOT / 'data/processed/matched-monitor-v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=4620)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--role', choices=('matrix', 'replay'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not args.diagnostic or not 1 <= args.max_seconds <= 4620 or args.output.exists()
            or args.worker != (args.role is not None) or not args.checkpoint.is_file()):
        parser.error('Use --diagnostic, an existing released checkpoint, fresh output and at most 4620 seconds.')
    from models.matched_diagnostic import session, worker
    if args.worker:
        worker(ROOT, args)
    else:
        result = session(ROOT, args)
        print('Matched diagnostic passed:', result['passed'])
        if not result['passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
