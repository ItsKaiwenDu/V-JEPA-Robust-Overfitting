#!/usr/bin/env python3
"""Compare original Conv3d execution options under strict determinism."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--cudnn-tf32', choices=('on', 'off'), default='on',
                        help='Explicit diagnostic precision control; does not change the research runner.')
    parser.add_argument('--max-seconds', type=float, default=480)
    parser.add_argument('--worker-seconds', type=float, default=45)
    parser.add_argument('--max-memory-gib', type=float, default=4)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--no-profiler', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--variant', help=argparse.SUPPRESS)
    parser.add_argument('--mode', help=argparse.SUPPRESS)
    parser.add_argument('--result', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (1 <= args.max_seconds <= 600 and .05 <= args.worker_seconds <= 45
            and .25 <= args.max_memory_gib <= 4 and 0 <= args.warmup <= 1 and 2 <= args.repeats <= 3):
        parser.error('Use <=600 seconds overall, <=45 per worker, <=4 GiB allocator, 0-1 warmups, and 2-3 repetitions.')
    if (args.device == 'cpu') != args.tiny:
        parser.error('CPU is available only for --tiny fixtures; CUDA uses representative shapes.')
    if args.worker:
        if not args.result or args.result.exists():
            parser.error('Worker needs a fresh --result path.')
    elif args.output is None or args.output.exists():
        parser.error('Use a fresh --output directory.')
    from models.cuda_convolution import MODES, VARIANTS, run_comparison, worker
    if args.worker and (args.variant not in VARIANTS or args.mode not in MODES):
        parser.error('Unknown worker variant or gradient mode.')
    if args.variants and (len(set(args.variants)) != len(args.variants) or 'contiguous' not in args.variants
                          or any(v not in VARIANTS for v in args.variants)):
        parser.error('Use unique known variants including contiguous.')
    if args.worker:
        worker(args, ROOT)
        return 0
    result = run_comparison(args, ROOT, started)
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
