#!/usr/bin/env python3
"""Bounded CUDA component timing and replay; no weights or video data needed."""

import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--tiny',action='store_true',help='Small CPU-fixture shapes only; not hardware evidence.')
    parser.add_argument('--max-seconds',type=float,default=480.)
    parser.add_argument('--worker-seconds',type=float,default=20.)
    parser.add_argument('--max-memory-gib',type=float,default=4.)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--parameter-gradients',action='store_true',help='Include isolated module weight gradients; no optimizer.')
    parser.add_argument('--no-profiler',action='store_true',help='Keep timed gradients/replay without the optional operator table.')
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--case',help=argparse.SUPPRESS)
    parser.add_argument('--policy',choices=('default','strict'),help=argparse.SUPPRESS)
    parser.add_argument('--result',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not (1<=args.max_seconds<=600 and .05<=args.worker_seconds<=45 and
            1<=args.repeats<=3 and 0<=args.warmup<=1 and .25<=args.max_memory_gib<=4):
        parser.error('Use <=600s overall, <=45s per worker, 1-3 repeats, 0-1 warmups, and <=4GiB allocator limit.')
    if (args.device=='cpu')!=args.tiny:
        parser.error('CPU is enabled only for --tiny fixtures; CUDA uses representative shapes.')
    if args.worker:
        if not args.case or not args.policy or not args.result or args.result.exists():
            parser.error('Workers need a case, policy, and fresh result path.')
    elif args.output is None or args.output.exists():
        parser.error('Use a fresh --output directory.')
    from models.cuda_profile import CASES,run_profile,worker
    if (args.cases and (len(set(args.cases))!=len(args.cases) or any(c not in CASES for c in args.cases))) or (args.worker and args.case not in CASES):
        parser.error('Unknown component case.')
    if args.worker:
        worker(args)
    else:
        result=run_profile(args,ROOT,started)
        return 0 if result['complete'] else 1


if __name__=='__main__':
    raise SystemExit(main())
