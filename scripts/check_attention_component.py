#!/usr/bin/env python3
"""Standalone synthetic attention components; no model or checkpoint loading."""
import argparse
from pathlib import Path
import sys
import time

ENTERED=time.time()
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic',action='store_true');p.add_argument('--tiny',action='store_true')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--role',choices=('predictor','encoder','all'),required=True)
    p.add_argument('--handoff-sha256',required=True);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--plan',type=Path);p.add_argument('--plan-sha256')
    p.add_argument('--output',type=Path);p.add_argument('--max-seconds',type=int);p.add_argument('--worker-seconds',type=int)
    p.add_argument('--launcher-started',type=float)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--case',choices=('predictor','encoder-0','encoder-1'),help=argparse.SUPPRESS)
    p.add_argument('--arm',choices=('efficient','math'),help=argparse.SUPPRESS)
    p.add_argument('--replay',type=int,choices=(0,1),help=argparse.SUPPRESS)
    p.add_argument('--worker-dir',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--authorization-sha256',help=argparse.SUPPRESS)
    args=p.parse_args()
    if not args.diagnostic or args.tiny!=(args.device=='cpu') or args.role=='all' and not args.tiny:
        p.error('Use --diagnostic and an explicit CPU fixture or single-role CUDA domain.')
    args.max_seconds=(300 if args.tiny else 1200) if args.max_seconds is None else args.max_seconds
    args.worker_seconds=(30 if args.tiny else 120) if args.worker_seconds is None else args.worker_seconds
    args.launcher_started=ENTERED if args.launcher_started is None else args.launcher_started
    if not args.upstream.is_dir():p.error('Existing pinned upstream source is required.')
    if not args.tiny and (args.plan is None or not args.plan.is_file() or args.plan_sha256 is None):
        p.error('CUDA requires a bound attended host/role plan.')
    if args.worker:
        if (args.case is None or args.arm is None or args.replay is None or args.worker_dir is None
                or not (args.worker_dir/'authorization.json').is_file() or (args.worker_dir/'worker.json').exists()
                or args.authorization_sha256 is None):p.error('Fresh parent authorization required.')
        if args.case not in (('predictor','encoder-0','encoder-1') if args.role=='all' else ('predictor',) if args.role=='predictor' else ('encoder-0','encoder-1')):
            p.error('Worker case differs from role.')
    elif args.output is None or args.output.exists():p.error('Use a fresh output path; no retries.')
    from models.attention_diagnostic import run,worker
    if args.worker:worker(args,ROOT)
    else:
        result=run(args,ROOT);print('Attention component diagnostic complete:',result['passed'],flush=True)
        print('Synthetic components only; full-model/backend adoption remains unqualified.',flush=True)


if __name__=='__main__':main()
