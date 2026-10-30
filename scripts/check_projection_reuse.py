#!/usr/bin/env python3
"""Explicit, separately identified CPU rehearsal or attended university diagnostic."""
import argparse
from pathlib import Path
import sys
import time

ENTERED=time.time()
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--tiny',action='store_true')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--handoff-sha256',required=True)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--plan',type=Path)
    p.add_argument('--plan-sha256')
    p.add_argument('--output',type=Path)
    p.add_argument('--max-seconds',type=int)
    p.add_argument('--worker-seconds',type=int)
    p.add_argument('--launcher-started',type=float)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--mode',choices=('recompute','reuse'),help=argparse.SUPPRESS)
    p.add_argument('--replay',type=int,choices=(0,1),help=argparse.SUPPRESS)
    p.add_argument('--worker-dir',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--authorization-sha256',help=argparse.SUPPRESS)
    p.add_argument('--initial-fixture',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--initial-fixture-sha256',help=argparse.SUPPRESS)
    args=p.parse_args()
    if not args.diagnostic or args.tiny!=(args.device=='cpu'):
        p.error('Use --diagnostic and either --tiny --device cpu or the separate CUDA launch.')
    args.max_seconds=(300 if args.tiny else 4800) if args.max_seconds is None else args.max_seconds
    args.worker_seconds=(60 if args.tiny else 1080) if args.worker_seconds is None else args.worker_seconds
    args.launcher_started=ENTERED if args.launcher_started is None else args.launcher_started
    if not args.upstream.is_dir():p.error('Use the existing pinned upstream checkout.')
    if not args.tiny and (args.plan is None or not args.plan.is_file() or args.plan_sha256 is None
            or args.checkpoint is None or not args.checkpoint.is_file()):
        p.error('CUDA requires an existing checkpoint and a separately reviewed launch plan/SHA.')
    if args.worker:
        if (args.mode is None or args.replay is None or args.worker_dir is None
                or not (args.worker_dir/'authorization.json').is_file()
                or (args.worker_dir/'worker.json').exists() or args.authorization_sha256 is None):
            p.error('Private worker requires its fresh parent authorization.')
        if args.tiny and (args.initial_fixture is None or args.initial_fixture_sha256 is None):
            p.error('CPU workers require the parent-pinned generated fixture.')
    elif args.output is None or args.output.exists():
        p.error('Use a fresh output directory; this diagnostic has no retries.')
    from models.reuse_diagnostic import run,worker
    if args.worker:worker(args,ROOT)
    else:
        result=run(args,ROOT)
        print('Projection-reuse diagnostic complete:',result['passed'],flush=True)
        print('Separate diagnostic only; no optimizer, study integration or research admission.',flush=True)


if __name__=='__main__':main()
