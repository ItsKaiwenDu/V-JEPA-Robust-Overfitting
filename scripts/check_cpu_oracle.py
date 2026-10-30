#!/usr/bin/env python3
"""CPU-only reference forensics. No CUDA initialization, SDPA or model execution."""
import argparse
import os
from pathlib import Path
import sys
import time

ENTERED=time.time()
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic',action='store_true');p.add_argument('--local',action='store_true',help='Explicit local CPU rehearsal; never a university-host qualification.')
    p.add_argument('--handoff-sha256',required=True);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--plan-sha256')
    p.add_argument('--launcher-started',type=float,default=ENTERED)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--threads',type=int,choices=(1,4),help=argparse.SUPPRESS)
    p.add_argument('--replay',type=int,choices=(0,1),help=argparse.SUPPRESS)
    p.add_argument('--worker-dir',type=Path,help=argparse.SUPPRESS);p.add_argument('--authorization-sha256',help=argparse.SUPPRESS)
    a=p.parse_args()
    if not a.diagnostic or not a.upstream.is_dir():p.error('Explicit diagnostic and existing pinned upstream source are required.')
    if not a.local and (a.plan is None or not a.plan.is_file() or a.plan_sha256 is None):p.error('Require a new attended CPU plan.')
    if a.worker:
        if (a.threads is None or a.replay is None or a.worker_dir is None or a.authorization_sha256 is None
                or not (a.worker_dir/'authorization.json').is_file() or (a.worker_dir/'worker.json').exists()):p.error('Fresh parent CPU authorization required.')
    elif a.output is None or a.output.exists():p.error('Fresh output required; no repeat attempt.')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from models.cpu_oracle_probe import run,worker
    result=worker(a,ROOT) if a.worker else run(a,ROOT)
    print('CPU capture complete:',result['capture_complete'],'numerical checks:',result['numerical_checks_passed'],flush=True)
    print('CPU forensics only; old failures, scientific gates and all paid/GPU limits remain unchanged.',flush=True)
    raise SystemExit(0 if result['capture_complete'] and result['numerical_checks_passed'] else 1)


if __name__=='__main__':main()
