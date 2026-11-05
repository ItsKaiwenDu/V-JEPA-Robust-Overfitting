#!/usr/bin/env python3
"""Cold packaged worker, or explicit local-only tiny math-pair rehearsal."""
import argparse
from pathlib import Path
import sys

OVERLAY=Path(__file__).resolve().parents[1];sys.path.insert(0,str(OVERLAY))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--control-sha256');p.add_argument('--task-index',type=int,choices=(0,1))
    p.add_argument('--worker-plan-sha256');p.add_argument('--local-tiny-rehearsal',action='store_true');a=p.parse_args()
    from models import original_attention_pair_runtime_v1 as runtime
    if a.task_index is None:
        if not a.local_tiny_rehearsal or a.worker_plan_sha256:p.error('Parent mode is only an explicit local tiny CPU rehearsal.')
        status=runtime.local_pair(a,OVERLAY)
    else:
        if not a.worker_plan_sha256:p.error('A bound parent-issued worker plan is required.')
        status=runtime.dispatch(a,OVERLAY)
    raise SystemExit(status)


if __name__=='__main__':main()
