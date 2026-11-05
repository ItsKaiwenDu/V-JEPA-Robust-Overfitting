#!/usr/bin/env python3
"""Cold bound replay-1 phase, or explicit small local CPU analogue."""
import argparse
from pathlib import Path
import sys

OVERLAY=Path(__file__).resolve().parents[1];sys.path.insert(0,str(OVERLAY))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','scratch','upstream'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--control-sha256');p.add_argument('--task-index',type=int,choices=(0,1))
    p.add_argument('--phase',choices=('local','worker','cohort'),default='local')
    p.add_argument('--worker-plan-sha256');p.add_argument('--local-tiny-analogue',action='store_true')
    for name in ('reference-archive','reference-spec','reference-manifest'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    from models import attention_replay1_runtime_v1 as runtime
    if a.phase=='local':
        if (not a.local_tiny_analogue or a.worker_plan_sha256 or a.control_sha256 or a.task_index is not None
                or any(getattr(a,n) is None for n in ('reference_archive','reference_spec','reference_manifest'))):
            p.error('Parent mode requires a local tiny analogue and explicit reference packet; no live grant.')
        status=runtime.local_analogue(a,OVERLAY)
    else:
        if not a.worker_plan_sha256 or (a.phase=='worker')!=(a.task_index is not None):
            p.error('A bound phase plan and the correct worker index are required.')
        status=runtime.dispatch(a,OVERLAY)
    raise SystemExit(status)


if __name__=='__main__':main()
