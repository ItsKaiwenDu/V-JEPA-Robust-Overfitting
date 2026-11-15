#!/usr/bin/env python3
"""Bound child: scope, source, continuing clock and live GPU gate before imports."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

overlay=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(overlay))
from models import predictor_pgd_target_operator_v1 as op


def main():
    p=argparse.ArgumentParser();p.add_argument('--scratch',type=Path,required=True);p.add_argument('--control-sha256',required=True);p.add_argument('--index',type=int,required=True)
    args=p.parse_args();s=args.scratch
    if s.is_symlink() or overlay!=s.resolve()/'overlay' or s.parent!=Path('/var/tmp') or not s.name.startswith(op.PREFIX):raise ValueError('Exact new scratch/overlay root required.')
    if op.old.digest(s/'control.json')!=args.control_sha256:raise ValueError('Control changed.')
    c=json.loads((s/'control.json').read_bytes());op.validate_control(c)
    op.validate_authorization(json.loads((s/'authorization.json').read_bytes()),args.control_sha256,c)
    for name,sha in c['remote_source_sha256'].items():
        path=overlay/op.old.safe_name(name)
        if path.is_symlink() or op.old.digest(path)!=sha:raise ValueError('Overlay changed.')
    for name,d in c['archive_manifest']['files'].items():
        path=s/'code'/op.old.safe_name(name)
        if path.is_symlink() or path.stat().st_size!=d['bytes'] or op.old.digest(path)!=d['sha256']:raise ValueError('Original packet changed.')
    session=json.loads((s/'session.json').read_bytes());plan=json.loads((s/'worker-plan-0.json').read_bytes())
    if args.index!=0 or session['tasks']!=[] or session['complete'] or plan['index']!=0 or plan['control_sha256']!=args.control_sha256:
        raise ValueError('One previously unstarted iterated worker only.')
    remaining=plan['deadline_unix']-time.time()
    if not 0<remaining<=1920 or plan['deadline_unix']>session['deadline_unix']-120:raise ValueError('Worker/continuing clock invalid.')
    signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,remaining)
    op.base.gpu_gate(c)
    sys.path.insert(1,str(s/'code'))
    from types import SimpleNamespace
    from models.predictor_pgd_target_worker_v1 import run
    existing=Path.home()/'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting'
    result=run(SimpleNamespace(output=s/'results/0',tiny=False,device='cuda',
        upstream=existing/'external/vjepa2',checkpoint=existing/'checkpoints/vitl.pt',packet=s/'code',
        proposal=overlay/op.CONTRACT,control_sha256=args.control_sha256))
    return 0 if result['passed'] else 1

if __name__=='__main__':sys.exit(main())
