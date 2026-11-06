#!/usr/bin/env python3
"""One source-bound child. Authorization and pins precede scientific imports."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

overlay=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(overlay))
from models import predictor_eot8_owned_operator_v1 as op


def main():
    p=argparse.ArgumentParser();p.add_argument('--scratch',type=Path,required=True);p.add_argument('--control-sha256',required=True);p.add_argument('--index',type=int,required=True)
    args=p.parse_args();s=args.scratch
    if s.is_symlink() or overlay!=s.resolve()/'overlay' or s.parent!=Path('/var/tmp') or not s.name.startswith(op.PREFIX):raise ValueError('Exact private scratch/overlay roots required.')
    if op.old.digest(s/'control.json')!=args.control_sha256:raise ValueError('Control changed.')
    c=json.loads((s/'control.json').read_text());op.validate_control(c)
    op.validate_authorization(json.loads((s/'authorization.json').read_text()),args.control_sha256)
    for name,sha in c['remote_source_sha256'].items():
        path=overlay/op.old.safe_name(name)
        if path.is_symlink() or op.old.digest(path)!=sha:raise ValueError('Operator source changed.')
    for name,item in c['archive_manifest']['files'].items():
        path=s/'code'/op.old.safe_name(name)
        if path.is_symlink() or path.stat().st_size!=item['bytes'] or op.old.digest(path)!=item['sha256']:raise ValueError('Original source/input changed.')
    session=json.loads((s/'session.json').read_text());plan=json.loads((s/('worker-plan-'+str(args.index)+'.json')).read_text())
    if args.index not in range(4) or len(session['tasks'])!=args.index or session['complete'] or plan['index']!=args.index or plan['control_sha256']!=args.control_sha256:
        raise ValueError('Worker order/phase changed.')
    remaining=plan['deadline_unix']-time.time()
    if not 0<remaining<=480 or plan['deadline_unix']>session['deadline_unix']-120:raise ValueError('Worker/continuing clock invalid.')
    signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,remaining)
    for index,task in enumerate(session['tasks']):
        if op.check_worker(s/'results'/str(index),index)!=task:raise ValueError('Earlier complete worker changed.')
    sys.path.insert(1,str(s/'code'))
    from types import SimpleNamespace
    from models.predictor_eot8_owned_worker_v1 import run
    existing=Path.home()/'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting'
    result=run(SimpleNamespace(output=s/'results'/str(args.index),index=args.index,tiny=False,device='cuda',
        upstream=existing/'external/vjepa2',checkpoint=existing/'checkpoints/vitl.pt',packet=s/'code',
        proposal=overlay/'configs/predictor_attention_eot8_proposal_v1.json',control_sha256=args.control_sha256,
        reference=s/'results'/str(args.index-1) if args.index in (1,3) else None,
        replay_reference=s/'results'/str(args.index-2) if args.index in (2,3) else None))
    return 0 if result['passed'] else 1

if __name__=='__main__':sys.exit(main())
