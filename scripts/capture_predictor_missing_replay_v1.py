#!/usr/bin/env python3
"""One bound candidate replay child; exact scope/pins/clock before model imports."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

overlay=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(overlay))
from models import predictor_missing_replay_operator_v1 as op


def main():
    p=argparse.ArgumentParser();p.add_argument('--scratch',type=Path,required=True);p.add_argument('--control-sha256',required=True);p.add_argument('--index',type=int,required=True)
    args=p.parse_args();s=args.scratch
    if s.is_symlink() or overlay!=s.resolve()/'overlay' or s.parent!=Path('/var/tmp') or not s.name.startswith(op.PREFIX):raise ValueError('Exact private scratch/overlay roots required.')
    if op.old.digest(s/'control.json')!=args.control_sha256:raise ValueError('Control changed.')
    c=json.loads((s/'control.json').read_text());op.validate_control(c)
    op.validate_authorization(json.loads((s/'authorization.json').read_text()),args.control_sha256,c)
    for name,sha in c['remote_source_sha256'].items():
        path=overlay/op.old.safe_name(name)
        if path.is_symlink() or op.old.digest(path)!=sha:raise ValueError('Operator source changed.')
    for name,item in c['archive_manifest']['files'].items():
        path=s/'code'/op.old.safe_name(name)
        if path.is_symlink() or path.stat().st_size!=item['bytes'] or op.old.digest(path)!=item['sha256']:raise ValueError('Original source/input changed.')
    session=json.loads((s/'session.json').read_text());plan=json.loads((s/'worker-plan-3.json').read_text())
    if (args.index!=3 or session['tasks']!=[] or session['complete'] or plan['index']!=3
            or plan['control_sha256']!=args.control_sha256 or session['reference_binding']!=op.records.binding(args.control_sha256)):
        raise ValueError('Only one previously unstarted candidate replay is allowed.')
    remaining=plan['deadline_unix']-time.time()
    if not 0<remaining<=480 or plan['deadline_unix']>session['deadline_unix']-120:raise ValueError('Worker/continuing clock invalid.')
    signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,remaining)
    with op.capsule.Capsule(s/'reference',manifest_sha256=op.records.MANIFEST_SHA) as reader:
        if reader.execution_binding(args.control_sha256)!=session['reference_binding']:raise ValueError('Old reference-to-new-execution binding changed.')
    op.base.gpu_gate(c)
    sys.path.insert(1,str(s/'code'))
    from types import SimpleNamespace
    from models.predictor_missing_replay_worker_v1 import run
    existing=Path.home()/'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting'
    result=run(SimpleNamespace(output=s/'results/3',index=3,tiny=False,device='cuda',
        upstream=existing/'external/vjepa2',checkpoint=existing/'checkpoints/vitl.pt',packet=s/'code',
        proposal=overlay/'configs/predictor_forward_preserving_handoff_v2.json',control_sha256=args.control_sha256,
        reference_capsule=s/'reference'))
    return 0 if result['passed'] else 1

if __name__=='__main__':sys.exit(main())
