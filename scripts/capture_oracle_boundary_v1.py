#!/usr/bin/env python3
"""Cold packaged entrypoint for exactly one original pre-transfer worker."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

OVERLAY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OVERLAY))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--control-sha256')
    p.add_argument('--launcher-started',type=float,required=True)
    p.add_argument('--local-tiny-rehearsal',action='store_true')
    a=p.parse_args()
    if (a.packet.is_symlink() or a.packet.resolve()!=(OVERLAY.parent/'code').resolve()
            or a.scratch.is_symlink() or a.scratch.resolve()!=OVERLAY.parent.resolve()):
        p.error('Original sibling packet/scratch directories required.')
    from models import oracle_boundary_operator_v1 as op
    manifest_path=a.packet/'FIXED-ATTENTION-HANDOFF.json';manifest=json.loads(manifest_path.read_text())
    sha=op.old.digest(manifest_path)
    if a.local_tiny_rehearsal:
        if manifest.get('tiny_fixture') is not True or platform.node() in op.old.HOSTS or a.control_sha256:
            p.error('Explicit tiny rehearsal is local only and cannot use live authorization.')
        control=None
    else:
        if not a.control_sha256 or op.old.digest(a.scratch/'operator-control.json')!=a.control_sha256:
            p.error('Missing/changed new boundary control.')
        control=json.loads((a.scratch/'operator-control.json').read_text());op.validate_control(control)
        op.validate_authorization(json.loads((a.scratch/'resource-authorization.json').read_text()),a.control_sha256)
        if sha!=op.HANDOFF_SHA or manifest.get('tiny_fixture') is not False:p.error('Require the unchanged original full packet.')
        for parent,items in [(OVERLAY,{n:{'sha256':v} for n,v in control['remote_source_sha256'].items()}),(a.packet,control['archive_manifest']['files'])]:
            for name,item in items.items():
                path=parent/op.old.safe_name(name)
                if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(parent.resolve()) or op.old.digest(path)!=item['sha256']:
                    p.error('Packaged source or fixture changed: '+name)
    if not 0<=time.time()-a.launcher_started<=120:p.error('Original outer clock is stale.')
    out=a.scratch/'results'
    if out.exists():p.error('One boundary capture only; never repeat a launch.')
    out.mkdir(mode=0o700);worker_dir=out/'original-worker';worker_dir.mkdir(mode=0o700)
    # Add the packet after overlay models. No numerical import/fixture loading
    # occurs until the unchanged original worker module is imported below.
    sys.path.insert(1,str(a.packet.resolve()))
    original_plan=None
    if control is not None:
        original_plan=out/'original-schema-plan.json'
        plan={'kind':'fixed-attention-launch-v1','expected_host':'linux10603','role':'encoder-1',
            'handoff_sha256':sha,'prior_consumed_attempts':manifest['prior_consumed_attempts'],
            'fixed_inputs':True,'generator_replay_qualified':False,'authorized':True,'attended':True,
            'diagnostic_only':True,'paid_compute':False,'research_admitted':False,
            'expected_gpu':control['expected_gpu'],'expected_gpu_uuid':control['expected_gpu_uuid'],
            'expected_driver':control['expected_driver'],**manifest['limits'],
            'outer_boundary_control_sha256':a.control_sha256,'outer_max_seconds':600,
            'original_parent_invoked':False,
            'authorization_basis':'NEW boundary exception only; this inner schema satisfies the unchanged worker validator and grants no original 1200-second parent run.'}
        op.old.write_json(original_plan,plan)
    seconds=30 if a.local_tiny_rehearsal else min(120,a.launcher_started+600-time.time()-5)
    if seconds<=0:p.error('No original allowance remains.')
    from models.oracle_boundary_observer_v1 import observe,record
    authorization={'case':'encoder-1','arm':'math','replay':0,'handoff_sha256':sha,
        'deadline_unix':time.time()+seconds,'reference':None,
        'scope':'new first-oracle-return capture only; no SDPA or full worker acceptance'}
    record(worker_dir/'authorization.json',authorization)
    from models import fixed_attention_diagnostic as original
    args=SimpleNamespace(worker_dir=worker_dir,authorization_sha256=op.old.digest(worker_dir/'authorization.json'),
        case='encoder-1',arm='math',replay=0,handoff_sha256=sha,tiny=a.local_tiny_rehearsal,
        device='cpu' if a.local_tiny_rehearsal else 'cuda',upstream=a.upstream,role='encoder-1',
        plan=original_plan,plan_sha256=op.old.digest(original_plan) if original_plan else None)
    state=observe(original,args,a.packet,out,local=a.local_tiny_rehearsal)
    print('Boundary outcome:',state['outcome'],flush=True)
    print('Original full GPU worker remains incomplete; no backend qualification.',flush=True)
    raise SystemExit(0 if state['outcome']=='captured-gate-pass' else 1 if state['outcome']=='captured-original-refusal' else 2)


if __name__=='__main__':main()
