"""Local CPU fixed-input component rehearsal; no university operator or grant."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))


def local_only(device):
    if device!='cpu' or platform.node() in ('linux10603','linux10609'):
        raise ValueError('Only the local CPU rehearsal is prepared; no university/CUDA work.')


def cohort(output):
    local_only('cpu')
    from models import predictor_sdpa_component_v2 as component
    from models import predictor_sdpa_component_records_v1 as record
    output=output.resolve();output.mkdir(mode=0o700,exist_ok=False)
    plan=component.make_local_plan(ROOT,output/'reference');plan_path=output/'component-plan.json'
    with plan_path.open('xb') as f:os.chmod(plan_path,0o600);f.write(record.encoded(plan));f.flush();os.fsync(f.fileno())
    state={'kind':'local-CPU-fixed-SDPA-four-fresh-workers-v1','plan_sha256':record.sha(plan_path),'complete':False,'workers':[],
        'CPU_only':True,'research_admitted':False,'performance_qualified':False,'CUDA_qualified':False,'automatic_retry':False}
    try:
        for index in range(4):
            deadline=time.time()+120
            cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--plan',str(plan_path),
                '--plan-sha256',state['plan_sha256'],'--index',str(index),'--output',str(output/str(index)),
                '--deadline-unix',str(deadline),'--device','cpu']
            with (output/f'{index}.log').open('xb') as log:
                os.chmod(log.name,0o600)
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=max(.001,deadline-time.time()))
            state['workers'].append({'index':index,'exit_status':result.returncode,'log_sha256':record.sha(output/f'{index}.log')})
            if result.returncode:raise RuntimeError('Component worker failed; preserve values and logs. No retry or next cell.')
        _,_,_,identity=component.capture(plan)
        verified=[component.load_worker(output/str(i),plan,state['plan_sha256'],identity,i) for i in range(4)]
        if len({w['pid'] for w in verified})!=4:raise ValueError('Four genuinely fresh CPU workers required.')
        state['complete']=True
    except BaseException as e:state.update(error_type=type(e).__name__,error=str(e));raise
    finally:
        with (output/'session.json').open('xb') as f:os.chmod(f.name,0o600);f.write(record.encoded(state))
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--index',type=int)
    parser.add_argument('--deadline-unix',type=float)
    args=parser.parse_args();local_only(args.device)
    if args.worker:
        if any(getattr(args,k) is None for k in ('plan','plan_sha256','index','deadline_unix')):parser.error('Worker requires its complete bound cell arguments.')
        from models.predictor_sdpa_component_v2 import run
        state=run(args)
    else:state=cohort(args.output)
    print(json.dumps({'complete':state['complete'],'CPU_only':True,'CUDA_qualified':False,'performance_qualified':False},sort_keys=True))


if __name__=='__main__':main()
