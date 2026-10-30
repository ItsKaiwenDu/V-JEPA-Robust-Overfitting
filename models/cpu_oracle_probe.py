"""Bounded attended CPU reference forensics; no model, SDPA or GPU work."""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from models.attention_component import fp64_oracle
from models.attention_fixture import INPUTS,OUTPUTS,BOOKKEEPING,load
from models.cpu_oracle_packet import HOSTS,LIMITS,TASKS,verify
from models.cuda_profile import tensor_signature
from models.execution import configure_execution,precision_contract
from models.oracle_trace import Observation,audit_observation,cpu_location,load_arrays,save_arrays,signature,safe_agreement
from models.pilot_bundle import digest,make_archive
from models.reuse_diagnostic import upstream_gate
from models.study_artifacts import read_record,write_record

KIND='cpu-oracle-reproducibility-probe-v1'


def validate_plan(plan,sha,prior):
    if (plan.get('kind')!='cpu-oracle-attended-plan-v1' or plan.get('expected_host') not in HOSTS or plan.get('handoff_sha256')!=sha
            or plan.get('authorized') is not True or plan.get('attended') is not True or plan.get('CPU_only') is not True
            or any(plan.get(k) is not False for k in ('GPU_work_allowed','paid_compute','research_admitted','old_refusals_waived'))
            or plan.get('prior_consumed_claims')!=prior or plan.get('tasks')!=[list(t) for t in TASKS]
            or any(plan.get(k)!=v for k,v in LIMITS.items())):
        raise ValueError('Require a separate, attended CPU-only reference plan with preserved history.')


def host_gate(plan):
    if platform.node()!=plan['expected_host']:raise ValueError('CPU control/target host differs.')


def context():
    captured={}
    for name,function in [('torch_build',torch.__config__.show),('torch_parallel',torch.__config__.parallel_info),('numpy_build',np.__config__.show)]:
        output=io.StringIO()
        with contextlib.redirect_stdout(output):result=function()
        value=output.getvalue()+(result if isinstance(result,str) else '')
        captured[name]={'text':value[:65536],'truncated':len(value)>65536}
    cpuinfo=Path('/proc/cpuinfo');selected=[]
    if cpuinfo.is_file():
        for line in cpuinfo.read_text()[:131072].splitlines():
            if ':' in line and line.split(':')[0].strip() in ('model name','cpu family','model','stepping','flags','bugs') and line not in selected:selected.append(line)
    return {'host':platform.node(),'platform':platform.platform(),'python':platform.python_version(),'architecture':platform.machine(),
       'torch':str(torch.__version__),'numpy':np.__version__,'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
       'precision':precision_contract(),'cpu_capability':torch.backends.cpu.get_cpu_capability(),
       'affinity_read_only':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
       'CPU_count':os.cpu_count(),'CPU_location':cpu_location(),'selected_cpuinfo':selected,
       'task_environment':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG')},
       'CUDA_initialized':torch.cuda.is_initialized(),'library_configuration':captured}


def worker(args,root):
    folder=Path(args.worker_dir);auth=read_record(folder/'authorization.json',expected_sha256=args.authorization_sha256)
    if (auth['threads'],auth['replay'],auth['handoff_sha256'])!=(args.threads,args.replay,args.handoff_sha256):raise ValueError('CPU worker authorization differs.')
    remaining=auth['deadline_unix']-time.time()
    if not 0<remaining<=90:raise ValueError('Expired or widened CPU worker deadline.')
    signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,remaining)
    started=time.monotonic();m=verify(root,args.handoff_sha256);record={'kind':KIND,'threads':args.threads,'replay':args.replay,
      'CPU_only':True,'GPU_work_allowed':False,'tiny_fixture':m['tiny_fixture'],'capture_complete':False,'numerical_checks_passed':False,'passes':{},'stage':'starting'}
    def progress(stage):
        record['stage']=stage;write_record(folder/'worker.json',record,replace=(folder/'worker.json').exists());print('CPU oracle:',args.threads,args.replay,stage,flush=True)
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CPU probe requires hidden and uninitialized CUDA.')
        if not args.local:
            plan=json.loads(Path(args.plan).read_text());validate_plan(plan,args.handoff_sha256,m['prior_consumed_claims']);host_gate(plan)
            if digest(Path(args.plan))!=args.plan_sha256:raise ValueError('Changed CPU plan.')
            if str(torch.__version__)!='2.8.0+cu128' or np.__version__!='2.4.6' or platform.python_version()!='3.11.17':
                raise ValueError('Keep the failed worker environment unchanged; no install or fallback.')
        configure_execution('shared-fp64-native',device='cpu');torch.set_num_threads(args.threads);upstream_gate(args.upstream,m)
        values,fixture=load(Path(root)/m['fixture']['folder'],m['fixture']['sha256'],case='encoder-1',tiny=m['tiny_fixture'])
        record.update(handoff_sha256=args.handoff_sha256,fixture=m['fixture'],input_signatures={n:tensor_signature(values[n]) for n in INPUTS},
                      bounds=BOOKKEEPING,context_before=context(),local_CPU_rehearsal=args.local)
        original=[values[n] for n in INPUTS];progress('inputs_bound_and_CPU_context_saved')
        for phase in ('before','observed','after'):
            progress(phase);location_before=cpu_location()
            if phase=='observed':
                observation=Observation(folder/'trace')
                with observation.active():computed=fp64_oracle(*original)
                record['trace_index_sha256']=digest(folder/'trace/index.json')
            else:computed=fp64_oracle(*original)
            # All three planned calls finish and persist before numerical gates.
            record['passes'][phase]={'artifact':save_arrays(folder/(phase+'.npz'),computed),
                                    'location_before':location_before,'location_after':cpu_location()}
            progress(phase+'-saved')
        checks={}
        for phase,item in record['passes'].items():
            computed=load_arrays(folder,item['artifact'],{n:list(values['q'].shape) for n in OUTPUTS})
            checks[phase]={n:safe_agreement(computed[n],values['oracle_'+n].numpy()) for n in OUTPUTS}
        traced=audit_observation(folder/'trace',record['trace_index_sha256'],{n:values[n].numpy() for n in INPUTS})
        record['checks']=checks;record['trace_summary']={k:v for k,v in traced.items() if k!='stage_comparisons'}
        write_record(folder/'stage-comparisons.json',traced)
        record['stage_comparisons_sha256']=digest(folder/'stage-comparisons.json')
        record['phase_bitwise_equal']=all(record['passes'][p]['artifact']['arrays']==record['passes']['before']['artifact']['arrays'] for p in ('observed','after'))
        record['context_after']=context()
        if record['context_after']['CUDA_initialized'] or sys.gettrace() is not None:raise ValueError('CPU/hook boundary not restored.')
        if any(tensor_signature(values[n])!=record['input_signatures'][n] for n in INPUTS):raise ValueError('Fixed inputs were mutated.')
        record['capture_complete']=True
        record['numerical_checks_passed']=traced['numerical_checks_passed'] and all(v['passed'] for group in checks.values() for v in group.values())
        record['stage']='complete' if record['numerical_checks_passed'] else 'captured-numerical-refusal'
    except BaseException as exc:
        record.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        record['wall_seconds']=time.monotonic()-started
        record['peak_RSS_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
        write_record(folder/'worker.json',record,replace=(folder/'worker.json').exists())
    return record


def summarize(root,output,sha):
    m=verify(root,sha);output=Path(output);session=read_record(output/'session.json');values,_=load(Path(root)/m['fixture']['folder'],m['fixture']['sha256'],case='encoder-1',tiny=m['tiny_fixture'])
    order=[list(t) for t in TASKS];rows=session['tasks']
    if (session['kind']!=KIND or session['handoff_sha256']!=sha or session['CPU_only'] is not True or session['GPU_work_allowed'] is not False
            or session['deadline_unix']-session['started_unix']!=LIMITS['max_seconds']
            or session.get('wall_seconds',0)>LIMITS['max_seconds']
            or not rows or [[r['threads'],r['replay']] for r in rows]!=order[:len(rows)] or len(rows)>len(order)):
        raise ValueError('Incomplete or relabelled CPU probe coverage.')
    workers=[]
    for i,item in enumerate(rows):
        folder=output/'workers'/str(i);w=read_record(folder/'worker.json')
        if (digest(folder/'worker.json')!=item['worker_sha256'] or not w['capture_complete'] or w['kind']!=KIND
                or w['handoff_sha256']!=sha or [w['threads'],w['replay']]!=order[i] or w['bounds']!=BOOKKEEPING
                or w['fixture']!=m['fixture'] or w['input_signatures']!={n:tensor_signature(values[n]) for n in INPUTS}
                or w['tiny_fixture']!=m['tiny_fixture'] or w['local_CPU_rehearsal']!=session['local_rehearsal']
                or w['CPU_only'] is not True or w['GPU_work_allowed'] is not False or not 0<w['wall_seconds']<=90):
            raise ValueError('CPU worker capture/source/clock/domain failed.')
        for ctx in (w['context_before'],w['context_after']):
            if (ctx['CUDA_initialized'] or ctx['threads']!=w['threads'] or ctx['host']!=session['host']
                    or ctx['task_environment']['CUDA_VISIBLE_DEVICES']!=''):raise ValueError('CPU process context changed.')
        observed=audit_observation(folder/'trace',w['trace_index_sha256'],{n:values[n].numpy() for n in INPUTS})
        checks={}
        if digest(folder/'stage-comparisons.json')!=w['stage_comparisons_sha256']:raise ValueError('Stage comparison receipt changed.')
        from models.precision_diagnostic import same_report
        if not same_report(observed,read_record(folder/'stage-comparisons.json')):raise ValueError('Stage comparisons do not reaggregate.')
        arrays={}
        for phase in ('before','observed','after'):
            a=load_arrays(folder,w['passes'][phase]['artifact'],{n:list(values['q'].shape) for n in OUTPUTS});arrays[phase]=w['passes'][phase]['artifact']['arrays']
            checks[phase]={n:safe_agreement(a[n],values['oracle_'+n].numpy()) for n in OUTPUTS}
        if not same_report(checks,w['checks']):raise ValueError('Saved CPU reference comparisons changed.')
        exact=all(arrays[p]==arrays['before'] for p in ('observed','after'))
        numeric=observed['numerical_checks_passed'] and all(v['passed'] for group in checks.values() for v in group.values())
        if exact!=w['phase_bitwise_equal'] or numeric!=w['numerical_checks_passed']:raise ValueError('CPU observer/rejection label disagrees with saved tensors.')
        if i<len(rows)-1 and not numeric:raise ValueError('Probe continued after a rejected worker.')
        workers.append({'threads':w['threads'],'replay':w['replay'],'numerical_checks_passed':numeric,'phase_bitwise_equal':exact,
            'trace_chunks':observed['chunks'],'first_captured_mismatch':observed['first_captured_mismatch'],'checks':checks,
            'wall_seconds':w['wall_seconds'],'peak_RSS_bytes':w['peak_RSS_bytes']})
    if len(rows)<len(order) and workers[-1]['numerical_checks_passed']:raise ValueError('Prematurely truncated CPU probe without a refusal.')
    return {'kind':KIND,'capture_integrity_verified':True,'planned_tasks_completed':len(rows)==len(order),
       'numerical_checks_passed':all(w['numerical_checks_passed'] for w in workers),'workers':workers,
       'CPU_only':True,'GPU_work':False,'tiny_fixture':m['tiny_fixture'],'local_rehearsal':session['local_rehearsal'],
       'root_cause_established':False,'no_anomaly_proves_host_reliable':False,
       'generator_replay_qualified':False,'old_refusals_waived':False,'optimization_adopted':False,'research_admitted':False}


def export(root,output):
    root=Path(root);output=Path(output);m=json.loads((root/'CPU-ORACLE-HANDOFF.json').read_text())
    files={'CPU-ORACLE-HANDOFF.json':root/'CPU-ORACLE-HANDOFF.json',**{'source/'+n:root/n for n in m['files']}}
    for p in output.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.npz','.partial','.log','.txt') and p.name!='evidence.tar.gz.manifest.json':
            if p.is_symlink():raise ValueError('No symlink evidence.')
            files['results/'+str(p.relative_to(output))]=p
    if sum(p.stat().st_size for p in files.values())>LIMITS['max_export_bytes']:raise ValueError('CPU export exceeds 1 GiB.')
    spec=make_archive(output/'evidence.tar.gz',files);(output/'evidence.tar.gz.manifest.json').write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n');return spec


def run(args,root):
    m=verify(root,args.handoff_sha256);upstream_gate(args.upstream,m)
    started=args.launcher_started
    if not math.isfinite(started) or not 0<=time.time()-started<=120:raise ValueError('Invalid original CPU launcher clock.')
    if not args.local:
        p=json.loads(Path(args.plan).read_text());validate_plan(p,args.handoff_sha256,m['prior_consumed_claims']);host_gate(p)
        if digest(Path(args.plan))!=args.plan_sha256 or m['tiny_fixture']:raise ValueError('Wrong CPU plan or tiny packet.')
        if (str(torch.__version__)!='2.8.0+cu128' or np.__version__!='2.4.6' or platform.python_version()!='3.11.17'):raise ValueError('Prior CPU library environment changed.')
        claims=Path(root)/'cpu-oracle-claims';claims.mkdir(exist_ok=True)
        with (claims/(args.plan_sha256+'.json')).open('x') as f:json.dump({'started_unix':started,'CPU_only':True},f)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False);deadline=started+LIMITS['max_seconds']
    session={'kind':KIND,'handoff_sha256':args.handoff_sha256,'CPU_only':True,'GPU_work_allowed':False,'local_rehearsal':args.local,
             'host':platform.node(),'tasks':[],'started_unix':started,'deadline_unix':deadline,'capture_complete':False,'numerical_checks_passed':False}
    write_record(output/'session.json',session);child=None
    def stop(*unused):raise RuntimeError('CPU probe interrupted or bounded deadline reached; no retry.')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGHUP,stop);signal.signal(signal.SIGTERM,stop)
    signal.setitimer(signal.ITIMER_REAL,max(.1,deadline-time.time()-5))
    try:
        for i,(threads,replay) in enumerate(TASKS):
            if deadline-time.time()<LIMITS['worker_seconds']+60:raise TimeoutError('No budget for CPU worker and evidence export.')
            folder=output/'workers'/str(i);folder.mkdir(parents=True)
            write_record(folder/'authorization.json',{'threads':threads,'replay':replay,'handoff_sha256':args.handoff_sha256,'deadline_unix':time.time()+90})
            command=[sys.executable,'-B','-u',str(Path(root)/'scripts/check_cpu_oracle.py'),'--diagnostic','--worker',
                '--threads',str(threads),'--replay',str(replay),'--worker-dir',str(folder),'--authorization-sha256',digest(folder/'authorization.json'),
                '--handoff-sha256',args.handoff_sha256,'--upstream',str(args.upstream)]
            if args.local:command+=['--local']
            else:command+=['--plan',str(args.plan),'--plan-sha256',args.plan_sha256]
            print('Fresh CPU oracle worker:',threads,replay,flush=True)
            with (folder/'worker.log').open('x') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':str(threads),'MKL_NUM_THREADS':str(threads),'OPENBLAS_NUM_THREADS':str(threads),'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'})
                try:child.wait(timeout=90)
                finally:
                    if child.poll() is None:
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                    child.wait()
                code=child.returncode;child=None
            w=read_record(folder/'worker.json')
            if not w['capture_complete'] or code not in (0,1) or bool(code==0)!=w['numerical_checks_passed']:
                raise RuntimeError('CPU worker capture failed; preserve incomplete evidence and stop.')
            session['tasks'].append({'threads':threads,'replay':replay,'worker_sha256':digest(folder/'worker.json')});write_record(output/'session.json',session,replace=True)
            if not w['numerical_checks_passed']:
                session['stop_reason']='captured_reference_mismatch';break
        summary=summarize(root,output,args.handoff_sha256);write_record(output/'summary.json',summary)
        session.update(capture_complete=True,numerical_checks_passed=summary['numerical_checks_passed'],summary_sha256=digest(output/'summary.json'))
    except BaseException as exc:session.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
        session['wall_seconds']=time.time()-started;write_record(output/'session.json',session,replace=True)
        try:
            verify(root,args.handoff_sha256);spec=export(root,output)
            print('CPU evidence:',json.dumps({'bytes':spec['bytes'],'sha256':spec['sha256']}),flush=True)
        finally:signal.setitimer(signal.ITIMER_REAL,0)
    return session
