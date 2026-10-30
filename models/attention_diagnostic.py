"""Bounded standalone attention comparison. No original model policy changes."""
import gc
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import torch

from models.attention_component import (KIND,CASES,ARMS,TOLERANCES,ORACLE_ID,operands,clone_layout,
    fp64_oracle,iteration,compare,backend_evidence,sdp_flags,force_backend)
from models.attention_packet import HOSTS,ROLES,LIMITS,verify
from models.cuda_profile import tensor_signature
from models.execution import configure_execution
from models.native_probe import profile_records
from models.pilot_bundle import digest,make_archive
from models.precision_diagnostic import same_report
from models.reuse_diagnostic import hardware_gate,upstream_gate
from models.study_artifacts import read_record,write_record
from models.training import runtime_contract


def tasks(role,tiny):
    cases=CASES if tiny and role=='all' else ROLES[role]
    return [(case,arm,replay) for case in cases for replay in (0,1) for arm in (('math',) if tiny else ARMS)]


def task_name(case,arm,replay):return f'{case}-{arm}-{replay}'


def validate_launch(plan,sha,role):
    host=plan.get('expected_host')
    if (plan.get('kind')!='attention-component-launch-v1' or host not in HOSTS or HOSTS[host]['role']!=role
            or plan.get('role')!=role or plan.get('handoff_sha256')!=sha or plan.get('authorized') is not True
            or plan.get('diagnostic_only') is not True or plan.get('paid_compute') is not False
            or plan.get('research_admitted') is not False or plan.get('attended') is not True
            or plan.get('expected_gpu')!='NVIDIA GeForce RTX 3070' or plan.get('expected_gpu_uuid')!=HOSTS[host]['uuid']
            or plan.get('expected_driver')!='580.178.04' or any(plan.get(k)!=v for k,v in LIMITS.items())):
        raise ValueError('Require the bound one-attempt host/role/component resource plan.')


def runtime_ok(runtime,device):
    return (runtime['device']==device and runtime['threads']==4 and runtime['deterministic_algorithms']
        and not runtime['deterministic_warn_only'] and runtime['matmul_precision']=='highest'
        and not runtime['matmul_tf32'] and runtime['cudnn_enabled'] and not runtime['cudnn_tf32']
        and not runtime['cudnn_benchmark'] and runtime['cudnn_deterministic']
        and runtime['cublas_workspace_config']==':4096:8'
        and (device=='cpu' or runtime['torch'].split('+')[0]=='2.8.0' and runtime['cuda']=='12.8'))


def worker(args,root):
    folder=Path(args.worker_dir);auth=read_record(folder/'authorization.json')
    if (digest(folder/'authorization.json')!=args.authorization_sha256 or auth['case']!=args.case
            or auth['arm']!=args.arm or auth['replay']!=args.replay or auth['handoff_sha256']!=args.handoff_sha256):
        raise ValueError('Worker authorization differs from parent.')
    remaining=auth['deadline_unix']-time.time()
    if not 0<remaining<=(30 if args.tiny else 120):raise ValueError('Worker deadline invalid.')
    signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,remaining)
    metadata,reference=verify(root,args.handoff_sha256);started=time.monotonic()
    result={'kind':KIND,'case':args.case,'arm':args.arm,'replay':args.replay,'tiny_fixture':args.tiny,
            'device':args.device,'complete':False,'passed':False,'research_admitted':False,'timings':[]}
    def progress(stage):
        result['stage']=stage;write_record(folder/'worker.json',result,replace=(folder/'worker.json').exists())
        print('Attention component:',args.case,args.arm,args.replay,stage,flush=True)
    try:
        progress('configure_and_bind_original_sources')
        configure_execution('shared-fp64-native',device=args.device);torch.set_num_threads(4);upstream_gate(args.upstream,metadata)
        hardware=None
        if not args.tiny:
            plan=json.loads(Path(args.plan).read_text());validate_launch(plan,args.handoff_sha256,args.role)
            if digest(Path(args.plan))!=args.plan_sha256:raise ValueError('Changed launch plan.')
            hardware=hardware_gate(plan,allowed_pid=os.getpid())
            if torch.__version__.split('+')[0]!='2.8.0' or torch.version.cuda!='12.8' or not torch.cuda.is_available():
                raise ValueError('Existing PyTorch2.8/CUDA12.8 required; no install/fallback.')
            torch.cuda.set_per_process_memory_fraction(min(.9,4*2**30/torch.cuda.get_device_properties(0).total_memory))
        elif args.arm!='math':raise ValueError('CPU fixtures use math only and cannot qualify CUDA efficient attention.')
        progress('capture_synthetic_post_rotary_operands_on_CPU')
        (q,k,v,g),shape=operands(args.upstream,args.case,reference,args.tiny)
        shape=dict(shape);positions=shape.pop('positions');shape['position_signature']=tensor_signature(positions)
        if not args.tiny:
            declared=reference['shapes'][args.case]
            if any(shape[k]!=declared[k] for k in ('heads','head_dim','tokens')):raise ValueError('Original geometry changed.')
            if args.case.startswith('encoder') and shape['position_signature']!=reference['first_draw_mask_signatures'][int(args.case[-1])][0]:
                raise ValueError('Original encoder mask positions changed.')
        common={'kind':KIND,'handoff_sha256':args.handoff_sha256,'case':args.case,'tiny_fixture':args.tiny,
            'runtime':runtime_contract(args.device),'hardware':hardware,'shape':shape,'synthetic_operands_only':True,
            'released_weights_or_pixels_loaded':False,'original_global_sdp_flags':sdp_flags(),
            'operand_signatures':{n:tensor_signature(t) for n,t in zip(('q','k','v','cotangent'),(q,k,v,g))},
            'tolerances':TOLERANCES,'oracle':ORACLE_ID,'oracle_query_chunk':128,'canonical_evaluation_artifact':False,
            'research_admitted':False}
        if not runtime_ok(common['runtime'],args.device):raise ValueError('Original precision/runtime gate rejected.')
        result['common']=common
        progress('independent_CPU_FP64_forward_and_analytical_QKV_oracle')
        oracle=fp64_oracle(q,k,v,g)
        leaves=[clone_layout(t,args.device,requires_grad=True) for t in (q,k,v)];cot=clone_layout(g,args.device)
        if [list(t.stride()) for t in leaves]!=[shape['strides'][n] for n in ('q','k','v')]:raise ValueError('Operand transfer changed post-RoPE strides.')
        progress('untimed_numerical_gate')
        current,_=iteration(leaves,cot,args.arm,args.device)
        result['oracle_checks']=compare(current,oracle)
        if not all(v['passed'] for v in result['oracle_checks'].values()):raise ValueError('Predeclared component FP64 oracle rejected; no tolerance widened.')
        actual_signature={n:tensor_signature(t) for n,t in current.items()}
        if any(actual_signature[n]['abs_max']==0 for n in actual_signature):raise ValueError('Require nonzero output and all QKV derivatives.')
        if auth['reference'] is not None:
            ref=auth['reference'];other=folder.parent/ref['worker'];previous=read_record(other/'worker.json')
            if (digest(other/'worker.json')!=ref['worker_sha256'] or not previous['passed'] or not previous['complete']
                    or previous['common']!=common or previous['arm']!='efficient'
                    or digest(other/'values.pt')!=ref['values_sha256']):raise ValueError('Reference worker is invalid or incomparable.')
            data=torch.load(other/'values.pt',map_location='cpu',weights_only=True)
            expected={n:data['actual_'+n] for n in TOLERANCES}
            if args.arm=='efficient':
                if any(tensor_signature(expected[n])!=actual_signature[n] for n in TOLERANCES):raise ValueError('Fresh reference replay changed before timing.')
                result['reference_comparison']={'exact_replay':True}
            else:
                result['reference_comparison']=compare(current,expected)
                if not all(v['passed'] for v in result['reference_comparison'].values()):raise ValueError('Candidate/reference component numerical gate rejected before timing.')
        progress('actual_backend_profile_before_timing')
        activities=[torch.profiler.ProfilerActivity.CPU]
        if args.device=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities,record_shapes=False,with_stack=False,profile_memory=False) as profile:
            profiled,_=iteration(leaves,cot,args.arm,args.device)
        rows=profile_records(profile);del profile;gc.collect()
        if len(rows)>20000:raise ValueError('Attention profiler event cap exceeded.')
        result['backend_profile_events']=rows;result['backend_evidence']=backend_evidence(rows,args.arm,args.device)
        if not result['backend_evidence']['passed']:raise ValueError('Requested backend was not observed; no fallback timing.')
        if {n:tensor_signature(t) for n,t in profiled.items()}!=actual_signature:raise ValueError('Profiler changed component output/derivatives.')
        progress('warmup_and_three_unprofiled_timed_passes')
        for i in range(4):
            values,timing=iteration(leaves,cot,args.arm,args.device)
            sig={n:tensor_signature(t) for n,t in values.items()}
            if sig!=actual_signature:raise ValueError('Within-process replay changed; do not compare timings.')
            if i:result['timings'].append(timing)
        result['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated() if args.device=='cuda' else None
        result['median_seconds']={n:statistics.median(t[n] for t in result['timings']) for n in ('forward_seconds','backward_seconds','total_seconds')}
        saved={**{n:t for n,t in zip(('q','k','v','cotangent'),(q,k,v,g))},
               **{'actual_'+n:t.detach().cpu() for n,t in current.items()},**{'oracle_'+n:t for n,t in oracle.items()}}
        with (folder/'values.pt').open('xb') as f:torch.save(saved,f)
        result['artifact']={'file':'values.pt','sha256':digest(folder/'values.pt'),
                            'tensor_signatures':{n:tensor_signature(t) for n,t in saved.items()}}
        if runtime_contract(args.device)!=common['runtime'] or sdp_flags()!=common['original_global_sdp_flags']:
            raise ValueError('Execution flags changed at component boundary.')
        verify(root,args.handoff_sha256)
        if not args.tiny:hardware_gate(plan,allowed_pid=os.getpid())
        result.update(complete=True,passed=True,stage='complete',within_process_exact=True)
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result['wall_seconds']=time.monotonic()-started
        write_record(folder/'worker.json',result,replace=(folder/'worker.json').exists())


def summarize(root,output,sha,role,tiny,*,upstream=None):
    metadata,reference=verify(root,sha);output=Path(output);session=read_record(output/'session.json')
    expected=tasks(role,tiny)
    # Downloads move between Linux and macOS. Bind source bytes, not the old
    # machine's path; callers may supply their verified local checkout.
    local_upstream=Path(session['upstream']) if upstream is None else Path(upstream)
    upstream_gate(local_upstream,metadata)
    if (session['kind']!=KIND or session['handoff_sha256']!=sha or session['role']!=role or session['tiny_fixture']!=tiny
            or session['research_admitted'] is not False or session['paid_compute'] is not False
            or [(v['case'],v['arm'],v['replay']) for v in session['tasks']]!=expected):raise ValueError('Incomplete or relabelled component coverage.')
    workers={};artifacts={};synthetic={};oracles={}
    for task in session['tasks']:
        key=(task['case'],task['arm'],task['replay']);folder=output/'workers'/task_name(*key);w=read_record(folder/'worker.json')
        if (digest(folder/'worker.json')!=task['worker_sha256'] or not w['passed'] or not w['complete'] or w['kind']!=KIND
                or (w['case'],w['arm'],w['replay'])!=key or w['device']!=('cpu' if tiny else 'cuda') or w['tiny_fixture']!=tiny
                or w['research_admitted'] is not False or not w['within_process_exact'] or w['stage']!='complete'):
            raise ValueError('Invalid component worker receipt.')
        c=w['common']
        if (c['kind']!=KIND or c['handoff_sha256']!=sha or c['case']!=key[0] or c['tiny_fixture']!=tiny
                or c['canonical_evaluation_artifact'] is not False or c['research_admitted'] is not False
                or c['tolerances']!=TOLERANCES or c['oracle']!=ORACLE_ID or c['oracle_query_chunk']!=128
                or not runtime_ok(c['runtime'],w['device']) or c['synthetic_operands_only'] is not True
                or c['released_weights_or_pixels_loaded'] is not False
                or set(c['original_global_sdp_flags'])!={'flash','efficient','math','cudnn'}
                or any(type(v) is not bool for v in c['original_global_sdp_flags'].values())):
            raise ValueError('Changed numerical bounds, oracle, runtime or artifact domain.')
        if not tiny and c['hardware']!=session['hardware']:raise ValueError('Worker GPU identity changed.')
        evidence=backend_evidence(w['backend_profile_events'],key[1],w['device'])
        if evidence!=w['backend_evidence'] or not evidence['passed']:raise ValueError('Actual backend evidence rejected.')
        item=w['artifact']
        if item['file']!='values.pt' or digest(folder/'values.pt')!=item['sha256']:raise ValueError('Component tensors changed.')
        values=torch.load(folder/'values.pt',map_location='cpu',weights_only=True)
        names={'q','k','v','cotangent'}|{'actual_'+n for n in TOLERANCES}|{'oracle_'+n for n in TOLERANCES}
        if set(values)!=names:raise ValueError('Incomplete or extra component tensor keys.')
        if {n:tensor_signature(t) for n,t in values.items()}!=item['tensor_signatures']:raise ValueError('Component tensor signatures changed.')
        if key[0] not in synthetic:
            tensors,shape=operands(local_upstream,key[0],reference,tiny)
            shape=dict(shape);shape['position_signature']=tensor_signature(shape.pop('positions'))
            synthetic[key[0]]=(tensors,shape);oracles[key[0]]=fp64_oracle(*tensors)
        tensors,shape=synthetic[key[0]]
        if c['shape']!=shape or c['operand_signatures']!={n:tensor_signature(t) for n,t in zip(('q','k','v','cotangent'),tensors)}:
            raise ValueError('Synthetic seed, upstream rotary geometry, strides or operand identity changed.')
        if any(not torch.equal(values[n],t) or values[n].dtype!=torch.float32 for n,t in zip(('q','k','v','cotangent'),tensors)):
            raise ValueError('Saved operands do not reconstruct from the pinned upstream synthetic capture.')
        if any(list(values[n].stride())!=shape['strides'][n] for n in ('q','k','v')):
            raise ValueError('Saved post-RoPE operand layout changed.')
        actual={n:values['actual_'+n] for n in TOLERANCES};oracle={n:values['oracle_'+n] for n in TOLERANCES}
        if any(actual[n].dtype!=torch.float32 or oracle[n].dtype!=torch.float64
                or not torch.allclose(oracle[n],oracles[key[0]][n],rtol=1e-12,atol=1e-14) for n in TOLERANCES):
            raise ValueError('FP64 oracle does not reconstruct independently or dtype changed.')
        for n,v in compare(actual,oracle).items():
            if not v['passed'] or not same_report(v,w['oracle_checks'][n]):raise ValueError('Saved FP64 accuracy gate rejected.')
        timing_names=('forward_seconds','backward_seconds','total_seconds')
        if len(w['timings'])!=3 or any(not math.isfinite(t[k]) or t[k]<=0 for t in w['timings'] for k in timing_names):
            raise ValueError('Invalid timing samples.')
        if (any(not math.isclose(t['forward_seconds']+t['backward_seconds'],t['total_seconds'],rel_tol=1e-10,abs_tol=1e-12) for t in w['timings'])
                or w['median_seconds']!={n:statistics.median(t[n] for t in w['timings']) for n in timing_names}):
            raise ValueError('Timing intervals or medians disagree with retained observations.')
        if not 0<w['wall_seconds']<=(30 if tiny else 120):raise ValueError('Worker exceeded its bound.')
        bad_memory=(w['peak_cuda_allocated_bytes'] is not None) if tiny else not 0<w['peak_cuda_allocated_bytes']<=4*2**30
        if bad_memory:
            raise ValueError('Component memory/domain gate rejected.')
        workers[key]=w;artifacts[key]=values
    summaries={}
    cases=CASES if role=='all' else ROLES[role]
    for case in cases:
        arms=('math',) if tiny else ARMS
        common=workers[(case,arms[0],0)]['common']
        for arm in arms:
            a,b=workers[(case,arm,0)],workers[(case,arm,1)]
            if a['common']!=common or b['common']!=common:raise ValueError('Fresh workers have different operand/source/runtime identities.')
            if a['artifact']['tensor_signatures']!=b['artifact']['tensor_signatures']:raise ValueError('Fresh-process tensors failed exact replay.')
        if not tiny:
            for replay in (0,1):
                ref=artifacts[(case,'efficient',replay)];candidate=artifacts[(case,'math',replay)]
                checks=compare({n:candidate['actual_'+n] for n in TOLERANCES},{n:ref['actual_'+n] for n in TOLERANCES})
                if any(not v['passed'] or not same_report(v,workers[(case,'math',replay)]['reference_comparison'][n]) for n,v in checks.items()):
                    raise ValueError('Saved candidate/reference component gate rejected.')
        timing={arm:{name:statistics.median(w['timings'][i][name] for replay in (0,1) for i in range(3) for w in [workers[(case,arm,replay)]])
                         for name in ('forward_seconds','backward_seconds','total_seconds')} for arm in arms}
        summaries[case]={'median_seconds':timing,'fresh_process_tensor_replay_exact':True,
                        'efficient_over_math_backward_speedup':None if tiny else timing['efficient']['backward_seconds']/timing['math']['backward_seconds'],
                        'efficient_over_math_total_speedup':None if tiny else timing['efficient']['total_seconds']/timing['math']['total_seconds']}
    return {'kind':KIND,'passed':True,'role':role,'tiny_fixture':tiny,'domain':'CPU-math-fixture' if tiny else 'university-CUDA-components',
            'cases':summaries,'all_numerical_and_actual_backend_gates_passed':True,'predeclared_component_tolerances':TOLERANCES,
            'CUDA_backend_comparison_qualified':not tiny,'full_model_correctness_or_speed_qualified':False,
            'optimization_adopted':False,'full_study_cost_approved':False,'research_admitted':False}


def export(root,output):
    root=Path(root);output=Path(output);m=json.loads((root/'ATTENTION-HANDOFF.json').read_text())
    files={'ATTENTION-HANDOFF.json':root/'ATTENTION-HANDOFF.json',**{'source/'+n:root/n for n in m['files']}}
    for p in output.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.jsonl','.log','.pt','.txt') and p.name!='evidence.tar.gz.manifest.json':
            if p.is_symlink():raise ValueError('Evidence cannot contain symlinks.')
            files['results/'+str(p.relative_to(output))]=p
    if sum(p.stat().st_size for p in files.values())>2**30:raise ValueError('Component export exceeds 1 GiB.')
    spec=make_archive(output/'evidence.tar.gz',files)
    (output/'evidence.tar.gz.manifest.json').write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n');return spec


def run(args,root):
    metadata,reference=verify(root,args.handoff_sha256);upstream_gate(args.upstream,metadata)
    started=args.launcher_started
    if not math.isfinite(started) or not 0<=time.time()-started<=120:raise ValueError('Invalid original component launcher clock.')
    if args.tiny:
        if not 0<args.max_seconds<=300 or not 0<args.worker_seconds<=30:raise ValueError('CPU fixture bounds exceeded.')
    else:
        plan=json.loads(Path(args.plan).read_text());validate_launch(plan,args.handoff_sha256,args.role)
        if digest(Path(args.plan))!=args.plan_sha256 or args.max_seconds!=1200 or args.worker_seconds!=120:raise ValueError('Component launch limits changed.')
        hardware_gate(plan)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    if not args.tiny:
        claims=Path(root)/'attention-claims';claims.mkdir(exist_ok=True)
        with (claims/(args.plan_sha256+'.json')).open('x') as f:json.dump({'started_unix':started,'output':str(output)},f)
    deadline=started+args.max_seconds;end=time.monotonic()+deadline-time.time()
    session={'kind':KIND,'role':args.role,'tiny_fixture':args.tiny,'handoff_sha256':args.handoff_sha256,
             'started_unix':started,'deadline_unix':deadline,'upstream':str(Path(args.upstream).resolve()),
             'hardware':None if args.tiny else hardware_gate(plan),
             'tasks':[],'passed':False,'complete':False,'research_admitted':False,'paid_compute':False}
    write_record(output/'session.json',session);child=None
    def alarm(*unused):raise TimeoutError('Independent component session deadline.')
    def interrupted(*unused):raise RuntimeError('Attended session interrupted; no retry.')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.1,deadline-time.time()-5))
    signal.signal(signal.SIGHUP,interrupted);signal.signal(signal.SIGTERM,interrupted)
    def monitor(row):
        with (output/'gpu-monitor.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
    try:
        for case,arm,replay in tasks(args.role,args.tiny):
            remaining=min(deadline-time.time(),end-time.monotonic())-(15 if args.tiny else 90)
            if remaining<=0:raise TimeoutError('No time remains for worker and evidence export.')
            seconds=min(args.worker_seconds,remaining);folder=output/'workers'/task_name(case,arm,replay);folder.mkdir(parents=True)
            ref=None
            if not args.tiny and (arm=='math' or replay==1):
                ref_name=task_name(case,'efficient',replay if arm=='math' else 0);previous=output/'workers'/ref_name
                w=read_record(previous/'worker.json')
                ref={'worker':ref_name,'worker_sha256':digest(previous/'worker.json'),'values_sha256':w['artifact']['sha256']}
            write_record(folder/'authorization.json',{'case':case,'arm':arm,'replay':replay,'handoff_sha256':args.handoff_sha256,'deadline_unix':time.time()+seconds,'reference':ref})
            command=[sys.executable,'-B','-u',str(Path(root)/'scripts/check_attention_component.py'),'--diagnostic','--device',args.device,
              '--role',args.role,'--worker','--case',case,'--arm',arm,'--replay',str(replay),'--worker-dir',str(folder),
              '--authorization-sha256',digest(folder/'authorization.json'),'--handoff-sha256',args.handoff_sha256,'--upstream',str(args.upstream)]
            if args.tiny:command+=['--tiny']
            else:command+=['--plan',str(args.plan),'--plan-sha256',args.plan_sha256]
            print('Fresh attention worker:',case,arm,replay,flush=True)
            if not args.tiny:hardware_gate(plan,retain=monitor)
            with (folder/'worker.log').open('x') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'})
                tick=time.monotonic();last_gate=0.
                try:
                    while child.poll() is None:
                        if time.monotonic()-tick>=seconds:raise TimeoutError('Component worker timed out; no retry.')
                        if not args.tiny and time.monotonic()-last_gate>=5:
                            hardware_gate(plan,allowed_pid=child.pid,retain=monitor);last_gate=time.monotonic()
                        time.sleep(.2)
                    if child.returncode:raise RuntimeError('Component worker rejected; preserve evidence, no retry or next case.')
                finally:
                    if child.poll() is None:
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                    child.wait();child=None
            session['tasks'].append({'case':case,'arm':arm,'replay':replay,'worker_sha256':digest(folder/'worker.json')})
            write_record(output/'session.json',session,replace=True)
        summary=summarize(root,output,args.handoff_sha256,args.role,args.tiny);write_record(output/'summary.json',summary)
        session.update(complete=True,passed=True,summary_sha256=digest(output/'summary.json'))
    except BaseException as exc:
        session.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        if child is not None:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            child.wait()
        session['wall_seconds']=time.time()-started;write_record(output/'session.json',session,replace=True)
        try:
            verify(root,args.handoff_sha256);spec=export(root,output)
            print('Component evidence:',json.dumps({'bytes':spec['bytes'],'sha256':spec['sha256']}),flush=True)
        except BaseException as exc:
            session.update(complete=False,passed=False,export_error=str(exc));write_record(output/'session.json',session,replace=True);raise
        finally:signal.setitimer(signal.ITIMER_REAL,0)
    return session
