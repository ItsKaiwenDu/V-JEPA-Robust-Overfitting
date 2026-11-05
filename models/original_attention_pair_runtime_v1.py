"""Standard-library task issuance and fresh-process dispatch; no numerical imports."""
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

from models.oracle_boundary_observer_v1 import record,read_record,digest

KIND='original-attention-pair-runtime-v1'
PHASES=('untimed','profiled','warmup','timed-1','timed-2','timed-3')
LIVE=(('efficient',0),('math',0))
LOCAL=(('math',0),('math',1))


def task(index,local):
    if type(index) is not int or index not in (0,1):raise ValueError('Exactly two planned worker slots.')
    arm,replay=(LOCAL if local else LIVE)[index]
    return {'case':'encoder-1','arm':arm,'replay':replay,'worker':f'encoder-1-{arm}-{replay}'}


def validate_packet_roots(overlay,packet,scratch):
    overlay,packet,scratch=Path(overlay),Path(packet),Path(scratch)
    if overlay.is_symlink() or packet.is_symlink() or scratch.is_symlink() or overlay.resolve()!=scratch.resolve()/'overlay' or packet.resolve()!=scratch.resolve()/'code':
        raise ValueError('Require original sibling overlay/code/scratch roots.')


def source_check(overlay,packet,c):
    from models import original_attention_pair_operator_v1 as op
    if digest(Path(packet)/'FIXED-ATTENTION-HANDOFF.json')!=op.HANDOFF_SHA:
        raise ValueError('Only the unchanged full original packet is valid for live work.')
    for root,items in [(Path(overlay),{n:{'sha256':s} for n,s in c['remote_source_sha256'].items()}),
            (Path(packet),c['archive_manifest']['files'])]:
        for name,item in items.items():
            path=root/op.base.old.safe_name(name)
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()) or digest(path)!=item['sha256']:
                raise ValueError('Bound source/fixture changed: '+name)


def initialize(scratch,packet,*,local,control_sha,started,deadline):
    scratch,packet=Path(scratch),Path(packet);out=scratch/'results'
    if out.exists():raise FileExistsError('Pair attempt is consumed; never repeat launch.')
    if not math.isfinite(started) or not math.isfinite(deadline) or deadline-started!=(120 if local else 600):
        raise ValueError('Wrong one continuing outer deadline.')
    out.mkdir(mode=0o700);(out/'workers').mkdir(mode=0o700)
    if not local:
        c=json.loads((scratch/'operator-control.json').read_text());manifest=json.loads((packet/'FIXED-ATTENTION-HANDOFF.json').read_text())
        plan={'kind':'fixed-attention-launch-v1','expected_host':'linux10603','role':'encoder-1',
            'handoff_sha256':digest(packet/'FIXED-ATTENTION-HANDOFF.json'),'prior_consumed_attempts':manifest['prior_consumed_attempts'],
            'fixed_inputs':True,'generator_replay_qualified':False,'authorized':True,'attended':True,
            'diagnostic_only':True,'paid_compute':False,'research_admitted':False,
            'expected_gpu':c['expected_gpu'],'expected_gpu_uuid':c['expected_gpu_uuid'],'expected_driver':c['expected_driver'],**manifest['limits'],
            'outer_pair_control_sha256':control_sha,'outer_max_seconds':600,'original_parent_invoked':False,
            'authorization_basis':'New pair exception only; compatibility schema grants no original 1200-second parent run.'}
        (out/'original-schema-plan.json').write_text(json.dumps(plan,sort_keys=True,allow_nan=False)+'\n')
    state={'kind':KIND,'local_tiny_rehearsal':local,'control_sha256':control_sha,'started_unix':started,'deadline_unix':deadline,
        'planned':[task(i,local) for i in (0,1)],'completed':[],'pair_checks_passed':False,'run_finished':False,
        'original_four_worker_session_qualified':False,'performance_qualified':False,'research_admitted':False,
        'paid_compute':False,'old_refusals_waived':False}
    record(out/'pair.json',state);return state


def worker_status(folder,*,local):
    folder=Path(folder);capture=read_record(folder/'capture.json');original=read_record(folder/'worker.json')
    matches=[task(i,local) for i in (0,1) if task(i,local)['worker']==folder.name]
    if len(matches)!=1:raise ValueError('Worker is outside this exact pair.')
    selected=matches[0];device='cpu' if local else 'cuda'
    if (any(original.get(k)!=selected[k] or capture.get(k)!=selected[k] for k in ('case','arm','replay'))
            or original.get('device')!=device or capture.get('device')!=device or original.get('tiny_fixture') is not local
            or original.get('research_admitted') is not False
            or any(capture.get(k) is not False for k in ('original_full_session_qualified','performance_qualified','research_admitted','old_refusals_waived'))):
        raise ValueError('Original worker/capture identity or qualification labels differ.')
    if (capture['kind']!='original-attention-pair-observer-v1' or capture['local_tiny_rehearsal'] is not local
            or capture['outcome']!='original-worker-passed' or not capture['capture_coverage_complete'] or not capture['saved_value_checks_passed']
            or not original['complete'] or not original['passed'] or original['stage']!='complete' or not original['within_process_exact']):
        raise ValueError('Original worker and complete typed capture must both pass before advancing.')
    if any(capture.get(k) is not True for k in ('profile_restored','trace_restored','profile_intact_until_original_exit','alarm_handler_restored')):
        raise ValueError('Original observation/alarm state did not restore.')
    if capture['CPU_oracle_calls']!=1 or capture['CPU_oracle_returns']!=1 or capture['attention_calls']!=6 or capture['SDPA_calls']!=6:
        raise ValueError('Wrong original call coverage.')
    if [r['phase'] for r in capture['attention_returns']]!=list(PHASES):raise ValueError('Missing return phase.')
    if digest(folder/'worker.json')!=capture['original_worker_sha256'] or digest(folder/'saved-audit.json')!=capture['saved_audit_sha256']:
        raise ValueError('Original worker or saved audit changed.')
    audit=read_record(folder/'saved-audit.json')
    if audit['saved_value_checks_passed'] is not True or audit['attention_returns']!=6 or audit['local_tiny_rehearsal'] is not local:
        raise ValueError('Saved worker audit rejected.')
    for item,dtype,limit in [(capture['CPU_return'],'torch.float64',16*2**20),
            *[(x['artifact'],'torch.float32',8*2**20) for x in capture['attention_returns']]]:
        path=folder/item['file']
        if Path(item['file']).name!=item['file'] or path.is_symlink() or not 0<item['bytes']<=limit or path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
            raise ValueError('Typed return artifact changed.')
        if item['dtype_preserved'] is not True or set(item['source_tensors'])!={'output','dq','dk','dv'}:
            raise ValueError('Return dtype/coverage changed.')
        for meta in item['source_tensors'].values():
            expected_device=device if dtype=='torch.float32' else 'cpu'
            if meta['dtype']!=dtype or meta['device'].split(':')[0]!=expected_device:
                raise ValueError('Original return dtype/device differs.')
    for key,name in [('artifact','values.pt')]:
        if digest(folder/name)!=original[key]['sha256']:raise ValueError('Original value snapshot changed.')
    return {'worker':folder.name,'original_worker_sha256':digest(folder/'worker.json'),
        'capture_sha256':digest(folder/'capture.json'),'values_sha256':digest(folder/'values.pt'),
        'saved_audit_sha256':digest(folder/'saved-audit.json'),'CPU_return_sha256':capture['CPU_return']['sha256'],
        'original_worker_passed':True,'capture_coverage_complete':True,'local_tiny_rehearsal':local}


def issue_task(scratch,packet,index,*,local,control_sha):
    scratch,packet=Path(scratch),Path(packet);state=read_record(scratch/'results/pair.json');selected=task(index,local)
    if state['local_tiny_rehearsal'] is not local or state['control_sha256']!=control_sha or state['run_finished'] or len(state['completed'])!=index:
        raise ValueError('Wrong pair sequence/domain or consumed task.')
    reference=None
    if index:
        previous=scratch/'results/workers'/task(0,local)['worker'];status=worker_status(previous,local=local)
        if status!=state['completed'][0]:raise ValueError('Completed fresh reference changed.')
        reference={'worker':previous.name,'worker_sha256':status['original_worker_sha256'],'values_sha256':status['values_sha256']}
    folder=scratch/'results/workers'/selected['worker'];folder.mkdir(mode=0o700)
    issued=time.time();end=min(issued+(30 if local else 120),state['deadline_unix']-5)
    if end<=issued:raise TimeoutError('No original pair allowance remains.')
    auth={'case':'encoder-1','arm':selected['arm'],'replay':selected['replay'],
        'handoff_sha256':digest(packet/'FIXED-ATTENTION-HANDOFF.json'),'deadline_unix':end,'reference':reference}
    record(folder/'authorization.json',auth)
    plan={'kind':'original-attention-pair-worker-plan-v1','index':index,'local_tiny_rehearsal':local,
        'control_sha256':control_sha,'outer_started_unix':state['started_unix'],'outer_deadline_unix':state['deadline_unix'],
        'issued_unix':issued,'original_authorization_sha256':digest(folder/'authorization.json'),
        'original_plan_sha256':None if local else digest(scratch/'results/original-schema-plan.json')}
    record(folder/'worker-plan.json',plan);return folder,plan


def collect_task(scratch,index,*,local):
    scratch=Path(scratch);state=read_record(scratch/'results/pair.json')
    if len(state['completed'])!=index or state['run_finished']:raise ValueError('Task completion order changed.')
    state['completed'].append(worker_status(scratch/'results/workers'/task(index,local)['worker'],local=local))
    record(scratch/'results/pair.json',state,replace=True);return state


def finish(scratch,*,error=None):
    path=Path(scratch)/'results/pair.json';state=read_record(path)
    if state['run_finished']:raise FileExistsError('Finished pair receipt cannot be overwritten.')
    state.update(run_finished=True,pair_checks_passed=error is None and len(state['completed'])==2,
        wall_seconds=time.time()-state['started_unix'])
    if error:state['error']=error
    record(path,state,replace=True);return state


def dispatch(args,overlay):
    from models import original_attention_pair_operator_v1 as op
    packet,scratch=Path(args.packet),Path(args.scratch);validate_packet_roots(overlay,packet,scratch)
    manifest=json.loads((packet/'FIXED-ATTENTION-HANDOFF.json').read_text());local=args.local_tiny_rehearsal
    if local:
        if platform.node() in ('linux10603','linux10609') or manifest['tiny_fixture'] is not True or args.control_sha256:
            raise ValueError('Explicit tiny CPU rehearsal is local only and accepts no live grant.')
    else:
        if not args.control_sha256 or digest(scratch/'operator-control.json')!=args.control_sha256:raise ValueError('Missing/changed new pair control.')
        c=json.loads((scratch/'operator-control.json').read_text());op.validate_control(c)
        op.validate_authorization(json.loads((scratch/'resource-authorization.json').read_text()),args.control_sha256)
        source_check(overlay,packet,c)
    selected=task(args.task_index,local);folder=scratch/'results/workers'/selected['worker']
    if digest(folder/'worker-plan.json')!=args.worker_plan_sha256:raise ValueError('Worker dispatch plan changed.')
    plan=read_record(folder/'worker-plan.json');auth=read_record(folder/'authorization.json')
    if (plan['kind']!='original-attention-pair-worker-plan-v1' or plan['index']!=args.task_index or plan['local_tiny_rehearsal'] is not local
            or plan['control_sha256']!=args.control_sha256 or digest(folder/'authorization.json')!=plan['original_authorization_sha256']
            or not 0<=time.time()-plan['issued_unix']<=120 or plan['outer_deadline_unix']-plan['outer_started_unix']!=(120 if local else 600)
            or auth['deadline_unix']>plan['outer_deadline_unix']-5):raise ValueError('Changed worker domain, authorization or original clock.')
    if args.task_index:
        baseline=scratch/'results/workers'/task(0,local)['worker'];status=worker_status(baseline,local=local)
        expected={'worker':baseline.name,'worker_sha256':status['original_worker_sha256'],'values_sha256':status['values_sha256']}
        if auth['reference']!=expected:raise ValueError('Candidate requires its fresh passing reference.')
    elif auth['reference'] is not None:raise ValueError('The first original worker has no substituted reference.')
    sys.path.insert(1,str(packet.resolve()))
    from models import fixed_attention_diagnostic as original
    from models.original_attention_pair_observer_v1 import observe_worker
    original_plan=None if local else scratch/'results/original-schema-plan.json'
    if not local and digest(original_plan)!=plan['original_plan_sha256']:raise ValueError('Original compatibility plan changed.')
    original_args=SimpleNamespace(worker_dir=folder,authorization_sha256=plan['original_authorization_sha256'],
        case='encoder-1',arm=selected['arm'],replay=selected['replay'],handoff_sha256=digest(packet/'FIXED-ATTENTION-HANDOFF.json'),
        tiny=local,device='cpu' if local else 'cuda',upstream=args.upstream,role='encoder-1',plan=original_plan,plan_sha256=plan['original_plan_sha256'])
    result=observe_worker(original,original_args,packet,folder,local=local)
    print('Original worker:',selected['worker'],'capture:',result['outcome'],flush=True)
    return 0 if result['outcome']=='original-worker-passed' else 1


def local_pair(args,overlay):
    packet,scratch=Path(args.packet),Path(args.scratch);validate_packet_roots(overlay,packet,scratch)
    manifest=json.loads((packet/'FIXED-ATTENTION-HANDOFF.json').read_text())
    if platform.node() in ('linux10603','linux10609') or manifest['tiny_fixture'] is not True or args.control_sha256:
        raise ValueError('The tiny math pair is local only; it cannot qualify efficient CUDA.')
    started=time.time();initialize(scratch,packet,local=True,control_sha=None,started=started,deadline=started+120)
    error=None
    try:
        for index in (0,1):
            folder,plan=issue_task(scratch,packet,index,local=True,control_sha=None)
            command=[sys.executable,'-I','-B',str(Path(overlay)/'scripts/capture_original_attention_pair_v1.py'),
                '--packet',str(packet),'--scratch',str(scratch),'--upstream',str(args.upstream),'--local-tiny-rehearsal',
                '--task-index',str(index),'--worker-plan-sha256',digest(folder/'worker-plan.json')]
            with (folder/'worker.log').open('xb') as log:
                child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(.01,read_record(folder/'authorization.json')['deadline_unix']-time.time()))
            if child.returncode:raise RuntimeError('Original tiny worker/capture rejected; no next worker or retry.')
            collect_task(scratch,index,local=True)
    except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)}
    state=finish(scratch,error=error);print('Local CPU math pair passed:',state['pair_checks_passed'],flush=True)
    return 0 if state['pair_checks_passed'] else 1
