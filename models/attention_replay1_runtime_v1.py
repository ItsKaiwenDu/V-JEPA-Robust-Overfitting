"""Issue only two new workers and one verifier; imported tasks stay separate."""
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

from models import attention_replay1_contract_v1 as refs
from models.oracle_boundary_observer_v1 import record,read_record,digest

KIND='attention-replay1-runtime-v1'
FLAGS={'original_parent_session_qualified':False,'original_four_worker_session_qualified':False,
    'performance_qualified':False,'research_admitted':False,'old_refusals_waived':False,'paid_compute':False}


def initialize(scratch,packet,manifest,*,local,control_sha,started):
    scratch,packet=Path(scratch),Path(packet);m=json.loads((packet/'FIXED-ATTENTION-HANDOFF.json').read_text())
    if (m['tiny_fixture'] is not local or local and (platform.node() in ('linux10603','linux10609') or control_sha is not None)):
        raise ValueError('Tiny CPU analogues cannot use live packets, hosts or grants.')
    refs.validate_manifest(manifest,local=local);out=scratch/'results'
    if out.exists():raise FileExistsError('Never reuse an attempt results directory.')
    if not 0<=time.time()-started<=110:raise ValueError('Initial setup clock is not fresh.')
    out.mkdir(mode=0o700);(out/'workers').mkdir(mode=0o700)
    if not local:
        c=json.loads((scratch/'operator-control.json').read_text())
        plan={'kind':'fixed-attention-launch-v1','expected_host':'linux10603','role':'encoder-1',
            'handoff_sha256':digest(packet/'FIXED-ATTENTION-HANDOFF.json'),'prior_consumed_attempts':m['prior_consumed_attempts'],
            'fixed_inputs':True,'generator_replay_qualified':False,'authorized':True,'attended':True,
            'diagnostic_only':True,'paid_compute':False,'research_admitted':False,'expected_gpu':c['expected_gpu'],
            'expected_gpu_uuid':c['expected_gpu_uuid'],'expected_driver':c['expected_driver'],**m['limits'],
            'outer_replay1_control_sha256':control_sha,'outer_max_seconds':600,'original_parent_invoked':False,
            'authorization_basis':'New replay1 exception only; no old parent/session or expired grant is resumed.'}
        (out/'original-schema-plan.json').write_text(json.dumps(plan,sort_keys=True,allow_nan=False)+'\n')
    state={'kind':KIND,'local_tiny_analogue':local,'control_sha256':control_sha,'started_unix':started,
        'deadline_unix':started+(120 if local else 600),'imported_workers':list(refs.imported(local)),
        'new_tasks':[refs.task(i,local) for i in (0,1)],'new_workers_completed':[],'verifier_completed':False,
        'cross_attempt_component_cohort_checks_passed':False,'run_finished':False,**FLAGS}
    record(out/'replay1.json',state);return state


def source_check(overlay,packet,c):
    from models import attention_replay1_operator_v1 as op
    if digest(Path(packet)/'FIXED-ATTENTION-HANDOFF.json')!=op.HANDOFF_SHA:raise ValueError('Require the unchanged original full packet.')
    for root,items in [(Path(overlay),{n:{'sha256':s} for n,s in c['remote_source_sha256'].items()}),
            (Path(packet),c['archive_manifest']['files'])]:
        for name,item in items.items():
            p=root/refs.safe(name)
            if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or not p.is_file() or digest(p)!=item['sha256']:
                raise ValueError('Bound source or fixture changed: '+name)


def worker_status(folder,*,local):
    folder=Path(folder);selected=next((refs.task(i,local) for i in (0,1) if refs.task(i,local)['worker']==folder.name),None)
    if selected is None:raise ValueError('Unknown new worker.')
    cap=read_record(folder/'capture.json');w=read_record(folder/'worker.json')
    if (cap['kind']!='attention-replay1-observer-v1' or cap['local_tiny_rehearsal'] is not local
            or cap['outcome']!='original-worker-passed' or not cap['capture_coverage_complete'] or not cap['saved_value_checks_passed']
            or not w['complete'] or not w['passed'] or w['stage']!='complete' or not w['within_process_exact']
            or any(cap.get(k) is not False for k in ('original_full_session_qualified','performance_qualified','research_admitted','old_refusals_waived'))
            or w['tiny_fixture'] is not local or w['device']!=('cpu' if local else 'cuda')
            or any(w[k]!=selected[k] or cap[k]!=selected[k] for k in ('case','arm','replay'))):raise ValueError('New worker or typed capture is incomplete.')
    if any(cap.get(k) is not True for k in ('profile_restored','trace_restored','profile_intact_until_original_exit','alarm_handler_restored')):
        raise ValueError('Observer/remaining alarm not restored.')
    if cap['CPU_oracle_calls']!=cap['CPU_oracle_returns'] or cap['CPU_oracle_returns']!=1 or cap['attention_calls']!=cap['SDPA_calls'] or cap['SDPA_calls']!=6:
        raise ValueError('Wrong original call coverage.')
    if [x['phase'] for x in cap['attention_returns']]!=list(refs.PHASES):raise ValueError('Missing original return phase.')
    if digest(folder/'worker.json')!=cap['original_worker_sha256'] or digest(folder/'saved-audit.json')!=cap['saved_audit_sha256']:
        raise ValueError('New worker/audit record changed.')
    audit=read_record(folder/'saved-audit.json')
    if audit['kind']!='attention-replay1-cohort-science-v1' or not audit['saved_value_checks_passed'] or audit['local_tiny_analogue'] is not local:
        raise ValueError('Saved scientific audit rejected.')
    for item,dtype,device,limit in [(cap['CPU_return'],'torch.float64','cpu',16*2**20),
            *[(x['artifact'],'torch.float32','cpu' if local else 'cuda',8*2**20) for x in cap['attention_returns']]]:
        p=folder/item['file']
        if (Path(item['file']).name!=item['file'] or p.is_symlink() or not 0<item['bytes']<=limit
                or p.stat().st_size!=item['bytes'] or digest(p)!=item['sha256'] or item['dtype_preserved'] is not True):raise ValueError('Typed return changed.')
        if set(item['source_tensors'])!={'output','dq','dk','dv'} or any(v['dtype']!=dtype or v['device'].split(':')[0]!=device for v in item['source_tensors'].values()):
            raise ValueError('Actual return dtype/device differs.')
    if digest(folder/'values.pt')!=w['artifact']['sha256']:raise ValueError('Original values snapshot changed.')
    return {'worker':folder.name,'worker_sha256':digest(folder/'worker.json'),'capture_sha256':digest(folder/'capture.json'),
        'values_sha256':digest(folder/'values.pt'),'saved_audit_sha256':digest(folder/'saved-audit.json'),
        'local_tiny_analogue':local,'original_worker_and_capture_passed':True}


def issue(scratch,packet,manifest,phase,*,local,control_sha,index=None):
    scratch,packet=Path(scratch),Path(packet);state=read_record(scratch/'results/replay1.json')
    if state['run_finished'] or state['local_tiny_analogue'] is not local or state['control_sha256']!=control_sha:
        raise ValueError('Wrong or consumed attempt.')
    refs.verify_imports(scratch/'results',manifest,local=local)
    if phase=='worker':
        selected=refs.task(index,local)
        if len(state['new_workers_completed'])!=index:raise ValueError('New worker order changed.')
        previous=scratch/'results/workers'/(refs.imported(local)[0] if index==0 else refs.task(0,local)['worker'])
        if index and worker_status(previous,local=local)!=state['new_workers_completed'][0]:raise ValueError('Fresh reference changed.')
        folder=scratch/'results/workers'/selected['worker'];folder.mkdir(mode=0o700)
        issued=time.time();end=min(issued+(30 if local else 120),state['deadline_unix']-5)
        auth={'case':'encoder-1','arm':selected['arm'],'replay':1,'handoff_sha256':digest(packet/'FIXED-ATTENTION-HANDOFF.json'),
            'deadline_unix':end,'reference':{'worker':previous.name,'worker_sha256':digest(previous/'worker.json'),'values_sha256':digest(previous/'values.pt')}}
        if end<=issued:raise TimeoutError('No remaining worker allowance.')
        record(folder/'authorization.json',auth);path=folder/'worker-plan.json'
    elif phase=='cohort':
        if index is not None or len(state['new_workers_completed'])!=2 or state['verifier_completed']:raise ValueError('Verifier requires both new accepted workers exactly once.')
        for i,prior in enumerate(state['new_workers_completed']):
            if worker_status(scratch/'results/workers'/refs.task(i,local)['worker'],local=local)!=prior:raise ValueError('New worker evidence changed.')
        folder=scratch/'results';issued=time.time();end=min(issued+(30 if local else 90),state['deadline_unix']-5)
        if end<=issued:raise TimeoutError('No remaining verifier allowance.')
        path=folder/'verifier-plan.json';auth=None
    else:raise ValueError('Only a planned new worker or one CPU verifier is supported.')
    plan={'kind':'attention-replay1-phase-plan-v1','phase':phase,'index':index,'local_tiny_analogue':local,
        'control_sha256':control_sha,'issued_unix':issued,'deadline_unix':end,'outer_started_unix':state['started_unix'],
        'outer_deadline_unix':state['deadline_unix'],'original_authorization_sha256':None if auth is None else digest(folder/'authorization.json'),
        'original_plan_sha256':None if local else digest(scratch/'results/original-schema-plan.json')}
    record(path,plan);return folder,plan,path


def collect_worker(scratch,index,*,local):
    scratch=Path(scratch);state=read_record(scratch/'results/replay1.json')
    if state['run_finished'] or len(state['new_workers_completed'])!=index:raise ValueError('Wrong completion order.')
    state['new_workers_completed'].append(worker_status(scratch/'results/workers'/refs.task(index,local)['worker'],local=local))
    record(scratch/'results/replay1.json',state,replace=True)


def verifier_status(scratch,*,local):
    folder=Path(scratch)/'results/cohort-verifier';v=read_record(folder/'verifier.json')
    if (v['kind']!='attention-replay1-cohort-science-v1' or v['local_tiny_analogue'] is not local or not v['complete'] or not v['passed']
            or v['NumPy_oracle_calls']!=v['NumPy_oracle_returns'] or v['NumPy_oracle_returns']!=1
            or v['CUDA_initialized_before'] or v['CUDA_initialized_after'] or not v['profile_restored'] or not v['trace_restored']
            or not v['return_saved_before_gate'] or not 0<v['wall_seconds']<=(30 if local else 90)):
        raise ValueError('CPU-only verifier is incomplete or out of scope.')
    if (v['phase_plan_sha256']!=digest(Path(scratch)/'results/verifier-plan.json')
            or v['control_sha256']!=read_record(Path(scratch)/'results/replay1.json')['control_sha256']):
        raise ValueError('CPU verifier plan changed.')
    for key,value in FLAGS.items():
        if key in v and v[key] is not value:raise ValueError('CPU verifier qualification scope changed.')
    item=v['NumPy_return'];p=folder/item['file']
    if (Path(item['file']).name!=item['file'] or p.is_symlink() or p.stat().st_size!=item['bytes'] or not 0<item['bytes']<=16*2**20
            or digest(p)!=item['sha256'] or item['source_dtype']!='float64' or item['dtype_preserved'] is not True
            or digest(folder/'saved-cohort-audit.json')!=v['saved_audit_sha256']):raise ValueError('NumPy reconstruction evidence changed.')
    audit=read_record(folder/'saved-cohort-audit.json')
    if (not audit['cross_attempt_component_cohort_checks_passed'] or not audit['old_subset_attestation_only']
            or audit['old_captures_directly_reread'] or audit['local_tiny_analogue'] is not local):raise ValueError('Wrong cohort evidence/provenance scope.')
    return {'verifier_sha256':digest(folder/'verifier.json'),'saved_audit_sha256':digest(folder/'saved-cohort-audit.json'),
        'NumPy_return_sha256':item['sha256'],'CPU_only_verifier_passed':True,'old_full_captures_must_be_reverified_locally':True}


def finish(scratch,*,local,error=None):
    p=Path(scratch)/'results/replay1.json';state=read_record(p)
    if state['run_finished']:raise FileExistsError('Preserve the finished receipt.')
    if error is None:
        if not 0<=time.time()-state['started_unix']<=state['deadline_unix']-state['started_unix']:raise TimeoutError('Outer work allowance expired.')
        if len(state['new_workers_completed'])!=2:raise ValueError('Cannot finish an incomplete cohort.')
        state['verifier']=verifier_status(scratch,local=local);state['verifier_completed']=True
    state.update(run_finished=True,cross_attempt_component_cohort_checks_passed=error is None,wall_seconds=time.time()-state['started_unix'])
    if error:state['error']=error
    record(p,state,replace=True);return state


def dispatch(args,overlay):
    from models import attention_replay1_operator_v1 as op
    scratch,packet,overlay=Path(args.scratch),Path(args.packet),Path(overlay);local=args.local_tiny_analogue
    if (scratch.is_symlink() or packet.is_symlink() or overlay.is_symlink() or packet.resolve()!=scratch.resolve()/'code'
            or overlay.resolve()!=scratch.resolve()/'overlay'):raise ValueError('Require exact sibling scratch/code/overlay roots.')
    metadata=json.loads((packet/'FIXED-ATTENTION-HANDOFF.json').read_text())
    if local:
        if platform.node() in ('linux10603','linux10609') or metadata['tiny_fixture'] is not True or args.control_sha256:
            raise ValueError('Explicit tiny analogue only; no live host, packet or authorization.')
        manifest=json.loads((scratch/'local-reference-manifest.json').read_text())
    else:
        if digest(scratch/'operator-control.json')!=args.control_sha256:raise ValueError('New replay1 control is required.')
        c=json.loads((scratch/'operator-control.json').read_text());op.validate_control(c)
        op.validate_authorization(json.loads((scratch/'resource-authorization.json').read_text()),args.control_sha256)
        source_check(overlay,packet,c);manifest=c['reference_manifest']
    refs.verify_imports(scratch/'results',manifest,local=local)
    if args.phase=='worker':
        selected=refs.task(args.task_index,local);folder=scratch/'results/workers'/selected['worker'];path=folder/'worker-plan.json'
    elif args.phase=='cohort':
        if args.task_index is not None:raise ValueError('CPU verifier has no GPU worker index.')
        folder=scratch/'results';path=folder/'verifier-plan.json'
    else:raise ValueError('Unknown bound scientific phase.')
    if digest(path)!=args.worker_plan_sha256:raise ValueError('New phase plan changed.')
    plan=read_record(path);state=read_record(scratch/'results/replay1.json');seconds=30 if local else (120 if args.phase=='worker' else 90)
    if (plan['kind']!='attention-replay1-phase-plan-v1' or plan['phase']!=args.phase or plan['index']!=args.task_index
            or plan['local_tiny_analogue'] is not local or plan['control_sha256']!=args.control_sha256 or state['run_finished']
            or plan['outer_started_unix']!=state['started_unix'] or plan['outer_deadline_unix']!=state['deadline_unix']
            or state['deadline_unix']-state['started_unix']!=(120 if local else 600)
            or not 0<=time.time()-plan['issued_unix']<=seconds or not 0<plan['deadline_unix']-time.time()<=seconds
            or plan['deadline_unix']>state['deadline_unix']-5):raise ValueError('Changed phase, domain or continuing clock.')
    if args.phase=='worker':
        if len(state['new_workers_completed'])!=args.task_index:raise ValueError('Worker ordering changed.')
        auth=read_record(folder/'authorization.json')
        previous=scratch/'results/workers'/(refs.imported(local)[0] if args.task_index==0 else refs.task(0,local)['worker'])
        expected={'worker':previous.name,'worker_sha256':digest(previous/'worker.json'),'values_sha256':digest(previous/'values.pt')}
        if args.task_index and worker_status(previous,local=local)!=state['new_workers_completed'][0]:raise ValueError('Fresh reference changed.')
        if digest(folder/'authorization.json')!=plan['original_authorization_sha256'] or auth['reference']!=expected or auth['deadline_unix']!=plan['deadline_unix']:
            raise ValueError('Original reference or worker authorization changed.')
    elif len(state['new_workers_completed'])!=2 or state['verifier_completed']:raise ValueError('Verifier cannot start before both workers or repeat.')
    sys.path.insert(1,str(packet.resolve()))
    if args.phase=='cohort':
        import signal
        signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,plan['deadline_unix']-time.time())
        from models.attention_replay1_cohort_v1 import verify_once
        return verify_once(scratch/'results',packet,local=local,plan_sha=args.worker_plan_sha256,control_sha=args.control_sha256)
    from models import fixed_attention_diagnostic as original
    from models.attention_replay1_observer_v1 import observe_worker
    oldplan=None if local else scratch/'results/original-schema-plan.json'
    if not local and digest(oldplan)!=plan['original_plan_sha256']:raise ValueError('Original compatibility plan changed.')
    original_args=SimpleNamespace(worker_dir=folder,authorization_sha256=plan['original_authorization_sha256'],case='encoder-1',
        arm=selected['arm'],replay=1,handoff_sha256=digest(packet/'FIXED-ATTENTION-HANDOFF.json'),tiny=local,
        device='cpu' if local else 'cuda',upstream=args.upstream,role='encoder-1',plan=oldplan,plan_sha256=plan['original_plan_sha256'])
    captured=observe_worker(original,original_args,packet,folder,local=local)
    return 0 if captured['outcome']=='original-worker-passed' else 1


def local_analogue(args,overlay):
    scratch,packet,overlay=Path(args.scratch),Path(args.packet),Path(overlay)
    if (scratch.is_symlink() or packet.is_symlink() or overlay.is_symlink() or packet.resolve()!=scratch.resolve()/'code'
            or overlay.resolve()!=scratch.resolve()/'overlay'):raise ValueError('Require exact sibling roots.')
    manifest=json.loads(Path(args.reference_manifest).read_text());spec=json.loads(Path(args.reference_spec).read_text())
    started=time.time();initialize(scratch,packet,manifest,local=True,control_sha=None,started=started)
    (scratch/'local-reference-manifest.json').write_bytes(refs.canonical(manifest))
    receipt=refs.extract_archive(args.reference_archive,spec,manifest,scratch/'results',local=True);record(scratch/'results/reference-import.json',receipt)
    error=None
    try:
        for phase,index in (('worker',0),('worker',1),('cohort',None)):
            folder,plan,path=issue(scratch,packet,manifest,phase,local=True,control_sha=None,index=index)
            command=[sys.executable,'-I','-B',str(Path(overlay)/'scripts/capture_attention_replay1_v1.py'),
                '--packet',str(packet),'--scratch',str(scratch),'--upstream',str(args.upstream),'--local-tiny-analogue',
                '--phase',phase,'--worker-plan-sha256',digest(path)]
            if index is not None:command+=['--task-index',str(index)]
            with (scratch/('local-'+phase+'-'+str(index)+'.log')).open('xb') as log:
                done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                    timeout=max(.01,plan['deadline_unix']-time.time()),env={**__import__('os').environ,'CUDA_VISIBLE_DEVICES':''})
            if done.returncode:raise RuntimeError('Tiny CPU analogue refused; no next phase or retry.')
            if phase=='worker':collect_worker(scratch,index,local=True)
        refs.verify_imports(scratch/'results',manifest,local=True)
    except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)}
    state=finish(scratch,local=True,error=error);print('Tiny CPU analogues passed:',state['cross_attempt_component_cohort_checks_passed'],flush=True)
    return 0 if state['cross_attempt_component_cohort_checks_passed'] else 1
