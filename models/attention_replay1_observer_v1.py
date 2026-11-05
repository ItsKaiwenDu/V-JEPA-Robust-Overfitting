"""Separate replay-1 scope, reusing only the frozen return-observation primitive."""
import json
from pathlib import Path
import platform
import signal
import sys
import time
from models.oracle_boundary_observer_v1 import record,read_record,digest,detailed_context
from models.original_attention_pair_observer_v1 import Returns
from models import attention_replay1_contract_v1 as contract
KIND='attention-replay1-observer-v1'


def observe_worker(original,args,packet,folder,*,local):
    packet,folder=Path(packet),Path(folder);manifest=json.loads((packet/'FIXED-ATTENTION-HANDOFF.json').read_text())
    if args.tiny is not local or manifest['tiny_fixture'] is not local or args.device!=('cpu' if local else 'cuda') or args.case!='encoder-1':
        raise ValueError('Replay-1 fixture/device domain differs.')
    if local and (platform.node() in ('linux10603','linux10609') or args.arm!='math' or args.replay!=1):
        raise ValueError('Tiny math rehearsal is local only.')
    if not local and (platform.node()!='linux10603' or args.arm not in ('efficient','math') or args.replay!=1):
        raise ValueError('Live replay-1 workers are encoder-1 replay 1 on linux10603 only.')
    if sys.getprofile() is not None or sys.gettrace() is not None:raise ValueError('Refuse to replace an existing observation hook.')
    path=folder/'capture.json'
    if path.exists() or (folder/'worker.json').exists() or list(folder.glob('*.npz*')):raise FileExistsError('Preserve all prior captures; no repeat worker.')
    if folder.name not in {contract.task(i,local)['worker'] for i in (0,1)}:raise ValueError('Unknown new worker slot.')
    observer=Returns(original,args,packet,folder)
    state={'kind':KIND,'local_tiny_rehearsal':local,'local_tiny_analogue':local,'case':args.case,'arm':args.arm,'replay':args.replay,'device':args.device,
        'process_id':__import__('os').getpid(),'parent_process_id':__import__('os').getppid(),
        'capture_coverage_complete':False,'saved_value_checks_passed':False,'outcome':'starting',
        'original_full_session_qualified':False,'performance_qualified':False,'research_admitted':False,
        'old_refusals_waived':False,'detailed_introspection_before_first_oracle':False,
        'function_identities':observer.identities}
    record(path,state);started=time.monotonic();previous_profile=sys.getprofile();previous_trace=sys.gettrace()
    previous_alarm=signal.getsignal(signal.SIGALRM);previous_timer=signal.getitimer(signal.ITIMER_REAL)
    installed=observer.callback;failure=None;hook_intact=False
    try:
        sys.setprofile(installed)
        try:original.worker(args,packet)
        except BaseException as exc:failure={'type':type(exc).__name__,'message':str(exc)}
        hook_intact=sys.getprofile() is installed
    finally:
        sys.setprofile(previous_profile);signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_alarm)
        if previous_timer[0]>0:signal.setitimer(signal.ITIMER_REAL,max(.000001,previous_timer[0]-(time.monotonic()-started)),previous_timer[1])
        elif previous_timer[1]>0:signal.setitimer(signal.ITIMER_REAL,0,previous_timer[1])
    state.update(CPU_oracle_calls=observer.cpu_calls,CPU_oracle_returns=observer.cpu_returns,CPU_return=observer.cpu,
        attention_calls=observer.calls,attention_returns=observer.entries,SDPA_calls=observer.sdpa,events=observer.events,
        observer_error=observer.error,original_exception=failure,profile_restored=sys.getprofile() is previous_profile,
        trace_restored=sys.gettrace() is previous_trace,profile_intact_until_original_exit=hook_intact,
        alarm_handler_restored=signal.getsignal(signal.SIGALRM) is previous_alarm)
    try:
        old=read_record(folder/'worker.json');state.update(original_worker_sha256=digest(folder/'worker.json'),
            original_worker_complete=old['complete'],original_worker_passed=old['passed'])
        if failure is not None:
            state['outcome']='technical-failure' if observer.error else 'original-worker-rejected'
            if old.get('error_type')!=failure['type'] or old.get('error')!=failure['message']:
                raise ValueError('Original exception record differs; preserve evidence.')
        elif (not hook_intact or observer.error or not old['complete'] or not old['passed']
                or observer.cpu_calls!=observer.cpu_returns or observer.cpu_calls!=1
                or observer.calls!=len(observer.entries) or observer.calls!=6 or observer.sdpa!=6):
            raise ValueError('Original pass and complete return coverage are both required.')
        else:
            state['capture_coverage_complete']=True
            record(path,state,replace=True)
            from models.attention_replay1_cohort_v1 import audit_new_worker
            audit=audit_new_worker(folder,packet,local=local);record(folder/'saved-audit.json',audit)
            state.update(saved_value_checks_passed=True,saved_audit_sha256=digest(folder/'saved-audit.json'),outcome='original-worker-passed')
        state['events'].append('detailed_context_after_original_worker_exit')
        state['post_worker_context']=detailed_context(original)
        state['post_worker_path_environment']={n:__import__('os').environ.get(n) for n in ('TMPDIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','CUDA_CACHE_MAXSIZE')}
        if not local and any(state['post_worker_context'][k]!=v for k,v in (
                ('python','3.11.17'),('torch','2.8.0+cu128'),('numpy','2.4.6'))):
            raise ValueError('The pinned live Python/PyTorch/NumPy environment differs; no fallback.')
    except BaseException as exc:
        state.update(saved_value_checks_passed=False,outcome='technical-failure',review_error={'type':type(exc).__name__,'message':str(exc)})
    state['wall_seconds']=time.monotonic()-started;record(path,state,replace=True)
    return state
