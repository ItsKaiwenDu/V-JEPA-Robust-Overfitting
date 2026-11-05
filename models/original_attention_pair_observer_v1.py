"""Observe original returns without replacing its calculations or acceptance gates."""
import ast
import json
from pathlib import Path
import platform
import signal
import sys
import time

from models.oracle_boundary_observer_v1 import bind_function,record,read_record,digest,detailed_context

KIND='original-attention-pair-observer-v1'
PHASES=('untimed','profiled','warmup','timed-1','timed-2','timed-3')
OUTPUTS=('output','dq','dk','dv')


class Returns:
    def __init__(self,original,args,packet,folder):
        from models import attention_component
        if any(getattr(original,n) is not getattr(attention_component,n) for n in ('fp64_oracle','iteration','clone_layout')):
            raise ValueError('An original function alias was replaced.')
        self.identities={n:bind_function(getattr(original,n),packet,json.loads((Path(packet)/'FIXED-ATTENTION-HANDOFF.json').read_text()),n)
            for n in ('worker','fp64_oracle','iteration','clone_layout')}
        tree=ast.parse((Path(packet)/'models/fixed_attention_diagnostic.py').read_text())
        worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        self.lines=sorted(n.lineno for n in ast.walk(worker) if isinstance(n,ast.Call) and ast.unparse(n.func)=='iteration')
        if len(self.lines)!=3:raise ValueError('Original iteration dispatch sites differ.')
        self.original=original;self.args=args;self.folder=Path(folder);self.worker=None;self.active=None
        self.cpu_calls=0;self.cpu_returns=0;self.calls=0;self.sdpa=0;self.entries=[];self.cpu=None;self.events=[];self.error=None

    def callback(self,frame,event,arg):
        try:self.handle(frame,event,arg)
        except BaseException as exc:
            self.error={'type':type(exc).__name__,'message':str(exc)};raise

    def handle(self,frame,event,arg):
        original=self.original
        if event=='call' and frame.f_code is original.worker.__code__:
            if self.worker is not None:raise ValueError('Exactly one original worker per process.')
            self.worker=frame
        if frame.f_code is original.fp64_oracle.__code__:
            if event=='call':
                self.cpu_calls+=1
                if self.cpu_calls!=1 or frame.f_back is not self.worker:raise ValueError('Unexpected original CPU oracle call.')
                self.events.append('original_CPU_oracle_call')
            elif event=='return' and arg is not None:
                self.cpu_returns+=1
                if self.cpu_returns!=1:raise ValueError('Unexpected CPU return count.')
                from models.attention_return_capture_v1 import save_return
                self.cpu=save_return(self.folder/'CPU-return.npz',arg,dtype='float64',device='cpu',
                    shape=list(arg['output'].shape),limit=16*2**20)
                self.events.append('CPU_return_saved_before_original_gate')
        if event=='c_call' and getattr(arg,'__name__','')=='scaled_dot_product_attention':
            if frame is not self.active:raise ValueError('SDPA outside the original iteration is not allowed.')
            self.sdpa+=1
            if self.sdpa!=self.calls:raise ValueError('Each original iteration permits one SDPA call.')
        if frame.f_code is original.iteration.__code__:
            if event=='call':
                if self.active is not None or frame.f_back is not self.worker or self.calls>=6 or self.cpu_returns!=1:
                    raise ValueError('Unexpected original attention iteration or missing CPU return.')
                gate=self.worker.f_locals.get('result',{}).get('oracle_bookkeeping',{})
                if set(gate)!=set(OUTPUTS) or not all(x.get('passed') is True for x in gate.values()):
                    raise ValueError('GPU transfer/iteration cannot bypass the original CPU gate.')
                line=self.lines[min(self.calls,2)]
                if self.worker.f_lineno!=line or self.calls>=2 and self.worker.f_locals.get('i')!=self.calls-2:
                    raise ValueError('Original attention phase dispatch differs.')
                self.calls+=1;self.active=frame;self.events.append('attention_call_'+PHASES[self.calls-1])
            elif event=='return':
                self.active=None
                if arg is None:return  # Preserve the original exception while unwinding.
                if not isinstance(arg,tuple) or len(arg)!=2 or self.sdpa!=self.calls:raise ValueError('Malformed original iteration return.')
                actual,timing=arg;phase=PHASES[self.calls-1]
                from models.attention_return_capture_v1 import save_return
                item=save_return(self.folder/('return-'+str(self.calls-1)+'-'+phase+'.npz'),actual,
                    dtype='float32',device=self.args.device,shape=list(actual['output'].shape),limit=8*2**20)
                self.entries.append({'phase':phase,'artifact':item,'original_timing_observation':dict(timing)})
                self.events.append('attention_return_saved_'+phase)


def observe_worker(original,args,packet,folder,*,local):
    packet,folder=Path(packet),Path(folder);manifest=json.loads((packet/'FIXED-ATTENTION-HANDOFF.json').read_text())
    if args.tiny is not local or manifest['tiny_fixture'] is not local or args.device!=('cpu' if local else 'cuda') or args.case!='encoder-1':
        raise ValueError('Pair fixture/device domain differs.')
    if local and (platform.node() in ('linux10603','linux10609') or args.arm!='math' or args.replay not in (0,1)):
        raise ValueError('Tiny math rehearsal is local only.')
    if not local and (platform.node()!='linux10603' or args.arm not in ('efficient','math') or args.replay!=0):
        raise ValueError('Live pair is encoder-1 replay 0 on linux10603 only.')
    if sys.getprofile() is not None or sys.gettrace() is not None:raise ValueError('Refuse to replace an existing observation hook.')
    path=folder/'capture.json'
    if path.exists() or (folder/'worker.json').exists() or list(folder.glob('*.npz*')):raise FileExistsError('Preserve all prior captures; no repeat worker.')
    observer=Returns(original,args,packet,folder)
    state={'kind':KIND,'local_tiny_rehearsal':local,'case':args.case,'arm':args.arm,'replay':args.replay,'device':args.device,
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
            from models.attention_return_capture_v1 import audit_worker
            audit=audit_worker(folder,packet,local=local);record(folder/'saved-audit.json',audit)
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
