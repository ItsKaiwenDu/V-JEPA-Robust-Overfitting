"""Observe an unchanged worker's actual CPU return and stop before transfer.

Importing this module imports only the standard library. Capture helpers and
detailed runtime inspection are deferred until the original computation returns.
"""
import ast
import contextlib
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import signal
import sys
import time
import types

KIND='oracle-boundary-observer-v1'
OUTPUTS=('output','dq','dk','dv')
BOOKKEEPING={'rtol':1e-12,'atol':1e-14}


class BoundaryReached(BaseException):
    """Intentional scope end; the original full GPU worker remains incomplete."""


def record(path,payload,*,replace=False):
    path=Path(path)
    if path.exists() and not replace:raise FileExistsError('Preserve existing boundary receipt.')
    canonical=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    data=json.dumps({'payload':payload,'sha256':hashlib.sha256(canonical).hexdigest()},sort_keys=True,allow_nan=False).encode()+b'\n'
    part=path.with_name(path.name+'.partial')
    with part.open('xb') as f:
        f.write(data);f.flush();__import__('os').fsync(f.fileno())
    if replace:part.replace(path)
    else:
        __import__('os').link(part,path);part.unlink()


def read_record(path):
    value=json.loads(Path(path).read_text())
    canonical=json.dumps(value['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest()!=value['sha256']:raise ValueError('Boundary record checksum differs.')
    return value['payload']


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def bind_function(function,packet,manifest,name):
    """Match loaded bytecode as well as the immutable source file."""
    path=Path(inspect.getsourcefile(function)).resolve();packet=Path(packet).resolve()
    if not path.is_relative_to(packet):raise ValueError('Original function is outside the packet: '+name)
    relative=str(path.relative_to(packet));item=manifest['files'].get(relative)
    if not item or path.is_symlink() or digest(path)!=item['sha256']:raise ValueError('Original function source differs: '+name)
    compiled=compile(path.read_text(),str(path),'exec',dont_inherit=True)
    matches=[c for c in compiled.co_consts if isinstance(c,types.CodeType) and c.co_name==name]
    if len(matches)!=1 or function.__code__!=matches[0]:raise ValueError('Loaded original function changed: '+name)
    return {'source':relative,'sha256':item['sha256'],'function':name,'bytecode_matches_bound_source':True}


def detailed_context(original):
    """Called only after the original gate/stop, never before the first return."""
    import os
    import platform
    from models.oracle_trace import cpu_location
    torch,np=original.torch,original.np
    configurations={}
    for name,fn in [('torch_build',torch.__config__.show),('torch_parallel',torch.__config__.parallel_info),('numpy_build',np.__config__.show)]:
        out=io.StringIO()
        with contextlib.redirect_stdout(out):result=fn()
        text=out.getvalue()+(result if isinstance(result,str) else '')
        configurations[name]={'text':text[:65536],'truncated':len(text)>65536}
    return {'collected_after_original_gate_or_technical_stop':True,'host':platform.node(),
        'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
        'architecture':platform.machine(),'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
        'CPU_count':os.cpu_count(),'cpu_capability':torch.backends.cpu.get_cpu_capability(),
        'affinity_read_only':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        'CPU_location':cpu_location(),'CUDA_initialized':torch.cuda.is_initialized(),
        'environment':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG')},
        'library_configuration':configurations}


class ReturnObserver:
    def __init__(self,original,packet,manifest,folder):
        from models import attention_component
        if original.fp64_oracle is not attention_component.fp64_oracle:
            raise ValueError('Original oracle alias was replaced.')
        self.identities={name:bind_function(fn,packet,manifest,name) for name,fn in (
            ('worker',original.worker),('fp64_oracle',original.fp64_oracle),('clone_layout',original.clone_layout),('iteration',original.iteration))}
        self.original=original;self.folder=Path(folder);self.worker_frame=None
        self.calls=0;self.returns=0;self.clone_interceptions=0;self.forbidden_calls=0
        self.artifact=None;self.events=[]

    def callback(self,frame,event,arg):
        if event=='c_call' and getattr(arg,'__name__','')=='scaled_dot_product_attention':
            self.forbidden_calls+=1;raise RuntimeError('SDPA is outside this observer scope.')
        if event=='call' and frame.f_code is self.original.worker.__code__:
            if self.worker_frame is not None:raise RuntimeError('Only one original worker invocation is allowed.')
            self.worker_frame=frame
        if frame.f_code is self.original.fp64_oracle.__code__:
            if event=='call':
                self.calls+=1
                if self.calls!=1 or frame.f_back is not self.worker_frame:
                    raise RuntimeError('Only the first original worker oracle call is allowed.')
                self.events.append('original_oracle_call')
            elif event=='return':
                if arg is None:
                    self.events.append('oracle_unwound_without_computed_return')
                    return  # Keep the original exception; do not mask it.
                self.returns+=1
                if self.returns!=1 or not isinstance(arg,dict) or set(arg)!=set(OUTPUTS):
                    raise RuntimeError('Original oracle did not return all four computed tensors.')
                torch=self.original.torch
                if any(not isinstance(v,torch.Tensor) or v.device.type!='cpu' or v.dtype!=torch.float64 for v in arg.values()):
                    raise ValueError('Actual return must be CPU FP64; no conversion or fallback.')
                # These imports, copies and writes occur AFTER the original math.
                from models.oracle_trace import save_arrays
                self.artifact=save_arrays(self.folder/'computed-oracle.npz',arg)
                self.events.append('actual_return_saved_before_original_gate')
        if event=='call' and frame.f_code is self.original.iteration.__code__:
            self.forbidden_calls+=1;raise RuntimeError('GPU attention iteration is outside scope.')
        if event=='call' and frame.f_code is self.original.clone_layout.__code__:
            self.clone_interceptions+=1
            result=self.worker_frame.f_locals.get('result',{}) if self.worker_frame else {}
            gate=result.get('oracle_bookkeeping',{})
            if self.calls!=1 or self.returns!=1 or self.artifact is None or set(gate)!=set(OUTPUTS) or not all(c.get('passed') is True for c in gate.values()):
                raise RuntimeError('Unexpected transfer boundary or missing original gate.')
            self.events.append('original_gate_passed_transfer_call_intercepted_before_body')
            raise BoundaryReached('CPU gate recorded; stop before input transfer and SDPA.')


def observe(original,args,packet,folder,*,local=False):
    """Invoke one unchanged original worker. No retries or acceptance overrides."""
    import platform
    packet,folder=Path(packet),Path(folder)
    manifest=json.loads((packet/'FIXED-ATTENTION-HANDOFF.json').read_text())
    if (local is not args.tiny or args.device!=('cpu' if local else 'cuda')
            or manifest['tiny_fixture'] is not local or args.case!='encoder-1' or args.arm!='math' or args.replay!=0):
        raise ValueError('Wrong boundary fixture, worker, arm label or device domain.')
    if local and platform.node() in ('linux10603','linux10609'):raise ValueError('Tiny observer rehearsal is local only.')
    if not local and platform.node()!='linux10603':raise ValueError('Live observer is target-host only.')
    if sys.getprofile() is not None or sys.gettrace() is not None:raise ValueError('Refuse to replace an existing trace or profile hook.')
    receipt=folder/'boundary.json'
    if receipt.exists() or (folder/'computed-oracle.npz').exists() or (folder/'computed-oracle.npz.partial').exists() or (Path(args.worker_dir)/'worker.json').exists():
        raise FileExistsError('Boundary capture is consumed; preserve all existing files.')
    observation=ReturnObserver(original,packet,manifest,folder)
    state={'kind':KIND,'local_tiny_rehearsal':local,'case':args.case,'arm_label':args.arm,'replay':args.replay,
        'capture_complete':False,'original_gate_passed':None,'outcome':'starting','original_full_worker_accepted':False,
        'GPU_attention_executed':False,'GPU_input_transfer_completed':False,'detailed_introspection_before_first_oracle':False,
        'old_failure_waived':False,'research_admitted':False,'function_identities':observation.identities}
    record(receipt,state);started=time.monotonic();previous_profile=sys.getprofile();previous_trace=sys.gettrace()
    previous_handler=signal.getsignal(signal.SIGALRM);previous_timer=signal.getitimer(signal.ITIMER_REAL)
    failure=None;stopped=False;installed=observation.callback
    try:
        sys.setprofile(installed)
        try:
            original.worker(args,packet)
            raise RuntimeError('Original worker escaped the intended stop boundary.')
        except BoundaryReached:
            stopped=True
        except BaseException as exc:
            failure={'type':type(exc).__name__,'message':str(exc)}
    finally:
        sys.setprofile(previous_profile)
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous_handler)
        if previous_timer[0]>0:
            signal.setitimer(signal.ITIMER_REAL,max(.000001,previous_timer[0]-(time.monotonic()-started)),previous_timer[1])
        elif previous_timer[1]>0:signal.setitimer(signal.ITIMER_REAL,0,previous_timer[1])
    state.update(profile_restored=sys.getprofile() is previous_profile,trace_restored=sys.gettrace() is previous_trace,
        original_oracle_calls=observation.calls,original_oracle_returns=observation.returns,
        clone_call_intercepted_before_body=observation.clone_interceptions==1,
        forbidden_attention_calls=observation.forbidden_calls,events=observation.events,
        original_worker_exception=failure,computed_return_artifact=observation.artifact)
    try:
        old_file=Path(args.worker_dir)/'worker.json';old=read_record(old_file)
        state['original_worker_sha256']=digest(old_file)
        state['original_worker_complete']=old['complete'];state['original_worker_passed']=old['passed']
        if old['complete'] or old['passed']:raise ValueError('An incomplete original GPU worker must never be promoted.')
        gate=old.get('oracle_bookkeeping',{})
        if set(gate)==set(OUTPUTS) and observation.artifact is not None and observation.calls==observation.returns==1:
            if any(c.get('rtol')!=1e-12 or c.get('atol')!=1e-14 or type(c.get('passed')) is not bool for c in gate.values()):
                raise ValueError('Original bookkeeping bounds or decisions changed.')
            state['original_gate_checks']=gate
            if all(c['passed'] for c in gate.values()) and stopped and old.get('error_type')=='BoundaryReached':
                state.update(capture_complete=True,original_gate_passed=True,outcome='captured-gate-pass')
            elif (not all(c['passed'] for c in gate.values()) and not stopped
                    and old.get('error_type')=='ValueError' and old.get('error')=='Fixed oracle bookkeeping failed; preserve input snapshot.'):
                state.update(capture_complete=True,original_gate_passed=False,outcome='captured-original-refusal')
        if not state['capture_complete']:state['outcome']='technical-failure'
        observation.events.append('detailed_introspection_after_original_gate_or_stop')
        state['post_gate_context']=detailed_context(original)
    except BaseException as exc:
        state.update(capture_complete=False,outcome='technical-failure',review_error={'type':type(exc).__name__,'message':str(exc)})
    state['wall_seconds']=time.monotonic()-started
    record(receipt,state,replace=True)
    return state
