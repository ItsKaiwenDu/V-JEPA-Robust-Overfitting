"""Observe one native post-RoPE SDPA call through a frozen instance-local route."""
import copy
import hashlib
import inspect
import os
from pathlib import Path
import platform
import threading
from types import FunctionType

from models.predictor_sdpa_capture_record_v1 import KIND,scope_check

ROUTE_SHA='ef3675e69ef56b0b4ef4f543cab96b7f2bc757a64202d74500f3802ccd5dfc12'
ADAPTER_SHA='89b7e0c67e88f2f776bb0bb6ad786a2202115e64ffc6c6f9a962381cc2e9a6b8'
_ACTIVE=set()


def closure(function):
    if type(function) is not FunctionType:raise ValueError('Frozen private Python function required.')
    return {name:cell.cell_contents for name,cell in zip(function.__code__.co_freevars,function.__closure__ or ())}


class NativeSDPAObserver:
    """No modified forward, incoming gradient, backend selection or GraphTask."""
    def __init__(self,route,store,*,control_sha256):
        import torch
        from models import predictor_eot8_owned_v1 as owned
        from models import predictor_eot8_adapter_v1 as base
        if type(route) is not owned.PredictorRoute or route.mode!='original' or not route.active or route.phase!='EOT8-before':
            raise ValueError('Active original native EOT8 route required.')
        if not isinstance(control_sha256,str) or len(control_sha256)!=64 or any(c not in '0123456789abcdef' for c in control_sha256):
            raise ValueError('Bound observation control hash required.')
        if route.tiny:
            if platform.node() in ('linux10603','linux10609'):raise ValueError('Tiny local CPU proof cannot run on a university host.')
        elif (platform.node()!='linux10603' or platform.python_version()!='3.11.17' or torch.__version__!='2.8.0+cu128'
                or torch.version.cuda!='12.8'):raise ValueError('Exact future target host/runtime required; no fallback.')
        for module,want in ((owned,ROUTE_SHA),(base,ADAPTER_SHA)):
            if hashlib.sha256(Path(inspect.getsourcefile(module.PredictorRoute)).read_bytes()).hexdigest()!=want:
                raise ValueError('Frozen route/adapter source changed.')
        route.check()
        counts=(len(route.model.encoder.blocks),len(route.model.predictor.predictor_blocks))
        if counts!=((2,2) if route.tiny else (24,12)):raise ValueError('Frozen owner geometry required.')
        self.route,self.store=route,store;self.pid=os.getpid();self.thread=threading.get_ident()
        self.device='cpu' if route.tiny else 'cuda';self.shape=[1,2,32,32] if route.tiny else [1,12,2048,32]
        self.row_index=counts[0];self.handles=[];self.refs={};self.versions={};self.groups={};self.storage_refs={}
        self.view=self.original=self.wrapper=None;self.active=self.closed=False;self.lock=threading.RLock()
        self.state={'kind':KIND,'local_tiny':route.tiny,'execution_host':platform.node(),'mode':'original',
            'pid':self.pid,'creator_thread':self.thread,'control_sha256':control_sha256,
            'capture_complete':False,'observer_restored':False,'alias_restored':False,'tensor_versions_unchanged':False,
            'native_forward_modified':False,'native_cotangent_replaced':False,'native_full_model_qualified':False,
            'CUDA_kernel_qualified':False,'research_admitted':False,'performance_qualified':False,
            'selected':{'owner':'predictor.0','draw':0,'family':0,'forward_row_index':self.row_index,'phase':'EOT8-before'},
            'artifacts':{},'source_storage':{},'gradient_observations':{},'finite':{},'selected_call':None,
            'frozen_route_sha256':ROUTE_SHA,'frozen_adapter_sha256':ADAPTER_SHA}

    def _owner(self,*,backward=False):
        route=self.route
        if not self.active or self.closed or os.getpid()!=self.pid or route.phase!='EOT8-before':
            raise ValueError('Observer inactive, foreign process or changed phase.')
        route._common()
        if self.view.route is not self.wrapper:raise ValueError('Observed private alias changed.')
        graph=route._graph_id();task=route._task;thread=threading.get_ident()
        if backward:
            if task is None or task.state!='running' or task.phase!=route.phase or task.graph_id is None or graph!=task.graph_id:
                raise ValueError('Observation callback is outside its actual native GraphTask.')
        elif task is None:
            if thread!=self.thread or graph!=-1:raise ValueError('Unowned original observation forward.')
        elif task.state!='running' or task.graph_id is None or graph!=task.graph_id:
            raise ValueError('Unowned observation recomputation.')
        return graph,thread

    def _save(self,name,tensor):
        import torch
        from models.predictor_eot8_capture_v1 import tensor_info
        if (not isinstance(tensor,torch.Tensor) or tensor.dtype!=torch.float32 or tensor.device.type!=self.device
                or list(tensor.shape)!=self.shape or tensor.ndim!=4 or any(s<=0 for s in tensor.stride())):
            raise ValueError('Actual selected native tensor domain/shape/layout changed.')
        version=tensor._version;before=tensor_info(tensor)
        storage=tensor.untyped_storage();key=storage._cdata
        if key not in self.groups:
            self.groups[key]=len(self.groups);self.storage_refs[key]=storage
        span=4*(1+sum((n-1)*s for n,s in zip(tensor.shape,tensor.stride())))
        offset=tensor.storage_offset()
        if span>16*2**20 or offset<0 or 4*offset+span>storage.nbytes():raise ValueError('Native source storage extent invalid.')
        self.state['source_storage'][name]={'alias_group':self.groups[key],'storage_offset_elements':offset,
            'storage_bytes':storage.nbytes(),'positive_stride_span_bytes':span,
            'data_address_mod_256':tensor.data_ptr()%256,'storage_address_mod_256':storage.data_ptr()%256,
            'version':version,'requires_grad':tensor.requires_grad,'source_logical_hash':before['sha256']}
        item=self.store.save(name,tensor);self.state['artifacts'][name]=item
        # Diagnostics must not add saved tensors to checkpoint bookkeeping.
        self.state['finite'][name]=bool(torch.isfinite(tensor.detach()).all().item())
        self.store.publish(self.state)
        if tensor._version!=version or tensor_info(tensor)!=before:raise ValueError('Observation changed actual native tensor bytes/version.')
        if not self.state['finite'][name]:raise ValueError('Nonfinite native observation saved before refusal.')

    def __enter__(self):
        from models import predictor_eot8_adapter_v1 as base
        import torch
        if self.active or self.closed or id(self.route) in _ACTIVE:raise ValueError('One single-use observer per native route.')
        target=self.route.model.predictor.predictor_blocks[0].attn
        bound=target.forward;outer=closure(bound.__func__)
        if bound.__self__ is not target or outer.get('self') is not self.route or outer.get('obj') is not target:
            raise ValueError('Frozen owned instance closure required.')
        private=outer.get('private')
        if type(private) is not FunctionType or private.__code__ is not type(target).forward.__code__:
            raise ValueError('Exact unmodified upstream forward bytecode required.')
        view=private.__globals__.get('F')
        if type(view) is not base.FunctionalView or view.original is not torch.nn.functional:
            raise ValueError('One frozen instance-local functional alias required.')
        original=view.route;inner=closure(original)
        if (original.__name__!='routed' or inner.get('self') is not self.route or inner.get('obj') is not target
                or inner.get('name')!='predictor.0' or inner.get('old_sdpa') is not torch._C._nn.scaled_dot_product_attention):
            raise ValueError('Original frozen native SDPA closure required.')
        self.view,self.original=view,original
        def observed(q,k,v,**kwargs):
            with self.lock:
                graph,thread=self._owner()
                selected=self.route._task is None and len(self.route.rows)==self.row_index
                if not selected:return self.original(q,k,v,**kwargs)
                if self.state['selected_call'] is not None:raise ValueError('Native selected call captured once.')
                if kwargs!={'dropout_p':0.,'is_causal':False,'attn_mask':None} or len({id(q),id(k),id(v)})!=3:
                    raise ValueError('Original SDPA arguments or tensor identities changed.')
                if not all(t.requires_grad for t in (q,k,v)):raise ValueError('Native selected operands must require gradients.')
                self.state['selected_call']={'forward_graph_id':graph,'forward_thread':thread,'pid':self.pid,'row_index':self.row_index}
                self.refs.update(q=q,k=k,v=v);self.versions.update({n:t._version for n,t in self.refs.items()})
                for name,t in zip(('q','k','v'),(q,k,v)):self._save(name,t)
                out=self.original(q,k,v,**kwargs)
                row=self.route.rows[self.row_index]
                if row['owner']!='predictor.0' or row['during_backward'] or row['outcome']!='returned':
                    raise ValueError('Selected actual native call did not return.')
                self.state['selected_call']['actual_call']=copy.deepcopy(row)
                if not out.requires_grad:raise ValueError('Actual native output must require gradients.')
                self.refs['output']=out;self.versions['output']=out._version;self._save('output',out)
                def hook(name):
                    def incoming(g):
                        with self.lock:
                            graph,thread=self._owner(backward=True)
                            if name in self.state['gradient_observations'] or g.requires_grad:raise ValueError('One first-order actual gradient observation required.')
                            self.state['gradient_observations'][name]={'graph_id':graph,'thread':thread,'pid':self.pid,'hook_removed':False}
                            self._save(name,g)
                            return None
                    return incoming
                for name,t in zip(('dq','dk','dv','grad_output'),(q,k,v,out)):
                    self.handles.append(t.register_hook(hook(name)))
                self.store.publish(self.state)
                return out
        self.wrapper=observed;self.active=True;view.route=observed;_ACTIVE.add(id(self.route))
        try:self.store.publish(self.state)
        except BaseException:
            view.route=original;self.active=False;self.closed=True;_ACTIVE.discard(id(self.route));raise
        return self

    def __exit__(self,typ,error,tb):
        cleanup_errors=[]
        for h in self.handles:
            try:h.remove()
            except BaseException as e:cleanup_errors.append(e)
        self.handles=[]
        alias_intact=self.view is not None and self.view.route is self.wrapper
        if self.view is not None:self.view.route=self.original
        self.state['alias_restored']=alias_intact and self.view.route is self.original
        self.state['tensor_versions_unchanged']=all(t._version==self.versions[n] for n,t in self.refs.items())
        self.state['observer_restored']=not cleanup_errors and self.state['alias_restored']
        for row in self.state['gradient_observations'].values():row['hook_removed']=not cleanup_errors
        self.active=False;self.closed=True;_ACTIVE.discard(id(self.route))
        original_error=error
        try:
            if typ is None:
                if not self.state['observer_restored'] or not self.state['tensor_versions_unchanged']:
                    raise ValueError('Native hook/alias/tensor-version restoration failed.')
                if set(self.state['artifacts'])!={'q','k','v','output','grad_output','dq','dk','dv'}:
                    raise ValueError('Incomplete actual native attention boundary; no reference.')
                self.state['scope_check']=scope_check(self.state,self.route.rows,self.route.scopes,self.route.refusals)
                scope=self.route.scopes[0]
                for row in self.state['gradient_observations'].values():
                    if row['graph_id']!=scope['graph_id'] or row['thread'] not in scope['call_threads']:
                        raise ValueError('Hook thread/graph absent from actual native recomputation.')
                self.state['capture_complete']=True
            else:
                self.state.update(error_type=typ.__name__,error=str(error)[:2048])
        except BaseException as failure:
            self.state.update(capture_complete=False,error_type=type(failure).__name__,error=str(failure)[:2048])
            original_error=failure
        finally:
            self.refs={};self.groups={};self.storage_refs={}
            try:self.store.finish(self.state,self.route)
            except BaseException as publish_error:
                if original_error is None:original_error=publish_error
        if typ is None and original_error is not None:raise original_error
        return False
