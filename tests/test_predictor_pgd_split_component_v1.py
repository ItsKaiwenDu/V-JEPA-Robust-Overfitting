"""Source/bytecode audits and explicit fake-engine callback simulations ONLY.

FakeTensor contains tiny opaque byte labels plus geometry metadata. No arrays,
Torch/NumPy, model, SDPA arithmetic, actual GraphTask or gradients are created.
"""
import ast
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from types import CodeType, SimpleNamespace as NS

import pytest
from models import predictor_pgd_split_component_v1 as builder

ROOT=Path(__file__).resolve().parents[1]
CPU=(ROOT/'models/predictor_forward_preserving_cpu_v1.py').read_bytes()
TARGET=(ROOT/'models/predictor_forward_preserving_target_v1.py').read_bytes()
SHAPE=(1,12,2048,32)


class Device:
    type='cuda';index=0
    def __str__(self):return 'cuda:0'
    def __eq__(self,other):return type(other) is Device and self.type==other.type and self.index==other.index


class Storage:
    serial=0
    def __init__(self,size):
        Storage.serial+=1;self.pointer=Storage.serial;self.size=size
    def data_ptr(self):return self.pointer
    def nbytes(self):return self.size


class Handle:
    serial=0
    def __init__(self,holder,callback,runtime,label):
        Handle.serial+=1;self.id=Handle.serial;self.holder=holder;self.runtime=runtime;self.label=label
        holder[self.id]=callback;runtime.live[self.id]=label
    def remove(self):
        self.runtime.removal_attempts.append(self.label)
        if self.runtime.should_fail_removal(self.label):raise RuntimeError('SIMULATED hook removal refusal')
        self.holder.pop(self.id,None);self.runtime.live.pop(self.id,None)


class FakeTensor:
    def __init__(self,runtime,data,*,storage=None,stride=None,offset=0,requires=True,leaf=True,node=None):
        self.runtime=runtime;self.data=data;self.shape=SHAPE;self.ndim=4;self.dtype='torch.float32';self.device=Device()
        self.storage=storage or Storage(3*2**20);self.strides=stride or (786432,65536,32,1);self.offset=offset
        self.requires_grad=requires;self.is_leaf=leaf;self.grad_fn=node;self.grad=None;self._version=0;self.finite=True
        self._backward_hooks={};self.label='value'
    def stride(self):return self.strides
    def storage_offset(self):return self.offset
    def untyped_storage(self):return self.storage
    def detach(self):
        value=FakeTensor(self.runtime,self.data,storage=self.storage,stride=self.strides,offset=self.offset,requires=False)
        value._version=self._version;value.finite=self.finite;value.dtype=self.dtype;value.device=self.device
        value.label=self.label.replace('outer-','inner-')
        return value
    def requires_grad_(self,value):self.requires_grad=value;return self
    def cpu(self):return self
    def contiguous(self):return self
    def numpy(self):return NS(tobytes=lambda:self.data)
    def register_hook(self,callback):
        if self.runtime.fault=='register-leaf' and self.label=='outer-1':raise RuntimeError('SIMULATED registration refusal')
        return Handle(self._backward_hooks,callback,self.runtime,self.label)


class Node:
    def __init__(self,runtime,kind,ctx=None):self.runtime=runtime;self.kind=kind;self.ctx=ctx;self.prehooks={}
    def register_prehook(self,callback):return Handle(self.prehooks,callback,self.runtime,self.kind+'-root')


class RNG:
    def __init__(self,data):self.data=data
    def clone(self):return RNG(self.data)


class Runtime:
    """Calls real generated control-flow with wholly synthetic task callbacks."""
    def __init__(self,*,outer_thread=False,inner_thread=False,fault=None):
        self.fault=fault;self.outer_thread=outer_thread;self.inner_thread=inner_thread
        self.local=threading.local();self.pid=os.getpid();self.live={};self.removal_attempts=[];self.failed_removals=set()
        self.flags={'flash':True,'mem_efficient':True,'math':True,'cudnn':True};self.initial=dict(self.flags)
        self.events=[];self.grad_entries=[];self.cpu_rng=RNG(b'CPU');self.cuda_rng=RNG(b'CUDA')
        shared=Storage(9*2**20);stride=(2359296,32,1152,1)
        self.values=tuple(FakeTensor(self,bytes([65+i]),storage=shared,stride=stride,offset=i*384) for i in range(3))
        for i,value in enumerate(self.values):value.label='outer-'+str(i)
        self.incoming=FakeTensor(self,b'COTANGENT',requires=False)
        self.sdpa=self._sdpa;self.grad=self._grad
        self.graph_api=lambda:getattr(self.local,'graph',-1)
        self.torch=NS(__version__='2.8.0+cu128',version=NS(cuda='12.8'),Tensor=FakeTensor,float32='torch.float32',
            nn=NS(functional=NS(scaled_dot_product_attention=self.sdpa)),autograd=NS(grad=self.grad),
            _C=NS(_nn=NS(scaled_dot_product_attention=self.sdpa),_current_graph_task_id=self.graph_api),
            is_grad_enabled=lambda:getattr(self.local,'enabled',True),enable_grad=self.enable_grad,
            is_inference_mode_enabled=lambda:False,is_autocast_enabled=lambda _:False,
            isfinite=lambda t:NS(all=lambda:t.finite),get_rng_state=lambda:self.cpu_rng,
            equal=lambda a,b:a.data==b.data)
        self.torch.cuda=NS(is_available=lambda:True,current_device=lambda:0,get_rng_state=lambda:self.cuda_rng)
        self.torch.backends=NS(cuda=NS(**{name+'_sdp_enabled':lambda n=name:self.flags[n] for name in self.flags}))
        rt=self
        class Function:
            def __init_subclass__(cls):
                cls._backward_cls=type(cls.__name__+'Backward',(Node,),{})
            @classmethod
            def apply(cls,route,index,guarded,*values):
                ctx=NS(save_for_backward=lambda *v:setattr(ctx,'saved_tensors',v))
                with rt.grad_mode(False):out=cls.forward(ctx,route,index,guarded,*values)
                node=cls._backward_cls(rt,'custom',ctx);node.backward=cls.backward
                out.requires_grad=True;out.is_leaf=False;out.grad_fn=node
                return out
        self.torch.autograd.Function=Function

    @contextmanager
    def grad_mode(self,value):
        old=getattr(self.local,'enabled',True);self.local.enabled=value
        try:yield
        finally:self.local.enabled=old

    def enable_grad(self):return self.grad_mode(True)

    @contextmanager
    def kernel(self,backend):
        assert backend=='MATH';old=dict(self.flags)
        self.flags.update(flash=False,mem_efficient=False,math=True,cudnn=False)
        try:yield
        finally:
            self.flags.clear();self.flags.update(old)
            if self.fault=='flags':self.flags['flash']=not self.flags['flash']

    def _sdpa(self,q,k,v,**kwargs):
        assert kwargs=={'dropout_p':0.,'is_causal':False,'attn_mask':None}
        kind='math' if self.flags=={'flash':False,'mem_efficient':False,'math':True,'cudnn':False} else 'native'
        self.events.append(('SDPA-STUB',kind,tuple(id(t) for t in (q,k,v)),self.torch.is_grad_enabled()))
        assert self.torch.is_grad_enabled()
        return FakeTensor(self,b'OUTPUT',requires=True,leaf=False,node=Node(self,kind))

    def should_fail_removal(self,label):
        fail=(self.fault=='outer-remove' and label=='outer-0' or self.fault=='inner-remove' and label=='inner-0')
        if fail and label not in self.failed_removals:self.failed_removals.add(label);return True
        return False

    def run_task(self,function,graph,foreign):
        def run():
            old=getattr(self.local,'graph',-1);self.local.graph=graph
            try:
                with self.grad_mode(False):return function()
            finally:self.local.graph=old
        if not foreign:return run()
        results=[];errors=[]
        def call():
            try:results.append(run())
            except BaseException as error:errors.append(error)
        worker=threading.Thread(target=call);worker.start();worker.join(timeout=3)
        assert not worker.is_alive(),'SIMULATED callback deadlock'
        if errors:raise errors[0]
        return results[0]

    def _grad(self,output,values,incoming,*,create_graph,retain_graph):
        assert create_graph is False and retain_graph is False
        inner=output.grad_fn.kind=='math'
        self.grad_entries.append(('SIMULATED-inner' if inner else 'SIMULATED-outer',tuple(id(t) for t in values)))
        graph=202 if inner else 101
        if inner and self.fault=='same-inner-graph':graph=101
        if not inner and self.fault=='negative-root':graph=-1
        def execute():
            root=list(output.grad_fn.prehooks.values())
            if self.fault!=('no-inner-root' if inner else 'no-outer-root'):
                for callback in root:callback((incoming,))
            if self.fault==('double-inner-root' if inner else 'double-outer-root'):
                for callback in root:callback((incoming,))
            if not inner and self.fault=='engine-error':raise RuntimeError('SIMULATED engine error')
            if output.grad_fn.kind=='custom':
                result=output.grad_fn.backward(output.grad_fn.ctx,incoming)[-3:]
            else:
                result=tuple(FakeTensor(self,('GRAD'+str(i)).encode(),requires=False) for i in range(3))
            if self.fault=='populated-grad' and not inner:values[0].grad=object()
            for i,value in enumerate(values):
                value.label=('inner-' if inner else 'outer-')+str(i)
                if self.fault==('no-inner-leaf' if inner else 'no-outer-leaf') and i==2:continue
                for callback in list(value._backward_hooks.values()):
                    if self.fault==('foreign-inner-leaf' if inner else 'foreign-outer-leaf') and i==1:
                        old=self.local.graph;self.local.graph=999
                        try:callback(result[i])
                        finally:self.local.graph=old
                    else:callback(result[i])
                    if self.fault==('double-inner-leaf' if inner else 'double-outer-leaf') and i==1:callback(result[i])
            if not inner:
                if self.fault=='input-bytes':self.values[0].data=b'CHANGED'
                if self.fault=='input-version':self.values[0]._version+=1
                if self.fault=='RNG':self.cuda_rng.data=b'CHANGED'
                if self.fault=='bad-return':return tuple(result[:2])
                if self.fault=='return-graph':result[0].requires_grad=True
                if self.fault=='return-nonfinite':result[0].finite=False
            return result
        return self.run_task(execute,graph,self.inner_thread if inner else self.outer_thread)


def namespace(runtime):
    module,audit=builder.prototype(CPU,TARGET)
    ns={'__name__':'EXPLICIT_COMPONENT_SIMULATION','torch':runtime.torch,'np':NS(__version__='2.4.6'),
        'os':NS(getpid=lambda:runtime.pid),'threading':threading,'platform':NS(node=lambda:'linux10603',system=lambda:'Linux',python_version=lambda:'3.11.17'),
        'copy':copy,'hashlib':hashlib,'json':json,'SimpleNamespace':NS,'sdpa_kernel':runtime.kernel,'SDPBackend':NS(MATH='MATH'),
        'DESIGN_SHA':builder.DESIGN_SHA,'KIND':builder.KIND}
    exec(compile(module,'<UNBOUND-SOURCE-FAKE-ENGINE-ONLY>','exec'),ns)
    return ns,audit


def route(runtime,ns,role):
    b={'design_sha256':builder.DESIGN_SHA,'control_sha256':'2'*64,'boundary_sha256':'3'*64,'probe_index':0 if role=='native' else 1}
    return ns['ComponentRoute'](role,runtime.values,runtime.incoming,binding=b)


def code_for(source,class_name,function):
    compiled=compile(source,'<FROZEN-SOURCE-COMPILE-NO-EXEC>','exec')
    c=next(x for x in compiled.co_consts if isinstance(x,CodeType) and x.co_name==class_name)
    return next(x for x in c.co_consts if isinstance(x,CodeType) and x.co_name==function)


def signature(code):
    # Ignore only file/line metadata; every instruction, constant, name, closure,
    # signature and exception table must match the frozen compiled source.
    return (code.co_code,code.co_argcount,code.co_posonlyargcount,code.co_kwonlyargcount,code.co_flags,
        code.co_names,code.co_varnames,code.co_cellvars,code.co_freevars,code.co_exceptiontable,
        tuple(signature(c) if isinstance(c,CodeType) else c for c in code.co_consts))


def test_frozen_arithmetic_AST_and_bytecode_are_exact_and_backward_alias_is_same_code():
    runtime=Runtime();ns,audit=namespace(runtime)
    assert audit['arithmetic_bodies_copied_exactly'] and audit['runtime_namespace_unbound']
    assert not audit['target_runner_available'] and not audit['scientific_qualified']
    pairs=[(CPU,'_OriginalForwardMathVJP','backward'),(TARGET,'_Attention','forward'),(TARGET,'_Attention','backward')]
    for src,cls,method in pairs:
        assert signature(getattr(ns[cls],method).__code__)==signature(code_for(src,cls,method))
    assert ns['_public_math_backward'].__code__ is ns['_OriginalForwardMathVJP'].backward.__code__
    module,_=builder.prototype(CPU,TARGET)
    assert isinstance(module.body[0],ast.If) and module.body[0].test.value is False
    imports=[n for n in ast.walk(module) if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert len(imports)==3 and module.body[0].body==imports
    assert audit['compiler_only_imports_in_literal_false_branch']==3 and not audit['generated_runtime_imports']
    for changed_cpu,changed_target in [(CPU+b'\n',TARGET),(CPU,TARGET+b'\n')]:
        with pytest.raises(ValueError):builder.prototype(changed_cpu,changed_target)


@pytest.mark.parametrize('role',['native','forward-preserving'])
@pytest.mark.parametrize('outer,inner',[(False,False),(True,False),(True,True)])
def test_actual_component_control_flow_with_fake_engine_keeps_single_use_and_real_callback_rules(role,outer,inner,tmp_path):
    rt=Runtime(outer_thread=outer,inner_thread=inner);ns,_=namespace(rt);c=route(rt,ns,role)
    original_pointers=(rt.torch.nn.functional.scaled_dot_product_attention,rt.torch.autograd.grad)
    with c:
        output=c.attention();result=c.gradient(output,c.values,c.incoming)
        provisional=c.snapshot();assert not provisional['route_complete'] and not provisional['scientific_qualified']
        assert len(result)==3
        with pytest.raises(ValueError):c.gradient(output,c.values,c.incoming)
    proof=c.snapshot()
    assert proof['route_complete'] and proof['outer']['state']=='complete'
    assert proof['outer']['root_calls']==1 and sorted(h['index'] for h in proof['outer']['leaf_hooks'])==[0,1,2]
    assert proof['outer']['root_hook_removed'] and proof['outer']['leaf_hooks_removed']
    assert len(rt.events)==(2 if role=='forward-preserving' else 1)
    assert len(rt.grad_entries)==(2 if role=='forward-preserving' else 1)
    assert rt.events[0][2]==tuple(id(t) for t in rt.values) and all(e[3] for e in rt.events)
    if role=='forward-preserving':
        assert proof['calls'][0]['inner_graph_id']==202 and proof['outer']['graph_id']==101
        assert sorted(h['index'] for h in proof['calls'][0]['leaf_hooks'])==[0,1,2]
        assert proof['calls'][0]['hooks_removed']
    assert not rt.live and rt.flags==rt.initial and all(v.grad is None and not v._backward_hooks for v in rt.values)
    assert (rt.torch.nn.functional.scaled_dot_product_attention,rt.torch.autograd.grad)==original_pointers
    assert not c.active and c.closed and c.values==() and c.output is None
    assert ns['_CLAIM']['consumed'] is True
    with pytest.raises(ValueError):route(rt,ns,role).__enter__()
    (tmp_path/'SIMULATED-route.json').write_text(json.dumps({'evidence_origin':'FAKE-ENGINE-SIMULATION','route':proof},sort_keys=True)+'\n')


@pytest.mark.parametrize('fault',['no-outer-root','double-outer-root','negative-root','no-outer-leaf','double-outer-leaf',
 'foreign-outer-leaf','no-inner-root','double-inner-root','same-inner-graph','no-inner-leaf','double-inner-leaf',
 'foreign-inner-leaf','engine-error','bad-return','return-graph','return-nonfinite','input-bytes','input-version',
 'populated-grad','flags','RNG','register-leaf','outer-remove','inner-remove'])
def test_fake_graph_or_restoration_failures_never_make_route_complete(fault,tmp_path):
    rt=Runtime(outer_thread=True,inner_thread=True,fault=fault);ns,_=namespace(rt);c=route(rt,ns,'forward-preserving')
    with pytest.raises((ValueError,RuntimeError,AssertionError)):
        with c:
            out=c.attention();c.gradient(out,c.values,c.incoming)
    proof=c.snapshot()
    assert not proof['route_complete'] and proof['phase']=='failed' and c.closed and not c.active
    assert not proof['scientific_qualified'] and not proof['backend_qualified'] and not proof['worker_complete']
    if fault not in ('outer-remove','inner-remove'):assert not rt.live
    if fault=='outer-remove':
        assert set(rt.removal_attempts)>={'custom-root','outer-0','outer-1','outer-2'}
        assert proof['outer']['root_hook_removed'] is False
    if fault=='inner-remove':assert not proof['calls'][0].get('hooks_removed')
    assert ns['_CLAIM']['consumed'] is True
    (tmp_path/'SIMULATED-failed-route.json').write_text(json.dumps({'evidence_origin':'FAKE-ENGINE-SIMULATION','fault':fault,'route':proof},sort_keys=True)+'\n')


@pytest.mark.parametrize('role',['native','forward-preserving'])
def test_incomplete_scope_without_gradient_refuses_at_exit(role):
    rt=Runtime();ns,_=namespace(rt);c=route(rt,ns,role)
    with pytest.raises(ValueError,match='Incomplete'):
        with c:c.attention()
    assert c.closed and not c.snapshot()['route_complete']


@pytest.mark.parametrize('field,value',[('device','cpu'),('dtype','torch.float64'),('shape',(1,2,32,32)),
 ('leaf',False),('requires',False),('offset',3000000),('stride',(1,0,1,1)),('version',1),('grad',object()),('nonfinite',True),('hooks',True)])
def test_invalid_or_changed_canonical_inputs_refuse_without_any_engine_work(field,value):
    rt=Runtime();ns,_=namespace(rt);c=route(rt,ns,'forward-preserving')
    if field=='version':
        with pytest.raises(ValueError):
            with c:
                rt.values[0]._version=1;c.attention()
        assert not rt.events and not rt.grad_entries and c.closed
        return
    t=rt.values[0]
    if field=='device':t.device=NS(type='cpu',index=None)
    elif field=='dtype':t.dtype=value
    elif field=='shape':t.shape=value
    elif field=='leaf':t.is_leaf=value
    elif field=='requires':t.requires_grad=value
    elif field=='offset':t.offset=value
    elif field=='stride':t.strides=value
    elif field=='grad':t.grad=value
    elif field=='nonfinite':t.finite=False
    else:t._backward_hooks={999:lambda g:g}
    with pytest.raises(ValueError):c.__enter__()
    assert not rt.events and not rt.grad_entries and ns['_CLAIM']['consumed'] is True


@pytest.mark.parametrize('create,retain',[(True,False),(False,True),(0,False),(False,None)])
def test_first_order_flags_and_exact_token_rejections_before_fake_gradient(create,retain):
    rt=Runtime();ns,_=namespace(rt);c=route(rt,ns,'native')
    try:
        with c:
            out=c.attention()
            with pytest.raises(ValueError):c.gradient(out,c.values,c.incoming,create_graph=create,retain_graph=retain)
            assert not rt.grad_entries
            c.gradient(out,c.values,c.incoming)
    finally:assert c.closed


def test_wrong_output_tuple_or_incoming_object_cannot_substitute_tokens():
    rt=Runtime();ns,_=namespace(rt);c=route(rt,ns,'native')
    with c:
        out=c.attention()
        for output,values,cot in [(out.detach(),c.values,c.incoming),(out,tuple(list(c.values)),c.incoming),(out,c.values,c.incoming.detach())]:
            with pytest.raises(ValueError):c.gradient(output,values,cot)
        assert not rt.grad_entries
        c.gradient(out,c.values,c.incoming)


@pytest.mark.parametrize('which',['public-SDPA','public-grad','graph-API','private-forward','private-math','extra-hook'])
def test_changed_execution_paths_or_foreign_hooks_refuse_before_dispatch(which):
    rt=Runtime();ns,_=namespace(rt);c=route(rt,ns,'native')
    with pytest.raises((ValueError,AttributeError)):
        with c:
            if which=='public-SDPA':rt.torch.nn.functional.scaled_dot_product_attention=lambda *a:None
            elif which=='public-grad':rt.torch.autograd.grad=lambda *a:None
            elif which=='graph-API':rt.torch._C._current_graph_task_id=lambda:-1
            elif which=='private-forward':c._guarded_forward=lambda *a:None
            elif which=='private-math':c._math_call=lambda *a:None
            else:rt.values[0]._backward_hooks[999]=lambda g:g
            c.attention()
    assert not rt.events and not rt.grad_entries and not c.snapshot()['route_complete']


def test_foreign_public_thread_or_pid_cannot_start_or_close_component():
    rt=Runtime();ns,_=namespace(rt);c=route(rt,ns,'native');errors=[]
    with c:
        output=c.attention()
        def foreign():
            for action in [lambda:c.gradient(output,c.values,c.incoming),lambda:c.__exit__(None,None,None)]:
                try:action()
                except ValueError as error:errors.append(error)
        t=threading.Thread(target=foreign);t.start();t.join(timeout=2);assert not t.is_alive() and len(errors)==2
        rt.pid+=1
        with pytest.raises(ValueError):c.gradient(output,c.values,c.incoming)
        rt.pid-=1
        assert not rt.grad_entries
        c.gradient(output,c.values,c.incoming)
    assert c.snapshot()['route_complete']


@pytest.mark.parametrize('fault',['host','system','python','torch','cuda','numpy','design','probe','binding_type'])
def test_unbound_domain_and_external_binding_guards_fail_before_tensor_operations(fault):
    rt=Runtime();ns,_=namespace(rt)
    b={'design_sha256':builder.DESIGN_SHA,'control_sha256':'2'*64,'boundary_sha256':'3'*64,'probe_index':1}
    if fault=='host':ns['platform'].node=lambda:'Kaiwens-MacBook-Pro'
    elif fault=='system':ns['platform'].system=lambda:'Darwin'
    elif fault=='python':ns['platform'].python_version=lambda:'3.11.16'
    elif fault=='torch':rt.torch.__version__='2.14.1'
    elif fault=='cuda':rt.torch.version.cuda='13.2'
    elif fault=='numpy':ns['np'].__version__='2.4.0'
    elif fault=='design':b['design_sha256']='0'*64
    elif fault=='probe':b['probe_index']=True
    else:b={}
    with pytest.raises(ValueError):ns['ComponentRoute']('forward-preserving',rt.values,rt.incoming,binding=b)
    assert not rt.events and not rt.grad_entries and ns['_CLAIM'] is None


def test_ownership_snapshot_cannot_be_used_as_backend_lineage_or_worker_admission():
    rt=Runtime();ns,_=namespace(rt);c=route(rt,ns,'native')
    with c:
        out=c.attention();c.gradient(out,c.values,c.incoming)
    proof=c.snapshot()
    for field in ('scientific_qualified','backend_qualified','actual_CUDA_execution_admitted','worker_complete','training_admitted','canonical_lineage_authenticated'):
        assert proof[field] is False
    assert len(json.dumps(proof).encode())<64*1024
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    source=ast.parse((ROOT/'models/predictor_pgd_split_component_v1.py').read_bytes())
    imports={n.module for n in ast.walk(source) if isinstance(n,ast.ImportFrom)}|{a.name for n in ast.walk(source) if isinstance(n,ast.Import) for a in n.names}
    assert imports<=set(sys.stdlib_module_names)
    assert not any(isinstance(n,ast.FunctionDef) and n.name in ('main','run','worker') for n in source.body)


def test_frozen_compiler_context_has_no_executable_import_in_any_code_object():
    import builtins
    import dis
    module,audit=builder.prototype(CPU,TARGET)
    code=compile(module,'<COMPONENT-NO-EXECUTABLE-IMPORTS>','exec')
    def walk(c):
        yield c
        for child in c.co_consts:
            if isinstance(child,CodeType):yield from walk(child)
    assert all(ins.opname not in ('IMPORT_NAME','IMPORT_FROM','IMPORT_STAR')
        for c in walk(code) for ins in dis.get_instructions(c))
    ns,_=namespace(Runtime())
    def refuse(*args,**kwargs):raise AssertionError('Any runtime import is forbidden in this prototype.')
    ns['__builtins__']=dict(vars(builtins),__import__=refuse)
    exec(code,ns)
    assert ns['_CLAIM'] is None and not audit['generated_runtime_imports']
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


@pytest.mark.parametrize('change',['incoming-bytes','incoming-hook','class-forward','class-math'])
def test_new_boundary_restoration_regressions_refuse_before_dispatch(change):
    rt=Runtime();ns,_=namespace(rt);c=route(rt,ns,'native')
    with pytest.raises(ValueError):
        with c:
            if change=='incoming-bytes':rt.incoming.data=b'ALTERED'
            elif change=='incoming-hook':rt.incoming._backward_hooks[999]=lambda g:g
            elif change=='class-forward':ns['ComponentRoute']._guarded_forward=lambda *args,**kwargs:None
            else:ns['ComponentRoute']._math_call=lambda *args,**kwargs:None
            c.attention()
    assert not rt.events and not rt.grad_entries and not c.snapshot()['route_complete']


def test_ownership_metadata_ceiling_refuses_without_truncating_or_publishing():
    rt=Runtime();ns,_=namespace(rt);c=route(rt,ns,'native')
    c.record['untrusted_addition']='x'*(64*1024)
    with pytest.raises(ValueError,match='Bounded component ownership'):c.snapshot()
    assert len(c.record['untrusted_addition'])==64*1024
    assert not c.record['route_complete'] and ns['_CLAIM'] is None
