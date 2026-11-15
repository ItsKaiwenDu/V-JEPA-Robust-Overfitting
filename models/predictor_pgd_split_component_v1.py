"""Source-only component route builder; importing it executes no scientific API.

Returns an UNBOUND AST. A future independently reviewed worker must supply the
exact runtime and authorization before this source can execute on a target.
Current tests supply explicit fake tensors/engines; no actual gradient is run.
"""
import ast
import copy
import hashlib
import textwrap

CPU_SHA='6ba308e948c0ce346512f17ef4fc11eccb542954048c859ad6c6a751db1fbb66'
TARGET_SHA='000591f52fd66e7b1d4a78ff0c38ed265a5870e4a8463f9d810031d56752710a'
DESIGN_SHA='c5daae30cbe4043cf4d5265de2a2a7420d015ab1e244cc3d74fcc93b5f0186af'
KIND='pgd-split-one-attention-route-v1'

ROUTE_SOURCE='''
class ComponentRoute:
    """One component forward/VJP per fresh process namespace; no model patch."""
    def __init__(self,role,values,incoming,*,binding):
        if (platform.node()!='linux10603' or platform.system()!='Linux' or platform.python_version()!='3.11.17'):
            raise ValueError('Unbound source requires separately authorized linux10603 runtime; no local analogue.')
        if (torch.__version__!='2.8.0+cu128' or torch.version.cuda!='12.8' or np.__version__!='2.4.6'
                or not torch.cuda.is_available() or torch.cuda.current_device()!=0):
            raise ValueError('Exact proposed CUDA runtime required; no installation/fallback.')
        keys={'design_sha256','control_sha256','boundary_sha256','probe_index'}
        if (role not in ('native','forward-preserving') or type(values) is not tuple or len(values)!=3
                or type(binding) is not dict or set(binding)!=keys or binding['design_sha256']!=DESIGN_SHA
                or any(type(binding[k]) is not str or len(binding[k])!=64 or any(c not in '0123456789abcdef' for c in binding[k]) for k in keys-{'probe_index'})
                or type(binding['probe_index']) is not int or binding['probe_index']!=(0 if role=='native' else 1)):
            raise ValueError('Exact component role/tuple/external binding required.')
        self.role,self.values,self.incoming=role,values,incoming
        self.pid,self.thread=os.getpid(),threading.get_ident()
        self.functional,self.autograd=torch.nn.functional,torch.autograd
        self._sdpa,self._grad=self.functional.scaled_dot_product_attention,self.autograd.grad
        self._graph_api=torch._C._current_graph_task_id
        self.forward_function,self.math_function=ComponentRoute._guarded_forward,ComponentRoute._math_call
        if not callable(self._graph_api) or self._sdpa is not torch._C._nn.scaled_dot_product_attention:
            raise ValueError('Original public SDPA and real GraphTask API required.')
        self.original_flags=flags();self.entered=self.active=self.closed=False
        self.forward_used=self.gradient_used=False
        self._outer=None;self.scope=None;self.calls=[];self.handles=[];self.hook_lock=threading.RLock()
        self.expected_hooks=[{} for _ in values];self.native_calls=self.math_calls=0
        self.output=self.root_node=None
        self.record={'kind':KIND,'role':role,'binding':dict(binding),'pid':self.pid,'creator_thread':self.thread,
            'phase':'new','route_complete':False,'scientific_qualified':False,'backend_qualified':False,
            'actual_CUDA_execution_admitted':False,'worker_complete':False,'training_admitted':False,
            'canonical_lineage_authenticated':False}

    def _graph_id(self):
        graph=self._graph_api()
        if type(graph) is not int or graph< -1:raise ValueError('Actual GraphTask integer required.')
        return graph

    def _creator(self):
        if os.getpid()!=self.pid or threading.get_ident()!=self.thread:
            raise ValueError('Component public entry/exit belongs to its creator process/thread.')

    def _same_flags(self):
        if flags()!=self.original_flags:raise ValueError('Original backend flags not restored.')

    def _hook_map(self,value):
        return dict(value._backward_hooks or {})

    def _common(self,*,math_context=False):
        if (not self.active or self.closed or os.getpid()!=self.pid
                or torch.nn.functional is not self.functional or torch.autograd is not self.autograd
                or self.functional.scaled_dot_product_attention is not self._sdpa
                or self._sdpa is not torch._C._nn.scaled_dot_product_attention
                or self.autograd.grad is not self._grad or torch._C._current_graph_task_id is not self._graph_api
                or self._guarded_forward.__func__ is not self.forward_function
                or self._math_call.__func__ is not self.math_function
                or torch.is_inference_mode_enabled() or torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda')):
            raise ValueError('Inactive/foreign/replaced public or private component execution path.')
        if not math_context:self._same_flags()
        for i,value in enumerate(self.values):
            got=self._hook_map(value);expected=self.expected_hooks[i]
            if set(got)!=set(expected) or any(got[k] is not expected[k] for k in got):
                raise ValueError('Unowned or changed input leaf hooks.')
            if value._version!=self.versions[i] or value.grad is not None:
                raise ValueError('Input version changed or input .grad populated.')
        if (self.incoming._version!=self.incoming_version or self.incoming.grad is not None
                or self._hook_map(self.incoming)):
            raise ValueError('Incoming cotangent version, hooks or .grad changed.')

    def _validate(self,values,*,incoming=None):
        if len(values)!=3 or len({id(t) for t in values})!=3:raise ValueError('Three distinct Q/K/V objects required.')
        for value in values:
            self._tensor(value,requires_grad=True)
        if incoming is not None:
            self._tensor(incoming,requires_grad=False)
            if fingerprint(incoming)!=self.incoming_fingerprint:
                raise ValueError('Actual incoming cotangent values/layout/version changed.')

    def _tensor(self,value,*,requires_grad):
        if (not isinstance(value,torch.Tensor) or value.device.type!='cuda' or value.device.index!=0
                or value.dtype!=torch.float32 or tuple(value.shape)!=(1,12,2048,32) or value.ndim!=4
                or not value.is_leaf or value.requires_grad is not requires_grad or value.grad is not None
                or any(type(s) is not int or s<=0 for s in value.stride())
                or not torch.isfinite(value.detach()).all()):
            raise ValueError('Original finite CUDA FP32 leaf geometry/gradient mode required.')
        span=1+sum((n-1)*s for n,s in zip(value.shape,value.stride()))
        if (value.storage_offset()<0 or value.untyped_storage().nbytes()>16*2**20
                or 4*(value.storage_offset()+span)>value.untyped_storage().nbytes()):
            raise ValueError('Original positive-stride storage extent required.')

    def _check_tokens(self):
        self._common();self._validate(self.values,incoming=self.incoming)
        if fingerprint(self.incoming)!=self.incoming_fingerprint:
            raise ValueError('Canonical incoming bytes/layout changed.')
        if ([fingerprint(t) for t in self.values]!=self.input_fingerprints or aliases(self.values)!=self.input_aliases):
            raise ValueError('Canonical input bytes/layout/aliases changed.')
        if self.output is not None and (fingerprint(self.output)!=self.output_fingerprint or self.output.grad_fn is not self.root_node):
            raise ValueError('Exact output/root token changed.')

    def __enter__(self):
        global _CLAIM
        self._creator()
        with _LOCK:
            if self.entered or self.closed or _CLAIM is not None or self._graph_id()!=-1:
                raise ValueError('One fresh component route per process namespace; no reset/retry.')
            self.entered=True
            _CLAIM={'pid':self.pid,'thread':self.thread,'role':self.role,'consumed':True}
        try:
            if (not torch.is_grad_enabled() or torch.is_inference_mode_enabled()
                    or torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda')):
                raise ValueError('Grad-enabled first-order component entry only.')
            self._validate(self.values)
            self._tensor(self.incoming,requires_grad=False)
            if any(self._hook_map(t) for t in self.values) or self._hook_map(self.incoming):
                raise ValueError('Fresh canonical leaves/cotangent must not carry external hooks.')
            self.versions=[t._version for t in self.values];self.incoming_version=self.incoming._version
            self.input_fingerprints=[fingerprint(t) for t in self.values];self.input_aliases=aliases(self.values)
            self.incoming_fingerprint=fingerprint(self.incoming)
            self.rng_cpu=torch.get_rng_state().clone();self.rng_cuda=torch.cuda.get_rng_state().clone()
            self.active=True;self.record['phase']='ready';self._check_tokens()
            return self
        except BaseException as error:
            self.active=False;self.closed=True;self.record.update(phase='failed',error_type=type(error).__name__)
            self.values=();self.incoming=self.output=self.root_node=None;self.rng_cpu=self.rng_cuda=None
            raise

    def _admit_forward(self):
        self._common()
        if self.record['phase']!='forward-running' or self._outer is not None or self._graph_id()!=-1 or threading.get_ident()!=self.thread:
            raise ValueError('Only the one original creator forward is admitted.')

    def _guarded_forward(self,q,k,v,**kwargs):
        self._admit_forward()
        if (kwargs!={'dropout_p':0.,'is_causal':False,'attn_mask':None} or self.native_calls
                or any(a is not b for a,b in zip((q,k,v),self.values)) or not torch.is_grad_enabled()):
            raise ValueError('One original forward at exact native eligibility and arguments.')
        self.native_calls+=1
        return self._sdpa(q,k,v,**kwargs)

    def attention(self):
        self._creator();self._check_tokens()
        if self.forward_used or self.gradient_used or self.record['phase']!='ready' or not torch.is_grad_enabled():
            raise ValueError('One fresh forward only; no warmup/replay.')
        self.forward_used=True;self.record['phase']='forward-running'
        row={'index':0,'forward_graph_id':self._graph_id(),'forward_thread':self.thread,
             'inputs':copy.deepcopy(self.input_fingerprints),'aliases':list(self.input_aliases),
             'entry_flags':flags(),'forward_complete':False,'backward_complete':False}
        self.calls.append(row)
        try:
            if self.role=='forward-preserving':
                output=_Attention.apply(self,0,self._guarded_forward,*self.values)
            else:
                output=self._guarded_forward(*self.values,dropout_p=0.,is_causal=False,attn_mask=None)
                row.update(forward_complete=True,original_forward_grad_enabled=torch.is_grad_enabled(),
                    output=fingerprint(output),exit_flags=flags())
            if (self.native_calls!=1 or not row['forward_complete'] or not row['original_forward_grad_enabled']
                    or not isinstance(output,torch.Tensor) or output.dtype!=torch.float32
                    or tuple(output.shape)!=(1,12,2048,32) or output.device!=self.values[0].device
                    or not output.requires_grad or output.is_leaf or output.grad_fn is None
                    or not torch.isfinite(output.detach()).all()):
                raise ValueError('Original finite forward/output graph required.')
            if self.role=='forward-preserving' and type(output.grad_fn) is not _Attention._backward_cls:
                raise ValueError('Exact custom attention output/root required.')
            self.output,self.root_node=output,output.grad_fn
            self.output_fingerprint=fingerprint(output);self.record['phase']='forward-returned';self._check_tokens()
            return output
        except BaseException as error:
            self.record.update(phase='failed',error_type=type(error).__name__);raise

    def _admit_backward(self):
        self._common();outer=self._outer
        if (self.record['phase']!='backward-running' or outer is None or outer['state']!='running'
                or outer['root_calls']!=1 or self._graph_id()!=outer['graph_id']):
            raise ValueError('Backward belongs to its one observed outer GraphTask.')
        return outer

    def _math_call(self,q,k,v):
        self._common(math_context=True);outer=self._outer
        if (self.role!='forward-preserving' or self.math_calls or outer is None or outer['state']!='running'
                or outer['root_calls']!=1 or self._graph_id()!=outer['graph_id'] or not torch.is_grad_enabled()
                or flags()!={'flash':False,'mem_efficient':False,'math':True,'cudnn':False}
                or [fingerprint(t) for t in (q,k,v)]!=self.input_fingerprints or aliases((q,k,v))!=self.input_aliases
                or any(self._hook_map(t) for t in (q,k,v))):
            raise ValueError('One owned original-layout public math recomputation only.')
        self.math_calls+=1
        return self._sdpa(q,k,v,dropout_p=0.,is_causal=False,attn_mask=None)

    def _remove_outer_hooks(self):
        errors=[]
        for handle in reversed(self.handles):
            try:handle.remove()
            except BaseException as error:errors.append(error)
        self.handles=[]
        self.expected_hooks=[{} for _ in self.values]
        if self.scope is not None:
            self.scope['root_hook_removed']=not errors;self.scope['leaf_hooks_removed']=not errors
        return errors

    def gradient(self,output,values,incoming,*,create_graph=False,retain_graph=False):
        self._creator();self._check_tokens()
        if (create_graph is not False or retain_graph is not False or self.gradient_used or not self.forward_used
                or output is not self.output or values is not self.values or incoming is not self.incoming
                or self.record['phase']!='forward-returned' or self._graph_id()!=-1 or not torch.is_grad_enabled()):
            raise ValueError('One exact output/QKV/cotangent token; first-order only.')
        self.gradient_used=True;self.record['phase']='backward-running'
        scope={'state':'running','graph_id':None,'root_calls':0,'leaf_hooks':[],
               'caller_thread':self.thread,'root_hook_removed':False,'leaf_hooks_removed':False}
        self.scope=self._outer=scope;problem=None
        def root_hook(gradients):
            with self.hook_lock:
                self._common();graph=self._graph_id()
                if (self._outer is not scope or scope['state']!='running' or graph<0 or scope['root_calls']
                        or len(gradients)!=1 or fingerprint(gradients[0])!=self.incoming_fingerprint):
                    raise ValueError('One exact output root and original arriving cotangent required.')
                scope.update(graph_id=graph,root_calls=1,root_thread=threading.get_ident())
            return None
        try:
            self.handles.append(self.root_node.register_prehook(root_hook))
            for index,value in enumerate(self.values):
                def leaf_hook(gradient,index=index):
                    with self.hook_lock:
                        self._admit_backward()
                        if any(h['index']==index for h in scope['leaf_hooks']):raise ValueError('Outer leaf observed twice.')
                        scope['leaf_hooks'].append({'index':index,'graph_id':self._graph_id(),'thread':threading.get_ident()})
                    return None
                handle=value.register_hook(leaf_hook);self.handles.append(handle)
                hooks=self._hook_map(value)
                if set(hooks)!={handle.id} or hooks[handle.id] is not leaf_hook:
                    raise ValueError('Only the owned outer leaf hook is permitted.')
                self.expected_hooks[index]=hooks
            result=self._grad(self.output,self.values,self.incoming,create_graph=False,retain_graph=False)
            if (type(result) is not tuple or len(result)!=3 or scope['root_calls']!=1
                    or sorted(h['index'] for h in scope['leaf_hooks'])!=[0,1,2]
                    or any(h['graph_id']!=scope['graph_id'] for h in scope['leaf_hooks']) or self._graph_id()!=-1):
                raise ValueError('Complete actual outer root and all three leaves required.')
            if any(not isinstance(t,torch.Tensor) or t.dtype!=torch.float32 or t.shape!=v.shape or t.device!=v.device
                   or t.requires_grad or not torch.isfinite(t).all() for t,v in zip(result,self.values)):
                raise ValueError('Finite first-order complete component returns required.')
            row=self.calls[0]
            if self.role=='forward-preserving':
                if (self.math_calls!=1 or not row.get('backward_complete') or not row.get('hooks_removed')
                        or row.get('outer_graph_id')!=scope['graph_id'] or type(row.get('inner_graph_id')) is not int
                        or row['inner_graph_id']<0 or row['inner_graph_id']==scope['graph_id']
                        or row.get('inner_root_calls')!=1 or sorted(h['index'] for h in row.get('leaf_hooks',[]))!=[0,1,2]
                        or any(h['graph_id']!=row['inner_graph_id'] for h in row['leaf_hooks'])):
                    raise ValueError('Distinct actual inner math task/root/three leaves and cleanup required.')
            elif self.math_calls:
                raise ValueError('Native component cannot execute a custom math VJP.')
            self._check_tokens();row.update(backward_complete=True,returned_component=[fingerprint(t) for t in result])
        except BaseException as error:
            problem=error;scope.update(state='failed',error_type=type(error).__name__)
            self.record.update(phase='failed',error_type=type(error).__name__);raise
        finally:
            errors=self._remove_outer_hooks();self._outer=None
            if errors:
                scope['state']='failed';self.record.update(phase='failed',error_type=type(errors[0]).__name__)
                if problem is not None:problem.add_note('Outer hook cleanup also failed: '+type(errors[0]).__name__)
                else:raise errors[0]
        scope['state']='complete';self.record['phase']='computed';return result

    def snapshot(self):
        result=copy.deepcopy(dict(self.record,calls=self.calls,outer=self.scope))
        if len((json.dumps(result,sort_keys=True,allow_nan=False)+'\\n').encode())>64*1024:
            raise ValueError('Bounded component ownership snapshot exceeded; no truncation.')
        return result

    def __exit__(self,kind,error,tb):
        self._creator()
        if self.closed or not self.entered or self._outer is not None or self._graph_id()!=-1:
            raise ValueError('Creator may close once, outside every running GraphTask.')
        try:
            self._check_tokens()
            if (not torch.equal(torch.get_rng_state(),self.rng_cpu) or not torch.equal(torch.cuda.get_rng_state(),self.rng_cuda)):
                raise ValueError('Global CPU/CUDA RNG changed.')
            if error is None and (self.record['phase']!='computed' or self.scope is None or self.scope['state']!='complete'
                    or not self.scope['root_hook_removed'] or not self.scope['leaf_hooks_removed'] or self.native_calls!=1):
                raise ValueError('Incomplete component route; no retry.')
            self.record.update(route_complete=error is None,phase='closed' if error is None else 'failed')
            if error is not None:self.record['error_type']=type(error).__name__
        except BaseException as failure:
            self.record.update(route_complete=False,phase='failed',error_type=type(failure).__name__);raise
        finally:
            self.active=False;self.closed=True;self.values=();self.incoming=self.output=self.root_node=None
            self.rng_cpu=self.rng_cuda=None
        return False
'''


def _node(tree,name,kind):
    return next(n for n in ast.walk(tree) if isinstance(n,kind) and n.name==name)


def prototype(cpu_bytes,target_bytes):
    """Return unbound definitions and source audit only; never import Torch."""
    if (hashlib.sha256(cpu_bytes).hexdigest()!=CPU_SHA or hashlib.sha256(target_bytes).hexdigest()!=TARGET_SHA):
        raise ValueError('Exact frozen candidate arithmetic source required.')
    cpu_tree=ast.parse(cpu_bytes);target_tree=ast.parse(target_bytes)
    cpu_class=copy.deepcopy(_node(cpu_tree,'_OriginalForwardMathVJP',ast.ClassDef))
    cpu_class.body=[copy.deepcopy(_node(cpu_class,'backward',ast.FunctionDef))]
    attention=copy.deepcopy(_node(target_tree,'_Attention',ast.ClassDef))
    helpers=[copy.deepcopy(_node(cpu_tree,n,ast.FunctionDef)) for n in ('flags','aliases')]
    helpers.append(copy.deepcopy(_node(target_tree,'fingerprint',ast.FunctionDef)))
    constants=ast.parse(textwrap.dedent("""
        cpu=SimpleNamespace(hashlib=hashlib,flags=flags)
        _public_math_backward=_OriginalForwardMathVJP.backward
        _LOCK=threading.RLock()
        _CLAIM=None
    """)).body
    # CPython 3.11 emits different attribute-call instructions for names known
    # as imported modules. Keep those frozen compiler declarations in a literal
    # false branch: they affect symbol analysis but compile to NO import opcode.
    # All runtime namespaces remain supplied explicitly, never imported here.
    compiler_imports=[copy.deepcopy(n) for n in cpu_tree.body if isinstance(n,ast.Import)
        and len(n.names)==1 and n.names[0].name in ('torch','threading')]
    compiler_imports += [copy.deepcopy(n) for n in target_tree.body if isinstance(n,ast.ImportFrom)
        and n.module=='models' and any(a.asname=='cpu' for a in n.names)]
    if len(compiler_imports)!=3:raise ValueError('Frozen compiler module-binding context changed.')
    compiler_context=ast.If(test=ast.Constant(value=False),body=compiler_imports,orelse=[])
    module=ast.fix_missing_locations(ast.Module(body=[compiler_context,*helpers,cpu_class,*constants,attention,
        *ast.parse(textwrap.dedent(ROUTE_SOURCE)).body],type_ignores=[]))
    def digest(n):return hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
    frozen={'public_math_backward_AST':digest(_node(cpu_class,'backward',ast.FunctionDef)),
            'original_forward_AST':digest(_node(attention,'forward',ast.FunctionDef)),
            'custom_backward_bridge_AST':digest(_node(attention,'backward',ast.FunctionDef))}
    audit={'kind':'pgd-split-component-unbound-source-audit-v1','cpu_source_sha256':CPU_SHA,'target_source_sha256':TARGET_SHA,
        'arithmetic_bodies_copied_exactly':True,'arithmetic_AST':frozen,'generated_module_has_imports':True,
        'compiler_only_imports_in_literal_false_branch':3,'generated_runtime_imports':False,
        'runtime_namespace_unbound':True,'target_runner_available':False,'scientific_qualified':False}
    return module,audit
