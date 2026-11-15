"""Pinned UNBOUND boundary/component providers; no scientific imports/execution.

Only the future reviewed worker namespace can supply Torch, the new numerical
readers, typed I/O and runtime bindings. Current checks compile/inspect this source.
"""
import ast
import copy
import hashlib
import textwrap
from models.predictor_pgd_split_numerical_v1 import Edits
from models import predictor_pgd_split_component_v1 as component

PINS={
 'observer':'5e6e00593f3befb1979b37f0fec155bb35322c9e4c196a7abe968d2f0ff68047',
 'boundary':'cdd02a830a65a2ab2185c9b5bc59e1026570f50542d78fe480750ab198198f06',
 'core':'70cc9e74c70980f598496783ea4c303f3d10b61537a329f9cc8ae94ff5dff955',
 'startup':'8359f9a1f8da77eeb13d02a55d56982e2998db67f48199a577f62ef95329dae3',
 'raw':'6da98072ac5f38e1939f0d470970b7f68b6d63b7d475c23ac5d4d3e60c6ac9e4',
 'cpu':component.CPU_SHA,'target':component.TARGET_SHA,
 'route_builder':'0ccb44c9978cfd352c04619abfd450e16e1c91c529df4a82ce63f4017b1e0533'}

ATTACH_SOURCE='''
class BoundOriginalBoundary:
    def __init__(self,route,source_value,source_objective_index,control_sha256,folder,result_store):
        import torch
        from models.cuda_profile import tensor_signature
        if (type(source_objective_index) is not int or source_objective_index!=0 or route.mode!='original'
                or route.phase!='EOT8-before' or not route.active or route.tiny
                or not isinstance(source_value,torch.Tensor) or source_value.device.type!='cuda'
                or source_value.dtype!=torch.float32 or tuple(source_value.shape)!=(1,3,16,256,256)
                or not source_value.is_leaf or not source_value.requires_grad or source_value.grad is not None):
            raise ValueError('Only the actual new original objective-zero input/route may attach.')
        self.value=source_value;self.signature=tensor_signature(source_value);self.version=source_value._version
        self.pid,self.thread=os.getpid(),threading.get_ident();self.used=self.closed=False
        folder=Path(folder);folder.mkdir(mode=0o700,exist_ok=False)
        self.inner=NativeSDPAObserver(route,split_boundary.BoundaryStore(folder,result_store=result_store),control_sha256=control_sha256)
        self.inner.state['split_lineage']={'kind':'pgd-split-original-objective-zero-source-v1',
            'source_objective_index':0,'source_value_id':id(source_value),'source_value_version':self.version,
            'source_input_signature':copy.deepcopy(self.signature),'frozen_observer_sha256':
                '5e6e00593f3befb1979b37f0fec155bb35322c9e4c196a7abe968d2f0ff68047'}
    def check(self):
        from models.cuda_profile import tensor_signature
        if ((os.getpid(),threading.get_ident())!=(self.pid,self.thread) or self.closed
                or self.value._version!=self.version or self.value.grad is not None
                or tensor_signature(self.value)!=self.signature):
            raise ValueError('Original full-input token/version/bytes/.grad changed around observation.')
    def __enter__(self):
        self.check()
        if self.used:raise ValueError('One original boundary scope; no retry.')
        self.used=True
        try:self.inner.__enter__()
        except BaseException:self.closed=True;self.value=None;raise
        return self
    def __exit__(self,kind,error,tb):
        primary=error
        try:self.check()
        except BaseException as changed:
            if primary is None:primary=changed
            else:primary.add_note('Boundary source check also failed: '+type(changed).__name__)
        try:
            self.inner.__exit__(None if primary is None else type(primary),primary,None if primary is None else primary.__traceback__)
        except BaseException as cleanup:
            if primary is None:primary=cleanup
            else:primary.add_note('Boundary cleanup also failed: '+type(cleanup).__name__)
        finally:self.closed=True;self.value=None
        if error is None and primary is not None:raise primary
        return False


def attach(route,*,source_value,source_objective_index,control_sha256,folder):
    return BoundOriginalBoundary(route,source_value,source_objective_index,control_sha256,folder,result_store)
'''

PROBE_SOURCE='''
def load_new_source(args):
    # All saved numerical/native gates run on THIS new result, not historical
    # arrays/cohorts. These are saved-value checks, never additional objectives.
    root=Path(args.result_root);numerical_folder=root/'results/numerical'
    receipt=numerical.verify_worker(numerical_folder,control_sha256=args.control_sha256)
    original,arrays=numerical.load_objective(numerical_folder/'objective-0',args.control_sha256,0)
    worker=json.loads(private_bytes(root/'results/run-status.json',256*1024,args.numerical_sha256))
    private_bytes(numerical_folder/'objective-0/numerical.json',256*1024,args.original_sha256)
    boundary_bytes=private_bytes(root/'results/boundary/boundary.json',128*1024,args.boundary_sha256)
    cap=boundary.load(root/'results/boundary',control_sha256=args.control_sha256)
    proof={'calls':[json.loads(line) for line in private_bytes(numerical_folder/'objective-0/native-calls.jsonl',1152*2048,
        original['proof_files']['native-calls.jsonl']['sha256']).splitlines()]}
    scoped=json.loads(private_bytes(numerical_folder/'objective-0/scopes.json',64*1024,original['proof_files']['scopes.json']['sha256']))
    proof.update(scopes=scoped['scopes'],refusals=scoped['refusals'])
    pixel=hashlib.sha256(arrays['pixels'].tobytes()).hexdigest()
    joined=lineage.join(worker,original,cap,proof,control_sha256=args.control_sha256,boundary_sha256=args.boundary_sha256,pixel_sha256=pixel)
    if (receipt['worker_sha256']!=args.numerical_sha256 or receipt.get('numerical_scoped') is not True
            or receipt.get('representative_backend_scoped') is not False):
        raise ValueError('New complete numerical receipt must match its immutable source bytes.')
    private_bytes(root/'results/run-status.json',256*1024,args.numerical_sha256)
    private_bytes(numerical_folder/'objective-0/numerical.json',256*1024,args.original_sha256)
    private_bytes(root/'results/boundary/boundary.json',128*1024,args.boundary_sha256)
    return cap,joined


def prepare_operands(args,cap):
    if flags()!=cap['selected_call']['actual_call']['entry_flags']:
        raise ValueError('Captured original attention flags changed before reconstruction; no fallback.')
    values,metadata=_operands(Path(args.result_root)/'results',cap,'cuda')
    return tuple(values[n] for n in ('q','k','v')),values['grad_output'],metadata


def compute_once(route,record_function,observe):
    with record_function('fixed-SDPA/forward'):
        output=observe('forward',route.attention)
    with record_function('fixed-SDPA/backward'):
        gradients=observe('backward',route.gradient,output,route.values,route.incoming,create_graph=False,retain_graph=False)
    observe('cuda_sync',torch.cuda.synchronize)
    return dict(zip(('output','dq','dk','dv'),(output.detach(),*(t.detach() for t in gradients))))


def make_profile():
    return torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,with_stack=False,profile_memory=False)


def raw_bytes(profile):
    return raw_capture.frozen.encoded(raw_capture.capture(profile,torch_version=str(torch.__version__)))


def serialize_return(value):
    import io
    import numpy as np
    if (not isinstance(value,torch.Tensor) or tuple(value.shape)!=(1,12,2048,32)
            or value.dtype!=torch.float32 or value.device.type!='cuda' or value.requires_grad
            or not torch.isfinite(value).all()):raise ValueError('One actual finite first-order component return required.')
    # Output diagnostics add no objective, cast or reference comparison.
    a=value.detach().cpu().contiguous().numpy();stream=io.BytesIO();np.save(stream,a,allow_pickle=False)
    data=stream.getvalue()
    if len(data)>4*2**20:raise ValueError('Original component output bound; no truncation.')
    return data
'''


def build(inputs):
    if type(inputs) is not dict or set(inputs)!=set(PINS):raise ValueError('Exact frozen provider source set required.')
    for key,want in PINS.items():
        if type(inputs[key]) is not bytes or hashlib.sha256(inputs[key]).hexdigest()!=want:raise ValueError('Frozen provider changed: '+key)
    observer=ast.parse(inputs['observer']);oe=Edits(observer)
    oe.apply('from models.predictor_sdpa_capture_record_v1 import KIND,scope_check','''
if False:
    from models.predictor_sdpa_capture_record_v1 import KIND,scope_check
KIND=split_boundary.KIND
scope_check=split_boundary.scope_check
''')
    observer_edits=oe.verify()
    observer.body.extend(ast.parse(textwrap.dedent(ATTACH_SOURCE)).body)
    boundary=ast.parse(inputs['boundary']);be=Edits(boundary)
    be.apply("KIND='original-SDPA-one-boundary-observation-v1'","KIND='pgd-split-original-boundary-v1'")
    be.apply("LIMITS={'boundary.json':128*1024,'boundary-complete.json':16*1024,'boundary-failed.json':16*1024,'attention-calls.jsonl':8*MiB,'attention-scopes.json':64*1024,**{n+'.pt':16*MiB for n in TENSORS}}",
        "LIMITS={'boundary.json':128*1024,'boundary-complete.json':16*1024,'boundary-failed.json':16*1024,'attention-calls.jsonl':1152*2048,'attention-scopes.json':64*1024,**{n+'.pt':4*MiB for n in TENSORS}}")
    # Keep the exact observer, save/load/rebuild and scope bodies. Only I/O
    # publication changes to the new typed bytes store and explicit 4 MiB cap.
    cls=next(n for n in boundary.body if isinstance(n,ast.ClassDef) and n.name=='BoundaryStore')
    old_ctor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    new_ctor=copy.deepcopy(old_ctor);new_ctor.args.kwonlyargs.append(ast.arg(arg='result_store'));new_ctor.args.kw_defaults.append(None)
    new_ctor.body.append(ast.parse('self.result_store=result_store').body[0])
    be.apply(ast.unparse(old_ctor),ast.unparse(new_ctor))
    old_write=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_write')
    new_write='''
def _write(self,name,write):
    import io
    if self.closed or self.poisoned or name not in LIMITS:raise ValueError('Closed/failed or unknown split boundary record.')
    self._check()
    try:
        data=io.BytesIO();write(data);raw=data.getvalue()
        if len(raw)>LIMITS[name]:raise ValueError('Split boundary serialization ceiling.')
        self._check()
        relative=(self.folder/name).relative_to(self.result_store.root).as_posix()
        item=self.result_store.write_bytes(relative,raw);self.files[name]=item
        return dict(item)
    except BaseException:self.poisoned=True;raise
'''
    be.apply(ast.unparse(old_write),new_write)
    be.apply('value.numel()*4+4096>16*MiB','value.numel()*4+4096>4*MiB')
    boundary_edits=be.verify()
    raw=ast.parse(inputs['raw']);re=Edits(raw)
    re.apply('MAX_BYTES=frozen.MAX_BYTES','MAX_BYTES=4*2**20');raw_edits=re.verify()
    # Error text reflects the NEW envelope; the original raw fields/API/rows
    # and whole-sequence no-truncation policy are retained.
    for n in ast.walk(raw):
        if isinstance(n,ast.Constant) and type(n.value) is str:n.value=n.value.replace('existing 8 MiB ceiling','split 4 MiB ceiling')
    core=ast.parse(inputs['core']);operands=copy.deepcopy(next(n for n in core.body if isinstance(n,ast.FunctionDef) and n.name=='_operands'))
    providers=ast.parse(textwrap.dedent(PROBE_SOURCE))
    compiler_imports=[copy.deepcopy(n) for n in core.body if isinstance(n,ast.ImportFrom) and n.module=='models'
        and any(a.asname in ('native','rec') for a in n.names)]
    if len(compiler_imports)!=2:raise ValueError('Frozen operand compiler context changed.')
    providers.body[:0]=[ast.If(test=ast.Constant(value=False),body=compiler_imports,orelse=[]),operands]
    route,route_audit=component.prototype(inputs['cpu'],inputs['target'])
    modules={'observer':observer,'boundary':boundary,'providers':providers,'startup':ast.parse(inputs['startup']),'raw':raw,'route':route}
    for k,t in modules.items():modules[k]=ast.fix_missing_locations(t)
    return modules,{'kind':'pgd-split-probe-provider-unbound-source-audit-v1','source_pins':dict(PINS),
        'observer_reversible_edits':observer_edits,'boundary_reversible_edits':boundary_edits,'raw_reversible_edits':raw_edits,
        'native_observer_class_AST_unchanged':True,'native_scope_check_and_tensor_load_rebuild_AST_unchanged':True,
        'device_operands_AST_unchanged':True,'startup_module_AST_unchanged':True,'route':route_audit,
        'compute_forwards':1,'compute_outer_VJPs':1,'extra_hybrid_inner_math_VJPs':1,
        'warmups_replays_or_reference_builds':0,'runtime_namespace_unbound':True,'target_runner_available':False,
        'actual_tensor_or_gradient_execution':False,'scientific_qualified':False}
