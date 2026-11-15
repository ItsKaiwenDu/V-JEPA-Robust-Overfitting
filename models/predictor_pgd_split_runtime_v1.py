"""LOCAL ONLY deterministic source assembly. No generated runtime is loaded here.

Builds concrete, separately named worker namespaces from pinned source bytes.
There is no CLI, subprocess, upload, authorization, model import or runtime call.
"""
import ast
import copy
import hashlib
import json
from pathlib import PurePosixPath
from models import predictor_pgd_split_numerical_v1 as numerical
from models import predictor_pgd_split_probe_sources_v1 as probes

# Filled from the reviewed local source contract, not supplied by a caller.
CONTRACT_SHA='4679b4c55a1f34232d9813bbbc89d2a27260fa6e7ddb7a831836f82dfef06b6d'
NUM_INPUTS={'worker':'predictor_pgd_target_worker_v1','records':'predictor_pgd_target_records_v1',
 'progress':'post_gradient_progress_v1','writer':'predictor_forward_preserving_records_v1'}
PROBE_INPUTS={'observer':'predictor_sdpa_observer_v1','boundary':'predictor_sdpa_capture_record_v1',
 'core':'predictor_sdpa_component_v7','startup':'predictor_sdpa_component_startup_v2','raw':'predictor_sdpa_raw_capture_v1',
 'cpu':'predictor_forward_preserving_cpu_v1','target':'predictor_forward_preserving_target_v1',
 'route_builder':'predictor_pgd_split_component_v1'}

def generated(role):return 'models/_pgd_split_'+role+'_v1.py'
def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(value).hexdigest()
def text_module(tree):return (ast.unparse(ast.fix_missing_locations(tree))+'\n').encode()
def parse(source):return ast.parse(source)

def prepend(tree,source):
    t=copy.deepcopy(tree);pos=1 if t.body and isinstance(t.body[0],ast.Expr) and isinstance(t.body[0].value,ast.Constant) and isinstance(t.body[0].value.value,str) else 0
    t.body[pos:pos]=parse(source).body
    return ast.fix_missing_locations(t)

IO_SOURCE='''
"""Single-owner byte I/O facade for this run's separate results/logs namespace."""
import hashlib
import os
from pathlib import Path
import stat
import threading
from models import predictor_eot8_record_v2 as original_json
from models.predictor_pgd_split_store_v1 import Inventory,Store,identity
_json=original_json._json
_parse=original_json._parse
_STORE=None
_CLAIMED=False

class SerialStore:
    # Real autograd hooks may publish on a different thread. Each serialized
    # operation gets the unchanged writer's current-thread ownership check;
    # the committed metadata/poison state remains shared by this one process.
    def __init__(self,inventory):
        self.inventory,self.root=inventory,inventory.root
        self.pid=os.getpid();self.lock=threading.RLock();self.poisoned=False
        self.written={};self.busy=False
    def writer(self):
        if self.pid!=os.getpid() or self.poisoned:raise ValueError('Owned unpoisoned serial process writer required.')
        writer=Store(self.inventory);writer.written=self.written
        return writer
    def check(self):
        with self.lock:return self.writer().check()
    def write_bytes(self,name,data):
        with self.lock:
            writer=self.writer()
            if self.busy:
                self.poisoned=True
                raise ValueError('Recursive publication refused; preserve failed namespace.')
            self.busy=True
            try:return writer.write_bytes(name,data)
            except BaseException:
                self.poisoned=True
                raise
            finally:self.busy=False

def bind(root):
    global _STORE,_CLAIMED
    if _CLAIMED:raise ValueError('One result namespace binding per fresh process; no reset.')
    _CLAIMED=True
    _STORE=SerialStore(Inventory(root,base_bytes=BASE_BYTES,design_bytes=DESIGN_BYTES))
    _STORE.check()
    return _STORE

def bound_store():
    if _STORE is None:raise ValueError('Result namespace is unbound.')
    _STORE.check()
    return _STORE

def relative(folder,name):
    if type(name) is not str or Path(name).name!=name or name in ('','.','..'):
        raise ValueError('One fixed result filename required.')
    store=bound_store();path=Path(folder).absolute()/name
    key=path.relative_to(store.root).as_posix()
    if key not in store.inventory.spec:raise ValueError('Unknown split result path: '+key)
    return key

def read(folder,name,limit):
    key=relative(folder,name);store=bound_store()
    if type(limit) is not int or limit<0:raise ValueError('Typed read cap required.')
    cap=min(limit,store.inventory.spec[key]['max_bytes'])
    row=store.inventory.scan()['files'][key];path=store.root/key
    if row['bytes']>cap:raise ValueError('Typed source read cap exceeded.')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:
        before=os.fstat(stream.fileno());data=stream.read(cap+1);after=os.fstat(stream.fileno())
    named=path.lstat()
    def row_id(s):return (identity(s),s.st_size,s.st_mtime_ns)
    want=(row['identity'],row['bytes'],row['mtime_ns'])
    if (row_id(before)!=want or row_id(after)!=want or row_id(named)!=want or len(data)!=row['bytes']
            or store.inventory.scan()['files'].get(key)!=row):
        raise ValueError('Typed source replaced or changed during read.')
    return data

def sha_file(path):
    path=Path(path);key=relative(path.parent,path.name)
    return hashlib.sha256(read(path.parent,path.name,bound_store().inventory.spec[key]['max_bytes'])).hexdigest()

def _atomic(folder,name,chunks,limit):
    key=relative(folder,name);store=bound_store()
    if type(limit) is not int or limit<0:raise ValueError('Publication cap required.')
    cap=min(limit,store.inventory.spec[key]['max_bytes']);data=bytearray()
    for chunk in chunks:
        if type(chunk) is not bytes or len(data)+len(chunk)>cap:raise ValueError('Typed whole byte publication cap exceeded.')
        data.extend(chunk)
    return store.write_bytes(key,bytes(data))
'''

ENTRY_SOURCE='''
"""Concrete but UNEXECUTED split worker bindings; future supervisor owns launch."""
from argparse import Namespace
import hashlib
import math
from pathlib import Path
import torch
from models import _pgd_split_io_v1 as io
from models import _pgd_split_numerical_v1 as numerical_worker
from models import _pgd_split_records_v1 as numerical
from models import _pgd_split_startup_v1 as startup
from models import _pgd_split_providers_v1 as provider
from models import _pgd_split_route_v1 as routes
from models.predictor_pgd_split_probe_job_v1 import ComponentJob
from models.predictor_pgd_split_lineage_v1 import DESIGN_SHA,sha
from models.training import runtime_contract
_USED=False

def claim(args,names):
    global _USED
    if _USED:raise ValueError('One entry per fresh worker namespace; no retry/reset.')
    _USED=True
    if type(args) is not Namespace or set(vars(args))!=set(names):raise ValueError('Exact bound worker arguments; no injectable ports.')
    if not sha(args.control_sha256):raise ValueError('Exact control digest required.')
    root=Path(args.result_root)
    if not root.is_absolute() or root.resolve(strict=True)!=root:raise ValueError('Canonical private result namespace required.')
    return io.bind(root)

def numerical_entry(args):
    store=claim(args,('result_root','control_sha256','packet','upstream','checkpoint','proposal','output','device','tiny'))
    if Path(args.output)!=store.root/'results/numerical' or args.device!='cuda' or args.tiny is not False:
        raise ValueError('Only the new numerical output/CUDA domain is bound.')
    result=numerical_worker.unbound_numerical_worker(args)
    receipt=numerical.verify_worker(store.root/'results/numerical',control_sha256=args.control_sha256)
    store.check()
    return result,receipt

def component_entry(args):
    store=claim(args,('result_root','control_sha256','numerical_sha256','original_sha256','boundary_sha256','role','deadline'))
    if (args.role not in ('native','forward-preserving') or any(not sha(getattr(args,k)) for k in ('numerical_sha256','original_sha256','boundary_sha256'))
            or type(args.deadline) not in (int,float) or not math.isfinite(args.deadline)):
        raise ValueError('Exact role/source digests and continuing parent deadline required.')
    binding={'design_sha256':DESIGN_SHA,'control_sha256':args.control_sha256,'boundary_sha256':args.boundary_sha256,
        'probe_index':0 if args.role=='native' else 1}
    state={};job=None
    def initialize():
        value=startup.initialize('cuda');torch.set_num_threads(4)
        if torch.get_num_threads()!=4:raise ValueError('Original four worker threads required.')
        state['startup']=value
        return value
    def source():
        cap,joined=provider.load_new_source(args)
        worker_bytes=io.read(store.root/'results','run-status.json',256*1024)
        if hashlib.sha256(worker_bytes).hexdigest()!=args.numerical_sha256:
            raise ValueError('Numerical runtime header hash changed after source verification.')
        worker=io._parse(worker_bytes)
        expected={k:v for k,v in worker['baseline']['runtime'].items() if k!='model_execution'}
        actual=runtime_contract('cuda')
        if actual!=expected or actual.get('threads')!=4:raise ValueError('Captured original runtime/precision changed before operands.')
        state.update(cap=cap,runtime=actual)
        job.state['runtime']=actual
        job.state['runtime_source_numerical_sha256']=args.numerical_sha256
        return cap,joined
    def restore():
        startup.validate(state['startup'],pid=job.state['pid'],device='cuda')
        if (runtime_contract('cuda')!=state['runtime'] or torch.get_num_threads()!=4
                or provider.flags()!=state['cap']['selected_call']['actual_call']['entry_flags']):
            raise ValueError('Original component runtime/precision/attention flags not restored.')
        store.check()
    ports={'startup':initialize,'source':source,'operands':lambda cap:provider.prepare_operands(args,cap),
        'route':lambda role,values,incoming,binding:routes.ComponentRoute(role,values,incoming,binding=binding),
        'profile':provider.make_profile,'compute':lambda route,observe:provider.compute_once(route,torch.profiler.record_function,observe),
        'raw':provider.raw_bytes,'serialize':provider.serialize_return,'restore':restore,'allocator':torch.cuda.max_memory_allocated}
    job=ComponentJob(args.role,binding,store,ports,deadline=args.deadline,evidence_origin='SOURCE-PINNED-SPLIT-RUNTIME-V1; LIVE-ADMISSIONS-PENDING')
    return job.execute()
'''


def imports(tree):
    """Literal project imports, including lazy functions; false compiler branches skip."""
    found=set();externals=set()
    def visit(n):
        if isinstance(n,ast.If) and isinstance(n.test,ast.Constant) and n.test.value is False:
            for item in n.orelse:visit(item)
            return
        if isinstance(n,ast.Import):
            for a in n.names:
                if a.name.split('.')[0] in ('models','attacks'):found.add(a.name)
                else:externals.add(a.name.split('.')[0])
        elif isinstance(n,ast.ImportFrom):
            if n.level:raise ValueError('Relative project source imports need explicit review.')
            name=n.module or ''
            if name in ('models','attacks'):
                found.update(name+'.'+a.name for a in n.names)
            elif name.split('.')[0] in ('models','attacks'):found.add(name)
            else:externals.add(name.split('.')[0])
        for child in ast.iter_child_nodes(n):visit(child)
    visit(tree)
    return found,externals


def validate_inputs(inputs,contract_bytes):
    if type(contract_bytes) is not bytes or digest(contract_bytes)!=CONTRACT_SHA:raise ValueError('Exact runtime source contract required.')
    contract=json.loads(contract_bytes)
    if type(inputs) is not dict or set(inputs)!=set(contract['source_sha256']):raise ValueError('Exact complete runtime source set required.')
    for name,want in contract['source_sha256'].items():
        p=PurePosixPath(name)
        if (p.is_absolute() or '..' in p.parts or str(p)!=name or type(inputs[name]) is not bytes or digest(inputs[name])!=want):
            raise ValueError('Pinned runtime source changed: '+name)
    return contract


def build(inputs,contract_bytes):
    """Emit pinned source bytes only; no import/exec/module injection occurs."""
    contract=validate_inputs(inputs,contract_bytes)
    nums,na=numerical.build({k:inputs['models/'+v+'.py'] for k,v in NUM_INPUTS.items()})
    ps,pa=probes.build({k:inputs['models/'+v+'.py'] for k,v in PROBE_INPUTS.items()})
    modules={}
    # Fixed records bind a private facade; no legacy global/module is mutated.
    records=copy.deepcopy(nums['records']);ed=numerical.Edits(records)
    ed.apply('from models import predictor_forward_preserving_records_v1 as frozen','from models import _pgd_split_facade_v1 as frozen')
    # Source arithmetic and saved-value validators remain bytecode/AST exact;
    # only serialization/publication goes through the new typed namespace.
    oldsave=next(n for n in records.body if isinstance(n,ast.FunctionDef) and n.name=='save_tensor')
    new=numerical.Edits(copy.deepcopy(oldsave))
    new.apply("""
with partial.open('xb') as f:
    import os
    os.fchmod(f.fileno(), 0o600);np.save(f, a, allow_pickle=False);f.flush();os.fsync(f.fileno())
if partial.stat().st_size > cap:
    raise ValueError('Serialized typed file cap; preserve partial.')
partial.rename(p)
""","""
import io
stream=io.BytesIO();np.save(stream,a,allow_pickle=False)
frozen.atomic._atomic(folder,p.name,[stream.getvalue()],cap)
""")
    tensor_edits=new.verify()
    ed.apply(ast.unparse(oldsave),ast.unparse(new.tree));record_edits=ed.verify()
    records=prepend(records,'from models import _pgd_split_progress_v1 as progress')
    modules['records']=records
    facade=parse(inputs['models/predictor_forward_preserving_records_v1.py'])
    facade.body+=parse('from models import _pgd_split_io_v1 as atomic\nread=atomic.read\nsha_file=atomic.sha_file').body
    modules['facade']=facade
    modules['numerical']=prepend(nums['numerical'],'''from models import _pgd_split_records_v1 as records
from models import _pgd_split_progress_v1 as progress
from models import _pgd_split_observer_v1 as boundary''')
    modules['progress']=nums['progress']
    modules['boundary']=ps['boundary']
    observer=copy.deepcopy(ps['observer']);oe=numerical.Edits(observer)
    oe.apply('return BoundOriginalBoundary(route,source_value,source_objective_index,control_sha256,folder,result_store)',
        'return BoundOriginalBoundary(route,source_value,source_objective_index,control_sha256,folder,bound_store())')
    observer_edits=oe.verify()
    modules['observer']=prepend(observer,'''from models import _pgd_split_boundary_v1 as split_boundary
from models._pgd_split_io_v1 import bound_store''')
    modules['providers']=prepend(ps['providers'],'''import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch as provider_torch
torch=provider_torch
from models import _pgd_split_boundary_v1 as boundary
from models import _pgd_split_boundary_v1 as native
from models import _pgd_split_records_v1 as numerical
from models import _pgd_split_raw_v1 as raw_capture
from models import predictor_pgd_split_lineage_v1 as lineage
from models.predictor_pgd_split_probe_job_v1 import private_bytes
from models.predictor_eot8_adapter_v1 import flags
rec=SimpleNamespace(MiB=2**20)''')
    modules['route']=prepend(ps['route'],'''import copy
import hashlib
import json
import os
import platform
import threading
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
DESIGN_SHA='c5daae30cbe4043cf4d5265de2a2a7420d015ab1e244cc3d74fcc93b5f0186af'
KIND='pgd-split-one-attention-route-v1' ''')
    modules['startup']=ps['startup'];modules['raw']=ps['raw']
    io_tree=parse(IO_SOURCE)
    io_tree.body+=parse('BASE_BYTES='+repr(inputs['configs/predictor_pgd_target_v1.json'])+'\nDESIGN_BYTES='+repr(inputs['configs/predictor_pgd_split_profile_v1.json'])).body
    modules['io']=io_tree;modules['entry']=parse(ENTRY_SOURCE)
    output={generated(k):text_module(t) for k,t in modules.items()}
    # Resolve every literal project import against exact bytes, never sys.path.
    pool={k:v for k,v in inputs.items() if k.endswith('.py')}|output
    queue=list(output);seen=set();edges={};external=set()
    while queue:
        name=queue.pop()
        if name in seen:continue
        seen.add(name);tree=parse(pool[name]);deps,ext=imports(tree);external|=ext
        edges[name]=[]
        for dep in sorted(deps):
            p=dep.replace('.','/')+'.py'
            if p not in pool:raise ValueError('Unresolved pinned project dependency: '+p+' from '+name)
            edges[name].append(p);queue.append(p)
            package=p.split('/')[0]+'/__init__.py'
            if package not in pool:raise ValueError('Missing pinned inert package initializer.')
            queue.append(package)
    for package in ('models/__init__.py','attacks/__init__.py'):
        tree=parse(pool[package])
        if any(not isinstance(n,ast.Expr) or not isinstance(n.value,ast.Constant) or type(n.value.value) is not str for n in tree.body):
            raise ValueError('Only inert pinned project package initializers are supported.')
    emitted={p:pool[p] for p in sorted(seen)}
    manifest={'kind':'pgd-split-concrete-runtime-source-assembly-v1','contract_sha256':CONTRACT_SHA,
        'files':{p:{'bytes':len(v),'sha256':digest(v)} for p,v in emitted.items()},'imports':edges,
        'external_literal_import_roots':sorted(external),'generated_names':sorted(output),
        'record_transport_reversible_edits':record_edits,'tensor_transport_reversible_edits':tensor_edits,
        'observer_binding_reversible_edits':observer_edits,'numerical_builder':na,'probe_builder':pa,
        'worker_threads':4,'module_global_mutation_of_historical_namespaces':False,'arbitrary_runtime_ports_accepted_by_entry':False,
        'actual_runtime_loaded_or_executed':False,'target_runner_available':False,'new_authorization_ready':False,
        'scientific_qualified':False,'training_admitted':False}
    return emitted,manifest
