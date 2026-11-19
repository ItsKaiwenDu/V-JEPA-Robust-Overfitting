"""Cold component-only source assembly. Generated scientific modules are NOT loaded.

The emitted entry has an explicit null execution binding and no full numerical
entry. Local checks use only source inspection, byte I/O and labeled fake ports.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
from models import predictor_pgd_split_runtime_v1 as cold

INPUTS='configs/predictor_pgd_component_runtime_inputs_v1.json'
INPUTS_SHA='13fd93783da85d20e978d7ab32c3c7045740cbefaea46356dd4f1560b4509173'
_IMPORTS=cold.imports
PREFIX='models/_pgd_component_'
FORBIDDEN=frozenset('models/_pgd_split_'+n+'_v1.py' for n in ('entry','numerical','records','observer','progress','facade','providers','io'))

SOURCE='''
"""Read-only historical capsule, separate from every new execution/result."""
import copy
import hashlib
import os
from pathlib import Path
import stat
from models import predictor_pgd_component_capsule_v1 as capsule
from models import _pgd_split_boundary_v1 as boundary


def runtime_layout():
    # JSON object keys are strings. Restore only this known, bounded group-ID
    # map for the existing Python layout gate; manifest bytes stay unchanged.
    result=copy.deepcopy(MANIFEST['layout']);groups=result['backing_groups']
    if (type(groups) is not dict or not groups or len(groups)>4
            or any(type(k) is not str or k not in tuple(str(i) for i in range(8))
                or type(v) is not int or not 0<v<=16*2**20 for k,v in groups.items())):
        raise ValueError('Canonical bounded serialized backing-group map required.')
    result['backing_groups']={int(k):v for k,v in groups.items()}
    return result


def binding(control_sha256):
    if (not capsule.lineage.sha(control_sha256) or control_sha256==MANIFEST['source_control_sha256']):
        raise ValueError('A distinct future execution control is required; historical control is immutable.')
    return {'kind':'pgd-component-continuation-source-binding-v1','control_sha256':control_sha256,
        'source_control_sha256':MANIFEST['source_control_sha256'],'design_sha256':capsule.lineage.DESIGN_SHA,
        'boundary_sha256':MANIFEST['boundary_sha256'],'source_objective_index':0,
        'capsule_contract_sha256':capsule.CONTRACT_SHA,'capsule_manifest_sha256':MANIFEST_SHA,
        'layout':runtime_layout(),'lineage_metadata_consistent':True,
        'canonical_lineage_authenticated':False,'actual_CUDA_execution_admitted':False,'scientific_qualified':False}


def verify_binding(got,execution_binding):
    expected=binding(execution_binding['control_sha256'])
    if capsule.encoded(got)!=capsule.encoded(expected):
        raise ValueError('Exact separate historical-source and new-execution binding required.')
    return expected


def private_directory(path,device=None):
    p=Path(path);s=p.lstat()
    if (p.resolve(strict=True)!=p or not stat.S_ISDIR(s.st_mode) or s.st_uid!=os.getuid()
            or stat.S_IMODE(s.st_mode)!=0o700 or device is not None and s.st_dev!=device):
        raise ValueError('Canonical private same-device capsule directory required.')
    return (s.st_dev,s.st_ino,s.st_uid,stat.S_IMODE(s.st_mode))


class Source:
    def __init__(self,root):
        self.root=Path(root).absolute();self.identity=private_directory(self.root)
        self.pid=os.getpid();self.used=False
        self._checked()
        self.cap=boundary.load(self.root/'boundary',control_sha256=MANIFEST['source_control_sha256'])
        self.run=capsule.parsed(capsule._read(self.root/'provenance/run-status.json',capsule.MAX_MEMBER,
            MANIFEST['files']['provenance/run-status.json']['sha256']))
        self.cap_snapshot=capsule.encoded(self.cap);self.run_snapshot=capsule.encoded(self.run)
        self.check()
    def _checked(self):
        if os.getpid()!=self.pid or private_directory(self.root)!=self.identity:
            raise ValueError('Historical capsule process/root changed.')
        allowed=set(MANIFEST['files'])|{'MANIFEST.json'};directories={'.'}
        for name in allowed:directories.update(str(p) for p in Path(name).parents)
        found=set()
        for current,dirs,names in os.walk(self.root,followlinks=False):
            here=Path(current);relative=str(here.relative_to(self.root))
            if relative not in directories:raise ValueError('Unknown capsule directory.')
            private_directory(here,self.identity[0])
            for name in dirs:private_directory(here/name,self.identity[0])
            for name in names:
                p=here/name;key=p.relative_to(self.root).as_posix();s=p.lstat()
                if (key not in allowed or key in found or not stat.S_ISREG(s.st_mode) or s.st_uid!=os.getuid()
                        or stat.S_IMODE(s.st_mode)!=0o600 or s.st_nlink!=1 or s.st_dev!=self.identity[0]):
                    raise ValueError('Closed private capsule inventory required.')
                found.add(key)
        if found!=allowed:raise ValueError('Complete capsule inventory required.')
        raw=capsule._read(self.root/'MANIFEST.json',capsule.MAX_MANIFEST,MANIFEST_SHA)
        if raw!=MANIFEST_BYTES:raise ValueError('Frozen capsule manifest required.')
        members={name:capsule._read(self.root/name,capsule.MAX_MEMBER,d['sha256']) for name,d in MANIFEST['files'].items()}
        checked=capsule.validate_members(members,CONTRACT_BYTES)
        if capsule.encoded(checked)!=MANIFEST_BYTES or private_directory(self.root)!=self.identity:
            raise ValueError('Canonical capsule metadata/root changed.')
    def check(self):
        self._checked()
        if capsule.encoded(self.cap)!=self.cap_snapshot or capsule.encoded(self.run)!=self.run_snapshot:
            raise ValueError('Loaded historical metadata mutated.')
    def join(self,control_sha256):
        self.check()
        if self.used:raise ValueError('One source binding per fresh component worker; no reset.')
        self.used=True
        return self.cap,binding(control_sha256)
'''

ENTRY='''
"""UNBOUND component-only entry. No numeric-cohort or training dispatch exists."""
from argparse import Namespace
import math
import os
from pathlib import Path
import platform
from types import SimpleNamespace
import torch
from models import _pgd_component_io_v1 as io
from models import _pgd_component_source_v1 as source
from models import _pgd_component_providers_v1 as provider
from models import _pgd_component_contract_v1 as contract
from models import _pgd_split_startup_v1 as startup
from models import _pgd_split_route_v1 as routes
from models._pgd_component_job_v1 import ComponentJob
EXECUTION_BINDING=None
_USED=False


def claim(args):
    global _USED
    if _USED:raise ValueError('One component entry per fresh namespace; no reset or retry.')
    _USED=True
    if EXECUTION_BINDING is None:raise ValueError('Execution identity is unbound; no target work is authorized.')
    fields={'kind','control_sha256','capsule_root','result_root','host','capsule_manifest_sha256'}
    if (type(EXECUTION_BINDING) is not dict or set(EXECUTION_BINDING)!=fields
            or EXECUTION_BINDING['kind']!='component-only-reviewed-execution-binding-v1'
            or type(args) is not Namespace or set(vars(args))!={'control_sha256','capsule_root','result_root','role','deadline'}):
        raise ValueError('Exact separately reviewed component-only binding and arguments required.')
    if (args.role not in ('native','forward-preserving') or args.control_sha256!=EXECUTION_BINDING['control_sha256']
            or args.capsule_root!=EXECUTION_BINDING['capsule_root'] or args.result_root!=EXECUTION_BINDING['result_root']
            or EXECUTION_BINDING['capsule_manifest_sha256']!=source.MANIFEST_SHA
            or platform.node()!=EXECUTION_BINDING['host']
            or type(args.deadline) not in (int,float) or not math.isfinite(args.deadline)):
        raise ValueError('Exact fresh component role/identity/source and continuing deadline required.')
    source.binding(args.control_sha256)
    parent=Path(args.capsule_root);result=Path(args.result_root)
    if (not parent.is_absolute() or not result.is_absolute() or parent==result
            or result.is_relative_to(parent) or parent.is_relative_to(result)):
        raise ValueError('Disjoint read-only capsule and fresh result roots required.')
    source.private_directory(parent);source.private_directory(result)
    return io.bind(result),source.Source(parent)


def component_entry(args):
    store,canonical=claim(args)
    binding={'design_sha256':source.capsule.lineage.DESIGN_SHA,'control_sha256':args.control_sha256,
        'boundary_sha256':source.MANIFEST['boundary_sha256'],'probe_index':0 if args.role=='native' else 1}
    state={};job=None
    def initialize():
        value=startup.initialize('cuda');torch.set_num_threads(4)
        if torch.get_num_threads()!=4:raise ValueError('Original four worker threads required.')
        state['startup']=value
        return value
    def load_source():
        cap,joined=canonical.join(args.control_sha256)
        expected={k:v for k,v in canonical.run['baseline']['runtime'].items() if k!='model_execution'}
        actual=contract.runtime_contract('cuda')
        if actual!=expected or actual.get('threads')!=4:
            raise ValueError('Captured original runtime/precision changed before operands.')
        state.update(cap=cap,runtime=actual)
        job.state['runtime']=actual
        job.state['runtime_source_numerical_sha256']=source.MANIFEST['files']['provenance/run-status.json']['sha256']
        return cap,joined
    def restore():
        canonical.check()
        startup.validate(state['startup'],pid=job.state['pid'],device='cuda')
        if (contract.runtime_contract('cuda')!=state['runtime'] or torch.get_num_threads()!=4
                or provider.flags()!=state['cap']['selected_call']['actual_call']['entry_flags']):
            raise ValueError('Original component runtime/precision/attention flags not restored.')
        store.check()
    ports={'startup':initialize,'source':load_source,'operands':lambda cap:provider.prepare_operands(canonical,cap),
        'route':lambda role,values,incoming,binding:routes.ComponentRoute(role,values,incoming,binding=binding),
        'profile':provider.make_profile,'compute':lambda route,observe:provider.compute_once(route,torch.profiler.record_function,observe),
        'raw':provider.raw_bytes,'serialize':provider.serialize_return,'restore':restore,'allocator':torch.cuda.max_memory_allocated}
    job=ComponentJob(args.role,binding,store,ports,deadline=args.deadline,
        evidence_origin='CAPSULE-COMPONENT-ONLY-V1; LIVE-ADMISSIONS-PENDING')
    return job.execute()
'''

PREPARE='''
def prepare_operands(source,cap):
    source.check()
    if flags()!=cap['selected_call']['actual_call']['entry_flags']:
        raise ValueError('Captured original attention flags changed before reconstruction; no fallback.')
    values,metadata=_operands(source.root,cap,'cuda')
    return tuple(values[n] for n in ('q','k','v')),values['grad_output'],metadata
'''


def digest(data):return hashlib.sha256(data).hexdigest()
def encoded(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def _require(value,message):
    if not value:raise ValueError(message)
def _text(tree):return (ast.unparse(ast.fix_missing_locations(tree))+'\n').encode()
def _node(tree,name):return next(n for n in tree.body if getattr(n,'name',None)==name)

def _patch(raw,start,end,new,edits):
    if type(new) is str:new=new.encode()
    edits.append({'start':start,'old':raw[start:end].decode(),'new':new.decode()})
    return raw[:start]+new+raw[end:]
def _literal(raw,old,new,edits):
    old=old.encode();_require(raw.count(old)==1,'One exact source bridge required: '+old.decode())
    start=raw.index(old);return _patch(raw,start,start+len(old),new,edits)
def _function(raw,name,new,edits):
    node=_node(ast.parse(raw),name);lines=raw.splitlines(keepends=True)
    start=sum(map(len,lines[:node.lineno-1]));end=sum(map(len,lines[:node.end_lineno]))
    return _patch(raw,start,end,new,edits)
def inverse(raw,edits):
    for e in reversed(edits):
        start=e['start'];end=start+len(e['new'].encode())
        _require(raw[start:end]==e['new'].encode(),'Exact source inverse required.')
        raw=raw[:start]+e['old'].encode()+raw[end:]
    return raw

def load(repository):
    root=Path(repository);cb=(root/INPUTS).read_bytes()
    _require(digest(cb)==INPUTS_SHA,'Exact cold source input contract required.')
    c=json.loads(cb);inputs={}
    for name,d in c['files'].items():
        raw=(root/d['path']).read_bytes()
        _require(len(raw)==d['bytes'] and digest(raw)==d['sha256'],'Pinned cold source changed: '+name)
        inputs[name]=raw
    return inputs,cb

def assemble(inputs,contract_bytes):
    _require(type(contract_bytes) is bytes and digest(contract_bytes)==INPUTS_SHA,'Exact cold source input contract required.')
    c=json.loads(contract_bytes)
    _require(type(inputs) is dict and set(inputs)==set(c['files']),'Complete fixed cold source set required.')
    for name,d in c['files'].items():
        _require(type(inputs[name]) is bytes and len(inputs[name])==d['bytes'] and digest(inputs[name])==d['sha256'],'Pinned cold source changed: '+name)
    _require(cold.imports is _IMPORTS and digest(Path(cold.__file__).read_bytes())==c['files']['COLD_IMPORT_READER']['sha256'],'Cold dependency reader changed.')
    pool={n:v for n,v in inputs.items() if n.endswith('.py')};edits={};generated={}
    def changed(name,old,work):
        e=[];new=work(old,e);_require(inverse(new,e)==old,'Complete component source byte inverse required.')
        generated[name]=new;edits[name]=e
    changed(PREFIX+'io_v1.py',inputs['models/_pgd_split_io_v1.py'],lambda raw,e:_literal(raw,
        'from models.predictor_pgd_split_store_v1 import Inventory, Store, identity',
        'from models._pgd_component_store_v1 import Inventory, Store, identity',e))
    def store(raw,e):
        tree=ast.parse(raw);cls=_node(tree,'Inventory');ctor=next(n for n in cls.body if getattr(n,'name',None)=='__init__')
        value=next(n for n in ctor.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='spec' for t in n.targets)).value
        expr=ast.get_source_segment(raw.decode(),value)
        return _literal(raw,expr,"{n:d for n,d in "+expr+".items() if n.startswith(('results/components/native/','results/components/forward-preserving/')) or n in ('logs/native.log','logs/forward-preserving.log')}",e)
    changed(PREFIX+'store_v1.py',inputs['models/predictor_pgd_split_store_v1.py'],store)
    def providers(raw,e):
        raw=_literal(raw,'from models import _pgd_split_records_v1 as numerical\n','',e)
        raw=_literal(raw,'from models.predictor_pgd_split_probe_job_v1 import private_bytes\n','',e)
        raw=_function(raw,'load_new_source','',e)
        return _function(raw,'prepare_operands',PREPARE.lstrip(),e)
    changed(PREFIX+'providers_v1.py',inputs['models/_pgd_split_providers_v1.py'],providers)
    source=ast.parse(SOURCE)
    source.body.extend(ast.parse('CONTRACT_BYTES='+repr(inputs['CAPSULE_CONTRACT'])+'\nMANIFEST_BYTES='+repr(inputs['CAPSULE_MANIFEST'])+'\nMANIFEST_SHA='+repr(digest(inputs['CAPSULE_MANIFEST']))+'\nMANIFEST=capsule.parsed(MANIFEST_BYTES)').body)
    generated[PREFIX+'source_v1.py']=_text(source)
    def job(raw,e):
        raw=_literal(raw,'from models import predictor_pgd_split_backend_v1 as backend','from models import predictor_pgd_split_backend_v3 as backend\nfrom models import _pgd_component_source_v1 as continuation',e)
        raw=_literal(raw,"KIND='pgd-split-component-worker-v1'","KIND='pgd-capsule-component-worker-v1'",e)
        raw=_literal(raw,"            self.state['source']=joined","            continuation.verify_binding(joined,self.binding)\n            self.state['source']=joined",e)
        raw=_literal(raw,"    resource=json.loads(verified(marker['resource'],'resource.json',16*1024))","    resource=json.loads(verified(marker['resource'],'resource.json',16*1024))\n    continuation.verify_binding(worker.get('source'),binding)",e)
        old="pgd-split-component-complete-v1";new="pgd-capsule-component-complete-v1"
        count=raw.count(old.encode());_require(count==3,'Three fixed completion-schema occurrences required.')
        for _ in range(count):
            start=raw.index(old.encode());raw=_patch(raw,start,start+len(old),new,e)
        return raw
    changed(PREFIX+'job_v1.py',inputs['models/predictor_pgd_split_probe_job_v1.py'],job)
    generated[PREFIX+'entry_v1.py']=_text(ast.parse(ENTRY))
    training=ast.parse(inputs['models/training.py']);runtime=copy.deepcopy(_node(training,'runtime_contract'))
    contract=ast.parse('import os\nimport platform\nimport numpy as np\nimport torch\nfrom models.execution import model_execution_contract,precision_contract\n');contract.body.append(runtime)
    generated[PREFIX+'contract_v1.py']=_text(contract)
    pool.update(generated)
    # Backend source authentication also reads pinned sibling files that need
    # not appear in Python import statements. Close those data dependencies.
    reader='models/predictor_pgd_split_backend_v3.py';tree=ast.parse(pool[reader])
    pins=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINNED' for t in n.targets)))
    data_dependencies={reader:{'models/'+name:want for name,want in pins.items()}}
    for name,want in data_dependencies[reader].items():
        _require(name in pool and digest(pool[name])==want,'Required non-import source pin missing or changed: '+name)
    seen=set();edges={};external=set();queue=[PREFIX+'entry_v1.py',*data_dependencies[reader]]
    while queue:
        name=queue.pop()
        if name in seen:continue
        _require(name in pool and name not in FORBIDDEN,'Unresolved or forbidden full-cohort source: '+name)
        seen.add(name);deps,roots=_IMPORTS(ast.parse(pool[name]));external.update(roots);edges[name]=[]
        for dep in sorted(deps):
            path=dep.replace('.','/')+'.py';_require(path in pool,'Unresolved pinned component dependency: '+path)
            edges[name].append(path);queue.append(path)
            package=path.split('/')[0]+'/__init__.py';_require(package in pool,'Pinned inert package initializer required.');queue.append(package)
    emitted={name:pool[name] for name in sorted(seen)}
    _require(set(generated)<=set(emitted) and not set(emitted)&FORBIDDEN,'Only closed component source graph may be emitted.')
    for package in (p for p in emitted if p.endswith('/__init__.py')):
        t=ast.parse(emitted[package]);_require(all(isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and type(n.value.value) is str for n in t.body),'Only inert package initializers required.')
    for name,raw in emitted.items():compile(raw,'<UNEXECUTED-COMPONENT-SOURCE-'+name+'>','exec')
    _require(sum(map(len,emitted.values()))<=2*2**20,'Original source envelope ceiling required.')
    manifest={'kind':'pgd-component-only-cold-runtime-source-v1','input_contract_sha256':INPUTS_SHA,
        'files':{n:{'bytes':len(v),'sha256':digest(v)} for n,v in emitted.items()},'imports':edges,'external_literal_import_roots':sorted(external),
        'generated_names':sorted(generated),'reversible_source_edits':edits,'all_source_byte_inverses_verified':True,
        'nonimport_source_pin_dependencies':data_dependencies,
        'source_capsule_archive_sha256':c['capsule_archive_sha256'],'source_capsule_manifest_sha256':digest(inputs['CAPSULE_MANIFEST']),
        'execution_binding':None,'new_full_model_objectives':0,'full_numerical_dispatch_absent':True,
        'source_and_result_namespaces_disjoint_required':True,'component_only_result_inventory':True,
        'frozen_arithmetic_reconstruction_startup_raw_and_runtime_contract_retained':True,
        'actual_runtime_loaded_or_executed':False,'target_supervisor_inventory_collection_integrated':False,
        'new_target_scope_handoff_or_authorization':False,'scientific_qualified':False,'training_admitted':False}
    return emitted,manifest
