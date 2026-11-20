"""Cold candidate-only worker adapter; no target entry, authorization or launch.

Archived scientific providers are copied exactly. A separately bound candidate
adds a read-only native reference, v4 backend metadata and native-relative checks.
"""
import ast
import hashlib
import json
from pathlib import Path
from models import predictor_pgd_split_runtime_v1 as cold
from models import predictor_pgd_candidate_reference_v1 as reference

INPUTS='configs/predictor_pgd_candidate_completion_inputs_v1.json'
INPUTS_SHA='b9541ed09da733a8000cf3f39fda884e88cab7b748e27d8e63ca60f9c0050ad1'
JOB='models/_pgd_component_job_v1.py'
ENTRY='models/_pgd_component_entry_v1.py'
STORE='models/_pgd_component_store_v1.py'
REF='models/predictor_pgd_candidate_reference_v1.py'
ACCURACY='models/predictor_pgd_candidate_accuracy_v1.py'
READER='models/predictor_pgd_split_backend_v4.py'
FORBIDDEN=frozenset('models/_pgd_split_'+n+'_v1.py' for n in ('entry','numerical','records','observer','progress','facade','providers','io'))


def digest(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def require(value,message):
    if not value:raise ValueError(message)

def inverse(raw,edits):
    for e in reversed(edits):
        start=e['start'];end=start+len(e['after'].encode());require(raw[start:end]==e['after'].encode(),'Exact cold inverse required.')
        raw=raw[:start]+e['before'].encode()+raw[end:]
    return raw


def load(repository):
    root=Path(repository);data=(root/INPUTS).read_bytes();require(digest(data)==INPUTS_SHA,'Exact candidate-only input contract required.')
    c=json.loads(data);raws={}
    for name,d in c['files'].items():
        raw=(root/name).read_bytes();require(len(raw)==d['bytes'] and digest(raw)==d['sha256'],'Pinned candidate input changed: '+name);raws[name]=raw
    require(digest(Path(cold.__file__).read_bytes())==c['extra_sources']['models/predictor_pgd_split_runtime_v1.py']['sha256'],'Frozen cold import reader changed.')
    return c,raws


def assemble(repository):
    root=Path(repository);c,raws=load(root)
    pool={name:raws[d['path']] for name,d in c['runtime_files'].items()};originals=dict(pool);changes={}
    members={name:raws[d['path']] for name,d in c['native_reference_files'].items()}
    spec=reference.describe_members(members,c['native_reference_files']);archive=reference.archive(members,spec)
    require(reference.inspect_archive(archive,spec)==members,'Native reference archive roundtrip required.')
    def edited(name,original,work):
        edits=[]
        def replace(raw,before,after):
            a=before.encode();require(raw.count(a)==1,'One exact candidate edit: '+before[:100]);start=raw.index(a)
            edits.append({'start':start,'before':before,'after':after});return raw[:start]+after.encode()+raw[start+len(a):]
        result=work(original,replace);require(inverse(result,edits)==original,'Complete candidate source inverse required.')
        pool[name]=result;changes[name]={'original_sha256':digest(original),'candidate_sha256':digest(result),'edits':edits}
    edited(REF,(root/REF).read_bytes(),lambda raw,e:e(e(raw,'SPEC_BYTES=None','SPEC_BYTES='+repr(reference.encoded(spec))),'SPEC_SHA=None','SPEC_SHA='+repr(digest(reference.encoded(spec)))))
    pool[ACCURACY]=(root/ACCURACY).read_bytes();pool[READER]=raws[READER]
    def job(raw,e):
        raw=e(raw,'from models import predictor_pgd_split_backend_v3 as backend','from models import predictor_pgd_split_backend_v4 as backend\nfrom models import predictor_pgd_candidate_reference_v1 as reference\nfrom models import predictor_pgd_candidate_accuracy_v1 as accuracy')
        raw=e(raw,"KIND='pgd-capsule-component-worker-v1'","KIND='pgd-candidate-component-worker-v1'")
        raw=e(raw,"'startup','source_verify','operands'","'startup','source_verify','reference_verify','operands'")
        raw=e(raw,"'backend_validate','backend_publish','restoration'","'backend_validate','backend_publish','accuracy_validate','accuracy_publish','restoration'")
        raw=e(raw,"'restore','allocator'","'restore','allocator','reference','accuracy'")
        raw=e(raw,"if role not in ('native','forward-preserving') or type(ports)","if role!='forward-preserving' or type(ports)")
        raw=e(raw,"            self.state['source']=joined", "            self.state['source']=joined;self.time_gate()\n            native_reference=self.log.call('reference_verify',self.ports['reference'],self.binding)\n            reference.verify_description(native_reference,self.binding)\n            self.state['native_reference']=native_reference;self.time_gate()")
        old="            self.state['backend']=self.log.call('backend_publish',self.save,'backend.json',observed)"
        new=old+"\n            def checked_values():\n                value=self.ports['accuracy'](self.binding,self.state['outputs'],self.folder)\n                return accuracy.validate_receipt(value,self.binding,self.state['outputs'],self.state['native_reference'])\n            value_check=self.log.call('accuracy_validate',checked_values)\n            self.state['accuracy']=self.log.call('accuracy_publish',self.save,'accuracy.json',value_check)\n            self.state['saved_native_relative_values_passed']=True;self.time_gate()"
        raw=e(raw,old,new)
        raw=e(raw,"    folder=Path(folder);p=folder/'complete.json'", "    if role!='forward-preserving':raise ValueError('Only a fresh candidate completion is admitted.')\n    folder=Path(folder);p=folder/'complete.json'")
        raw=e(raw,"    continuation.verify_binding(worker.get('source'),binding)","    continuation.verify_binding(worker.get('source'),binding)\n    reference.verify_description(worker.get('native_reference'),binding)")
        old="    ownership=json.loads(verified(worker['ownership'],'ownership.json',64*1024))"
        raw=e(raw,old,"    value_check=json.loads(verified(worker['accuracy'],'accuracy.json',16*1024))\n    accuracy.validate_receipt(value_check,binding,worker['outputs'],worker['native_reference'])\n    if worker.get('saved_native_relative_values_passed') is not True:raise ValueError('Passing native-relative candidate values required.')\n"+old)
        old='    """Join already-verified receipts; no dict alone authenticates execution."""'
        raw=e(raw,old,old+"\n    if role!='forward-preserving':raise ValueError('No native rerun/parent admission through candidate adapter.')")
        # Three distinct schema positions are changed, with exact inverses.
        before='pgd-capsule-component-complete-v1';after='pgd-candidate-component-complete-v1'
        require(raw.count(before.encode())==3,'Three fixed completion-schema occurrences required.')
        for name in ('execute','read_completion','bind_parent_metadata'):
            node=next(n for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.FunctionDef) and n.name==name)
            segment=ast.get_source_segment(raw.decode(),node)
            require(segment.count(before)==1,'One completion schema in each selected function required.')
            raw=e(raw,segment,segment.replace(before,after))
        return raw
    edited(JOB,originals[JOB],job)
    def entry(raw,e):
        tree=ast.parse(raws['models/predictor_pgd_component_runtime_v1.py'])
        template=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ENTRY' for t in n.targets)))
        require((ast.unparse(ast.parse(template))+'\n').encode()==raw,'Original archived entry must equal the pinned template AST emission.')
        raw=e(raw,raw.decode(),template)
        raw=e(raw,'from models._pgd_component_job_v1 import ComponentJob','from models._pgd_component_job_v1 import ComponentJob\nfrom models import predictor_pgd_candidate_reference_v1 as reference\nfrom models import predictor_pgd_candidate_accuracy_v1 as accuracy')
        raw=e(raw,"fields={'kind','control_sha256','capsule_root','result_root','host','capsule_manifest_sha256'}", "fields={'kind','control_sha256','capsule_root','result_root','reference_root','host','capsule_manifest_sha256','reference_manifest_sha256'}")
        raw=e(raw,"'component-only-reviewed-execution-binding-v1'","'candidate-only-reviewed-execution-binding-v1'")
        raw=e(raw,"{'control_sha256','capsule_root','result_root','role','deadline'}", "{'control_sha256','capsule_root','result_root','reference_root','role','deadline'}")
        raw=e(raw,"args.role not in ('native','forward-preserving')", "args.role!='forward-preserving'")
        raw=e(raw,"or EXECUTION_BINDING['capsule_manifest_sha256']!=source.MANIFEST_SHA", "or args.reference_root!=EXECUTION_BINDING['reference_root']\n            or EXECUTION_BINDING['reference_manifest_sha256']!=reference.SPEC_SHA\n            or EXECUTION_BINDING['capsule_manifest_sha256']!=source.MANIFEST_SHA")
        old="    source.private_directory(parent);source.private_directory(result)\n    return io.bind(result),source.Source(parent)"
        new="    native_root=Path(args.reference_root)\n    roots=(parent,result,native_root)\n    if any(not p.is_absolute() for p in roots) or any(a==b or a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(roots) for b in roots[i+1:]):\n        raise ValueError('Three disjoint read-only source/reference and fresh result roots required.')\n    source.private_directory(parent);source.private_directory(result);source.private_directory(native_root)\n    binding={'design_sha256':source.capsule.lineage.DESIGN_SHA,'control_sha256':args.control_sha256,'boundary_sha256':source.MANIFEST['boundary_sha256'],'probe_index':1}\n    reference.expected_description(binding)\n    return io.bind(result),source.Source(parent),reference.Reference(native_root)"
        raw=e(raw,old,new)
        raw=e(raw,'    store,canonical=claim(args)','    store,canonical,native_reference=claim(args)')
        raw=e(raw,"'probe_index':0 if args.role=='native' else 1", "'probe_index':1")
        raw=e(raw,"        canonical.check()\n        startup.validate", "        canonical.check();native_reference.check()\n        startup.validate")
        raw=e(raw,"'restore':restore,'allocator':torch.cuda.max_memory_allocated}", "'restore':restore,'allocator':torch.cuda.max_memory_allocated,\n        'reference':lambda binding:native_reference.describe(binding),\n        'accuracy':lambda binding,outputs,folder:accuracy.check_values(binding,outputs,folder,native_reference)}")
        raw=e(raw,'CAPSULE-COMPONENT-ONLY-V1; LIVE-ADMISSIONS-PENDING','CANDIDATE-ONLY-COMPLETION-V1; LIVE-ADMISSIONS-PENDING')
        return e(raw,raw.decode(),ast.unparse(ast.parse(raw))+'\n')
    edited(ENTRY,originals[ENTRY],entry)
    def store(raw,e):
        tree=ast.parse(raw);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Inventory');ctor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        assign=next(n for n in ctor.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='spec' for t in n.targets));old=ast.get_source_segment(raw.decode(),assign.value)
        new="{**{n:d for n,d in ("+old+").items() if n.startswith('results/components/forward-preserving/') or n=='logs/forward-preserving.log'},'results/components/forward-preserving/accuracy.json':{'category':'metadata','max_bytes':16*1024,'mutable':False,'partial_allowed':True}}"
        return e(raw,old,new)
    edited(STORE,originals[STORE],store)
    pins=ast.literal_eval(next(n.value for n in ast.parse(pool[READER]).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINNED' for t in n.targets)))
    dependencies={'models/'+n:want for n,want in pins.items()}
    for n,want in dependencies.items():require(n in pool and digest(pool[n])==want,'Frozen backend data dependency changed.')
    seen=set();edges={};external=set();queue=[ENTRY,*dependencies]
    while queue:
        name=queue.pop()
        if name in seen:continue
        require(name in pool and name not in FORBIDDEN,'Unresolved or full-cohort candidate dependency: '+name)
        seen.add(name);deps,roots=cold.imports(ast.parse(pool[name]));external.update(roots);edges[name]=[]
        for dep in sorted(deps):
            p=dep.replace('.','/')+'.py';require(p in pool,'Closed candidate source dependency required: '+p);edges[name].append(p);queue.append(p)
            package=p.split('/')[0]+'/__init__.py';require(package in pool,'Inert package initializer required.');queue.append(package)
    files={n:pool[n] for n in sorted(seen)}
    require(REF in files and ACCURACY in files and READER in files and 'models/predictor_pgd_split_backend_v3.py' not in files,'Only the corrected candidate reader may bind.')
    for n,raw in files.items():compile(raw,'<UNEXECUTED-CANDIDATE-COMPLETION-'+n+'>','exec')
    require(sum(map(len,files.values()))<=2*2**20,'Original source envelope ceiling required.')
    for n in originals:
        if n in files and n not in (JOB,ENTRY,STORE):require(files[n]==originals[n],'Frozen scientific/source provider changed: '+n)
    manifest={'kind':'pgd-candidate-only-cold-runtime-v1','input_contract_sha256':INPUTS_SHA,
        'files':{n:{'bytes':len(v),'sha256':digest(v)} for n,v in files.items()},'imports':edges,'external_literal_import_roots':sorted(external),
        'reversible_source_edits':changes,'all_byte_inverses_verified':True,'nonimport_file_pins':dependencies,
        'native_reference_manifest_sha256':digest(reference.encoded(spec)),'native_reference_archive_bytes':len(archive),'native_reference_archive_sha256':digest(archive),
        'unchanged_capsule_archive':c['unchanged_capsule_archive'],'execution_binding':None,'roles':['forward-preserving'],
        'new_full_model_objectives':0,'native_or_completed_cohort_reruns':0,'new_FP64_reference_or_full_model_trace':False,
        'scientific_provider_route_startup_raw_layout_runtime_bytes_unchanged':True,'parent_completion_metadata_join_preserved':True,
        'OS_supervisor_scratch_cache_collection_integration_complete':False,'new_target_scope_handoff_or_authorization':False,
        'actual_tensor_array_model_gradient_GPU_or_target_execution':False,'training_admitted':False}
    return files,manifest,archive,spec
