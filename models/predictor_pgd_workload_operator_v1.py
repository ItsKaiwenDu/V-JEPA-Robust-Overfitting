"""Cold whole-PGD OS/bootstrap/client derivation; no scope, arming or launch API."""
import ast
import base64
import json
from pathlib import Path
import stat
from models import predictor_pgd_workload_runtime_v1 as frozen
from models.predictor_pgd_workload_session_v1 import result_specs

INPUTS='configs/predictor_pgd_workload_operator_inputs_v2.json'
INPUTS_SHA='2b833c7cfb22880a897f2bde4042f2bd331db0b6292f58265af68d7e78ef3a60'
LIVE_EXECUTION_BINDING=None
ROLES=('proof','original-unit','candidate-unit','candidate-replay')
SECONDS={'proof':60,'original-unit':720,'candidate-unit':720,'candidate-replay':720}
BASE_SLOTS=('bootstrap.py','predictor_pgd_split_scratch_v1.py','predictor_pgd_split_coordinator_v1.py',
    'predictor_pgd_split_collection_v1.py','predictor_pgd_split_watch_v1.py','scratch_allocation_v2.py')
EXTRA_SLOTS={'models.predictor_pgd_workload_watch_v1':'workload_watch_v1.py',
             'models.predictor_pgd_workload_os_v1':'workload_os_v1.py',
             'models.predictor_pgd_split_plan_v1':'workload_plan_v1.py'}
OUTER='models/predictor_pgd_workload_outer_v1.py'
CLIENT='models/predictor_pgd_workload_client_v1.py'
READERS={'bootstrap.py':'models/predictor_pgd_workload_bootstrap_v1.py',
    'predictor_pgd_split_scratch_v1.py':'models/predictor_pgd_workload_scratch_v1.py',
    'predictor_pgd_split_collection_v1.py':'models/predictor_pgd_workload_collection_v1.py'}
encoded=frozen.encoded;digest=frozen.digest;require=frozen.require;inverse=frozen.inverse
KINDS={
    'pgd-candidate-supervised-control-v1':'whole-pgd-supervised-control-v1',
    'pgd-candidate-supervised-university-exception-v1':'whole-pgd-supervised-university-exception-v1',
    'pgd-candidate-parent-session-v1':'whole-pgd-parent-session-v1',
    'pgd-candidate-child-launch-v1':'whole-pgd-child-launch-v1',
    'pgd-candidate-work-receipt-v1':'whole-pgd-work-receipt-v1',
    'pgd-candidate-evidence-stream-v1':'whole-pgd-evidence-stream-v1',
    'pgd-candidate-outer-control-v1':'whole-pgd-outer-control-v1',
    'pgd-candidate-outer-university-exception-v1':'whole-pgd-outer-university-exception-v1',
    'candidate-only-runtime-policy-v1':'whole-pgd-runtime-policy-v1',
    'candidate-only-whole-scratch-v1':'whole-pgd-whole-scratch-v1',
    'vjepa-pgd-candidate-v1-':'vjepa-pgd-workload-v1-',
    'Both proof and candidate roles must finish before a complete-work receipt.':
        'Proof and all three whole-PGD roles must finish before a complete-work receipt.',
}


class Editor(frozen.Editor):
    def all(self,before,after):
        require(before not in after,'Substitutions must terminate.')
        while before.encode() in self.raw:
            a=self.raw.index(before.encode());self.raw=self.raw[:a]+after.encode()+self.raw[a+len(before.encode()):]
            self.edits.append({'start':a,'before':before,'after':after})
    def node(self,name,replacement,*,assignment=False):
        nodes=[n for n in ast.parse(self.raw).body if (
            assignment and isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
            or (not assignment and getattr(n,'name',None)==name)]
        require(len(nodes)==1,'One exact top-level definition required: '+name)
        self.one(ast.get_source_segment(self.raw.decode(),nodes[0]),replacement.strip())
    def append(self,text):
        self.edits.append({'start':len(self.raw),'before':'','after':text});self.raw+=text.encode()
    def schemas(self):
        for before,after in KINDS.items():self.all(before,after)
        return self


def assigned(raw,name):
    return ast.literal_eval(next(n.value for n in ast.parse(raw).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))


def load(repository):
    root=Path(repository);raw=(root/INPUTS).read_bytes()
    require(INPUTS_SHA is not None and digest(raw)==INPUTS_SHA,'Pinned whole-PGD operator inputs required.')
    cfg=json.loads(raw);files={}
    for name,d in cfg['files'].items():
        p=root/name;require(stat.S_ISREG(p.lstat().st_mode),'Never open a special operator input.')
        value=p.read_bytes();require(len(value)==d['bytes'] and digest(value)==d['sha256'],'Pinned operator input changed: '+name)
        files[name]=value
    return cfg,files


STATIC_CONTRACT="""def static_contract(supervisors):
    assets=dict(STATIC['assets'])
    assets.update({'supervisor/'+n:dict(d) for n,d in supervisors.items()})
    return {'kind':'whole-pgd-whole-scratch-v1','assets':assets,
        'runtime_files':STATIC['runtime_files'],'result':STATIC['result'],
        'root_metadata':STATIC['root_metadata'],'caps':STATIC['caps'],'packet':STATIC['packet']}
"""

VERIFY="""def verify(self,role,plan,observed):
    o=self.owner;g=self.guard
    if (role not in ROLES or observed.get('pid') is None or observed.get('exit_status')!=0
            or observed.get('parent_observation_passed') is not True):
        raise ValueError('Successful new owned whole-unit/proof observation required.')
    if role=='proof':
        raw=g.read('allocation-result-v2.json',16*1024);receipt=g.verify_proof(boot.digest(raw))
        self.sources=dict(SOURCE_HASHES);self.whole_units=[]
    else:
        if self.sources!=SOURCE_HASHES or plan['source_hashes']!=SOURCE_HASHES:
            raise ValueError('Exact new full-runtime source plan required.')
        g.verify_assets(phase='capture')
        from models import predictor_pgd_workload_os_v1 as worker
        from models import predictor_pgd_workload_runtime_records_v1 as records
        row=worker.read_complete(g,plan,o.policy['scope'],observed);self.whole_units.append(row)
        receipt={'work_metadata_verified':True,'metadata_authenticates_execution':False}
        if role=='candidate-replay':
            join=records.join_units(*self.whole_units)
            _metadata_write(g,'capture/results/replay.json',join,replace=False)
    _metadata_write(g,'parent-'+role+'.json',observed,replace=False)
    g.scan(phase='capture',settled=True)
    return {'role':role,'pid':observed['pid'],'start_ticks':observed['start_ticks'],
        'saved_gates_returned':True,'receipt':receipt,'source_hashes':dict(self.sources),
        'scientific_qualified':False,'training_admitted':False}
"""

DISPATCH="""def dispatch_child(root,binding,cbytes,abytes,plan):
    import resource
    c,policy,csha=boot.control(cbytes);boot.authorization(abytes,csha)
    target_gate(root,binding);gpu_gate();boot.require_namespace(root,binding,csha)
    boot.validate_plan(plan,csha,boot.parsed(boot.private_read(root,binding,'session.json',128*1024)),time.monotonic())
    ib,isha=boot.installation(c,csha)
    if boot.private_read(root,binding,'installation.json',64*1024)!=ib:
        raise ValueError('Child installation bytes changed.')
    layout=scratch.Layout(policy['scratch_contract_text'].encode(),policy['base_text'].encode(),policy['design_text'].encode(),ib,isha)
    guard=scratch.Guard(root,binding,layout);guard.verify_assets(phase='setup' if plan['role']=='proof' else 'capture')
    if time.monotonic()>=plan['deadline']:raise TimeoutError('Original child clock exhausted before work.')
    limit=3*2**30 if plan['role']=='proof' else 16*2**20
    resource.setrlimit(resource.RLIMIT_FSIZE,(limit,limit))
    if plan['role']=='proof':
        guard.proof_readiness()
        from models.scratch_allocation_v2 import bounded_write_proof
        bounded_write_proof(root,deadline=plan['deadline']);return 0
    if plan['role'] not in ROLES[1:] or plan['source_hashes']!=SOURCE_HASHES:
        raise ValueError('Only a new complete-unit role may bind the runtime.')
    from models import predictor_pgd_workload_os_v1 as worker
    return worker.run_unit(root,guard,plan,policy['scope'])
"""

MONITOR="""def monitor_for(role):
    if role in ROLES[1:]:
        from models import predictor_pgd_workload_watch_v1 as unit_watch
        return unit_watch.watch
    if role!='proof':raise ValueError('Exact new proof/unit role required.')
    source=Path(frozen_watch.__file__).read_bytes()
    tree,_=watch_source(source);namespace=dict(frozen_watch.__dict__)
    exec(compile(tree,'<FROZEN-ORIGINAL-PROOF-MONITOR>','exec'),namespace)
    return lambda child,lease,**kw:namespace['watch_stage'](child,lease,role=role,**kw)
"""

DIRECTORIES="""def _setup_directory_plan(layout):
    claims={r:'capture/results/workload/'+r for r in ('original-unit','candidate-unit','candidate-replay')}
    selected={'.','capture','capture/logs','capture/results','capture/results/workload','cache','cache/cuda','tmp'}
    for name in layout.assets:selected.update(str(p) for p in PurePosixPath(name).parents)
    if any(n==p or n.startswith(p+'/') for n in selected for p in claims.values()):
        raise ValueError('Worker directories must not be precreated by setup.')
    for name in selected:layout.directory(name)
    return {'kind':'whole-pgd-minimal-directory-plan-v1',
        'setup_directories':sorted(selected,key=lambda n:(len(PurePosixPath(n).parts),n)),
        'worker_claim_directories':claims,'authorized':False,'scientific_qualified':False,'training_admitted':False}
"""


def method_node(raw,cls,method):
    tree=ast.parse(raw)
    n=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return ast.get_source_segment(raw.decode(),next(x for x in n.body if getattr(x,'name',None)==method))


def assemble(repository,*,method,step_size_255=1.,sensitivity_banks=0):
    require(LIVE_EXECUTION_BINDING is None,'Cold operator execution binding must stay null.')
    require(method in ('primary','convergence') and type(step_size_255) in (int,float)
        and step_size_255 in (.5,1,2) and type(sensitivity_banks) is int and sensitivity_banks in (0,2),'Declared method/diagnostic step/audit banks required.')
    cfg,f=load(repository);root=Path(repository)
    m=json.loads(f[cfg['runtime_manifest']]);runtime={n:d for n,d in m['files'].items() if not n.startswith('supervisor/')}
    require(len(m['files'])==47 and len(runtime)==46 and m['execution_binding'] is None,'Exact reviewed null 47-file closure required.')
    workers={n:f[cfg['runtime_paths'][n]] for n in runtime}
    for n,data in workers.items():require(digest(data)==runtime[n]['sha256'] and len(data)==runtime[n]['bytes'],'Frozen workload source changed.')
    old_payload=json.loads(f[cfg['old_payload']]);old={n:base64.b64decode(old_payload['supervisor/'+n],validate=True) for n in BASE_SLOTS}
    static=assigned(old['bootstrap.py'],'STATIC')
    # Keep immutable original base/design data as lineage, not a grant for this workload.
    static={k:v for k,v in static.items() if k not in ('reference','reference_manifest_sha256','capsule_manifest_sha256')}
    static['runtime_files']=runtime
    static['packet']=cfg['packet']
    static['assets']={'original-packet.tar.gz':cfg['packet']['archive']}
    static['assets'].update({'code/'+n:d for n,d in cfg['packet']['files'].items()})
    static['assets'].update({'overlay/'+n:d for n,d in runtime.items()})
    for name,text in (('base',static['base_text']),('design',static['design_text'])):
        data=text.encode();static['assets']['configs/'+name+'.json']={'bytes':len(data),'sha256':digest(data)}
    static['result']=result_specs(method)
    static['result'].pop('logs/proof.log')
    for n in tuple(static['root_metadata']):
        if n.startswith(('worker-plan-','parent-')):static['root_metadata'].pop(n)
    for r in ROLES:
        static['root_metadata']['worker-plan-'+r+'.json']=[16*1024,False,True]
        static['root_metadata']['parent-'+r+'.json']=[128*1024,False,True]
    static['source_hashes']={'runtime_manifest_sha256':digest(f[cfg['runtime_manifest']]),
        'runtime_contract_sha256':digest(f[cfg['runtime_contract']])}
    static['scope']={'kind':'whole-pgd-source-interface-not-a-grant','method':method,
        'step_size_255':step_size_255,'sensitivity_banks':sensitivity_banks,
        'runtime_manifest_sha256':digest(f[cfg['runtime_manifest']]),
        'design_sha256':digest(static['design_text'].encode()),'research_admitted':False,'training_admitted':False}
    slots=dict(assigned(old['bootstrap.py'],'SUPERVISORS'),**EXTRA_SLOTS)
    static['supervisor_names']=list(slots.values())
    unitwatch=f[cfg['runtime_paths']['supervisor/workload_watch_v1.py']]
    support={'predictor_pgd_split_watch_v1.py':old['predictor_pgd_split_watch_v1.py'],
             'scratch_allocation_v2.py':old['scratch_allocation_v2.py'],
             'workload_watch_v1.py':unitwatch,'workload_os_v1.py':f['models/predictor_pgd_workload_os_v1.py'],
             'workload_plan_v1.py':f['models/predictor_pgd_split_plan_v1.py']}
    static['fixed_support']={n:digest(data) for n,data in support.items()}
    sources=dict(support);changes={}
    def finish(name,e):
        sources[name],changes[name]=e.finish()
    e=Editor(old['bootstrap.py']).schemas()
    for name,value in (('ROLES',ROLES),('SECONDS',SECONDS),('SUPERVISORS',slots),('STATIC',static)):
        e.node(name,name+' = '+repr(value),assignment=True)
    e.node('static_contract',STATIC_CONTRACT)
    e.one('len(mapping) != 45','len(mapping) != 55')
    e.all("('models', 'attacks')","('models', 'attacks', 'datasets')")
    e.one("STATIC['source_hashes'] if plan['role'] == 'forward-preserving' else {}","STATIC['source_hashes'] if plan['role'] != 'proof' else {}")
    e.one('Exact39candidate runtime and six supervisor names required.','Exact46whole-PGD runtime and nine supervisor names required.')
    e.one('Six exact supervisor slots required.','Nine exact supervisor slots required.')
    finish('bootstrap.py',e)
    e=Editor(old['predictor_pgd_split_scratch_v1.py']).schemas()
    e.node('STATIC','STATIC = '+repr(static),assignment=True);e.node('static_contract',STATIC_CONTRACT)
    # Only the derived Layout classification changes; original scanner/assessment/Guard remain exact.
    e.one("partial and plain in ('component-inputs.tar', 'native-reference.tar')","partial and plain == 'original-packet.tar.gz'")
    e.one("plain in ('component-inputs.tar', 'native-reference.tar') or plain.startswith(('source/', 'reference/'))","plain == 'original-packet.tar.gz' or plain.startswith('code/input/')")
    e.one("partial_allowed=plain in ('component-inputs.tar', 'native-reference.tar')","partial_allowed=plain == 'original-packet.tar.gz'")
    finish('predictor_pgd_split_scratch_v1.py',e)
    e=Editor(old['predictor_pgd_split_coordinator_v1.py']).schemas()
    for name,value in (('ROLES',ROLES),('SOURCE_HASHES',static['source_hashes'])):
        e.node(name,name+' = '+repr(value),assignment=True)
    e.node('monitor_for',MONITOR);e.node('dispatch_child',DISPATCH)
    e.one(method_node(e.raw,'SystemDriver','verify'),'\n'.join('    '+x if x else '' for x in VERIFY.strip().splitlines()).lstrip())
    e.one("self.work_deadline - 240 - later","self.work_deadline - 120")
    e.one("if role == 'forward-preserving' else {}","if role != 'proof' else {}")
    e.one("('setup', 'proof', 'numerical', 'native', 'forward-preserving', 'finalization')",
        "('setup', 'proof', 'original-unit', 'candidate-unit', 'candidate-replay', 'finalization')")
    finish('predictor_pgd_split_coordinator_v1.py',e)
    e=Editor(old['predictor_pgd_split_collection_v1.py']).schemas()
    finish('predictor_pgd_split_collection_v1.py',e)
    desc={n:{'bytes':len(data),'sha256':digest(data)} for n,data in sources.items()}
    require(len(desc)==9,'Nine closed supervisor slots required.')
    catalogue={'kind':'whole-pgd-whole-scratch-v1','assets':dict(static['assets'],
        **{'supervisor/'+n:d for n,d in desc.items()}),'runtime_files':runtime,'result':static['result'],
        'root_metadata':static['root_metadata'],'caps':static['caps'],'packet':static['packet']}
    policy={'kind':'whole-pgd-runtime-policy-v1','scratch_contract_text':encoded(catalogue).decode(),
        'base_text':static['base_text'],'design_text':static['design_text'],'scope':static['scope']}
    control=encoded({'kind':'whole-pgd-supervised-control-v1','policy_text':encoded(policy).decode(),
        'policy_sha256':digest(encoded(policy)),'supervisor_sources':desc})
    e=Editor(f['models/predictor_pgd_candidate_outer_v1.py']).schemas()
    # Remove the second archive protocol; the frozen original single-packet functions are reused.
    e.node('receive_reference','def receive_reference(*a,**kw):\n    raise ValueError("No component reference archive belongs to a full-unit workload.")')
    e.node('extract_reference','def extract_reference(*a,**kw):\n    raise ValueError("No component reference archive belongs to a full-unit workload.")')
    original_full=f['models/predictor_pgd_split_outer_v4.py']
    for name in ('receive_packet','extract_packet'):
        n=next(n for n in ast.parse(original_full).body if getattr(n,'name',None)==name)
        e.node(name,ast.get_source_segment(original_full.decode(),n))
    local_sources=(OUTER,CLIENT,'models/predictor_pgd_workload_operator_v1.py',INPUTS,
        'models/predictor_pgd_workload_os_v1.py',*READERS.values(),'models/__init__.py','models/predictor_pgd_split_plan_v1.py')
    for name,value in (('POLICY_SHA',digest(encoded(policy))),('BOOT_SHA',desc['bootstrap.py']['sha256']),
        ('COLLECTOR_SHA',desc['predictor_pgd_split_collection_v1.py']['sha256']),('LOCAL_SOURCES',local_sources),
        ('STATIC_RUNTIME',runtime),('SUPERVISOR_NAMES',tuple(desc)),('PACKET_PATH',cfg['packet_path'])):
        e.node(name,name+' = '+repr(value),assignment=True)
    e.one("{'reference_path', 'reference_manifest_sha256', 'kind'","{'kind'")
    e.one(" or c['reference_path'] != REFERENCE_PATH or c['reference_manifest_sha256'] != REFERENCE_MANIFEST_SHA",'')
    e.one("len(c['runtime_files']) != 39","len(c['runtime_files']) != 46")
    e.one("len(mapping) != 45","len(mapping) != 55")
    e.one("All 45 exact candidate and supervisor source files required.","All 55 exact whole-PGD and supervisor source files required.")
    e.one("'full_model_gradients': 0","'full_model_gradients': "+str(120 if method=='primary' else 300))
    e.one("'extra_component_VJPs': 1","'extra_component_VJPs': 0")
    e.node('_setup_directory_plan',DIRECTORIES)
    e.one("    receive_reference(incoming, root, contract['reference']['archive'], check)\n    extract_reference(root, contract['reference']['files'], check)\n",'')
    outer,changes['outer']=e.finish()
    e=Editor(f['models/predictor_pgd_candidate_client_v1.py'])
    for before,after in (('predictor_pgd_candidate_outer_v1','predictor_pgd_workload_outer_v1'),
        ('predictor_pgd_candidate_bootstrap_v1','predictor_pgd_workload_bootstrap_v1'),
        ('predictor_pgd_candidate_collection_v1','predictor_pgd_workload_collection_v1'),
        ('CONTROL-CANDIDATE.json','CONTROL-WORKLOAD.json')):
        e.one(before,after)
    e.one("    reference = root / c['reference_path']\n    wanted_reference = contract['reference']['archive']\n    file_bytes(reference, wanted_reference['bytes'], wanted_reference['sha256'])\n    return (c, b, p, csha, raw, araw, a, mapping, code, (packet, reference))",
        "    return (c, b, p, csha, raw, araw, a, mapping, code, (packet,))")
    client,changes['client']=e.finish()
    generated={OUTER:outer,CLIENT:client}
    for name,path in READERS.items():
        data=sources[name]
        if name=='predictor_pgd_split_collection_v1.py':
            e=Editor(data);e.one('predictor_pgd_split_bootstrap_v1 as boot','predictor_pgd_workload_bootstrap_v1 as boot')
            e.one('predictor_pgd_split_scratch_v1 as scratch','predictor_pgd_workload_scratch_v1 as scratch')
            data,changes['local_collection_bridge']=e.finish()
        generated[path]=data
    uploads={'overlay/'+n:data for n,data in workers.items()}
    uploads.update({'supervisor/'+n:data for n,data in sources.items()})
    require(len(uploads)==55 and sum(map(len,uploads.values()))<=2*2**20,'Original shared source envelope retained.')
    facts={'kind':'whole-pgd-operator-cold-derivation-v1','method':method,'execution_binding':None,
        'runtime_files_unchanged':47,'runtime_bytes_unchanged':450166,'cache_modules':46,
        'supervisor_slots':9,'uploaded_sources':55,'uploaded_bytes':sum(map(len,uploads.values())),
        'whole_byte_inverses':changes,'new_scope_handoff_approval_or_target_work':False,'training_admitted':False}
    return {'control':control,'policy':policy,'catalogue':catalogue,'static':static,
        'sources':sources,'workers':workers,'uploads':uploads,'generated':generated,'derivation':facts}


def write_local(repository,destination,**settings):
    built=assemble(repository,**settings);dest=Path(destination);dest.mkdir(mode=0o700)
    for name,value in {'CONTROL-UNBOUND.json':built['control'],'CATALOGUE.json':encoded(built['catalogue']),
        'DERIVATION.json':encoded(built['derivation']),
        'SOURCE-PAYLOAD.json':encoded({n:base64.b64encode(v).decode() for n,v in built['uploads'].items()})}.items():
        with (dest/name).open('xb') as out:out.write(value)
    return built

