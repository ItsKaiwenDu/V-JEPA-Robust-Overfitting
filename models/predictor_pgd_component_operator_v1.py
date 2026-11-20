"""Cold assembly of a component-only operator; never launches, connects or arms.

The emitted OS interfaces are exercised with explicit fake bindings in local tests.
Every emitted byte and all changed AST nodes are recorded for independent review.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
from models import predictor_pgd_component_session_v1 as local

INPUTS='configs/predictor_pgd_component_operator_inputs_v1.json'
INPUTS_SHA='d1c09c1c056556d8d4be5a61d7b303c75b056a6274bbd3f7f361938f547c81d7'
SAVED='runs/pretrained/diagnostic/pgd-joint-handoff-v1-20261118/handoff-parent/linux10614-final-unarmed/attended-session/saved/supervisor/predictor_pgd_split_coordinator_v1.py'
NAMES=('bootstrap.py','predictor_pgd_split_scratch_v1.py','predictor_pgd_split_coordinator_v1.py','predictor_pgd_split_collection_v1.py','predictor_pgd_split_watch_v1.py','scratch_allocation_v2.py')
OUTER='models/predictor_pgd_component_outer_v1.py'
CLIENT='models/predictor_pgd_component_client_v1.py'
CLI='scripts/run_predictor_pgd_component_v1.py'
READERS={'bootstrap.py':'models/predictor_pgd_component_bootstrap_v1.py','predictor_pgd_split_scratch_v1.py':'models/predictor_pgd_component_scratch_v1.py','predictor_pgd_split_collection_v1.py':'models/predictor_pgd_component_collection_v1.py'}
KINDS={'pgd-split-supervised-control-v1':'pgd-component-supervised-control-v1','pgd-split-supervised-university-exception-v1':'pgd-component-supervised-university-exception-v1','pgd-split-parent-session-v1':'pgd-component-parent-session-v1','pgd-split-child-launch-v1':'pgd-component-child-launch-v1','pgd-split-combined-work-receipt-v1':'pgd-component-work-receipt-v1','pgd-split-evidence-stream-v1':'pgd-component-evidence-stream-v1','pgd-joint-evidence-outer-control-v1':'pgd-component-outer-control-v1','pgd-joint-evidence-university-exception-v1':'pgd-component-outer-university-exception-v1','vjepa-pgd-split-v1-':'vjepa-pgd-component-v1-','All four saved roles must finish before a complete-work receipt.':'All three component-only roles must finish before a complete-work receipt.'}
enc,digest,require=local.encoded,local.digest,local.require


def replace(tree,name,source,edits):
    matches=[(parent,index,n) for parent in ast.walk(tree) if isinstance(getattr(parent,'body',None),list) for field in ('body',) for index,n in enumerate(getattr(parent,field,[])) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name]
    require(len(matches)==1,'One explicit function/class replacement: '+name)
    parent,index,old=matches[0];new=ast.parse(source).body[0];parent.body[index]=new
    edits.append({'name':name,'before':ast.unparse(old),'after':ast.unparse(new)})


def assignment(tree,name,value,edits):
    rows=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    require(len(rows)==1,'One explicit assignment: '+name)
    old=rows[0].value;new=ast.parse(repr(value),mode='eval').body;rows[0].value=new
    edits.append({'assignment':name,'before':ast.unparse(old),'after':ast.unparse(new)})


def renamed(tree):
    class Rename(ast.NodeTransformer):
        def visit_Constant(self,n):
            if type(n.value) is str and n.value in KINDS:return ast.copy_location(ast.Constant(KINDS[n.value]),n)
            return n
    return Rename().visit(tree)


def emit(tree):
    raw=(ast.unparse(ast.fix_missing_locations(tree))+'\n').encode();compile(raw,'<UNINVOKED-COMPONENT-OPERATOR>','exec');return raw


CONTROL='''
def control(data):
    if type(data) is not bytes or len(data)>256*1024:raise ValueError('Bounded component control required.')
    c=parsed(data)
    if type(c) is not dict or set(c)!={'kind','policy_text','policy_sha256','supervisor_sources'} or c['kind']!='pgd-component-supervised-control-v1' or encoded(c)!=data:raise ValueError('Exact canonical component control required.')
    p=parsed(c['policy_text']);validate_policy(p,c['supervisor_sources'])
    if digest(c['policy_text'].encode())!=c['policy_sha256']:raise ValueError('Component policy digest changed.')
    return c,p,digest(data)
'''
POLICY='''
def validate_policy(p,supervisors):
    if type(p) is not dict or set(p)!={'kind','scratch_contract_text','base_text','design_text','scope'} or p['kind']!='component-only-runtime-policy-v1':raise ValueError('Exact component-only policy required.')
    if p['base_text']!=STATIC['base_text'] or p['design_text']!=STATIC['design_text'] or p['scope']!=STATIC['scope']:raise ValueError('Original scientific/host/resource policy changed.')
    if type(supervisors) is not dict or set(supervisors)!=set(SUPERVISORS.values()):raise ValueError('Six exact supervisor slots required.')
    for name,d in supervisors.items():
        if type(d) is not dict or set(d)!={'bytes','sha256'} or type(d['bytes']) is not int or not 0<d['bytes']<=2*2**20 or not sha(d['sha256']):raise ValueError('Bounded supervisor descriptor required.')
    if any(supervisors[n]['sha256']!=want for n,want in STATIC['fixed_support'].items()):raise ValueError('Frozen proof/owned-monitor bytes changed.')
    expected=static_contract(supervisors)
    if p['scratch_contract_text']!=encoded(expected).decode():raise ValueError('Exact whole component scratch catalogue required.')
    return p
'''
STATIC_CONTRACT='''
def static_contract(supervisors):
    assets=dict(STATIC['assets'])
    for name,d in supervisors.items():assets['supervisor/'+name]=dict(d)
    return {'kind':'component-only-whole-scratch-v1','assets':assets,'runtime_files':STATIC['runtime_files'],'result':STATIC['result'],'root_metadata':STATIC['root_metadata'],'caps':STATIC['caps'],'packet':STATIC['packet']}
'''
LAYOUT='''
class Layout(_OriginalLayout):
    def __init__(self,contract_bytes,base_bytes,design_bytes,installation_bytes,installation_sha256):
        c=parse_json(contract_bytes);install=parse_json(installation_bytes)
        if base_bytes.decode()!=STATIC['base_text'] or design_bytes.decode()!=STATIC['design_text'] or encoded(c)!=contract_bytes or install.get('kind')!='pgd-split-scratch-installation-v1' or hashlib.sha256(installation_bytes).hexdigest()!=installation_sha256:raise ValueError('Exact component layout/config/source installation required.')
        files=install.get('files')
        if type(files) is not dict or set(files)!=set(STATIC['supervisor_names']) or c!=static_contract(files):raise ValueError('Complete exact component catalogue and supervisor source set required.')
        self.contract=c;self.result=c['result'];self.assets=dict(c['assets']);self.installation=install;self.installation_sha256=installation_sha256
        self.assets['installation.json']={'bytes':len(installation_bytes),'sha256':installation_sha256}
        self.directories={'.','capture','cache','cache/cuda','tmp'}
        for name in (*self.assets,*STATIC['root_metadata'],*(('capture/'+n) for n in self.result)):self.directories.update(str(p) for p in PurePosixPath(name).parents)
    def classify(self,name):
        safe(name)
        if name.startswith('capture/'):
            d,plain,partial=plan.classify(name[8:],self.result);return dict(d,plain='capture/'+plain,partial=partial,space='result')
        if name in ('space-proof-v2.bin.partial','emergency-space.bin'):
            proof=name.startswith('space-proof');return dict(category='proof' if proof else 'emergency',max_bytes=5*GiB//2 if proof else 16*MiB,plain=name,partial=False,mutable=False,partial_allowed=False,space='proof' if proof else 'emergency')
        if name.startswith(('cache/','tmp/')):
            if len(PurePosixPath(name).parts)>8:raise ValueError('Cache directory depth ceiling.')
            return dict(category='cache',max_bytes=32*MiB,plain=name,partial=False,mutable=True,partial_allowed=False,space='cache')
        partial=name.endswith('.partial');plain=name.removesuffix('.partial')
        if plain in STATIC['root_metadata']:
            cap,mutable,allow=STATIC['root_metadata'][plain]
            if partial and not allow:raise ValueError('Disallowed supervisor partial.')
            return dict(category='metadata',max_bytes=cap,plain=plain,partial=partial,mutable=mutable,partial_allowed=allow,space='metadata')
        if name in self.assets or partial and plain=='component-inputs.tar':
            d=self.assets[plain];return dict(category='input' if plain=='component-inputs.tar' or plain.startswith('source/') else 'code',max_bytes=d['bytes'],plain=plain,partial=partial,mutable=False,partial_allowed=plain=='component-inputs.tar',space='asset')
        raise ValueError('Unknown component scratch path: '+name)
'''
IMPORTER_CHECK='''
def require_namespace(root,binding,csha):
    candidates=[v for v in sys.meta_path if getattr(v,'kind',None)=='pgd-split-pinned-importer-v1' and getattr(v,'root',None)==Path(root) and getattr(v,'binding',None)==binding and getattr(v,'control_sha256',None)==csha and getattr(v,'installed',False)]
    if len(candidates)!=1:raise ValueError('One trusted fresh pinned component importer required.')
    finder=candidates[0]
    for name,module in list(sys.modules.items()):
        if name.split('.')[0] not in ('models','attacks'):continue
        if name not in finder.files or name not in finder.loaded:raise ValueError('Unpinned/preloaded project namespace refused.')
        path,data,package=finder.files[name];expected=str(Path(root)/path)
        if getattr(module,'__file__',None)!=expected or getattr(module,'__loader__',None) is not finder or getattr(getattr(module,'__spec__',None),'loader',None) is not finder or getattr(module.__spec__,'origin',None)!=expected or package and getattr(module,'__path__',None)!=[str((Path(root)/path).parent)]:raise ValueError('Pinned project origin/loader/path changed.')
        if private_read(root,binding,path,len(data))!=data:raise ValueError('Loaded source bytes changed.')
        if '.' in name:
            parent,child=name.rsplit('.',1)
            if getattr(sys.modules.get(parent),child,None) is not module:raise ValueError('Pinned project parent-child identity changed.')
    return finder
'''
VERIFY='''
def verify(self,role,plan,observed):
    o=self.owner;g=self.guard
    if role not in ROLES or observed.get('pid') is None or observed.get('exit_status')!=0 or observed.get('parent_observation_passed') is not True:raise ValueError('Successful owned component/proof observation required.')
    if role=='proof':
        raw=g.read('allocation-result-v2.json',16*1024);receipt=g.verify_proof(boot.digest(raw))
        self.sources=dict(SOURCE_HASHES)
    else:
        from models import _pgd_component_io_v1 as io
        from models import _pgd_component_job_v1 as job
        if io._STORE is None:io.bind(o.root/'capture')
        prefix='capture/results/components/'+role+'/'
        marker_bytes=g.read(prefix+'complete.json',16*1024);marker=boot.parsed(marker_bytes)
        progress=g.read(prefix+'progress.jsonl',64*1024)
        binding={'design_sha256':o.policy['scope']['design_sha256'],'control_sha256':o.csha,'boundary_sha256':SOURCE_HASHES['boundary_sha256'],'probe_index':int(role=='forward-preserving')}
        job.read_completion(o.root/prefix,role=role,binding=binding,marker_sha256=boot.digest(marker_bytes),progress_sha256=boot.digest(progress))
        worker=boot.parsed(g.read(prefix+'worker.json',256*1024));resource=boot.parsed(g.read(prefix+'resource.json',16*1024))
        receipt=job.bind_parent_metadata(marker,worker,resource,observed,role=role,binding=binding)
    _metadata_write(g,'parent-'+role+'.json',observed,replace=False)
    g.scan(phase='capture',settled=True)
    return {'role':role,'pid':observed['pid'],'start_ticks':observed['start_ticks'],'saved_gates_returned':True,'receipt':receipt,'source_hashes':dict(self.sources),'scientific_qualified':False,'training_admitted':False}
'''
DISPATCH='''
def dispatch_child(root,binding,cbytes,abytes,plan):
    import resource
    c,policy,csha=boot.control(cbytes);boot.authorization(abytes,csha);target_gate(root,binding);gpu_gate()
    boot.validate_plan(plan,csha,boot.parsed(boot.private_read(root,binding,'session.json',128*1024)),time.monotonic())
    ib,isha=boot.installation(c,csha)
    if boot.private_read(root,binding,'installation.json',64*1024)!=ib:raise ValueError('Child installation bytes changed.')
    layout=scratch.Layout(policy['scratch_contract_text'].encode(),policy['base_text'].encode(),policy['design_text'].encode(),ib,isha)
    guard=scratch.Guard(root,binding,layout);guard.verify_assets(phase='setup' if plan['role']=='proof' else 'capture')
    if not boot.finite(time.monotonic()) or time.monotonic()>=plan['deadline']:raise TimeoutError('Original child phase exhausted before work.')
    limit=3*2**30 if plan['role']=='proof' else 16*2**20
    resource.setrlimit(resource.RLIMIT_FSIZE,(limit,limit))
    if plan['role']=='proof':
        guard.proof_readiness()
        from models.scratch_allocation_v2 import bounded_write_proof
        bounded_write_proof(root,deadline=plan['deadline']);return 0
    if plan['role'] not in ('native','forward-preserving') or plan['source_hashes']!=SOURCE_HASHES:raise ValueError('Only source-bound component workers are admitted.')
    from models import _pgd_component_entry_v1 as runtime
    if runtime.EXECUTION_BINDING is not None:raise ValueError('One fresh null component entry binding required.')
    runtime.EXECUTION_BINDING={'kind':'component-only-reviewed-execution-binding-v1','control_sha256':csha,'capsule_root':str(root/'source'),'result_root':str(root/'capture'),'host':HOST['hostname'],'capsule_manifest_sha256':boot.STATIC['capsule_manifest_sha256']}
    runtime.component_entry(Namespace(capsule_root=str(root/'source'),result_root=str(root/'capture'),control_sha256=csha,role=plan['role'],deadline=plan['deadline']))
    return 0
'''
VALIDATE_OUTER='''
def validate_control(raw,code_sha256):
    if type(raw) is not bytes or len(raw)>256*1024:raise ValueError('Bounded component outer control required.')
    c=parsed(raw)
    keys={'kind','supervised_control_text','supervised_control_sha256','outer_source_sha256','local_source_sha256','runtime_files','repository','local_handoff','packet_path','known_hosts_text','known_hosts_sha256','expected_ssh_fingerprint','financial_state_unchanged','authorized_at_preparation','target_helper_reviewed'}
    if type(c) is not dict or set(c)!=keys or c['kind']!='pgd-component-outer-control-v1' or encoded(c)!=raw or c['outer_source_sha256']!=code_sha256 or not sha(code_sha256) or c['packet_path']!=PACKET_PATH or c['known_hosts_text']!=KNOWN_HOSTS or c['known_hosts_sha256']!=digest(KNOWN_HOSTS.encode()) or c['expected_ssh_fingerprint']!=FINGERPRINT or c['financial_state_unchanged'] is not True or c['authorized_at_preparation'] is not False or c['target_helper_reviewed'] is not False:raise ValueError('Exact new component operator/control/host binding required.')
    if type(c['local_source_sha256']) is not dict or set(c['local_source_sha256'])!=set(LOCAL_SOURCES) or any(not sha(v) for v in c['local_source_sha256'].values()) or c['local_source_sha256'][LOCAL_SOURCES[0]]!=code_sha256:raise ValueError('Complete exact component operator inputs required.')
    for key in ('repository','local_handoff'):
        value=c[key]
        if type(value) is not str or not Path(value).is_absolute() or str(Path(value))!=value or '..' in Path(value).parts:raise ValueError('Exact canonical local handoff/repository required.')
    text=c['supervised_control_text']
    if type(text) is not str or len(text.encode())>256*1024 or digest(text.encode())!=c['supervised_control_sha256']:raise ValueError('Component supervised control digest changed.')
    b=parsed(text);p=parsed(b['policy_text'])
    if digest(b['policy_text'].encode())!=POLICY_SHA or b['policy_sha256']!=POLICY_SHA or c['runtime_files']!=STATIC_RUNTIME or len(c['runtime_files'])!=37:raise ValueError('Pinned component-only policy/runtime changed.')
    if set(b['supervisor_sources'])!=set(SUPERVISOR_NAMES) or b['supervisor_sources']['bootstrap.py']['sha256']!=BOOT_SHA or b['supervisor_sources']['predictor_pgd_split_collection_v1.py']['sha256']!=COLLECTOR_SHA:raise ValueError('Pinned component supervisors changed.')
    return c,b,p,digest(raw)
'''
DIRECTORIES='''
def _setup_directory_plan(layout):
    if len(layout.assets)!=83 or len(layout.result)!=30:raise ValueError('Exact closed component source/result catalogue required.')
    selected={'.','capture','capture/logs','capture/results','capture/results/components','cache','cache/cuda','tmp','supervisor','configs','source','overlay'}
    for name in layout.assets:selected.update(str(p) for p in PurePosixPath(name).parents)
    claims={r:'capture/results/components/'+r for r in ('native','forward-preserving')}
    if any(n==p or n.startswith(p+'/') for n in selected for p in claims.values()):raise ValueError('Component output directories belong only to their workers.')
    for name in selected:layout.directory(name)
    return {'kind':'component-only-minimal-directory-plan-v1','setup_directories':sorted(selected,key=lambda n:(len(PurePosixPath(n).parts),n)),'worker_claim_directories':claims,'authorized':False,'scientific_qualified':False,'training_admitted':False}
'''
EXTRACT='''
def extract_packet(root,files,check):
    if files!=CAPSULE_FILES or len(files)!=36:raise ValueError('Exact closed capsule member set required.')
    seen=set()
    with tarfile.open(Path(root)/'component-inputs.tar',mode='r:') as archive:
        for item in archive:
            check();name=safe(item.name)
            if name in seen or name not in files or not item.isfile() or item.size!=files[name]['bytes']:raise ValueError('Unexpected duplicate/nonregular capsule member.')
            with archive.extractfile(item) as source:data=read_exact(source,item.size,check)
            if digest(data)!=files[name]['sha256']:raise ValueError('Pinned capsule member changed.')
            destination=Path(root)/'source'/name;destination.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            q=destination.parent
            while q!=Path(root):
                if q.is_symlink() or stat.S_IMODE(q.stat().st_mode)!=0o700 or q.stat().st_uid!=os.getuid():raise ValueError('Private capsule ancestry required.')
                q=q.parent
            with os.fdopen(os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as out:write_all(out,data,check);out.flush();os.fsync(out.fileno())
            seen.add(name)
    if seen!=set(files):raise ValueError('Incomplete capsule; no proof or component work.')
    check()
'''


def assemble(repository):
    root=Path(repository);contract=(root/INPUTS).read_bytes();require(digest(contract)==INPUTS_SHA,'Exact cold operator inputs required.')
    inputs=json.loads(contract);f={}
    for name,d in inputs['files'].items():
        raw=(root/name).read_bytes();require(len(raw)==d['bytes'] and digest(raw)==d['sha256'],'Pinned operator input changed: '+name);f[name]=raw
    b=local.Blueprint(root);edits={};static={'base_text':b.base.decode(),'design_text':b.design.decode(),'assets':{n:d for n,d in b.assets.items() if n!='installation.json'},'runtime_files':b.runtime,'result':{n:d for n,d in b.result.items() if n!='logs/proof.log'},'root_metadata':dict(b.root_meta),'caps':local.frozen_scratch.CAPS,'fixed_support':local.frozen_scratch.PINNED_SUPPORT,'supervisor_names':list(NAMES),'capsule_manifest_sha256':digest(b.inputs['capsule_manifest'])}
    static['root_metadata'].pop('local-binding.json');static['root_metadata']['authorization.json']=(64*1024,False,True);static['root_metadata']['proof.log']=(b.result['logs/proof.log']['max_bytes'],False,False)
    static['scope']={'kind':'component-only-source-interface-not-a-grant','design_sha256':digest(b.design),'new_full_model_objectives':0,'completed_objectives_repeated':0,'components':['native','forward-preserving'],'component_seconds':120,'proof_seconds':60,'work_seconds':2400,'collection_seconds':600,'connection_seconds':3000,'setup_bytes':3*2**30,'capture_bytes':9*2**30//4,'response_bytes':3*2**30,'GPU_bytes':7*2**30,'component_RSS_bytes':4*2**30,'financial_state_unchanged':True}
    members={n.removeprefix('source/'):d for n,d in static['assets'].items() if n.startswith('source/')}
    static['packet']={'archive':static['assets']['component-inputs.tar'],'files':members}
    static=json.loads(enc(static))
    source_hashes={k:b.capsule['files'][v]['sha256'] for k,v in [('numerical_sha256','provenance/run-status.json'),('original_sha256','provenance/objective-0/numerical.json'),('boundary_sha256','boundary/boundary.json')]}
    sources={};bt=ast.parse(f['models/predictor_pgd_joint_bootstrap_v1.py']);bt.body=bt.body[:next(i for i,n in enumerate(bt.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_STARTUP_PACKET_IMPORTS' for t in n.targets))];be=[]
    assignment(bt,'ROLES',local.ROLES,be);assignment(bt,'SECONDS',local.SECONDS,be)
    replace(bt,'control',CONTROL,be);replace(bt,'require_namespace',IMPORTER_CHECK,be)
    # Closed43-source graph, no packet-level datasets imports or numerical entry.
    importer=next(n for n in bt.body if isinstance(n,ast.ClassDef) and n.name=='PinnedImporter')
    for n in ast.walk(importer):
        if isinstance(n,ast.Constant) and n.value==69:n.value=43
        elif isinstance(n,ast.Constant) and n.value=='Exact 63 runtime and six supervisor names required.':n.value='Exact37component runtime and six supervisor names required.'
    bt.body.extend(ast.parse('STATIC='+repr(static)+'\n'+STATIC_CONTRACT+POLICY).body);finder=next(n for n in importer.body if isinstance(n,ast.FunctionDef) and n.name=='find_spec')
    finder.body[0:0]=ast.parse('''
if fullname.split('.')[0] in ('models','attacks'):
    if target is not None:raise ImportError('Pinned project reload refused.')
    if '.' in fullname:
        parent=fullname.rsplit('.',1)[0];module=sys.modules.get(parent)
        if module is None or getattr(module,'__loader__',None) is not self or path!=getattr(module,'__path__',None):raise ImportError('Exact already-loaded private project parent/search path required.')
    elif path is not None:raise ImportError('Top-level pinned package cannot use an ambient path.')
''').body
    executor=next(n for n in importer.body if isinstance(n,ast.FunctionDef) and n.name=='exec_module')
    index=next(i for i,n in enumerate(executor.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='exec')
    check=ast.parse("if private_read(self.root,self.binding,relative,len(data))!=data:raise ImportError('Source changed around pinned module execution.')").body
    executor.body[index:index]=copy.deepcopy(check);executor.body.extend(check)
    sources[NAMES[0]]=emit(renamed(bt));edits['bootstrap']=be
    st=ast.parse(f['models/predictor_pgd_joint_scratch_v1.py']);se=[]
    old=next(n for n in st.body if isinstance(n,ast.ClassDef) and n.name=='Layout');old.name='_OriginalLayout'
    st.body.extend(ast.parse('STATIC='+repr(static)+'\n'+STATIC_CONTRACT+LAYOUT).body);sources[NAMES[1]]=emit(st);edits['scratch']=se
    ct=ast.parse(f[SAVED]);ce=[];replace(ct,'verify',VERIFY,ce);replace(ct,'dispatch_child',DISPATCH,ce)
    # Keep the repaired outer monitor and cleanup exact apart from proof log placement.
    cls=next(n for n in ct.body if isinstance(n,ast.ClassDef) and n.name=='OuterDriver');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_outer_owned_launch')
    target=next(n for n in method.body if isinstance(n,ast.Try))
    log=next(n for n in target.body if isinstance(n,ast.Assign) and any(isinstance(k,ast.Name) and k.id=='log' for k in n.targets));old_log=ast.unparse(log.value);log.value=ast.parse("o.root/'proof.log' if role=='proof' else o.root/'capture/logs'/(role+'.log')",mode='eval').body
    ce.append({'function':'OuterDriver._outer_owned_launch','log_before':old_log,'log_after':ast.unparse(log.value)})
    # Guarded roles have no numerical branch; exact proof adapter body remains frozen.
    m=next(n for n in ct.body if isinstance(n,ast.FunctionDef) and n.name=='monitor_for')
    for n in ast.walk(m):
        if isinstance(n,ast.Tuple) and len(n.elts)==2 and [getattr(x,'value',None) for x in n.elts]==['proof','numerical']:n.elts=[ast.Constant('proof')]
    final_driver=[n for n in ct.body if isinstance(n,ast.ClassDef) and n.name=='OuterDriver'][-1]
    final_launch=next(n for n in final_driver.body if isinstance(n,ast.FunctionDef) and n.name=='launch')
    note=next(n for n in ast.walk(final_launch) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_note')
    ce.append({'function':'OuterDriver.launch','diagnostic_before':ast.unparse(note.args[0])})
    note.args[0]=ast.parse("'Owned '+role+' child exit status: '+str(status)[:64]+'; original child log retained at '+('proof.log' if role=='proof' else 'capture/logs/'+role+'.log')",mode='eval').body
    ce[-1]['diagnostic_after']=ast.unparse(note.args[0])
    ct.body.append(ast.parse('SOURCE_HASHES='+repr(source_hashes)).body[0]);sources[NAMES[2]]=emit(renamed(ct));edits['coordinator']=ce
    sources[NAMES[3]]=emit(renamed(ast.parse(f['models/predictor_pgd_joint_collection_v1.py'])))
    # Target codec imports resolve through the43-source pinned cache.
    sources[NAMES[3]]=sources[NAMES[3]].replace(b'predictor_pgd_joint_bootstrap_v1 as boot',b'predictor_pgd_split_bootstrap_v1 as boot').replace(b'predictor_pgd_joint_scratch_v1 as scratch',b'predictor_pgd_split_scratch_v1 as scratch')
    sources[NAMES[4]]=b.inputs['monitor'];sources[NAMES[5]]=b.inputs['proof']
    descriptors={n:{'bytes':len(v),'sha256':digest(v)} for n,v in sources.items()}
    catalogue={'kind':'component-only-whole-scratch-v1','assets':dict(static['assets'],**{'supervisor/'+n:d for n,d in descriptors.items()}),'runtime_files':b.runtime,'result':static['result'],'root_metadata':static['root_metadata'],'caps':static['caps'],'packet':static['packet']}
    policy={'kind':'component-only-runtime-policy-v1','scratch_contract_text':enc(catalogue).decode(),'base_text':b.base.decode(),'design_text':b.design.decode(),'scope':static['scope']}
    control=enc({'kind':'pgd-component-supervised-control-v1','policy_text':enc(policy).decode(),'policy_sha256':digest(enc(policy)),'supervisor_sources':descriptors})
    ot=ast.parse(f['models/predictor_pgd_joint_outer_v1.py']);oe=[]
    local_sources=(OUTER,CLIENT,CLI,'models/__init__.py','models/predictor_pgd_split_plan_v1.py','models/predictor_pgd_component_operator_v1.py',INPUTS,local.INPUTS,'models/predictor_pgd_component_session_v1.py',*READERS.values())
    for name,value in [('POLICY_SHA',digest(enc(policy))),('BOOT_SHA',descriptors[NAMES[0]]['sha256']),('COLLECTOR_SHA',descriptors[NAMES[3]]['sha256']),('PACKET_PATH','runs/pretrained/diagnostic/pgd-component-capsule-v1-20261119/component-inputs.tar'),('LOCAL_SOURCES',local_sources)]:assignment(ot,name,value,oe)
    for name,text in [('validate_control',VALIDATE_OUTER),('_setup_directory_plan',DIRECTORIES),('extract_packet',EXTRACT)]:replace(ot,name,text,oe)
    # Component-only scope never requests completed full-model gradients.
    auth=next(n for n in ot.body if isinstance(n,ast.FunctionDef) and n.name=='authorization')
    for n in ast.walk(auth):
        if isinstance(n,ast.Dict):
            for index,k in enumerate(n.keys):
                if isinstance(k,ast.Constant) and k.value=='full_model_gradients':n.values[index]=ast.Constant(0)
    dec=next(n for n in ot.body if isinstance(n,ast.FunctionDef) and n.name=='decode_sources')
    for n in ast.walk(dec):
        if isinstance(n,ast.Constant) and n.value==69:n.value=43
    # receive_packet differs only in its fixed archive extent/hash and file literal.
    rcv=next(n for n in ot.body if isinstance(n,ast.FunctionDef) and n.name=='receive_packet')
    class Packet(ast.NodeTransformer):
        def visit_Constant(self,n):
            mapping={17192095:static['packet']['archive']['bytes'],'2278bce2b519943c0f83a49c556bb73dd928fe880fba0f6c56de2a4c3c2ec0da':static['packet']['archive']['sha256'],'original-packet.tar.gz.partial':'component-inputs.tar.partial','original-packet.tar.gz':'component-inputs.tar'}
            if type(n.value) in (str,int) and n.value in mapping:return ast.copy_location(ast.Constant(mapping[n.value]),n)
            return n
    Packet().visit(rcv)
    ot.body.extend(ast.parse('STATIC_RUNTIME='+repr(b.runtime)+'\nCAPSULE_FILES='+repr(members)+'\nSUPERVISOR_NAMES='+repr(NAMES)).body)
    removed=[n for n in ot.body if isinstance(n,ast.FunctionDef) and n.name in ('prepare_package','verify_directory_policy_sources','verify_refusal_policy_sources')]
    ot.body=[n for n in ot.body if n not in removed];oe.append({'removed_obsolete_local_preparation_functions':[n.name for n in removed]})
    outer=emit(renamed(ot));edits['outer']=oe
    readers={path:sources[name] for name,path in READERS.items()}
    readers[READERS[NAMES[3]]]=readers[READERS[NAMES[3]]].replace(b'predictor_pgd_split_bootstrap_v1 as boot',b'predictor_pgd_component_bootstrap_v1 as boot').replace(b'predictor_pgd_split_scratch_v1 as scratch',b'predictor_pgd_component_scratch_v1 as scratch')
    client=f['models/predictor_pgd_joint_client_v1.py']
    for old,new in [('predictor_pgd_joint_outer_v1','predictor_pgd_component_outer_v1'),('predictor_pgd_joint_bootstrap_v1','predictor_pgd_component_bootstrap_v1'),('predictor_pgd_joint_collection_v1','predictor_pgd_component_collection_v1'),('vjepa-pgd-split-v1-','vjepa-pgd-component-v1-')]:client=client.replace(old.encode(),new.encode())
    client=emit(renamed(ast.parse(client)))
    # Exact isolated-Python CLI scaffold; only its client import changes.
    cli=b'#!/usr/bin/env python3\n"""One attended outer attempt, only after a new exact authorization is present."""\nimport argparse\nfrom pathlib import Path\nimport sys\n\n\ndef main(argv=None):\n    parser=argparse.ArgumentParser(description=__doc__)\n    parser.add_argument(\'handoff\',type=Path)\n    args=parser.parse_args(argv)\n    if not (args.handoff/\'AUTHORIZATION.json\').is_file():\n        raise SystemExit(\'Unarmed: no new scoped authorization. No timer, claim, SSH or model work started.\')\n    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))\n    from models.predictor_pgd_component_client_v1 import run\n    return run(args.handoff)\n\n\nif __name__==\'__main__\':\n    sys.exit(main())\n'
    runtime={'overlay/'+n:b.inputs['runtime/'+n] for n in b.runtime};uploads={**runtime,**{'supervisor/'+n:v for n,v in sources.items()}}
    require(len(uploads)==43 and len(catalogue['assets'])==82 and len(static['result'])==30,'Exact closed component runtime/support/catalogue required.')
    return {'control':control,'policy':policy,'catalogue':catalogue,'sources':sources,'uploads':uploads,'generated':{**readers,OUTER:outer,CLIENT:client,CLI:cli},'static':static,'source_hashes':source_hashes,'derivation':{'kind':'component-operator-cold-source-v1','edits':edits,'literal_kinds':KINDS,'runtime_files':37,'runtime_bytes':sum(d['bytes'] for d in b.runtime.values()),'uploaded_sources':43,'uploaded_source_bytes':sum(map(len,uploads.values())),'capsule_bytes':static['packet']['archive']['bytes'],'capsule_sha256':static['packet']['archive']['sha256'],'component_log_isolation':'proof.log outside capture; original worker inventory unchanged','scientific_sources_unchanged':True,'execution_authorized':False,'new_scope_prepared':False,'training_admitted':False}}


def write_local(repository,destination):
    result=assemble(repository);root=Path(repository);dest=Path(destination);dest.mkdir(mode=0o700)
    for name,raw in result['generated'].items():
        p=root/name
        with p.open('xb') as f:f.write(raw)
    for name,value in [('CONTROL-UNARMED.json',result['control']),('CATALOGUE.json',enc(result['catalogue'])),('DERIVATION.json',enc(result['derivation'])),('SOURCE-PAYLOAD.json',enc({n:__import__('base64').b64encode(v).decode() for n,v in result['uploads'].items()}))]:
        with (dest/name).open('xb') as f:f.write(value)
    return result


def prepare_unarmed(repository,destination):
    """Create concrete local bytes only. Never creates an authorization or claim."""
    import base64,os,stat
    root=Path(repository);dest=Path(destination)
    require(root.is_absolute() and root.resolve(strict=True)==root and dest.is_absolute() and dest.parent.resolve(strict=True)==dest.parent and not os.path.lexists(dest),'Fresh canonical local package required.')
    parent=dest.parent.lstat();require(stat.S_ISDIR(parent.st_mode) and parent.st_uid==os.getuid() and stat.S_IMODE(parent.st_mode)==0o700,'Private owned local package parent required.')
    r=assemble(root)
    for name,data in r['generated'].items():require((root/name).read_bytes()==data,'Reviewed generated operator changed: '+name)
    outer={};exec(compile(r['generated'][OUTER],'<UNINVOKED-COMPONENT-PACKAGER>','exec'),outer)
    names=outer['LOCAL_SOURCES'];pins={n:digest((root/n).read_bytes()) for n in names}
    for name in ('models/__init__.py','models/predictor_pgd_split_plan_v1.py'):require(pins[name]==r['static']['runtime_files'][name]['sha256'],'Local reader project dependency changed.')
    c={'kind':'pgd-component-outer-control-v1','supervised_control_text':r['control'].decode(),'supervised_control_sha256':digest(r['control']),'outer_source_sha256':pins[OUTER],'local_source_sha256':pins,'runtime_files':r['static']['runtime_files'],'repository':str(root),'local_handoff':str(dest),'packet_path':outer['PACKET_PATH'],'known_hosts_text':outer['KNOWN_HOSTS'],'known_hosts_sha256':digest(outer['KNOWN_HOSTS'].encode()),'expected_ssh_fingerprint':outer['FINGERPRINT'],'financial_state_unchanged':True,'authorized_at_preparation':False,'target_helper_reviewed':False}
    raw=enc(c);outer['validate_control'](raw,pins[OUTER]);payload=enc({n:base64.b64encode(v).decode() for n,v in r['uploads'].items()});outer['decode_sources'](json.loads(payload),c,json.loads(r['control']))
    require(len(raw)+len(payload)+65536<2*2**20,'Original upload frame cap retained.')
    package={'kind':'component-only-unarmed-local-package-v1','outer_control_sha256':digest(raw),'source_files':43,'source_bytes':sum(map(len,r['uploads'].values())),'source_envelope_bytes':len(payload),'capsule_bytes':r['static']['packet']['archive']['bytes'],'capsule_sha256':r['static']['packet']['archive']['sha256'],'authorization_created':False,'new_scope_approval_requested':False,'target_or_scientific_work':False,'completed_full_model_gradients_repeated':0,'training_admitted':False}
    dest.mkdir(mode=0o700)
    for name,data in [('CONTROL-CANDIDATE.json',raw),('SOURCE-PAYLOAD.json',payload),('known-hosts',outer['KNOWN_HOSTS'].encode()),('PACKAGE.json',enc(package))]:outer['private_write'](dest,name,data,lambda:None)
    return package
