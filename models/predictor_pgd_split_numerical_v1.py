"""LOCAL source-only numerical/progress integration; no scientific runtime entry.

The returned ASTs are UNBOUND. Only metadata progress is executed in current
checks; numerical Capture uses explicit opaque fake callbacks. No old grant,
worker, proof, profiler, gradient or cohort can run through this builder.
"""
import ast
import copy
import hashlib
import textwrap
from models.predictor_pgd_progress_adapter_v1 import _replace, dump

PINS={
 'worker':'0eb5b56e6e3f74bfdc53e4f1f0f5861e0f066340961fbbfc2b134cc75536f3a0',
 'records':'d2b98fe16a0d1efd997d74092be68cbbcd3f2e73eef7fb44c5fcb2d0843204e8',
 'progress':'fd7458b29118ca6171dee3a3c99eefc0c10dd2ccd238c230ff1a84fa43a4da1a',
 'writer':'24639c3b3d306cc67a02a2327ce2c40b4829788e2da48e373c6edab86255eb0f',
}
WORKER_KIND='pretrained-pgd-split-numerical-worker-v1'
OBJECTIVE_KIND='pretrained-pgd-split-numerical-objective-v1'
ALGORITHM='pretrained-pgd2-r2-eot8-split-observation-v1'


def parse(s):return ast.parse(textwrap.dedent(s))


def replace_exact(tree,before,after):
    left,right=parse(before).body,parse(after).body
    if len(left)==len(right)==1 and isinstance(left[0],ast.Expr) and isinstance(right[0],ast.Expr):
        wanted,new=left[0].value,right[0].value;found=0
        def visit(n):
            nonlocal found
            for field,value in ast.iter_fields(n):
                if isinstance(value,ast.AST):
                    if dump(value)==dump(wanted):setattr(n,field,copy.deepcopy(new));found+=1
                    else:visit(value)
                elif isinstance(value,list):
                    for i,child in enumerate(value):
                        if isinstance(child,ast.AST):
                            if dump(child)==dump(wanted):value[i]=copy.deepcopy(new);found+=1
                            else:visit(child)
        visit(tree)
        if found!=1:raise ValueError('Exactly one frozen expression anchor required: '+str(found))
    else:_replace(tree,before,after)


class Edits:
    def __init__(self,tree):self.tree=tree;self.original=dump(tree);self.edits=[]
    def apply(self,before,after):
        replace_exact(self.tree,before,after);self.edits.append((before,after))
    def verify(self):
        reverted=copy.deepcopy(self.tree)
        for before,after in reversed(self.edits):replace_exact(reverted,after,before)
        if dump(reverted)!=self.original:raise ValueError('Undeclared source edit; refuse.')
        return len(self.edits)


WORKER_EDITS=(
("from models import predictor_pgd_target_records_v1 as records", '''
if False:
    from models import predictor_pgd_target_records_v1 as records
'''),
("folder = Path(args.output);folder.mkdir(mode=0o700,exist_ok=False)", '''
if records.KIND != 'pretrained-pgd-split-numerical-worker-v1' or records.OBJECTIVE != 'pretrained-pgd-split-numerical-objective-v1' or progress.FORMAT != 'post-gradient-progress-v2':
    raise ValueError('New split numerical records/progress bindings required; no legacy worker completion.')
folder = Path(args.output);folder.mkdir(mode=0o700,exist_ok=False)
'''),
('''
def progress():
    records.save_json(folder,'worker.json',state,replace=(folder/'worker.json').exists())
''','''
def save_progress():
    records.save_json(folder.parent,'run-status.json',state,replace=(folder.parent/'run-status.json').exists())
'''),
("self.stack=ExitStack();self.input_version=self.value._version;self.input_hash=tensor_signature(self.value)",
 "self.progress_journal=None;self.observer=None;self.stack=ExitStack();self.input_version=self.value._version;self.input_hash=tensor_signature(self.value)"),
("if self.step==0:\n    self.profile=self.stack.enter_context(torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False,profile_memory=False))", '''
self.progress_journal=progress.Journal(self.folder,control_sha256=args.control_sha256,
    objective_index=self.index,provisional=self.index<2)
self.observer=progress.Observer(self.progress_journal)
self.stack=progress.ObservedExitStack(self.observer)
'''),
('self.stack.enter_context(self.route)', '''
self.stack.named_enter_context('native_exit',self.route)
if self.index==0:
    self.route.phase='EOT8-before'
    observed=boundary.attach(self.route,source_value=self.value,source_objective_index=self.index,
        control_sha256=args.control_sha256,folder=folder.parent/'boundary')
    self.stack.named_enter_context('boundary_exit',observed)
else:
    self.stack.named_callback('boundary_exit',lambda:None)
'''),
('self.stack.callback(remove)',"self.stack.named_callback('hook_removal',remove)"),
('''
self.save('input',gradient)
if len(self.arrays)!=37 or not self.arrays['input'].any():raise ValueError('All actual returns and nonzero full gradient required.')
if self.reference is not None:self.record['comparison']=records.compare(self.arrays,self.reference.arrays)
''','''
self.save('input',gradient)
self.observer.note_gradient(self.record['artifacts']['input'])
def _original_gradient_checks():
    if len(self.arrays)!=37 or not self.arrays['input'].any():raise ValueError('All actual returns and nonzero full gradient required.')
    if self.reference is not None:self.record['comparison']=records.compare(self.arrays,self.reference.arrays)
self.observer.call('gradient_checks',_original_gradient_checks)
if self.index<2:
    def _provisional():
        n=self.route.native if self.candidate else self.route
        return progress.provisional_snapshot(self.progress_journal,calls=n.rows,scopes=n.scopes,refusals=n.refusals,
            vjps=self.route.prototype.calls if self.candidate else [],borrowed=self.route.prototype.scope if self.candidate else None)
    self.observer.call('partial_snapshot',_provisional)
'''),
("self.stack.__exit__(kind,error,tb);self.record['events'].append('exit');active[0]=None;idle()", '''
self.stack.__exit__(kind,error,tb);self.record['events'].append('exit');active[0]=None
if self.observer is None:idle()
else:self.observer.call('restoration_checks',idle)
'''),
('''
if self.value._version!=self.input_version or tensor_signature(self.value)!=self.input_hash or self.value.grad is not None:
    raise ValueError('Input/version changed or input .grad populated.')
''','''
def _original_input_checks():
    if self.value._version!=self.input_version or tensor_signature(self.value)!=self.input_hash or self.value.grad is not None:
        raise ValueError('Input/version changed or input .grad populated.')
if self.observer is None:_original_input_checks()
else:self.observer.call('input_restore_checks',_original_input_checks)
'''),
('''
self.record['ownership']=records.check_scope(self.record,n.rows,n.scopes,n.refusals,
    self.route.prototype.calls if self.candidate else [],self.route.prototype.scope if self.candidate else None)
if self.profile is not None:
    observed=records.profile_summary(self.profile,self.record['mode'])
    records.save_json(self.folder,'backend.json',observed,cap=4*2**20)
    self.record['backend_sha256']=records.frozen.sha_file(self.folder/'backend.json')
    records.validate_backend(observed,self.record['mode'])
''','''
self.record['ownership']=self.observer.call('native_proof_validation',records.check_scope,self.record,n.rows,n.scopes,n.refusals,
    self.route.prototype.calls if self.candidate else [],self.route.prototype.scope if self.candidate else None)
'''),
('''
records.write_final(self.folder,self.record,calls=[] if n is None else n.rows,scopes=[] if n is None else n.scopes,refusals=[] if n is None else n.refusals,
    vjps=[] if self.route is None or not self.candidate else self.route.prototype.calls,borrowed=None if self.route is None or not self.candidate else self.route.prototype.scope)
''','''
try:
    proof_args=dict(calls=[] if n is None else n.rows,scopes=[] if n is None else n.scopes,refusals=[] if n is None else n.refusals,
        vjps=[] if self.route is None or not self.candidate else self.route.prototype.calls,
        borrowed=None if self.route is None or not self.candidate else self.route.prototype.scope)
    if self.observer is None:
        if not failed:raise ValueError('No numerical completion without its durable observer.')
        records.write_final(self.folder,self.record,**proof_args)
    else:self.observer.publish(records.write_final,success=not failed,args=(self.folder,self.record),kwargs=proof_args)
finally:
    if self.progress_journal is not None:self.progress_journal.close()
'''),
)


def literal_rewrite(tree,mapping):
    counts={k:0 for k in mapping}
    for n in ast.walk(tree):
        if isinstance(n,ast.Constant) and type(n.value) is str and n.value in mapping:
            old=n.value;n.value=mapping[old];counts[old]+=1
    return counts


def _progress(raw):
    t=ast.parse(raw);e=Edits(t)
    e.apply("FORMAT = 'post-gradient-progress-v1'","FORMAT = 'post-gradient-progress-v2'")
    e.apply("FILE = 'post-gradient-progress-v1.jsonl'","FILE = 'post-gradient-progress-v2.jsonl'")
    e.apply("PHASES = frozenset(('gradient_checks', 'partial_snapshot', 'hook_removal', 'native_exit', 'profiler_stop', 'restoration_checks', 'input_restore_checks', 'native_proof_validation', 'events_materialize', 'events_aggregate', 'backend_publish', 'backend_validate', 'record_publish'))",
        "PHASES = frozenset(('gradient_checks','partial_snapshot','hook_removal','boundary_exit','native_exit','restoration_checks','input_restore_checks','native_proof_validation','record_publish'))")
    e.apply("CLEANUP = frozenset(('hook_removal', 'native_exit', 'profiler_stop'))","CLEANUP = frozenset(('hook_removal','boundary_exit','native_exit'))")
    e.apply("REQUIRED = frozenset(('gradient_checks', 'hook_removal', 'native_exit', 'restoration_checks', 'input_restore_checks', 'native_proof_validation'))",
        "REQUIRED = frozenset(('gradient_checks','hook_removal','boundary_exit','native_exit','restoration_checks','input_restore_checks','native_proof_validation'))")
    e.apply("PROFILE_REQUIRED = frozenset(('partial_snapshot', 'profiler_stop', 'events_materialize', 'events_aggregate', 'backend_publish', 'backend_validate'))","PROFILE_REQUIRED = frozenset(('partial_snapshot',))")
    e.apply("SNAPSHOT_NAMES = ('provisional-native-calls-v1.jsonl', 'provisional-VJP-calls-v1.jsonl', 'provisional-scopes-v1.json', 'provisional-capture-v1.json')",
        "SNAPSHOT_NAMES = ('provisional-native-calls-v2.jsonl','provisional-VJP-calls-v2.jsonl','provisional-scopes-v2.json','provisional-capture-v2.json')")
    e.apply("if (type(calls) is not list or len(calls) > 4096 or type(vjps) is not list or len(vjps) > 384 or type(scopes) is not list or len(scopes) > 1 or type(refusals) is not list or len(refusals) > 32):\n    raise ProgressError('Original per-objective metadata row caps remain unchanged.')",
        "if (type(calls) is not list or len(calls)>1152 or type(vjps) is not list or len(vjps)>384 or (journal.identity['objective_index']==0 and vjps) or type(scopes) is not list or len(scopes)>1 or type(refusals) is not list or len(refusals)>32):\n    raise ProgressError('Split per-objective metadata row caps; no VJPs for the original.')")
    e.apply("((SNAPSHOT_NAMES[0], calls, 2048, 8*2**20), (SNAPSHOT_NAMES[1], vjps, 8192, 8*2**20))",
        "((SNAPSHOT_NAMES[0],calls,2048,1152*2048),(SNAPSHOT_NAMES[1],vjps,8192,384*8192 if journal.identity['objective_index']==1 else 0))")
    count=e.verify()
    # This is an explicit v2 schema rename, not a profile flag set to imply work.
    for n in ast.walk(t):
        if isinstance(n,ast.Name) and n.id=='profiled':n.id='provisional'
        if isinstance(n,ast.arg) and n.arg=='profiled':n.arg='provisional'
    renamed=literal_rewrite(t,{'profiled':'provisional','provisional-native-capture-v1':'provisional-native-capture-v2','post-gradient-progress-read-v1':'post-gradient-progress-read-v2'})
    ctor=next(n for n in ast.walk(t) if isinstance(n,ast.ClassDef) and n.name=='Journal')
    init=next(n for n in ctor.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    init.body.insert(0,parse("if type(objective_index) is not int or not 0<=objective_index<8 or provisional is not (objective_index<2):\n    raise ProgressError('Only first two numerical objectives retain provisional metadata.')").body[0])
    completion=parse("""
def verify_completion(value,*,control_sha256,objective_index):
    if type(value) is not dict or value.get('kind')!='post-gradient-progress-read-v2':
        raise ProgressError('Read-only validated v2 progress required.')
    identity=value.get('identity')
    if (type(objective_index) is not int or not 0<=objective_index<8 or type(identity) is not dict
            or identity.get('control_sha256')!=control_sha256 or identity.get('objective_index')!=objective_index
            or identity.get('provisional') is not (objective_index<2) or value.get('incomplete_tail_bytes')!=0
            or value.get('unreturned_observed_phase') is not None):
        raise ProgressError('Exact completed numerical progress identity required.')
    rows=value.get('rows')
    if type(rows) is not list or not 0<len(rows)<=MAX_ROWS:raise ProgressError('Bounded progress rows required.')
    returned={r['phase'] for r in rows if r['event']=='returned'}
    need=REQUIRED | {'record_publish'} | (PROFILE_REQUIRED if objective_index<2 else frozenset())
    if (not need<=returned or any(r['event']=='raised' for r in rows)
            or not any(r['phase']=='gradient_saved' and r['event']=='observed' for r in rows)
            or rows[-1]['phase']!='record_publish' or rows[-1]['event']!='returned'):
        raise ProgressError('All durable gates and final publication return required.')
    return {'progress_gates_returned':True,'scientific_qualified':False}
""").body
    t.body.extend(completion)
    return ast.fix_missing_locations(t),{'reversible_edits':count,'schema_rename_counts':renamed,'original_restored_before_declared_schema_rename':True,'new_read_only_completion_gate':True}


def _records(raw,writer):
    t=ast.parse(raw);e=Edits(t)
    source=ast.parse(writer)
    original_writer=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='write_final')
    e.apply("write_final = FunctionType(frozen.write_final.__code__, dict(frozen.write_final.__globals__, identity=identity))",
        'atomic=frozen.atomic\nsha_file=frozen.sha_file\n'+ast.unparse(original_writer))
    e.apply("identity(s)", """
identity(s)
progress.verify_completion(progress.read_progress(Path(folder)/'post-gradient-progress-v2.jsonl'),
    control_sha256=sha,objective_index=index)
if (Path(folder)/'failure.json').exists():raise ValueError('Failed numerical marker remains; no completion.')
""")
    e.apply("LIMITS = frozen.LIMITS", "LIMITS = {'worker.json':256*1024,'native-calls.jsonl':1152*2048,'VJP-calls.jsonl':384*8192,'scopes.json':64*1024,'success.json':16*1024,'failure.json':16*1024}")
    e.apply("[('native-calls.jsonl', calls, 4096, 2048), ('VJP-calls.jsonl', vjps, 384, 8192)]", "[('native-calls.jsonl',calls,1152,2048),('VJP-calls.jsonl',vjps,384 if state['index']%2 else 0,8192)]")
    e.apply("not 0 < s.get('wall_seconds',0) <= 1920","not 0 < s.get('wall_seconds',0) <= 1680")
    e.apply("s = json.loads(frozen.read(folder,'worker.json',256*1024))", "s = json.loads(frozen.read(folder.parent,'run-status.json',256*1024))")
    oldblock='''
if step == 0:
    for role, objective in (('original', original),('candidate',candidate)):
        path=folder/('objective-'+str(0 if role=='original' else 1))/'backend.json'
        profile=json.loads(frozen.read(path.parent,path.name,4*2**20))
        if frozen.sha_file(path) != objective['backend_sha256']:
            raise ValueError('Backend aggregate hash changed.')
        validate_backend(profile,role)
'''
    e.apply(oldblock,"if step == 0:\n    pass # Detailed profiling is separate; numerical proof never admits a backend.")
    oldret="return {'worker_sha256':frozen.sha_file(folder/'worker.json'),'pid':s['pid'],'gradient_objectives':8,'same_input_checks':checks,'selection':selected,'numerical_and_backend_scoped':True,'fresh_rollout_replay_qualified':False,'performance_qualified':False,'research_admitted':False}"
    newret="return {'worker_sha256':frozen.sha_file(folder.parent/'run-status.json'),'pid':s['pid'],'gradient_objectives':8,'same_input_checks':checks,'selection':selected,'numerical_scoped':True,'representative_backend_scoped':False,'full_EOT8_backend_qualified':False,'fresh_rollout_replay_qualified':False,'performance_qualified':False,'research_admitted':False}"
    e.apply(oldret,newret)
    count=e.verify()
    rename={'pretrained-pgd2-r2-iterated-worker-v1':WORKER_KIND,'pretrained-pgd2-r2-owned-objective-v1':OBJECTIVE_KIND,
        'pretrained-pgd2-r2-eot8-native-input-context-diagnostic-v1':ALGORITHM,
        'worker.json':'numerical.json','success.json':'numerical-complete.json','failure.json':'numerical-failed.json'}
    changes=literal_rewrite(t,rename)
    removed={n.name:ast.unparse(n) for n in t.body if isinstance(n,ast.FunctionDef) and n.name in ('profile_summary','validate_backend')}
    t.body=[n for n in t.body if not (isinstance(n,ast.FunctionDef) and n.name in removed)]
    return ast.fix_missing_locations(t),{'reversible_edits':count,'explicit_identity_filename_renames':changes,'old_full_model_profile_functions_removed':sorted(removed),'original_restored_before_declared_renames_and_profile_removal':True}


def build(inputs):
    if type(inputs) is not dict or set(inputs)!=set(PINS):raise ValueError('Four exact frozen source inputs required.')
    for name,want in PINS.items():
        if type(inputs[name]) is not bytes or hashlib.sha256(inputs[name]).hexdigest()!=want:
            raise ValueError('Frozen input changed: '+name)
    w=ast.parse(inputs['worker']);e=Edits(w)
    for before,after in WORKER_EDITS:e.apply(before,after)
    # All existing calls to the run-local save callback keep their order.
    callback=next(n for n in ast.walk(w) if isinstance(n,ast.FunctionDef) and n.name=='save_progress')
    calls=0
    for n in ast.walk(w):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='progress':
            n.func.id='save_progress';calls+=1
    reverted=copy.deepcopy(w)
    for n in ast.walk(reverted):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='save_progress':n.func.id='progress'
    audit_e=Edits.__new__(Edits);audit_e.tree=reverted;audit_e.original=e.original;audit_e.edits=e.edits
    count=audit_e.verify()
    names=literal_rewrite(w,{'worker.json':'numerical.json'})
    run=next(n for n in w.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    run.name='unbound_numerical_worker'
    capture=copy.deepcopy(next(n for n in ast.walk(run) if isinstance(n,ast.ClassDef) and n.name=='Capture'))
    p,pa=_progress(inputs['progress']);r,ra=_records(inputs['records'],inputs['writer'])
    output={'numerical':ast.fix_missing_locations(w),'capture':ast.fix_missing_locations(ast.Module(body=[capture],type_ignores=[])),
        'progress':p,'records':r}
    forbidden={'profile_summary','validate_backend'}
    if any(isinstance(n,ast.Attribute) and n.attr in forbidden or isinstance(n,ast.Attribute) and n.attr=='profiler' for n in ast.walk(w)):
        raise ValueError('Numerical source must contain no detailed profiling call.')
    return output,{'kind':'pgd-split-numerical-source-audit-v1','pinned_sources':dict(PINS),'worker_reversible_edits':count,
        'worker_AST_inverse_exact_before_explicit_callback_filename_and_entry_renames':True,
        'save_progress_calls_renamed':calls,'objective_filename_renames':names,'progress':pa,'records':ra,
        'actual_model_or_gradient_execution':False,'target_runner_available':False,'scientific_qualified':False}
