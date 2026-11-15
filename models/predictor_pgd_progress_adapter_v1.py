"""LOCAL-ONLY source prototype for post-gradient observation, with pinned inputs.

Builds an unbound class AST, not a worker/launcher. No imports or execution of the
frozen model, its profiler, or its records module. A future target integration
requires its own reviewed identity, storage accounting and exact authorization.
"""
import ast
import copy
import hashlib
import textwrap

WORKER_SHA256 = '0eb5b56e6e3f74bfdc53e4f1f0f5861e0f066340961fbbfc2b134cc75536f3a0'
RECORDS_SHA256 = 'd2b98fe16a0d1efd997d74092be68cbbcd3f2e73eef7fb44c5fcb2d0843204e8'


def tree(source):
    return ast.parse(textwrap.dedent(source))


def dump(node):
    return ast.dump(node, include_attributes=False)


def _replace(root, before, after):
    old, new = tree(before).body, tree(after).body
    found = 0
    def visit(node):
        nonlocal found
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                i = 0
                while i < len(value):
                    if [dump(x) for x in value[i:i+len(old)] if isinstance(x,ast.AST)] == [dump(x) for x in old]:
                        value[i:i+len(old)] = copy.deepcopy(new)
                        found += 1
                        i += len(new)
                    else:
                        if isinstance(value[i], ast.AST):
                            visit(value[i])
                        i += 1
            elif isinstance(value,ast.AST):
                visit(value)
    visit(root)
    if found != 1:
        raise ValueError('Exactly one frozen source anchor required, found '+str(found))


CHANGES = (
('self.stack=ExitStack()', '''
self.progress_journal=progress.Journal(self.folder,control_sha256=args.control_sha256,
    objective_index=self.index,profiled=self.step==0)
self.observer=progress.Observer(self.progress_journal)
self.stack=progress.ObservedExitStack(self.observer)
'''),
("self.profile=self.stack.enter_context(torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False,profile_memory=False))",
 "self.profile=self.stack.named_enter_context('profiler_stop',torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False,profile_memory=False))"),
('self.stack.enter_context(self.route)', "self.stack.named_enter_context('native_exit',self.route)"),
('self.stack.callback(remove)', "self.stack.named_callback('hook_removal',remove)"),
('''
self.save('input',gradient)
if len(self.arrays)!=37 or not self.arrays['input'].any():raise ValueError('All actual returns and nonzero full gradient required.')
if self.reference is not None:self.record['comparison']=records.compare(self.arrays,self.reference.arrays)
''', '''
self.save('input',gradient)
self.observer.note_gradient(self.record['artifacts']['input'])
def _unchanged_gradient_checks():
    if len(self.arrays)!=37 or not self.arrays['input'].any():raise ValueError('All actual returns and nonzero full gradient required.')
    if self.reference is not None:self.record['comparison']=records.compare(self.arrays,self.reference.arrays)
self.observer.call('gradient_checks',_unchanged_gradient_checks)
if self.profile is not None:
    def _save_provisional():
        n=self.route.native if self.candidate else self.route
        return progress.provisional_snapshot(self.progress_journal,calls=n.rows,scopes=n.scopes,refusals=n.refusals,
            vjps=self.route.prototype.calls if self.candidate else [],borrowed=self.route.prototype.scope if self.candidate else None)
    self.observer.call('partial_snapshot',_save_provisional)
'''),
('''
idle()
if self.value._version!=self.input_version or tensor_signature(self.value)!=self.input_hash or self.value.grad is not None:
    raise ValueError('Input/version changed or input .grad populated.')
''', '''
self.observer.call('restoration_checks',idle)
def _unchanged_input_checks():
    if self.value._version!=self.input_version or tensor_signature(self.value)!=self.input_hash or self.value.grad is not None:
        raise ValueError('Input/version changed or input .grad populated.')
self.observer.call('input_restore_checks',_unchanged_input_checks)
'''),
('''
self.record['ownership']=records.check_scope(self.record,n.rows,n.scopes,n.refusals,
    self.route.prototype.calls if self.candidate else [],self.route.prototype.scope if self.candidate else None)
''', '''
self.record['ownership']=self.observer.call('native_proof_validation',records.check_scope,self.record,n.rows,n.scopes,n.refusals,
    self.route.prototype.calls if self.candidate else [],self.route.prototype.scope if self.candidate else None)
'''),
("observed=records.profile_summary(self.profile,self.record['mode'])", '''
materialized=self.observer.call('events_materialize',_materialize,self.profile)
observed=self.observer.call('events_aggregate',_aggregate,materialized,self.record['mode'])
'''),
("records.save_json(self.folder,'backend.json',observed,cap=4*2**20)",
 "self.observer.call('backend_publish',records.save_json,self.folder,'backend.json',observed,cap=4*2**20)"),
("records.validate_backend(observed,self.record['mode'])",
 "self.observer.call('backend_validate',records.validate_backend,observed,self.record['mode'])"),
('''
records.write_final(self.folder,self.record,calls=[] if n is None else n.rows,scopes=[] if n is None else n.scopes,refusals=[] if n is None else n.refusals,
    vjps=[] if self.route is None or not self.candidate else self.route.prototype.calls,borrowed=None if self.route is None or not self.candidate else self.route.prototype.scope)
''', '''
try:
    self.observer.publish(records.write_final,success=not failed,args=(self.folder,self.record),kwargs={
        'calls':[] if n is None else n.rows,'scopes':[] if n is None else n.scopes,'refusals':[] if n is None else n.refusals,
        'vjps':[] if self.route is None or not self.candidate else self.route.prototype.calls,
        'borrowed':None if self.route is None or not self.candidate else self.route.prototype.scope})
finally:
    self.progress_journal.close()
'''),
)


def prototype(worker_bytes, records_bytes):
    """Return unbound AST plus reversible-edit evidence; execute nothing."""
    if (hashlib.sha256(worker_bytes).hexdigest() != WORKER_SHA256
            or hashlib.sha256(records_bytes).hexdigest() != RECORDS_SHA256):
        raise ValueError('Frozen worker/records source changed; no fallback.')
    original = next(n for n in ast.walk(ast.parse(worker_bytes)) if isinstance(n,ast.ClassDef) and n.name=='Capture')
    observed = copy.deepcopy(original)
    for before, after in CHANGES:
        _replace(observed,before,after)
    reverted = copy.deepcopy(observed)
    for before, after in reversed(CHANGES):
        _replace(reverted,after,before)
    if dump(reverted) != dump(original):
        raise ValueError('Scientific source changed outside declared observation edits.')
    observed.name='UnboundCapturePrototype'
    summary = next(n for n in ast.parse(records_bytes).body if isinstance(n,ast.FunctionDef) and n.name=='profile_summary')
    # Retain the exact producer body: groups={}, list(events()), the existing
    # post-materialization cap, then its original ancestry/kernel aggregation.
    body=copy.deepcopy(summary.body[1:])  # Original docstring has no runtime gates.
    if not (len(body)>=5 and isinstance(body[0],ast.Assign) and dump(body[0])==dump(tree('groups={}').body[0])
            and dump(body[1])==dump(tree('events=list(profile.events())').body[0]) and isinstance(body[2],ast.If)):
        raise ValueError('Frozen profiler boundary source changed.')
    materialize=tree('def _materialize(profile):\n    pass').body[0]
    materialize.body=copy.deepcopy(body[:3])+tree('return events,groups').body
    aggregate=tree('def _aggregate(materialized,role):\n    pass').body[0]
    aggregate.body=tree('events,groups=materialized').body+copy.deepcopy(body[3:])
    if dump(ast.Module(body=materialize.body[:-1]+aggregate.body[1:],type_ignores=[])) != dump(ast.Module(body=body,type_ignores=[])):
        raise ValueError('Frozen profiler predicates/ancestry/kernel aggregation changed.')
    module=ast.fix_missing_locations(ast.Module(body=[materialize,aggregate,observed],type_ignores=[]))
    return module, {'kind':'pgd-finalization-observer-source-audit-v1','source_edits':len(CHANGES),
                    'frozen_worker_sha256':WORKER_SHA256,'frozen_records_sha256':RECORDS_SHA256,
                    'inverse_ast_exact':True,'profiler_body_reassembled_exact':True,
                    'scientific_qualified':False,'target_runner_available':False}
