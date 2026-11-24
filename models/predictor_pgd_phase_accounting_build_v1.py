"""Cold opt-in phase-accounting derivation; no client, scope or launch API."""
import ast
import base64
import copy
import json
from pathlib import Path
import stat
from models import predictor_pgd_workload_shutdown_v1 as frozen

INPUTS = 'configs/predictor_pgd_phase_accounting_inputs_v2.json'
INPUTS_SHA = '005e7809c66f5035968a5e0a03e926e61fd0222429b010c9654539dad8f8c92f'
OUTER = 'outer-unbound.py'
LIVE_EXECUTION_BINDING = None
HELPER = 'models/predictor_pgd_phase_accounting_v1.py'
encoded, digest, require = frozen.encoded, frozen.digest, frozen.require
Editor, inverse, assigned = frozen.Editor, frozen.inverse, frozen.assigned


def function(editor, name, transform):
    node = next(n for n in ast.parse(editor.raw).body if isinstance(n, ast.FunctionDef) and n.name == name)
    before = ast.get_source_segment(editor.raw.decode(), node)
    transform(node)
    editor.one(before, ast.unparse(ast.fix_missing_locations(node)))


def method(editor, cls, name, transform):
    frozen.edit_method(editor, cls, name, transform)


def wrap(expression, body):
    return ast.With(items=[ast.withitem(context_expr=ast.parse(expression, mode='eval').body)], body=body)


def target_name(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


def call_name(node, suffix):
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func).endswith(suffix)


def load(repository):
    root = Path(repository)
    raw = (root/INPUTS).read_bytes()
    require(INPUTS_SHA is not None and digest(raw) == INPUTS_SHA, 'Exact phase-accounting input pins required.')
    cfg = json.loads(raw)
    files = {}
    for name, wanted in cfg['files'].items():
        path = root/name
        require(stat.S_ISREG(path.lstat().st_mode), 'Never open a special phase input.')
        data = path.read_bytes()
        require(wanted == {'bytes': len(data), 'sha256': digest(data)}, 'Phase input changed: '+name)
        files[name] = data
    return cfg, files


PARENT_HELPERS = '''
def _phase_parent_open(owner, started):
    from models import predictor_pgd_phase_accounting_v1 as phases
    value = phases.PhaseLedger(phases.FileSink(owner.driver.guard, 'phase-parent-events.jsonl'),
        role='parent', control_sha256=owner.csha, plan_sha256=owner.csha,
        deadline=owner.work_deadline, scope_started=started, clock=owner.clock)
    owner.phase_accounting = value
    value.adopt_startup('parent_prepare', started, deadline=owner.started+180)

def _phase_parent_finish(owner, primary=None):
    value = getattr(owner, 'phase_accounting', None)
    if value is not None:
        value.finalize(lambda name, row: _metadata_write(owner.driver.guard, name, row, replace=False),
            'phase-parent-summary.json', primary=primary)

def record_outer_setup(guard, control_sha256, started, ended):
    if not boot.finite(started) or not boot.finite(ended) or not 0 <= started <= ended:
        raise ValueError('Same-parent monotonic outer setup interval required.')
    row = {'kind':'whole-pgd-outer-setup-interval-v1', 'pid':os.getpid(),
        'clock_domain':'process-monotonic/'+str(os.getpid()), 'control_sha256':control_sha256,
        'started':started,'ended':ended,'seconds':ended-started,
        'includes':'prepare_tree archive/source/cache/storage installation and checks',
        'excludes':'own setup-record publication and earlier handshake/root claim',
        'scientific_qualified':False,'training_admitted':False,'performance_qualified':False,
        'observation_authenticates_execution':False}
    return _metadata_write(guard,'phase-outer-setup.json',row,replace=False)
'''


def assemble(repository):
    require(LIVE_EXECUTION_BINDING is None, 'Phase route must remain unbound.')
    cfg, files = load(repository)
    pins = cfg['pins']
    baseline = {name: base64.b64decode(value, validate=True)
                for name, value in json.loads(files[pins['payload']]).items()}
    require(len(baseline) == 55 and sum(map(len, baseline.values())) == 905563, 'Exact shutdown graph required.')
    workers = {n.removeprefix('overlay/'): raw for n, raw in baseline.items() if n.startswith('overlay/')}
    sources = {n.removeprefix('supervisor/'): raw for n, raw in baseline.items() if n.startswith('supervisor/')}
    changes = {}

    name = 'models/_pgd_workload_entry_v1.py'
    e = Editor(workers[name])
    e.one("    from models import predictor_pgd_workload_target_v1 as unit",
        "    from models import predictor_pgd_phase_accounting_v1 as phases\n    phases.require_ledger(PHASE_ACCOUNTING,args)\n    from models import predictor_pgd_workload_target_v1 as unit")
    e.one('unit.EXECUTION_BINDING=binding;unit.OBJECTIVE_TIMER=objective_timer',
        'unit.EXECUTION_BINDING=binding;unit.OBJECTIVE_TIMER=objective_timer;unit.PHASE_ACCOUNTING=PHASE_ACCOUNTING')
    e.append('\nPHASE_ACCOUNTING = None\n')
    workers[name], changes['overlay/'+name] = e.raw, e.edits

    name = 'models/_pgd_workload_inputs_v1.py'
    e = Editor(workers[name])
    def provider(node):
        start = next(i for i,n in enumerate(node.body) if target_name(n, 'started'))
        at = next(i for i,n in enumerate(node.body) if target_name(n, 'result'))
        ret = node.body[-1]
        node.body = node.body[:start]+[
            wrap("unit.PHASE_ACCOUNTING.span('model_preparation')", node.body[start:at]),
            wrap("unit.PHASE_ACCOUNTING.span('prepared_unit')", [node.body[at]]),
            wrap("unit.PHASE_ACCOUNTING.span('post_unit_guards')", node.body[at+1:-1]), ret]
    function(e, 'run_prepared', provider)
    workers[name], changes['overlay/'+name] = e.raw, e.edits

    name = 'models/predictor_pgd_workload_target_v1.py'
    e = Editor(workers[name])
    e.append('\nPHASE_ACCOUNTING = None\n')
    def binding_init(node):
        node.body.append(ast.parse('self.phases = PHASE_ACCOUNTING').body[0])
    method(e, 'Binding', '__init__', binding_init)
    def context(node):
        inner = ast.FunctionDef(name='_phase_context_create',
            args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),
            body=node.body, decorator_list=[])
        node.body = [inner, ast.parse('return self.phases.context(_phase_context_create,index=self.started)').body[0]]
    method(e, 'Binding', 'context', context)
    def objective(node):
        inner = next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name == 'score')
        inner.body = [wrap("self.phases.span('score',index=index)", inner.body)]
    method(e, 'Binding', 'objective', objective)
    def unit_run(node):
        at = next(i for i,n in enumerate(node.body) if isinstance(n,ast.Try))
        main = node.body[at]
        setup = node.body[:at]
        evaluation = next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
            and isinstance(n.value,ast.Call) and ast.unparse(n.value.func) == 'evaluate_local_unit')
        publication = next(i for i,n in enumerate(main.body) if call_name(n,'store.put'))
        main.body = [
            wrap("PHASE_ACCOUNTING.span('unit_setup')", main.body[:evaluation]),
            wrap("PHASE_ACCOUNTING.span('evaluation')", [main.body[evaluation]]),
            wrap("PHASE_ACCOUNTING.span('evaluation_checks')", main.body[evaluation+1:publication]),
            wrap("PHASE_ACCOUNTING.span('unit_publication')", [main.body[publication]]),
            *main.body[publication+1:]]
        for handler in main.handlers:
            for i,part in enumerate(handler.body):
                if isinstance(part,ast.If) and ast.unparse(part.test) == 'baseline is not None':
                    handler.body[i] = wrap("PHASE_ACCOUNTING.restoring('failure_checks')", [part])
        node.body = [wrap("PHASE_ACCOUNTING.span('unit_setup')", setup), main]
    method(e, 'TargetUnit', 'run', unit_run)
    workers[name], changes['overlay/'+name] = e.raw, e.edits

    name = 'models/predictor_pgd_workload_target_evaluation_v1.py'
    e = Editor(workers[name])
    def evaluator(node):
        main = next(n for n in node.body if isinstance(n,ast.Try) and n.finalbody)
        main.finalbody = [wrap("binding.phases.restoring('rng_mode_restore')", main.finalbody)]
    function(e, 'evaluate_local_unit', evaluator)
    workers[name], changes['overlay/'+name] = e.raw, e.edits

    name = 'attacks/pgd_workload_target_v1.py'
    e = Editor(workers[name])
    def attack(node):
        at = next(i for i,n in enumerate(node.body) if isinstance(n,ast.FunctionDef) and n.name == 'consider')
        inner = node.body[at]
        inner.body = [wrap("_phase_ledger.span('selection')", inner.body)]
        node.body.insert(at,ast.parse('_phase_ledger = gradient_context.__self__.phases').body[0])
    function(e, 'pgd_linf_workload_v1', attack)
    workers[name], changes['overlay/'+name] = e.raw, e.edits
    workers[HELPER] = files[pins['helper']]

    # Runtime manifest is a new cold identity; the previous graph is untouched.
    old_manifest = json.loads(files[pins['manifest']])
    runtime = {n:{'bytes':len(raw),'sha256':digest(raw)} for n,raw in workers.items()}
    manifest = {'kind':'whole-pgd-phase-accounting-runtime-manifest-v1','execution_binding':None,
        'prior_manifest':pins['manifest'],'prior_manifest_sha256':digest(files[pins['manifest']]),
        'files':dict(runtime, **{'supervisor/workload_watch_v1.py':old_manifest['files']['supervisor/workload_watch_v1.py']}),
        'scientific_function_bodies_require_only_explicit_phase_bridge_inverse':True,
        'changes':copy.deepcopy(changes),'research_admitted':False,'training_admitted':False}
    manifest_raw = encoded(manifest)

    e = Editor(sources['workload_os_v1.py'])
    e.one('import time\n', 'import time\nfrom models import predictor_pgd_phase_accounting_v1 as phase_api\nPHASE_ACCOUNTING = None\n')
    e.one('    result,_=entry.entry(args,objective_timer=timers.objective)',
        '    entry.PHASE_ACCOUNTING = PHASE_ACCOUNTING\n    result,_=entry.entry(args,objective_timer=timers.objective)')
    def os_run(node):
        # Preserve the old startup/directory claim and entire timer body in order.
        node.args.kwonlyargs.append(ast.arg(arg='entry_started'));node.args.kw_defaults.append(ast.Constant(None))
        at = next(i for i,n in enumerate(node.body) if isinstance(n,ast.With))
        main = node.body[at]
        body = main.body
        compute = body[0]
        pub = next(i for i,n in enumerate(body) if target_name(n,'work'))
        main.body = [compute,wrap("phases.span('worker_validation')",body[1:pub]),
                     wrap("phases.span('worker_publication')",body[pub:])]
        prelude = ast.parse('''global PHASE_ACCOUNTING
if (PHASE_ACCOUNTING is not None or not phase_api.finite(entry_started)
        or not plan['launch_started'] <= entry_started <= started < plan['deadline']):
    raise ValueError('One live same-child entry sample and phase ledger required.')
phases = phase_api.PhaseLedger(phase_api.FileSink(guard,prefix+'phase-events.jsonl'),
    role=plan['role'],control_sha256=plan['control_sha256'],plan_sha256=work_plan(plan,scope)['plan_sha256'],
    deadline=plan['deadline'],scope_started=entry_started,clock=time.monotonic)
PHASE_ACCOUNTING = phases
''').body
        success = ast.parse("phases.finalize(lambda name,row:publish(guard,name,row),prefix+'phase-summary.json',require_counts=True)").body
        failure = ast.parse("""try:
    pass
except BaseException as primary:
    phases.finalize(lambda name,row:publish(guard,name,row),prefix+'phase-summary.json',primary=primary)
    raise
""").body[0]
        failure.body = ast.parse("phases.adopt_startup('child_cache_dispatch',entry_started)").body+[
            wrap("phases.span('worker_body')",[main])]+success
        node.body = node.body[:at]+prelude+[failure,node.body[-1]]
    function(e,'run_unit',os_run)
    def read_complete(node):
        extra = ast.parse("""summary=parsed(guard.read(prefix+'phase-summary.json',65536))
phase_api.validate_saved(guard.read(prefix+'phase-events.jsonl',phase_api.MAX_BYTES),summary,
    role=plan['role'],pid=observed['pid'],control_sha256=plan['control_sha256'],
    plan_sha256=work_plan(plan,scope)['plan_sha256'],deadline=plan['deadline'])
if any(summary['completed_spans'].get(name)!=count for name,count in (('owned_context',40),('selection',42),('score',6))):
    raise ValueError('Complete primary phase counts required by parent reader.')
""").body
        node.body[-1:-1] = extra
    function(e,'read_complete',read_complete)
    sources['workload_os_v1.py'], changes['supervisor/workload_os_v1.py'] = e.raw,e.edits

    # New catalogue paths only; original scanner/assessment/Guard are unchanged.
    static = assigned(sources['bootstrap.py'],'STATIC')
    for name in tuple(static['assets']):
        if name.startswith('overlay/'):static['assets'].pop(name)
    static['assets'].update({'overlay/'+name:row for name,row in runtime.items()})
    static['runtime_files'] = runtime
    static['source_hashes'] = {'runtime_manifest_sha256':digest(manifest_raw),
                              'runtime_contract_sha256':digest(files[pins['design']])}
    static['scope']['runtime_manifest_sha256'] = digest(manifest_raw)
    static['fixed_support']['workload_os_v1.py'] = digest(sources['workload_os_v1.py'])
    for role in ('original-unit','candidate-unit','candidate-replay'):
        prefix='results/workload/'+role+'/'
        static['result'][prefix+'phase-events.jsonl']={'category':'metadata','max_bytes':512*1024,'mutable':True,'partial_allowed':False}
        static['result'][prefix+'phase-summary.json']={'category':'metadata','max_bytes':65536,'mutable':False,'partial_allowed':True}
    static['root_metadata'].update({'phase-parent-events.jsonl':[512*1024,True,False],
        'phase-parent-summary.json':[65536,False,True],'phase-outer-setup.json':[16384,False,True]})
    reservation=sum(d['max_bytes']*(2 if d['partial_allowed'] else 1) for d in static['result'].values())
    reservation+=sum(d[0]*(2 if d[2] else 1) for d in static['root_metadata'].values())
    require(reservation<32*2**20,'Original metadata cap retained including all permitted partials.')
    for name in ('bootstrap.py','predictor_pgd_split_scratch_v1.py'):
        e=Editor(sources[name]);e.assign('STATIC',static)
        if name=='bootstrap.py':
            e.one('len(mapping) != 55','len(mapping) != 56')
            e.one('Exact46whole-PGD runtime and nine supervisor names required.','Exact47whole-PGD runtime and nine supervisor names required.')
            def child_entry(node):
                at=next(i for i,n in enumerate(node.body) if target_name(n,'_CHILD_USED'))
                node.body.insert(at+1,ast.parse('_phase_entry_started = time.monotonic()').body[0])
                node.body[-1].value.keywords.append(ast.keyword(arg='entry_started',value=ast.Name(id='_phase_entry_started',ctx=ast.Load())))
            function(e,'child_entry',child_entry)
        sources[name],changes['supervisor/'+name]=e.raw,e.edits

    e=Editor(sources['predictor_pgd_split_coordinator_v1.py'])
    e.assign('SOURCE_HASHES',static['source_hashes'])
    def dispatch(node):
        node.args.kwonlyargs.append(ast.arg(arg='entry_started'));node.args.kw_defaults.append(ast.Constant(None))
        node.body[-1].value.keywords.append(ast.keyword(arg='entry_started',value=ast.Name(id='entry_started',ctx=ast.Load())))
    function(e,'dispatch_child',dispatch)
    def parent_init(node):
        node.body.append(ast.parse('self.phase_accounting = None').body[0])
    method(e,'Coordinator','__init__',parent_init)
    def parent_run(node):
        main=next(n for n in node.body if isinstance(n,ast.Try))
        first=main.body[0];assert call_name(first,'self.check')
        main.body[0]=ast.Assign(targets=[ast.Name(id='_phase_prepare_started',ctx=ast.Store())],value=first.value)
        main.body.insert(3,ast.parse('_phase_parent_open(self,_phase_prepare_started)').body[0])
        loop=next(n for n in main.body if isinstance(n,ast.For))
        loop.body[4:]=[wrap('self.phase_accounting.span(role,deadline=deadline)',loop.body[4:])]
        main.body[-1:-1]=ast.parse('_phase_parent_finish(self)').body
        handler=main.handlers[0]
        handler.body[1:1]=ast.parse('_phase_parent_finish(self,error)').body
    method(e,'Coordinator','run_work',parent_run)
    e.append(PARENT_HELPERS)
    sources['predictor_pgd_split_coordinator_v1.py'],changes['supervisor/predictor_pgd_split_coordinator_v1.py']=e.raw,e.edits

    descriptors={name:{'bytes':len(raw),'sha256':digest(raw)} for name,raw in sources.items()}
    catalogue={'kind':static['kind'] if 'kind' in static else 'whole-pgd-whole-scratch-v1',
        'assets':dict(static['assets'],**{'supervisor/'+name:d for name,d in descriptors.items()}),
        'runtime_files':runtime,'result':static['result'],'root_metadata':static['root_metadata'],
        'caps':static['caps'],'packet':static['packet']}
    old_control=json.loads(files[pins['control']])
    policy=json.loads(old_control['policy_text'])
    policy.update(scratch_contract_text=encoded(catalogue).decode(),scope=static['scope'])
    control=encoded({'kind':old_control['kind'],'policy_text':encoded(policy).decode(),
        'policy_sha256':digest(encoded(policy)),'supervisor_sources':descriptors})
    e=Editor(files[pins['outer']])
    for name,value in [('POLICY_SHA',digest(encoded(policy))),('BOOT_SHA',descriptors['bootstrap.py']['sha256']),
        ('COLLECTOR_SHA',descriptors['predictor_pgd_split_collection_v1.py']['sha256']),('STATIC_RUNTIME',runtime)]:
        e.assign(name,value)
    e.one("len(c['runtime_files']) != 46","len(c['runtime_files']) != 47")
    e.one('len(mapping) != 55','len(mapping) != 56')
    e.one('All 55 exact whole-PGD and supervisor source files required.','All 56 exact whole-PGD and supervisor source files required.')
    e.one("        coordinator, scratch, guard, binding, derived = prepare_tree(root, sources, c, b, p, a, raw, araw, boot, incoming, alarms.check)",
        "        phase_setup_started = time.monotonic()\n        coordinator, scratch, guard, binding, derived = prepare_tree(root, sources, c, b, p, a, raw, araw, boot, incoming, alarms.check)\n        coordinator.record_outer_setup(guard,c['supervised_control_sha256'],phase_setup_started,time.monotonic())\n        alarms.check()")
    outer=e.raw;changes['outer']=e.edits
    uploads={'overlay/'+name:raw for name,raw in workers.items()}
    uploads.update({'supervisor/'+name:raw for name,raw in sources.items()})
    require(len(uploads)==56 and sum(map(len,uploads.values()))<2*2**20,'Original source envelope retained.')
    for name,edits in changes.items():
        raw=outer if name=='outer' else uploads[name]
        before=files[pins['outer']] if name=='outer' else baseline[name]
        require(inverse(raw,edits)==before,'Phase byte inverse failed: '+name)
    for name,raw in uploads.items():compile(raw,'<UNBOUND-PHASE-'+name+'>','exec')
    compile(outer,'<UNBOUND-PHASE-OUTER>','exec')
    facts={'kind':'whole-pgd-phase-accounting-cold-derivation-v1','execution_binding':None,
        'changes':changes,'uploaded_sources':len(uploads),'uploaded_bytes':sum(map(len,uploads.values())),
        'runtime_files':len(manifest['files']),'runtime_bytes':sum(d['bytes'] for d in manifest['files'].values()),
        'cached_modules':len(runtime),'supervisor_slots':len(sources),'primary_result_paths':len(static['result']),
        'maximum_metadata_bytes_including_partials':reservation,'compute_caps_or_budget_gate_opened':False,
        'new_scope_handoff_or_target_work':False,'research_admitted':False,'training_admitted':False}
    return {'sources':sources,'workers':workers,'uploads':uploads,'control':control,'policy':policy,
        'catalogue':catalogue,'static':static,'manifest':manifest_raw,'derivation':facts,'generated':{OUTER:outer}}


def write_local(repository,destination):
    built=assemble(repository);dest=Path(destination);dest.mkdir(mode=0o700)
    values={'CONTROL-UNBOUND.json':built['control'],'SOURCE-MANIFEST-v1.json':built['manifest'],
        'DERIVATION.json':encoded(built['derivation']),'CATALOGUE.json':encoded(built['catalogue']),
        OUTER:built['generated'][OUTER],
        'SOURCE-PAYLOAD.json':encoded({n:base64.b64encode(v).decode() for n,v in built['uploads'].items()})}
    values.update({'sources/'+n:v for n,v in built['sources'].items()})
    values.update({'workers/'+n:v for n,v in built['workers'].items() if n in (
        HELPER,'models/_pgd_workload_entry_v1.py','models/_pgd_workload_inputs_v1.py',
        'models/predictor_pgd_workload_target_v1.py','models/predictor_pgd_workload_target_evaluation_v1.py',
        'attacks/pgd_workload_target_v1.py')})
    for name,raw in values.items():
        p=dest/name;p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        with p.open('xb') as handle:handle.write(raw)
    return built
