"""Cold shutdown-only derivation of the reviewed v3 workload; no handoff API."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import stat
import textwrap

INPUTS = 'configs/predictor_pgd_workload_shutdown_inputs_v1.json'
INPUTS_SHA = 'acde7da70e989e8c72eb890ef7e9a11e68617ab98eaf141ef27db11f43380a2c'
OUTER = 'outer-unbound.py'
LIVE_EXECUTION_BINDING = None


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def assigned(raw, name):
    return ast.literal_eval(next(n.value for n in ast.parse(raw).body if isinstance(n, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))


class Editor:
    def __init__(self, raw):
        self.raw = raw
        self.edits = []

    def one(self, before, after):
        require(self.raw.count(before.encode()) == 1, 'One exact shutdown edit required.')
        start = self.raw.index(before.encode())
        self.raw = self.raw[:start] + after.encode() + self.raw[start + len(before.encode()):]
        self.edits.append({'start': start, 'before': before, 'after': after})

    def assign(self, name, value):
        node = next(n for n in ast.parse(self.raw).body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        self.one(ast.get_source_segment(self.raw.decode(), node), name + ' = ' + repr(value))

    def definition(self, name, text):
        node = next(n for n in ast.parse(self.raw).body if getattr(n, 'name', None) == name)
        self.one(ast.get_source_segment(self.raw.decode(), node), text.strip())

    def append(self, text):
        self.edits.append({'start': len(self.raw), 'before': '', 'after': text})
        self.raw += text.encode()


def inverse(raw, edits):
    for edit in reversed(edits):
        a = edit['start']
        b = a + len(edit['after'].encode())
        require(raw[a:b] == edit['after'].encode(), 'Exact shutdown inverse required.')
        raw = raw[:a] + edit['before'].encode() + raw[b:]
    return raw


def edit_method(editor, cls, name, transform):
    tree = ast.parse(editor.raw)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in owner.body if getattr(n, 'name', None) == name)
    before = ast.get_source_segment(editor.raw.decode(), node)
    transform(node)
    after = textwrap.indent(ast.unparse(ast.fix_missing_locations(node)), '    ').lstrip()
    editor.one(before, after)


CLEANUP_CALL = '''cleanup_owned(child, leases, root_lease, primary,
    work_deadline=work_deadline, phase_deadline=phase_deadline,
    clock=clock, pause=pause, last=last, samples=count, peak=peak)'''

MONITOR = '''def monitor_for(role):
    from models import predictor_pgd_workload_watch_v1 as unit_watch
    if role in ROLES[1:]:
        return unit_watch.watch
    if role != 'proof':
        raise ValueError('Exact new proof/unit role required.')
    source = Path(frozen_watch.__file__).read_bytes()
    tree, _ = watch_source(source)
    node = tree.body[0]
    if not isinstance(node.body[-1], ast.Try) or not node.body[-1].finalbody:
        raise ValueError('Exact frozen proof monitor cleanup required.')
    node.body[-1].finalbody = ast.parse(PROOF_CLEANUP_CALL).body
    namespace = dict(frozen_watch.__dict__, cleanup_owned=unit_watch.cleanup_owned)
    exec(compile(ast.fix_missing_locations(tree), '<SHUTDOWN-ONLY-PROOF-MONITOR>', 'exec'), namespace)
    return lambda child, lease, **kw: namespace['watch_stage'](child, lease, role=role, **kw)
'''

LAUNCH = '''def launch(self, role, plan):
    self.shutdown_observation = None
    self.owner.outer_alarm.arm(plan['deadline'], hard_deadline=self.owner.work_deadline, phase=role)
    try:
        return self._outer_owned_launch(role, plan)
    except BaseException as error:
        try:
            self.assert_idle()
        except BaseException as idle:
            error.add_note('Clock handoff refused: ' + type(idle).__name__ + ': ' + str(idle)[:256])
        else:
            try:
                self.owner.outer_alarm.arm(min(self.owner.work_deadline, self.owner.clock() + 120),
                                          hard_deadline=self.owner.work_deadline, phase='finalization')
            except BaseException as timer_error:
                error.add_note('Finalization timer refused: '+type(timer_error).__name__)
        raise
'''

LAUNCH_FINALLY = '''cleanup_errors = []
for fd in (readfd, writefd, logfd):
    if fd is not None:
        try:
            os.close(fd)
        except BaseException as extra:
            cleanup_errors.append(extra)
if lease is not None:
    if not monitored:
        from models import predictor_pgd_workload_watch_v1 as unit_watch
        unit_watch.cleanup_owned(child, {(lease.pid, lease.birth): lease}, lease, primary,
            work_deadline=work, phase_deadline=deadline, clock=o.clock, pause=time.sleep,
            last=start, samples=0, peak=0)
    self.shutdown_observation = getattr(lease, 'shutdown_observation', None)
elif child is not None:
    # No held handle means release never occurred. Close the gate (above) and
    # wait only; never guess ownership or send a numeric PID signal.
    terminal = None
    try:
        child.wait(timeout=max(0., min(2., work-o.clock())))
        terminal = child.poll()
    except BaseException as extra:
        cleanup_errors.append(extra)
    self.shutdown_observation = {'kind': 'whole-pgd-unreleased-child-exit-v1',
        'exit_status': terminal, 'confirmed_idle': type(terminal) is int,
        'released_without_held_handle': False, 'scientific_qualified': False}
self.current = None
if cleanup_errors:
    if primary is not None:
        for extra in cleanup_errors[:4]:
            primary.add_note('Launch descriptor/wait cleanup: '+type(extra).__name__)
    else:
        raise cleanup_errors[0]
'''

DIAGNOSTIC_APPEND = '''
_ShutdownPriorErrorRecord = _startup_error_record
def _startup_error_record(error):
    from models import predictor_pgd_workload_watch_v1 as unit_watch
    record = _ShutdownPriorErrorRecord(error)
    record['exception_chain'] = unit_watch.error_chain(error)
    if len(boot.encoded(record)) > 16*1024:
        record['exception_chain'] = {'exceptions': [unit_watch.bounded_error(error)], 'truncated_or_cycle': True}
    return record
'''


def load(repository):
    root = Path(repository)
    raw = (root / INPUTS).read_bytes()
    require(INPUTS_SHA is not None and digest(raw) == INPUTS_SHA, 'Pinned shutdown inputs required.')
    cfg = json.loads(raw)
    files = {}
    for name, descriptor in cfg['files'].items():
        path = root / name
        require(stat.S_ISREG(path.lstat().st_mode), 'Never open a special shutdown input.')
        value = path.read_bytes()
        require(descriptor == {'bytes': len(value), 'sha256': digest(value)}, 'Shutdown input drift: ' + name)
        files[name] = value
    return cfg, files


def assemble(repository, *, method='primary', sensitivity_banks=0):
    require(LIVE_EXECUTION_BINDING is None, 'Cold shutdown execution must stay unbound.')
    require(method == 'primary' and sensitivity_banks == 0, 'This adapter retains the exact v3 primary method.')
    cfg, files = load(repository)
    pins = cfg['pins']
    old_sources = {name: base64.b64decode(value, validate=True)
                   for name, value in json.loads(files[pins['payload']]).items()}
    require(len(old_sources) == 55 and sum(map(len, old_sources.values())) == 896039, 'Exact reviewed v3 closure required.')
    old_control = json.loads(files[pins['control']])
    policy = json.loads(old_control['policy_text'])
    catalogue = json.loads(policy['scratch_contract_text'])
    sources = {name.removeprefix('supervisor/'): raw for name, raw in old_sources.items() if name.startswith('supervisor/')}
    workers = {name.removeprefix('overlay/'): raw for name, raw in old_sources.items() if name.startswith('overlay/')}
    bases = dict(sources)
    changes = {}

    name = 'workload_watch_v1.py'
    e = Editor(sources[name])
    node = next(n for n in ast.parse(e.raw).body if isinstance(n, ast.FunctionDef) and n.name == 'watch')
    before = ast.get_source_segment(e.raw.decode(), node)
    require(isinstance(node.body[-1], ast.Try), 'Exact reviewed unit monitor structure required.')
    node.body[-1].finalbody = ast.parse(CLEANUP_CALL).body
    e.one(before, ast.unparse(ast.fix_missing_locations(node)))
    e.append('\n\n' + files[pins['logic']].decode())
    sources[name] = e.raw
    changes[name] = e.edits

    old_manifest = json.loads(files[pins['manifest']])
    manifest = {'kind': 'whole-pgd-shutdown-runtime-source-manifest-v1',
        'execution_binding': None, 'prior_manifest': pins['manifest'],
        'prior_manifest_sha256': digest(files[pins['manifest']]),
        'files': copy.deepcopy(old_manifest['files']), 'scientific_runtime_files_byte_exact': 46,
        'shutdown_only_monitor_changes': changes[name], 'research_admitted': False, 'training_admitted': False}
    manifest['files']['supervisor/workload_watch_v1.py'] = {'bytes': len(sources[name]), 'sha256': digest(sources[name])}
    manifest_raw = encoded(manifest)
    static = assigned(sources['bootstrap.py'], 'STATIC')
    static['fixed_support']['workload_watch_v1.py'] = digest(sources[name])
    static['source_hashes'] = {'runtime_manifest_sha256': digest(manifest_raw),
                             'runtime_contract_sha256': digest(files[pins['design']])}
    static['scope']['runtime_manifest_sha256'] = digest(manifest_raw)
    for name in ('bootstrap.py', 'predictor_pgd_split_scratch_v1.py'):
        e = Editor(sources[name])
        e.assign('STATIC', static)
        sources[name] = e.raw
        changes[name] = e.edits

    name = 'predictor_pgd_split_coordinator_v1.py'
    e = Editor(sources[name])
    e.assign('SOURCE_HASHES', static['source_hashes'])
    e.definition('monitor_for', MONITOR)
    e.append('\nPROOF_CLEANUP_CALL = ' + repr(CLEANUP_CALL) + '\n')

    def edit_launch(node):
        body = ast.parse(LAUNCH).body[0].body
        node.body = body
    edit_method(e, 'OuterDriver', 'launch', edit_launch)

    def edit_idle(node):
        node.body[:0] = ast.parse("if self.shutdown_observation is not None and not self.shutdown_observation.get('confirmed_idle'):\n    raise RuntimeError('Owned shutdown is unconfirmed; no clock handoff or collection.')").body
    edit_method(e, 'OuterDriver', 'assert_idle', edit_idle)

    def edit_init(node):
        node.body.append(ast.parse('self.shutdown_observation = None').body[0])
    edit_method(e, 'OuterDriver', '__init__', edit_init)

    def edit_owned(node):
        node.body[:0] = ast.parse('primary = None\nself.shutdown_observation = None').body
        block = node.body[-1]
        require(isinstance(block, ast.Try) and not block.handlers, 'Exact owned launch/finally required.')
        block.handlers = ast.parse('try:\n    pass\nexcept BaseException as error:\n    primary = error\n    raise').body[0].handlers
        block.finalbody = ast.parse(LAUNCH_FINALLY).body
    edit_method(e, 'OuterDriver', '_outer_owned_launch', edit_owned)
    e.one("self.state['parent_error_diagnostic'] = _startup_error_record(error)",
          "self.state['parent_error_diagnostic'] = _startup_error_record(error)\n                observation = self.driver.shutdown_observation\n                if observation is not None and len(boot.encoded(observation)) <= 16*1024:\n                    self.state['owned_shutdown'] = observation")
    e.append(DIAGNOSTIC_APPEND)
    sources[name] = e.raw
    changes[name] = e.edits

    descriptors = {name: {'bytes': len(raw), 'sha256': digest(raw)} for name, raw in sources.items()}
    catalogue['assets'] = {name: value for name, value in catalogue['assets'].items() if not name.startswith('supervisor/')}
    catalogue['assets'].update({'supervisor/' + name: value for name, value in descriptors.items()})
    policy['scratch_contract_text'] = encoded(catalogue).decode()
    policy['scope'] = static['scope']
    control = encoded({'kind': old_control['kind'], 'policy_text': encoded(policy).decode(),
                       'policy_sha256': digest(encoded(policy)), 'supervisor_sources': descriptors})

    e = Editor(files[pins['outer']])
    e.assign('POLICY_SHA', digest(encoded(policy)))
    e.assign('BOOT_SHA', descriptors['bootstrap.py']['sha256'])
    e.assign('COLLECTOR_SHA', descriptors['predictor_pgd_split_collection_v1.py']['sha256'])
    e.one('''except BaseException as error:
            owner.driver.assert_idle()
            work_error = error''', '''except BaseException as error:
            try:
                owner.driver.assert_idle()
            except BaseException as idle:
                error.add_note('Collection refused: '+type(idle).__name__+': '+str(idle)[:256])
                raise error
            work_error = error''')
    e.one('''if owner is not None:
            owner.driver.assert_idle()
            owner.driver.outer_journal.close()''', '''if owner is not None:
            try:
                owner.driver.assert_idle()
            except BaseException as idle:
                error.add_note('Outer refusal retains primary error; idle unconfirmed: '+type(idle).__name__)
                raise error
            owner.driver.outer_journal.close()''')
    outer = e.raw
    changes['outer'] = e.edits
    uploads = {'overlay/' + name: raw for name, raw in workers.items()}
    uploads.update({'supervisor/' + name: raw for name, raw in sources.items()})
    require(len(uploads) == 55 and sum(map(len, uploads.values())) <= 2*2**20, 'Original source envelope required.')
    for name, edits in changes.items():
        raw = outer if name == 'outer' else sources[name]
        original = files[pins['outer']] if name == 'outer' else bases[name]
        require(inverse(raw, edits) == original, 'Whole-byte inverse failed: ' + name)
    for name, raw in sources.items():
        compile(raw, '<UNBOUND-SHUTDOWN-' + name + '>', 'exec')
    compile(outer, '<UNBOUND-SHUTDOWN-OUTER>', 'exec')
    facts = {'kind': 'whole-pgd-shutdown-cold-derivation-v1', 'execution_binding': None,
        'whole_byte_inverses': changes, 'uploaded_sources': len(uploads),
        'uploaded_bytes': sum(map(len, uploads.values())), 'scientific_runtime_unchanged': 46,
        'supervisor_slots': 9, 'cache_modules': 46, 'result_catalogue_unchanged': True,
        'cleanup_total_seconds': 2., 'TERM_grace_seconds': .5,
        'compute_resource_caps_and_scientific_methods_unchanged': True,
        'new_client_binding_scope_package_or_approval': False, 'research_admitted': False, 'training_admitted': False}
    return {'sources': sources, 'workers': workers, 'uploads': uploads, 'control': control,
        'policy': policy, 'catalogue': catalogue, 'static': static,
        'manifest': manifest_raw, 'generated': {OUTER: outer}, 'derivation': facts}


def write_local(repository, destination):
    built = assemble(repository)
    dest = Path(destination)
    dest.mkdir(mode=0o700)
    files = {'CONTROL-UNBOUND.json': built['control'], 'SOURCE-MANIFEST-v1.json': built['manifest'],
             'DERIVATION.json': encoded(built['derivation']), 'CATALOGUE.json': encoded(built['catalogue']),
             OUTER: built['generated'][OUTER],
             'SOURCE-PAYLOAD.json': encoded({name: base64.b64encode(raw).decode() for name, raw in built['uploads'].items()})}
    files.update({'sources/' + name: raw for name, raw in built['sources'].items()})
    for name, raw in files.items():
        path = dest / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(raw)
    return built
