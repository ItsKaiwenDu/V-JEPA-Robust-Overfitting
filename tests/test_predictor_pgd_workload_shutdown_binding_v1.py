"""New v4 client/shutdown integration with explicit fake budget and OS ports."""
import ast
import base64
import builtins
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from types import SimpleNamespace as NS

import pytest
from models import predictor_pgd_workload_shutdown_binding_v1 as h
from models import predictor_pgd_workload_handoff_outer_v4 as outer
from models import predictor_pgd_workload_handoff_client_v4 as client
from models import predictor_pgd_workload_budget_v1 as budget
from models import predictor_pgd_workload_runtime_records_v1  # Metadata definitions only.

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'runs/pretrained/diagnostic/pgd-workload-shutdown-binding-v1-20261123'
EVIDENCE = Path(os.environ['VJEPA_SHUTDOWN_BINDING_EVIDENCE'])


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


support = module('_new_shutdown_client_ports', STAGE/'fake-client-support-v1.py')
worker = module('_new_shutdown_delay_ports', STAGE/'fake-delayed-worker-v1.py')
preparation = module('_new_shutdown_preparation_ports', STAGE/'fake-preparation-support-v1.py')
ORIGIN = support.ORIGIN
put = support.put


@pytest.fixture(scope='module', autouse=True)
def forbid_actual_work():
    yield from support.forbid_work.__wrapped__()


@pytest.fixture
def area(request):
    path = EVIDENCE/request.node.name
    path.mkdir(mode=0o700)
    put(path/'FAKE-ORIGIN.json', {'origin': ORIGIN, 'actual_user_grant': False,
        'actual_target_work': False, 'actual_budget_qualification': False})
    return path


class AttachedEngine(worker.Engine):
    """Exercise installed source bytes in a private fake process/cache namespace."""
    def __init__(self, server, actual, sources, control, policy, authorization, connection_started):
        self.folder = server.area
        self.method = 'primary'
        self.fault = {'integrated-deadline': 'deadline', 'integrated-rss': 'rss',
            'integrated-interrupt': 'interrupt', 'integrated-unconfirmed': 'never'}.get(server.area.name)
        self.clock = worker.support.Clock()
        self.clock.now = server.clock.now
        self.events = []
        self.parent = server.pid
        self.pid = self.parent
        self.children = {}
        self.current = None
        self.pipes = {}
        self.nextfd = 900000
        self.root = actual
        self.built = {'sources': {n.removeprefix('supervisor/'): v for n, v in sources.items() if n.startswith('supervisor/')},
            'workers': {n.removeprefix('overlay/'): v for n, v in sources.items() if n.startswith('overlay/')},
            'uploads': sources, 'control': control, 'policy': policy}
        self.os = NS(**vars(os))
        self.time = NS(monotonic=self.clock, sleep=self.pause)
        self.os.getpid = lambda: self.pid
        self.os.getppid = lambda: self.parent if self.pid != self.parent else os.getppid()
        self.os.getsid = lambda pid: self.pid
        self.os.getpgrp = lambda: self.pid
        self.os.pipe2 = self.pipe
        self.os.close = self.close
        self.os.write = self.write
        self.os.read = self.read
        self.os.fstat = lambda fd: NS(st_mode=stat.S_IFIFO) if fd in self.pipes else os.fstat(fd)
        self.os.fstatvfs = lambda fd: NS(f_bavail=20*2**30, f_frsize=1, f_favail=100000)
        self.os.urandom = lambda n: bytes([len(self.children)+1])*n
        self.os._exit = lambda code: (_ for _ in ()).throw(SystemExit(code))
        self.subprocess = NS(PIPE=-1, DEVNULL=-3, TimeoutExpired=worker.support.subprocess.TimeoutExpired,
            check_output=self.query, Popen=self.popen)
        self.v = worker.support.Virtual(self)
        self.bootstrap = self.v.module(self.built['sources']['bootstrap.py'], '_new_bound_bootstrap', actual/'supervisor/bootstrap.py')
        self.csha = h.digest(control)
        c = json.loads(control)
        self.auth = authorization
        self.binding = {'root': str(actual), 'root_identity': list(self.bootstrap.dir_id(actual.lstat())),
            'parent': str(actual.parent), 'parent_identity': list(self.bootstrap.dir_id(actual.parent.lstat()))}
        self.finder = self.bootstrap.PinnedImporter(actual, self.binding, c, policy).install()
        self.v.finder = self.finder
        self.scratch = self.v.load('models.predictor_pgd_split_scratch_v1')
        self.coordinator = self.v.load('models.predictor_pgd_split_coordinator_v1')
        ib, isha = self.bootstrap.installation(c, self.csha)
        layout = self.scratch.Layout(policy['scratch_contract_text'].encode(), policy['base_text'].encode(),
            policy['design_text'].encode(), ib, isha)
        self.guard = self.scratch.Guard(actual, self.binding, layout)
        definitions = self.v.module((ROOT/h.OUTER).read_bytes(), '_new_fake_outer', actual/'_unemitted-outer')
        self.alarm = definitions.Alarms(clock=self.clock)
        self.owner = self.coordinator.OuterCoordinator(actual, self.binding, control, authorization,
            connection_started=connection_started, clock=self.clock, outer_guard=self.guard, outer_alarm=self.alarm)

    def fake_compute(self, root, plan, scope, timers):
        return preparation.Engine.fake_compute(self, root, plan, scope, timers)


BaseServer = support.ServerFixture


class IntegratedServer(BaseServer):
    def setup(self, alias, sources, c, b, p, auth, raw, araw, bootstrap, incoming, check):
        _, scratch, guard, binding, derived = super().setup(alias, sources, c, b, p, auth, raw, araw, bootstrap, incoming, check)
        parent = self

        class Owner:
            def __init__(self, root, binding, control, authorization, *, connection_started, **kw):
                engine = parent.engine = AttachedEngine(parent, alias.physical, sources, control, p, authorization, connection_started)
                self.inner = engine.owner
                self.claimed = self.inner.claimed
                self.state = self.inner.state
                self.driver = NS(**vars(self.inner.driver))
                self.driver.guard = support.GuardView(engine.guard, alias.virtual)
                self.driver.assert_idle = self.inner.driver.assert_idle
                self.driver.save = self.inner.driver.save

            def run_work(self):
                try:
                    return self.inner.run_work()
                finally:
                    self.claimed = self.inner.claimed
                    self.state = self.inner.state
                    parent.clock.now = parent.engine.clock.now
                    put(parent.area/'CORE-EVENTS.json', {'origin': ORIGIN, 'events': parent.engine.events,
                        'all_OS_compute_proof_timer_GPU_and_budget_ports_fake': True})

        return NS(OuterCoordinator=Owner), scratch, guard, binding, derived


support.ServerFixture = IntegratedServer


def undo(raw, edits):
    for edit in reversed(edits):
        a = edit['start']
        b = a+len(edit['after'].encode())
        assert raw[a:b] == edit['after'].encode()
        raw = raw[:a]+edit['before'].encode()+raw[b:]
    return raw


def test_new_binding_inverse_ast_shutdown_cache_and_source_identity(area):
    generated, derivation, cfg, files = h.assemble(ROOT)
    sources = {n: base64.b64decode(v) for n, v in json.loads(files[cfg['pins']['payload']]).items()}
    bases = {h.OUTER: files[cfg['pins']['outer']], h.CLIENT: files[cfg['pins']['client']], h.CLI: files[cfg['pins']['cli']]}
    bases.update({path: sources['supervisor/'+name] for name, path in h.READERS.items()})
    for name, raw in generated.items():
        assert raw == (ROOT/name).read_bytes() and undo(raw, derivation['edits'][name]) == bases[name]
    before = ast.parse(bases[h.OUTER])
    after = ast.parse(generated[h.OUTER])
    definitions = lambda tree: [ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert definitions(before) == definitions(after)
    assert len(outer.LOCAL_SOURCES) == 18 and len(cfg['files']) == 17
    assert len(sources) == 55 and sum(map(len, sources.values())) == 905563
    assert h.digest(files[cfg['pins']['control']]) == 'bbb6e9789e34998d2f07b5b2ece487baa32b22b7638086dee7f1358a2a7b194c'
    assert not any(hasattr(h, name) for name in ('prepare_unarmed', 'material', 'authorization'))
    put(area/'RESULT.json', {'origin': ORIGIN, 'six_inverses': True, 'all_outer_function_class_ASTs_exact': True,
        'local_pins': 18, 'input_pins': 17, 'new_real_package_API': False})


def test_real_new_local_preflight_checks_fake_grant_and_exact_shutdown_bytes(area):
    folder = support.package(area)
    checked = client.check_local(folder)
    assert len(checked[0]['local_source_sha256']) == 18 and len(checked[7]) == 55
    assert checked[6]['user_basis'] == ORIGIN and checked[6]['full_model_gradients'] == 120
    assert checked[-1] == (ROOT/outer.PACKET_PATH,)
    assert not (folder/'ATTEMPT.json').exists()
    assert 'cleanup_owned' in base64.b64decode(checked[7]['supervisor/workload_watch_v1.py']).decode()
    put(area/'RESULT.json', {'origin': ORIGIN, 'preflight': 'PASS', 'execution_or_budget_qualified': False})


@pytest.mark.parametrize('armed', [False, True])
def test_real_budget_gate_refuses_before_clock_attempt_or_network(area, monkeypatch, armed):
    folder = support.package(area) if armed else area/'missing'
    def forbidden(*args, **kw):
        pytest.fail('No time, timer, protocol or connection before budget qualification')
    monkeypatch.setattr(client, 'time', NS(monotonic=forbidden))
    monkeypatch.setattr(outer, 'Alarms', forbidden)
    monkeypatch.setattr(client, '_budget_unqualified_protocol', forbidden)
    with pytest.raises(ValueError, match='budget is unqualified'):
        client.run(folder)
    assert not (folder/'ATTEMPT.json').exists() and not (folder/'attended-session').exists()
    put(area/'RESULT.json', {'origin': ORIGIN, 'budget_gate_refused': True,
        'fake_grant_present': armed, 'clock_attempt_timer_or_network_created': False})


def test_metadata_complete_flags_cannot_open_budget_gate(area, monkeypatch):
    monkeypatch.setattr(budget, 'REVIEWED_COMPLETE_UNIT_BUDGET',
        {'complete': True, 'approved': True, 'unit_seconds': 1, 'source_authenticated': True})
    with pytest.raises(ValueError, match='unqualified'):
        budget.require_execution()
    decision = json.loads((ROOT/'configs/predictor_pgd_workload_budget_v1.json').read_bytes())
    assert decision['decision'] == 'REJECT_CURRENT_ROUTE_FOR_NEW_SCOPE'
    assert not decision['scope_preparation_allowed'] and decision['maximum_new_attempts'] == 0
    put(area/'RESULT.json', {'origin': ORIGIN, 'untrusted_metadata_is_not_budget_or_execution_qualification': True})


@pytest.mark.parametrize('armed', [False, True])
def test_isolated_new_CLI_refuses_or_dispatches_closed_new_client(area, armed):
    folder = support.package(area) if armed else area/'missing'
    events = []
    fake_sys = NS(path=[])
    original = builtins.__import__
    def importing(name, *args, **kw):
        if name == 'sys':
            return fake_sys
        if name == 'models.predictor_pgd_workload_handoff_client_v4':
            assert fake_sys.path == [str(ROOT)]
            events.append('v4')
            return NS(run=client.run)
        assert not name.startswith('models.')
        return original(name, *args, **kw)
    namespace = {'__name__': '_FAKE_ISOLATED_V4_CLI', '__file__': str(ROOT/h.CLI),
        '__builtins__': {**vars(builtins), '__import__': importing}}
    exec(compile((ROOT/h.CLI).read_bytes(), str(ROOT/h.CLI), 'exec'), namespace)
    if armed:
        with pytest.raises(ValueError, match='budget is unqualified'):
            namespace['main']([str(folder)])
        assert events == ['v4']
    else:
        with pytest.raises(SystemExit, match='Unarmed'):
            namespace['main']([str(folder)])
        assert not events and not fake_sys.path
    assert not (folder/'ATTEMPT.json').exists()


@pytest.mark.parametrize('version', ['v1', 'v2', 'v3'])
def test_all_consumed_real_packages_refuse_new_preflight_without_mutation(area, version):
    tail = 'linux10614-primary-final-unarmed' if version == 'v1' else 'linux10614-primary-'+version+'-final-unarmed'
    folder = ROOT/f'runs/pretrained/diagnostic/pgd-workload-handoff-{version}-20261123/handoff-parent'/tail
    before = (folder/'ATTEMPT.json').read_bytes()
    with pytest.raises(ValueError):
        client.check_local(folder)
    assert (folder/'ATTEMPT.json').read_bytes() == before and json.loads(before)['consumed'] is True
    put(area/'RESULT.json', {'origin': ORIGIN, 'old_consumed_package_preserved': version})


@pytest.fixture(scope='module')
def pin_baseline():
    parent = EVIDENCE/'preflight-baseline'
    parent.mkdir(mode=0o700)
    support.package(parent)
    project = parent/'project'
    project.mkdir(mode=0o700)
    for name in [*outer.LOCAL_SOURCES, outer.PACKET_PATH]:
        put(project/name, (ROOT/name).read_bytes())
    return parent


@pytest.mark.parametrize('fault', ['client', 'bootstrap', 'budget-module', 'budget-design', 'payload', 'runtime-source', 'archive', 'grant-kind'])
def test_new_pins_source_budget_and_grant_drift_refuse_without_attempt(area, pin_baseline, fault):
    copied = area/'copied'
    shutil.copytree(pin_baseline, copied)
    folder = copied/'fake-package'
    project = copied/'project'
    control = json.loads((folder/'CONTROL-WORKLOAD.json').read_bytes())
    control.update(repository=str(project), local_handoff=str(folder))
    raw = h.enc(control)
    (folder/'CONTROL-WORKLOAD.json').write_bytes(raw)
    grant = outer.authorization(control, h.digest(raw), ORIGIN)
    if fault in ('client', 'bootstrap', 'budget-module', 'budget-design'):
        name = {'client': h.CLIENT, 'bootstrap': h.READERS['bootstrap.py'],
            'budget-module': h.BUDGET, 'budget-design': 'configs/predictor_pgd_workload_budget_v1.json'}[fault]
        (project/name).write_bytes(b'EXPLICIT FAKE PIN DRIFT\n')
    elif fault == 'payload':
        (folder/'SOURCE-PAYLOAD.json').write_bytes(b'{}\n')
    elif fault == 'runtime-source':
        payload = json.loads((folder/'SOURCE-PAYLOAD.json').read_bytes())
        payload['supervisor/workload_watch_v1.py'] = base64.b64encode(b'FAKE SHUTDOWN DRIFT').decode()
        (folder/'SOURCE-PAYLOAD.json').write_bytes(h.enc(payload))
    elif fault == 'archive':
        (project/outer.PACKET_PATH).write_bytes(b'FAKE ARCHIVE DRIFT')
    else:
        grant['kind'] = 'pgd-candidate-outer-university-exception-v1'
    (folder/'AUTHORIZATION.json').write_bytes(h.enc(grant))
    with pytest.raises(ValueError):
        client.check_local(folder)
    assert not (folder/'ATTEMPT.json').exists() and not (folder/'attended-session').exists()


@pytest.fixture(scope='module')
def transcript():
    area = EVIDENCE/'integrated-primary'
    area.mkdir(mode=0o700)
    folder = support.package(area)
    fixture = support.ClientFixture(area, folder)
    before = ({k: v for k, v in sys.modules.items() if k.split('.')[0] in ('models', 'attacks', 'datasets')}, list(sys.meta_path))
    fixture.result = fixture.run()
    after = ({k: v for k, v in sys.modules.items() if k.split('.')[0] in ('models', 'attacks', 'datasets')}, list(sys.meta_path))
    assert before == after
    return fixture


def test_new_client_server_worker_parent_and_transport_complete_fake_flow(transcript):
    fixture = transcript
    assert fixture.result == fixture.server.status == 0 and not fixture.proc.terminated
    assert bytes(fixture.proc.stdin.data) == fixture.checked[8]+fixture.server.input
    session = fixture.folder/'attended-session'
    receipt = json.loads((session/'COLLECTION-RECEIPT.json').read_bytes())
    state = json.loads((session/'saved/session.json').read_bytes())
    assert receipt['transport_verified'] and receipt['SSH_exit_status'] == 0
    assert state['completed'] == ['proof', 'original-unit', 'candidate-unit', 'candidate-replay'] and state['complete'] is False
    assert not json.loads((session/'saved-transport/PARTIAL-RECEIPT.json').read_bytes())['transport_verified']
    for role in state['completed'][1:]:
        preparation_row = json.loads((fixture.area/('preparation-'+role+'.json')).read_bytes())
        assert preparation_row['events'].index('transfer') < preparation_row['events'].index('checkpoint') < preparation_row['events'].index('unit')
    assert ['FAKE-BUDGET-PORT-NOT-ACTUAL-QUALIFICATION'] in fixture.events
    put(fixture.area/'TRANSCRIPT-RESULT.json', {'origin': ORIGIN, 'result': 'PASS', 'files': len(receipt['files']),
        'wire_bytes': receipt['wire_bytes'], 'wire_sha256': receipt['wire_sha256'],
        'exact_client_outbound_matches_server_input': True, 'private_project_namespaces_unchanged': True,
        'actual_target_budget_or_scientific_execution_qualified': False})


@pytest.mark.parametrize('fault,kind', [('deadline', 'TimeoutError'), ('rss', 'MemoryError'), ('interrupt', 'KeyboardInterrupt')])
def test_client_collects_delayed_exit_failure_with_original_cause(area, fault, kind):
    work = area/('integrated-'+fault)
    work.mkdir(mode=0o700)
    fixture = support.ClientFixture(work, support.package(work))
    assert fixture.run() == fixture.server.status == 1
    assert bytes(fixture.proc.stdin.data) == fixture.checked[8]+fixture.server.input
    session = fixture.folder/'attended-session'
    state = json.loads((session/'saved/session.json').read_bytes())
    assert state['phase'] == 'failed' and state['completed'] == ['proof'] and not state['complete']
    assert state['error_type'] == state['parent_error_diagnostic']['error_type'] == kind
    assert state['owned_shutdown']['primary']['type'] == kind
    assert state['owned_shutdown']['confirmed_idle'] and state['owned_shutdown']['exit_status'] == -9
    receipt = json.loads((session/'COLLECTION-RECEIPT.json').read_bytes())
    assert receipt['transport_verified'] and receipt['SSH_exit_status'] == 1
    assert [row[2] for row in fixture.server.engine.events if row[0] == 'FAKE-Popen'] == ['proof', 'original-unit']
    put(work/'TRANSCRIPT-RESULT.json', {'origin': ORIGIN, 'result': 'PASS', 'files': len(receipt['files']),
        'wire_bytes': receipt['wire_bytes'], 'wire_sha256': receipt['wire_sha256'], 'primary': kind,
        'exact_client_outbound_matches_server_input': True, 'actual_target_or_budget_qualified': False})


def test_unconfirmed_exit_refuses_ready_and_client_preserves_claim_and_failure(area):
    work = area/'integrated-unconfirmed'
    work.mkdir(mode=0o700)
    fixture = support.ClientFixture(work, support.package(work))
    with pytest.raises(ValueError, match='truncated outer transport'):
        fixture.run()
    session = fixture.folder/'attended-session'
    assert (session/'ROOT-RECEIPT.json').exists() and (session/'CLIENT-REFUSAL.json').exists()
    assert not (session/'COLLECTION-READY.json').exists() and not (session/'COLLECTION-RECEIPT.json').exists()
    server_error = json.loads((work/'SERVER-RAISED.json').read_bytes())
    state = json.loads((work/'server-files/session.json').read_bytes())
    assert server_error['type'] == state['error_type'] == 'TimeoutError'
    assert not state['owned_shutdown']['confirmed_idle'] and not fixture.server.engine.owner.driver.outer_lease.closed
    assert fixture.proc.terminated
    put(work/'TRANSCRIPT-RESULT.json', {'origin': ORIGIN, 'result': 'PASS', 'collection_refused': True,
        'original_primary_preserved': True, 'retained_claim_and_client_refusal': True,
        'actual_target_or_budget_qualified': False})


@pytest.mark.parametrize('fault', ['prefix', 'work-clock', 'header-root', 'header-origin', 'partial', 'trailing'])
def test_saved_response_mutations_refuse_without_new_worker_lifecycle(area, transcript, fault):
    source = io.BytesIO(transcript.server.output)
    frames = [outer.frame(source, lambda: None) for _ in range(3)]
    codec = transcript.server.codec
    _, header = codec.WireReader(source, NS(check=lambda: None)).frame(codec.MAGIC, codec.HEADER_CAP)
    tail = source.read()
    if fault == 'prefix':
        frames[1]['scratch'] = frames[1]['scratch'].replace('workload-v1-', 'candidate-v1-')
    elif fault == 'work-clock':
        frames[1]['work_deadline'] += 1
    elif fault == 'header-root':
        header['base']['root_binding']['root'] += '-other'
    elif fault == 'header-origin':
        header['base']['connection_started'] += 1
        header['base']['connection_deadline'] += 1
    elif fault == 'partial':
        tail = tail[:200]
    else:
        tail += b'EXPLICIT-FAKE-TRAILING'
    response = io.BytesIO()
    for item in frames:
        outer.send_frame(response, item, lambda: None)
    raw = h.enc(header)
    response.write(codec.MAGIC+len(raw).to_bytes(4, 'big')+raw+tail)
    raw = response.getvalue()
    put(area/'MUTATED-SERVER-OUTPUT.bin', raw)
    fixture = support.ClientFixture(area, golden=transcript, response=raw)
    with pytest.raises((ValueError, EOFError)):
        fixture.run()
    session = fixture.folder/'attended-session'
    assert (session/'CLAIM-OBSERVED.json').exists() and (session/'CLIENT-REFUSAL.json').exists()
    assert not (session/'COLLECTION-RECEIPT.json').exists() and fixture.proc.terminated
    if fault in ('prefix', 'work-clock'):
        assert not (session/'ROOT-RECEIPT.json').exists()
    else:
        assert (session/'ROOT-RECEIPT.json').exists()
    if fault in ('partial', 'trailing'):
        assert not json.loads((session/'saved-transport/PARTIAL-RECEIPT.json').read_bytes())['transport_verified']
    put(area/'RESULT.json', {'origin': ORIGIN, 'result': 'PASS', 'fault': fault, 'no_new_worker_lifecycle': True})
