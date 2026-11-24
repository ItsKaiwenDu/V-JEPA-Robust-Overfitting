"""New delayed-exit controls: explicit fake process/clock/compute ports only."""
import ast
import base64
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess

import pytest
from models import predictor_pgd_workload_shutdown_logic_v1 as logic
from models import predictor_pgd_workload_shutdown_v1 as adapter

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'runs/pretrained/diagnostic/pgd-workload-shutdown-v1-20261123'
EVIDENCE = Path(os.environ['VJEPA_SHUTDOWN_EVIDENCE'])
spec = importlib.util.spec_from_file_location('_new_shutdown_fake_ports', STAGE / 'fake-shutdown-support-v1.py')
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)
ORIGIN = support.ORIGIN
put = support.put


@pytest.fixture(autouse=True)
def no_actual_work(monkeypatch):
    support.no_actual_work.__wrapped__(monkeypatch)


@pytest.fixture
def area(request):
    path = EVIDENCE / request.node.name
    path.mkdir(mode=0o700)
    put(path / 'FAKE-ORIGIN.json', {'origin': ORIGIN, 'actual_target_work': False, 'actual_user_grant': False})
    return path


class Ports:
    def __init__(self, fault=None):
        self.now = 10.
        self.events = []
        self.fault = fault
        self.killed = None
        owner = self

        class Lease:
            pid = 970001
            birth = 42
            closed = False

            def dead(self):
                assert not self.closed, 'No closed handle observation'
                if owner.fault == 'dead':
                    raise OSError('FAKE held death observation failure')
                return owner.killed is not None and owner.now >= owner.killed + .15

            def send(self, number):
                assert not self.closed, 'No closed/reused handle signal'
                owner.events.append(['held-signal', int(number), owner.now])
                if owner.fault == ('term' if number == signal.SIGTERM else 'kill'):
                    raise OSError('FAKE owned signal failure')
                if number == signal.SIGKILL and owner.fault != 'never':
                    owner.killed = owner.now

            def close(self):
                assert self.dead() and owner.child.poll() is not None
                if owner.fault == 'close':
                    raise OSError('FAKE handle close failure')
                self.closed = True
                owner.events.append(['held-close', owner.now])

        class Child:
            pid = Lease.pid

            def poll(self):
                if owner.fault == 'poll':
                    raise OSError('FAKE child poll failure')
                if owner.killed is not None and owner.now >= owner.killed + .4:
                    return -9
                return None

            def wait(self, timeout):
                owner.events.append(['bounded-wait', timeout, owner.now])
                assert 0 < timeout <= .1
                if owner.fault == 'wait':
                    raise InterruptedError('FAKE wait interruption')
                owner.now += timeout
                if self.poll() is None:
                    raise subprocess.TimeoutExpired('EXPLICIT FAKE', timeout)
                return -9

        self.lease = Lease()
        self.child = Child()

    def clock(self):
        if self.fault == 'backward':
            return 9.
        if self.fault == 'nan':
            return float('nan')
        return self.now

    def pause(self, seconds):
        assert 0 <= seconds <= .1
        self.now += seconds

    def run(self, primary, work=100.):
        return logic.cleanup_owned(self.child, {(self.lease.pid, self.lease.birth): self.lease}, self.lease, primary,
            work_deadline=work, phase_deadline=10., clock=self.clock, pause=self.pause,
            last=10., samples=31, peak=12345)

    def save(self, area, record):
        put(area / 'RESULT.json', {'origin': ORIGIN, 'observation': record, 'events': self.events})


def test_delayed_KILL_exit_and_reap_preserve_primary_and_held_identity(area):
    ports = Ports()
    primary = TimeoutError('FAKE original phase deadline')
    record = ports.run(primary)
    assert record['confirmed_idle'] and record['exit_status'] == -9
    assert primary.owned_shutdown is record and record['primary']['type'] == 'TimeoutError'
    assert record['cleanup_deadline'] == 12. and ports.now <= 12.
    assert record['RSS_samples_before_failure'] == 31 and record['peak_observed_RSS_bytes'] == 12345
    assert [row[1] for row in ports.events if row[0] == 'held-signal'] == [signal.SIGTERM, signal.SIGKILL]
    assert ports.events[-1][0] == 'held-close'
    ports.save(area, record)


@pytest.mark.parametrize('fault', ['dead', 'term', 'kill', 'wait', 'poll', 'close'])
def test_cleanup_faults_never_replace_original_error_or_reset_clock(area, fault):
    ports = Ports(fault)
    primary = MemoryError('FAKE original RSS refusal')
    record = ports.run(primary)
    assert record['primary']['type'] == 'MemoryError' and primary.owned_shutdown is record
    assert record['error_count'] > 0 and len(record['errors']) <= 8
    assert ports.now <= 12. and record['wait_iterations'] <= 32
    if not record['confirmed_idle']:
        assert not ports.lease.closed
    ports.save(area, record)


@pytest.mark.parametrize('fault,work', [(None, 10.2), ('backward', 100.), ('nan', 100.)])
def test_cleanup_is_clamped_to_original_work_and_rejects_bad_clock(area, fault, work):
    ports = Ports(fault)
    primary = TimeoutError('FAKE original phase failure')
    record = ports.run(primary, work)
    assert not record['confirmed_idle'] and not ports.lease.closed
    assert ports.now <= min(12., work)
    if fault is None:
        assert record['cleanup_deadline'] == work
    ports.save(area, record)


def test_unconfirmed_exit_without_primary_refuses_and_retains_handles(area):
    ports = Ports('never')
    with pytest.raises(RuntimeError, match='confirmed idle=False') as caught:
        ports.run(None)
    record = caught.value.owned_shutdown
    assert not record['confirmed_idle'] and record['exit_status'] is None
    assert not ports.lease.closed and ports.now <= 12.
    ports.save(area, record)


def test_bounded_explicit_cause_context_cycle_and_notes(area):
    first = TimeoutError('FAKE first deadline')
    second = RuntimeError('FAKE idle refusal')
    second.__cause__ = first
    first.__context__ = second
    second.add_note('x' * 10000)
    record = logic.error_chain(second)
    assert [row['type'] for row in record['exceptions']] == ['RuntimeError', 'TimeoutError']
    assert record['truncated_or_cycle'] and len(record['exceptions'][0]['notes'][0]) == 256
    put(area / 'RESULT.json', {'origin': ORIGIN, 'chain': record})


def test_cold_inverses_fixed_science_resource_guards_and_monitor_body(area):
    built = adapter.assemble(ROOT)
    cfg, files = adapter.load(ROOT)
    original = {name: base64.b64decode(value) for name, value in json.loads(files[cfg['pins']['payload']]).items()}
    for name, edits in built['derivation']['whole_byte_inverses'].items():
        raw = built['generated'][adapter.OUTER] if name == 'outer' else built['sources'][name]
        expected = files[cfg['pins']['outer']] if name == 'outer' else original['supervisor/' + name]
        assert adapter.inverse(raw, edits) == expected
    for name, raw in built['workers'].items():
        assert raw == original['overlay/' + name]
    for name in ['predictor_pgd_split_watch_v1.py', 'scratch_allocation_v2.py', 'workload_os_v1.py', 'workload_plan_v1.py', 'predictor_pgd_split_collection_v1.py']:
        assert built['sources'][name] == original['supervisor/' + name]
    nodes = []
    for raw in [original['supervisor/workload_watch_v1.py'], built['sources']['workload_watch_v1.py']]:
        node = next(n for n in ast.parse(raw).body if getattr(n, 'name', None) == 'watch')
        node.body[-1].finalbody = []
        nodes.append(ast.dump(node, include_attributes=False))
    assert nodes[0] == nodes[1]
    old_policy = json.loads(json.loads(files[cfg['pins']['control']])['policy_text'])
    old_cat = json.loads(old_policy['scratch_contract_text'])
    for key in ('caps', 'result', 'root_metadata', 'packet', 'runtime_files'):
        assert built['catalogue'][key] == old_cat[key]
    assert built['static']['scope']['research_admitted'] is False
    put(area / 'RESULT.json', {'origin': ORIGIN, 'runtime_science_exact': 46,
        'source_bytes': built['derivation']['uploaded_bytes'], 'all_inverses': True})


class DelayedChild(support.Child):
    def poll(self):
        if getattr(self, 'kill_at', None) is not None and self.e.clock.now >= self.kill_at + .4:
            self.exited = True
            self.status = -9
        return super().poll()

    def wait(self, timeout=None):
        assert timeout is not None and timeout >= 0
        self.e.events.append(['FAKE-BOUNDED-WAIT', self.pid, timeout, self.e.clock.now])
        if self.poll() is None:
            self.e.clock.now += timeout
        if self.poll() is None:
            raise subprocess.TimeoutExpired('EXPLICIT FAKE', timeout)
        return self.status


class Engine(support.Engine):
    def popen(self, command, **kw):
        child = super().popen(command, **kw)
        child.__class__ = DelayedChild
        return child

    def identity(self, pid):
        row = super().identity(pid)
        if self.fault == 'identity' and self.children[pid].plan['role'] == 'original-unit':
            row['parent_pid'] = 1
        return row

    def close(self, fd):
        if (fd in self.pipes and self.current is not None and not self.current.released
                and fd == self.current.fd + 1):
            # Explicit fake release-pipe EOF, not an actual pipe/OS exit.
            self.current.exited = True
            self.current.status = 124
        return super().close(fd)

    def write(self, fd, data):
        if self.fault == 'release' and self.current.plan['role'] == 'original-unit' and fd in self.pipes:
            raise BrokenPipeError('FAKE release failure')
        return super().write(fd, data)

    def sample(self, pid, birth):
        child = self.children[pid]
        if child.plan['role'] == 'original-unit' and self.fault in ('deadline', 'never', 'rss', 'interrupt'):
            if self.fault == 'interrupt':
                raise KeyboardInterrupt('FAKE primary interruption')
            rss = 4*2**30 + 1 if self.fault == 'rss' else 1048576
            if self.fault in ('deadline', 'never'):
                self.clock.now = child.plan['deadline']
            return {'processes': {pid: dict(self.identity(pid), rss_bytes=rss)}, 'rss_bytes': rss}
        return super().sample(pid, birth)

    def handle_send(self, fd, sig):
        child = self.children[fd - 1000000]
        self.events.append(['FAKE-HELD-SIGNAL', child.pid, int(sig), self.clock.now])
        if sig == signal.SIGKILL and self.fault != 'never':
            child.kill_at = self.clock.now

    def handle_dead(self, fd):
        child = self.children[fd - 1000000]
        return child.exited or (getattr(child, 'kill_at', None) is not None and self.clock.now >= child.kill_at + .15)


def collect(engine, area):
    engine.owner.driver.assert_idle()
    engine.owner.driver.outer_journal.close()
    codec = engine.v.load('models.predictor_pgd_split_collection_v1')
    collector = codec.Collector(engine.guard, engine.built['control'], engine.auth,
                                connection_started=100., clock=engine.clock)
    collector.prepare()
    stream = io.BytesIO()
    collector.stream(stream)
    data = stream.getvalue()
    put(area / 'WIRE.bin', data)
    received = codec.receive(io.BytesIO(data), area / 'received', engine.built['control'], engine.auth,
        window=codec.Window(100., collector.window.started, clock=engine.clock))
    assert received['transport_verified'] and not received['scientific_qualified']
    put(area / 'TRANSPORT.json', received)
    return received


def test_complete_new_bootstrap_cache_timer_parent_and_transport_fake_lifecycle(area):
    engine = Engine(area)
    engine.owner.run_work()
    assert engine.owner.state['completed'] == ['proof', 'original-unit', 'candidate-unit', 'candidate-replay']
    assert engine.owner.state['complete'] is False
    assert len(engine.finder.files) == 55
    received = collect(engine, area)
    with pytest.raises(ValueError):
        engine.v.original_proof(engine.guard, adapter.digest((engine.root / 'allocation-result-v2.json').read_bytes()))
    put(area / 'RESULT.json', {'origin': ORIGIN, 'complete_fake_roles': engine.owner.state['completed'],
        'transport_files': len(received['files']), 'actual_proof_bytes': 0, 'real_proof_reader_refused': True})
    put(area / 'EVENTS.json', engine.events)


@pytest.mark.parametrize('fault,kind', [('deadline', TimeoutError), ('rss', MemoryError), ('interrupt', KeyboardInterrupt)])
def test_delayed_exit_preserves_primary_session_and_collects_failed_work(area, fault, kind):
    engine = Engine(area, fault=fault)
    with pytest.raises(kind):
        engine.owner.run_work()
    state = engine.owner.state
    assert state['phase'] == 'failed' and state['completed'] == ['proof'] and state['complete'] is False
    assert state['error_type'] == kind.__name__
    assert state['parent_error_diagnostic']['error_type'] == kind.__name__
    assert state['owned_shutdown']['confirmed_idle'] and state['owned_shutdown']['exit_status'] == -9
    assert state['owned_shutdown']['all_handles_closed']
    assert len(engine.children) == 2
    collect(engine, area)
    put(area / 'RESULT.json', {'origin': ORIGIN, 'primary_type': kind.__name__,
        'late_exit_confirmed': True, 'later_roles_started': False, 'failed_status_unchanged': True})
    put(area / 'EVENTS.json', engine.events)


def test_unconfirmed_child_keeps_handles_and_refuses_collection_without_masking(area):
    engine = Engine(area, fault='never')
    with pytest.raises(TimeoutError):
        engine.owner.run_work()
    state = engine.owner.state
    assert state['error_type'] == 'TimeoutError' and not state['owned_shutdown']['confirmed_idle']
    assert not engine.owner.driver.outer_lease.closed
    with pytest.raises(RuntimeError, match='unconfirmed'):
        engine.owner.driver.assert_idle()
    assert not (engine.root / 'capture/results/final-receipt.json').exists()
    record = state['owned_shutdown']
    assert record['cleanup_deadline'] == min(record['work_deadline'], record['cleanup_started'] + 2.)
    assert record['wait_iterations'] <= 32
    put(area / 'RESULT.json', {'origin': ORIGIN, 'primary_preserved': True,
        'unconfirmed_owned_handles_retained': True, 'collection_refused': True})
    put(area / 'EVENTS.json', engine.events)


def test_descendant_must_be_dead_before_any_held_handle_is_closed(area):
    ports = Ports()

    class Descendant(type(ports.lease)):
        pid = 970002
        birth = 43

        def dead(self):
            assert not self.closed
            return ports.killed is not None and ports.now >= ports.killed + 1.

    descendant = Descendant()
    leases = {(value.pid, value.birth): value for value in (ports.lease, descendant)}
    primary = TimeoutError('FAKE full owned tree deadline')
    record = logic.cleanup_owned(ports.child, leases, ports.lease, primary,
        work_deadline=100., phase_deadline=10., clock=ports.clock, pause=ports.pause,
        last=10., samples=3, peak=12345)
    assert record['confirmed_idle'] and len(record['owned_identities']) == 2
    assert all(row['dead'] and row['handle_closed'] for row in record['owned_identities'])
    assert min(row[-1] for row in ports.events if row[0] == 'held-close') >= ports.killed + 1.
    ports.save(area, record)


@pytest.mark.parametrize('fault,kind', [('identity', ValueError), ('release', BrokenPipeError)])
def test_pre_monitor_failure_never_releases_unowned_or_masks_primary(area, fault, kind):
    engine = Engine(area, fault=fault)
    with pytest.raises(kind):
        engine.owner.run_work()
    assert engine.owner.state['error_type'] == kind.__name__ and len(engine.children) == 2
    assert not engine.current.released
    engine.owner.driver.assert_idle()
    put(area / 'RESULT.json', {'origin': ORIGIN, 'primary_type': kind.__name__, 'unowned_release': False})
    put(area / 'EVENTS.json', engine.events)
