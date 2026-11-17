"""Focused byte-only checks. No target, real child, GPU, model, or proof work."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import types
import pytest
from models import pgd_byte_close_probe_v1 as client
from models import pgd_byte_close_core_v1 as core

REPO = Path(__file__).resolve().parents[1]
RAW = REPO/'runs/pretrained/diagnostic/pgd-byte-close-probe-v1-20261116'
NONCE, CSHA = '1'*64, '2'*64


def sources():
    return {n: (REPO/n).read_bytes() for n in client.SOURCES}


@pytest.fixture
def ns():
    n = {'__name__': 'inert_local_byte_probe'}
    exec(compile(client.build(sources()), '<LOCAL-INERT-BYTE-DEFINITIONS>', 'exec'), n)
    # Explicit small-byte test adaptation only; generated target source is fixed.
    n['FILE_SIZE'] = 4224
    n['DATA_SHA'] = hashlib.sha256(b'\xa5'*4224).hexdigest()
    return n


def setup_case(tmp_path, n, cls='FrozenStore'):
    root = tmp_path/'probe'
    root.mkdir(mode=0o700)
    budget = n['Budget'](root, 60, lambda: 0)
    case = root/'frozen'
    case.mkdir(mode=0o700)
    store = n[cls](n['OneFileInventory'](case, budget))
    return root, budget, store


def outcome(tmp_path, n):
    parent = tmp_path/'scratch'
    parent.mkdir(mode=0o700)
    observed = {'host': n['EXPECTED_HOST'], 'uid': os.getuid(), 'device': parent.stat().st_dev,
                'filesystem': 'local-test-NOT-XFS', 'parent_identity': n['identity'](parent.stat())[:4],
                'free_bytes_observed': 2**30, 'free_inodes_observed': 1000}
    return n['run_probe'](NONCE, CSHA, '3'*64, clock=lambda: 0, inspect=lambda: (observed, parent))


def fixture_package(tmp_path):
    repo = tmp_path/'repo'
    repo.mkdir(mode=0o700)
    for name, data in sources().items():
        dest = repo/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        dest.chmod(0o600)
    parent = tmp_path/'handoffs'
    parent.mkdir(mode=0o700)
    dest = parent/'final'
    client.prepare(repo, dest)
    c, csha = client.control((dest/'CONTROL.json').read_bytes())
    return dest, c, csha


def arm_test(dest, csha):
    client.write_new(dest, 'BYTE-AUTHORIZATION.json', client.encoded(client.authorization(csha, 'INERT LOCAL TEST ONLY')))


def response(c, csha, nonce):
    return dict(kind='pgd-byte-close-observation-v1', request_nonce=nonce, control_sha256=csha,
                program_sha256=c['program_sha256'], checked_utc='2026-11-16T23:00:00+00:00',
                preflight=None, root=None, cases=[], footprint=None,
                error={'type': 'ValueError', 'message': 'INERT refusal', 'notes': []},
                assessment='inconclusive', elapsed_seconds=0.1, **core.FLAGS)


def test_exact_build_class_methods_helpers_and_no_project_import():
    src = sources()
    program = client.build(src)
    tree = ast.parse(program)
    for old, new in ((client.FROZEN, 'FrozenStore'), (client.CANDIDATE, 'CandidateStore')):
        text = client.definition(src[old], 'Store').replace(b'class Store:', ('class '+new+':').encode(), 1)
        assert client.definition(program, new) == text
    for name in ('identity', 'private'):
        assert client.definition(program, name) == client.definition(src[client.FROZEN], name)
    roots = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    roots |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert roots <= {'datetime', 'hashlib', 'json', 'math', 'os', 'pathlib', 'platform', 'stat', 'sys', 'threading', 'time', 'types'}
    assert not {'models', 'scripts', 'numpy', 'torch', 'subprocess', 'socket'} & roots
    assert len(program) <= 65536
    assert core.FILE_SIZE == 12583040
    assert hashlib.sha256(b'\xa5'*core.FILE_SIZE).hexdigest() == core.DATA_SHA


def test_unknown_writer_revision_refused():
    src = sources()
    src[client.FROZEN] += b'\n'
    with pytest.raises(ValueError, match='reviewed writer'):
        client.build(src)


def test_real_small_byte_roundtrip_and_serial_case_order(tmp_path, ns):
    r = outcome(tmp_path, ns)
    (tmp_path/'LOCAL-NOT-XFS-observation.json').write_bytes(ns['encoded'](r))
    assert r['assessment'] == 'failure_not_reproduced' and r['error'] is None
    assert [c['kind'] for c in r['cases']] == ['frozen', 'candidate']
    for c in r['cases']:
        assert c['classification'] == 'consistent'
        assert c['readback']['sha256'] == ns['DATA_SHA']
        o = c['close_observations'][0]
        assert o['before_held'] and o['after_named'] and not o['observation_errors']
        assert (o['committed_at_close'] is None) == (c['kind'] == 'candidate')
    assert r['footprint']['files'] == 3 and r['footprint']['directories'] == 9
    assert Path(r['root'], 'RESULT.json').is_file()


def test_recording_proxy_keeps_original_write_code(tmp_path, ns):
    _, _, store = setup_case(tmp_path, ns)
    original = store.write_bytes.__func__
    proxy = ns['bind_observer'](store)
    assert store.write_bytes.__func__.__code__ is original.__code__
    assert store.write_bytes.__func__.__globals__['os'] is proxy
    assert original.__globals__['os'] is os
    assert store.check.__func__.__globals__['os'] is os


def test_explicit_close_allocation_injection_classification(tmp_path, ns):
    original = ns['OneFileInventory'].scan
    def scan(inventory):
        r = original(inventory)
        p = inventory.root/ns['KEY']
        if p.exists() and inventory.root.name == 'frozen' and not closed[0]:
            r['files'][ns['KEY']]['allocated_bytes'] += 4096
        return r
    closed = [False]
    ns['OneFileInventory'].scan = scan
    original_close = ns['RecordingOS'].close
    def close(proxy, fd):
        tracked = fd in proxy.owned and proxy.store.root.name == 'frozen'
        if tracked:
            closed[0] = True
        original_close(proxy, fd)
        if tracked:
            o = proxy.observations[0]
            o['before_held']['allocated_bytes'] += 4096
            o['before_named']['allocated_bytes'] += 4096
    ns['RecordingOS'].close = close
    r = outcome(tmp_path, ns)
    (tmp_path/'EXPLICIT-INJECTION-NOT-XFS.json').write_bytes(ns['encoded'](r))
    assert r['error'] is None and r['assessment'] == 'allocation_transition_observed'
    assert r['cases'][0]['classification'] == 'allocation_only_refusal'
    assert r['cases'][1]['classification'] == 'consistent'


@pytest.mark.parametrize('fault', ['unknown', 'symlink', 'hardlink', 'mode', 'extent', 'allocation', 'deadline'])
def test_one_file_and_root_budget_negative_guards(tmp_path, ns, monkeypatch, fault):
    root, budget, store = setup_case(tmp_path, ns)
    if fault == 'unknown':
        (root/'foreign').write_bytes(b'x')
    elif fault == 'symlink':
        (root/'candidate').symlink_to(root/'frozen', target_is_directory=True)
    elif fault == 'hardlink':
        store.write_bytes(ns['KEY'], b'\xa5'*ns['FILE_SIZE'])
        os.link(store.root/ns['KEY'], root/'foreign')
    elif fault == 'mode':
        (root/'frozen').chmod(0o755)
    elif fault == 'extent':
        p = root/'RESULT.json'
        p.write_bytes(b'x'*65537); p.chmod(0o600)
    elif fault == 'allocation':
        store.write_bytes(ns['KEY'], b'\xa5'*ns['FILE_SIZE'])
        monkeypatch.setitem(ns['LIMITS'], 'allocated_bytes', 0)
    elif fault == 'deadline':
        budget.clock = lambda: 60
    with pytest.raises((ValueError, TimeoutError)):
        budget.check()


def test_logical_failure_stops_candidate(tmp_path, ns):
    def fail(store, *a):
        raise ValueError('unexpected logical identity mutation')
    ns['FrozenStore'].write_bytes = fail
    r = outcome(tmp_path, ns)
    assert len(r['cases']) == 1 and r['assessment'] == 'inconclusive'
    assert r['cases'][0]['write_error']['message'] == 'unexpected logical identity mutation'


def test_read_hash_failure_stops_candidate(tmp_path, ns):
    def fail(*a):
        raise ValueError('inert bytes differ')
    ns['readback'] = fail
    r = outcome(tmp_path, ns)
    assert len(r['cases']) == 1 and r['cases'][0]['read_error']


def test_exclusive_root_preserves_existing_contents(tmp_path, ns):
    parent = tmp_path/'scratch'; parent.mkdir(mode=0o700)
    root = parent/('vjepa-byte-close-v1-'+str(os.getuid())+'-'+CSHA)
    root.mkdir(mode=0o700)
    (root/'KEEP').write_bytes(b'preserve')
    r = ns['run_probe'](NONCE, CSHA, '3'*64, clock=lambda: 0,
                        inspect=lambda: ({'parent_identity': ns['identity'](parent.stat())[:4]}, parent))
    assert r['error']['type'] == 'FileExistsError' and not r['cases']
    assert (root/'KEEP').read_bytes() == b'preserve' and r['root'] is None


def test_wrong_runtime_preflight_has_no_write(ns, monkeypatch):
    class WrongPlatform:
        node = staticmethod(lambda: 'linux10603')
        system = staticmethod(lambda: 'Linux')
        python_version = staticmethod(lambda: '3.11.17')
    monkeypatch.setitem(ns, 'platform', WrongPlatform)
    r = ns['run_probe'](NONCE, CSHA, '3'*64, clock=lambda: 0)
    assert r['preflight'] is None and r['root'] is None and not r['cases']
    assert 'before writes' in r['error']['message']


def test_unarmed_old_grants_do_not_touch_clock_or_capture(tmp_path):
    for name in ('AUTHORIZATION.json', 'READ-AUTHORIZATION.json'):
        (tmp_path/name).write_bytes(b'{}')
    def forbidden(*a, **kw):
        raise AssertionError('No clock or SSH before new byte approval')
    with pytest.raises(ValueError, match='Unarmed'):
        client.run(tmp_path, clock=forbidden, capture_fn=forbidden)


def test_prepare_unarmed_pins_exact_sources(tmp_path):
    dest, c, csha = fixture_package(tmp_path)
    assert c['source_pins'][client.FROZEN] == client.FROZEN_SHA
    assert c['source_pins'][client.CANDIDATE] == client.CANDIDATE_SHA
    assert (dest/'PROBE.py').read_bytes() == client.build(sources())
    assert not (dest/'BYTE-AUTHORIZATION.json').exists() and not (dest/'BYTE-ATTEMPT.json').exists()
    assert stat.S_IMODE(dest.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in dest.iterdir())
    with pytest.raises(ValueError, match='Unarmed'):
        client.check_local(dest)


@pytest.mark.parametrize('fault', ['copied', 'source', 'hostkey', 'limits_bool', 'authorization', 'consumed'])
def test_bound_grant_source_path_and_limits(tmp_path, fault):
    dest, c, csha = fixture_package(tmp_path)
    arm_test(dest, csha)
    if fault == 'copied':
        copydir = dest.parent/'copy'; shutil.copytree(dest, copydir); dest = copydir
    elif fault == 'source':
        (Path(c['repository'])/client.SOURCES[0]).write_bytes(b'changed')
    elif fault == 'hostkey':
        (dest/'known-hosts').write_bytes(b'unknown key')
    elif fault == 'limits_bool':
        c['limits']['connections'] = True
        (dest/'CONTROL.json').write_bytes(client.encoded(c))
    elif fault == 'authorization':
        (dest/'BYTE-AUTHORIZATION.json').write_bytes(b'{}')
    elif fault == 'consumed':
        (dest/'BYTE-ATTEMPT.json').write_bytes(b'{}')
    with pytest.raises(ValueError):
        client.check_local(dest)


def test_one_fake_connection_and_byte_claim_preserve_refusal(tmp_path):
    dest, c, csha = fixture_package(tmp_path); arm_test(dest, csha)
    calls = []
    def capture(argv, **kwargs):
        calls.append((argv, kwargs))
        return 1, client.encoded(response(c, csha, NONCE)), b'INERT stderr'
    assert client.run(dest, clock=lambda: 0, capture_fn=capture, nonce_fn=lambda: NONCE) == 1
    assert len(calls) == 1
    claim = json.loads((dest/'BYTE-ATTEMPT.json').read_bytes())
    assert claim['consumed'] is True and claim['deadline_monotonic'] == 180
    receipt = json.loads((dest/'byte-response/RESULT.json').read_bytes())
    assert receipt['response_validated'] and receipt['assessment'] == 'inconclusive'
    assert (dest/'byte-response/STDERR.bin').read_bytes() == b'INERT stderr'
    with pytest.raises(ValueError, match='consumed'):
        client.run(dest, clock=lambda: 0, capture_fn=capture)
    assert len(calls) == 1


def test_capture_error_keeps_partial_response_no_retry(tmp_path):
    dest, c, csha = fixture_package(tmp_path); arm_test(dest, csha)
    def capture(*a, **kw):
        raise client.transport.CaptureError('INERT timeout', b'partial-json', b'partial-stderr')
    assert client.run(dest, clock=lambda: 0, capture_fn=capture, nonce_fn=lambda: NONCE) == 1
    assert (dest/'byte-response/STDOUT.bin').read_bytes() == b'partial-json'
    assert (dest/'byte-response/STDERR.bin').read_bytes() == b'partial-stderr'
    r = json.loads((dest/'byte-response/RESULT.json').read_bytes())
    assert not r['response_validated'] and 'INERT timeout' in r['error']


def test_pinned_ssh_argv_timeout_and_no_forwarding(tmp_path):
    dest, c, sha = fixture_package(tmp_path)
    argv = client.ssh_command(dest, c, sha, (dest/'PROBE.py').read_bytes(), NONCE)
    assert argv[:4] == ['ssh', '-F', '/dev/null', '-T']
    assert argv[-2] == 'kdu@linux10609.dc.engr.scu.edu'
    assert 'StrictHostKeyChecking=yes' in argv and 'ForwardAgent=no' in argv
    assert 'GlobalKnownHostsFile=/dev/null' in argv and 'ConnectionAttempts=1' in argv
    assert '--kill-after=2s 65s' in argv[-1] and ' -I -S -B -u -c ' in argv[-1]
    assert len(argv[-1].encode()) <= 100000


@pytest.mark.parametrize('fault', ['binding', 'false_boolean', 'false_admission', 'clock', 'claim_success', 'extra', 'oversize'])
def test_response_binding_bounds_and_no_false_admission(tmp_path, fault):
    dest, c, sha = fixture_package(tmp_path)
    r = response(c, sha, NONCE)
    if fault == 'binding': r['control_sha256'] = '4'*64
    elif fault == 'false_boolean': r['byte_only'] = 1
    elif fault == 'false_admission': r['training_admitted'] = True
    elif fault == 'clock': r['elapsed_seconds'] = 66
    elif fault == 'claim_success': r['assessment'] = 'allocation_transition_observed'
    elif fault == 'extra': r['extra'] = 'unrecognized'
    raw = client.encoded(r) if fault != 'oversize' else b'x'*65537
    with pytest.raises(ValueError):
        client.validate_response(raw, c, sha, NONCE)


def test_finalization_never_resets_work_deadline(tmp_path, ns):
    parent = tmp_path/'scratch'; parent.mkdir(mode=0o700)
    observed = {'parent_identity': ns['identity'](parent.stat())[:4]}
    times = iter((0, 0, 60, 65))
    clock = lambda: next(times, 65)
    r = ns['run_probe'](NONCE, CSHA, '3'*64, clock=clock, inspect=lambda: (observed, parent))
    assert r['error']['type'] == 'TimeoutError' and r['assessment'] == 'inconclusive' and not r['cases']


@pytest.mark.parametrize('fault', ['none', 'filesystem', 'readonly', 'device', 'home_device', 'mode', 'free_bytes', 'free_inodes'])
def test_preflight_mount_and_scratch_guards_with_explicit_fakes(ns, monkeypatch, fault):
    import io
    from types import SimpleNamespace as N
    parent_stat = N(st_dev=66307, st_ino=10, st_uid=0, st_mode=stat.S_IFDIR|0o1777, st_nlink=2)
    home_stat = N(st_dev=55)
    free = N(f_bavail=128*2**20, f_frsize=1, f_favail=100)
    filesystem, options = 'xfs', 'rw'
    if fault == 'filesystem': filesystem = 'nfs4'
    elif fault == 'readonly': options = 'ro'
    elif fault == 'device': parent_stat.st_dev = 55
    elif fault == 'home_device': home_stat.st_dev = 66307
    elif fault == 'mode': parent_stat.st_mode = stat.S_IFDIR|0o755
    elif fault == 'free_bytes': free.f_bavail = 1
    elif fault == 'free_inodes': free.f_favail = 1
    mount = ('1 0 '+str(os.major(66307))+':'+str(os.minor(66307))+' / / '+options+' - '+filesystem+' /dev/test '+options+'\n').encode()
    class FakePath:
        def __init__(self, name): self.name = name
        def lstat(self): return parent_stat
        def stat(self): return home_stat
        def resolve(self, **kw): return self
        @staticmethod
        def home(): return FakePath('FAKE_HOME')
    proxy = types.SimpleNamespace(**{n: getattr(os, n) for n in ('major', 'minor', 'getuid')})
    proxy.statvfs = lambda p: free
    monkeypatch.setitem(ns, 'Path', FakePath)
    monkeypatch.setitem(ns, 'os', proxy)
    monkeypatch.setitem(ns, 'platform', N(node=lambda: 'linux10609', system=lambda: 'Linux', python_version=lambda: '3.11.17'))
    monkeypatch.setitem(ns, 'open', lambda path, mode: io.BytesIO(mount))
    if fault == 'none':
        r, p = ns['preflight']()
        assert r['filesystem'] == 'xfs' and r['device'] == 66307
        assert r['parent_identity'] == (66307, 10, 0, 0o1777)
    else:
        with pytest.raises(ValueError): ns['preflight']()


def test_response_success_recomputed_from_inert_record_fixture(tmp_path, ns):
    dest, c, csha = fixture_package(tmp_path)
    r = outcome(tmp_path, ns)
    # Wire-validation-only adaptation of real small local byte records. This is
    # not an assertion that this Mac is XFS or that target data was observed.
    r['control_sha256'], r['program_sha256'] = csha, c['program_sha256']
    p = r['preflight']
    p.update(host=core.EXPECTED_HOST, device=66307, filesystem='xfs', parent_identity=[66307, 10, 0, 0o1777])
    r['root'] = '/var/tmp/vjepa-byte-close-v1-'+str(p['uid'])+'-'+csha
    for case in r['cases']:
        case['receipt'].update(bytes=core.FILE_SIZE, sha256=core.DATA_SHA)
        case['readback'].update(bytes=core.FILE_SIZE, sha256=core.DATA_SHA)
        rows = [case['committed'], case['before_read'], case['readback']['before'], case['readback']['after']]
        rows += [case['close_observations'][0][k] for k in ('before_held', 'before_named', 'after_named', 'committed_at_close')]
        for row in rows:
            if row is not None: row['bytes'] = core.FILE_SIZE
    validated = client.validate_response(client.encoded(r), c, csha, NONCE)
    assert validated['assessment'] == 'failure_not_reproduced'
    forged = copy.deepcopy(r)
    forged['cases'][1]['committed']['allocated_bytes'] += 4096
    with pytest.raises(ValueError, match='classification'):
        client.validate_response(client.encoded(forged), c, csha, NONCE)


def test_cli_unarmed_refuses_before_import_or_sys_path_change(tmp_path):
    import sys
    namespace = {'__name__': 'inert_cli_definition'}
    exec(compile((REPO/'scripts/run_pgd_byte_close_probe_v1.py').read_bytes(), '<INERT-CLI>', 'exec'), namespace)
    before = list(sys.path)
    with pytest.raises(SystemExit, match='Unarmed'):
        namespace['main']([str(tmp_path)])
    assert sys.path == before


@pytest.mark.parametrize('close_error', [False, True])
def test_observer_diagnostics_never_prevent_owned_close(ns, monkeypatch, close_error):
    calls = []
    def fstat(fd): raise OSError('INERT diagnostic fstat failure')
    def close(fd):
        calls.append(fd)
        if close_error: raise OSError('INERT owned close failure')
    proxy_os = types.SimpleNamespace(fstat=fstat, close=close)
    class MissingPath:
        def __truediv__(self, key): return self
        def lstat(self): raise FileNotFoundError('INERT missing file')
    store = types.SimpleNamespace(root=MissingPath(), written={})
    observer = ns['RecordingOS'](store); observer.owned.add(99)
    monkeypatch.setitem(ns, 'os', proxy_os)
    if close_error:
        with pytest.raises(OSError, match='owned close failure'): observer.close(99)
    else:
        observer.close(99)
    assert calls == [99] and observer.owned == set()
    assert len(observer.observations) == 1 and len(observer.observations[0]['observation_errors']) == 2
    assert (observer.observations[0]['close_error'] is not None) == close_error


def test_observer_open_fstat_failure_closes_only_new_fd(ns, monkeypatch):
    closed = []
    def fstat(fd): raise OSError('INERT new-fd fstat failure')
    proxy_os = types.SimpleNamespace(open=lambda *a, **kw: 99, close=lambda fd: closed.append(fd),
                                     fstat=fstat, O_WRONLY=os.O_WRONLY)
    observer = ns['RecordingOS'](types.SimpleNamespace(written={}))
    monkeypatch.setitem(ns, 'os', proxy_os)
    with pytest.raises(OSError, match='new-fd fstat failure'): observer.open('INERT', os.O_WRONLY)
    assert closed == [99] and not observer.owned
