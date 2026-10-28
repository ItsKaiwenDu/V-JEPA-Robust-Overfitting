"""CPU-only original-review publication, failure-history, clock and real-lock tests."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/Users/kaiwendu/Documents/GitHub/V-JEPA-Robust-Overfitting')
CORE = Path(os.environ.get('ARCHERY_SYNC_TEST_CORE', str(REPO / 'runs/pretrained/diagnostic/archery-recovery-20261027/extracted/archery-recovery-20261027')))
REFERENCE = Path(os.environ.get('ARCHERY_SYNC_TEST_REFERENCE', str(REPO / 'runs/pretrained/diagnostic/two-host-panel-20261026/extracted/two-host-execution-20261026')))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result

m = module('archery_sync_unit_tests', ROOT / 'models/archery_review_sync.py')
core = module('archery_sync_original_core', ROOT / 'support/archery_recovery.py' if (ROOT / 'support/archery_recovery.py').exists() else CORE / 'models/archery_recovery.py')
prior = module('archery_sync_original_records', CORE / core.RECORD_SUPPORT)
ops = module('archery_sync_original_locks', REFERENCE / 'models/two_host_panel.py')
FIX = json.loads((ROOT / 'tests/fixtures/archery_review_sync_metadata.json').read_text())


def operation():
    return dict(m.policy(), inputs_file_sha256='1' * 64,
        sources_sha256={n: m.CORE_SOURCE if n.startswith('support/') else '1' * 64 for n in m.SOURCES})


def fixture(tmp_path):
    transfer = tmp_path / 'shared'; root = transfer / m.CORE_ROOT; root.mkdir(parents=True)
    for n, text in FIX['case_metadata'].items():
        path = root / n; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
    packet = transfer / m.CORE_PREFIX / 'unpacked' / m.CORE_PREFIX
    launcher = packet / 'launchers' / (m.HOST + '-' + m.CASE); launcher.mkdir(parents=True)
    for n, text in FIX['launcher_files'].items(): (launcher / n).write_text(text)
    case = root / 'cases' / m.CASE / m.ATTEMPT; case.mkdir(parents=True)
    # The real envelope is also retained in the fixture so extracted tests are standalone.
    ops.write_record(case / 'case-envelope.json', FIX['case_envelope'])
    legacy = transfer / 'two-host-panel-20261026'; legacy.mkdir()
    old_pins = {}
    for n in ('two-host-panel-20261026/STOP.json', 'reviewed-panel-20261026/STOP.json', 'archery-timecarry-execution-20261027/STOP.json'):
        p = transfer / n; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('Generated immutable old failure fixture: ' + n)
        old_pins[n] = m.digest(p)
    context = {'op': FIX['core_operation'], 'data': FIX['core_data'], 'root': root,
        'prior': prior, 'ops': ops, 'legacy': legacy, 'packet': packet}
    c = {'op': operation(), 'data': copy.deepcopy(FIX['input']), 'root': root, 'transfer': transfer,
        'prior': prior, 'ops': ops, 'core': core, 'context': context}
    return c, lambda: prior.check_preserved(transfer, old_pins)


def completed_receipt(c):
    before = core.Registry(c['context']).status()
    return {'kind': m.KIND + '-receipt', 'operation_sha256': m.canonical(c['op']),
        'inputs_file_sha256': c['op']['inputs_file_sha256'], 'created': [m.TARGET], 'preexisting_identical': [],
        'records_readback': {m.TARGET: m.targets(c['data'])[m.TARGET].decode()},
        'fresh_preserved_files_sha256': dict(c['data']['preserved_files_sha256']),
        'status_before': before, 'status_after': dict(before, accepted_case_ids=m.ORDER[:15]),
        'saved_cases_reverified': 1, 'time_before': m.time_readback(c['data']), 'time_after': m.time_readback(c['data']),
        'historical_STOP_records_unchanged': True, 'models_started': False, 'test_suite_started': False,
        'new_GPU_dispatch_authorized': False, 'additional_attempt_authorized': False, 'research_admitted': False}


def test_original_recovery_marker_and_578_history_pins():
    m.validate_input(FIX['input']); assert len(FIX['input']['preserved_files_sha256']) == 578
    assert m.marker(FIX['input'])['execution_sha256'] == m.EXECUTION
    assert set(m.targets(FIX['input'])) == {m.TARGET}


@pytest.mark.parametrize('key,value', [
    ('host', 'linux10603'), ('coordinator_namespace', 'archery-timecarry-execution-20261027'),
    ('case_ids', ['clip-07-bank-1']), ('attempt', 'attempt-01'), ('accepted_after_sync', m.BEFORE),
    ('preserved_file_count', 470), ('max_seconds', 4800), ('kill_grace_seconds', 16),
    ('metadata_only', False), ('models_started', True), ('test_suite_started', True),
    ('new_GPU_dispatch_authorized', True), ('historical_STOP_records_unchanged', False),
    ('automatic_queue', True), ('research_admitted', True), ('core_module_sha256', '0' * 64),
    ('actual_activation_receipt_file_sha256', '0' * 64), ('original_review_marker_file_sha256', '0' * 64),
    ('consumed_grant_id', 'another-grant'), ('previous_charged_seconds', 0),
    ('recovery_charged_seconds', 0), ('total_charged_seconds', 2599), ('old_remainder_added', True),
    ('accounting_file_sha256', '0' * 64), ('additional_attempt_authorized', True)])
def test_changes_to_scope_failure_history_time_or_authority_rejected(key, value):
    op = operation(); op[key] = value
    with pytest.raises(ValueError): m.validate_operation(op, m.canonical(op))


def test_source_external_pin_and_full_inventory_required():
    op = operation(); m.validate_operation(op, m.canonical(op))
    with pytest.raises(ValueError): m.validate_operation(op, '0' * 64)
    op['sources_sha256'].pop(m.SOURCES[0])
    with pytest.raises(ValueError): m.validate_operation(op, m.canonical(op))


@pytest.mark.parametrize('change', ['marker-text', 'marker-pin', 'history-drop', 'archive-pin', 'outcome-pin', 'unsafe-path', 'precredited-review', 'account-bytes', 'account-pin'])
def test_mixed_unpinned_or_rewritten_originals_rejected(change):
    d = copy.deepcopy(FIX['input']); pins = d['preserved_files_sha256']
    if change == 'marker-text': d['review']['text'] += ' '
    elif change == 'marker-pin': d['review']['sha256'] = '0' * 64
    elif change == 'history-drop': pins.pop(next(iter(pins)))
    elif change == 'archive-pin': pins[m.CORE_ROOT + '/cases/' + m.CASE + '/recovery-01/continuation-evidence.tar.gz'] = '0' * 64
    elif change == 'outcome-pin': pins[m.CORE_ROOT + '/outcomes/' + m.CASE + '.json'] = '0' * 64
    elif change == 'unsafe-path': pins['../escape'] = pins.pop(next(iter(pins)))
    elif change == 'precredited-review': pins[m.CORE_ROOT + '/' + m.TARGET] = pins.pop(next(iter(pins)))
    elif change == 'account-bytes': d['accounting']['text'] += '\n'
    else: d['accounting']['sha256'] = '0' * 64
    with pytest.raises(ValueError): m.validate_input(d)


def test_atomic_append_and_lost_acknowledgment_are_idempotent(tmp_path):
    c, preserve = fixture(tmp_path); values = m.targets(c['data'])
    old = {p: p.read_bytes() for p in c['transfer'].rglob('*') if p.is_file()}
    before = core.Registry(c['context']).status(); m.check_state(before, [])
    calls = []
    def lost_ack():
        preserve(); calls.append(1)
        if len(calls) == 2: raise RuntimeError('Generated lost acknowledgment after append')
    with pytest.raises(RuntimeError): m.publish(c['root'], values, prior, lost_ack)
    created, existing = m.publish(c['root'], values, prior, preserve)
    assert created == [] and existing == [m.TARGET]
    m.check_state(core.Registry(c['context']).status(), [m.TARGET])
    assert old == {p: p.read_bytes() for p in old}


def test_failure_before_atomic_append_leaves_no_review(tmp_path, monkeypatch):
    c, preserve = fixture(tmp_path)
    def fail(*args): raise RuntimeError('Generated prepublication interruption')
    monkeypatch.setattr(prior, 'exclusive_bytes', fail)
    with pytest.raises(RuntimeError): m.publish(c['root'], m.targets(c['data']), prior, preserve)
    assert not (c['root'] / m.TARGET).exists()


@pytest.mark.parametrize('change', ['conflict', 'symlink', 'directory'])
def test_conflicting_or_redirected_review_is_never_replaced(tmp_path, change):
    c, preserve = fixture(tmp_path); p = c['root'] / m.TARGET; p.parent.mkdir()
    if change == 'conflict': p.write_text('{}')
    elif change == 'symlink': p.symlink_to(c['root'] / 'imports.json')
    else: p.mkdir()
    with pytest.raises(ValueError): m.publish(c['root'], m.targets(c['data']), prior, preserve)


@pytest.mark.parametrize('change', ['missing', 'changed', 'symlink'])
def test_old_failed_stop_drift_blocks_publication(tmp_path, change):
    c, preserve = fixture(tmp_path); p = c['context']['legacy'] / 'STOP.json'
    if change == 'missing': p.unlink()
    elif change == 'changed': p.write_text('changed')
    else: p.unlink(); p.symlink_to(c['root'] / 'imports.json')
    with pytest.raises(ValueError): m.publish(c['root'], m.targets(c['data']), prior, preserve)
    assert not (c['root'] / m.TARGET).exists()


@pytest.mark.parametrize('key,value', [('stop_dispatch', True), ('failed_case_ids', [m.CASE]),
    ('claimed_case_ids', []), ('computed_case_ids', []), ('accepted_case_ids', m.ORDER[:14]),
    ('complete', True), ('scientific_aggregate', {}), ('research_admitted', True)])
def test_unknown_or_unreviewed_state_not_accepted(tmp_path, key, value):
    c, _ = fixture(tmp_path); state = core.Registry(c['context']).status(); state[key] = value
    with pytest.raises(ValueError): m.check_state(state, [])


@pytest.mark.parametrize('change', ['readback', 'created-overlap', 'pin', 'before-count', 'after-count', 'unchecked', 'model', 'attempt', 'time-before', 'time-after'])
def test_receipt_requires_exact_marker_time_and_states(tmp_path, change):
    c, _ = fixture(tmp_path); r = completed_receipt(c); assert m.verify_receipt(r, c['op'], c['data'])['verified']
    if change == 'readback': r['records_readback'][m.TARGET] += ' '
    elif change == 'created-overlap': r['preexisting_identical'] = [m.TARGET]
    elif change == 'pin': r['fresh_preserved_files_sha256'].pop(next(iter(r['fresh_preserved_files_sha256'])))
    elif change == 'before-count': r['status_before']['accepted_case_ids'] = m.ORDER[:14]
    elif change == 'after-count': r['status_after']['accepted_case_ids'] = m.BEFORE
    elif change == 'unchecked': r['saved_cases_reverified'] = 0
    elif change == 'model': r['models_started'] = True
    elif change == 'attempt': r['additional_attempt_authorized'] = True
    elif change == 'time-before': r['time_before']['previous_charged_seconds'] = 0
    else: r['time_after']['old_remainder_added'] = True
    with pytest.raises(ValueError): m.verify_receipt(r, c['op'], c['data'])


@pytest.mark.parametrize('name,value', [('started-epoch.txt', 'nan'), ('started-epoch.txt', '1793159370.0821397'),
    ('ended-utc.txt', '2026-10-28T04:32:48Z'), ('exit-status.txt', '1')])
def test_recovery_clock_cannot_be_reset(tmp_path, name, value):
    c, _ = fixture(tmp_path); p = c['context']['packet'] / 'launchers' / (m.HOST + '-' + m.CASE) / name; p.write_text(value)
    with pytest.raises((ValueError, ArithmeticError)): m.check_time(c)


def test_original_clock_and_consumed_grant_readback(tmp_path):
    c, _ = fixture(tmp_path); assert m.check_time(c) == m.time_readback(c['data'])
    assert m.check_time(c)['total_charged_seconds'] == 4947


def test_apply_uses_real_locks_and_rechecks_saved_evidence_before_append(tmp_path, monkeypatch):
    c, _ = fixture(tmp_path); monkeypatch.setattr(m, 'load', lambda *a: c)
    monkeypatch.setattr(m, 'check_preserved', lambda c: dict(c['data']['preserved_files_sha256']))
    calls = []
    def saved(c, temporary):
        with pytest.raises(BlockingIOError):
            with ops.file_lock(c['root'] / '.registry.lock', wait_seconds=0): pass
        assert core.Registry(c['context']).status()['accepted_case_ids'] == m.BEFORE
        calls.append(temporary); return 1
    monkeypatch.setattr(m, 'verify_saved', saved)
    r = m.apply(None, m.canonical(c['op']), tmp_path, 'linux10609')
    assert len(calls) == 1 and r['created'] == [m.TARGET]
    assert m.verify_receipt(r, c['op'], c['data'])['accepted_cases'] == 15


def test_exclusive_lock_descriptors_satisfy_real_record_lock_access_rules(tmp_path, monkeypatch):
    import fcntl
    c, _ = fixture(tmp_path); monkeypatch.setattr(m, 'load', lambda *a: c)
    monkeypatch.setattr(m, 'check_preserved', lambda c: dict(c['data']['preserved_files_sha256']))
    monkeypatch.setattr(m, 'verify_saved', lambda *a: 1); monkeypatch.setattr(fcntl, 'flock', fcntl.lockf)
    r = m.apply(None, m.canonical(c['op']), tmp_path, 'linux10609')
    assert r['created'] == [m.TARGET] and m.verify_receipt(r, c['op'], c['data'])['verified']


@pytest.mark.parametrize('where', ['linux10609', 'linux10603', 'registry'])
def test_another_process_holding_any_lease_prevents_review_append(tmp_path, monkeypatch, where):
    c, _ = fixture(tmp_path); monkeypatch.setattr(m, 'load', lambda *a: c)
    monkeypatch.setattr(m, 'check_preserved', lambda c: dict(c['data']['preserved_files_sha256']))
    monkeypatch.setattr(m, 'verify_saved', lambda *a: 1)
    path = c['root'] / '.registry.lock' if where == 'registry' else c['context']['legacy'] / 'leases' / where
    path.parent.mkdir(parents=True, exist_ok=True)
    code = 'import fcntl,sys;f=open(sys.argv[1],"a+");fcntl.flock(f,fcntl.LOCK_EX);print("locked",flush=True);sys.stdin.read()'
    p = subprocess.Popen([sys.executable, '-B', '-c', code, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == 'locked'
        if where == 'registry':
            original = ops.file_lock
            monkeypatch.setattr(ops, 'file_lock', lambda p, **kw: original(p, wait_seconds=0))
        with pytest.raises(BlockingIOError): m.apply(None, m.canonical(c['op']), tmp_path, 'linux10609')
        assert not (c['root'] / m.TARGET).exists()
    finally:
        p.stdin.close(); p.wait(timeout=5); p.stdout.close()


def test_wrong_host_never_loads_or_publishes(monkeypatch, tmp_path):
    def forbidden(*a): raise AssertionError('Wrong host must stop before loading')
    monkeypatch.setattr(m, 'load', forbidden)
    with pytest.raises(ValueError): m.apply(None, '0' * 64, tmp_path, 'linux10603')


@pytest.mark.parametrize('change', ['missing', 'changed', 'symlink'])
def test_damaged_history_is_rejected_before_an_older_constructor_can_initialize(tmp_path, monkeypatch, change):
    import shutil
    from types import SimpleNamespace
    home = tmp_path / 'home'; transfer = home / m.BASE
    packet = transfer / m.PREFIX / 'unpacked' / m.PREFIX; packet.mkdir(parents=True)
    for name in m.SOURCES:
        source = ROOT / name if not name.startswith('support/') else CORE / 'models/archery_recovery.py'
        dest = packet / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, dest)
    (packet / 'INPUT.json').write_text(json.dumps(FIX['input'], sort_keys=True, indent=2) + '\n')
    source = REPO / 'runs/pretrained/diagnostic/archery-recovery-20261027/activation-session-20261028T034037Z.23bdUE/receipt.json'
    # Use a pinned copy supplied through the real core fixture environment after extraction.
    receipt_source = Path(os.environ.get('ARCHERY_SYNC_TEST_ACTIVATION', str(source)))
    shutil.copyfile(receipt_source, packet / 'activation-receipt.json')
    op = dict(m.policy(), inputs_file_sha256=m.digest(packet / 'INPUT.json'),
        sources_sha256={n: m.digest(packet / n) for n in m.SOURCES})
    (packet / 'SYNC.json').write_text(json.dumps(op))
    first = transfer / next(iter(FIX['input']['preserved_files_sha256']))
    first.parent.mkdir(parents=True, exist_ok=True)
    if change == 'changed': first.write_text('Changed immutable original')
    elif change == 'symlink': first.symlink_to(packet / 'INPUT.json')
    def constructor_forbidden(*args): raise AssertionError('Must not initialize or repair any older coordinator')
    guarded = SimpleNamespace(boundary=lambda *a: {'prior': prior}, load=constructor_forbidden)
    monkeypatch.setattr(m, 'module', lambda *a: guarded)
    before = {p: p.read_bytes() for p in transfer.rglob('*') if p.is_file() and not p.is_symlink()}
    with pytest.raises(ValueError): m.load(packet, m.canonical(op), home)
    assert before == {p: p.read_bytes() for p in transfer.rglob('*') if p.is_file() and not p.is_symlink()}
    assert not (transfer / m.CORE_ROOT / m.TARGET).exists()
