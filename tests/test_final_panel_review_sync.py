"""Original final reviews, interrupted appends, carried clocks and shared-home locks."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/Users/kaiwendu/Documents/GitHub/V-JEPA-Robust-Overfitting')
CORE = Path(os.environ.get('FINAL_REVIEW_SYNC_TEST_CORE', str(REPO / 'runs/pretrained/diagnostic/last-two-panel-20261028/extracted/last-two-panel-20261028')))
REFERENCE = REPO / 'runs/pretrained/diagnostic/two-host-panel-20261026/extracted/two-host-execution-20261026'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


m = module('final_review_tests', ROOT / 'models/final_panel_review_sync.py')
core = module('original_final_case_coordinator', ROOT / 'support/reviewed_panel_last_two.py' if (ROOT / 'support/reviewed_panel_last_two.py').exists() else CORE / 'models/reviewed_panel_last_two.py')
prior = core.boundary(CORE, m.CORE_PIN)['prior']
ops = module('final_review_test_locks', REFERENCE / 'models/two_host_panel.py')
FIX = json.loads((ROOT / 'tests/fixtures/final_panel_review_sync_metadata.json').read_text())


def operation():
    return dict(m.policy(), inputs_file_sha256='1' * 64,
                sources_sha256={n: m.CORE_SOURCE if n.startswith('support/') else '1' * 64 for n in m.SOURCES})


def fixture(tmp_path):
    transfer = tmp_path / 'shared'
    root = transfer / m.CORE_ROOT
    root.mkdir(parents=True)
    for name, text in FIX['case_metadata'].items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    packet = transfer / m.CORE_PREFIX / 'unpacked' / m.CORE_PREFIX
    for host, cid in m.CASES.items():
        launcher = packet / 'launchers' / (host + '-' + cid)
        launcher.mkdir(parents=True)
        for name, text in FIX['launcher_files'][host].items():
            (launcher / name).write_text(text)
        case = root / 'cases' / cid / 'attempt-01'
        case.mkdir(parents=True)
        ops.write_record(case / 'case-envelope.json', FIX['case_envelopes'][cid])
    legacy = transfer / 'two-host-panel-20261026'
    legacy.mkdir()
    old_pins = {}
    for name in ('two-host-panel-20261026/STOP.json', 'reviewed-panel-20261026/STOP.json', 'archery-timecarry-execution-20261027/STOP.json'):
        p = transfer / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('Generated immutable failure fixture: ' + name)
        old_pins[name] = m.digest(p)
    context = dict(op=FIX['core_operation'], data=FIX['core_data'], root=root,
                   prior=prior, ops=ops, legacy=legacy, packet=packet)
    return dict(op=operation(), data=copy.deepcopy(FIX['input']), root=root, transfer=transfer,
                prior=prior, ops=ops, core=core, context=context), lambda: prior.check_preserved(transfer, old_pins)


def receipt(c):
    before = core.Registry(c['context']).status()
    return dict(kind=m.KIND + '-receipt', operation_sha256=m.canonical(c['op']),
                inputs_file_sha256=c['op']['inputs_file_sha256'], created=sorted(m.TARGETS), preexisting_identical=[],
                records_readback={n: b.decode() for n, b in m.targets(c['data']).items()},
                fresh_preserved_files_sha256=dict(c['data']['preserved_files_sha256']),
                status_before=before, status_after=dict(before, accepted_case_ids=m.ORDER, complete=True),
                saved_cases_reverified=2, time_before=m.time_readback(c['data']), time_after=m.time_readback(c['data']),
                historical_STOP_records_unchanged=True, models_started=False, test_suite_started=False,
                new_GPU_dispatch_authorized=False, additional_attempt_authorized=False, research_admitted=False)


@pytest.mark.parametrize('key,value', [('host', 'linux10603'), ('coordinator_namespace', 'another-root'),
    ('case_ids', ['clip-09-bank-1']), ('attempt', 'attempt-02'), ('accepted_after_sync', m.BEFORE),
    ('preserved_file_count', 928), ('max_seconds', 4800), ('metadata_only', False), ('models_started', True),
    ('test_suite_started', True), ('new_GPU_dispatch_authorized', True), ('additional_attempt_authorized', True),
    ('historical_STOP_records_unchanged', False), ('automatic_queue', True), ('research_admitted', True),
    ('core_module_sha256', '0' * 64), ('actual_activation_receipt_file_sha256', '0' * 64),
    ('previous_consumed_history_seconds', 0), ('bowling_startup_charge_seconds', 0)])
def test_changed_scope_or_authority_is_rejected(key, value):
    op = operation()
    op[key] = value
    with pytest.raises(ValueError):
        m.validate_operation(op, m.canonical(op))


def test_external_pin_and_frozen_coordinator_source_required():
    op = operation()
    m.validate_operation(op, m.canonical(op))
    with pytest.raises(ValueError):
        m.validate_operation(op, '0' * 64)
    op['sources_sha256']['support/reviewed_panel_last_two.py'] = '0' * 64
    with pytest.raises(ValueError):
        m.validate_operation(op, m.canonical(op))


@pytest.mark.parametrize('cid', ['clip-08-bank-1', 'clip-09-bank-1'])
@pytest.mark.parametrize('change', ['marker-text', 'marker-pin', 'archive-pin', 'outcome-pin', 'unsafe-path', 'precredit', 'clock', 'carry'])
def test_both_original_reviews_and_clocks_must_remain_exact(cid, change):
    d = copy.deepcopy(FIX['input'])
    pins = d['preserved_files_sha256']
    if change == 'marker-text': d['reviews'][cid]['text'] += ' '
    elif change == 'marker-pin': d['reviews'][cid]['sha256'] = '0' * 64
    elif change == 'archive-pin': pins[m.CORE_ROOT + '/cases/' + cid + '/attempt-01/continuation-evidence.tar.gz'] = '0' * 64
    elif change == 'outcome-pin': pins[m.CORE_ROOT + '/outcomes/' + cid + '.json'] = '0' * 64
    elif change == 'unsafe-path': pins['../escape'] = pins.pop(next(iter(pins)))
    elif change == 'precredit': pins[m.CORE_ROOT + '/reviews/' + cid + '.json'] = pins.pop(next(iter(pins)))
    elif change == 'clock': d['clocks'][cid]['conservative_charge_upper_seconds'] = 0
    else: d['clocks'][cid]['carried_startup_seconds'] += 4800
    with pytest.raises(ValueError): m.validate_input(d)


def test_lost_acknowledgment_after_first_append_resumes_without_rewriting(tmp_path, monkeypatch):
    c, preserve = fixture(tmp_path)
    old = {p: p.read_bytes() for p in c['transfer'].rglob('*') if p.is_file()}
    atomic = prior.exclusive_bytes
    def interrupted(path, body):
        atomic(path, body)
        raise RuntimeError('Generated lost acknowledgment after the first atomic append')
    with monkeypatch.context() as scope:
        scope.setattr(prior, 'exclusive_bytes', interrupted)
        with pytest.raises(RuntimeError): m.publish(c['root'], m.targets(c['data']), prior, preserve)
    first = sorted(m.TARGETS)[0]
    m.check_state(core.Registry(c['context']).status(), [first])
    assert len(core.Registry(c['context']).status()['accepted_case_ids']) == 19
    assert m.publish(c['root'], m.targets(c['data']), prior, preserve) == ([sorted(m.TARGETS)[1]], [first])
    m.check_state(core.Registry(c['context']).status(), m.TARGETS)
    assert m.publish(c['root'], m.targets(c['data']), prior, preserve) == ([], sorted(m.TARGETS))
    assert old == {p: p.read_bytes() for p in old}


@pytest.mark.parametrize('change', ['conflict', 'symlink', 'directory'])
def test_conflicting_second_marker_prevents_first_append(tmp_path, change):
    c, preserve = fixture(tmp_path)
    first, second = sorted(m.TARGETS)
    p = c['root'] / second
    p.parent.mkdir()
    if change == 'conflict': p.write_text('{}')
    elif change == 'symlink': p.symlink_to(c['root'] / 'imports.json')
    else: p.mkdir()
    with pytest.raises(ValueError): m.publish(c['root'], m.targets(c['data']), prior, preserve)
    assert not (c['root'] / first).exists()


@pytest.mark.parametrize('change', ['missing', 'changed', 'symlink'])
def test_damaged_failure_history_blocks_publication(tmp_path, change):
    c, preserve = fixture(tmp_path)
    p = c['context']['legacy'] / 'STOP.json'
    if change == 'missing': p.unlink()
    elif change == 'changed': p.write_text('changed')
    else: p.unlink(); p.symlink_to(c['root'] / 'imports.json')
    with pytest.raises(ValueError): m.publish(c['root'], m.targets(c['data']), prior, preserve)
    assert all(not (c['root'] / n).exists() for n in m.TARGETS)


@pytest.mark.parametrize('change', ['readback', 'overlap', 'history', 'before-count', 'after-count', 'unchecked', 'model', 'attempt', 'time-before', 'time-after', 'complete'])
def test_receipt_requires_twenty_exact_reviews_and_unchanged_time(tmp_path, change):
    c, _ = fixture(tmp_path)
    r = receipt(c)
    assert m.verify_receipt(r, c['op'], c['data'])['accepted_cases'] == 20
    if change == 'readback': r['records_readback'][sorted(m.TARGETS)[0]] += ' '
    elif change == 'overlap': r['preexisting_identical'] = [sorted(m.TARGETS)[0]]
    elif change == 'history': r['fresh_preserved_files_sha256'].pop(next(iter(r['fresh_preserved_files_sha256'])))
    elif change == 'before-count': r['status_before']['accepted_case_ids'] = m.ORDER[:18]
    elif change == 'after-count': r['status_after']['accepted_case_ids'] = m.BEFORE
    elif change == 'unchecked': r['saved_cases_reverified'] = 1
    elif change == 'model': r['models_started'] = True
    elif change == 'attempt': r['additional_attempt_authorized'] = True
    elif change == 'time-before': r['time_before']['completed_original_attempts']['clip-08-bank-1']['carried_startup_seconds'] = 0
    elif change == 'time-after': r['time_after']['time_carry']['additional_allowance_seconds'] = 4800
    else: r['status_after']['complete'] = False
    with pytest.raises(ValueError): m.verify_receipt(r, c['op'], c['data'])


@pytest.mark.parametrize('host', ['linux10603', 'linux10609'])
@pytest.mark.parametrize('name,value', [('started-epoch.txt', 'nan'), ('ended-utc.txt', '2026-10-28T22:05:19Z'), ('exit-status.txt', '1')])
def test_actual_launch_clocks_cannot_reset_or_expand(tmp_path, host, name, value):
    c, _ = fixture(tmp_path)
    p = c['context']['packet'] / 'launchers' / (host + '-' + m.CASES[host]) / name
    p.write_text(value)
    with pytest.raises((ValueError, ArithmeticError)): m.check_time(c)


def test_completed_charges_keep_bowling_startup_and_older_consumed_history(tmp_path):
    c, _ = fixture(tmp_path)
    t = m.check_time(c)
    assert t['completed_original_attempts']['clip-08-bank-1']['total_charged_original_case_seconds'] == 3204
    assert t['completed_original_attempts']['clip-09-bank-1']['total_charged_original_case_seconds'] == 3007
    assert t['bowling_startup_accounting']['conservative_startup_charge_seconds'] == 206
    assert t['previous_consumed_history']['completed_original_attempts']['clip-09-bank-0']['conservative_charge_upper_seconds'] == 2951
    assert t['previous_consumed_history']['previous_consumed_history']['previous_consumed_history']['total_charged_seconds'] == 4947


def test_apply_holds_real_mutex_and_both_leases_during_cpu_verification(tmp_path, monkeypatch):
    c, _ = fixture(tmp_path)
    monkeypatch.setattr(m, 'load', lambda *a: c)
    monkeypatch.setattr(m, 'check_preserved', lambda c: dict(c['data']['preserved_files_sha256']))
    def saved(c, temporary):
        for p in (c['root'] / '.registry.lock', *(c['context']['legacy'] / 'leases' / host for host in m.LEASE_HOSTS)):
            with pytest.raises(BlockingIOError):
                with ops.file_lock(p, wait_seconds=0): pass
        return 2
    monkeypatch.setattr(m, 'verify_saved', saved)
    r = m.apply(None, m.canonical(c['op']), tmp_path, 'linux10609')
    assert r['created'] == sorted(m.TARGETS)
    assert m.verify_receipt(r, c['op'], c['data'])['accepted_cases'] == 20


@pytest.mark.parametrize('change', ['incomplete-review', 'concurrent-stop'])
def test_failed_cpu_check_or_new_stop_prevents_all_appends(tmp_path, monkeypatch, change):
    c, _ = fixture(tmp_path)
    monkeypatch.setattr(m, 'load', lambda *a: c)
    monkeypatch.setattr(m, 'check_preserved', lambda c: dict(c['data']['preserved_files_sha256']))
    def saved(c, temporary):
        if change == 'concurrent-stop': ops.write_record(c['root'] / 'STOP.json', {'generated_stop': True})
        return 1 if change == 'incomplete-review' else 2
    monkeypatch.setattr(m, 'verify_saved', saved)
    with pytest.raises(ValueError): m.apply(None, m.canonical(c['op']), tmp_path, 'linux10609')
    assert all(not (c['root'] / n).exists() for n in m.TARGETS)


@pytest.mark.parametrize('where', ['linux10609', 'linux10603', 'registry'])
def test_existing_owner_blocks_registration(tmp_path, monkeypatch, where):
    c, _ = fixture(tmp_path)
    monkeypatch.setattr(m, 'load', lambda *a: c)
    monkeypatch.setattr(m, 'check_preserved', lambda c: dict(c['data']['preserved_files_sha256']))
    monkeypatch.setattr(m, 'verify_saved', lambda *a: 2)
    p = c['root'] / '.registry.lock' if where == 'registry' else c['context']['legacy'] / 'leases' / where
    p.parent.mkdir(parents=True, exist_ok=True)
    code = 'import fcntl,sys;f=open(sys.argv[1],"a+");fcntl.flock(f,fcntl.LOCK_EX);print("locked",flush=True);sys.stdin.read()'
    child = subprocess.Popen([sys.executable, '-B', '-c', code, str(p)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'locked'
        if where == 'registry':
            original = ops.file_lock
            monkeypatch.setattr(ops, 'file_lock', lambda p, **kw: original(p, wait_seconds=0))
        with pytest.raises(BlockingIOError): m.apply(None, m.canonical(c['op']), tmp_path, 'linux10609')
        assert all(not (c['root'] / n).exists() for n in m.TARGETS)
    finally:
        child.stdin.close(); child.wait(timeout=5); child.stdout.close()


def test_real_NFS_record_lock_access(tmp_path, monkeypatch):
    import fcntl
    c, _ = fixture(tmp_path)
    monkeypatch.setattr(m, 'load', lambda *a: c)
    monkeypatch.setattr(m, 'check_preserved', lambda c: dict(c['data']['preserved_files_sha256']))
    monkeypatch.setattr(m, 'verify_saved', lambda *a: 2)
    monkeypatch.setattr(fcntl, 'flock', fcntl.lockf)
    assert m.apply(None, m.canonical(c['op']), tmp_path, 'linux10609')['created'] == sorted(m.TARGETS)


@pytest.mark.parametrize('change', ['missing', 'changed', 'symlink'])
def test_damaged_history_rejected_before_older_constructors(tmp_path, monkeypatch, change):
    home = tmp_path / 'home'
    transfer = home / m.BASE
    packet = transfer / m.PREFIX / 'unpacked' / m.PREFIX
    packet.mkdir(parents=True)
    for name in m.SOURCES:
        source = ROOT / name if not name.startswith('support/') else CORE / 'models/reviewed_panel_last_two.py'
        target = packet / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (packet / 'INPUT.json').write_text(json.dumps(FIX['input'], sort_keys=True, indent=2) + '\n')
    source = REPO / 'runs/pretrained/diagnostic/last-two-panel-20261028/activation-session-20261028T210439Z.lJLToN/receipt.json'
    shutil.copyfile(source, packet / 'activation-receipt.json')
    op = dict(m.policy(), inputs_file_sha256=m.digest(packet / 'INPUT.json'), sources_sha256={n: m.digest(packet / n) for n in m.SOURCES})
    (packet / 'SYNC.json').write_text(json.dumps(op))
    first = transfer / next(iter(FIX['input']['preserved_files_sha256']))
    first.parent.mkdir(parents=True, exist_ok=True)
    if change == 'changed': first.write_text('changed original')
    elif change == 'symlink': first.symlink_to(packet / 'INPUT.json')
    def forbidden(*a): raise AssertionError('History must be checked before older constructors')
    monkeypatch.setattr(m, 'module', lambda *a: SimpleNamespace(boundary=lambda *a: dict(prior=prior), load=forbidden))
    with pytest.raises(ValueError): m.load(packet, m.canonical(op), home)
    assert all(not (transfer / m.CORE_ROOT / n).exists() for n in m.TARGETS)


def test_wrong_host_rejected_without_loading(monkeypatch, tmp_path):
    def forbidden(*a): raise AssertionError('wrong metadata host')
    monkeypatch.setattr(m, 'load', forbidden)
    with pytest.raises(ValueError): m.apply(None, '0' * 64, tmp_path, 'linux10603')
