"""Metadata recovery, original identity, elapsed-time and real-lock checks."""
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
CORE = Path(os.environ.get('PAIR_REVIEW_SYNC_TEST_CORE', str(REPO / 'runs/pretrained/diagnostic/bowling-handstand-pair-20261028/extracted/bowling-handstand-pair-20261028')))
REFERENCE = Path(os.environ.get('PAIR_REVIEW_SYNC_TEST_REFERENCE', str(REPO / 'runs/pretrained/diagnostic/two-host-panel-20261026/extracted/two-host-execution-20261026')))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value

m = module('pair_original_review_tests', ROOT / 'models/handstand_review_sync.py')
core = module('completed_pair_test_core', ROOT / 'support/reviewed_panel_final_pair.py' if (ROOT / 'support/reviewed_panel_final_pair.py').exists() else CORE / 'models/reviewed_panel_final_pair.py')
prior = core.boundary(CORE, m.CORE_PIN)['prior']
ops = module('original_pair_test_locks', REFERENCE / 'models/two_host_panel.py')
FIX = json.loads((ROOT / 'tests/fixtures/handstand_review_sync_metadata.json').read_text())


def operation():
    return dict(m.policy(), inputs_file_sha256='1' * 64,
        sources_sha256={n: m.CORE_SOURCE if n.startswith('support/') else '1' * 64 for n in m.SOURCES})


def fixture(tmp_path):
    transfer = tmp_path / 'shared'; root = transfer / m.CORE_ROOT; root.mkdir(parents=True)
    for n, text in FIX['case_metadata'].items():
        path = root / n; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
    packet = transfer / m.CORE_PREFIX / 'unpacked' / m.CORE_PREFIX
    for host, cid in {'linux10609': 'clip-09-bank-0', 'linux10603': 'clip-08-bank-1'}.items():
        launcher = packet / 'launchers' / (host + '-' + cid); launcher.mkdir(parents=True)
        for n, text in FIX['launcher_files'][host].items(): (launcher / n).write_text(text)
        if cid in FIX['case_envelopes']:
            case = root / 'cases' / cid / 'attempt-01'; case.mkdir(parents=True)
            ops.write_record(case / 'case-envelope.json', FIX['case_envelopes'][cid])
    legacy = transfer / 'two-host-panel-20261026'; legacy.mkdir()
    old_pins = {}
    for n in ('two-host-panel-20261026/STOP.json', 'reviewed-panel-20261026/STOP.json', 'archery-timecarry-execution-20261027/STOP.json'):
        p = transfer / n; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('Generated immutable failure fixture: ' + n)
        old_pins[n] = m.digest(p)
    context = {'op': FIX['core_operation'], 'data': FIX['core_data'], 'root': root,
        'prior': prior, 'ops': ops, 'legacy': legacy, 'packet': packet, 'package': transfer / 'unpacked/matched-panel-20261024'}
    c = {'op': operation(), 'data': copy.deepcopy(FIX['input']), 'root': root, 'transfer': transfer,
        'prior': prior, 'ops': ops, 'core': core, 'context': context}
    return c, lambda: prior.check_preserved(transfer, old_pins)


def completed_receipt(c, existing=()):
    before = core.Registry(c['context']).status()
    return {'kind': m.KIND + '-receipt', 'operation_sha256': m.canonical(c['op']),
        'inputs_file_sha256': c['op']['inputs_file_sha256'],
        'created': sorted(m.TARGETS - set(existing)), 'preexisting_identical': sorted(existing),
        'records_readback': {n: b.decode() for n, b in m.targets(c['data']).items()},
        'fresh_preserved_files_sha256': dict(c['data']['preserved_files_sha256']),
        'status_before': before, 'status_after': dict(before, accepted_case_ids=m.AFTER),
        'saved_cases_reverified': 1, 'time_before': m.time_readback(c['data']), 'time_after': m.time_readback(c['data']),
        'historical_STOP_records_unchanged': True, 'models_started': False, 'test_suite_started': False,
        'new_GPU_dispatch_authorized': False, 'additional_attempt_authorized': False, 'research_admitted': False}


@pytest.mark.parametrize('key,value', [
    ('host', 'linux10603'), ('coordinator_namespace', 'another-root'), ('case_ids', ['clip-08-bank-1']),
    ('attempt', 'attempt-02'), ('accepted_after_sync', m.ORDER[:20]), ('preserved_file_count', 588),
    ('max_seconds', 4800), ('metadata_only', False), ('models_started', True), ('test_suite_started', True),
    ('new_GPU_dispatch_authorized', True), ('additional_attempt_authorized', True),
    ('historical_STOP_records_unchanged', False), ('automatic_queue', True), ('research_admitted', True),
    ('core_module_sha256', '0' * 64), ('actual_activation_receipt_file_sha256', '0' * 64),
    ('previous_consumed_history_seconds', 0), ('original_attempt_charge_upper_seconds', {'clip-09-bank-0': 0})])
def test_scope_or_authority_drift_is_rejected(key, value):
    op = operation(); op[key] = value
    with pytest.raises(ValueError): m.validate_operation(op, m.canonical(op))


def test_external_pin_and_complete_source_inventory_are_required():
    op = operation(); m.validate_operation(op, m.canonical(op))
    with pytest.raises(ValueError): m.validate_operation(op, '0' * 64)
    op['sources_sha256'].pop(m.SOURCES[0])
    with pytest.raises(ValueError): m.validate_operation(op, m.canonical(op))


@pytest.mark.parametrize('change', ['marker-text', 'marker-pin', 'marker-kind', 'history-drop', 'archive-pin', 'outcome-pin', 'unsafe-path', 'precredited-review', 'clock-reset', 'old-time-reset'])
def test_rewritten_mixed_or_unpinned_originals_are_rejected(change):
    d = copy.deepcopy(FIX['input']); cid = 'clip-09-bank-0'; pins = d['preserved_files_sha256']
    if change == 'marker-text': d['reviews'][cid]['text'] += ' '
    elif change == 'marker-pin': d['reviews'][cid]['sha256'] = '0' * 64
    elif change == 'marker-kind': d['reviews'][cid]['text'] = d['reviews'][cid]['text'].replace('reviewed-seventeen-pair', 'reviewed-fifteen-pair')
    elif change == 'history-drop': pins.pop(next(iter(pins)))
    elif change == 'archive-pin': pins[m.CORE_ROOT + '/cases/' + cid + '/attempt-01/continuation-evidence.tar.gz'] = '0' * 64
    elif change == 'outcome-pin': pins[m.CORE_ROOT + '/outcomes/' + cid + '.json'] = '0' * 64
    elif change == 'unsafe-path': pins['../escape'] = pins.pop(next(iter(pins)))
    elif change == 'precredited-review': pins[m.CORE_ROOT + '/reviews/' + cid + '.json'] = pins.pop(next(iter(pins)))
    elif change == 'clock-reset': d['clocks'][cid]['conservative_charge_upper_seconds'] = 0
    else: d['previous_consumed_history']['previous_consumed_history']['total_charged_seconds'] = 2599
    with pytest.raises(ValueError): m.validate_input(d)


def test_lost_acknowledgment_preserves_original_marker_and_idempotent_retry(tmp_path, monkeypatch):
    c, preserve = fixture(tmp_path); values = m.targets(c['data'])
    old = {p: p.read_bytes() for p in c['transfer'].rglob('*') if p.is_file()}
    original = prior.exclusive_bytes
    def interrupted(path, body):
        original(path, body); raise RuntimeError('Generated lost acknowledgment after the atomic append')
    with monkeypatch.context() as scope:
        scope.setattr(prior, 'exclusive_bytes', interrupted)
        with pytest.raises(RuntimeError): m.publish(c['root'], values, prior, preserve)
    name = next(iter(values)); assert (c['root'] / name).read_bytes() == values[name]
    m.check_state(core.Registry(c['context']).status(), m.TARGETS)
    assert m.publish(c['root'], values, prior, preserve) == ([], [name])
    assert old == {p: p.read_bytes() for p in old}


@pytest.mark.parametrize('change', ['conflict', 'symlink', 'directory'])
def test_any_conflicting_original_marker_prevents_publication(tmp_path, change):
    c, preserve = fixture(tmp_path); name = next(iter(m.TARGETS)); p = c['root'] / name; p.parent.mkdir()
    if change == 'conflict': p.write_text('{}')
    elif change == 'symlink': p.symlink_to(c['root'] / 'imports.json')
    else: p.mkdir()
    with pytest.raises(ValueError): m.publish(c['root'], m.targets(c['data']), prior, preserve)
    assert p.is_symlink() or p.is_dir() or p.read_text() == '{}'


@pytest.mark.parametrize('change', ['missing', 'changed', 'symlink'])
def test_old_failure_history_drift_prevents_publication(tmp_path, change):
    c, preserve = fixture(tmp_path); p = c['context']['legacy'] / 'STOP.json'
    if change == 'missing': p.unlink()
    elif change == 'changed': p.write_text('changed')
    else: p.unlink(); p.symlink_to(c['root'] / 'imports.json')
    with pytest.raises(ValueError): m.publish(c['root'], m.targets(c['data']), prior, preserve)
    assert all(not (c['root'] / n).exists() for n in m.TARGETS)


@pytest.mark.parametrize('change', ['readback', 'overlap', 'history', 'before-count', 'after-count', 'unchecked', 'model', 'attempt', 'time-before', 'time-after'])
def test_receipt_requires_both_originals_exact_states_and_original_time(tmp_path, change):
    c, _ = fixture(tmp_path); r = completed_receipt(c); assert m.verify_receipt(r, c['op'], c['data'])['verified']
    if change == 'readback': r['records_readback'][sorted(m.TARGETS)[0]] += ' '
    elif change == 'overlap': r['preexisting_identical'] = [sorted(m.TARGETS)[0]]
    elif change == 'history': r['fresh_preserved_files_sha256'].pop(next(iter(r['fresh_preserved_files_sha256'])))
    elif change == 'before-count': r['status_before']['accepted_case_ids'] = m.ORDER[:16]
    elif change == 'after-count': r['status_after']['accepted_case_ids'] = m.BEFORE
    elif change == 'unchecked': r['saved_cases_reverified'] = 0
    elif change == 'model': r['models_started'] = True
    elif change == 'attempt': r['additional_attempt_authorized'] = True
    elif change == 'time-before': r['time_before']['completed_original_attempts']['clip-09-bank-0']['conservative_charge_upper_seconds'] = 0
    else: r['time_after']['previous_consumed_history']['previous_consumed_history']['old_remainder_added'] = True
    with pytest.raises(ValueError): m.verify_receipt(r, c['op'], c['data'])


@pytest.mark.parametrize('host,name,value', [
    ('linux10609', 'started-epoch.txt', 'nan'), ('linux10603', 'started-epoch.txt', '1793167297.9180472'),
    ('linux10609', 'ended-utc.txt', '2026-10-28T06:48:45Z'), ('linux10603', 'exit-status.txt', '0')])
def test_each_original_clock_cannot_be_reset(tmp_path, host, name, value):
    c, _ = fixture(tmp_path); cid = 'clip-08-bank-1' if host == 'linux10603' else 'clip-09-bank-0'
    p = c['context']['packet'] / 'launchers' / (host + '-' + cid) / name; p.write_text(value)
    with pytest.raises((ValueError, ArithmeticError)): m.check_time(c)


def test_original_completed_and_refused_clocks_preserve_prior_consumed_history(tmp_path):
    c, _ = fixture(tmp_path); value = m.check_time(c)
    assert value['completed_original_attempts']['clip-09-bank-0']['conservative_charge_upper_seconds'] == 2951
    assert value['bowling_startup_accounting']['conservative_startup_charge_seconds'] == 206
    assert value['bowling_startup_accounting']['remaining_original_allowance_seconds'] == 4594
    assert value['previous_consumed_history']['previous_consumed_history']['total_charged_seconds'] == 4947
    assert value['previous_consumed_history']['completed_original_attempts']['clip-07-bank-1']['conservative_charge_upper_seconds'] == 2794
    assert value['previous_consumed_history']['completed_original_attempts']['clip-08-bank-0']['conservative_charge_upper_seconds'] == 2634


def test_apply_holds_real_locks_and_requires_both_saved_reviews(tmp_path, monkeypatch):
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
    assert len(calls) == 1 and r['created'] == sorted(m.TARGETS)
    assert m.verify_receipt(r, c['op'], c['data'])['accepted_cases'] == 18


@pytest.mark.parametrize('change', ['incomplete-review', 'concurrent-stop'])
def test_incomplete_saved_review_or_new_stop_leaves_reviews_absent(tmp_path, monkeypatch, change):
    c, _ = fixture(tmp_path); monkeypatch.setattr(m, 'load', lambda *a: c)
    monkeypatch.setattr(m, 'check_preserved', lambda c: dict(c['data']['preserved_files_sha256']))
    def saved(c, temporary):
        if change == 'concurrent-stop': ops.write_record(c['root'] / 'STOP.json', {'generated_stop': True})
        return 0 if change == 'incomplete-review' else 1
    monkeypatch.setattr(m, 'verify_saved', saved)
    with pytest.raises(ValueError): m.apply(None, m.canonical(c['op']), tmp_path, 'linux10609')
    assert all(not (c['root'] / n).exists() for n in m.TARGETS)


def test_exclusive_descriptors_support_real_NFS_record_lock_access(tmp_path, monkeypatch):
    import fcntl
    c, _ = fixture(tmp_path); monkeypatch.setattr(m, 'load', lambda *a: c)
    monkeypatch.setattr(m, 'check_preserved', lambda c: dict(c['data']['preserved_files_sha256']))
    monkeypatch.setattr(m, 'verify_saved', lambda *a: 1); monkeypatch.setattr(fcntl, 'flock', fcntl.lockf)
    r = m.apply(None, m.canonical(c['op']), tmp_path, 'linux10609')
    assert r['created'] == sorted(m.TARGETS)


@pytest.mark.parametrize('where', ['linux10609', 'linux10603', 'registry'])
def test_live_lease_or_registry_owner_blocks_all_appends(tmp_path, monkeypatch, where):
    c, _ = fixture(tmp_path); monkeypatch.setattr(m, 'load', lambda *a: c)
    monkeypatch.setattr(m, 'check_preserved', lambda c: dict(c['data']['preserved_files_sha256']))
    monkeypatch.setattr(m, 'verify_saved', lambda *a: 1)
    path = c['root'] / '.registry.lock' if where == 'registry' else c['context']['legacy'] / 'leases' / where
    path.parent.mkdir(parents=True, exist_ok=True)
    code = 'import fcntl,sys;f=open(sys.argv[1],"a+");fcntl.flock(f,fcntl.LOCK_EX);print("locked",flush=True);sys.stdin.read()'
    process = subprocess.Popen([sys.executable, '-B', '-c', code, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'locked'
        if where == 'registry':
            original = ops.file_lock
            monkeypatch.setattr(ops, 'file_lock', lambda p, **kw: original(p, wait_seconds=0))
        with pytest.raises(BlockingIOError): m.apply(None, m.canonical(c['op']), tmp_path, 'linux10609')
        assert all(not (c['root'] / n).exists() for n in m.TARGETS)
    finally:
        process.stdin.close(); process.wait(timeout=5); process.stdout.close()


def test_wrong_metadata_host_rejected_before_any_load(monkeypatch, tmp_path):
    def forbidden(*a): raise AssertionError('Wrong host must fail before load')
    monkeypatch.setattr(m, 'load', forbidden)
    with pytest.raises(ValueError): m.apply(None, '0' * 64, tmp_path, 'linux10603')


@pytest.mark.parametrize('change', ['missing', 'changed', 'symlink'])
def test_damaged_history_blocks_older_constructors_before_side_effects(tmp_path, monkeypatch, change):
    home = tmp_path / 'home'; transfer = home / m.BASE
    packet = transfer / m.PREFIX / 'unpacked' / m.PREFIX; packet.mkdir(parents=True)
    for name in m.SOURCES:
        source = ROOT / name if not name.startswith('support/') else CORE / 'models/reviewed_panel_final_pair.py'
        target = packet / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    (packet / 'INPUT.json').write_text(json.dumps(FIX['input'], sort_keys=True, indent=2) + '\n')
    source = Path(os.environ.get('PAIR_REVIEW_SYNC_TEST_ACTIVATION', str(REPO / 'runs/pretrained/diagnostic/bowling-handstand-pair-20261028/activation-session-20261028T173712Z.3GpVt9/receipt.json')))
    shutil.copyfile(source, packet / 'activation-receipt.json')
    op = dict(m.policy(), inputs_file_sha256=m.digest(packet / 'INPUT.json'), sources_sha256={n: m.digest(packet / n) for n in m.SOURCES})
    (packet / 'SYNC.json').write_text(json.dumps(op))
    first = transfer / next(iter(FIX['input']['preserved_files_sha256'])); first.parent.mkdir(parents=True, exist_ok=True)
    if change == 'changed': first.write_text('Changed original')
    elif change == 'symlink': first.symlink_to(packet / 'INPUT.json')
    def forbidden(*a): raise AssertionError('History must be checked before an older constructor runs')
    monkeypatch.setattr(m, 'module', lambda *a: SimpleNamespace(boundary=lambda *a: {'prior': prior}, load=forbidden))
    before = {p: p.read_bytes() for p in transfer.rglob('*') if p.is_file() and not p.is_symlink()}
    with pytest.raises(ValueError): m.load(packet, m.canonical(op), home)
    assert before == {p: p.read_bytes() for p in before}
    assert all(not (transfer / m.CORE_ROOT / n).exists() for n in m.TARGETS)


def test_returned_clock_and_policy_objects_cannot_mutate_frozen_history():
    data = copy.deepcopy(FIX['input']); original = copy.deepcopy(data)
    expected_policy = copy.deepcopy(m.policy()); expected_history = copy.deepcopy(m.PRIOR_TIME)
    readback = m.time_readback(data)
    readback['completed_original_attempts']['clip-09-bank-0']['conservative_charge_upper_seconds'] = 0
    readback['previous_consumed_history']['previous_consumed_history']['old_remainder_added'] = True
    policy = m.policy(); policy['original_attempt_charge_upper_seconds']['clip-09-bank-0'] = 0
    policy['original_review_markers_file_sha256'].clear()
    assert data == original and m.PRIOR_TIME == expected_history and m.policy() == expected_policy
    m.validate_input(data)


@pytest.mark.parametrize('namespace', ('bowling-handstand-execution-20261028', 'two-host-panel-20261026', 'archery-timecarry-execution-20261027'))
@pytest.mark.parametrize('relative', ('claims/clip-08-bank-1', 'cases/clip-08-bank-1', 'outcomes/clip-08-bank-1.json', 'exports/clip-09-bank-1.json', 'reviews/clip-09-bank-1.json'))
def test_pending_final_case_artifacts_block_registration(tmp_path, namespace, relative):
    c, _ = fixture(tmp_path); p = c['transfer'] / namespace / relative
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text('Generated previous pending state')
    with pytest.raises(ValueError): m.check_bowling(c)


@pytest.mark.parametrize('field,value', [('conservative_startup_charge_seconds', 0), ('remaining_original_allowance_seconds', 4800), ('numeric_attempts_started', 1), ('additional_allowance_seconds', 4800), ('operation_sha256', '0'*64)])
def test_bowling_accounting_cannot_reset_or_add_authority(field,value):
    d = copy.deepcopy(FIX['input']); d['bowling_startup_accounting'][field] = value
    with pytest.raises(ValueError): m.validate_input(d)


def test_bowling_startup_files_cannot_be_replaced_with_same_count_unrelated_pin():
    d = copy.deepcopy(FIX['input']); pins = d['preserved_files_sha256']
    name = m.CORE_PREFIX+'/unpacked/'+m.CORE_PREFIX+'/launchers/linux10603-clip-08-bank-1/cpu-new.xml'
    pins['unrelated-generated-file'] = pins.pop(name)
    with pytest.raises(ValueError): m.validate_input(d)


def test_eighteen_case_set_is_gapped_and_preserves_original_seventeen():
    assert len(m.AFTER) == 18 and set(m.AFTER) == set(m.BEFORE) | {'clip-09-bank-0'}
    assert 'clip-08-bank-1' not in m.AFTER and 'clip-09-bank-1' not in m.AFTER
    assert m.AFTER != m.ORDER[:18]
