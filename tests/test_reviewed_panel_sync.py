"""CPU-only publication, preservation and real-lock checks for the two reviewed markers."""
import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/Users/kaiwendu/Documents/GitHub/V-JEPA-Robust-Overfitting')
CORE = Path(os.environ.get('REVIEWED_SYNC_TEST_CORE', str(REPO / 'runs/pretrained/diagnostic/reviewed-continuation-20261026/extracted/reviewed-continuation-20261026')))
REFERENCE = Path(os.environ.get('REVIEWED_SYNC_TEST_REFERENCE', str(REPO / 'runs/pretrained/diagnostic/two-host-panel-20261026/extracted/two-host-execution-20261026')))

def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

m = module('reviewed_sync_tests', ROOT / 'models/reviewed_panel_sync.py')
core = module('sync_prior_core', ROOT / 'support/reviewed_panel_continuation.py')
prior = module('sync_prior_record_tools', CORE / 'support/reviewed_panel.py')
ops = module('sync_original_lock_tools', REFERENCE / 'models/two_host_panel.py')
FIX = json.loads((ROOT / 'tests/fixtures/reviewed_panel_sync_metadata.json').read_text())


def operation():
    return {'kind': m.KIND, 'host': 'linux10609', 'core_operation_sha256': m.CORE_PIN,
        'core_module_sha256': m.CORE_SOURCE, 'actual_activation_receipt_file_sha256': m.ACTIVATION_RECEIPT,
        'coordinator_namespace': m.CORE_ROOT, 'case_ids': list(m.CASES.values()),
        'accepted_after_sync': m.ORDER[:13], 'preserved_file_count': 275,
        'max_seconds': 300, 'kill_grace_seconds': 15, 'metadata_only': True,
        'models_started': False, 'test_suite_started': False, 'new_GPU_dispatch_authorized': False,
        'historical_STOP_records_unchanged': True, 'automatic_queue': False, 'research_admitted': False,
        'inputs_file_sha256': '1' * 64,
        'sources_sha256': {n: m.CORE_SOURCE if n.startswith('support/') else '1' * 64 for n in m.SOURCES}}


def fixture(tmp_path):
    transfer = tmp_path / 'shared'
    root = transfer / m.CORE_ROOT
    root.mkdir(parents=True)
    for name, text in FIX['activation_record_texts'].items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    for records in FIX['case_metadata'].values():
        for name, text in records.items():
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    old = transfer / 'two-host-panel-20261026'
    old.mkdir()
    stop = old / 'STOP.json'
    stop.write_text('Generated CPU fixture: preserve historical stop.\n')
    old_pins = {'two-host-panel-20261026/STOP.json': m.digest(stop)}
    context = {'op': FIX['core_operation'], 'data': FIX['core_data'], 'root': root,
        'prior': prior, 'ops': ops, 'legacy': old}
    c = {'op': operation(), 'data': copy.deepcopy(FIX['input']), 'root': root, 'transfer': transfer,
        'prior': prior, 'ops': ops, 'core': core, 'context': context}
    def preserved():
        return prior.check_preserved(transfer, old_pins)
    return c, preserved


def test_genuine_scope_contains_only_the_two_original_markers_and_275_pins():
    m.validate_input(FIX['input'])
    assert len(FIX['input']['preserved_files_sha256']) == 275
    assert set(m.targets(FIX['input'])) == m.TARGETS
    for cid, row in FIX['input']['reviews'].items():
        assert row['sha256'] == m.MARKER_PINS[cid]
        assert m.marker_payload(row)['case_id'] == cid


@pytest.mark.parametrize('key,value', [
    ('host', 'linux10603'), ('coordinator_namespace', 'two-host-panel-20261026'),
    ('case_ids', ['clip-04-bank-1']), ('accepted_after_sync', m.ORDER[:14]),
    ('preserved_file_count', 148), ('max_seconds', 4800), ('kill_grace_seconds', 16),
    ('metadata_only', False), ('models_started', True), ('test_suite_started', True),
    ('new_GPU_dispatch_authorized', True), ('historical_STOP_records_unchanged', False),
    ('automatic_queue', True), ('research_admitted', True), ('core_module_sha256', '0' * 64),
    ('actual_activation_receipt_file_sha256', '0' * 64)])
def test_changed_scope_or_model_authority_is_rejected(key, value):
    op = operation()
    op[key] = value
    with pytest.raises(ValueError):
        m.validate_operation(op, m.canonical(op))


def test_external_source_pin_and_full_inventory_required():
    op = operation()
    m.validate_operation(op, m.canonical(op))
    with pytest.raises(ValueError): m.validate_operation(op, '0' * 64)
    op['sources_sha256'].pop(m.SOURCES[0])
    with pytest.raises(ValueError): m.validate_operation(op, m.canonical(op))


@pytest.mark.parametrize('change', ['marker-text', 'marker-pin', 'drop-case', 'add-case', 'missing-preserved', 'archive-pin', 'outcome-pin'])
def test_changed_marker_or_preserved_original_is_rejected(change):
    data = copy.deepcopy(FIX['input'])
    cid = next(iter(data['reviews']))
    if change == 'marker-text': data['reviews'][cid]['text'] += ' '
    elif change == 'marker-pin': data['reviews'][cid]['sha256'] = '0' * 64
    elif change == 'drop-case': data['reviews'].pop(cid)
    elif change == 'add-case': data['reviews']['clip-09-bank-1'] = data['reviews'][cid]
    elif change == 'missing-preserved': data['preserved_files_sha256'].pop(next(iter(data['preserved_files_sha256'])))
    elif change == 'archive-pin': data['preserved_files_sha256'][m.CORE_ROOT + '/cases/' + cid + '/attempt-01/continuation-evidence.tar.gz'] = '0' * 64
    else: data['preserved_files_sha256'][m.CORE_ROOT + '/outcomes/' + cid + '.json'] = '0' * 64
    with pytest.raises(ValueError): m.validate_input(data)


def test_two_publications_preserve_existing_files_and_are_idempotent(tmp_path):
    c, preserve = fixture(tmp_path)
    values = m.targets(c['data'])
    before = {str(p): p.read_bytes() for p in c['root'].rglob('*') if p.is_file()}
    state = core.Registry(c['context']).status()
    m.check_state(state, [])
    created, existing = m.publish(c['root'], values, prior, preserve)
    assert set(created) == m.TARGETS and existing == []
    m.check_state(core.Registry(c['context']).status(), m.TARGETS)
    assert len(core.Registry(c['context']).status()['accepted_case_ids']) == 13
    again, existing = m.publish(c['root'], values, prior, preserve)
    assert again == [] and set(existing) == m.TARGETS
    assert before == {name: Path(name).read_bytes() for name in before}


def test_interrupted_sync_only_completes_identical_missing_review(tmp_path, monkeypatch):
    c, preserve = fixture(tmp_path)
    original = prior.exclusive_bytes
    calls = []
    def fail_after_one(path, data):
        if calls: raise RuntimeError('Generated CPU interruption')
        original(path, data)
        calls.append(path)
    monkeypatch.setattr(prior, 'exclusive_bytes', fail_after_one)
    with pytest.raises(RuntimeError): m.publish(c['root'], m.targets(c['data']), prior, preserve)
    monkeypatch.setattr(prior, 'exclusive_bytes', original)
    created, existing = m.publish(c['root'], m.targets(c['data']), prior, preserve)
    assert len(created) == len(existing) == 1
    m.check_state(core.Registry(c['context']).status(), m.TARGETS)


@pytest.mark.parametrize('target', sorted(m.TARGETS))
def test_conflicting_review_stops_before_any_publication(tmp_path, target):
    c, preserve = fixture(tmp_path)
    path = c['root'] / target
    path.parent.mkdir()
    path.write_text('{}')
    before = {str(p): p.read_bytes() for p in c['root'].rglob('*') if p.is_file()}
    with pytest.raises(ValueError): m.publish(c['root'], m.targets(c['data']), prior, preserve)
    assert before == {str(p): p.read_bytes() for p in c['root'].rglob('*') if p.is_file()}


@pytest.mark.parametrize('change', ['missing', 'changed', 'symlink'])
def test_historical_stop_drift_prevents_review_writes(tmp_path, change):
    c, preserve = fixture(tmp_path)
    stop = c['context']['legacy'] / 'STOP.json'
    if change == 'missing': stop.unlink()
    elif change == 'changed': stop.write_text('changed')
    else:
        stop.unlink()
        stop.symlink_to(c['root'] / 'genesis.json')
    with pytest.raises(ValueError): m.publish(c['root'], m.targets(c['data']), prior, preserve)
    assert not (c['root'] / 'reviews').exists()


def test_constructor_and_sync_use_real_locks_without_a_nested_constructor_deadlock(tmp_path, monkeypatch):
    c, _ = fixture(tmp_path)
    monkeypatch.setattr(m, 'load', lambda *a: c)
    monkeypatch.setattr(m, 'check_preserved', lambda context: dict(context['data']['preserved_files_sha256']))
    called = []
    def saved(context, temporary):
        with pytest.raises(BlockingIOError):
            with ops.file_lock(context['root'] / '.registry.lock', wait_seconds=0): pass
        assert core.Registry(context['context']).status()['accepted_case_ids'] == m.ORDER[:11]
        called.append(temporary)
        return 2
    monkeypatch.setattr(m, 'verify_saved', saved)
    receipt = m.apply(None, m.canonical(c['op']), tmp_path, 'linux10609')
    assert len(called) == 1
    assert m.verify_receipt(receipt, c['op'], c['data'])['accepted_cases'] == 13
    assert set(receipt['created']) == m.TARGETS


@pytest.mark.parametrize('change', ['new-stop', 'new-failure', 'wrong-claimed', 'wrong-computed', 'unreviewed-acceptance'])
def test_unknown_state_never_qualifies_for_this_sync(tmp_path, change):
    c, _ = fixture(tmp_path)
    state = core.Registry(c['context']).status()
    if change == 'new-stop': state['stop_dispatch'] = True
    elif change == 'new-failure': state['failed_case_ids'] = [next(iter(m.CASES.values()))]
    elif change == 'wrong-claimed': state['claimed_case_ids'] = []
    elif change == 'wrong-computed': state['computed_case_ids'] = []
    else: state['accepted_case_ids'] = m.ORDER[:12]
    with pytest.raises(ValueError): m.check_state(state, [])


def completed_receipt(c):
    return {'kind': m.KIND + '-receipt', 'operation_sha256': m.canonical(c['op']),
        'inputs_file_sha256': c['op']['inputs_file_sha256'], 'created': sorted(m.TARGETS), 'preexisting_identical': [],
        'records_readback': {n: b.decode() for n, b in m.targets(c['data']).items()},
        'fresh_preserved_files_sha256': dict(c['data']['preserved_files_sha256']),
        'status_before': core.Registry(c['context']).status(),
        'status_after': dict(core.Registry(c['context']).status(), accepted_case_ids=m.ORDER[:13]),
        'saved_cases_reverified': 2, 'historical_STOP_records_unchanged': True,
        'models_started': False, 'test_suite_started': False, 'new_GPU_dispatch_authorized': False, 'research_admitted': False}


@pytest.mark.parametrize('change', ['readback', 'created-overlap', 'pin', 'before-count', 'after-count', 'unchecked-case', 'model', 'dispatch'])
def test_receipt_must_prove_exact_records_states_and_fresh_preservation(tmp_path, change):
    c, _ = fixture(tmp_path)
    r = completed_receipt(c)
    assert m.verify_receipt(r, c['op'], c['data'])['verified']
    if change == 'readback': r['records_readback'][next(iter(m.TARGETS))] = '{}'
    elif change == 'created-overlap': r['preexisting_identical'] = [r['created'][0]]
    elif change == 'pin': r['fresh_preserved_files_sha256'].pop(next(iter(r['fresh_preserved_files_sha256'])))
    elif change == 'before-count': r['status_before']['accepted_case_ids'] = m.ORDER[:12]
    elif change == 'after-count': r['status_after']['accepted_case_ids'] = m.ORDER[:12]
    elif change == 'unchecked-case': r['saved_cases_reverified'] = 1
    elif change == 'model': r['models_started'] = True
    else: r['new_GPU_dispatch_authorized'] = True
    with pytest.raises(ValueError): m.verify_receipt(r, c['op'], c['data'])
