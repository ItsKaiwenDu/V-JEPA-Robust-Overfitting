"""CPU checks for the new ledger, single-use claims, original limits and source identities."""
from contextlib import contextmanager
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

m = module('reviewed_continuation_tests', ROOT / 'models/reviewed_panel_final_pair.py')
SUPPORT = Path(os.environ.get('REVIEWED_FINAL_PAIR_SUPPORT', str(ROOT / 'support')))
prior = module('reviewed_prior_tests', SUPPORT / 'core-packet/support/core-packet/support/core-packet/support/core-packet/support/core-packet/support/reviewed_panel.py')
sync = module('reviewed_sync_tests', SUPPORT / 'yoyo_bowling_review_sync.py')
REFERENCE = Path(os.environ.get('REVIEWED_TEST_REFERENCE', str(ROOT)))
ops = module('reviewed_reference_tests', REFERENCE / 'models/two_host_panel.py')
FIX = json.loads((ROOT / 'tests/fixtures/reviewed_final_pair_metadata.json').read_text())


def operation():
    return {'kind': m.KIND, 'sync_operation_sha256': m.SYNC_OP,
        'sync_receipt_file_sha256': m.PRIOR_RECEIPT, 'reference_revision_sha256': m.RP,
        'reference_implementation_sha256': m.MP, 'imported_case_ids': m.ACCEPTED,
        'first_requests': dict(m.TARGETS), 'coordinator_namespace': m.TARGET,
        'historical_STOP_sha256': m.STOP, 'preserve_all_three_historical_STOP_records': True,
        'numeric_settings_changed': False, 'case_seconds': 4800, 'kill_grace_seconds': 15,
        'matrix_seconds': 3600, 'replay_seconds': 900, 'parent_seconds': 4620, 'max_attempts': 1,
        'additional_allowance_seconds': 0, 'automatic_GPU_retries': 0, 'max_parallel_cases': 2,
        'one_case_per_host': True, 'automatic_queue': False, 'paid_compute': False, 'research_admitted': False,
        'preserved_file_count': m.PRESERVED,
        'sources_sha256': {n: (m.SYNC_SOURCE if n == 'support/yoyo_bowling_review_sync.py' else m.CORE_SOURCE if n == 'support/core-packet/models/reviewed_panel_next_pair.py' else '1' * 64) for n in m.SOURCES}}


def context(tmp_path):
    transfer = tmp_path / 'shared'
    transfer.mkdir()
    root = transfer / m.TARGET
    legacy = transfer / m.LEGACY
    legacy.mkdir()
    package = transfer / m.PACKAGE
    package.mkdir(parents=True)
    return {'op': operation(), 'prior': prior, 'ops': ops, 'root': root, 'legacy': legacy,
        'package': package, 'transfer': transfer, 'data': copy.deepcopy(FIX['inputs']),
        'sync_op': FIX['sync_operation'], 'sync_input': FIX['sync_input'], 'sync_receipt': FIX['receipt'],
        'previous': {'root': transfer / m.CORE_TARGET},
        'revision': FIX['previous_fixture']['reference_revision'], 'manifest': FIX['previous_fixture']['reference_manifest']}


@pytest.fixture
def active(tmp_path, monkeypatch):
    c = context(tmp_path)
    observed = m.ancestor_pins(c)
    monkeypatch.setattr(m, 'check_ancestors', lambda context: dict(observed))
    receipt = m.activate(c, 'linux10609')
    return c, receipt


def test_actual_seventeen_case_sync_is_the_only_imported_boundary():
    sync.validate_operation(FIX['sync_operation'], m.SYNC_OP)
    assert sync.verify_receipt(FIX['receipt'], FIX['sync_operation'], FIX['sync_input'])['verified']
    assert len(FIX['receipt']['status_after']['accepted_case_ids']) == 17
    assert len(FIX['inputs']['preserved_files_sha256']) == 774
    index = FIX['inputs']['case_index']
    assert index['clip-04-bank-1']['origin'] == 'explicit-attended-recovery'
    assert index['clip-05-bank-0']['origin'] == 'two-host-revision'
    for cid, row in FIX['old_data']['case_index'].items(): assert index[cid] == row
    recovered = index['clip-06-bank-1']
    assert recovered['origin'] == 'explicit-Archery-additional-attempt'
    assert recovered['consumed_grant_id'] == sync.PRIOR_TIME['consumed_grant_id']
    assert recovered['authorization_file_sha256'] == sync.PRIOR_TIME['authorization_file_sha256']
    assert recovered['review_file_sha256'] == 'ba9adb7e31588dd85a697436c637a3c602997fb4f6c701957ee775af0c9f401a'
    assert recovered['execution_sha256'] == '87bd519d87e4aae442d4837dac6f165d9459f4fcb2903328b928488c69db4760'


@pytest.mark.parametrize('key,value', [
    ('first_requests', {'linux10603': 'clip-04-bank-1'}), ('coordinator_namespace', m.LEGACY),
    ('case_seconds', 4801), ('kill_grace_seconds', 16), ('matrix_seconds', 3601), ('replay_seconds', 901),
    ('parent_seconds', 4621), ('max_attempts', 2), ('additional_allowance_seconds', 4800),
    ('automatic_GPU_retries', 1), ('max_parallel_cases', 3), ('one_case_per_host', False),
    ('automatic_queue', True), ('paid_compute', True), ('research_admitted', True),
    ('numeric_settings_changed', True), ('preserve_all_three_historical_STOP_records', False),
    ('historical_STOP_sha256', '0' * 64), ('sync_receipt_file_sha256', '0' * 64),
    ('reference_implementation_sha256', '0' * 64), ('imported_case_ids', m.ACCEPTED[:-1])])
def test_policy_and_scope_changes_rejected(key, value):
    op = operation()
    op[key] = value
    with pytest.raises(ValueError):
        m.validate_operation(op, m.canonical(op))


def test_external_operation_and_complete_source_pins_required():
    op = operation()
    m.validate_operation(op, m.canonical(op))
    with pytest.raises(ValueError):
        m.validate_operation(op, '0' * 64)
    op['sources_sha256'].pop(m.SOURCES[0])
    with pytest.raises(ValueError):
        m.validate_operation(op, m.canonical(op))


@pytest.mark.parametrize('host', list(m.TARGETS))
def test_new_request_has_no_recovery_or_extra_allowance(host):
    op = operation()
    q = m.request(op, host)
    assert m.validate_request(q, m.canonical(q), op, host) == m.TARGETS[host]
    assert q['case_id'] not in m.ACCEPTED
    assert q['max_attempts'] == 1 and q['additional_allowance_seconds'] == 0
    q['case_id'] = 'clip-04-bank-1'
    with pytest.raises(ValueError):
        m.validate_request(q, m.canonical(q), op, host)


@pytest.mark.parametrize('started,now', [(None, 1), (float('nan'), 1), (True, 1), (100, 99), (100, 4900), (0, float('inf'))])
def test_time_cannot_be_unknown_exhausted_or_reset(started, now):
    with pytest.raises((ValueError, TimeoutError)):
        m.elapsed(started, now)


def test_cpu_setup_counts_against_the_original_case_limit():
    assert m.elapsed(100, 350) == 250
    assert min(4620, m.LIMIT - m.elapsed(100, 350) - 120) == 4430


def test_five_record_activation_keeps_all_seventeen_original_identities(active):
    c, receipt = active
    assert len(receipt['created']) == 5 and receipt['created'][-1] == 'genesis.json'
    assert len(receipt['preserved_files_sha256']) == 774
    assert m.verify_activation(receipt, c['op'], c['data'], prior)['verified']
    imports = ops.read_record(c['root'] / 'imports.json')
    assert imports['case_origins'] == FIX['inputs']['case_index']
    assert m.Registry(c).status()['accepted_case_ids'] == m.ACCEPTED
    assert not (c['root'] / 'STOP.json').exists()
    again = m.activate(c, 'linux10609')
    assert again['created'] == [] and len(again['preexisting_identical']) == 5


@pytest.mark.parametrize('change', ['receipt', 'record', 'preserved', 'created', 'GPU', 'cases'])
def test_activation_receipt_cannot_claim_changed_state(active, change):
    c, r = active
    r = copy.deepcopy(r)
    if change == 'receipt': r['prior_receipt_file_sha256'] = '0' * 64
    elif change == 'record': r['record_texts']['resolution.json'] = '{}'
    elif change == 'preserved': r['preserved_files_sha256'].pop(next(iter(r['preserved_files_sha256'])))
    elif change == 'created': r['created'].append(r['created'][0])
    elif change == 'GPU': r['models_started'] = True
    elif change == 'cases': r['accepted_case_ids'] = m.ACCEPTED[:-1]
    with pytest.raises(ValueError):
        m.verify_activation(r, c['op'], c['data'], prior)


@pytest.mark.parametrize('change', ['conflict', 'STOP', 'claim', 'symlink'])
def test_activation_never_overwrites_conflicts_or_resets_attempts(active, change):
    c, _ = active
    root = c['root']
    if change == 'conflict': (root / 'resolution.json').write_text('{}')
    elif change == 'STOP': ops.write_record(root / 'STOP.json', {'failure': 'new'})
    elif change == 'claim': (root / 'claims/clip-07-bank-1').mkdir(parents=True)
    else:
        (root / 'resolution.json').unlink()
        (root / 'resolution.json').symlink_to(root / 'imports.json')
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        m.activate(c, 'linux10609')
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_registry_does_not_nest_its_constructor_inside_a_file_lock(active):
    c, _ = active
    with ops.file_lock(c['root'] / '.registry.lock', wait_seconds=0):
        assert m.Registry(c).status()['accepted_case_ids'] == m.ACCEPTED
    r = m.Registry(c)
    r.claim('linux10609')
    with pytest.raises(ValueError):
        r.claim('linux10609')
    assert r.status()['claimed_case_ids'] == ['clip-09-bank-0']


def test_distinct_claims_and_failure_stop_preserve_running_peer(active):
    c, _ = active
    r = m.Registry(c)
    with ops.file_lock(c['legacy'] / 'leases/linux10609', wait_seconds=0):
        with ops.file_lock(c['legacy'] / 'leases/linux10603', wait_seconds=0):
            r.claim('linux10609')
            r.claim('linux10603')
    r.fail('linux10603', 'foreign CUDA process', 20)
    state = r.status()
    assert state['stop_dispatch'] and state['failed_case_ids'] == ['clip-08-bank-1']
    assert state['claimed_case_ids'] == ['clip-08-bank-1', 'clip-09-bank-0']
    assert state['accepted_case_ids'] == m.ACCEPTED
    with pytest.raises(ValueError): r.claim('linux10609')


@pytest.mark.parametrize('place', ['claims', 'cases', 'outcomes', 'exports', 'reviews', 'old-v1'])
def test_old_or_unreviewed_pending_case_state_cannot_be_reset(active, place):
    c, _ = active
    cid = m.TARGETS['linux10609']
    if place == 'old-v1':
        (c['package'] / 'runs/panel.qIXeDn/cases' / cid).mkdir(parents=True)
    elif place in ('claims', 'cases'):
        (c['legacy'] / place / cid).mkdir(parents=True)
    else:
        path = c['legacy'] / place / (cid + '.json')
        path.parent.mkdir(parents=True)
        path.write_text('{}')
    with pytest.raises(ValueError, match='Prior pending-case'):
        m.Registry(c).claim('linux10609')
    assert not (c['root'] / 'claims').exists()


def test_global_old_host_lease_is_shared_between_both_namespaces(active):
    c, _ = active
    with ops.file_lock(c['legacy'] / 'leases/linux10609', wait_seconds=0):
        command = [sys.executable, '-B', '-c',
            "import fcntl,sys;f=open(sys.argv[1],'a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)",
            str(c['legacy'] / 'leases/linux10609')]
        result = subprocess.run(command, capture_output=True)
        assert result.returncode != 0 and b'BlockingIOError' in result.stderr


@pytest.mark.parametrize('host', list(m.TARGETS))
def test_new_numeric_identity_binds_actual_wrapper_and_retains_old_reference(active, host):
    c, _ = active
    old = ops.execution_identity(c['revision'], c['manifest'], host)
    new = m.execution_identity(c, host)
    assert new['reference_execution_sha256'] == m.canonical(old)
    assert new['continuation_operation_sha256'] == m.canonical(c['op'])
    assert new['sync_receipt_file_sha256'] == m.PRIOR_RECEIPT
    assert new['checkpoint_sha256'] == old['checkpoint_sha256']
    assert new['expected_runtime'] == old['expected_runtime']
    assert new['expected_weights'] == old['expected_weights']
    assert m.canonical(new) != m.canonical(old)


def test_original_strict_GPU_guard_rejects_any_non_self_compute_PID():
    for host in m.TARGETS:
        ops.validate_hardware(host, ops.HOSTS[host], '123\n', allowed_pid=123)
        with pytest.raises(ValueError, match='Another CUDA compute process'):
            ops.validate_hardware(host, ops.HOSTS[host], '123\n1162024\n', allowed_pid=123)


def test_rejected_GPU_boundary_is_retained_without_a_valid_observation(active, tmp_path, monkeypatch):
    c, _ = active
    def reject(*args, **kwargs):
        raise ValueError('Another CUDA compute process is active.')
    monkeypatch.setattr(c['ops'], 'hardware_gate', reject)
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout='Generated read-only GPU snapshot', stderr=''))
    with pytest.raises(ValueError):
        m.strict_gate(c, 'linux10603', tmp_path, 'matrix-after', allow_self=True)
    event = ops.read_record(tmp_path / 'hardware-matrix-after.json')
    assert event['passed'] is False and event['strict_guard_source_unchanged']
    assert not (tmp_path / 'matrix/observation.json').exists()


@pytest.mark.parametrize('change', ['unknown', 'claim-symlink', 'outcome-symlink', 'metadata'])
def test_unknown_or_redirected_registry_state_stops_before_claim(active, change):
    c, _ = active
    root = c['root']
    if change == 'unknown':
        (root / 'unknown-grant').mkdir()
    elif change == 'metadata':
        (root / 'imports.json').write_text('{}')
    else:
        if change == 'claim-symlink': path = root / 'claims/clip-07-bank-1/claim.json'
        else: path = root / 'outcomes/clip-07-bank-1.json'
        path.parent.mkdir(parents=True)
        path.symlink_to(root / 'genesis.json')
    with pytest.raises(ValueError):
        m.Registry(c).claim('linux10609')


def test_complete_outcome_alone_never_confers_independent_acceptance(active):
    c, _ = active
    registry = m.Registry(c)
    registry.claim('linux10609')
    ops.write_record(c['root'] / 'outcomes/clip-09-bank-0.json', {
        'kind': m.KIND + '-outcome', 'operation_sha256': m.canonical(c['op']),
        'case_id': 'clip-09-bank-0', 'host': 'linux10609', 'complete': True, 'research_admitted': False})
    state = registry.status()
    assert state['computed_case_ids'] == ['clip-09-bank-0']
    assert state['accepted_case_ids'] == m.ACCEPTED


def test_full_run_failure_consumes_once_stops_peer_and_counts_setup(active, monkeypatch):
    c, _ = active
    monkeypatch.setattr(m.platform, 'node', lambda: 'linux10609')
    monkeypatch.setattr(c['ops'], 'hardware_gate', lambda *a, **kw: ops.HOSTS['linux10609'])
    captured = []
    def reject(context, host, directory, budget):
        captured.append((host, budget))
        raise RuntimeError('synthetic CPU failure before model work')
    monkeypatch.setattr(m, 'numerical_case', reject)
    with pytest.raises(RuntimeError, match='synthetic CPU failure'):
        m.run(c, 'linux10609', m.time.time() - 250)
    assert len(captured) == 1 and 4420 < captured[0][1] <= 4430
    state = m.Registry(c).status()
    assert state['stop_dispatch'] and state['failed_case_ids'] == ['clip-09-bank-0']
    assert state['accepted_case_ids'] == m.ACCEPTED
    with pytest.raises(ValueError): m.Registry(c).claim('linux10603')
    with pytest.raises(ValueError): m.run(c, 'linux10609', m.time.time())
    assert len(captured) == 1


def test_worker_claim_role_is_one_use_even_if_boundary_rejects(active, monkeypatch):
    c, _ = active
    host = 'linux10609'
    cid = m.TARGETS[host]
    m.Registry(c).claim(host)
    directory = c['root'] / 'cases' / cid / 'attempt-01'
    directory.mkdir(parents=True)
    ops.write_record(directory / 'case-envelope.json', {
        'operation_sha256': m.canonical(c['op']), 'active_role': 'matrix', 'complete': False})
    monkeypatch.setattr(m.platform, 'node', lambda: host)
    def reject(*a, **kw): raise RuntimeError('CPU boundary sentinel')
    monkeypatch.setattr(m, 'strict_gate', reject)
    with pytest.raises(ValueError): m.worker(c, host, 'replay')
    with pytest.raises(RuntimeError, match='CPU boundary sentinel'): m.worker(c, host, 'matrix')
    with pytest.raises(FileExistsError): m.worker(c, host, 'matrix')
    assert not (directory / 'matrix').exists()


def test_abandoned_claim_blocks_the_other_new_case_without_reset(active):
    c, _ = active
    registry = m.Registry(c)
    registry.claim('linux10609')
    with pytest.raises(ValueError, match='no live host lease'):
        registry.claim('linux10603')
    assert registry.status()['claimed_case_ids'] == ['clip-09-bank-0']


def test_unclaimed_case_directory_does_not_consume_a_fresh_claim(active):
    c, _ = active
    (c['root'] / 'cases/clip-09-bank-0/attempt-01').mkdir(parents=True)
    with pytest.raises(ValueError, match='Prior pending-case'):
        m.Registry(c).claim('linux10609')
    assert not (c['root'] / 'claims').exists()


def test_extra_request_is_rejected_instead_of_ignored(active):
    c, _ = active
    (c['root'] / 'requests/clip-09-bank-1.json').write_text('{}')
    with pytest.raises(ValueError, match='Unknown continuation'):
        m.Registry(c).status()


def test_unknown_previous_claim_rejects_before_hashes_or_new_writes(active, monkeypatch):
    c, _ = active
    state = copy.deepcopy(FIX['receipt']['status_after'])
    state['claimed_case_ids'].append('clip-09-bank-0')
    c['core'] = SimpleNamespace(check_ancestors=lambda previous: None,
        Registry=lambda previous: SimpleNamespace(status=lambda: state))
    real = module('unpatched_ancestor_check', ROOT / 'models/reviewed_panel_final_pair.py')
    with pytest.raises(ValueError, match='Previous coordinator differs'):
        real.check_ancestors(c)
    assert not (c['root'] / 'claims').exists()


@pytest.mark.parametrize('folder', ['requests', 'claims', 'cases', 'outcomes', 'exports', 'reviews'])
def test_already_touched_target_in_previous_namespace_blocks_new_attempt(active, folder):
    c, _ = active
    cid = m.TARGETS['linux10609']
    path = c['previous']['root'] / folder / (cid if folder in ('claims','cases') else cid+'.json')
    if folder in ('claims','cases'): path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True)
        path.write_text('{}')
    with pytest.raises(ValueError, match='Prior pending-case'):
        m.Registry(c).claim('linux10609')
    assert not (c['root']/'claims').exists()


def test_existing_fifteen_origins_cannot_be_relabelled():
    index=FIX['inputs']['case_index']
    assert {k:index[k] for k in FIX['old_data']['case_index']} == FIX['old_data']['case_index']
    assert index['clip-06-bank-0']['execution_sha256']=='4841ebbea9c41017808cc8f157f43cab11bbf5ec5cd1f85049c13034e9e07323'
    assert index['clip-05-bank-1']['execution_sha256']=='76b79e50c33e160b6587cc1095d357ee63e5af2b723aae643de898c828846b61'


def test_model_and_numerical_worker_body_match_qualified_predecessor():
    import ast
    before=ast.parse((SUPPORT/'core-packet/models/reviewed_panel_next_pair.py').read_text())
    after=ast.parse((ROOT/'models/reviewed_panel_final_pair.py').read_text())
    functions=lambda tree:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    b,a=functions(before),functions(after)
    assert ast.dump(a['worker'])==ast.dump(b['worker'])
    # Sole parent-worker change is the honest new CLI path; all numerical work/deadlines stay fixed.
    old=ast.unparse(b['numerical_case']).replace('scripts/continue_reviewed_next_pair.py','scripts/continue_reviewed_final_pair.py')
    assert ast.unparse(a['numerical_case'])==old


@pytest.mark.parametrize('change', ['sync-receipt','source','pins','scope'])
def test_new_policy_requires_the_actual_sync_and_774_file_boundary(change):
    op=operation()
    if change=='sync-receipt': op['sync_receipt_file_sha256']='0'*64
    elif change=='source': op['sources_sha256']['support/yoyo_bowling_review_sync.py']='0'*64
    elif change=='pins': op['preserved_file_count']=773
    else: op['sync_operation_sha256']='0'*64
    with pytest.raises(ValueError): m.validate_operation(op,m.canonical(op))


@pytest.mark.parametrize('namespace', m.HISTORICAL_ROOTS)
def test_any_older_coordinator_target_artifact_blocks_a_fresh_slot(active, namespace):
    c, _ = active
    cid = m.TARGETS['linux10603']
    p = c['transfer'] / namespace / 'claims' / cid / 'claim.json'
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')
    with pytest.raises(ValueError, match='Prior pending-case'):
        m.Registry(c).claim('linux10603')
    assert not (c['root'] / 'claims').exists()


def test_current_two_new_requests_are_original_unused_slots(active):
    c, _ = active
    for host, cid in m.TARGETS.items():
        assert cid not in c['data']['case_index']
        q = m.request(c['op'], host)
        assert q['max_attempts'] == 1 and q['additional_allowance_seconds'] == 0
    recovered = c['data']['case_index']['clip-06-bank-1']
    assert recovered['consumed_grant_id'] == 'fd4b4cb8d42e442f8d9ee8ac7130945e'
    assert '/recovery-01/' in recovered['archive_path']


@pytest.mark.parametrize('field,value', [('all_three_historical_STOP_records_unchanged',False)])
def test_activation_receipt_cannot_hide_any_old_STOP(active, field, value):
    c, receipt = active; bad = copy.deepcopy(receipt); bad[field] = value
    with pytest.raises(ValueError): m.verify_activation(bad, c['op'], c['data'], prior)


def test_exclusive_lock_descriptors_satisfy_real_NFS_record_lock_access_rules(tmp_path, monkeypatch):
    import fcntl
    c = context(tmp_path); observed = m.ancestor_pins(c)
    monkeypatch.setattr(m, 'check_ancestors', lambda context: dict(observed))
    monkeypatch.setattr(fcntl, 'flock', fcntl.lockf)
    result = m.activate(c, 'linux10609')
    assert m.verify_activation(result, c['op'], c['data'], prior)['accepted_cases'] == 17
    registry = m.Registry(c); registry.claim('linux10609')
    with pytest.raises(ValueError): registry.claim('linux10609')


@pytest.mark.parametrize('cut', [1, 4])
def test_partial_activation_can_finish_without_touching_old_history(tmp_path, monkeypatch, cut):
    c = context(tmp_path); observed = m.ancestor_pins(c)
    monkeypatch.setattr(m, 'check_ancestors', lambda context: dict(observed))
    values = m.activation_records(c['op'], c['data'], prior)
    c['root'].mkdir()
    for name in sorted(set(values) - {'genesis.json'})[:cut]: prior.exclusive_bytes(prior.bound(c['root'],name),values[name])
    result = m.activate(c,'linux10609')
    assert result['created'][-1] == 'genesis.json' and len(result['preexisting_identical']) == cut
    assert m.verify_activation(result,c['op'],c['data'],prior)['verified']


@pytest.mark.parametrize('change', ['missing', 'changed', 'symlink'])
def test_preserved_history_rejects_before_any_older_loader_may_initialize(tmp_path, monkeypatch, change):
    c = context(tmp_path)
    home = tmp_path / 'home'; transfer = home / m.BASE
    packet = transfer / m.PREFIX / 'unpacked' / m.PREFIX; packet.mkdir(parents=True)
    first = transfer / next(iter(c['data']['preserved_files_sha256']))
    first.parent.mkdir(parents=True, exist_ok=True)
    if change == 'changed': first.write_text('Changed immutable source')
    elif change == 'symlink': first.symlink_to(packet)
    def forbidden(*args): raise AssertionError('An older Registry loader must not initialize over damaged history')
    c['core'] = SimpleNamespace(load=forbidden)
    monkeypatch.setattr(m, 'trusted', lambda *a: c)
    before = {p: p.read_bytes() for p in transfer.rglob('*') if p.is_file() and not p.is_symlink()}
    with pytest.raises(ValueError): m.load(packet, m.canonical(c['op']), home)
    assert before == {p: p.read_bytes() for p in transfer.rglob('*') if p.is_file() and not p.is_symlink()}
    assert not (transfer / m.TARGET).exists()


def boundary_packet(tmp_path):
    import shutil
    packet = tmp_path / 'packet'; packet.mkdir()
    for name in m.SOURCES:
        source = ROOT / name
        if name.startswith('support/'): source = SUPPORT / name.removeprefix('support/')
        dest = packet / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, dest)
    for name, value in [('INPUT.json', FIX['inputs']), ('prior-OPERATION.json', FIX['sync_operation']),
                        ('prior-INPUT.json', FIX['sync_input']), ('prior-receipt.json', FIX['receipt'])]:
        (packet / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    op = operation()
    op.update(inputs_file_sha256=m.digest(packet / 'INPUT.json'), sources_sha256={n: m.digest(packet / n) for n in m.SOURCES})
    (packet / 'CONTINUATION.json').write_text(json.dumps(op, sort_keys=True, indent=2) + '\n')
    return packet, op


def test_real_source_metadata_boundary_reconstructs_all_seventeen_origins(tmp_path):
    packet, op = boundary_packet(tmp_path)
    c = m.boundary(packet, m.canonical(op))
    assert c['data'] == FIX['inputs'] and len(c['data']['case_index']) == 17
    assert c['data']['case_index']['clip-06-bank-1']['consumed_grant_id'] == sync.PRIOR_TIME['consumed_grant_id']
    assert len(c['data']['preserved_files_sha256']) == 774


@pytest.mark.parametrize('change', ['source', 'origin', 'prior-receipt'])
def test_real_boundary_rejects_drift_or_a_relabelled_recovery(tmp_path, change):
    packet, op = boundary_packet(tmp_path)
    if change == 'source': (packet / 'support/yoyo_bowling_review_sync.py').write_text('Changed source')
    elif change == 'prior-receipt': (packet / 'prior-receipt.json').write_text('{}')
    else:
        value = json.loads((packet / 'INPUT.json').read_text())
        value['case_index']['clip-06-bank-1']['origin'] = 'ordinary-fresh-case'
        (packet / 'INPUT.json').write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
        op['inputs_file_sha256'] = m.digest(packet / 'INPUT.json')
        (packet / 'CONTINUATION.json').write_text(json.dumps(op))
    with pytest.raises(ValueError): m.boundary(packet, m.canonical(op))


@pytest.mark.parametrize('cid', ['clip-07-bank-1', 'clip-08-bank-0'])
def test_registered_pair_origins_retain_their_actual_original_identities(cid):
    row = FIX['inputs']['case_index'][cid]
    original = sync.marker(FIX['sync_input'], cid)
    assert row['origin'] == 'reviewed-fifteen-case-pair'
    assert row['review_file_sha256'] == sync.MARKERS[cid]
    assert row['execution_sha256'] == original['execution_sha256']
    assert row['numeric_receipt_sha256'] == original['numeric_receipt_sha256']
    assert '/attempt-01/' in row['archive_path'] and 'consumed_grant_id' not in row


def test_host_assignments_cannot_be_swapped_or_admit_the_final_unrequested_bank():
    op=operation()
    op['first_requests']={'linux10609':'clip-08-bank-1','linux10603':'clip-09-bank-0'}
    with pytest.raises(ValueError):m.validate_operation(op,m.canonical(op))
    op=operation();op['first_requests']['linux10609']='clip-09-bank-1'
    with pytest.raises(ValueError):m.validate_operation(op,m.canonical(op))
