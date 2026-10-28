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

m = module('reviewed_continuation_tests', ROOT / 'models/archery_recovery.py')
SUPPORT = Path(os.environ.get('ARCHERY_RECOVERY_SUPPORT', str(ROOT / 'support')))
prior = module('reviewed_prior_tests', SUPPORT / 'core-packet/support/core-packet/support/core-packet/support/reviewed_panel.py')
core = module('qualified_failed_case_tests', SUPPORT / 'core-packet/models/archery_continuation.py')
REFERENCE = Path(os.environ.get('REVIEWED_TEST_REFERENCE', str(ROOT)))
ops = module('reviewed_reference_tests', REFERENCE / 'models/two_host_panel.py')
FIX = json.loads((ROOT / 'tests/fixtures/archery_recovery_metadata.json').read_text())


def operation():
    return dict({'kind': m.KIND, 'core_operation_sha256': m.CORE_OP, 'core_module_sha256': m.CORE_SOURCE, 'prior_activation_receipt_file_sha256': m.PRIOR_RECEIPT, 'proposal_sha256': m.PROPOSAL, 'authorization_file_sha256': m.AUTH_FILE, 'grant_id': m.GRANT_ID, 'reference_revision_sha256': m.RP, 'reference_implementation_sha256': m.MP, 'imported_case_ids': m.ACCEPTED, 'first_requests': m.TARGETS, 'coordinator_namespace': m.TARGET, 'historical_STOP_sha256': m.STOP, 'failed_STOP_file_sha256': m.FAILED_STOP, 'all_existing_STOP_records_preserved': True, 'numeric_settings_changed': False, 'case_seconds': m.LIMIT, 'previous_charged_seconds': m.CARRIED, 'old_remaining_seconds': m.OLD_REMAINING, 'old_remainder_added': False, 'prior_numeric_attempts': 1, 'additional_numeric_attempts': 1, 'kill_grace_seconds': 15, 'matrix_seconds': 3600, 'replay_seconds': 900, 'parent_seconds': 4620, 'max_attempts': 1, 'extra_case_seconds': 4800, 'automatic_GPU_retries': 0, 'max_parallel_cases': 1, 'one_case_per_host': True, 'automatic_queue': False, 'paid_compute': False, 'research_admitted': False, 'preserved_file_count': m.PRESERVED}, sources_sha256={n: m.CORE_SOURCE if n == 'support/core-packet/models/archery_continuation.py' else '1'*64 for n in m.SOURCES})


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
        'sync_op': FIX['old_operation'], 'sync_input': FIX['old_data'], 'sync_receipt': FIX['receipt'],
        'previous': {'root': transfer / m.CORE_TARGET},
        'core': SimpleNamespace(TARGETS={'linux10609':'clip-07-bank-0','linux10603':'clip-06-bank-1'}),
        'revision': FIX['previous_fixture']['reference_revision'], 'manifest': FIX['previous_fixture']['reference_manifest']}


@pytest.fixture
def active(tmp_path, monkeypatch):
    c = context(tmp_path)
    observed = m.ancestor_pins(c)
    monkeypatch.setattr(m, 'check_ancestors', lambda context: dict(observed))
    receipt = m.activate(c, 'linux10609')
    return c, receipt


def test_explicit_authorization_and_failed_case_are_the_only_boundary():
    assert m.validate_authorization(FIX['authorization'])['approved'] is True
    assert m.canonical(FIX['proposal']) == m.PROPOSAL
    assert FIX['proposal']['approved'] is False
    assert FIX['inputs']['case_index'] == FIX['old_data']['case_index']
    assert len(FIX['inputs']['preserved_files_sha256']) == 470
    assert 'clip-06-bank-1' not in m.ACCEPTED and 'clip-07-bank-0' in m.ACCEPTED
    assert FIX['inputs']['past_time_accounting']['total_original_allowance_charged_seconds'] == 2348


@pytest.mark.parametrize('key,value', [
    ('first_requests', {'linux10609': 'clip-06-bank-1'}), ('coordinator_namespace', m.CORE_TARGET),
    ('case_seconds', 4801), ('previous_charged_seconds', 0), ('old_remaining_seconds', 2453),
    ('old_remainder_added', True), ('prior_numeric_attempts', 0), ('additional_numeric_attempts', 2),
    ('extra_case_seconds', 4801), ('grant_id', 'old-grant'), ('authorization_file_sha256','0'*64),
    ('proposal_sha256','0'*64), ('failed_STOP_file_sha256','0'*64), ('all_existing_STOP_records_preserved',False),
    ('kill_grace_seconds', 16), ('matrix_seconds', 3601), ('replay_seconds', 901),
    ('parent_seconds', 4621), ('max_attempts', 2), ('automatic_GPU_retries', 1),
    ('max_parallel_cases', 2), ('one_case_per_host', False), ('automatic_queue', True),
    ('paid_compute', True), ('research_admitted', True), ('numeric_settings_changed', True),
    ('historical_STOP_sha256', '0'*64), ('reference_implementation_sha256', '0'*64),
    ('imported_case_ids', m.ACCEPTED[:-1])])
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
def test_new_request_binds_exactly_one_approved_extra_attempt(host):
    op=operation();q=m.request(op,host)
    assert m.validate_request(q,m.canonical(q),op,host)==m.TARGETS[host]
    assert q['max_attempts']==1 and q['additional_numeric_attempts']==1 and q['prior_numeric_attempts']==1
    assert q['grant_id']==m.GRANT_ID and q['extra_case_seconds']==4800 and not q['old_remainder_added']
    q['grant_id']='another-grant'
    with pytest.raises(ValueError):m.validate_request(q,m.canonical(q),op,host)


@pytest.mark.parametrize('started,now', [(None, 1), (float('nan'), 1), (True, 1), (100, 99), (100, 4900), (0, float('inf'))])
def test_time_cannot_be_unknown_exhausted_or_reset(started, now):
    with pytest.raises((ValueError, TimeoutError)):
        m.elapsed(started, now)


def test_cpu_setup_counts_against_the_original_case_limit():
    assert m.elapsed(100, 350) == 250
    assert min(4620, m.LIMIT - m.elapsed(100, 350) - 120) == 4430


def test_five_record_activation_keeps_all_thirteen_original_identities(active):
    c, receipt = active
    assert len(receipt['created']) == 5 and receipt['created'][-1] == 'genesis.json'
    assert len(receipt['preserved_files_sha256']) == 470
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
    elif change == 'claim': (root / 'claims/clip-06-bank-1').mkdir(parents=True)
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
    r.claim('linux10603')
    with pytest.raises(ValueError):
        r.claim('linux10603')
    assert r.status()['claimed_case_ids'] == ['clip-06-bank-1']


def test_only_unused_archery_request_can_be_claimed(active):
    c, _ = active
    registry = m.Registry(c)
    with pytest.raises(ValueError): registry.claim('linux10609')
    registry.claim('linux10603')
    registry.fail('linux10603','retained CPU fixture failure',25)
    state=registry.status()
    assert state['failed_case_ids']==['clip-06-bank-1'] and state['stop_dispatch']
    assert state['accepted_case_ids']==m.ACCEPTED
    with pytest.raises(ValueError): registry.claim('linux10603')


@pytest.mark.parametrize('place', ['claims', 'cases', 'outcomes', 'exports', 'reviews', 'old-v1'])
def test_old_or_unreviewed_pending_case_state_cannot_be_reset(active, place):
    c, _ = active
    cid = m.TARGETS['linux10603']
    if place == 'old-v1':
        (c['root'] / 'cases' / cid).mkdir(parents=True)
    elif place in ('claims', 'cases'):
        (c['root'] / place / cid).mkdir(parents=True)
    else:
        path = c['root'] / place / (cid + '.json')
        path.parent.mkdir(parents=True)
        path.write_text('{}')
    before={str(p):p.read_bytes() for p in c['root'].rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        m.Registry(c).claim('linux10603')
    assert before=={str(p):p.read_bytes() for p in c['root'].rglob('*') if p.is_file()}
    assert not (c['root']/'claims'/cid/'claim.json').exists()


def test_global_old_host_lease_is_shared_between_both_namespaces(active):
    c, _ = active
    with ops.file_lock(c['legacy'] / 'leases/linux10603', wait_seconds=0):
        command = [sys.executable, '-B', '-c',
            "import fcntl,sys;f=open(sys.argv[1],'a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)",
            str(c['legacy'] / 'leases/linux10603')]
        result = subprocess.run(command, capture_output=True)
        assert result.returncode != 0 and b'BlockingIOError' in result.stderr


@pytest.mark.parametrize('host', list(m.TARGETS))
def test_new_numeric_identity_binds_actual_wrapper_and_retains_old_reference(active, host):
    c, _ = active
    old = ops.execution_identity(c['revision'], c['manifest'], host)
    new = m.execution_identity(c, host)
    assert new['reference_execution_sha256'] == m.canonical(old)
    assert new['continuation_operation_sha256'] == m.canonical(c['op'])
    assert new['prior_activation_receipt_file_sha256'] == m.PRIOR_RECEIPT
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
        if change == 'claim-symlink': path = root / 'claims/clip-06-bank-1/claim.json'
        else: path = root / 'outcomes/clip-06-bank-1.json'
        path.parent.mkdir(parents=True)
        path.symlink_to(root / 'genesis.json')
    with pytest.raises(ValueError):
        m.Registry(c).claim('linux10603')


def test_complete_outcome_alone_never_confers_independent_acceptance(active):
    c, _ = active
    registry = m.Registry(c)
    registry.claim('linux10603')
    ops.write_record(c['root'] / 'outcomes/clip-06-bank-1.json', {
        'kind': m.KIND + '-outcome', 'operation_sha256': m.canonical(c['op']),
        'case_id': 'clip-06-bank-1', 'host': 'linux10603', 'complete': True, 'research_admitted': False})
    state = registry.status()
    assert state['computed_case_ids'] == ['clip-06-bank-1']
    assert state['accepted_case_ids'] == m.ACCEPTED


def test_full_run_failure_consumes_once_stops_peer_and_counts_setup(active, monkeypatch):
    c, _ = active
    monkeypatch.setattr(m.platform, 'node', lambda: 'linux10603')
    monkeypatch.setattr(c['ops'], 'hardware_gate', lambda *a, **kw: ops.HOSTS['linux10603'])
    captured = []
    def reject(context, host, directory, budget):
        captured.append((host, budget))
        raise RuntimeError('synthetic CPU failure before model work')
    monkeypatch.setattr(m, 'numerical_case', reject)
    with pytest.raises(RuntimeError, match='synthetic CPU failure'):
        m.run(c, 'linux10603', m.time.time() - 250)
    assert len(captured) == 1 and 4420 < captured[0][1] <= 4430
    state = m.Registry(c).status()
    assert state['stop_dispatch'] and state['failed_case_ids'] == ['clip-06-bank-1']
    assert state['accepted_case_ids'] == m.ACCEPTED
    with pytest.raises(ValueError): m.Registry(c).claim('linux10603')
    with pytest.raises(ValueError): m.run(c, 'linux10603', m.time.time())
    assert len(captured) == 1


def test_worker_claim_role_is_one_use_even_if_boundary_rejects(active, monkeypatch):
    c, _ = active
    host = 'linux10603'
    cid = m.TARGETS[host]
    m.Registry(c).claim(host)
    directory = c['root'] / 'cases' / cid / m.ATTEMPT
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


def test_unused_old_request_is_retained_and_does_not_consume_numeric_claim(active):
    c, _ = active
    path=c['previous']['root']/'requests/clip-06-bank-1.json'
    path.parent.mkdir(parents=True)
    text='Already externally pinned and verified by check_ancestors; unused request retained.'
    path.write_text(text)
    m.Registry(c).claim('linux10603')
    assert path.read_text()==text
    assert m.Registry(c).status()['claimed_case_ids']==['clip-06-bank-1']


def test_unclaimed_case_directory_does_not_consume_a_fresh_claim(active):
    c, _ = active
    (c['root'] / 'cases/clip-06-bank-1/recovery-01').mkdir(parents=True)
    with pytest.raises(ValueError, match='pending artifacts'):
        m.Registry(c).claim('linux10603')
    assert not (c['root'] / 'claims').exists()


def test_extra_request_is_rejected_instead_of_ignored(active):
    c, _ = active
    (c['root'] / 'requests/clip-09-bank-0.json').write_text('{}')
    with pytest.raises(ValueError, match='Unknown continuation'):
        m.Registry(c).status()


def test_unknown_previous_claim_rejects_before_hashes_or_new_writes(active, monkeypatch):
    c,_=active
    state={'claimed_case_ids':['clip-06-bank-1','clip-09-bank-0']}
    c['core']=SimpleNamespace(KIND='matched-panel-archery-time-carry-v1',check_ancestors=lambda previous:None,
        Registry=lambda previous:SimpleNamespace(status=lambda:state))
    real=module('unpatched_recovery_ancestor_check',ROOT/'models/archery_recovery.py')
    with pytest.raises(ValueError,match='prior failed coordinator changed'):real.check_ancestors(c)
    assert not (c['root']/'claims').exists()


@pytest.mark.parametrize('folder', ['claims', 'cases', 'outcomes', 'exports', 'reviews', 'failures'])
def test_original_failed_namespace_artifacts_are_preserved_by_new_claim(active, folder):
    c,_=active;cid=m.TARGETS['linux10603'];path=c['previous']['root']/folder/(cid if folder in ('claims','cases') else cid+'.json')
    if folder in ('claims','cases'):path.mkdir(parents=True)
    else:path.parent.mkdir(parents=True);path.write_text('Preserved fixture bytes already pinned by ancestor check.')
    before={str(p):p.read_bytes() for p in c['previous']['root'].rglob('*') if p.is_file()}
    m.Registry(c).claim('linux10603')
    assert {str(p):p.read_bytes() for p in c['previous']['root'].rglob('*') if p.is_file()}==before


def test_existing_fourteen_origins_cannot_be_relabelled():
    assert FIX['inputs']['case_index']==FIX['old_data']['case_index']
    assert FIX['inputs']['case_index']['clip-07-bank-0']['execution_sha256']=='71b000ed2e55e2169b715c1f2fdf9f27ba5748438072f889ea918367caf5f763'
    assert FIX['proposal']['rejected_execution_sha256']==m.FAILED_EXECUTION


def test_model_and_numerical_worker_body_match_qualified_predecessor():
    import ast
    before=ast.parse((SUPPORT/'core-packet/models/archery_continuation.py').read_text())
    after=ast.parse((ROOT/'models/archery_recovery.py').read_text())
    functions=lambda tree:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    b,a=functions(before),functions(after)
    # Only the operational recovery-directory expression changes inside the neural worker.
    expected=ast.parse(ast.unparse(b['worker']).replace("'/attempt-01'", "'/' + ATTEMPT")).body[0]
    assert ast.dump(a['worker'])==ast.dump(expected)
    assert ast.unparse(a['numerical_case'])==ast.unparse(b['numerical_case']).replace('scripts/continue_archery.py','scripts/recover_archery.py')


@pytest.mark.parametrize('change', ['sync-receipt','source','pins','scope'])
def test_new_policy_requires_actual_approval_failed_state_and_470_pins(change):
    op=operation()
    if change=='sync-receipt':op['prior_activation_receipt_file_sha256']='0'*64
    elif change=='source':op['sources_sha256']['support/core-packet/models/archery_continuation.py']='0'*64
    elif change=='pins':op['preserved_file_count']=469
    else:op['core_operation_sha256']='0'*64
    with pytest.raises(ValueError):m.validate_operation(op,m.canonical(op))


def test_grant_preserves_past_charge_without_stacking_unused_remainder(active):
    c,r=active;grant=ops.read_record(c['root']/'recovery-grant.json')
    assert grant==m.recovery_grant() and m.CARRIED==2348 and m.OLD_REMAINING==2452
    assert m.CARRIED+m.OLD_REMAINING==4800 and m.LIMIT==4800
    assert not grant['old_remainder_added'] and grant['prior_numeric_attempts']==grant['additional_numeric_attempts']==1
    result=m.verify_activation(r,c['op'],c['data'],prior)
    assert result['fresh_attempt_seconds']==4800 and result['previous_charged_seconds']==2348
    assert m.elapsed(100,4899.99)<4800
    with pytest.raises(TimeoutError):m.elapsed(100,4900)
    assert m.request(c['op'],'linux10603')['recovery_grant_sha256']==m.canonical(grant)


def test_changed_grant_cannot_reset_claim_or_clock(active):
    c,_=active
    ops.write_record(c['root']/'recovery-grant.json',dict(m.recovery_grant(),fresh_attempt_seconds=7252),replace=True)
    with pytest.raises(ValueError):m.Registry(c).claim('linux10603')
    assert not (c['root']/'claims').exists()


def test_original_exclusive_lock_uses_readwrite_descriptor_for_NFS(active,monkeypatch):
    import fcntl
    c,_=active
    def record_lock(fd,flags):
        # A true POSIX write lock reproduces the NFS access-mode rule.
        fcntl.lockf(fd,flags)
    monkeypatch.setattr(ops.fcntl,'flock',record_lock)
    with ops.file_lock(c['legacy']/'leases/linux10603',wait_seconds=0):
        assert m.Registry(c).status()['accepted_case_ids']==m.ACCEPTED


@pytest.mark.parametrize('field,value',[('approved',False),('extra_case_seconds',4801),('additional_attempts',2),('grant_id','old-floor-grant'),('old_remainder_added',True),('host','linux10609')])
def test_authorization_cannot_be_reused_or_widened(field,value):
    authorization=copy.deepcopy(FIX['authorization']);authorization[field]=value
    with pytest.raises(ValueError):m.validate_authorization(authorization)


def test_all_five_raw_failed_matrix_tensors_are_pinned():
    old=FIX['proposal']['preserved_files_sha256'];new=FIX['inputs']['preserved_files_sha256']
    assert {n:new[n] for n in old}==old and len(old)==465 and len(new)==470
    for name in ('inputs.pt','A.pt','B.pt','C.pt','D.pt'):
        relative=m.CORE_TARGET+'/cases/clip-06-bank-1/attempt-01/matrix/'+name
        assert new[relative]==FIX['failed_manifest']['files']['matrix/'+name]['sha256']


def test_guard_rejection_keeps_process_snapshots_without_whitelisting(active,tmp_path,monkeypatch):
    c,_=active;calls=[]
    def reject(*args,**kwargs):raise ValueError('Another CUDA compute process is active on this host.')
    def snapshot(command,**kwargs):
        calls.append(command);return subprocess.CompletedProcess(command,0,'2260645, /usr/NX/bin/nxnode.bin, 183\n','')
    monkeypatch.setattr(c['ops'],'hardware_gate',reject);monkeypatch.setattr(m.subprocess,'run',snapshot)
    with pytest.raises(ValueError,match='Another CUDA'):m.strict_gate(c,'linux10603',tmp_path,'matrix-after',allow_self=True)
    event=ops.read_record(tmp_path/'hardware-matrix-after.json')
    assert event['passed'] is False and len(calls)==2 and '2260645' in event['compute_processes']['stdout']
    assert not (c['root']/'claims').exists()
