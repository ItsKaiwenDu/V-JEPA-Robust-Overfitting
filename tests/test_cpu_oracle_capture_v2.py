"""Local integration and failure tests. No university job or full-size write."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import time

import pytest

from models import cpu_oracle_capture_v2 as remote
from models import cpu_oracle_capture_client_v2 as client
from models import cpu_oracle_capture_runtime_v2 as runtime
from models.cpu_oracle_packet import build
from models.study_artifacts import read_record

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'runs/pretrained/diagnostic/cpu-oracle-scratch-recovery-20261104/linux10603'


def control():
    value = json.loads((OLD/'CONTROL.json').read_text())
    value.update(kind=remote.KIND, role='target', policy=remote.POLICY,
        authorization_present_at_preparation=False, reuse_old_grant=False, original_refusals_waived=False)
    names = set(remote.REMOTE_NAMES)-{'models/__init__.py'} | {
        'models/cpu_oracle_capture_client_v2.py', 'scripts/run_cpu_oracle_capture_v2.py'}
    value['local_source_sha256'] = {n: remote.old.digest(ROOT/n) for n in names}
    value['remote_source_sha256'] = {n: hashlib.sha256(remote.OVERLAY_INIT).hexdigest()
        if n == 'models/__init__.py' else remote.old.digest(ROOT/n) for n in remote.REMOTE_NAMES}
    value['prior_claims_sha256']['runs/pretrained/diagnostic/cpu-oracle-scratch-recovery-20261104/linux10603/ATTEMPT.json'] = remote.old.digest(OLD/'ATTEMPT.json')
    return value


def authorization(sha):
    return {'kind': remote.AUTH_KIND, 'approved': True, 'control_sha256': sha,
        'host': 'linux10603', 'maximum_new_attempts': 1, 'max_seconds': 600,
        'worker_seconds': 90, 'write_test_bytes': remote.POLICY['write_test_bytes'],
        'setup_ceiling': remote.POLICY['setup_ceiling'], 'capture_ceiling': remote.POLICY['capture_ceiling'],
        'collection_seconds': 600, 'CPU_only': True, 'paid_compute': False,
        'reset_old_claims': False, 'automatic_retries': 0,
        'user_basis': 'LOCAL TEST FIXTURE ONLY. No authorization for any university resource.'}


def base(tmp_path, *, armed=False):
    folder = tmp_path/'client'; folder.mkdir(); (folder/'known-hosts').write_text('public test host key\n')
    value = control(); value['known_hosts_sha256'] = remote.old.digest(folder/'known-hosts')
    (folder/'CONTROL.json').write_bytes(remote.old.json_bytes(value)); sha = remote.old.digest(folder/'CONTROL.json')
    if armed: (folder/'AUTHORIZATION.json').write_bytes(remote.old.json_bytes(authorization(sha)))
    return folder, value, sha


@pytest.mark.parametrize('key,value', [('kind', remote.old.KIND), ('host', 'linux10609'),
    ('role', 'control'), ('reuse_old_grant', True), ('original_refusals_waived', True),
    ('authorization_present_at_preparation', True), ('archive_bytes', 1)])
def test_old_or_changed_scope_is_not_a_v2_target(key, value):
    c = control(); c[key] = value
    with pytest.raises(ValueError): remote.validate_control(c)


@pytest.mark.parametrize('field', ['setup_ceiling', 'minimum_free_bytes', 'write_test_bytes', 'max_seconds', 'worker_seconds'])
def test_policy_cannot_be_relaxed(field):
    c = control(); c['policy'] = {**c['policy'], field: c['policy'][field]+1}
    with pytest.raises(ValueError): remote.validate_control(c)


def test_consumed_v1_authorization_cannot_arm_v2():
    auth = json.loads((OLD/'AUTHORIZATION.json').read_text())
    with pytest.raises(ValueError): remote.validate_authorization(auth, remote.old.digest(OLD/'CONTROL.json'), 'linux10603')


@pytest.mark.parametrize('field,value', [('maximum_new_attempts', 2), ('max_seconds', 601),
    ('worker_seconds', 91), ('setup_ceiling', 4*2**30), ('collection_seconds', 601),
    ('CPU_only', False), ('paid_compute', True), ('reset_old_claims', True),
    ('approved', 1), ('automatic_retries', 1)])
def test_new_exception_requires_the_exact_reviewed_bound(field, value):
    auth = authorization('1'*64); auth[field] = value
    with pytest.raises(ValueError): remote.validate_authorization(auth, '1'*64, 'linux10603')


def test_unarmed_client_stops_before_claim_session_or_connection(tmp_path, monkeypatch):
    folder, _, sha = base(tmp_path)
    def denied(*a, **k): pytest.fail('An unarmed client attempted a subprocess.')
    monkeypatch.setattr(client.subprocess, 'Popen', denied)
    with pytest.raises(ValueError, match='UNARMED'): client.run_client(folder, sha)
    assert set(p.name for p in folder.iterdir()) == {'CONTROL.json', 'known-hosts'}


def test_bound_source_change_refuses_before_claim(tmp_path, monkeypatch):
    folder, c, _ = base(tmp_path, armed=True)
    c['local_source_sha256']['models/cpu_oracle_capture_v2.py'] = '0'*64
    (folder/'CONTROL.json').write_bytes(remote.old.json_bytes(c)); sha = remote.old.digest(folder/'CONTROL.json')
    with pytest.raises(ValueError, match='source/history'): client.run_client(folder, sha)
    assert not (folder/'ATTEMPT.json').exists()


def test_overlay_preserves_the_original_packet_and_refuses_changed_new_source(tmp_path):
    c = control(); blobs = client.sources(c)
    remote.write_overlay(tmp_path, blobs, c)
    for name, data in blobs.items(): assert (tmp_path/'overlay'/name).read_bytes() == data
    assert not (tmp_path/'code').exists()
    altered = dict(blobs); altered['models/scratch_allocation_v2.py'] += b'\n# changed'
    fresh = tmp_path/'fresh'; fresh.mkdir()
    with pytest.raises(ValueError, match='source changed'): remote.write_overlay(fresh, altered, c)
    assert not (fresh/'overlay').exists()


def scratch(tmp_path):
    folder = tmp_path/'private'; folder.mkdir(mode=0o700); info = tmp_path.stat()
    return folder, {'device': info.st_dev, 'inode': info.st_ino}


def test_physical_directory_blocks_are_counted(tmp_path):
    folder, identity = scratch(tmp_path); (folder/'subdir').mkdir()
    (folder/'subdir/witness').write_bytes(b'keep')
    sample = remote.storage_gate(folder, identity, parent_path=tmp_path, phase='setup')
    expected = sum(p.stat().st_blocks*512 for p in (folder, folder/'subdir', folder/'subdir/witness'))
    assert sample['allocated_bytes'] == expected


def test_recorded_four_gib_footprint_refuses_in_the_actual_supervisor_gate(tmp_path, monkeypatch):
    folder, identity = scratch(tmp_path)
    monkeypatch.setattr(remote.old, 'tree_inventory', lambda *a, **k: ({}, 4294901760))
    with pytest.raises(ValueError, match='allocated-byte ceiling'):
        remote.storage_gate(folder, identity, parent_path=tmp_path, phase='setup')


def test_capture_phase_cannot_retain_a_setup_proof(tmp_path):
    folder, identity = scratch(tmp_path); (folder/'space-proof-v2.bin.partial').write_bytes(b'failed proof')
    assert remote.storage_gate(folder, identity, parent_path=tmp_path, phase='setup')
    with pytest.raises(ValueError, match='category'): remote.storage_gate(folder, identity, parent_path=tmp_path)


def test_supervisor_stops_a_blocked_child_within_the_original_phase_deadline(tmp_path):
    folder, identity = scratch(tmp_path)
    command = [sys.executable, '-B', '-u', '-c', 'import os,time;print(os.getpid(),flush=True);time.sleep(10)']
    began = time.monotonic()
    with pytest.raises(TimeoutError):
        remote.supervise(command, folder, role='proof', deadline=began+.35,
            gate=lambda: remote.storage_gate(folder, identity, parent_path=tmp_path, phase='setup'), file_limit=8*2**20)
    assert time.monotonic()-began < 3
    result = json.loads((folder/'supervisor-proof.json').read_text())
    assert result['complete'] is False and result['error_type'] == 'TimeoutError'
    pid = int((folder/'proof.log').read_text().strip())
    with pytest.raises(ProcessLookupError): os.kill(pid, 0)


def test_child_file_size_limit_is_enforced_by_the_kernel(tmp_path):
    folder, identity = scratch(tmp_path); target = folder/'too-large.bin'
    result = remote.supervise([sys.executable, '-B', '-c',
        'from pathlib import Path;Path('+repr(str(target))+').write_bytes(b"x"*65536)'],
        folder, role='bounded-child', deadline=time.monotonic()+5,
        gate=lambda: remote.storage_gate(folder, identity, parent_path=tmp_path, phase='setup'), file_limit=8192)
    assert result['exit_status'] != 0 and target.stat().st_size <= 8192


def proof_record():
    return {'kind': 'scratch-allocation-proof-v2', 'passed': True, 'complete': True,
        'proof_removed': True, 'fsync_succeeded': True, 'preallocator_injected_for_test': False,
        'requested_bytes': 2684354560, 'bytes_written': 2684354560, 'bytes_read_back': 2684354560,
        'setup_ceiling_bytes': 3*2**30, 'peak_observed_allocated_bytes': 2684362752, 'wall_seconds': 20}


@pytest.mark.parametrize('field,value', [('passed', False), ('preallocator_injected_for_test', True),
    ('bytes_written', 16384), ('bytes_read_back', 0), ('setup_ceiling_bytes', 4*2**30),
    ('peak_observed_allocated_bytes', 4294901760), ('wall_seconds', 61)])
def test_reference_gate_refuses_injected_small_or_over_budget_proof(tmp_path, field, value):
    data = proof_record(); data[field] = value
    (tmp_path/'allocation-result-v2.json').write_text(json.dumps(data))
    with pytest.raises(ValueError): remote.validate_proof(tmp_path)


def test_proof_refusal_never_dispatches_reference_but_collects_failure_files(tmp_path, monkeypatch):
    c = control(); sha = hashlib.sha256(remote.old.json_bytes(c)).hexdigest()
    folder, identity = scratch(tmp_path); roles = []
    monkeypatch.setattr(remote, 'readiness', lambda _: identity)
    monkeypatch.setattr(remote.tempfile, 'mkdtemp', lambda **kwargs: str(folder))
    def refuse(command, root, **kwargs):
        roles.append(kwargs['role']); (Path(root)/'space-proof-v2.bin.partial').write_bytes(b'preserve partial')
        raise ValueError('simulated physical allocation excess')
    monkeypatch.setattr(remote, 'supervise', refuse)
    stream = io.BytesIO()
    result = remote.run_remote(c, authorization(sha), io.BytesIO(), stream, client.sources(c))
    assert roles == ['proof'] and result['run_phase_finished'] and not result['reference_process_started']
    assert 'allocation excess' in result['error']
    assert (folder/'space-proof-v2.bin.partial').read_bytes() == b'preserve partial'
    assert not (folder/'results').exists() and not (folder/'emergency-space.bin').exists()
    stream.seek(0); prelude = json.loads(stream.readline()); manifest = json.loads(stream.readline())
    assert prelude['kind'] == remote.KIND and 'operator-result.json' in manifest['files']


@pytest.fixture(scope='module')
def tiny_capture(tmp_path_factory):
    folder = tmp_path_factory.mktemp('capture-v2-tiny-local')
    manifest, _ = build(ROOT, folder/'packet', tiny=True)
    packet = folder/'packet/handoff'; scratch_dir = folder/'scratch'; scratch_dir.mkdir(mode=0o700)
    state = runtime.run_workers(packet, scratch_dir, ROOT/'external/vjepa2', time.time(), local=True)
    return packet, scratch_dir, state


def test_four_fresh_original_tiny_workers_and_v2_audit_export_complete(tiny_capture):
    packet, folder, state = tiny_capture
    assert state['complete'] and state['saved_value_checks_passed'] and state['all_planned_workers_completed']
    assert not state['original_parent_invoked'] and not state['original_refusals_waived']
    assert len(state['tasks']) == 4 and state['local_tiny_rehearsal']
    summary = read_record(folder/'results/summary-v2.json', expected_sha256=state['summary_sha256'])
    assert summary['bounds'] == {'rtol': 1e-12, 'atol': 1e-14} and summary['tiny_fixture']
    assert all(row['passed'] for row in summary['workers']) and not summary['research_admitted']
    assert (folder/'results/evidence.tar.gz').is_file()
    assert read_record(folder/'results/workers/0/worker.json')['local_CPU_rehearsal']


def test_saved_tiny_output_reaudits_without_starting_workers(tiny_capture, tmp_path, monkeypatch):
    packet, folder, state = tiny_capture
    def denied(*a, **k): pytest.fail('Saved-value review started a process.')
    monkeypatch.setattr(runtime.subprocess, 'Popen', denied)
    result = runtime.review_captures(packet, folder/'results/workers', state['tasks'], tmp_path/'audit',
        expected_host=__import__('platform').node(), tiny=True)
    assert result['saved_value_checks_passed']


@pytest.mark.parametrize('defect', ['truncated', 'reordered', 'changed_hash', 'wrong_host'])
def test_integration_summary_does_not_accept_incomplete_or_changed_capture(tiny_capture, tmp_path, defect):
    packet, folder, state = tiny_capture; rows = copy.deepcopy(state['tasks']); host = __import__('platform').node()
    if defect == 'truncated': rows.pop()
    elif defect == 'reordered': rows.reverse()
    elif defect == 'changed_hash': rows[0]['worker_sha256'] = '0'*64
    else: host = 'another-host'
    with pytest.raises(ValueError): runtime.review_captures(packet, folder/'results/workers', rows,
        tmp_path/'audit', expected_host=host, tiny=True)


def test_runtime_rejects_expired_original_clock_before_any_worker(tiny_capture, tmp_path):
    packet, _, _ = tiny_capture
    with pytest.raises(ValueError, match='freshness'): runtime.run_workers(packet, tmp_path, ROOT/'external/vjepa2', time.time()-121, local=True)
    assert not (tmp_path/'results').exists()


def test_bootstrap_code_and_overlay_imports_are_syntactically_valid(tmp_path):
    c = control(); command = client.command_for(tmp_path, c)
    assert command[0] == 'ssh' and command.count('kdu@linux10603.dc.engr.scu.edu') == 1
    assert 'StrictHostKeyChecking=yes' in command and 'ClearAllForwardings=yes' in command
    python_code = shlex.split(command[-1])[-1]
    compile(python_code, '<v2-bootstrap>', 'exec')
    assert len(python_code.encode()) < 100000


def test_actual_bootstrap_rejects_an_old_grant_before_any_remote_operation(tmp_path):
    code = shlex.split(client.command_for(tmp_path, control())[-1])[-1]
    old_auth = (OLD/'AUTHORIZATION.json').read_bytes()
    result = subprocess.run([sys.executable, '-B', '-c', code], input=old_auth,
        capture_output=True, timeout=15)
    assert result.returncode != 0 and result.stdout == b''
    assert b'separate explicit v2 target resource exception' in result.stderr
    assert not list(tmp_path.iterdir())


def response_bytes(folder, c, sha, *, defect=None):
    folder.mkdir(mode=0o700); (folder/'operator-control.json').write_bytes(remote.old.json_bytes(c))
    state = {'kind': remote.KIND, 'control_sha256': sha, 'host': c['host'], 'run_phase_finished': True,
        'reference_process_started': False, 'CPU_only': True, 'GPU_work': False, 'paid_compute': False,
        'scientific_review_complete': False, 'old_refusals_waived': False, 'error': 'LOCAL TRANSPORT FIXTURE'}
    (folder/'operator-result.json').write_bytes(remote.old.json_bytes(state)); (folder/'retained.partial').write_bytes(b'keep failed data')
    result = io.BytesIO(); result.write(remote.old.json_bytes({'kind': remote.KIND, 'phase': 'scratch-created',
        'host': c['host'], 'control_sha256': sha, 'scratch_directory': '/var/tmp/'+remote.PREFIX+c['host']+'-fixture'}))
    remote.result_stream(folder, sha, result); data = result.getvalue()
    if defect == 'truncated': return data[:-10240]
    if defect == 'changed': return data.replace(b'keep failed data', b'keep BAD!!! data')
    return data


def test_transport_preserves_failure_fragments_and_timestamps_without_numerical_acceptance(tmp_path):
    c = control(); sha = hashlib.sha256(remote.old.json_bytes(c)).hexdigest()
    data = response_bytes(tmp_path/'source', c, sha); response = tmp_path/'response.bin'; response.write_bytes(data)
    receipt = client.verify_response(response, tmp_path/'saved', c, sha)
    assert receipt['transport_verified'] and not receipt['scientific_review_complete']
    assert not receipt['model_restarted_for_collection'] and not receipt['remote_scratch_deleted']
    assert (tmp_path/'saved/retained.partial').read_bytes() == b'keep failed data'
    assert (tmp_path/'saved/retained.partial').stat().st_mtime_ns == (tmp_path/'source/retained.partial').stat().st_mtime_ns


@pytest.mark.parametrize('defect', ['truncated', 'changed'])
def test_partial_or_changed_transport_does_not_verify(tmp_path, defect):
    c = control(); sha = hashlib.sha256(remote.old.json_bytes(c)).hexdigest()
    response = tmp_path/'response.bin'; response.write_bytes(response_bytes(tmp_path/'source', c, sha, defect=defect))
    with pytest.raises((ValueError, tarfile.TarError)): client.verify_response(response, tmp_path/'saved', c, sha)


def test_one_connection_client_and_download_only_never_relaunches_reference(tmp_path):
    folder, c, sha = base(tmp_path, armed=True)
    data = response_bytes(tmp_path/'source', c, sha); fixture = tmp_path/'fixture.bin'; fixture.write_bytes(data)
    stub = [sys.executable, '-B', '-c', 'import sys;sys.stdin.buffer.read();sys.stdout.buffer.write(open('+repr(str(fixture))+',"rb").read())']
    receipt = client.run_client(folder, sha, ssh_override=stub)
    claim = (folder/'ATTEMPT.json').read_bytes()
    assert receipt['transport_verified'] and not receipt['reference_process_started']
    with pytest.raises(FileExistsError): client.run_client(folder, sha, ssh_override=stub)
    second = client.run_client(folder, sha, download_only=True, ssh_override=stub)
    assert second['transport_verified'] and (folder/'ATTEMPT.json').read_bytes() == claim
    assert len(list(folder.glob('attended-*'))) == len(list(folder.glob('download-only-*'))) == 1
    assert not list(folder.glob('*/upload-stream.bin'))
