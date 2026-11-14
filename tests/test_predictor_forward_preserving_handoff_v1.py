"""New packaged science and bounded-control tests; no target/paid work."""
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from models import predictor_forward_preserving_operator_v1 as op
from models import predictor_forward_preserving_client_v1 as client
from models import predictor_forward_preserving_records_v1 as records

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ['VJEPA_FORWARD_TARGET_EVIDENCE'])
HANDOFF = Path(os.environ['VJEPA_FORWARD_TARGET_HANDOFF'])


@pytest.fixture(scope='session')
def control():
    return json.loads((HANDOFF/'CONTROL.json').read_text())


@pytest.fixture(scope='session')
def cohort(control):
    # Use the actual prepared source overlay and unchanged 99-member input packet.
    parent = RAW/'packaged-CPU-cohort'; parent.mkdir(mode=0o700)
    overlay, code = parent/'overlay', parent/'code'
    for name in op.REMOTE_NAMES:
        data = op.INIT if name == 'models/__init__.py' else (ROOT/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == control['remote_source_sha256'][name]
        p = overlay/name; p.parent.mkdir(mode=0o700, parents=True, exist_ok=True); p.write_bytes(data)
    op.extract(ROOT/control['archive_path'], control['archive_manifest'], code)
    results = []
    for index in range(4):
        output = parent/('worker-'+str(index))
        kwargs = dict(output=str(output), index=index, tiny=True, device='cpu', upstream=str(ROOT/'external/vjepa2'),
                      checkpoint=None, packet=str(code), proposal=str(overlay/op.CONTRACT), control_sha256='c'*64,
                      reference=str(parent/('worker-'+str(index-1))) if index in (1, 3) else None,
                      replay_reference=str(parent/('worker-'+str(index-2))) if index in (2, 3) else None)
        program = 'import sys;sys.path[:0]='+repr([str(overlay), str(code)])+';from types import SimpleNamespace;from models.predictor_forward_preserving_worker_v1 import run;run(SimpleNamespace(**'+repr(kwargs)+'))'
        with (parent/('worker-'+str(index)+'.log')).open('xb') as log:
            result = subprocess.run([sys.executable, '-I', '-B', '-c', program], stdout=log, stderr=subprocess.STDOUT, timeout=45)
        assert result.returncode == 0, (parent/('worker-'+str(index)+'.log')).read_text()[-9000:]
        state = records.load(output, control_sha256='c'*64, index=index, live=False)
        results.append((output, state))
    return results


def test_four_fresh_packaged_workers_one_EOT8_each_and_conditional_exact_replay(cohort):
    from models.predictor_forward_preserving_worker_v1 import arrays_from, compare_arrays
    assert len({s['pid'] for p, s in cohort}) == 4
    assert all(s['complete'] and s['passed'] and s['model_unchanged'] for p, s in cohort)
    for index, (folder, state) in enumerate(cohort):
        actual = arrays_from(folder, state)
        assert len(actual) == 36 and len([k for k in actual if k.startswith('prediction-')]) == 16
        if index in (1, 3):
            assert compare_arrays(actual, arrays_from(*cohort[index-1]))['input']['passed']
        if index in (2, 3):
            assert compare_arrays(actual, arrays_from(*cohort[index-2]), replay=True) == {}
        if index % 2:
            assert state['ownership']['custom_VJPs'] == 32 and state['ownership']['predictor_forward_cap'] == 64
        else:
            assert state['ownership']['custom_VJPs'] == 0
    out = {'passed': True, 'fresh_processes': 4, 'EOT8_passes': 4, 'profiles_attacks_or_optimizer_steps': 0,
           'artifacts': 144, 'FP32_arrays': 140, 'int64_mask_arrays': 4, 'reference_controls_and_exact_replays': True,
           'CUDA_qualification': False}
    (RAW/'cohort-check.json').write_text(json.dumps(out, sort_keys=True, indent=2)+'\n')


@pytest.mark.parametrize('fault', ('marker', 'native-proof', 'VJP-proof', 'scope-proof', 'tensor', 'wrong-index', 'live-domain'))
def test_cold_receipt_refuses_incomplete_altered_or_wrong_domain(cohort, tmp_path, fault):
    source, _ = cohort[1]; folder = tmp_path/'copy'; shutil.copytree(source, folder)
    if fault == 'marker':
        (folder/'success.json').unlink()
    elif fault in ('native-proof', 'VJP-proof', 'scope-proof'):
        file = {'native-proof': 'native-calls.jsonl', 'VJP-proof': 'VJP-calls.jsonl', 'scope-proof': 'scopes.json'}[fault]
        with (folder/file).open('ab') as f:
            f.write(b' ')
    elif fault == 'tensor':
        with (folder/'input.npy').open('ab') as f:
            f.write(b'x')
    with pytest.raises((ValueError, FileNotFoundError)):
        records.load(folder, control_sha256='c'*64, index=0 if fault == 'wrong-index' else 1, live=fault == 'live-domain')


def test_changed_cotangent_refuses_even_when_loss_input_unchanged(cohort):
    from models.predictor_forward_preserving_worker_v1 import arrays_from, compare_arrays
    original = arrays_from(*cohort[0]); modified = {k: a.copy() for k, a in original.items()}
    a = modified['cotangent-0']; a.flat[0] = np.nextafter(a.flat[0], np.float32(np.inf))
    with pytest.raises(ValueError, match='cotangent'):
        compare_arrays(modified, original)


def test_gradient_within_numerical_bound_still_requires_exact_fresh_replay(cohort):
    from models.predictor_forward_preserving_worker_v1 import arrays_from, compare_arrays
    original = arrays_from(*cohort[0]); modified = {k: a.copy() for k, a in original.items()}
    modified['input'].flat[0] = np.nextafter(modified['input'].flat[0], np.float32(np.inf))
    assert compare_arrays(modified, original)['input']['passed']
    with pytest.raises(ValueError, match='replay'):
        compare_arrays(modified, original, replay=True)


def test_input_rejection_keeps_original_bound(cohort):
    from models.predictor_forward_preserving_worker_v1 import arrays_from, compare_arrays
    original = arrays_from(*cohort[0]); modified = {k: a.copy() for k, a in original.items()}
    modified['input'].flat[0] += np.float32(.01)
    with pytest.raises(ValueError, match='numerical gate'):
        compare_arrays(modified, original)


def test_frozen_math_code_is_reused_and_observation_preserves_layout():
    import torch
    from models import predictor_forward_preserving_target_v1 as target
    from models import predictor_forward_preserving_cpu_v1 as cpu
    assert target._public_math_backward.__code__ is cpu._OriginalForwardMathVJP.backward.__code__
    t = torch.arange(64, dtype=torch.float32).reshape(2, 4, 8).transpose(0, 2)
    before = cpu.fingerprint(t); after = target.fingerprint(t)
    assert {k: v for k, v in after.items() if k != 'device'} == before
    assert after['device'] == 'cpu'


@pytest.mark.parametrize('host,tiny,device', [('linux10603', True, 'cpu'), ('linux10609', True, 'cpu'),
                                             ('local-only', False, 'cuda'), ('linux10609', False, 'cuda')])
def test_wrong_worker_domain_refused_before_output(monkeypatch, tmp_path, host, tiny, device):
    from models import predictor_forward_preserving_worker_v1 as worker
    monkeypatch.setattr(worker.platform, 'node', lambda: host)
    args = SimpleNamespace(index=0, tiny=tiny, device=device, output=tmp_path/'must-not-exist')
    with pytest.raises(ValueError, match='domain'):
        worker.run(args)
    assert not args.output.exists()


def test_candidate_without_fresh_original_refuses_before_output(tmp_path):
    from models.predictor_forward_preserving_worker_v1 import run
    args = SimpleNamespace(index=1, tiny=True, device='cpu', reference=None, replay_reference=None, control_sha256='c'*64, output=tmp_path/'absent')
    with pytest.raises(ValueError, match='before scientific imports'):
        run(args)
    assert not args.output.exists()


def test_unarmed_client_refuses_before_provenance_claim_or_transport(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError('Unarmed client reached provenance/transport.')
    monkeypatch.setattr(client, 'check_local', forbidden)
    with pytest.raises(ValueError, match='Unarmed'):
        client.run(tmp_path, popen=forbidden)
    assert list(tmp_path.iterdir()) == []


def test_exact_control_and_actual_local_binding(control):
    op.validate_control(control)
    c, sha = client.check_local(HANDOFF)
    assert c == control and sha == op.old.digest(HANDOFF/'CONTROL.json')


def test_copied_handoff_cannot_reset_claim(control, tmp_path):
    copy_to = tmp_path/'copied'; copy_to.mkdir()
    for name in ('CONTROL.json', 'known-hosts'):
        shutil.copy2(HANDOFF/name, copy_to/name)
    with pytest.raises(ValueError, match='Exact local handoff'):
        client.check_local(copy_to)


@pytest.mark.parametrize('fault', ('old-kind', 'more-time', 'more-attempts', 'retries', 'different-host', 'reset-claim'))
def test_expanded_or_old_authorization_refused(control, fault):
    sha = op.old.digest(HANDOFF/'CONTROL.json'); a = op.authorization_template(control, sha, 'test fixture only; never written as authorization')
    if fault == 'old-kind': a['kind'] = op.envelope.KIND
    elif fault == 'more-time': a['resource'] = {**a['resource'], 'work_seconds': 2401}
    elif fault == 'more-attempts': a['maximum_new_attempts'] = 2
    elif fault == 'retries': a['automatic_retries'] = 1
    elif fault == 'different-host': a['host'] = 'linux10609'
    else: a['reset_old_claims'] = True
    with pytest.raises(ValueError, match='new exact scoped'):
        op.validate_authorization(a, sha, control)


@pytest.mark.parametrize('name,cap', [('loss.npy', 65536), ('input.npy', 16*2**20), ('targets.npy', 12*2**20),
                                     ('VJP-calls.jsonl', 8*2**20), ('worker.json', 256*2**10), ('success.json', 16*2**10)])
def test_every_new_record_and_partial_has_explicit_file_cap(tmp_path, name, cap):
    folder = tmp_path/'private'; folder.mkdir(mode=0o700)
    path = folder/(name+'.partial')
    with path.open('xb') as f:
        f.truncate(cap+1)
    with pytest.raises(ValueError, match='file cap'):
        op.inventory(folder, running=True)


def test_NPY_is_raw_and_graph_records_are_metadata(tmp_path):
    folder = tmp_path/'private'; folder.mkdir(mode=0o700)
    (folder/'results/0').mkdir(mode=0o700, parents=True)
    (folder/'results/0/prediction-0.npy').write_bytes(b'x'*4096)
    (folder/'results/0/VJP-calls.jsonl').write_bytes(b'{}\n')
    st = tmp_path.stat(); identity = {'device': st.st_dev, 'inode': st.st_ino}
    got = op.storage(folder, identity, parent=tmp_path)
    assert got['categories']['raw'] == 4096 and got['categories']['metadata'] == 3


def test_bootstrap_cold_import_refuses_without_scientific_imports_or_target_work(control):
    body = client.payload(control).decode()
    guard = "import importlib.abc,sys\nclass NoTorch(importlib.abc.MetaPathFinder):\n def find_spec(self,name,path=None,target=None):\n  if name.split('.')[0] in ('torch','numpy','timm'):raise RuntimeError('SCIENTIFIC IMPORT FORBIDDEN')\nsys.meta_path.insert(0,NoTorch())\n"
    result = subprocess.run([sys.executable, '-I', '-B', '-c', guard+body], input=b'{}\n', capture_output=True, timeout=10)
    (RAW/'cold-bootstrap-refusal.log').write_bytes(result.stdout+result.stderr)
    assert result.returncode != 0
    assert b'SCIENTIFIC IMPORT FORBIDDEN' not in result.stderr
    assert b'new exact scoped' in result.stderr


def test_empty_collection_response_is_explicit_and_preserved(control, tmp_path):
    response = tmp_path/'response.bin'; response.write_bytes(b'')
    with pytest.raises(ValueError, match='No collection response'):
        client.unpack(response, tmp_path/'saved', control, 'c'*64)
    assert response.exists() and not (tmp_path/'saved').exists()


def test_same_SSH_stream_roundtrip_retains_bytes_modes_times(control, tmp_path):
    folder = tmp_path/'source'; folder.mkdir(mode=0o700)
    p = folder/'partial-note.log'; p.write_bytes(b'partial failure evidence\n'); p.chmod(0o600)
    sha = 'c'*64
    response = io.BytesIO(); response.write(op.old.json_bytes({'kind': op.KIND, 'phase': 'scratch-created', 'control_sha256': sha}))
    op.stream(folder, sha, response)
    binary = tmp_path/'response.bin'; binary.write_bytes(response.getvalue())
    out = tmp_path/'saved'; client.unpack(binary, out, control, sha)
    assert (out/p.name).read_bytes() == p.read_bytes() and (out/p.name).stat().st_mtime_ns == p.stat().st_mtime_ns
    assert (out/p.name).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('failure_index', (None, 0, 1, 2, 3))
def test_serial_supervisor_stops_on_failure_and_collects_without_retry(control, tmp_path, monkeypatch, failure_index):
    # Hardware, allocation proof, worker processes and clocks are explicitly simulated.
    calls = []; checked = []; streamed = []
    parent = tmp_path/'scratch'; parent.mkdir(mode=0o700); st = parent.stat()
    monkeypatch.setattr(op, 'readiness', lambda c: {'parent': str(parent), 'device': st.st_dev, 'inode': st.st_ino})
    monkeypatch.setattr(op.base, 'gpu_gate', lambda *a: {'simulated_GPU': True})
    monkeypatch.setattr(op, 'storage', lambda *a, **k: {'simulated_storage': True})
    monkeypatch.setattr(op.base.primitives, 'validate_proof', lambda *a: calls.append('proof-validated'))
    monkeypatch.setattr(op.old, 'receive_archive', lambda *a: calls.append('archive-received'))
    monkeypatch.setattr(op, 'extract', lambda *a: calls.append('archive-extracted'))
    monkeypatch.setattr(op.signal, 'setitimer', lambda *a: None)
    monkeypatch.setattr(op.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(op, 'stream', lambda folder, sha, out: streamed.append(json.loads((folder/'session.json').read_text())))
    def supervise(command, folder, *, role, deadline, gate, file_limit, **kw):
        calls.append(role); gate()
        if role == 'proof':
            assert file_limit == 3*op.GiB
        else:
            assert file_limit == 16*op.MiB
        return {'exit_status': 1 if role == 'worker-'+str(failure_index) else 0}
    monkeypatch.setattr(op.base.primitives, 'supervise', supervise)
    def accept(folder, index, **kw):
        checked.append(index)
        return {'pid': 1000+index, 'index': index}
    monkeypatch.setattr(op, 'check_worker', accept)
    sources = {n: op.INIT if n == 'models/__init__.py' else (ROOT/n).read_bytes() for n in op.REMOTE_NAMES}
    start = op.time.monotonic(); rc = op._execute(control, {}, io.BytesIO(), io.BytesIO(), sources, 'c'*64, start, start+2400, lambda *a: None)
    expected = 4 if failure_index is None else failure_index+1
    assert [r for r in calls if r.startswith('worker-')] == ['worker-'+str(i) for i in range(expected)]
    assert calls.index('proof-validated') < calls.index('archive-received')
    assert len(streamed) == 1 and streamed[0]['passed'] is (failure_index is None)
    assert rc == (0 if failure_index is None else 1)
    if failure_index == 1:
        assert 2 not in checked and 3 not in checked  # No conditional replay after the initial candidate fails.
