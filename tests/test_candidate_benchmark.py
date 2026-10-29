"""Offline bounds, artifact barriers, and real tiny-model CPU workflow rehearsals."""

import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import signal
import sys

import pytest
import torch

from models.candidate_benchmark import PHASES, benchmark_proposal, launch_limits, logical_signature
from models.candidate_contract import binding, bind_provenance, ROOT
from models.candidate_evaluation import model_identity
from models.candidate_evaluation_checkpoint import load_anchor
from models.candidate_features import extract_features, verify_features
from models.checkpointing import CheckpointStore
from models.study_artifacts import read_record
from models.training import TrainingSettings
from test_evaluation import tiny_masks, clip
from test_training import make_engine
from test_vjepa2 import model


def valid_limits():
    return ({'kind': 'candidate_eot8_benchmark_launch_v1', 'candidate_binding': binding(), 'authorized': True,
        'diagnostic_only': True, 'required_phases': list(PHASES), 'execution_policy': 'shared-fp64-native',
        'container_gb': 30, 'persistent_gb': 80, 'volume_mount': '/workspace',
        'minimum_system_ram_gib': 40,
        'total_hourly_usd': '.61', 'inclusive_session_budget_usd': '1.22',
        'deployed_unix': 1000., 'deadline_unix': 8200.},
        {'reconciled': True, 'all_prior_pods_terminated': True, 'compute_cap_usd': 160,
         'total_cap_usd': 200, 'pilot_cap_usd': 10, 'noncompute_reserve_usd': 40,
         'pilot_spent_and_other_holds_usd': '6.10'})


def test_new_limits_count_setup_and_idle_and_keep_single_pilot_envelope():
    plan, ledger = valid_limits()
    seconds, rate, cap = launch_limits(plan, ledger, now=2200.)
    assert seconds == 5820. and rate == .61 and cap == 1.22
    assert launch_limits(plan, ledger, now=3200.)[0] == seconds - 1000
    assert benchmark_proposal()['unknowns']['primary_seconds'] is None
    assert not benchmark_proposal()['launch_authorized']


@pytest.mark.parametrize('defect', ['unapproved', 'stale_source', 'missing_phase', 'wrong_precision',
    'unreconciled', 'duplicate_storage', 'total_cap', 'pilot_overrun', 'underfunded',
    'too_long', 'expired', 'future_deployment', 'nan_rate', 'zero_rate', 'bool_rate'])
def test_launch_refuses_unreviewed_or_unbounded_work(defect):
    plan, ledger = valid_limits()
    if defect == 'unapproved': plan['authorized'] = False
    elif defect == 'stale_source': plan['candidate_binding']['source_sha256']['models/training.py'] = 'stale'
    elif defect == 'missing_phase': plan['required_phases'].pop()
    elif defect == 'wrong_precision': plan['execution_policy'] = 'native-fp32'
    elif defect == 'unreconciled': ledger['reconciled'] = False
    elif defect == 'duplicate_storage': ledger['all_prior_pods_terminated'] = False
    elif defect == 'total_cap': ledger['total_cap_usd'] = 300
    elif defect == 'pilot_overrun': ledger['pilot_spent_and_other_holds_usd'] = '9.5'
    elif defect == 'underfunded': plan['inclusive_session_budget_usd'] = '1.21'
    elif defect == 'too_long': plan['deadline_unix'] += 1
    elif defect == 'expired': plan['deadline_unix'] = 2300.
    elif defect == 'future_deployment': plan['deployed_unix'] = 3000.
    elif defect == 'nan_rate': plan['total_hourly_usd'] = 'NaN'
    elif defect == 'zero_rate': plan['total_hourly_usd'] = 0
    else: plan['total_hourly_usd'] = True
    with pytest.raises(ValueError):
        launch_limits(plan, ledger, now=2200.)


def test_signature_detects_dtype_type_single_bit_and_nonfinite():
    value = {'moments': torch.arange(4, dtype=torch.float32), 'order': (1, 2)}
    expected = logical_signature(value)
    assert expected == logical_signature(copy.deepcopy(value))
    for other in ({**value, 'moments': value['moments'].double()}, {**value, 'order': [1, 2]},
                  {**value, 'moments': value['moments'] + 1e-7}):
        assert logical_signature(other) != expected
    with pytest.raises(ValueError):
        logical_signature(torch.tensor(float('nan')))


@pytest.mark.parametrize('change', ['gpu', 'uuid', 'driver_version', 'missing'])
def test_cuda_artifacts_bind_physical_gpu_and_driver_without_using_cuda(monkeypatch, change):
    from models import candidate_contract as contract
    original = {'gpu': 'fixture-A40', 'uuid': 'GPU-fixture', 'driver_version': 'fixture-driver'}
    monkeypatch.setattr(contract, 'cuda_identity', lambda: original)
    provenance = contract.bind_provenance({}, cuda=True)
    contract.verify_runtime({'device': 'cuda'}, provenance)
    if change == 'missing':
        provenance.pop('candidate_cuda_identity')
    else:
        provenance['candidate_cuda_identity'] = {**original, change: 'different'}
    with pytest.raises(ValueError, match='not portable'):
        contract.verify_runtime({'device': 'cuda'}, provenance)


@pytest.mark.parametrize('defect', ['legacy', 'source', 'runtime', 'weights', 'expected_source'])
def test_feature_cache_refusal_before_use(model, clip, defect):
    panel = TinyPanel(clip.repeat(8, 1, 1, 1, 1))
    provenance = bind_provenance({'fixture': True})
    bundle = extract_features(model, panel, provenance=provenance)
    if defect == 'legacy': bundle['contract']['provenance'].pop('candidate_eot8')
    elif defect == 'source': bundle['contract']['provenance']['candidate_eot8']['source_sha256']['models/features.py'] = 'other'
    elif defect == 'runtime': bundle['contract']['runtime']['threads'] += 1
    elif defect == 'weights': bundle['contract']['encoder_sha256'] = 'other'
    else: provenance['fixture'] = False
    with pytest.raises(ValueError):
        verify_features(bundle, expected_provenance=provenance, model=model)


def test_legacy_anchor_and_resume_refused_before_loading_learned_tensors(model, tmp_path):
    old = make_engine(copy.deepcopy(model), settings=TrainingSettings(epochs=2, effective_batch=3, decay_after_epochs=(1,)),
                      provenance={'checkpoint_sha256': 'original'})
    old.update()
    store = CheckpointStore(tmp_path)
    store.save(old.state(), anchor_epoch=1)
    before = model_identity(model)
    expected = bind_provenance({'checkpoint_sha256': 'original'})
    with pytest.raises(ValueError):
        load_anchor(model, tmp_path, 1, initial_checkpoint_sha256='original', expected_provenance=expected)
    candidate = make_engine(model, settings=old.settings, provenance=expected)
    with pytest.raises(ValueError, match='Resume contract'):
        candidate.restore(old.state())
    assert model_identity(model) == before


def test_unapproved_cli_refuses_before_hardware_data_or_model_access(tmp_path):
    from models.checkpointing import file_sha256
    plan, ledger = valid_limits()
    plan['authorized'] = False
    ledger_path, plan_path = tmp_path / 'ledger.json', tmp_path / 'plan.json'
    ledger_path.write_text(json.dumps(ledger))
    plan['ledger_sha256'] = file_sha256(ledger_path)
    plan_path.write_text(json.dumps(plan))
    child = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/run_candidate_benchmark.py'),
        '--plan', str(plan_path), '--plan-sha256', file_sha256(plan_path), '--ledger', str(ledger_path),
        '--checkpoint', str(tmp_path / 'absent.pt'), '--output', str(tmp_path / 'run')],
        capture_output=True, text=True, timeout=30)
    assert child.returncode != 0 and 'no GPU work permitted' in child.stderr
    assert not (tmp_path / 'run').exists()


def test_independent_worker_alarm_ends_orphanable_work_without_parent_monitor():
    code = ('import sys,time; sys.path.insert(0,sys.argv[1]); '
            'from scripts.run_candidate_benchmark import arm_worker_deadline; '
            'arm_worker_deadline(.1); time.sleep(5)')
    child = subprocess.run([sys.executable, '-B', '-c', code, str(ROOT)], timeout=10)
    assert child.returncode == -signal.SIGALRM


@pytest.mark.parametrize('timeout', [False, True])
def test_cpu_parent_orchestration_and_timeout_never_initializes_cuda(tmp_path, monkeypatch, timeout):
    """Mock hardware/results ONLY to check parent dispatch and deadline behavior."""
    from scripts import run_candidate_benchmark as cli, run_workflow_pilot
    from models.checkpointing import file_sha256
    from models.study_artifacts import write_record
    plan, ledger = valid_limits()
    plan.update(expected_gpu='fixture GPU', expected_gpu_uuid='fixture UUID')
    ledger_path, plan_path = tmp_path / 'ledger.json', tmp_path / 'plan.json'
    ledger_path.write_text(json.dumps(ledger))
    plan['ledger_sha256'] = file_sha256(ledger_path)
    plan_path.write_text(json.dumps(plan))
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['candidate', '--plan', str(plan_path), '--plan-sha256', file_sha256(plan_path),
        '--ledger', str(ledger_path), '--checkpoint', '/absent/not-used', '--output', str(output)])
    monkeypatch.setattr(cli.time, 'time', lambda: 2200.)
    monkeypatch.setattr(cli.os, 'sysconf', lambda name: 4096 if name == 'SC_PAGE_SIZE' else 32 * 2**22)
    def read_only_nvidia(command, **kwargs):
        return 'fixture GPU, fixture UUID, 24576\n' if '--query-gpu=name,uuid,memory.total' in command else ''
    monkeypatch.setattr(cli.subprocess, 'check_output', read_only_nvidia)
    def forbidden(*args, **kwargs):
        raise AssertionError('Parent must stay CPU-only.')
    monkeypatch.setattr(torch.cuda, 'get_device_name', forbidden)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', forbidden)
    phases = []
    def cpu_dispatch_fixture(command, log, seconds):
        phase = command[command.index('--phase') + 1]
        phases.append(phase)
        assert 0 < seconds <= 5820
        if timeout:
            raise TimeoutError('fixture deadline')
        result = dict(phase=phase, candidate_binding=binding(), fixture_only=True)
        if phase.endswith('/reference'):
            result['exact_recovery'] = True
            write_record(output / ('recovery-' + phase.split('/')[0] + '.json'),
                         {'exact_match': True, 'fixture_only': True})
        write_record(output / 'phase-results' / (phase.replace('/', '-') + '.json'), result)
    monkeypatch.setattr(run_workflow_pilot, 'run_child', cpu_dispatch_fixture)
    if timeout:
        with pytest.raises(TimeoutError, match='fixture deadline'):
            cli.main()
        assert phases == ['preflight']
        assert read_record(output / 'session.json')['complete'] is False
    else:
        cli.main()
        assert phases == list(PHASES)
        assert read_record(output / 'session.json')['complete'] is True


class TinyPanel:
    panel = 'train'
    def __init__(self, pixels, indices=None):
        self.data = pixels
        self.indices = indices if indices is not None else list(range(8))
        self.rows = [dict(video_id=f'fixture-{i}', source_group_id=f'group-{i}',
                          class_id=str(i % 2), class_name=f'class-{i % 2}') for i in self.indices]
        self.entries = [{'fixture': i} for i in self.indices]
        self.paths = [Path(f'/fixture/{i}') for i in self.indices]
        self.identity = {'fixture': 'candidate-benchmark-v1'}
    def pixels(self, index):
        # panel_view copies rows/paths; the first selected representative is
        # resolved by its video ID rather than the original positions.
        i = int(self.rows[index]['video_id'].split('-')[-1])
        return self.data[i:i+1]


def cpu_phase(output, phase, upstream):
    """The full phase implementation, real tiny transformers, synthetic pixels.

    This explicitly bypasses CUDA preflight and launch approval for CPU testing;
    it never creates a qualified GPU session or a usable launch authorization.
    """
    from models.vjepa2 import use_upstream
    from models import workflow_diagnostic
    from models.candidate_benchmark import run_phase
    use_upstream(upstream)
    torch.set_num_threads(2)
    original_factory = workflow_diagnostic.prepare_context
    def fixture_context(root, args):
        instance = model.__wrapped__(upstream)
        pixels = .2 + .6 * torch.rand((8, 3, 4, 16, 16), generator=torch.Generator().manual_seed(22))
        training, monitor = TinyPanel(pixels), TinyPanel(pixels, [0])
        monitor.panel = 'train_monitor'
        return SimpleNamespace(model=instance, training=training, monitor=monitor, sampler=tiny_masks,
            settings=TrainingSettings(), checkpoint=dict(epoch=0, condition='pretrained', seed=None,
                training_contract_sha256=None, file_sha256='original'),
            provenance={'checkpoint_sha256': 'original', 'fixture_only': True})
    workflow_diagnostic.prepare_context = fixture_context
    try:
        run_phase(ROOT, SimpleNamespace(output=Path(output), phase=phase, device='cpu'))
    finally:
        workflow_diagnostic.prepare_context = original_factory


def test_tiny_cpu_phases_with_fresh_process_recovery_for_both_conditions(tmp_path, upstream):
    cpu_phase(tmp_path, 'calibration-bundle', upstream)
    cpu_phase(tmp_path, 'primary-pretrained', upstream)
    for condition in ('clean-finetune', 'adversarial-finetune'):
        for segment in ('first', 'resume', 'reference'):
            code = ('import sys; sys.path[:0]=[sys.argv[1],sys.argv[1]+"/tests"]; '
                    'from test_candidate_benchmark import cpu_phase; '
                    'cpu_phase(sys.argv[2],sys.argv[3],sys.argv[4])')
            child = subprocess.run([sys.executable, '-B', '-c', code, str(ROOT), str(tmp_path),
                condition + '/' + segment, str(upstream)], capture_output=True, text=True, timeout=120)
            assert child.returncode == 0, child.stdout + child.stderr
        comparison = read_record(tmp_path / ('recovery-' + condition + '.json'))
        assert comparison['exact_match'] and comparison['compared_leaves'] > 100
        assert comparison['logical_signatures'][0] == comparison['logical_signatures'][1]
    for phase in ('primary-trained', 'controls', 'features-initial', 'features-trained'):
        cpu_phase(tmp_path, phase, upstream)
    assert len(list((tmp_path / 'phase-results').glob('*.json'))) == len(PHASES) - 1
    assert read_record(tmp_path / 'phase-results/controls.json')['convergence']['records'][0]['settings']['attack_masks'] == 8
