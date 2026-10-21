import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from models.checkpointing import CheckpointStore
from models.pilot_measurement import Timings, compare_recovery, compare_trees
from models.study_artifacts import read_record, write_record
from models.training import TrainingSettings
from models.workflow_pilot import CONDITIONS, MeasuredEvaluator, run_phase
from test_evaluation import tiny_masks
from test_schedule import create
from test_training import assert_tree_equal
from test_vjepa2 import model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pilot_fixture(model, tmp_path, monkeypatch):
    rows = [dict(video_id=f'v{i}', source_group_id=f'g{i}', class_id=str(i % 2)) for i in range(8)]
    pixels = .2 + .6 * torch.rand(8, 3, 4, 16, 16, generator=torch.Generator().manual_seed(17))
    training = SimpleNamespace(panel='train', rows=rows, identity={'panel': 'train', 'data': 'fixture'},
                               pixels=lambda i: pixels[i:i+1])
    monitor = SimpleNamespace(panel='train_monitor', rows=rows[:1], identity={'panel': 'train_monitor', 'data': 'fixture'},
                              pixels=lambda i: pixels[i:i+1])
    context = SimpleNamespace(model=model, training=training, monitor=monitor, sampler=tiny_masks,
        settings=TrainingSettings(), plan={'evaluation': {'primary_epochs': [1, 15, 16, 23, 24, 30], 'mask_seed': 20261020}},
        checkpoint={'epoch': 0, 'condition': 'pretrained', 'seed': None,
                    'training_contract_sha256': None, 'file_sha256': 'original'},
        provenance={'checkpoint_sha256': 'original', 'fixture': 'workflow-pilot'})
    monkeypatch.setattr('models.workflow_pilot.prepare_context', lambda root, args: copy.deepcopy(context))
    args = SimpleNamespace(device='cpu', output=tmp_path / 'pilot', phase='calibrate')
    run_phase(ROOT, args)
    return args


@pytest.mark.parametrize('condition', CONDITIONS)
def test_full_pilot_recovery_phases_match_weights_optimizer_rng_and_evaluations(pilot_fixture, condition):
    args = pilot_fixture
    for segment in ('split-first', 'split-resume', 'reference'):
        args.phase = condition + '/' + segment
        result = run_phase(ROOT, args)
        if segment == 'split-first':
            assert result['expected_pause_after_saved_step'] == 1
            # Interruption actually falls after checkpoint commit and before
            # the first trained epoch's evaluation, not after the whole epoch.
            ledger = read_record(args.output / condition / 'split/workflow.json')
            assert len(ledger['completed_evaluations']) == 2
    comparison = compare_recovery(args.output / condition / 'reference', args.output / condition / 'split')
    assert comparison['exact_match'] and comparison['compared_leaves'] > 100
    assert not comparison['research_result']
    # A validly reserialized optimizer mutation must fail logical equality.
    store = CheckpointStore(args.output / condition / 'split/checkpoints', create=False)
    state, _ = store.load()
    state['optimizer']['state'][0]['exp_avg'].add_(1e-5)
    manifest = store._manifest()
    manifest['latest'] = store._atomic('tampered.pt', state)
    store._atomic('manifest.json', manifest, json_file=True)
    assert not compare_recovery(args.output / condition / 'reference', args.output / condition / 'split')['exact_match']


def test_pilot_features_trained_anchor_and_controls_have_real_artifacts(pilot_fixture):
    args = pilot_fixture
    for phase in ('features-initial', 'adversarial-finetune/split-first',
                  'adversarial-finetune/split-resume', 'features-trained', 'controls'):
        args.phase = phase
        result = run_phase(ROOT, args)
        assert not result['research_result'] and not result['budget_approved']
    for phase in ('features-initial', 'features-trained'):
        result = read_record(args.output / 'phase-results' / (phase + '.json'))
        assert not result['held_out_accuracy_measured']
        assert (args.output / phase / 'features.pt').is_file()
    convergence = read_record(args.output / 'controls/convergence/evaluation.json')
    assert convergence['complete'] and convergence['contract']['sensitivity_banks'] == 2


def test_measuring_training_preserves_state_and_records_failures(model, tmp_path):
    from test_training import make_engine
    reference = make_engine(copy.deepcopy(model))
    reference.update()
    expected = reference.state()
    measured = make_engine(copy.deepcopy(model))
    timer = Timings(tmp_path / 'times.jsonl', 'cpu')
    timer.instrument(measured, ('update', 'state'), 'engine')
    measured.update()
    assert_tree_equal(expected, measured.state())
    with pytest.raises(RuntimeError):
        with timer.span('broken'):
            raise RuntimeError('fixture')
    records = [json.loads(line) for line in timer.path.read_text().splitlines()]
    assert records[-1]['status'] == 'failed' and records[-1]['seconds'] >= 0


def test_cached_evaluation_does_not_claim_reused_gpu_seconds(model, tmp_path):
    timer = Timings(tmp_path / 'times.jsonl', 'cpu')
    def factory(*args, **kwargs):
        return MeasuredEvaluator(*args, timings=timer, **kwargs)
    first = create(copy.deepcopy(model), tmp_path / 'first', evaluator_class=factory)
    first.run(max_updates=0, max_evaluations=1)
    second = create(copy.deepcopy(model), tmp_path / 'second', evaluator_class=factory)
    second.run(max_updates=0, max_evaluations=1)
    records = [json.loads(line) for line in timer.path.read_text().splitlines()]
    assert records[0]['new_clips'] == 1 and records[0]['new_native_seconds'] > 0
    assert records[1]['new_clips'] == 0 and records[1]['new_native_seconds'] == 0
    assert records[1]['reused_clips'] == 1


@pytest.mark.parametrize('defect', ['wrong_step', 'missing_evaluation', 'corrupt_receipt'])
def test_recovery_comparison_rejects_incomplete_evidence(model, tmp_path, defect):
    workflow = create(model, tmp_path / 'run')
    workflow.run(max_updates=2, max_evaluations=3)
    ledger = read_record(workflow.record_path)
    if defect == 'missing_evaluation':
        ledger['completed_evaluations'].pop()
        write_record(workflow.record_path, ledger, replace=True)
    elif defect == 'corrupt_receipt':
        Path(ledger['completed_evaluations'][0]['receipt']['path']).write_text('{}')
    with pytest.raises((ValueError, KeyError)):
        compare_recovery(workflow.root, workflow.root, expected_step=3 if defect == 'wrong_step' else 2)


@pytest.mark.parametrize('changed', [torch.tensor([float('nan')]), torch.tensor([float('inf')]),
                                   torch.tensor([1.0001]), torch.tensor([1.], dtype=torch.float64)])
def test_exact_comparison_rejects_nonfinite_and_small_differences(changed):
    assert not compare_trees(torch.tensor([1.]), changed)['exact_match']


def test_worker_deadline_terminates_a_real_subprocess(tmp_path):
    spec = importlib.util.spec_from_file_location('pilot_runner', ROOT / 'scripts/run_workflow_pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    marker = tmp_path / 'should-not-exist'
    program = 'import time,pathlib; time.sleep(10); pathlib.Path(' + repr(str(marker)) + ').touch()'
    with pytest.raises(subprocess.TimeoutExpired):
        module.run_child([sys.executable, '-c', program], tmp_path / 'worker.log', .1)
    assert not marker.exists()


def test_cli_refuses_non_diagnostic_before_model_or_output(tmp_path):
    output = tmp_path / 'not-created'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_workflow_pilot.py'),
        '--checkpoint', 'missing', '--clips', 'missing', '--manifests', 'missing', '--output', str(output)],
        capture_output=True, text=True)
    assert result.returncode == 2 and '--diagnostic' in result.stderr
    assert not output.exists()
