import copy
from pathlib import Path
import subprocess
import sys

import pytest

from models.calibration import calibration_record
from models.evaluation import model_identity
from models.evaluation_store import json_hash
from models.schedule import EpochEvaluator, EpochSchedule, ScheduledTraining
from models.study_artifacts import read_record, write_record
from models.training import TrainingSettings, runtime_contract
from test_calibration import candidates
from test_evaluation import tiny_masks
from test_training import assert_tree_equal, make_engine
from test_vjepa2 import model


ROWS = [dict(video_id=f'video-{v}', source_group_id=f'group-{v}', class_id=str(i % 2))
        for i, v in enumerate('abc')]
PROVENANCE = {'fixture': 'schedule-v1'}
SETTINGS = TrainingSettings(epochs=2, effective_batch=2, decay_after_epochs=(1,))


def create(original, root, *, resume=False, condition='clean-finetune', evaluator_class=EpochEvaluator,
           artifacts=None, monitor_count=1, calibration_override=None, schedule=None):
    calibration_rows = ROWS[:monitor_count]
    calibration = calibration_record(candidates(model_identity(original), calibration_rows,
                                     provenance=PROVENANCE), calibration_rows, diagnostic=True)
    if calibration_override:
        calibration_override(calibration)
    provenance = dict(checkpoint_sha256='original', training_rows=ROWS,
                      evaluation_provenance_sha256=json_hash(PROVENANCE), calibration_sha256=json_hash(calibration))
    engine = make_engine(copy.deepcopy(original), condition=condition, settings=SETTINGS, provenance=provenance)
    panel = type('Panel', (), {'panel': 'train_monitor', 'identity': {'data': 'fixture'},
        'rows': calibration_rows, 'pixels': lambda self, i: engine.load_example(i)[0]})()
    evaluator = evaluator_class({'train_monitor': panel}, tiny_masks, artifacts or root.parent / 'evaluations',
                               primary_step_size=calibration['chosen_step_size_255'], provenance=PROVENANCE)
    workflow = ScheduledTraining(engine, schedule or EpochSchedule(2, (2,), panels=('train_monitor',)), evaluator, root,
        calibration=calibration, calibration_rows=calibration_rows, diagnostic=True, resume=resume)
    return workflow


def test_schedule_declares_all_routine_epochs_initialization_and_primary_anchors():
    schedule = EpochSchedule(30, (1, 15, 16, 23, 24, 30))
    tasks = schedule.tasks()
    for panel in schedule.panels:
        assert [t['epoch'] for t in tasks if t['panel'] == panel and t['method'] == 'routine'] == list(range(31))
        assert [t['epoch'] for t in tasks if t['panel'] == panel and t['method'] == 'primary'] == [0, 1, 15, 16, 23, 24, 30]


@pytest.mark.parametrize('condition', ['clean-finetune', 'adversarial-finetune'])
def test_scheduled_interruption_resume_matches_unmonitored_trajectory_exactly(model, tmp_path, condition):
    workflow = create(model, tmp_path / 'trajectory', condition=condition)
    reference = make_engine(copy.deepcopy(model), condition=condition, settings=SETTINGS,
                            provenance=workflow.engine.contract['provenance'])
    while not reference.finished:
        reference.update()
    expected = reference.state()
    partial = workflow.run(max_updates=1, max_evaluations=2)
    assert partial['completed_step'] == 1 and partial['reason'] == 'update_limit'
    resumed = create(model, tmp_path / 'trajectory', condition=condition, resume=True)
    result = resumed.run(max_updates=4, max_evaluations=8)
    assert result['complete'] and result['completed_step'] == 4
    assert_tree_equal(expected, resumed.engine.state())
    assert resumed.store._manifest()['anchors'].keys() == {'2'}
    ledger = read_record(resumed.record_path)
    assert len(ledger['completed_evaluations']) == 5
    assert len({r['receipt']['path'] for r in ledger['completed_evaluations']}) == 5


def test_training_cannot_advance_while_initial_evaluation_is_pending(model, tmp_path):
    workflow = create(model, tmp_path / 'run')
    result = workflow.run(max_updates=8, max_evaluations=1)
    assert result['completed_step'] == 0 and result['reason'] == 'evaluation_limit'
    result = workflow.run(max_updates=0, max_evaluations=1)
    assert result['completed_step'] == 0 and result['reason'] == 'update_limit'
    assert len(read_record(workflow.record_path)['completed_evaluations']) == 2


def test_recovery_after_saved_epoch_before_evaluation_does_not_skip_boundary(model, tmp_path):
    workflow = create(model, tmp_path / 'run')
    def interrupt(event):
        if event['step'] == 2:
            raise RuntimeError('lost process after epoch checkpoint')
    with pytest.raises(RuntimeError, match='lost process'):
        workflow.run(max_updates=8, max_evaluations=8, on_update=interrupt)
    assert workflow.store.load()[0]['step'] == 2
    assert len(read_record(workflow.record_path)['completed_evaluations']) == 2
    resumed = create(model, tmp_path / 'run', resume=True)
    result = resumed.run(max_updates=0, max_evaluations=1)
    assert result['completed_step'] == 2
    assert resumed.record['completed_evaluations'][-1]['task']['epoch'] == 1


def test_lost_workflow_commit_reuses_completed_panel_without_reloading_pixels(model, tmp_path, monkeypatch):
    import models.schedule as module
    workflow = create(model, tmp_path / 'run')
    original_write = module.write_record
    def fail(*args, **kwargs):
        raise OSError('lost workflow commit')
    monkeypatch.setattr(module, 'write_record', fail)
    with pytest.raises(OSError, match='lost workflow'):
        workflow.run(max_updates=0, max_evaluations=1)
    assert not read_record(workflow.record_path)['completed_evaluations']
    monkeypatch.setattr(module, 'write_record', original_write)
    resumed = create(model, tmp_path / 'run', resume=True)
    def fail_load(i):
        raise AssertionError('Completed panel must be reused')
    resumed.evaluator.panels['train_monitor'].pixels = fail_load
    result = resumed.run(max_updates=0, max_evaluations=1)
    assert result['evaluations_this_call'] == 1


def test_recovery_inside_panel_restarts_at_next_unfinished_clip(model, tmp_path, monkeypatch):
    import models.schedule as module
    workflow = create(model, tmp_path / 'run', monitor_count=2)
    original_evaluate = module.evaluate_panel
    def interrupted(*args, **kwargs):
        def fail(i, r):
            raise RuntimeError('interrupted after clip')
        return original_evaluate(*args, on_clip=fail, **kwargs)
    monkeypatch.setattr(module, 'evaluate_panel', interrupted)
    with pytest.raises(RuntimeError, match='after clip'):
        workflow.run(max_updates=4, max_evaluations=1)
    monkeypatch.setattr(module, 'evaluate_panel', original_evaluate)
    resumed = create(model, tmp_path / 'run', resume=True, monitor_count=2)
    panel = resumed.evaluator.panels['train_monitor']
    original_pixels, loaded = panel.pixels, []
    def load(i):
        loaded.append(i)
        return original_pixels(i)
    panel.pixels = load
    resumed.run(max_updates=0, max_evaluations=1)
    assert loaded == [1]
    assert resumed.engine.step == 0


def test_shared_initialization_is_reused_across_matched_conditions(model, tmp_path):
    first = create(model, tmp_path / 'clean')
    first.run(max_updates=0, max_evaluations=2)
    second = create(model, tmp_path / 'adversarial', condition='adversarial-finetune')
    def fail(i):
        raise AssertionError('Initialization should be shared')
    second.evaluator.panels['train_monitor'].pixels = fail
    second.run(max_updates=0, max_evaluations=2)
    assert [r['receipt'] for r in first.record['completed_evaluations']] == [r['receipt'] for r in second.record['completed_evaluations']]


@pytest.mark.parametrize('defect', ['provenance', 'runtime', 'seed', 'checkpoint'])
def test_stale_calibration_refused_before_trajectory_directory_created(model, tmp_path, defect):
    def mutate(record):
        c = record['common_contract']
        if defect == 'provenance':
            c['provenance'] = {'changed': True}
        elif defect == 'runtime':
            c['runtime']['threads'] += 1
        elif defect == 'seed':
            c['base_seed'] += 1
        else:
            c['checkpoint']['file_sha256'] = 'another'
    with pytest.raises(ValueError, match='Calibration runtime'):
        create(model, tmp_path / 'run', calibration_override=mutate)
    assert not (tmp_path / 'run').exists()


def test_calibration_cache_identity_cannot_be_replaced_with_same_video_ids(model, tmp_path):
    def mutate(record):
        record['common_contract']['panel_identity'] = {'data': 'redecoded-differently'}
    with pytest.raises(ValueError, match='pixel-cache identity'):
        create(model, tmp_path / 'run', calibration_override=mutate)
    assert not (tmp_path / 'run').exists()


@pytest.mark.parametrize('defect', ['schedule', 'missing_receipt', 'changed_receipt', 'skipped_evaluation', 'mixed_run'])
def test_resume_rejects_changed_schedule_or_invalid_evaluation_ledger(model, tmp_path, defect):
    workflow = create(model, tmp_path / 'run')
    workflow.run(max_updates=2, max_evaluations=3)
    if defect == 'schedule':
        with pytest.raises(ValueError, match='contract changed'):
            create(model, tmp_path / 'run', resume=True, schedule=EpochSchedule(2, (1, 2), panels=('train_monitor',)))
        return
    record = read_record(workflow.record_path)
    receipt = Path(record['completed_evaluations'][-1]['receipt']['path'])
    if defect == 'missing_receipt':
        receipt.unlink()
    elif defect == 'changed_receipt':
        receipt.write_text(receipt.read_text() + ' ')
    elif defect == 'skipped_evaluation':
        record['completed_evaluations'] = record['completed_evaluations'][:1]
        write_record(workflow.record_path, record, replace=True)
    else:
        record['completed_evaluations'][-1]['checkpoint']['seed'] += 1
        write_record(workflow.record_path, record, replace=True)
    resumed = create(model, tmp_path / 'run', resume=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        resumed.run(max_updates=8, max_evaluations=8)


@pytest.mark.parametrize('mode', ['features', 'calibrate', 'train'])
def test_cli_refuses_research_mode_before_reading_data_or_model(tmp_path, mode):
    script = Path(__file__).resolve().parents[1] / 'scripts/check_workflow.py'
    result = subprocess.run([sys.executable, str(script), mode, '--checkpoint', 'missing', '--manifests', 'missing',
        '--clips', 'missing', '--output', str(tmp_path / 'output')], capture_output=True, text=True)
    assert result.returncode == 2 and 'Only --diagnostic' in result.stderr
    assert not (tmp_path / 'output').exists()
