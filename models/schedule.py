"""Train/evaluate epoch barriers with recoverable progress and shared initialization."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

from models.calibration import verify_calibration
from models.checkpointing import CheckpointStore, file_sha256
from models.evaluation import EvaluationSettings, SEED_SCHEMA, model_identity
from models.evaluation_store import EvaluationStore, evaluate_panel, json_hash, validate_records
from models.study_artifacts import read_record, write_record


@dataclass(frozen=True)
class EpochSchedule:
    epochs: int
    primary_epochs: tuple
    panels: tuple = ('train_monitor', 'exploration_validation')

    def __post_init__(self):
        if (type(self.epochs) is not int or self.epochs < 1
                or self.primary_epochs != tuple(sorted(set(self.primary_epochs)))
                or not self.primary_epochs or self.primary_epochs[-1] != self.epochs
                or any(type(e) is not int or not 1 <= e <= self.epochs for e in self.primary_epochs)
                or not self.panels or len(set(self.panels)) != len(self.panels)
                or any(p not in ('train_monitor', 'exploration_validation') for p in self.panels)):
            raise ValueError('Invalid exploration epoch schedule.')

    def tasks(self):
        tasks = []
        for epoch in range(self.epochs + 1):
            methods = ('routine', 'primary') if epoch == 0 or epoch in self.primary_epochs else ('routine',)
            for panel in self.panels:
                for method in methods:
                    tasks.append({'key': f'epoch-{epoch:03d}/{panel}/{method}', 'epoch': epoch,
                                  'panel': panel, 'method': method})
        return tasks


class EpochEvaluator:
    """Reuse complete initialization evaluations only for identical contracts."""

    def __init__(self, panels, masks_for, output, *, primary_step_size, provenance, base_seed=20261020):
        if set(panels) - {'train_monitor', 'exploration_validation'} or not panels:
            raise ValueError('Only declared exploration panels may be evaluated.')
        for name, panel in panels.items():
            if name != panel.panel:
                raise ValueError('Panel labels disagree.')
        self.panels, self.masks_for, self.root = panels, masks_for, Path(output).resolve()
        self.primary_step_size, self.provenance, self.base_seed = primary_step_size, provenance, base_seed
        self.binding = {'panels': {name: {'identity': panel.identity, 'rows': panel.rows} for name, panel in panels.items()},
                        'primary_step_size_255': primary_step_size, 'provenance': provenance,
                        'base_seed': base_seed, 'seed_schema': SEED_SCHEMA}

    def settings(self, task):
        return EvaluationSettings.for_method(task['method'], self.primary_step_size if task['method'] == 'primary' else 1.)

    def __call__(self, model, task, checkpoint):
        panel = self.panels[task['panel']]
        settings = self.settings(task)
        key = json_hash({'checkpoint': checkpoint, 'panel': panel.panel, 'panel_identity': panel.identity,
                         'ids': [r['video_id'] for r in panel.rows], 'settings': asdict(settings),
                         'provenance': self.provenance, 'base_seed': self.base_seed})
        directory = self.root / key
        store = EvaluationStore(directory, resume=directory.exists())
        evaluate_panel(model, panel, self.masks_for, settings, store,
                        checkpoint=checkpoint, provenance=self.provenance, base_seed=self.base_seed)
        return {'path': str(directory / 'evaluation.json'), 'sha256': file_sha256(directory / 'evaluation.json')}

    def validate_receipt(self, receipt, task, checkpoint, *, identity=None):
        path = Path(receipt['path']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Evaluation receipt lies outside its declared artifact directory.')
        payload = read_record(path, expected_sha256=receipt['sha256'])
        validate_records(payload)
        c, panel = payload['contract'], self.panels[task['panel']]
        if (not payload['complete'] or c['checkpoint'] != checkpoint or c['panel'] != panel.panel
                or c['panel_identity'] != panel.identity or c['settings'] != asdict(self.settings(task))
                or c['video_ids'] != [r['video_id'] for r in panel.rows]
                or c['source_group_ids'] != [r['source_group_id'] for r in panel.rows]
                or c['provenance'] != self.provenance or c['base_seed'] != self.base_seed
                or c['seed_schema'] != SEED_SCHEMA or c['sensitivity_banks'] != 0
                or (identity is not None and c['weights_sha256'] != identity)):
            raise ValueError('Completed evaluation does not match its scheduled checkpoint and panel.')
        return payload


class ScheduledTraining:
    """Evaluation must finish at saved boundaries before the next update.

    In-place recovery is explicit, checksum-bound continuation of one trajectory.
    New trajectories/reruns still need a fresh directory. The older train.py
    diagnostic's fresh-attempt recovery behavior is unchanged.
    """

    def __init__(self, engine, schedule, evaluator, directory, *, calibration, calibration_rows,
                 diagnostic=False, resume=False):
        if engine.step != 0:
            raise ValueError('Construct a fresh engine; the workflow owns checkpoint recovery.')
        if schedule.epochs != engine.settings.epochs or set(schedule.panels) != set(evaluator.panels):
            raise ValueError('Training settings, schedule, and evaluation panels disagree.')
        if not diagnostic and schedule.panels != ('train_monitor', 'exploration_validation'):
            raise ValueError('Full exploration requires both train-monitor and validation panels.')
        original_identity = model_identity(engine.model)
        verify_calibration(calibration, original_identity=original_identity,
                           expected_rows=calibration_rows, diagnostic=diagnostic)
        if evaluator.primary_step_size != calibration['chosen_step_size_255']:
            raise ValueError('Scheduled primary attack step differs from calibration.')
        common = calibration['common_contract']
        if (common['runtime'] != engine.contract['runtime']
                or common['provenance'] != evaluator.provenance
                or common['base_seed'] != evaluator.base_seed
                or common['checkpoint']['file_sha256'] != engine.contract['provenance']['checkpoint_sha256']
                or engine.contract['provenance']['evaluation_provenance_sha256'] != json_hash(evaluator.provenance)):
            raise ValueError('Calibration runtime, source provenance, seeds, or checkpoint pin changed.')
        training_rows = engine.contract['provenance']['training_rows']
        training_by_id = {r['video_id']: r for r in training_rows}
        if [r['video_id'] for r in training_rows] != engine.ids:
            raise ValueError('Training identities disagree with their recorded rows.')
        monitor = evaluator.panels['train_monitor']
        monitor_by_id = {r['video_id']: r for r in monitor.rows}
        if (any(training_by_id.get(r['video_id']) != r for r in monitor.rows)
                or any(monitor_by_id.get(r['video_id']) != r for r in calibration_rows)):
            raise ValueError('Monitor/calibration rows must belong to this training set.')
        if not diagnostic and calibration_rows != monitor.rows[:10]:
            raise ValueError('Research calibration must use the first ten declared monitor rows.')
        subset_keys = {'ordered_ids_sha256', 'diagnostic_selection', 'calibration_selection'}
        if ({k: v for k, v in common['panel_identity'].items() if k not in subset_keys}
                != {k: v for k, v in monitor.identity.items() if k not in subset_keys}):
            raise ValueError('Calibration pixel-cache identity differs from the scheduled monitor.')
        if 'exploration_validation' in evaluator.panels:
            validation = evaluator.panels['exploration_validation']
            if (set(training_by_id) & {r['video_id'] for r in validation.rows}
                    or {r['source_group_id'] for r in training_rows} & {r['source_group_id'] for r in validation.rows}):
                raise ValueError('Validation overlaps training videos or source groups.')
        self.engine, self.schedule, self.evaluator = engine, schedule, evaluator
        self.root = Path(directory).resolve()
        self.record_path = self.root / 'workflow.json'
        self.contract = json.loads(json.dumps({'schema_version': 1, 'kind': 'scheduled_native_training',
            'training_contract_sha256': engine.contract_sha256, 'schedule': asdict(schedule),
            'evaluation_binding': evaluator.binding, 'calibration_sha256': json_hash(calibration),
            'original_identity': original_identity, 'diagnostic_only': diagnostic}, sort_keys=True))
        self.tasks = schedule.tasks()
        if resume:
            self.record = read_record(self.record_path)
            if self.record['contract'] != self.contract:
                raise ValueError('Workflow resume contract changed.')
            self.store = CheckpointStore(self.root / 'checkpoints', create=False)
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            self.store = CheckpointStore(self.root / 'checkpoints')
            self.record = {'contract': self.contract, 'completed_evaluations': [], 'complete': False}
            write_record(self.record_path, self.record)
            self.store.save(engine.state())
        self.failed = False
        self.last_saved = 0
        # A separate lock covers this whole workflow; the checkpoint store has
        # its own per-transaction lock, so the same lock is not nested.
        self.workflow_lock = CheckpointStore(self.root)

    def _checkpoint(self, epoch):
        if epoch == 0:
            return {'epoch': 0, 'condition': 'pretrained', 'seed': None,
                    'training_contract_sha256': None,
                    'file_sha256': self.engine.contract['provenance']['checkpoint_sha256']}
        manifest = self.store._manifest()
        ref = manifest['anchors'].get(str(epoch), manifest['latest'])
        return {'epoch': epoch, 'condition': self.engine.condition, 'seed': self.engine.seed,
                'training_contract_sha256': self.engine.contract_sha256, 'file_sha256': ref['sha256']}

    def _restore(self):
        state, _ = self.store.load()
        self.engine.restore(state)
        self.last_saved = self.engine.step
        # Reload after locking; another process may have progressed since init.
        self.record = read_record(self.record_path)
        if self.record['contract'] != self.contract:
            raise ValueError('Workflow contract changed while waiting for its lock.')
        completed = self.record['completed_evaluations']
        if [r['task'] for r in completed] != self.tasks[:len(completed)]:
            raise ValueError('Workflow evaluations are not an ordered schedule prefix.')
        for row in completed:
            epoch = row['task']['epoch']
            if epoch * self.engine.updates_per_epoch > self.engine.step:
                raise ValueError('Evaluation ledger is ahead of the committed training checkpoint.')
            expected = self._checkpoint(0) if epoch == 0 else {
                'epoch': epoch, 'condition': self.engine.condition, 'seed': self.engine.seed,
                'training_contract_sha256': self.engine.contract_sha256,
                'file_sha256': row['checkpoint']['file_sha256']}
            if row['checkpoint'] != expected:
                raise ValueError('Evaluation ledger mixes training runs or initialization identities.')
            # Anchor weights remain available even after rolling resume files
            # are retired; their references must match every anchor receipt.
            if epoch in self.schedule.primary_epochs:
                anchor = self.store._manifest()['anchors'].get(str(epoch))
                if anchor is None or anchor['sha256'] != row['checkpoint']['file_sha256']:
                    raise ValueError('Evaluation ledger differs from its saved anchor.')
            payload = self.evaluator.validate_receipt(row['receipt'], row['task'], row['checkpoint'])
            if payload['contract']['runtime'] != self.engine.contract['runtime']:
                raise ValueError('Evaluation runtime differs from training runtime.')
            if payload['contract']['weights_sha256']['target_encoder'] != self.engine.target_sha256:
                raise ValueError('Evaluation target differs from the frozen training target.')
            if epoch == 0 and payload['contract']['weights_sha256'] != self.contract['original_identity']:
                raise ValueError('Initialization evaluation used different pretrained weights.')
        if len(completed) < len(self.tasks):
            next_step = self.tasks[len(completed)]['epoch'] * self.engine.updates_per_epoch
            if next_step < self.engine.step:
                raise ValueError('Training advanced past an unfinished evaluation boundary.')
        if self.record['complete'] and (not self.engine.finished or len(completed) != len(self.tasks)):
            raise ValueError('Invalid workflow completion marker.')

    def _save(self):
        if self.engine.step == self.last_saved:
            return
        epoch, partial = divmod(self.engine.step, self.engine.updates_per_epoch)
        anchor = epoch if not partial and epoch in self.schedule.primary_epochs else None
        self.store.save(self.engine.state(), anchor_epoch=anchor)
        self.last_saved = self.engine.step

    def run(self, *, max_updates, max_evaluations, max_seconds=3600., stop_requested=lambda: False,
            on_update=None):
        if (self.failed or type(max_updates) is not int or max_updates < 0
                or type(max_evaluations) is not int or max_evaluations < 0
                or not 0 < max_seconds <= 86400):
            raise ValueError('Invalid workflow bounds or failed in-memory workflow.')
        started, updates, evaluations = time.monotonic(), 0, 0
        with self.workflow_lock._lock(exclusive=True):
            self._restore()
            try:
                while True:
                    if stop_requested() or time.monotonic() - started >= max_seconds:
                        reason = 'stop_or_time_boundary'
                        break
                    index = len(self.record['completed_evaluations'])
                    task = self.tasks[index] if index < len(self.tasks) else None
                    if task is not None and task['epoch'] * self.engine.updates_per_epoch == self.engine.step:
                        if evaluations >= max_evaluations:
                            reason = 'evaluation_limit'
                            break
                        self._save()
                        checkpoint = self._checkpoint(task['epoch'])
                        identity = model_identity(self.engine.model)
                        receipt = self.evaluator(self.engine.model, task, checkpoint)
                        payload = self.evaluator.validate_receipt(receipt, task, checkpoint, identity=identity)
                        if payload['contract']['runtime'] != self.engine.contract['runtime']:
                            raise ValueError('Evaluation runtime differs from training runtime.')
                        self.record['completed_evaluations'].append({'task': task, 'checkpoint': checkpoint,
                                                                      'receipt': receipt})
                        write_record(self.record_path, self.record, replace=True)
                        evaluations += 1
                        continue
                    if self.engine.finished:
                        self.record['complete'] = True
                        write_record(self.record_path, self.record, replace=True)
                        reason = 'training_and_scheduled_evaluations_complete'
                        break
                    if updates >= max_updates:
                        reason = 'update_limit'
                        break
                    event = self.engine.update()
                    updates += 1
                    if self.engine.step % self.engine.updates_per_epoch == 0:
                        self._save()
                    if on_update is not None:
                        on_update(event)
                self._save()
            except BaseException:
                self.failed = True
                # No partial update or uncommitted in-memory state is serialized.
                raise
        return {'reason': reason, 'updates_this_call': updates, 'evaluations_this_call': evaluations,
                'completed_step': self.engine.step, 'complete': self.record['complete'],
                'wall_seconds': time.monotonic() - started, 'research_result': False, 'budget_approved': False}
