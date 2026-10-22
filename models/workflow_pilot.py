"""Fixed training-only pilot phases; scientific implementations are reused unchanged."""

from pathlib import Path

from models.calibration import load_calibration, run_calibration
from models.checkpointing import CheckpointStore, file_sha256
from models.evaluation import EvaluationSettings, model_identity
from models.evaluation_checkpoint import load_anchor
from models.evaluation_store import EvaluationStore, evaluate_panel, json_hash
from models.features import extract_features, fit_ridge_probe, score_ridge_probe, feature_digest
from models.pilot_measurement import Timings
from models.runtime import MemoryMonitor
from models.schedule import EpochEvaluator, EpochSchedule, ScheduledTraining
from models.study_artifacts import read_record, write_record
from models.training import TrainingEngine
from models.workflow_diagnostic import prepare_context, cached_training_loader

PILOT_SOURCES = ('models/workflow_pilot.py', 'models/pilot_measurement.py',
                 'scripts/run_workflow_pilot.py')
CONDITIONS = ('clean-finetune', 'adversarial-finetune')


class SavedBoundaryPause(Exception):
    """An intentional process exit after a saved epoch and before its evaluation."""


class MeasuredEvaluator(EpochEvaluator):
    def __init__(self, *args, timings, **kwargs):
        super().__init__(*args, **kwargs)
        self.timings = timings

    def __call__(self, model, task, checkpoint):
        # The artifact directory is private to this pilot; count only newly
        # evaluated clips. Reused native seconds are not new GPU measurements.
        before = {str(p.resolve()): len(read_record(p)['records'])
                  for p in self.root.glob('*/evaluation.json')}
        with self.timings.span('scheduled_evaluation', task=task) as event:
            receipt = super().__call__(model, task, checkpoint)
            payload = read_record(receipt['path'], expected_sha256=receipt['sha256'])
            previous = before.get(receipt['path'], 0)
            fresh = payload['records'][previous:]
            event.update(new_clips=len(fresh), reused_clips=previous,
                         new_native_seconds=sum(r['seconds'] for r in fresh))
        return receipt


def run_phase(root, args):
    output = Path(args.output)
    timings = Timings(output / 'timings' / (args.phase + '.jsonl'), args.device)
    with timings.span('prepare_context'):
        context = prepare_context(root, args)
        context.provenance['pilot_sources'] = {name: file_sha256(root / name) for name in PILOT_SOURCES}
        context.provenance['pilot_schema'] = 'cuda-workflow-diagnostic-v1'
    result = {'phase': args.phase, 'diagnostic_only': True, 'research_result': False,
              'budget_approved': False, 'provenance': context.provenance}
    with MemoryMonitor(args.device) as memory:
        if args.phase == 'calibrate':
            with timings.span('calibration_three_primary_candidates'):
                value = run_calibration(context.model, context.monitor, context.sampler, output / 'calibration',
                    checkpoint=context.checkpoint, provenance=context.provenance, diagnostic=True)
            result['chosen_step_size_255'] = value['chosen_step_size_255']
        elif args.phase in ('features-initial', 'features-trained'):
            checkpoint = context.checkpoint
            if args.phase == 'features-trained':
                with timings.span('load_anchor'):
                    checkpoint = load_anchor(context.model, output / CONDITIONS[1] / 'split' / 'checkpoints', 1,
                                              initial_checkpoint_sha256=context.checkpoint['file_sha256'])
            with timings.span('extract_features_eight_training_clips'):
                bundle = extract_features(context.model, context.training,
                                          provenance={**context.provenance, 'checkpoint': checkpoint})
            with timings.span('fit_probe'):
                probe = fit_ridge_probe(bundle)
            with timings.span('score_training_resubstitution'):
                result['training_resubstitution'] = score_ridge_probe(probe, bundle)
            with timings.span('save_features_and_probe'):
                store = CheckpointStore(output / args.phase)
                result['artifacts'] = {'features': store._atomic('features.pt', bundle),
                                       'probe': store._atomic('probe.pt', probe)}
            result.update(feature_sha256=feature_digest(bundle), held_out_accuracy_measured=False)
        elif args.phase == 'controls':
            calibration = load_calibration(output / 'calibration', context.monitor.rows,
                                          original_identity=model_identity(context.model), diagnostic=True, model=context.model)
            for method in ('convergence', 'fgsm', 'random'):
                with timings.span('control_evaluation', method=method):
                    destination = output / 'controls' / method
                    evaluate_panel(context.model, context.monitor, context.sampler,
                        EvaluationSettings.for_method(method, calibration['chosen_step_size_255']),
                        EvaluationStore(destination), checkpoint=context.checkpoint,
                        provenance=context.provenance, sensitivity_banks=2 if method == 'convergence' else 0)
        else:
            condition, segment = args.phase.rsplit('/', 1)
            if condition not in CONDITIONS or segment not in ('split-first', 'split-resume', 'reference'):
                raise ValueError('Unknown pilot phase.')
            calibration = load_calibration(output / 'calibration', context.monitor.rows,
                                          original_identity=model_identity(context.model), diagnostic=True, model=context.model)
            provenance = {**context.provenance, 'training_rows': context.training.rows,
                          'calibration_sha256': json_hash(calibration),
                          'evaluation_provenance_sha256': json_hash(context.provenance)}
            with timings.span('cache_fixed_targets_eight_training_clips'):
                loader = cached_training_loader(context.model, context.training, args.device)
            engine = TrainingEngine(context.model, context.settings, [r['video_id'] for r in context.training.rows],
                loader, context.sampler, condition=condition, seed=0, provenance=provenance, device=args.device)
            timings.instrument(engine, ('update', 'state', 'restore'), 'engine')
            variant = 'reference' if segment == 'reference' else 'split'
            evaluator = MeasuredEvaluator({'train_monitor': context.monitor}, context.sampler,
                output / ('evaluations-' + variant), primary_step_size=calibration['chosen_step_size_255'],
                provenance=context.provenance, base_seed=context.plan['evaluation']['mask_seed'], timings=timings)
            schedule = EpochSchedule(context.settings.epochs, tuple(context.plan['evaluation']['primary_epochs']),
                                     panels=('train_monitor',))
            with timings.span('construct_workflow', includes_initial_save=segment != 'split-resume'):
                workflow = ScheduledTraining(engine, schedule, evaluator, output / condition / variant,
                    calibration=calibration, calibration_rows=context.monitor.rows, diagnostic=True,
                    resume=segment == 'split-resume')
            timings.instrument(workflow.store, ('save', 'load'), 'checkpoint')

            def boundary(event):
                print(f"Completed {condition} update {event['step']}", flush=True)
                if segment == 'split-first':
                    # ScheduledTraining commits the epoch checkpoint before
                    # calling this hook. No partial update is ever saved.
                    raise SavedBoundaryPause()

            try:
                with timings.span('scheduled_training_attempt'):
                    value = workflow.run(max_updates=2 if segment == 'reference' else 1,
                        max_evaluations=8, max_seconds=3600, on_update=boundary)
                if value['completed_step'] != 2:
                    raise RuntimeError('Bounded attempt ended before the required comparison boundary.')
                result['workflow'] = value
            except SavedBoundaryPause:
                if segment != 'split-first' or workflow.store._manifest()['step'] != 1:
                    raise
                result['expected_pause_after_saved_step'] = 1
        result['memory'] = memory.report()
    write_record(output / 'phase-results' / (args.phase + '.json'), result)
    return result
