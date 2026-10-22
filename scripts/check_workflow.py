#!/usr/bin/env python3
"""Bounded feature, calibration, and scheduled-training checks; no research launch."""

import argparse
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('features', 'calibrate', 'train'))
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--clips', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=ROOT / 'external/vjepa2')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    parser.add_argument('--limit', type=int, default=1, help='One or two training-monitor clips.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Continue this exact calibration or trajectory directory.')
    parser.add_argument('--calibration', type=Path, help='Completed diagnostic calibration directory (training only).')
    parser.add_argument('--evaluations', type=Path, help='Artifact directory, shareable across matched conditions.')
    parser.add_argument('--condition', choices=('clean-finetune', 'adversarial-finetune'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-updates', type=int, default=1, help='0-8 additional updates, preserving the epoch schedule.')
    parser.add_argument('--max-evaluations', type=int, default=4, help='0-16 additional panel evaluations.')
    parser.add_argument('--max-seconds', type=float, default=600., help='Training/evaluation boundary limit; excludes setup and does not stop billing.')
    parser.add_argument('--anchor-store', type=Path, help='Features only: saved anchor checkpoint directory.')
    parser.add_argument('--anchor-epoch', type=int)
    parser.add_argument('--execution-policy', choices=('original', 'cudnn-fp32', 'native-fp32'),
                        default='original', help='Explicit diagnostic backend/precision; does not adopt a study policy.')
    args = parser.parse_args()
    if not args.diagnostic or args.limit not in (1, 2):
        parser.error('Only --diagnostic with one or two training-monitor clips is enabled.')
    if not (0 <= args.max_updates <= 8 and 0 <= args.max_evaluations <= 16
            and 0 < args.max_seconds <= 3600 and 0 <= args.seed < 2**31 - 1):
        parser.error('Invalid diagnostic bounds or seed.')
    if (args.anchor_store is None) != (args.anchor_epoch is None) or (args.anchor_store and args.mode != 'features'):
        parser.error('Anchor store/epoch must be supplied together, for features only.')
    if args.mode == 'train' and any(x is None for x in (args.calibration, args.evaluations, args.condition)):
        parser.error('Training requires --calibration, --evaluations, and --condition.')
    if args.resume:
        required = 'workflow.json' if args.mode == 'train' else None
        if args.mode == 'features' or not args.output.is_dir() or (required and not (args.output / required).is_file()):
            parser.error('No compatible diagnostic directory to resume.')
    elif args.output.exists():
        parser.error('Use a fresh --output, or explicitly --resume calibration/training.')

    from models.calibration import load_calibration, run_calibration
    from models.checkpointing import CheckpointStore
    from models.evaluation import model_identity
    from models.evaluation_checkpoint import load_anchor
    from models.evaluation_store import json_hash
    from models.features import extract_features, fit_ridge_probe, score_ridge_probe, feature_digest
    from models.schedule import EpochEvaluator, EpochSchedule, ScheduledTraining
    from models.study_artifacts import write_record
    from models.training import TrainingEngine
    from models.workflow_diagnostic import prepare_context, cached_training_loader

    from models.execution import configure_execution
    configure_execution(args.execution_policy, device=args.device)
    context = prepare_context(ROOT, args)
    if args.mode == 'features':
        checkpoint = context.checkpoint
        if args.anchor_store:
            checkpoint = load_anchor(context.model, args.anchor_store, args.anchor_epoch,
                                     initial_checkpoint_sha256=context.checkpoint['file_sha256'])
        bundle = extract_features(context.model, context.training,
                                  provenance={**context.provenance, 'checkpoint': checkpoint})
        probe = fit_ridge_probe(bundle)
        # This small integration check deliberately scores training examples.
        # It cannot estimate held-out accuracy or justify scientific conclusions.
        score = score_ridge_probe(probe, bundle)
        args.output.mkdir(parents=True, exist_ok=False)
        store = CheckpointStore(args.output)
        refs = {'features': store._atomic('features.pt', bundle), 'probe': store._atomic('probe.pt', probe)}
        result = {'kind': 'clean_feature_probe_diagnostic', 'diagnostic_only': True,
                  'feature_sha256': feature_digest(bundle), 'extraction_seconds': bundle['seconds'],
                  'checkpoint': checkpoint, 'training_resubstitution': score, 'artifacts': refs,
                  'held_out_accuracy_measured': False, 'research_result': False, 'budget_approved': False}
        write_record(args.output / 'summary.json', result)
    elif args.mode == 'calibrate':
        result = run_calibration(context.model, context.monitor, context.sampler, args.output,
                                 checkpoint=context.checkpoint, provenance=context.provenance, diagnostic=True)
        result = {key: result[key] for key in ('chosen_step_size_255', 'diagnostic_only', 'research_result', 'budget_approved')}
    else:
        calibration = load_calibration(args.calibration, context.monitor.rows,
                                       original_identity=model_identity(context.model), diagnostic=True, model=context.model)
        train_provenance = {**context.provenance, 'training_rows': context.training.rows,
                            'calibration_sha256': json_hash(calibration),
                            'evaluation_provenance_sha256': json_hash(context.provenance)}
        engine = TrainingEngine(context.model, context.settings, [r['video_id'] for r in context.training.rows],
            cached_training_loader(context.model, context.training, args.device), context.sampler,
            condition=args.condition, seed=args.seed, provenance=train_provenance, device=args.device)
        evaluator = EpochEvaluator({'train_monitor': context.monitor}, context.sampler, args.evaluations,
            primary_step_size=calibration['chosen_step_size_255'], provenance=context.provenance,
            base_seed=context.plan['evaluation']['mask_seed'])
        schedule = EpochSchedule(context.settings.epochs, tuple(context.plan['evaluation']['primary_epochs']),
                                 panels=('train_monitor',))
        workflow = ScheduledTraining(engine, schedule, evaluator, args.output, calibration=calibration,
            calibration_rows=context.monitor.rows, diagnostic=True, resume=args.resume)
        stop = []
        handlers = {s: signal.signal(s, lambda signum, frame: stop.append(signum))
                    for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            result = workflow.run(max_updates=args.max_updates, max_evaluations=args.max_evaluations,
                max_seconds=args.max_seconds, stop_requested=lambda: bool(stop),
                on_update=lambda event: print(f"Completed update {event['step']}, epoch {event['epoch']}", flush=True))
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
