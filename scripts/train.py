#!/usr/bin/env python3
"""Bounded diagnostic of the training/resume runner; full study admission is pending."""

import argparse
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/budget_study_plan.yaml')
    parser.add_argument('--upstream', type=Path, default=ROOT / 'external/vjepa2')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--clips', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/manifests/ucf101-split1-v1/train.csv')
    parser.add_argument('--condition', choices=['clean-finetune', 'adversarial-finetune'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--max-updates', type=int, required=True, help='1-8 additional updates for this diagnostic attempt.')
    parser.add_argument('--max-seconds', type=float, default=600, help='Boundary-checked wall-time limit; does not stop cloud billing.')
    parser.add_argument('--output', type=Path, required=True, help='Fresh attempt directory; existing directories are refused.')
    parser.add_argument('--resume-from', type=Path, help='Previous attempt directory; write recovery into fresh --output.')
    parser.add_argument('--resume-generation', choices=['latest', 'previous'], default='latest')
    args = parser.parse_args()
    if not args.diagnostic:
        parser.error('Only --diagnostic is implemented until evaluation, manifests and measured budget are admitted.')
    if not 1 <= args.max_updates <= 8 or not 0 < args.max_seconds <= 3600:
        parser.error('Use 1-8 updates and a finite time limit of at most 3600 seconds.')
    if not 0 <= args.seed < 2**31 - 1:
        parser.error('Use a seed in [0, 2**31 - 1).')
    if args.output.exists():
        parser.error('Use a fresh --output directory; previous attempts are preserved.')
    if args.resume_generation != 'latest' and args.resume_from is None:
        parser.error('--resume-generation requires --resume-from.')

    import torch
    import yaml
    from datasets.prepared import PreparedTrainingClips
    from models.checkpointing import CheckpointStore, file_sha256
    from models.runtime import require_device, synchronize
    from models.training import TrainingEngine, TrainingMaskSampler, TrainingSettings
    from models.vjepa2 import UPSTREAM_COMMIT, PRETRAIN_CONFIG, build_vitl, load_checkpoint

    plan = yaml.safe_load(args.plan.read_text())
    settings = TrainingSettings.from_plan(plan)
    study_path = ROOT / plan['parent_study']
    if file_sha256(study_path) != plan['source_sha256'][plan['parent_study']]:
        parser.error('Parent study hash changed; revise the proposal first.')
    study = yaml.safe_load(study_path.read_text())
    if study['model']['upstream_commit'] != UPSTREAM_COMMIT:
        parser.error('Upstream pin differs from the adapter.')
    # Only the diagnostic parent training pool is admitted at this stage.
    expected_manifest = ROOT / study['data']['manifest_directory'] / 'train.csv'
    if (args.manifest.resolve() != expected_manifest.resolve()
            or file_sha256(args.manifest) != plan['source_sha256'][str(expected_manifest.relative_to(ROOT))]):
        parser.error('Use the pinned parent training manifest for this bounded diagnostic.')
    require_device(args.device)
    torch.set_num_threads(4)
    clips = PreparedTrainingClips(args.clips, args.manifest)
    if len(clips.entries) > 8:
        parser.error('This diagnostic currently accepts at most eight prepared training clips.')
    clips.verify_all()
    resume = None
    parent_ref = None
    if args.resume_from:
        resume, parent_ref = CheckpointStore(args.resume_from / 'checkpoints', create=False).load(args.resume_generation)

    sampler = TrainingMaskSampler(args.upstream)
    model = build_vitl(args.upstream, activation_checkpointing=True)
    checkpoint = load_checkpoint(model, args.checkpoint, study['model']['checkpoint_sha256'])
    model.to(args.device)
    load_example = clips.target_loader(model, args.device)
    sources = ['scripts/train.py', 'models/training.py', 'models/checkpointing.py', 'datasets/prepared.py',
               'models/pilot.py', 'models/vjepa2.py', 'models/diagnostics.py', 'models/runtime.py', 'attacks/pgd.py']
    provenance = {'checkpoint_sha256': checkpoint['sha256'], 'upstream_commit': UPSTREAM_COMMIT,
                  'plan_sha256': file_sha256(args.plan), 'clip_index_sha256': clips.index_sha256,
                  'manifest_sha256': clips.manifest_sha256,
                  'upstream_mask_config_sha256': file_sha256(args.upstream / PRETRAIN_CONFIG),
                  'source_sha256': {name: file_sha256(ROOT / name) for name in sources}}
    engine = TrainingEngine(model, settings, clips.video_ids, load_example, sampler,
                            condition=args.condition, seed=args.seed, provenance=provenance, device=args.device)
    if resume:
        engine.restore(resume)
        del resume
    args.output.mkdir(parents=True, exist_ok=False)
    store = CheckpointStore(args.output / 'checkpoints')
    beginning_step = engine.step
    store.save(engine.state())
    last_saved = engine.step
    stop_requested = []

    def stop(signum, frame):
        stop_requested.append(signum)

    handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    record = {'kind': 'training_runner_diagnostic', 'research_result': False,
              'contract': engine.contract, 'contract_sha256': engine.contract_sha256,
              'parent_checkpoint': parent_ref, 'parent_attempt': str(args.resume_from) if args.resume_from else None,
              'starting_step': beginning_step, 'budget_approved': False}
    (args.output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    reason = 'update_limit'
    try:
        while engine.step - beginning_step < args.max_updates and not engine.finished:
            if stop_requested or time.monotonic() - started >= args.max_seconds:
                reason = 'signal' if stop_requested else 'time_limit'
                break
            synchronize(args.device)
            tick = time.perf_counter()
            event = engine.update()
            synchronize(args.device)
            seconds = time.perf_counter() - tick
            with (args.output / 'events.jsonl').open('a') as handle:
                handle.write(json.dumps({**event, 'update_seconds': seconds}) + '\n')
            print(f"Update {engine.step}, epoch {event['epoch']}, {len(event['video_ids'])} clips, "
                  f"loss={event['optimizer_loss']:.6f}, {seconds:.3f}s", flush=True)
            if engine.step % engine.updates_per_epoch == 0:
                epoch = engine.step // engine.updates_per_epoch
                anchor = epoch if epoch in plan['evaluation']['primary_epochs'] else None
                store.save(engine.state(), anchor_epoch=anchor)
                last_saved = engine.step
        if engine.step != last_saved:
            store.save(engine.state())
        if engine.finished:
            reason = 'training_epochs_complete_without_research_evaluation'
        record.update(status='stopped_cleanly', stop_reason=reason, ending_step=engine.step,
                      wall_seconds=time.monotonic() - started)
    except BaseException as exc:
        # Never serialize a partially accumulated or partially applied update.
        record.update(status='failed', ending_step_last_committed=last_saved,
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        (args.output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Diagnostic saved in {args.output}. No validation/test selection or cloud shutdown was performed.')


if __name__ == '__main__':
    main()
