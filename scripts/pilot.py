#!/usr/bin/env python3
"""Measure a small FP32 CUDA/MPS pilot; no cloud rental or confirmatory training.

Requires a checkpoint and prepared training clips. Records update and full
PGD/EOT evaluation time. It does not authorize spending or choose a protocol.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=ROOT / 'external/vjepa2')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--clips', type=Path, required=True, help='Prepared training clip directory with index.json.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3, help='Measured updates per condition, after one warmup.')
    parser.add_argument('--device', choices=['cuda', 'mps'], default='cuda')
    parser.add_argument('--patch-embedding', choices=['conv3d', 'linear'], default='conv3d')
    parser.add_argument('--accumulation', type=int, default=8, help='1-8 clips per update; use 8 for the planned effective batch.')
    parser.add_argument('--evaluation-steps', type=int, default=20)
    parser.add_argument('--evaluation-restarts', type=int, default=2)
    parser.add_argument('--evaluation-masks', type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1 or args.output.exists():
        parser.error('Use positive --steps and a fresh output filename.')
    if not 1 <= args.accumulation <= 8 or min(args.evaluation_steps, args.evaluation_restarts, args.evaluation_masks) < 1:
        parser.error('Use accumulation 1-8 and positive evaluation steps/restarts/masks.')
    import torch
    from models.runtime import MemoryMonitor, device_metadata, empty_cache, require_device, synchronize
    try:
        require_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    device = args.device
    torch.set_num_threads(4)
    from attacks.pgd import native_objective, pgd_linf
    from models.diagnostics import state_digest
    from models.pilot import effective_update, optimizer_for
    from models.vjepa2 import UPSTREAM_COMMIT, build_vitl, load_checkpoint, sample_masks
    import yaml

    study = yaml.safe_load((ROOT / 'configs/study.yaml').read_text())
    expected_hash = study['model']['checkpoint_sha256']
    if not expected_hash or study['model']['upstream_commit'] != UPSTREAM_COMMIT:
        parser.error('Study configuration must pin the checkpoint hash and matching upstream revision.')

    index_bytes = (args.clips / 'index.json').read_bytes()
    index = json.loads(index_bytes)
    if index['split'] != 'train' or len(index['clips']) < 8:
        parser.error('The pilot needs at least eight prepared training clips.')
    entries = index['clips'][:8]
    paths = []
    for entry in entries:
        path = (args.clips / entry['file']).resolve()
        if not path.is_relative_to(args.clips.resolve()) or not path.is_file():
            parser.error('Missing or invalid clip cache path.')
        paths.append(path)

    # Validate cached input identity before any measurements.
    def pixels_at(i):
        payload = torch.load(paths[i % len(paths)], weights_only=True, map_location='cpu')
        pixels = payload['pixels']
        expected = entries[i % len(entries)]['pixel_sha256']
        if hashlib.sha256(pixels.numpy().tobytes()).hexdigest() != expected:
            raise ValueError('Cached pixels disagree with the clip manifest.')
        return pixels.unsqueeze(0)

    torch.manual_seed(0)
    model = build_vitl(args.upstream, patch_embedding=args.patch_embedding)
    checkpoint = load_checkpoint(model, args.checkpoint, expected_hash)
    model.to(device)
    targets = [model.clean_targets(pixels_at(i).to(device)).cpu() for i in range(args.accumulation)]
    frozen_target_digest = state_digest(model.target_encoder)
    conditions = {}
    for condition in ('clean-finetune', 'adversarial-finetune'):
        # Both conditions start from identical released states, not from the
        # preceding pilot's updated weights. Loading is outside the timer.
        load_checkpoint(model, args.checkpoint, checkpoint['sha256'])
        encoder_before = model.encoder.patch_embed.proj.weight.detach().cpu().clone()
        predictor_before = model.predictor.predictor_embed.weight.detach().cpu().clone()
        optimizer = optimizer_for(model)
        timings = []
        with MemoryMonitor(device) as memory:
            for step in range(args.steps + 1):
                def load_example(i):
                    return pixels_at(i).to(device), targets[i].to(device)
                def masks_for(i):
                    return [m.to(device) for m in sample_masks(args.upstream, batch_size=1, seed=step*8+i)]
                synchronize(device)
                started = time.perf_counter()
                loss = effective_update(model, optimizer, load_example, masks_for,
                                        adversarial=condition == 'adversarial-finetune',
                                        accumulation=args.accumulation, seed=step*8)
                synchronize(device)
                elapsed = time.perf_counter() - started
                print(f'{condition}: update {step}, {elapsed:.3f}s, loss={loss:.6f}', flush=True)
                if step:
                    timings.append(elapsed)
        checks = {'target_unchanged': state_digest(model.target_encoder) == frozen_target_digest,
                  'encoder_updated': not torch.equal(encoder_before, model.encoder.patch_embed.proj.weight.detach().cpu()),
                  'predictor_updated': not torch.equal(predictor_before, model.predictor.predictor_embed.weight.detach().cpu())}
        if not all(checks.values()):
            raise RuntimeError(f'Pilot optimizer check failed: {checks}')
        conditions[condition] = {'update_seconds': timings, 'mean_seconds': statistics.mean(timings),
                                 'memory': memory.report(), 'optimizer_checks': checks}
        del optimizer
        empty_cache(device)

    # Include the expensive primary evaluation attack, on a training pilot
    # clip; this is a timing measurement, never final-test evaluation.
    model.eval()
    pixels, target = pixels_at(0).to(device), targets[0].to(device)
    bank = [[m.to(device) for m in sample_masks(args.upstream, batch_size=1, seed=1000+i)] for i in range(args.evaluation_masks)]
    scoring = [[m.to(device) for m in sample_masks(args.upstream, batch_size=1, seed=2000+i)] for i in range(4)]
    print(f'Timing PGD-{args.evaluation_steps}, {args.evaluation_restarts} restarts, '
          f'{args.evaluation_masks} attack-mask draws and four scoring draws...', flush=True)
    with MemoryMonitor(device) as evaluation_memory:
        synchronize(device)
        started = time.perf_counter()
        result = pgd_linf(native_objective(model, target, bank), pixels, epsilon=4/255,
                          step_size=1/255, steps=args.evaluation_steps,
                          restarts=args.evaluation_restarts, seed=3000)
        with torch.no_grad():
            score = native_objective(model, target, scoring)(result.adversarial).item()
        synchronize(device)
        evaluation_seconds = time.perf_counter() - started
    projected = math.ceil(2020 / 8) * 30 * 3 * (8/args.accumulation) * sum(c['mean_seconds'] for c in conditions.values()) / 3600
    report = {'kind': 'fp32_gpu_feasibility_pilot', 'research_result': False,
              'upstream_commit': UPSTREAM_COMMIT, 'checkpoint': checkpoint,
              'runtime': device_metadata(device),
              'packages': {n: importlib.metadata.version(n) for n in ('torch', 'torchvision', 'timm', 'einops', 'numpy')},
              'source_files_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                      for name in ('scripts/pilot.py', 'models/pilot.py', 'models/vjepa2.py',
                                                   'attacks/pgd.py', 'configs/study.yaml', 'models/runtime.py',
                                                   'models/diagnostics.py', 'models/patch_embed.py')},
              'context_patch_embedding': args.patch_embedding,
              'clip_index_sha256': hashlib.sha256(index_bytes).hexdigest(), 'conditions': conditions,
              'microbatch': 1, 'effective_batch': args.accumulation, 'precision': 'float32',
              'measured_updates_per_condition': args.steps, 'warmup_updates_per_condition': 1,
              'training_projection': {'videos': 2020, 'updates_per_epoch': math.ceil(2020/8),
                                      'epochs': 30, 'seeds': 3, 'conditions': 2, 'planned_effective_batch': 8,
                                      'measured_effective_batch': args.accumulation,
                                      'method': 'Linear scaling of measured update cost; scales optimizer overhead too.'},
              'evaluation_attack': {'steps': args.evaluation_steps, 'restarts': args.evaluation_restarts,
                                    'eot_masks': args.evaluation_masks, 'scoring_masks': 4,
                                    'epsilon': 4/255, 'step_size': 1/255},
              'primary_evaluation_measured': (args.evaluation_steps, args.evaluation_restarts, args.evaluation_masks) == (20,2,4),
              'evaluation_seconds_per_clip': evaluation_seconds,
              'evaluation_memory': evaluation_memory.report(),
              'pilot_scoring_loss': score, 'projected_training_only_hours': projected,
              'budget_approved': False,
              'total_pilot_wall_seconds': time.perf_counter() - run_started,
              'limitations': [f'Only {args.accumulation} cached training clips used; small warm-cache throughput sample.',
                              'If accumulation is below eight, training projection extrapolates to the planned batch size and may misestimate optimizer overhead.',
                              'Reduced evaluation settings do not measure primary PGD-20/2-restart/EOT-4 runtime.',
                              'Training projection excludes evaluation, target-cache construction, checkpoints, and probe fitting.',
                              'FP32 pilot settings are not the locked training/evaluation protocol.',
                              'PGD-50/EOT-8 convergence, multiple shapes, checkpoint I/O, and end-to-end costs remain to be measured.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(f'Pilot recorded at {args.output}; review total cost before any full run.')


if __name__ == '__main__':
    main()
