#!/usr/bin/env python3
"""Measure a small FP32 CUDA pilot; no cloud rental and no confirmatory training.

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=ROOT / 'external/vjepa2')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--clips', type=Path, required=True, help='Prepared training clip directory with index.json.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3, help='Measured updates per condition, after one warmup.')
    args = parser.parse_args()
    if args.steps < 1 or args.output.exists():
        parser.error('Use positive --steps and a fresh output filename.')
    import torch
    if not torch.cuda.is_available():
        parser.error('This cost pilot requires CUDA. No GPU is available in this runtime.')
    from attacks.pgd import native_objective, pgd_linf
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
    model = build_vitl(args.upstream)
    checkpoint = load_checkpoint(model, args.checkpoint, expected_hash)
    model.cuda()
    targets = [model.clean_targets(pixels_at(i).cuda()).cpu() for i in range(8)]
    conditions = {}
    for condition in ('clean-finetune', 'adversarial-finetune'):
        # Both conditions start from identical released states, not from the
        # preceding pilot's updated weights. Loading is outside the timer.
        load_checkpoint(model, args.checkpoint, checkpoint['sha256'])
        optimizer = optimizer_for(model)
        timings = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(args.steps + 1):
            def load_example(i):
                return pixels_at(i).cuda(), targets[i].cuda()
            def masks_for(i):
                return [m.to('cuda') for m in sample_masks(args.upstream, batch_size=1, seed=step*8+i)]
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss = effective_update(model, optimizer, load_example, masks_for,
                                    adversarial=condition == 'adversarial-finetune', seed=step*8)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            print(f'{condition}: update {step}, {elapsed:.3f}s, loss={loss:.6f}', flush=True)
            if step:
                timings.append(elapsed)
        conditions[condition] = {'update_seconds': timings, 'mean_seconds': statistics.mean(timings),
                                 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        del optimizer
        torch.cuda.empty_cache()

    # Include the expensive primary evaluation attack, on a training pilot
    # clip; this is a timing measurement, never final-test evaluation.
    model.eval()
    pixels, target = pixels_at(0).cuda(), targets[0].cuda()
    bank = [[m.to('cuda') for m in sample_masks(args.upstream, batch_size=1, seed=1000+i)] for i in range(4)]
    scoring = [[m.to('cuda') for m in sample_masks(args.upstream, batch_size=1, seed=2000+i)] for i in range(4)]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = pgd_linf(native_objective(model, target, bank), pixels, epsilon=4/255,
                      step_size=1/255, steps=20, restarts=2, seed=3000)
    with torch.no_grad():
        score = native_objective(model, target, scoring)(result.adversarial).item()
    torch.cuda.synchronize()
    evaluation_seconds = time.perf_counter() - started
    projected = math.ceil(2020 / 8) * 30 * 3 * sum(c['mean_seconds'] for c in conditions.values()) / 3600
    report = {'kind': 'fp32_cuda_feasibility_pilot', 'research_result': False,
              'upstream_commit': UPSTREAM_COMMIT, 'checkpoint': checkpoint,
              'gpu': torch.cuda.get_device_name(), 'vram_bytes': torch.cuda.get_device_properties(0).total_memory,
              'packages': {n: importlib.metadata.version(n) for n in ('torch', 'torchvision', 'timm', 'einops', 'numpy')},
              'source_files_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                      for name in ('scripts/pilot.py', 'models/pilot.py', 'models/vjepa2.py',
                                                   'attacks/pgd.py', 'configs/study.yaml')},
              'clip_index_sha256': hashlib.sha256(index_bytes).hexdigest(), 'conditions': conditions,
              'microbatch': 1, 'effective_batch': 8, 'precision': 'float32',
              'evaluation_seconds_per_clip': evaluation_seconds,
              'evaluation_cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'pilot_scoring_loss': score, 'projected_training_only_hours': projected,
              'budget_approved': False,
              'limitations': ['Only eight cached training clips; small warm-cache throughput sample.',
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
