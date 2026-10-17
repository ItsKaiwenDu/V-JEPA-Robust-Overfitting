#!/usr/bin/env python3
"""Sanity-check FGSM/PGD on one prepared training clip; not a robustness benchmark."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=ROOT / 'external/vjepa2')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--clip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output file.')
    if not args.clip.is_file() or not args.checkpoint.is_file():
        parser.error('Both a prepared clip and the checkpoint must already exist.')
    import torch
    from attacks.pgd import fgsm, native_objective, pgd_linf
    from models.diagnostics import state_digest
    from models.vjepa2 import UPSTREAM_COMMIT, build_vitl, load_checkpoint, sample_masks

    torch.set_num_threads(4)
    model = build_vitl(args.upstream)
    checkpoint = load_checkpoint(model, args.checkpoint)
    model.to(args.device).eval()
    cache = torch.load(args.clip, map_location='cpu', weights_only=True)
    if hashlib.sha256(cache['pixels'].numpy().tobytes()).hexdigest() != cache['metadata']['pixel_sha256']:
        raise ValueError('Cached clip pixels do not match their recorded hash.')
    clean = cache['pixels'].unsqueeze(0).to(args.device)
    targets = model.clean_targets(clean)
    masks = [m.to(args.device) for m in sample_masks(args.upstream, batch_size=1, seed=0)]
    scoring_masks = [m.to(args.device) for m in sample_masks(args.upstream, batch_size=1, seed=100000)]
    objective = native_objective(model, targets, [masks])
    scoring = native_objective(model, targets, [scoring_masks])
    before = state_digest(model)
    with torch.no_grad():
        clean_loss = objective(clean).item()
        clean_scoring = scoring(clean).item()
    measured = {}
    for name in ('fgsm', 'pgd5'):
        started = time.perf_counter()
        attack = (fgsm(objective, clean, epsilon=4/255) if name == 'fgsm'
                  else pgd_linf(objective, clean, epsilon=4/255, step_size=1/255, steps=5, seed=0))
        delta = (attack.adversarial - clean).abs().max().item()
        if delta > 4/255 + 1e-7 or attack.adversarial.min() < 0 or attack.adversarial.max() > 1:
            raise RuntimeError('Attack violates its pixel-space bounds.')
        with torch.no_grad():
            independent_loss = scoring(attack.adversarial).item()
        measured[name] = {'attack_bank_loss': attack.losses.item(), 'independent_mask_loss': independent_loss,
                          'max_pixel_delta': delta, 'seconds': time.perf_counter() - started}
    if state_digest(model) != before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Input attacks changed model state or parameter gradients.')
    if not any(value['attack_bank_loss'] > clean_loss for value in measured.values()):
        raise RuntimeError('No native-loss ascent was observed on this diagnostic clip.')
    report = {'kind': 'one_training_clip_attack_sanity', 'research_result': False,
              'upstream_commit': UPSTREAM_COMMIT, 'checkpoint_sha256': checkpoint['sha256'],
              'packages': {n: importlib.metadata.version(n) for n in ('torch', 'torchvision', 'numpy', 'timm', 'einops')},
              'source_files_sha256': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
                                      for n in ('attacks/pgd.py', 'models/vjepa2.py', 'datasets/video.py', 'scripts/check_attacks.py')},
              'device': args.device, 'clip': cache['metadata'], 'epsilon': 4/255,
              'attack_mask_seed': 0, 'independent_scoring_mask_seed': 100000,
              'clean_attack_bank_loss': clean_loss, 'clean_independent_mask_loss': clean_scoring,
              'attacks': measured, 'weights_and_parameter_gradients_unchanged': True,
              'limitations': ['One selected training clip; no held-out robustness estimate.',
                              'One mask draw (two families) per bank; EOT/convergence sweeps remain pending.',
                              'No adversarial fine-tuning, downstream evaluation, or overfitting claim.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'clean_loss': clean_loss, 'attacks': measured}, indent=2))


if __name__ == '__main__':
    main()
