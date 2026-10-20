#!/usr/bin/env python3
"""Compare pretrained CPU/MPS outputs and selected gradients on one real clip."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def comparison(reference, observed):
    import torch
    a, b = reference.flatten().double(), observed.flatten().double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError('Nonfinite CPU/MPS comparison values.')
    scale = a.norm()
    if scale == 0 or b.norm() == 0:
        raise RuntimeError('Expected nonzero reference and observed values.')
    return {'relative_l2_error': ((a-b).norm()/scale).item(),
            'cosine_similarity': torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-30).item(),
            'max_absolute_error': (a-b).abs().max().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=ROOT/'external/vjepa2')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--clip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mps-patch-embedding', choices=['conv3d', 'linear'], default='conv3d',
                        help='CPU reference always uses the original Conv3d patch map.')
    args = parser.parse_args()
    if args.output.exists() or not args.checkpoint.is_file() or not args.clip.is_file():
        parser.error('Use existing checkpoint/clip files and a new output filename.')
    import torch
    import yaml
    from models.runtime import device_metadata, require_device, synchronize
    from models.diagnostics import state_digest
    from models.vjepa2 import UPSTREAM_COMMIT, build_vitl, load_checkpoint, sample_masks
    require_device('mps')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    clip = torch.load(args.clip, weights_only=True, map_location='cpu')
    if hashlib.sha256(clip['pixels'].numpy().tobytes()).hexdigest() != clip['metadata']['pixel_sha256']:
        raise ValueError('Cached pixels disagree with their provenance.')
    expected = yaml.safe_load((ROOT/'configs/study.yaml').read_text())['model']['checkpoint_sha256']
    model = build_vitl(args.upstream)
    checkpoint = load_checkpoint(model, args.checkpoint, expected)
    target_before = state_digest(model.target_encoder)
    masks = sample_masks(args.upstream, batch_size=1, seed=0)
    selected = ('encoder.patch_embed.proj.weight', 'predictor.predictor_embed.weight')

    def snapshot(device):
        print(f'Checking real-clip loss and gradients on {device}...', flush=True)
        model.to(device).train()
        clean = clip['pixels'].unsqueeze(0).to(device)
        pixels = clean.detach().clone().requires_grad_(True)
        targets = model.clean_targets(clean)
        model.zero_grad(set_to_none=True)
        loss = model(pixels, targets, [m.to(device) for m in masks])
        loss.backward()
        synchronize(device)
        if any(p.grad is not None for p in model.target_encoder.parameters()):
            raise RuntimeError('Target encoder acquired gradients.')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite model gradient.')
        parameters = dict(model.named_parameters())
        result = {'loss': loss.detach().cpu(), 'targets': targets.detach().cpu(),
                  'input_gradient': pixels.grad.detach().cpu(),
                  **{name: parameters[name].grad.detach().cpu().clone() for name in selected}}
        model.zero_grad(set_to_none=True)
        return result

    cpu = snapshot('cpu')
    if args.mps_patch_embedding == 'linear':
        from models.patch_embed import LinearPatchEmbed3D
        model.encoder.patch_embed = LinearPatchEmbed3D(model.encoder.patch_embed)
    mps = snapshot('mps')
    thresholds = {'loss_rtol': 1e-4, 'loss_atol': 1e-5,
                  'target_relative_l2': 1e-3, 'gradient_relative_l2': 1e-2,
                  'gradient_cosine_minimum': 0.999}
    # Thresholds are declared before inspecting the device comparison.
    metrics = {name: comparison(cpu[name], mps[name]) for name in ('targets', 'input_gradient', *selected)}
    loss_passed = bool(torch.isclose(cpu['loss'], mps['loss'], rtol=thresholds['loss_rtol'], atol=thresholds['loss_atol']))
    target_passed = metrics['targets']['relative_l2_error'] <= thresholds['target_relative_l2']
    gradients_passed = all(metrics[name]['relative_l2_error'] <= thresholds['gradient_relative_l2']
                           and metrics[name]['cosine_similarity'] >= thresholds['gradient_cosine_minimum']
                           for name in ('input_gradient', *selected))
    target_unchanged = state_digest(model.target_encoder) == target_before
    passed = loss_passed and target_passed and gradients_passed and target_unchanged
    report = {'kind': 'one_training_clip_cpu_mps_parity', 'research_result': False, 'passed': passed,
              'thresholds': thresholds, 'cpu_loss': cpu['loss'].item(), 'mps_loss': mps['loss'].item(),
              'comparisons': metrics, 'frozen_target_unchanged': target_unchanged,
              'runtime': device_metadata('mps'), 'checkpoint': checkpoint, 'upstream_commit': UPSTREAM_COMMIT,
              'clip': clip['metadata'], 'mask_seed': 0,
              'packages': {n: importlib.metadata.version(n) for n in ('torch','torchvision','timm','einops','numpy')},
              'source_files_sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                                      for n in ('scripts/check_mps.py','models/runtime.py','models/vjepa2.py','models/patch_embed.py')},
              'cpu_context_patch_embedding': 'conv3d', 'mps_context_patch_embedding': args.mps_patch_embedding,
              'limitations': ['One training clip and mask draw, FP32.',
                              'Full input gradients and two selected parameter tensors compared, not every parameter gradient.',
                              'No optimizer step or cross-device training-trajectory equivalence.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'passed': passed, 'cpu_loss': report['cpu_loss'], 'mps_loss': report['mps_loss'], 'comparisons': metrics}, indent=2))
    if not passed:
        raise RuntimeError('CPU/MPS correctness comparison failed; inspect the report before timing training.')


if __name__ == '__main__':
    main()
