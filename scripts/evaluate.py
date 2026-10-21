#!/usr/bin/env python3
"""Bounded primary/routine attack diagnostic on staged training-monitor clips."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--clips', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=ROOT / 'external/vjepa2')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    parser.add_argument('--method', choices=('routine', 'primary', 'convergence', 'fgsm', 'random'), default='primary')
    parser.add_argument('--step-size-255', type=float, choices=(.5, 1., 2.), default=1.)
    parser.add_argument('--limit', type=int, default=1, help='One or two training-monitor clips; no held-out metrics.')
    parser.add_argument('--sensitivity-banks', type=int, choices=(0, 2), default=0)
    parser.add_argument('--anchor-store', type=Path)
    parser.add_argument('--anchor-epoch', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Continue only the identical checkpoint/panel evaluation.')
    args = parser.parse_args()
    if not args.diagnostic or not 1 <= args.limit <= 2:
        parser.error('Only --diagnostic on one or two training-monitor clips is currently enabled.')
    if (args.anchor_store is None) != (args.anchor_epoch is None):
        parser.error('--anchor-store and --anchor-epoch must be provided together.')
    if not args.resume and args.output.exists():
        parser.error('Use a fresh output or explicit --resume.')
    if args.resume and not (args.output / 'evaluation.json').is_file():
        parser.error('No committed evaluation exists to resume.')

    import torch
    import yaml
    from datasets.staged_preparation import PreparedStagedPanel
    from models.checkpointing import file_sha256
    from models.evaluation import EvaluationMaskSampler, EvaluationSettings
    from models.evaluation_checkpoint import load_anchor
    from models.evaluation_store import EvaluationStore, evaluate_panel
    from models.runtime import require_device
    from models.vjepa2 import UPSTREAM_COMMIT, PRETRAIN_CONFIG, build_vitl, load_checkpoint

    torch.set_num_threads(4)
    require_device(args.device)
    settings = EvaluationSettings.for_method(args.method, args.step_size_255)
    panel = PreparedStagedPanel(ROOT, args.manifests, args.clips, 'train_monitor')
    # Preserve selection order; identity explicitly records the bounded subset.
    panel.rows, panel.entries, panel.paths = panel.rows[:args.limit], panel.entries[:args.limit], panel.paths[:args.limit]
    panel.identity['diagnostic_first_n'] = args.limit
    panel.verify_all()
    study = yaml.safe_load((ROOT / 'configs/study.yaml').read_text())
    model = build_vitl(args.upstream, activation_checkpointing=True)
    original = load_checkpoint(model, args.checkpoint, study['model']['checkpoint_sha256'])
    checkpoint = {'epoch': 0, 'condition': 'pretrained', 'seed': None,
                  'training_contract_sha256': None, 'file_sha256': original['sha256']}
    if args.anchor_store:
        checkpoint = load_anchor(model, args.anchor_store, args.anchor_epoch,
                                 initial_checkpoint_sha256=original['sha256'])
    model.to(args.device)
    sampler = EvaluationMaskSampler(args.upstream)
    sources = ['scripts/evaluate.py', 'models/evaluation.py', 'models/evaluation_store.py',
               'models/evaluation_checkpoint.py', 'models/checkpointing.py', 'models/training.py',
               'datasets/staged_files.py', 'datasets/staged_preparation.py', 'datasets/staged_subset.py',
               'datasets/video.py', 'models/vjepa2.py', 'models/diagnostics.py', 'models/runtime.py', 'attacks/pgd.py']
    provenance = {'diagnostic': True, 'research_admitted': False,
                  'primary_step_size_calibrated': False, 'upstream_commit': UPSTREAM_COMMIT,
                  'mask_config_sha256': file_sha256(args.upstream / PRETRAIN_CONFIG),
                  'source_sha256': {name: file_sha256(ROOT / name) for name in sources}}
    store = EvaluationStore(args.output, resume=args.resume)
    result = evaluate_panel(model, panel, sampler, settings, store, checkpoint=checkpoint,
                            provenance=provenance, sensitivity_banks=args.sensitivity_banks,
                            on_clip=lambda i, r: print(f"Completed {r['video_id']}: {r['seconds']:.3f}s", flush=True))
    print(json.dumps({'complete': result['complete'], 'clips': len(result['records']),
                      'research_result': False, 'budget_approved': False}, indent=2))


if __name__ == '__main__':
    main()
