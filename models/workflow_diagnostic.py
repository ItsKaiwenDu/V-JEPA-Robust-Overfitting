"""Local setup for a bounded eight-training-clip integration diagnostic."""

import copy
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from datasets.staged_preparation import PreparedStagedPanel
from models.checkpointing import file_sha256
from models.evaluation import EvaluationMaskSampler
from models.evaluation_store import json_hash
from models.runtime import require_device
from models.execution import apply_model_policy, ExecutionGuard, model_execution_contract
from models.training import TrainingSettings
from models.vjepa2 import UPSTREAM_COMMIT, PRETRAIN_CONFIG, build_vitl, load_checkpoint

SOURCES = (
    'scripts/check_workflow.py', 'models/workflow_diagnostic.py', 'models/features.py',
    'models/calibration.py', 'models/schedule.py', 'models/study_artifacts.py',
    'models/training.py', 'models/checkpointing.py', 'models/evaluation.py',
    'models/evaluation_store.py', 'models/evaluation_checkpoint.py', 'models/vjepa2.py',
    'models/pilot.py', 'models/runtime.py', 'models/diagnostics.py', 'attacks/pgd.py',
    'datasets/staged_files.py', 'datasets/staged_subset.py', 'datasets/staged_preparation.py',
    'datasets/video.py', 'models/execution.py', 'models/native_patch.py', 'models/patch_embed.py',
)


def panel_view(panel, indices, *, selection):
    if (not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(panel.rows) for i in indices)):
        raise ValueError('Invalid diagnostic panel indices.')
    result = copy.copy(panel)
    for name in ('rows', 'entries', 'paths'):
        values = getattr(panel, name)
        setattr(result, name, [values[i] for i in indices])
    result.identity = {**panel.identity, 'diagnostic_selection': selection,
                       'ordered_ids_sha256': json_hash([r['video_id'] for r in result.rows])}
    return result


def diagnostic_panels(root, manifests, clips, *, monitor_limit=1):
    if monitor_limit not in (1, 2):
        raise ValueError('Only one or two monitoring clips are allowed for this diagnostic.')
    training = PreparedStagedPanel(root, manifests, clips, 'train')
    monitor = PreparedStagedPanel(root, manifests, clips, 'train_monitor')
    # Include the first two declared monitors, then fill with training rows in
    # existing manifest order. This guarantees two classes for the clean probe.
    selected = {r['video_id'] for r in monitor.rows[:2]}
    for row in training.rows:
        if len(selected) >= 8:
            break
        selected.add(row['video_id'])
    indices = [i for i, row in enumerate(training.rows) if row['video_id'] in selected]
    training = panel_view(training, indices, selection='first-two-monitors-then-manifest-fill-to-eight-v1')
    monitor = panel_view(monitor, list(range(monitor_limit)), selection='first-declared-training-monitors-v1')
    if (len(training.rows) != 8 or len({r['class_id'] for r in training.rows}) < 2
            or not {r['video_id'] for r in monitor.rows} <= selected):
        raise ValueError('Diagnostic needs eight training clips including two classes and its monitors.')
    training.verify_all()
    return training, monitor


def prepare_context(root, args):
    root = Path(root)
    torch.set_num_threads(4)
    require_device(args.device)
    plan_path = root / 'configs/budget_study_plan.yaml'
    plan = yaml.safe_load(plan_path.read_text())
    study_path = root / plan['parent_study']
    if file_sha256(study_path) != plan['source_sha256'][plan['parent_study']]:
        raise ValueError('Parent protocol changed; revise the proposal first.')
    study = yaml.safe_load(study_path.read_text())
    if study['model']['upstream_commit'] != UPSTREAM_COMMIT:
        raise ValueError('Upstream revision differs from the model adapter.')
    training, monitor = diagnostic_panels(root, args.manifests, args.clips, monitor_limit=args.limit)
    model = build_vitl(args.upstream, activation_checkpointing=True)
    original = load_checkpoint(model, args.checkpoint, study['model']['checkpoint_sha256'])
    apply_model_policy(model, getattr(args, 'execution_policy', 'original'))
    model.to(args.device)
    checkpoint = {'epoch': 0, 'condition': 'pretrained', 'seed': None,
                  'training_contract_sha256': None, 'file_sha256': original['sha256']}
    provenance = {'diagnostic_only': True, 'research_admitted': False,
                  'checkpoint_sha256': original['sha256'], 'upstream_commit': UPSTREAM_COMMIT,
                  'plan_sha256': file_sha256(plan_path), 'study_sha256': file_sha256(study_path),
                  'model_execution': model_execution_contract(model),
                  'training_panel': training.identity, 'monitor_panel': monitor.identity,
                  'upstream_mask_config_sha256': file_sha256(args.upstream / PRETRAIN_CONFIG),
                  'source_sha256': {name: file_sha256(root / name) for name in SOURCES}}
    return SimpleNamespace(model=model, training=training, monitor=monitor,
                           sampler=EvaluationMaskSampler(args.upstream),
                           checkpoint=checkpoint, provenance=provenance, plan=plan,
                           settings=TrainingSettings.from_plan(plan))


def cached_training_loader(model, panel, device):
    guard = ExecutionGuard(model, device, target_versions=True)
    targets = [model.clean_targets(panel.pixels(i).to(device)).cpu() for i in range(len(panel.rows))]
    guard.check()

    def load(index):
        guard.check()
        return panel.pixels(index), targets[index]

    return load
