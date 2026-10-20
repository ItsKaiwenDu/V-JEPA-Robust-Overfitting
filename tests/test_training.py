"""Actual upstream CPU models exercise interrupted native training and recovery."""

import copy
from dataclasses import replace
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from models.checkpointing import CheckpointStore, cpu_tree
from models.diagnostics import state_digest
from models.training import TrainingEngine, TrainingMaskSampler, TrainingSettings
from models.vjepa2 import MaskPair, sample_masks
from test_vjepa2 import model


def make_engine(model, *, condition='clean-finetune', seed=19, settings=None, provenance=None):
    settings = settings or TrainingSettings(epochs=3, effective_batch=2, decay_after_epochs=(1, 2))
    pixels = .2 + .6 * torch.rand(3, 3, 4, 16, 16, generator=torch.Generator().manual_seed(17))
    targets = model.clean_targets(pixels)

    def masks(seed):
        indices = torch.randperm(32, generator=torch.Generator().manual_seed(seed)).unsqueeze(0)
        return [MaskPair(indices[:, :8], indices[:, 8:]), MaskPair(indices[:, :24], indices[:, 24:])]

    return TrainingEngine(model, settings, ['video-a', 'video-b', 'video-c'],
        lambda i: (pixels[i:i+1], targets[i:i+1]), masks,
        condition=condition, seed=seed, provenance=provenance or {'fixture': 'v1'})


def assert_tree_equal(a, b):
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('condition', ['clean-finetune', 'adversarial-finetune'])
@pytest.mark.parametrize('cut', [1, 2, 5])
def test_resume_matches_uninterrupted_training_through_warmup_drops_and_partial_batches(model, tmp_path, condition, cut):
    original = copy.deepcopy(model)
    full = make_engine(copy.deepcopy(original), condition=condition)
    while not full.finished:
        full.update()
    expected = full.state()
    continuation_random = (random.random(), np.random.rand(), torch.rand(3))

    interrupted = make_engine(copy.deepcopy(original), condition=condition)
    for _ in range(cut):
        interrupted.update()
    store = CheckpointStore(tmp_path / 'checkpoints')
    store.save(interrupted.state())
    saved, _ = store.load()
    assert 'target_encoder' not in saved  # Reconstruct only from pinned fixed target.
    resumed = make_engine(copy.deepcopy(original), condition=condition)
    # Emulate a new process consuming RNG before restoration.
    random.random()
    np.random.rand()
    torch.rand(4)
    resumed.restore(saved)
    while not resumed.finished:
        resumed.update()
    assert_tree_equal(expected, resumed.state())
    assert random.random() == continuation_random[0]
    assert np.random.rand() == continuation_random[1]
    assert torch.equal(torch.rand(3), continuation_random[2])
    assert state_digest(resumed.model.target_encoder) == state_digest(original.target_encoder)


def test_last_partial_batch_matches_an_independent_single_example_update(model):
    engine = make_engine(model)
    engine.update()
    reference = make_engine(copy.deepcopy(model))
    reference.optimizer.load_state_dict(cpu_tree(engine.optimizer.state_dict()))
    index = engine.epoch_order(0)[2]
    event = engine.update()  # Final batch of epoch 1 contains one example.
    pixels, targets = reference.load_example(index)
    masks = reference.masks_for(event['mask_seeds'][0])
    for group, rate in zip(reference.optimizer.param_groups, event['learning_rates']):
        group['lr'] = rate
    reference.model.train()
    reference.optimizer.zero_grad(set_to_none=True)
    reference.model(pixels, targets, masks).backward()  # Divide by one, not configured batch two.
    reference.optimizer.step()
    assert state_digest(engine.model) == state_digest(reference.model)
    assert len(event['video_ids']) == 1


def test_learning_rate_boundaries_and_matched_condition_randomness(model):
    clean = make_engine(copy.deepcopy(model))
    adversarial = make_engine(copy.deepcopy(model), condition='adversarial-finetune')
    assert [clean.learning_rates(i)[0] / 1e-5 for i in range(6)] == pytest.approx([.5, 1, .1, .1, .01, .01])
    a, b = clean.update(), adversarial.update()
    for field in ('video_ids', 'mask_seeds', 'attack_seed', 'learning_rates'):
        assert a[field] == b[field]


def test_partial_update_failure_cannot_be_serialized(model, tmp_path):
    engine = make_engine(model)
    store = CheckpointStore(tmp_path)
    store.save(engine.state())
    original_load, calls = engine.load_example, []

    def fail_second(index):
        calls.append(index)
        if len(calls) == 2:
            raise RuntimeError('Simulated interruption inside accumulation')
        return original_load(index)

    engine.load_example = fail_second
    with pytest.raises(RuntimeError, match='Simulated interruption'):
        engine.update()
    with pytest.raises(RuntimeError, match='completed'):
        engine.state()
    assert store.load()[0]['step'] == 0
    assert engine.step == 0


@pytest.mark.parametrize('change', ['condition', 'seed', 'settings', 'provenance', 'target', 'tensor'])
def test_resume_mismatch_fails_before_changing_context(model, change):
    original = make_engine(copy.deepcopy(model))
    original.update()
    state = original.state()
    kwargs = {}
    if change == 'condition':
        kwargs['condition'] = 'adversarial-finetune'
    elif change == 'seed':
        kwargs['seed'] = 20
    elif change == 'settings':
        kwargs['settings'] = replace(original.settings, weight_decay=.1)
    elif change == 'provenance':
        kwargs['provenance'] = {'fixture': 'changed'}
    elif change == 'target':
        with torch.no_grad():
            next(model.target_encoder.parameters()).add_(.1)
    else:
        state['encoder'][next(iter(state['encoder']))] = torch.zeros(1)
    restored = make_engine(model, **kwargs)
    before = state_digest(restored.model)
    with pytest.raises(ValueError):
        restored.restore(state)
    assert state_digest(restored.model) == before


def test_failed_manifest_commit_preserves_last_good_state_and_resume_generations(model, tmp_path, monkeypatch):
    import models.checkpointing as checkpointing
    engine = make_engine(model)
    store = CheckpointStore(tmp_path)
    store.save(engine.state())
    engine.update()
    store.save(engine.state())
    engine.update()
    store.save(engine.state(), anchor_epoch=1)
    assert len(list(tmp_path.glob('resume-*.pt'))) == 2
    assert len(list(tmp_path.glob('epoch-*.pt'))) == 1
    manifest_before = (tmp_path / 'manifest.json').read_bytes()
    replace_file = checkpointing.os.replace

    def fail_manifest(source, destination):
        if Path(destination).name == 'manifest.json':
            raise OSError('Simulated manifest write failure')
        return replace_file(source, destination)

    monkeypatch.setattr(checkpointing.os, 'replace', fail_manifest)
    engine.update()
    with pytest.raises(OSError, match='Simulated'):
        store.save(engine.state())
    assert (tmp_path / 'manifest.json').read_bytes() == manifest_before
    assert store.load()[0]['step'] == 2
    assert store.load('previous')[0]['step'] == 1
    assert len(list(tmp_path.glob('epoch-*.pt'))) == 1


def test_corrupt_latest_is_rejected_without_silent_fallback(model, tmp_path):
    engine = make_engine(model)
    store = CheckpointStore(tmp_path)
    store.save(engine.state())
    engine.update()
    reference = store.save(engine.state())
    with (tmp_path / reference['file']).open('ab') as handle:
        handle.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        store.load()
    assert store.load('previous')[0]['step'] == 0


@pytest.mark.parametrize('defect', ['missing_moments', 'changed_betas'])
def test_resume_rejects_invalid_optimizer_before_model_changes(model, defect):
    engine = make_engine(copy.deepcopy(model))
    engine.update()
    state = engine.state()
    if defect == 'missing_moments':
        state['optimizer']['state'] = {}
    else:
        state['optimizer']['param_groups'][0]['betas'] = (.8, .9)
    restored = make_engine(model)
    before = state_digest(restored.model)
    with pytest.raises(ValueError, match='[Oo]ptimizer'):
        restored.restore(state)
    assert state_digest(restored.model) == before


def test_low_disk_space_does_not_replace_a_committed_checkpoint(model, tmp_path, monkeypatch):
    from collections import namedtuple
    import models.checkpointing as checkpointing
    engine = make_engine(model)
    store = CheckpointStore(tmp_path)
    store.save(engine.state())
    engine.update()
    usage = namedtuple('Usage', ['total', 'used', 'free'])
    monkeypatch.setattr(checkpointing.shutil, 'disk_usage', lambda path: usage(100, 99, 1))
    with pytest.raises(OSError, match='Insufficient free space'):
        store.save(engine.state())
    assert store.load()[0]['step'] == 0


def test_reading_a_missing_resume_directory_does_not_create_it(tmp_path):
    with pytest.raises(ValueError, match='no committed'):
        CheckpointStore(tmp_path / 'missing', create=False)
    assert not (tmp_path / 'missing').exists()


def test_cached_mask_sampler_matches_existing_upstream_sampler(upstream):
    sampler = TrainingMaskSampler(upstream)
    rng = torch.get_rng_state().clone()
    for seed in (4, 17, 4):
        actual, expected = sampler(seed), sample_masks(upstream, batch_size=1, seed=seed)
        for a, b in zip(actual, expected):
            assert torch.equal(a.context, b.context)
            assert torch.equal(a.target, b.target)
    assert torch.equal(torch.get_rng_state(), rng)


def test_cli_rejects_non_diagnostic_work_before_loading_models(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/train.py'
    result = subprocess.run([sys.executable, str(script), '--checkpoint', '/missing.pt', '--clips', '/missing',
        '--condition', 'clean-finetune', '--max-updates', '8', '--output', str(tmp_path / 'run')],
        capture_output=True, text=True)
    assert result.returncode == 2 and 'Only --diagnostic' in result.stderr
    assert not (tmp_path / 'run').exists()
