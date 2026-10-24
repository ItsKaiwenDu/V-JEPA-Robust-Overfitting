"""Opt-in shared-forward integration; small CPU models are not CUDA acceptance."""

import copy
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.checkpointing import CheckpointStore
from models.diagnostics import state_digest
from models.execution import (ExecutionGuard, SHARED_POLICIES, apply_model_policy,
                              configure_execution, model_execution_contract, precision_contract)
from models.precision_patch import FORWARD_ID, PrecisionPatchEmbed3D
from models.training import rng_state, runtime_contract
from models.workflow_diagnostic import cached_training_loader
from test_native_execution import restore_flags
from test_training import assert_tree_equal, make_engine
from test_vjepa2 import model, pixels

ROOT = Path(__file__).resolve().parents[1]


def integrated(model, policy='shared-fp64-native'):
    configure_execution(policy, device='cpu')
    apply_model_policy(model, policy)
    model.encoder.use_activation_checkpointing = True
    model.predictor.use_activation_checkpointing = True
    return model


@pytest.mark.parametrize('policy', SHARED_POLICIES)
def test_explicit_integration_preserves_original_parameters_keys_and_targets(model, pixels, policy):
    projection, target = model.encoder.patch_embed.proj, model.target_encoder.patch_embed
    ids, state = [id(p) for p in model.parameters()], copy.deepcopy(model.state_dict())
    clean_targets = model.clean_targets(pixels)
    integrated(model, policy)
    identity = model_execution_contract(model)
    assert model.encoder.patch_embed.proj is projection and model.target_encoder.patch_embed is target
    assert [id(p) for p in model.parameters()] == ids
    assert_tree_equal(state, model.state_dict())
    model.load_state_dict(state, strict=True)
    assert torch.equal(clean_targets, model.clean_targets(pixels))
    assert identity['schema_version'] == 2 and identity['diagnostic_only'] and not identity['study_adopted']
    precision = identity['precision']
    assert precision['forward'] == FORWARD_ID and precision['patch_accumulation'] == 'torch.float64'
    assert precision['patch_tokens'] == precision['parameters'] == precision['backward_accumulation'] == 'torch.float32'
    assert precision['backward'] == policy.removeprefix('shared-fp64-')
    assert identity == json.loads(json.dumps(identity))


@pytest.mark.parametrize('condition', ['clean-finetune', 'adversarial-finetune'])
@pytest.mark.parametrize('policy', SHARED_POLICIES)
@pytest.mark.parametrize('cut', [1, 2, 5])
def test_full_adamw_checkpoint_recovery_is_exact_at_partial_batch_warmup_and_decay(model, tmp_path, condition, policy, cut):
    original = integrated(model, policy)
    initial = {name: state_digest(getattr(original, name)) for name in ('encoder', 'predictor', 'target_encoder')}
    reference = make_engine(copy.deepcopy(original), condition=condition)
    while not reference.finished:
        reference.update()
    expected = reference.state()
    draws = (random.random(), np.random.rand(), torch.rand(3))
    interrupted = make_engine(copy.deepcopy(original), condition=condition)
    for _ in range(cut):
        interrupted.update()
    store = CheckpointStore(tmp_path / 'checkpoints')
    store.save(interrupted.state())
    saved, _ = store.load()
    resumed = make_engine(copy.deepcopy(original), condition=condition)
    random.random(), np.random.rand(), torch.rand(7)
    resumed.restore(saved)
    while not resumed.finished:
        resumed.update()
    assert_tree_equal(expected, resumed.state())
    assert random.random() == draws[0] and np.random.rand() == draws[1] and torch.equal(torch.rand(3), draws[2])
    assert state_digest(resumed.model.target_encoder) == initial['target_encoder']
    assert state_digest(resumed.model.encoder) != initial['encoder']
    assert state_digest(resumed.model.predictor) != initial['predictor']
    for group in resumed.optimizer.param_groups:
        assert any(p in resumed.optimizer.state and resumed.optimizer.state[p]['exp_avg'].abs().max() > 0 for p in group['params'])
    assert all(p.grad is None for p in resumed.model.parameters())
    assert all(not p.requires_grad for p in resumed.model.target_encoder.parameters())


@pytest.mark.parametrize('policy', SHARED_POLICIES)
@pytest.mark.parametrize('condition', ['clean-finetune', 'adversarial-finetune'])
def test_fresh_process_training_restore_matches_full_adamw_and_rng(tmp_path, policy, condition):
    from models.native_diagnostic import save_tiny_fixture
    fixture_path = tmp_path / 'initial-fixture.pt'
    initial = save_tiny_fixture(ROOT/'external/vjepa2', fixture_path, seed=20261022)
    worker = ROOT / 'tests/fixtures/shared_recovery_worker.py'
    for mode in ('reference', 'split', 'resume'):
        result = subprocess.run([sys.executable, str(worker), str(ROOT), str(tmp_path), policy, condition, mode, str(fixture_path), initial['sha256']],
                                cwd=ROOT, capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
    expected = torch.load(tmp_path / 'reference.pt', weights_only=True)
    actual = torch.load(tmp_path / 'resume.pt', weights_only=True)
    assert_tree_equal(expected, actual)
    records = [json.loads((tmp_path/(mode+'-initialization.json')).read_text()) for mode in ('reference','split','resume')]
    assert len({row['constructed_state_sha256'] for row in records}) == 3
    assert all(row['initial_fixture'] == initial and row['loaded_state_sha256'] == initial['state_sha256'] for row in records)


@pytest.mark.parametrize('policy', SHARED_POLICIES)
@pytest.mark.parametrize('condition', ['clean-finetune', 'adversarial-finetune'])
def test_scheduled_evaluations_and_interrupted_recovery_match_uninterrupted(model, tmp_path, policy, condition):
    from models.pilot_measurement import compare_recovery
    from test_schedule import create
    original = integrated(model, policy)
    full = create(original, tmp_path / 'reference', condition=condition)
    assert full.run(max_updates=8, max_evaluations=8)['complete']
    split = create(original, tmp_path / 'split', condition=condition)
    assert split.run(max_updates=1, max_evaluations=2)['completed_step'] == 1
    resumed = create(original, tmp_path / 'split', condition=condition, resume=True)
    assert resumed.run(max_updates=8, max_evaluations=8)['complete']
    result = compare_recovery(tmp_path / 'reference', tmp_path / 'split', expected_step=4)
    assert result['exact_match'], result
    assert_tree_equal(full.engine.state(), resumed.engine.state())


@pytest.mark.parametrize('fault', ['inside_panel', 'after_epoch', 'after_evaluation'])
def test_evaluation_barrier_survives_failures_under_shared_forward(model, tmp_path, monkeypatch, fault):
    import models.schedule as module
    from test_schedule import create
    original = integrated(model)
    workflow = create(original, tmp_path / 'run', monitor_count=2)
    if fault == 'inside_panel':
        original_evaluate = module.evaluate_panel
        def interrupted(*args, **kwargs):
            def stop(i, record):
                raise RuntimeError('shared integration interrupted')
            return original_evaluate(*args, on_clip=stop, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(module, 'evaluate_panel', interrupted)
            with pytest.raises(RuntimeError, match='interrupted'):
                workflow.run(max_updates=4, max_evaluations=1)
    elif fault == 'after_epoch':
        def stop(event):
            if event['step'] == 2:
                raise RuntimeError('shared integration interrupted')
        with pytest.raises(RuntimeError, match='interrupted'):
            workflow.run(max_updates=4, max_evaluations=8, on_update=stop)
    else:
        def stop(*args, **kwargs):
            raise OSError('shared integration interrupted')
        with monkeypatch.context() as patch:
            patch.setattr(module, 'write_record', stop)
            with pytest.raises(OSError, match='interrupted'):
                workflow.run(max_updates=4, max_evaluations=1)
    resumed = create(original, tmp_path / 'run', monitor_count=2, resume=True)
    before_step = resumed.store.load()[0]['step']  # Restoration is lazy until run() takes the lock.
    loaded, panel = [], resumed.evaluator.panels['train_monitor']
    pixels_for = panel.pixels
    panel.pixels = lambda i: (loaded.append(i), pixels_for(i))[1]
    resumed.run(max_updates=0, max_evaluations=1)
    assert resumed.engine.step == before_step
    assert loaded == {'inside_panel': [1], 'after_epoch': [0, 1], 'after_evaluation': []}[fault]


@pytest.mark.parametrize('defect', ['forward', 'backward', 'geometry', 'precision', 'normalization',
                                  'target', 'target_mode', 'checkpointing', 'parameter', 'module', 'autocast'])
def test_cache_and_training_reject_execution_drift_before_update(model, pixels, defect):
    integrated(model)
    panel = SimpleNamespace(rows=[{'video_id': 'a'}], pixels=lambda i: pixels[:1])
    loader = cached_training_loader(model, panel, 'cpu')
    engine = make_engine(model)
    if defect == 'forward': model.encoder.patch_embed.forward_id = 'different-forward'
    elif defect == 'backward': model.encoder.patch_embed.backward_backend = 'cudnn'
    elif defect == 'geometry': model.encoder.patch_embed.proj.stride = (1, 4, 4)
    elif defect == 'precision': model.encoder.patch_embed.proj.double()
    elif defect == 'normalization': model.pixel_std.add_(.01)
    elif defect == 'target':
        with torch.no_grad(): next(model.target_encoder.parameters()).add_(.01)
    elif defect == 'target_mode': model.target_encoder.train()
    elif defect == 'checkpointing': model.encoder.use_activation_checkpointing = False
    elif defect == 'parameter':
        p = model.encoder.patch_embed.proj
        p.weight = torch.nn.Parameter(p.weight.detach().clone())
    elif defect == 'module': model.encoder.patch_embed = copy.copy(model.encoder.patch_embed)
    before, optimizer = state_digest(model), copy.deepcopy(engine.optimizer.state_dict())
    with torch.autocast('cpu', enabled=defect == 'autocast'):
        for operation in (lambda: loader(0), engine.update, engine.state, lambda: engine.restore({})):
            with pytest.raises(ValueError): operation()
    assert engine.step == 0 and state_digest(model) == before
    assert_tree_equal(engine.optimizer.state_dict(), optimizer)


@pytest.mark.parametrize('source_policy', ['original', 'native-fp32', 'shared-fp64-cudnn'])
def test_resume_and_anchor_cannot_cross_forward_or_backward_identity(model, tmp_path, source_policy, monkeypatch):
    from models.evaluation_checkpoint import load_anchor
    from test_evaluation_checkpoint import trained_anchor
    source = copy.deepcopy(model)
    configure_execution(source_policy, device='cpu')
    apply_model_policy(source, source_policy)
    source.encoder.use_activation_checkpointing = source.predictor.use_activation_checkpointing = True
    trained = trained_anchor(source, tmp_path)
    saved = trained.state()  # Serialize before changing the process-global execution flags.
    destination = integrated(model)
    engine = make_engine(destination, provenance={'checkpoint_sha256': 'original'})
    before = engine.state()
    def forbidden(*args, **kwargs):
        raise AssertionError('Mismatch must reject before loading any trainable tensor or optimizer state')
    monkeypatch.setattr(engine.model.encoder, 'load_state_dict', forbidden)
    monkeypatch.setattr(engine.model.predictor, 'load_state_dict', forbidden)
    monkeypatch.setattr(engine.optimizer, 'load_state_dict', forbidden)
    with pytest.raises(ValueError, match='Resume contract differs'): engine.restore(saved)
    with pytest.raises(ValueError, match='execution runtime/backend'): load_anchor(destination, tmp_path, 1, initial_checkpoint_sha256='original')
    assert_tree_equal(before, engine.state())


@pytest.mark.parametrize('policy', SHARED_POLICIES)
def test_matching_saved_anchor_loads_trained_weights_and_keeps_original_target(model, tmp_path, policy):
    from models.evaluation_checkpoint import load_anchor
    from models.evaluation import model_identity
    from test_evaluation_checkpoint import trained_anchor
    integrated(model, policy)
    trained = trained_anchor(model, tmp_path)
    before_target = state_digest(model.target_encoder)
    reference = load_anchor(model, tmp_path, 1, initial_checkpoint_sha256='original')
    assert model_identity(model) == model_identity(trained.model)
    assert state_digest(model.target_encoder) == before_target
    assert reference['training_contract_sha256'] == trained.contract_sha256


def test_precision_free_resume_or_anchor_is_rejected_before_weights_load(model, tmp_path, monkeypatch):
    from models.evaluation_checkpoint import load_anchor
    from models.checkpointing import file_sha256
    from test_evaluation_checkpoint import trained_anchor
    integrated(model)
    trained = trained_anchor(model, tmp_path)
    invalid = trained.state()
    del invalid['contract']['runtime']['model_execution']['precision']
    engine = make_engine(model, provenance={'checkpoint_sha256': 'original'})
    with pytest.raises(ValueError, match='Resume contract differs'): engine.restore(invalid)
    # Preserve valid serialization checksums so rejection tests the precision
    # contract itself rather than corruption of the envelope.
    manifest_path = tmp_path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    ref = manifest['anchors']['1']
    path = tmp_path / ref['file']
    payload = torch.load(path, weights_only=True)
    del payload['contract']['runtime']['model_execution']['precision']
    import hashlib
    digest = hashlib.sha256(json.dumps(payload['contract'], sort_keys=True).encode()).hexdigest()
    payload['contract_sha256'] = manifest['contract_sha256'] = digest
    torch.save(payload, path)
    ref['sha256'] = file_sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    before = state_digest(model)
    def forbidden(*args, **kwargs): raise AssertionError('No trainable weights may be loaded')
    monkeypatch.setattr(model.encoder, 'load_state_dict', forbidden)
    monkeypatch.setattr(model.predictor, 'load_state_dict', forbidden)
    with pytest.raises(ValueError, match='execution runtime/backend'): load_anchor(model, tmp_path, 1, initial_checkpoint_sha256='original')
    assert state_digest(model) == before


@pytest.mark.parametrize('source_policy', ['cudnn-fp32', 'shared-fp64-cudnn'])
def test_calibration_evaluation_and_feature_probe_refuse_different_precision_policy(model, pixels, tmp_path, source_policy):
    from models.calibration import load_calibration, run_calibration
    from models.evaluation import EvaluationSettings, model_identity
    from models.evaluation_store import EvaluationStore, evaluate_panel
    from models.features import extract_features, fit_ridge_probe, score_ridge_probe
    from test_evaluation import tiny_masks
    source = copy.deepcopy(model)
    configure_execution(source_policy, device='cpu')
    apply_model_policy(source, source_policy)
    rows = [dict(video_id=f'v{i}', source_group_id=f'g{i}', class_id=str(i)) for i in range(2)]
    panel = SimpleNamespace(panel='train_monitor', identity={'data': 'generated-fixture'}, rows=rows,
                            pixels=lambda i: pixels[i:i+1])
    checkpoint = dict(epoch=0, condition='pretrained', seed=None, training_contract_sha256=None, file_sha256='original')
    run_calibration(source, panel, tiny_masks, tmp_path/'calibration', checkpoint=checkpoint, provenance={}, diagnostic=True)
    store = EvaluationStore(tmp_path/'evaluation')
    evaluate_panel(source, panel, tiny_masks, EvaluationSettings.for_method('fgsm'), store, checkpoint=checkpoint, provenance={})
    panel.panel = 'train'
    probe = fit_ridge_probe(extract_features(source, panel, provenance={}))
    integrated(model)
    assert model_identity(source) == model_identity(model)  # Equal weights do not make execution identities equal.
    with pytest.raises(ValueError, match='Calibration execution'):
        load_calibration(tmp_path/'calibration', rows, original_identity=model_identity(model), diagnostic=True, model=model)
    panel.panel = 'train_monitor'
    with pytest.raises(ValueError, match='resume contract changed'):
        evaluate_panel(model, panel, tiny_masks, EvaluationSettings.for_method('fgsm'),
                       EvaluationStore(tmp_path/'evaluation', resume=True), checkpoint=checkpoint, provenance={})
    with pytest.raises(ValueError): score_ridge_probe(probe, extract_features(model, panel, provenance={}))


@pytest.mark.parametrize('policy', SHARED_POLICIES)
def test_shared_policy_rejects_mps_unvalidated_cuda_and_undeclared_wrapper(model, monkeypatch, policy):
    with pytest.raises(ValueError, match='CPU fixtures and CUDA'): configure_execution(policy, device='mps')
    with monkeypatch.context() as patch:
        patch.setattr(torch, '__version__', '2.9.0+cu130')
        with pytest.raises(ValueError, match='PyTorch 2.8.0'): configure_execution(policy, device='cuda')
    integrated(model, policy)
    model._diagnostic_execution_policy = 'cudnn-fp32'
    with pytest.raises(ValueError, match='Actual patch backend'): model_execution_contract(model)


def test_shared_policy_is_explicit_in_bounded_cli_and_source_provenance():
    from models.workflow_diagnostic import SOURCES
    assert 'models/precision_patch.py' in SOURCES
    for name in ('check_workflow.py', 'run_workflow_pilot.py'):
        result = subprocess.run([sys.executable, str(ROOT/'scripts'/name), '--help'], capture_output=True, text=True)
        assert result.returncode == 0
        assert all(policy in result.stdout for policy in SHARED_POLICIES)
