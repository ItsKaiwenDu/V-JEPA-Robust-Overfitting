import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from models.evaluation import (EvaluationMaskSampler, EvaluationSettings, evaluate_clip,
                               evaluation_seed, model_identity)
from models.evaluation_store import EvaluationStore, evaluate_panel, select_primary_checkpoint
from models.vjepa2 import MaskPair
from test_vjepa2 import model


def tiny_masks(seed):
    order = torch.randperm(32, generator=torch.Generator().manual_seed(seed % (2**31 - 1))).view(1, -1)
    return [MaskPair(order[:, :8], order[:, 8:]), MaskPair(order[:, :24], order[:, 24:])]


@pytest.fixture
def clip():
    return .2 + .6 * torch.rand((1, 3, 4, 16, 16), generator=torch.Generator().manual_seed(27))


def scientific(record):
    return {k: v for k, v in record.items() if k != 'seconds'}


def test_primary_replay_preserves_modes_rng_parameters_and_targets(model, clip):
    model.train()
    before = model_identity(model)
    rng = torch.get_rng_state().clone()
    settings = EvaluationSettings.for_method('primary')
    first, a = evaluate_clip(model, clip, tiny_masks, settings, video_id='first', sensitivity_banks=2)
    second, b = evaluate_clip(model, clip, tiny_masks, settings, video_id='first', sensitivity_banks=2)
    assert torch.equal(a, b) and scientific(first) == scientific(second)
    assert model_identity(model) == before and torch.equal(rng, torch.get_rng_state())
    assert model.training and model.encoder.training and not model.target_encoder.training
    assert all(p.grad is None for p in model.parameters())
    assert first['attack_loss'] >= first['attack_clean_loss']
    assert first['max_pixel_delta'] <= 4/255 + 1e-7
    assert len(first['scoring_audits']) == 2
    assert set(first['seeds']['attack']).isdisjoint(first['seeds']['score'])


def test_scoring_bank_cannot_change_selected_perturbation(model, clip):
    settings = EvaluationSettings.for_method('routine')
    score_seed = evaluation_seed(20261020, 'routine_score', 'video')
    def altered(seed):
        return tiny_masks(seed + 73 if seed == score_seed else seed)
    first, a = evaluate_clip(model, clip, tiny_masks, settings, video_id='video')
    second, b = evaluate_clip(model, clip, altered, settings, video_id='video')
    assert torch.equal(a, b) and first['attack_loss'] == second['attack_loss']
    assert first['adversarial_loss'] != second['adversarial_loss']


@pytest.mark.parametrize('method', ['convergence', 'fgsm', 'random'])
def test_diagnostics_share_primary_scoring_seeds_and_keep_clean_targets(model, clip, method):
    settings = EvaluationSettings.for_method(method)
    record, adversarial = evaluate_clip(model, clip, tiny_masks, settings, video_id='video')
    assert record['seeds']['score'] == [evaluation_seed(20261020, 'primary_score', 'video', draw=i) for i in range(4)]
    assert not adversarial.requires_grad
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_evaluation_sampler_accepts_hash_range_without_corrupting_upstream_counter(upstream):
    sampler = EvaluationMaskSampler(upstream, frames=4, crop_size=16, patch_size=4)
    seed = evaluation_seed(20261020, 'primary_attack', 'video')
    assert seed > 2**31
    rng = torch.get_rng_state().clone()
    a, b = sampler(seed), sampler(seed)
    assert all(torch.equal(x.context, y.context) and torch.equal(x.target, y.target) for x, y in zip(a, b))
    assert torch.equal(rng, torch.get_rng_state())


def test_evaluation_preserves_training_state_even_after_failed_attack(model, clip, monkeypatch):
    model.train()
    before = torch.get_rng_state().clone()
    def failure(*args, **kwargs):
        torch.rand(10)
        raise RuntimeError('interrupted')
    monkeypatch.setattr('models.evaluation.pgd_linf', failure)
    with pytest.raises(RuntimeError, match='interrupted'):
        evaluate_clip(model, clip, tiny_masks, EvaluationSettings.for_method('routine'), video_id='a')
    assert model.training and not model.target_encoder.training
    assert torch.equal(before, torch.get_rng_state())


class Panel:
    panel = 'train_monitor'
    identity = {'manifest_sha256': 'fixture', 'index_sha256': 'fixture'}
    rows = [dict(video_id='a', source_group_id='g1'), dict(video_id='b', source_group_id='g2')]
    def __init__(self, clip):
        self.clip, self.loaded = clip, []
    def pixels(self, index):
        self.loaded.append(index)
        return self.clip.clone()


CHECKPOINT = dict(epoch=0, condition='pretrained', seed=None, training_contract_sha256=None)


def test_interrupted_evaluation_resumes_only_unfinished_clips(model, clip, tmp_path):
    settings = EvaluationSettings.for_method('routine')
    panel = Panel(clip)
    directory = tmp_path / 'interrupted'
    def interrupt(index, record):
        raise RuntimeError('lost connection')
    kwargs = dict(checkpoint=CHECKPOINT, provenance={'fixture': True})
    with pytest.raises(RuntimeError, match='lost connection'):
        evaluate_panel(model, panel, tiny_masks, settings, EvaluationStore(directory), on_clip=interrupt, **kwargs)
    assert panel.loaded == [0]
    panel.loaded.clear()
    resumed = evaluate_panel(model, panel, tiny_masks, settings, EvaluationStore(directory, resume=True), **kwargs)
    assert panel.loaded == [1]
    uninterrupted = evaluate_panel(model, Panel(clip), tiny_masks, settings, EvaluationStore(tmp_path / 'reference'), **kwargs)
    assert resumed['contract'] == uninterrupted['contract']
    assert [scientific(r) for r in resumed['records']] == [scientific(r) for r in uninterrupted['records']]
    assert resumed['complete']


@pytest.mark.parametrize('defect', ['weights', 'settings', 'panel', 'provenance', 'corruption'])
def test_evaluation_cannot_resume_stale_or_corrupted_results(model, clip, tmp_path, defect):
    panel = Panel(clip)
    settings = EvaluationSettings.for_method('routine')
    provenance = {'fixture': True}
    evaluate_panel(model, panel, tiny_masks, settings, EvaluationStore(tmp_path / 'run'),
                   checkpoint=CHECKPOINT, provenance=provenance)
    panel.loaded.clear()
    if defect == 'weights':
        with torch.no_grad():
            next(model.encoder.parameters()).add_(.01)
    elif defect == 'settings':
        settings = EvaluationSettings.for_method('primary')
    elif defect == 'panel':
        panel.rows = list(reversed(panel.rows))
    elif defect == 'provenance':
        provenance['fixture'] = False
    else:
        path = tmp_path / 'run/evaluation.json'
        wrapper = json.loads(path.read_text())
        wrapper['payload']['records'][0]['adversarial_loss'] += 1
        path.write_text(json.dumps(wrapper))
    with pytest.raises(ValueError):
        evaluate_panel(model, panel, tiny_masks, settings, EvaluationStore(tmp_path / 'run', resume=True),
                       checkpoint=CHECKPOINT, provenance=provenance)
    assert not panel.loaded


def selection_fixture():
    values = []
    settings = asdict(EvaluationSettings.for_method('primary'))
    for epoch, losses in [(0, [2., 4.]), (1, [1., 3.]), (15, [1., 3.]), (30, [2., 3.])]:
        checkpoint = dict(epoch=epoch, condition='adversarial-finetune' if epoch else 'pretrained',
                          seed=0 if epoch else None, training_contract_sha256='training' if epoch else None)
        contract = dict(checkpoint=checkpoint, panel='exploration_validation', settings=settings,
                        video_ids=['a', 'b'], weights_sha256={'target_encoder': 'fixed', 'encoder': str(epoch)},
                        panel_identity='fixed-panel', provenance='fixed-source')
        records = [dict(video_id=v, settings=settings, method='primary', clean_loss=1., adversarial_loss=loss,
                        attack_clean_loss=1., attack_loss=4., max_pixel_delta=.01, seconds=.1)
                   for v, loss in zip(['a', 'b'], losses)]
        values.append(dict(contract=contract, records=records, complete=True))
    return values


def select(values):
    return select_primary_checkpoint(values, phase='exploration', condition='adversarial-finetune',
                                      seed=0, epochs=[0, 1, 15, 30], training_contract_sha256='training')


def test_primary_selection_includes_initialization_earliest_ties_and_correct_gap():
    values = selection_fixture()
    chosen = select(values)
    assert chosen['selected']['epoch'] == 1
    assert chosen['final_minus_selected_loss'] == .5
    assert not chosen['test_data_used'] and not chosen['research_result']
    for record in values[0]['records']:
        record['adversarial_loss'] = .2
    assert select(values)['selected']['epoch'] == 0


@pytest.mark.parametrize('defect', ['missing', 'routine', 'test', 'train', 'incomplete', 'seed',
                                  'target', 'settings', 'data', 'duplicate', 'weak_primary', 'nonfinite'])
def test_invalid_selection_input_is_rejected(defect):
    values = selection_fixture()
    if defect == 'missing':
        values.pop()
    elif defect == 'routine':
        values[1]['contract']['settings'] = asdict(EvaluationSettings.for_method('routine'))
    elif defect in ('test', 'train'):
        values[1]['contract']['panel'] = defect
    elif defect == 'incomplete':
        values[1]['complete'] = False
    elif defect == 'seed':
        values[1]['contract']['checkpoint']['seed'] = 11
    elif defect == 'target':
        values[1]['contract']['weights_sha256']['target_encoder'] = 'drifted'
    elif defect == 'settings':
        # The fixture intentionally shares its original settings object.
        values[1]['contract']['settings'] = {**values[1]['contract']['settings'], 'step_size_255': 2.}
    elif defect == 'data':
        values[1]['contract']['panel_identity'] = 'changed'
    elif defect == 'duplicate':
        values.append(copy.deepcopy(values[1]))
    elif defect == 'weak_primary':
        values[1]['contract']['settings']['steps'] = 5
    else:
        values[1]['records'][0]['adversarial_loss'] = float('nan')
    with pytest.raises(ValueError):
        select(values)


def test_failed_journal_commit_preserves_previous_completed_records(model, clip, tmp_path, monkeypatch):
    panel = Panel(clip)
    settings = EvaluationSettings.for_method('routine')
    directory = tmp_path / 'journal'
    def interrupt(index, record):
        raise RuntimeError('interrupt')
    with pytest.raises(RuntimeError):
        evaluate_panel(model, panel, tiny_masks, settings, EvaluationStore(directory),
                       checkpoint=CHECKPOINT, provenance={}, on_clip=interrupt)
    committed = (directory / 'evaluation.json').read_bytes()
    def fail(*args):
        raise OSError('rename failed')
    monkeypatch.setattr('models.evaluation_store.os.replace', fail)
    with pytest.raises(OSError, match='rename failed'):
        evaluate_panel(model, panel, tiny_masks, settings, EvaluationStore(directory, resume=True),
                       checkpoint=CHECKPOINT, provenance={})
    assert (directory / 'evaluation.json').read_bytes() == committed


def test_cli_refuses_unbounded_research_before_model_or_data_loading(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/evaluate.py'
    result = subprocess.run([sys.executable, str(script), '--checkpoint', 'missing', '--manifests', 'missing',
                             '--clips', 'missing', '--output', str(tmp_path / 'output')], capture_output=True, text=True)
    assert result.returncode == 2 and 'Only --diagnostic' in result.stderr
    assert not (tmp_path / 'output').exists()
