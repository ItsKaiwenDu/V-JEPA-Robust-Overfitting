"""CPU plumbing fixtures, never pretrained/GPU/calibration research evidence."""

import copy
from dataclasses import asdict
import json

import pytest
import torch

from models.candidate_contract import BASE_SEED, ROOT, bind_provenance, binding
from models.candidate_evaluation import EvaluationSettings, evaluate_clip, model_identity, seeds_for, SEED_SCHEMA
from models.candidate_evaluation_store import EvaluationStore, evaluate_panel, validate_records, select_primary_checkpoint
from models.candidate_calibration import calibration_record, run_calibration, load_calibration, verify_calibration
from models.evaluation import EvaluationSettings as LegacySettings
from models.evaluation_store import validate_records as legacy_validate
from models.training import runtime_contract
from models.study_artifacts import read_record, write_record
from test_evaluation import Panel, clip, tiny_masks, scientific
from test_vjepa2 import model


def candidates(identity, rows, *, provenance=None, runtime=None):
    """Fully explicit synthetic losses; not substituted for real calibration."""
    result = []
    for size, loss in zip((.5, 1., 2.), (2., 3., 3.)):
        settings = EvaluationSettings.for_method('primary', size)
        values = asdict(settings)
        contract = dict(kind='candidate_eot8_panel_evaluation', schema_version=2,
            checkpoint=dict(epoch=0, condition='pretrained', seed=None, training_contract_sha256=None, file_sha256='original'),
            panel='train_monitor', panel_identity={'data': 'fixture'},
            video_ids=[r['video_id'] for r in rows], source_group_ids=[r['source_group_id'] for r in rows],
            weights_sha256=copy.deepcopy(identity), settings=values, seed_schema=SEED_SCHEMA,
            base_seed=BASE_SEED, sensitivity_banks=0, provenance=copy.deepcopy(provenance or bind_provenance({})),
            runtime=copy.deepcopy(runtime or runtime_contract('cpu')))
        records = [dict(video_id=r['video_id'], settings=values, method='primary', clean_loss=1.,
            adversarial_loss=loss, attack_clean_loss=1., attack_loss=4., max_pixel_delta=.01, seconds=.1,
            seeds=seeds_for(settings, r['video_id']), scoring_audits=[]) for r in rows]
        result.append(dict(contract=contract, records=records, complete=True))
    return result


def test_explicit_primary_eight_and_unchanged_legacy_guard():
    candidate = EvaluationSettings.for_method('primary')
    assert (candidate.steps, candidate.restarts, candidate.attack_masks, candidate.scoring_masks) == (20, 2, 8, 4)
    assert LegacySettings.for_method('primary').attack_masks == 4
    with pytest.raises(ValueError):
        LegacySettings('primary', 20, 2, 8, 4)
    with pytest.raises(ValueError):
        EvaluationSettings('primary', 20, 2, 4, 4)


@pytest.mark.parametrize('change', [dict(profile='legacy'), dict(step_size_255=True),
    dict(epsilon_255=True), dict(steps=True), dict(attack_masks=16), dict(step_size_255=4)])
def test_strength_and_version_refused(change):
    with pytest.raises(ValueError):
        EvaluationSettings(**{**asdict(EvaluationSettings.for_method('primary')), **change})


def test_all_two_hundred_actual_seeds_match_frozen_proposal_and_step_only_control():
    candidate = json.loads((ROOT / 'configs/research_attack_candidate_v1.json').read_text())
    actual = []
    for row in candidate['calibration']['rows']:
        primary = seeds_for(EvaluationSettings.for_method('primary'), row['video_id'], sensitivity_banks=2)
        assert primary == seeds_for(EvaluationSettings.for_method('convergence'), row['video_id'], sensitivity_banks=2)
        assert primary['attack'] == row['attack_seeds'] and primary['start'] == row['start_seed']
        assert primary['score'] == row['scoring_seeds']['main']
        assert primary['audit_0'] == row['scoring_seeds']['audit_0']
        assert primary['audit_1'] == row['scoring_seeds']['audit_1']
        actual.extend(s % (2**31 - 1) for k, group in primary.items() if k != 'start' for s in group)
    assert len(actual) == len(set(actual)) == 200


def test_real_primary_replay_keeps_rng_modes_weights_target_and_pixel_bound(model, clip):
    model.train()
    identity, rng = model_identity(model), torch.get_rng_state().clone()
    settings = EvaluationSettings.for_method('primary')
    first, a = evaluate_clip(model, clip, tiny_masks, settings, video_id='a', sensitivity_banks=2)
    second, b = evaluate_clip(model, clip, tiny_masks, settings, video_id='a', sensitivity_banks=2)
    assert torch.equal(a, b) and scientific(first) == scientific(second)
    assert model_identity(model) == identity and torch.equal(rng, torch.get_rng_state())
    assert model.training and not model.target_encoder.training
    assert all(p.grad is None for p in model.parameters())
    assert first['attack_loss'] >= first['attack_clean_loss']
    assert first['max_pixel_delta'] <= 4/255 + 1e-7


def test_independent_scoring_and_audits_cannot_select_perturbations(model, clip):
    settings = EvaluationSettings.for_method('primary')
    seed = seeds_for(settings, 'a', sensitivity_banks=2)
    scoring = set(seed['score'] + seed['audit_0'] + seed['audit_1'])
    def altered(s):
        return tiny_masks(s + 17 if s in scoring else s)
    first, a = evaluate_clip(model, clip, tiny_masks, settings, video_id='a', sensitivity_banks=2)
    second, b = evaluate_clip(model, clip, altered, settings, video_id='a', sensitivity_banks=2)
    assert torch.equal(a, b) and first['attack_loss'] == second['attack_loss']
    assert first['adversarial_loss'] != second['adversarial_loss']


def test_failure_restores_rng_and_modes(model, clip, monkeypatch):
    model.train()
    rng = torch.get_rng_state().clone()
    def fail(*args, **kwargs):
        torch.rand(3)
        raise RuntimeError('interrupted')
    monkeypatch.setattr('models.candidate_evaluation.pgd_linf', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        evaluate_clip(model, clip, tiny_masks, EvaluationSettings.for_method('primary'), video_id='a')
    assert model.training and not model.target_encoder.training and torch.equal(rng, torch.get_rng_state())


def test_atomic_per_clip_resume_and_legacy_cache_refusal(model, clip, tmp_path):
    panel = Panel(clip)
    kwargs = dict(checkpoint={'epoch': 0}, provenance=bind_provenance({'fixture': True}))
    settings = EvaluationSettings.for_method('routine')
    def interrupt(*args):
        raise RuntimeError('disconnect')
    with pytest.raises(RuntimeError):
        evaluate_panel(model, panel, tiny_masks, settings, EvaluationStore(tmp_path / 'run'), on_clip=interrupt, **kwargs)
    panel.loaded.clear()
    value = evaluate_panel(model, panel, tiny_masks, settings, EvaluationStore(tmp_path / 'run', resume=True), **kwargs)
    assert value['complete'] and panel.loaded == [1]
    with pytest.raises((ValueError, TypeError)):
        legacy_validate(value)
    old = copy.deepcopy(value)
    old['contract'].update(kind='native_panel_evaluation', schema_version=1)
    old['contract']['settings'].pop('profile')
    with pytest.raises(ValueError):
        validate_records(old)


@pytest.mark.parametrize('defect', ['old_strength', 'old_kind', 'seed', 'source', 'runtime', 'weights',
                                  'rows', 'incomplete', 'missing', 'audit', 'nonfinite'])
def test_calibration_evidence_cannot_be_mixed_or_relabeled(defect):
    rows = [dict(video_id='a', source_group_id='g')]
    values = candidates({'encoder': 'initial'}, rows)
    c, r = values[1]['contract'], values[1]['records'][0]
    if defect == 'old_strength': c['settings']['attack_masks'] = 4
    elif defect == 'old_kind': c['kind'] = 'native_panel_evaluation'
    elif defect == 'seed': r['seeds']['attack'][0] += 1
    elif defect == 'source': c['provenance']['candidate_eot8']['source_sha256']['attacks/pgd.py'] = 'stale'
    elif defect == 'runtime': c['runtime']['threads'] += 1
    elif defect == 'weights': c['weights_sha256']['encoder'] = 'other'
    elif defect == 'rows': c['source_group_ids'] = ['other']
    elif defect == 'incomplete': values[1]['complete'] = False
    elif defect == 'missing': values.pop()
    elif defect == 'audit': r['scoring_audits'] = [{'clean_loss': 1., 'adversarial_loss': 10000.}]
    else: r['adversarial_loss'] = float('nan')
    with pytest.raises(ValueError):
        calibration_record(values, rows, diagnostic=True)


def test_smaller_exact_tie_uses_main_mean_only_and_original_ten_rows():
    config = json.loads((ROOT / 'configs/research_attack_candidate_v1.json').read_text())
    rows = config['calibration']['rows']
    values = candidates({'encoder': 'initial'}, rows)
    values[0]['records'][0]['attack_loss'] = 10000.
    record = calibration_record(values, rows)
    assert record['chosen_step_size_255'] == 1. and not record['diagnostic_only']
    verify_calibration(record, original_identity={'encoder': 'initial'}, expected_rows=rows)
    with pytest.raises(ValueError, match='original first ten'):
        calibration_record(values, list(reversed(rows)))


def test_real_diagnostic_bundle_resume_and_pinned_candidate_files(model, clip, tmp_path):
    panel = Panel(clip)
    panel.rows = panel.rows[:1]
    kwargs = dict(checkpoint=dict(epoch=0, condition='pretrained', seed=None,
        training_contract_sha256=None, file_sha256='original'), provenance=bind_provenance({}), diagnostic=True)
    record = run_calibration(model, panel, tiny_masks, tmp_path / 'bundle', **kwargs)
    assert panel.loaded == [0, 0, 0]
    panel.loaded.clear()
    assert run_calibration(model, panel, tiny_masks, tmp_path / 'bundle', **kwargs) == record and not panel.loaded
    assert load_calibration(tmp_path / 'bundle', panel.rows, original_identity=model_identity(model),
                            model=model, diagnostic=True) == record
    path = tmp_path / 'bundle' / record['evaluation_files'][0]['file']
    altered = read_record(path)
    altered['records'][0]['adversarial_loss'] += 1
    write_record(path, altered, replace=True)
    with pytest.raises(ValueError, match='checksum'):
        load_calibration(tmp_path / 'bundle', panel.rows, original_identity=model_identity(model), model=model, diagnostic=True)


def test_full_calibration_and_heldout_execution_remain_gated_before_pixels(model, clip, tmp_path):
    panel = Panel(clip)
    with pytest.raises(ValueError, match='gated'):
        run_calibration(model, panel, tiny_masks, tmp_path / 'full', checkpoint={}, provenance=bind_provenance({}))
    for name in ('confirmation_validation', 'test'):
        panel.panel = name
        with pytest.raises(ValueError, match='not open'):
            evaluate_panel(model, panel, tiny_masks, EvaluationSettings.for_method('primary'),
                           EvaluationStore(tmp_path / name), checkpoint={}, provenance=bind_provenance({}))
    assert not panel.loaded and not (tmp_path / 'full').exists()


def test_selection_needs_complete_eot8_anchors_and_keeps_initialization_and_earliest_tie():
    rows = [dict(video_id='a', source_group_id='g')]
    values = []
    for epoch, score in ((0, 4.), (1, 2.), (15, 2.), (30, 3.)):
        value = candidates({'encoder': str(epoch), 'target_encoder': 'fixed'}, rows)[1]
        value['contract']['panel'] = 'exploration_validation'
        value['contract']['checkpoint'] = dict(epoch=epoch, condition='clean-finetune' if epoch else 'pretrained',
            seed=0 if epoch else None, training_contract_sha256='training' if epoch else None)
        value['records'][0]['adversarial_loss'] = score
        values.append(value)
    kwargs = dict(phase='exploration', condition='clean-finetune', seed=0, epochs=[0, 1, 15, 30],
                  training_contract_sha256='training')
    result = select_primary_checkpoint(values, **kwargs)
    assert result['selected']['epoch'] == 1 and result['final_minus_selected_loss'] == 1
    with pytest.raises(ValueError, match='Missing'):
        select_primary_checkpoint(values[1:], **kwargs)
