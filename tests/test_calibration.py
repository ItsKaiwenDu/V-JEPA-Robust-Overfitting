import copy
from dataclasses import asdict

import pytest

from models.calibration import CANDIDATES, calibration_record, load_calibration, run_calibration
from models.evaluation import EvaluationSettings, SEED_SCHEMA, model_identity
from models.evaluation_store import json_hash
from models.study_artifacts import read_record, write_record
from models.training import runtime_contract
from test_evaluation import Panel, clip, tiny_masks
from test_vjepa2 import model


def candidates(identity, rows, *, provenance=None, runtime=None):
    """Synthetic selection fixtures, not measured attack results."""
    values = []
    for size, score in zip(CANDIDATES, [2., 3., 3.]):
        settings = asdict(EvaluationSettings.for_method('primary', size))
        contract = dict(schema_version=1, kind='native_panel_evaluation',
            checkpoint=dict(epoch=0, condition='pretrained', seed=None, training_contract_sha256=None, file_sha256='original'),
            panel='train_monitor', panel_identity={'data': 'fixture'},
            video_ids=[r['video_id'] for r in rows], source_group_ids=[r['source_group_id'] for r in rows],
            weights_sha256=copy.deepcopy(identity), settings=settings, seed_schema=SEED_SCHEMA, base_seed=20261020,
            sensitivity_banks=0, provenance=copy.deepcopy(provenance or {}),
            runtime=copy.deepcopy(runtime or runtime_contract('cpu')))
        records = [dict(video_id=r['video_id'], settings=settings, method='primary', clean_loss=1.,
                        adversarial_loss=score, attack_clean_loss=1., attack_loss=4., max_pixel_delta=.01, seconds=.1)
                   for r in rows]
        values.append(dict(contract=contract, records=records, complete=True))
    return values


def test_step_choice_uses_independent_mean_and_smaller_step_for_ties():
    rows = [dict(video_id=f'v{i}', source_group_id=f'g{i}') for i in range(10)]
    values = candidates({'encoder': 'initial'}, rows)
    values[0]['records'][0]['attack_loss'] = 1000.  # Attack-bank score is not the selection score.
    record = calibration_record(values, rows)
    assert record['chosen_step_size_255'] == 1.
    assert not record['diagnostic_only'] and not record['validation_or_test_used']
    assert record['training_attack_step_size_255'] == 1.
    assert record['candidates'][0]['evaluation_sha256'] == json_hash(values[0])


@pytest.mark.parametrize('defect', ['incomplete', 'missing_candidate', 'duplicate_candidate', 'validation', 'test',
                                  'wrong_rows', 'different_weights', 'different_runtime', 'different_seeds',
                                  'different_source', 'trained_checkpoint', 'weak_primary', 'nonfinite'])
def test_invalid_calibration_evidence_is_rejected(defect):
    rows = [dict(video_id='a', source_group_id='g1')]
    values = copy.deepcopy(candidates({'encoder': 'initial'}, rows))
    c = values[1]['contract']
    if defect == 'incomplete':
        values[1]['complete'] = False
    elif defect == 'missing_candidate':
        values.pop()
    elif defect == 'duplicate_candidate':
        values.append(copy.deepcopy(values[0]))
    elif defect in ('validation', 'test'):
        c['panel'] = defect
    elif defect == 'wrong_rows':
        c['source_group_ids'] = ['other']
    elif defect == 'different_weights':
        c['weights_sha256']['encoder'] = 'changed'
    elif defect == 'different_runtime':
        c['runtime']['threads'] += 1
    elif defect == 'different_seeds':
        c['base_seed'] += 1
    elif defect == 'different_source':
        c['provenance']['changed'] = True
    elif defect == 'trained_checkpoint':
        c['checkpoint']['epoch'] = 1
    elif defect == 'weak_primary':
        c['settings']['steps'] = 5
    else:
        values[1]['records'][0]['adversarial_loss'] = float('nan')
    with pytest.raises(ValueError):
        calibration_record(values, rows, diagnostic=True)


def test_diagnostic_subset_cannot_be_promoted_to_ten_clip_calibration():
    rows = [dict(video_id='a', source_group_id='g1')]
    with pytest.raises(ValueError, match='ten'):
        calibration_record(candidates({'encoder': 'initial'}, rows), rows)


def test_real_calibration_reuses_complete_attacks_and_verifies_saved_evidence(model, clip, tmp_path):
    panel = Panel(clip)
    panel.rows = panel.rows[:1]
    checkpoint = dict(epoch=0, condition='pretrained', seed=None, training_contract_sha256=None, file_sha256='original')
    kwargs = dict(checkpoint=checkpoint, provenance={'fixture': True}, diagnostic=True)
    record = run_calibration(model, panel, tiny_masks, tmp_path, **kwargs)
    assert panel.loaded == [0, 0, 0]
    panel.loaded.clear()
    assert run_calibration(model, panel, tiny_masks, tmp_path, **kwargs) == record
    assert not panel.loaded
    identity = model_identity(model)
    assert load_calibration(tmp_path, panel.rows, original_identity=identity, diagnostic=True, model=model) == record
    evidence = tmp_path / record['evaluation_files'][0]['file']
    payload = read_record(evidence)
    payload['records'][0]['adversarial_loss'] += .1
    write_record(evidence, payload, replace=True)  # Even internally checksummed replacement must fail the pinned reference.
    with pytest.raises(ValueError, match='checksum'):
        load_calibration(tmp_path, panel.rows, original_identity=identity, diagnostic=True, model=model)
