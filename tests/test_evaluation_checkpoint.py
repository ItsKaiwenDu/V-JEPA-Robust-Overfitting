import copy
import json

import pytest
import torch

from models.checkpointing import CheckpointStore, file_sha256
from models.evaluation import model_identity
from models.evaluation_checkpoint import load_anchor
from test_training import make_engine
from test_vjepa2 import model


def trained_anchor(model, directory):
    engine = make_engine(copy.deepcopy(model), provenance={'checkpoint_sha256': 'original'})
    for _ in range(engine.updates_per_epoch):
        engine.update()
    store = CheckpointStore(directory)
    store.save(engine.state(), anchor_epoch=1)
    return engine


def test_anchor_load_matches_trained_context_and_preserves_target(model, tmp_path):
    trained = trained_anchor(model, tmp_path)
    before = model_identity(model)
    reference = load_anchor(model, tmp_path, 1, initial_checkpoint_sha256='original')
    assert model_identity(model) == model_identity(trained.model)
    assert model_identity(model)['target_encoder'] == before['target_encoder']
    assert reference['epoch'] == 1 and reference['condition'] == 'clean-finetune'
    assert reference['training_contract_sha256'] == trained.contract_sha256


@pytest.mark.parametrize('defect', ['checksum', 'original', 'target', 'tensor', 'epoch'])
def test_bad_anchor_rejected_before_context_or_predictor_mutation(model, tmp_path, defect):
    trained_anchor(model, tmp_path)
    manifest_path = tmp_path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    ref = manifest['anchors']['1']
    path = tmp_path / ref['file']
    original = 'original'
    if defect == 'checksum':
        with path.open('ab') as handle:
            handle.write(b'corrupt')
    elif defect == 'original':
        original = 'different-original'
    elif defect == 'target':
        with torch.no_grad():
            next(model.target_encoder.parameters()).add_(.01)
    else:
        payload = torch.load(path, weights_only=True)
        if defect == 'tensor':
            next(iter(payload['predictor'].values())).flatten()[0] = float('nan')
        else:
            payload['step'] += 1
        torch.save(payload, path)
        ref['sha256'] = file_sha256(path)
        manifest_path.write_text(json.dumps(manifest))
    before = model_identity(model)
    with pytest.raises(ValueError):
        load_anchor(model, tmp_path, 1, initial_checkpoint_sha256=original)
    assert model_identity(model) == before
