"""Read trained anchor weights without changing the fixed target encoder."""

import hashlib
import json
import math

import torch

from models.checkpointing import CheckpointStore, file_sha256
from models.diagnostics import state_digest


def load_anchor(model, directory, epoch, *, initial_checkpoint_sha256):
    if type(epoch) is not int or epoch <= 0:
        raise ValueError('Use a positive trained anchor epoch.')
    store = CheckpointStore(directory, create=False)
    with store._lock(exclusive=False):
        manifest = store._manifest()
        reference = manifest['anchors'].get(str(epoch))
        if reference is None:
            raise ValueError('Requested anchor is not committed.')
        path = store._path(reference)
        if file_sha256(path) != reference['sha256']:
            raise ValueError('Anchor checksum mismatch.')
        payload = torch.load(path, map_location='cpu', weights_only=True)
    contract = payload['contract']
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if (contract_hash != payload['contract_sha256'] or contract_hash != manifest['contract_sha256']
            or contract['provenance']['checkpoint_sha256'] != initial_checkpoint_sha256
            or payload['target_sha256'] != contract['target_sha256']
            or payload['target_sha256'] != state_digest(model.target_encoder)):
        raise ValueError('Anchor contract, original weights, or fixed target differs.')
    updates = math.ceil(len(contract['ids']) / contract['settings']['effective_batch'])
    if payload['step'] != epoch * updates or epoch > contract['settings']['epochs']:
        raise ValueError('Anchor epoch disagrees with completed updates.')
    for name in ('encoder', 'predictor'):
        actual, expected = payload[name], getattr(model, name).state_dict()
        if (set(actual) != set(expected) or any(actual[k].shape != expected[k].shape
                or actual[k].dtype != expected[k].dtype or not torch.isfinite(actual[k]).all() for k in expected)):
            raise ValueError('Invalid anchor tensors; no weights loaded.')
    for name in ('encoder', 'predictor'):
        getattr(model, name).load_state_dict(payload[name], strict=True)
    return {'epoch': epoch, 'condition': contract['condition'], 'seed': contract['seed'],
            'training_contract_sha256': contract_hash, 'file_sha256': reference['sha256']}
