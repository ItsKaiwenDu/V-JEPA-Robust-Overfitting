"""Atomic per-clip evaluation progress and strict primary-validation selection."""

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from models.evaluation import EvaluationSettings, SEED_SCHEMA, evaluate_clip, model_identity
from models.training import runtime_contract


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


class EvaluationStore:
    """Resume only the identical checkpoint/panel/settings/source contract."""

    def __init__(self, directory, *, resume=False):
        self.root = Path(directory)
        if resume:
            if not (self.root / 'evaluation.json').is_file():
                raise ValueError('No committed evaluation to resume.')
        else:
            self.root.mkdir(parents=True, exist_ok=False)

    @contextmanager
    def lock(self):
        with (self.root / '.writer.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def read(self, contract):
        path = self.root / 'evaluation.json'
        if not path.exists():
            return {'contract': contract, 'records': [], 'complete': False}
        wrapper = json.loads(path.read_text())
        payload = wrapper['payload']
        if json_hash(payload) != wrapper['sha256']:
            raise ValueError('Evaluation checksum mismatch.')
        if payload['contract'] != contract:
            raise ValueError('Evaluation resume contract changed.')
        validate_records(payload)
        return payload

    def write(self, payload):
        validate_records(payload)
        wrapper = {'sha256': json_hash(payload), 'payload': payload}
        path = self.root / ('.evaluation-' + uuid.uuid4().hex + '.tmp')
        try:
            with path.open('x') as handle:
                json.dump(wrapper, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(path, self.root / 'evaluation.json')
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            path.unlink(missing_ok=True)


def validate_records(payload):
    contract, records = payload['contract'], payload['records']
    EvaluationSettings(**contract['settings'])
    ids = contract['video_ids']
    if (not ids or len(set(ids)) != len(ids) or len(records) > len(ids)
            or [r['video_id'] for r in records] != ids[:len(records)]
            or (payload['complete'] and len(records) != len(ids))):
        raise ValueError('Evaluation records are incomplete, duplicated, or reordered.')
    for record in records:
        if (record['settings'] != contract['settings'] or record['method'] != contract['settings']['method']
                or any(not math.isfinite(record[k]) for k in
                       ('clean_loss', 'adversarial_loss', 'attack_clean_loss', 'attack_loss', 'max_pixel_delta', 'seconds'))
                or not 0 <= record['max_pixel_delta'] <= contract['settings']['epsilon_255'] / 255 + 1e-7):
            raise ValueError('Invalid evaluation record.')


def evaluate_panel(model, panel, masks_for, settings, store, *, checkpoint, provenance,
                   base_seed=20261020, sensitivity_banks=0, on_clip=None):
    """Preserve progress after each completed clip; unopened panels are refused."""
    if panel.panel not in ('train', 'train_monitor', 'exploration_validation'):
        raise ValueError('Confirmation and test evaluation is not open at this stage.')
    device = next(model.encoder.parameters()).device.type
    identity = model_identity(model)
    contract = {'schema_version': 1, 'kind': 'native_panel_evaluation',
                'checkpoint': checkpoint, 'weights_sha256': identity,
                'panel': panel.panel, 'panel_identity': panel.identity,
                'video_ids': [r['video_id'] for r in panel.rows],
                'source_group_ids': [r['source_group_id'] for r in panel.rows],
                'settings': asdict(settings), 'seed_schema': SEED_SCHEMA,
                'base_seed': base_seed, 'sensitivity_banks': sensitivity_banks,
                'provenance': provenance, 'runtime': runtime_contract(device)}
    # Normalize tuples etc. to the JSON representation used on disk.
    contract = json.loads(json.dumps(contract, allow_nan=False))
    with store.lock():
        payload = store.read(contract)
        if not (store.root / 'evaluation.json').exists():
            store.write(payload)
        for index in range(len(payload['records']), len(panel.rows)):
            record, _ = evaluate_clip(model, panel.pixels(index).to(device), masks_for, settings,
                                       video_id=contract['video_ids'][index], base_seed=base_seed,
                                       sensitivity_banks=sensitivity_banks)
            payload['records'].append(record)
            store.write(payload)
            if on_clip is not None:
                on_clip(index, record)
        if model_identity(model) != identity:
            raise RuntimeError('Model or fixed target changed during evaluation.')
        payload['complete'] = True
        store.write(payload)
        return payload


def select_primary_checkpoint(evaluations, *, phase, condition, seed, epochs,
                              training_contract_sha256):
    """Select only a complete anchor sequence, with initialization and earliest ties.

    This function computes a provisional choice; it does not admit a research
    run, open test pixels, or infer that overfitting has occurred.
    """
    if phase not in ('exploration', 'confirmation') or condition not in ('clean-finetune', 'adversarial-finetune'):
        raise ValueError('Unknown study phase or training condition.')
    epochs = list(epochs)
    if epochs != sorted(set(epochs)) or not epochs or epochs[0] != 0:
        raise ValueError('Selection schedule must include initialization and sorted unique anchors.')
    candidates, common = {}, None
    for payload in evaluations:
        validate_records(payload)
        contract = payload['contract']
        checkpoint = contract['checkpoint']
        epoch = checkpoint['epoch']
        if (not payload['complete'] or contract['panel'] != phase + '_validation'
                or contract['settings']['method'] != 'primary' or epoch not in epochs
                or epoch in candidates):
            raise ValueError('Selection requires one complete primary validation panel at every declared anchor.')
        if epoch == 0:
            if checkpoint['condition'] != 'pretrained' or checkpoint['seed'] is not None:
                raise ValueError('Initialization must be the shared pretrained reference.')
        elif (checkpoint['condition'] != condition or checkpoint['seed'] != seed
                or checkpoint['training_contract_sha256'] != training_contract_sha256):
            raise ValueError('Mixed training runs cannot select a checkpoint.')
        comparison = {k: v for k, v in contract.items() if k not in ('checkpoint', 'weights_sha256')}
        if common is not None and common != comparison:
            raise ValueError('Evaluation panels, settings, sources, or runtime changed across anchors.')
        common = comparison
        # Initialization and every trained checkpoint must retain the same target.
        target = contract['weights_sha256']['target_encoder']
        if candidates and target != next(iter(candidates.values()))['target_sha256']:
            raise ValueError('Target identity changed across anchors.')
        candidates[epoch] = {'epoch': epoch, 'checkpoint': checkpoint,
                             'mean_adversarial_loss': math.fsum(r['adversarial_loss'] for r in payload['records'])
                                                       / len(payload['records']),
                             'target_sha256': target, 'evaluation_sha256': json_hash(payload)}
    if set(candidates) != set(epochs):
        raise ValueError('Missing primary validation anchors; refusing a partial best checkpoint.')
    best = min(candidates.values(), key=lambda row: (row['mean_adversarial_loss'], row['epoch']))
    return {'kind': 'provisional_primary_validation_selection', 'phase': phase, 'condition': condition,
            'seed': seed, 'selected': best, 'initial': candidates[0], 'final': candidates[epochs[-1]],
            'final_minus_selected_loss': candidates[epochs[-1]]['mean_adversarial_loss'] - best['mean_adversarial_loss'],
            'candidates': [candidates[e] for e in epochs], 'test_data_used': False,
            'research_result': False, 'budget_approved': False}
