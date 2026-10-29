"""Select and record primary PGD step size using only initialization/train clips."""

import copy
import math
from pathlib import Path

from models.candidate_evaluation import EvaluationSettings, SEED_SCHEMA
from models.candidate_evaluation_store import EvaluationStore, evaluate_panel, json_hash, validate_records
from models.checkpointing import file_sha256
from models.study_artifacts import read_record, write_record
from models.candidate_contract import BASE_SEED, PROFILE, declared_calibration_rows, verify_provenance, verify_runtime

CANDIDATES = (.5, 1., 2.)


def verify_rows(rows, diagnostic):
    if not diagnostic and [{'video_id': r['video_id'], 'source_group_id': r['source_group_id']}
                           for r in rows] != declared_calibration_rows():
        raise ValueError('Research calibration rows differ from the original first ten monitor rows.')


def calibration_record(evaluations, expected_rows, *, diagnostic=False):
    verify_rows(expected_rows, diagnostic)
    expected_ids = [r['video_id'] for r in expected_rows]
    if (len(set(expected_ids)) != len(expected_ids) or not expected_ids
            or (not diagnostic and len(expected_ids) != 10)):
        raise ValueError('Research calibration needs exactly the declared ten training-monitor clips.')
    common, results = None, {}
    for payload in evaluations:
        validate_records(payload)
        c = payload['contract']
        settings, checkpoint = c['settings'], c['checkpoint']
        size = settings['step_size_255']
        if (not payload['complete'] or c['panel'] != 'train_monitor'
                or c['video_ids'] != expected_ids
                or c['source_group_ids'] != [r['source_group_id'] for r in expected_rows]
                or settings['method'] != 'primary' or size not in CANDIDATES or size in results
                or checkpoint['epoch'] != 0 or checkpoint['condition'] != 'pretrained'
                or checkpoint['seed'] is not None or checkpoint['training_contract_sha256'] is not None
                or c['seed_schema'] != SEED_SCHEMA
                or c['sensitivity_banks'] != 0):
            raise ValueError('Calibration requires complete primary attacks on the declared training initialization.')
        comparison = copy.deepcopy(c)
        del comparison['settings']['step_size_255']
        if common is not None and common != comparison:
            raise ValueError('Calibration candidates changed weights, data, seeds, provenance, or runtime.')
        common = comparison
        # Per-candidate adversaries were selected on attack masks only. This
        # independent scoring mean chooses the hyperparameter on training data.
        results[size] = {'step_size_255': size,
                         'mean_independent_loss': math.fsum(r['adversarial_loss'] for r in payload['records']) / len(expected_ids),
                         'evaluation_sha256': json_hash(payload)}
    if set(results) != set(CANDIDATES):
        raise ValueError('Missing calibration candidates; no partial choice is permitted.')
    chosen = min(results.values(), key=lambda r: (-r['mean_independent_loss'], r['step_size_255']))
    return {'kind': 'candidate_eot8_training_step_calibration', 'schema_version': 2,
            'diagnostic_only': diagnostic, 'common_contract': common,
            'candidates': [results[s] for s in CANDIDATES], 'chosen_step_size_255': chosen['step_size_255'],
            'selection_rule': 'highest_mean_independent_loss_then_smaller_step_on_exact_tie',
            'validation_or_test_used': False, 'training_attack_step_size_255': 1.,
            'research_result': False, 'budget_approved': False}


def run_calibration(model, panel, masks_for, directory, *, checkpoint, provenance, diagnostic=False):
    # Running all thirty units before runtime/economic qualification would make
    # the resulting choice nonportable and spend the remaining pilot allowance.
    if not diagnostic:
        raise ValueError('Full research calibration is gated pending chosen-runtime and economic qualification.')
    verify_provenance(provenance)
    if panel.panel != 'train_monitor' or (not diagnostic and len(panel.rows) != 10):
        raise ValueError('Use the fixed ten training-monitor rows, or an explicitly bounded diagnostic.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payloads, refs = [], []
    for size in CANDIDATES:
        location = directory / ('step-' + str(size).replace('.', 'p'))
        # A directory without a committed journal is an incomplete initialization,
        # not permission to silently overwrite an earlier attempt.
        store = EvaluationStore(location, resume=location.exists())
        payload = evaluate_panel(model, panel, masks_for, EvaluationSettings.for_method('primary', size),
                                 store, checkpoint=checkpoint, provenance=provenance)
        payloads.append(payload)
        refs.append({'file': str((location / 'evaluation.json').relative_to(directory)),
                     'sha256': file_sha256(location / 'evaluation.json')})
    record = calibration_record(payloads, panel.rows, diagnostic=diagnostic)
    record['evaluation_files'] = refs
    output = directory / 'calibration.json'
    if output.exists():
        if read_record(output) != record:
            raise ValueError('Existing frozen calibration record differs; refusing replacement.')
    else:
        write_record(output, record)
    return record


def verify_calibration(record, *, original_identity, expected_rows, diagnostic=False):
    verify_rows(expected_rows, diagnostic)
    if (record.get('kind') != 'candidate_eot8_training_step_calibration' or record.get('schema_version') != 2
            or record.get('diagnostic_only') != diagnostic
            or record['chosen_step_size_255'] not in CANDIDATES
            or record['common_contract']['weights_sha256'] != original_identity
            or record['common_contract']['video_ids'] != [r['video_id'] for r in expected_rows]
            or record['common_contract']['source_group_ids'] != [r['source_group_id'] for r in expected_rows]
            or record['common_contract']['panel'] != 'train_monitor'
            or record['common_contract']['checkpoint']['epoch'] != 0
            or record['common_contract']['checkpoint']['condition'] != 'pretrained'
            or record['common_contract']['checkpoint']['seed'] is not None
            or record['common_contract']['checkpoint']['training_contract_sha256'] is not None
            or record['common_contract']['seed_schema'] != SEED_SCHEMA
            or record['selection_rule'] != 'highest_mean_independent_loss_then_smaller_step_on_exact_tie'
            or record['validation_or_test_used'] or record['training_attack_step_size_255'] != 1
            or (not diagnostic and len(expected_rows) != 10)):
        raise ValueError('Calibration is not valid for these initial weights, training rows, or study mode.')
    c = record['common_contract']
    verify_runtime(c['runtime'], c['provenance'])
    EvaluationSettings(**{**c['settings'], 'step_size_255': 1.})
    if (c['settings'].get('profile') != PROFILE or c['settings']['method'] != 'primary'
            or c.get('kind') != 'candidate_eot8_panel_evaluation' or c.get('schema_version') != 2
            or c['base_seed'] != BASE_SEED or c['sensitivity_banks'] != 0):
        raise ValueError('Calibration candidate profile or strength differs.')
    candidates = record['candidates']
    if ([r['step_size_255'] for r in candidates] != list(CANDIDATES)
            or any(not math.isfinite(r['mean_independent_loss']) for r in candidates)):
        raise ValueError('Invalid calibration candidate results.')
    best = min(candidates, key=lambda r: (-r['mean_independent_loss'], r['step_size_255']))
    if record['chosen_step_size_255'] != best['step_size_255']:
        raise ValueError('Calibration choice disagrees with its recorded rule.')


def load_calibration(directory, expected_rows, *, original_identity, model, diagnostic=False):
    directory = Path(directory)
    record = read_record(directory / 'calibration.json')
    payloads = []
    for ref in record['evaluation_files']:
        path = (directory / ref['file']).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError('Calibration evaluation path escapes its directory.')
        payloads.append(read_record(path, expected_sha256=ref['sha256']))
    rebuilt = calibration_record(payloads, expected_rows, diagnostic=diagnostic)
    if {k: v for k, v in record.items() if k != 'evaluation_files'} != rebuilt:
        raise ValueError('Calibration record disagrees with its evaluated candidates.')
    verify_calibration(record, original_identity=original_identity, expected_rows=expected_rows, diagnostic=diagnostic)
    from models.training import runtime_contract
    device = next(model.encoder.parameters()).device.type
    if record['common_contract']['runtime'] != runtime_contract(device, model=model):
        raise ValueError('Calibration execution runtime/backend differs.')
    return record
