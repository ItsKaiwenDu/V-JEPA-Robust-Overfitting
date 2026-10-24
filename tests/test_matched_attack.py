"""Matched attack mechanics on actual small upstream CPU transformers."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from models.checkpointing import file_sha256
from models.evaluation import EvaluationSettings, model_identity
from models.execution import apply_model_policy, configure_execution
from models.fixture_state import save_fixture_state
from models.matched_attack import (ARMS, KIND, POLICY, _seeds, compare_replay, pairing_gate,
                                   run_attacks, tensor_hash, verify_worker)
from models.matched_diagnostic import bounded_child, monitor_input, session
from models.native_diagnostic import tiny_fixture
from models.pilot_measurement import compare_trees
from models.study_artifacts import read_record, write_record
from models.training import rng_state

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def checked(tmp_path_factory, upstream):
    configure_execution(POLICY, device='cpu')
    torch.set_num_threads(2)
    torch.manual_seed(617)
    model, pixels, sampler = tiny_fixture(upstream)
    output = tmp_path_factory.mktemp('matched')
    initial = save_fixture_state(model, output / 'initial.pt')
    apply_model_policy(model, POLICY)
    masks = lambda seed: sampler(seed % (2**31 - 1))
    before, rng = model_identity(model), rng_state('cpu')
    modes = [m.training for m in model.modules()]
    run_attacks(model, pixels, masks, video_id='generated-training-only', provenance={'fixture': initial},
                output=output / 'matrix')
    assert model_identity(model) == before and [m.training for m in model.modules()] == modes
    assert compare_trees(rng, rng_state('cpu'))['exact_match']
    contract, records = verify_worker(output / 'matrix')
    return SimpleNamespace(root=output, initial=initial, model=model, pixels=pixels, masks=masks,
                           contract=contract, records=records)


def test_all_four_arms_share_starts_nested_masks_and_exact_prefixes(checked):
    contract, records = checked.contract, checked.records
    assert contract['kind'] == KIND and not contract['canonical_evaluation_artifact']
    assert pairing_gate(records, contract)['starts_and_first_20_steps_exact']
    assert [r['settings'] for r in records.values()] == [dict(steps=s, attack_masks=m) for s, m in ARMS.values()]
    actual = [s % (2**31 - 1) for n, group in contract['seeds'].items() if n != 'start' for s in group]
    assert len(actual) == len(set(actual)) == 20
    assert records['B']['attack_loss'] >= records['A']['attack_loss']
    assert records['D']['attack_loss'] >= records['C']['attack_loss']
    for row in records.values():
        assert row['max_pixel_delta'] <= 4 / 255 + 1e-7
        assert row['attack_recomputed_exact']
        assert all(g['gradient_abs_max'] > 0 for g in row['trace'] if g['with_grad'])
        assert all(g['gradient_finite'] for g in row['trace'] if g['with_grad'])


def test_saved_adversary_scores_recompute_from_clean_targets(checked):
    from attacks.pgd import native_objective
    model = checked.model
    model.eval()
    inputs = torch.load(checked.root / 'matrix/inputs.pt', weights_only=True)
    assert tensor_hash(model.clean_targets(inputs['clean'])) == tensor_hash(inputs['targets'])
    banks = {name: [[p.to('cpu') for p in checked.masks(seed)] for seed in checked.contract['seeds'][name]]
             for name in ('main', 'audit_0', 'audit_1')}
    for arm, record in checked.records.items():
        adversarial = torch.load(checked.root / ('matrix/' + arm + '.pt'), weights_only=True)['adversarial']
        with torch.no_grad():
            for name, bank in banks.items():
                objective = native_objective(model, inputs['targets'], bank)
                assert objective(adversarial).item() == record['scores'][name]['adversarial']
                assert objective(inputs['clean']).item() == record['scores'][name]['clean']
    model.train()


def test_independent_scoring_never_selects_an_adversary(checked, tmp_path):
    seeds = _seeds('generated-training-only')
    score_seeds = set(seeds['main'] + seeds['audit_0'] + seeds['audit_1'])
    def different(seed):
        return checked.masks(seed + 73 if seed in score_seeds else seed)
    run_attacks(checked.model, checked.pixels, different, video_id='generated-training-only',
                provenance={'fixture': checked.initial}, output=tmp_path / 'changed', arms=('A',))
    _, rows = verify_worker(tmp_path / 'changed', expected_arms=('A',))
    assert checked.records['A']['adversarial_sha256'] == rows['A']['adversarial_sha256']
    assert checked.records['A']['trace'] == rows['A']['trace']
    assert checked.records['A']['scores'] != rows['A']['scores']


def test_fresh_cpu_process_replay_loads_pinned_parent_weights(checked, tmp_path):
    output = tmp_path / 'fresh'
    subprocess.run([sys.executable, str(ROOT / 'tests/fixtures/matched_attack_worker.py'), str(ROOT),
                    str(output), 'replay', str(checked.root / 'initial.pt'), checked.initial['sha256']],
                   check=True, timeout=180, capture_output=True)
    result = compare_replay(checked.root / 'matrix', output)
    assert result['passed'] and not result['all_four_arms_fresh_replayed']


def test_bad_fixture_hash_cannot_fall_back_to_random_initialization(checked, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'tests/fixtures/matched_attack_worker.py'), str(ROOT),
                             str(tmp_path / 'bad'), 'replay', str(checked.root / 'initial.pt'), '0' * 64],
                            timeout=60, capture_output=True)
    assert result.returncode != 0 and b'checkpoint SHA-256 mismatch' in result.stderr
    assert not (tmp_path / 'bad/worker.json').exists()


@pytest.mark.parametrize('change', ['prefix', 'missing', 'contract', 'settings', 'loss', 'gradient', 'bound', 'scores'])
def test_pairing_rejects_incomplete_or_mixed_evidence(checked, change):
    rows = copy.deepcopy(checked.records)
    if change == 'prefix': rows['B']['trace'][1]['gradient_sha256'] = 'a' * 64
    if change == 'missing': del rows['D']
    if change == 'contract': rows['C']['contract_sha256'] = 'b' * 64
    if change == 'settings': rows['C']['settings']['attack_masks'] = 4
    if change == 'loss': rows['B']['attack_loss'] -= 1
    if change == 'gradient': rows['A']['trace'][1]['gradient_finite'] = False
    if change == 'bound': rows['D']['max_pixel_delta'] = float('nan')
    if change == 'scores': rows['A']['scores'].pop('audit_1')
    with pytest.raises(ValueError):
        pairing_gate(rows, checked.contract)


@pytest.mark.parametrize('name', ['A.pt', 'inputs.pt', 'A.json', 'contract.json', 'worker.json'])
def test_tampered_artifacts_are_rejected(checked, tmp_path, name):
    destination = tmp_path / 'copy'
    shutil.copytree(checked.root / 'matrix', destination)
    path = destination / name
    path.write_bytes(path.read_bytes() + b'broken')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        verify_worker(destination)


def test_replay_requires_exact_contract_not_just_same_loss(checked, tmp_path):
    copy_path = tmp_path / 'copy'
    shutil.copytree(checked.root / 'matrix', copy_path)
    worker = read_record(copy_path / 'worker.json')
    worker['arms'] = ['A']
    worker['pairing'] = None
    worker['arm_records'] = {'A': worker['arm_records']['A']}
    write_record(copy_path / 'worker.json', worker, replace=True)
    # This is a logical same-process copy, only used to exercise rejection.
    assert compare_replay(checked.root / 'matrix', copy_path)['passed']
    contract = read_record(copy_path / 'contract.json')
    contract['provenance']['different_source'] = 'changed'
    write_record(copy_path / 'contract.json', contract, replace=True)
    with pytest.raises(ValueError):
        compare_replay(checked.root / 'matrix', copy_path)


def test_upstream_seed_collision_rejected_before_attack(checked, tmp_path, monkeypatch):
    monkeypatch.setattr('models.matched_attack.evaluation_seed', lambda *_a, **_k: 7)
    with pytest.raises(ValueError, match='collision'):
        run_attacks(checked.model, checked.pixels, checked.masks, video_id='collision', provenance={}, output=tmp_path / 'bad')
    assert not (tmp_path / 'bad/worker.json').exists()


def test_failure_restores_rng_modes_weights_and_parameter_grads(checked, tmp_path):
    model = checked.model
    model.train()
    before, rng = model_identity(model), rng_state('cpu')
    modes = [m.training for m in model.modules()]
    def failure(_seed):
        torch.rand(8)
        raise RuntimeError('deliberate interruption')
    with pytest.raises(RuntimeError, match='deliberate interruption'):
        run_attacks(model, checked.pixels, failure, video_id='failure', provenance={}, output=tmp_path / 'failed')
    assert before == model_identity(model) and compare_trees(rng, rng_state('cpu'))['exact_match']
    assert modes == [m.training for m in model.modules()] and all(p.grad is None for p in model.parameters())
    assert not (tmp_path / 'failed/worker.json').exists()


def test_canonical_methods_still_reject_noncanonical_arms():
    with pytest.raises(ValueError): EvaluationSettings('primary', 20, 2, 8, 4)
    with pytest.raises(ValueError): EvaluationSettings('convergence', 50, 2, 4, 4)


def test_worker_timeout_kills_process_and_cannot_pass(tmp_path):
    task = bounded_child([sys.executable, '-c', 'import time; time.sleep(60)'], tmp_path / 'timeout.log', .1)
    assert task == {'exit_status': None, 'timed_out': True, 'complete': False}


def test_failed_session_exports_partial_evidence_without_passing(tmp_path, monkeypatch):
    monkeypatch.setattr('models.matched_diagnostic.bounded_child', lambda *_a: {'exit_status': 1, 'complete': False, 'timed_out': False})
    output = tmp_path / 'session'
    args = SimpleNamespace(output=output, max_seconds=120, checkpoint=Path('absent.pt'),
                           upstream=ROOT / 'external/vjepa2', manifests=Path('none'), clips=Path('none'))
    result = session(ROOT, args)
    assert not result['passed'] and not result['complete'] and set(result['tasks']) == {'matrix'}
    assert (output / 'evidence.tar.gz').is_file()
    assert not (output / 'replay').exists()


def test_training_monitor_loader_opens_one_pinned_training_tensor_only(tmp_path, monkeypatch):
    import hashlib
    from datasets.staged_preparation import RECIPE
    directory = tmp_path / 'prepared'
    (directory / 'train').mkdir(parents=True)
    first = {'video_id': 'first', 'class_id': '0', 'source_group_id': 'g1', 'relative_path': 'class/first.avi'}
    second = {**first, 'video_id': 'second', 'relative_path': 'class/second.avi', 'source_group_id': 'g2'}
    clean = torch.full((3, 16, 256, 256), .5)
    entry = {**first, 'file': 'first.pt', 'recipe': RECIPE,
             'pixel_sha256': hashlib.sha256(clean.numpy().tobytes()).hexdigest()}
    index = {'split': 'train', 'source_manifest_sha256': 'metadata-only',
             'clips': [entry, {**entry, **second, 'file': 'second.pt'}]}
    index_path = directory / 'train/index.json'
    index_path.write_text(json.dumps(index))
    (directory / 'preparation.json').write_text(json.dumps({'kind': 'exploration_preparation',
        'manifest_sha256': 'reviewed', 'clip_indices_sha256': {'train': file_sha256(index_path)}}))
    torch.save({'pixels': clean, 'metadata': {k: v for k, v in entry.items() if k != 'file'}}, directory / 'train/first.pt')
    monkeypatch.setattr('models.matched_diagnostic.verify_manifests', lambda *_a:
                        ({'train': [first, second], 'train_monitor': [first]},
                         {'manifest_sha256': 'reviewed', 'files_sha256': {'train.csv': 'metadata-only'}}))
    pixels, data = monitor_input(ROOT, tmp_path / 'manifests', directory)
    assert tuple(pixels.shape) == (1, 3, 16, 256, 256) and data['row'] == first
    # No second training file exists; no held-out pixel file was provided.
    assert not (directory / 'train/second.pt').exists()
    torch.save({'pixels': clean + .01, 'metadata': {k: v for k, v in entry.items() if k != 'file'}}, directory / 'train/first.pt')
    with pytest.raises(ValueError, match='checksum'):
        monitor_input(ROOT, tmp_path / 'manifests', directory)


def test_production_cli_has_no_tiny_or_random_weight_fallback(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/check_matched_attack.py'), '--diagnostic',
                             '--checkpoint', str(tmp_path / 'missing.pt'), '--output', str(tmp_path / 'new')],
                            capture_output=True, timeout=30)
    assert result.returncode == 2 and b'existing released checkpoint' in result.stderr
    assert not (tmp_path / 'new').exists()
