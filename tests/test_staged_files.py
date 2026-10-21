import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from datasets.staged_files import expected_export, export_manifests, sha256, verify_manifests
from datasets.staged_preparation import PreparedStagedPanel, duplicate_hashes
from datasets.video import RECIPE

ROOT = Path(__file__).resolve().parents[1]


def test_export_matches_proposal_and_preserves_parent_identities(tmp_path):
    output = tmp_path / 'manifests'
    metadata = export_manifests(ROOT, output)
    rows, checked = verify_manifests(ROOT, output)
    assert metadata == checked
    assert metadata['counts'] == dict(train=100, train_monitor=20, exploration_validation=20,
                                     confirmation_validation=20, test=60)
    assert len({r['source_group_id'] for r in rows['test']}) == 59
    assert {r['video_id'] for r in rows['train_monitor']} <= {r['video_id'] for r in rows['train']}
    assert not metadata['confirmation_and_test_pixels_opened'] and not metadata['protocol_frozen']
    with pytest.raises(FileExistsError):
        export_manifests(ROOT, output)


@pytest.mark.parametrize('filename', ['train.csv', 'exploration_validation.csv', 'test.csv', 'metadata.json'])
def test_self_consistent_export_tampering_cannot_replace_proposed_rows(tmp_path, filename):
    output = tmp_path / 'manifests'
    metadata = export_manifests(ROOT, output)
    if filename == 'metadata.json':
        metadata['counts']['train'] = 99
    else:
        path = output / filename
        path.write_bytes(path.read_bytes() + b'\n')
        metadata['files_sha256'][filename] = sha256(path)
    (output / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        verify_manifests(ROOT, output)


@pytest.mark.parametrize('panel', ['confirmation_validation', 'test'])
def test_unopened_panel_rejected_before_reading_any_files(tmp_path, panel):
    with pytest.raises(ValueError, match='remain closed'):
        PreparedStagedPanel(tmp_path / 'missing', tmp_path / 'missing', tmp_path / 'missing', panel)


@pytest.fixture
def staged_pixels(tmp_path, monkeypatch):
    rows = {'train': [dict(video_id='v_One', relative_path='Class/v_One.avi', class_id='4', source_group_id='Class_g01'),
                      dict(video_id='v_Two', relative_path='Class/v_Two.avi', class_id='4', source_group_id='Class_g02')]}
    rows['train_monitor'] = [rows['train'][1]]
    metadata = {'manifest_sha256': 'all-manifests', 'files_sha256': {'train.csv': 'train-hash',
                                                                   'train_monitor.csv': 'monitor-hash'}}
    monkeypatch.setattr('datasets.staged_preparation.verify_manifests', lambda *a: (copy.deepcopy(rows), metadata))
    cache = tmp_path / 'train'
    cache.mkdir()
    entries = []
    for i, row in enumerate(rows['train']):
        pixels = torch.full((3, 4, 16, 16), .3 + .2 * i)
        entry = {**row, 'recipe': RECIPE, 'pixel_sha256': hashlib.sha256(pixels.numpy().tobytes()).hexdigest()}
        torch.save({'pixels': pixels, 'metadata': entry}, cache / (row['video_id'] + '.pt'))
        entries.append({'file': row['video_id'] + '.pt', **entry})
    index = dict(split='train', source_manifest_sha256='train-hash', clips=entries)
    (cache / 'index.json').write_text(json.dumps(index))
    preparation = dict(kind='exploration_preparation', manifest_sha256='all-manifests',
                       clip_indices_sha256={'train': sha256(cache / 'index.json')})
    (tmp_path / 'preparation.json').write_text(json.dumps(preparation))
    return tmp_path, rows, index, preparation


def test_monitor_reuses_the_exact_verified_training_cache(staged_pixels):
    path, rows, _, _ = staged_pixels
    panel = PreparedStagedPanel(ROOT, path, path, 'train_monitor', shape=(3, 4, 16, 16))
    panel.verify_all()
    assert panel.rows == [rows['train'][1]]
    assert torch.equal(panel.pixels(0), torch.full((1, 3, 4, 16, 16), .5))
    payload = torch.load(path / 'train/v_Two.pt', weights_only=True)
    payload['pixels'][0, 0, 0, 0] = .9
    torch.save(payload, path / 'train/v_Two.pt')
    with pytest.raises(ValueError, match='checksum'):
        panel.pixels(0)


@pytest.mark.parametrize('defect', ['index_hash', 'missing_row', 'order', 'recipe', 'escape', 'incomplete'])
def test_invalid_preparation_cannot_be_used(staged_pixels, defect):
    path, _, index, preparation = staged_pixels
    if defect == 'incomplete':
        (path / 'preparation.json').unlink()
    else:
        if defect == 'missing_row':
            index['clips'].pop()
        elif defect == 'order':
            index['clips'].reverse()
        elif defect == 'recipe':
            index['clips'][0]['recipe'] = 'different'
        elif defect == 'escape':
            index['clips'][0]['file'] = '../outside.pt'
        else:
            index['ignored'] = 'index modified'
        (path / 'train/index.json').write_text(json.dumps(index))
        if defect != 'index_hash':
            preparation['clip_indices_sha256']['train'] = sha256(path / 'train/index.json')
        (path / 'preparation.json').write_text(json.dumps(preparation))
    with pytest.raises((ValueError, FileNotFoundError)):
        PreparedStagedPanel(ROOT, path, path, 'train', shape=(3, 4, 16, 16))


def test_exact_duplicate_audit_retains_group_and_panel_information():
    rows = [dict(video_id='a', source_group_id='g1', panel='train', pixel_sha256='same'),
            dict(video_id='b', source_group_id='g2', panel='exploration_validation', pixel_sha256='same'),
            dict(video_id='c', source_group_id='g3', panel='train', pixel_sha256='different')]
    matches = duplicate_hashes(rows, 'pixel_sha256')
    assert len(matches) == 1
    assert {row['panel'] for row in matches[0]} == {'train', 'exploration_validation'}
