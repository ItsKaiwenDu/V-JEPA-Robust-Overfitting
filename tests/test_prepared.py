import csv
import hashlib
import json

import pytest
import torch

from datasets.prepared import PreparedTrainingClips


@pytest.fixture
def prepared(tmp_path):
    row = dict(video_id='example', relative_path='Class/example.avi', class_id='0', source_group_id='Class_g01')
    manifest = tmp_path / 'train.csv'
    with manifest.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / 'metadata.json').write_text(json.dumps({'files_sha256': {'train.csv': digest}}))
    pixels = torch.full((3, 4, 16, 16), .5)
    entry = {**row, 'pixel_sha256': hashlib.sha256(pixels.numpy().tobytes()).hexdigest()}
    torch.save({'pixels': pixels, 'metadata': entry}, tmp_path / 'example.pt')
    index = {'split': 'train', 'source_manifest_sha256': digest, 'clips': [{**entry, 'file': 'example.pt'}]}
    (tmp_path / 'index.json').write_text(json.dumps(index))
    return tmp_path, manifest, index


def test_prepared_inputs_verify_shape_identity_and_changed_content(prepared):
    path, manifest, index = prepared
    loader = PreparedTrainingClips(path, manifest, shape=(3, 4, 16, 16))
    loader.verify_all()
    assert loader.pixels(0).shape == (1, 3, 4, 16, 16)
    payload = torch.load(path / 'example.pt', weights_only=True)
    payload['pixels'][0, 0, 0, 0] += .1
    torch.save(payload, path / 'example.pt')
    with pytest.raises(ValueError, match='checksum'):
        loader.pixels(0)


@pytest.mark.parametrize('defect', ['test_split', 'wrong_member', 'wrong_manifest', 'duplicate', 'escaped_path'])
def test_data_provenance_failures_are_rejected(prepared, defect):
    path, manifest, index = prepared
    if defect == 'test_split':
        index['split'] = 'test'
    elif defect == 'wrong_member':
        index['clips'][0]['source_group_id'] = 'held-out-group'
    elif defect == 'wrong_manifest':
        index['source_manifest_sha256'] = '0' * 64
    elif defect == 'duplicate':
        index['clips'] *= 2
    else:
        index['clips'][0]['file'] = '../outside.pt'
    (path / 'index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError):
        PreparedTrainingClips(path, manifest)
