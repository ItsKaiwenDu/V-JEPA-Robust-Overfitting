"""Checksummed JSON records using the existing atomic file-write primitive."""

import json
from pathlib import Path

from models.checkpointing import CheckpointStore, file_sha256
from models.evaluation_store import json_hash


def write_record(path, payload, *, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
    wrapper = {'sha256': json_hash(payload), 'payload': payload}
    store = CheckpointStore(path.parent)
    store._atomic(path.name, wrapper, json_file=True)
    return {'path': str(path.resolve()), 'sha256': file_sha256(path)}


def read_record(path, *, expected_sha256=None):
    path = Path(path)
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError('Artifact file checksum mismatch.')
    wrapper = json.loads(path.read_text())
    if json_hash(wrapper['payload']) != wrapper['sha256']:
        raise ValueError('Artifact payload checksum mismatch.')
    return wrapper['payload']
