"""Atomic, checksummed checkpoints with two resume generations and anchor weights."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import torch


def file_sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


class CheckpointStore:
    """The manifest rename commits a transaction; old files survive failed writes.

    No automatic fallback from a damaged latest generation. Orphans left by a
    failed transaction are not treated as committed states or silently deleted.
    """

    def __init__(self, directory, *, create=True):
        self.root = Path(directory)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not (self.root / 'manifest.json').is_file():
            raise ValueError('Resume directory has no committed checkpoint manifest.')

    @contextmanager
    def _lock(self, exclusive):
        # NFS maps flock to record locks: shared reads require a readable fd.
        # Read/write append preserves the lock inode and existing contents.
        with (self.root / '.writer.lock').open('a+') as handle:
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(handle, mode | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _sync_directory(self):
        descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _atomic(self, filename, value, *, json_file=False):
        path = self.root / filename
        temporary = self.root / ('.' + filename + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('xb') as handle:
                if json_file:
                    handle.write((json.dumps(value, sort_keys=True, indent=2) + '\n').encode())
                else:
                    torch.save(value, handle)
                handle.flush()
                os.fsync(handle.fileno())
            checksum = file_sha256(temporary)
            os.replace(temporary, path)
            self._sync_directory()
            return {'file': filename, 'sha256': checksum}
        finally:
            temporary.unlink(missing_ok=True)

    def _manifest(self):
        path = self.root / 'manifest.json'
        if not path.exists():
            return None
        manifest = json.loads(path.read_text())
        if manifest.get('schema_version') != 1:
            raise ValueError('Unsupported checkpoint manifest.')
        return manifest

    def _path(self, ref):
        name = ref['file']
        path = (self.root / name).resolve()
        if Path(name).name != name or not path.is_relative_to(self.root.resolve()):
            raise ValueError('Invalid checkpoint reference path.')
        return path

    def save(self, payload, *, anchor_epoch=None):
        with self._lock(exclusive=True):
            old = self._manifest()
            if old is not None and (old['contract_sha256'] != payload['contract_sha256']
                                    or old['step'] >= payload['step']):
                raise ValueError('Refusing a changed contract or non-increasing checkpoint step.')
            anchors = dict(old['anchors']) if old else {}
            if anchor_epoch is not None and str(anchor_epoch) in anchors:
                raise ValueError('Anchor already exists; use a new run attempt.')
            anchor = {key: payload[key] for key in ('encoder', 'predictor', 'contract',
                      'contract_sha256', 'target_sha256', 'step')} if anchor_epoch is not None else None
            needed = tensor_bytes(payload) + tensor_bytes(anchor)
            if shutil.disk_usage(self.root).free < needed * 1.1 + 256 * 2**20:
                raise OSError('Insufficient free space for a checkpoint transaction and reserve.')
            generation = uuid.uuid4().hex
            latest = self._atomic(f'resume-{payload["step"]:08d}-{generation}.pt', payload)
            if anchor is not None:
                anchors[str(anchor_epoch)] = self._atomic(f'epoch-{anchor_epoch:03d}-{generation}.pt', anchor)
            manifest = {'schema_version': 1, 'step': payload['step'],
                        'contract_sha256': payload['contract_sha256'],
                        'latest': latest, 'previous': old['latest'] if old else None,
                        'anchors': anchors}
            self._atomic('manifest.json', manifest, json_file=True)
            # Retire only the oldest resume file named by the former manifest,
            # after the new transaction commits. Never prune anchors here.
            if old and old['previous']:
                self._path(old['previous']).unlink(missing_ok=True)
            return latest

    def load(self, generation='latest'):
        if generation not in ('latest', 'previous'):
            raise ValueError('Choose latest or previous explicitly.')
        with self._lock(exclusive=False):
            manifest = self._manifest()
            if manifest is None or not manifest[generation]:
                raise ValueError('Requested committed checkpoint is unavailable.')
            ref = manifest[generation]
            path = self._path(ref)
            if file_sha256(path) != ref['sha256']:
                raise ValueError('Checkpoint checksum mismatch; no automatic fallback performed.')
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload['contract_sha256'] != manifest['contract_sha256']:
                raise ValueError('Checkpoint contract differs from manifest.')
            if generation == 'latest' and payload['step'] != manifest['step']:
                raise ValueError('Checkpoint step differs from manifest.')
            return payload, dict(ref)
