"""Verified training-only input cache for the diagnostic runner."""

import csv
import hashlib
import json
from pathlib import Path

import torch


class PreparedTrainingClips:
    def __init__(self, directory, manifest, *, shape=(3, 16, 256, 256)):
        self.root = Path(directory).resolve()
        manifest = Path(manifest)
        if manifest.name != 'train.csv':
            raise ValueError('This runner accepts only a train.csv manifest.')
        self.manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        metadata = json.loads((manifest.parent / 'metadata.json').read_text())
        if metadata['files_sha256']['train.csv'] != self.manifest_sha256:
            raise ValueError('Training manifest checksum differs from metadata.')
        with manifest.open() as handle:
            rows = list(csv.DictReader(handle))
        allowed = {row['video_id']: row for row in rows}
        if len(allowed) != len(rows):
            raise ValueError('Duplicate video ID in training manifest.')
        raw = (self.root / 'index.json').read_bytes()
        index = json.loads(raw)
        self.index_sha256 = hashlib.sha256(raw).hexdigest()
        if index['split'] != 'train' or index['source_manifest_sha256'] != self.manifest_sha256:
            raise ValueError('Prepared clips must come from this exact training manifest.')
        self.entries = index['clips']
        ids = [entry['video_id'] for entry in self.entries]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Prepared training IDs must be nonempty and unique.')
        self.paths = []
        self.shape = tuple(shape)
        for entry in self.entries:
            row = allowed.get(entry['video_id'])
            if row is None or any(entry[k] != v for k, v in row.items()):
                raise ValueError('A prepared clip is not a member of the training manifest.')
            path = (self.root / entry['file']).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError('Missing or escaped prepared clip path.')
            self.paths.append(path)

    @property
    def video_ids(self):
        return [entry['video_id'] for entry in self.entries]

    def pixels(self, index):
        payload = torch.load(self.paths[index], weights_only=True, map_location='cpu')
        pixels, metadata = payload['pixels'], payload['metadata']
        entry = self.entries[index]
        if any(metadata[k] != v for k, v in entry.items() if k != 'file'):
            raise ValueError('Prepared file metadata differs from its index.')
        if (pixels.dtype != torch.float32 or tuple(pixels.shape) != self.shape
                or not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1):
            raise ValueError('Prepared pixels must have the expected FP32 raw-pixel shape/range.')
        if hashlib.sha256(pixels.contiguous().numpy().tobytes()).hexdigest() != entry['pixel_sha256']:
            raise ValueError('Prepared pixel checksum mismatch.')
        return pixels.unsqueeze(0)

    def verify_all(self):
        for index in range(len(self.entries)):
            self.pixels(index)

    def target_loader(self, model, device):
        # Cache only fixed clean target features, in CPU RAM. Never use an
        # unverified target cache from another model/preprocessing recipe.
        targets = [model.clean_targets(self.pixels(i).to(device)).cpu()
                   for i in range(len(self.entries))]

        def load(index):
            return self.pixels(index), targets[index]

        return load
