"""Exploration-only deterministic preparation and verified cached panel access."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from datasets.staged_files import OPEN_PANELS, sha256, verify_manifests
from datasets.video import RECIPE, decode_clip


def duplicate_hashes(records, field):
    groups = defaultdict(list)
    for entry in records:
        groups[entry[field]].append({'video_id': entry['video_id'],
                                    'source_group_id': entry['source_group_id'], 'panel': entry['panel']})
    return [values for values in groups.values() if len(values) > 1]


def prepare_exploration(root, manifests, archive, output, *, decoder=decode_clip):
    """Extract only train/exploration members. Decode twice, fail on any mismatch.

    Completion is marked by preparation.json, written last. Partial output is
    preserved on failure and must never be used as a completed preparation.
    """
    started = time.monotonic()
    rows, metadata = verify_manifests(root, manifests)
    provenance = json.loads((Path(root) / 'report/checks/2026-10-17-data-preparation.json').read_text())
    expected = provenance['download_provenance']['ucf101_archive']
    archive = Path(archive).resolve(strict=True)
    if archive.stat().st_size != expected['bytes'] or sha256(archive) != expected['sha256']:
        raise ValueError('Dataset archive checksum differs from the original preparation record.')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_root = output / 'sources'
    source_root.mkdir()
    panels = ('train', 'exploration_validation')
    members = ['UCF-101/' + row['relative_path'] for panel in panels for row in rows[panel]]
    listing = output / 'extracted-members.txt'
    listing.write_text('\n'.join(members) + '\n')
    subprocess.run(['unrar', 'x', '-idq', '-o-', str(archive), '@' + str(listing), str(source_root) + '/'],
                   check=True, capture_output=True, text=True, timeout=600)
    records, indices, index_hashes = [], {}, {}
    for panel in panels:
        directory = output / panel
        directory.mkdir()
        entries = []
        for row in rows[panel]:
            path = (source_root / 'UCF-101' / row['relative_path']).resolve(strict=True)
            if not path.is_relative_to(source_root):
                raise ValueError('Extracted source escapes preparation directory.')
            pixels, record = decoder(path)
            repeated, repeated_record = decoder(path)
            if not torch.equal(pixels, repeated) or record != repeated_record:
                raise ValueError(f"Decode replay differs: {row['video_id']}")
            if (pixels.dtype != torch.float32 or pixels.shape != (3, 16, 256, 256)
                    or not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1
                    or record['recipe'] != RECIPE):
                raise ValueError('Decoded clip has invalid shape, range, dtype, or recipe.')
            entry = {**record, **row}
            filename = row['video_id'] + '.pt'
            torch.save({'pixels': pixels, 'metadata': entry}, directory / filename)
            entries.append({'file': filename, **entry})
            records.append({**entry, 'panel': panel})
            print(f"Verified two decodes: {panel}/{row['video_id']}", flush=True)
        index = {'split': panel, 'source_manifest_sha256': metadata['files_sha256'][panel + '.csv'],
                 'selection': 'all staged manifest rows in manifest order', 'clips': entries}
        (directory / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
        indices[panel] = index
        index_hashes[panel] = sha256(directory / 'index.json')
    duplicates = {key: duplicate_hashes(records, key) for key in ('source_sha256', 'pixel_sha256')}
    if any(duplicates.values()):
        (output / 'duplicate-failure.json').write_text(json.dumps(duplicates, indent=2) + '\n')
        raise ValueError('Duplicate source bytes or prepared pixels; review before admitting this subset.')
    record = {'schema_version': 1, 'kind': 'exploration_preparation',
              'manifest_sha256': metadata['manifest_sha256'], 'archive_sha256': expected['sha256'],
              'clip_indices_sha256': index_hashes, 'counts': {p: len(indices[p]['clips']) for p in panels},
              'identical_redecodes': len(records), 'exact_duplicates': duplicates,
              'recipe': RECIPE, 'wall_seconds': time.monotonic() - started,
              'source_and_pixel_identity_checked': True, 'near_duplicate_content_checked': False,
              'confirmation_and_test_pixels_opened': False, 'protocol_frozen': False,
              'budget_approved': False, 'research_result': False}
    (output / 'preparation.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


class PreparedStagedPanel:
    """Only fully prepared training/exploration panels; unopened panels fail early."""

    def __init__(self, root, manifests, prepared, panel, *, shape=(3, 16, 256, 256)):
        if panel not in OPEN_PANELS:
            raise ValueError('Confirmation/test pixels remain closed at the exploration preparation stage.')
        rows, metadata = verify_manifests(root, manifests)
        self.panel = panel
        self.rows = rows[panel]
        prepared = Path(prepared).resolve()
        preparation = json.loads((prepared / 'preparation.json').read_text())
        if (preparation.get('kind') != 'exploration_preparation'
                or preparation['manifest_sha256'] != metadata['manifest_sha256']):
            raise ValueError('Preparation and staged manifest identities disagree.')
        cache_panel = 'train' if panel == 'train_monitor' else panel
        self.directory = prepared / cache_panel
        index_path = self.directory / 'index.json'
        if sha256(index_path) != preparation['clip_indices_sha256'][cache_panel]:
            raise ValueError('Prepared index checksum mismatch.')
        index = json.loads(index_path.read_text())
        entries = index['clips']
        if (index['split'] != cache_panel
                or index['source_manifest_sha256'] != metadata['files_sha256'][cache_panel + '.csv']
                or len(entries) != len(rows[cache_panel])):
            raise ValueError('Prepared cache panel or completeness mismatch.')
        for entry, row in zip(entries, rows[cache_panel]):
            if any(entry[k] != v for k, v in row.items()) or entry['recipe'] != RECIPE:
                raise ValueError('Prepared identities or preprocessing recipe differ from the manifest.')
        by_id = {entry['video_id']: entry for entry in entries}
        self.entries = [by_id[row['video_id']] for row in self.rows]
        self.shape = tuple(shape)
        self.paths = []
        for entry in self.entries:
            path = (self.directory / entry['file']).resolve()
            if not path.is_relative_to(self.directory) or not path.is_file():
                raise ValueError('Missing or escaped prepared pixel file.')
            self.paths.append(path)
        self.identity = {'panel': panel, 'manifest_sha256': metadata['files_sha256'][panel + '.csv'],
                         'preparation_sha256': sha256(prepared / 'preparation.json'),
                         'index_sha256': sha256(index_path)}

    def pixels(self, index):
        payload = torch.load(self.paths[index], map_location='cpu', weights_only=True)
        entry = self.entries[index]
        if payload['metadata'] != {k: v for k, v in entry.items() if k != 'file'}:
            raise ValueError('Prepared metadata differs from its index.')
        pixels = payload['pixels']
        if (pixels.dtype != torch.float32 or tuple(pixels.shape) != self.shape
                or not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1):
            raise ValueError('Prepared pixels have invalid shape, dtype, or range.')
        if hashlib.sha256(pixels.contiguous().numpy().tobytes()).hexdigest() != entry['pixel_sha256']:
            raise ValueError('Prepared pixel checksum mismatch.')
        return pixels.unsqueeze(0)

    def verify_all(self):
        for index in range(len(self.entries)):
            self.pixels(index)
