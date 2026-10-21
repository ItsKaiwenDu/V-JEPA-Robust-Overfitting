"""Reproducible staged CSV exports tied to the existing metadata-only proposal."""

import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath

import yaml

from datasets.staged_subset import propose_subset

PLAN = 'configs/budget_study_plan.yaml'
PROPOSAL = 'report/checks/2026-10-20-staged-study-projection.json'
FIELDS = ('video_id', 'relative_path', 'class_id', 'source_group_id')
PANELS = ('train', 'train_monitor', 'exploration_validation', 'confirmation_validation', 'test')
OPEN_PANELS = ('train', 'train_monitor', 'exploration_validation')


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def csv_bytes(rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def expected_export(root):
    """Recreate the proposed rows without accessing any video content."""
    root = Path(root)
    proposal = json.loads((root / PROPOSAL).read_text())
    for name, expected in proposal['source_sha256'].items():
        if sha256(root / name) != expected:
            raise ValueError(f'Proposal source checksum changed: {name}')
    plan = yaml.safe_load((root / PLAN).read_text())
    parent = {}
    for split in ('train', 'validation', 'test'):
        with (root / plan['data']['parent_manifests'] / f'{split}.csv').open() as handle:
            parent[split] = list(csv.DictReader(handle))
    subset = propose_subset(parent, **plan['data']['variants'][plan['data']['recommended_variant']],
                            seed=plan['data']['subset_seed'],
                            excluded_groups=plan['data']['exclude_from_exploration_holdout_groups'])
    if subset != proposal['proposed_subset']:
        raise ValueError('Recomputed subset disagrees with the reviewed metadata proposal.')
    rows = dict(subset['rows'])
    by_id = {row['video_id']: row for row in rows['train']}
    rows['train_monitor'] = [by_id[identifier] for identifier in subset['train_monitor_video_ids']]
    for panel in PANELS:
        for row in rows[panel]:
            path = PurePosixPath(row['relative_path'])
            if (path.is_absolute() or len(path.parts) != 2 or '..' in path.parts
                    or path.suffix != '.avi' or path.stem != row['video_id']):
                raise ValueError('Invalid dataset-relative video path.')
    contents = {panel + '.csv': csv_bytes(rows[panel]) for panel in PANELS}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    metadata = {
        'schema_version': 1, 'recipe': 'staged-20x5-manifests-v1',
        'status': 'exported_proposal_not_research_admission',
        'variant': plan['data']['recommended_variant'], 'classes': subset['classes'],
        'counts': {panel: len(rows[panel]) for panel in PANELS},
        'class_ids': 'original zero-based UCF101 indices; not reindexed',
        'files_sha256': hashes,
        'manifest_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        'proposal_sha256': sha256(root / PROPOSAL), 'plan_sha256': sha256(root / PLAN),
        'parent_files_sha256': {k: v for k, v in proposal['source_sha256'].items()
                                if k.startswith(plan['data']['parent_manifests'] + '/')},
        'exploration_groups_excluded_from_training': subset['exploration_groups_excluded_entirely_from_actual_training'],
        'source_groups_per_class': subset['source_groups_per_class'],
        'train_monitor_is_training_subset': True,
        'partitions_video_and_source_group_disjoint': True,
        'video_content_checked_by_this_export': False,
        'confirmation_and_test_pixels_opened': False,
        'protocol_frozen': False, 'budget_approved': False,
    }
    return rows, contents, metadata


def export_manifests(root, output):
    _, contents, metadata = expected_export(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in contents.items():
        (output / name).write_bytes(data)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    return metadata


def verify_manifests(root, directory):
    """Check the exported bytes against a fresh deterministic proposal replay."""
    rows, contents, metadata = expected_export(root)
    directory = Path(directory)
    for name, data in contents.items():
        if (directory / name).read_bytes() != data:
            raise ValueError(f'Staged manifest differs from the proposed identities: {name}')
    if json.loads((directory / 'metadata.json').read_text()) != metadata:
        raise ValueError('Staged manifest metadata differs from proposal replay.')
    return rows, metadata
