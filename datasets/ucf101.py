"""Reproducible source-group-separated subsets of official UCF101 split 1."""

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import re


@dataclass(frozen=True)
class Video:
    video_id: str
    relative_path: str
    class_id: int
    source_group_id: str


def read_official(lists: Path) -> tuple[dict[str, int], list[Video], list[Video]]:
    classes = {}
    for line in (lists / 'classInd.txt').read_text().splitlines():
        if not line.strip():
            continue
        number, name = line.split()
        if name in classes or int(number) < 1:
            raise ValueError('Duplicate or invalid class index.')
        classes[name] = int(number) - 1
    if sorted(classes.values()) != list(range(len(classes))):
        raise ValueError('Class indices must be unique and contiguous starting at 1.')

    def parse(filename, labelled):
        rows = []
        seen = set()
        for line in (lists / filename).read_text().splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != (2 if labelled else 1):
                raise ValueError(f'Malformed row in {filename}: {line}')
            path = PurePosixPath(fields[0])
            if path.is_absolute() or len(path.parts) != 2 or '..' in path.parts:
                raise ValueError('Expected a dataset-relative Class/video.avi path.')
            name = path.parts[0]
            match = re.fullmatch(r'v_(.+)_g(\d{2})_c(\d{2})\.avi', path.name)
            # Official annotations use HandstandPushups/v_HandStandPushups...
            # Preserve the original path while accepting this case-only alias.
            if not match or match[1].casefold() != name.casefold() or name not in classes:
                raise ValueError(f'Invalid UCF101 filename: {path}')
            if labelled and int(fields[1]) - 1 != classes[name]:
                raise ValueError(f'Class label disagrees with path: {path}')
            if str(path) in seen:
                raise ValueError(f'Duplicate official video: {path}')
            seen.add(str(path))
            rows.append(Video(path.stem, str(path), classes[name], f'{name}_g{match[2]}'))
        return rows

    train, test = parse('trainlist01.txt', True), parse('testlist01.txt', False)
    if {v.source_group_id for v in train} & {v.source_group_id for v in test}:
        raise ValueError('Official train/test source groups overlap.')
    return classes, train, test


def select_subsets(classes, official_train, official_test, *, seed=20261016,
                   train_count=20, validation_count=5, test_count=5):
    if any(type(n) is not int or n < 1 for n in (train_count, validation_count, test_count)):
        raise ValueError('All subset counts must be positive integers.')
    selected = {'train': [], 'validation': [], 'test': []}
    held_out = {}
    for name, class_id in sorted(classes.items(), key=lambda item: item[1]):
        rng = random.Random(f'ucf101-split1-v1:{seed}:{class_id}')
        groups = defaultdict(list)
        for video in official_train:
            if video.class_id == class_id:
                groups[video.source_group_id].append(video)
        group_ids = sorted(groups)
        rng.shuffle(group_ids)
        validation_pool, validation_groups = [], []
        for group_id in group_ids:
            validation_groups.append(group_id)
            validation_pool.extend(groups[group_id])
            if len(validation_pool) >= validation_count:
                break
        training_pool = [v for group_id, rows in groups.items()
                         if group_id not in validation_groups for v in rows]
        test_pool = [v for v in official_test if v.class_id == class_id]
        pools = [('train', training_pool, train_count), ('validation', validation_pool, validation_count),
                 ('test', test_pool, test_count)]
        for split, pool, count in pools:
            if len(pool) < count:
                raise ValueError(f'{name}: insufficient {split} videos after whole-group holdout.')
            selected[split].extend(rng.sample(sorted(pool, key=lambda v: v.video_id), count))
        held_out[name] = sorted(validation_groups)
    for rows in selected.values():
        rows.sort(key=lambda v: (v.class_id, v.video_id))
    audit_subsets(selected)
    return selected, held_out


def audit_subsets(splits):
    seen_videos, seen_groups = set(), set()
    for split in ('train', 'validation', 'test'):
        rows = splits[split]
        videos = {v.video_id for v in rows}
        groups = {v.source_group_id for v in rows}
        if len(videos) != len(rows) or videos & seen_videos or groups & seen_groups:
            raise ValueError(f'Duplicate video or source-group leakage involving {split}.')
        seen_videos.update(videos)
        seen_groups.update(groups)


def write_manifests(lists: Path, output: Path, *, seed=20261016, train_count=20,
                    validation_count=5, test_count=5, expected_classes=101) -> dict:
    classes, official_train, official_test = read_official(lists)
    if len(classes) != expected_classes:
        raise ValueError(f'Expected {expected_classes} classes, got {len(classes)}.')
    splits, held_out = select_subsets(classes, official_train, official_test, seed=seed,
                                    train_count=train_count, validation_count=validation_count,
                                    test_count=test_count)
    contents = {}
    for name, rows in splits.items():
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=list(Video.__dataclass_fields__), lineterminator='\n')
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
        contents[f'{name}.csv'] = buffer.getvalue().encode()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    combined = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    metadata = {'recipe': 'ucf101-split1-grouped-v1', 'official_split': 1, 'seed': seed,
                'class_ids': 'zero-based; official classInd.txt indices minus one',
                'classes': classes, 'counts': {key: len(rows) for key, rows in splits.items()},
                'per_class': {'train': train_count, 'validation': validation_count, 'test': test_count},
                'held_out_validation_groups': held_out, 'manifest_sha256': combined,
                'files_sha256': hashes,
                'source_files_sha256': {name: hashlib.sha256((lists / name).read_bytes()).hexdigest()
                                        for name in ('classInd.txt', 'trainlist01.txt', 'testlist01.txt')},
                'source_url': 'https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip',
                'video_content_checked': False, 'preprocessing_locked': False}
    output.mkdir(parents=True, exist_ok=False)
    for name, content in contents.items():
        (output / name).write_bytes(content)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    return metadata
