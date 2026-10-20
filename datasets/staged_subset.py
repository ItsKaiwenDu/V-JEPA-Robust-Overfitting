"""Metadata-only proposed subsets; never opens videos or changes parent manifests."""

from collections import defaultdict
import hashlib
from itertools import combinations


def propose_subset(parent, *, classes, train_per_class, seed, excluded_groups=()):
    """Hold out exploration groups before selecting actual training videos.

    Confirmation validation and test remain in their original partitions.
    Ranking uses only identifiers, never losses or image content. For larger
    alternatives, round robin maximizes group diversity before reusing groups.
    """
    if any(type(n) is not int or n <= 0 for n in (classes, train_per_class)):
        raise ValueError('Class and training counts must be positive integers.')
    required = ('train', 'validation', 'test')
    if set(parent) != set(required):
        raise ValueError('Need the three original partitions.')
    for split, rows in parent.items():
        if len({r['video_id'] for r in rows}) != len(rows):
            raise ValueError(f'Duplicate video in {split}.')
    for a, b in combinations(required, 2):
        for key in ('video_id', 'source_group_id'):
            if {r[key] for r in parent[a]} & {r[key] for r in parent[b]}:
                raise ValueError(f'Parent partitions overlap by {key}.')

    def rank(namespace, value):
        return hashlib.sha256(f'{seed}|{namespace}|{value}'.encode()).hexdigest()

    by_class = {}
    for split, rows in parent.items():
        grouped = defaultdict(list)
        for row in rows:
            grouped[row['relative_path'].split('/')[0]].append(row)
        by_class[split] = grouped
    names = set(by_class['train'])
    if any(set(by_class[s]) != names for s in required) or classes > len(names):
        raise ValueError('Class coverage differs or the request exceeds the pool.')
    chosen = sorted(names, key=lambda name: (rank('class', name), name))[:classes]

    def spread(rows, count, namespace):
        if len(rows) < count:
            raise ValueError(f'Insufficient videos for {namespace}.')
        groups = defaultdict(list)
        for row in rows:
            groups[row['source_group_id']].append(row)
        ordered = sorted(groups, key=lambda g: (rank(namespace + ':group', g), g))
        for group in ordered:
            groups[group].sort(key=lambda r: (rank(namespace + ':video', r['video_id']), r['video_id']))
        result, depth = [], 0
        while len(result) < count:
            for group in ordered:
                if depth < len(groups[group]):
                    result.append(dict(groups[group][depth]))
                    if len(result) == count:
                        break
            depth += 1
        return result

    selected = {s: [] for s in ('train', 'exploration_validation', 'confirmation_validation', 'test')}
    monitor, held_out = [], {}
    excluded = set(excluded_groups)
    for name in chosen:
        pool = by_class['train'][name]
        groups = {r['source_group_id'] for r in pool} - excluded
        if not groups:
            raise ValueError(f'No unused exploration group for {name}.')
        group = min(groups, key=lambda g: (rank('exploration_group', g), g))
        held_out[name] = group
        selected['exploration_validation'] += spread(
            [r for r in pool if r['source_group_id'] == group], 1, 'exploration_validation')
        train = spread([r for r in pool if r['source_group_id'] != group], train_per_class, 'train')
        selected['train'] += train
        monitor += spread(train, 1, 'train_monitor')
        selected['confirmation_validation'] += spread(by_class['validation'][name], 1, 'confirmation_validation')
        selected['test'] += spread(by_class['test'][name], 3, 'test')

    for a, b in combinations(selected, 2):
        for key in ('video_id', 'source_group_id'):
            if {r[key] for r in selected[a]} & {r[key] for r in selected[b]}:
                raise ValueError(f'Proposed partitions overlap by {key}.')
    group_counts = {
        split: {name: len({r['source_group_id'] for r in rows
                           if r['relative_path'].split('/')[0] == name}) for name in chosen}
        for split, rows in selected.items()
    }
    return {
        'status': 'proposed_metadata_only_not_frozen',
        'classes': chosen,
        'counts': {s: len(rows) for s, rows in selected.items()},
        'source_groups_per_class': group_counts,
        'exploration_groups_excluded_entirely_from_actual_training': held_out,
        'partitions_video_and_source_group_disjoint': True,
        'train_monitor_video_ids': [r['video_id'] for r in monitor],
        'rows': selected,
        'video_content_checked': False,
        'test_pixels_or_outcomes_accessed': False,
    }
