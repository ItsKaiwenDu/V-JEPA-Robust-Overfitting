#!/usr/bin/env python3
"""Decode a training/validation manifest into deterministic raw-pixel clips.

Use --limit for a small pilot; test-manifest preprocessing is deliberately unavailable here.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from datasets.video import decode_clip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.manifest.name not in ('train.csv', 'validation.csv'):
        parser.error('Only train.csv or validation.csv can be used for preparation at this stage.')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive.')
    metadata = json.loads((args.manifest.parent / 'metadata.json').read_text())
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if metadata['files_sha256'][args.manifest.name] != digest:
        parser.error('Manifest checksum disagrees with its metadata.')
    rows = list(csv.DictReader(args.manifest.open()))
    if args.limit is not None and args.limit < len(rows):
        rows = [rows[i * len(rows) // args.limit] for i in range(args.limit)]
    root = args.data_root.resolve()
    paths = [(root / row['relative_path']).resolve() for row in rows]
    if any(not path.is_relative_to(root) or not path.is_file() for path in paths):
        parser.error('A selected relative video path is missing or outside the dataset root.')
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for row, path in zip(rows, paths):
        clip, record = decode_clip(path)
        record.update(row)
        filename = row['video_id'] + '.pt'
        torch.save({'pixels': clip, 'metadata': record}, args.output / filename)
        records.append({'file': filename, **record})
        print(f"Prepared {row['video_id']}", flush=True)
    index = {'source_manifest_sha256': digest, 'split': args.manifest.stem,
             'selection': 'evenly spaced manifest rows' if args.limit else 'all rows', 'clips': records}
    (args.output / 'index.json').write_text(json.dumps(index, indent=2) + '\n')


if __name__ == '__main__':
    main()
