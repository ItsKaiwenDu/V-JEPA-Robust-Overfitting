#!/usr/bin/env python3
"""Create source-group-separated study manifests from official UCF101 lists.

Does not download, decode, evaluate, or copy videos. Output must be a new directory.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.ucf101 import write_manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lists-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--train-per-class', type=int, default=20)
    args = parser.parse_args()
    metadata = write_manifests(args.lists_dir, args.output, seed=args.seed, train_count=args.train_per_class)
    print(json.dumps({'counts': metadata['counts'], 'manifest_sha256': metadata['manifest_sha256']}, indent=2))


if __name__ == '__main__':
    main()
