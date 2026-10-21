#!/usr/bin/env python3
"""Prepare the 100 training and 20 exploration clips, with two deterministic decodes."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from datasets.staged_preparation import prepare_exploration
    torch.set_num_threads(4)
    report = prepare_exploration(ROOT, args.manifests, args.archive, args.output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
