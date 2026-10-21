#!/usr/bin/env python3
"""Build a local offline diagnostic package; no cloud API or GPU execution."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wheelhouse', type=Path, required=True)
    parser.add_argument('--extra', action='append', default=[], help='Explicit new source path not yet tracked by Git.')
    args = parser.parse_args()
    from models.pilot_bundle import build, digest
    value = build(ROOT, args.output, args.wheelhouse, args.extra)
    print('Source files:', len(value['archives']['source']['files']))
    print('Training clips:', value['packaged_training_clips'])
    print('Manifest SHA256:', digest(args.output / 'manifest.json'))


if __name__ == '__main__':
    main()
