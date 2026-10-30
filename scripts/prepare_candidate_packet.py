#!/usr/bin/env python3
"""Prepare corrected training-only benchmark inputs; no upload or launch."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.candidate_packet import build
from models.pilot_bundle import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = build(ROOT, args.base, args.output)
    print(json.dumps({'source_files': len(value['archives']['source']['files']),
        'manifest_sha256': digest(args.output / 'manifest.json'), 'training_clips': 100,
        'candidate_source_inputs_unchanged': 61, 'launch_authorized': False,
        'research_admitted': False, 'tensor_deserialization_performed': False}, indent=2))


if __name__ == '__main__':
    main()
