#!/usr/bin/env python3
"""Export the reviewed staged video identities; no video pixels are accessed."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.staged_files import export_manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = export_manifests(ROOT, args.output)
    print(f"Exported {metadata['counts']}. Proposal only; no study or spending admission.")


if __name__ == '__main__':
    main()
