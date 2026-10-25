#!/usr/bin/env python3
"""Freeze or verify the metadata-only broader matched diagnostic specification."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--output', type=Path, help='Create a fresh plan; existing files are refused.')
    action.add_argument('--verify', type=Path, help='Verify an existing plan without modifying it.')
    args = parser.parse_args()
    from models.matched_panel_plan import freeze_plan, validate_plan
    result = (freeze_plan(ROOT, args.output) if args.output else
              validate_plan(ROOT, json.loads(args.verify.read_text())))
    print(json.dumps(result, indent=2, sort_keys=True))
    print('Specification only. No pixels/weights opened, GPU job, handoff or study admission.')


if __name__ == '__main__':
    main()
