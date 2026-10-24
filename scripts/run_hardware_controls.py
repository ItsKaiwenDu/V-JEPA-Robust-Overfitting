#!/usr/bin/env python3
"""Operate a separate controls-only attempt using an unchanged frozen project."""

import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('offline', 'launch', 'worker', 'gate', 'combine', 'export'), required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.mode in ('worker', 'gate', 'combine', 'export') and args.run is None:
        parser.error('This mode needs --run.')
    # Load the operational module by file, avoiding any substitution of frozen
    # scientific modules by the newer working repository's models namespace.
    path = Path(__file__).resolve().parents[1] / 'models/hardware_controls.py'
    spec = importlib.util.spec_from_file_location('vjepa_controls_operations', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dispatch(args)


if __name__ == '__main__':
    raise SystemExit(main())
