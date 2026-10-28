#!/usr/bin/env python3
"""Register the original Archery recovery review; no numerical dispatch."""
import argparse
import importlib.util
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('archery_review_registration', ROOT / 'models/archery_review_sync.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation-sha256', required=True)
    args = parser.parse_args()
    print('Rechecking saved Archery recovery evidence on CPU; append its original review only. Preserve failed attempts and consumed grant; no model or test suite starts.', file=sys.stderr, flush=True)
    receipt = control.apply(ROOT, args.operation_sha256, Path.home(), socket.gethostname())
    print(json.dumps(receipt, sort_keys=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
