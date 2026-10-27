#!/usr/bin/env python3
"""Register the original YoYo review and return gapped fourteen-case progress; no model starts."""
import argparse
import importlib.util
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reviewed_pair_sync_control', ROOT / 'models/yoyo_review_sync.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation-sha256', required=True)
    args = parser.parse_args()
    print('Rechecking saved YoYo evidence on CPU and registering its original review. Preserve the unused Archery case and startup charge; no model or test suite starts.', file=sys.stderr, flush=True)
    receipt = control.apply(ROOT, args.operation_sha256, Path.home(), socket.gethostname())
    print(json.dumps(receipt, sort_keys=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
