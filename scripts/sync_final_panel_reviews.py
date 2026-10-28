#!/usr/bin/env python3
"""Register the final two original diagnostic reviews; no numerical dispatch."""
import argparse
import importlib.util
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('final_panel_review_registration', ROOT / 'models/final_panel_review_sync.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation-sha256', required=True)
    args = parser.parse_args()
    print('Rechecking both saved cases on CPU; append their exact original reviews only. Preserve all attempts and elapsed history; no model or test suite starts.', file=sys.stderr, flush=True)
    receipt = control.apply(ROOT, args.operation_sha256, Path.home(), socket.gethostname())
    print(json.dumps(receipt, sort_keys=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
