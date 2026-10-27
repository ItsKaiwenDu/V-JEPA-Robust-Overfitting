#!/usr/bin/env python3
"""Recheck saved artifacts on CPU and append the paused eleven-case view."""
import argparse
import importlib.util
import json
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reviewed_panel_control',ROOT/'models/reviewed_panel.py')
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--operation-sha256',required=True)
    args=parser.parse_args()
    print('Rechecking eleven saved cases on CPU; legacy records and STOP remain intact. No model starts.',file=sys.stderr,flush=True)
    result=control.apply(ROOT,args.operation_sha256,Path.home(),socket.gethostname())
    print(json.dumps(result,sort_keys=True,indent=2),flush=True)


if __name__=='__main__':main()
