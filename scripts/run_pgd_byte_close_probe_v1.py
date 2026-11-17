#!/usr/bin/env python3
"""Run one attended inert-byte probe after its exact separate approval."""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('handoff', type=Path)
    args = p.parse_args(argv)
    if not (args.handoff/'BYTE-AUTHORIZATION.json').is_file():
        raise SystemExit('Unarmed: new exact byte-probe approval required. No connection or model work started.')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.pgd_byte_close_probe_v1 import run
    return run(args.handoff)


if __name__ == '__main__':
    sys.exit(main())
