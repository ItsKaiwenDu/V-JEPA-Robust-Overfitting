#!/usr/bin/env python3
"""One attended read-only workstation observation after its separate approval."""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('handoff',type=Path);args=p.parse_args(argv)
    if not (args.handoff/'READ-AUTHORIZATION.json').is_file():raise SystemExit('Unarmed: read-only approval required. No SSH, query or model work started.')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from models.university_gpu_availability_v1 import run
    return run(args.handoff)


if __name__=='__main__':sys.exit(main())
