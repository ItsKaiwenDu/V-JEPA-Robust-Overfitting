#!/usr/bin/env python3
"""One attended outer attempt, only after a new exact authorization is present."""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('handoff',type=Path)
    args=parser.parse_args(argv)
    if not (args.handoff/'AUTHORIZATION.json').is_file():
        raise SystemExit('Unarmed: no new scoped authorization. No timer, claim, SSH or model work started.')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from models.predictor_pgd_host10614_client_v1 import run
    return run(args.handoff)


if __name__=='__main__':
    sys.exit(main())
