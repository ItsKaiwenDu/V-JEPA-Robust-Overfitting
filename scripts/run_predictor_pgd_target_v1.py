#!/usr/bin/env python3
"""Use only the selected, separately armed handoff; no automatic retries."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.predictor_pgd_target_client_v1 import run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--handoff',type=Path,required=True)
    a=p.parse_args();sys.exit(run(a.handoff))
