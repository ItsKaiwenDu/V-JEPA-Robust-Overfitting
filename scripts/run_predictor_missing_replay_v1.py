#!/usr/bin/env python3
"""Run only a separately authorized, unused single-replay attended handoff."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.predictor_missing_replay_client_v1 import run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--handoff',type=Path,required=True);a=p.parse_args()
    sys.exit(run(a.handoff))
