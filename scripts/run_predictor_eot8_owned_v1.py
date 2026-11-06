#!/usr/bin/env python3
"""Local Mac entry; an unarmed handoff refuses before SSH."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.predictor_eot8_owned_client_v1 import run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--handoff',type=Path,required=True);p.add_argument('--download-only',action='store_true');a=p.parse_args()
    sys.exit(run(a.handoff,download_only=a.download_only))
