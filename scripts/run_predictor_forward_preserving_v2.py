#!/usr/bin/env python3
"""One reviewed unarmed-by-default attended client; no retry/download mode."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictor_forward_preserving_client_v2 import run

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--handoff', type=Path, required=True)
    args = p.parse_args(); sys.exit(run(args.handoff))
