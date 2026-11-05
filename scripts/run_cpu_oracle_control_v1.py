#!/usr/bin/env python3
"""Run a separately authorized control capture, or retrieve only its saved evidence."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.cpu_oracle_control_client_v1 import main

if __name__ == '__main__': main()
