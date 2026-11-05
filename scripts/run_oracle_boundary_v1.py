#!/usr/bin/env python3
"""Run one separately authorized boundary capture or collect its existing files."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.oracle_boundary_client_v1 import main
if __name__=='__main__':main()
