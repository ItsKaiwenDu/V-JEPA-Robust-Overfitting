"""One attended fixed-input handoff; unarmed refuses before claim/SSH."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.predictor_sdpa_continuation_client_v6 import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--handoff',type=Path,required=True);a=p.parse_args()
    sys.exit(run(a.handoff))
