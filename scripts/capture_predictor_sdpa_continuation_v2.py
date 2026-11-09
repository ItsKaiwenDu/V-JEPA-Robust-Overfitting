"""Source/authorization/deadline checks precede any target scientific import."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
overlay=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(overlay))
from models.predictor_sdpa_continuation_operator_v2 import child,ROLES

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--control-sha256',required=True);p.add_argument('--role',choices=ROLES,required=True);a=p.parse_args()
    sys.exit(child(a.scratch,a.control_sha256,a.role,overlay))
