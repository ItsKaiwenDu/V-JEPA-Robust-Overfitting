"""Explicit local CPU native observer rehearsal, never a target launcher."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runner',choices=('creator','other-thread'),default='creator')
    args=parser.parse_args();args.device='cpu';args.upstream=ROOT/'external/vjepa2'
    old=ROOT/'runs/pretrained/diagnostic/predictor-eot8-10603-v1-20261105/cpu-four-worker'
    args.initial=old/'initial.pt';args.reference=old/'0'
    from models.predictor_sdpa_observer_cpu_v1 import run
    state=run(args)
    print('Local CPU native boundary complete:',state['complete'],'Original values replay exactly:',state['original_value_replay_exact'])

if __name__=='__main__':main()
