"""Thin attended CLI; one separately authorized metadata-only SSH connection."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def main():
 p=argparse.ArgumentParser();p.add_argument('handoff',type=Path);args=p.parse_args()
 if not (args.handoff/'AUTHORIZATION.json').is_file():raise ValueError('Unarmed metadata-only handoff; new exact approval required.')
 from models.pgd_publication_metadata_v1 import run
 return run(args.handoff)
if __name__=='__main__':sys.exit(main())
