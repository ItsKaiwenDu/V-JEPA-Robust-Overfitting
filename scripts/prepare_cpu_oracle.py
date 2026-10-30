#!/usr/bin/env python3
"""Build a CPU-only handoff; no remote connection or worker/resource launch."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.cpu_oracle_packet import build


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--tiny',action='store_true')
    a=p.parse_args();m,spec=build(ROOT,a.output,tiny=a.tiny)
    print('Unarmed CPU-only handoff:',len(m['files']),'files;',spec['bytes'],'bytes; SHA256',spec['sha256'])


if __name__=='__main__':main()
