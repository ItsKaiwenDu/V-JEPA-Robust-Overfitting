#!/usr/bin/env python3
"""Build only: no SSH, model execution or resource deployment."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.fixed_attention_packet import build


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tiny',action='store_true',help='Build only small CPU fixtures; never qualify CUDA.')
    args=p.parse_args();metadata,archive=build(ROOT,args.output,tiny=args.tiny)
    print('Unarmed immutable-input attention handoff:',len(metadata['files']),'files;',archive['bytes'],'bytes; SHA256',archive['sha256'])


if __name__=='__main__':main()
