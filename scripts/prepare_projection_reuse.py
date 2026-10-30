#!/usr/bin/env python3
"""Build the one-training-clip source handoff without SSH or GPU execution."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from models.reuse_handoff import build
    metadata,archive=build(ROOT,args.output)
    print('Handoff prepared; no upload or launch.')
    print('Files:',len(metadata['files']),'training tensors:',metadata['packaged_training_clips'])
    print('Archive bytes:',archive['bytes'],'SHA256:',archive['sha256'])


if __name__=='__main__':main()
