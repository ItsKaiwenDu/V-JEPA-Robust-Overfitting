"""Compare only the already-collected component NPY returns locally."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.predictor_pgd_component_returns_v1 import analyze,encoded


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();folder=args.output.absolute()
    if folder.parent.resolve(strict=True)!=ROOT/'runs/pretrained/diagnostic/pgd-component-returns-v1-20261119':
        raise ValueError('Fresh local saved-return analysis directory required.')
    folder.mkdir(mode=0o700,exist_ok=False)
    result=analyze(ROOT)
    with os.fdopen(os.open(folder/'comparison.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(encoded(result))
    print(json.dumps({'analysis_complete':True,'saved_pair_original_bounds_passed':result['saved_pair_original_bounds_passed'],
        'checks':result['checks'],'new_GPU_or_model_work':False},sort_keys=True))
    return 0 if result['saved_pair_original_bounds_passed'] else 1


if __name__=='__main__':sys.exit(main())
