#!/usr/bin/env python3
"""Reassess authenticated saved arrays locally; no SSH or experiment launch."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.cpu_oracle_reassessment_v2 import reassess,verify_reassessment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--review-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-output',action='store_true')
    parser.add_argument('--summary-sha256')
    args=parser.parse_args()
    if args.verify_output:
        if not args.summary_sha256:parser.error('Verification requires a pinned existing summary SHA256.')
        result=verify_reassessment(ROOT,args.review,args.review_sha256,args.output,args.summary_sha256)
    else:
        if args.summary_sha256:parser.error('Summary SHA256 is only for verification.')
        result=reassess(ROOT,args.review,args.review_sha256,args.output)
    print('Offline array reassessment:',result['array_checks_passed'])
    print('Original parent/setup refusals remain; no launch or research authorization.')
    print(json.dumps(result,sort_keys=True))
    raise SystemExit(0 if result['array_checks_passed'] else 1)


if __name__=='__main__':main()
