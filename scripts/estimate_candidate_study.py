#!/usr/bin/env python3
"""Verify saved EOT8 evidence and screen the original full study; offline only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.candidate_budget import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output; historical evidence must remain unchanged.')
    value = build_report(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    for name, scenario in value['economics']['scenarios'].items():
        print(f"{name}: ${scenario['total_with_historical_allowance_usd']:.2f}; "
              f"same-sample-throughput rate ceiling ${scenario['break_even_all_in_rate_at_same_sample_throughput_usd']:.4f}/hour. "
              'Incomplete screen; no research or paid launch authorized.')


if __name__ == '__main__':
    main()
