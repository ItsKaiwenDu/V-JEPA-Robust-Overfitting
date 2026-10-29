#!/usr/bin/env python3
"""Analyze the complete registered diagnostic panel from saved evidence on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from models.registered_panel_analysis import analyze, markdown, path_in
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, help='Repository-relative frozen input manifest.')
    parser.add_argument('--inputs-sha256', required=True)
    parser.add_argument('--output', required=True, help='New repository-relative output directory.')
    args = parser.parse_args()
    output = path_in(ROOT, args.output)
    if output.exists():
        raise FileExistsError('Preserve the previous analysis; choose a new output directory.')
    import torch
    torch.set_num_threads(4)
    result = analyze(ROOT, path_in(ROOT, args.inputs), args.inputs_sha256)
    output.mkdir(parents=True, exist_ok=False)
    text = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n'
    with (output / 'analysis.json').open('x') as stream:
        stream.write(text)
    with (output / 'analysis.md').open('x') as stream:
        stream.write(markdown(result))
    print(json.dumps(dict(complete=True, cases=20, clips=10,
        output=str(output.relative_to(ROOT)), analysis_sha256=hashlib.sha256(text.encode()).hexdigest(),
        summary=result['scientific_aggregate']['summary'], research_admitted=False), sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
