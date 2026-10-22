#!/usr/bin/env python3
"""Export small diagnostic evidence, excluding model/optimizer/pixel tensors."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-preflight-tensors', action='store_true',
                        help='Retain only bounded representative-check tensors; never optimizer states.')
    args = parser.parse_args()
    from models.pilot_bundle import make_archive
    roots = {'pilot': args.run.resolve(), 'setup': ROOT / 'runs/pretrained/diagnostic/workflow-setup'}
    files = {}
    for prefix, directory in roots.items():
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.txt', '.log', '.xml'):
                if path.is_symlink() or path.stat().st_size > 10 * 2**20:
                    raise ValueError('Unexpected evidence file; inspect before exporting.')
                files[prefix + '/' + str(path.relative_to(directory))] = path
    if args.include_preflight_tensors:
        preflight = roots['pilot'] / 'preflight'
        tensors = sorted(preflight.glob('clip-*.pt'))
        if (len(tensors) > 6 or sum(p.stat().st_size for p in tensors) > 2 * 2**30
                or any(p.is_symlink() or not p.is_file() or p.stat().st_size > 256 * 2**20 for p in tensors)):
            raise ValueError('Unexpected preflight tensor export; inspect before copying.')
        for path in tensors:
            files['pilot/preflight/' + path.name] = path
    if 'pilot/session.json' not in files:
        raise ValueError('No pilot session record to export.')
    if (ROOT / 'handoff-manifest.json').exists():
        files['handoff-manifest.json'] = ROOT / 'handoff-manifest.json'
    manifest = make_archive(args.output, files)
    import json
    args.output.with_suffix(args.output.suffix + '.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Export SHA256:', manifest['sha256'])
    print('Bytes:', manifest['bytes'])
    print('Tensor checkpoints are excluded. Keep the Pod until this export is verified locally.')


if __name__ == '__main__':
    main()
