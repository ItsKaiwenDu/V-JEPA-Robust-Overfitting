#!/usr/bin/env python3
"""Build a training-only, offline matched-attack handoff; never launch a GPU job."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build(root, output):
    from models.matched_diagnostic import NEW_SOURCES, monitor_input
    from models.pilot_bundle import CHECKPOINT, UPSTREAM, digest, make_archive, safe_name
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    prior = root / 'runs/pretrained/diagnostic/a40-controls-session-20261023/extracted/package/prior/extracted/handoff-manifest.json'
    frozen = json.loads(prior.read_text())['archives']['source']['files']
    science = {n: r['sha256'] for n, r in frozen.items() if not n.endswith('.md')}
    if len(science) != 198 or any(digest(root / n) != sha for n, sha in science.items()):
        raise ValueError('The existing 198 scientific source pins must remain unchanged.')
    _, data = monitor_input(root, root / 'data/manifests/ucf101-staged-20x5-v1',
                            root / 'data/processed/staged-exploration-v1')
    tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z'], text=True).split('\0')
    new = (*NEW_SOURCES, 'tests/test_matched_attack.py', 'tests/fixtures/matched_attack_worker.py',
           'docs/matched_attack_diagnostic.md')
    files = {}
    for name in sorted(set(n for n in tracked if n) | set(new)):
        safe_name(name)
        path = root / name
        if (path.is_symlink() or not path.resolve().is_relative_to(root)
                or not path.is_file() or path.stat().st_size > 10 * 2**20
                or path.suffix in ('.pt', '.pth', '.whl', '.key', '.pem')
                or any(part in ('.git', '.venv', '.ssh') for part in path.parts)
                or path.name == '.env' or path.name.startswith('.env.')):
            raise ValueError('Unexpected non-source/private input: ' + name)
        files[name] = path
    prepared = root / 'data/processed/staged-exploration-v1'
    index = json.loads((prepared / 'train/index.json').read_text())
    entry, = [e for e in index['clips'] if e['video_id'] == data['row']['video_id']]
    for name in ('preparation.json', 'train/index.json', 'train/' + entry['file']):
        files['data/processed/matched-monitor-v1/' + name] = prepared / name
    upstream = root / 'external/vjepa2'
    if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != UPSTREAM:
        raise ValueError('Upstream revision changed.')
    output.mkdir(parents=True)
    # Reuse the existing standard-library archive verifier, with a CLI for
    # this single-archive format rather than the older three-archive pilot.
    helper = (root / 'models/pilot_bundle.py').read_text().split("\nif __name__ == '__main__':", 1)[0]
    helper += '''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify and extract the matched attack handoff.')
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    extract_verified(args.archive, json.loads(args.manifest.read_text()), args.destination)
    print('Matched handoff extracted and verified. No GPU job started.')
'''
    helper_path = output / 'verify_extract.py'
    helper_path.write_text(helper)
    subprocess.run(['git', '-C', str(upstream), 'bundle', 'create', str(output / 'upstream.bundle'), 'HEAD'], check=True)
    files['upstream.bundle'] = output / 'upstream.bundle'
    files['verify_extract.py'] = helper_path
    manifest = {'kind': 'matched-training-attack-handoff-v1', 'diagnostic_only': True, 'research_admitted': False,
                'source_is_working_tree_snapshot': True, 'source_commit': subprocess.check_output(
                    ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                'existing_science_pins_verified': len(science), 'existing_science_sha256': science,
                'checkpoint_sha256': CHECKPOINT, 'checkpoint_included': False, 'upstream_commit': UPSTREAM,
                'training_clip': data, 'training_tensor_count': 1, 'held_out_pixel_tensors_included': False,
                'fresh_gpu_status_verified': False, 'gpu_job_started': False, 'paid_compute_authorized': False,
                'files': {n: {'bytes': p.stat().st_size, 'sha256': digest(p)} for n, p in files.items()}}
    manifest_path = output / 'MATCHED-HANDOFF.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    files['MATCHED-HANDOFF.json'] = manifest_path
    archive = output / 'vjepa-matched-attack-20261023.tar.gz'
    prefix = 'matched-attack-20261023/'
    specification = make_archive(archive, {prefix + n: p for n, p in files.items()})
    specification_path = output / (archive.name + '.manifest.json')
    specification_path.write_text(json.dumps(specification, indent=2, sort_keys=True) + '\n')
    (output / 'SHA256SUMS').write_text(''.join(digest(path) + '  ' + path.name + '\n'
                                            for path in (archive, specification_path, helper_path)))
    return {'archive': str(archive), 'bytes': specification['bytes'], 'sha256': specification['sha256'],
            'members': len(specification['files']), 'training_tensor_count': 1, 'existing_science_pins': len(science)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(ROOT, args.output), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
