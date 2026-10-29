#!/usr/bin/env python3
"""Offline source/inputs handoff, preserving the original training-only archives."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build(root, output, base):
    from models.candidate_benchmark import benchmark_proposal
    from models.candidate_contract import SOURCES
    from models.pilot_bundle import digest, make_archive, safe_name
    root, output, base = Path(root), Path(output), Path(base)
    manifest = json.loads((base / 'manifest.json').read_text())
    BASE_MANIFEST = 'e42164317e16fbf6e05f4143fd466f00105371284fc9e769d09d5c71ba82608a'
    if digest(base / 'manifest.json') != BASE_MANIFEST:
        raise ValueError('Use the verified frozen training-only handoff B.')
    for name in ('training', 'wheels'):
        if digest(base / (name + '.tar.gz')) != manifest['archives'][name]['sha256']:
            raise ValueError('Frozen input archive changed.')
    if digest(base / 'upstream.bundle') != manifest['upstream_bundle_sha256']:
        raise ValueError('Frozen upstream bundle changed.')
    # Use an explicit source allowlist, with just the package initializers and
    # metadata required by these adapters. No home scan or old run output.
    names = set(SOURCES) | {
        'configs/research_attack_candidate_v1.json', 'scripts/prepare_candidate_benchmark.py',
        'scripts/run_workflow_pilot.py', 'models/pilot_bundle.py',
        'models/__init__.py', 'attacks/__init__.py', 'datasets/__init__.py',
    }
    # PreparedStagedPanel reads these metadata manifests, never held-out pixels.
    names |= {str(p.relative_to(root)) for p in (root / 'data/manifests/ucf101-staged-20x5-v1').glob('*') if p.is_file()}
    names |= {str(p.relative_to(root)) for p in (root / 'data/manifests/ucf101-split1-v1').glob('*') if p.is_file()}
    names |= {str(p.relative_to(root)) for p in (root / 'tests').glob('test_candidate_*.py')}
    names |= {'tests/conftest.py', 'tests/test_vjepa2.py', 'tests/test_evaluation.py',
              'tests/test_calibration.py', 'tests/test_training.py', 'tests/test_schedule.py'}
    # Restore.py is reused byte-for-byte; it performs verification/extraction,
    # local git clone, and never installs dependencies or starts a GPU worker.
    output.mkdir(parents=True, exist_ok=False)
    files = {safe_name(n): root / n for n in sorted(names)}
    manifest['archives']['source'] = make_archive(output / 'source.tar.gz', files)
    for name in ('training.tar.gz', 'wheels.tar.gz', 'upstream.bundle', 'restore.py'):
        shutil.copyfile(base / name, output / name)
    if digest(output / 'restore.py') != manifest['restore_sha256']:
        raise ValueError('Original restore implementation changed.')
    manifest.update(source_is_working_tree_snapshot=True, base_commit=subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        candidate=benchmark_proposal(), executed_training_clips=0, cuda_execution_performed=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    plan = {'kind': 'candidate_eot8_benchmark_launch_v1', 'candidate_binding': manifest['candidate']['candidate_binding'],
            'authorized': False, 'diagnostic_only': True, 'execution_policy': 'shared-fp64-native',
            'required_phases': manifest['candidate']['required_phases'], 'container_gb': 30, 'persistent_gb': 80,
            'volume_mount': '/workspace', 'total_hourly_usd': None, 'inclusive_session_budget_usd': None,
            'minimum_system_ram_gib': 40,
            'deployed_unix': None, 'deadline_unix': None, 'expected_gpu': None, 'expected_gpu_uuid': None,
            'ledger_sha256': None}
    (output / 'launch.template.json').write_text(json.dumps(plan, sort_keys=True, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen-handoff', type=Path, required=True)
    args = parser.parse_args()
    value = build(ROOT, args.output, args.frozen_handoff)
    print(json.dumps({'source_files': len(value['archives']['source']['files']),
                      'cuda_execution_performed': False, 'paid_launch_authorized': False}))
