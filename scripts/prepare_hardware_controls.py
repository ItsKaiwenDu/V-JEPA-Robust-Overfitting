#!/usr/bin/env python3
"""Freeze a compact controls completion package alongside the unchanged handoff."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build(root, output, frozen, evidence):
    from models.hardware_controls import BASE_MANIFEST, PRIOR_ARCHIVE, PROFILE, sha, save
    from models.pilot_bundle import make_archive
    if sha(frozen / 'manifest.json') != BASE_MANIFEST:
        raise ValueError('Use the verified, unchanged hardware handoff B.')
    original = json.loads((evidence / 'downloaded/evidence.tar.gz.manifest.json').read_text())
    archive_path = evidence / 'downloaded/evidence.tar.gz'
    if (original['sha256'] != PRIOR_ARCHIVE or sha(archive_path) != PRIOR_ARCHIVE
            or archive_path.stat().st_size != original['bytes']):
        raise ValueError('Prior evidence archive differs.')
    # Rehash all60 original members, including the six local tensor files.
    from models.hardware_controls import verify_files
    verify_files(evidence / 'extracted', original['files'])
    output.mkdir(parents=True, exist_ok=False)
    base_package = output / 'base-handoff'
    base_package.mkdir()
    frozen_manifest = json.loads((frozen / 'manifest.json').read_text())
    for name in ('source', 'training', 'wheels'):
        path = frozen / (name + '.tar.gz')
        if sha(path) != frozen_manifest['archives'][name]['sha256']:
            raise ValueError('Frozen archive changed: ' + name)
    for name, checksum in (('restore.py', frozen_manifest['restore_sha256']),
                           ('upstream.bundle', frozen_manifest['upstream_bundle_sha256'])):
        if sha(frozen / name) != checksum:
            raise ValueError('Frozen restoration input changed.')
    for name in ('manifest.json', 'source.tar.gz', 'training.tar.gz', 'wheels.tar.gz', 'restore.py', 'upstream.bundle'):
        shutil.copyfile(frozen / name, base_package / name)
    prior = output / 'prior'
    prior.mkdir()
    shutil.copyfile(evidence / 'downloaded/evidence.tar.gz.manifest.json', prior / 'archive-manifest.json')
    for name in original['files']:
        if not name.endswith('.pt'):
            target = prior / 'extracted' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(evidence / 'extracted' / name, target)
    for name in ('archive-verification.json', 'partial-verification.json',
                 'preflight-verification.json', 'feature-verification.json'):
        shutil.copyfile(evidence / name, prior / name)
    source_names = ('models/hardware_controls.py', 'scripts/run_hardware_controls.py',
                    'scripts/prepare_hardware_controls.py', 'tests/test_hardware_controls.py',
                    'docs/hardware_controls_completion.md')
    for name in source_names:
        target = output / 'operations' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    ledger = root / 'report/checks/2026-10-23-runpod-billing-reconciliation.json'
    shutil.copyfile(ledger, output / 'budget-ledger.json')
    template = dict(profile=PROFILE, base_manifest_sha256=BASE_MANIFEST, ledger_sha256=sha(ledger),
        expected_gpu='NVIDIA A40', container_gb=30, volume_gb=80, volume_mount='/workspace',
        quoted_all_in_hourly_usd=None, new_session_hold_usd='2.00', deployment_epoch=None,
        deadline_epoch=None, fresh_quote_confirmed=False, manual_deployment_authorized=False,
        host_fingerprint_verified=False)
    save(output / 'deployment-plan.template.json', template)
    files = {str(path.relative_to(output)): path for path in output.rglob('*')
             if path.is_file() and not path.is_relative_to(base_package)}
    spec = dict(schema_version=1, profile=PROFILE, base_manifest_sha256=BASE_MANIFEST,
        prior_archive_sha256=PRIOR_ARCHIVE, paid_compute_authorized=False, research_admitted=False,
        historical_timeout_preserved=True, proposed_new_session_hold_usd='2.00',
        reconciled_diagnostic_allowance_remaining_usd='4.53',
        files={name: dict(sha256=sha(path), bytes=path.stat().st_size) for name, path in sorted(files.items())})
    save(output / 'controls-manifest.json', spec)
    # The small overlay travels independently of the unchanged ~800MB inputs.
    overlay = make_archive(output / 'controls-overlay.tar.gz',
                           {**files, 'controls-manifest.json': output / 'controls-manifest.json'})
    save(output / 'controls-overlay.tar.gz.manifest.json', overlay)
    print(json.dumps(dict(controls_manifest_sha256=sha(output / 'controls-manifest.json'),
        overlay_sha256=overlay['sha256'], overlay_bytes=overlay['bytes'], overlay_members=len(overlay['files']),
        frozen_manifest_sha256=sha(base_package / 'manifest.json'), new_paid_job_started=False), indent=2))
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen-handoff', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    build(ROOT, args.output, args.frozen_handoff, args.evidence)


if __name__ == '__main__':
    main()
