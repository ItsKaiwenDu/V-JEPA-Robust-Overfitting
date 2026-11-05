#!/usr/bin/env python3
"""Packaged control entrypoint. Preserve overlay models and resolve sibling packages."""
import argparse
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--launcher-started', type=float, required=True)
    p.add_argument('--control-sha256')
    p.add_argument('--local-tiny-rehearsal', action='store_true', help='Local tiny fixtures only; refused on university hosts.')
    a = p.parse_args()
    if (a.packet.is_symlink() or a.packet.resolve() != (ROOT.parent/'code').resolve()
            or a.scratch.is_symlink() or a.scratch.resolve() != ROOT.parent.resolve()):
        p.error('Packet and scratch must be the original sibling package directories.')
    # models remains the overlay package, with its original models fallback.
    # The full packet root is separately required by attacks/datasets imports.
    sys.path.insert(1, str(a.packet.resolve()))
    from models import cpu_oracle_control_v1 as operator
    if a.local_tiny_rehearsal:
        manifest = json.loads((a.packet/'CPU-ORACLE-HANDOFF.json').read_text())
        if platform.node() in operator.old.HOSTS or manifest.get('tiny_fixture') is not True or a.control_sha256:
            p.error('Explicit tiny rehearsal is local only and cannot use a live control.')
    else:
        if not a.control_sha256 or operator.old.digest(a.scratch/'operator-control.json') != a.control_sha256:
            p.error('Changed or missing new control control.')
        c = json.loads((a.scratch/'operator-control.json').read_text()); operator.validate_control(c)
        operator.validate_authorization(json.loads((a.scratch/'resource-authorization.json').read_text()),
            a.control_sha256, c['host'])
        for parent, entries in ((ROOT, {n: {'sha256': s} for n, s in c['remote_source_sha256'].items()}),
                                (a.packet, c['archive_manifest']['files'])):
            for name, spec in entries.items():
                path = parent/operator.old.safe_name(name)
                if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(parent.resolve())
                        or operator.old.digest(path) != spec['sha256']):
                    p.error('Packaged source or fixture changed: '+name)
    from models.cpu_oracle_control_runtime_v1 import run_workers
    result = run_workers(a.packet, a.scratch, a.upstream, a.launcher_started, local=a.local_tiny_rehearsal)
    raise SystemExit(0 if result['complete'] and result['saved_value_checks_passed'] else 1)


if __name__ == '__main__': main()
