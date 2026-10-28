#!/usr/bin/env python3
"""Activate the new two-case namespace, run one attended case, or review saved evidence."""
import argparse
import importlib.util
import json
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reviewed_case_control', ROOT / 'models/reviewed_panel_last_two.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('activate', 'run', 'worker', 'review'))
    parser.add_argument('--operation-sha256', required=True)
    parser.add_argument('--host')
    parser.add_argument('--role', choices=('matrix', 'replay'))
    parser.add_argument('--launcher-started', type=float)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--archive-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'review':
        if any(v is None for v in (args.package, args.reference, args.archive, args.archive_sha256, args.output)):
            parser.error('Review requires trusted package/reference, archive pin and a fresh output directory.')
        context = control.trusted(ROOT, args.operation_sha256, args.package, args.reference)
        manifest = json.loads(Path(str(args.archive) + '.manifest.json').read_text())
        result = control.review(context, args.archive, manifest, args.archive_sha256, args.output)
        context['ops'].write_record(args.output / 'independent-review.json', result)
        print(json.dumps(result, sort_keys=True, indent=2))
        return
    host = socket.gethostname()
    if args.host is not None and args.host != host:
        parser.error('Requested host differs from the actual workstation.')
    context = control.load(ROOT, args.operation_sha256, Path.home())
    if args.mode == 'activate':
        result = control.activate(context, host)
        print(json.dumps(result, sort_keys=True, indent=2), flush=True)
    elif args.mode == 'run':
        control.run(context, host, args.launcher_started)
    else:
        control.worker(context, host, args.role)


if __name__ == '__main__':
    main()
