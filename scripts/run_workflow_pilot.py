#!/usr/bin/env python3
"""Bounded CUDA workflow diagnostic. This command never provisions or stops a Pod."""

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PHASES = ('calibrate', 'features-initial',
          'clean-finetune/split-first', 'clean-finetune/split-resume', 'clean-finetune/reference',
          'adversarial-finetune/split-first', 'adversarial-finetune/split-resume', 'adversarial-finetune/reference',
          'features-trained', 'controls')


def run_child(command, log, seconds):
    """A wall deadline kills the whole worker process group, including on Ctrl-C."""
    if seconds <= 0:
        raise TimeoutError('Pilot wall deadline reached.')
    with Path(log).open('x') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        previous_handler = signal.getsignal(signal.SIGTERM)

        def interrupted(signum, frame):
            raise InterruptedError('Pilot parent received SIGTERM.')

        signal.signal(signal.SIGTERM, interrupted)
        try:
            status = process.wait(timeout=seconds)
            if status:
                raise RuntimeError(f'Pilot worker failed with exit status {status}; see {log}.')
        finally:
            try:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            finally:
                signal.signal(signal.SIGTERM, previous_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--clips', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=ROOT / 'external/vjepa2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=5400,
                        help='Worker wall deadline, at most 90 minutes; does not stop cloud billing.')
    parser.add_argument('--phase', choices=PHASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.diagnostic or not 1 <= args.max_seconds <= 5400:
        parser.error('Only --diagnostic with a wall deadline up to 5400 seconds is enabled.')
    args.device, args.limit = 'cuda', 1
    args.output = args.output.resolve()
    if not args.phase and args.output.exists():
        parser.error('Use a fresh output. Failed sessions need inspection before another bounded pilot.')
    if args.phase and not (args.output / 'session.json').is_file():
        parser.error('Worker phases require a parent diagnostic session.')

    from models.checkpointing import file_sha256
    from models.pilot_measurement import compare_recovery
    from models.runtime import configure_cuda_reproducibility, device_metadata, require_device
    from models.study_artifacts import write_record
    from models.training import runtime_contract
    from models.workflow_pilot import run_phase

    configure_cuda_reproducibility()
    require_device('cuda')
    if args.phase:
        print(json.dumps(run_phase(ROOT, args), indent=2))
        return

    # Four trajectories retain two rolling AdamW states and epoch-one weights.
    # The extra margin covers a simultaneous atomic write and diagnostic caches.
    existing_parent = args.output.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < 55 * 2**30:
        parser.error('Need at least 55 GiB free after inputs/setup; use an 80 GB persistent volume.')
    metadata = device_metadata('cuda')
    if '5090' not in metadata['gpu'] or metadata['vram_bytes'] < 30 * 2**30:
        parser.error('This handoff targets one RTX 5090 with at least 30 GiB visible VRAM.')
    args.output.mkdir(parents=True)
    (args.output / 'logs').mkdir()
    started = time.monotonic()
    session = {'schema_version': 1, 'diagnostic_only': True, 'research_result': False,
               'budget_approved': False, 'cloud_billing_stopped_by_runner': False,
               'max_worker_wall_seconds': args.max_seconds, 'device': metadata,
               'runtime': runtime_contract('cuda'), 'phases': [], 'complete': False,
               'started_unix_seconds': time.time()}
    session_path = args.output / 'session.json'
    write_record(session_path, session)
    try:
        for phase in PHASES:
            print(f'Pilot phase: {phase}', flush=True)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--diagnostic', '--phase', phase,
                       '--output', str(args.output), '--checkpoint', str(args.checkpoint.resolve()),
                       '--manifests', str(args.manifests.resolve()), '--clips', str(args.clips.resolve()),
                       '--upstream', str(args.upstream.resolve())]
            before = time.monotonic()
            run_child(command, args.output / 'logs' / (phase.replace('/', '-') + '.log'),
                      args.max_seconds - (before - started))
            result_path = args.output / 'phase-results' / (phase + '.json')
            session['phases'].append({'phase': phase, 'process_wall_seconds': time.monotonic() - before,
                                      'result_sha256': file_sha256(result_path)})
            if phase.endswith('/reference'):
                condition = phase.split('/')[0]
                comparison = compare_recovery(args.output / condition / 'reference', args.output / condition / 'split')
                write_record(args.output / (condition + '-comparison.json'), comparison)
                if not comparison['exact_match']:
                    raise RuntimeError(f'{condition}: exact recovery comparison failed; no tolerance was relaxed.')
            write_record(session_path, session, replace=True)
        session['complete'] = True
        print('Bounded diagnostic complete. Export evidence, then stop/terminate the Pod in Runpod.', flush=True)
    except BaseException as exc:
        session.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        session['wall_seconds'] = time.monotonic() - started
        write_record(session_path, session, replace=True)


if __name__ == '__main__':
    main()
