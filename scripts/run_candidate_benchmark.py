#!/usr/bin/env python3
"""Candidate benchmark; requires a new reviewed launch/ledger, never deploys a Pod."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arm_worker_deadline(seconds):
    """Worker's own fatal deadline survives loss of its SSH/parent process."""
    if not 0 < seconds <= 7200:
        raise ValueError('Invalid independent worker deadline.')
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, default=ROOT / 'data/manifests/ucf101-staged-20x5-v1')
    parser.add_argument('--clips', type=Path, default=ROOT / 'data/processed/staged-exploration-v1')
    parser.add_argument('--upstream', type=Path, default=ROOT / 'external/vjepa2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', help=argparse.SUPPRESS)
    args = parser.parse_args()
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest(args.plan) != args.plan_sha256:
        parser.error('Reviewed launch file changed.')
    plan, ledger = json.loads(args.plan.read_text()), json.loads(args.ledger.read_text())
    if digest(args.ledger) != plan.get('ledger_sha256'):
        parser.error('Reviewed ledger file changed.')
    from models.candidate_benchmark import PHASES, launch_limits, run_phase
    seconds, rate, allowance = launch_limits(plan, ledger, now=time.time())
    if args.phase:
        arm_worker_deadline(seconds)
    args.output = args.output.resolve()
    if args.phase:
        if args.phase not in PHASES or not (args.output / 'session.json').is_file():
            parser.error('Worker requires a committed parent session and declared phase.')
    elif args.output.exists():
        parser.error('Use a fresh output directory; there are no automatic retries.')
    # Read-only checks before loading checkpoint tensors or initializing CUDA.
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total',
                                   '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(rows) != 1:
        parser.error('Exactly one visible GPU is required.')
    gpu, uuid, mib = [s.strip() for s in rows[0].split(',')]
    if gpu != plan.get('expected_gpu') or uuid != plan.get('expected_gpu_uuid') or float(mib) < 22 * 1024:
        parser.error('GPU identity or sufficient-memory boundary differs from reviewed plan.')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'], text=True).strip()
    if active:
        parser.error('Another CUDA compute process is present; leave it alone and stop here.')
    ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    for cap in (Path('/sys/fs/cgroup/memory.max'), Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')):
        if cap.is_file() and cap.read_text().strip().isdigit():
            ram = min(ram, int(cap.read_text().strip()))
    if ram < plan['minimum_system_ram_gib'] * 2**30:
        parser.error('Need at least 40 GiB usable system RAM for state comparison and checkpoint IO.')
    from models.checkpointing import file_sha256
    from models.study_artifacts import read_record, write_record
    from scripts.run_workflow_pilot import run_child
    args.device, args.limit, args.execution_policy = 'cuda', 1, 'shared-fp64-native'
    if args.phase:
        session = read_record(args.output / 'session.json')
        if (session['plan_sha256'] != args.plan_sha256 or session['complete']
                or PHASES[len(session['phases'])] != args.phase):
            parser.error('Worker phase does not match the parent session prefix.')
        from models.execution import configure_execution
        configure_execution('shared-fp64-native', device='cuda', strict_original=True)
        run_phase(ROOT, args)
        other = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True).strip().splitlines()
        if any(p.strip() and p.strip() != str(os.getpid()) for p in other):
            raise RuntimeError('Additional CUDA process appeared; phase not accepted, leave all processes alone.')
        return
    parent = args.output.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < 55 * 2**30:
        parser.error('Need 55 GiB free after inputs/setup on the persistent volume.')
    args.output.mkdir(parents=True)
    (args.output / 'logs').mkdir()
    started = time.monotonic()
    # Monotonic time cannot be reset by a backwards wall-clock adjustment.
    end = started + seconds
    session = {'kind': 'candidate_eot8_benchmark_session_v1', 'candidate_binding': plan['candidate_binding'],
        'plan_sha256': args.plan_sha256, 'ledger_sha256': plan['ledger_sha256'],
        'required_phases': list(PHASES), 'phases': [], 'complete': False,
        'diagnostic_only': True, 'research_admitted': False, 'budget_approved': False,
        'cloud_billing_stopped_by_runner': False, 'deployed_unix': plan['deployed_unix'],
        'deadline_unix': plan['deadline_unix'], 'rate_usd': rate, 'session_allowance_usd': allowance,
        # Parent stays CPU-only so its CUDA context cannot contend with workers.
        'device': {'gpu': gpu, 'uuid': uuid, 'vram_mib': float(mib)},
        'runtime': 'recorded_by_each_model_worker'}
    record = args.output / 'session.json'
    write_record(record, session)
    try:
        for phase in PHASES:
            remaining, _, _ = launch_limits(plan, ledger, now=time.time())
            remaining = min(remaining, end - time.monotonic())
            print('Candidate phase: ' + phase, flush=True)
            command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--phase', phase,
                '--plan', str(args.plan.resolve()), '--plan-sha256', args.plan_sha256,
                '--ledger', str(args.ledger.resolve()), '--checkpoint', str(args.checkpoint.resolve()),
                '--manifests', str(args.manifests.resolve()), '--clips', str(args.clips.resolve()),
                '--upstream', str(args.upstream.resolve()), '--output', str(args.output)]
            run_child(command, args.output / 'logs' / (phase.replace('/', '-') + '.log'),
                      min(1200, remaining) if phase == 'preflight' else remaining)
            path = args.output / 'phase-results' / (phase.replace('/', '-') + '.json')
            value = read_record(path)
            if value['phase'] != phase or value['candidate_binding'] != plan['candidate_binding']:
                raise ValueError('Worker result profile/phase changed.')
            if phase.endswith('/reference'):
                comparison = read_record(args.output / ('recovery-' + phase.split('/')[0] + '.json'))
                if not comparison['exact_match'] or value.get('exact_recovery') is not True:
                    raise RuntimeError('Exact optimizer recovery rejected; no tolerance was changed.')
            session['phases'].append({'phase': phase, 'result_sha256': file_sha256(path)})
            write_record(record, session, replace=True)
        session['complete'] = True
        print('Diagnostic complete. Copy and verify evidence, then terminate the Pod manually.', flush=True)
    except BaseException as exc:
        session.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        session['wall_seconds'] = time.monotonic() - started
        write_record(record, session, replace=True)


if __name__ == '__main__':
    main()
