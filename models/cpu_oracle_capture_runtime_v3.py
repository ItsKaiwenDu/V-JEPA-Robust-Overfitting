"""Fresh original workers with a separately versioned saved-value audit.

This runner never invokes or changes the frozen v1 parent acceptance function.
It preserves every v1 worker receipt and all numerical bounds.
"""
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from models.attention_fixture import load, BOOKKEEPING
from models.cpu_oracle_packet import verify, TASKS
from models.cpu_oracle_probe import export, validate_plan, host_gate
from models.cpu_oracle_reassessment_v2 import audit_worker
from models.pilot_bundle import digest
from models.study_artifacts import read_record, write_record
from models.reuse_diagnostic import upstream_gate
from models import cpu_oracle_capture_v3 as operator

KIND = 'cpu-oracle-capture-runtime-v3'


def review_captures(packet, workers, rows, report_dir, *, expected_host, tiny=False):
    """Recompute authenticated saved arrays; writing only new v3 audit receipts."""
    packet, workers, report_dir = Path(packet), Path(workers), Path(report_dir)
    if report_dir.resolve().is_relative_to(workers.resolve()):
        raise ValueError('V3 audit must not write inside an original worker capture.')
    manifest_path = packet/'CPU-ORACLE-HANDOFF.json'; sha = digest(manifest_path)
    if not tiny and sha != operator.old.HANDOFF_SHA: raise ValueError('Full packet identity changed.')
    manifest = verify(packet, sha)
    if manifest['tiny_fixture'] is not tiny: raise ValueError('Tiny/full capture domain differs.')
    if (not rows or len(rows) > 4
            or [[r['threads'], r['replay']] for r in rows] != [list(t) for t in TASKS[:len(rows)]]):
        raise ValueError('Wrong or reordered worker coverage.')
    values, _ = load(packet/manifest['fixture']['folder'], manifest['fixture']['sha256'], case='encoder-1', tiny=tiny)
    report_dir.mkdir(parents=True, exist_ok=False); checked = []
    for i, row in enumerate(rows):
        result = audit_worker(workers/str(i), row['worker_sha256'], values, manifest['fixture'], tiny=tiny, handoff_sha=sha)
        if result['producer_context']['host'] != expected_host: raise ValueError('Worker producer host changed.')
        if [result['threads'], result['replay']] != list(TASKS[i]): raise ValueError('Worker identity differs from planned order.')
        if not tiny and (result['producer_context']['python'], result['producer_context']['torch'], result['producer_context']['numpy']) != ('3.11.17', '2.8.0+cu128', '2.4.6'):
            raise ValueError('Original university worker runtime changed.')
        if i < len(rows)-1 and not result['saved_value_evidence_passed']:
            raise ValueError('A later worker ran after a numerical/integrity refusal.')
        write_record(report_dir/f'worker-{i}.json', result)
        checked.append({'threads': row['threads'], 'replay': row['replay'],
            'worker_sha256': row['worker_sha256'], 'v3_audit_sha256': digest(report_dir/f'worker-{i}.json'),
            'passed': result['saved_value_evidence_passed'],
            'diagnostic_difference_count': result['stage_receipt_comparison']['diagnostic_difference_count'],
            'phase_bitwise_equal': result['phase_bitwise_equal']})
    if len(rows) < 4 and checked[-1]['passed']:
        raise ValueError('Missing planned workers without a preserved numerical refusal.')
    return {'kind': KIND, 'handoff_sha256': sha, 'host': expected_host, 'tiny_fixture': tiny,
        'capture_integrity_verified': True, 'all_planned_workers_completed': len(rows) == 4,
        'saved_value_checks_passed': len(rows) == 4 and all(r['passed'] for r in checked),
        'workers': checked, 'bounds': BOOKKEEPING, 'CPU_only': True, 'GPU_work': False,
        'original_parent_invoked': False, 'original_refusals_waived': False,
        'root_cause_established': False, 'host_reliability_established': False,
        'optimization_adopted': False, 'research_admitted': False}


def run_workers(packet, scratch, upstream, started, *, local=False):
    packet, scratch, upstream = Path(packet), Path(scratch), Path(upstream)
    sha = digest(packet/'CPU-ORACLE-HANDOFF.json'); manifest = verify(packet, sha)
    if not math.isfinite(started) or not 0 <= time.time()-started <= 120: raise ValueError('Original launcher freshness failed.')
    if local:
        if not manifest['tiny_fixture'] or platform.node() in operator.old.HOSTS:
            raise ValueError('Local rehearsal is tiny and cannot run on a university host.')
    else:
        control = json.loads((scratch/'operator-control.json').read_text()); operator.validate_control(control)
        auth = json.loads((scratch/'resource-authorization.json').read_text())
        operator.validate_authorization(auth, digest(scratch/'operator-control.json'), control['host'])
        if sha != operator.old.HANDOFF_SHA or manifest['tiny_fixture']: raise ValueError('Production requires the original full packet.')
        plan = json.loads((scratch/'new-attended-plan.json').read_text()); validate_plan(plan, sha, manifest['prior_consumed_claims']); host_gate(plan)
        operator.validate_proof(scratch)
    upstream_gate(upstream, manifest)
    output = scratch/'results'; output.mkdir(exist_ok=False); rows = []; child = None
    deadline = started+600
    state = {'kind': KIND, 'handoff_sha256': sha, 'started_unix': started, 'deadline_unix': deadline,
        'CPU_only': True, 'GPU_work': False, 'local_tiny_rehearsal': local, 'tasks': rows,
        'complete': False, 'saved_value_checks_passed': False, 'original_parent_invoked': False,
        'original_refusals_waived': False, 'research_admitted': False}
    write_record(output/'session-v3.json', state)
    def stop(*unused): raise RuntimeError('CPU v3 runtime interrupted; no retry.')
    previous = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGHUP, signal.SIGTERM)}
    for s in previous: signal.signal(s, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.time()-5))
    try:
        for i, (threads, replay) in enumerate(TASKS):
            if deadline-time.time() < 150: raise TimeoutError('No original time for a worker plus audit/export.')
            folder = output/'workers'/str(i); folder.mkdir(parents=True)
            write_record(folder/'authorization.json', {'threads': threads, 'replay': replay,
                'handoff_sha256': sha, 'deadline_unix': time.time()+90})
            command = [sys.executable, '-B', '-u', str(packet/'scripts/check_cpu_oracle.py'),
                '--diagnostic', '--worker', '--threads', str(threads), '--replay', str(replay),
                '--worker-dir', str(folder), '--authorization-sha256', digest(folder/'authorization.json'),
                '--handoff-sha256', sha, '--upstream', str(upstream)]
            if local: command.append('--local')
            else: command += ['--plan', str(scratch/'new-attended-plan.json'), '--plan-sha256', digest(scratch/'new-attended-plan.json')]
            print('V3 fresh original CPU worker:', threads, replay, flush=True)
            with (folder/'worker.log').open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': str(threads),
                        'MKL_NUM_THREADS': str(threads), 'OPENBLAS_NUM_THREADS': str(threads),
                        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'})
                try: child.wait(timeout=90)
                finally:
                    if child.poll() is None:
                        try: os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                    child.wait()
                code = child.returncode; child = None
            worker = read_record(folder/'worker.json')
            if (worker.get('capture_complete') is not True or code not in (0, 1)
                    or (code == 0) != worker['numerical_checks_passed']):
                raise RuntimeError('Original worker capture failed; preserve files, no retry.')
            rows.append({'threads': threads, 'replay': replay, 'worker_sha256': digest(folder/'worker.json')})
            write_record(output/'session-v3.json', state, replace=True)
            # Audit immediately, before dispatching any later worker. This avoids
            # carrying forward a false producer pass after a reaggregation error.
            values, _ = load(packet/manifest['fixture']['folder'], manifest['fixture']['sha256'], case='encoder-1', tiny=local)
            independent = audit_worker(folder, rows[-1]['worker_sha256'], values, manifest['fixture'], tiny=local, handoff_sha=sha)
            write_record(folder/'parent-v3-audit.json', independent)
            if not independent['saved_value_evidence_passed']:
                state['stop_reason'] = 'saved_value_refusal'; break
        summary = review_captures(packet, output/'workers', rows, output/'v3-audit', expected_host=platform.node(), tiny=local)
        write_record(output/'summary-v3.json', summary)
        state.update(complete=True, saved_value_checks_passed=summary['saved_value_checks_passed'],
            all_planned_workers_completed=summary['all_planned_workers_completed'], summary_sha256=digest(output/'summary-v3.json'))
    except BaseException as exc:
        state.update(error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        if child is not None:
            if child.poll() is None:
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            child.wait()
        state['wall_seconds'] = time.time()-started; write_record(output/'session-v3.json', state, replace=True)
        try:
            verify(packet, sha); spec = export(packet, output)
            print('V3 evidence archive:', spec['bytes'], spec['sha256'], flush=True)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s, handler in previous.items(): signal.signal(s, handler)
    return state
