"""Standalone setup/export operator for one explicitly authorized RTX 3090 diagnostic."""
import argparse
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

PACKET = 'bf2b124b9fff819cc9f1b3acbc095c208c8a4df1293cda24267b802edd5d0b8a'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    tmp = path.with_name('.' + path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def validate(plan, ledger, manifest, now):
    if (plan.get('authorized') is not True or plan.get('diagnostic_only') is not True
            or plan.get('kind') != 'candidate_eot8_benchmark_launch_v1'
            or plan.get('expected_gpu') != 'NVIDIA GeForce RTX 3090'
            or not isinstance(plan.get('expected_gpu_uuid'), str)
            or not plan['expected_gpu_uuid'].startswith('GPU-')
            or plan.get('candidate_binding') != manifest['candidate']['candidate_binding']
            or plan.get('required_phases') != manifest['candidate']['required_phases']
            or plan.get('execution_policy') != 'shared-fp64-native'
            or (plan.get('container_gb'), plan.get('persistent_gb'), plan.get('minimum_system_ram_gib')) != (30, 80, 40)
            or plan.get('volume_mount') != '/workspace'
            or ledger.get('reconciled') is not True or ledger.get('all_prior_pods_terminated') is not True
            or (ledger.get('total_cap_usd'), ledger.get('compute_cap_usd'), ledger.get('pilot_cap_usd'), ledger.get('noncompute_reserve_usd')) != (200, 160, 10, 40)):
        raise ValueError('Missing authorization, hardware binding or original contract.')
    rate = Decimal(str(plan['total_hourly_usd'])); allowance = Decimal(str(plan['inclusive_session_budget_usd']))
    prior = Decimal(str(ledger['pilot_spent_and_other_holds_usd']))
    if (not all(v.is_finite() for v in (rate, allowance, prior)) or rate != Decimal('.52')
            or not 0 < allowance <= Decimal('1.30') or prior < Decimal('8.6788')
            or allowance + prior > 10):
        raise ValueError('Fresh reviewed quote/allowance/other holds do not fit.')
    start, deadline = plan['deployed_unix'], plan['deadline_unix']
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, deadline, now))
            or not start <= now < deadline - 180 or not 300 < deadline - start <= 7200
            or Decimal(str(deadline - start)) * rate / 3600 > allowance):
        raise ValueError('Invalid or expired deployment-inclusive deadline.')
    return deadline - now


def remaining(deadline, monotonic_end, now, tick, cap):
    value = min(deadline - now - 180, monotonic_end - tick - 180, cap)
    if value <= 0:
        raise TimeoutError('Original deployment-inclusive deadline reached.')
    return value


def command(base, label):
    p = str(base / 'env/bin/python')
    code = base / 'code'; setup = base / 'setup'
    choices = {
        'restore': [sys.executable, '-B', str(base / 'package/restore.py'), '--package', str(base / 'package'), '--destination', str(code)],
        'venv': [sys.executable, '-m', 'venv', '--system-site-packages', str(base / 'env')],
        'dependencies': [p, '-m', 'pip', 'install', '--no-index', '--no-deps', *map(str, sorted((code / 'offline-wheels').glob('*.whl')))],
        'checkpoint-download': ['curl', '--fail', '--location', '--retry', '2', '--max-time', '900', '--output', str(base / 'assets/vitl.pt'), 'https://dl.fbaipublicfiles.com/vjepa2/vitl.pt'],
        'cpu-fixtures': [p, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_candidate_evaluation.py', 'tests/test_candidate_schedule.py', 'tests/test_candidate_benchmark.py', '--basetemp=' + str(base / 'cpu-fixtures-temp'), '--junitxml=' + str(setup / 'cpu-fixtures.xml')],
    }
    return choices[label]


def run_command(argv, log, seconds, cwd):
    with log.open('x') as f:
        child = subprocess.Popen(argv, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'})
        try:
            code = child.wait(timeout=seconds)
            if code:
                raise subprocess.CalledProcessError(code, argv)
        finally:
            # This session owns this process group; never touch unrelated jobs.
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            child.wait()


def export(base, manifest):
    package = base / 'package'
    if sha(package / 'manifest.json') != PACKET or sha(package / 'restore.py') != manifest['restore_sha256']:
        raise ValueError('Changed package before export.')
    spec = importlib.util.spec_from_file_location('verified_export', package / 'restore.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    files = {'package-manifest.json': package / 'manifest.json'}
    for prefix in ('setup', 'run'):
        for path in (base / prefix).rglob('*'):
            if path.is_file() and (path.suffix in ('.json', '.jsonl', '.xml', '.txt', '.log', '.py')
                    or path.suffix == '.pt' and path.parent.name == 'preflight'):
                if path.is_symlink(): raise ValueError('Export refuses symlinks.')
                files[prefix + '/' + str(path.relative_to(base / prefix))] = path
    if (base / 'code').exists():
        for name, item in manifest['archives']['source']['files'].items():
            path = base / 'code' / name
            if sha(path) != item['sha256']: raise ValueError('Source changed before export: ' + name)
            files['source/' + name] = path
    result = helper.make_archive(base / 'evidence.tar.gz', files)
    save(base / 'evidence.tar.gz.manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args(); base = args.base.resolve()
    if not base.is_relative_to('/workspace') or base == Path('/workspace'):
        parser.error('Require a new session below /workspace.')
    setup = base / 'setup'; package = base / 'package'
    if sha(setup / 'launch.json') != args.plan_sha256 or sha(package / 'manifest.json') != PACKET:
        parser.error('Changed reviewed plan or packet.')
    plan = json.loads((setup / 'launch.json').read_text()); ledger = json.loads((setup / 'budget-envelope.json').read_text())
    if sha(setup / 'budget-envelope.json') != plan['ledger_sha256']: parser.error('Changed ledger.')
    manifest = json.loads((package / 'manifest.json').read_text())
    seconds = validate(plan, ledger, manifest, time.time()); end = time.monotonic() + seconds
    for name in ('code', 'env', 'run', 'assets', 'setup/operator-claim.json'):
        if (base / name).exists(): parser.error('No retry or reuse of an existing attempt.')
    with (setup / 'operator-claim.json').open('x') as f: json.dump({'plan_sha256': args.plan_sha256}, f)
    status = 1; state = {'complete': False, 'benchmark_started': False, 'steps': [], 'deadline_unix': plan['deadline_unix'], 'deployed_unix': plan['deployed_unix']}
    def alarm(*_): raise TimeoutError('Independent setup/export deadline reached.')
    signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, seconds - 60)
    try:
        for label, cap in [('restore', 300), ('venv', 120), ('dependencies', 300), ('checkpoint-download', 930), ('cpu-fixtures', 900), ('benchmark', 7200)]:
            state['current_stage'] = label; save(setup / 'operator-state.json', state)
            if label == 'checkpoint-download': (base / 'assets').mkdir()
            if label == 'benchmark':
                state['benchmark_started'] = True
                argv = [str(base / 'env/bin/python'), '-B', '-u', str(base / 'code/scripts/run_candidate_benchmark.py'), '--plan', str(setup / 'launch.json'), '--plan-sha256', args.plan_sha256, '--ledger', str(setup / 'budget-envelope.json'), '--checkpoint', str(base / 'assets/vitl.pt'), '--output', str(base / 'run')]
            else: argv = command(base, label)
            print('Stage: ' + label, flush=True)
            before = time.monotonic()
            run_command(argv, setup / (label + '.log'), remaining(plan['deadline_unix'], end, time.time(), time.monotonic(), cap), base / 'code' if label in ('cpu-fixtures', 'benchmark') else base)
            state['steps'].append({'stage': label, 'seconds': time.monotonic() - before})
            if label == 'checkpoint-download' and (sha(base / 'assets/vitl.pt') != manifest['checkpoint_sha256'] or (base / 'assets/vitl.pt').stat().st_size != manifest['checkpoint_bytes']):
                raise ValueError('Released checkpoint differs.')
            if label == 'cpu-fixtures':
                cases = ET.parse(setup / 'cpu-fixtures.xml').getroot().findall('.//testcase')
                if len(cases) != 77 or any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped')): raise ValueError('Need all 77 CPU checks.')
        state['complete'] = True; status = 0
    except BaseException as exc:
        state.update(error_type=type(exc).__name__, error=str(exc))
        print('Stopped: ' + repr(exc), flush=True)
    finally:
        state.update(exit_status=status, ended_unix=time.time()); save(setup / 'operator-state.json', state)
        try:
            result = export(base, manifest); print('Evidence: ' + json.dumps({'bytes': result['bytes'], 'sha256': result['sha256']}), flush=True)
        except BaseException as exc:
            print('Preserve workspace; export incomplete: ' + repr(exc), flush=True); status = status or 2
        (base / 'exit-status.txt').write_text(str(status) + '\n')
        signal.setitimer(signal.ITIMER_REAL, 0)
    return status


if __name__ == '__main__': sys.exit(main())
