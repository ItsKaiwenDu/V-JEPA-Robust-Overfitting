"""Separate operational completion of missing controls on frozen science sources.

The old failed session is an immutable input, never a session to resume or mark
successful. Scientific attacks/calibration and cost arithmetic live in the
frozen project; this module only checks, launches, and combines their evidence.
"""

from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

BASE_MANIFEST = 'e42164317e16fbf6e05f4143fd466f00105371284fc9e769d09d5c71ba82608a'
PRIOR_ARCHIVE = 'a2c7de5655d3872d5ee625952fb11fb4ea5e31ee8f6be619feacedb0d35f39d0'
CHECKPOINT = '5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012'
CHECKPOINT_BYTES = 5127726842
PROFILE = 'a40-missing-controls-v1'
EXPORT_RESERVE = 300
OLD_PHASES = ('preflight', 'calibrate', 'features-initial',
              'clean-finetune/reference', 'adversarial-finetune/reference', 'features-trained')
CONTROLS = ('convergence', 'fgsm', 'random')
UNIT_NAMES = ('routine', 'primary', *CONTROLS, 'clean-finetune',
              'adversarial-finetune', 'engine.state', 'checkpoint.save',
              'calibration_three_primary_candidates', 'load_anchor')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    """Atomic operational record; reject non-finite metadata."""
    path = Path(path)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('x') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_wrapped(path, value, *, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
    checksum = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode()).hexdigest()
    save(path, dict(payload=value, sha256=checksum))


def safe_file(root, name):
    root = Path(root).resolve()
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or '..' in relative.parts
            or '\\' in name or str(relative) != name or name == '.'):
        raise ValueError('Unsafe input path.')
    path = root / name
    if (not path.is_file() or not path.resolve().is_relative_to(root)
            or any((root / Path(*relative.parts[:i])).is_symlink()
                   for i in range(1, len(relative.parts) + 1))):
        raise ValueError('Require a regular, contained input: ' + name)
    return path


def verify_files(root, files):
    for name, expected in files.items():
        path = safe_file(root, name)
        if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise ValueError('Input checksum/size changed: ' + name)
    return len(files)


def source_gate(base):
    base = Path(base)
    if sha(base / 'handoff-manifest.json') != BASE_MANIFEST:
        raise ValueError('Frozen scientific manifest changed.')
    spec = json.loads((base / 'handoff-manifest.json').read_text())
    return verify_files(base, spec['archives']['source']['files'])


def package_gate(package):
    package = Path(package)
    spec = json.loads((package / 'controls-manifest.json').read_text())
    if (spec['profile'] != PROFILE or spec['base_manifest_sha256'] != BASE_MANIFEST
            or spec['prior_archive_sha256'] != PRIOR_ARCHIVE
            or spec['paid_compute_authorized'] is not False):
        raise ValueError('Controls package identity differs.')
    verify_files(package, spec['files'])
    return spec


def money(value):
    if isinstance(value, bool):
        raise ValueError('Invalid monetary amount.')
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('Invalid monetary amount.') from exc
    if not result.is_finite() or result < 0:
        raise ValueError('Invalid monetary amount.')
    return result


def review_plan(plan, ledger, ledger_sha256, now, *, launching=True):
    """New deployment allowance, never the settled historical $2.80 hold."""
    if (plan.get('profile') != PROFILE or plan.get('ledger_sha256') != ledger_sha256
            or plan.get('base_manifest_sha256') != BASE_MANIFEST):
        raise ValueError('Require current ledger and frozen source identities.')
    used = money(ledger['conservative_cumulative_credit_usage_usd'])
    available = money(ledger['shared_diagnostic_allowance_usd']) - used
    if (used != money(ledger['displayed_total_debits_usd'])
            or available != money(ledger['conservative_remaining_diagnostic_allowance_usd'])
            or money(ledger['remaining_unsettled_internal_reservation_usd']) != 0
            or used + money(ledger['last_wallet_screenshot_available_credit_usd'])
               != money(ledger['initial_account_credit_usd'])):
        raise ValueError('Ledger is unreconciled; historical holds cannot be reused.')
    rate = money(plan.get('quoted_all_in_hourly_usd'))
    hold = money(plan.get('new_session_hold_usd'))
    if (not 0 < rate <= Decimal('.70') or hold != Decimal('2.00')
            or hold > min(available, money(ledger['last_wallet_screenshot_available_credit_usd']))
            or rate * 2 > hold or used + hold > money(ledger['project_cap_usd'])):
        raise ValueError('Quote or new inclusive reservation exceeds the reviewed allowance.')
    if any(plan.get(key) is not True for key in
           ('fresh_quote_confirmed', 'manual_deployment_authorized', 'host_fingerprint_verified')):
        raise ValueError('Deployment, quote and host identity must be explicitly reviewed.')
    if (plan.get('expected_gpu') != 'NVIDIA A40' or plan.get('container_gb') != 30
            or plan.get('volume_gb') != 80 or plan.get('volume_mount') != '/workspace'):
        raise ValueError('Require the reviewed A40 and 30/80 GB storage layout.')
    deploy, deadline = plan.get('deployment_epoch'), plan.get('deadline_epoch')
    if (type(deploy) is not int or type(deadline) is not int or type(now) is not int
            or deadline != deploy + 7200 or not deploy <= now < deadline - EXPORT_RESERVE
            or (launching and deadline - now < 3600)):
        raise ValueError('Use the original deployment-based two-hour deadline with setup/export time.')
    return deadline - now


def suite_gate(path, expected=509):
    suites = list(ET.parse(path).getroot().iter('testsuite'))
    if (len(suites) != 1 or int(suites[0].get('tests', -1)) != expected
            or any(suites[0].get(k) != '0' for k in ('failures', 'errors', 'skipped'))):
        raise ValueError('Fresh complete CUDA-host suite must pass with zero skips.')
    return suites[0].attrib


def same_runtime(prior, actual):
    if prior != actual:
        raise ValueError('Hardware/software/precision runtime changed; timings cannot be pooled.')


def reviewed_run_plan(package, run):
    """Every child binds the same reviewed plan and immutable absolute deadline."""
    metadata = json.loads((run / 'launcher-metadata.json').read_text())
    path = run / 'deployment-plan.json'
    if (sha(path) != metadata['deployment_plan_sha256']
            or sha(package / 'controls-manifest.json') != metadata['controls_manifest_sha256']
            or sha(__file__) != metadata['source_sha256']):
        raise ValueError('Reviewed plan or operational source changed after launch.')
    plan = json.loads(path.read_text())
    ledger_path = package / 'budget-ledger.json'
    review_plan(plan, json.loads(ledger_path.read_text()), sha(ledger_path), int(time.time()), launching=False)
    return plan


def shared_contract(contract):
    return {k: v for k, v in contract.items()
            if k not in ('checkpoint', 'weights_sha256', 'settings', 'sensitivity_banks')}


def check_evaluation(value, common, method, step, *, pretrained=False):
    """Recompute settings, draw namespaces and scoring-bank independence."""
    from dataclasses import asdict
    from models.evaluation import EvaluationSettings, evaluation_seed
    from models.evaluation_store import validate_records
    validate_records(value)
    c = value['contract']
    settings = asdict(EvaluationSettings.for_method(method, step))
    audits = 2 if method == 'convergence' else 0
    if (not value['complete'] or c['settings'] != settings
            or c['sensitivity_banks'] != audits or shared_contract(c) != shared_contract(common)
            or (pretrained and any(c[k] != common[k] for k in ('checkpoint', 'weights_sha256')))
            or c['weights_sha256']['target_encoder'] != common['weights_sha256']['target_encoder']):
        raise ValueError('Incomplete/mixed data, sources, checkpoint, runtime or attack settings.')
    for row in value['records']:
        namespaces = {'attack': (method + '_attack', settings['attack_masks']),
                      'score': ('routine_score' if method == 'routine' else 'primary_score', settings['scoring_masks'])}
        namespaces.update({f'audit_{i}': (f'primary_score_audit_{i}', settings['scoring_masks'])
                           for i in range(audits)})
        seeds = {key: [evaluation_seed(c['base_seed'], namespace, row['video_id'], draw=i)
                       for i in range(count)] for key, (namespace, count) in namespaces.items()}
        seeds['start'] = evaluation_seed(c['base_seed'], method + '_start', row['video_id'])
        draws = [n % (2**31 - 1) for key, values in seeds.items() if key != 'start' for n in values]
        if (row['seeds'] != seeds or len(set(draws)) != len(draws)
                or len(row['scoring_audits']) != audits
                or any(set(audit) != {'clean_loss', 'adversarial_loss'} for audit in row['scoring_audits'])
                or any(not math.isfinite(n) for audit in row['scoring_audits'] for n in audit.values())
                or not math.isfinite(row['seconds']) or row['seconds'] <= 0):
            raise ValueError('Invalid independent scoring banks, seeds or native timing.')
    return [row['seconds'] for row in value['records']]


def collect_prior(pilot):
    """Only declared complete phases contribute units; rejected control stays visible."""
    from models.calibration import calibration_record
    from models.schedule import EpochSchedule
    from models.study_artifacts import read_record
    pilot = Path(pilot)
    session = read_record(pilot / 'session.json')
    if (session['complete'] or session.get('error_type') != 'TimeoutExpired'
            or [p['phase'] for p in session['phases']] != list(OLD_PHASES)
            or (pilot / 'economic-screen.json').exists()):
        raise ValueError('Expected the preserved six-phase partial session, not a relabeled success.')
    calibration = read_record(pilot / 'calibration/calibration.json')
    common = calibration['common_contract']
    same_runtime(session['runtime'], {k: v for k, v in common['runtime'].items() if k != 'model_execution'})
    rows = [dict(video_id=v, source_group_id=g) for v, g in zip(common['video_ids'], common['source_group_ids'])]
    candidates = [read_record(safe_file(pilot / 'calibration', r['file']), expected_sha256=r['sha256'])
                  for r in calibration['evaluation_files']]
    if calibration_record(candidates, rows, diagnostic=True) != {k: v for k, v in calibration.items() if k != 'evaluation_files'}:
        raise ValueError('Saved calibration did not recompute.')
    observations = {name: [] for name in UNIT_NAMES}
    for phase in session['phases']:
        path = safe_file(pilot, 'phase-results/' + phase['phase'] + '.json')
        if sha(path) != phase['result_sha256']:
            raise ValueError('Completed phase result changed.')
        result = read_record(path)
        if (result['phase'] != phase['phase'] or result['provenance'] != common['provenance']
                or result['diagnostic_only'] is not True or result['research_result'] is not False):
            raise ValueError('Mixed completed phase provenance.')
        timing = safe_file(pilot, 'timings/' + phase['phase'] + '.jsonl')
        for line in timing.read_text().splitlines():
            event = json.loads(line)
            if event['status'] != 'ok' or not math.isfinite(event['seconds']) or event['seconds'] <= 0:
                raise ValueError('Failed or invalid timing span in a completed phase.')
            name = event['name']
            if name == 'engine.update':
                observations[phase['phase'].split('/')[0]].append(event['seconds'])
            elif name in observations:
                observations[name].append(event['seconds'])
    evaluation_paths = sorted((pilot / 'calibration').glob('*/evaluation.json'))
    evaluation_paths += sorted((pilot / 'evaluations-reference').glob('*/evaluation.json'))
    for path in evaluation_paths:
        value = read_record(path)
        method = value['contract']['settings']['method']
        if method not in ('routine', 'primary'):
            raise ValueError('Unexpected inherited method.')
        step = value['contract']['settings']['step_size_255']
        if method == 'primary' and path.parent.parent.name != 'calibration' and step != calibration['chosen_step_size_255']:
            raise ValueError('Scheduled primary differs from frozen calibration.')
        observations[method].extend(check_evaluation(value, common, method, step))
    # Verify shared initialization receipts once per unique file, not once per
    # condition. Two workflows deliberately reference the same initialization.
    receipts = set()
    for condition in ('clean-finetune', 'adversarial-finetune'):
        directory = pilot / condition / 'reference'
        workflow = read_record(directory / 'workflow.json')
        schedule = workflow['contract']['schedule']
        tasks = [t for t in EpochSchedule(schedule['epochs'], tuple(schedule['primary_epochs']), tuple(schedule['panels'])).tasks() if t['epoch'] <= 2]
        if [r['task'] for r in workflow['completed_evaluations']] != tasks:
            raise ValueError('Incomplete scheduled workflow receipts.')
        for row in workflow['completed_evaluations']:
            # Absolute old paths remain unchanged in the artifact. Only the
            # checksummed evaluation ID is mapped into this read-only copy.
            old = PurePosixPath(row['receipt']['path'])
            if old.parent.parent.name != 'evaluations-reference' or old.name != 'evaluation.json':
                raise ValueError('Unexpected historical receipt path.')
            path = safe_file(pilot, 'evaluations-reference/' + old.parent.name + '/evaluation.json')
            payload = read_record(path, expected_sha256=row['receipt']['sha256'])
            if not payload['complete'] or payload['contract']['checkpoint'] != row['checkpoint']:
                raise ValueError('Mixed scheduled checkpoint receipt.')
            receipts.add(path.resolve())
    if receipts != {p.resolve() for p in (pilot / 'evaluations-reference').glob('*/evaluation.json')}:
        raise ValueError('Unreferenced or duplicate historical evaluation file.')
    rejected = read_record(pilot / 'controls/convergence/evaluation.json')
    if rejected['complete'] or rejected['records']:
        raise ValueError('Historical timed-out control must retain zero completed records.')
    if (len(observations['primary']) != 6 or len(observations['routine']) != 5
            or len(observations['clean-finetune']) != 2 or len(observations['adversarial-finetune']) != 2):
        raise ValueError('Historical unique measurement counts differ.')
    return session, calibration, observations


def prior_gate(package, base):
    """Bind the compact evidence copy to the original archive and local audits."""
    source_gate(base)
    package_gate(package)
    prior = Path(package) / 'prior'
    original = json.loads((prior / 'archive-manifest.json').read_text())
    if original['sha256'] != PRIOR_ARCHIVE or len(original['files']) != 60:
        raise ValueError('Original evidence archive identity changed.')
    non_tensors = {n: r for n, r in original['files'].items() if not n.endswith('.pt')}
    verify_files(prior / 'extracted', non_tensors)
    if sha(prior / 'extracted/handoff-manifest.json') != BASE_MANIFEST:
        raise ValueError('Historical source manifest changed.')
    for name in ('preflight-verification.json', 'feature-verification.json'):
        if json.loads((prior / name).read_text()).get('passed') is not True:
            raise ValueError('Independent local verification was not accepted.')
    partial = json.loads((prior / 'partial-verification.json').read_text())
    if (partial['archive_integrity_verified'] is not True or partial['completed_phase_evidence_verified'] is not True
            or partial['complete_screen_passed'] is not False
            or partial['archive']['sha256'] != PRIOR_ARCHIVE):
        raise ValueError('Partial-screen audit changed.')
    pilot = prior / 'extracted/pilot'
    session, calibration, observations = collect_prior(pilot)
    if (session['expected_gpu'] != 'NVIDIA A40' or session['device']['gpu'] != 'NVIDIA A40'
            or session['runtime']['gpu'] != 'NVIDIA A40'):
        raise ValueError('Only the verified A40 observations are reusable.')
    return session, calibration, observations


def combine(past_pilot, current_pilot, reference, rate):
    """Produce a separate cost screen; neither session is rewritten here."""
    from models.hardware_benchmark import economic_screen, summarize
    from models.study_artifacts import read_record
    past, calibration, observations = collect_prior(past_pilot)
    current_pilot = Path(current_pilot)
    current = read_record(current_pilot / 'session.json')
    same_runtime(past['runtime'], current['runtime'])
    same_runtime(past['device'], current['device'])
    if (current['profile'] != PROFILE or current.get('error') or current['required_phases'] != ['controls']
            or [r['phase'] for r in current['phases']] != ['controls']
            or current['execution_policy'] != past['execution_policy']
            or current['quoted_hourly_usd'] != rate):
        raise ValueError('The independent controls session has not completed its required phase.')
    phase_path = current_pilot / 'phase-results/controls.json'
    if sha(phase_path) != current['phases'][0]['result_sha256']:
        raise ValueError('New control phase checksum changed.')
    phase = read_record(phase_path)
    if phase['provenance'] != calibration['common_contract']['provenance']:
        raise ValueError('New control phase has changed science/data provenance.')
    copied = current_pilot / 'calibration/calibration.json'
    if sha(copied) != sha(Path(past_pilot) / 'calibration/calibration.json'):
        raise ValueError('New controls changed the inherited calibration.')
    expected = {current_pilot / 'controls' / method / 'evaluation.json' for method in CONTROLS}
    if set((current_pilot / 'controls').rglob('evaluation.json')) != expected:
        raise ValueError('Missing, duplicate or unexpected new control evaluation.')
    for method in CONTROLS:
        observations[method] = check_evaluation(read_record(current_pilot / 'controls' / method / 'evaluation.json'),
            calibration['common_contract'], method, calibration['chosen_step_size_255'], pretrained=True)
    events = [json.loads(line) for line in (current_pilot / 'timings/controls.jsonl').read_text().splitlines()]
    if [event.get('method') for event in events if event['name'] == 'control_evaluation'] != list(CONTROLS):
        raise ValueError('Missing or duplicate completed control spans.')
    for event in events:
        if event['status'] != 'ok' or not math.isfinite(event['seconds']) or event['seconds'] <= 0:
            raise ValueError('New controls contain a failed timing span.')
    value = economic_screen(reference, observations, rate)
    value.update(profile=PROFILE, diagnostic_only=True, combined_economic_measurements_complete=True,
        historical_session_complete=False, historical_timeout_preserved=True,
        historical_session_sha256=sha(Path(past_pilot) / 'session.json'),
        controls_phase_sha256=sha(phase_path), observations=observations,
        timing_summaries={k: summarize(v) for k, v in observations.items()},
        excluded_failed_measurements=['historical controls/convergence/evaluation.json (zero completed clips)',
                                      'historical timings/controls.jsonl (interrupted phase)'],
        unknown_costs_and_research_gates_remain=True, confirmation_or_test_pixels_opened=False)
    return value


def actual_runtime(base):
    import platform
    import torch
    import torchvision
    from models.execution import configure_execution
    from models.runtime import device_metadata
    from models.training import runtime_contract
    configure_execution('shared-fp64-native', device='cuda', strict_original=True)
    torch.set_num_threads(4)
    limits = [int(p.read_text().strip()) for p in
              (Path('/sys/fs/cgroup/memory.max'), Path('/sys/fs/cgroup/memory/memory.limit_in_bytes'))
              if p.exists() and p.read_text().strip().isdigit()]
    if (platform.system() != 'Linux' or platform.machine() != 'x86_64'
            or not limits or min(limits) < 45_000_000_000
            or str(torchvision.__version__) != '0.23.0+cu128'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError('Require the matching Linux A40 CUDA stack and at least45GB host RAM.')
    device, runtime = device_metadata('cuda'), runtime_contract('cuda')
    if device['gpu'] != 'NVIDIA A40' or device['vram_bytes'] < 44 * 2**30:
        raise ValueError('The new host does not contain the reviewed A40.')
    return dict(device=device, runtime=runtime, cgroup_ram_limit_bytes=min(limits),
                torchvision=str(torchvision.__version__))


def export_evidence(base, package, run):
    """The partial-failure path uses only the standard library."""
    from models.pilot_bundle import make_archive
    files = {}
    for prefix, directory in (('run', run), ('setup', base / 'runs/pretrained/diagnostic/workflow-setup')):
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.xml', '.txt', '.log'):
                if path.is_symlink() or path.stat().st_size > 10 * 2**20:
                    raise ValueError('Unexpected export input.')
                files[prefix + '/' + str(path.relative_to(directory))] = path
    spec = package_gate(package)
    files['controls-manifest.json'] = package / 'controls-manifest.json'
    # Carry the immutable compact prior and operational sources so an export is
    # sufficient to audit composition without reconnecting to either old Pod.
    for name in spec['files']:
        files['package/' + name] = package / name
    manifest = make_archive(run / 'evidence.tar.gz', files)
    save(run / 'evidence.tar.gz.manifest.json', manifest)
    return manifest


def worker(base, package, run, plan):
    from scripts.run_workflow_pilot import run_child
    remaining = lambda: plan['deadline_epoch'] - time.time()
    pilot = run / 'pilot'
    status, stage = 1, 'setup'
    session = dict(schema_version=1, profile=PROFILE, diagnostic_only=True, research_result=False,
        research_admitted=False, budget_approved=False, complete=False, required_phases=['controls'], phases=[],
        execution_policy='shared-fp64-native', historical_session_complete=False,
        inherited_completed_phases=list(OLD_PHASES), quoted_hourly_usd=float(plan['quoted_all_in_hourly_usd']),
        expected_gpu='NVIDIA A40', deadline_epoch=plan['deadline_epoch'],
        cloud_billing_stopped_by_runner=False, exact_optimizer_recovery_checked=False)
    write_wrapped(pilot / 'session.json', session)
    try:
        print('Stage: frozen full-suite setup (no measurement phases)', flush=True)
        os.environ['VJEPA_BENCHMARK_GPU'] = 'NVIDIA A40'
        run_child(['bash', 'scripts/setup_hardware_benchmark.sh'], run / 'setup-supervisor.log',
                  min(2400, remaining() - EXPORT_RESERVE - 1200))
        suite = suite_gate(base / 'runs/pretrained/diagnostic/workflow-setup/tests.xml')
        stage = 'runtime-and-input-gate'
        command = [str(base / '.venv/bin/python'), str(package / 'operations/scripts/run_hardware_controls.py'),
                   '--mode', 'gate', '--base', str(base), '--package', str(package), '--run', str(run)]
        run_child(command, run / 'runtime-gate.log', min(300, remaining() - EXPORT_RESERVE - 900))
        gate = json.loads((run / 'runtime-gate.json').read_text())
        session.update(device=gate['device'], runtime=gate['runtime'], fresh_full_suite=suite,
                       source_sha256=gate['source_sha256'], deployment_plan=plan)
        write_wrapped(pilot / 'session.json', session, replace=True)
        prior_calibration = package / 'prior/extracted/pilot/calibration'
        shutil.copytree(prior_calibration, pilot / 'calibration')
        stage = 'controls'
        print('Stage: missing PGD50 / FGSM / random controls only', flush=True)
        command = [str(base / '.venv/bin/python'), '-u', 'scripts/run_hardware_benchmark.py', '--diagnostic',
            '--phase', 'controls', '--expected-gpu', 'NVIDIA A40', '--hourly-rate-usd',
            str(plan['quoted_all_in_hourly_usd']), '--deadline-epoch', str(plan['deadline_epoch']),
            '--checkpoint', 'checkpoints/vitl.pt', '--upstream', 'external/vjepa2',
            '--manifests', 'data/manifests/ucf101-staged-20x5-v1', '--clips',
            'data/processed/staged-exploration-v1', '--output', str(pilot)]
        before = time.monotonic()
        run_child(command, run / 'controls.log', remaining() - EXPORT_RESERVE)
        session['phases'] = [dict(phase='controls', process_wall_seconds=time.monotonic() - before,
                                 result_sha256=sha(pilot / 'phase-results/controls.json'))]
        write_wrapped(pilot / 'session.json', session, replace=True)
        stage = 'combine'
        command = [str(base / '.venv/bin/python'), str(package / 'operations/scripts/run_hardware_controls.py'),
                   '--mode', 'combine', '--base', str(base), '--package', str(package), '--run', str(run)]
        run_child(command, run / 'combine.log', min(120, remaining() - 180))
        session.update(complete=True, combined_screen_sha256=sha(pilot / 'combined-economic-screen.json'))
        status = 0
    except BaseException as exc:
        session.update(error_type=type(exc).__name__, error=str(exc), failed_stage=stage)
        print('Controls attempt failed:', stage, type(exc).__name__, str(exc), flush=True)
    finally:
        session.update(worker_exit_status=status, finished_epoch=time.time())
        write_wrapped(pilot / 'session.json', session, replace=True)
        save(run / 'exit-status.json', dict(status=status, failed_stage=None if status == 0 else stage))
        try:
            # Export in a separately bounded process; no scientific imports are
            # needed when setup failed before the virtual environment existed.
            run_child([sys.executable, str(package / 'operations/scripts/run_hardware_controls.py'),
                '--mode', 'export', '--base', str(base), '--package', str(package), '--run', str(run)],
                run / 'export.log', min(120, max(1, remaining() - 20)))
        except BaseException as exc:
            print('Export incomplete; preserve volume:', str(exc), flush=True)
        print('Exit status:', status, '; copy/verify export, then user Stop/Terminate to end billing.', flush=True)
    return status


def dispatch(args):
    base, package = args.base.resolve(), args.package.resolve()
    os.chdir(base)
    sys.path.insert(0, str(base))
    spec = package_gate(package)
    if sha(__file__) != spec['files']['operations/models/hardware_controls.py']['sha256']:
        raise ValueError('Executed operational source differs from the package.')
    if args.mode == 'offline':
        session, calibration, observations = prior_gate(package, base)
        print(json.dumps(dict(offline_only=True, paid_job_started=False, inherited_phases=list(OLD_PHASES),
            historical_timeout_preserved=not session['complete'], chosen_step_size_255=calibration['chosen_step_size_255'],
            observation_counts={k: len(v) for k, v in observations.items()}), indent=2))
        return 0
    if args.mode == 'export':
        print(json.dumps(export_evidence(base, package, args.run.resolve()), indent=2)); return 0
    if args.mode == 'gate':
        reviewed_run_plan(package, args.run)
        past, _, _ = prior_gate(package, base)
        runtime = actual_runtime(base)
        same_runtime(past['runtime'], runtime['runtime'])
        same_runtime(past['device'], runtime['device'])
        checkpoint = safe_file(base, 'checkpoints/vitl.pt')
        if checkpoint.stat().st_size != CHECKPOINT_BYTES or sha(checkpoint) != CHECKPOINT:
            raise ValueError('Released model differs.')
        runtime.update(source_sha256={name: sha(base / name) for name in past['source_sha256']},
                       frozen_manifest_sha256=BASE_MANIFEST, checkpoint_sha256=CHECKPOINT)
        save(args.run / 'runtime-gate.json', runtime); return 0
    if args.mode == 'combine':
        plan = reviewed_run_plan(package, args.run)
        prior_gate(package, base)
        from models.study_artifacts import write_record
        reference_path = base / 'report/checks/2026-10-22-measured-budget-review.json'
        result = combine(package / 'prior/extracted/pilot', args.run / 'pilot',
                         json.loads(reference_path.read_text()), float(plan['quoted_all_in_hourly_usd']))
        result.update(reference_sha256=sha(reference_path), ledger_sha256=plan['ledger_sha256'],
                      controls_manifest_sha256=sha(package / 'controls-manifest.json'))
        write_record(args.run / 'pilot/combined-economic-screen.json', result); return 0
    if args.plan is None:
        raise ValueError('Supply reviewed deployment metadata; the template cannot launch.')
    plan = json.loads(args.plan.read_text())
    ledger_path = package / 'budget-ledger.json'
    seconds = review_plan(plan, json.loads(ledger_path.read_text()), sha(ledger_path), int(time.time()),
                          launching=args.mode == 'launch')
    source_gate(base)
    if args.mode == 'worker':
        if args.plan.resolve() != (args.run / 'deployment-plan.json').resolve():
            raise ValueError('Worker must use the recorded deployment plan.')
        reviewed_run_plan(package, args.run)
        return worker(base, package, args.run.resolve(), plan)
    if not shutil.which('timeout'):
        raise ValueError('A finite external timeout supervisor is required.')
    parent = base / 'runs/pretrained/diagnostic'
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / ('controls-claimed-' + str(plan['deployment_epoch']) + '.json')).open('x') as handle:
        json.dump(plan, handle)
    run = Path(tempfile.mkdtemp(prefix='hardware-controls.', dir=parent))
    (run / 'pilot').mkdir()
    save(run / 'deployment-plan.json', plan)
    save(run / 'launcher-metadata.json', dict(profile=PROFILE, source_sha256=sha(__file__),
        deployment_plan_sha256=sha(run / 'deployment-plan.json'),
        frozen_science_manifest_sha256=BASE_MANIFEST, controls_manifest_sha256=sha(package / 'controls-manifest.json'),
        deadline_epoch=plan['deadline_epoch'], export_reserve_seconds=EXPORT_RESERVE,
        maximum_inclusive_session_usd='2.00', new_session_hold_usd='2.00',
        cloud_billing_stopped_automatically=False, old_session_holds_reused=False))
    command = ['timeout', '--signal=TERM', '--kill-after=15s', str(seconds), sys.executable,
        str(package / 'operations/scripts/run_hardware_controls.py'), '--mode', 'worker', '--base', str(base),
        '--package', str(package), '--plan', str(run / 'deployment-plan.json'), '--run', str(run)]
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'PYTHONPATH': ''}
    with (run / 'launcher.log').open('x') as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    save(run / 'launcher-process.json', dict(pid=process.pid, command=command))
    print(json.dumps(dict(run_directory=str(run), supervisor_pid=process.pid,
                         deadline_epoch=plan['deadline_epoch'], billing_stopped_automatically=False), indent=2))
    return 0
