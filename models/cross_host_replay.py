"""Isolated two-process reference replay; never writes a frozen panel journal.

The unchanged numerical worker is reused with an explicit candidate execution
identity. Its retained records are shadow evidence, never a complete v1 case.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import time

KIND = 'cross-host-reference-shadow-v1'
PLAN_KIND = 'cross-host-reference-shadow-plan-v1'
HOST = 'linux10603'
GPU = 'NVIDIA GeForce RTX 3070, GPU-e54fae90-feeb-8106-d8c0-1974fe48555f, 8192, 580.178.04'
REFERENCE_HOST = 'linux10609'
REFERENCE_GPU = 'NVIDIA GeForce RTX 3070, GPU-4e2a7c77-19bd-689f-6c08-60eb774c0d52, 8192, 580.178.04'
REFERENCE_PLAN = 'f6b81dec841aeee703413dba23ea16afe3dbf14736edecadc0f7830f3ded34ae'
MANIFEST = '73f361fc072a053439bf71f35835bb91227932edd2db739adde2db5f96c6b0f4'
PACKAGE = 'Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024/unpacked/matched-panel-20261024'
FOLDER = 'cross-host-replay-10603-20261025'
PACKET = 'Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024/' + FOLDER + '/unpacked/' + FOLDER
REFERENCE_CASE = 'clip-02-bank-1'
WORKER_FILES = ('contract.json', 'inputs.json', 'inputs.pt', 'A.json', 'A.pt', 'worker.json', 'observation.json')
SOURCES = ('models/cross_host_replay.py', 'scripts/check_cross_host_replay.py',
           'tests/test_cross_host_replay.py', 'tests/fixtures/cross_host_reference_records.json')
LIMIT = 1200
WORKER_LIMIT = 510
EXPORT_LIMIT = 256 * 2**20


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_record(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or set(value) != {'payload', 'sha256'} or canonical(value['payload']) != value['sha256']:
        raise ValueError('Record checksum mismatch: ' + str(path))
    return value['payload']


def write_record(path, payload, *, replace=False):
    path = Path(path)
    body = json.dumps({'payload': payload, 'sha256': canonical(payload)}, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if not replace:
        with path.open('x') as stream:
            stream.write(body)
    else:
        temporary = path.with_name(path.name + '.pending')
        with temporary.open('x') as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def bound(root, name):
    root = Path(root).resolve()
    relative = PurePosixPath(name)
    path = root / name
    if (not name or relative.is_absolute() or '..' in relative.parts or '\\' in name or str(relative) != name
            or name == '.' or not path.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in [path, *list(path.parents)[:len(relative.parts)-1]])):
        raise ValueError('Unsafe shadow evidence path: ' + name)
    return path


def validate_plan(plan, expected):
    if not isinstance(expected, str) or re.fullmatch('[a-f0-9]{64}', expected) is None or canonical(plan) != expected:
        raise ValueError('Shadow plan differs from the externally pinned request.')
    policy = {'kind': PLAN_KIND, 'candidate_host': HOST, 'candidate_GPU_identity': GPU,
              'reference_host': REFERENCE_HOST, 'reference_GPU_identity': REFERENCE_GPU,
              'reference_plan_sha256': REFERENCE_PLAN, 'handoff_manifest_sha256': MANIFEST,
              'reference_case_id': REFERENCE_CASE, 'arm': 'A', 'fresh_workers': 2,
              'whole_attempt_seconds': LIMIT, 'worker_seconds': WORKER_LIMIT, 'kill_grace_seconds': 15,
              'max_attempts': 1, 'automatic_GPU_retries': 0, 'panel_admitted': False,
              'remaining_queue': False, 'research_admitted': False, 'paid_compute': False}
    if any(k not in plan or canonical(plan[k]) != canonical(v) for k, v in policy.items()):
        raise ValueError('Use only the separately declared bounded two-worker shadow replay.')
    if set(plan.get('shadow_sources_sha256', {})) != set(SOURCES):
        raise ValueError('Missing shadow source pins.')
    for pins in (plan['shadow_sources_sha256'], plan.get('reference_worker_files_sha256', {})):
        if any(not isinstance(v, str) or re.fullmatch('[a-f0-9]{64}', v) is None for v in pins.values()):
            raise ValueError('Invalid evidence/source pin.')
    if set(plan['reference_worker_files_sha256']) != set(WORKER_FILES):
        raise ValueError('Missing reference replay inventory.')
    execution = plan['reference_execution']
    if (execution['expected_host'] != REFERENCE_HOST or execution['expected_GPU_identity'] != REFERENCE_GPU
            or execution['plan_sha256'] != REFERENCE_PLAN or execution['research_admitted'] is not False):
        raise ValueError('Wrong independently accepted reference execution.')
    if plan['reference_case_directory'] != 'runs/panel.qIXeDn/cases/clip-02-bank-1/attempt-01':
        raise ValueError('Reference case selection changed.')
    return plan


def candidate_execution(plan):
    execution = copy.deepcopy(plan['reference_execution'])
    execution.update(kind='cross-host-shadow-execution-v1', expected_host=HOST, expected_GPU_identity=GPU,
        scope=KIND, panel_admitted=False, reference_execution_sha256=canonical(plan['reference_execution']),
        shadow_plan_sha256=canonical(plan), shadow_operator_sources_sha256=plan['shadow_sources_sha256'])
    if canonical(execution) == canonical(plan['reference_execution']):
        raise ValueError('Candidate must have its own truthful execution identity.')
    return execution


def validate_hardware(host, identity, active, *, allowed_pid=None):
    if host != HOST or identity.strip() != GPU:
        raise ValueError('Candidate host/GPU UUID/driver differs from the shadow plan.')
    pids = [int(p.strip()) for p in active.splitlines() if p.strip()]
    allowed = set() if allowed_pid is None else {allowed_pid}
    if any(pid not in allowed for pid in pids):
        raise ValueError('Another CUDA compute process is active on the candidate GPU.')


def hardware_gate(*, allow_self=False):
    identity = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                    text=True, timeout=10)
    validate_hardware(platform.node(), identity, active, allowed_pid=os.getpid() if allow_self else None)
    return identity


def validate_locations(package, packet, home):
    if (Path(package).resolve() != (Path(home) / PACKAGE).resolve()
            or Path(packet).resolve() != (Path(home) / PACKET).resolve()):
        raise ValueError('Use the fixed shared package and candidate operation; no copied-root attempt reset.')


def reserve_worker(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    claim = output.parent / 'claims' / output.name
    claim.mkdir(parents=True, exist_ok=False)


def compare_records(reference_contract, observed_contract, reference_arm, observed_arm, execution):
    expected = copy.deepcopy(reference_contract)
    expected['execution_sha256'] = canonical(execution)
    if canonical(expected) != canonical(observed_contract):
        raise ValueError('Reference pixels/targets/starts/masks/weights/runtime/settings differ.')
    omitted = {'contract_sha256', 'artifact', 'instrumented_attack_seconds'}
    if canonical({k: v for k, v in reference_arm.items() if k not in omitted}) != canonical({k: v for k, v in observed_arm.items() if k not in omitted}):
        raise ValueError('Reference PGD trace, gradients, selection or scoring is not exact.')
    if canonical(reference_arm['artifact']['tensors']) != canonical(observed_arm['artifact']['tensors']):
        raise ValueError('Reference adversarial tensor hashes differ.')
    return True


def check_sources(package, packet, plan, pin):
    validate_plan(plan, pin)
    package, packet = Path(package).resolve(), Path(packet).resolve()
    for name, expected in plan['shadow_sources_sha256'].items():
        if digest(bound(packet, name)) != expected:
            raise ValueError('Shadow source changed: ' + name)
    if digest(package / 'PANEL-HANDOFF.json') != MANIFEST:
        raise ValueError('Reference handoff changed.')
    sys.path.insert(0, str(package))
    from models.matched_panel_io import load_plan, execution_identity
    original_plan = load_plan(package)
    original_execution = execution_identity(package, original_plan)
    if canonical(original_execution) != canonical(plan['reference_execution']):
        raise ValueError('Reference source/runtime/weight identity changed.')
    if digest(packet / 'reference-review.json') != plan['reference_review_file_sha256']:
        raise ValueError('Independent reference acceptance changed.')
    acceptance = read_record(packet / 'reference-review.json')
    if (acceptance['case_id'] != REFERENCE_CASE or acceptance['completed_case_count'] != 6
            or not acceptance['reviewed_export'] or acceptance['research_admitted'] is not False
            or acceptance['case_receipt_file_sha256'] != plan['reference_receipt_file_sha256']):
        raise ValueError('Reference lacks its independently accepted six-case boundary.')
    return original_plan, candidate_execution(plan)


def reference_gate(case_root, original_plan, plan):
    from models.matched_panel_case import verify_worker
    case_root = Path(case_root)
    if digest(case_root / 'receipt.json') != plan['reference_receipt_file_sha256']:
        raise ValueError('Accepted reference case receipt changed.')
    for name, pin in plan['reference_worker_files_sha256'].items():
        if digest(bound(case_root / 'replay', name)) != pin:
            raise ValueError('Accepted reference replay changed: ' + name)
    return verify_worker(case_root / 'replay', plan=original_plan, case_id=REFERENCE_CASE,
                         execution=plan['reference_execution'], role='replay')


def worker(package, packet, plan, pin, index):
    if type(index) is not int or index not in (0, 1):
        raise ValueError('Only the two declared fresh workers may run.')
    original_plan, execution = check_sources(package, packet, plan, pin)
    validate_locations(package, packet, Path.home())
    output = Path(packet) / 'runs/attempt-01' / ('worker-' + str(index))
    session = read_record(output.parent / 'session.json')
    if session['shadow_plan_sha256'] != pin or session['active_worker'] != index or session['complete']:
        raise ValueError('Worker is not owned by the bounded shadow supervisor.')
    reserve_worker(output)
    hardware_gate()
    import numpy as np
    import torch
    from models.evaluation import EvaluationMaskSampler
    from models.execution import apply_model_policy, configure_execution
    from models.matched_panel_case import KIND as NUMERICAL_KIND, run_worker
    from models.matched_panel_io import load_clip
    from models.precision_diagnostic import native_probe
    from models.runtime import MemoryMonitor
    from models.training import runtime_contract
    from models.vjepa2 import build_vitl, load_checkpoint
    expected = execution['expected_runtime']
    configure_execution('shared-fp64-native', device='cuda')
    torch.set_num_threads(expected['threads'])
    if (platform.python_version() != expected['python'] or str(torch.__version__) != expected['torch']
            or np.__version__ != expected['numpy'] or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError('Candidate CUDA access or runtime differs from the accepted reference.')
    total = torch.cuda.get_device_properties(0).total_memory
    limit = min(7 * 2**30, int(.9 * total))
    torch.cuda.set_per_process_memory_fraction(min(.9, 7 * 2**30 / total))
    print('Shadow worker', index, ': native CUDA probe, then original checkpoint and arm A', flush=True)
    probe = native_probe('cuda')
    hardware_gate(allow_self=True)
    torch.manual_seed(20261024)
    clean = load_clip(package, original_plan, REFERENCE_CASE)
    model = build_vitl(Path(package) / 'external/vjepa2', activation_checkpointing=True)
    checkpoint = Path.home() / 'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting/checkpoints/vitl.pt'
    load_checkpoint(model, checkpoint, execution['checkpoint_sha256'])
    apply_model_policy(model, 'shared-fp64-native')
    model.to('cuda')
    if canonical(runtime_contract('cuda', model=model)) != canonical(expected):
        raise ValueError('Actual candidate model/runtime/precision differs from reference.')
    with MemoryMonitor('cuda') as memory:
        run_worker(model, clean.to('cuda'), EvaluationMaskSampler(Path(package) / 'external/vjepa2'),
                   plan=original_plan, case_id=REFERENCE_CASE, execution=execution, output=output, role='replay')
    write_record(output / 'observation.json', {'kind': NUMERICAL_KIND, 'case_id': REFERENCE_CASE,
        'role': 'replay', 'execution_sha256': canonical(execution), 'actual_host': platform.node(),
        'actual_GPU_identity': hardware_gate(allow_self=True), 'memory': memory.report(),
        'allocator_limit_bytes': limit, 'native_probe': probe, 'research_admitted': False,
        'A40_cost_pool_eligible': False, 'shadow_scope': KIND, 'panel_admitted': False})


def verify_results(package, packet, plan, pin, output, reference_case):
    import torch
    from models.matched_attack import _load_tensors
    from models.matched_panel_case import verify_worker
    original_plan, execution = check_sources(package, packet, plan, pin)
    ref_contract, ref_rows, _ = reference_gate(reference_case, original_plan, plan)
    reference_inputs = _load_tensors(Path(reference_case) / 'replay', read_record(Path(reference_case) / 'replay/inputs.json')['artifact'])
    reference_adv = _load_tensors(Path(reference_case) / 'replay', ref_rows['A']['artifact'])['adversarial']
    results = []
    for index in (0, 1):
        directory = Path(output) / ('worker-' + str(index))
        contract, rows, _ = verify_worker(directory, plan=original_plan, case_id=REFERENCE_CASE,
                                          execution=execution, role='replay')
        compare_records(ref_contract, contract, ref_rows['A'], rows['A'], execution)
        observed = _load_tensors(directory, read_record(directory / 'inputs.json')['artifact'])
        if set(observed) != set(reference_inputs) or any(not torch.equal(observed[n], reference_inputs[n]) for n in observed):
            raise ValueError('Retained reference inputs/targets/starts/masks are not bitwise equal.')
        adv = _load_tensors(directory, rows['A']['artifact'])['adversarial']
        if not torch.equal(adv, reference_adv):
            raise ValueError('Retained reference adversary is not bitwise equal.')
        observation = read_record(directory / 'observation.json')
        if observation.get('shadow_scope') != KIND or observation.get('panel_admitted') is not False:
            raise ValueError('Candidate evidence is not explicitly shadow-only.')
        results.append({'worker': index, 'reference_exact': True,
            'logical_tensors_verified': len(observed) + 1,
            'finite_nonzero_input_gradients_verified': sum(bool(r['with_grad']) for r in rows['A']['trace']),
            'actual_host': observation['actual_host'], 'actual_GPU_identity': observation['actual_GPU_identity'],
            'peak_allocated_bytes': observation['memory']['cuda_peak_allocated_bytes'],
            'native_CUDA_probe_verified': True})
    return {'kind': KIND + '-comparison', 'shadow_plan_sha256': pin, 'passed': True,
            'reference_case_id': REFERENCE_CASE, 'reference_execution_sha256': canonical(plan['reference_execution']),
            'candidate_execution_sha256': canonical(execution), 'workers': results,
            'candidate_fresh_replay_exact': True, 'scope': 'one training clip, PGD20/EOT4 arm A only',
            'other_clips_or_arms_qualified': False, 'panel_admitted': False, 'research_admitted': False,
            'parallel_dispatch_enabled': False, 'scientific_aggregate': None}


def reserve_output(packet):
    output = Path(packet) / 'runs/attempt-01'
    output.mkdir(parents=True, exist_ok=False)
    return output


def remaining_worker_seconds(started, now):
    remaining = LIMIT - (now - started) - 60
    if not math.isfinite(remaining) or remaining < 1:
        raise TimeoutError('No time remains in the original shadow allowance.')
    return min(WORKER_LIMIT, remaining)


def validate_session(session, pin):
    task = {'complete': True, 'exit_status': 0, 'timed_out': False}
    if (session.get('kind') != KIND + '-session' or session.get('shadow_plan_sha256') != pin
            or session.get('complete') is not True or session.get('passed') is not True
            or session.get('active_worker') is not None or canonical(session.get('tasks')) != canonical({'0': task, '1': task})
            or session.get('panel_admitted') is not False or session.get('research_admitted') is not False
            or session.get('whole_attempt_seconds') != LIMIT
            or type(session.get('elapsed_seconds')) not in (int, float)
            or not math.isfinite(session['elapsed_seconds']) or not 0 < session['elapsed_seconds'] <= LIMIT):
        raise ValueError('Missing, failed, mixed or over-limit shadow completion.')


def export_evidence(packet, output, reference_case, plan):
    from models.pilot_bundle import make_archive
    files = {'shadow-plan.json': Path(packet) / 'shadow-plan.json',
             'reference-review.json': Path(packet) / 'reference-review.json',
             'reference/receipt.json': Path(reference_case) / 'receipt.json'}
    for name in SOURCES:
        files['shadow-source/' + name] = Path(packet) / name
    for name in ('session.json', 'candidate-execution.json', 'comparison.json', 'worker-0.log', 'worker-1.log'):
        path = Path(output) / name
        if path.is_file(): files[name] = path
    for index in (0, 1):
        for name in WORKER_FILES:
            path = Path(output) / ('worker-' + str(index)) / name
            if path.is_file(): files['worker-' + str(index) + '/' + name] = path
    for name in WORKER_FILES:
        files['reference/replay/' + name] = Path(reference_case) / 'replay' / name
    if any(p.is_symlink() or p.stat().st_size > 65 * 2**20 for p in files.values()) or sum(p.stat().st_size for p in files.values()) > EXPORT_LIMIT:
        raise ValueError('Shadow export exceeds its fixed regular-file bound.')
    archive = Path(output) / 'evidence.tar.gz'
    manifest = make_archive(archive, files)
    with Path(str(archive) + '.manifest.json').open('x') as stream:
        stream.write(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    print('Shadow export:', json.dumps({'bytes': manifest['bytes'], 'sha256': manifest['sha256'], 'members': len(files)}), flush=True)
    return manifest


def review_export(package, packet, plan, pin, archive, manifest, expected_sha, output):
    check_sources(package, packet, plan, pin)
    from models.matched_panel_io import verify_archive
    from models.pilot_bundle import extract_verified
    verify_archive(archive, manifest, expected_sha)
    extract_verified(archive, manifest, output)
    output = Path(output)
    if canonical(json.loads((output / 'shadow-plan.json').read_text())) != pin:
        raise ValueError('Exported shadow request changed.')
    if digest(output / 'reference-review.json') != plan['reference_review_file_sha256']:
        raise ValueError('Exported reference acceptance changed.')
    for name, expected in plan['shadow_sources_sha256'].items():
        if digest(bound(output / 'shadow-source', name)) != expected:
            raise ValueError('Exported shadow source changed: ' + name)
    if canonical(read_record(output / 'candidate-execution.json')) != canonical(candidate_execution(plan)):
        raise ValueError('Candidate was relabeled with a different execution identity.')
    validate_session(read_record(output / 'session.json'), pin)
    expected_names = {'shadow-plan.json', 'reference-review.json', 'reference/receipt.json',
        'session.json', 'candidate-execution.json', 'comparison.json', 'worker-0.log', 'worker-1.log'}
    expected_names.update('shadow-source/' + name for name in SOURCES)
    expected_names.update('reference/replay/' + name for name in WORKER_FILES)
    expected_names.update('worker-' + str(index) + '/' + name for index in (0, 1) for name in WORKER_FILES)
    if set(manifest['files']) != expected_names:
        raise ValueError('Incomplete or unexpected shadow export inventory.')
    comparison = verify_results(package, packet, plan, pin, output, output / 'reference')
    if canonical(read_record(output / 'comparison.json')) != canonical(comparison):
        raise ValueError('Saved shadow comparison disagrees with independent read-back.')
    return comparison


def run(package, packet, plan, pin):
    original_plan, execution = check_sources(package, packet, plan, pin)
    validate_locations(package, packet, Path.home())
    if platform.node() != HOST:
        raise ValueError('Use only the separately declared candidate host and shared package.')
    reference = bound(package, plan['reference_case_directory'])
    reference_gate(reference, original_plan, plan)
    if shutil.disk_usage(packet).free < 2 * 2**30:
        raise OSError('Insufficient free space for bounded shadow artifacts and export reserve.')
    output = reserve_output(packet)
    write_record(output / 'candidate-execution.json', execution)
    started = time.monotonic()
    session = {'kind': KIND + '-session', 'shadow_plan_sha256': pin, 'complete': False, 'passed': False,
               'active_worker': None, 'tasks': {}, 'elapsed_seconds': 0.0, 'research_admitted': False,
               'panel_admitted': False, 'whole_attempt_seconds': LIMIT}
    write_record(output / 'session.json', session)
    from models.matched_diagnostic import bounded_child
    from models.matched_panel_runner import case_deadline
    try:
        with case_deadline(LIMIT, kill_grace_seconds=15):
            hardware_gate()
            for index in (0, 1):
                session['active_worker'] = index
                write_record(output / 'session.json', session, replace=True)
                print('Fresh candidate shadow worker:', index, flush=True)
                command = [sys.executable, '-B', '-u', str(Path(packet) / 'scripts/check_cross_host_replay.py'),
                    'worker', '--package', str(package), '--plan', str(Path(packet) / 'shadow-plan.json'),
                    '--expected-plan-sha256', pin, '--worker-index', str(index)]
                task = bounded_child(command, output / ('worker-' + str(index) + '.log'), remaining_worker_seconds(started, time.monotonic()))
                session['tasks'][str(index)] = task
                session['elapsed_seconds'] = time.monotonic() - started
                write_record(output / 'session.json', session, replace=True)
                if not task['complete']:
                    raise RuntimeError('Shadow worker failed/timed out; stop without retry or dispatch.')
            comparison = verify_results(package, packet, plan, pin, output, reference)
            write_record(output / 'comparison.json', comparison)
            session.update(complete=True, passed=True, active_worker=None, elapsed_seconds=time.monotonic() - started)
    except Exception as error:
        session.update(complete=False, passed=False, error=type(error).__name__ + ': ' + str(error),
                       active_worker=None, elapsed_seconds=time.monotonic() - started)
    finally:
        write_record(output / 'session.json', session, replace=True)
        export_evidence(packet, output, reference, plan)
    print('Shadow finished; passed:', session['passed'], '; no panel cases admitted or dispatched.', flush=True)
    return session
