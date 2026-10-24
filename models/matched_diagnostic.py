"""Bounded released-checkpoint workers for the separate matched attack check."""

import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

import torch
import yaml

from datasets.staged_files import verify_manifests
from datasets.staged_preparation import RECIPE
from models.checkpointing import file_sha256
from models.evaluation import EvaluationMaskSampler
from models.execution import apply_model_policy, configure_execution
from models.matched_attack import ARMS, KIND, POLICY, compare_replay, run_attacks
from models.pilot_bundle import CHECKPOINT, make_archive
from models.runtime import MemoryMonitor
from models.study_artifacts import write_record
from models.vjepa2 import PRETRAIN_CONFIG, UPSTREAM_COMMIT, build_vitl, load_checkpoint

NEW_SOURCES = ('models/matched_attack.py', 'models/matched_diagnostic.py',
               'scripts/check_matched_attack.py', 'scripts/university_matched_attack.sh',
               'scripts/prepare_matched_attack_handoff.py')


def monitor_input(root, manifests, prepared):
    """Validate the complete metadata, but open only the first training monitor."""
    rows, metadata = verify_manifests(root, manifests)
    prepared = Path(prepared).resolve()
    preparation = json.loads((prepared / 'preparation.json').read_text())
    index_path = prepared / 'train/index.json'
    index = json.loads(index_path.read_text())
    if (preparation['kind'] != 'exploration_preparation'
            or preparation['manifest_sha256'] != metadata['manifest_sha256']
            or file_sha256(index_path) != preparation['clip_indices_sha256']['train']
            or index['split'] != 'train'
            or index['source_manifest_sha256'] != metadata['files_sha256']['train.csv']
            or len(index['clips']) != len(rows['train'])):
        raise ValueError('Training preparation/index identity differs.')
    for entry, row in zip(index['clips'], rows['train']):
        if any(entry.get(k) != v for k, v in row.items()) or entry['recipe'] != RECIPE:
            raise ValueError('Training index differs from the declared recipe/manifest.')
    first = rows['train_monitor'][0]
    entries = [entry for entry in index['clips'] if entry['video_id'] == first['video_id']]
    if len(entries) != 1 or any(entries[0][k] != v for k, v in first.items()):
        raise ValueError('First monitor is not a unique member of actual training.')
    entry = entries[0]
    if Path(entry['file']).name != entry['file']:
        raise ValueError('Unsafe training tensor filename.')
    path = prepared / 'train' / entry['file']
    if path.is_symlink() or not path.resolve().is_relative_to(prepared):
        raise ValueError('Escaped training tensor.')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    clean = payload['pixels']
    import hashlib
    if (payload['metadata'] != {k: v for k, v in entry.items() if k != 'file'}
            or clean.dtype != torch.float32 or tuple(clean.shape) != (3, 16, 256, 256)
            or not torch.isfinite(clean).all() or clean.min() < 0 or clean.max() > 1
            or hashlib.sha256(clean.contiguous().numpy().tobytes()).hexdigest() != entry['pixel_sha256']):
        raise ValueError('First training monitor tensor/metadata checksum mismatch.')
    return clean.unsqueeze(0), {'panel': 'train_monitor', 'selection': 'first-declared-monitor-only-v1',
        'row': first, 'pixel_sha256': entry['pixel_sha256'], 'file_sha256': file_sha256(path),
        'manifest_sha256': metadata['manifest_sha256'], 'index_sha256': file_sha256(index_path),
        'preparation_sha256': file_sha256(prepared / 'preparation.json')}


def source_contract(root, upstream):
    from models.workflow_diagnostic import SOURCES
    modules = {p.relative_to(root).as_posix() for name in ('models', 'attacks', 'datasets')
               for p in (root / name).rglob('*.py')}
    sources = sorted(set(SOURCES) | set(NEW_SOURCES) | modules)
    names = sorted(p.relative_to(upstream).as_posix() for p in (upstream / 'src').rglob('*.py'))
    names += ['app/vjepa/train.py', PRETRAIN_CONFIG]
    return {'source_sha256': {n: file_sha256(root / n) for n in sources},
            'handoff_manifest_sha256': file_sha256(root / 'MATCHED-HANDOFF.json')
                if (root / 'MATCHED-HANDOFF.json').is_file() else None,
            'upstream_commit': UPSTREAM_COMMIT,
            'upstream_sha256': {n: file_sha256(upstream / n) for n in names}}


def worker(root, args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    # Configure before the allocator/device is initialized. No backend fallback.
    configure_execution(POLICY, device='cuda')
    torch.set_num_threads(4)
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or str(torch.__version__) != '2.8.0+cu128'):
        raise ValueError('Use exactly one CUDA GPU and the existing PyTorch 2.8.0+cu128 runtime.')
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.9, 7 * 2**30 / total))
    from models.precision_diagnostic import native_probe
    probe = native_probe('cuda')
    torch.manual_seed(20261023)
    plan_path = root / 'configs/budget_study_plan.yaml'
    plan = yaml.safe_load(plan_path.read_text())
    study_path = root / plan['parent_study']
    study = yaml.safe_load(study_path.read_text())
    if (file_sha256(study_path) != plan['source_sha256'][plan['parent_study']]
            or study['model']['checkpoint_sha256'] != CHECKPOINT
            or study['model']['upstream_commit'] != UPSTREAM_COMMIT):
        raise ValueError('Original study/checkpoint/upstream pins changed.')
    clean, data = monitor_input(root, args.manifests, args.clips)
    model = build_vitl(args.upstream, activation_checkpointing=True)
    original = load_checkpoint(model, args.checkpoint, CHECKPOINT)
    apply_model_policy(model, POLICY)
    model.to('cuda')
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version',
                                        '--format=csv,noheader,nounits'], text=True).strip()
    provenance = {**source_contract(root, args.upstream), 'data': data, 'host': platform.node(),
                  'nvidia_smi_identity': hardware,
                  'checkpoint_sha256': original['sha256'], 'study_sha256': file_sha256(study_path),
                  'plan_sha256': file_sha256(plan_path), 'allocator_limit_bytes': min(int(.9 * total), 7 * 2**30)}
    with MemoryMonitor('cuda') as memory:
        run_attacks(model, clean.to('cuda'), EvaluationMaskSampler(args.upstream),
                    video_id=data['row']['video_id'], provenance=provenance, output=output,
                    arms=tuple(ARMS) if args.role == 'matrix' else ('A',))
    write_record(output / 'observation.json', {'kind': KIND, 'role': args.role, 'memory': memory.report(),
                 'native_probe': probe, 'timing_pool_eligible': False, 'research_admitted': False})


def bounded_child(command, log, seconds):
    """Kill the full process group at its deadline, including on parent TERM."""
    process = None
    old_handler = signal.getsignal(signal.SIGTERM)
    def stop(*_):
        raise TimeoutError('Matched diagnostic supervisor terminated.')
    signal.signal(signal.SIGTERM, stop)
    try:
        with Path(log).open('xb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                code = process.wait(timeout=seconds)
                return {'exit_status': code, 'timed_out': False, 'complete': code == 0}
            except subprocess.TimeoutExpired:
                return {'exit_status': None, 'timed_out': True, 'complete': False}
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        signal.signal(signal.SIGTERM, old_handler)


def export_evidence(root, output):
    """Allowlist small diagnostic artifacts; never collect checkpoints or keys."""
    files = {}
    for role in ('matrix', 'replay'):
        directory = output / role
        for name in ('contract.json', 'inputs.json', 'inputs.pt', 'worker.json', 'observation.json',
                     *[a + suffix for a in ARMS for suffix in ('.pt', '.json')]):
            path = directory / name
            if path.is_file() and not path.is_symlink():
                if path.stat().st_size > 65 * 2**20:
                    raise ValueError('Unexpected diagnostic artifact size; preserve raw files.')
                files[role + '/' + name] = path
    for name in ('session.json', 'matrix.log', 'replay.log'):
        path = output / name
        if path.is_file() and not path.is_symlink():
            files[name] = path
    for name in NEW_SOURCES:
        files['source/' + name] = root / name
    manifest_path = root / 'MATCHED-HANDOFF.json'
    if manifest_path.is_file():
        files['MATCHED-HANDOFF.json'] = manifest_path
    if sum(p.stat().st_size for p in files.values()) > 256 * 2**20:
        raise ValueError('Diagnostic evidence exceeds the declared 256 MiB bound.')
    manifest = make_archive(output / 'evidence.tar.gz', files)
    (output / 'evidence.tar.gz.manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return {'bytes': manifest['bytes'], 'sha256': manifest['sha256'], 'members': len(files)}


def session(root, args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = {'kind': KIND, 'diagnostic_only': True, 'research_admitted': False, 'passed': False,
              'complete': False, 'tasks': {}, 'replay_scope': 'A-only', 'a40_cost_pool_eligible': False}
    started = time.monotonic()
    old_handler = signal.getsignal(signal.SIGTERM)
    def stop(*_):
        raise TimeoutError('Matched diagnostic session reached its external deadline.')
    signal.signal(signal.SIGTERM, stop)
    try:
        for role, cap in (('matrix', 3600), ('replay', 900)):
            remaining = args.max_seconds - (time.monotonic() - started) - 30
            if remaining < 1:
                raise TimeoutError('No time remains for the required fresh worker.')
            command = [sys.executable, '-u', str(root / 'scripts/check_matched_attack.py'), '--diagnostic',
                       '--role', role, '--worker', '--checkpoint', str(args.checkpoint),
                       '--upstream', str(args.upstream), '--manifests', str(args.manifests),
                       '--clips', str(args.clips), '--output', str(output / role)]
            result['tasks'][role] = bounded_child(command, output / (role + '.log'), min(cap, remaining))
            write_record(output / 'session.json', result, replace=True)
            if not result['tasks'][role]['complete']:
                raise RuntimeError('Worker failed/timed out; retain partial evidence, do not call it a pass.')
        result['replay'] = compare_replay(output / 'matrix', output / 'replay')
        from models.matched_attack import verify_worker
        _, rows = verify_worker(output / 'matrix')
        result['paired_independent_loss_contrasts'] = {
            name: {bank: rows[high]['scores'][bank]['adversarial'] - rows[low]['scores'][bank]['adversarial']
                   for bank in ('main', 'audit_0', 'audit_1')}
            for name, low, high in (('50_minus_20_at_EOT4', 'A', 'B'), ('50_minus_20_at_EOT8', 'C', 'D'),
                                    ('EOT8_minus_EOT4_at_20', 'A', 'C'), ('EOT8_minus_EOT4_at_50', 'B', 'D'))}
        result.update(passed=True, complete=True)
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        write_record(output / 'session.json', result, replace=True)
        export = export_evidence(root, output)
        print('Matched diagnostic evidence:', export, flush=True)
        signal.signal(signal.SIGTERM, old_handler)
    return result
