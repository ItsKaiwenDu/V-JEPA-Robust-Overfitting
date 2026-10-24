"""Training-only full-model input-gradient/attack check; no optimizer or rental."""

import copy
from functools import partial
import importlib
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn
import yaml

from attacks.pgd import fgsm, native_objective, pgd_linf
from datasets.prepared import PreparedTrainingClips
from models.cuda_convolution import agreement
from models.cuda_profile import file_hash, run_bounded, tensor_signature, write_json
from models.diagnostics import state_digest
from models.execution import ExecutionGuard, apply_model_policy, configure_execution
from models.fixture_state import load_fixture_state, read_fixture_state, save_fixture_state
from models.native_probe import native_backward_probe, validate_probe
from models.runtime import synchronize
from models.training import runtime_contract
from models.vjepa2 import FixedTargetVJEPA, MaskPair, build_vitl, load_checkpoint, sample_masks, use_upstream

POLICIES = ('cudnn-fp32', 'native-fp32')
TOLERANCES = {'clean_loss': {'rtol': 1e-5, 'atol': 2e-6},
              'input_gradient': {'rtol': 2e-3, 'atol': 2e-7}}
SOURCES = ('models/fixture_state.py', 'models/native_diagnostic.py', 'models/native_probe.py', 'models/native_patch.py', 'models/execution.py',
           'models/patch_embed.py', 'models/runtime.py', 'models/vjepa2.py', 'models/training.py',
           'models/diagnostics.py', 'models/pilot.py', 'models/checkpointing.py',
           'models/cuda_convolution.py', 'models/cuda_profile.py', 'attacks/pgd.py',
           'datasets/prepared.py', 'scripts/check_native_model.py', 'scripts/university_native.sh')


def provenance(root, upstream):
    upstream = use_upstream(upstream)
    upstream_names = sorted(p.relative_to(upstream).as_posix() for p in (upstream / 'src').rglob('*.py'))
    return {'source_sha256': {n: file_hash(Path(root) / n) for n in SOURCES},
            'upstream_sha256': {n: file_hash(upstream / n) for n in upstream_names}}


def tiny_fixture(upstream):
    """Small actual upstream transformers, explicitly not a pretrained result."""
    use_upstream(upstream)
    common = dict(img_size=16, patch_size=4, num_frames=4, tubelet_size=2,
                  use_rope=True, num_heads=2, depth=2, use_sdpa=True)
    encoder = importlib.import_module('src.models.vision_transformer').VisionTransformer(
        embed_dim=128, norm_layer=partial(nn.LayerNorm, eps=1e-6), **common)
    predictor = importlib.import_module('src.models.predictor').vit_predictor(
        embed_dim=128, predictor_embed_dim=64, use_mask_tokens=True, num_mask_tokens=10, **common)
    model = FixedTargetVJEPA(encoder, predictor, copy.deepcopy(encoder))
    model.encoder.use_activation_checkpointing = True
    model.predictor.use_activation_checkpointing = True
    pixels = .2 + .6 * torch.rand(1, 3, 4, 16, 16, generator=torch.Generator().manual_seed(17))
    def masks(seed):
        order = torch.randperm(32, generator=torch.Generator().manual_seed(seed)).unsqueeze(0)
        return [MaskPair(order[:, :8], order[:, 8:]), MaskPair(order[:, :24], order[:, 24:])]
    return model, pixels, masks


def save_tiny_fixture(upstream, path, *, seed=20261021):
    """One initial checkpoint shared by fresh workers; preserve caller RNG/threads."""
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model, _, _ = tiny_fixture(upstream)
            return save_fixture_state(model, path)
    finally:
        torch.set_num_threads(threads)


def worker(args, root):
    started = time.monotonic()
    result = {'schema_version': 2, 'finished': False, 'passed': False, 'stage': 'configure',
              'diagnostic_only': True, 'research_result': False, 'tiny_fixture': args.tiny}
    write_json(args.result, result)
    try:
        configure_execution(args.execution_policy, device=args.device)
        torch.set_num_threads(4)
        if not args.tiny:
            if args.device != 'cuda' or str(torch.__version__).split('+')[0] != '2.8.0' or not torch.cuda.is_available():
                raise ValueError('Full-model diagnostic requires existing CUDA PyTorch 2.8.0; no fallback/install.')
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(.9, args.max_memory_gib * 2**30 / total))
        result['provenance'] = provenance(root, args.upstream)
        if args.execution_policy == 'native-fp32':
            result['stage'] = 'native_backward_kernel_probe'
            write_json(args.result, result)
            def retain_probe(evidence):
                result['native_backward_probe'] = evidence
                write_json(args.result, result)
            native_backward_probe(args.device, on_evidence=retain_probe)
        torch.manual_seed(20261021)  # Same construction in every fresh worker.
        result['stage'] = 'load_model_and_one_training_clip'
        write_json(args.result, result)
        if args.tiny:
            model, clean, masks_for = tiny_fixture(args.upstream)
            initial = load_fixture_state(model, args.initial_fixture, args.initial_fixture_sha256)
            data = {'fixture': 'tiny-upstream-model-and-generated-training-pixels-v2', 'initial_fixture': initial}
        else:
            plan = yaml.safe_load((Path(root) / 'configs/budget_study_plan.yaml').read_text())
            study_path = Path(root) / plan['parent_study']
            if file_hash(study_path) != plan['source_sha256'][plan['parent_study']]:
                raise ValueError('Parent study pin changed.')
            study = yaml.safe_load(study_path.read_text())
            manifest_key = study['data']['manifest_directory'] + '/train.csv'
            if args.manifest.name != 'train.csv' or file_hash(args.manifest) != plan['source_sha256'][manifest_key]:
                raise ValueError('Use only the pinned original training manifest.')
            clips = PreparedTrainingClips(args.clips, args.manifest)
            clean = clips.pixels(0)  # Exactly one training clip; no held-out pixel files opened.
            model = build_vitl(args.upstream, activation_checkpointing=True)
            checkpoint = load_checkpoint(model, args.checkpoint, study['model']['checkpoint_sha256'])
            data = {'checkpoint_sha256': checkpoint['sha256'], 'manifest_sha256': clips.manifest_sha256,
                    'clip_index_sha256': clips.index_sha256, 'training_clip': clips.entries[0],
                    'study_sha256': file_hash(study_path), 'plan_sha256': file_hash(Path(root) / 'configs/budget_study_plan.yaml')}
            masks_for = lambda seed: sample_masks(args.upstream, batch_size=1, seed=seed)
        apply_model_policy(model, args.execution_policy)
        model.to(args.device).eval()
        clean = clean.to(args.device)
        guard = ExecutionGuard(model, args.device, target_versions=True)
        before = state_digest(model)
        targets = model.clean_targets(clean)
        target_before = tensor_signature(targets)
        attack_masks = [m.to(args.device) for m in masks_for(0)]
        score_masks = [m.to(args.device) for m in masks_for(100000)]
        objective = native_objective(model, targets, [attack_masks])
        scoring = native_objective(model, targets, [score_masks])
        result['contract'] = {'execution_policy': args.execution_policy, 'runtime': runtime_contract(args.device, model=model),
            'data': data, 'pixels': tensor_signature(clean), 'weights_sha256': before,
            'targets': target_before, 'mask_seeds': [0, 100000], 'attack_seed': 0,
            'masks': [[tensor_signature(m.context), tensor_signature(m.target)] for m in attack_masks + score_masks],
            'epsilon_255': 4, 'pgd_step_size_255': 1, 'pgd_steps': 5, 'tiny_fixture': args.tiny,
            'allocator_limit_gib': None if args.tiny else args.max_memory_gib, 'tolerances': TOLERANCES}
        tensors, science, seconds = {}, {}, {}
        result['stage'] = 'full_model_input_gradient'
        write_json(args.result, result)
        synchronize(args.device)
        tick = time.monotonic()
        x = clean.detach().clone().requires_grad_(True)
        loss = objective(x).sum()
        gradient, = torch.autograd.grad(loss, x)
        synchronize(args.device)
        seconds['clean_loss_and_input_backward'] = time.monotonic() - tick
        if not torch.isfinite(gradient).all() or not gradient.abs().max() > 0:
            raise RuntimeError('Invalid or zero full-model input gradient.')
        tensors.update(clean_loss=loss.detach().cpu(), input_gradient=gradient.detach().cpu())
        science.update(clean_loss=float(loss.detach()), input_gradient=tensor_signature(gradient))
        del x, loss, gradient
        for method in ('fgsm', 'pgd5'):
            result['stage'] = method
            write_json(args.result, result)
            synchronize(args.device)
            tick = time.monotonic()
            attack = (fgsm(objective, clean, epsilon=4/255) if method == 'fgsm' else
                      pgd_linf(objective, clean, epsilon=4/255, step_size=1/255, steps=5, seed=0))
            with torch.no_grad():
                independent_loss = float(scoring(attack.adversarial))
            synchronize(args.device)
            seconds[method] = time.monotonic() - tick
            delta = float((attack.adversarial - clean).abs().max())
            if (delta > 4/255 + 1e-7 or attack.adversarial.min() < 0 or attack.adversarial.max() > 1
                    or attack.losses.item() < science['clean_loss']):
                raise RuntimeError('Attack failed projection or clean-candidate retention checks.')
            science[method] = {'attack_loss': float(attack.losses), 'independent_mask_loss': independent_loss,
                               'max_pixel_delta': delta, 'adversarial': tensor_signature(attack.adversarial)}
            tensors[method] = attack.adversarial.cpu()
        if not any(science[m]['attack_loss'] > science['clean_loss'] for m in ('fgsm', 'pgd5')):
            raise RuntimeError('No native loss ascent observed on this one training clip.')
        guard.check()
        if (state_digest(model) != before or tensor_signature(targets) != target_before
                or targets.requires_grad or any(p.grad is not None for p in model.parameters())):
            raise RuntimeError('Input-gradient diagnostics changed weights/targets or accumulated parameter gradients.')
        tensor_file = args.result.with_suffix('.pt')
        with tensor_file.open('xb') as handle:
            torch.save(tensors, handle)
        result.update(scientific=science, seconds=seconds, tensors_file=tensor_file.name,
                      tensors_sha256=file_hash(tensor_file), finished=True, passed=True, stage='complete',
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                      weights_targets_and_parameter_gradients_preserved=True)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), finished=True, passed=False)
        raise
    finally:
        result['worker_wall_seconds'] = time.monotonic() - started
        write_json(args.result, result)


def summarize(directory, tasks, expected_provenance, *, tiny, initial_fixture=None):
    """Incomplete, mixed-runtime/source, or altered artifacts never pass."""
    directory = Path(directory).resolve()
    if (len(tasks) != 4 or {(t['policy'], t['replay']) for t in tasks} != {(p, i) for p in POLICIES for i in (0, 1)}
            or any(t['status'] != 'ok' for t in tasks)):
        raise ValueError('Four successful fresh-process workers are required.')
    fixture_identity = None
    if tiny:
        if not initial_fixture or initial_fixture.get('file') != 'initial-fixture.pt':
            raise ValueError('Pinned initial CPU fixture is required.')
        _, fixture_identity = read_fixture_state(directory / initial_fixture['file'], initial_fixture['sha256'])
        if {k:v for k,v in initial_fixture.items() if k != 'file'} != fixture_identity:
            raise ValueError('Parent initial fixture identity differs.')
    elif initial_fixture is not None:
        raise ValueError('Generated fixture is not valid for a pretrained diagnostic.')
    workers, tensors = {}, {}
    for task in tasks:
        path = (directory / task['result']).resolve()
        if not path.is_relative_to(directory) or file_hash(path) != task['result_sha256']:
            raise ValueError('Worker result path/hash changed.')
        value = json.loads(path.read_text())
        contract = value['contract']
        if (value['schema_version'] != 2 or not value['passed'] or not value['finished'] or value['tiny_fixture'] != tiny
                or value['provenance'] != expected_provenance or contract['execution_policy'] != task['policy']
                or contract['runtime']['model_execution']['policy'] != task['policy']
                or contract['tolerances'] != TOLERANCES):
            raise ValueError('Worker evidence policy/source/scale differs or failed.')
        if tiny and (contract['data'].get('initial_fixture') != fixture_identity
                     or contract['weights_sha256'] != fixture_identity['state_sha256']):
            raise ValueError('Worker did not use the pinned initial fixture.')
        tensor_path = (path.parent / value['tensors_file']).resolve()
        if not tensor_path.is_relative_to(directory) or file_hash(tensor_path) != value['tensors_sha256']:
            raise ValueError('Worker tensors path/hash changed.')
        saved = torch.load(tensor_path, map_location='cpu', weights_only=True)
        if (set(saved) != {'clean_loss', 'input_gradient', 'fgsm', 'pgd5'}
                or float(saved['clean_loss']) != value['scientific']['clean_loss']
                or tensor_signature(saved['input_gradient']) != value['scientific']['input_gradient']
                or any(tensor_signature(saved[m]) != value['scientific'][m]['adversarial'] for m in ('fgsm', 'pgd5'))):
            raise ValueError('Recorded tensor signatures disagree with saved evidence.')
        key = task['policy'], task['replay']
        workers[key], tensors[key] = value, saved
    for policy in POLICIES:
        a, b = workers[policy, 0], workers[policy, 1]
        if a['contract'] != b['contract'] or a['scientific'] != b['scientific']:
            raise ValueError('Full input-gradient/attack fresh-process replay is not exact.')
        if policy == 'native-fp32':
            for value in (a, b):
                validate_probe(value['native_backward_probe'], device='cpu' if tiny else 'cuda')
    baseline, native = (copy.deepcopy(workers[p, 0]['contract']) for p in POLICIES)
    for contract in (baseline, native):
        policy = contract.pop('execution_policy')
        runtime, execution = contract['runtime'], contract['runtime']['model_execution']
        expected_module = ('models.native_patch.NativePatchEmbed3D' if policy == 'native-fp32'
                           else 'src.models.utils.patch_embed.PatchEmbed3D')
        if (runtime['device'] != ('cpu' if tiny else 'cuda') or contract['tiny_fixture'] != tiny
                or execution['context_patch']['module'] != expected_module
                or execution['native_backward'] != ('aten.convolution_backward-first-order-v1' if policy == 'native-fp32' else None)
                or not runtime['deterministic_algorithms'] or runtime['deterministic_warn_only']
                or runtime['matmul_precision'] != 'highest' or runtime['matmul_tf32']
                or runtime['cudnn_tf32'] or not runtime['cudnn_enabled']
                or runtime['cudnn_benchmark'] or not runtime['cudnn_deterministic']
                or runtime['cublas_workspace_config'] != ':4096:8'
                or (not tiny and runtime['torch'].split('+')[0] != '2.8.0')):
            raise ValueError('Actual backend, precision, scale, or deterministic runtime differs.')
        # Only the declared context dispatch may differ. Retain target/patch
        # geometry and actual activation checkpointing in the comparison.
        del execution['policy'], execution['native_backward'], execution['context_patch']['module']
    if baseline != native:
        raise ValueError('Backend comparison changed data, weights, targets, masks, runtime, precision, or bounds.')
    comparison = {name: agreement(tensors['native-fp32', 0][name], tensors['cudnn-fp32', 0][name], tolerance)
                  for name, tolerance in TOLERANCES.items()}
    return {'passed': all(v['passed'] for v in comparison.values()), 'cross_backend': comparison,
            'within_backend_exact_replay': {p: True for p in POLICIES},
            'attacks_compared_for_exact_replay_only_within_each_backend': True,
            'full_model_optimizer_or_recovery_validated': False, 'tiny_fixture': tiny,
            'single_attempt_seconds': {p: workers[p, 0]['seconds'] for p in POLICIES}}


def run(args, root):
    args.output.mkdir(parents=True, exist_ok=False)
    expected = provenance(root, args.upstream)
    session = {'schema_version': 2, 'diagnostic_only': True, 'research_result': False,
               'budget_approved': False, 'tiny_fixture': args.tiny, 'complete': False, 'passed': False,
               'tolerances': TOLERANCES, 'provenance': expected, 'tasks': [],
               'worker_seconds': args.worker_seconds, 'max_seconds': args.max_seconds,
               'full_model_optimizer_or_recovery_validated': False}
    session_path = args.output / 'session.json'
    write_json(session_path, session)
    started = time.monotonic()
    try:
        if args.tiny:
            identity = save_tiny_fixture(args.upstream, args.output / 'initial-fixture.pt')
            session['initial_fixture'] = {'file': 'initial-fixture.pt', **identity}
            write_json(session_path, session)
        # Fail a rejected native probe before repeating the expensive reference.
        for policy in reversed(POLICIES):
            for replay in (0, 1):
                print(f'Full-model input/attack diagnostic: {policy}; replay: {replay}', flush=True)
                name = f'{policy}-{replay}'
                result = args.output / (name + '.json')
                command = [sys.executable, '-u', str(Path(root) / 'scripts/check_native_model.py'),
                    '--diagnostic', '--worker', '--execution-policy', policy, '--result', str(result),
                    '--upstream', str(args.upstream.resolve()), '--device', args.device,
                    '--max-memory-gib', str(args.max_memory_gib)]
                if args.tiny:
                    command += ['--tiny', '--initial-fixture', str((args.output/'initial-fixture.pt').resolve()),
                                '--initial-fixture-sha256', session['initial_fixture']['sha256']]
                else:
                    for key in ('checkpoint', 'clips', 'manifest'):
                        command += ['--' + key, str(getattr(args, key).resolve())]
                task = {'policy': policy, 'replay': replay, 'result': result.name}
                task.update(run_bounded(command, args.output / (name + '.log'),
                                        min(args.worker_seconds, args.max_seconds - (time.monotonic() - started))))
                if result.is_file():
                    task['result_sha256'] = file_hash(result)
                session['tasks'].append(task)
                write_json(session_path, session)
                if task['status'] != 'ok':
                    raise RuntimeError('Worker failed/timed out; preserve evidence and inspect its log.')
        session['comparison'] = summarize(args.output, session['tasks'], expected, tiny=args.tiny,
                                          initial_fixture=session.get('initial_fixture'))
        session['complete'] = True
        session['passed'] = session['comparison']['passed']
        if not session['passed']:
            raise RuntimeError('Full-model FP32 numerical agreement gate rejected; no tolerance was relaxed.')
    except BaseException as exc:
        session.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        session['wall_seconds'] = time.monotonic() - started
        write_json(session_path, session)
    return session
