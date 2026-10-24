"""Bounded matched-forward comparison; distinct precision/reference, no study adoption."""

import copy
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from torch import nn
import yaml

from attacks.pgd import fgsm, native_objective, pgd_linf
from datasets.prepared import PreparedTrainingClips
from models.cuda_convolution import analytical_reference, agreement, TOLERANCES as PATCH_TOLERANCES
from models.cuda_profile import file_hash, run_bounded, tensor_signature, write_json
from models.diagnostics import state_digest
from models.execution import apply_model_policy, configure_execution, precision_contract
from models.gradient_attribution import load_observed, tail_derivative, verify_identity
from models.native_diagnostic import provenance as prior_provenance, tiny_fixture, save_tiny_fixture, TOLERANCES
from models.fixture_state import load_fixture_state, read_fixture_state
from models.native_probe import profile_records, kernel_boundary_checks, validate_probe
from models.precision_patch import BACKWARDS, FORWARD_ID, PrecisionPatchEmbed3D, precision_scope
from models.runtime import synchronize
from models.training import runtime_contract
from models.vjepa2 import build_vitl, load_checkpoint, sample_masks

SOURCES = ('models/precision_patch.py', 'models/precision_diagnostic.py', 'models/gradient_attribution.py',
           'scripts/check_precision.py', 'scripts/university_precision.sh')
REFERENCE = 'shared FP64 patch forward / FP32 tokens / cuDNN FP32 backward'


def provenance(root, upstream):
    return {**prior_provenance(root, upstream),
            'precision_sources_sha256': {n: file_hash(Path(root) / n) for n in SOURCES}}


def native_probe(device, retain=lambda v: None):
    """Observe this new delegator's native backward, outside its FP64 forward."""
    flags = precision_contract()
    p = nn.Conv3d(3, 16, (2, 4, 4), stride=(2, 4, 4)).to(device)
    module = PrecisionPatchEmbed3D(SimpleNamespace(proj=p, patch_size=4, tubelet_size=2, training=True), 'native')
    x = torch.ones(1, 3, 4, 16, 16, device=device, requires_grad=True)
    y = module(x)
    after_forward = precision_contract()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profile:
        y.square().mean().backward()
        synchronize(device)
    rows = profile_records(profile)
    checks = kernel_boundary_checks(rows)
    evidence = {'schema_version': 2, 'device': device, 'forward_id': FORWARD_ID,
                'delegator': 'models.precision_patch._PrecisionProjection',
                'backward_outside_forward_context': True,
                'parameter_and_input_gradients_finite': all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in (x, *p.parameters())),
                'flags_before': flags, 'flags_after_forward': after_forward,
                'flags_after_backward': precision_contract(), 'profiler_events': rows,
                'kernel_boundary_checks': checks, 'actual_cuda_kernel_evidence': device == 'cuda' and checks['passed']}
    retain(evidence)
    validate_probe(evidence, device=device)
    return evidence


def projection_check(model, clean, targets, masks):
    """Trained-weight forward and input/weight/bias oracle gates before attacks."""
    patch = model.encoder.patch_embed
    with torch.no_grad():
        normalized = model.normalize(clean)
        tokens = patch(normalized)
    tail = tail_derivative(model, clean, targets, masks, tokens)
    cot = torch.stack([v.double() for v in tail['cotangents']]).sum(0)
    oracle = analytical_reference(normalized, patch.proj.weight, patch.proj.bias, cot)
    oracle['input'] = oracle['input'] / model.pixel_std.detach().cpu().double()
    x = clean.detach().clone().requires_grad_(True)
    normalized = model.normalize(x)
    outputs = [patch(normalized) for _ in masks]
    parameters = [patch.proj.weight] + ([] if patch.proj.bias is None else [patch.proj.bias])
    grads = torch.autograd.grad(outputs, [x, *parameters], grad_outputs=[v.to(x.device) for v in tail['cotangents']])
    actual = {'output': tokens.detach().cpu(), **{n: g.detach().cpu() for n, g in zip(('input', 'weight', 'bias'), grads)}}
    validation = {name: agreement(value, oracle[name], TOLERANCES['input_gradient'] if name == 'input' else PATCH_TOLERANCES[name])
                  for name, value in actual.items()}
    saved = {'patch_tokens': actual['output'], 'patch_input_gradient': actual['input'],
             'patch_weight_gradient': actual['weight'], 'fp64_weight_gradient': oracle['weight'], 'fp64_tokens': oracle['output'],
             'fp64_input_gradient': oracle['input'], 'tail_loss': tail['loss']}
    if 'bias' in actual:
        saved['patch_bias_gradient'] = actual['bias']
        saved['fp64_bias_gradient'] = oracle['bias']
    return validation, saved


def worker(args, root):
    started = time.monotonic()
    result = {'schema_version': 1, 'finished': False, 'passed': False, 'tiny_fixture': args.tiny,
              'diagnostic_only': True, 'research_result': False, 'study_adopted': False,
              'historical_fp32_failures_preserved': True, 'full_model_optimizer_or_recovery_validated': False,
              'artifacts': {}}
    def progress(stage):
        result['stage'] = stage
        result['elapsed_seconds'] = time.monotonic() - started
        write_json(args.result, result)
        print('Shared-forward control:', args.backward, stage, flush=True)
    def save(name, tensors):
        path = args.result.with_name(args.result.stem + '-' + name + '.pt')
        with path.open('xb') as handle:
            torch.save(tensors, handle)
        result['artifacts'][name] = {'file': path.name, 'sha256': file_hash(path),
                                   'tensors': {n: tensor_signature(v) for n, v in tensors.items()}}
        write_json(args.result, result)
    try:
        progress('configure_and_bind_original_sources')
        configure_execution('cudnn-fp32', device=args.device)
        torch.set_num_threads(4)
        result['provenance'] = provenance(root, args.upstream)
        observed = None
        if not args.tiny:
            if not torch.cuda.is_available() or str(torch.__version__).split('+')[0] != '2.8.0':
                raise ValueError('Existing CUDA PyTorch 2.8.0 required; no install/fallback.')
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(.9, 7 * 2**30 / total))
            old_provenance = {k: result['provenance'][k] for k in ('source_sha256', 'upstream_sha256')}
            observed = load_observed(args.observed, args.reference_run, old_provenance)
            result['observed_spec_sha256'] = file_hash(args.observed)
        # Identical sanity-probe and initialization order in both arms, before
        # resetting construction RNG. Keep allocator/BLAS warm-up symmetric.
        progress('common_new_native_delegator_probe')
        def retain_probe(value):
            result['native_backward_probe'] = value
            write_json(args.result, result)
        native_probe(args.device, retain_probe)
        torch.manual_seed(20261021)
        progress('load_original_model_and_first_training_clip')
        if args.tiny:
            model, clean, masks_for = tiny_fixture(args.upstream)
            initial = load_fixture_state(model, args.initial_fixture, args.initial_fixture_sha256)
            data = {'fixture': 'tiny-upstream-model-and-generated-training-pixels-v2', 'initial_fixture': initial}
        else:
            plan_path = Path(root) / 'configs/budget_study_plan.yaml'
            plan = yaml.safe_load(plan_path.read_text())
            study_path = Path(root) / plan['parent_study']
            study = yaml.safe_load(study_path.read_text())
            manifest_key = study['data']['manifest_directory'] + '/train.csv'
            if (file_hash(study_path) != plan['source_sha256'][plan['parent_study']]
                    or args.manifest.name != 'train.csv' or file_hash(args.manifest) != plan['source_sha256'][manifest_key]):
                raise ValueError('Pinned original training manifest/study differs.')
            clips = PreparedTrainingClips(args.clips, args.manifest)
            clean = clips.pixels(0)
            model = build_vitl(args.upstream, activation_checkpointing=True)
            checkpoint = load_checkpoint(model, args.checkpoint, study['model']['checkpoint_sha256'])
            data = {'checkpoint_sha256': checkpoint['sha256'], 'manifest_sha256': clips.manifest_sha256,
                    'clip_index_sha256': clips.index_sha256, 'training_clip': clips.entries[0],
                    'study_sha256': file_hash(study_path), 'plan_sha256': file_hash(plan_path)}
            masks_for = lambda seed: sample_masks(args.upstream, batch_size=1, seed=seed)
        apply_model_policy(model, 'cudnn-fp32')
        model.to(args.device).eval()
        clean = clean.to(args.device)
        weights = state_digest(model)
        targets = model.clean_targets(clean)
        masks = [m.to(args.device) for m in masks_for(0)]
        scoring_masks = [m.to(args.device) for m in masks_for(100000)]
        if observed is not None:
            verify_identity(runtime_contract(args.device, model=model), data, clean, weights, targets, masks, observed)
        base_identity = {'data': data, 'weights_sha256': weights, 'pixels': tensor_signature(clean),
                         'targets': tensor_signature(targets), 'mask_seeds': [0, 100000],
                         'masks': [[tensor_signature(m.context), tensor_signature(m.target)] for m in masks + scoring_masks]}
        seconds = {}
        with precision_scope(model, args.backward, args.device) as guard:
            result['contract'] = {**base_identity, 'precision': guard.identity, 'reference': REFERENCE,
                'input_and_loss_tolerances': TOLERANCES, 'patch_tolerances': PATCH_TOLERANCES,
                'epsilon_255': 4, 'pgd_steps': 5, 'pgd_step_size_255': 1, 'attack_seed': 0,
                'allocator_cap_gib': None if args.tiny else 7, 'tiny_fixture': args.tiny,
                'initialization_order': 'common native probe, reset seed, construct/load model'}
            progress('trained_patch_forward_and_vjp_oracles')
            tick = time.monotonic()
            validation, tensors = projection_check(model, clean, targets, masks)
            seconds['oracle_control'] = time.monotonic() - tick
            result['oracle_validation'] = validation
            result['comparison_with_historical_fp32'] = 'identity pins only; no claim of historical loss/gradient equality'
            save('projection', tensors)
            if not all(v['passed'] for v in validation.values()):
                raise RuntimeError('Shared-forward projection oracle rejected before attacks; unchanged bounds.')
            progress('full_model_clean_input_gradient')
            objective = native_objective(model, targets, [masks])
            scoring = native_objective(model, targets, [scoring_masks])
            synchronize(args.device)
            tick = time.monotonic()
            x = clean.detach().clone().requires_grad_(True)
            loss = objective(x).sum()
            gradient, = torch.autograd.grad(loss, x)
            synchronize(args.device)
            seconds['clean_loss_and_input_backward'] = time.monotonic() - tick
            clean_tensors = {'clean_loss': loss.detach().cpu(), 'input_gradient': gradient.detach().cpu()}
            result['reconstruction'] = {'loss': agreement(loss, tensors['tail_loss'], TOLERANCES['clean_loss']),
                'gradient': agreement(gradient, tensors['patch_input_gradient'], TOLERANCES['input_gradient'])}
            save('clean', clean_tensors)
            if (not all(v['passed'] for v in result['reconstruction'].values()) or not gradient.abs().max() > 0):
                raise RuntimeError('Full-model chain-rule reconstruction rejected.')
            del x, loss, gradient
            attacks, science = {}, {}
            for method in ('fgsm', 'pgd5'):
                progress(method)
                guard.check()
                synchronize(args.device)
                tick = time.monotonic()
                attack = fgsm(objective, clean, epsilon=4/255) if method == 'fgsm' else pgd_linf(
                    objective, clean, epsilon=4/255, step_size=1/255, steps=5, seed=0)
                with torch.no_grad():
                    score = float(scoring(attack.adversarial))
                synchronize(args.device)
                seconds[method] = time.monotonic() - tick
                delta = float((attack.adversarial - clean).abs().max())
                if (delta > 4/255+1e-7 or attack.adversarial.min() < 0 or attack.adversarial.max() > 1
                        or attack.losses.item() < float(clean_tensors['clean_loss'])):
                    raise RuntimeError('Attack failed projection/clean-candidate checks.')
                attacks[method] = attack.adversarial.detach().cpu()
                science[method] = {'attack_loss': float(attack.losses), 'independent_mask_loss': score, 'max_pixel_delta': delta}
            if not any(v['attack_loss'] > float(clean_tensors['clean_loss']) for v in science.values()):
                raise RuntimeError('No native L1 loss ascent observed.')
            save('attacks', attacks)
            result['scientific'] = {'clean_loss': float(clean_tensors['clean_loss']), 'attacks': science}
            guard.check()
            if state_digest(model) != weights or tensor_signature(targets) != base_identity['targets'] or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('Weights/targets changed or parameter gradients accumulated.')
        result.update(finished=True, passed=True, stage='complete', seconds=seconds,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                      original_module_policy_and_tensor_identities_restored=True)
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), finished=True, passed=False)
        raise
    finally:
        result['wall_seconds'] = time.monotonic() - started
        write_json(args.result, result)


def same_report(actual, stored):
    if set(actual) != set(stored):
        return False
    return all(math.isclose(v, stored[k], rel_tol=1e-12, abs_tol=1e-15) if k == 'relative_l2_error'
               else v == stored[k] for k, v in actual.items())


def validate_contract(contract, *, tiny):
    identity = contract['precision']
    runtime = identity['runtime']
    if (identity['schema_version'] != 1 or not identity['diagnostic_only'] or identity['study_adopted']
            or identity['forward'] != FORWARD_ID or identity['backward'] not in BACKWARDS
            or identity['parameters_and_tokens'] != 'torch.float32'
            or identity['transient_patch_accumulation'] != 'torch.float64'
            or identity['context_module'] != 'models.precision_patch.PrecisionPatchEmbed3D'
            or runtime['device'] != ('cpu' if tiny else 'cuda')
            or not runtime['deterministic_algorithms'] or runtime['deterministic_warn_only']
            or runtime['matmul_precision'] != 'highest' or runtime['matmul_tf32']
            or not runtime['cudnn_enabled'] or runtime['cudnn_tf32'] or runtime['cudnn_benchmark']
            or not runtime['cudnn_deterministic'] or runtime['cublas_workspace_config'] != ':4096:8'
            or runtime['threads'] != 4 or contract['tiny_fixture'] != tiny
            or (not tiny and runtime['torch'].split('+')[0] != '2.8.0')):
        raise ValueError('Declared precision/device/runtime is not the specified control.')


def summarize(directory, tasks, expected, *, tiny, initial_fixture=None):
    directory = Path(directory).resolve()
    if (len(tasks) != 4 or {(t['backward'], t['replay']) for t in tasks} != {(b, i) for b in BACKWARDS for i in (0, 1)}
            or any(t['status'] != 'ok' for t in tasks)):
        raise ValueError('Four successful fresh workers required.')
    fixture_identity = None
    if tiny:
        if not initial_fixture or initial_fixture.get('file') != 'initial-fixture.pt':
            raise ValueError('Pinned initial CPU fixture is required.')
        _, fixture_identity = read_fixture_state(directory / initial_fixture['file'], initial_fixture['sha256'])
        if {k:v for k,v in initial_fixture.items() if k != 'file'} != fixture_identity:
            raise ValueError('Parent initial fixture identity differs.')
    elif initial_fixture is not None:
        raise ValueError('Generated fixture is not valid for a pretrained diagnostic.')
    workers, saved = {}, {}
    for task in tasks:
        path = (directory / task['result']).resolve()
        if not path.is_relative_to(directory) or file_hash(path) != task['result_sha256']:
            raise ValueError('Worker JSON path/hash differs.')
        v = json.loads(path.read_text())
        if (v['schema_version'] != 1 or not v['finished'] or not v['passed'] or v['tiny_fixture'] != tiny
                or v['provenance'] != expected or v['contract']['reference'] != REFERENCE
                or v['contract']['precision']['backward'] != task['backward']
                or v['contract']['precision']['forward'] != FORWARD_ID
                or v['contract']['input_and_loss_tolerances'] != TOLERANCES
                or v['contract']['patch_tolerances'] != PATCH_TOLERANCES
                or not all(row['passed'] for row in v['oracle_validation'].values())
                or not all(row['passed'] for row in v['reconstruction'].values())):
            raise ValueError('Rejected/mixed scientific identity or gate.')
        if tiny and (v['contract']['data'].get('initial_fixture') != fixture_identity
                     or v['contract']['weights_sha256'] != fixture_identity['state_sha256']):
            raise ValueError('Worker did not use the pinned initial fixture.')
        validate_contract(v['contract'], tiny=tiny)
        probe = v['native_backward_probe']
        if probe['forward_id'] != FORWARD_ID or probe['delegator'] != 'models.precision_patch._PrecisionProjection':
            raise ValueError('Probe does not observe this delegator.')
        validate_probe(probe, device='cpu' if tiny else 'cuda')
        tensors = {}
        if set(v['artifacts']) != {'projection', 'clean', 'attacks'}:
            raise ValueError('Incomplete projection/clean/attack evidence.')
        for row in v['artifacts'].values():
            tensor_path = (directory / row['file']).resolve()
            if not tensor_path.is_relative_to(directory) or file_hash(tensor_path) != row['sha256']:
                raise ValueError('Worker tensor path/hash differs.')
            values = torch.load(tensor_path, weights_only=True, map_location='cpu')
            if set(values) & set(tensors) or {n: tensor_signature(x) for n, x in values.items()} != row['tensors']:
                raise ValueError('Saved tensor signatures/keys disagree.')
            tensors.update(values)
        if float(tensors['clean_loss']) != v['scientific']['clean_loss']:
            raise ValueError('Reported clean loss differs from saved tensor.')
        # Recompute scientific gates from saved evidence; never trust pass booleans.
        checks = {
            'output': agreement(tensors['patch_tokens'], tensors['fp64_tokens'], PATCH_TOLERANCES['output']),
            'input': agreement(tensors['patch_input_gradient'], tensors['fp64_input_gradient'], TOLERANCES['input_gradient'])}
        for name in ('weight', 'bias'):
            if 'patch_' + name + '_gradient' in tensors:
                checks[name] = agreement(tensors['patch_' + name + '_gradient'], tensors['fp64_' + name + '_gradient'], PATCH_TOLERANCES[name])
        if set(checks) != set(v['oracle_validation']):
            raise ValueError('Projection gate set changed.')
        for name in checks:
            if not same_report(checks[name], v['oracle_validation'][name]):
                raise ValueError('Stored forward/input oracle gate disagrees with tensors.')
        for name, actual, reference, tol in (
            ('loss', tensors['clean_loss'], tensors['tail_loss'], TOLERANCES['clean_loss']),
            ('gradient', tensors['input_gradient'], tensors['patch_input_gradient'], TOLERANCES['input_gradient'])):
            if not same_report(agreement(actual, reference, tol), v['reconstruction'][name]):
                raise ValueError('Stored reconstruction gate disagrees with tensors.')
        key = task['backward'], task['replay']
        workers[key], saved[key] = v, tensors
    for backend in BACKWARDS:
        a, b = workers[backend, 0], workers[backend, 1]
        if (a['contract'] != b['contract'] or a['scientific'] != b['scientific']
                or set(saved[backend, 0]) != set(saved[backend, 1])
                or any(not torch.equal(x, saved[backend, 1][n]) for n, x in saved[backend, 0].items())):
            raise ValueError('Within-backend fresh-process replay is not exact.')
    contracts = [copy.deepcopy(workers[b, 0]['contract']) for b in BACKWARDS]
    for c in contracts:
        del c['precision']['backward']
    if contracts[0] != contracts[1]:
        raise ValueError('Cross-backward comparison changed forward/data/runtime/targets/masks.')
    a, b = saved['cudnn', 0], saved['native', 0]
    tokens_exact = torch.equal(a['patch_tokens'], b['patch_tokens'])
    comparisons = {n: agreement(b[n], a[n], tol) for n, tol in TOLERANCES.items()}
    return {'passed': tokens_exact and all(v['passed'] for v in comparisons.values()),
            'shared_forward_tokens_bitwise_equal': tokens_exact, 'cross_backward': comparisons,
            'within_backend_fresh_process_exact': {b: True for b in BACKWARDS},
            'reference': REFERENCE, 'historical_fp32_failures_preserved': True,
            'full_model_optimizer_or_recovery_validated': False, 'study_adopted': False,
            'single_attempt_seconds': {b: workers[b, 0]['seconds'] for b in BACKWARDS}}


def run(args, root):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.monotonic()
    expected = provenance(root, args.upstream)
    session = {'schema_version': 1, 'complete': False, 'passed': False, 'research_result': False,
               'study_adopted': False, 'reference': REFERENCE, 'provenance': expected, 'tasks': [],
               'max_seconds': args.max_seconds, 'worker_seconds': args.worker_seconds}
    path = args.output / 'session.json'
    write_json(path, session)
    try:
        if args.tiny:
            identity = save_tiny_fixture(args.upstream, args.output / 'initial-fixture.pt')
            session['initial_fixture'] = {'file': 'initial-fixture.pt', **identity}
            write_json(path, session)
        for backend in reversed(BACKWARDS):
            for replay in (0, 1):
                name = f'{backend}-{replay}'
                print('Shared-forward fresh worker:', name, flush=True)
                result = args.output / (name + '.json')
                command = [sys.executable, '-u', str(Path(root) / 'scripts/check_precision.py'), '--diagnostic',
                           '--worker', '--backward', backend, '--result', str(result),
                           '--device', args.device, '--upstream', str(args.upstream.resolve())]
                if args.tiny:
                    command += ['--tiny', '--initial-fixture', str((args.output/'initial-fixture.pt').resolve()),
                                '--initial-fixture-sha256', session['initial_fixture']['sha256']]
                else:
                    for key in ('checkpoint', 'clips', 'manifest', 'observed', 'reference_run'):
                        command += ['--'+key.replace('_', '-'), str(getattr(args, key).resolve())]
                task = {'backward': backend, 'replay': replay, 'result': result.name}
                task.update(run_bounded(command, args.output / (name + '.log'),
                                        min(args.worker_seconds, args.max_seconds-(time.monotonic()-started))))
                if result.exists():
                    task['result_sha256'] = file_hash(result)
                session['tasks'].append(task)
                write_json(path, session)
                if task['status'] != 'ok':
                    raise RuntimeError('Shared-forward worker rejected/timed out; inspect its JSON/log.')
        session['comparison'] = summarize(args.output, session['tasks'], expected, tiny=args.tiny,
                                          initial_fixture=session.get('initial_fixture'))
        session['complete'] = True
        session['passed'] = session['comparison']['passed']
        if not session['passed']:
            raise RuntimeError('Matched-forward backward comparison rejected; no bounds relaxed.')
    except BaseException as exc:
        session.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        session['wall_seconds'] = time.monotonic() - started
        write_json(path, session)
    return session
