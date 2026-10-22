"""Read-only patch/tail derivative attribution, never backend acceptance/training."""

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
import time

import torch
import yaml

from datasets.prepared import PreparedTrainingClips
from models.cuda_convolution import analytical_reference, agreement, TOLERANCES as PATCH_TOLERANCES
from models.cuda_profile import file_hash, run_bounded, tensor_signature, write_json
from models.diagnostics import state_digest
from models.execution import ExecutionGuard, apply_model_policy, configure_execution
from models.native_diagnostic import provenance, tiny_fixture, TOLERANCES, POLICIES
from models.native_patch import NativePatchEmbed3D
from models.native_probe import native_backward_probe
from models.training import rng_state, restore_rng, runtime_contract
from models.vjepa2 import build_vitl, load_checkpoint, sample_masks

SOURCES = ('models/gradient_attribution.py', 'scripts/check_gradient_attribution.py',
           'scripts/university_gradient.sh')


def difference(a, b):
    """Descriptive metrics, with the unchanged full-model derivative bounds."""
    result = agreement(a, b, TOLERANCES['input_gradient'])
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    active = (a != 0) & (b != 0)
    result.update(exact=torch.equal(a, b),
                  zero_support_disagreements=int(((a == 0) != (b == 0)).sum()),
                  active_sign_disagreements=int(((a.sign() != b.sign()) & active).sum()))
    return result


@contextmanager
def retained_rng(device):
    state = rng_state(device)
    try:
        yield
    finally:
        restore_rng(state, device)


def tail_derivative(model, pixels, targets, masks, tokens):
    """Intervene only at patch outputs; separate leaf/cotangent per mask family.

    Preserves original dense transposed layout. The incoming derivatives remain
    separate to reproduce the original two-branch accumulation/normalization.
    Hooks are removed even on failure; no module, mode or parameter is replaced.
    """
    patch = model.encoder.patch_embed
    if any(m.training for m in model.modules()) or targets.requires_grad:
        raise ValueError('Tail control requires evaluation mode and detached clean targets.')
    leaves = [tokens.detach().clone(memory_format=torch.preserve_format).requires_grad_(True) for _ in masks]
    if any(v.stride() != tokens.stride() for v in leaves):
        raise ValueError('Leaf substitution changed patch-token layout.')
    calls, predictions = [], []
    def replace(module, inputs, output):
        if (len(calls) >= len(leaves) or output.shape != tokens.shape
                or output.dtype != tokens.dtype or output.device != tokens.device
                or output.stride() != tokens.stride()):
            raise ValueError('Unexpected patch call/shape/layout during tail intervention.')
        leaf = leaves[len(calls)]
        calls.append(True)
        return leaf
    def capture(module, inputs, output):
        predictions.append(output.detach().cpu())
    handles = []
    with retained_rng(pixels.device.type):
        try:
            handles.append(patch.register_forward_hook(replace))
            handles.append(model.predictor.register_forward_hook(capture))
            loss = model(pixels.detach(), targets, masks)
            cotangents = torch.autograd.grad(loss, leaves)
            if len(calls) != len(masks) or len(predictions) != len(masks):
                raise ValueError('Tail intervention did not observe exactly one call per family.')
            residuals = []
            for prediction, pair in zip(predictions, masks):
                indices = pair.target.cpu().unsqueeze(-1).expand(-1, -1, targets.shape[-1])
                residuals.append(prediction - targets.detach().cpu().gather(1, indices))
            return {'loss': loss.detach().cpu(), 'cotangents': [c.detach().cpu() for c in cotangents],
                    'predictions': predictions, 'residuals': residuals,
                    'token_stride': list(tokens.stride()), 'leaf_stride': list(leaves[0].stride()),
                    'patch_calls': len(calls)}
        finally:
            for handle in reversed(handles):
                handle.remove()


def pixel_vjp(model, patch, pixels, cotangents):
    """Fixed incoming derivatives, original branch order and shared normalization."""
    x = pixels.detach().clone().requires_grad_(True)
    normalized = model.normalize(x)
    outputs = [patch(normalized) for _ in cotangents]
    gradient, = torch.autograd.grad(outputs, x, grad_outputs=[v.to(x.device) for v in cotangents])
    return gradient.detach().cpu()


def fp64_reference(model, pixels, cotangents):
    """Independent unfolded CPU FP64 VJP, including raw RGB normalization."""
    patch = model.encoder.patch_embed.proj
    # FP64 sum is intentional: oracle for the mathematical chain rule, without
    # introducing an FP32 branch-accumulation rounding into the reference.
    cotangent = torch.stack([v.double() for v in cotangents]).sum(0)
    ref = analytical_reference(model.normalize(pixels).detach().cpu(),
                               patch.weight, patch.bias, cotangent)
    return {'tokens': ref['output'], 'raw_pixel_vjp': ref['input'] / model.pixel_std.detach().cpu().double()}


def residual_changes(a, b):
    if len(a) != len(b):
        raise ValueError('Residual families differ.')
    rows = []
    for left, right in zip(a, b):
        if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError('Residual shape/finiteness differs.')
        changed = left.sign() != right.sign()
        magnitude = torch.maximum(left.abs(), right.abs())
        rows.append({'elements': left.numel(), 'sign_or_zero_changes': int(changed.sum()),
                     'nonzero_sign_crossings': int((changed & (left != 0) & (right != 0)).sum()),
                     'left_zeros': int((left == 0).sum()), 'right_zeros': int((right == 0).sum()),
                     'largest_magnitude_at_changed_sign': float(magnitude[changed].max()) if changed.any() else 0.,
                     'max_residual_difference': float((left - right).abs().max())})
    return rows


def load_observed(path, reference_run, expected_provenance):
    """Bind existing rejected evidence before opening any model/video assets."""
    spec = json.loads(Path(path).read_text())
    if spec['schema_version'] != 1 or spec['prior_attempt_passed'] is not False or spec['provenance'] != expected_provenance:
        raise ValueError('Observed failed-run source identity changed.')
    directory = Path(reference_run).resolve() / 'results'
    session_path = directory / 'session.json'
    if not session_path.is_file() or file_hash(session_path) != spec['prior_session_sha256']:
        raise ValueError('Observed failed-run session hash changed.')
    session = json.loads(session_path.read_text())
    if not session['complete'] or session['passed']:
        raise ValueError('Expected the complete, rejected prior comparison.')
    observed = {}
    for policy in POLICIES:
        row = spec['workers'][policy]
        paths = [(directory / row[k]).resolve() for k in ('json_file', 'tensor_file')]
        if any(not p.is_relative_to(directory) or not p.is_file() for p in paths):
            raise ValueError('Observed evidence path missing/escapes results.')
        meta_path, tensor_path = paths
        if file_hash(meta_path) != row['json_sha256'] or file_hash(tensor_path) != row['tensor_sha256']:
            raise ValueError('Observed failed-run artifact hash changed.')
        meta = json.loads(meta_path.read_text())
        values = torch.load(tensor_path, weights_only=True, map_location='cpu')
        if (not meta['finished'] or not meta['passed'] or meta['provenance'] != expected_provenance
                or meta['contract'] != row['contract'] or meta['contract']['execution_policy'] != policy
                or meta['tensors_sha256'] != row['tensor_sha256']
                or tensor_signature(values['input_gradient']) != meta['scientific']['input_gradient']
                or float(values['clean_loss']) != meta['scientific']['clean_loss']):
            raise ValueError('Observed worker contract/tensor signatures changed.')
        observed[policy] = {'contract': meta['contract'], 'gradient': values['input_gradient'], 'loss': values['clean_loss']}
    return observed


def verify_identity(runtime, data, pixels, weights, targets, masks, observed):
    baseline = observed['cudnn-fp32']['contract']
    current = {'runtime': runtime, 'data': data, 'pixels': tensor_signature(pixels), 'weights_sha256': weights,
               'targets': tensor_signature(targets)}
    if any(current[k] != baseline[k] for k in current):
        raise ValueError('Current runtime/data/weights/targets differ from the observed failed run.')
    actual = [[tensor_signature(m.context), tensor_signature(m.target)] for m in masks]
    if actual != baseline['masks'][:len(masks)] or baseline['mask_seeds'][0] != 0:
        raise ValueError('Mask bank differs from the observed failed run.')
    # Only the previously declared context dispatch may differ between sources.
    native = copy.deepcopy(observed['native-fp32']['contract'])
    original = copy.deepcopy(baseline)
    for contract in (native, original):
        contract.pop('execution_policy')
        execution = contract['runtime']['model_execution']
        del execution['policy'], execution['native_backward'], execution['context_patch']['module']
    if native != original:
        raise ValueError('Observed backends changed more than context dispatch.')


def measure(model, pixels, targets, masks, *, observed=None, retain=lambda name, values: None,
            progress=lambda stage: None):
    """Two token sources × two backwards, plus independent mathematical oracles."""
    guard = ExecutionGuard(model, pixels.device.type, target_versions=True)
    before = state_digest(model)
    target_signature = tensor_signature(targets)
    original = model.encoder.patch_embed
    patches = {'cudnn-fp32': original, 'native-fp32': NativePatchEmbed3D(original)}
    tokens, tails, vjps, oracles = {}, {}, {}, {}
    result = {'order': {'token_sources': list(POLICIES), 'backwards_per_source': list(POLICIES)},
              'tail_intervention': 'distinct preserve-format patch leaf per family; unchanged transformer/predictor/L1',
              'training_or_backend_adoption': False, 'comparisons': {}, 'reconstruction': {}}
    for source in POLICIES:
        progress('capture_tokens_and_tail_' + source)
        with torch.no_grad():
            tokens[source] = patches[source](model.normalize(pixels)).detach()
        tails[source] = tail_derivative(model, pixels, targets, masks, tokens[source])
        retain('tail-' + source, {'tokens': tokens[source].cpu(), **tails[source]})
        progress('fp64_analytical_oracle_' + source)
        oracles[source] = fp64_reference(model, pixels, tails[source]['cotangents'])
        retain('oracle-' + source, oracles[source])
        result['comparisons'][source] = {'tokens_vs_fp64': agreement(tokens[source], oracles[source]['tokens'], PATCH_TOLERANCES['output']),
                                         'fixed_cotangent_vjps': {}}
        for backend in POLICIES:
            progress('fixed_cotangent_' + source + '_backward_' + backend)
            vjp = pixel_vjp(model, patches[backend], pixels, tails[source]['cotangents'])
            vjps[source, backend] = vjp
            retain('vjp-' + source + '-via-' + backend, {'raw_pixel_vjp': vjp})
            result['comparisons'][source]['fixed_cotangent_vjps'][backend] = difference(vjp, oracles[source]['raw_pixel_vjp'])
        result['comparisons'][source]['same_cotangent_cross_backend'] = difference(vjps[source, 'native-fp32'], vjps[source, 'cudnn-fp32'])
        if observed is not None:
            result['reconstruction'][source] = {
                'loss': agreement(tails[source]['loss'], observed[source]['loss'], TOLERANCES['clean_loss']),
                'input_gradient': difference(vjps[source, source], observed[source]['gradient'])}
    # Repeat the identical supplied original tokens, to detect state/order drift.
    progress('repeat_identical_token_tail_control')
    repeat = tail_derivative(model, pixels, targets, masks, tokens['cudnn-fp32'])
    retain('tail-original-repeat', repeat)
    result['identical_token_tail_exact'] = all(torch.equal(a, b) for key in ('cotangents', 'predictions', 'residuals')
                                              for a, b in zip(tails['cudnn-fp32'][key], repeat[key])) and torch.equal(repeat['loss'], tails['cudnn-fp32']['loss'])
    result['residual_changes'] = residual_changes(tails['cudnn-fp32']['residuals'], tails['native-fp32']['residuals'])
    result['incoming_derivatives'] = [difference(n, c) for n, c in zip(tails['native-fp32']['cotangents'], tails['cudnn-fp32']['cotangents'])]
    result['forward_tokens'] = difference(tokens['native-fp32'], tokens['cudnn-fp32'])
    cc, nc, nn = (vjps['cudnn-fp32', 'cudnn-fp32'].double(), vjps['native-fp32', 'cudnn-fp32'].double(),
                  vjps['native-fp32', 'native-fp32'].double())
    result['decomposition_l2'] = {'total': float(torch.linalg.vector_norm(nn - cc)),
        'incoming_cotangent_effect_at_cudnn_backward': float(torch.linalg.vector_norm(nc - cc)),
        'backward_effect_at_native_cotangent': float(torch.linalg.vector_norm(nn - nc)),
        'note': 'Effects are vectors and their norms need not add. Descriptive attribution, not backend admission.'}
    guard.check()
    if (model.encoder.patch_embed is not original or state_digest(model) != before
            or tensor_signature(targets) != target_signature or any(p.grad is not None for p in model.parameters())):
        raise RuntimeError('Intervention changed model/targets or accumulated parameter gradients.')
    result['preservation_checks'] = True
    result['faithful_reconstruction'] = observed is not None and all(v[k]['passed'] for v in result['reconstruction'].values() for k in ('loss', 'input_gradient'))
    result['fixed_cotangent_oracles_agree'] = all(v['passed'] for row in result['comparisons'].values() for v in row['fixed_cotangent_vjps'].values())
    result['forward_oracles_agree'] = all(row['tokens_vs_fp64']['passed'] for row in result['comparisons'].values())
    # No automatic cause claim, even if this narrowly scoped control succeeds.
    result['control_checks_passed'] = (result['identical_token_tail_exact'] and result['fixed_cotangent_oracles_agree']
                                      and result['forward_oracles_agree'] and (observed is None or result['faithful_reconstruction']))
    return result


def worker(args, root):
    started = time.monotonic()
    result = {'schema_version': 1, 'diagnostic_only': True, 'research_result': False, 'backend_adopted': False,
              'prior_attempt_passed': False, 'tiny_fixture': args.tiny, 'complete': False, 'control_checks_passed': False,
              'artifacts': {}, 'full_model_optimizer_or_recovery_validated': False}
    def progress(stage):
        result['stage'] = stage
        result['elapsed_seconds'] = time.monotonic() - started
        write_json(args.result, result)
        print('Derivative control:', stage, flush=True)
    def retain(name, values):
        path = args.result.parent / (name + '.pt')
        with path.open('xb') as handle:
            torch.save(values, handle)
        result['artifacts'][name] = {'file': path.name, 'sha256': file_hash(path)}
        write_json(args.result, result)
    try:
        progress('configure_and_bind_observed_sources')
        configure_execution('cudnn-fp32', device=args.device)
        torch.set_num_threads(4)
        previous = provenance(root, args.upstream)
        result['provenance'] = {**previous, 'control_source_sha256': {n: file_hash(Path(root) / n) for n in SOURCES}}
        observed = None
        if not args.tiny:
            if str(torch.__version__).split('+')[0] != '2.8.0' or not torch.cuda.is_available():
                raise ValueError('Existing CUDA PyTorch 2.8.0 required; no install/fallback.')
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(.9, 7 * 2**30 / total))
            observed = load_observed(args.observed, args.reference_run, previous)
            result['observed_spec_sha256'] = file_hash(args.observed)
            progress('actual_native_backward_kernel_probe')
            def retain_probe(evidence):
                result['native_backward_probe'] = evidence
                write_json(args.result, result)
            native_backward_probe(args.device, on_evidence=retain_probe)
        torch.manual_seed(20261021)
        progress('load_original_model_and_first_training_clip')
        if args.tiny:
            model, clean, masks_for = tiny_fixture(args.upstream)
            data = {'fixture': 'tiny-upstream-model-and-generated-training-pixels-v1'}
        else:
            plan_path = Path(root) / 'configs/budget_study_plan.yaml'
            plan = yaml.safe_load(plan_path.read_text())
            study_path = Path(root) / plan['parent_study']
            study = yaml.safe_load(study_path.read_text())
            manifest_key = study['data']['manifest_directory'] + '/train.csv'
            if (file_hash(study_path) != plan['source_sha256'][plan['parent_study']]
                    or args.manifest.name != 'train.csv' or file_hash(args.manifest) != plan['source_sha256'][manifest_key]):
                raise ValueError('Pinned training manifest/study differs.')
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
        targets = model.clean_targets(clean)
        masks = [m.to(args.device) for m in masks_for(0)]
        runtime = runtime_contract(args.device, model=model)
        weights = state_digest(model)
        result['identity'] = {'runtime': runtime, 'data': data, 'pixels': tensor_signature(clean),
                              'weights_sha256': weights, 'targets': tensor_signature(targets),
                              'masks': [[tensor_signature(m.context), tensor_signature(m.target)] for m in masks],
                              'allocator_limit_gib': None if args.tiny else 7, 'mask_seed': 0,
                              'input_vjp_tolerances': TOLERANCES['input_gradient'], 'patch_output_tolerances': PATCH_TOLERANCES['output']}
        if observed is not None:
            verify_identity(runtime, data, clean, weights, targets, masks, observed)
        result['measurements'] = measure(model, clean, targets, masks, observed=observed, retain=retain, progress=progress)
        result.update(complete=True, control_checks_passed=result['measurements']['control_checks_passed'], stage='complete',
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None)
        if not result['control_checks_passed']:
            raise RuntimeError('Derivative control checks rejected; preserve all evidence; no tolerance changed.')
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result['wall_seconds'] = time.monotonic() - started
        write_json(args.result, result)


def run(args, root):
    args.output.mkdir(parents=True, exist_ok=False)
    session = {'schema_version': 1, 'complete': False, 'control_checks_passed': False,
               'research_result': False, 'backend_adopted': False, 'max_seconds': args.max_seconds}
    command = [sys.executable, '-u', str(Path(root) / 'scripts/check_gradient_attribution.py'),
               '--diagnostic', '--worker', '--device', args.device, '--upstream', str(args.upstream.resolve()),
               '--result', str(args.output / 'control.json')]
    if args.tiny:
        command += ['--tiny']
    else:
        for name in ('checkpoint', 'clips', 'manifest', 'observed', 'reference_run'):
            command += ['--' + name.replace('_', '-'), str(getattr(args, name).resolve())]
    write_json(args.output / 'session.json', session)
    try:
        session['task'] = run_bounded(command, args.output / 'control.log', args.max_seconds)
        path = args.output / 'control.json'
        if path.exists():
            value = json.loads(path.read_text())
            session.update(result_file=path.name, result_sha256=file_hash(path), complete=value['complete'],
                           control_checks_passed=value['control_checks_passed'])
        if session['task']['status'] != 'ok' or not session['complete'] or not session['control_checks_passed']:
            raise RuntimeError('Derivative control rejected/timed out; inspect control.json/control.log.')
    finally:
        write_json(args.output / 'session.json', session)
    return session
