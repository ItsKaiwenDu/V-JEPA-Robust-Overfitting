"""Strict, bounded comparisons of execution options for the original Conv3d.

Synthetic diagnostics only: this module never patches a research model.
"""

from contextlib import nullcontext
import copy
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch

from models.cuda_profile import (file_hash, run_bounded, runtime, tensor_signature,
                                 UPSTREAM_SOURCES, write_json)
from models.runtime import configure_cuda_reproducibility, synchronize
from models.vjepa2 import UPSTREAM_COMMIT, use_upstream

VARIANTS = ('contiguous', 'channels-last-3d', 'native')
MODES = ('input', 'parameters')
SOURCES = ('models/cuda_convolution.py', 'scripts/compare_cuda_convolution.py',
           'models/cuda_profile.py', 'models/runtime.py', 'models/vjepa2.py')
# Fixed before CUDA execution. A failure stays a failure, never widens these.
TOLERANCES = {'output': {'rtol': 1e-4, 'atol': 2e-5},
              'input': {'rtol': 1e-4, 'atol': 2e-5},
              'weight': {'rtol': 1e-4, 'atol': 2e-4},
              'bias': {'rtol': 1e-4, 'atol': 1e-4}}


def analytical_reference(x, weight, bias, cotangent):
    """CPU FP64 patch matrix and analytic derivatives, without conv/autograd.

Only nonoverlapping unpadded patches; callers use the fixed upstream geometry.
The inverse permutation explicitly reconstructs the pixel-gradient layout.
"""
    x, weight, bias, cotangent = (v.detach().cpu().double() for v in
                                 (x, weight, bias, cotangent))
    batch, channels, frames, height, width = x.shape
    outputs, wc, kt, kh, kw = weight.shape
    if channels != wc or frames % kt or height % kh or width % kw:
        raise ValueError('Invalid nonoverlapping patch geometry.')
    nt, nh, nw = frames // kt, height // kh, width // kw
    if tuple(cotangent.shape) != (batch, nt * nh * nw, outputs):
        raise ValueError('Cotangent shape differs from the patch output.')
    patches = x.unfold(2, kt, kt).unfold(3, kh, kh).unfold(4, kw, kw)
    matrix = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, channels * kt * kh * kw)
    w = weight.flatten(1)
    g = cotangent.reshape(-1, outputs)
    prediction = (matrix @ w.T + bias).reshape_as(cotangent)
    dx = (g @ w).reshape(batch, nt, nh, nw, channels, kt, kh, kw)
    dx = dx.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape_as(x)
    return {'output': prediction, 'input': dx,
            'weight': (g.T @ matrix).reshape_as(weight), 'bias': g.sum(0)}


def agreement(actual, expected, tolerance):
    """Finite FP64 error accounting with fixed elementwise tolerances."""
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError('Invalid or nonfinite comparison tensors.')
    error = (actual - expected).abs()
    allowed = tolerance['atol'] + tolerance['rtol'] * expected.abs()
    return {'passed': bool((error <= allowed).all()), 'shape': list(actual.shape),
            'max_absolute_error': float(error.max()),
            'max_tolerance_ratio': float((error / allowed).max()),
            'failed_elements': int((error > allowed).sum()),
            'relative_l2_error': float(torch.linalg.vector_norm(error) /
                                      torch.linalg.vector_norm(expected).clamp_min(1e-300)),
            **tolerance}


def execution_context(variant):
    if variant == 'native':
        # Cover both forward and backward; explicitly retain strict/precision flags.
        return torch.backends.cudnn.flags(enabled=False, benchmark=False,
                                         deterministic=True, allow_tf32=torch.backends.cudnn.allow_tf32)
    return nullcontext()


def make_operands(upstream, tiny):
    use_upstream(upstream)
    patch = importlib.import_module('src.models.utils.patch_embed').PatchEmbed3D
    torch.manual_seed(20261021)
    frames, crop, size, width = (4, 16, 4, 16) if tiny else (16, 256, 16, 1024)
    module = patch(patch_size=size, tubelet_size=2, embed_dim=width)
    x = torch.randn(1, 3, frames, crop, crop)
    tokens = (frames // 2) * (crop // size) ** 2
    cotangent = torch.randn((1, tokens, width), generator=torch.Generator().manual_seed(73))
    return module, x, cotangent


def prepare_execution(original, x_cpu, cotangent_cpu, variant, mode, device):
    if variant not in VARIANTS or mode not in MODES:
        raise ValueError('Unknown convolution variant or gradient mode.')
    module = copy.deepcopy(original).to(device).eval().requires_grad_(mode == 'parameters')
    x = x_cpu.to(device).detach().requires_grad_(True)
    cotangent = cotangent_cpu.to(device)
    synchronize(device)
    start = time.perf_counter()
    if variant == 'channels-last-3d':
        module.to(memory_format=torch.channels_last_3d)
    synchronize(device)
    weight_layout_setup_seconds = time.perf_counter() - start
    # Include conversion in every forward and its derivative in input backward.
    def operation():
        value = x.contiguous(memory_format=torch.channels_last_3d) if variant == 'channels-last-3d' else x
        return module(value)
    leaves = [x]
    if mode == 'parameters':
        leaves.extend((module.proj.weight, module.proj.bias))
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        value = x.contiguous(memory_format=torch.channels_last_3d) if variant == 'channels-last-3d' else x
    synchronize(device)
    input_conversion_seconds = time.perf_counter() - start
    layout = {'input_leaf_stride': list(x.stride()), 'convolution_input_stride': list(value.stride()),
              'weight_stride': list(module.proj.weight.stride()),
              'weight_layout_setup_seconds': weight_layout_setup_seconds,
              'standalone_input_conversion_seconds': input_conversion_seconds,
              'input_conversion_included_in_forward': variant == 'channels-last-3d',
              'state_keys': list(module.state_dict()),
              'parameters': {n: tensor_signature(p) for n, p in module.named_parameters()}}
    return operation, leaves, cotangent, layout


def evaluate(operation, leaves, cotangent, variant, device, progress=lambda stage: None):
    with execution_context(variant):
        progress('forward')
        synchronize(device)
        start = time.perf_counter()
        y = operation()
        synchronize(device)
        forward = time.perf_counter() - start
        progress('backward')
        start = time.perf_counter()
        grads = torch.autograd.grad(y, leaves, grad_outputs=cotangent)
        synchronize(device)
        backward = time.perf_counter() - start
    values = {'output': y.detach().cpu(), **{n: g.detach().cpu() for n, g in
              zip(('input', 'weight', 'bias'), grads)}}
    return values, {'forward_seconds': forward, 'backward_seconds': backward,
                    'total_seconds': forward + backward}


def observed_runtime(device, variant):
    with execution_context(variant):
        return {**runtime(device, 'strict'), 'cudnn_enabled': torch.backends.cudnn.enabled}


def provenance(root, upstream):
    return {'source_sha256': {n: file_hash(Path(root) / n) for n in SOURCES},
            'upstream_sha256': {n: file_hash(Path(upstream) / n) for n in UPSTREAM_SOURCES}}


def worker(args, root):
    configure_cuda_reproducibility()
    # Explicit diagnostic control, isolated in this fresh worker process.
    torch.backends.cudnn.allow_tf32 = args.cudnn_tf32 == 'on'
    torch.set_num_threads(4)
    if args.device == 'cuda':
        if not torch.cuda.is_available() or str(torch.__version__).split('+')[0] != '2.8.0':
            raise RuntimeError('CUDA PyTorch 2.8.0 required; no fallback or install.')
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.9, args.max_memory_gib * 2**30 / total))
    def progress(stage):
        write_json(args.result.with_suffix('.progress.json'),
                   {'variant': args.variant, 'mode': args.mode, 'stage': stage, 'unix_seconds': time.time()})
    progress('construct_operands')
    original, x, cotangent = make_operands(args.upstream, args.tiny)
    operands = {'input': tensor_signature(x), 'cotangent': tensor_signature(cotangent),
                'parameters': {n: tensor_signature(p) for n, p in original.named_parameters()}}
    contract = {'variant': args.variant, 'mode': args.mode, 'operands': operands,
                'cudnn_tf32_policy': args.cudnn_tf32,
                'scale': 'tiny_cpu_fixture' if args.tiny else 'representative',
                'tolerances': copy.deepcopy(TOLERANCES), 'original_conv3d': True,
                'runtime': observed_runtime(args.device, args.variant),
                'allocator_limit_gib': args.max_memory_gib if args.device == 'cuda' else None,
                'warmup': args.warmup, 'repeats': args.repeats}
    result = {'schema_version': 2, 'diagnostic_only': True, 'research_result': False,
              'finished': False, 'passed': False, 'contract': contract,
              'provenance': provenance(root, args.upstream), 'profiler': {'requested': not args.no_profiler}}
    write_json(args.result, result)
    try:
        progress('cpu_fp64_analytical_reference')
        oracle = analytical_reference(x, original.proj.weight, original.proj.bias, cotangent)
        result['oracle'] = {'kind': 'CPU FP64 unfolded patches with analytic derivatives; no conv or autograd',
                            'signatures': {n: tensor_signature(v) for n, v in oracle.items()}}
        op, leaves, cot, layout = prepare_execution(original, x, cotangent, args.variant, args.mode, args.device)
        assert layout['parameters'] == operands['parameters']
        result['layout'] = layout
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        actual, _ = evaluate(op, leaves, cot, args.variant, args.device,
                             lambda stage: progress('validate_candidate_' + stage))
        if args.variant == 'contiguous':
            reference = actual
        else:
            ref_op, ref_leaves, ref_cot, _ = prepare_execution(original, x, cotangent, 'contiguous', args.mode, args.device)
            reference, _ = evaluate(ref_op, ref_leaves, ref_cot, 'contiguous', args.device,
                                    lambda stage: progress('validate_conv_reference_' + stage))
            del ref_op, ref_leaves, ref_cot
        result['validation'] = {
            'candidate_vs_oracle': {n: agreement(v, oracle[n], TOLERANCES[n]) for n, v in actual.items()},
            'conv_reference_vs_oracle': {n: agreement(v, oracle[n], TOLERANCES[n]) for n, v in reference.items()},
            'candidate_vs_conv_reference': {n: agreement(v, reference[n], TOLERANCES[n]) for n, v in actual.items()}}
        validation_passed = all(v['passed'] for group in result['validation'].values() for v in group.values())
        result['validation_passed'] = validation_passed
        result['validation_signatures'] = {
            'candidate': {n: tensor_signature(v) for n, v in actual.items()},
            'conv_reference': {n: tensor_signature(v) for n, v in reference.items()}}
        # A numerically rejected option is never benchmarked or given a speed ratio.
        if not validation_passed:
            result.update(finished=True, timing_skipped_reason='Numerical validation failed at fixed tolerances.')
            return result
        validation_signature = result['validation_signatures']['candidate']
        result['signature'] = validation_signature
        for _ in range(args.warmup):
            evaluate(op, leaves, cot, args.variant, args.device, lambda stage: progress('warmup_' + stage))
        result['iterations'] = []
        signatures = []
        for _ in range(args.repeats):
            values, timing = evaluate(op, leaves, cot, args.variant, args.device,
                                      lambda stage: progress('measured_' + stage))
            signatures.append({n: tensor_signature(v) for n, v in values.items()})
            result['iterations'].append(timing)
        result['within_process_exact'] = all(v == validation_signature for v in signatures)
        result['median_seconds'] = {k: statistics.median(v[k] for v in result['iterations'])
                                   for k in ('forward_seconds', 'backward_seconds', 'total_seconds')}
        write_json(args.result, result)
        if not args.no_profiler:
            progress('operator_profiler')
            try:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if args.device == 'cuda':
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
                    with execution_context(args.variant):
                        with torch.profiler.record_function('convolution.forward'):
                            y = op()
                        with torch.profiler.record_function('convolution.backward'):
                            torch.autograd.grad(y, leaves, grad_outputs=cot)
                        synchronize(args.device)
                rows = [{'name': e.key, 'count': e.count, 'self_cpu_us': e.self_cpu_time_total,
                         'self_device_us': e.self_device_time_total, 'total_device_us': e.device_time_total}
                        for e in prof.key_averages()]
                result['profiler'] = {'requested': True, 'available': True,
                                     'cuda_events_observed': any(v['self_device_us'] > 0 for v in rows),
                                     'top_device': sorted(rows, key=lambda v: v['self_device_us'], reverse=True)[:15],
                                     'top_cpu': sorted(rows, key=lambda v: v['self_cpu_us'], reverse=True)[:15],
                                     'profiled_iteration_excluded_from_medians': True}
            except Exception as exc:
                result['profiler'] = {'requested': True, 'available': False,
                                     'error_type': type(exc).__name__, 'error': str(exc)}
        result.update(finished=True, passed=result['within_process_exact'])
        return result
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if args.device == 'cuda':
            result['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        write_json(args.result, result)
        progress('complete' if result['finished'] else 'failed')


def summarize(directory, tasks, variants, expected_provenance, *, cudnn_tf32='on'):
    if cudnn_tf32 not in ('on', 'off'):
        raise ValueError('Unknown declared cuDNN TF32 policy.')
    comparisons = {}
    for variant in variants:
        comparisons[variant] = {}
        for mode in MODES:
            selected = [t for t in tasks if t['variant'] == variant and t['mode'] == mode]
            result = {'complete': False, 'qualified': False}
            comparisons[variant][mode] = result
            try:
                if len(selected) != 2 or {t['replay'] for t in selected} != {0, 1} or any(t['status'] != 'ok' for t in selected):
                    continue
                values = []
                for task in selected:
                    path = (Path(directory) / task['result']).resolve()
                    if not path.is_relative_to(Path(directory).resolve()) or file_hash(path) != task['result_sha256']:
                        raise ValueError('Worker path/checksum mismatch.')
                    value = json.loads(path.read_text())
                    contract = value['contract']
                    if (value['schema_version'] != 2 or value['provenance'] != expected_provenance or contract['variant'] != variant
                            or contract['mode'] != mode or not value['diagnostic_only'] or value['research_result']
                            or contract['tolerances'] != TOLERANCES or not contract['original_conv3d']
                            or contract['cudnn_tf32_policy'] != cudnn_tf32):
                        raise ValueError('Worker identity, tolerance, or source mismatch.')
                    rt = contract['runtime']
                    required = {'policy': 'strict', 'deterministic_algorithms': True, 'warn_only': False,
                                'cudnn_deterministic': True, 'cudnn_benchmark': False,
                                'cudnn_tf32': cudnn_tf32 == 'on', 'matmul_tf32': False, 'matmul_precision': 'highest',
                                'cublas_workspace_config': ':4096:8', 'cudnn_enabled': variant != 'native'}
                    if any(rt[k] != expected for k, expected in required.items()):
                        raise ValueError('Strict runtime/precision contract mismatch.')
                    if not value['finished']:
                        continue
                    values.append(value)
                if len(values) != 2:
                    continue
                result['complete'] = True
                same = values[0]['contract'] == values[1]['contract'] and values[0]['oracle'] == values[1]['oracle']
                result['same_operands_runtime_and_oracle'] = same
                names = {'output', 'input', 'weight', 'bias'} if mode == 'parameters' else {'output', 'input'}
                valid = all(set(v['validation']) == {'candidate_vs_oracle', 'conv_reference_vs_oracle', 'candidate_vs_conv_reference'}
                            and all(set(group) == names and all(item['passed'] and item['failed_elements'] == 0
                                and math.isfinite(item['max_tolerance_ratio']) and item['max_tolerance_ratio'] <= 1
                                and all(item[k] == TOLERANCES[n][k] for k in ('rtol', 'atol'))
                                for n, item in group.items()) for group in v['validation'].values()) for v in values)
                result['numerical_validation_passed'] = valid
                exact = valid and same and all(v.get('within_process_exact', False) for v in values)
                exact = exact and values[0].get('signature') == values[1].get('signature')
                result['observed_exact_replay'] = exact
                result['qualified'] = exact
                if exact:
                    for v in values:
                        if len(v['iterations']) != v['contract']['repeats']:
                            raise ValueError('Timed iteration count mismatch.')
                        for key, med in v['median_seconds'].items():
                            if not math.isfinite(med) or med <= 0 or med != statistics.median(row[key] for row in v['iterations']):
                                raise ValueError('Invalid timing median.')
                    result['median_seconds'] = {k: statistics.median(v['median_seconds'][k] for v in values)
                                               for k in ('forward_seconds', 'backward_seconds', 'total_seconds')}
                    result['operands'] = values[0]['contract']['operands']
                    result['runtime'] = values[0]['contract']['runtime']
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result.update(complete=False, qualified=False, error=str(exc))
    for variant in variants:
        for mode in MODES:
            value = comparisons[variant][mode]
            ref = comparisons.get('contiguous', {}).get(mode, {})
            if value['qualified'] and ref.get('qualified'):
                runtime_a, runtime_b = (dict(v['runtime']) for v in (value, ref))
                runtime_a.pop('cudnn_enabled'); runtime_b.pop('cudnn_enabled')
                comparable = value['operands'] == ref['operands'] and runtime_a == runtime_b
                value['same_operands_precision_and_shared_runtime'] = comparable
                if comparable:
                    value['candidate_to_reference_total_ratio'] = value['median_seconds']['total_seconds'] / ref['median_seconds']['total_seconds']
    return comparisons


def run_comparison(args, root, started):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    upstream = use_upstream(args.upstream)
    variants = args.variants or list(VARIANTS)
    identity = provenance(root, upstream)
    session = {'schema_version': 2, 'diagnostic_only': True, 'research_result': False,
               'budget_approved': False, 'full_model_validated': False, 'model_changed': False,
               'device': args.device, 'tiny_fixture': args.tiny, 'tolerances': TOLERANCES,
               'cudnn_tf32_policy': args.cudnn_tf32,
               'provenance': identity, 'upstream_commit': UPSTREAM_COMMIT,
               'bounds': {'worker_seconds': args.worker_seconds, 'overall_seconds': args.max_seconds,
                          'allocator_gib': args.max_memory_gib},
               'started_unix_seconds': time.time(), 'tasks': [], 'complete': False,
               'all_checks_passed': False, 'finished': False, 'selected_variant': None}
    write_json(output / 'session.json', session)
    try:
        if args.device == 'cuda' and (not torch.cuda.is_available() or str(torch.__version__).split('+')[0] != '2.8.0'):
            raise RuntimeError('CUDA PyTorch 2.8.0 required; no fallback or install.')
        for variant in variants:
            for mode in MODES:
                for replay in (0, 1):
                    remaining = args.max_seconds - (time.monotonic() - started)
                    row = {'variant': variant, 'mode': mode, 'replay': replay}
                    relative = f'workers/{variant}/{mode}-{replay}.json'
                    path = output / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if remaining <= 0:
                        row['status'] = 'not_started_deadline'
                    else:
                        print(f'Convolution: {variant}; gradients: {mode}; replay: {replay}', flush=True)
                        command = [sys.executable, '-u', str(Path(root) / 'scripts/compare_cuda_convolution.py'),
                                   '--worker', '--variant', variant, '--mode', mode, '--device', args.device,
                                   '--upstream', str(upstream), '--result', str(path),
                                   '--warmup', str(args.warmup), '--repeats', str(args.repeats),
                                   '--cudnn-tf32', args.cudnn_tf32,
                                   '--max-memory-gib', str(args.max_memory_gib)]
                        if args.tiny:
                            command.append('--tiny')
                        if args.no_profiler:
                            command.append('--no-profiler')
                        session['active_task'] = {**row, 'result': relative}
                        write_json(output / 'session.json', session)
                        before = time.monotonic()
                        row.update(run_bounded(command, output / 'logs' / f'{variant}-{mode}-{replay}.log',
                                               min(remaining, args.worker_seconds)))
                        row['worker_wall_seconds'] = time.monotonic() - before
                        if path.exists():
                            row.update(result=relative, result_sha256=file_hash(path))
                        elif row['status'] == 'ok':
                            row['status'] = 'missing_result'
                    session.pop('active_task', None)
                    session['tasks'].append(row)
                    write_json(output / 'session.json', session)
    except BaseException as exc:
        session.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        session['finished'] = True
        session['wall_seconds'] = time.monotonic() - started
        session['comparisons'] = summarize(output, session['tasks'], variants, identity,
                                            cudnn_tf32=args.cudnn_tf32)
        session['complete'] = all(v['complete'] for modes in session['comparisons'].values() for v in modes.values())
        session['all_checks_passed'] = session['complete'] and all(v['qualified'] for modes in session['comparisons'].values() for v in modes.values())
        write_json(output / 'session.json', session)
    print('Comparison finished. Inspect session.json; rejected or incomplete candidates are not passes.', flush=True)
    return session
