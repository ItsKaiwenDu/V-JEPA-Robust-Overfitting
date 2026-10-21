"""Bounded component diagnostics using pinned upstream code and synthetic tensors.

No weights, video pixels, optimizer, or research trajectory are loaded.
"""

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from models.runtime import configure_cuda_reproducibility, device_metadata, synchronize
from models.vjepa2 import PRETRAIN_CONFIG, UPSTREAM_COMMIT, sample_masks, use_upstream

CASES = ('patch-conv', 'encoder-mask-0', 'encoder-mask-1', 'predictor-mask-0',
         'predictor-mask-1', 'predictor-sort', 'sdpa-encoder-0', 'sdpa-encoder-1',
         'sdpa-predictor', 'rope-encoder-0', 'rope-encoder-1', 'rope-predictor')
POLICIES = ('default', 'strict')
# First primary-attack mask seed from the failed pilot, reduced for upstream's counter.
MASK_SEED = 8913831894998997572 % (2**31 - 1)
SOURCES = ('models/cuda_profile.py', 'models/runtime.py', 'models/vjepa2.py',
           'scripts/profile_cuda.py')
UPSTREAM_SOURCES = (PRETRAIN_CONFIG, 'src/masks/utils.py',
                    'src/masks/multiseq_multiblock3d.py',
                    'src/models/utils/patch_embed.py', 'src/models/utils/modules.py',
                    'src/models/predictor.py')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Keep committed progress on interruption without an optional dependency."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.pending')
    with temp.open('w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def tensor_signature(value):
    value = value.detach().cpu().contiguous()
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite diagnostic output/gradient.')
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            'abs_max': float(value.abs().max())}


def masks_and_shape(upstream, tiny=False):
    if tiny:
        order = torch.randperm(32, generator=torch.Generator().manual_seed(17)).unsqueeze(0)
        masks = [(order[:, :8], order[:, 8:]), (order[:, :24], order[:, 24:])]
        return masks, {'frames': 4, 'crop': 16, 'patch': 4, 'tokens': 32,
                       'encoder_dim': 128, 'encoder_heads': 2,
                       'predictor_dim': 64, 'predictor_heads': 2}
    masks = [(p.context, p.target) for p in sample_masks(upstream, batch_size=1, seed=MASK_SEED)]
    return masks, {'frames': 16, 'crop': 256, 'patch': 16, 'tokens': 2048,
                   'encoder_dim': 1024, 'encoder_heads': 16,
                   'predictor_dim': 384, 'predictor_heads': 12}


def make_case(case, upstream, device, *, tiny=False, parameter_gradients=False):
    """Return the actual upstream component, its leaves, and a fixed cotangent."""
    if case not in CASES:
        raise ValueError('Unknown component case.')
    use_upstream(upstream)
    pairs, shape = masks_and_shape(upstream, tiny)
    # Construct on CPU, identically in every policy/process; then transfer.
    torch.manual_seed(20261021)
    modules = importlib.import_module('src.models.utils.modules')
    apply_masks = importlib.import_module('src.masks.utils').apply_masks
    patch = importlib.import_module('src.models.utils.patch_embed').PatchEmbed3D
    family = int(case[-1]) if case.endswith(('-0', '-1')) else 0
    context, target = (v.to(device) for v in pairs[family])
    width = shape['predictor_dim'] if 'predictor' in case else shape['encoder_dim']
    heads = shape['predictor_heads'] if 'predictor' in case else shape['encoder_heads']
    n = shape['tokens'] if 'predictor' in case else context.shape[1]
    module = None
    operands = {}

    if case == 'patch-conv':
        module = patch(patch_size=shape['patch'], tubelet_size=2, embed_dim=shape['encoder_dim'])
        x = torch.randn(1, 3, shape['frames'], shape['crop'], shape['crop'])
        operation = lambda: module(x)
    elif '-mask-' in case:
        x = torch.randn(1, shape['tokens'], width)
        indices = target if 'predictor' in case else context
        operands['mask'] = tensor_signature(indices)
        operation = lambda: apply_masks(x, [indices])
    elif case == 'predictor-sort':
        x = torch.randn(1, shape['tokens'], width)
        indices = torch.argsort(torch.cat((context, target), dim=1), dim=1)
        operands['sort_indices'] = tensor_signature(indices)
        # Same indexing expression used before the upstream predictor blocks.
        operation = lambda: torch.stack([x[i, row, :] for i, row in enumerate(indices)], dim=0)
    elif case.startswith('sdpa-'):
        # Preserve the interleaved qkv layout produced by upstream Linear.
        x = torch.randn(1, n, 3 * width)
        def operation():
            q, k, v = x.unflatten(-1, (3, heads, width // heads)).permute(2, 0, 3, 1, 4)
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
    elif case.startswith('rope-'):
        module = modules.RoPEAttention(dim=width, num_heads=heads, qkv_bias=True,
                                      use_sdpa=True, grid_size=shape['crop']//shape['patch'])
        indices = torch.arange(shape['tokens'], device=device).unsqueeze(0) if 'predictor' in case else context
        operands['position_indices'] = tensor_signature(indices)
        x = torch.randn(1, n, width)
        operation = lambda: module(x, mask=indices)
    else:
        raise AssertionError(case)

    x = x.to(device).requires_grad_(True)
    leaves = [x]
    if module is not None:
        module.to(device).eval().requires_grad_(parameter_gradients)
        operands['parameters'] = {name: tensor_signature(p) for name, p in module.named_parameters()}
        if parameter_gradients:
            leaves.extend(module.parameters())
    operands['input'] = tensor_signature(x)
    # Shapes are explicit so a slow forward cannot hide in unmeasured setup.
    if case=='patch-conv': output_shape=(1,shape['tokens'],shape['encoder_dim'])
    elif '-mask-' in case: output_shape=(1,indices.shape[1],width)
    elif case.startswith('sdpa-'): output_shape=(1,heads,n,width//heads)
    else: output_shape=(1,n,width)
    cotangent = torch.randn(output_shape, generator=torch.Generator().manual_seed(73)).to(device)
    operands['cotangent'] = tensor_signature(cotangent)
    metadata = {'case': case, 'scale': 'tiny_cpu_fixture' if tiny else 'representative',
                'shape': shape, 'output_shape': list(output_shape),
                'context_tokens': context.shape[1], 'target_tokens': target.shape[1],
                'mask_seed': None if tiny else MASK_SEED, 'operands': operands,
                'parameter_gradients': parameter_gradients,
                'scope': 'Isolated synthetic component; no full-model or research claim.'}
    return operation, leaves, cotangent, metadata


def runtime(device, policy):
    result = {'policy': policy, 'torch': str(torch.__version__), 'python': platform.python_version(),
              'threads': torch.get_num_threads(), 'device': device,
              'packages': {name:importlib.metadata.version(name) for name in ('torch','timm','einops','PyYAML')},
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
              'cudnn_benchmark': torch.backends.cudnn.benchmark,
              'cudnn_deterministic': torch.backends.cudnn.deterministic,
              'cudnn_tf32': torch.backends.cudnn.allow_tf32,
              'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
              'matmul_precision': torch.get_float32_matmul_precision(),
              'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    if device == 'cuda':
        result.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      hardware=device_metadata(device))
    return result


def iteration(operation, leaves, cotangent, device, progress=lambda stage: None):
    progress('forward')
    synchronize(device)
    before = time.perf_counter()
    y = operation()
    synchronize(device)
    forward_seconds = time.perf_counter()-before
    progress('backward')
    before_backward = time.perf_counter()
    grads = torch.autograd.grad(y, leaves, grad_outputs=cotangent)
    synchronize(device)
    backward_seconds = time.perf_counter()-before_backward
    return y, grads, {'forward_seconds': forward_seconds, 'backward_seconds': backward_seconds,
                      'total_seconds': forward_seconds+backward_seconds}


def worker(args):
    if args.policy == 'strict':
        configure_cuda_reproducibility()
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
    torch.set_num_threads(4)
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU fallback is enabled.')
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.9, args.max_memory_gib*2**30/total))
    def progress(stage):
        write_json(args.result.with_suffix('.progress.json'),
                   {'case':args.case,'policy':args.policy,'stage':stage,'unix_seconds':time.time()})
    progress('construct_component')
    operation, leaves, cotangent, metadata = make_case(args.case, args.upstream, args.device,
        tiny=args.tiny, parameter_gradients=args.parameter_gradients)
    provenance = {'source_sha256': {n:file_hash(Path(__file__).resolve().parents[1]/n) for n in SOURCES},
                  'upstream_sha256': {n:file_hash(Path(args.upstream)/n) for n in UPSTREAM_SOURCES}}
    result = {'schema_version': 1, 'diagnostic_only': True, 'research_result': False,
              'provenance': provenance,
              'metadata': metadata, 'runtime': runtime(args.device, args.policy),
              'allocator_limit_gib': args.max_memory_gib if args.device == 'cuda' else None,
              'iterations': [], 'profiler': {'requested': not args.no_profiler}}
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):
        iteration(operation, leaves, cotangent, args.device,lambda stage:progress('warmup_'+stage))
    signatures = []
    for _ in range(args.repeats):
        y, grads, timing = iteration(operation, leaves, cotangent, args.device,lambda stage:progress('measured_'+stage))
        signature = {'output': tensor_signature(y), 'gradients': [tensor_signature(g) for g in grads]}
        if not signature['gradients'][0]['abs_max'] > 0:
            raise ValueError('The diagnostic input gradient is zero.')
        signatures.append(signature)
        result['iterations'].append(timing)
    result['signature'] = signatures[0]
    result['within_process_exact'] = all(s == signatures[0] for s in signatures)
    result['median_seconds'] = {k: statistics.median(r[k] for r in result['iterations'])
                                for k in ('forward_seconds','backward_seconds','total_seconds')}
    # Commit timing/replay before optional CUPTI work, which can be unavailable.
    write_json(args.result, result)
    if not args.no_profiler:
        progress('operator_profiler')
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if args.device == 'cuda': activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
                with torch.profiler.record_function('component.forward'):
                    y = operation()
                with torch.profiler.record_function('component.backward'):
                    torch.autograd.grad(y, leaves, grad_outputs=cotangent)
                synchronize(args.device)
            rows = [{'name': e.key, 'count': e.count, 'input_shapes': e.input_shapes,
                     'self_cpu_us': e.self_cpu_time_total,
                     'self_device_us': e.self_device_time_total,
                     'total_device_us': e.device_time_total}
                    for e in prof.key_averages(group_by_input_shape=True)]
            result['profiler'] = {'requested': True, 'available': True,
                                 'cuda_events_observed': any(r['self_device_us']>0 for r in rows),
                                 'top_device': sorted(rows,key=lambda v:v['self_device_us'],reverse=True)[:25],
                                 'top_cpu': sorted(rows,key=lambda v:v['self_cpu_us'],reverse=True)[:25],
                                 'profiled_iteration_excluded_from_medians': True}
        except Exception as exc:
            result['profiler'] = {'requested': True, 'available': False,
                                 'error_type': type(exc).__name__, 'error': str(exc)}
    if args.device == 'cuda':
        result['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    write_json(args.result, result)
    progress('complete')
    return result


def run_bounded(command, log, seconds):
    """Kill the worker process group on timeout/parent interruption."""
    if seconds <= 0:
        return {'status': 'not_started_deadline'}
    with Path(log).open('x') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        previous = signal.getsignal(signal.SIGTERM)
        def interrupted(signum, frame):
            raise InterruptedError('Component profile parent received SIGTERM.')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            code = process.wait(timeout=seconds)
            return {'status': 'ok' if code == 0 else 'worker_error', 'exit_status': code}
        except subprocess.TimeoutExpired:
            return {'status': 'timeout'}
        finally:
            try:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            except ProcessLookupError:
                pass
            finally:
                signal.signal(signal.SIGTERM, previous)


def summarize(directory, tasks, cases, expected_provenance=None):
    """Timeouts/partial results never count as complete successful comparisons."""
    comparisons = {}
    for case in cases:
        policies = {}
        for policy in POLICIES:
            selected = [t for t in tasks if t['case']==case and t['policy']==policy]
            if len(selected)!=2 or any(t['status']!='ok' for t in selected):
                policies[policy] = {'complete': False}
                continue
            values = []
            try:
                if {t['replay'] for t in selected}!={0,1}:
                    raise ValueError('Duplicate replay IDs.')
                for task in selected:
                    path = (Path(directory)/task['result']).resolve()
                    if not path.is_relative_to(Path(directory).resolve()) or file_hash(path)!=task['result_sha256']:
                        raise ValueError('Worker result checksum or location mismatch.')
                    value = json.loads(path.read_text())
                    if (value['metadata']['case']!=case or value['runtime']['policy']!=policy
                            or not value['diagnostic_only'] or value['research_result']):
                        raise ValueError('Worker result identity mismatch.')
                    if any(not math.isfinite(t) or t<=0 for t in value['median_seconds'].values()):
                        raise ValueError('Invalid timing observation.')
                    if expected_provenance is not None and value['provenance']!=expected_provenance:
                        raise ValueError('Worker source differs from the parent session.')
                    values.append(value)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                policies[policy] = {'complete': False, 'error':str(exc)}
                continue
            same_operands = values[0]['metadata']==values[1]['metadata']
            same_runtime = (values[0]['runtime']==values[1]['runtime']
                            and values[0]['provenance']==values[1]['provenance'])
            exact = same_operands and same_runtime and all(v['within_process_exact'] for v in values) and values[0]['signature']==values[1]['signature']
            policies[policy] = {'complete': True, 'same_operands_and_runtime': same_operands and same_runtime,
                                'observed_exact_replay': exact,
                                'median_seconds': {k:statistics.median(v['median_seconds'][k] for v in values)
                                   for k in ('forward_seconds','backward_seconds','total_seconds')}}
        if all(policies[p]['complete'] for p in POLICIES):
            reference = [json.loads((Path(directory)/t['result']).read_text())
                         for t in tasks if t['case']==case and t['replay']==0]
            same = reference[0]['metadata']==reference[1]['metadata']
            policies['same_operands_between_policies'] = same
            same_source = reference[0]['provenance']==reference[1]['provenance']
            policies['same_source_between_policies'] = same_source
            precision_keys = ('matmul_precision','matmul_tf32','cudnn_tf32')
            same_precision = all(reference[0]['runtime'][k]==reference[1]['runtime'][k] for k in precision_keys)
            policies['same_precision_between_policies'] = same_precision
            if same and same_precision and same_source:
                policies['strict_to_default_total_ratio'] = policies['strict']['median_seconds']['total_seconds']/policies['default']['median_seconds']['total_seconds']
        comparisons[case] = policies
    return comparisons


def run_profile(args, root, started):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'logs').mkdir()
    upstream = use_upstream(args.upstream)
    cases = args.cases or list(CASES)
    session = {'schema_version': 1, 'diagnostic_only': True, 'research_result': False,
               'budget_approved': False, 'full_model_validated': False,
               'device': args.device, 'tiny_fixture': args.tiny,
               'upstream_commit': UPSTREAM_COMMIT,
               'upstream_sha256': {n:file_hash(upstream/n) for n in UPSTREAM_SOURCES},
               'source_sha256': {n:file_hash(Path(root)/n) for n in SOURCES},
               'max_seconds': args.max_seconds, 'per_worker_seconds': args.worker_seconds,
               'warmup': args.warmup, 'repeats': args.repeats,
               'parameter_gradients': args.parameter_gradients,
               'started_unix_seconds':time.time(),
               'tasks': [], 'complete': False, 'finished': False,
               'limitations': ['Synthetic tensors and isolated components, not a full-model replay.',
                              'Representative shapes sampled from one declared upstream mask seed.',
                              'Two fresh workers per policy provide observed replay, not a universal guarantee.',
                              'CUDA allocator limit excludes driver allocations; OOM/errors/timeouts remain failures.',
                              'Profiler iteration is excluded from timing medians; event timings may overlap.']}
    try:
        # Clear only this declared variable in child environments, not the caller.
        # The default policy models the original run; strict configures its own value.
        if args.device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU fallback is enabled.')
        if args.device=='cuda' and str(torch.__version__).split('+')[0]!='2.8.0':
            raise RuntimeError('Use the already validated PyTorch 2.8.0 environment.')
        for case in cases:
            for policy in POLICIES:
                for replay in range(2):
                    remaining = args.max_seconds - (time.monotonic()-started)
                    row = {'case':case,'policy':policy,'replay':replay}
                    relative = f'workers/{case}/{policy}-{replay}.json'
                    result = output/relative;result.parent.mkdir(parents=True,exist_ok=True)
                    if remaining<=0:
                        row['status']='not_started_deadline'
                    else:
                        print(f'Component: {case}; policy: {policy}; replay: {replay}',flush=True)
                        command=[sys.executable,'-u',str(Path(root)/'scripts/profile_cuda.py'),
                            '--worker','--case',case,'--policy',policy,'--device',args.device,
                            '--upstream',str(upstream),'--result',str(result),
                            '--repeats',str(args.repeats),'--warmup',str(args.warmup),
                            '--max-memory-gib',str(args.max_memory_gib)]
                        if args.tiny:command.append('--tiny')
                        if args.parameter_gradients:command.append('--parameter-gradients')
                        if args.no_profiler:command.append('--no-profiler')
                        # Clear this one declared variable only in the child;
                        # the caller's environment is unchanged.
                        launcher=['env','-u','CUBLAS_WORKSPACE_CONFIG',*command]
                        session['active_task']={**row,'result':relative}
                        write_json(output/'session.json',session)
                        before=time.monotonic()
                        row.update(run_bounded(launcher,output/'logs'/f'{case}-{policy}-{replay}.log',
                                               min(remaining,args.worker_seconds)))
                        row['worker_wall_seconds']=time.monotonic()-before
                        if result.exists():
                            row.update(result=relative,result_sha256=file_hash(result))
                        elif row['status']=='ok':row['status']='missing_result'
                    session.pop('active_task',None)
                    session['tasks'].append(row)
                    session['wall_seconds']=time.monotonic()-started
                    write_json(output/'session.json',session)
        session['complete']=all(t['status']=='ok' for t in session['tasks'])
    except BaseException as exc:
        session.update(error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        session['finished']=True
        session['wall_seconds']=time.monotonic()-started
        session['comparisons']=summarize(output,session['tasks'],cases,
            {'source_sha256':session['source_sha256'],'upstream_sha256':session['upstream_sha256']})
        session['strict_replay_passed']=all(p['strict'].get('observed_exact_replay',False)
                                             for p in session['comparisons'].values())
        session['complete'] = session['complete'] and session['strict_replay_passed']
        write_json(output/'session.json',session)
    print('Component diagnostic finished. Inspect session.json; timeouts and incomplete replay are not passes.',flush=True)
    return session
