"""One actual native EOT8 objective per fresh process; no profiling/warmup/training."""
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from models import predictor_forward_preserving_records_v1 as records

BOUNDS = {'loss': {'rtol': 1e-5, 'atol': 2e-6}, 'input': {'rtol': .002, 'atol': 2e-7}}


def compare_arrays(actual, reference, *, replay=False):
    import numpy as np
    if set(actual) != set(reference):
        raise ValueError('Complete actual value coverage required.')
    checks = {}
    for name, value in actual.items():
        expected = reference[name]
        if value.shape != expected.shape or value.dtype != expected.dtype or not np.isfinite(value).all() or not np.isfinite(expected).all():
            raise ValueError('Actual/reference dtype/shape/finite gate rejected.')
        if replay or name not in ('loss', 'input'):
            if value.tobytes() != expected.tobytes():
                raise ValueError('Exact native forward/cotangent/fresh replay gate rejected: '+name)
        else:
            error = np.abs(value.astype(np.float64)-expected.astype(np.float64))
            allowed = BOUNDS[name]['atol']+BOUNDS[name]['rtol']*np.abs(expected.astype(np.float64))
            check = {'passed': bool(np.all(error <= allowed)), 'failed_elements': int(np.count_nonzero(error > allowed)),
                     'max_absolute_error': float(error.max()), 'max_tolerance_ratio': float((error/allowed).max()), **BOUNDS[name]}
            checks[name] = check
            if not check['passed']:
                raise ValueError('Original full-model numerical gate rejected: '+name)
    return checks


def arrays_from(folder, state):
    import numpy as np
    return {name: np.load(Path(folder)/item['file'], allow_pickle=False) for name, item in state['artifacts'].items()}


def run(args):
    host = platform.node().split('.')[0]
    if type(args.tiny) is not bool or type(args.index) is not int or args.index not in range(4):
        raise ValueError('Exact worker domain/index required.')
    if (args.tiny and (host in ('linux10603', 'linux10609') or args.device != 'cpu')) or (not args.tiny and
            (host != 'linux10603' or args.device != 'cuda' or platform.python_version() != '3.11.17')):
        raise ValueError('Wrong worker domain; no CPU fallback or university CPU analogue.')
    mode, replay = ('original' if args.index % 2 == 0 else 'candidate'), args.index//2
    references = {}
    for name, path, index in [('original', args.reference, args.index-1), ('replay', args.replay_reference, args.index-2)]:
        needed = mode == 'candidate' if name == 'original' else replay == 1
        if needed:
            if path is None:
                raise ValueError('Fresh complete passing reference required before scientific imports.')
            references[name] = records.load(path, control_sha256=args.control_sha256, index=index, live=not args.tiny)
        elif path is not None:
            raise ValueError('Unexpected extra reference.')
    import numpy as np
    import torch
    from attacks.pgd import native_objective
    from models.candidate_evaluation import EvaluationSettings, seeds_for
    from models.cuda_profile import tensor_signature
    from models.diagnostics import state_digest
    from models.execution import ExecutionGuard, configure_execution, apply_model_policy, precision_contract
    from models.native_diagnostic import tiny_fixture
    from models.training import runtime_contract
    from models import predictor_eot8_owned_v1 as native
    from models import predictor_forward_preserving_target_v1 as target
    folder = Path(args.output); folder.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    state = {'kind': records.WORKER_KIND, 'execution_host': host, 'local_tiny': args.tiny, 'mode': mode,
             'index': args.index, 'replay': replay, 'pid': os.getpid(), 'control_sha256': args.control_sha256,
             'complete': False, 'passed': False, 'artifacts': {}, 'research_admitted': False,
             'performance_or_kernel_coverage_qualified': False, 'numerical_bounds': BOUNDS}
    route = None; handles = []; arrays = {}; borrowed = None
    def progress():
        records.atomic._atomic(folder, 'worker.json', [records.atomic._json(state)], records.LIMITS['worker.json'])
    def save(name, tensor):
        a = tensor.detach().cpu().contiguous().numpy().copy()
        path = folder/(name+'.npy')
        cap = 65536 if name == 'loss' else 2**20 if name == 'masks' else 12*2**20 if name == 'targets' else 16*2**20
        if a.nbytes+4096 > cap:
            raise ValueError('Actual return exceeds typed file cap.')
        partial = path.with_name(path.name+'.partial')
        with partial.open('xb') as f:
            np.save(f, a, allow_pickle=False); f.flush(); os.fsync(f.fileno())
        if partial.stat().st_size > cap:
            raise ValueError('Actual serialized return exceeds cap; preserve partial.')
        os.rename(partial, path)
        state['artifacts'][name] = {'file': path.name, 'bytes': path.stat().st_size, 'sha256': records.sha_file(path),
                                  'shape': list(a.shape), 'dtype': str(a.dtype), 'source_device': str(tensor.device), 'saved_before_gate': True}
        arrays[name] = a; progress()
        if not np.isfinite(a).all():
            raise ValueError('Saved nonfinite value rejected.')
    progress()
    try:
        configure_execution('shared-fp64-native', device=args.device); torch.set_num_threads(4); torch.manual_seed(20261029)
        if args.tiny:
            model, clean, masks_for = tiny_fixture(args.upstream)
            video = 'tiny-predictor-eot8'; data = {'kind': 'local-tiny-CPU'}; prior = None
        else:
            if torch.__version__ != '2.8.0+cu128' or torch.version.cuda != '12.8' or np.__version__ != '2.4.6' or not torch.cuda.is_available():
                raise ValueError('Exact live runtime required; no installation/fallback.')
            torch.cuda.set_per_process_memory_fraction(7*2**30/torch.cuda.get_device_properties(0).total_memory)
            from models.eot_profile_packet import verify
            from models.reuse_diagnostic import upstream_gate
            from models.reuse_handoff import training_input
            from models.vjepa2 import build_vitl, load_checkpoint, sample_masks
            _, parent = verify(args.packet, 'c95c0b21ee018dfd6c897756baf453dc7d69eb1d52b5c64d99e1ac4b4b945dbd')
            upstream_gate(args.upstream, parent)
            model = build_vitl(args.upstream, activation_checkpointing=True)
            checkpoint = load_checkpoint(model, args.checkpoint, '5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012')
            clean = training_input(args.packet, parent)
            masks_for = lambda seed: sample_masks(args.upstream, batch_size=1, seed=seed)
            video = parent['training_row']['video_id']
            data = {'kind': 'original-training', 'video_id': video, 'checkpoint_sha256': checkpoint['sha256']}
            prior = torch.load(Path(args.packet)/'input/prior-EOT8-derivatives.pt', weights_only=True, map_location='cpu')
        apply_model_policy(model, 'shared-fp64-native'); model.to(args.device).eval(); clean = clean.to(args.device)
        guard = ExecutionGuard(model, args.device, target_versions=True)
        weights = state_digest(model); targets = model.clean_targets(clean)
        seeds = seeds_for(EvaluationSettings.for_method('primary', 2.), video)['attack']
        banks = [[m.to(args.device) for m in masks_for(seed % (2**31-1))] for seed in seeds]
        masks = [[[tensor_signature(m.context), tensor_signature(m.target)] for m in draw] for draw in banks]
        baseline = {'weights_sha256': weights, 'targets': tensor_signature(targets), 'pixels': tensor_signature(clean),
                    'seeds': seeds, 'masks': masks, 'runtime': runtime_contract(args.device, model=model), 'data': data}
        if len(banks) != 8 or any(len(draw) != 2 for draw in banks):
            raise ValueError('Exactly eight draws/two original families required.')
        if not args.tiny:
            expected = json.loads(Path(args.proposal).read_text())['reference']
            for key, alias in [('weights_sha256', 'weights_sha256'), ('targets', 'clean_targets'), ('pixels', 'clean_pixels'),
                               ('seeds', 'attack_seeds'), ('masks', 'full_mask_tensor_identities')]:
                if baseline[key] != expected[alias]:
                    raise ValueError('Frozen original weights/pixels/targets/masks/seeds changed.')
            old = json.loads((Path(args.packet)/'input/prior-contract.json').read_text())
            if baseline['runtime'] != old['runtime']:
                raise ValueError('Original full-model runtime changed.')
        if any(ref['baseline'] != baseline for ref in references.values()):
            raise ValueError('Fresh original/replay baseline changed.')
        state['baseline'] = baseline
        state['mask_context_lengths'] = [[int(m.context.shape[1]) for m in draw] for draw in banks]
        save('targets', targets)
        packed = torch.stack([torch.stack([torch.cat((m.context, m.target), dim=1) for m in draw]) for draw in banks])
        save('masks', packed)
        versions = tuple(p._version for p in model.parameters()); target_version = targets._version; precision = precision_contract()
        predictions = []
        def observe(module, inputs, output):
            index = len(predictions); predictions.append(index)
            if index >= 16:
                raise ValueError('No extra predictor objective calls allowed.')
            save('prediction-'+str(index), output)
            def cotangent(g):
                save('cotangent-'+str(index), g)
                return None
            handles.append(output.register_hook(cotangent))
            return None
        handles.append(model.predictor.register_forward_hook(observe))
        value = clean.detach().clone().requires_grad_(True)
        route = native.PredictorRoute(model, 'original', tiny=args.tiny) if mode == 'original' else target.TargetRoute(model, tiny=args.tiny)
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        with route:
            if mode == 'original':
                route.phase = 'EOT8-before'
                loss = native_objective(model, targets, banks)(value).sum()
            else:
                loss = route.loss(value, targets, banks)
            save('loss', loss)
            if mode == 'original':
                gradient, = route.prepare_backward(loss, value).run()
            else:
                gradient = route.gradient(loss, value)
            save('input', gradient)
        for h in handles:
            h.remove()
        handles.clear()
        if len(predictions) != 16 or not arrays['input'].any():
            raise ValueError('All predictions and nonzero full input gradient required.')
        if (state_digest(model) != weights or tuple(p._version for p in model.parameters()) != versions
                or targets._version != target_version or precision_contract() != precision
                or any(p.grad is not None for p in model.parameters()) or value.grad is not None):
            raise ValueError('Model/target/versions/precision or parameter gradient state changed.')
        guard.check()
        if args.device == 'cuda':
            state['peak_allocator_bytes'] = torch.cuda.max_memory_allocated()
            if not 0 < state['peak_allocator_bytes'] <= 7*2**30:
                raise ValueError('GPU allocator cap exceeded.')
        # Actual arrays exist before any acceptance, and remain on numerical refusal.
        if prior is not None:
            state['anchor_checks'] = compare_arrays({k: arrays[k] for k in ('loss', 'input')},
                                                    {k: prior[k].numpy() for k in ('loss', 'input')})
        for name, ref in references.items():
            path = args.reference if name == 'original' else args.replay_reference
            state[name+'_checks'] = compare_arrays(arrays, arrays_from(path, ref), replay=name == 'replay')
        native_route = route if mode == 'original' else route.native
        vjps = [] if mode == 'original' else route.prototype.calls
        borrowed = None if mode == 'original' else route.prototype.scope
        state['ownership'] = records.check_scope(state, native_route.rows, native_route.scopes, native_route.refusals, vjps, borrowed)
        state.update(model_unchanged=True, adapter_restored=True, complete=True, passed=True)
    except BaseException as error:
        state.update(complete=False, passed=False, error_type=type(error).__name__, error=str(error)[:2048])
        raise
    finally:
        for handle in handles:
            handle.remove()
        native_route = None if route is None else route if mode == 'original' else route.native
        state['wall_seconds'] = time.monotonic()-started
        records.write_final(folder, state, calls=[] if native_route is None else native_route.rows,
                            scopes=[] if native_route is None else native_route.scopes,
                            refusals=[] if native_route is None else native_route.refusals,
                            vjps=[] if route is None or mode == 'original' else route.prototype.calls,
                            borrowed=None if route is None or mode == 'original' else route.prototype.scope)
    return state
