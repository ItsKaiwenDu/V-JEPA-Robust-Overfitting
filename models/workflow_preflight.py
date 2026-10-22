"""Representative shared-forward input checks before expensive workflow phases."""

from contextlib import contextmanager
from pathlib import Path

import torch

from attacks.pgd import fgsm, native_objective, pgd_linf
from models.cuda_convolution import agreement
from models.cuda_profile import tensor_signature
from models.diagnostics import state_digest
from models.execution import ExecutionGuard, SHARED_POLICIES
from models.native_diagnostic import TOLERANCES
from models.precision_diagnostic import native_probe, projection_check
from models.precision_patch import PrecisionPatchEmbed3D
from models.runtime import synchronize
from models.study_artifacts import write_record
from models.checkpointing import file_sha256
from models.training import restore_rng, rng_state, runtime_contract


@contextmanager
def matched_backward(model, backend, device):
    """Deliberate comparison of declared integrated policies; never relabel artifacts."""
    if (getattr(model, '_diagnostic_execution_policy', None) not in SHARED_POLICIES
            or type(model.encoder.patch_embed) is not PrecisionPatchEmbed3D
            or backend not in ('cudnn', 'native')):
        raise ValueError('Preflight requires an explicit integrated shared-forward policy.')
    baseline = ExecutionGuard(model, device, target_versions=True)
    patch = model.encoder.patch_embed
    previous = patch.backward_backend, model._diagnostic_execution_policy
    try:
        patch.backward_backend = backend
        model._diagnostic_execution_policy = 'shared-fp64-' + backend
        guard = ExecutionGuard(model, device, target_versions=True)
        yield guard
        guard.check()
    finally:
        patch.backward_backend, model._diagnostic_execution_policy = previous
        baseline.check()


def _retained_tensor(directory, name, values):
    path = directory / name
    # Exclusive file creation, and a committed record only after a full write.
    with path.open('xb') as handle:
        torch.save(values, handle)
    return {'file': name, 'sha256': file_sha256(path), 'bytes': path.stat().st_size,
            'tensors': {n: tensor_signature(v) for n, v in values.items()}}


def run_preflight(context, directory, *, device, timings):
    """Three prescribed already-open training rows; original numerical limits."""
    if getattr(context.model, '_diagnostic_execution_policy', None) not in SHARED_POLICIES:
        raise ValueError('Representative preflight requires an explicit integrated shared-forward policy.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model, panel = context.model, context.training
    if len(panel.rows) < 3:
        raise ValueError('Preflight requires at least three diagnostic training rows.')
    indices = (0, len(panel.rows) // 2, len(panel.rows) - 1)
    record = {'schema_version': 1, 'kind': 'shared_forward_workflow_preflight',
              'diagnostic_only': True, 'research_result': False, 'study_adopted': False,
              'complete': False, 'passed': False, 'device': device,
              'indices': list(indices), 'selection': 'first-middle-last-of-existing-eight-training-rows-v1',
              'mask_seed': 0, 'score_mask_seed': 100000, 'epsilon_255': 4, 'pgd_step_255': 1,
              'source_provenance': context.provenance, 'clips': []}
    path = directory / 'preflight.json'
    write_record(path, record)
    original_rng = rng_state(device)
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    before = state_digest(model)
    baseline = ExecutionGuard(model, device, target_versions=True)
    try:
        def retain(evidence):
            record['native_backward_probe'] = evidence
            write_record(path, record, replace=True)
        with timings.span('new_shared_forward_native_backward_probe'):
            native_probe(device, retain)
        for index in indices:
            clip = {'index': index, 'video_id': panel.rows[index]['video_id'], 'arms': {}, 'passed': False}
            record['clips'].append(clip)
            write_record(path, record, replace=True)
            clean = panel.pixels(index).to(device)
            targets = model.clean_targets(clean)
            target_before = tensor_signature(targets)
            clip.update(pixels=tensor_signature(clean), clean_targets=target_before)
            masks = [m.to(device) for m in context.sampler(0)]
            score_masks = [m.to(device) for m in context.sampler(100000)]
            clip['masks'] = [{k: tensor_signature(getattr(m, k)) for k in ('context', 'target')} for m in masks]
            clip['score_masks'] = [{k: tensor_signature(getattr(m, k)) for k in ('context', 'target')} for m in score_masks]
            values = {}
            # Match probe/warmup and arithmetic order; target inputs never change.
            for backend in ('cudnn', 'native'):
                with matched_backward(model, backend, device) as guard:
                    arm = {'runtime': runtime_contract(device, model=model), 'passed': False}
                    clip['arms'][backend] = arm
                    with timings.span('preflight_projection_oracle', index=index, backward=backend):
                        validation, tensors = projection_check(model, clean, targets, masks)
                    arm['oracle_validation'] = validation
                    # Retain all reference tensors even when the gate rejects.
                    if not all(v['passed'] for v in validation.values()):
                        arm['artifact'] = _retained_tensor(directory, f'clip-{index}-{backend}.pt', tensors)
                        write_record(path, record, replace=True)
                        raise RuntimeError('Representative projection oracle rejected; original tolerances retained.')
                    x = clean.detach().clone().requires_grad_(True)
                    with timings.span('preflight_clean_input_gradient', index=index, backward=backend):
                        loss = model(x, targets, masks)
                        gradient, = torch.autograd.grad(loss, x)
                        synchronize(device)
                    tensors.update(full_loss=loss.detach().cpu(), full_input_gradient=gradient.detach().cpu())
                    arm['input_gradient_nonzero'] = bool(gradient.abs().max() > 0)
                    reconstruction = agreement(tensors['full_input_gradient'], tensors['patch_input_gradient'],
                                               TOLERANCES['input_gradient'])
                    arm['full_chain_reconstruction'] = reconstruction
                    objective = native_objective(model, targets, [masks])
                    score = native_objective(model, targets, [score_masks])
                    arm['attacks'] = {}
                    with torch.no_grad():
                        clean_objective = objective(clean)
                    for method in ('fgsm', 'pgd5'):
                        with timings.span('preflight_attack_and_score', index=index, backward=backend, method=method):
                            result = (fgsm(objective, clean, epsilon=4/255) if method == 'fgsm' else
                                      pgd_linf(objective, clean, epsilon=4/255, step_size=1/255, steps=5, seed=0))
                            with torch.no_grad():
                                independent = score(result.adversarial)
                            delta = float((result.adversarial - clean).abs().max())
                            passed = (bool(torch.isfinite(independent).all()) and delta <= 4/255 + 1e-7
                                      and float(result.adversarial.min()) >= 0 and float(result.adversarial.max()) <= 1
                                      and bool((result.losses >= clean_objective).all()))
                            arm['attacks'][method] = {'attack_loss': result.losses.tolist(),
                                'independent_loss': independent.tolist(), 'max_pixel_delta': delta, 'passed': passed}
                            tensors[method] = result.adversarial.detach().cpu()
                    arm['artifact'] = _retained_tensor(directory, f'clip-{index}-{backend}.pt', tensors)
                    arm['native_loss_ascent_observed'] = any(
                        any(value > float(tensors['full_loss']) for value in attack['attack_loss'])
                        for attack in arm['attacks'].values())
                    arm['passed'] = (reconstruction['passed'] and all(v['passed'] for v in arm['attacks'].values())
                                     and torch.equal(tensors['full_loss'], tensors['tail_loss'])
                                     and arm['input_gradient_nonzero'] and arm['native_loss_ascent_observed'])
                    guard.check()
                    if (any(p.grad is not None for p in model.parameters()) or targets.requires_grad
                            or tensor_signature(targets) != target_before):
                        raise RuntimeError('Preflight changed target features or accumulated parameter gradients.')
                    values[backend] = tensors
                    write_record(path, record, replace=True)
                    if not arm['passed']:
                        raise RuntimeError('Representative reconstruction/attack gate rejected.')
            ref, candidate = values['cudnn'], values['native']
            clip['comparison'] = {'tokens_bitwise_equal': torch.equal(ref['patch_tokens'], candidate['patch_tokens']),
                'loss': agreement(candidate['full_loss'], ref['full_loss'], TOLERANCES['clean_loss']),
                'input_gradient': agreement(candidate['full_input_gradient'], ref['full_input_gradient'], TOLERANCES['input_gradient']),
                'attacks_bitwise_equal': {m: torch.equal(ref[m], candidate[m]) for m in ('fgsm', 'pgd5')}}
            # Cross-backend attack equality is descriptive, never a new acceptance limit.
            c = clip['comparison']
            clip['passed'] = c['tokens_bitwise_equal'] and c['loss']['passed'] and c['input_gradient']['passed']
            baseline.check()
            write_record(path, record, replace=True)
            if not clip['passed']:
                raise RuntimeError('Representative full-model numerical comparison rejected.')
        if state_digest(model) != before:
            raise RuntimeError('Preflight changed original learned or fixed target tensors.')
        record.update(complete=True, passed=True, original_weights_unchanged=True)
        write_record(path, record, replace=True)
        return record
    except BaseException as exc:
        record.update(error_type=type(exc).__name__, error=str(exc))
        write_record(path, record, replace=True)
        raise
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(original_rng, device)
