"""Bounded candidate phases, fresh-process recovery and deployment-inclusive limits.

This does not provision a Pod, grant spending permission, adopt an attack, or
qualify a full research trajectory. All measurements start unknown.
"""

import json
import math
from pathlib import Path
from decimal import Decimal, InvalidOperation

from models.candidate_contract import binding, bind_provenance, verify_binding, verify_provenance
from models.evaluation_store import json_hash

CONDITIONS = ('clean-finetune', 'adversarial-finetune')
PHASES = ('preflight', 'calibration-bundle', 'primary-pretrained',
          'clean-finetune/first', 'clean-finetune/resume', 'clean-finetune/reference',
          'adversarial-finetune/first', 'adversarial-finetune/resume', 'adversarial-finetune/reference',
          'primary-trained', 'controls', 'features-initial', 'features-trained')


def positive_decimal(value):
    if isinstance(value, bool):
        raise ValueError('Invalid money value.')
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('Invalid money value.') from exc
    if not result.is_finite() or result <= 0:
        raise ValueError('Use positive finite money values.')
    return result


def launch_limits(plan, ledger, *, now):
    """A new reviewed launch record must count deployment, setup and idle time.

    Raw ledger bytes must be checksum-verified by the CLI first. No historical
    recovery windows or financial holds can supply these new fields.
    """
    verify_binding(plan.get('candidate_binding'))
    if (plan.get('kind') != 'candidate_eot8_benchmark_launch_v1' or plan.get('authorized') is not True
            or plan.get('diagnostic_only') is not True or plan.get('required_phases') != list(PHASES)
            or plan.get('execution_policy') != 'shared-fp64-native'
            or plan.get('container_gb') != 30 or plan.get('persistent_gb') != 80
            or plan.get('volume_mount') != '/workspace'
            or plan.get('minimum_system_ram_gib') != 40
            or ledger.get('reconciled') is not True or ledger.get('all_prior_pods_terminated') is not True
            or ledger.get('compute_cap_usd') != 160 or ledger.get('total_cap_usd') != 200
            or ledger.get('pilot_cap_usd') != 10 or ledger.get('noncompute_reserve_usd') != 40):
        raise ValueError('Fresh reviewed launch/ledger contract is incomplete; no GPU work permitted.')
    rate = positive_decimal(plan['total_hourly_usd'])
    allowance = positive_decimal(plan['inclusive_session_budget_usd'])
    prior = Decimal(str(ledger['pilot_spent_and_other_holds_usd']))
    if not prior.is_finite() or prior < 0 or allowance + prior > Decimal(10):
        raise ValueError('Session would exceed the inclusive ten-dollar pilot envelope.')
    start, deadline = plan['deployed_unix'], plan['deadline_unix']
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, deadline, now))
            or start > now or not 300 < deadline - start <= 7200
            or Decimal(str(deadline - start)) * rate / 3600 > allowance
            or deadline - now <= 180):
        raise ValueError('Deployment-inclusive time or dollar limit is invalid/expired.')
    # Leave three minutes for final evidence collection and manual termination.
    # The runner cannot stop Runpod billing, so this is not a billing guarantee.
    return min(deadline - now - 180, 7200), float(rate), float(allowance)


def benchmark_proposal():
    return {'kind': 'candidate_eot8_benchmark_preparation_v1', 'candidate_binding': binding(),
            'diagnostic_only': True, 'research_admitted': False, 'budget_approved': False,
            'launch_authorized': False, 'required_phases': list(PHASES),
            'execution_policy': 'shared-fp64-native', 'proposed_maximum_billed_hours': 2,
            'container_gb': 30, 'persistent_gb': 80, 'minimum_free_gib_after_inputs': 55,
            'minimum_system_ram_gib': 40,
            'volume_mount': '/workspace', 'primary_diagnostic_step_255': 'one-clip training bundle only',
            'unknowns': {'gpu': None, 'current_total_hourly_usd': None, 'deployment_unix': None,
                         'inclusive_session_budget_usd': None, 'research_primary_step_255': None,
                         'primary_seconds': None, 'three_candidate_bundle_seconds': None,
                         'new_hardware_exact_recovery_pass': None, 'complete_study_cost_usd': None},
            'needs_before_launch': ['source-and-input-verified handoff on selected host',
                'fresh all-in quote and sufficient system RAM/VRAM',
                'reconciled ledger with other holds and stopped-storage costs',
                'new reviewed launch contract and public host fingerprint'],
            'full_ten_clip_calibration_enabled': False,
            'historical_timing_reused_as_new_measurement': False,
            'automatic_billing_stop': False}


def run_phase(root, args):
    """Called in a fresh process for every listed phase; no implicit retries."""
    from models.workflow_diagnostic import prepare_context, cached_training_loader, panel_view
    from models.candidate_calibration import run_calibration, load_calibration
    from models.candidate_evaluation import EvaluationSettings, model_identity
    from models.candidate_evaluation_store import EvaluationStore, evaluate_panel
    from models.candidate_evaluation_checkpoint import load_anchor
    from models.checkpointing import CheckpointStore
    from models.pilot_measurement import Timings
    from models.runtime import MemoryMonitor
    from models.study_artifacts import write_record
    from models.training import TrainingEngine
    output = Path(args.output)
    timings = Timings(output / 'timings' / (args.phase.replace('/', '-') + '.jsonl'), args.device)
    with timings.span('prepare_context'):
        context = prepare_context(root, args)
        context.provenance = bind_provenance(context.provenance, cuda=args.device == 'cuda')
    result = {'kind': 'candidate_eot8_benchmark_phase_v1', 'phase': args.phase,
              'candidate_binding': binding(), 'provenance': context.provenance,
              'diagnostic_only': True, 'research_result': False, 'budget_approved': False}
    with MemoryMonitor(args.device) as memory:
        if args.phase == 'preflight':
            from models.workflow_preflight import run_preflight
            result['preflight'] = run_preflight(context, output / 'preflight', device=args.device, timings=timings)
        elif args.phase == 'calibration-bundle':
            with timings.span('candidate_three_step_bundle'):
                value = run_calibration(context.model, context.monitor, context.sampler,
                    output / 'calibration-diagnostic', checkpoint=context.checkpoint,
                    provenance=context.provenance, diagnostic=True)
            result['diagnostic_chosen_step_255'] = value['chosen_step_size_255']
            result['research_step_selected'] = False
        else:
            calibration = load_calibration(output / 'calibration-diagnostic', context.monitor.rows,
                original_identity=model_identity(context.model), model=context.model, diagnostic=True)
            step = calibration['chosen_step_size_255']
            provenance = {**context.provenance, 'training_rows': context.training.rows,
                'calibration_sha256': json_hash(calibration),
                'evaluation_provenance_sha256': json_hash(context.provenance)}
            def evaluate(model, panel, name, method='primary', checkpoint=None, audits=0):
                with timings.span('candidate_evaluation', unit=name, method=method):
                    payload = evaluate_panel(model, panel, context.sampler,
                        EvaluationSettings.for_method(method, 1. if method == 'routine' else step),
                        EvaluationStore(output / 'evaluations' / name),
                        checkpoint=checkpoint or context.checkpoint, provenance=context.provenance,
                        sensitivity_banks=audits)
                return {'records': payload['records'], 'contract_sha256': json_hash(payload['contract'])}
            if args.phase == 'primary-pretrained':
                n = len(context.training.rows)
                panel = panel_view(context.training, [0, n // 2, n - 1],
                                   selection='candidate-benchmark-first-middle-last-v1')
                result['primary'] = evaluate(context.model, panel, args.phase)
                result['routine'] = evaluate(context.model, context.monitor, 'routine-pretrained', method='routine')
            elif args.phase.split('/')[0] in CONDITIONS:
                condition, segment = args.phase.split('/')
                with timings.span('cache_fixed_training_targets'):
                    loader = cached_training_loader(context.model, context.training, args.device)
                engine = TrainingEngine(context.model, context.settings,
                    [r['video_id'] for r in context.training.rows], loader, context.sampler,
                    condition=condition, seed=0, provenance=provenance, device=args.device)
                if engine.updates_per_epoch != 1:
                    raise ValueError('Benchmark requires the original eight-clip/effective-eight boundary.')
                timings.instrument(engine, ('update', 'state', 'restore'), 'engine')
                path = output / condition / ('reference' if segment == 'reference' else 'split')
                if segment == 'resume':
                    store = CheckpointStore(path, create=False)
                    with timings.span('checkpoint_load'):
                        state, _ = store.load()
                    if state['step'] != 1:
                        raise ValueError('Fresh-process resume requires exactly saved update one.')
                    engine.restore(state)
                else:
                    path.mkdir(parents=True, exist_ok=False)
                    store = CheckpointStore(path)
                timings.instrument(store, ('save',), 'checkpoint')
                required_step = 1 if segment == 'first' else 2
                while engine.step < required_step:
                    engine.update()
                    store.save(engine.state(), anchor_epoch=engine.step)
                result.update(condition=condition, completed_step=engine.step,
                              training_contract_sha256=engine.contract_sha256)
                if segment == 'reference':
                    # The comparison is inside this worker's enforced deadline.
                    # It is never an unbounded post-processing step in the parent.
                    with timings.span('exact_fresh_process_recovery_comparison'):
                        comparison = compare_recovery(output, condition)
                    write_record(output / ('recovery-' + condition + '.json'), comparison)
                    if not comparison['exact_match']:
                        raise RuntimeError('Exact optimizer recovery rejected; no tolerance changed.')
                    result['exact_recovery'] = True
            elif args.phase == 'primary-trained':
                # Each model starts at the original checkpoint in its own phase.
                # Both anchor loads validate complete candidate provenance before
                # loading any learned tensors. The fixed target never changes.
                for condition in CONDITIONS:
                    with timings.span('candidate_anchor_load', condition=condition):
                        checkpoint = load_anchor(context.model, output / condition / 'reference', 2,
                            initial_checkpoint_sha256=context.checkpoint['file_sha256'], expected_provenance=provenance)
                    result[condition] = evaluate(context.model, context.monitor, 'trained-' + condition, checkpoint=checkpoint)
            elif args.phase == 'controls':
                for method in ('primary', 'convergence', 'fgsm', 'random'):
                    result[method] = evaluate(context.model, context.monitor, 'controls-' + method,
                                              method=method, audits=2 if method in ('primary', 'convergence') else 0)
                if result['primary']['records'][0]['seeds'] != result['convergence']['records'][0]['seeds']:
                    raise ValueError('Step-only controls did not retain identical masks and starts.')
            elif args.phase in ('features-initial', 'features-trained'):
                from models.candidate_features import extract_features, fit_ridge_probe, score_ridge_probe
                if args.phase == 'features-trained':
                    with timings.span('candidate_anchor_load', condition='adversarial-finetune'):
                        load_anchor(context.model, output / 'adversarial-finetune' / 'reference', 2,
                            initial_checkpoint_sha256=context.checkpoint['file_sha256'], expected_provenance=provenance)
                with timings.span('feature_extraction_eight_training_clips'):
                    features = extract_features(context.model, context.training, provenance=context.provenance)
                with timings.span('fit_training_only_probe'):
                    probe = fit_ridge_probe(features, expected_provenance=context.provenance, model=context.model)
                result['training_resubstitution'] = score_ridge_probe(probe, features,
                    expected_provenance=context.provenance, model=context.model)
                result['held_out_accuracy_measured'] = False
            else:
                raise ValueError('Unknown candidate benchmark phase.')
        result['memory'] = memory.report()
    write_record(output / 'phase-results' / (args.phase.replace('/', '-') + '.json'), result)
    return result


def compare_recovery(output, condition):
    from models.checkpointing import CheckpointStore
    from models.pilot_measurement import compare_trees
    values = []
    for variant in ('reference', 'split'):
        state, _ = CheckpointStore(Path(output) / condition / variant, create=False).load()
        verify_provenance(state['contract']['provenance'])
        if state['step'] != 2 or state['contract']['condition'] != condition:
            raise ValueError('Recovery comparison did not reach the exact two-update condition boundary.')
        values.append(state)
    signatures = [logical_signature(v) for v in values]
    result = compare_trees(*values)
    if result['exact_match'] != (signatures[0] == signatures[1]):
        raise ValueError('Independent logical-signature comparison disagrees with exact tree comparison.')
    return {'kind': 'candidate_eot8_two_update_exact_recovery_v1',
            'condition': condition, **result, 'logical_signatures': signatures,
            'research_result': False}


def logical_signature(value):
    """Retain every logical leaf for off-host comparison without copying AdamW tensors."""
    import hashlib
    import torch
    if isinstance(value, torch.Tensor):
        v = value.detach().cpu().contiguous()
        if not torch.isfinite(v).all():
            raise ValueError('Nonfinite recovery tensor.')
        return {'type': 'tensor', 'shape': list(v.shape), 'dtype': str(v.dtype),
                'sha256': hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {'type': 'dict', 'items': [[logical_signature(k), logical_signature(v)]
            for k, v in sorted(value.items(), key=lambda item: (type(item[0]).__name__, str(item[0])))]}
    if isinstance(value, (list, tuple)):
        return {'type': type(value).__name__, 'items': [logical_signature(v) for v in value]}
    if value is None or type(value) in (str, bool, int, float):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Nonfinite recovery scalar.')
        return {'type': type(value).__name__, 'value': value}
    raise TypeError('Unsupported recovery logical leaf: ' + type(value).__name__)
