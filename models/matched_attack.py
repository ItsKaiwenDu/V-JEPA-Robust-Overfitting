"""Paired training-only PGD diagnostic; deliberately outside EvaluationSettings."""

import hashlib
import math
from pathlib import Path
import time

import torch

from attacks.pgd import native_objective, pgd_linf
from models.checkpointing import CheckpointStore, file_sha256
from models.evaluation import evaluation_seed, model_identity
from models.evaluation_store import json_hash
from models.execution import ExecutionGuard
from models.runtime import synchronize
from models.study_artifacts import read_record, write_record
from models.training import restore_rng, rng_state, runtime_contract

KIND = 'matched-training-attack-diagnostic-v1'
POLICY = 'shared-fp64-native'
ARMS = {'A': (20, 4), 'B': (50, 4), 'C': (20, 8), 'D': (50, 8)}
BASE_SEED = 20261023
EPSILON, STEP, RESTARTS = 4 / 255, 2 / 255, 2
MAX_TENSOR_BYTES = 64 * 2**20


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    header = json_hash({'dtype': str(value.dtype), 'shape': list(value.shape)})
    return hashlib.sha256(header.encode() + value.numpy().tobytes()).hexdigest()


def _seeds(video_id):
    seeds = {name: [evaluation_seed(BASE_SEED, 'matched_diagnostic_' + name, video_id, draw=i)
                    for i in range(count)]
             for name, count in (('attack', 8), ('main', 4), ('audit_0', 4), ('audit_1', 4))}
    actual = [s % (2**31 - 1) for group in seeds.values() for s in group]
    if len(actual) != len(set(actual)):
        raise ValueError('Matched diagnostic upstream mask seed collision.')
    seeds['start'] = evaluation_seed(BASE_SEED, 'matched_diagnostic_start', video_id)
    return seeds


def _starts(clean, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    lower, upper = (clean - EPSILON).clamp(0, 1), (clean + EPSILON).clamp(0, 1)
    return [(clean + (2 * torch.rand(clean.shape, generator=generator,
                    dtype=torch.float32).to(clean.device) - 1) * EPSILON).clamp(lower, upper)
            for _ in range(RESTARTS)]


def _save_tensors(directory, name, tensors):
    if any(not isinstance(t, torch.Tensor) or not torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError('Only finite named diagnostic tensors may be saved.')
    if sum(t.numel() * t.element_size() for t in tensors.values()) > MAX_TENSOR_BYTES:
        raise ValueError('Diagnostic tensor file exceeds the declared bound.')
    store = CheckpointStore(directory)
    if (store.root / name).exists():
        raise FileExistsError(store.root / name)
    reference = store._atomic(name, {n: t.detach().cpu() for n, t in tensors.items()})
    return {**reference, 'bytes': (store.root / name).stat().st_size,
            'tensors': {n: tensor_hash(t) for n, t in tensors.items()}}


def _load_tensors(directory, reference):
    name = reference['file']
    if Path(name).name != name:
        raise ValueError('Escaped diagnostic artifact.')
    path = Path(directory) / name
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != reference['bytes']
            or path.stat().st_size > MAX_TENSOR_BYTES + 2**20 or file_sha256(path) != reference['sha256']):
        raise ValueError('Diagnostic artifact checksum or size mismatch.')
    tensors = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(tensors, dict) or set(tensors) != set(reference['tensors'])
            or any(not isinstance(t, torch.Tensor) or not torch.isfinite(t).all()
                   or tensor_hash(t) != reference['tensors'][n] for n, t in tensors.items())):
        raise ValueError('Diagnostic logical tensor checksum mismatch.')
    return tensors


class _Trace:
    """Observe unchanged PGD calls and input VJPs without retaining their graphs."""

    def __init__(self, objective, steps):
        self.objective, self.steps, self.rows = objective, steps, []
        self.handles = []

    def __call__(self, pixels):
        index = len(self.rows)
        if index == 0:
            restart, step, expected_grad = None, None, False
        else:
            restart, step = divmod(index - 1, self.steps + 1)
            expected_grad = step < self.steps
        if restart is not None and restart >= RESTARTS:
            raise ValueError('Unexpected PGD objective call.')
        if torch.is_grad_enabled() != expected_grad or pixels.requires_grad != expected_grad:
            raise ValueError('PGD call ordering or gradient mode changed.')
        loss = self.objective(pixels)
        if loss.shape != (1,) or not torch.isfinite(loss).all():
            raise ValueError('Matched diagnostic requires one finite native loss.')
        row = {'restart': restart, 'step': step, 'with_grad': expected_grad,
               'input_sha256': tensor_hash(pixels), 'loss': loss.detach().item()}
        if expected_grad:
            def retain(gradient):
                row['gradient_sha256'] = tensor_hash(gradient)
                row['gradient_abs_max'] = gradient.detach().abs().max().item()
                row['gradient_finite'] = bool(torch.isfinite(gradient).all())
            self.handles.append(pixels.register_hook(retain))
        self.rows.append(row)
        return loss

    def close(self):
        for handle in self.handles:
            handle.remove()


def _trace_gate(record, contract):
    steps, draws = ARMS[record['arm']]
    trace = record['trace']
    if (record['settings'] != {'steps': steps, 'attack_masks': draws}
            or len(trace) != 1 + RESTARTS * (steps + 1)):
        raise ValueError('Incomplete or mismatched diagnostic arm.')
    winner = trace[0]
    for index, row in enumerate(trace):
        restart, step = (None, None) if index == 0 else divmod(index - 1, steps + 1)
        grad = index != 0 and step < steps
        if (row['restart'] != restart or row['step'] != step or row['with_grad'] != grad
                or not math.isfinite(row['loss'])
                or (grad and (not row.get('gradient_finite')
                              or not 0 < row.get('gradient_abs_max', 0) < math.inf
                              or not isinstance(row.get('gradient_sha256'), str)))):
            raise ValueError('Incomplete/nonfinite PGD trace or input gradient.')
        if step == 0 and row['input_sha256'] != contract['starts_sha256'][restart]:
            raise ValueError('Random starts differ across matched arms.')
        if row['loss'] > winner['loss']:
            winner = row
    if (trace[0]['input_sha256'] != contract['clean_sha256']
            or record['attack_loss'] != winner['loss']
            or record['adversarial_sha256'] != winner['input_sha256']
            or not record['attack_recomputed_exact']):
        raise ValueError('Attack-bank selection or saved adversary reconstruction failed.')
    if (not math.isfinite(record['max_pixel_delta'])
            or record['max_pixel_delta'] > EPSILON + 1e-7 or record['max_pixel_delta'] < 0
            or not 0 < record['instrumented_attack_seconds'] < math.inf
            or set(record['scores']) != {'main', 'audit_0', 'audit_1'}
            or any(set(row) != {'clean', 'adversarial'} or any(not math.isfinite(v) for v in row.values())
                   for row in record['scores'].values())):
        raise ValueError('Invalid bounds or independent scoring evidence.')


def pairing_gate(records, contract):
    if set(records) != set(ARMS):
        raise ValueError('All four matched diagnostic arms are required.')
    for record in records.values():
        if record['contract_sha256'] != json_hash(contract):
            raise ValueError('Mixed diagnostic contracts.')
        _trace_gate(record, contract)
        if any(record['scores'][name]['clean'] != records['A']['scores'][name]['clean']
               for name in ('main', 'audit_0', 'audit_1')):
            raise ValueError('Matched arms have different clean scoring references.')
    for short, long in (('A', 'B'), ('C', 'D')):
        for restart in range(RESTARTS):
            a = records[short]['trace'][1 + restart * 21:1 + restart * 21 + 21]
            b = records[long]['trace'][1 + restart * 51:1 + restart * 51 + 21]
            if a[:20] != b[:20] or a[20]['input_sha256'] != b[20]['input_sha256']:
                raise ValueError('Matched attack start/prefix replay is not exact.')
        if records[long]['attack_loss'] < records[short]['attack_loss']:
            raise ValueError('Extending matched PGD reduced its best attack-bank loss.')
    return {'starts_and_first_20_steps_exact': True, 'attack_bank_best_nondecreasing': True,
            'independent_scores_required_monotone': False}


def run_attacks(model, clean, masks_for, *, video_id, provenance, output, arms=tuple(ARMS)):
    """Save four paired attacks or an explicitly scoped A-only fresh replay.

    No restart/iterate is selected by independent scoring. Targets and every
    mask/start are made once before the first arm. Failures preserve committed
    partial artifacts but cannot write a passing worker record.
    """
    if tuple(arms) not in (tuple(ARMS), ('A',)):
        raise ValueError('Use all four arms or the declared A-only replay.')
    if (clean.dtype != torch.float32 or clean.ndim != 5 or clean.shape[0] != 1
            or clean.requires_grad or not torch.isfinite(clean).all() or clean.min() < 0 or clean.max() > 1
            or any(p.grad is not None for p in model.parameters())
            or any(p.requires_grad for p in model.target_encoder.parameters())
            or getattr(model, '_diagnostic_execution_policy', None) != POLICY):
        raise ValueError('Use one detached clean FP32 clip, gradient-free model and explicit frozen-target policy.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    saved_rng, modes = rng_state(clean.device.type), [(m, m.training) for m in model.modules()]
    guard = ExecutionGuard(model, clean.device.type, target_versions=True)
    before, clean_hash = model_identity(model), tensor_hash(clean)
    targets = None
    try:
        model.eval()
        targets = model.clean_targets(clean)
        target_hash = tensor_hash(targets)
        if targets.requires_grad or not torch.isfinite(targets).all():
            raise ValueError('Clean targets must be finite and detached.')
        seeds = _seeds(video_id)
        banks = {name: [[pair.to(clean.device) for pair in masks_for(seed)] for seed in values]
                 for name, values in seeds.items() if name != 'start'}
        starts = _starts(clean, seeds['start'])
        tensors = {'clean': clean, 'targets': targets,
                   **{'start_' + str(i): t for i, t in enumerate(starts)}}
        for bank, draws in banks.items():
            for i, families in enumerate(draws):
                for j, pair in enumerate(families):
                    tensors[f'{bank}_{i}_{j}_context'] = pair.context
                    tensors[f'{bank}_{i}_{j}_target'] = pair.target
        contract = {'kind': KIND, 'schema_version': 1, 'diagnostic_only': True, 'research_admitted': False,
                    'canonical_evaluation_artifact': False, 'policy': POLICY,
                    'video_id': video_id, 'base_seed': BASE_SEED, 'epsilon': EPSILON,
                    'step_size': STEP, 'restarts': RESTARTS, 'arms': {k: list(v) for k, v in ARMS.items()},
                    'seeds': seeds, 'clean_sha256': clean_hash, 'targets_sha256': target_hash,
                    'starts_sha256': [tensor_hash(t) for t in starts],
                    'mask_tensors_sha256': {n: tensor_hash(t) for n, t in tensors.items()
                                           if n not in ('clean', 'targets', 'start_0', 'start_1')},
                    'weights': before, 'runtime': runtime_contract(clean.device.type, model=model),
                    'provenance': provenance, 'selection': 'attack-loss-only-strict-greater-best-over-clean-starts-iterates',
                    'timing_is_instrumented_diagnostic_only': True, 'a40_cost_pool_eligible': False}
        write_record(output / 'contract.json', contract)
        evidence = _save_tensors(output, 'inputs.pt', tensors)
        write_record(output / 'inputs.json', {'contract_sha256': json_hash(contract), 'artifact': evidence})
        scores = {name: native_objective(model, targets, banks[name]) for name in ('main', 'audit_0', 'audit_1')}
        with torch.no_grad():
            clean_scores = {name: objective(clean).item() for name, objective in scores.items()}
        records = {}
        for arm in arms:
            print('Matched diagnostic arm:', arm, ARMS[arm], flush=True)
            steps, draws = ARMS[arm]
            objective = native_objective(model, targets, banks['attack'][:draws])
            trace = _Trace(objective, steps)
            synchronize(clean.device.type)
            started = time.monotonic()
            try:
                attack = pgd_linf(trace, clean, epsilon=EPSILON, step_size=STEP, steps=steps,
                                  restarts=RESTARTS, seed=seeds['start'])
            finally:
                trace.close()
            synchronize(clean.device.type)
            seconds = time.monotonic() - started
            adversarial = attack.adversarial
            if not torch.isfinite(adversarial).all() or adversarial.min() < 0 or adversarial.max() > 1:
                raise ValueError('Invalid adversarial pixels.')
            with torch.no_grad():
                reconstructed = objective(adversarial).item()
                independent = {name: {'clean': clean_scores[name], 'adversarial': score(adversarial).item()}
                               for name, score in scores.items()}
            guard.check()
            if (model_identity(model) != before or tensor_hash(clean) != clean_hash
                    or tensor_hash(targets) != target_hash or any(p.grad is not None for p in model.parameters())):
                raise ValueError('Attack mutated model, target, clean pixels or parameter gradients.')
            artifact = _save_tensors(output, arm + '.pt', {'adversarial': adversarial})
            record = {'arm': arm, 'contract_sha256': json_hash(contract),
                      'settings': {'steps': steps, 'attack_masks': draws}, 'trace': trace.rows,
                      'attack_loss': attack.losses.item(), 'attack_recomputed_exact': reconstructed == attack.losses.item(),
                      'adversarial_sha256': tensor_hash(adversarial),
                      'max_pixel_delta': (adversarial - clean).abs().max().item(), 'scores': independent,
                      'artifact': artifact, 'instrumented_attack_seconds': seconds}
            _trace_gate(record, contract)
            write_record(output / (arm + '.json'), record)
            records[arm] = record
        pairing = pairing_gate(records, contract) if tuple(arms) == tuple(ARMS) else None
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(saved_rng, clean.device.type)
        guard.check()
        if (model_identity(model) != before or tensor_hash(clean) != clean_hash
                or any(p.grad is not None for p in model.parameters())
                or (targets is not None and tensor_hash(targets) != target_hash)):
            raise ValueError('Diagnostic state preservation failed; evidence remains rejected.')
    result = {'kind': KIND, 'complete': True, 'passed': True, 'diagnostic_only': True,
              'research_admitted': False, 'contract_sha256': json_hash(contract), 'arms': list(arms),
              'pairing': pairing, 'state_and_rng_restored': True,
              'arm_records': {a: file_sha256(output / (a + '.json')) for a in arms}}
    write_record(output / 'worker.json', result)
    return result


def verify_worker(directory, *, expected_arms=tuple(ARMS)):
    """Validate all retained bytes, pairing, selection and raw-pixel bounds offline."""
    directory = Path(directory)
    worker, contract = read_record(directory / 'worker.json'), read_record(directory / 'contract.json')
    if (worker.get('kind') != KIND or not worker.get('complete') or not worker.get('passed')
            or worker['arms'] != list(expected_arms) or worker['research_admitted']
            or not worker['state_and_rng_restored'] or not worker['diagnostic_only']
            or worker['contract_sha256'] != json_hash(contract)
            or contract['kind'] != KIND or contract['policy'] != POLICY or contract['research_admitted']
            or not contract['diagnostic_only'] or contract['canonical_evaluation_artifact'] or contract['a40_cost_pool_eligible']
            or contract['seeds'] != _seeds(contract['video_id'])
            or (contract['epsilon'], contract['step_size'], contract['restarts']) != (EPSILON, STEP, RESTARTS)
            or contract['arms'] != {k: list(v) for k, v in ARMS.items()}):
        raise ValueError('Incomplete or mixed matched diagnostic contract.')
    inputs = read_record(directory / 'inputs.json')
    if inputs['contract_sha256'] != json_hash(contract):
        raise ValueError('Mixed diagnostic input contract.')
    tensors = _load_tensors(directory, inputs['artifact'])
    if (tensor_hash(tensors['clean']) != contract['clean_sha256']
            or tensor_hash(tensors['targets']) != contract['targets_sha256']
            or [tensor_hash(tensors['start_' + str(i)]) for i in range(RESTARTS)] != contract['starts_sha256']
            or [tensor_hash(t) for t in _starts(tensors['clean'], contract['seeds']['start'])] != contract['starts_sha256']
            or {n: tensor_hash(t) for n, t in tensors.items() if n not in ('clean', 'targets', 'start_0', 'start_1')}
            != contract['mask_tensors_sha256']):
        raise ValueError('Diagnostic clean/target/mask/start identities disagree.')
    records = {}
    for arm in expected_arms:
        record = read_record(directory / (arm + '.json'), expected_sha256=worker['arm_records'][arm])
        if record['arm'] != arm or record['contract_sha256'] != json_hash(contract):
            raise ValueError('Mixed matched arm.')
        _trace_gate(record, contract)
        adversarial = _load_tensors(directory, record['artifact'])['adversarial']
        if (adversarial.shape != tensors['clean'].shape or adversarial.dtype != torch.float32
                or tensor_hash(adversarial) != record['adversarial_sha256']
                or adversarial.min() < 0 or adversarial.max() > 1
                or (adversarial - tensors['clean']).abs().max().item() != record['max_pixel_delta']):
            raise ValueError('Saved adversarial pixels disagree with the arm record.')
        records[arm] = record
    if tuple(expected_arms) == tuple(ARMS):
        if worker['pairing'] != pairing_gate(records, contract):
            raise ValueError('Paired diagnostic gate differs.')
    if contract['runtime']['device'] == 'cuda':
        from models.native_probe import validate_probe
        observation = read_record(directory / 'observation.json')
        if (observation['kind'] != KIND or observation['research_admitted']
                or observation['role'] != ('matrix' if tuple(expected_arms) == tuple(ARMS) else 'replay')
                or observation['timing_pool_eligible']
                or not 0 < observation['memory']['cuda_peak_allocated_bytes']
                          <= contract['provenance']['allocator_limit_bytes']):
            raise ValueError('Missing/invalid bounded CUDA observation.')
        validate_probe(observation['native_probe'], device='cuda')
    return contract, records


def compare_replay(reference, replay):
    contract, rows = verify_worker(reference)
    replay_contract, replay_rows = verify_worker(replay, expected_arms=('A',))
    if contract != replay_contract:
        raise ValueError('Fresh replay runtime/source/model/data contract differs.')
    omit = {'artifact', 'instrumented_attack_seconds'}
    a = {k: v for k, v in rows['A'].items() if k not in omit}
    b = {k: v for k, v in replay_rows['A'].items() if k not in omit}
    if a != b or rows['A']['artifact']['tensors'] != replay_rows['A']['artifact']['tensors']:
        raise ValueError('Fresh A-only replay is not exact.')
    return {'passed': True, 'scope': 'fresh-process-A-only-20-steps-EOT4',
            'all_four_arms_fresh_replayed': False}
