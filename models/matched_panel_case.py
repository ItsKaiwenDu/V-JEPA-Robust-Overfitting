"""Explicit plan-bound four-arm cases; no canonical evaluation or global seed edits."""

import json
import math
from pathlib import Path
import time

import torch

from attacks.pgd import native_objective, pgd_linf
from models.evaluation import model_identity
from models.evaluation_store import json_hash
from models.execution import ExecutionGuard
from models.matched_attack import (_Trace, _load_tensors, _save_tensors, _starts, _trace_gate,
                                   pairing_gate, tensor_hash)
from models.matched_panel_plan import ARMS, KIND as PLAN_KIND, plan_digest
from models.pilot_bundle import digest
from models.runtime import synchronize
from models.study_artifacts import read_record, write_record
from models.training import restore_rng, rng_state, runtime_contract

KIND = 'matched-panel-case-v1'
POLICY = 'shared-fp64-native'
FIRST_CASE = 'clip-00-bank-0'
PLAN_SHA256 = 'f6b81dec841aeee703413dba23ea16afe3dbf14736edecadc0f7830f3ded34ae'
PLAN_FILE_SHA256 = '996717379099048f1ceb2691b2eaf62914710cead6aef92bf96d0af46408bb84'


def exact(a, b):
    return plan_digest(a) == plan_digest(b)


def case_spec(plan, case_id):
    if plan.get('kind') != PLAN_KIND:
        raise ValueError('Wrong panel plan namespace.')
    selected = [c for c in plan['cases'] if c['case_id'] == case_id]
    if len(selected) != 1:
        raise ValueError('Case is not uniquely declared in the frozen plan.')
    return selected[0]


def pixel_digest(pixels):
    import hashlib
    return hashlib.sha256(pixels.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def execution_gate(plan, execution, *, allow_fixture=False):
    if execution.get('plan_sha256') != plan_digest(plan):
        raise ValueError('Execution belongs to a different panel plan.')
    domain = execution.get('domain')
    if domain == 'university-cuda':
        if plan_digest(plan) != PLAN_SHA256 or execution['expected_runtime']['device'] != 'cuda':
            raise ValueError('Use the frozen production plan and declared CUDA runtime.')
    elif domain != 'generated-cpu-fixture' or not allow_fixture or execution['expected_runtime']['device'] != 'cpu':
        raise ValueError('Generated CPU evidence cannot qualify as a university CUDA case.')
    if execution.get('research_admitted') is not False:
        raise ValueError('Execution identity cannot admit research.')


def run_worker(model, clean, masks_for, *, plan, case_id, execution, output, role,
               allow_fixture=False):
    """All seeds are explicit data; actual video IDs never carry bank labels."""
    execution_gate(plan, execution, allow_fixture=allow_fixture)
    case = case_spec(plan, case_id)
    clip = plan['ordered_clips'][case['clip_index']]
    arms = tuple(ARMS) if role == 'matrix' else ('A',)
    if role not in ('matrix', 'replay'):
        raise ValueError('Unknown worker role.')
    if (clean.dtype != torch.float32 or clean.ndim != 5 or clean.shape[0] != 1 or clean.requires_grad
            or not torch.isfinite(clean).all() or clean.min() < 0 or clean.max() > 1
            or pixel_digest(clean[0]) != clip['pixel_sha256']
            or case['video_id'] != clip['video_id']
            or any(p.grad is not None for p in model.parameters())
            or any(p.requires_grad for p in model.target_encoder.parameters())
            or getattr(model, '_diagnostic_execution_policy', None) != POLICY):
        raise ValueError('Require the exact declared clean clip and gradient-free fixed-target model.')
    device = clean.device.type
    runtime = runtime_contract(device, model=model)
    weights = model_identity(model)
    if not exact(runtime, execution['expected_runtime']) or weights != execution['expected_weights']:
        raise ValueError('Actual runtime or loaded weights differ from the execution identity.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    saved_rng, modes = rng_state(device), [(m, m.training) for m in model.modules()]
    guard = ExecutionGuard(model, device, target_versions=True)
    clean_hash, targets, target_hash = tensor_hash(clean), None, None
    try:
        model.eval()
        targets = model.clean_targets(clean)
        target_hash = tensor_hash(targets)
        if targets.requires_grad or not torch.isfinite(targets).all():
            raise ValueError('Clean target features must be finite and detached.')
        seeds = {'attack': case['attack_seeds'], **case['scoring_seeds']}
        banks = {name: [[p.to(clean.device) for p in masks_for(seed)] for seed in values]
                 for name, values in seeds.items()}
        starts = _starts(clean, case['start_seed'])
        tensors = {'clean': clean, 'targets': targets,
                   **{'start_' + str(i): value for i, value in enumerate(starts)}}
        for name, draws in banks.items():
            for draw, families in enumerate(draws):
                for family, pair in enumerate(families):
                    tensors[f'{name}_{draw}_{family}_context'] = pair.context
                    tensors[f'{name}_{draw}_{family}_target'] = pair.target
        contract = {'kind': KIND, 'schema_version': 1, 'plan_sha256': plan_digest(plan), 'case': case,
            'execution_sha256': json_hash(execution), 'runtime': runtime, 'weights': weights,
            'clip': clip, 'clean_sha256': clean_hash, 'targets_sha256': target_hash,
            'starts_sha256': [tensor_hash(t) for t in starts],
            'mask_tensors_sha256': {n: tensor_hash(t) for n, t in tensors.items()
                                    if n not in ('clean', 'targets', 'start_0', 'start_1')},
            'diagnostic_only': True, 'research_admitted': False,
            'canonical_evaluation_artifact': False, 'A40_cost_pool_eligible': False,
            'instrumented_timing_only': True, 'settings': plan['settings']}
        write_record(output / 'contract.json', contract)
        artifact = _save_tensors(output, 'inputs.pt', tensors)
        write_record(output / 'inputs.json', {'contract_sha256': json_hash(contract), 'artifact': artifact})
        scores = {name: native_objective(model, targets, banks[name]) for name in ('main', 'audit_0', 'audit_1')}
        with torch.no_grad():
            clean_scores = {name: score(clean).item() for name, score in scores.items()}
        records = {}
        for arm in arms:
            setting = ARMS[arm]
            print('Panel case:', case_id, 'arm:', arm, setting, flush=True)
            objective = native_objective(model, targets, banks['attack'][:setting['attack_masks']])
            traced = _Trace(objective, setting['steps'])
            synchronize(device)
            started = time.monotonic()
            try:
                attack = pgd_linf(traced, clean, epsilon=4/255, step_size=2/255,
                                  steps=setting['steps'], restarts=2, seed=case['start_seed'])
            finally:
                traced.close()
            synchronize(device)
            seconds = time.monotonic() - started
            adv = attack.adversarial
            if not torch.isfinite(adv).all() or adv.min() < 0 or adv.max() > 1:
                raise ValueError('Invalid adversarial pixels.')
            with torch.no_grad():
                reconstructed = objective(adv).item()
                independent = {n: {'clean': clean_scores[n], 'adversarial': score(adv).item()}
                               for n, score in scores.items()}
            guard.check()
            if (model_identity(model) != weights or tensor_hash(clean) != clean_hash
                    or tensor_hash(targets) != target_hash or any(p.grad is not None for p in model.parameters())):
                raise ValueError('Attack changed model, targets, pixels or parameter gradients.')
            reference = _save_tensors(output, arm + '.pt', {'adversarial': adv})
            record = {'arm': arm, 'contract_sha256': json_hash(contract), 'settings': setting,
                'trace': traced.rows, 'attack_loss': attack.losses.item(),
                'attack_recomputed_exact': reconstructed == attack.losses.item(),
                'adversarial_sha256': tensor_hash(adv), 'max_pixel_delta': (adv-clean).abs().max().item(),
                'scores': independent, 'artifact': reference, 'instrumented_attack_seconds': seconds}
            _trace_gate(record, contract)
            write_record(output / (arm + '.json'), record)
            records[arm] = record
        pairing = pairing_gate(records, contract) if role == 'matrix' else None
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(saved_rng, device)
        guard.check()
        if (model_identity(model) != weights or tensor_hash(clean) != clean_hash
                or any(p.grad is not None for p in model.parameters())
                or (targets is not None and tensor_hash(targets) != target_hash)):
            raise ValueError('Panel worker state-preservation gate failed.')
    result = {'kind': KIND, 'role': role, 'complete': True, 'passed': True, 'diagnostic_only': True,
              'research_admitted': False, 'contract_sha256': json_hash(contract), 'arms': list(arms),
              'pairing': pairing, 'state_and_rng_restored': True,
              'arm_records': {a: digest(output / (a + '.json')) for a in arms}}
    write_record(output / 'worker.json', result)
    return result


def verify_worker(directory, *, plan, case_id, execution, role, allow_fixture=False):
    execution_gate(plan, execution, allow_fixture=allow_fixture)
    directory = Path(directory)
    contract, worker = read_record(directory / 'contract.json'), read_record(directory / 'worker.json')
    case = case_spec(plan, case_id)
    arms = list(ARMS) if role == 'matrix' else ['A']
    expected = {'kind': KIND, 'plan_sha256': plan_digest(plan), 'case': case,
                'execution_sha256': json_hash(execution), 'runtime': execution['expected_runtime'],
                'weights': execution['expected_weights'], 'clip': plan['ordered_clips'][case['clip_index']],
                'settings': plan['settings'], 'diagnostic_only': True, 'research_admitted': False,
                'canonical_evaluation_artifact': False, 'A40_cost_pool_eligible': False,
                'instrumented_timing_only': True, 'schema_version': 1}
    if any(k not in contract or not exact(contract[k], value) for k, value in expected.items()):
        raise ValueError('Mixed plan, case, runtime, weights or execution-source contract.')
    if (worker['kind'] != KIND or worker['role'] != role or worker['arms'] != arms
            or worker['complete'] is not True or worker['passed'] is not True
            or worker['diagnostic_only'] is not True or worker['research_admitted'] is not False
            or worker['state_and_rng_restored'] is not True
            or worker['contract_sha256'] != json_hash(contract) or set(worker['arm_records']) != set(arms)):
        raise ValueError('Incomplete panel worker or missing replay.')
    input_record = read_record(directory / 'inputs.json')
    if input_record['contract_sha256'] != json_hash(contract):
        raise ValueError('Input tensors belong to a different case.')
    tensors = _load_tensors(directory, input_record['artifact'])
    expected_names = {'clean', 'targets', 'start_0', 'start_1'}
    for bank, count in (('attack',8), ('main',4), ('audit_0',4), ('audit_1',4)):
        expected_names.update(f'{bank}_{draw}_{family}_{which}' for draw in range(count)
                              for family in range(2) for which in ('context','target'))
    if set(tensors) != expected_names:
        raise ValueError('Incomplete mask/target/start tensor inventory.')
    clean = tensors['clean']
    if (clean.dtype != torch.float32 or clean.ndim != 5 or clean.shape[0] != 1
            or clean.min()<0 or clean.max()>1 or tensor_hash(clean) != contract['clean_sha256']
            or pixel_digest(clean[0]) != contract['clip']['pixel_sha256']
            or tensor_hash(tensors['targets']) != contract['targets_sha256']
            or [tensor_hash(tensors[f'start_{i}']) for i in range(2)] != contract['starts_sha256']
            or {n:tensor_hash(t) for n,t in tensors.items() if n not in ('clean','targets','start_0','start_1')}
            != contract['mask_tensors_sha256']):
        raise ValueError('Declared pixels, clean targets, masks or starts disagree.')
    patch=execution['expected_runtime']['model_execution']['context_patch']
    kt,kh,kw=patch['kernel']
    tokens=(clean.shape[2]//kt)*(clean.shape[3]//kh)*(clean.shape[4]//kw)
    if tensors['targets'].dtype!=torch.float32 or tuple(tensors['targets'].shape)!=(1,tokens,patch['weight_shape'][0]):
        raise ValueError('Wrong clean-target dtype or token geometry.')
    for i in range(2):
        start=tensors[f'start_{i}']
        if (start.dtype!=torch.float32 or start.shape!=clean.shape or start.min()<0 or start.max()>1
                or (start-clean).abs().max()>4/255+1e-7):
            raise ValueError('Invalid bounded start pixels.')
    for bank,count in (('attack',8),('main',4),('audit_0',4),('audit_1',4)):
        for draw in range(count):
            for family in range(2):
                ctx=tensors[f'{bank}_{draw}_{family}_context']
                target=tensors[f'{bank}_{draw}_{family}_target']
                for value in (ctx,target):
                    if (value.dtype!=torch.int64 or value.ndim!=2 or value.shape[0]!=1 or value.shape[1]==0
                            or value.min()<0 or value.max()>=tokens or value.unique().numel()!=value.numel()):
                        raise ValueError('Invalid retained mask geometry or indices.')
                if torch.isin(ctx,target).any():raise ValueError('Context and target masks overlap.')
    # Same-runtime readers can independently regenerate starts. A Mac reader
    # must not silently add a new cross-version CPU-RNG acceptance requirement.
    start_regenerated = str(torch.__version__) == contract['runtime']['torch']
    if start_regenerated and [tensor_hash(t) for t in _starts(clean,case['start_seed'])] != contract['starts_sha256']:
        raise ValueError('Saved starts differ from their declared seed in the qualified runtime.')
    if execution['domain']=='university-cuda' and tuple(clean.shape)!=(1,3,16,256,256):
        raise ValueError('Wrong production clip geometry.')
    records={}
    for arm in arms:
        record=read_record(directory/(arm+'.json'),expected_sha256=worker['arm_records'][arm])
        if record['arm']!=arm or record['contract_sha256']!=json_hash(contract):
            raise ValueError('Mixed arm record.')
        _trace_gate(record,contract)
        adv=_load_tensors(directory,record['artifact'])['adversarial']
        if (adv.shape!=clean.shape or adv.dtype!=torch.float32 or adv.min()<0 or adv.max()>1
                or tensor_hash(adv)!=record['adversarial_sha256']
                or (adv-clean).abs().max().item()!=record['max_pixel_delta']):
            raise ValueError('Saved adversary differs from the selected bounded candidate.')
        records[arm]=record
    if role=='matrix' and worker['pairing']!=pairing_gate(records,contract):
        raise ValueError('Exact within-EOT prefix gate differs.')
    if execution['domain']=='university-cuda':
        from models.native_probe import validate_probe
        obs=read_record(directory/'observation.json')
        if (obs['role']!=role or obs['case_id']!=case_id or obs['execution_sha256']!=json_hash(execution)
                or obs['kind']!=KIND or obs['research_admitted'] is not False or obs['A40_cost_pool_eligible'] is not False
                or obs['actual_host']!=execution['expected_host']
                or obs['actual_GPU_identity']!=execution['expected_GPU_identity']
                or not 0<obs['memory']['cuda_peak_allocated_bytes']<=obs['allocator_limit_bytes']<=7*2**30):
            raise ValueError('Invalid/mixed CUDA observation or memory bound.')
        validate_probe(obs['native_probe'],device='cuda')
    return contract,records,{'starts_regenerated_in_matching_runtime':start_regenerated}


def verify_pair(directory, *, plan, case_id, execution, allow_fixture=False):
    directory=Path(directory)
    a,rows,_=verify_worker(directory/'matrix',plan=plan,case_id=case_id,execution=execution,
                          role='matrix',allow_fixture=allow_fixture)
    b,replay,_=verify_worker(directory/'replay',plan=plan,case_id=case_id,execution=execution,
                            role='replay',allow_fixture=allow_fixture)
    ignored={'artifact','instrumented_attack_seconds'}
    if (not exact(a,b) or {k:v for k,v in rows['A'].items() if k not in ignored}
            !={k:v for k,v in replay['A'].items() if k not in ignored}
            or rows['A']['artifact']['tensors']!=replay['A']['artifact']['tensors']):
        raise ValueError('Fresh A-only replay is not exact.')
    return a,rows


def verify_case(directory, *, plan, case_id, execution, allow_fixture=False):
    directory=Path(directory)
    a,rows=verify_pair(directory,plan=plan,case_id=case_id,execution=execution,allow_fixture=allow_fixture)
    session=read_record(directory/'session.json')
    if (session['kind']!=KIND or session['plan_sha256']!=plan_digest(plan) or session['case_id']!=case_id
            or session['execution_sha256']!=json_hash(execution) or session['passed'] is not True
            or session['complete'] is not True or session['research_admitted'] is not False
            or set(session['tasks'])!={'matrix','replay'}
            or any(not exact(t,{'complete':True,'exit_status':0,'timed_out':False}) for t in session['tasks'].values())
            or not isinstance(session['elapsed_seconds'],(int,float)) or not math.isfinite(session['elapsed_seconds'])
            or not 0<session['elapsed_seconds']<=plan['execution_requirements']['parent_seconds']):
        raise ValueError('Missing, failed or mixed case session.')
    return {'kind':KIND,'case_id':case_id,'plan_sha256':plan_digest(plan),
        'execution_sha256':json_hash(execution),'contract':a,
        'scores':{arm:r['scores'] for arm,r in rows.items()},
        'session_sha256':digest(directory/'session.json'),
        'worker_files_sha256':{str(p.relative_to(directory)):digest(p)
            for name in ('matrix','replay') for p in (directory/name).iterdir()
            if p.is_file() and p.suffix in ('.json','.pt')},
        'fresh_A_replay_exact':True,'research_admitted':False}
