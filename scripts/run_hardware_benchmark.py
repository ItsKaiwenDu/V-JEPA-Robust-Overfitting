#!/usr/bin/env python3
"""Bounded seven-phase GPU cost screen on an already reviewed, deployed host."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.hardware_benchmark import GPU_MIN_GIB, SCREEN_PHASES, economic_screen, validate_gpu, validate_launch
from scripts.run_workflow_pilot import run_child


def collect_observations(output):
    from models.study_artifacts import read_record
    from models.evaluation_store import validate_records
    observations = {name: [] for name in ('routine','primary','convergence','fgsm','random',
        'clean-finetune','adversarial-finetune','engine.state','checkpoint.save',
        'calibration_three_primary_candidates','load_anchor')}
    for path in sorted(output.glob('**/evaluation.json')):
        value = read_record(path)
        validate_records(value)
        if not value['complete']:
            raise ValueError('Incomplete native evaluation is not a timing pass.')
        for row in value['records']:
            if row['method'] == 'convergence' and len(row['scoring_audits']) != 2:
                raise ValueError('Require both convergence scoring-bank audits.')
            observations[row['method']].append(row['seconds'])
    for path in sorted((output/'timings').glob('**/*.jsonl')):
        phase = str(path.relative_to(output/'timings'))
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event['status'] != 'ok':
                raise ValueError('A failed timing span prevents a completed economic screen.')
            name = event['name']
            if name == 'engine.update':
                condition = phase.split('/')[0]
                observations[condition].append(event['seconds'])
            elif name in observations:
                observations[name].append(event['seconds'])
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--expected-gpu', choices=tuple(GPU_MIN_GIB), required=True)
    parser.add_argument('--hourly-rate-usd', type=float, required=True)
    parser.add_argument('--deadline-epoch', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--clips', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=ROOT/'external/vjepa2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=5400)
    parser.add_argument('--phase', choices=SCREEN_PHASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.diagnostic or not 1 <= args.max_seconds <= 6600:
        parser.error('Use --diagnostic with a positive worker allowance at most 6600 seconds.')
    if not args.phase:
        try:
            contract = validate_launch(args.expected_gpu,args.hourly_rate_usd,int(time.time()),args.deadline_epoch)
        except ValueError as exc:
            parser.error(str(exc))
    args.output = args.output.resolve()
    if (not args.phase and args.output.exists()) or (args.phase and not (args.output/'session.json').is_file()):
        parser.error('Parent needs a fresh output; phases require its existing session.')

    import torch
    from models.checkpointing import file_sha256
    from models.execution import configure_execution
    from models.runtime import device_metadata, require_device
    from models.study_artifacts import read_record, write_record
    from models.training import runtime_contract
    from models.workflow_pilot import run_phase

    configure_execution('shared-fp64-native',device='cuda',strict_original=True)
    require_device('cuda')
    metadata = device_metadata('cuda')
    validate_gpu(args.expected_gpu,metadata['gpu'],metadata['vram_bytes'],torch.cuda.device_count())
    args.device, args.limit, args.execution_policy = 'cuda', 1, 'shared-fp64-native'
    args.feature_anchor_variant, args.hardware_screen_only = 'reference', True
    if args.phase:
        parent = read_record(args.output/'session.json')
        if (parent['expected_gpu'] != args.expected_gpu or parent['deadline_epoch'] != args.deadline_epoch
                or parent['quoted_hourly_usd'] != args.hourly_rate_usd or parent['complete']
                or time.time() >= args.deadline_epoch - 240):
            raise ValueError('Worker does not match the live parent hardware/deadline contract.')
        print(json.dumps(run_phase(ROOT,args),indent=2))
        return
    existing = args.output.parent
    while not existing.exists(): existing = existing.parent
    if shutil.disk_usage(existing).free < 55*2**30:
        parser.error('Retain 80 GB persistent storage and at least 55 GiB free after setup.')
    args.output.mkdir(parents=True)
    (args.output/'logs').mkdir()
    started = time.monotonic()
    session = {**contract, 'schema_version':1,'diagnostic_only':True,'research_result':False,
        'budget_approved':False,'complete':False,'execution_policy':'shared-fp64-native',
        'device':metadata,'runtime':runtime_contract('cuda'),'required_phases':list(SCREEN_PHASES),
        'phases':[],'exact_optimizer_recovery_checked':False,'profile':'hardware-cost-screen-v1',
        'source_sha256':{name:file_sha256(ROOT/name) for name in ('models/hardware_benchmark.py',
            'scripts/run_hardware_benchmark.py','models/workflow_pilot.py')},
        'started_unix_seconds':time.time()}
    path = args.output/'session.json'
    write_record(path,session)
    try:
        for phase in SCREEN_PHASES:
            remaining = min(args.max_seconds-(time.monotonic()-started),args.deadline_epoch-time.time()-240)
            if remaining <= 0: raise TimeoutError('No reviewed worker/export allowance remains.')
            print('Hardware screen phase:',phase,flush=True)
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--diagnostic','--phase',phase,
                '--expected-gpu',args.expected_gpu,'--hourly-rate-usd',str(args.hourly_rate_usd),
                '--deadline-epoch',str(args.deadline_epoch),'--checkpoint',str(args.checkpoint.resolve()),
                '--manifests',str(args.manifests.resolve()),'--clips',str(args.clips.resolve()),
                '--upstream',str(args.upstream.resolve()),'--output',str(args.output)]
            before = time.monotonic()
            run_child(command,args.output/'logs'/(phase.replace('/','-')+'.log'),min(1200,remaining) if phase=='preflight' else remaining)
            result_path=args.output/'phase-results'/(phase+'.json')
            result=read_record(result_path)
            if phase=='preflight' and (not result['preflight']['passed'] or not result['preflight']['complete']):
                raise ValueError('Preflight failed; do not continue to timing.')
            session['phases'].append({'phase':phase,'process_wall_seconds':time.monotonic()-before,
                                       'result_sha256':file_sha256(result_path)})
            write_record(path,session,replace=True)
        reference_path=ROOT/'report/checks/2026-10-22-measured-budget-review.json'
        observations=collect_observations(args.output)
        summary=economic_screen(json.loads(reference_path.read_text()),observations,args.hourly_rate_usd)
        summary.update(reference_sha256=file_sha256(reference_path),observations=observations)
        write_record(args.output/'economic-screen.json',summary)
        session['economic_screen_sha256']=file_sha256(args.output/'economic-screen.json')
        session['complete']=True
        print('Cost screen complete; full recovery and research admission remain pending.',flush=True)
    except BaseException as exc:
        session.update(error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        session['wall_seconds']=time.monotonic()-started
        write_record(path,session,replace=True)


if __name__=='__main__': main()
