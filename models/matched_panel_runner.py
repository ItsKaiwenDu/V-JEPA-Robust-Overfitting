"""Bounded university workers and serial whole-case scheduling; no rental API."""

from contextlib import contextmanager
import datetime
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import uuid

import numpy as np
import torch

from models.evaluation import EvaluationMaskSampler
from models.evaluation_store import json_hash
from models.execution import apply_model_policy,configure_execution
from models.matched_diagnostic import bounded_child
from models.matched_panel import PanelJournal
from models.matched_panel_case import FIRST_CASE,KIND,POLICY,run_worker,verify_case,verify_pair
from models.matched_panel_io import (execution_identity,export_case,load_clip,load_plan,verify_archive)
from models.pilot_bundle import digest,make_archive
from models.runtime import MemoryMonitor
from models.study_artifacts import read_record,write_record
from models.training import runtime_contract
from models.vjepa2 import build_vitl,load_checkpoint


@contextmanager
def case_deadline(seconds,*,kill_grace_seconds=15):
    """Bound the whole serial case, including verification/export/pointer I/O.

    The first alarm unwinds bounded_child, which terminates its process group.
    A second alarm ends this supervisor if cleanup itself exceeds the grace.
    """
    if signal.getitimer(signal.ITIMER_REAL)!=(0.,0.):
        raise RuntimeError('Refusing to replace another active case deadline.')
    old_handler=signal.getsignal(signal.SIGALRM)
    def hard_stop(*_):os._exit(124)
    def stop(*_):
        signal.signal(signal.SIGALRM,hard_stop)
        signal.setitimer(signal.ITIMER_REAL,kill_grace_seconds)
        raise TimeoutError('Whole-case outer deadline reached; stop the queue and preserve evidence.')
    signal.signal(signal.SIGALRM,stop)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_handler)


def hardware_gate(execution,*,allow_self=False):
    identity=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version',
                                      '--format=csv,noheader,nounits'],text=True).strip()
    if platform.node()!=execution['expected_host'] or identity!=execution['expected_GPU_identity']:
        raise ValueError('Host/GPU UUID/driver identity differs from the frozen university reference.')
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    allowed={os.getpid()} if allow_self else set()
    if any(int(p.strip()) not in allowed for p in output.splitlines() if p.strip()):
        raise ValueError('Another CUDA compute process is active; stop before this case.')
    return identity


def worker(root,args):
    root=Path(root)
    plan=load_plan(root)
    execution=execution_identity(root,plan)
    expected=execution['expected_runtime']
    configure_execution(POLICY,device='cuda')
    torch.set_num_threads(expected['threads'])
    if (platform.python_version()!=expected['python'] or str(torch.__version__)!=expected['torch']
            or np.__version__!=expected['numpy']):
        raise ValueError('Use the existing frozen university software; no install/fallback.')
    hardware=hardware_gate(execution)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Exactly one CUDA GPU is required.')
    total=torch.cuda.get_device_properties(0).total_memory
    limit=min(7*2**30,int(.9*total))
    torch.cuda.set_per_process_memory_fraction(min(.9,7*2**30/total))
    from models.precision_diagnostic import native_probe
    probe=native_probe('cuda')
    hardware_gate(execution,allow_self=True)
    torch.manual_seed(20261024)
    clean=load_clip(root,plan,args.case_id)
    model=build_vitl(root/'external/vjepa2',activation_checkpointing=True)
    load_checkpoint(model,args.checkpoint,execution['checkpoint_sha256'])
    apply_model_policy(model,POLICY)
    model.to('cuda')
    if runtime_contract('cuda',model=model)!=expected:
        raise ValueError('Actual model/runtime/precision differs from the frozen reference.')
    with MemoryMonitor('cuda') as memory:
        run_worker(model,clean.to('cuda'),EvaluationMaskSampler(root/'external/vjepa2'),plan=plan,
            case_id=args.case_id,execution=execution,output=args.output,role=args.role)
    write_record(Path(args.output)/'observation.json',{'kind':KIND,'case_id':args.case_id,
        'role':args.role,'execution_sha256':json_hash(execution),'actual_host':platform.node(),
        'actual_GPU_identity':hardware,'memory':memory.report(),'allocator_limit_bytes':limit,
        'native_probe':probe,'research_admitted':False,'A40_cost_pool_eligible':False})


def case_session(root,plan,execution,case_id,directory,checkpoint,archive,*,parent_seconds=4620):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    result={'kind':KIND,'case_id':case_id,'plan_sha256':execution['plan_sha256'],
            'execution_sha256':json_hash(execution),'complete':False,'passed':False,
            'research_admitted':False,'tasks':{},'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    started=time.monotonic()
    def stop(*_):raise TimeoutError('Case supervisor terminated; preserve partial evidence.')
    old_handler=signal.signal(signal.SIGTERM,stop)
    finalized=False
    try:
        write_record(directory/'session.json',{**result,'elapsed_seconds':0.})
        for role,cap in [('matrix',3600),('replay',900)]:
            remaining=parent_seconds-(time.monotonic()-started)-30
            if remaining<1:raise TimeoutError('No case time remains for a required worker.')
            command=[sys.executable,'-u',str(Path(root)/'scripts/check_matched_panel.py'),'worker',
                     '--diagnostic','--case-id',case_id,'--role',role,'--output',str(directory/role),
                     '--checkpoint',str(checkpoint)]
            result['tasks'][role]=bounded_child(command,directory/(role+'.log'),min(cap,remaining))
            write_record(directory/'session.json',{**result,'elapsed_seconds':time.monotonic()-started},replace=True)
            if not result['tasks'][role]['complete']:
                raise RuntimeError('Worker failed/timed out; queue must stop without skipping it.')
        verify_pair(directory,plan=plan,case_id=case_id,execution=execution)
        result.update(complete=True,passed=True,elapsed_seconds=time.monotonic()-started)
        write_record(directory/'session.json',result,replace=True)
        receipt=verify_case(directory,plan=plan,case_id=case_id,execution=execution)
        write_record(directory/'receipt.json',receipt)
        finalized=True
    except Exception as error:
        result.update(complete=False,passed=False,error=type(error).__name__+': '+str(error),
                      elapsed_seconds=time.monotonic()-started)
        write_record(directory/'session.json',result,replace=True)
    finally:
        try:
            manifest=export_case(root,directory,plan,execution,destination=archive)
            verify_archive(archive,manifest,manifest['sha256'])
            print('Panel case export:',case_id,{'archive':str(archive),'bytes':manifest['bytes'],
                  'sha256':manifest['sha256'],'members':len(manifest['files']),'passed':finalized},flush=True)
        finally:
            signal.signal(signal.SIGTERM,old_handler)
    if not finalized:raise RuntimeError(result['error'])
    return result


def next_attempt(journal,case_id,*,explicit_retry=False):
    """Recover only a wholly verified case. Partial attempts never retry themselves."""
    state=journal.read()
    if case_id in state['completed']:
        journal.verify_completed(case_id)
        return None,None,None
    container=journal.root/'cases'/case_id
    previous=sorted(container.glob('attempt-*'))
    spent=0.
    for attempt in previous:
        archive=journal.root/'exports'/(case_id+'-'+attempt.name+'.tar.gz')
        try:
            journal.commit(case_id,attempt,archive=archive)
            print('Recovered verified uncommitted case:',case_id,flush=True)
            return None,None,None
        except (ValueError,FileNotFoundError,KeyError,json.JSONDecodeError):
            try:
                receipt=verify_case(attempt,plan=journal.plan,case_id=case_id,execution=journal.execution,
                                    allow_fixture=journal.allow_fixture)
                raw_complete=read_record(attempt/'receipt.json')==receipt
            except (ValueError,FileNotFoundError,KeyError,json.JSONDecodeError):raw_complete=False
            if raw_complete:
                raise ValueError('Completed numerical evidence needs export/pointer review; do not repeat GPU work to repair it.')
            if not explicit_retry:
                raise ValueError('Incomplete/failed attempt exists; preserve it and request an explicit retry after review.')
            try:elapsed=read_record(attempt/'session.json')['elapsed_seconds']
            except (FileNotFoundError,KeyError,ValueError):elapsed=4800
            if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:
                elapsed=4800
            spent+=min(4800,elapsed+180)
    remaining=4800-spent
    if remaining<=180:
        raise ValueError('The original per-case allowance is exhausted; no deadline reset is allowed.')
    container.mkdir(parents=True,exist_ok=True)
    name='attempt-'+str(len(previous)+1).zfill(2)
    return container/name,journal.root/'exports'/(case_id+'-'+name+'.tar.gz'),min(4620,remaining-180)


def export_index(journal):
    state=journal.read()
    summary=journal.aggregate()
    write_record(journal.root/'panel.json',summary,replace=True)
    files={'journal.json':journal.root/'journal.json','panel.json':journal.root/'panel.json'}
    for case_id,pointer in state['completed'].items():
        files['receipts/'+case_id+'.json']=journal.root/pointer['directory']/'receipt.json'
        files['manifests/'+case_id+'.json']=journal.root/pointer['archive_manifest_path']
    if state['first_acceptance'] is not None:
        write_record(journal.root/'first-case-acceptance.json',state['first_acceptance'],replace=True)
        files['first-case-acceptance.json']=journal.root/'first-case-acceptance.json'
    path=journal.root/'exports'/('panel-index-'+str(len(state['completed'])).zfill(2)+'-'+uuid.uuid4().hex[:8]+'.tar.gz')
    path.parent.mkdir(parents=True,exist_ok=True)
    manifest=make_archive(path,files)
    Path(str(path)+'.manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    write_record(journal.root/'latest-index.json',{'file':path.name,'sha256':manifest['sha256'],
                 'complete':summary['complete'],'completed_cases':len(state['completed'])},replace=True)
    print('Panel index:',path,manifest['sha256'],'complete:',summary['complete'],flush=True)
    return summary


def run_panel(root,args):
    root=Path(root)
    plan=load_plan(root)
    execution=execution_identity(root,plan)
    hardware_gate(execution)
    journal=PanelJournal(args.output,plan=plan,execution=execution)
    with journal.lease():
        state=journal.read()
        remaining=20-len(state['completed'])
        needed=(12*2**30 if not state['completed'] else max(2*2**30,(remaining*512*2**20)+2*2**30))
        if shutil.disk_usage(journal.root).free<needed:
            raise OSError('Insufficient free space for the declared remaining evidence and reserve.')
        if args.mode=='remaining':
            if args.acceptance is None:raise ValueError('Supply the independent first-case export review receipt.')
            journal.accept_first(read_record(args.acceptance))
            ids=journal.header['case_order']
        elif args.mode=='first': ids=[FIRST_CASE]
        elif args.mode=='retry':
            if args.case_id not in journal.header['case_order']:
                raise ValueError('Unknown retry case.')
            pending=[c for c in journal.header['case_order'] if c not in state['completed']]
            if args.case_id not in state['completed'] and (not pending or args.case_id!=pending[0]):
                raise ValueError('Only the next declared incomplete case may be explicitly retried.')
            if args.case_id!=FIRST_CASE and args.acceptance is None:
                raise ValueError('First-case review is required before retrying later cases.')
            if args.acceptance is not None:journal.accept_first(read_record(args.acceptance))
            ids=[args.case_id]
        else:raise ValueError('Unknown panel execution mode.')
        error=None
        try:
            for case_id in ids:
                state=journal.read()
                if case_id!=FIRST_CASE and state['first_acceptance'] is None:
                    raise ValueError('First export has not been independently accepted.')
                attempt,archive,seconds=next_attempt(journal,case_id,explicit_retry=args.mode=='retry')
                if attempt is None:continue
                if args.mode=='retry' and case_id not in [c['case_id'] for c in plan['cases']]:
                    raise ValueError('Unknown retry case.')
                hardware_gate(execution)
                print('Starting panel case:',case_id,'at',attempt,flush=True)
                with case_deadline(seconds+180):
                    case_session(root,plan,execution,case_id,attempt,args.checkpoint,archive,parent_seconds=seconds)
                    journal.commit(case_id,attempt,archive=archive)
        except Exception as caught:
            error=caught
        finally:
            summary=export_index(journal)
        if error is not None:raise error
        return summary
