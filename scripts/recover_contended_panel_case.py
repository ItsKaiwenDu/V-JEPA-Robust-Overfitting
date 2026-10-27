#!/usr/bin/env python3
"""One explicitly authorized, attended retry; keep the original ledger stopped.

The scientific plan, checkpoint, attack code, numerical verifiers, and strict
two-host GPU guard are imported from their frozen packets. This separate
operator has its own source/authorization identity and never edits old events.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

KIND='matched-panel-contended-recovery-lockfix-v1'
HOST='linux10603';CASE='clip-04-bank-1';ATTEMPT='attempt-02'
RP='5ca7596fe95ecb8f6b6c6752426e56bb354b1f019d3576c6618f7ede98ae4116'
MP='da5aeb93bd6d3e46040b158cc14f0d6feb1284a83b2b2778a3639b420db0acf9'
OP='b5501e3fbc4c76b2d66d49ddc063fe64dd70f22885ef3d506ce51c4bcf391a05'
STOP='776e80c041599c1c1a35ee31348312b4f696eec03f0e1e8787cc27dd231b11f5'
BASE='Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024'
PREFIX='contended-panel-recovery-lockfix-20261026'
PRIOR_PREFIX='contended-panel-recovery-20261026'
ORIGINAL_PROPOSAL='2d841e90f4378507b80e4805e1096e17ed2231c1257c9b57d405b802f32164f4'
ORIGINAL_AUTH='721bbf9e77804b7acc7bdf542c63e2d29cda3f56b148dec7b24fea810c0f9d2d'
ORIGINAL_GRANT='814965108dd544eba53ebc5e34496d27'
STARTUP_TRANSPORT='ceb8d98eb7e0fe09c1aece046c794702fe77628c7d8990496bd987d5882cf746'
STARTUP_MANIFEST='e375fb1e4b80dfc3cc9ab736c2a23dbe2f9d95a68245f5227c2e07bdfa9d25d9'
STARTUP_CHARGE=36
CASE_LIMIT=4800-STARTUP_CHARGE
SOURCES=('scripts/recover_contended_panel_case.py','scripts/prepare_contended_panel_recovery.py',
         'tests/test_contended_panel_recovery.py','tests/fixtures/contended_panel_metadata.json')
SUCCESS=('clip-03-bank-1','clip-04-bank-0','clip-05-bank-0')
ABSENT=('matrix/observation.json','replay','receipt.json','revision-evidence.tar.gz','revision-evidence.tar.gz.manifest.json')
PRIOR_LAUNCH=('started-utc.txt','ended-utc.txt','exit-status.txt','case.log','cpu.log','cpu-results.xml','cpu-summary.log',
             'cpu-old.log','cpu-new.log','cpu-old.xml','cpu-new.xml','source-launch.sh','source-pipeline.sh')
PRIOR_FILES=tuple(sorted(
    ['packet/'+n for n in (*SOURCES,'RECOVERY.json','authorization.json','biking-independent-review.json','remote-launch.sh','remote-pipeline.sh')]
    +['launcher/'+n for n in PRIOR_LAUNCH]
    +['ledger/'+n for n in ('STOP.json','genesis.json','imports.json','failed-claim.json','failed-outcome.json','failed-session.json','failed-envelope.json')]
    +['reference/two_host_panel.py','reference/IMPLEMENTATION.json']))


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def regular_bound(root,name):
    root=Path(root).resolve();relative=Path(name);path=root/relative
    if (relative.is_absolute() or '..' in relative.parts or str(relative)!=name or '\\' in name
        or not name or name=='.' or not path.resolve().is_relative_to(root)
        or any(p.is_symlink() for p in [path,*list(path.parents)[:len(relative.parts)-1]])):
        raise ValueError('Escaped or symlinked recovery input: '+name)
    return path


def budget_assessment(numeric_seconds,envelope_seconds,launcher_seconds):
    values=(numeric_seconds,envelope_seconds,launcher_seconds)
    if any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in values):
        raise ValueError('Unknown old runtime cannot authorize a deadline reset.')
    charge=min(4800,math.ceil(max(numeric_seconds+180,envelope_seconds,launcher_seconds)))
    return {'old_charge_seconds':charge,'old_remaining_seconds':4800-charge,
            'old_remaining_parent_seconds':max(0,4800-charge-180)}


def validate_spec(spec,pin):
    expected={'kind':KIND,'case_id':CASE,'host':HOST,'attempt':ATTEMPT,
        'reference_revision_sha256':RP,'reference_implementation_sha256':MP,'reference_operation_sha256':OP,
        'failed_STOP_sha256':STOP,'extra_case_seconds':4800,'case_seconds':CASE_LIMIT,'kill_grace_seconds':15,
        'parent_seconds':4620,'matrix_seconds':3600,'replay_seconds':900,'additional_attempts':1,
        'reuse_old_tensors':False,'old_remainder_added':False,'automatic_GPU_retries':0,'automatic_queue':False,
        'strict_GPU_guard_unchanged':True,'preserve_global_STOP':True,'one_case_only':True,
        'paid_compute':False,'research_admitted':False,'numeric_settings_changed':False,
        'original_proposal_sha256':ORIGINAL_PROPOSAL,'original_authorization_file_sha256':ORIGINAL_AUTH,
        'original_grant_id':ORIGINAL_GRANT,'prior_startup_charge_seconds':STARTUP_CHARGE,
        'prior_startup_transport_sha256':STARTUP_TRANSPORT,'prior_startup_manifest_sha256':STARTUP_MANIFEST}
    if canonical(spec)!=pin or any(k not in spec or canonical(spec[k])!=canonical(v) for k,v in expected.items()):
        raise ValueError('Recovery scope, source identity or resource bound changed.')
    if set(spec['sources_sha256'])!=set(SOURCES) or any(re.fullmatch('[a-f0-9]{64}',v) is None for v in spec['sources_sha256'].values()):
        raise ValueError('Incomplete recovery source inventory.')
    if spec['old_budget']!=budget_assessment(2065.4081135171,2086.5223807520233,2113):
        raise ValueError('Old consumed time must remain charged separately.')
    if spec['preserved_coordinator_files_sha256'].get('STOP.json')!=STOP:
        raise ValueError('The reviewed failure stop must remain pinned.')
    if set(spec['prior_startup_files_sha256'])!=set(PRIOR_FILES):raise ValueError('Incomplete failed-startup evidence inventory.')
    startup_pins={n.removeprefix('packet/'):p for n,p in spec['prior_startup_files_sha256'].items() if n.startswith('packet/')}
    startup_pins.update({n:p for n,p in spec['prior_startup_files_sha256'].items() if n.startswith('launcher/')})
    if spec['preserved_failed_startup_files_sha256']!=startup_pins:raise ValueError('The original failed startup must stay intact.')


def validate_authorization(spec,authorization):
    expected={'kind':'matched-panel-contended-recovery-v1-authorization','approved':True,'proposal_sha256':ORIGINAL_PROPOSAL,
        'case_id':CASE,'host':HOST,'attempt':ATTEMPT,'extra_case_seconds':4800,
        'additional_attempts':1,'fresh_matrix_and_replay':True,'paid_compute':False,'research_admitted':False}
    if any(k not in authorization or canonical(authorization[k])!=canonical(v) for k,v in expected.items()):
        raise ValueError('An explicit matching one-attempt authorization is required.')
    grant=authorization.get('grant_id','')
    if grant!=ORIGINAL_GRANT:
        raise ValueError('Preserve the original unused grant; no new allowance is authorized.')
    return grant


def validate_startup_manifest(manifest,spec):
    expected={'kind':'read-only-recovery-lock-startup-v1','host':HOST,'case_id':CASE,
        'proposal_sha256':ORIGINAL_PROPOSAL,'authorization_file_sha256':ORIGINAL_AUTH,
        'coordinator_pins_unchanged':True,'university_files_changed':False,
        'model_or_test_job_started':False,'scientific_case_accepted':False}
    if any(k not in manifest or canonical(manifest[k])!=canonical(v) for k,v in expected.items()):
        raise ValueError('Failed-startup observation does not permit this continuation.')
    if set(manifest['files'])!=set(PRIOR_FILES) or {n:v['sha256'] for n,v in manifest['files'].items()}!=spec['prior_startup_files_sha256']:
        raise ValueError('Failed-startup file pins differ.')
    expected_locations={'recovery_grant':{'relative_path':'recovery-grants/'+CASE,'exists':False,'is_symlink':False},
                        'attempt_02':{'relative_path':'cases/'+CASE+'/'+ATTEMPT,'exists':False,'is_symlink':False}}
    if manifest['recovery_locations']!=expected_locations:raise ValueError('The original grant or attempt already existed at capture.')
    rows=manifest['coordinator_file_hashes'];pins=spec['preserved_coordinator_files_sha256']
    if set(rows)!=set(pins) or any(v['observed_sha256']!=v['expected_sha256'] or v['observed_sha256']!=pins[n] for n,v in rows.items()):
        raise ValueError('Prior startup changed preserved coordinator evidence.')


def verify_prior_startup(directory,spec,authorization):
    """Read only authentic captured files; no old process or model is restarted."""
    directory=Path(directory)
    if digest(regular_bound(directory,'transport-manifest.json'))!=STARTUP_MANIFEST:raise ValueError('Failed-startup manifest changed.')
    manifest=json.loads((directory/'transport-manifest.json').read_text());validate_startup_manifest(manifest,spec)
    for name,row in manifest['files'].items():
        path=regular_bound(directory,name)
        if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:raise ValueError('Failed-startup file changed: '+name)
    previous=json.loads((directory/'packet/RECOVERY.json').read_text())
    if (canonical(previous)!=ORIGINAL_PROPOSAL or digest(directory/'packet/authorization.json')!=ORIGINAL_AUTH
        or json.loads((directory/'packet/authorization.json').read_text())!=authorization
        or previous['preserved_coordinator_files_sha256']!=spec['preserved_coordinator_files_sha256']
        or previous['failed_attempt_files_sha256']!=spec['failed_attempt_files_sha256']
        or previous['old_budget']!=spec['old_budget']):raise ValueError('Continuation must preserve the original authorization and scope.')
    for name,pin in previous['sources_sha256'].items():
        if digest(directory/'packet'/name)!=pin:raise ValueError('Captured original source changed.')
    for name in ('launch','pipeline'):
        if (directory/'launcher'/('source-'+name+'.sh')).read_bytes()!=(directory/'packet'/('remote-'+name+'.sh')).read_bytes():
            raise ValueError('Captured failed launcher differs from its original source.')
    import xml.etree.ElementTree as E
    for name,count in [('old',76),('new',63),('results',139)]:
        filename='cpu-results.xml' if name=='results' else 'cpu-'+name+'.xml'
        xml=E.parse(directory/'launcher'/filename).getroot()
        counts={k:sum(int(s.get(k,0)) for s in xml.iter('testsuite')) for k in ('tests','failures','errors','skipped')}
        if (counts!={'tests':count,'failures':0,'errors':0,'skipped':0} or len(list(xml.iter('testcase')))!=count
            or any(s.get('hostname')!=HOST for s in xml.iter('testsuite'))):raise ValueError('Original startup CPU results are incomplete.')
    start=datetime.fromisoformat((directory/'launcher/started-utc.txt').read_text().strip().replace('Z','+00:00'))
    end=datetime.fromisoformat((directory/'launcher/ended-utc.txt').read_text().strip().replace('Z','+00:00'))
    if ((end-start).total_seconds()!=35 or (directory/'launcher/exit-status.txt').read_text().strip()!='1'
        or STARTUP_CHARGE!=math.ceil((end-start).total_seconds())+1):raise ValueError('Do not reset or undercharge the failed startup clock.')
    log=(directory/'launcher/case.log').read_text()
    if 'BlockingIOError' not in log or 'recovery starting matrix worker' in log:raise ValueError('Unexpected startup failure boundary.')
    return manifest


def load_context(packet,pin,authorization_pin,home):
    packet,home=Path(packet).resolve(),Path(home)
    if packet!=(home/BASE/PREFIX/'unpacked'/PREFIX).resolve():raise ValueError('Wrong fixed recovery packet path.')
    spec=json.loads((packet/'RECOVERY.json').read_text());validate_spec(spec,pin)
    if digest(packet/'authorization.json')!=authorization_pin:raise ValueError('External authorization file pin differs.')
    authorization=json.loads((packet/'authorization.json').read_text());grant=validate_authorization(spec,authorization)
    if authorization_pin!=ORIGINAL_AUTH:raise ValueError('The unchanged original authorization file is required.')
    for name,pin in spec['sources_sha256'].items():
        if digest(regular_bound(packet,name))!=pin:raise ValueError('Recovery source changed: '+name)
    if digest(regular_bound(packet,'biking-independent-review.json'))!=spec['peer_review_file_sha256']:
        raise ValueError('Independent peer review marker changed.')
    verify_prior_startup(packet/'prior-startup',spec,authorization)
    transfer=(home/BASE).resolve();original=transfer/'two-host-execution-20261026/unpacked/two-host-execution-20261026'
    revision=json.loads(regular_bound(original,'configs/matched_panel_two_host_v1.json').read_text())
    manifest=json.loads(regular_bound(original,'IMPLEMENTATION.json').read_text())
    if canonical(revision)!=RP or canonical(manifest)!=MP:raise ValueError('Frozen reference identity changed.')
    for name,pin in manifest['source_sha256'].items():
        if digest(regular_bound(original,name))!=pin:raise ValueError('Frozen reference source changed before import.')
    module=importlib.util.spec_from_file_location('frozen_contended_reference',original/'models/two_host_panel.py')
    ops=importlib.util.module_from_spec(module);module.loader.exec_module(ops)
    package=transfer/'unpacked/matched-panel-20261024';root=transfer/'two-host-panel-20261026'
    numeric,_,old=ops.check_sources(package,original,revision,manifest,RP,MP)
    ops.validate_locations(package,original,root,home)
    prior=transfer/'two-host-continuation-20261026/unpacked/two-host-continuation-20261026'
    operation=json.loads((prior/'OPERATION.json').read_text())
    if canonical(operation)!=OP:raise ValueError('Original operational identity changed.')
    for name,pin in operation['sources_sha256'].items():
        if digest(ops.bound(prior,name))!=pin:raise ValueError('Original operational source changed.')
    return spec,authorization,grant,ops,revision,manifest,numeric,old,package,original,root


def execution_identity(context):
    spec,authorization,grant,ops,revision,manifest,*_=context
    result=ops.execution_identity(revision,manifest,HOST)
    result.update(kind=KIND+'-numeric-execution',origin='explicit-attended-contention-recovery',
        reference_execution_sha256=canonical(result),recovery_proposal_sha256=canonical(spec),
        recovery_authorization_sha256=canonical(authorization),recovery_grant_id=grant,
        recovery_sources_sha256=spec['sources_sha256'],recovery_attempt=ATTEMPT)
    return result


def check_preserved(root,spec,ops):
    for name,pin in spec['preserved_coordinator_files_sha256'].items():
        path=ops.bound(root,name)
        if not path.is_file() or digest(path)!=pin:raise ValueError('Preserved coordinator artifact changed: '+name)
    old=ops.bound(root,'cases/'+CASE+'/attempt-01')
    files={str(p.relative_to(old)):p for p in old.rglob('*') if p.is_file() or p.is_symlink()}
    if set(files)!=set(spec['failed_attempt_files_sha256']):raise ValueError('Failed attempt inventory changed.')
    for name,pin in spec['failed_attempt_files_sha256'].items():
        if digest(ops.bound(old,name))!=pin:raise ValueError('Failed attempt bytes changed: '+name)
    if any((old/name).exists() or (old/name).is_symlink() for name in ABSENT):
        raise ValueError('Failed attempt gained evidence after review; inspect it before any retry.')
    prior=Path(root).parent/PRIOR_PREFIX/'unpacked'/PRIOR_PREFIX
    for name,pin in spec.get('preserved_failed_startup_files_sha256',{}).items():
        if digest(ops.bound(prior,name))!=pin:raise ValueError('The failed startup was altered: '+name)


def preflight(context,actual_host,*,return_registry=False):
    spec,_,_,ops,revision,manifest,numeric,old,package,_,root=context
    if actual_host!=HOST:raise ValueError('Recovery must stay on its assigned GPU host.')
    check_preserved(root,spec,ops)
    for relative in ('recovery-grants/'+CASE,'cases/'+CASE+'/'+ATTEMPT):
        path=ops.bound(root,relative)
        if path.exists() or path.is_symlink():raise ValueError('Recovery is already claimed; never repeat the GPU launch.')
    registry=ops.Registry(root,revision,MP,ops.verify_imports(package,old));status=registry.status()
    if (status['accepted_case_ids']!=ops.CASE_ORDER[:9] or status['computed_new_case_ids']!=list(SUCCESS)
        or status['failed_case_ids']!=[CASE] or not status['stop_dispatch'] or status['scientific_aggregate'] is not None):
        raise ValueError('Recovery is confined to the reviewed stopped checkpoint.')
    from models.matched_panel_case import verify_case
    from models.matched_panel_io import verify_archive
    for cid in SUCCESS:
        host=revision['case_hosts'][cid];directory=ops.bound(root,'cases/'+cid+'/attempt-01')
        receipt=verify_case(directory,plan=numeric,case_id=cid,execution=ops.execution_identity(revision,manifest,host))
        if ops.read_record(directory/'receipt.json')!=receipt:raise ValueError('Saved successful receipt changed.')
        outcome=ops.read_record(root/'outcomes'/(cid+'.json'));event=ops.read_record(root/'exports'/(cid+'.json'))
        if not outcome['complete'] or outcome['numeric_receipt_sha256']!=canonical(receipt):raise ValueError('Successful outcome changed.')
        archive=ops.bound(root,event['archive_path']);sidecar=ops.bound(root,event['archive_manifest_path'])
        verify_archive(archive,json.loads(sidecar.read_text()),event['archive_sha256'])
        marker=json.loads((root.parent/PREFIX/'unpacked'/PREFIX/'biking-independent-review.json').read_text()) if cid==SUCCESS[-1] else None
        if marker is not None:
            marker=marker['payload']
            if (marker['case_id']!=cid or marker['numeric_receipt_sha256']!=canonical(receipt)
                or marker['outcome_file_sha256']!=digest(root/'outcomes'/(cid+'.json')) or marker['archive_sha256']!=event['archive_sha256']):
                raise ValueError('Locally reviewed peer marker does not match the saved successful peer.')
    if shutil.disk_usage(root).free<4*2**30:raise OSError('Insufficient artifact/export space.')
    return (registry,status) if return_registry else status


def consume(root,spec,authorization,ops):
    directory=ops.bound(root,'recovery-grants/'+CASE);directory.mkdir(parents=True,exist_ok=False)
    ops.write_record(directory/'claim.json',{'kind':KIND+'-claim','case_id':CASE,'host':HOST,'attempt':ATTEMPT,
        'grant_id':authorization['grant_id'],'proposal_sha256':canonical(spec),'authorization_sha256':canonical(authorization),
        'case_seconds':CASE_LIMIT,'prior_startup_charge_seconds':STARTUP_CHARGE,
        'claimed_utc':datetime.now(timezone.utc).isoformat(),'research_admitted':False})
    return directory


def strict_gate(ops,root,stage,*,allow_self=False):
    """Retain observations around the unchanged guard, including a rejection."""
    event={'kind':KIND+'-GPU-check','stage':stage,'utc':datetime.now(timezone.utc).isoformat(),
        'strict_guard_source_unchanged':True,'snapshot_is_not_a_reservation':True,'research_admitted':False}
    try:
        identity=ops.hardware_gate(HOST,allow_self=allow_self);event['passed']=True
        return identity
    except Exception as error:
        event.update(passed=False,error=type(error).__name__+': '+str(error))
        for name,command in [('nvidia_smi',['nvidia-smi']),('compute_pids',['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])]:
            try:
                result=subprocess.run(command,capture_output=True,text=True,timeout=10)
                event[name]={'exit_status':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
                             'captured_after_rejection_not_exact_failure_PID_proof':True}
            except Exception as capture_error:event[name]={'capture_error':str(capture_error)}
        raise
    finally:ops.write_record(Path(root)/('hardware-'+stage+'.json'),event)


def live_worker_gate(context,role):
    spec,authorization,_,ops,*rest=context;root=rest[-1]
    if role not in ('matrix','replay') or platform.node()!=HOST:raise ValueError('Wrong recovery worker host or role.')
    grant_dir=ops.bound(root,'recovery-grants/'+CASE);claim=ops.read_record(grant_dir/'claim.json')
    directory=ops.bound(root,'cases/'+CASE+'/'+ATTEMPT);control=ops.read_record(directory/'case-envelope.json')
    if (claim['proposal_sha256']!=canonical(spec) or claim['authorization_sha256']!=canonical(authorization)
        or claim['grant_id']!=authorization['grant_id'] or control['proposal_sha256']!=canonical(spec)
        or control['complete'] is not False or control['active_role']!=role):raise ValueError('Worker lacks a live matching recovery grant.')
    check_preserved(root,spec,ops)
    ops.bound(grant_dir,'workers/'+role).mkdir(parents=True,exist_ok=False)
    strict_gate(ops,directory,role+'-before')
    return directory


def worker(context,role):
    directory=live_worker_gate(context,role)
    spec,_,_,ops,_,_,numeric,_,package,_,_=context
    import numpy as np
    import torch
    from models.evaluation import EvaluationMaskSampler
    from models.execution import apply_model_policy,configure_execution
    from models.matched_panel_case import KIND as NUMERIC_KIND,run_worker
    from models.matched_panel_io import load_clip
    from models.precision_diagnostic import native_probe
    from models.runtime import MemoryMonitor
    from models.training import runtime_contract
    from models.vjepa2 import build_vitl,load_checkpoint
    execution=execution_identity(context);expected=execution['expected_runtime']
    configure_execution('shared-fp64-native',device='cuda');torch.set_num_threads(expected['threads'])
    if (platform.python_version()!=expected['python'] or str(torch.__version__)!=expected['torch']
        or np.__version__!=expected['numpy'] or not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise ValueError('Use the exact existing frozen numerical runtime.')
    total=torch.cuda.get_device_properties(0).total_memory;limit=min(7*2**30,int(.9*total))
    torch.cuda.set_per_process_memory_fraction(min(.9,7*2**30/total))
    probe=native_probe('cuda');strict_gate(ops,directory,role+'-probe',allow_self=True)
    torch.manual_seed(20261024);clean=load_clip(package,numeric,CASE)
    model=build_vitl(package/'external/vjepa2',activation_checkpointing=True)
    checkpoint=Path.home()/'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting/checkpoints/vitl.pt'
    load_checkpoint(model,checkpoint,execution['checkpoint_sha256'])
    apply_model_policy(model,'shared-fp64-native');model.to('cuda')
    if canonical(runtime_contract('cuda',model=model))!=canonical(expected):raise ValueError('Actual model/precision policy changed.')
    with MemoryMonitor('cuda') as memory:
        run_worker(model,clean.to('cuda'),EvaluationMaskSampler(package/'external/vjepa2'),
            plan=numeric,case_id=CASE,execution=execution,output=directory/role,role=role)
    identity=strict_gate(ops,directory,role+'-after',allow_self=True)
    ops.write_record(directory/role/'observation.json',{'kind':NUMERIC_KIND,'case_id':CASE,'role':role,
        'execution_sha256':canonical(execution),'actual_host':platform.node(),'actual_GPU_identity':identity,
        'memory':memory.report(),'allocator_limit_bytes':limit,'native_probe':probe,
        'research_admitted':False,'A40_cost_pool_eligible':False,'origin':KIND,'recovery_proposal_sha256':canonical(spec)})


def child_command(packet,pin,authorization_pin,role):
    if role not in ('matrix','replay'):raise ValueError('Unknown recovery worker role.')
    return [sys.executable,'-B','-u',str(Path(packet)/'scripts/recover_contended_panel_case.py'),'worker',
            '--proposal-sha256',pin,'--authorization-sha256',authorization_pin,'--role',role]


def worker_limit(role,budget,elapsed):
    if role not in ('matrix','replay') or not 0<budget<=4620 or not math.isfinite(elapsed) or elapsed<0:
        raise ValueError('Invalid recovery worker budget.')
    remaining=budget-elapsed-30
    if remaining<1:raise TimeoutError('No time remains for a required fresh worker.')
    return min(3600 if role=='matrix' else 900,remaining)


def run_session(context,packet,pin,authorization_pin,directory,budget):
    spec,_,_,ops,_,_,numeric,_,package,_,root=context
    from models.matched_panel_case import KIND as NUMERIC_KIND,verify_pair,verify_case
    from models.matched_panel_io import export_case,verify_archive
    execution=execution_identity(context);started=time.monotonic()
    session={'kind':NUMERIC_KIND,'case_id':CASE,'plan_sha256':ops.NUMERIC_PLAN_SHA,'execution_sha256':canonical(execution),
        'complete':False,'passed':False,'research_admitted':False,'tasks':{},'elapsed_seconds':0.,
        'started_utc':datetime.now(timezone.utc).isoformat()}
    ops.write_record(directory/'session.json',session)
    control=ops.read_record(directory/'case-envelope.json');prior=signal.getsignal(signal.SIGTERM)
    def stop(*_):raise TimeoutError('Recovery terminated; preserve partial evidence and consumed grant.')
    signal.signal(signal.SIGTERM,stop)
    try:
        for role in ('matrix','replay'):
            seconds=worker_limit(role,budget,time.monotonic()-started)
            control['active_role']=role;ops.write_record(directory/'case-envelope.json',control,replace=True)
            print(HOST,CASE,'recovery starting',role,'worker',flush=True)
            session['tasks'][role]=ops.logged_child(child_command(packet,pin,authorization_pin,role),directory/(role+'.log'),seconds)
            session['elapsed_seconds']=time.monotonic()-started;ops.write_record(directory/'session.json',session,replace=True)
            if not session['tasks'][role]['complete']:raise RuntimeError('Recovery worker failed/timed out; no retry or next case.')
        verify_pair(directory,plan=numeric,case_id=CASE,execution=execution)
        session.update(complete=True,passed=True,elapsed_seconds=time.monotonic()-started)
        ops.write_record(directory/'session.json',session,replace=True)
        receipt=verify_case(directory,plan=numeric,case_id=CASE,execution=execution);ops.write_record(directory/'receipt.json',receipt)
    except BaseException as error:
        session.update(complete=False,passed=False,error=type(error).__name__+': '+str(error),elapsed_seconds=time.monotonic()-started)
        ops.write_record(directory/'session.json',session,replace=True)
        raise
    finally:
        control['active_role']=None;ops.write_record(directory/'case-envelope.json',control,replace=True)
        try:
            archive=export_case(package,directory,numeric,execution,destination=directory/'numeric-evidence.tar.gz')
            verify_archive(directory/'numeric-evidence.tar.gz',archive,archive['sha256'])
        finally:signal.signal(signal.SIGTERM,prior)
    return receipt,archive


def export_recovery(context,packet,directory,grant_dir):
    spec,_,_,ops,_,_,_,_,_,original,root=context
    from models.pilot_bundle import make_archive
    from models.matched_panel_io import verify_archive
    files={'RECOVERY.json':Path(packet)/'RECOVERY.json','authorization.json':Path(packet)/'authorization.json',
        'biking-independent-review.json':Path(packet)/'biking-independent-review.json',
        'recovery-claim.json':grant_dir/'claim.json','recovery-outcome.json':grant_dir/'outcome.json',
        'case-envelope.json':directory/'case-envelope.json','failed-STOP.json':root/'STOP.json'}
    for name in SOURCES:files['recovery-source/'+name]=Path(packet)/name
    for name in (*PRIOR_FILES,'transport-manifest.json'):files['prior-startup/'+name]=Path(packet)/'prior-startup'/name
    for name in ops.SOURCES:files['reference-source/'+name]=original/name
    files['reference-implementation.json']=original/'IMPLEMENTATION.json'
    for name in ('numeric-evidence.tar.gz','numeric-evidence.tar.gz.manifest.json'):
        if (directory/name).is_file():files['numeric/'+name]=directory/name
    for path in sorted(directory.glob('hardware-*.json')):files['hardware/'+path.name]=path
    if any(p.is_symlink() for p in files.values()) or sum(p.stat().st_size for p in files.values())>256*2**20:
        raise ValueError('Recovery export exceeds the fixed regular-file bound.')
    path=directory/'recovery-evidence.tar.gz';manifest=make_archive(path,files)
    Path(str(path)+'.manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    verify_archive(path,manifest,manifest['sha256'])
    print('Recovery evidence:',json.dumps({'archive':str(path),**{k:manifest[k] for k in ('bytes','sha256')}},sort_keys=True),flush=True)
    return manifest


def review_case(packet,reference,package,pin,authorization_pin,archive,sidecar,archive_pin,output):
    """Verify a saved recovery from trusted local sources; never run a model."""
    packet,reference,package,output=map(Path,(packet,reference,package,output))
    spec=json.loads((packet/'RECOVERY.json').read_text());validate_spec(spec,pin)
    if digest(packet/'authorization.json')!=authorization_pin:raise ValueError('Trusted authorization changed.')
    authorization=json.loads((packet/'authorization.json').read_text());grant=validate_authorization(spec,authorization)
    if authorization_pin!=ORIGINAL_AUTH:raise ValueError('Original authorization file required for review.')
    verify_prior_startup(packet/'prior-startup',spec,authorization)
    for name,p in spec['sources_sha256'].items():
        if digest(regular_bound(packet,name))!=p:raise ValueError('Trusted recovery verifier source changed.')
    manifest=json.loads((reference/'IMPLEMENTATION.json').read_text())
    revision=json.loads((reference/'configs/matched_panel_two_host_v1.json').read_text())
    if canonical(manifest)!=MP or canonical(revision)!=RP:raise ValueError('Trusted reference identity changed.')
    for name,p in manifest['source_sha256'].items():
        if digest(regular_bound(reference,name))!=p:raise ValueError('Trusted reference verifier source changed.')
    m=importlib.util.spec_from_file_location('frozen_contended_review',reference/'models/two_host_panel.py')
    ops=importlib.util.module_from_spec(m);m.loader.exec_module(ops)
    numeric,_,_=ops.check_sources(package,reference,revision,manifest,RP,MP)
    from models.matched_panel_io import verify_archive,review_export
    from models.pilot_bundle import extract_verified
    verify_archive(archive,sidecar,archive_pin)
    fixed={'RECOVERY.json','authorization.json','biking-independent-review.json','recovery-claim.json',
           'recovery-outcome.json','case-envelope.json','failed-STOP.json','reference-implementation.json',
           'numeric/numeric-evidence.tar.gz','numeric/numeric-evidence.tar.gz.manifest.json'}
    fixed|={'recovery-source/'+n for n in SOURCES}|{'reference-source/'+n for n in ops.SOURCES}
    fixed|={'prior-startup/'+n for n in (*PRIOR_FILES,'transport-manifest.json')}
    stages=[role+'-'+stage for role in ('matrix','replay') for stage in ('before','probe','after')]
    fixed|={'hardware/hardware-'+stage+'.json' for stage in stages}
    if set(sidecar['files'])!=fixed:raise ValueError('Missing or unexpected recovery evidence inventory.')
    extract_verified(archive,sidecar,output)
    if (json.loads((output/'RECOVERY.json').read_text())!=spec or digest(output/'authorization.json')!=authorization_pin
        or digest(output/'failed-STOP.json')!=STOP
        or digest(output/'biking-independent-review.json')!=spec['peer_review_file_sha256']
        or canonical(json.loads((output/'reference-implementation.json').read_text()))!=MP):
        raise ValueError('Exported recovery boundary differs from its independent authorization.')
    for prefix,hashes in [('recovery-source',spec['sources_sha256']),('reference-source',manifest['source_sha256'])]:
        for name,p in hashes.items():
            if digest(output/prefix/name)!=p:raise ValueError('Captured operator source changed.')
    verify_prior_startup(output/'prior-startup',spec,authorization)
    claim=ops.read_record(output/'recovery-claim.json');outcome=ops.read_record(output/'recovery-outcome.json')
    envelope=ops.read_record(output/'case-envelope.json')
    for body,suffix in ((claim,'claim'),(outcome,'outcome'),(envelope,'envelope')):
        if (body['kind']!=KIND+'-'+suffix or body['case_id']!=CASE or body['host']!=HOST or body['attempt']!=ATTEMPT
            or body['grant_id']!=grant or body['proposal_sha256']!=pin or body['research_admitted'] is not False):
            raise ValueError('Mixed recovery ownership or grant.')
    if (claim['authorization_sha256']!=canonical(authorization) or claim['case_seconds']!=CASE_LIMIT
        or claim['prior_startup_charge_seconds']!=STARTUP_CHARGE
        or envelope['authorization_sha256']!=canonical(authorization) or envelope['complete'] is not True
        or envelope['active_role'] is not None or envelope['case_seconds']!=CASE_LIMIT or outcome['complete'] is not True
        or outcome['independently_reviewed'] is not False
        or any(type(x['elapsed_seconds']) not in (int,float) or not 0<x['elapsed_seconds']<=CASE_LIMIT for x in (envelope,outcome))):
        raise ValueError('Incomplete, unbounded or mixed recovery envelope.')
    for stage in stages:
        event=ops.read_record(output/'hardware'/('hardware-'+stage+'.json'))
        if (event['kind']!=KIND+'-GPU-check' or event['stage']!=stage or event['passed'] is not True
            or event['strict_guard_source_unchanged'] is not True or event['research_admitted'] is not False):
            raise ValueError('A required strict GPU guard was rejected or missing.')
    context=(spec,authorization,grant,ops,revision,manifest);execution=execution_identity(context)
    inner=output/'numeric/numeric-evidence.tar.gz';inner_spec=json.loads(Path(str(inner)+'.manifest.json').read_text())
    receipt=review_export(inner,inner_spec,outcome['numeric_archive_sha256'],output/'numeric-reviewed',plan=numeric,execution=execution)
    if (outcome['execution_sha256']!=canonical(execution) or outcome['numeric_receipt_sha256']!=canonical(receipt)):
        raise ValueError('Recovery outcome differs from saved numerical evidence.')
    marker={'kind':KIND+'-independent-review','case_id':CASE,'host':HOST,'attempt':ATTEMPT,'grant_id':grant,
        'proposal_sha256':pin,'authorization_file_sha256':authorization_pin,'archive_sha256':archive_pin,
        'numeric_receipt_sha256':canonical(receipt),'execution_sha256':canonical(execution),'reviewed_export':True,
        'original_failure_preserved':True,'global_STOP_not_cleared':True,'remote_ledger_updated':False,'research_admitted':False}
    ops.write_record(output/'independent-review.json',marker)
    return marker


def run(context,packet,pin,authorization_pin,launcher_started):
    spec,authorization,grant,ops,_,_,_,_,_,_,root=context
    if type(launcher_started) not in (float,int) or not math.isfinite(launcher_started) or not 0<=time.monotonic()-launcher_started<CASE_LIMIT:
        raise ValueError('A current monotonic launcher origin is required; no silent deadline reset.')
    from models.matched_panel_runner import case_deadline
    with case_deadline(CASE_LIMIT-(time.monotonic()-launcher_started),kill_grace_seconds=15),ExitStack() as stack:
        stack.enter_context(ops.file_lock(root/'leases'/HOST,wait_seconds=0))
        # Registry.__init__ acquires .registry.lock itself. Construct/read it
        # before the claim transaction; its status() method does not lock.
        registry,prior_status=preflight(context,platform.node(),return_registry=True)
        with ops.file_lock(root/'.registry.lock'):
            check_preserved(root,spec,ops)
            if canonical(registry.status())!=canonical(prior_status):
                raise ValueError('Coordinator changed during recovery preflight; inspect it before dispatch.')
            # The same unchanged PID/UUID/driver guard runs before consuming a grant.
            ops.hardware_gate(HOST)
            grant_dir=consume(root,spec,authorization,ops)
            directory=ops.bound(root,'cases/'+CASE+'/'+ATTEMPT);directory.mkdir(parents=True,exist_ok=False)
            control={'kind':KIND+'-envelope','case_id':CASE,'host':HOST,'attempt':ATTEMPT,
                'proposal_sha256':pin,'authorization_sha256':canonical(authorization),'grant_id':grant,
                'case_seconds':CASE_LIMIT,'prior_startup_charge_seconds':STARTUP_CHARGE,
                'complete':False,'active_role':None,'elapsed_seconds':0.,'research_admitted':False}
            ops.write_record(directory/'case-envelope.json',control)
        try:
            budget=min(4620,CASE_LIMIT-(time.monotonic()-launcher_started)-120)
            if budget<1:raise TimeoutError('Preflight consumed the additional case allowance.')
            receipt,archive=run_session(context,packet,pin,authorization_pin,directory,budget)
            check_preserved(root,spec,ops)
            control.update(complete=True,elapsed_seconds=time.monotonic()-launcher_started)
            outcome={'kind':KIND+'-outcome','case_id':CASE,'host':HOST,'attempt':ATTEMPT,
                'complete':True,'numeric_receipt_sha256':canonical(receipt),'numeric_archive_sha256':archive['sha256'],
                'execution_sha256':receipt['execution_sha256'],'elapsed_seconds':control['elapsed_seconds'],
                'grant_id':grant,'proposal_sha256':pin,'independently_reviewed':False,'research_admitted':False}
        except BaseException as error:
            control.update(complete=False,elapsed_seconds=time.monotonic()-launcher_started,error=type(error).__name__+': '+str(error))
            outcome={'kind':KIND+'-outcome','case_id':CASE,'host':HOST,'attempt':ATTEMPT,'complete':False,
                'grant_id':grant,'proposal_sha256':pin,'elapsed_seconds':control['elapsed_seconds'],
                'error':control['error'],'independently_reviewed':False,'research_admitted':False}
            raise
        finally:
            control['active_role']=None;ops.write_record(directory/'case-envelope.json',control,replace=True)
            ops.write_record(grant_dir/'outcome.json',outcome)
            check_preserved(root,spec,ops)
            export_recovery(context,packet,directory,grant_dir)
        print('One recovery saved; global STOP and old failure retained. Independent review comes next; no queue starts.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('run','worker','preflight'))
    parser.add_argument('--proposal-sha256',required=True);parser.add_argument('--authorization-sha256',required=True)
    parser.add_argument('--role',choices=('matrix','replay'));parser.add_argument('--launcher-started',type=float)
    args=parser.parse_args();packet=Path(__file__).resolve().parents[1]
    context=load_context(packet,args.proposal_sha256,args.authorization_sha256,Path.home())
    if args.mode=='worker':worker(context,args.role)
    elif args.mode=='run':run(context,packet,args.proposal_sha256,args.authorization_sha256,args.launcher_started)
    else:
        result=preflight(context,platform.node());context[3].hardware_gate(HOST)
        print(json.dumps({'kind':KIND+'-preflight','verified':True,'status':result,'models_started':False,'grant_consumed':False},sort_keys=True))


if __name__=='__main__':main()
