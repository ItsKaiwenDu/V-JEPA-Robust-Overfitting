"""Explicit two-host execution revision with an append-only shared case ledger.

Scientific workers and the original panel are immutable. Claims are exclusive,
outcomes never confer independent acceptance, and every run stops after one case.
"""
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

KIND='matched-panel-two-host-revision-v1'
MANIFEST_KIND='matched-panel-two-host-implementation-v1'
REQUEST_KIND='matched-panel-two-host-single-case-request-v1'
REVIEW_KIND='matched-panel-two-host-case-review-v1'
NUMERIC_PLAN_SHA='f6b81dec841aeee703413dba23ea16afe3dbf14736edecadc0f7830f3ded34ae'
HANDOFF_SHA='73f361fc072a053439bf71f35835bb91227932edd2db739adde2db5f96c6b0f4'
BASE_EXECUTION_SHA='1e59a735e2f0b6f5485cf0867851c638dcd027715d19270422f390f7745aa70e'
IMPORT_JOURNAL_SHA='cf851847721b40a24b45563ee4b06f589024269c94d4cbf73ce96af79a3e6814'
IMPORT_REVIEW_SHA='b08361f743f71b2066cb3ac63f4179891c4b1f3b06f293e5199760f346a8acff'
BRIDGE_REVIEW_SHA='2a7a52034d30a9740c9ebd5f190bd214ed124307307afabb92c5f918cd12d846'
HOSTS={
 'linux10609':'NVIDIA GeForce RTX 3070, GPU-4e2a7c77-19bd-689f-6c08-60eb774c0d52, 8192, 580.178.04',
 'linux10603':'NVIDIA GeForce RTX 3070, GPU-e54fae90-feeb-8106-d8c0-1974fe48555f, 8192, 580.178.04'}
CASE_ORDER=[f'clip-{i:02d}-bank-{b}' for i in range(10) for b in (0,1)]
ASSIGNMENTS={cid:('linux10609' if i<4 or i%2 else 'linux10603') for i in range(10) for cid in (f'clip-{i:02d}-bank-0',f'clip-{i:02d}-bank-1')}
FIRST_BATCH={'linux10609':'clip-03-bank-1','linux10603':'clip-04-bank-0'}
PACKAGE='Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024/unpacked/matched-panel-20261024'
COORDINATOR='Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024/two-host-panel-20261026'
PREFIX='two-host-execution-20261026'
PACKET='Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024/'+PREFIX+'/unpacked/'+PREFIX
SOURCES=('models/two_host_panel.py','scripts/check_two_host_panel.py','tests/test_two_host_panel.py',
         'tests/fixtures/two_host_bootstrap_metadata.json','configs/matched_panel_two_host_v1.json',
         'scripts/prepare_two_host_panel.py','scripts/install_two_host_packet.py')
LIMIT=4800


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read_record(path):
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict) or set(value)!={'payload','sha256'} or canonical(value['payload'])!=value['sha256']:
        raise ValueError('Record checksum mismatch: '+str(path))
    return value['payload']


def write_record(path,payload,*,replace=False):
    """Fsync then publish atomically; an exclusive link never overwrites an event."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump({'payload':payload,'sha256':canonical(payload)},stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        if replace:os.replace(temporary,path)
        else:os.link(temporary,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temporary.unlink(missing_ok=True)


def bound(root,name):
    root=Path(root).resolve();relative=PurePosixPath(name);path=root/name
    if (not name or relative.is_absolute() or '..' in relative.parts or '\\' in name or str(relative)!=name
        or name=='.' or not path.resolve().is_relative_to(root)
        or any(p.is_symlink() for p in [path,*list(path.parents)[:len(relative.parts)-1]])):
        raise ValueError('Unsafe revision artifact path: '+name)
    return path


@contextmanager
def file_lock(path,*,wait_seconds=10):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        deadline=time.monotonic()+wait_seconds
        while True:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>=deadline:raise
                time.sleep(.025)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def validate_revision(revision,pin):
    if not isinstance(pin,str) or re.fullmatch('[a-f0-9]{64}',pin) is None or canonical(revision)!=pin:
        raise ValueError('Execution revision differs from its external pin.')
    policy={'kind':KIND,'numeric_plan_sha256':NUMERIC_PLAN_SHA,'reference_handoff_sha256':HANDOFF_SHA,
        'max_parallel_cases':2,'max_cases_per_host':1,'case_seconds':4800,'kill_grace_seconds':15,
        'matrix_seconds':3600,'replay_seconds':900,'parent_seconds':4620,'max_attempts_per_case':1,
        'automatic_GPU_retries':0,'one_case_per_invocation':True,'automatic_queue':False,
        'hosts':HOSTS,'case_order':CASE_ORDER,'case_hosts':ASSIGNMENTS,'first_batch':FIRST_BATCH,
        'imported_case_ids':CASE_ORDER[:7],'imported_journal_sha256':IMPORT_JOURNAL_SHA,
        'imported_review_sha256':IMPORT_REVIEW_SHA,'bridge_review_sha256':BRIDGE_REVIEW_SHA,
        'numeric_settings_changed':False,'research_admitted':False,'paid_compute':False}
    if any(k not in revision or canonical(revision[k])!=canonical(v) for k,v in policy.items()):
        raise ValueError('Use the explicit two-host revision with original case limits and fixed assignments.')
    return revision


def host_queue(revision,host):
    if host not in HOSTS:raise ValueError('Unknown host.')
    return [cid for cid in CASE_ORDER[7:] if revision['case_hosts'][cid]==host]


def dispatch_gate(revision,host,cid,accepted,claimed,*,stopped=False):
    if stopped:raise ValueError('A prior failure stopped further dispatch; preserve evidence for review.')
    if host not in HOSTS or cid not in CASE_ORDER[7:] or revision['case_hosts'].get(cid)!=host:
        raise ValueError('Imported case, unknown case or wrong assigned GPU.')
    queue=host_queue(revision,host);pending=[c for c in queue if c not in accepted]
    if not pending or pending[0]!=cid:raise ValueError('Only the next independently unreviewed host case may run.')
    if cid in claimed:raise ValueError('Case already claimed; never reset an unknown or exhausted allowance.')
    if cid!=FIRST_BATCH[host] and not set(FIRST_BATCH.values())<=set(accepted):
        raise ValueError('Independently review both first-batch exports before further dispatch.')
    return cid


def validate_request(request,pin,revision,manifest_sha):
    if canonical(request)!=pin:raise ValueError('Case request differs from the external pin.')
    policy={'kind':REQUEST_KIND,'revision_sha256':canonical(revision),'implementation_sha256':manifest_sha,
        'case_seconds':4800,'kill_grace_seconds':15,'max_attempts':1,'one_case_only':True,
        'additional_allowance_seconds':0,'automatic_queue':False,'paid_compute':False,'research_admitted':False}
    if any(k not in request or canonical(request[k])!=canonical(v) for k,v in policy.items()):
        raise ValueError('Use one fresh assigned case within its original allowance.')
    host,cid=request.get('host'),request.get('case_id')
    if host not in HOSTS or cid not in CASE_ORDER[7:] or ASSIGNMENTS[cid]!=host:
        raise ValueError('Case request does not match the fixed host assignment.')
    return host,cid


def validate_hardware(host,identity,active,*,allowed_pid=None):
    if host not in HOSTS or identity.strip()!=HOSTS[host]:raise ValueError('Host/GPU UUID/driver changed.')
    allowed=set() if allowed_pid is None else {allowed_pid}
    if any(int(p.strip()) not in allowed for p in active.splitlines() if p.strip()):
        raise ValueError('Another CUDA compute process is active on this host.')


def hardware_gate(host,*,allow_self=False):
    identity=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    if platform.node()!=host:raise ValueError('SSH session is on the wrong assigned workstation.')
    validate_hardware(host,identity,active,allowed_pid=os.getpid() if allow_self else None)
    return identity


def validate_locations(package,packet,root,home):
    if any(Path(p).resolve()!=(Path(home)/name).resolve() for p,name in [(package,PACKAGE),(packet,PACKET),(root,COORDINATOR)]):
        raise ValueError('Use the fixed shared package, revision packet and coordinator; no copied-root resets.')


class Registry:
    """Independent immutable per-case events avoid lost concurrent ledger updates."""
    def __init__(self,root,revision,implementation_sha,imports):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.revision=revision;self.implementation_sha=implementation_sha
        self.header={'kind':KIND+'-ledger','revision_sha256':canonical(revision),
            'implementation_sha256':implementation_sha,'imports_sha256':canonical(imports),'research_admitted':False}
        if set(imports)!=set(CASE_ORDER[:7]):raise ValueError('Exactly the seven reviewed v1 imports are required.')
        with file_lock(self.root/'.registry.lock'):
            if (self.root/'genesis.json').exists():
                if read_record(self.root/'genesis.json')!=self.header or read_record(self.root/'imports.json')!=imports:
                    raise ValueError('Existing coordinator belongs to another revision or implementation.')
            else:
                if any((self.root/n).exists() for n in ('claims','outcomes','reviews')):
                    raise ValueError('Unjournaled events exist; never initialize over them.')
                if (self.root/'imports.json').exists():
                    if read_record(self.root/'imports.json')!=imports:raise ValueError('Interrupted import identity differs.')
                else:write_record(self.root/'imports.json',imports)
                write_record(self.root/'genesis.json',self.header)

    @contextmanager
    def host_lease(self,host):
        if host not in HOSTS:raise ValueError('Unknown GPU host.')
        with file_lock(self.root/'leases'/host,wait_seconds=0):yield

    def status(self):
        if read_record(self.root/'genesis.json')!=self.header:raise ValueError('Ledger identity changed.')
        imports=read_record(self.root/'imports.json');accepted=set(imports);computed=set();claimed=set();failed=set()
        if canonical(imports)!=self.header['imports_sha256']:raise ValueError('Imported artifacts changed.')
        for cid in CASE_ORDER[7:]:
            if (self.root/'claims'/cid).exists():claimed.add(cid)
            outcome=self.root/'outcomes'/(cid+'.json')
            if outcome.exists():
                body=read_record(outcome)
                if (body['kind']!=KIND+'-outcome' or body['case_id']!=cid or body['host']!=ASSIGNMENTS[cid]
                    or body['revision_sha256']!=canonical(self.revision) or type(body['complete']) is not bool
                    or body['research_admitted'] is not False):raise ValueError('Mixed case outcome.')
                (computed if body['complete'] else failed).add(cid)
            review=self.root/'reviews'/(cid+'.json')
            if review.exists():
                marker=read_record(review);export=read_record(self.root/'exports'/(cid+'.json'))
                if (cid not in computed or marker['kind']!=REVIEW_KIND or marker['reviewed_export'] is not True
                    or marker['case_id']!=cid or marker['host']!=ASSIGNMENTS[cid]
                    or marker['implementation_sha256']!=self.implementation_sha
                    or marker['numeric_receipt_sha256']!=body['numeric_receipt_sha256']
                    or marker['execution_sha256']!=body['execution_sha256']
                    or marker['revision_sha256']!=canonical(self.revision) or marker['archive_sha256']!=export['archive_sha256']
                    or marker['outcome_file_sha256']!=digest(outcome) or marker['research_admitted'] is not False):
                    raise ValueError('Independent case review does not match retained outcome/export.')
                accepted.add(cid)
        return {'kind':KIND+'-progress','accepted_case_ids':[c for c in CASE_ORDER if c in accepted],
            'computed_new_case_ids':[c for c in CASE_ORDER if c in computed],'claimed_case_ids':[c for c in CASE_ORDER if c in claimed],
            'failed_case_ids':[c for c in CASE_ORDER if c in failed],'complete':len(accepted)==20,
            'stop_dispatch':(self.root/'STOP.json').exists(),'scientific_aggregate':None,'research_admitted':False}

    def claim(self,request,pin):
        host,cid=validate_request(request,pin,self.revision,self.implementation_sha)
        with file_lock(self.root/'.registry.lock'):
            state=self.status()
            dispatch_gate(self.revision,host,cid,state['accepted_case_ids'],state['claimed_case_ids'],stopped=state['stop_dispatch'])
            directory=bound(self.root,'claims/'+cid);directory.mkdir(parents=True,exist_ok=False)
            body={'kind':KIND+'-claim','case_id':cid,'host':host,'revision_sha256':canonical(self.revision),
                'implementation_sha256':self.implementation_sha,'request_sha256':pin,'case_seconds':4800,
                'attempt':'attempt-01','claimed_utc':datetime.now(timezone.utc).isoformat(),'research_admitted':False}
            write_record(directory/'claim.json',body)
            return body

    def fail(self,cid,host,error,elapsed):
        body={'kind':KIND+'-outcome','case_id':cid,'host':host,'revision_sha256':canonical(self.revision),
            'complete':False,'error':str(error),'elapsed_seconds':elapsed,'research_admitted':False}
        with file_lock(self.root/'.registry.lock'):
            path=self.root/'outcomes'/(cid+'.json')
            if path.exists():
                if read_record(path).get('complete') is True:
                    write_record(self.root/'failures'/(cid+'.json'),body)
            else:write_record(path,body)
            if not (self.root/'STOP.json').exists():write_record(self.root/'STOP.json',body)
        return body

    def publish(self,cid,body):
        if (body['kind']!=KIND+'-outcome' or body['case_id']!=cid or body['host']!=ASSIGNMENTS[cid]
            or body['revision_sha256']!=canonical(self.revision) or body['complete'] is not True
            or body['research_admitted'] is not False or not (self.root/'claims'/cid/'claim.json').exists()):
            raise ValueError('Invalid or unowned numerical completion.')
        write_record(self.root/'outcomes'/(cid+'.json'),body)


def execution_identity(revision,manifest,host):
    if host not in HOSTS:raise ValueError('Unknown execution host.')
    execution=copy.deepcopy(manifest['reference_execution'])
    execution.update(kind='matched-panel-two-host-numeric-execution-v1',expected_host=host,expected_GPU_identity=HOSTS[host],
        execution_revision_sha256=canonical(revision),implementation_sha256=canonical(manifest),
        revision_sources_sha256=manifest['source_sha256'],origin='two-host-revision',v1_execution_admitted=False)
    return execution


def check_sources(package,packet,revision,manifest,revision_pin,manifest_pin):
    validate_revision(revision,revision_pin)
    if (canonical(manifest)!=manifest_pin or manifest['kind']!=MANIFEST_KIND
        or manifest['revision_sha256']!=revision_pin or set(manifest['source_sha256'])!=set(SOURCES)):
        raise ValueError('Implementation manifest differs from its external pin.')
    for name,pin in manifest['source_sha256'].items():
        if digest(bound(packet,name))!=pin:raise ValueError('New revision source changed: '+name)
    if digest(Path(package)/'PANEL-HANDOFF.json')!=HANDOFF_SHA:raise ValueError('Frozen scientific handoff changed.')
    sys.path.insert(0,str(Path(package).resolve()))
    from models.matched_panel_io import load_plan,execution_identity as original_execution
    numeric_plan=load_plan(package);base=original_execution(package,numeric_plan)
    if canonical(base)!=BASE_EXECUTION_SHA or canonical(base)!=canonical(manifest['reference_execution']):
        raise ValueError('Original scientific/runtime/checkpoint identity changed.')
    for name,pin in [('bootstrap-journal.json',IMPORT_JOURNAL_SHA),('bootstrap-review.json',IMPORT_REVIEW_SHA),('bridge-review.json',BRIDGE_REVIEW_SHA)]:
        if digest(Path(packet)/name)!=pin:raise ValueError('Independent bootstrap or bridge review changed.')
    old=read_record(Path(packet)/'bootstrap-journal.json');review=read_record(Path(packet)/'bootstrap-review.json')
    bridge=read_record(Path(packet)/'bridge-review.json')
    if (set(old['completed'])!=set(CASE_ORDER[:7]) or old['execution_sha256']!=BASE_EXECUTION_SHA
        or review['completed_case_count']!=7 or review['journal_file_sha256']!=IMPORT_JOURNAL_SHA
        or not review['reviewed_export'] or not bridge['reviewed_export'] or bridge['fresh_workers_reference_exact']!=2
        or bridge['candidate_host']!='linux10603' or bridge['parallel_dispatch_authorized'] is not False):
        raise ValueError('Missing seven-case independent review or scoped second-host reference bridge.')
    return numeric_plan,base,old


def import_metadata(old):
    return {cid:{'origin':'frozen-v1','host':'linux10609','old_pointer':pointer,
        'receipt_sha256':pointer['receipt_file_sha256'],'execution_sha256':BASE_EXECUTION_SHA,
        'old_journal_sha256':IMPORT_JOURNAL_SHA,'independent_review_sha256':IMPORT_REVIEW_SHA}
        for cid,pointer in old['completed'].items()}


def verify_imports(package,old):
    """Read every old case/export through the original verifier; never rewrite v1."""
    from models.matched_panel import PanelJournal
    from models.matched_panel_io import load_plan,execution_identity as original_execution
    old_root=Path(package)/'runs/panel.qIXeDn'
    if digest(old_root/'journal.json')!=IMPORT_JOURNAL_SHA:raise ValueError('Frozen v1 import boundary changed.')
    numeric_plan=load_plan(package);base=original_execution(package,numeric_plan)
    journal=PanelJournal(old_root,plan=numeric_plan,execution=base)
    if journal.progress()['completed_case_ids']!=CASE_ORDER[:7]:raise ValueError('Old reviewed prefix changed.')
    return import_metadata(old)


def load_context(package,packet,revision_pin,manifest_pin,request_path,request_pin):
    packet=Path(packet)
    revision=json.loads((packet/'configs/matched_panel_two_host_v1.json').read_text())
    manifest=json.loads((packet/'IMPLEMENTATION.json').read_text())
    numeric,base,old=check_sources(package,packet,revision,manifest,revision_pin,manifest_pin)
    request=read_record(request_path)
    host,cid=validate_request(request,request_pin,revision,manifest_pin)
    if Path(request_path).resolve()!=bound(packet,'requests/'+cid+'.json'):
        raise ValueError('Use the case-specific request in the verified packet.')
    return revision,manifest,numeric,old,request,host,cid


def numeric_worker(package,packet,root,revision_pin,manifest_pin,request_path,request_pin,role):
    revision,manifest,numeric,_,request,host,cid=load_context(package,packet,revision_pin,manifest_pin,request_path,request_pin)
    validate_locations(package,packet,root,Path.home())
    if role not in ('matrix','replay'):raise ValueError('Unknown numerical worker role.')
    claim=read_record(bound(root,'claims/'+cid+'/claim.json'))
    directory=bound(root,'cases/'+cid+'/attempt-01')
    control=read_record(directory/'case-envelope.json')
    if (claim['request_sha256']!=request_pin or control['request_sha256']!=request_pin
        or control['active_role']!=role or control['complete'] is not False):
        raise ValueError('Worker has no matching live case owner.')
    bound(root,'claims/'+cid+'/workers/'+role).mkdir(parents=True,exist_ok=False)
    hardware_gate(host)
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
    execution=execution_identity(revision,manifest,host);expected=execution['expected_runtime']
    configure_execution('shared-fp64-native',device='cuda');torch.set_num_threads(expected['threads'])
    if (platform.python_version()!=expected['python'] or str(torch.__version__)!=expected['torch']
        or np.__version__!=expected['numpy'] or not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise ValueError('Actual numerical worker runtime differs from the frozen reference.')
    total=torch.cuda.get_device_properties(0).total_memory;limit=min(7*2**30,int(.9*total))
    torch.cuda.set_per_process_memory_fraction(min(.9,7*2**30/total))
    probe=native_probe('cuda');hardware_gate(host,allow_self=True)
    torch.manual_seed(20261024)
    clean=load_clip(package,numeric,cid)
    model=build_vitl(Path(package)/'external/vjepa2',activation_checkpointing=True)
    checkpoint=Path.home()/'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting/checkpoints/vitl.pt'
    load_checkpoint(model,checkpoint,execution['checkpoint_sha256'])
    apply_model_policy(model,'shared-fp64-native');model.to('cuda')
    if canonical(runtime_contract('cuda',model=model))!=canonical(expected):
        raise ValueError('Actual model/precision policy differs from the reference.')
    with MemoryMonitor('cuda') as memory:
        run_worker(model,clean.to('cuda'),EvaluationMaskSampler(Path(package)/'external/vjepa2'),
            plan=numeric,case_id=cid,execution=execution,output=directory/role,role=role)
    write_record(directory/role/'observation.json',{'kind':NUMERIC_KIND,'case_id':cid,'role':role,
        'execution_sha256':canonical(execution),'actual_host':platform.node(),'actual_GPU_identity':hardware_gate(host,allow_self=True),
        'memory':memory.report(),'allocator_limit_bytes':limit,'native_probe':probe,
        'research_admitted':False,'A40_cost_pool_eligible':False,'execution_revision_sha256':revision_pin,
        'origin':'two-host-revision','v1_execution_admitted':False})


def logged_child(command,path,seconds):
    """Stream stage messages while retaining the unchanged bounded-child primitive."""
    from models.matched_diagnostic import bounded_child
    path=Path(path);stop=threading.Event();cursor=0
    def show():
        nonlocal cursor
        if path.exists():
            with path.open(errors='replace') as stream:
                stream.seek(cursor);text=stream.read();cursor=stream.tell()
            if text:print(text,end='',flush=True)
    def watch():
        while not stop.wait(1):show()
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    try:return bounded_child(command,path,seconds)
    finally:stop.set();thread.join(timeout=2);show()


def run_numeric_case(package,packet,root,revision,manifest,numeric,request,request_pin,directory,budget):
    from models.matched_panel_case import KIND as NUMERIC_KIND,verify_pair,verify_case
    from models.matched_panel_io import export_case,verify_archive
    directory=Path(directory);host,cid=request['host'],request['case_id']
    execution=execution_identity(revision,manifest,host);started=time.monotonic()
    session={'kind':NUMERIC_KIND,'case_id':cid,'plan_sha256':NUMERIC_PLAN_SHA,'execution_sha256':canonical(execution),
        'complete':False,'passed':False,'research_admitted':False,'tasks':{},'elapsed_seconds':0.0,
        'started_utc':datetime.now(timezone.utc).isoformat()}
    write_record(directory/'session.json',session)
    control=read_record(directory/'case-envelope.json')
    old_handler=signal.getsignal(signal.SIGTERM)
    def terminated(*_):raise TimeoutError('Case terminated; preserve evidence and elapsed allowance.')
    signal.signal(signal.SIGTERM,terminated)
    try:
        for role,cap in [('matrix',3600),('replay',900)]:
            remaining=budget-(time.monotonic()-started)-30
            if remaining<1:raise TimeoutError('No time remains for a required numerical worker.')
            control['active_role']=role;write_record(directory/'case-envelope.json',control,replace=True)
            command=[sys.executable,'-B','-u',str(Path(packet)/'scripts/check_two_host_panel.py'),'worker',
                '--package',str(package),'--root',str(root),'--revision-sha256',canonical(revision),
                '--implementation-sha256',canonical(manifest),'--request',str(Path(packet)/'requests'/(cid+'.json')),
                '--request-sha256',request_pin,'--role',role]
            print(host,cid,'starting',role,'worker',flush=True)
            session['tasks'][role]=logged_child(command,directory/(role+'.log'),min(cap,remaining))
            session['elapsed_seconds']=time.monotonic()-started;write_record(directory/'session.json',session,replace=True)
            if not session['tasks'][role]['complete']:raise RuntimeError('Numerical worker failed/timed out; stop dispatch.')
        verify_pair(directory,plan=numeric,case_id=cid,execution=execution)
        session.update(complete=True,passed=True,elapsed_seconds=time.monotonic()-started)
        write_record(directory/'session.json',session,replace=True)
        receipt=verify_case(directory,plan=numeric,case_id=cid,execution=execution)
        write_record(directory/'receipt.json',receipt)
    except Exception as error:
        session.update(complete=False,passed=False,error=type(error).__name__+': '+str(error),elapsed_seconds=time.monotonic()-started)
        write_record(directory/'session.json',session,replace=True)
    finally:
        control.update(active_role=None);write_record(directory/'case-envelope.json',control,replace=True)
        try:
            archive=export_case(package,directory,numeric,execution,destination=directory/'numeric-evidence.tar.gz')
            verify_archive(directory/'numeric-evidence.tar.gz',archive,archive['sha256'])
        finally:signal.signal(signal.SIGTERM,old_handler)
    if not session['passed']:raise RuntimeError('Case incomplete; partial numerical export retained, no retry.')
    return receipt,archive,session


OUTER_FILES=('revision.json','implementation.json','case-request.json','case-claim.json','case-outcome.json',
    'case-envelope.json','bootstrap-journal.json','bootstrap-review.json','bridge-review.json',
    'ledger-genesis.json','ledger-imports.json','progress-at-export.json',
    'numeric/evidence.tar.gz','numeric/evidence.tar.gz.manifest.json')


def export_revision_case(packet,registry,directory):
    from models.pilot_bundle import make_archive
    directory=Path(directory);cid=read_record(directory/'case-claim.json')['case_id']
    files={
        'revision.json':Path(packet)/'configs/matched_panel_two_host_v1.json',
        'implementation.json':Path(packet)/'IMPLEMENTATION.json',
        'case-request.json':directory/'case-request.json','case-claim.json':directory/'case-claim.json',
        'case-outcome.json':registry.root/'outcomes'/(cid+'.json'),'case-envelope.json':directory/'case-envelope.json',
        'ledger-genesis.json':registry.root/'genesis.json','ledger-imports.json':registry.root/'imports.json',
        'numeric/evidence.tar.gz':directory/'numeric-evidence.tar.gz',
        'numeric/evidence.tar.gz.manifest.json':directory/'numeric-evidence.tar.gz.manifest.json'}
    for name in ('bootstrap-journal.json','bootstrap-review.json','bridge-review.json'):files[name]=Path(packet)/name
    for name in SOURCES:files['source/'+name]=Path(packet)/name
    write_record(directory/'progress-at-export.json',registry.status())
    files['progress-at-export.json']=directory/'progress-at-export.json'
    if set(files)!=set(OUTER_FILES)|{'source/'+n for n in SOURCES}:raise ValueError('Unexpected outer export inventory.')
    if any(p.is_symlink() for p in files.values()) or sum(p.stat().st_size for p in files.values())>256*2**20:
        raise ValueError('Outer export exceeds the fixed regular-file bound.')
    archive=directory/'revision-evidence.tar.gz';manifest=make_archive(archive,files)
    sidecar=Path(str(archive)+'.manifest.json');sidecar.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    event={'kind':KIND+'-export','case_id':cid,'revision_sha256':canonical(registry.revision),
        'archive_path':str(archive.relative_to(registry.root)),'archive_sha256':manifest['sha256'],
        'archive_manifest_path':str(sidecar.relative_to(registry.root)),'archive_manifest_sha256':digest(sidecar),
        'bytes':manifest['bytes'],'members':len(manifest['files'])}
    write_record(registry.root/'exports'/(cid+'.json'),event)
    print('Two-host case export:',json.dumps(event,sort_keys=True),flush=True)
    return event


def run_one(package,packet,root,revision_pin,manifest_pin,request_path,request_pin):
    started=time.monotonic()
    revision,manifest,numeric,old,request,host,cid=load_context(package,packet,revision_pin,manifest_pin,request_path,request_pin)
    validate_locations(package,packet,root,Path.home())
    from models.matched_panel_runner import case_deadline
    with case_deadline(LIMIT,kill_grace_seconds=15):
        imports=verify_imports(package,old)
        registry=Registry(root,revision,manifest_pin,imports)
        with registry.host_lease(host):
            hardware_gate(host)
            if __import__('shutil').disk_usage(root).free<4*2**30:raise OSError('Insufficient artifact/export space.')
            # Also refuse any old v1 attempt/operator/export of a pending case.
            old_root=Path(package)/'runs/panel.qIXeDn'
            if any((old_root/name).exists() for name in ['cases/'+cid,'operator-attended/'+cid,
                'exports/'+cid+'-attempt-01.tar.gz','exports/'+cid+'-attempt-01.tar.gz.manifest.json']):
                raise ValueError('Pending case has old unreviewed evidence; never reset its allowance.')
            claim=registry.claim(request,request_pin)
            directory=bound(root,'cases/'+cid+'/attempt-01');directory.mkdir(parents=True,exist_ok=False)
            write_record(directory/'case-request.json',request);write_record(directory/'case-claim.json',claim)
            control={'kind':KIND+'-case-envelope','revision_sha256':revision_pin,'implementation_sha256':manifest_pin,
                'request_sha256':request_pin,'case_id':cid,'host':host,'complete':False,'active_role':None,
                'case_seconds':LIMIT,'elapsed_seconds':0.0,'research_admitted':False}
            write_record(directory/'case-envelope.json',control)
            try:
                budget=min(4620,LIMIT-(time.monotonic()-started)-120)
                if budget<1:raise TimeoutError('Preflight consumed the original case allowance.')
                receipt,inner,session=run_numeric_case(package,packet,root,revision,manifest,numeric,request,request_pin,directory,budget)
                control.update(complete=True,active_role=None,elapsed_seconds=time.monotonic()-started)
                write_record(directory/'case-envelope.json',control,replace=True)
                body={'kind':KIND+'-outcome','case_id':cid,'host':host,'revision_sha256':revision_pin,
                    'implementation_sha256':manifest_pin,'request_sha256':request_pin,'complete':True,
                    'numeric_receipt_sha256':canonical(receipt),'numeric_receipt_file_sha256':digest(directory/'receipt.json'),
                    'execution_sha256':receipt['execution_sha256'],'numeric_archive_sha256':inner['sha256'],
                    'elapsed_seconds':control['elapsed_seconds'],'research_admitted':False}
                registry.publish(cid,body)
                export=export_revision_case(packet,registry,directory)
                print(host,cid,'numerically verified; independent export review is still required. No next case starts.',flush=True)
                return export
            except BaseException as error:
                control.update(complete=False,active_role=None,elapsed_seconds=time.monotonic()-started,
                               error=type(error).__name__+': '+str(error))
                write_record(directory/'case-envelope.json',control,replace=True)
                registry.fail(cid,host,control['error'],control['elapsed_seconds'])
                raise


def review_case(package,packet,revision_pin,manifest_pin,archive,sidecar,expected_sha,output):
    revision=json.loads((Path(packet)/'configs/matched_panel_two_host_v1.json').read_text())
    manifest=json.loads((Path(packet)/'IMPLEMENTATION.json').read_text())
    numeric,_,old=check_sources(package,packet,revision,manifest,revision_pin,manifest_pin)
    from models.pilot_bundle import extract_verified
    from models.matched_panel_io import verify_archive,review_export
    verify_archive(archive,sidecar,expected_sha)
    if set(sidecar['files'])!=set(OUTER_FILES)|{'source/'+n for n in SOURCES}:raise ValueError('Unexpected revision export inventory.')
    extract_verified(archive,sidecar,output);output=Path(output)
    if (canonical(json.loads((output/'revision.json').read_text()))!=revision_pin
        or canonical(json.loads((output/'implementation.json').read_text()))!=manifest_pin):
        raise ValueError('Exported revision or implementation changed.')
    for name,pin in manifest['source_sha256'].items():
        if digest(output/'source'/name)!=pin:raise ValueError('Captured new execution source changed.')
    for name,pin in [('bootstrap-journal.json',IMPORT_JOURNAL_SHA),('bootstrap-review.json',IMPORT_REVIEW_SHA),('bridge-review.json',BRIDGE_REVIEW_SHA)]:
        if digest(output/name)!=pin:raise ValueError('Captured independent bootstrap changed.')
    request=read_record(output/'case-request.json');claim=read_record(output/'case-claim.json')
    host,cid=validate_request(request,canonical(request),revision,manifest_pin)
    if (claim['case_id']!=cid or claim['host']!=host or claim['request_sha256']!=canonical(request)
        or claim['revision_sha256']!=revision_pin or claim['implementation_sha256']!=manifest_pin):
        raise ValueError('Case ownership differs from its prepared request.')
    envelope=read_record(output/'case-envelope.json');outcome=read_record(output/'case-outcome.json')
    if (envelope['case_id']!=cid or envelope['host']!=host or envelope['request_sha256']!=canonical(request)
        or envelope['revision_sha256']!=revision_pin or envelope['implementation_sha256']!=manifest_pin
        or envelope['complete'] is not True or envelope['active_role'] is not None
        or envelope['case_seconds']!=LIMIT or envelope['research_admitted'] is not False
        or type(envelope['elapsed_seconds']) not in (int,float) or not 0<envelope['elapsed_seconds']<=LIMIT):
        raise ValueError('Case envelope is incomplete, mixed or over limit.')
    expected=execution_identity(revision,manifest,host)
    inner=output/'numeric/evidence.tar.gz';inner_spec=json.loads(Path(str(inner)+'.manifest.json').read_text())
    receipt=review_export(inner,inner_spec,outcome['numeric_archive_sha256'],output/'numeric-reviewed',plan=numeric,execution=expected)
    if (outcome['kind']!=KIND+'-outcome' or outcome['case_id']!=cid or outcome['host']!=host
        or outcome['revision_sha256']!=revision_pin or outcome['implementation_sha256']!=manifest_pin
        or outcome['request_sha256']!=canonical(request) or outcome['complete'] is not True
        or outcome['numeric_receipt_sha256']!=canonical(receipt)
        or outcome['numeric_receipt_file_sha256']!=digest(output/'numeric-reviewed/receipt.json')
        or outcome['execution_sha256']!=canonical(expected) or outcome['research_admitted'] is not False):
        raise ValueError('Numerical outcome differs from independent retained evidence.')
    imports=read_record(output/'ledger-imports.json');genesis=read_record(output/'ledger-genesis.json')
    if (imports!=import_metadata(old) or genesis['revision_sha256']!=revision_pin
        or genesis['implementation_sha256']!=manifest_pin or genesis['imports_sha256']!=canonical(imports)):
        raise ValueError('Exported ledger does not preserve the reviewed imports.')
    return {'kind':REVIEW_KIND,'revision_sha256':revision_pin,'implementation_sha256':manifest_pin,
        'case_id':cid,'host':host,'archive_sha256':expected_sha,'outcome_file_sha256':digest(output/'case-outcome.json'),
        'numeric_receipt_sha256':canonical(receipt),'execution_sha256':canonical(expected),
        'reviewed_export':True,'research_admitted':False,'automatic_queue_authorized':False}


def register_review(package,packet,root,revision_pin,manifest_pin,marker_path,marker_sha):
    if digest(marker_path)!=marker_sha:raise ValueError('Review differs from its external file pin.')
    marker=read_record(marker_path)
    revision=json.loads((Path(packet)/'configs/matched_panel_two_host_v1.json').read_text())
    manifest=json.loads((Path(packet)/'IMPLEMENTATION.json').read_text())
    numeric,_,old=check_sources(package,packet,revision,manifest,revision_pin,manifest_pin)
    validate_locations(package,packet,root,Path.home())
    registry=Registry(root,revision,manifest_pin,verify_imports(package,old))
    cid=marker.get('case_id')
    if (marker.get('kind')!=REVIEW_KIND or cid not in CASE_ORDER[7:] or marker.get('host')!=ASSIGNMENTS[cid]
        or marker.get('revision_sha256')!=revision_pin or marker.get('implementation_sha256')!=manifest_pin
        or marker.get('reviewed_export') is not True or marker.get('research_admitted') is not False):
        raise ValueError('Independent review does not belong to this revision case.')
    export=read_record(bound(root,'exports/'+cid+'.json'))
    if (marker['archive_sha256']!=export['archive_sha256']
        or marker['outcome_file_sha256']!=digest(bound(root,'outcomes/'+cid+'.json'))
        or digest(bound(root,export['archive_path']))!=export['archive_sha256']
        or digest(bound(root,export['archive_manifest_path']))!=export['archive_manifest_sha256']):
        raise ValueError('Reviewed export/outcome changed.')
    from models.matched_panel_case import verify_case
    rebuilt=verify_case(bound(root,'cases/'+cid+'/attempt-01'),plan=numeric,case_id=cid,
        execution=execution_identity(revision,manifest,ASSIGNMENTS[cid]))
    if canonical(rebuilt)!=marker['numeric_receipt_sha256']:raise ValueError('Retained numerical evidence changed after review.')
    target=bound(root,'reviews/'+cid+'.json')
    with file_lock(Path(root)/'.registry.lock'):
        if target.exists():
            if read_record(target)!=marker:raise ValueError('Refusing to replace an existing independent review.')
        else:write_record(target,marker)
    return registry.status()


def preflight(package,packet,root,revision_pin,manifest_pin,request_path,request_pin):
    revision,manifest,_,old,request,host,cid=load_context(package,packet,revision_pin,manifest_pin,request_path,request_pin)
    validate_locations(package,packet,root,Path.home());hardware_gate(host)
    if digest(Path(package)/'runs/panel.qIXeDn/journal.json')!=IMPORT_JOURNAL_SHA:
        raise ValueError('Original seven-case import boundary changed.')
    return {'host':host,'case_id':cid,'revision_sha256':revision_pin,'implementation_sha256':manifest_pin,
        'scientific_plan_sha256':NUMERIC_PLAN_SHA,'imported_cases':len(old['completed']),
        'GPU_job_started':False,'research_admitted':False}
