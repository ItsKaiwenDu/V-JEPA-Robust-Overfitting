#!/usr/bin/env python3
"""Explicit canonical-path continuation; original numerical sources stay pinned."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import uuid

KIND='matched-panel-canonical-continuation-v1'
RP='5ca7596fe95ecb8f6b6c6752426e56bb354b1f019d3576c6618f7ede98ae4116'
MP='da5aeb93bd6d3e46040b158cc14f0d6feb1284a83b2b2778a3639b420db0acf9'
SUPPORT='0ba6950f342c6e9a4801e9fcb7e2575c6e9b5b88134da41ac3b4d9b6c49d1d12'
PRIOR='62a0b5a7222d50d670dc1c90cc5e23682aa02fe8d10f8a479c0bd81d2bf38fc0'
UPLOAD='aefe6fb4b6882bce5ab5102441ce9b2adb64ceb6503d1f1486a785af56214be3'
STOP='b42a023f4b7e3bf0c9183f7af384d4abe3cccecaf05d727495b9782107c0f892'
BASE='Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024'
PACKET=BASE+'/two-host-continuation-20261026/unpacked/two-host-continuation-20261026'
TARGETS={'linux10609':'clip-05-bank-0','linux10603':'clip-04-bank-1'}
ACCEPTED=[f'clip-{i:02d}-bank-{b}' for i in range(10) for b in (0,1)][:9]
OLD_CASES=['clip-03-bank-1','clip-04-bank-0']
ARCHIVE='archived-stops/'+STOP+'.json'
RESOLUTION='stop-resolutions/export-alias-20261026.json'
SOURCES=('scripts/continue_two_host_panel.py','tests/test_panel_continuation.py',
 'tests/fixtures/panel_continuation_metadata.json','support/sync_bookkeeping.py')


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def validate_operation(operation,pin):
    if canonical(operation)!=pin:raise ValueError('Operational revision differs from its external pin.')
    fixed={'kind':KIND,'numeric_plan_changed':False,'reference_revision_sha256':RP,'reference_implementation_sha256':MP,
        'prior_sync_receipt_sha256':PRIOR,'resolved_stop_sha256':STOP,'accepted_before':ACCEPTED,'targets':TARGETS,
        'root_path_mode':'canonical','case_seconds':4800,'kill_grace_seconds':15,'matrix_seconds':3600,
        'replay_seconds':900,'parent_seconds':4620,'max_attempts':1,'additional_allowance_seconds':0,
        'automatic_GPU_retries':0,'automatic_queue':False,'one_case_per_host':True,'paid_compute':False,'research_admitted':False}
    if any(k not in operation or canonical(operation[k])!=canonical(v) for k,v in fixed.items()):
        raise ValueError('Use the fixed next pair and original numerical/resource policy.')
    if set(operation['sources_sha256'])!=set(SOURCES):raise ValueError('Incomplete operational source inventory.')


def request(operation,host):
    if host not in TARGETS:raise ValueError('Unknown assigned host.')
    return {'kind':'matched-panel-two-host-single-case-request-v1','revision_sha256':RP,'implementation_sha256':MP,
        'host':host,'case_id':TARGETS[host],'case_seconds':4800,'kill_grace_seconds':15,'max_attempts':1,
        'one_case_only':True,'additional_allowance_seconds':0,'automatic_queue':False,'paid_compute':False,
        'research_admitted':False,'operational_revision_sha256':canonical(operation)}


def resolution(operation):
    return {'kind':KIND+'-stop-resolution','operation_sha256':canonical(operation),'old_STOP_sha256':STOP,
        'archived_STOP_path':ARCHIVE,'accepted_case_ids':ACCEPTED,'prior_sync_receipt_sha256':PRIOR,
        'reason':'Both saved cases independently verified and bookkeeping synced; canonical coordinator paths prevent the reviewed export alias error.',
        'targets':TARGETS,'root_path_mode':'canonical','scientific_settings_changed':False,
        'GPU_retry_authorized':False,'automatic_queue':False,'research_admitted':False}


def exclusive_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes()!=data:raise ValueError('Conflicting historical stop copy.')
        return
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def same_record(path,payload,ops):
    if Path(path).exists() and (Path(path).is_symlink() or ops.read_record(path)!=payload):
        raise ValueError('Conflicting continuation metadata: '+str(path))


def publish_resolution(root,original_packet,operation,ops):
    """Caller holds both leases + registry lock. STOP is removed last."""
    root=Path(root).resolve();original_packet=Path(original_packet).resolve();stop=root/'STOP.json'
    archive=ops.bound(root,ARCHIVE);record=ops.bound(root,RESOLUTION);body=resolution(operation)
    requests={host:request(operation,host) for host in TARGETS}
    if stop.exists():
        if stop.is_symlink() or digest(stop)!=STOP:raise ValueError('A different stop must be reviewed; never clear it.')
        stop_bytes=stop.read_bytes()
    else:
        if not archive.is_file() or digest(archive)!=STOP or not record.exists():raise ValueError('Missing reviewed stop history or resolution.')
        stop_bytes=archive.read_bytes()
    if archive.exists() and (archive.is_symlink() or archive.read_bytes()!=stop_bytes):raise ValueError('Historical stop copy differs.')
    same_record(record,body,ops)
    for host,q in requests.items():same_record(ops.bound(original_packet,'requests/'+q['case_id']+'.json'),q,ops)
    # Preflight every immutable destination before the first metadata write.
    exclusive_bytes(archive,stop_bytes)
    for host,q in requests.items():
        target=original_packet/'requests'/(q['case_id']+'.json')
        if not target.exists():ops.write_record(target,q)
    if not record.exists():ops.write_record(record,body)
    if digest(archive)!=STOP or ops.read_record(record)!=body:raise ValueError('Stop-resolution readback failed.')
    if stop.exists():
        if digest(stop)!=STOP:raise ValueError('Stop changed during resolution.')
        stop.unlink()
        fd=os.open(root,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    return body,requests


def load(packet,pin,home):
    packet=Path(packet).resolve();home=Path(home)
    if packet!=(home/PACKET).resolve():raise ValueError('Use the fixed operational packet; no copied-root resets.')
    operation=json.loads((packet/'OPERATION.json').read_text());validate_operation(operation,pin)
    for name,sha in operation['sources_sha256'].items():
        if digest(packet/name)!=sha:raise ValueError('Operational source changed: '+name)
    if digest(packet/'support/sync_bookkeeping.py')!=SUPPORT:raise ValueError('Reviewed support operator changed.')
    spec=importlib.util.spec_from_file_location('reviewed_sync_support',packet/'support/sync_bookkeeping.py')
    support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
    plan,_=support.decode((packet/'reviewed-sync-upload.json').read_bytes(),UPLOAD)
    if digest(packet/'prior-sync-receipt.json')!=PRIOR:raise ValueError('Confirmed checkpoint changed.')
    receipt=json.loads((packet/'prior-sync-receipt.json').read_text());support.verify_receipt(receipt,plan)
    package=(home/support.PACKAGE).resolve();original=(home/support.PACKET).resolve();root=(home/support.COORDINATOR).resolve()
    ops,revision,manifest,numeric,old=support.load_frozen(package,original)
    ops.validate_locations(package,original,root,home)
    preserved=support.source_paths(home)
    if not (root/'STOP.json').exists():preserved['ledger/STOP.json']=root/ARCHIVE
    support.check_preserved(preserved,plan['preserve_remote_files_sha256'])
    return operation,support,plan,receipt,ops,revision,manifest,numeric,old,package,original,root


def verify_nine(context,home):
    operation,support,plan,receipt,ops,revision,manifest,numeric,old,package,original,root=context
    import torch
    from models.matched_panel_case import verify_case
    from models.matched_panel_io import verify_archive
    torch.set_num_threads(4)
    registry=ops.Registry(root,revision,MP,ops.verify_imports(package,old))
    paths=support.source_paths(home)
    if not (root/'STOP.json').exists():paths['ledger/STOP.json']=root/ARCHIVE
    support.check_preserved(paths,plan['preserve_remote_files_sha256'])
    for name,pin in receipt['records_file_sha256'].items():
        if digest(ops.bound(root,name))!=pin:raise ValueError('Confirmed nine-case record changed.')
    for host,cid in support.CASES.items():
        event=ops.read_record(root/'exports'/(cid+'.json'));marker=ops.read_record(root/'reviews'/(cid+'.json'))
        archive=ops.bound(root,event['archive_path']);sidecar=ops.bound(root,event['archive_manifest_path'])
        verify_archive(archive,json.loads(sidecar.read_text()),event['archive_sha256'])
        rebuilt=verify_case(root/'cases'/cid/'attempt-01',plan=numeric,case_id=cid,execution=ops.execution_identity(revision,manifest,host))
        if canonical(rebuilt)!=marker['numeric_receipt_sha256']:raise ValueError('Saved numerical result changed.')
    state=registry.status()
    if (state['accepted_case_ids']!=ACCEPTED or state['claimed_case_ids']!=OLD_CASES
        or state['computed_new_case_ids']!=OLD_CASES or state['failed_case_ids'] or state['scientific_aggregate'] is not None):
        raise ValueError('Activation requires the exact reviewed nine-case checkpoint.')
    for host,cid in TARGETS.items():
        if any((root/n).exists() for n in ['cases/'+cid,'claims/'+cid,'outcomes/'+cid+'.json','reviews/'+cid+'.json','exports/'+cid+'.json']):
            raise ValueError('Next case already has evidence or a claim; never reset its allowance.')
        old_root=package/'runs/panel.qIXeDn'
        if any((old_root/n).exists() for n in ['cases/'+cid,'operator-attended/'+cid,'exports/'+cid+'-attempt-01.tar.gz']):
            raise ValueError('Next case has old evidence; inspect it before any attempt.')
    return registry,paths


def activate(packet,pin,home,host):
    if host!='linux10609':raise ValueError('Use linux10609 for the shared metadata activation.')
    context=load(packet,pin,home);operation,support,plan,prior,ops,revision,manifest,numeric,old,package,original,root=context
    # Import/read only. No original numerical model entry point is called here.
    with ExitStack() as stack:
        for assigned in TARGETS:stack.enter_context(ops.file_lock(root/'leases'/assigned,wait_seconds=0))
        registry,paths=verify_nine(context,home)
        with ops.file_lock(root/'.registry.lock'):
            body,requests=publish_resolution(root,original,operation,ops)
            paths['ledger/STOP.json']=root/ARCHIVE;support.check_preserved(paths,plan['preserve_remote_files_sha256'])
            status=registry.status()
            if status['accepted_case_ids']!=ACCEPTED or status['stop_dispatch']:raise ValueError('Activation state readback failed.')
    return {'kind':KIND+'-activation-receipt','operation_sha256':pin,'host':host,'canonical_root':str(root),
        'archived_STOP_text':(root/ARCHIVE).read_text(),'resolution_record_text':(root/RESOLUTION).read_text(),
        'request_record_texts':{h:(original/'requests'/(q['case_id']+'.json')).read_text() for h,q in requests.items()},
        'status_after':status,'preserved_prior_file_hashes':46,'prior_numeric_implementation_unchanged':True,
        'models_started':False,'GPU_retry':False,'automatic_queue':False,'research_admitted':False}


def run_command(packet,pin,home,host):
    context=load(packet,pin,home);operation,support,plan,prior,ops,revision,manifest,numeric,old,package,original,root=context
    if host not in TARGETS:raise ValueError('Unknown assigned host.')
    if (root/'STOP.json').exists() or ops.read_record(root/RESOLUTION)!=resolution(operation) or digest(root/ARCHIVE)!=STOP:
        raise ValueError('Resolve only the reviewed stop before this case; new stops always block dispatch.')
    ops.hardware_gate(host)
    q=request(operation,host);path=original/'requests'/(q['case_id']+'.json')
    if ops.read_record(path)!=q:raise ValueError('Prepared case request changed.')
    registry=ops.Registry(root,revision,MP,ops.verify_imports(package,old));state=registry.status()
    if not set(ACCEPTED)<=set(state['accepted_case_ids']):raise ValueError('A reviewed predecessor is missing.')
    ops.dispatch_gate(revision,host,q['case_id'],state['accepted_case_ids'],state['claimed_case_ids'],stopped=state['stop_dispatch'])
    return [sys.executable,'-B','-u',str(original/'scripts/check_two_host_panel.py'),'run',
        '--package',str(package),'--root',str(root),'--revision-sha256',RP,'--implementation-sha256',MP,
        '--request',str(path),'--request-sha256',canonical(q)]


def payload_text(text):
    value=json.loads(text)
    if set(value)!={'payload','sha256'} or canonical(value['payload'])!=value['sha256']:raise ValueError('Receipt record checksum mismatch.')
    return value['payload']


def verify_activation(receipt,operation,pin):
    validate_operation(operation,pin);status=receipt['status_after']
    if (receipt['kind']!=KIND+'-activation-receipt' or receipt['operation_sha256']!=pin or receipt['host']!='linux10609'
        or not Path(receipt['canonical_root']).is_absolute() or not receipt['canonical_root'].endswith(BASE+'/two-host-panel-20261026')
        or hashlib.sha256(receipt['archived_STOP_text'].encode()).hexdigest()!=STOP
        or payload_text(receipt['resolution_record_text'])!=resolution(operation)
        or set(receipt['request_record_texts'])!=set(TARGETS)
        or status['accepted_case_ids']!=ACCEPTED or status['claimed_case_ids']!=OLD_CASES
        or status['computed_new_case_ids']!=OLD_CASES or status['failed_case_ids'] or status['stop_dispatch'] is not False
        or status['complete'] is not False or status['scientific_aggregate'] is not None or status['research_admitted'] is not False
        or receipt['preserved_prior_file_hashes']!=46 or receipt['prior_numeric_implementation_unchanged'] is not True
        or any(receipt[k] is not False for k in ['models_started','GPU_retry','automatic_queue','research_admitted'])):
        raise ValueError('Activation does not match the bounded reviewed stop resolution.')
    for host,text in receipt['request_record_texts'].items():
        if payload_text(text)!=request(operation,host):raise ValueError('Actual next-case request changed.')
    return {'kind':KIND+'-activation-verification','verified':True,'accepted_cases':9,'STOP_archived_and_resolved':True,
        'next_case_ids':list(TARGETS.values()),'models_started':False,'automatic_queue':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('activate','run'))
    parser.add_argument('--operation-sha256',required=True);args=parser.parse_args();packet=Path(__file__).resolve().parents[1]
    if args.mode=='activate':print(json.dumps(activate(packet,args.operation_sha256,Path.home(),socket.gethostname()),sort_keys=True,indent=2))
    else:
        command=run_command(packet,args.operation_sha256,Path.home(),socket.gethostname())
        print('One original-limit case via the unchanged runner; canonical root:',command[command.index('--root')+1],flush=True)
        os.execv(command[0],command)


if __name__=='__main__':main()
