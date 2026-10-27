#!/usr/bin/env python3
"""Append four independently reviewed records; retain STOP and every old file.

This operator checks saved evidence on CPU. It starts no model, CUDA worker,
test job, retry or queue, and never installs a replacement numerical source.
"""
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys

BASE='Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024'
PACKAGE=BASE+'/unpacked/matched-panel-20261024'
PACKET=BASE+'/two-host-execution-20261026/unpacked/two-host-execution-20261026'
COORDINATOR=BASE+'/two-host-panel-20261026'
RP='5ca7596fe95ecb8f6b6c6752426e56bb354b1f019d3576c6618f7ede98ae4116'
MP='da5aeb93bd6d3e46040b158cc14f0d6feb1284a83b2b2778a3639b420db0acf9'
PLAN_PIN='9755c213846be5cc6a97fa374eff0c80728d3c1f376f79324d3a1a53df6c005a'
ACCEPTANCE_PIN='dfe57c2ea0d7c910e3755aafde62705a9559d2fedb442d59892b2cd15e034b64'
CASES={'linux10609':'clip-03-bank-1','linux10603':'clip-04-bank-0'}
TARGETS={prefix+'/'+cid+'.json' for prefix in ('exports','reviews') for cid in CASES.values()}
LIMIT=256*1024


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def record_text(text,pin):
    if hashlib.sha256(text.encode()).hexdigest()!=pin:raise ValueError('Externally reviewed record changed.')
    value=json.loads(text)
    if set(value)!={'payload','sha256'} or canonical(value['payload'])!=value['sha256']:
        raise ValueError('Record checksum mismatch.')
    return value['payload']


def record_bytes(payload):
    return (json.dumps({'payload':payload,'sha256':canonical(payload)},sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def decode(body,pin):
    if len(body)>LIMIT or hashlib.sha256(body).hexdigest()!=pin:raise ValueError('Pinned metadata upload changed.')
    bundle=json.loads(body)
    if set(bundle)!={'kind','plan_text','acceptance_text','audit_report_text'} or bundle['kind']!='two-host-reviewed-ledger-sync-upload-v1':
        raise ValueError('Wrong bookkeeping upload inventory.')
    plan=record_text(bundle['plan_text'],PLAN_PIN);acceptance=record_text(bundle['acceptance_text'],ACCEPTANCE_PIN)
    report=json.loads(bundle['audit_report_text'])
    if (hashlib.sha256(bundle['audit_report_text'].encode()).hexdigest()!=acceptance['independent_report_sha256']
        or report['verified'] is not True or report['independently_verified_case_count']!=9
        or report['new_GPU_or_model_work_during_review'] is not False or report['remote_bookkeeping_synchronized'] is not False):
        raise ValueError('Independent retained-evidence audit changed.')
    if (plan['kind']!='two-host-export-bookkeeping-repair-plan-v1' or plan['revision_sha256']!=RP or plan['implementation_sha256']!=MP
        or set(plan['proposed_records'])!=TARGETS or len(plan['preserve_remote_files_sha256'])!=46
        or plan['stop_dispatch_remains'] is not True or plan['model_or_test_execution'] is not False
        or plan['GPU_retry'] is not False or plan['remote_changes_performed'] is not False or plan['research_admitted'] is not False):
        raise ValueError('Only the four pinned append-only bookkeeping records are allowed.')
    if (acceptance['revision_sha256']!=RP or acceptance['implementation_sha256']!=MP
        or acceptance['ledger_repair_plan_sha256']!=PLAN_PIN or acceptance['STOP_preserved'] is not True
        or acceptance['future_GPU_work_authorized_by_this_marker'] is not False):
        raise ValueError('Independent audit acceptance does not match the bounded repair.')
    for name,item in plan['proposed_records'].items():
        if hashlib.sha256(record_bytes(item['payload'])).hexdigest()!=item['file_sha256']:
            raise ValueError('Proposed record serialization differs from its pin.')
        if name.startswith('reviews/') and acceptance['case_review_files_sha256'][item['payload']['case_id']]!=item['file_sha256']:
            raise ValueError('Case marker differs from independent review.')
    return plan,acceptance


def source_paths(home):
    home=Path(home);root=home/COORDINATOR;packet=home/PACKET
    paths={'ledger/genesis.json':root/'genesis.json','ledger/imports.json':root/'imports.json','ledger/STOP.json':root/'STOP.json',
        'original-v1/journal.json':home/PACKAGE/'runs/panel.qIXeDn/journal.json',
        'packet/IMPLEMENTATION.json':packet/'IMPLEMENTATION.json','packet/revision.json':packet/'configs/matched_panel_two_host_v1.json'}
    for host,cid in CASES.items():
        directory=root/'cases'/cid/'attempt-01'
        for name in ('revision-evidence.tar.gz','revision-evidence.tar.gz.manifest.json','receipt.json','session.json',
                     'case-envelope.json','case-claim.json','case-request.json','progress-at-export.json'):
            paths['cases/'+cid+'/'+name]=directory/name
        paths['ledger/claims/'+cid+'.json']=root/'claims'/cid/'claim.json'
        for name in ('outcomes','failures'):paths['ledger/'+name+'/'+cid+'.json']=root/name/(cid+'.json')
        for name in ('started-utc.txt','ended-utc.txt','exit-status.txt','case.log','cpu.log','cpu-results.xml','cpu-summary.log'):
            paths['launchers/'+host+'/'+name]=packet/'launchers'/(host+'-'+cid)/name
        for name in ('launch','pipeline'):paths['launchers/'+host+'/source-'+name+'.sh']=packet/(name+'-'+host+'.sh')
    return paths


def check_preserved(paths,pins):
    if set(paths)!=set(pins):raise ValueError('Preserved input inventory changed.')
    for name,path in paths.items():
        if path.is_symlink() or not path.is_file() or digest(path)!=pins[name]:
            raise ValueError('Saved evidence changed or missing: '+name)


def check_targets(root,proposed):
    """Inspect every destination before publishing any record, including retries."""
    if set(proposed)!=TARGETS:raise ValueError('Only four reviewed destinations are allowed.')
    existing=[]
    for name,item in sorted(proposed.items()):
        path=Path(root)/name
        if hashlib.sha256(record_bytes(item['payload'])).hexdigest()!=item['file_sha256']:
            raise ValueError('Proposed bookkeeping bytes differ from their external pin.')
        if not path.resolve().is_relative_to(Path(root).resolve()) or path.is_symlink() or path.parent.is_symlink():
            raise ValueError('Escaped bookkeeping destination.')
        if path.exists():
            if not path.is_file() or digest(path)!=item['file_sha256']:raise ValueError('Existing bookkeeping record conflicts: '+name)
            existing.append(name)
    return existing


def publish(root,proposed,preserved,ops):
    """An interrupted sync may leave an exact subset; repetition only fills gaps."""
    paths,pins=preserved;check_preserved(paths,pins);existing=check_targets(root,proposed);created=[]
    for name,item in sorted(proposed.items()):
        if name not in existing:
            ops.write_record(Path(root)/name,item['payload']);created.append(name)
        if digest(Path(root)/name)!=item['file_sha256']:raise ValueError('Published bookkeeping readback failed.')
    check_preserved(paths,pins)
    return created,existing


def load_frozen(package,packet):
    manifest=json.loads((Path(packet)/'IMPLEMENTATION.json').read_text())
    if canonical(manifest)!=MP:raise ValueError('Use the original completed-batch implementation.')
    for name in ('models/two_host_panel.py','scripts/check_two_host_panel.py'):
        if digest(Path(packet)/name)!=manifest['source_sha256'][name]:raise ValueError('Deployed trusted entry point changed.')
    spec=importlib.util.spec_from_file_location('completed_batch_operator',Path(packet)/'models/two_host_panel.py')
    ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
    revision=json.loads((Path(packet)/'configs/matched_panel_two_host_v1.json').read_text())
    numeric,_,old=ops.check_sources(package,packet,revision,manifest,RP,MP)
    return ops,revision,manifest,numeric,old


def check_state(status):
    expected=list(CASES.values())
    if (status['computed_new_case_ids']!=expected or status['claimed_case_ids']!=expected or status['failed_case_ids']
        or status['stop_dispatch'] is not True or status['complete'] is not False
        or status['scientific_aggregate'] is not None or status['research_admitted'] is not False):
        raise ValueError('The coordinator differs from the reviewed two-case stopped state.')


def run(home,host,plan,acceptance):
    if host!='linux10609':raise ValueError('Use the explicitly pinned host for this shared-home repair.')
    home=Path(home);package=home/PACKAGE;packet=home/PACKET;root=(home/COORDINATOR).resolve()
    ops,revision,manifest,numeric,old=load_frozen(package,packet)
    ops.validate_locations(package,packet,root,home)
    paths=source_paths(home);pins=plan['preserve_remote_files_sha256']
    check_preserved(paths,pins)
    registry=ops.Registry(root,revision,MP,ops.verify_imports(package,old))
    with ExitStack() as stack:
        for assigned in CASES:stack.enter_context(registry.host_lease(assigned))
        check_preserved(paths,pins);check_targets(root,plan['proposed_records'])
        check_state(registry.status())
        import torch
        from models.matched_panel_case import verify_case
        from models.matched_panel_io import verify_archive
        torch.set_num_threads(4)
        for assigned,cid in CASES.items():
            event=plan['proposed_records']['exports/'+cid+'.json']['payload']
            marker=plan['proposed_records']['reviews/'+cid+'.json']['payload']
            archive=ops.bound(root,event['archive_path']);sidecar=ops.bound(root,event['archive_manifest_path'])
            if digest(sidecar)!=event['archive_manifest_sha256']:raise ValueError('Saved sidecar changed.')
            verify_archive(archive,json.loads(sidecar.read_text()),event['archive_sha256'])
            receipt=verify_case(root/'cases'/cid/'attempt-01',plan=numeric,case_id=cid,execution=ops.execution_identity(revision,manifest,assigned))
            if (ops.canonical(receipt)!=marker['numeric_receipt_sha256'] or receipt!=ops.read_record(root/'cases'/cid/'attempt-01/receipt.json')
                or marker['outcome_file_sha256']!=digest(root/'outcomes'/(cid+'.json'))):
                raise ValueError('Retained result differs from its independent review.')
        with ops.file_lock(root/'.registry.lock'):
            before=registry.status()
            check_state(before)
            created,existing=publish(root,plan['proposed_records'],(paths,pins),ops)
            after=registry.status()
            check_state(after)
            if (after['accepted_case_ids']!=plan['after_repair_accepted_case_ids'] or not after['stop_dispatch']
                or after['scientific_aggregate'] is not None):raise ValueError('Readback did not preserve the reviewed stopped state.')
    return {'kind':'two-host-reviewed-ledger-sync-receipt-v1','revision_sha256':RP,'implementation_sha256':MP,
        'plan_file_sha256':PLAN_PIN,'audit_acceptance_file_sha256':ACCEPTANCE_PIN,'host':host,
        'created':created,'preexisting_identical':existing,'records_file_sha256':{n:v['file_sha256'] for n,v in plan['proposed_records'].items()},
        'records_readback':{n:(root/n).read_text() for n in plan['proposed_records']},'STOP_readback':(root/'STOP.json').read_text(),
        'preserved_files_verified':len(pins),'preserved_files_map_sha256':canonical(pins),'STOP_sha256':digest(root/'STOP.json'),
        'original_imports_reverified':7,'saved_new_cases_reverified':2,'accepted_before':len(before['accepted_case_ids']),
        'accepted_after':len(after['accepted_case_ids']),'status_after':after,'GPU_or_model_job_started':False,
        'test_job_started':False,'old_records_changed':False,'research_admitted':False}


def verify_receipt(receipt,plan):
    expected={n:item['file_sha256'] for n,item in plan['proposed_records'].items()}
    status=receipt['status_after'];created=receipt['created'];existing=receipt['preexisting_identical']
    check_state(status)
    if (receipt['kind']!='two-host-reviewed-ledger-sync-receipt-v1' or receipt['revision_sha256']!=RP
        or receipt['implementation_sha256']!=MP or receipt['plan_file_sha256']!=PLAN_PIN
        or receipt['audit_acceptance_file_sha256']!=ACCEPTANCE_PIN or receipt['host']!='linux10609'
        or set(created)&set(existing) or set(created)|set(existing)!=TARGETS or len(created)+len(existing)!=4
        or receipt['records_file_sha256']!=expected or set(receipt['records_readback'])!=TARGETS
        or receipt['preserved_files_verified']!=46 or receipt['preserved_files_map_sha256']!=canonical(plan['preserve_remote_files_sha256'])
        or receipt['STOP_sha256']!=plan['preserve_remote_files_sha256']['ledger/STOP.json']
        or hashlib.sha256(receipt['STOP_readback'].encode()).hexdigest()!=receipt['STOP_sha256']
        or receipt['original_imports_reverified']!=7 or receipt['saved_new_cases_reverified']!=2
        or type(receipt['accepted_before']) is not int or receipt['accepted_before'] not in (7,8,9)
        or type(receipt['accepted_after']) is not int or receipt['accepted_after']!=9
        or status['accepted_case_ids']!=plan['after_repair_accepted_case_ids'] or status['stop_dispatch'] is not True
        or status['complete'] is not False or status['scientific_aggregate'] is not None or status['research_admitted'] is not False
        or any(receipt[k] is not False for k in ('GPU_or_model_job_started','test_job_started','old_records_changed','research_admitted'))):
        raise ValueError('Sync receipt does not match the reviewed stopped state.')
    for name,text in receipt['records_readback'].items():
        if record_text(text,expected[name])!=plan['proposed_records'][name]['payload']:
            raise ValueError('Published record readback differs from its independent pin.')
    return {'kind':'two-host-ledger-sync-receipt-verification-v1','verified':True,'accepted_cases':9,
        'records_verified':4,'preserved_files_verified':46,'STOP_still_active':True,'GPU_or_model_job_started':False}


def main():
    if len(sys.argv)!=2:raise ValueError('Provide the external metadata-upload SHA-256.')
    plan,acceptance=decode(sys.stdin.buffer.read(LIMIT+1),sys.argv[1])
    print('Rechecking saved cases on CPU and syncing four records; STOP stays active.',file=sys.stderr,flush=True)
    result=run(Path.home(),socket.gethostname(),plan,acceptance)
    print(json.dumps(result,sort_keys=True,indent=2),flush=True)


if __name__=='__main__':main()
