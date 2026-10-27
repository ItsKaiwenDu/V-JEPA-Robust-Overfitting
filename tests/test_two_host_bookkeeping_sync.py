"""CPU metadata publication guards; these tests create no CUDA evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reviewed_sync',ROOT/'scripts/sync_two_host_bookkeeping.py')
sync=importlib.util.module_from_spec(spec);spec.loader.exec_module(sync)
RAW=ROOT/'runs/pretrained/diagnostic/two-host-export-incident-20261026'
DOWNLOAD=Path((RAW/'latest-download.txt').read_text().strip())
FROZEN=ROOT/'runs/pretrained/diagnostic/two-host-panel-20261026/extracted/two-host-execution-20261026'
spec=importlib.util.spec_from_file_location('completed_operator',FROZEN/'models/two_host_panel.py')
ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)


def bundle():
    return {'kind':'two-host-reviewed-ledger-sync-upload-v1',
        'plan_text':(DOWNLOAD/'ledger-repair-plan.json').read_text(),
        'acceptance_text':(DOWNLOAD/'first-pair-review-acceptance.json').read_text(),
        'audit_report_text':(ROOT/'report/checks/2026-10-26-two-host-first-pair-verification.json').read_text()}


def encode(value):return (json.dumps(value,sort_keys=True)+'\n').encode()


def plan():
    body=encode(bundle());return sync.decode(body,hashlib.sha256(body).hexdigest())[0]


def fixture(root):
    root.mkdir(exist_ok=True)
    # These two inert files stand for immutable records; no numerical pass is generated.
    stop=root/'STOP.json';stop.write_text('CPU publication fixture: stopped\n')
    old=root/'old-evidence.json';old.write_text('CPU publication fixture: preserve\n')
    paths={'stop':stop,'old':old};pins={n:sync.digest(p) for n,p in paths.items()}
    return plan()['proposed_records'],(paths,pins)


def test_genuine_independent_plan_allows_only_four_records_and_retains_stop():
    asked=plan()
    assert set(asked['proposed_records'])==sync.TARGETS
    assert len(asked['preserve_remote_files_sha256'])==46
    assert asked['stop_dispatch_remains'] and not asked['model_or_test_execution'] and not asked['GPU_retry']
    assert set(sync.source_paths(Path('/home/kdu')))==set(asked['preserve_remote_files_sha256'])


@pytest.mark.parametrize('change',['upload-pin','plan','acceptance','audit','extra-key','kind','oversized'])
def test_changed_upload_or_review_cannot_authorize_a_different_mutation(change):
    value=bundle()
    if change=='plan':value['plan_text']=value['plan_text'].replace('two-host-export-bookkeeping','other-bookkeeping')
    elif change=='acceptance':value['acceptance_text']=value['acceptance_text'].replace('"STOP_preserved": true','"STOP_preserved": false')
    elif change=='audit':value['audit_report_text']=value['audit_report_text'].replace('"verified": true','"verified": false')
    elif change=='extra-key':value['run_model']=True
    elif change=='kind':value['kind']='run-model'
    elif change=='oversized':value['extra']='x'*sync.LIMIT
    body=encode(value);pin=hashlib.sha256(body).hexdigest()
    if change=='upload-pin':pin='0'*64
    with pytest.raises(ValueError):sync.decode(body,pin)


def test_publication_adds_exactly_four_files_and_is_idempotent(tmp_path):
    proposed,preserved=fixture(tmp_path);before={n:p.read_bytes() for n,p in preserved[0].items()}
    created,existing=sync.publish(tmp_path,proposed,preserved,ops)
    assert set(created)==sync.TARGETS and existing==[]
    again,existing=sync.publish(tmp_path,proposed,preserved,ops)
    assert again==[] and set(existing)==sync.TARGETS
    assert {n:p.read_bytes() for n,p in preserved[0].items()}==before
    assert {str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*') if p.is_file()}==sync.TARGETS|{'STOP.json','old-evidence.json'}


@pytest.mark.parametrize('cut',[1,2,3])
def test_interrupted_publication_only_fills_missing_identical_records_on_repeat(tmp_path,cut):
    proposed,preserved=fixture(tmp_path);calls=[]
    def interrupted(path,payload):
        if len(calls)==cut:raise OSError('simulated interrupted metadata write')
        calls.append(path);ops.write_record(path,payload)
    with pytest.raises(OSError):sync.publish(tmp_path,proposed,preserved,SimpleNamespace(write_record=interrupted))
    assert len(calls)==cut
    created,existing=sync.publish(tmp_path,proposed,preserved,ops)
    assert len(created)==4-cut and len(existing)==cut
    assert (tmp_path/'STOP.json').read_text()=='CPU publication fixture: stopped\n'


@pytest.mark.parametrize('target',sorted(sync.TARGETS))
def test_any_conflicting_existing_record_rejects_before_new_records_are_added(tmp_path,target):
    proposed,preserved=fixture(tmp_path);path=tmp_path/target;path.parent.mkdir(parents=True);path.write_text('conflict\n')
    with pytest.raises(ValueError,match='conflicts'):sync.publish(tmp_path,proposed,preserved,ops)
    assert {str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*.json')}=={target,'STOP.json','old-evidence.json'}
    assert path.read_text()=='conflict\n'


@pytest.mark.parametrize('change',['missing','changed','symlink'])
def test_changed_stop_or_preserved_record_prevents_all_publication(tmp_path,change):
    proposed,preserved=fixture(tmp_path);stop=tmp_path/'STOP.json'
    if change=='missing':stop.unlink()
    elif change=='changed':stop.write_text('cleared\n')
    else:stop.unlink();stop.symlink_to(tmp_path/'old-evidence.json')
    with pytest.raises(ValueError):sync.publish(tmp_path,proposed,preserved,ops)
    assert not (tmp_path/'exports').exists() and not (tmp_path/'reviews').exists()


@pytest.mark.parametrize('change',['escape','missing-target','added-target','changed-payload'])
def test_target_inventory_and_record_bytes_are_fixed_before_writing(tmp_path,change):
    proposed,preserved=fixture(tmp_path);proposed=copy.deepcopy(proposed);name=next(iter(proposed))
    if change=='escape':proposed['../escape.json']=proposed.pop(name)
    elif change=='missing-target':proposed.pop(name)
    elif change=='added-target':proposed['STOP.json']=proposed[name]
    else:proposed[name]['payload']['case_id']='other'
    with pytest.raises(ValueError):sync.publish(tmp_path,proposed,preserved,ops)
    assert not (tmp_path/'exports').exists() and not (tmp_path/'reviews').exists()


def test_destination_directory_symlink_cannot_redirect_the_sync(tmp_path):
    proposed,preserved=fixture(tmp_path);outside=tmp_path/'elsewhere';outside.mkdir();(tmp_path/'exports').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError):sync.publish(tmp_path,proposed,preserved,ops)
    assert not list(outside.iterdir())


def test_home_alias_has_the_same_publication_destinations(tmp_path):
    real=tmp_path/'canonical';proposed,preserved=fixture(real);alias=tmp_path/'home';alias.symlink_to(real,target_is_directory=True)
    created,_=sync.publish(alias,proposed,preserved,ops)
    assert len(created)==4 and all((real/n).is_file() for n in sync.TARGETS)


def test_production_entry_rejects_another_host_before_import_or_any_write(tmp_path,monkeypatch):
    monkeypatch.setattr(sync,'load_frozen',lambda *_:pytest.fail('must reject before import'))
    with pytest.raises(ValueError,match='pinned host'):sync.run(tmp_path,'linux10603',plan(),{})
    assert not list(tmp_path.iterdir())


def receipt_fixture():
    asked=plan();report=json.loads(bundle()['audit_report_text'])
    return {'kind':'two-host-reviewed-ledger-sync-receipt-v1','revision_sha256':sync.RP,'implementation_sha256':sync.MP,
        'plan_file_sha256':sync.PLAN_PIN,'audit_acceptance_file_sha256':sync.ACCEPTANCE_PIN,'host':'linux10609',
        'created':sorted(sync.TARGETS),'preexisting_identical':[],
        'records_file_sha256':{n:v['file_sha256'] for n,v in asked['proposed_records'].items()},
        'records_readback':{n:sync.record_bytes(v['payload']).decode() for n,v in asked['proposed_records'].items()},
        'STOP_readback':(DOWNLOAD/'downloaded/ledger/STOP.json').read_text(),
        'STOP_sha256':asked['preserve_remote_files_sha256']['ledger/STOP.json'],
        'preserved_files_verified':46,'preserved_files_map_sha256':sync.canonical(asked['preserve_remote_files_sha256']),
        'original_imports_reverified':7,'saved_new_cases_reverified':2,'accepted_before':7,'accepted_after':9,
        'status_after':copy.deepcopy(report['local_coordinator_reconstruction']['status']),
        'GPU_or_model_job_started':False,'test_job_started':False,'old_records_changed':False,'research_admitted':False}


def test_receipt_byte_readback_requires_exact_pinned_reviews_and_retained_stop():
    result=sync.verify_receipt(receipt_fixture(),plan())
    assert result['verified'] and result['accepted_cases']==9 and result['STOP_still_active']


@pytest.mark.parametrize('change',['plan-pin','implementation','host','stop-hash','stop-bytes','accepted-after','accepted-before',
    'GPU-job','test-job','old-record','record-bytes','record-hash','added-record','duplicate-created','overlapping-records',
    'stop-cleared','extra-computation','missing-claim','numerical-failure','complete','aggregate','research'])
def test_receipt_rejects_changed_scope_file_bytes_and_unexpected_coordinator_state(change):
    receipt=receipt_fixture();name=next(iter(sync.TARGETS));state=receipt['status_after']
    if change=='plan-pin':receipt['plan_file_sha256']='0'*64
    elif change=='implementation':receipt['implementation_sha256']='0'*64
    elif change=='host':receipt['host']='linux10603'
    elif change=='stop-hash':receipt['STOP_sha256']='0'*64
    elif change=='stop-bytes':receipt['STOP_readback']='changed'
    elif change=='accepted-after':receipt['accepted_after']=10
    elif change=='accepted-before':receipt['accepted_before']=6
    elif change=='GPU-job':receipt['GPU_or_model_job_started']=True
    elif change=='test-job':receipt['test_job_started']=True
    elif change=='old-record':receipt['old_records_changed']=True
    elif change=='record-bytes':receipt['records_readback'][name]='changed'
    elif change=='record-hash':receipt['records_file_sha256'][name]='0'*64
    elif change=='added-record':receipt['records_readback']['STOP.json']='clear'
    elif change=='duplicate-created':receipt['created'].append(name)
    elif change=='overlapping-records':receipt['preexisting_identical'].append(name)
    elif change=='stop-cleared':state['stop_dispatch']=False
    elif change=='extra-computation':state['computed_new_case_ids'].append('clip-04-bank-1')
    elif change=='missing-claim':state['claimed_case_ids'].pop()
    elif change=='numerical-failure':state['failed_case_ids'].append('clip-04-bank-0')
    elif change=='complete':state['complete']=True
    elif change=='aggregate':state['scientific_aggregate']={'partial':1}
    else:state['research_admitted']=True
    with pytest.raises(ValueError):sync.verify_receipt(receipt,plan())
