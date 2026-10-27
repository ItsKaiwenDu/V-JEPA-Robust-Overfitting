"""CPU-only stop-resolution/path guards; no fixture qualifies CUDA work."""
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('continuation_tests',ROOT/'scripts/continue_two_host_panel.py')
new=importlib.util.module_from_spec(spec);spec.loader.exec_module(new)
OLD=Path(os.environ.get('CONTINUATION_TEST_ORIGINAL',str(ROOT/'runs/pretrained/diagnostic/two-host-panel-20261026/extracted/two-host-execution-20261026')))
spec=importlib.util.spec_from_file_location('unchanged_operator',OLD/'models/two_host_panel.py')
ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
FIX=json.loads((ROOT/'tests/fixtures/panel_continuation_metadata.json').read_text())


def operation():
    return {'kind':new.KIND,'numeric_plan_changed':False,'reference_revision_sha256':new.RP,'reference_implementation_sha256':new.MP,
        'prior_sync_receipt_sha256':new.PRIOR,'resolved_stop_sha256':new.STOP,'accepted_before':new.ACCEPTED,'targets':new.TARGETS,
        'root_path_mode':'canonical','case_seconds':4800,'kill_grace_seconds':15,'matrix_seconds':3600,'replay_seconds':900,
        'parent_seconds':4620,'max_attempts':1,'additional_allowance_seconds':0,'automatic_GPU_retries':0,
        'automatic_queue':False,'one_case_per_host':True,'paid_compute':False,'research_admitted':False,
        'sources_sha256':{n:'1'*64 for n in new.SOURCES}}


def fixture(tmp_path):
    canonical=tmp_path/'canonical-home';canonical.mkdir();home=tmp_path/'home-alias';home.symlink_to(canonical,target_is_directory=True)
    root=home/new.BASE/'two-host-panel-20261026';root.mkdir(parents=True)
    packet=home/new.BASE/'two-host-execution-20261026/unpacked/two-host-execution-20261026';(packet/'requests').mkdir(parents=True)
    (root/'STOP.json').write_text(FIX['STOP_text'])
    return home,root,packet


def receipt(root,packet,op):
    state=copy.deepcopy(FIX['prior_status']);state['stop_dispatch']=False
    return {'kind':new.KIND+'-activation-receipt','operation_sha256':new.canonical(op),'host':'linux10609',
        'canonical_root':str(root.resolve()),'archived_STOP_text':(root/new.ARCHIVE).read_text(),
        'resolution_record_text':(root/new.RESOLUTION).read_text(),
        'request_record_texts':{h:(packet/'requests'/(cid+'.json')).read_text() for h,cid in new.TARGETS.items()},
        'status_after':state,'preserved_prior_file_hashes':46,'prior_numeric_implementation_unchanged':True,
        'models_started':False,'GPU_retry':False,'automatic_queue':False,'research_admitted':False}


def test_genuine_checkpoint_and_explicit_scope():
    op=operation();new.validate_operation(op,new.canonical(op))
    assert FIX['prior_status']['accepted_case_ids']==new.ACCEPTED and FIX['prior_status']['stop_dispatch']
    assert new.canonical(FIX['reference_revision'])==new.RP
    assert new.TARGETS=={'linux10609':'clip-05-bank-0','linux10603':'clip-04-bank-1'}
    assert new.canonical(FIX['prior_receipt']['status_after'])==new.canonical(FIX['prior_status'])


@pytest.mark.parametrize('key,value',[
    ('root_path_mode','alias'),('numeric_plan_changed',True),('reference_implementation_sha256','0'*64),
    ('resolved_stop_sha256','0'*64),('accepted_before',new.ACCEPTED[:8]),('targets',{'linux10609':'clip-03-bank-1'}),
    ('case_seconds',4801),('kill_grace_seconds',16),('matrix_seconds',3601),('replay_seconds',901),
    ('parent_seconds',4621),('max_attempts',2),('additional_allowance_seconds',1),('automatic_GPU_retries',1),
    ('automatic_queue',True),('one_case_per_host',False),('paid_compute',True),('research_admitted',True)])
def test_revision_cannot_switch_cases_sources_or_extend_resources(key,value):
    op=operation();op[key]=value
    with pytest.raises(ValueError):new.validate_operation(op,new.canonical(op))


def test_external_pin_and_source_inventory_are_required():
    op=operation();pin=new.canonical(op);op['note']='changed'
    with pytest.raises(ValueError):new.validate_operation(op,pin)
    op=operation();op['sources_sha256'].pop(next(iter(op['sources_sha256'])))
    with pytest.raises(ValueError):new.validate_operation(op,new.canonical(op))


@pytest.mark.parametrize('host',new.TARGETS)
def test_new_requests_are_accepted_by_unchanged_runner_and_trace_the_operational_revision(host):
    op=operation();q=new.request(op,host)
    assert ops.validate_request(q,new.canonical(q),FIX['reference_revision'],new.MP)==(host,new.TARGETS[host])
    assert q['operational_revision_sha256']==new.canonical(op) and q['additional_allowance_seconds']==0


def test_resolution_preserves_stop_bytes_and_is_idempotent(tmp_path):
    home,root,packet=fixture(tmp_path);op=operation();before=(root/'STOP.json').read_bytes()
    new.publish_resolution(root,packet,op,ops)
    assert not (root/'STOP.json').exists() and (root/new.ARCHIVE).read_bytes()==before
    files={str(p.relative_to(home)):p.read_bytes() for p in home.rglob('*.json')}
    new.publish_resolution(root,packet,op,ops)
    assert {str(p.relative_to(home)):p.read_bytes() for p in home.rglob('*.json')}==files
    assert new.verify_activation(receipt(root,packet,op),op,new.canonical(op))['verified']


@pytest.mark.parametrize('which',['other-stop','missing-stop','conflicting-request','conflicting-resolution','archive-conflict','archive-symlink'])
def test_unknown_or_conflicting_state_is_not_resolved(tmp_path,which):
    home,root,packet=fixture(tmp_path);op=operation()
    if which=='other-stop':(root/'STOP.json').write_text('new unrelated failure')
    elif which=='missing-stop':(root/'STOP.json').unlink()
    elif which=='conflicting-request':ops.write_record(packet/'requests/clip-05-bank-0.json',{'other':True})
    elif which=='conflicting-resolution':ops.write_record(root/new.RESOLUTION,{'other':True})
    elif which=='archive-conflict':
        p=root/new.ARCHIVE;p.parent.mkdir();p.write_text('different')
    else:
        p=root/'archived-stops';p.symlink_to(packet,target_is_directory=True)
    before={str(p):p.read_bytes() for p in home.rglob('*.json') if p.is_file()}
    with pytest.raises(ValueError):new.publish_resolution(root,packet,op,ops)
    assert {str(p):p.read_bytes() for p in home.rglob('*.json') if p.is_file()}==before


@pytest.mark.parametrize('cut',[0,1,2])
def test_metadata_interruption_keeps_stop_until_all_records_exist_and_resumes_safely(tmp_path,cut):
    _,root,packet=fixture(tmp_path);op=operation();calls=[]
    def interrupted(path,payload):
        if len(calls)==cut:raise OSError('CPU metadata interruption')
        calls.append(path);ops.write_record(path,payload)
    wrapped=SimpleNamespace(bound=ops.bound,read_record=ops.read_record,write_record=interrupted)
    with pytest.raises(OSError):new.publish_resolution(root,packet,op,wrapped)
    assert (root/'STOP.json').exists() and new.digest(root/'STOP.json')==new.STOP
    new.publish_resolution(root,packet,op,ops)
    assert not (root/'STOP.json').exists() and new.digest(root/new.ARCHIVE)==new.STOP


def command_context(home,root,packet,op,monkeypatch):
    state=copy.deepcopy(FIX['prior_status']);state['stop_dispatch']=False;observed=[]
    delegated=SimpleNamespace(read_record=ops.read_record,Registry=lambda *_:SimpleNamespace(status=lambda:state),
        verify_imports=lambda *_:{},hardware_gate=lambda host:observed.append(host),dispatch_gate=ops.dispatch_gate)
    context=(op,None,None,None,delegated,FIX['reference_revision'],{},None,{},home/'package',packet.resolve(),root.resolve())
    monkeypatch.setattr(new,'load',lambda *_:context)
    return observed,state


@pytest.mark.parametrize('host',new.TARGETS)
def test_run_command_keeps_original_sources_and_passes_the_canonical_root(tmp_path,monkeypatch,host):
    home,root,packet=fixture(tmp_path);op=operation();new.publish_resolution(root,packet,op,ops)
    seen,state=command_context(home,root,packet,op,monkeypatch)
    command=new.run_command(tmp_path,new.canonical(op),home,host)
    assert command[command.index('--root')+1]==str(root.resolve())
    assert command[command.index('--implementation-sha256')+1]==new.MP
    assert command[command.index('--request-sha256')+1]==new.canonical(new.request(op,host))
    assert seen==[host] and 'models_started' not in state


def test_a_new_failure_always_blocks_and_is_never_auto_resolved(tmp_path,monkeypatch):
    home,root,packet=fixture(tmp_path);op=operation();new.publish_resolution(root,packet,op,ops)
    seen,_=command_context(home,root,packet,op,monkeypatch);(root/'STOP.json').write_text('new failure')
    with pytest.raises(ValueError):new.run_command(tmp_path,new.canonical(op),home,'linux10609')
    assert seen==[] and (root/'STOP.json').read_text()=='new failure'
    with pytest.raises(ValueError):new.publish_resolution(root,packet,op,ops)


def test_existing_case_claim_cannot_reset_its_allowance(tmp_path,monkeypatch):
    home,root,packet=fixture(tmp_path);op=operation();new.publish_resolution(root,packet,op,ops)
    _,state=command_context(home,root,packet,op,monkeypatch);state['claimed_case_ids'].append('clip-05-bank-0')
    with pytest.raises(ValueError,match='already claimed'):new.run_command(tmp_path,new.canonical(op),home,'linux10609')


@pytest.mark.parametrize('change',['archive','resolution','request','operation','stop','accepted','models','implementation','queue'])
def test_activation_readback_rejects_scope_or_history_drift(tmp_path,change):
    _,root,packet=fixture(tmp_path);op=operation();new.publish_resolution(root,packet,op,ops);r=receipt(root,packet,op)
    if change=='archive':r['archived_STOP_text']='changed'
    elif change=='resolution':r['resolution_record_text']='{}'
    elif change=='request':r['request_record_texts']['linux10609']='{}'
    elif change=='operation':r['operation_sha256']='0'*64
    elif change=='stop':r['status_after']['stop_dispatch']=True
    elif change=='accepted':r['status_after']['accepted_case_ids'].pop()
    elif change=='models':r['models_started']=True
    elif change=='implementation':r['prior_numeric_implementation_unchanged']=False
    else:r['automatic_queue']=True
    with pytest.raises(ValueError):new.verify_activation(r,op,new.canonical(op))


def test_old_exporter_accepts_canonical_root_with_a_real_home_alias(tmp_path):
    # Incomplete CPU export fixture, deliberately without a numerical receipt.
    home,root,packet=fixture(tmp_path);root=root.resolve();(root/'STOP.json').unlink()
    imports=ops.import_metadata(ops.read_record(OLD/'bootstrap-journal.json'))
    registry=ops.Registry(root,FIX['reference_revision'],new.MP,imports);cid='clip-05-bank-0'
    directory=ops.bound(root,'cases/'+cid+'/attempt-01');directory.mkdir(parents=True)
    for name,payload in [('case-claim.json',{'case_id':cid}),('case-request.json',{'CPU_fixture_only':True}),('case-envelope.json',{'complete':False})]:
        ops.write_record(directory/name,payload)
    registry.fail(cid,'linux10609','CPU fixture only',0)
    (directory/'numeric-evidence.tar.gz').write_bytes(b'CPU fixture, no numerical evidence')
    (directory/'numeric-evidence.tar.gz.manifest.json').write_text('{}')
    event=ops.export_revision_case(OLD,registry,directory)
    assert event['archive_path']=='cases/'+cid+'/attempt-01/revision-evidence.tar.gz'
    assert new.digest(root/event['archive_path'])==event['archive_sha256']
    assert not (directory/'receipt.json').exists()
