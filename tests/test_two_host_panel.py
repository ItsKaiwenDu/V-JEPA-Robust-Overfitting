"""CPU coordination/failure tests with genuine prior acceptance metadata.

No test creates CUDA evidence or qualifies any new host/clip/arm. Synthetic
ledger events below exercise scheduling semantics only, never model receipts.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,os.environ.get('TWO_HOST_TEST_PACKAGE',str(ROOT)))
spec=importlib.util.spec_from_file_location('two_host_coordination_tests',ROOT/'models/two_host_panel.py')
ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
FIXTURE=json.loads((ROOT/'tests/fixtures/two_host_bootstrap_metadata.json').read_text())


def revision():return json.loads((ROOT/'configs/matched_panel_two_host_v1.json').read_text())


def request(host):
    return {'kind':ops.REQUEST_KIND,'revision_sha256':ops.canonical(revision()),'implementation_sha256':'1'*64,
        'host':host,'case_id':ops.FIRST_BATCH[host],'case_seconds':4800,'kill_grace_seconds':15,
        'max_attempts':1,'one_case_only':True,'additional_allowance_seconds':0,'automatic_queue':False,
        'paid_compute':False,'research_admitted':False}


def registry(root):return ops.Registry(root,revision(),'1'*64,ops.import_metadata(FIXTURE['journal']))


def outcome(cid):
    return {'kind':ops.KIND+'-outcome','case_id':cid,'host':ops.ASSIGNMENTS[cid],
        'revision_sha256':ops.canonical(revision()),'implementation_sha256':'1'*64,'complete':True,
        'numeric_receipt_sha256':'2'*64,'execution_sha256':'3'*64,'research_admitted':False}


def metadata_review(reg,cid):
    """Coordinator event fixture only. Never a production numerical pass."""
    export={'archive_sha256':'4'*64}
    ops.write_record(reg.root/'exports'/(cid+'.json'),export)
    marker={'kind':ops.REVIEW_KIND,'case_id':cid,'host':ops.ASSIGNMENTS[cid],
        'implementation_sha256':'1'*64,'numeric_receipt_sha256':'2'*64,'execution_sha256':'3'*64,
        'revision_sha256':ops.canonical(revision()),'archive_sha256':'4'*64,'reviewed_export':True,
        'outcome_file_sha256':ops.digest(reg.root/'outcomes'/(cid+'.json')),'research_admitted':False}
    ops.write_record(reg.root/'reviews'/(cid+'.json'),marker)


def test_genuine_original_seven_and_scoped_bridge_are_preserved():
    old=FIXTURE['journal'];imports=ops.import_metadata(old)
    assert list(sorted(imports))==ops.CASE_ORDER[:7]
    assert ops.canonical(FIXTURE['reference_execution'])==ops.BASE_EXECUTION_SHA
    assert old['execution_sha256']==ops.BASE_EXECUTION_SHA
    assert FIXTURE['review']['completed_case_count']==7
    assert FIXTURE['bridge']['fresh_workers_reference_exact']==2
    assert FIXTURE['bridge']['other_clips_or_arms_qualified'] is False
    assert FIXTURE['bridge']['parallel_dispatch_authorized'] is False
    assert imports['clip-01-bank-0']['old_pointer']['directory'].endswith('attempt-02')


def test_revision_declares_separate_execution_only_and_thirteen_remaining():
    r=revision();assert ops.validate_revision(r,ops.canonical(r))==r
    assert len(ops.host_queue(r,'linux10609'))==7 and len(ops.host_queue(r,'linux10603'))==6
    assert sum(len(ops.host_queue(r,h)) for h in ops.HOSTS)==13
    for i in range(10):assert ops.ASSIGNMENTS[f'clip-{i:02d}-bank-0']==ops.ASSIGNMENTS[f'clip-{i:02d}-bank-1']
    assert ops.ASSIGNMENTS['clip-03-bank-1']=='linux10609'


@pytest.mark.parametrize('key,value',[
    ('kind','other'),('numeric_plan_sha256','0'*64),('reference_handoff_sha256','0'*64),
    ('case_seconds',4801),('kill_grace_seconds',16),('matrix_seconds',3601),('replay_seconds',901),
    ('parent_seconds',4621),('max_attempts_per_case',2),('automatic_GPU_retries',1),
    ('max_parallel_cases',3),('max_cases_per_host',2),('one_case_per_invocation',False),
    ('automatic_queue',True),('numeric_settings_changed',True),('research_admitted',True),('paid_compute',True),
    ('max_parallel_cases',True),('imported_case_ids',ops.CASE_ORDER[:6]),('bridge_review_sha256','0'*64)])
def test_revision_cannot_extend_or_relabel_the_frozen_scope(key,value):
    r=revision();r[key]=value
    with pytest.raises(ValueError):ops.validate_revision(r,ops.canonical(r))


def test_assignment_cannot_split_a_pair_even_with_a_new_checksum():
    r=revision();r['case_hosts']['clip-03-bank-1']='linux10603'
    with pytest.raises(ValueError):ops.validate_revision(r,ops.canonical(r))


def test_external_revision_pin_prevents_self_consistent_replacement():
    r=revision();pin=ops.canonical(r);r['note']='replacement'
    with pytest.raises(ValueError):ops.validate_revision(r,pin)


@pytest.mark.parametrize('host',ops.HOSTS)
def test_one_host_specific_first_request_is_bounded(host):
    r=request(host);assert ops.validate_request(r,ops.canonical(r),revision(),'1'*64)==(host,ops.FIRST_BATCH[host])


@pytest.mark.parametrize('key,value',[
    ('case_seconds',4801),('kill_grace_seconds',16),('max_attempts',2),('one_case_only',False),
    ('additional_allowance_seconds',1),('automatic_queue',True),('paid_compute',True),('research_admitted',True),
    ('implementation_sha256','0'*64),('revision_sha256','0'*64),('case_id','clip-00-bank-0'),('host','linux10603')])
def test_case_request_cannot_reset_increase_or_switch_its_allowance(key,value):
    r=request('linux10609');r[key]=value
    with pytest.raises(ValueError):ops.validate_request(r,ops.canonical(r),revision(),'1'*64)


def test_original_seven_are_readonly_imports_not_new_computations(tmp_path):
    reg=registry(tmp_path);before=(tmp_path/'imports.json').read_bytes()
    registry(tmp_path)
    assert (tmp_path/'imports.json').read_bytes()==before
    state=reg.status();assert state['accepted_case_ids']==ops.CASE_ORDER[:7]
    assert state['computed_new_case_ids']==[] and state['scientific_aggregate'] is None


def test_import_inventory_and_implementation_cannot_change_on_reconnect(tmp_path):
    reg=registry(tmp_path)
    with pytest.raises(ValueError):ops.Registry(tmp_path,revision(),'9'*64,ops.import_metadata(FIXTURE['journal']))
    changed=ops.import_metadata(FIXTURE['journal']);changed['clip-00-bank-0']['host']='linux10603'
    with pytest.raises(ValueError):ops.Registry(tmp_path,revision(),'1'*64,changed)
    assert len(reg.status()['accepted_case_ids'])==7


def test_two_first_claims_are_independent_and_unknown_claims_never_retry(tmp_path):
    reg=registry(tmp_path)
    for host in ops.HOSTS:
        r=request(host);reg.claim(r,ops.canonical(r))
        with pytest.raises(ValueError,match='already claimed'):reg.claim(r,ops.canonical(r))
    assert len(reg.status()['claimed_case_ids'])==2
    assert len(reg.status()['accepted_case_ids'])==7


def test_crash_between_claim_mkdir_and_record_cannot_reset_allowance(tmp_path):
    reg=registry(tmp_path);r=request('linux10609')
    (tmp_path/'claims'/r['case_id']).mkdir(parents=True)
    with pytest.raises(ValueError,match='already claimed'):reg.claim(r,ops.canonical(r))


def test_numerical_completion_waits_for_independent_review_and_both_first_exports(tmp_path):
    reg=registry(tmp_path)
    for host,cid in ops.FIRST_BATCH.items():
        r=request(host);reg.claim(r,ops.canonical(r));reg.publish(cid,outcome(cid))
    state=reg.status();assert len(state['computed_new_case_ids'])==2 and len(state['accepted_case_ids'])==7
    metadata_review(reg,'clip-03-bank-1');state=reg.status()
    with pytest.raises(ValueError,match='both first-batch'):
        ops.dispatch_gate(revision(),'linux10609','clip-05-bank-0',state['accepted_case_ids'],state['claimed_case_ids'])
    metadata_review(reg,'clip-04-bank-0');state=reg.status()
    assert len(state['accepted_case_ids'])==9 and state['scientific_aggregate'] is None
    for host,cid in [('linux10609','clip-05-bank-0'),('linux10603','clip-04-bank-1')]:
        assert ops.dispatch_gate(revision(),host,cid,state['accepted_case_ids'],state['claimed_case_ids'])==cid


@pytest.mark.parametrize('key,value',[
    ('host','linux10603'),('case_id','clip-04-bank-0'),('revision_sha256','0'*64),
    ('implementation_sha256','0'*64),('numeric_receipt_sha256','0'*64),('execution_sha256','0'*64),
    ('archive_sha256','0'*64),('outcome_file_sha256','0'*64),('research_admitted',True),('reviewed_export',False)])
def test_mixed_independent_review_never_advances_progress(tmp_path,key,value):
    reg=registry(tmp_path);r=request('linux10609');cid=r['case_id'];reg.claim(r,ops.canonical(r));reg.publish(cid,outcome(cid))
    metadata_review(reg,cid);path=tmp_path/'reviews'/(cid+'.json');marker=ops.read_record(path);marker[key]=value
    ops.write_record(path,marker,replace=True)
    with pytest.raises(ValueError):reg.status()


def test_failure_stops_future_dispatch_but_preserves_inflight_peer_result(tmp_path):
    reg=registry(tmp_path)
    for host in ops.HOSTS:
        r=request(host);reg.claim(r,ops.canonical(r))
    reg.fail('clip-03-bank-1','linux10609','worker timeout',4800)
    reg.publish('clip-04-bank-0',outcome('clip-04-bank-0'))
    state=reg.status();assert state['stop_dispatch'] and state['computed_new_case_ids']==['clip-04-bank-0']
    with pytest.raises(ValueError,match='prior failure'):
        ops.dispatch_gate(revision(),'linux10603','clip-04-bank-1',state['accepted_case_ids'],state['claimed_case_ids'],stopped=True)


def test_export_failure_preserves_numeric_outcome_and_opens_global_stop(tmp_path):
    reg=registry(tmp_path);r=request('linux10609');cid=r['case_id'];reg.claim(r,ops.canonical(r));reg.publish(cid,outcome(cid))
    before=(tmp_path/'outcomes'/(cid+'.json')).read_bytes();reg.fail(cid,r['host'],'export failed',2500)
    assert (tmp_path/'outcomes'/(cid+'.json')).read_bytes()==before
    assert (tmp_path/'failures'/(cid+'.json')).exists() and reg.status()['stop_dispatch']


def test_unowned_or_duplicate_numerical_publish_is_rejected(tmp_path):
    reg=registry(tmp_path);cid='clip-03-bank-1'
    with pytest.raises(ValueError):reg.publish(cid,outcome(cid))
    r=request('linux10609');reg.claim(r,ops.canonical(r));reg.publish(cid,outcome(cid))
    with pytest.raises(FileExistsError):reg.publish(cid,outcome(cid))


def test_host_lock_is_exclusive_in_real_child_process_and_released_on_exit(tmp_path):
    reg=registry(tmp_path);code="import fcntl,sys;f=open(sys.argv[1],'a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
    for host in ops.HOSTS:
        with reg.host_lease(host):
            child=subprocess.run([sys.executable,'-c',code,str(tmp_path/'leases'/host)],capture_output=True)
            assert child.returncode!=0 and b'BlockingIOError' in child.stderr
        assert subprocess.run([sys.executable,'-c',code,str(tmp_path/'leases'/host)],capture_output=True).returncode==0


@pytest.mark.parametrize('same_host',[False,True])
def test_real_concurrent_claims_have_no_duplicate_or_lost_events(tmp_path,same_host):
    code='''import importlib.util,json,sys
from pathlib import Path
s=importlib.util.spec_from_file_location('coordinator',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=json.loads(Path(sys.argv[2]).read_text());f=json.loads(Path(sys.argv[3]).read_text());h=sys.argv[5]
reg=m.Registry(sys.argv[4],r,'1'*64,m.import_metadata(f['journal']))
q={'kind':m.REQUEST_KIND,'revision_sha256':m.canonical(r),'implementation_sha256':'1'*64,'host':h,'case_id':m.FIRST_BATCH[h],
'case_seconds':4800,'kill_grace_seconds':15,'max_attempts':1,'one_case_only':True,'additional_allowance_seconds':0,
'automatic_queue':False,'paid_compute':False,'research_admitted':False}
reg.claim(q,m.canonical(q))
'''
    args=[sys.executable,'-c',code,str(ROOT/'models/two_host_panel.py'),str(ROOT/'configs/matched_panel_two_host_v1.json'),
          str(ROOT/'tests/fixtures/two_host_bootstrap_metadata.json'),str(tmp_path)]
    hosts=['linux10609','linux10609' if same_host else 'linux10603']
    children=[subprocess.Popen([*args,h],stdout=subprocess.PIPE,stderr=subprocess.PIPE) for h in hosts]
    results=[(c.communicate(timeout=15),c.returncode) for c in children]
    assert sum(code==0 for _,code in results)==(1 if same_host else 2),results
    assert len(registry(tmp_path).status()['claimed_case_ids'])==(1 if same_host else 2)


@pytest.mark.parametrize('host',ops.HOSTS)
def test_actual_host_gpu_driver_and_foreign_compute_guards(host):
    ops.validate_hardware(host,ops.HOSTS[host],'')
    ops.validate_hardware(host,ops.HOSTS[host],'123\n',allowed_pid=123)
    for identity,active in [(ops.HOSTS[host].replace('580.178.04','other'),''),(ops.HOSTS[host],'123\n456\n')]:
        with pytest.raises(ValueError):ops.validate_hardware(host,identity,active,allowed_pid=123)
    with pytest.raises(ValueError):ops.validate_hardware('other',ops.HOSTS[host],'')


def test_actual_observation_cannot_be_replaced_with_the_reference_host_identity():
    r=revision();manifest={'reference_execution':copy.deepcopy(FIXTURE['reference_execution']),
        'source_sha256':{n:'1'*64 for n in ops.SOURCES}}
    before=copy.deepcopy(manifest);a=ops.execution_identity(r,manifest,'linux10609');b=ops.execution_identity(r,manifest,'linux10603')
    assert manifest==before and a['source_sha256']==b['source_sha256']==FIXTURE['reference_execution']['source_sha256']
    assert a['kind']=='matched-panel-two-host-numeric-execution-v1' and not a['v1_execution_admitted']
    assert a['expected_GPU_identity']!=b['expected_GPU_identity'] and ops.canonical(a)!=ops.canonical(b)
    assert a['plan_sha256']==b['plan_sha256']==ops.NUMERIC_PLAN_SHA


def test_copied_or_other_coordinator_roots_cannot_reset_cases(tmp_path):
    ops.validate_locations(tmp_path/ops.PACKAGE,tmp_path/ops.PACKET,tmp_path/ops.COORDINATOR,tmp_path)
    with pytest.raises(ValueError,match='copied-root'):
        ops.validate_locations(tmp_path/ops.PACKAGE,tmp_path/ops.PACKET,tmp_path/'new-ledger',tmp_path)


@pytest.mark.parametrize('name',['../escape','/absolute','a/../b','a\\b','./x','.'])
def test_escape_paths_reject(tmp_path,name):
    with pytest.raises(ValueError):ops.bound(tmp_path,name)


def test_symlink_path_and_checksum_reject(tmp_path):
    (tmp_path/'link').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError):ops.bound(tmp_path,'link/x')
    ops.write_record(tmp_path/'record.json',{'phase':'incomplete'})
    value=json.loads((tmp_path/'record.json').read_text());value['payload']['phase']='passed'
    (tmp_path/'record.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):ops.read_record(tmp_path/'record.json')


def test_immutable_event_collision_never_overwrites_the_original(tmp_path):
    path=tmp_path/'event.json';ops.write_record(path,{'complete':False});before=path.read_bytes()
    with pytest.raises(FileExistsError):ops.write_record(path,{'complete':True})
    assert path.read_bytes()==before and not list(tmp_path.glob('.*.tmp'))


def test_failed_and_timed_out_children_stop_without_running_a_second_role(tmp_path,monkeypatch):
    import models.matched_diagnostic as primitive
    calls=[]
    def failure(command,path,seconds):
        calls.append(command);path.write_text('failed child\n')
        return {'complete':False,'exit_status':124,'timed_out':True}
    monkeypatch.setattr(primitive,'bounded_child',failure)
    result=ops.logged_child(['no-GPU-command'],tmp_path/'worker.log',.1)
    assert len(calls)==1 and result['timed_out'] and (tmp_path/'worker.log').read_text()=='failed child\n'


def test_real_child_deadline_terminates_child_and_retains_log(tmp_path):
    result=ops.logged_child([sys.executable,'-u','-c',"import time;print('started');time.sleep(10)"],tmp_path/'worker.log',.15)
    assert result['timed_out'] and not result['complete'] and 'started' in (tmp_path/'worker.log').read_text()


@pytest.mark.parametrize('timed_out',[False,True])
def test_numeric_supervisor_stops_after_failed_matrix_and_preserves_partial_export(tmp_path,monkeypatch,timed_out):
    calls=[];r=request('linux10609')
    manifest={'reference_execution':FIXTURE['reference_execution'],'source_sha256':{n:'1'*64 for n in ops.SOURCES}}
    numeric=json.loads((ROOT/'configs/matched_panel_diagnostic_v1.json').read_text()) if (ROOT/'configs/matched_panel_diagnostic_v1.json').exists() else None
    if numeric is None:
        package=Path(os.environ['TWO_HOST_TEST_PACKAGE']);numeric=json.loads((package/'configs/matched_panel_diagnostic_v1.json').read_text())
    else:package=Path(os.environ.get('TWO_HOST_TEST_PACKAGE',str(ROOT)))
    ops.write_record(tmp_path/'case-envelope.json',{'complete':False,'active_role':None})
    def failed(command,path,seconds):
        calls.append((command,seconds));path.write_text('CPU failure injection; no CUDA ran.\n')
        return {'complete':False,'exit_status':124 if timed_out else 1,'timed_out':timed_out}
    monkeypatch.setattr(ops,'logged_child',failed)
    with pytest.raises(RuntimeError,match='partial numerical export retained'):
        ops.run_numeric_case(package,ROOT,tmp_path/'ledger',revision(),manifest,numeric,r,ops.canonical(r),tmp_path,4620)
    assert len(calls)==1 and calls[0][1]<=3600
    session=ops.read_record(tmp_path/'session.json')
    assert not session['complete'] and not session['passed'] and set(session['tasks'])=={'matrix'}
    assert (tmp_path/'numeric-evidence.tar.gz').exists() and not (tmp_path/'receipt.json').exists()
    assert ops.read_record(tmp_path/'case-envelope.json')['active_role'] is None


def test_cli_rejects_changed_core_before_importing_or_starting_any_model(tmp_path):
    packet=tmp_path/'packet';(packet/'models').mkdir(parents=True);(packet/'scripts').mkdir()
    cli=packet/'scripts/check_two_host_panel.py';cli.write_bytes((ROOT/'scripts/check_two_host_panel.py').read_bytes())
    evil=packet/'models/two_host_panel.py';evil.write_text("raise RuntimeError('MUST NOT IMPORT')")
    manifest={'source_sha256':{'models/two_host_panel.py':'0'*64,'scripts/check_two_host_panel.py':ops.digest(cli)}}
    (packet/'IMPLEMENTATION.json').write_text(json.dumps(manifest))
    r=subprocess.run([sys.executable,str(cli),'run','--package',str(tmp_path),'--revision-sha256','0'*64,
        '--implementation-sha256',ops.canonical(manifest)],capture_output=True,text=True)
    assert r.returncode!=0 and 'source changed' in r.stderr and 'MUST NOT IMPORT' not in r.stderr
