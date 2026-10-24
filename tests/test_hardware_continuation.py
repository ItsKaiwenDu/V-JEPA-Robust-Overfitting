"""Restart guards use small self-contained fixtures; no private runs or weights."""
import json
from pathlib import Path
import pytest
import scripts.continue_hardware_benchmark as continuation

def valid_plan():
    return dict(restart_epoch=1000,deadline_epoch=3940,prior_running_seconds=4260,
        quoted_all_in_hourly_usd=.61,fresh_quote_confirmed=True,new_hold_usd=0,existing_hold_usd=2.8)

def test_deadline_uses_remaining_49_minutes_instead_of_new_two_hour_window():
    plan=valid_plan()
    assert continuation.plan_gate(plan,1010)==2930
    assert continuation.plan_gate(plan,2400)==1540
    with pytest.raises(ValueError,match='Insufficient'):continuation.plan_gate(plan,2501)

@pytest.mark.parametrize('key,value',[
    ('prior_running_seconds',0),('prior_running_seconds',4200),('prior_running_seconds',7200),
    ('prior_running_seconds',True),('restart_epoch',1011),('deadline_epoch',8200),
    ('deadline_epoch',1001),('quoted_all_in_hourly_usd',0),('quoted_all_in_hourly_usd',True),
    ('quoted_all_in_hourly_usd',float('nan')),('quoted_all_in_hourly_usd',float('inf')),
    ('quoted_all_in_hourly_usd',.71),('fresh_quote_confirmed',False),('new_hold_usd',3),
    ('existing_hold_usd',3)])
def test_unreviewed_or_reset_time_quote_and_reservation_are_rejected(key,value):
    plan=valid_plan();plan[key]=value
    with pytest.raises(ValueError):continuation.plan_gate(plan,1010)

@pytest.fixture
def pinned_inputs(tmp_path,monkeypatch):
    base=tmp_path/'project';base.mkdir()
    source=base/'fixture.txt';source.write_text('declared source\n')
    manifest=base/'handoff-manifest.json'
    manifest.write_text(json.dumps({'archives':{'source':{'files':{'fixture.txt':{'sha256':continuation.sha(source)}}}}}))
    monkeypatch.setattr(continuation,'MANIFEST',continuation.sha(manifest))
    setup=tmp_path/'setup';setup.mkdir();junit=setup/'tests.xml'
    junit.write_text('<testsuites><testsuite tests="509" failures="0" errors="0" skipped="0"/></testsuites>')
    monkeypatch.setattr(continuation,'JUNIT',continuation.sha(junit))
    receipt=tmp_path/'receipt.json'
    receipt.write_text(json.dumps({'manifest_sha256':continuation.MANIFEST,
        'junit_sha256':continuation.JUNIT,'checkpoint_sha256':continuation.CHECKPOINT,
        'original_supervisor_exit_status':124}))
    return base,setup,receipt

def test_source_drift_and_symlinks_are_rejected(pinned_inputs,tmp_path):
    base,_,_=pinned_inputs
    assert continuation.source_gate(base)==1
    source=base/'fixture.txt';original=source.read_bytes();source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Pinned source changed'):continuation.source_gate(base)
    target=tmp_path/'same-bytes.txt';target.write_bytes(original);source.unlink();source.symlink_to(target)
    with pytest.raises(ValueError,match='Pinned source changed'):continuation.source_gate(base)

def test_corrupt_weights_reject_before_loading_model(pinned_inputs,tmp_path):
    base,setup,receipt=pinned_inputs;bad=tmp_path/'bad.pt';bad.write_bytes(b'not the released checkpoint')
    with pytest.raises(ValueError,match='checkpoint differs'):
        continuation.input_gate(base,setup,bad,receipt)

def test_modified_junit_rejects_before_weights_are_opened(pinned_inputs,tmp_path):
    base,setup,receipt=pinned_inputs
    (setup/'tests.xml').write_bytes((setup/'tests.xml').read_bytes()+b' ')
    with pytest.raises(ValueError,match='JUnit changed'):
        continuation.input_gate(base,setup,tmp_path/'unopened.pt',receipt)

def test_changed_receipt_rejects_before_weights_are_opened(pinned_inputs,tmp_path):
    base,setup,receipt=pinned_inputs
    value=json.loads(receipt.read_text());value['checkpoint_sha256']='0'*64;receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='evidence changed'):
        continuation.input_gate(base,setup,tmp_path/'unopened.pt',receipt)
