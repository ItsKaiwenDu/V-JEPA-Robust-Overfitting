"""CPU-only operator-envelope tests; no simulated neural/CUDA pass receipts."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('guarded_panel_recovery',ROOT/'scripts/guarded_panel_recovery.py')
ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)


@pytest.mark.parametrize('session',[{}, {'complete':False,'passed':False,'elapsed_seconds':0,'tasks':{}},
    {'complete':False,'passed':False,'elapsed_seconds':2000,'tasks':{'matrix':{'complete':True}}}])
def test_unfinalized_session_conservatively_consumes_original_allowance(session):
    assert ops.assess_old_time(session)=={'old_elapsed_known':False,'old_allowance_charged_seconds':4800,
        'old_allowance_remaining_seconds':0,'additional_authorization_required':True}


@pytest.mark.parametrize('elapsed',[True,-1,float('nan'),float('inf'),'0'])
def test_invalid_elapsed_cannot_grant_time(elapsed):
    assert ops.assess_old_time({'complete':False,'passed':False,'error':'failed','elapsed_seconds':elapsed})['old_allowance_remaining_seconds']==0


def test_finalized_failure_charges_elapsed_and_reserve():
    value=ops.assess_old_time({'complete':False,'passed':False,'error':'worker failed','elapsed_seconds':120.5})
    assert value['old_elapsed_known'] and value['old_allowance_charged_seconds']==300.5
    assert value['old_allowance_remaining_seconds']==4499.5


def authorization(proposal):
    return {'kind':'panel-recovery-extra-time-authorization-v1','approved':True,
        'proposal_sha256':ops.canonical(proposal),'case_id':ops.CASE,'extra_case_seconds':4800,
        'fresh_attempt_from_scratch':True,'research_admitted':False,'grant_id':'a'*32}


@pytest.mark.parametrize('field,value',[('approved',False),('approved',1),('case_id','clip-02-bank-0'),
    ('proposal_sha256','0'*64),('extra_case_seconds',4801),('extra_case_seconds',True),
    ('fresh_attempt_from_scratch',False),('research_admitted',True),('grant_id','../escape')])
def test_missing_changed_or_unbounded_authorization_rejects(field,value):
    proposal={'fixture':'operator authorization only'};grant=authorization(proposal);grant[field]=value
    with pytest.raises(ValueError):ops.validate_authorization(proposal,grant)


def test_matching_authorization_is_single_use(tmp_path):
    proposal={'fixture':'metadata only'};grant=authorization(proposal)
    identity=ops.validate_authorization(proposal,grant)
    first=ops.consume(tmp_path,identity,proposal,grant)
    assert ops.read_record(first/'authorization.json')==grant
    with pytest.raises(FileExistsError):ops.consume(tmp_path,identity,proposal,grant)


@pytest.mark.parametrize('change',[None,'changed','extra','symlink'])
def test_partial_evidence_is_pinned_and_never_edited(tmp_path,change):
    p=tmp_path/'case';p.mkdir();(p/'record').write_text('fixture bytes')
    expected={'record':ops.digest(p/'record')}
    if change=='changed':(p/'record').write_text('different')
    if change=='extra':(p/'additional').write_text('new evidence')
    if change=='symlink':
        original=tmp_path/'outside';original.write_text('fixture bytes');(p/'record').unlink();(p/'record').symlink_to(original)
    if change is None:ops.verify_inventory(p,expected)
    else:
        with pytest.raises(ValueError):ops.verify_inventory(p,expected)


def test_process_completion_gets_fresh_status_without_claiming_case_verification(tmp_path):
    d=tmp_path/'operation';d.mkdir()
    state=ops.supervise([sys.executable,'-c','print("CPU process fixture")'],d,5,heartbeat_seconds=.01)
    assert state['state']=='process_finished' and state['exit_status']==0
    assert state['scientific_case_verified'] is False
    assert ops.read_record(d/'status.json')==state
    assert 'CPU process fixture' in (d/'operator.log').read_text()


def test_nonzero_child_is_failure_not_success(tmp_path):
    state=ops.supervise([sys.executable,'-c','raise SystemExit(7)'],tmp_path,5,heartbeat_seconds=.01)
    assert state['state']=='stopped' and state['exit_status']==7 and not state['scientific_case_verified']


@pytest.mark.parametrize('seconds',[True,0,-1,float('nan'),float('inf')])
def test_unbounded_or_invalid_supervision_refuses_before_launch(tmp_path,seconds):
    with pytest.raises(ValueError,match='Finite positive'):
        ops.supervise(['never-launch'],tmp_path,seconds)
    assert not (tmp_path/'status.json').exists()


def test_timeout_terminates_real_process_group_and_descendant(tmp_path):
    marker=tmp_path/'survived'
    grandchild='import time,pathlib;time.sleep(1);pathlib.Path('+repr(str(marker))+').write_text("bad")'
    code='import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",'+repr(grandchild)+']);time.sleep(10)'
    state=ops.supervise([sys.executable,'-c',code],tmp_path,.25,heartbeat_seconds=.01,kill_grace_seconds=.2)
    assert state['state']=='stopped' and state['stop_reason']=='deadline'
    time.sleep(1.1)
    assert not marker.exists()


@pytest.mark.parametrize('number',[signal.SIGTERM,signal.SIGINT,signal.SIGHUP])
def test_signal_interrupt_is_recorded_and_handlers_restored(tmp_path,number):
    before=signal.getsignal(number)
    code=f'import os,signal,time;time.sleep(.1);os.kill({os.getpid()},{int(number)});time.sleep(10)'
    state=ops.supervise([sys.executable,'-c',code],tmp_path,5,heartbeat_seconds=.01,kill_grace_seconds=.2)
    assert state['state']=='stopped' and state['stop_reason']=='signal-'+str(number)
    assert signal.getsignal(number)==before


def test_production_entry_has_no_unapproved_or_random_fixture_mode(tmp_path):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/guarded_panel_recovery.py'),
        '--package',str(tmp_path),'--panel',str(tmp_path),'--proposal',str(tmp_path/'missing'),
        '--authorization',str(tmp_path/'missing-grant')],capture_output=True,timeout=15)
    assert result.returncode!=0
    assert not (tmp_path/'operator-recovery').exists()
