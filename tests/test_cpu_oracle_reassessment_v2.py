"""New decision contract: strict array bounds, authenticated context/stat drift."""
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

from models.cpu_oracle_packet import build
from models.cpu_oracle_reassessment_v2 import (KIND,decision,compare_numeric_receipts,compare_stage_receipts,audit_worker,validate_summary)
from models.attention_fixture import BOOKKEEPING,load
from models.oracle_trace import load_arrays,save_arrays,audit_observation
from models.pilot_bundle import digest
from models.study_artifacts import read_record,write_record
from models.precision_diagnostic import same_report

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def tiny_worker(tmp_path_factory):
    root=tmp_path_factory.mktemp('v2-local-worker')
    manifest,_=build(ROOT,root/'packet',tiny=True);packet=root/'packet/handoff'
    sha=digest(packet/'CPU-ORACLE-HANDOFF.json');folder=root/'worker';folder.mkdir()
    write_record(folder/'authorization.json',{'threads':1,'replay':0,'handoff_sha256':sha,'deadline_unix':time.time()+90})
    command=[sys.executable,'-B',str(packet/'scripts/check_cpu_oracle.py'),'--diagnostic','--local','--worker',
        '--threads','1','--replay','0','--worker-dir',str(folder),'--handoff-sha256',sha,
        '--authorization-sha256',digest(folder/'authorization.json'),'--upstream',str(ROOT/'external/vjepa2')]
    result=subprocess.run(command,capture_output=True,text=True,timeout=95)
    assert result.returncode==0,result.stdout+result.stderr
    values,_=load(packet/manifest['fixture']['folder'],manifest['fixture']['sha256'],case='encoder-1',tiny=True)
    return folder,values,manifest['fixture'],sha


def audit(folder,values,fixture,sha):
    return audit_worker(folder,digest(folder/'worker.json'),values,fixture,tiny=True,handoff_sha=sha)


def test_actual_saved_one_four_thread_statistic_pair_no_longer_requires_exact_statistics():
    source=json.loads((ROOT/'tests/fixtures/cpu_oracle_refusals_v1.json').read_text())
    left,right=source['four_thread_statistic'],source['one_thread_statistic']
    assert same_report(left,right) is False
    result=compare_numeric_receipts(left,right)
    assert result['decisions_match'] and len(result['diagnostic_differences'])==3
    assert left['rtol']==right['rtol']==1e-12 and left['atol']==right['atol']==1e-14


def test_actual_tiny_saved_worker_is_reassessed_with_original_bounds(tiny_worker):
    result=audit(*tiny_worker)
    assert result['capture_integrity_verified'] and result['numerical_values_passed']
    assert result['producer_decisions_match_reassessment'] and result['saved_value_evidence_passed']
    assert result['bounds']=={'rtol':1e-12,'atol':1e-14}
    assert not result['original_parent_failure_waived'] and not result['research_admitted']


def test_changed_statistics_are_retained_without_modifying_original(tiny_worker):
    folder,_,_,_=tiny_worker;left=read_record(folder/'stage-comparisons.json');right=copy.deepcopy(left)
    metric=right['stage_comparisons'][0]['comparisons']['dq']
    metric['max_absolute_error']+=1e-20;metric['relative_l2_error']+=1e-20
    compared=compare_stage_receipts(left,right)
    assert compared['decisions_match'] and not compared['raw_reports_identical']
    assert compared['diagnostic_difference_count']==2
    assert read_record(folder/'stage-comparisons.json')==left


@pytest.mark.parametrize('field,value',[('passed',False),('failed_elements',1),('shape',[1,1]),('rtol',1e-10),('atol',1e-10)])
def test_decisions_shapes_and_bounds_cannot_be_treated_as_statistic_drift(field,value):
    data=json.loads((ROOT/'tests/fixtures/cpu_oracle_refusals_v1.json').read_text())
    left=data['four_thread_statistic'];right=copy.deepcopy(left);right[field]=value
    try:result=compare_numeric_receipts(left,right)
    except ValueError:return
    assert result['decisions_match'] is False


@pytest.mark.parametrize('field,value',[('max_absolute_error',float('nan')),('max_tolerance_ratio',2),('relative_l2_error',-1),('failed_elements',True),('shape',[0])])
def test_invalid_statistics_or_inconsistent_pass_label_refuse(field,value):
    d=json.loads((ROOT/'tests/fixtures/cpu_oracle_refusals_v1.json').read_text())['four_thread_statistic'];d[field]=value
    with pytest.raises(ValueError):decision(d)


def test_numerical_failure_is_not_accepted_as_diagnostic_drift():
    left=json.loads((ROOT/'tests/fixtures/cpu_oracle_refusals_v1.json').read_text())['four_thread_statistic']
    right=copy.deepcopy(left);right.update(passed=False,failed_elements=1,max_tolerance_ratio=2,max_absolute_error=1e-4)
    assert not compare_numeric_receipts(left,right)['decisions_match']
    assert decision(right)['passed'] is False


@pytest.mark.parametrize('defect',['missing_chunk','duplicate_chunk','source_hash','file_bytes','wrong_worker_sha','changed_fixture'])
def test_capture_integrity_refusals_remain(tiny_worker,tmp_path,defect):
    source,values,fixture,sha=tiny_worker;folder=tmp_path/'changed';shutil.copytree(source,folder)
    w=read_record(folder/'worker.json');index=read_record(folder/'trace/index.json')
    if defect=='missing_chunk':index['chunks'].pop()
    elif defect=='duplicate_chunk':index['chunks'].append(copy.deepcopy(index['chunks'][-1]))
    elif defect=='source_hash':index['oracle_file_sha256']='0'*64
    elif defect=='file_bytes':
        with (folder/'trace'/index['chunks'][0]['artifact']['file']).open('ab') as handle:handle.write(b'tamper')
    elif defect=='changed_fixture':fixture={**fixture,'sha256':'0'*64}
    if defect in ('missing_chunk','duplicate_chunk','source_hash'):
        write_record(folder/'trace/index.json',index,replace=True);w['trace_index_sha256']=digest(folder/'trace/index.json')
        write_record(folder/'worker.json',w,replace=True)
    with pytest.raises(ValueError):
        audit_worker(folder,'0'*64 if defect=='wrong_worker_sha' else digest(folder/'worker.json'),values,fixture,tiny=True,handoff_sha=sha)


def change_saved_trace(folder,*,nonfinite=False):
    w=read_record(folder/'worker.json');index=read_record(folder/'trace/index.json');item=index['chunks'][0]
    arrays=load_arrays(folder/'trace',item['artifact'],{n:d['shape'] for n,d in item['artifact']['arrays'].items()})
    arrays['output'][0,0]=float('nan') if nonfinite else arrays['output'][0,0]+.01
    item['artifact']=save_arrays(folder/'trace/modified.npz',arrays)
    write_record(folder/'trace/index.json',index,replace=True);w['trace_index_sha256']=digest(folder/'trace/index.json')
    write_record(folder/'worker.json',w,replace=True)
    return w


@pytest.mark.parametrize('nonfinite',[False,True])
def test_authenticated_bad_array_still_fails_actual_math_check(tiny_worker,tmp_path,nonfinite):
    source,values,fixture,sha=tiny_worker;folder=tmp_path/'changed';shutil.copytree(source,folder)
    change_saved_trace(folder,nonfinite=nonfinite)
    result=audit(folder,values,fixture,sha)
    assert result['capture_integrity_verified'] and not result['numerical_values_passed']
    assert not result['producer_decisions_match_reassessment'] and not result['saved_value_evidence_passed']
    assert result['fresh_stage_checks']['first_captured_mismatch']['stage']=='output'


def test_matching_two_refusals_is_not_a_numeric_pass(tiny_worker,tmp_path):
    source,values,fixture,sha=tiny_worker;folder=tmp_path/'bad';shutil.copytree(source,folder)
    w=change_saved_trace(folder)
    fresh=audit_observation(folder/'trace',w['trace_index_sha256'],{n:values[n].numpy() for n in ('q','k','v','cotangent')})
    write_record(folder/'stage-comparisons.json',fresh,replace=True)
    w['stage_comparisons_sha256']=digest(folder/'stage-comparisons.json')
    w['trace_summary']={k:v for k,v in fresh.items() if k!='stage_comparisons'}
    w['numerical_checks_passed']=False;w['stage']='captured-numerical-refusal'
    write_record(folder/'worker.json',w,replace=True)
    result=audit(folder,values,fixture,sha)
    assert result['producer_decisions_match_reassessment']
    assert not result['numerical_values_passed'] and not result['saved_value_evidence_passed']


def test_final_reference_value_corruption_is_not_ignored(tiny_worker,tmp_path):
    source,values,fixture,sha=tiny_worker;folder=tmp_path/'changed';shutil.copytree(source,folder)
    w=read_record(folder/'worker.json');item=w['passes']['before']['artifact']
    arrays=load_arrays(folder,item,{n:list(values['q'].shape) for n in ('output','dq','dk','dv')})
    arrays['dk'][0,0,0,0]+=.01
    w['passes']['before']['artifact']=save_arrays(folder/'changed-before.npz',arrays)
    w['phase_bitwise_equal']=False
    write_record(folder/'worker.json',w,replace=True)
    result=audit(folder,values,fixture,sha)
    assert not result['saved_value_evidence_passed'] and not result['fresh_final_checks']['before']['dk']['passed']


@pytest.mark.parametrize('flag',['capture_complete','CPU_only'])
def test_incomplete_or_changed_domain_refuses(tiny_worker,tmp_path,flag):
    source,values,fixture,sha=tiny_worker;folder=tmp_path/'changed';shutil.copytree(source,folder)
    w=read_record(folder/'worker.json');w[flag]=False;write_record(folder/'worker.json',w,replace=True)
    with pytest.raises(ValueError):audit(folder,values,fixture,sha)


def test_reassessment_cli_has_no_remote_or_allocation_launch_mode(tmp_path):
    result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/reassess_cpu_oracle_v2.py'),'--launch'],capture_output=True,text=True)
    assert result.returncode==2 and not list(tmp_path.iterdir())


def summary_fixture():
    return {'kind':KIND,'source_review_sha256':'1'*64,'source_response_sha256':'2'*64,
        'worker_reassessments':[{'worker_sha256':'3'*64,'reassessment_sha256':'4'*64,'threads':t,'replay':r,
            'saved_value_evidence_passed':True,'diagnostic_difference_count':0} for t,r in ((4,0),(1,0),(4,1),(1,1))],
        'array_checks_passed':True,'stage_comparisons':2048,'final_reference_comparisons':48,'bounds':dict(BOOKKEEPING),
        'original_parent_accepted':False,'original_parent_refusal':'Stage comparisons do not reaggregate.',
        'original_setup_accepted':False,'original_refusals_waived':False,'launch_authorized':False,
        'control_host_ready':False,'research_admitted':False,'historical_discrepancy_resolved':False,
        'new_university_work_started':False,'sources':{n:'5'*64 for n in ('models/cpu_oracle_reassessment_v2.py','models/oracle_trace.py','models/attention_fixture.py','models/attention_saved_evidence.py')},
        'scope':'Offline test fixture only.'}


@pytest.mark.parametrize('key',['original_parent_accepted','original_setup_accepted','original_refusals_waived',
    'launch_authorized','control_host_ready','research_admitted','historical_discrepancy_resolved','new_university_work_started'])
def test_new_report_cannot_waive_a_refusal_or_grant_resources(key):
    s=summary_fixture();s[key]=True
    with pytest.raises(ValueError):validate_summary(s,'1'*64,'2'*64)


@pytest.mark.parametrize('defect',['coverage','bounds','source','dropped_dependency','reordered_worker','missing_worker',
    'extra_field','numeric_boolean','changed_worker_label','invalid_hash'])
def test_reassessment_header_remains_strict(defect):
    s=summary_fixture()
    if defect=='coverage':s['stage_comparisons']=2047
    elif defect=='bounds':s['bounds']['atol']=1e-10
    elif defect=='source':s['source_review_sha256']='0'*64
    elif defect=='dropped_dependency':s['sources'].pop('models/oracle_trace.py')
    elif defect=='reordered_worker':s['worker_reassessments'].reverse()
    elif defect=='missing_worker':s['worker_reassessments'].pop()
    elif defect=='extra_field':s['override_failure']=True
    elif defect=='numeric_boolean':s['array_checks_passed']=1
    elif defect=='changed_worker_label':s['worker_reassessments'][0]['saved_value_evidence_passed']=False
    elif defect=='invalid_hash':s['worker_reassessments'][0]['worker_sha256']='not a hash'
    with pytest.raises(ValueError):validate_summary(s,'1'*64,'2'*64)


def test_authentic_negative_reassessment_is_not_promoted_to_success():
    s=summary_fixture();s['array_checks_passed']=False;s['worker_reassessments'][0]['saved_value_evidence_passed']=False
    validate_summary(s,'1'*64,'2'*64)
    assert s['array_checks_passed'] is False and not s['original_parent_accepted']
