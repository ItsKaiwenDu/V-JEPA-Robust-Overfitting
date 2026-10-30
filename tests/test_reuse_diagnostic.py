"""CPU-only preparation/rehearsal and independent rejection of corrupted evidence."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from models.cuda_profile import tensor_signature
from models.pilot_bundle import digest
from models.reuse_diagnostic import TASKS,hardware_gate,summarize,validate_launch
from models.reuse_handoff import build,training_input,verify
from models.study_artifacts import read_record,write_record

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def handoff(tmp_path_factory):
    output=tmp_path_factory.mktemp('reuse-handoff')/'packet'
    m,archive=build(ROOT,output)
    return output/'handoff',m,digest(output/'handoff/REUSE-HANDOFF.json')


def plan(sha):
    return {'kind':'projection-reuse-university-launch-v1','authorized':True,'diagnostic_only':True,
        'paid_compute':False,'research_admitted':False,'handoff_sha256':sha,
        'expected_host':'linux10609','expected_gpu':'NVIDIA GeForce RTX 3070',
        'expected_gpu_uuid':'GPU-4e2a7c77-19bd-689f-6c08-60eb774c0d52','expected_driver':'580.178.04',
        'allocator_gib':7,'max_seconds':4800,'worker_seconds':1080,'automatic_retries':0,'attended':True}


def test_handoff_contains_one_original_training_input_and_no_launch_authorization(handoff):
    root,m,sha=handoff
    assert verify(root,sha)==m
    assert m['launch_authorized'] is False and m['packaged_training_clips']==1
    assert [n for n in m['files'] if n.endswith('.pt')]==['input/training-clip.pt']
    pixels=training_input(root,m)
    assert tuple(pixels.shape)==(1,3,16,256,256)
    assert tensor_signature(pixels)['sha256']==m['training_row']['pixel_sha256']


@pytest.mark.parametrize('name',['models/reuse_execution.py','input/training-clip.pt','REUSE-HANDOFF.json'])
def test_changed_handoff_member_is_rejected_before_model_work(handoff,tmp_path,name):
    root,_,sha=handoff;copy=tmp_path/'copy';shutil.copytree(root,copy)
    with (copy/name).open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError):verify(copy,sha)


@pytest.mark.parametrize('key,value',[('authorized',False),('paid_compute',True),('max_seconds',9600),
    ('attended',False),('automatic_retries',1),('allocator_gib',8),('handoff_sha256','wrong')])
def test_launch_proposal_or_changed_resource_bounds_are_not_authorization(key,value):
    value_plan=plan('fixture-sha');value_plan[key]=value
    with pytest.raises(ValueError):validate_launch(value_plan,'fixture-sha')


@pytest.mark.parametrize('defect',['host','uuid','driver','active','memory','multiple'])
def test_read_only_gpu_identity_and_contention_refusals(monkeypatch,defect):
    import models.reuse_diagnostic as d
    launch=plan('fixture-sha');monkeypatch.setattr(d.platform,'node',lambda:'other' if defect=='host' else 'linux10609')
    row='NVIDIA GeForce RTX 3070, GPU-4e2a7c77-19bd-689f-6c08-60eb774c0d52, 8192, 94, 580.178.04\n'
    if defect=='uuid':row=row.replace('GPU-4e2a7c77','GPU-wrong')
    elif defect=='driver':row=row.replace('580.178.04','other')
    elif defect=='memory':row=row.replace(', 94,',', 900,')
    elif defect=='multiple':row=row+row
    def query(argv,**kwargs):return '123, /usr/NX/bin/nxnode.bin, 175\n' if 'query-compute' in ' '.join(argv) and defect=='active' else '' if 'query-compute' in ' '.join(argv) else row
    monkeypatch.setattr(d.subprocess,'check_output',query)
    with pytest.raises(ValueError):hardware_gate(launch)


@pytest.fixture(scope='module')
def rehearsal(handoff,tmp_path_factory):
    root,m,sha=handoff;output=tmp_path_factory.mktemp('reuse-workers')/'results'
    run=subprocess.run([sys.executable,'-B',str(root/'scripts/check_projection_reuse.py'),
        '--diagnostic','--tiny','--device','cpu','--handoff-sha256',sha,
        '--upstream',str(ROOT/'external/vjepa2'),'--output',str(output)],capture_output=True,text=True,timeout=310)
    assert run.returncode==0,run.stdout+run.stderr
    return root,m,sha,output,run.stdout,run.stderr


def test_four_actual_fresh_tiny_workers_have_exact_attack_and_replay(rehearsal):
    root,m,sha,output,_,_=rehearsal
    summary=summarize(output,m,sha,True)
    assert summary['passed'] and summary['four_fresh_workers_complete']
    assert summary['attack_and_scoring_bitwise_equal_between_modes']
    assert summary['fresh_process_tensor_and_science_replay_exact']
    assert not summary['full_parameter_optimizer_recovery_qualified'] and not summary['research_admitted']
    assert summary==read_record(output/'summary.json')
    manifest=json.loads((output/'evidence.tar.gz.manifest.json').read_text())
    assert digest(output/'evidence.tar.gz')==manifest['sha256']


@pytest.mark.parametrize('defect',['missing_worker','worker_hash','oracle_claim','tolerance','score_bank',
    'pixels','draws','GPU_label','seed','delegator','runtime'])
def test_independent_summary_rejects_incomplete_corrupt_or_relabelled_results(rehearsal,tmp_path,defect):
    _,metadata,sha,source,_,_=rehearsal;output=tmp_path/'results';shutil.copytree(source,output)
    folder=output/'workers/reuse-0';path=folder/'worker.json';w=read_record(path)
    if defect=='missing_worker':shutil.rmtree(folder)
    elif defect=='worker_hash':
        w['passed']=False;write_record(path,w,replace=True)
    else:
        if defect=='oracle_claim':w['oracle_checks']['output']['passed']=False
        elif defect=='tolerance':w['contract']['patch_tolerances']['output']['atol']=1.
        elif defect=='score_bank':w['science']['scores'].pop('audit_1')
        elif defect=='draws':w['attack_receipts'][0]['requests']=8
        elif defect=='GPU_label':w['tiny_fixture']=False;w['device']='cuda'
        elif defect=='seed':w['contract']['seeds']['attack'][0]+=1
        elif defect=='delegator':w['native_probe']['delegator']='old-delegator'
        elif defect=='runtime':w['contract']['runtime']['cudnn_tf32']=True
        elif defect=='pixels':
            item=w['artifacts']['attack'];p=folder/item['file']
            tensors=torch.load(p,map_location='cpu',weights_only=True);tensors['adversarial']=torch.ones_like(tensors['adversarial'])
            torch.save(tensors,p);item['sha256']=digest(p)
            item['tensors']={n:tensor_signature(t) for n,t in tensors.items()}
        write_record(path,w,replace=True)
        parent=read_record(output/'session.json')
        next(t for t in parent['tasks'] if (t['mode'],t['replay'])==('reuse',0))['worker_sha256']=digest(path)
        write_record(output/'session.json',parent,replace=True)
    with pytest.raises((ValueError,FileNotFoundError)):summarize(output,metadata,sha,True)


@pytest.mark.parametrize('flags',[[],['--device','cpu'],['--diagnostic','--device','cpu'],
    ['--diagnostic','--tiny','--device','cpu','--max-seconds','0'],
    ['--diagnostic','--tiny','--device','cpu','--worker-seconds','61']])
def test_cli_scope_and_resource_limits_start_no_worker(handoff,tmp_path,flags):
    root,_,sha=handoff;output=tmp_path/'unused'
    run=subprocess.run([sys.executable,'-B',str(root/'scripts/check_projection_reuse.py'),*flags,
        '--handoff-sha256',sha,'--upstream',str(ROOT/'external/vjepa2'),'--output',str(output)],
        capture_output=True,text=True,timeout=15)
    assert run.returncode!=0 and not output.exists(),run.stdout+run.stderr
