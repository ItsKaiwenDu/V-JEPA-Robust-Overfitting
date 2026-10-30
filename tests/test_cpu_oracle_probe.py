"""Fresh CPU probe workers, retained reference anomalies and bounded scope."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from models.cpu_oracle_packet import LIMITS,TASKS,build,verify
from models.cpu_oracle_probe import KIND,summarize,validate_plan
from models.pilot_bundle import digest
from models.study_artifacts import read_record,write_record

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def packet(tmp_path_factory):
    folder=tmp_path_factory.mktemp('cpu-oracle-packet')/'packet';m,_=build(ROOT,folder,tiny=True)
    return folder/'handoff',m,digest(folder/'handoff/CPU-ORACLE-HANDOFF.json')


def test_packet_binds_only_failed_shape_and_preserves_all_attempts(packet):
    root,m,sha=packet;assert verify(root,sha)==m
    assert m['CPU_only'] and not m['GPU_work_allowed'] and not m['launch_authorized']
    assert len(m['original_scientific_binding']['source_sha256'])==61 and len(m['prior_consumed_claims'])==8
    assert [n for n in m['files'] if n.endswith('fixture.json')]==['fixtures/encoder-1/fixture.json']
    assert len([n for n in m['files'] if n.endswith('.npy')])==8


def plan(m,sha):
    return {'kind':'cpu-oracle-attended-plan-v1','expected_host':'linux10603','handoff_sha256':sha,'authorized':True,'attended':True,
        'CPU_only':True,'GPU_work_allowed':False,'paid_compute':False,'research_admitted':False,'old_refusals_waived':False,
        'prior_consumed_claims':m['prior_consumed_claims'],'tasks':[list(t) for t in TASKS],**LIMITS}


@pytest.mark.parametrize('key,value',[('GPU_work_allowed',True),('CPU_only',False),('max_seconds',1200),('worker_seconds',180),
    ('calls_per_worker',4),('automatic_retries',1),('stop_after_rejected_worker',False),('prior_consumed_claims',{}),
    ('authorized',False),('paid_compute',True),('kind','fixed-attention-launch-v1'),('tasks',[[4,0]])])
def test_previous_gpu_plan_or_wider_cpu_scope_refuses(packet,key,value):
    _,m,sha=packet;p=plan(m,sha);p[key]=value
    with pytest.raises(ValueError):validate_plan(p,sha,m['prior_consumed_claims'])


@pytest.fixture(scope='module')
def rehearsal(packet,tmp_path_factory):
    root,_,sha=packet;out=tmp_path_factory.mktemp('cpu-oracle-workers')/'results'
    child=subprocess.run([sys.executable,'-B',str(root/'scripts/check_cpu_oracle.py'),'--diagnostic','--local','--handoff-sha256',sha,
                         '--upstream',str(ROOT/'external/vjepa2'),'--output',str(out)],capture_output=True,text=True,timeout=610)
    assert child.returncode==0,child.stdout+child.stderr+'\n'+'\n'.join(p.read_text() for p in out.rglob('worker.log'))
    return root,out,sha


def test_four_fresh_cpu_workers_capture_before_observed_after_without_cuda(rehearsal):
    root,out,sha=rehearsal;report=summarize(root,out,sha)
    assert report==read_record(out/'summary.json') and report['numerical_checks_passed'] and report['planned_tasks_completed']
    assert report['CPU_only'] and not report['GPU_work'] and not report['root_cause_established']
    assert not report['no_anomaly_proves_host_reliable'] and not report['old_refusals_waived']
    assert len(report['workers'])==4
    for i in range(4):
        w=read_record(out/f'workers/{i}/worker.json');assert w['capture_complete'] and not w['context_before']['CUDA_initialized']
        assert set(w['passes'])=={'before','observed','after'} and (out/f'workers/{i}/trace/converted.npz').is_file()


def test_numerical_refusal_keeps_all_three_passes_and_original_trace(packet,tmp_path):
    root,_,sha=packet;out=tmp_path/'refusal';folder=out/'workers/0';folder.mkdir(parents=True)
    write_record(folder/'authorization.json',{'threads':4,'replay':0,'handoff_sha256':sha,'deadline_unix':time.time()+90})
    command=['--diagnostic','--local','--worker','--threads','4','--replay','0','--worker-dir',str(folder),
             '--authorization-sha256',digest(folder/'authorization.json'),'--handoff-sha256',sha,'--upstream',str(ROOT/'external/vjepa2')]
    code='''import sys,os
sys.path.insert(0,sys.argv[1]);os.environ['CUDA_VISIBLE_DEVICES']=''
import models.cpu_oracle_probe as probe
original=probe.fp64_oracle;calls=0
def injected(*args,**kwargs):
 global calls
 result=original(*args,**kwargs);calls+=1
 if calls==1:result['output'].add_(1e-7)
 return result
probe.fp64_oracle=injected
sys.argv=[sys.argv[1]+'/scripts/check_cpu_oracle.py']+sys.argv[2:]
from scripts.check_cpu_oracle import main
main()
'''
    child=subprocess.run([sys.executable,'-B','-c',code,str(root),*command],capture_output=True,text=True,timeout=100)
    assert child.returncode==1,child.stdout+child.stderr
    w=read_record(folder/'worker.json');assert w['capture_complete'] and not w['numerical_checks_passed']
    assert all((folder/(p+'.npz')).is_file() for p in ('before','observed','after'))
    write_record(out/'session.json',{'kind':KIND,'handoff_sha256':sha,'CPU_only':True,'GPU_work_allowed':False,
        'local_rehearsal':True,'host':w['context_before']['host'],'started_unix':1,'deadline_unix':601,
        'tasks':[{'threads':4,'replay':0,'worker_sha256':digest(folder/'worker.json')}]})
    report=summarize(root,out,sha)
    assert report['capture_integrity_verified'] and not report['planned_tasks_completed'] and not report['numerical_checks_passed']
    assert not report['workers'][0]['checks']['before']['output']['passed']


@pytest.mark.parametrize('defect',['cuda','clock','thread','source','phase','summary_label','history'])
def test_reaggregation_rejects_changed_cpu_domain_capture_or_history(rehearsal,tmp_path,defect):
    root,source,sha=rehearsal;out=tmp_path/'changed';shutil.copytree(source,out)
    path=out/'workers/0/worker.json';w=read_record(path);s=read_record(out/'session.json')
    if defect=='cuda':w['context_after']['CUDA_initialized']=True
    elif defect=='clock':w['wall_seconds']=91
    elif defect=='thread':w['context_before']['threads']=8
    elif defect=='source':w['handoff_sha256']='0'*64
    elif defect=='phase':w['passes'].pop('after')
    elif defect=='summary_label':w['numerical_checks_passed']=False
    elif defect=='history':s['handoff_sha256']='0'*64
    if defect!='history':
        write_record(path,w,replace=True);s['tasks'][0]['worker_sha256']=digest(path)
    write_record(out/'session.json',s,replace=True)
    with pytest.raises((ValueError,KeyError)):summarize(root,out,sha)


def test_cli_requires_cpu_diagnostic_opt_in_and_never_accepts_a_device_switch(packet,tmp_path):
    root,_,sha=packet;command=[sys.executable,'-B',str(root/'scripts/check_cpu_oracle.py'),'--handoff-sha256',sha,
      '--upstream',str(ROOT/'external/vjepa2'),'--output',str(tmp_path/'refused')]
    assert subprocess.run(command,capture_output=True).returncode==2
    assert subprocess.run(command+['--diagnostic','--local','--device','cuda'],capture_output=True).returncode==2
    assert not (tmp_path/'refused').exists()
