"""Actual fresh CPU loading/replay, early failure snapshots and strict review."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest
import torch

from models.attention_fixture import INPUTS,load
from models.cuda_profile import tensor_signature
from models.fixed_attention_diagnostic import summarize,validate_launch
from models.fixed_attention_packet import HOSTS,LIMITS,build,verify
from models.pilot_bundle import digest
from models.study_artifacts import read_record,write_record
from test_projection_reuse import restore_process_settings

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def packet(tmp_path_factory):
    path=tmp_path_factory.mktemp('fixed-packet')/'packet';metadata,_=build(ROOT,path,tiny=True)
    return path/'handoff',metadata,digest(path/'handoff/FIXED-ATTENTION-HANDOFF.json')


def test_new_packet_preserves_science_and_both_consumed_attempts(packet):
    root,m,sha=packet;actual,_=verify(root,sha)
    assert actual==m and not m['launch_authorized'] and not m['runtime_generation_allowed']
    assert len(m['original_scientific_binding']['source_sha256'])==61
    assert len([n for n in m['files'] if n.endswith('.npy')])==24
    assert set(m['prior_consumed_attempts'])=={'linux10609','linux10603'}
    assert HOSTS['linux10609']['role']=='encoder-0' and HOSTS['linux10603']['role']=='encoder-1'


@pytest.mark.parametrize('member',['models/fixed_attention_diagnostic.py','fixtures/encoder-1/q.npy','fixtures/encoder-0/fixture.json'])
def test_changed_handoff_source_or_input_refuses(packet,tmp_path,member):
    source,_,sha=packet;target=tmp_path/'changed';shutil.copytree(source,target)
    with (target/member).open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError):verify(target,sha)


def plan(metadata,sha,host='linux10609'):
    return {'kind':'fixed-attention-launch-v1','authorized':True,'attended':True,'paid_compute':False,
        'research_admitted':False,'diagnostic_only':True,'fixed_inputs':True,'generator_replay_qualified':False,
        'handoff_sha256':sha,'expected_host':host,'expected_gpu_uuid':HOSTS[host]['uuid'],'expected_gpu':'NVIDIA GeForce RTX 3070',
        'expected_driver':'580.178.04','role':HOSTS[host]['role'],'prior_consumed_attempts':metadata['prior_consumed_attempts'],**LIMITS}


@pytest.mark.parametrize('key,value',[('kind','attention-component-launch-v1'),('prior_consumed_attempts',{}),('fixed_inputs',False),
    ('role','encoder-1'),('max_seconds',2400),('worker_seconds',480),('allocator_gib',8),('automatic_retries',1),
    ('authorized',False),('paid_compute',True),('generator_replay_qualified',True)])
def test_old_plan_relabel_time_reset_or_wrong_host_is_not_a_new_fixed_plan(packet,key,value):
    _,m,sha=packet;p=plan(m,sha);p[key]=value
    with pytest.raises(ValueError):validate_launch(p,sha,'encoder-0',m['prior_consumed_attempts'])


@pytest.fixture(scope='module')
def rehearsal(packet,tmp_path_factory):
    root,_,sha=packet;out=tmp_path_factory.mktemp('fixed-workers')/'results'
    child=subprocess.run([sys.executable,'-B',str(root/'scripts/check_fixed_attention.py'),'--diagnostic','--tiny','--device','cpu',
        '--role','all','--handoff-sha256',sha,'--upstream',str(ROOT/'external/vjepa2'),'--output',str(out)],capture_output=True,text=True,timeout=310)
    assert child.returncode==0,child.stdout+child.stderr+'\n'+'\n'.join(p.read_text() for p in out.rglob('worker.log'))
    return root,out,sha


def test_six_fresh_workers_load_exact_fixed_bytes_and_math_vjps(rehearsal):
    root,out,sha=rehearsal;s=summarize(root,out,sha,'all',True)
    assert s==read_record(out/'summary.json') and s['fixed_input_bytes_and_replay_verified']
    assert not s['generator_replay_qualified'] and not s['old_refusals_waived'] and not s['CUDA_backend_comparison_qualified']
    assert len(read_record(out/'session.json')['tasks'])==6
    for path in out.glob('workers/*/worker.json'):
        w=read_record(path)
        assert w['evidence']['inputs']['tensor_signatures']==w['evidence']['device_inputs']['tensor_signatures']==w['common']['operand_signatures']


def test_saved_input_review_is_independent_of_remote_path_and_float_regeneration(rehearsal,tmp_path,monkeypatch):
    root,source,sha=rehearsal;out=tmp_path/'results';shutil.copytree(source,out)
    s=read_record(out/'session.json');s['upstream']='/different/remote/check-out';write_record(out/'session.json',s,replace=True)
    def forbidden(*args,**kwargs):raise AssertionError('No regenerated operands are allowed in immutable-input review.')
    monkeypatch.setattr('models.attention_component.operands',forbidden)
    assert summarize(root,out,sha,'all',True,upstream=ROOT/'external/vjepa2')==read_record(out/'summary.json')


@pytest.mark.parametrize('failure',['numerical','reference'])
def test_rejected_worker_retains_inputs_device_inputs_and_untimed_values(rehearsal,tmp_path,failure):
    root,source,sha=rehearsal;folder=tmp_path/'workers'/'encoder-0-math-1';folder.mkdir(parents=True)
    ref=None
    if failure=='reference':
        other=folder.parent/'encoder-0-math-0';shutil.copytree(source/'workers/encoder-0-math-0',other)
        previous=read_record(other/'worker.json');previous['common']['operand_signatures']['q']['sha256']='0'*64
        write_record(other/'worker.json',previous,replace=True)
        ref={'worker':other.name,'worker_sha256':digest(other/'worker.json'),'values_sha256':previous['artifact']['sha256']}
    write_record(folder/'authorization.json',{'case':'encoder-0','arm':'math','replay':1,'handoff_sha256':sha,'deadline_unix':time.time()+30,'reference':ref})
    args=['--diagnostic','--tiny','--device','cpu','--role','all','--worker','--case','encoder-0','--arm','math','--replay','1',
          '--worker-dir',str(folder),'--authorization-sha256',digest(folder/'authorization.json'),'--handoff-sha256',sha,'--upstream',str(ROOT/'external/vjepa2')]
    # Explicit CPU-only fault injection; no test hooks are added to production.
    script='''import sys
sys.path.insert(0,sys.argv[1])
import models.fixed_attention_diagnostic as diagnostic
if sys.argv[2]=='numerical':
 original=diagnostic.iteration
 def altered(*args,**kwargs):
  values,timing=original(*args,**kwargs);values['dq']=values['dq']+.1;return values,timing
 diagnostic.iteration=altered
sys.argv=[sys.argv[1]+'/scripts/check_fixed_attention.py']+sys.argv[3:]
from scripts.check_fixed_attention import main
main()
'''
    child=subprocess.run([sys.executable,'-B','-c',script,str(root),failure,*args],capture_output=True,text=True,timeout=40)
    assert child.returncode!=0,child.stdout+child.stderr
    w=read_record(folder/'worker.json');assert not w['passed'] and not w['complete'] and w['timings']==[]
    assert 'oracle rejected' in w['error'] if failure=='numerical' else 'incomparable' in w['error']
    assert set(w['evidence'])=={'inputs','device_inputs'} and (folder/'values.pt').is_file()
    assert all((folder/name).is_file() for name in ('inputs.pt','device-inputs.pt','values.pt'))
    for item in [*w['evidence'].values(),w['artifact']]:assert digest(folder/item['file'])==item['sha256']


@pytest.mark.parametrize('defect',['coverage','fixture_id','input_signature','device_stride','tolerance','runtime','backend','phase',
    'replay','clock','memory','timing','oracle_tensor'])
def test_independent_review_refuses_resigned_incomplete_or_changed_evidence(rehearsal,tmp_path,defect):
    root,source,sha=rehearsal;out=tmp_path/'changed';shutil.copytree(source,out)
    path=out/'workers/predictor-math-1/worker.json';w=read_record(path);s=read_record(out/'session.json')
    if defect=='coverage':s['tasks'].pop()
    elif defect=='fixture_id':w['common']['fixture']['sha256']='0'*64
    elif defect=='input_signature':w['evidence']['inputs']['tensor_signatures']['q']['sha256']='0'*64
    elif defect=='device_stride':w['device_input_strides']['v'][-2]+=1
    elif defect=='tolerance':w['common']['tolerances']['dq']['atol']=1
    elif defect=='runtime':w['common']['runtime']['matmul_tf32']=True
    elif defect=='backend':w['backend_profile_events']=[]
    elif defect=='phase':w['complete']=False
    elif defect=='replay':w['replay_signatures'].pop()
    elif defect=='clock':w['wall_seconds']=31
    elif defect=='memory':w['peak_cuda_allocated_bytes']=1
    elif defect=='timing':w['timings'][0]['total_seconds']+=1
    elif defect=='oracle_tensor':
        values=torch.load(path.parent/'values.pt',weights_only=True);values['oracle_output'].add_(.1);torch.save(values,path.parent/'values.pt')
        w['artifact']['sha256']=digest(path.parent/'values.pt');w['artifact']['tensor_signatures']={n:tensor_signature(t) for n,t in values.items()}
    if defect!='coverage':
        write_record(path,w,replace=True);s['tasks'][1]['worker_sha256']=digest(path)
    write_record(out/'session.json',s,replace=True)
    with pytest.raises((ValueError,KeyError)):summarize(root,out,sha,'all',True)
