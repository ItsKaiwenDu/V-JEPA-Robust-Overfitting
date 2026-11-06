"""Full-model record handoff: local CPU integration, proof consumption and refusal gates."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
from models import predictor_eot8_record_operator_v2 as op
from models import predictor_eot8_record_client_v2 as client
from models import predictor_eot8_record_v2 as records

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'runs/pretrained/diagnostic/predictor-eot8-10603-v1-20261105/cpu-four-worker'
UPSTREAM=ROOT/'external/vjepa2';CONTROL='b'*64
INITIAL_SHA='989665a5a4f8ea90600214be5536b4600c37fc448108a5bd177b3416e5ae0338'


def authorization():
    return {'kind':'predictor-clean-eot8-record-university-exception-v2','approved':True,'control_sha256':'new','host':'linux10603','maximum_new_attempts':1,
        'work_seconds':2400,'worker_seconds':480,'workers':4,'collection_seconds':600,'collection_bytes':3*2**30,
        'write_proof_bytes':5*2**29,'setup_bytes':3*2**30,'capture_bytes':9*2**28,'raw_bytes':1792*2**20,'allocator_bytes':7*2**30,
        'bootstrap_bytes':2*2**20,'paid_compute':False,'reset_old_claims':False,'automatic_retries':0,
        'prior_consumed_control_sha256':op.PRIOR_CONSUMED_CONTROL,'prior_consumed_claim_sha256':op.PRIOR_CONSUMED_CLAIM,
        'user_basis':'Local test fixture only, not a target resource grant.'}


def test_unarmed_refuses_before_claim_folder_or_SSH(tmp_path):
    with pytest.raises(ValueError,match='Unarmed'):
        client.run(tmp_path,popen=lambda *a,**k:(_ for _ in ()).throw(AssertionError('SSH attempted')))
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('field,value',[
 ('kind','predictor-clean-eot8-owned-university-exception-v1'),('host','linux10609'),('maximum_new_attempts',2),
 ('work_seconds',2401),('paid_compute',True),('reset_old_claims',True),('control_sha256','old'),('prior_consumed_claim_sha256','old')])
def test_old_or_expanded_authorization_refused(field,value):
    a=authorization();a[field]=value
    with pytest.raises(ValueError):op.validate_authorization(a,'new')


@pytest.fixture(scope='module')
def four_workers(tmp_path_factory):
    folder=tmp_path_factory.mktemp('record-target-four-cpu')
    code='''import sys,os
os.umask(0o077)
from pathlib import Path
from types import SimpleNamespace
from models.predictor_eot8_record_worker_v2 import run
b=Path(sys.argv[1]);i=int(sys.argv[2]);run(SimpleNamespace(output=b/str(i),index=i,tiny=True,device='cpu',upstream=Path(sys.argv[3]),initial=Path(sys.argv[4]),initial_sha=sys.argv[5],control_sha256=sys.argv[6],reference=b/str(i-1) if i in (1,3) else None,replay_reference=b/str(i-2) if i in (2,3) else None))'''
    assert op.old.digest(OLD/'initial.pt')==INITIAL_SHA
    for i in range(4):
        p=subprocess.run([sys.executable,'-B','-c',code,str(folder),str(i),str(UPSTREAM),str(OLD/'initial.pt'),INITIAL_SHA,CONTROL],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        (folder/('worker-'+str(i)+'.log')).write_bytes(p.stdout)
        assert p.returncode==0,p.stdout.decode()[-5000:]
    return folder


def test_four_real_workers_exact_40_returns_with_reread_backend_receipts(four_workers):
    pids=[];count=0
    for i in range(4):
        folder=four_workers/str(i);w=records.load_worker(folder,expected_control=CONTROL);old=json.loads((OLD/str(i)/'worker.json').read_text())
        assert w['record_worker_revision']=='predictor-eot8-record-worker-v2' and w['local_tiny_analogue'] and not w['backend']['CUDA_qualified']
        assert w['baseline']==old['baseline'];pids.append(w['pid'])
        proof=op.verify_outputs(folder,i,control_sha256=CONTROL,live=False)
        (four_workers/('verified-'+str(i)+'.json')).write_text(json.dumps(proof,sort_keys=True))
        assert proof['pid']==w['pid'] and not proof['CUDA_qualified']
        for n,item in w['artifacts'].items():
            value=torch.load(folder/item['file'],weights_only=True,map_location='cpu')
            prior=torch.load(OLD/str(i)/old['artifacts'][n]['file'],weights_only=True,map_location='cpu')
            assert torch.equal(value,prior);count+=1
    assert count==40 and len(set(pids))==4


def rebind_test_header(folder,mutate):
    """Test-owned copy only: let semantic checks run after valid header hashing."""
    p=folder/records.HEADER;w=json.loads(p.read_text());mutate(w);p.write_bytes(records._json(w))
    m=folder/records.COMPLETE;marker=json.loads(m.read_text())
    marker['header']={'file':records.HEADER,'bytes':p.stat().st_size,'sha256':op.old.digest(p)}
    m.write_bytes(records._json(marker))


@pytest.mark.parametrize('fault',['revision','artifact_bytes','artifact_shape','artifact_device','backend_summary','raw_trace','missing_marker','evidence_changed'])
def test_output_refusal_before_reference_or_subsequent_worker(four_workers,tmp_path,fault):
    folder=tmp_path/'copy';shutil.copytree(four_workers/'0',folder)
    if fault=='revision':rebind_test_header(folder,lambda w:w.update(record_worker_revision='old-worker'))
    elif fault=='artifact_bytes':rebind_test_header(folder,lambda w:w['artifacts']['EOT8-before-input'].update(bytes=1))
    elif fault=='artifact_shape':rebind_test_header(folder,lambda w:w['artifacts']['EOT8-before-input']['source'].update(shape=[1]))
    elif fault=='artifact_device':rebind_test_header(folder,lambda w:w['artifacts']['EOT8-before-input']['source'].update(device='cuda:0'))
    elif fault=='backend_summary':rebind_test_header(folder,lambda w:w['backend'].update(CUDA_qualified=True))
    elif fault=='raw_trace':
        p=folder/'trace.jsonl';old=p.read_bytes();new=old.replace(b'predictor-eot8/call/',b'predictor-xot8/call/');assert len(old)==len(new) and old!=new;p.write_bytes(new)
    elif fault=='missing_marker':(folder/records.COMPLETE).unlink()
    elif fault=='evidence_changed':
        p=folder/records.CALLS;old=p.read_bytes();p.write_bytes(old[:-1]+b'x')
    with pytest.raises((ValueError,FileNotFoundError)):op.verify_outputs(folder,0,control_sha256=CONTROL,live=False)


def test_live_supervisor_never_admits_CPU_or_wrong_control(four_workers):
    with pytest.raises(ValueError):op.check_worker(four_workers/'0',0,control_sha256=CONTROL)
    with pytest.raises(ValueError,match='control binding'):op.verify_outputs(four_workers/'0',0,control_sha256='a'*64,live=False)


def test_live_worker_retains_frozen_model_objective_and_numerical_expression_blocks():
    old=(ROOT/'models/predictor_eot8_owned_worker_v1.py').read_text();new=(ROOT/'models/predictor_eot8_record_worker_v2.py').read_text()
    for begin,end in [("        configure_execution(","        state['baseline']=baseline;record()"),
                      ("                x=clean.detach()","                if args.device=='cuda':torch.cuda.reset_peak_memory_stats()")]:
        assert old[old.index(begin):old.index(end,old.index(begin))] in new
    assert "compare(t,prior[name],BOUNDS[name])" in new
    assert 'gradient,=route.input_gradient(loss,x)' in new
    assert "state['call_rows']=route.rows" not in new


def run_args(folder,*,index=0,reference=None):
    return SimpleNamespace(output=folder,index=index,tiny=True,device='cpu',upstream=UPSTREAM,initial=OLD/'initial.pt',initial_sha=INITIAL_SHA,
        control_sha256=CONTROL,reference=reference,replay_reference=None)


def test_missing_completion_refuses_candidate_before_model_setup(four_workers,tmp_path,monkeypatch):
    from models.predictor_eot8_record_worker_v2 import run
    import models.execution as execution
    reference=tmp_path/'reference';shutil.copytree(four_workers/'0',reference);(reference/records.COMPLETE).unlink()
    monkeypatch.setattr(execution,'configure_execution',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Model setup reached')))
    with pytest.raises(FileNotFoundError):run(run_args(tmp_path/'candidate',index=1,reference=reference))
    state=records.load_worker(tmp_path/'candidate',require_passed=False);assert not state['passed'] and not state['artifacts']


def test_live_worker_integration_retains_primary_backward_failure(tmp_path,monkeypatch):
    from models.predictor_eot8_record_worker_v2 import run
    from models.predictor_eot8_owned_v1 import PredictorRoute
    original=PredictorRoute._admit
    def refuse(self):
        if self._task is not None and self._task.state=='running':self._fail('record-live-integration-refusal')
        return original(self)
    monkeypatch.setattr(PredictorRoute,'_admit',refuse);folder=tmp_path/'failure'
    with pytest.raises(ValueError,match='record-live-integration-refusal'):run(run_args(folder))
    state=records.load_worker(folder,require_passed=False)
    assert not state['passed'] and state['adapter_restored'] and set(state['artifacts'])=={'EOT8-before-loss'}
    scope=state['backward_scopes'][0];assert scope['state']=='failed' and scope['hook_removed'] and scope['graph_id']>=0
    assert state['ownership_refusals'][0]['current_graph_id']==scope['graph_id']
    assert not (folder/records.COMPLETE).exists()


def test_wrong_live_domain_refuses_before_scientific_import(tmp_path):
    program='''import sys
from types import SimpleNamespace
from models.predictor_eot8_record_worker_v2 import run
def deny(event,args):
 if event=='import' and args[0].split('.')[0] in ('torch','numpy'):raise AssertionError('Premature scientific import')
sys.addaudithook(deny)
try:run(SimpleNamespace(index=0,tiny=False,device='cuda',output=sys.argv[1]))
except ValueError as e:assert 'reviewed university host' in str(e)
else:raise AssertionError('Live worker reached')
'''
    p=subprocess.run([sys.executable,'-B','-c',program,str(tmp_path/'none')],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
    assert p.returncode==0,p.stderr.decode();assert not (tmp_path/'none').exists()
