"""Live-handoff integration on local CPU plus receipt/domain refusal tests."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
from models import predictor_eot8_owned_operator_v1 as op
from models import predictor_eot8_owned_client_v1 as client
from models.predictor_eot8_owned_receipt_v1 import verify_ownership

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'external/vjepa2'
OLD=ROOT/'runs/pretrained/diagnostic/predictor-eot8-10603-v1-20261105/cpu-four-worker'
INITIAL_SHA='989665a5a4f8ea90600214be5536b4600c37fc448108a5bd177b3416e5ae0338'

def authorization():
    return {'kind':'predictor-clean-eot8-owned-university-exception-v1','approved':True,'control_sha256':'new','host':'linux10603','maximum_new_attempts':1,
            'work_seconds':2400,'worker_seconds':480,'workers':4,'collection_seconds':600,'collection_bytes':3*2**30,'write_proof_bytes':5*2**29,
            'setup_bytes':3*2**30,'capture_bytes':9*2**28,'raw_bytes':1792*2**20,'allocator_bytes':7*2**30,'bootstrap_bytes':2*2**20,
            'paid_compute':False,'reset_old_claims':False,'automatic_retries':0,'user_basis':'test fixture only, never live',
            'prior_consumed_control_sha256':op.PRIOR_CONSUMED_CONTROL,'prior_consumed_claim_sha256':op.PRIOR_CONSUMED_CLAIM}


def test_unarmed_refuses_before_claim_session_or_SSH(tmp_path):
    def forbidden(*a,**k):raise AssertionError('SSH attempted')
    with pytest.raises(ValueError,match='Unarmed'):client.run(tmp_path,popen=forbidden)
    assert list(tmp_path.iterdir())==[]

@pytest.mark.parametrize('field,value',[('kind','predictor-clean-eot8-linux10603-exception-v1'),('host','linux10609'),('work_seconds',2401),('maximum_new_attempts',2),('reset_old_claims',True),('paid_compute',True),('prior_consumed_claim_sha256','reset'),('control_sha256','old')])
def test_old_or_expanded_authority_refused(field,value):
    a=authorization();a[field]=value
    with pytest.raises(ValueError):op.validate_authorization(a,'new')

@pytest.fixture(scope='module')
def four_workers(tmp_path_factory):
    folder=tmp_path_factory.mktemp('owned-target-four-workers')
    code='''import sys
from pathlib import Path
from types import SimpleNamespace
from models.predictor_eot8_owned_worker_v1 import run
b=Path(sys.argv[1]);i=int(sys.argv[2]);run(SimpleNamespace(output=b/str(i),index=i,tiny=True,device='cpu',upstream=Path(sys.argv[3]),initial=Path(sys.argv[4]),initial_sha=sys.argv[5],control_sha256='local-CPU-test-only',reference=b/str(i-1) if i in (1,3) else None,replay_reference=b/str(i-2) if i in (2,3) else None))'''
    assert op.old.digest(OLD/'initial.pt')==INITIAL_SHA
    for i in range(4):
        result=subprocess.run([sys.executable,'-B','-c',code,str(folder),str(i),str(UPSTREAM),str(OLD/'initial.pt'),INITIAL_SHA],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        (folder/('worker-'+str(i)+'.log')).write_bytes(result.stdout)
        assert result.returncode==0,result.stdout.decode()[-5000:]
    return folder


def test_fresh_live_worker_CPU_analogue_exact_against_saved_returns(four_workers):
    pids=[];count=0
    for i in range(4):
        w=json.loads((four_workers/str(i)/'worker.json').read_text());old=json.loads((OLD/str(i)/'worker.json').read_text());pids.append(w['pid'])
        assert w['complete'] and w['passed'] and w['local_tiny_analogue'] and not w['backend']['CUDA_qualified']
        assert w['baseline']==old['baseline'] and w['ownership_check']==verify_ownership(w)
        assert len(w['artifacts'])==10 and len(w['phases'])==5 and w['adapter_restored']
        for name,item in w['artifacts'].items():
            t=torch.load(four_workers/str(i)/item['file'],weights_only=True,map_location='cpu')
            prior=torch.load(OLD/str(i)/old['artifacts'][name]['file'],weights_only=True,map_location='cpu')
            assert torch.equal(t,prior) and item['saved_before_numerical_gate'];count+=1
    assert len(set(pids))==4 and count==40

@pytest.mark.parametrize('fault',['missing_scopes','refusal','hook','root','duplicate_graph','forward_thread','backward_graph','backward_scope','call_thread','PID','missing_draw','incomplete_call'])
def test_forged_or_incomplete_scope_receipt_refused(four_workers,fault):
    w=json.loads((four_workers/'0/worker.json').read_text());s=w['backward_scopes'][0]
    forward=next(r for r in w['call_rows'] if not r['during_backward']);back=next(r for r in w['call_rows'] if r['during_backward'])
    if fault=='missing_scopes':w.pop('backward_scopes')
    elif fault=='refusal':w['ownership_refusals']=[{'predicate':'foreign-autograd-graph'}]
    elif fault=='hook':s['hook_removed']=False
    elif fault=='root':s['graph_id']=-1
    elif fault=='duplicate_graph':w['backward_scopes'][1]['graph_id']=s['graph_id']
    elif fault=='forward_thread':forward['thread']+=1
    elif fault=='backward_graph':back['graph_id']+=9999
    elif fault=='backward_scope':back['backward_scope']=4
    elif fault=='call_thread':back['thread']+=9999
    elif fault=='PID':back['pid']+=1
    elif fault=='missing_draw':w['call_rows'].remove(forward)
    elif fault=='incomplete_call':forward['outcome']='entered'
    with pytest.raises(ValueError,match='Graph ownership receipt'):verify_ownership(w)


def test_failed_backward_retains_actual_loss_and_refusal_and_removed_hook(tmp_path,monkeypatch):
    from models.predictor_eot8_owned_worker_v1 import run
    from models.predictor_eot8_owned_v1 import PredictorRoute
    original=PredictorRoute._admit
    def refuse(self):
        if self._task is not None and self._task.state=='running':self._fail('injected-integration-refusal')
        return original(self)
    monkeypatch.setattr(PredictorRoute,'_admit',refuse)
    folder=tmp_path/'failure'
    with pytest.raises(ValueError,match='injected-integration-refusal'):
        run(SimpleNamespace(output=folder,index=0,tiny=True,device='cpu',upstream=UPSTREAM,initial=OLD/'initial.pt',initial_sha=INITIAL_SHA,reference=None,replay_reference=None))
    w=json.loads((folder/'worker.json').read_text())
    assert not w['passed'] and not w['complete'] and w['adapter_restored'] and not w['phases']
    assert set(w['artifacts'])=={'EOT8-before-loss'} and len(w['backward_scopes'])==1
    s=w['backward_scopes'][0];assert s['state']=='failed' and s['hook_removed'] and s['graph_id']>=0
    r=w['ownership_refusals'][0];assert r['predicate']=='injected-integration-refusal' and r['current_graph_id']==s['graph_id']
    assert r['current_thread']>0 and r['creator_thread']>0
    assert not (folder/'EOT8-before-input.pt').exists()


def test_candidate_requires_graph_qualified_fresh_reference(four_workers,tmp_path,monkeypatch):
    import models.execution as execution
    from models.predictor_eot8_owned_worker_v1 import run
    saved=json.loads((four_workers/'0/worker.json').read_text());saved['backward_scopes'][0]['hook_removed']=False
    reference=tmp_path/'reference';reference.mkdir()
    # Hard links are local immutable CPU fixture inputs only, not remote payload.
    import os
    for p in (four_workers/'0').iterdir():
        if p.name!='worker.json':os.link(p,reference/p.name)
    (reference/'worker.json').write_text(json.dumps(saved))
    monkeypatch.setattr(execution,'configure_execution',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Model setup reached')))
    with pytest.raises(ValueError,match='hook not removed'):
        run(SimpleNamespace(output=tmp_path/'candidate',index=1,tiny=True,device='cpu',reference=reference,control_sha256='local-CPU-test-only'))
    w=json.loads((tmp_path/'candidate/worker.json').read_text());assert not w['passed'] and not w['artifacts']


def test_wrong_live_domain_refused_before_scientific_import_or_output(tmp_path):
    program='''import sys
from types import SimpleNamespace
from models.predictor_eot8_owned_worker_v1 import run
def audit(event,args):
 if event=='import' and args[0].split('.')[0] in ('torch','numpy'):raise AssertionError('Scientific import before domain refusal')
sys.addaudithook(audit)
try:run(SimpleNamespace(output=sys.argv[1],index=0,tiny=False,device='cuda'))
except ValueError as e:
 assert 'reviewed university host' in str(e)
else:raise AssertionError('Live worker reached')
'''
    p=subprocess.run([sys.executable,'-B','-c',program,str(tmp_path/'none')],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
    assert p.returncode==0,p.stderr.decode();assert not (tmp_path/'none').exists()


def test_pending_10609_remains_unselected(tmp_path):
    pending=tmp_path/'runs/pretrained/diagnostic/predictor-eot8-v1-20261105/linux10609-final';pending.mkdir(parents=True)
    client.ensure_previous_unarmed(tmp_path)
    (pending/'AUTHORIZATION.json').write_text('{}')
    with pytest.raises(ValueError,match='armed/claimed'):client.ensure_previous_unarmed(tmp_path)
