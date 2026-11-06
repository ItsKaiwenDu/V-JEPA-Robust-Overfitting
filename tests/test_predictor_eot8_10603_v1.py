"""Actual cold CPU/checkpoint analogues and adversarial evidence/guard checks."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest
import torch

from models import predictor_eot8_adapter_v1 as adapter
from models import predictor_eot8_events_v1 as events
from models import predictor_eot8_operator_10603_v1 as op
from models import predictor_eot8_client_10603_v1 as client
from models.predictor_eot8_capture_v1 import save_return,compare
from models.execution import configure_execution,apply_model_policy,ExecutionGuard
from models.native_diagnostic import tiny_fixture,save_tiny_fixture
from models.pilot_bundle import digest

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'external/vjepa2'




def test_unarmed_refuses_before_any_side_effect(tmp_path):
    def forbidden(*a,**k):raise AssertionError('SSH attempted')
    with pytest.raises(ValueError,match='Unarmed'):client.run(tmp_path,popen=forbidden)
    assert list(tmp_path.iterdir())==[]

@pytest.mark.parametrize('field,value',[('approved',False),('host','linux10609'),('work_seconds',600),('raw_bytes',400*2**20),('reset_old_claims',True),('control_sha256','old')])
def test_authorization_exact_new_scope(field,value):
    a=authorization();a[field]=value
    with pytest.raises(ValueError):op.validate_authorization(a,'new')

def authorization():
    return {'kind':'predictor-clean-eot8-linux10603-exception-v1','approved':True,'control_sha256':'new','host':'linux10603','maximum_new_attempts':1,
            'work_seconds':2400,'worker_seconds':480,'workers':4,'collection_seconds':600,'collection_bytes':3*2**30,'write_proof_bytes':5*2**29,
            'setup_bytes':3*2**30,'capture_bytes':9*2**28,'raw_bytes':1792*2**20,'allocator_bytes':7*2**30,'bootstrap_bytes':2*2**20,
            'paid_compute':False,'reset_old_claims':False,'automatic_retries':0,'user_basis':'test only, never live','replaces_pending_control_sha256':op.REPLACES_CONTROL}

def test_local_four_fresh_real_workers(tmp_path):
    initial=tmp_path/'initial.pt';save_tiny_fixture(UPSTREAM,initial)
    code='''import sys\nfrom pathlib import Path\nfrom types import SimpleNamespace\nfrom models.predictor_eot8_worker_10603_v1 import run\nb=Path(sys.argv[1]);i=int(sys.argv[2]);run(SimpleNamespace(output=b/str(i),index=i,tiny=True,device='cpu',upstream=Path(sys.argv[3]),initial=b/'initial.pt',initial_sha=sys.argv[4],reference=b/str(i-1) if i in (1,3) else None,replay_reference=b/str(i-2) if i in (2,3) else None))'''
    pids=[]
    for i in range(4):
        result=subprocess.run([sys.executable,'-B','-c',code,str(tmp_path),str(i),str(UPSTREAM),digest(initial)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        (tmp_path/('log-'+str(i))).write_bytes(result.stdout)
        assert result.returncode==0,result.stdout.decode()[-5000:]
        w=json.loads((tmp_path/str(i)/'worker.json').read_text());pids.append(w['pid'])
        assert w['complete'] and w['passed'] and w['local_tiny_analogue'] and not w['backend']['CUDA_qualified']
        assert len(w['artifacts'])==10 and len(w['phases'])==5
    assert len(set(pids))==4

def test_candidate_requires_new_passing_reference_before_any_model_work(tmp_path,monkeypatch):
    import models.execution as execution
    from models.predictor_eot8_worker_10603_v1 import run
    monkeypatch.setattr(execution,'configure_execution',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Model setup reached')))
    with pytest.raises(ValueError,match='reference missing before model work'):
        run(SimpleNamespace(output=tmp_path/'worker',index=1,tiny=True,device='cpu',reference=None))
    state=json.loads((tmp_path/'worker/worker.json').read_text())
    assert not state['complete'] and state['error_type']=='ValueError' and not state['artifacts']


def test_worker_tensor_operations_unchanged_from_frozen_10609_source():
    old=(ROOT/'models/predictor_eot8_worker_v1.py').read_text()
    expected=old.replace("platform.node()!='linux10609'","platform.node()!='linux10603'").replace("'kind':'predictor-clean-eot8-worker-v1'","'kind':'predictor-clean-eot8-worker-linux10603-v1','execution_host':platform.node()")
    assert (ROOT/'models/predictor_eot8_worker_10603_v1.py').read_text()==expected


@pytest.mark.parametrize('field,value',[('kind','predictor-clean-eot8-exception-v1'),('replaces_pending_control_sha256','wrong-pending-control')])
def test_10609_authority_or_wrong_replacement_refused(field,value):
    a=authorization();a[field]=value
    with pytest.raises(ValueError):op.validate_authorization(a,'new')


def test_other_pending_handoff_must_be_unarmed_and_unclaimed(tmp_path):
    pending=tmp_path/'runs/pretrained/diagnostic/predictor-eot8-v1-20261105/linux10609-final'
    pending.mkdir(parents=True)
    client.ensure_previous_unarmed(tmp_path)
    (pending/'AUTHORIZATION.json').write_text('{}')
    with pytest.raises(ValueError,match='armed/claimed'):client.ensure_previous_unarmed(tmp_path)
    (pending/'AUTHORIZATION.json').unlink()
    (pending/'ATTEMPT.json').write_text('{}')
    with pytest.raises(ValueError,match='armed/claimed'):client.ensure_previous_unarmed(tmp_path)
