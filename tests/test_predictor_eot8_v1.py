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
from models import predictor_eot8_operator_v1 as op
from models import predictor_eot8_client_v1 as client
from models.predictor_eot8_capture_v1 import save_return,compare
from models.execution import configure_execution,apply_model_policy,ExecutionGuard
from models.native_diagnostic import tiny_fixture,save_tiny_fixture
from models.pilot_bundle import digest

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'external/vjepa2'


@pytest.fixture
def model():
    configure_execution('shared-fp64-native',device='cpu');torch.set_num_threads(2)
    torch.manual_seed(17);m,x,masks=tiny_fixture(UPSTREAM);apply_model_policy(m,'shared-fp64-native');m.eval()
    return m,x,masks


def test_private_route_checkpoint_restoration_and_real_cpu_events(model,tmp_path):
    m,x,masks=model;target=m.clean_targets(x);guard=ExecutionGuard(m,'cpu',target_versions=True)
    original=m.predictor.predictor_blocks[0].attn.forward.__func__;original_F=original.__globals__['F'];sdpa=original_F.scaled_dot_product_attention
    with adapter.PredictorRoute(m,'predictor-math',tiny=True) as route:
        with pytest.raises(ValueError):guard.check()
        route.phase='draw-profiled'
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
            value=x.detach().clone().requires_grad_(True);loss=m.loss_per_example(value,target,masks(10)).sum();route.backward_active=True;torch.autograd.grad(loss,value)
        assert len(route.rows)>=8 and any(r['during_backward'] for r in route.rows)
        assert all(r['effective_flags']['math'] and not r['effective_flags']['flash'] for r in route.rows if r['role']=='predictor')
        assert original_F.scaled_dot_product_attention is sdpa and original.__globals__['F'] is original_F
        events.save(p,tmp_path/'events.jsonl')
        result=events.verify(events.read(tmp_path/'events.jsonl'),route.rows,cuda=False,mode='predictor-math',owners=['encoder.0','encoder.1','predictor.0','predictor.1'])
        assert result['CPU_analogue_only'] and not result['CUDA_qualified']
        route.check()
    guard.check();assert m.predictor.predictor_blocks[0].attn.forward.__func__ is original
    assert all(p.grad is None for p in m.parameters())


@pytest.mark.parametrize('fault',['exception','identity','method','nested','source','alias'])
def test_restore_and_refuse_routing_faults(model,monkeypatch,fault):
    m,x,masks=model;original=m.predictor.predictor_blocks[0].attn.forward.__func__
    if fault=='source':monkeypatch.setattr(adapter,'MODULE_SHA','0'*64)
    if fault in ('source','alias'):
        if fault=='alias':
            # The alias is bound at install; change it after install to test active drift.
            target=m.clean_targets(x)
            with adapter.PredictorRoute(m,'original',tiny=True) as route:
                monkeypatch.setattr(original.__globals__['F'],'scaled_dot_product_attention',lambda *a,**k:None)
                with pytest.raises(ValueError):m.loss_per_example(x,target,masks(1))
        else:
            with pytest.raises(ValueError):
                with adapter.PredictorRoute(m,'original',tiny=True):pass
    else:
        with pytest.raises((ValueError,RuntimeError)):
            with adapter.PredictorRoute(m,'predictor-math',tiny=True) as route:
                if fault=='exception':raise RuntimeError('injected')
                if fault=='identity':m._diagnostic_execution_policy='original';route.check()
                if fault=='method':m.predictor.predictor_blocks[0].attn.forward=lambda x:x;route.check()
                if fault=='nested':
                    with adapter.PredictorRoute(m,'original',tiny=True):pass
    assert m._diagnostic_execution_policy=='shared-fp64-native'
    assert m.predictor.predictor_blocks[0].attn.forward.__func__ is original


@pytest.mark.parametrize('kind,shape,dtype,nonfinite',[('loss',[],torch.float32,True),('loss',[1],torch.float32,False),('loss',[],torch.float64,False),('input',[1,2],torch.float32,False)])
def test_actual_returns_retained_before_gate(tmp_path,kind,shape,dtype,nonfinite):
    value=torch.full(shape,float('nan') if nonfinite else 1.,dtype=dtype)
    path=tmp_path/'value.pt'
    if not nonfinite:
        with pytest.raises(ValueError):save_return(path,value,kind,[1,3])
    else:
        item=save_return(path,value,kind,[1,3]);assert item['saved_before_numerical_gate']
        with pytest.raises(ValueError):compare(value,torch.ones_like(value),{'rtol':1e-5,'atol':2e-6})
    actual=torch.load(path,weights_only=True);assert actual.dtype==dtype and actual.shape==value.shape
    if nonfinite:assert torch.isnan(actual)


def test_capture_disk_failure_preserves_partial(tmp_path,monkeypatch):
    def broken(value,f):f.write(b'partial');raise OSError('injected ENOSPC')
    monkeypatch.setattr(torch,'save',broken)
    with pytest.raises(OSError):save_return(tmp_path/'loss.pt',torch.tensor(1.),'loss',[])
    assert (tmp_path/'loss.pt.partial').read_bytes()==b'partial'


def row(i,name,parent=None,seq=-1,scope=0,thread=1,fwd=0,device='cpu',start=0,end=10,kernels=()):
    return {'index':i,'name':name,'device':device,'device_index':0,'parent':parent,'thread':thread,'fwd_thread':fwd,'sequence_nr':seq,'scope':scope,
            'user_annotation':False,'start_us':start,'end_us':end,'legacy':False,'linked_kernels':list(kernels)}


def synthetic():
    rows=[];calls=[]
    for j in range(2):
        label=adapter.PREFIX+str(j)+'/predictor.0';base=len(rows);offset=j*20
        calls.append({'index':j,'owner':'predictor.0','phase':'draw-profiled','outcome':'returned'})
        rows += [row(base,label,start=offset,end=offset+10),row(base+1,'aten::_scaled_dot_product_attention_math',base,seq=10+j,start=offset,end=offset+10,kernels=[{'name':'kernel','device':0,'duration_us':1}]),
                 row(base+2,'BmmBackward0',seq=10+j,scope=1,fwd=1,start=offset+11,end=offset+15),row(base+3,'SoftmaxBackward0',base+2,seq=10+j,scope=1,fwd=1,start=offset+11,end=offset+15,kernels=[{'name':'kernel','device':0,'duration_us':1}])]
    rows.append(row(len(rows),'kernel',device='cuda',start=1,end=2))
    return rows,calls


@pytest.mark.parametrize('fault',['none','missing_sequence','wrong_fwd_thread','ambiguous','missing_backend','no_kernels','cycle','GPU_annotation_only','missing_call'])
def test_owned_backend_association_refuses_missing_or_ambiguous(fault):
    rows,calls=synthetic()
    if fault=='missing_sequence':rows[1]['sequence_nr']=-1
    if fault=='wrong_fwd_thread':rows[2]['fwd_thread']=rows[3]['fwd_thread']=99
    if fault=='ambiguous':rows[5]['sequence_nr']=10
    if fault=='missing_backend':rows[1]['name']='unknown'
    if fault=='no_kernels':
        for r in rows:r['linked_kernels']=[]
    if fault=='cycle':rows[2]['parent']=2
    if fault=='GPU_annotation_only':rows[-1]['user_annotation']=True
    if fault=='missing_call':calls.pop()
    if fault=='none':
        result=events.verify(rows,calls,cuda=True,mode='predictor-math',owners=['predictor.0']);assert result['CUDA_qualified']
    else:
        with pytest.raises(ValueError):events.verify(rows,calls,cuda=True,mode='predictor-math',owners=['predictor.0'])


def test_unarmed_refuses_before_any_side_effect(tmp_path):
    def forbidden(*a,**k):raise AssertionError('SSH attempted')
    with pytest.raises(ValueError,match='Unarmed'):client.run(tmp_path,popen=forbidden)
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('field,value',[('approved',False),('host','linux10603'),('work_seconds',600),('raw_bytes',400*2**20),('reset_old_claims',True),('control_sha256','old')])
def test_authorization_exact_new_scope(field,value):
    a=authorization();a[field]=value
    with pytest.raises(ValueError):op.validate_authorization(a,'new')


def authorization():
    return {'kind':'predictor-clean-eot8-exception-v1','approved':True,'control_sha256':'new','host':'linux10609','maximum_new_attempts':1,
            'work_seconds':2400,'worker_seconds':480,'workers':4,'collection_seconds':600,'collection_bytes':3*2**30,'write_proof_bytes':5*2**29,
            'setup_bytes':3*2**30,'capture_bytes':9*2**28,'raw_bytes':1792*2**20,'allocator_bytes':7*2**30,'bootstrap_bytes':2*2**20,
            'paid_compute':False,'reset_old_claims':False,'automatic_retries':0,'user_basis':'test only, never live'}


def test_local_four_fresh_real_workers(tmp_path):
    initial=tmp_path/'initial.pt';save_tiny_fixture(UPSTREAM,initial)
    code='''import sys\nfrom pathlib import Path\nfrom types import SimpleNamespace\nfrom models.predictor_eot8_worker_v1 import run\nb=Path(sys.argv[1]);i=int(sys.argv[2]);run(SimpleNamespace(output=b/str(i),index=i,tiny=True,device='cpu',upstream=Path(sys.argv[3]),initial=b/'initial.pt',initial_sha=sys.argv[4],reference=b/str(i-1) if i in (1,3) else None,replay_reference=b/str(i-2) if i in (2,3) else None))'''
    pids=[]
    for i in range(4):
        result=subprocess.run([sys.executable,'-B','-c',code,str(tmp_path),str(i),str(UPSTREAM),digest(initial)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        (tmp_path/('log-'+str(i))).write_bytes(result.stdout)
        assert result.returncode==0,result.stdout.decode()[-5000:]
        w=json.loads((tmp_path/str(i)/'worker.json').read_text());pids.append(w['pid'])
        assert w['complete'] and w['passed'] and w['local_tiny_analogue'] and not w['backend']['CUDA_qualified']
        assert len(w['artifacts'])==10 and len(w['phases'])==5
    assert len(set(pids))==4


def test_recompute_early_stop_restores_private_context(model):
    from torch.utils.checkpoint import _StopRecomputationError
    m,x,masks=model;owner=m.predictor.predictor_blocks[0].attn;original=owner.forward.__func__
    with adapter.PredictorRoute(m,'predictor-math',tiny=True) as route:
        view=owner.forward.__func__.__globals__['F']
        view.route=lambda *a,**k:(_ for _ in ()).throw(_StopRecomputationError())
        with pytest.raises(_StopRecomputationError):owner(torch.rand(1,32,64))
        # The checkpoint stop can unwind an owned method without leaking a global alias.
        assert original.__globals__['F'] is torch.nn.functional
    assert owner.forward.__func__ is original and not route.active


def test_route_rejects_actual_FP64_without_cast(model):
    m,x,masks=model;m.double();attn=m.encoder.blocks[0].attn
    with adapter.PredictorRoute(m,'predictor-math',tiny=True) as route:
        with pytest.raises(ValueError,match='actual FP32'):attn(torch.rand(1,32,128,dtype=torch.float64))
    assert next(attn.parameters()).dtype==torch.float64


def test_candidate_requires_new_passing_reference_before_any_model_work(tmp_path,monkeypatch):
    import models.execution as execution
    from models.predictor_eot8_worker_v1 import run
    monkeypatch.setattr(execution,'configure_execution',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Model setup reached')))
    with pytest.raises(ValueError,match='reference missing before model work'):
        run(SimpleNamespace(output=tmp_path/'worker',index=1,tiny=True,device='cpu',reference=None))
    state=json.loads((tmp_path/'worker/worker.json').read_text())
    assert not state['complete'] and state['error_type']=='ValueError' and not state['artifacts']
