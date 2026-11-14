"""Saved/synthetic metadata regressions; no model, SDPA, profiler or GPU work."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from models import predictor_sdpa_launch_evidence_v2 as new
from models import predictor_sdpa_launch_evidence_v1 as frozen
from scripts import check_predictor_sdpa_launch_evidence_v2 as audit
from tests.test_predictor_sdpa_launch_evidence_v1 import fixture,at,gpu,ORIGIN

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'runs/pretrained/diagnostic/predictor-sdpa-continuation-v4-20261113/linux10603-final'
RAW=TARGET/'attended-6ub5e6wx/saved/results/components/0/events.jsonl'
RAW_SHA='d1b2006c583c6cfc4d8dc0bb0b12e823293222079c740d6d78acb78a72382686'
CLAIM_SHA='3829d1b634b32053f73560610f2e78d5a835e7193eaea885b55211e6c2536c32'


def overhead(t,**changes):
    # Handwritten signature, not an authenticated native activity-type witness.
    r={'index':len(t['records']),'name':new.OVERHEAD_NAME,'device':'cpu','device_index':-1,
       'resource_id':0,'thread':1,'end_thread':1,'fwd_thread':0,'sequence_nr':-1,'scope':0,
       'start_ns':ORIGIN,'end_ns':ORIGIN+1000000,'correlation_id':999999,'linked_correlation_id':0,
       'is_async':False,'user_annotation':False,'legacy_cuda_elapsed_us':-1}
    r.update(changes);t['records'].append(r);return r


def test_backend_and_autograd_validator_body_stays_exactly_frozen():
    a=ast.parse((ROOT/'models/predictor_sdpa_launch_evidence_v1.py').read_text())
    b=ast.parse((ROOT/'models/predictor_sdpa_launch_evidence_v2.py').read_text())
    f=next(x for x in a.body if isinstance(x,ast.FunctionDef) and x.name=='verify')
    g=next(x for x in b.body if isinstance(x,ast.FunctionDef) and x.name=='_verify_view');g.name='verify'
    assert ast.dump(f,include_attributes=False)==ast.dump(g,include_attributes=False)
    assert new.associated_regions is frozen.associated_regions and new.FAMILIES==frozen.FAMILIES
    assert new.MAX_EVENTS==frozen.MAX_EVENTS==20000 and new.MAX_BYTES==frozen.MAX_BYTES==8*2**20


@pytest.mark.parametrize('family',('efficient','math','cpu-flash'))
@pytest.mark.parametrize('cross',(False,True))
def test_existing_backend_families_keep_all_rows_and_links(family,cross):
    t=fixture(family,cross=cross);before=new.encoded(t)
    p=new.verify(t,arm='math' if family=='math' else 'original',device='cpu' if family=='cpu-flash' else 'cuda')
    assert p['observed_family']==family and bool(p['cross_thread_anchors']) is cross
    assert p['raw_records']==p['projected_records']==len(t['records']) and not p['retained_overhead_indices']
    assert p['metadata_only'] and not p['actual_CUDA_execution_admitted'] and not p['worker_completion_inferred']
    assert not p['CUDA_forward_and_backward_qualified'] and new.encoded(t)==before


def test_signature_retained_but_cannot_parent_or_own_operations(tmp_path):
    t=fixture();r=overhead(t);before=new.encoded(t)
    with pytest.raises(ValueError):frozen.verify(t,arm='original',device='cuda')
    rows=new.project(t);p=new.verify(t,arm='original',device='cuda')
    assert len(rows)==len(t['records']) and rows[-1]['index']==r['index'] and rows[-1]['device_index']==-1
    assert rows[-1]['record_role']=='profiler-overhead-signature' and rows[-1]['parent'] is None
    assert not rows[-1]['linked_kernels'] and all(x['parent']!=r['index'] for x in rows)
    assert p['retained_overhead_indices']==[r['index']] and not p['native_activity_kind_observed']
    assert not p['raw_rows_removed_or_rewritten'] and not p['overhead_can_own_launches']
    assert all(w['host_event_index']!=r['index'] for w in p['kernel_witnesses'])
    assert new.encoded(t)==before
    (tmp_path/'synthetic-overhead.json').write_bytes(before)
    (tmp_path/'synthetic-proof.json').write_bytes(new.encoded(p))


@pytest.mark.parametrize('field,value',(('name','unknown overhead'),('name',new.FORWARD),('name','aten::_scaled_dot_product_efficient_attention'),
 ('device','cuda'),('resource_id',1),('thread',2),('end_thread',2),('fwd_thread',7),('sequence_nr',1),('scope',1),
 ('correlation_id',0),('linked_correlation_id',100),('is_async',True),('user_annotation',True),('legacy_cuda_elapsed_us',0),
 ('device_index',-2),('device_index',-1.0),('device_index',True),('resource_id',False),('correlation_id',True),
 ('start_ns',ORIGIN-1),('end_ns',-1),('scope','0'),('sequence_nr',-2)))
def test_unknown_or_malformed_negative_device_signatures_refuse(field,value):
    t=fixture();overhead(t,**{field:value});before=new.encoded(t)
    with pytest.raises(ValueError):new.verify(t,arm='original',device='cuda')
    assert new.encoded(t)==before


@pytest.mark.parametrize('version',('2.8.0','2.8.0+cu129','2.9.0+cu128','2.14.1','2.8.0+cu128\x00',''))
def test_overhead_signature_needs_exact_supported_runtime(version):
    t=fixture();overhead(t);t['torch_version']=version
    with pytest.raises(ValueError):new.project(t)


def test_positive_process_row_is_not_excluded_just_because_of_name():
    t=fixture();f=at(t,new.FAMILIES['efficient'][0]);r=copy.deepcopy(f)
    r.update(index=len(t['records']),name=new.OVERHEAD_NAME,correlation_id=999999,start_ns=f['start_ns']+1,end_ns=f['start_ns']+2,sequence_nr=-1,fwd_thread=0)
    t['records'].append(r);row=new.project(t)[-1]
    assert row['record_role']=='frontend-CPU' and row['parent']==f['index']
    assert not new.verify(t,arm='original',device='cuda')['retained_overhead_indices']


@pytest.mark.parametrize('role',('CUDA','runtime','unknown-negative-op'))
def test_overhead_cannot_replace_or_ambiguously_alias_a_real_owner(role):
    t=fixture();h=overhead(t)
    if role=='CUDA':gpu(t)['linked_correlation_id']=h['correlation_id']
    if role=='runtime':
        r=copy.deepcopy(at(t,new.FAMILIES['efficient'][0]));r.update(index=len(t['records']),linked_correlation_id=h['correlation_id'],correlation_id=88888);t['records'].append(r)
    if role=='unknown-negative-op':h.update(name='aten::bmm')
    with pytest.raises(ValueError):new.verify(t,arm='original',device='cuda')


@pytest.mark.parametrize('fault',('unknown-owner','duplicate-owner','duplicate-device','foreign-device','GPU-annotation',
 'legacy-fallback','async-owner','unstable-owner','wrong-phase-owner','cross-process','ancestry-tie','broken-sequence',
 'missing-kernel','unlinked-kernel','extra-backend','zero-kernel'))
def test_existing_strict_failures_still_refuse_with_overhead_present(fault):
    t=fixture();overhead(t);f=at(t,new.FAMILIES['efficient'][0]);g=gpu(t)
    if fault=='unknown-owner':g['linked_correlation_id']=777777
    if fault=='duplicate-owner':at(t,new.FORWARD)['correlation_id']=f['correlation_id']
    if fault=='duplicate-device':r=copy.deepcopy(g);r['index']=len(t['records']);t['records'].append(r)
    if fault=='foreign-device':g['device_index']=1
    if fault=='GPU-annotation':g['user_annotation']=True
    if fault=='legacy-fallback':f['legacy_cuda_elapsed_us']=2.
    if fault=='async-owner':f['is_async']=True
    if fault=='unstable-owner':f['end_thread']=99
    if fault=='wrong-phase-owner':g['linked_correlation_id']=at(t,new.BACKWARD)['correlation_id']
    if fault=='cross-process':f['device_index']=18
    if fault=='ancestry-tie':
        r=copy.deepcopy(f);r.update(index=len(t['records']),correlation_id=88888,name='unrelated');t['records'].append(r)
    if fault=='broken-sequence':f['sequence_nr']=99
    if fault=='missing-kernel':g['name']='unrelated'
    if fault=='unlinked-kernel':g['linked_correlation_id']=0
    if fault=='extra-backend':f['name']='aten::_scaled_dot_product_cudnn_attention'
    if fault=='zero-kernel':g['end_ns']=g['start_ns']
    with pytest.raises(ValueError):new.verify(t,arm='original',device='cuda')



def test_inherited_overhead_ID_cannot_shadow_the_real_frontend_owner():
    t=fixture();f=at(t,new.FAMILIES['efficient'][0]);h=overhead(t,correlation_id=f['correlation_id'])
    another=overhead(t,correlation_id=f['correlation_id'],start_ns=ORIGIN+1)
    p=new.verify(t,arm='original',device='cuda')
    assert p['retained_overhead_indices']==[h['index'],another['index']]
    assert any(w['host_event_index']==f['index'] for w in p['kernel_witnesses'])
    assert all(w['host_event_index'] not in p['retained_overhead_indices'] for w in p['kernel_witnesses'])
    rows=new.project(t)
    assert all(r['parent'] not in p['retained_overhead_indices'] for r in rows)



def gpu_range(t,**changes):
    phase=at(t,new.FORWARD)
    r=copy.deepcopy(gpu(t));r.update(index=len(t['records']),name=new.FORWARD,user_annotation=True,
        correlation_id=phase['correlation_id'],linked_correlation_id=0,scope=0,sequence_nr=-1,fwd_thread=0,
        thread=1,end_thread=1,is_async=False,legacy_cuda_elapsed_us=-1)
    r.update(changes);t['records'].append(r);return r


def test_source_supported_GPU_range_is_retained_and_never_a_kernel_witness():
    t=fixture();overhead(t);r=gpu_range(t);before=new.encoded(t)
    rows=new.project(t);p=new.verify(t,arm='original',device='cuda')
    assert rows[r['index']]['annotation_host_event_index']==at(t,new.FORWARD)['index']
    assert rows[r['index']]['record_role']=='profiler-GPU-range' and not rows[r['index']]['linked_kernels']
    assert p['retained_GPU_annotation_indices']==[r['index']] and not p['annotations_qualify_as_kernels']
    assert all(w['CUDA_event_index']!=r['index'] for w in p['kernel_witnesses'])
    assert new.encoded(t)==before


@pytest.mark.parametrize('field,value',(('name','unknown phase'),('linked_correlation_id',100),('correlation_id',9999999),
 ('device_index',1),('scope',1),('sequence_nr',1),('fwd_thread',1),('thread',2),('end_thread',2),('is_async',True),
 ('legacy_cuda_elapsed_us',0),('user_annotation',False)))
def test_unknown_or_malformed_GPU_range_cannot_enter_kernel_proof(field,value):
    t=fixture();gpu_range(t,**{field:value})
    with pytest.raises(ValueError):new.verify(t,arm='original',device='cuda')


def test_GPU_annotation_cannot_replace_required_real_kernel_or_wrong_phase():
    for fault in ('replace','non-label-owner','wrong-label','duplicate-stream','runtime'):
        t=fixture();r=gpu_range(t)
        if fault=='replace':
            k=gpu(t);k.update(name=new.FORWARD,user_annotation=True,correlation_id=at(t,new.FORWARD)['correlation_id'],
                linked_correlation_id=0,thread=1,end_thread=1,scope=0,sequence_nr=-1,fwd_thread=0,resource_id=99)
        if fault=='non-label-owner':r['correlation_id']=at(t,new.FAMILIES['efficient'][0])['correlation_id']
        if fault=='wrong-label':r['name']=new.BACKWARD
        if fault=='duplicate-stream':gpu_range(t,start_ns=r['start_ns']+1)
        if fault=='runtime':t['torch_version']='2.14.1'
        with pytest.raises(ValueError):new.verify(t,arm='original',device='cuda')


def test_bounded_schema_count_and_bytes_remain_strict(monkeypatch):
    t=fixture();overhead(t)
    monkeypatch.setattr(new,'MAX_BYTES',len(new.encoded(t))-1)
    with pytest.raises(ValueError,match='byte ceiling'):new.project(t)
    monkeypatch.setattr(new,'MAX_BYTES',frozen.MAX_BYTES);monkeypatch.setattr(new,'MAX_EVENTS',len(t['records'])-1)
    with pytest.raises(ValueError):new.project(t)
    monkeypatch.setattr(new,'MAX_EVENTS',frozen.MAX_EVENTS);t['extra']=True
    with pytest.raises(ValueError):new.project(t)


def test_actual_saved_trace_keeps_frozen_refusal_and_incomplete_claim(tmp_path):
    data=RAW.read_bytes();assert hashlib.sha256(data).hexdigest()==RAW_SHA
    claim=TARGET/'ATTEMPT.json';before=claim.stat().st_mtime_ns
    assert hashlib.sha256(claim.read_bytes()).hexdigest()==CLAIM_SHA
    t=json.loads(data)
    with pytest.raises(ValueError,match='Invalid raw event fields'):frozen.project(t)
    p=new.verify(t,arm='original',device='cuda')
    assert p['retained_overhead_indices']==[10] and p['retained_GPU_annotation_indices']==[20]
    assert p['raw_records']==p['projected_records']==106
    assert p['raw_trace_sha256']==RAW_SHA and not p['worker_completion_inferred']
    assert not p['CUDA_forward_and_backward_qualified'] and p['observed_family']=='efficient'
    w=json.loads((RAW.parent/'worker.json').read_text())
    assert not w['complete'] and not w['profile_replay_exact'] and not w['flags_restored'] and not w['inputs_unchanged']
    assert not (RAW.parent/'complete.json').exists() and not (RAW.parent/'component-resource.json').exists()
    assert RAW.read_bytes()==data and claim.stat().st_mtime_ns==before
    (tmp_path/'actual-saved-metadata-proof.json').write_bytes(new.encoded(p))


def test_cold_saved_assessment_blocks_science_profiler_process_and_network(tmp_path):
    code='''import importlib.abc,sys,subprocess,socket,os,runpy
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','numpy','timm'):raise RuntimeError('Science import in metadata assessment')
sys.meta_path.insert(0,Deny())
def no(*a,**k):raise RuntimeError('process/network in metadata assessment')
subprocess.Popen=subprocess.run=socket.socket=socket.create_connection=os.system=os.fork=no
sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')'''
    out=tmp_path/'cold-assessment'
    r=subprocess.run([sys.executable,'-I','-B','-c',code,str(ROOT/'scripts/check_predictor_sdpa_launch_evidence_v2.py'),
        '--input',str(RAW),'--output',str(out),'--selection','original','--device','cuda'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
    assert r.returncode==0,r.stderr.decode()
    p=json.loads((out/'assessment.json').read_text())
    assert p['passed'] and p['metadata_only'] and not p['actual_CUDA_execution_admitted']
    assert not p['worker_completion_inferred'] and not p['new_model_autograd_SDPA_profiler_CUDA_reference_or_university_work']


def test_assessment_refuses_previous_format_and_preserves_existing_output(tmp_path):
    path=tmp_path/'invalid.json';path.write_text('[]')
    out=tmp_path/'refusal';r=audit.assess(path,out,arm='original',device='cuda')
    assert not r['passed'] and r['error_type']=='ValueError'
    with pytest.raises(FileExistsError):audit.assess(path,out,arm='original',device='cuda')


def test_every_projected_original_index_and_provenance_field_is_retained():
    t=fixture();overhead(t);t['records'].reverse()
    for i,r in enumerate(t['records']):r['index']=i
    rows=new.project(t);p=new.verify(t,arm='original',device='cuda')
    assert p['retained_overhead_indices']==[0]
    for a,b in zip(t['records'],rows):
        for k in ('index','name','device','device_index','scope','fwd_thread','sequence_nr','user_annotation','start_ns','end_ns','correlation_id','linked_correlation_id','resource_id'):
            assert a[k]==b[k]
    assert p['raw_records']==p['projected_records']==len(t['records'])
