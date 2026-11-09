"""ID-correlation metadata, pinned collision and read-only adapter regressions."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from models import predictor_sdpa_launch_evidence_v1 as new
from models import predictor_sdpa_component_events_v2 as old
from scripts import check_predictor_sdpa_launch_evidence_v1 as audit
from tests.test_predictor_sdpa_component_events_v2 import fixture as legacy_fixture

ROOT=Path(__file__).resolve().parents[1]
COLLISION=ROOT/'runs/pretrained/diagnostic/predictor-sdpa-integration-v3-20261108/linux10603-final/attended-1o9_nbog/saved/results/components/0/events.jsonl'
COLLISION_SHA='e120e86d4a6bfbdb4b2d31911c27be8ef6a15063875c14e0c1bb9c7ff78a2449'
FILL='void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)'
ORIGIN=1_700_000_000_000_000_000


def fixture(family='efficient',*,cross=True,fills=True):
    # All correlation IDs here are HANDWRITTEN. None are recovered from old traces.
    device='cpu' if family=='cpu-flash' else 'cuda'
    rows=legacy_fixture(family,cross=cross,device=device)
    raw=[];owner={}
    for r in rows:
        if r['device']=='cpu':
            for k in r['linked_kernels']:owner[k['name']]=100+r['index']
    for i,r in enumerate(rows):
        raw.append({'index':i,'name':r['name'],'device':r['device'],'device_index':17 if r['device']=='cpu' else 0,
            'resource_id':r['thread'] if r['device']=='cpu' else 5,'thread':r['thread'],'end_thread':r['thread'],
            'fwd_thread':r['fwd_thread'],'sequence_nr':r['sequence_nr'],'scope':r['scope'],
            'start_ns':ORIGIN+round(r['start_us']*1000),'end_ns':ORIGIN+round(r['end_us']*1000),
            'correlation_id':100+i,'linked_correlation_id':owner[r['name']] if r['device']=='cuda' else 0,
            'is_async':False,'user_annotation':r['user_annotation'],'legacy_cuda_elapsed_us':-1.})
    if fills and device=='cuda':
        for start,thread in ((4000,7),(60000,8 if cross else 7)):
            template=next(r for r in raw if r['device']=='cpu' and r['thread']==thread and not r['user_annotation'])
            row=copy.deepcopy(template);row.update(index=len(raw),name='aten::fill_',start_ns=ORIGIN+start,
                end_ns=ORIGIN+start+500,correlation_id=100+len(raw),sequence_nr=-1,fwd_thread=0,scope=0)
            raw.append(row);host=row['correlation_id']
            gpu=copy.deepcopy(next(r for r in raw if r['device']=='cuda'))
            gpu.update(index=len(raw),name=FILL,correlation_id=900+len(raw),linked_correlation_id=host,
                start_ns=ORIGIN+200000+start,end_ns=ORIGIN+201376+start)
            raw.append(gpu)
    return {'kind':new.KIND,'source_api':new.API,'torch_version':'2.8.0+cu128','trace_start_ns':ORIGIN,'records':raw}


def at(trace,name):return next(r for r in trace['records'] if r['name']==name)
def gpu(trace):return next(r for r in trace['records'] if r['device']=='cuda')
def save(path,value):path.write_bytes(new.encoded(value))


def test_pinned_real_collision_stays_rejected_and_missing_ids_are_not_invented():
    data=COLLISION.read_bytes();assert hashlib.sha256(data).hexdigest()==COLLISION_SHA
    rows=[json.loads(s) for s in data.splitlines()]
    with pytest.raises(ValueError,match='ambiguous positive CUDA'):old.verify(rows,arm='original',device='cuda')
    for value in (rows,{'records':rows}):
        with pytest.raises(ValueError,match='old traces have no recoverable IDs'):new.verify(value,arm='original',device='cuda')
    assert COLLISION.read_bytes()==data
    assert [old.duration_ns(rows[i]['end_us']-rows[i]['start_us']) for i in (24,93)]==[1376,1376]


@pytest.mark.parametrize('family',('efficient','math','cpu-flash'))
@pytest.mark.parametrize('cross',(False,True))
def test_synthetic_raw_ID_links_preserve_backend_and_autograd_rules(family,cross):
    t=fixture(family,cross=cross);before=new.encoded(t);device='cpu' if family=='cpu-flash' else 'cuda'
    p=new.verify(t,arm='math' if family=='math' else 'original',device=device)
    assert p['observed_family']==family and p['metadata_only'] and not p['actual_CUDA_execution_admitted']
    assert bool(p['cross_thread_anchors']) is cross and not p['worker_completion_inferred']
    assert p['recorded_host_correlation_links_available'] and not p['per_kernel_global_unique_correlation_id_assumed']
    assert not p['kernel_name_or_duration_used_as_identity'] and new.encoded(t)==before


def test_equal_name_device_duration_launches_have_distinct_recorded_ID_owners(tmp_path):
    t=fixture();p=new.verify(t,arm='original',device='cuda');raw=t['records']
    fills=[w for w in p['kernel_witnesses'] if w['name']==FILL]
    assert len(fills)==2 and len({w['host_correlation_id'] for w in fills})==2
    assert len({w['CUDA_event_index'] for w in fills})==2 and all(w['end_ns']-w['start_ns']==1376 for w in fills)
    for w in fills:
        assert raw[w['CUDA_event_index']]['linked_correlation_id']==raw[w['host_event_index']]['correlation_id']==w['host_correlation_id']
    # A weaker tuple rule rejects this synthetic collision; the new proof never uses that tuple as identity.
    with pytest.raises(ValueError,match='ambiguous positive CUDA'):old.verify(new.project(t),arm='original',device='cuda')
    save(tmp_path/'synthetic-collision.json',t);save(tmp_path/'proof.json',p)


def test_individual_GPU_correlation_ids_need_not_be_globally_unique():
    t=fixture();devices=[r for r in t['records'] if r['device']=='cuda']
    for r in devices:r['correlation_id']=777
    p=new.verify(t,arm='original',device='cuda')
    assert len({w['CUDA_event_index'] for w in p['kernel_witnesses']})==len(devices)
    assert len({w['GPU_correlation_id'] for w in p['kernel_witnesses']})==1


@pytest.mark.parametrize('fault',('missing-id','unknown-owner','duplicate-owner','duplicate-device','foreign-device',
 'GPU-annotation','legacy-fallback','async-owner','unstable-owner-thread','wrong-forward-owner','cross-process',
 'CPU-ancestry-tie','broken-sequence','missing-attention-kernel','unlinked-attention-kernel','extra-backend','zero-kernel'))
def test_missing_ambiguous_foreign_or_misassociated_metadata_refuses(fault):
    t=fixture();rows=t['records'];f=at(t,new.FAMILIES['efficient'][0]);g=gpu(t)
    if fault=='missing-id':del g['linked_correlation_id']
    if fault=='unknown-owner':g['linked_correlation_id']=999999
    if fault=='duplicate-owner':at(t,new.FORWARD)['correlation_id']=f['correlation_id']
    if fault=='duplicate-device':r=copy.deepcopy(g);r['index']=len(rows);rows.append(r)
    if fault=='foreign-device':g['device_index']=1
    if fault=='GPU-annotation':g['user_annotation']=True
    if fault=='legacy-fallback':f['legacy_cuda_elapsed_us']=2.
    if fault=='async-owner':f['is_async']=True
    if fault=='unstable-owner-thread':f['end_thread']=99
    if fault=='wrong-forward-owner':g['linked_correlation_id']=at(t,new.BACKWARD)['correlation_id']
    if fault=='cross-process':f['device_index']=18
    if fault=='CPU-ancestry-tie':
        r=copy.deepcopy(f);r.update(index=len(rows),correlation_id=999,name='unrelated equal-interval op');rows.append(r)
    if fault=='broken-sequence':f['sequence_nr']=99
    if fault=='missing-attention-kernel':g['name']='unrelated_filler'
    if fault=='unlinked-attention-kernel':g['linked_correlation_id']=0
    if fault=='extra-backend':f['name']='aten::_scaled_dot_product_cudnn_attention'
    if fault=='zero-kernel':g['end_ns']=g['start_ns']
    with pytest.raises(ValueError):new.verify(t,arm='original',device='cuda')


@pytest.mark.parametrize('field,value',(('correlation_id',True),('linked_correlation_id',-1),('start_ns',-1),
 ('end_ns',2**63),('device_index',0.0),('sequence_nr',-2),('is_async',0),('user_annotation',1),
 ('legacy_cuda_elapsed_us',float('nan')),('name',None),('index',True),('scope','0')))
def test_strict_raw_field_types_and_integer_clocks(field,value):
    t=fixture();t['records'][0][field]=value
    with pytest.raises(ValueError):new.project(t)


def test_linked_runtime_thread_is_derived_explicitly_without_mutating_raw_record():
    t=fixture();f=at(t,new.FAMILIES['efficient'][0]);r=copy.deepcopy(f)
    r.update(index=len(t['records']),name='cudaLaunchKernel',thread=99,end_thread=99,sequence_nr=-1,
        correlation_id=888,linked_correlation_id=f['correlation_id'],start_ns=f['start_ns']+10,end_ns=f['start_ns']+20)
    t['records'].append(r);before=new.encoded(t);derived=new.project(t)[-1]
    assert derived['raw_thread']==99 and derived['thread']==f['thread'] and derived['parent']==f['index']
    assert new.encoded(t)==before and new.verify(t,arm='original',device='cuda')['observed_family']=='efficient'


def test_runtime_outside_its_linked_CPU_interval_refuses():
    t=fixture();f=at(t,new.FAMILIES['efficient'][0]);r=copy.deepcopy(f)
    r.update(index=len(t['records']),name='cudaLaunchKernel',correlation_id=888,
        linked_correlation_id=f['correlation_id'],start_ns=f['end_ns']+1,end_ns=f['end_ns']+2)
    t['records'].append(r)
    with pytest.raises(ValueError,match='outside'):new.project(t)


def fake_result(t):
    class Event:
        def __init__(self,row):self.row=row
        def __getattr__(self,method):
            if method=='device_type':return lambda:'DeviceType.'+self.row['device'].upper()
            field=next((k for k,v in new.METHODS.items() if v==method),None)
            if field is None:raise AttributeError(method)
            return lambda:self.row[field]
    return SimpleNamespace(events=lambda:[Event(r) for r in t['records']],trace_start_ns=lambda:t['trace_start_ns'])


def test_readonly_adapter_uses_exact_v28_raw_apis_and_retains_integer_epochs():
    t=fixture();result=fake_result(t);profile=SimpleNamespace(profiler=SimpleNamespace(kineto_results=result))
    captured=new.capture(profile,torch_version='2.8.0+cu128')
    assert captured==t and new.verify(captured,arm='original',device='cuda')['observed_family']=='efficient'


@pytest.mark.parametrize('profile',(None,SimpleNamespace(),SimpleNamespace(profiler=SimpleNamespace())))
def test_adapter_never_falls_back_to_processed_or_missing_events(profile):
    with pytest.raises(ValueError,match='Completed raw Kineto'):new.capture(profile,torch_version='2.8.0+cu128')


def test_actual_local_CPU_metadata_adapter(tmp_path):
    # Tiny CPU API probe only, not SDPA, autograd, a model or target qualification.
    import torch
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        with torch.profiler.record_function('launch-evidence/CPU-adapter-probe'):
            value=torch.ones(8);value=value+1
    t=new.capture(profile,torch_version=str(torch.__version__));rows=new.project(t)
    assert all(r['device']=='cpu' for r in rows) and any(r['name']=='aten::add' for r in rows)
    assert any(r['name']=='launch-evidence/CPU-adapter-probe' for r in rows)
    assert all(type(r['start_ns']) is int and r['start_ns']>2**53 for r in t['records'])
    save(tmp_path/'actual-local-CPU-raw.json',t)
    save(tmp_path/'probe-scope.json',{'CPU_only':True,'torch_version':str(torch.__version__),
        'SDPA_calls':0,'backwards':0,'model_calls':0,'target_runtime_qualified':False,'CUDA_qualified':False})


def test_cold_cli_blocks_science_process_and_network_and_saves_metadata_only(tmp_path):
    t=fixture();path=tmp_path/'fixture.json';save(path,t);out=tmp_path/'assessment'
    code='''import importlib.abc,sys,subprocess,socket,os,runpy
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','numpy','timm'):raise RuntimeError('Science import in metadata assessment')
sys.meta_path.insert(0,Deny())
def no(*a,**k):raise RuntimeError('process/network in assessment')
subprocess.Popen=subprocess.run=socket.socket=socket.create_connection=os.system=os.fork=no
sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')'''
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(ROOT/'scripts/check_predictor_sdpa_launch_evidence_v1.py'),
        '--input',str(path),'--output',str(out),'--selection','original','--device','cuda'],cwd=ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
    assert result.returncode==0,result.stderr.decode()
    r=json.loads((out/'assessment.json').read_text());assert r['passed'] and r['metadata_only'] and not r['actual_CUDA_execution_admitted']
    assert not r['worker_completion_inferred'] and not r['new_model_autograd_SDPA_profiler_CUDA_reference_or_university_work']


def test_cli_retains_refusal_for_old_trace_and_preserves_existing_output(tmp_path):
    out=tmp_path/'refused';r=audit.assess(COLLISION,out,arm='original',device='cuda')
    assert not r['passed'] and r['error_type']=='JSONDecodeError'
    with pytest.raises(FileExistsError):audit.assess(COLLISION,out,arm='original',device='cuda')


def test_collector_refuses_missing_required_raw_api_and_oversized_results():
    result=fake_result(fixture());result.events=lambda:[SimpleNamespace()]
    with pytest.raises(ValueError,match='lacks name'):new.capture_results(result,torch_version='2.8.0+cu128')
    result.events=lambda:[None]*(new.MAX_EVENTS+1)
    with pytest.raises(ValueError,match='never truncate'):new.capture_results(result,torch_version='2.8.0+cu128')
    t=fixture();t['extra']='not in the raw schema'
    with pytest.raises(ValueError):new.project(t)


def test_nested_duplicate_CPU_names_retain_the_exact_inner_kernel_owner():
    t=fixture();outer=at(t,'aten::fill_');inner=copy.deepcopy(outer)
    inner.update(index=len(t['records']),correlation_id=999,start_ns=outer['start_ns']+100,end_ns=outer['end_ns']-100)
    t['records'].append(inner)
    g=next(r for r in t['records'] if r['device']=='cuda' and r['linked_correlation_id']==outer['correlation_id'])
    g['linked_correlation_id']=999
    rows=new.project(t);assert rows[-1]['parent']==outer['index'] and rows[-1]['name']==outer['name']
    proof=new.verify(t,arm='original',device='cuda')
    assert any(w['host_correlation_id']==999 and w['host_event_index']==inner['index'] for w in proof['kernel_witnesses'])
    assert not any(w['host_correlation_id']==outer['correlation_id'] for w in proof['kernel_witnesses'])


def test_raw_enumeration_order_is_not_a_timestamp_or_duration_identity():
    t=fixture();t['records'].reverse()
    for i,r in enumerate(t['records']):r['index']=i
    proof=new.verify(t,arm='original',device='cuda')
    assert proof['observed_family']=='efficient' and len(proof['cross_thread_anchors'])==1
    assert len({w['host_correlation_id'] for w in proof['kernel_witnesses'] if w['name']==FILL})==2


def test_positive_GPU_intervals_can_outlive_CPU_labels_and_change_without_changing_links():
    t=fixture()
    for i,r in enumerate(t['records']):
        if r['device']=='cuda':r['start_ns']=ORIGIN+1000000+i*10000;r['end_ns']=r['start_ns']+3000
    proof=new.verify(t,arm='original',device='cuda')
    assert proof['observed_family']=='efficient' and not proof['timings_usable_for_speed_or_cost']
    assert all(w['start_ns']>at(t,new.BACKWARD)['end_ns'] for w in proof['kernel_witnesses'])


def test_byte_ceiling_and_unsupported_raw_schema_refuse_without_truncation(monkeypatch):
    t=fixture();monkeypatch.setattr(new,'MAX_BYTES',len(new.encoded(t))-1)
    with pytest.raises(ValueError,match='byte ceiling'):new.project(t)
    t['source_api']='processed FunctionEvent name/duration list'
    with pytest.raises(ValueError,match='Exact raw correlation schema'):new.project(t)


def test_wrong_selection_or_device_cannot_reuse_an_observed_efficient_proof():
    t=fixture()
    for arm,device in (('math','cuda'),('original','cpu'),('invented','cuda'),('original','unknown')):
        with pytest.raises(ValueError):new.verify(t,arm=arm,device=device)


def test_raw_adapter_requires_exact_API_even_if_processed_events_exist():
    class ProcessedOnly:
        def events(self):raise AssertionError('Never read a processed-event fallback')
    with pytest.raises(ValueError):new.capture(ProcessedOnly(),torch_version='2.8.0+cu128')
    with pytest.raises(ValueError,match='Raw Kineto event APIs'):new.capture_results(SimpleNamespace(events=lambda:[]),torch_version='2.8.0+cu128')
