"""Extraction retains bounded invalid scalars without changing frozen acceptance."""
import copy,json,math,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from models import predictor_sdpa_raw_capture_v1 as capture
from models import predictor_sdpa_launch_evidence_v1 as frozen
from models import predictor_sdpa_component_records_v1 as rec
from tests.test_predictor_sdpa_launch_evidence_v1 import fixture,fake_result
ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('family',('efficient','math','cpu-flash'))
def test_complete_extraction_matches_frozen_raw_values_and_acceptance(family):
    t=fixture(family);before=frozen.encoded(t);result=fake_result(t)
    actual=capture.capture_results(result,torch_version=t['torch_version'])
    assert frozen.encoded(actual)==before==frozen.encoded(frozen.capture_results(result,torch_version=t['torch_version']))
    device='cpu' if family=='cpu-flash' else 'cuda';arm='math' if family=='math' else 'original'
    assert frozen.verify(actual,arm=arm,device=device)==frozen.verify(t,arm=arm,device=device)


@pytest.mark.parametrize('field,value',(('device_index',-1),('thread',True),('end_thread',-1),('fwd_thread',-1),
 ('scope','0'),('sequence_nr',-2),('start_ns',-1),('end_ns',2**63),('correlation_id',0),
 ('linked_correlation_id',-1),('is_async',0),('user_annotation',1),('legacy_cuda_elapsed_us',1.),('name',None)))
def test_invalid_serializable_values_are_saved_exactly_then_still_refuse(tmp_path,field,value):
    t=fixture();t['records'][0][field]=value
    actual=capture.capture_results(fake_result(t),torch_version=t['torch_version'])
    assert actual['records'][0][field]==value and type(actual['records'][0][field]) is type(value)
    folder=tmp_path/'raw';folder.mkdir(mode=0o700);store=rec.Store(folder)
    item=store.write('events.jsonl',lambda f:f.write(frozen.encoded(actual)))
    data=rec.verified(folder,item);assert json.loads(data)==t
    with pytest.raises(ValueError):frozen.project(json.loads(data))
    assert not (folder/'complete.json').exists()


@pytest.mark.parametrize('field,value,detail',(('legacy_cuda_elapsed_us',float('nan'),'nonfinite float nan'),
 ('legacy_cuda_elapsed_us',float('inf'),'nonfinite float inf'),('legacy_cuda_elapsed_us',float('-inf'),'nonfinite float -inf'),
 ('thread',object(),'unsupported scalar type object'),('thread',2**300,'integer exceeds 256 bits'),
 ('name','x'*16385,'string length 16385 exceeds 16384')))
def test_nonportable_or_oversize_values_refuse_with_exact_record_and_field(field,value,detail):
    t=fixture();t['records'][0][field]=value
    with pytest.raises(ValueError) as e:capture.capture_results(fake_result(t),torch_version=t['torch_version'])
    assert 'record 0 field '+field in str(e.value) and detail in str(e.value)


def test_missing_API_names_its_actual_record_and_field():
    t=fixture();result=fake_result(t);rows=result.events();rows[0].row.pop('scope')
    with pytest.raises(ValueError,match='record 0 field scope: API scope raised KeyError'):
        capture.capture_results(SimpleNamespace(events=lambda:rows,trace_start_ns=result.trace_start_ns),torch_version=t['torch_version'])


def test_missing_profile_or_result_APIs_never_fall_back():
    with pytest.raises(ValueError,match='no processed-event fallback'):capture.capture(SimpleNamespace(),torch_version='fixture')
    with pytest.raises(ValueError,match='missing API events'):capture.capture_results(SimpleNamespace(),torch_version='fixture')
    with pytest.raises(ValueError,match='trace_start_ns: missing API'):capture.capture_results(SimpleNamespace(events=lambda:[None]),torch_version='fixture')


@pytest.mark.parametrize('value',([],None,'events',tuple([None]*(capture.MAX_EVENTS+1))))
def test_unbounded_or_incomplete_event_sequence_cannot_be_truncated(value):
    with pytest.raises(ValueError,match='never truncate'):capture.capture_results(SimpleNamespace(events=lambda:value),torch_version='fixture')


def test_existing_8MiB_extent_is_enforced_before_publication_without_partial_trace():
    t=fixture('cpu-flash');row=t['records'][0];row['name']='x'*16384
    rows=[copy.deepcopy(row) for _ in range(600)]
    assert len(rows)<=capture.MAX_EVENTS
    large={**t,'records':rows}
    with pytest.raises(ValueError,match='8 MiB ceiling; no truncation'):capture.capture_results(fake_result(large),torch_version=t['torch_version'])


def test_valid_frozen_field_limits_are_within_capture_transport_limits():
    # Broader retention bounds never replace frozen semantic bounds.
    assert capture.MAX_BYTES==frozen.MAX_BYTES==8*2**20 and capture.MAX_EVENTS==frozen.MAX_EVENTS
    assert capture.MAX_FIELD_CHARS>=4096 and capture.MAX_INTEGER_BITS>=64
    for value in (-1,0,2**63-1,2**63,False,None,'x'*4096):assert capture.scalar(value,'fixture')==value


def test_cold_extractor_import_performs_no_scientific_process_or_network_work(tmp_path):
    program="""import importlib.abc,os,socket,subprocess,sys
sys.path.insert(0,sys.argv[1])
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','numpy','timm'):raise RuntimeError('No scientific import')
sys.meta_path.insert(0,Deny())
def no(*a,**k):raise RuntimeError('No process or network')
subprocess.Popen=subprocess.run=os.system=os.fork=socket.socket=socket.create_connection=no
from models import predictor_sdpa_raw_capture_v1
print('PASS: standard-library extraction module imported without scientific execution.')"""
    r=subprocess.run([sys.executable,'-I','-B','-c',program,str(ROOT)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
    (tmp_path/'cold-import.log').write_bytes(r.stdout+r.stderr);assert r.returncode==0,r.stderr.decode()
