"""Observe the actual frozen function, retain failures and recheck each stage."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from models.attention_component import fp64_oracle
from models.oracle_trace import Observation,audit_observation,load_arrays,save_arrays
from models.pilot_bundle import digest
from models.study_artifacts import read_record,write_record
from test_projection_reuse import restore_process_settings


@pytest.fixture
def capture(tmp_path,request):
    torch.set_num_threads(2);tokens=getattr(request,'param',9)
    gen=torch.Generator().manual_seed(31);inputs=[torch.randn(2,2,tokens,6,generator=gen) for _ in range(4)]
    before=fp64_oracle(*inputs);o=Observation(tmp_path/'observed')
    prior=sys.gettrace()
    with o.active():after=fp64_oracle(*inputs)
    assert sys.gettrace() is prior
    return o,inputs,before,after


@pytest.mark.parametrize('capture',[9,129],indirect=True)
def test_actual_reference_chunks_include_partial_tile_all_heads_and_accumulations(capture):
    o,inputs,before,after=capture
    for n in before:torch.testing.assert_close(before[n],after[n],rtol=1e-12,atol=1e-14)
    r=audit_observation(o.folder,digest(o.folder/'index.json'),dict(zip(('q','k','v','cotangent'),[t.numpy() for t in inputs])))
    assert r['capture_integrity_verified'] and r['numerical_checks_passed'] and r['converted_inputs_exact']
    assert r['chunks']==(4 if inputs[0].shape[-2]==9 else 8) and r['first_captured_mismatch'] is None


def test_first_saved_stage_divergence_is_reported_without_waiving_gate(capture):
    o,inputs,_,_=capture;index=read_record(o.folder/'index.json');item=index['chunks'][0]['artifact']
    arrays={n:v.copy() for n,v in np.load(o.folder/item['file'],allow_pickle=False).items()};arrays['scores_centered'][0,0]+=.01
    index['chunks'][0]['artifact']=save_arrays(o.folder/'altered.npz',arrays);write_record(o.folder/'index.json',index,replace=True)
    report=audit_observation(o.folder,digest(o.folder/'index.json'),dict(zip(('q','k','v','cotangent'),[t.numpy() for t in inputs])))
    assert report['capture_integrity_verified'] and not report['numerical_checks_passed']
    assert report['first_captured_mismatch']['stage']=='scores_centered' and not report['first_mismatch_establishes_root_cause']


@pytest.mark.parametrize('defect',['missing_chunk','source_hash','index_checksum','file_bytes'])
def test_partial_or_altered_capture_cannot_be_accepted(capture,defect):
    o,inputs,_,_=capture;index=read_record(o.folder/'index.json');expected=digest(o.folder/'index.json')
    if defect=='missing_chunk':index['chunks'].pop()
    elif defect=='source_hash':index['oracle_file_sha256']='0'*64
    elif defect=='index_checksum':
        with (o.folder/'index.json').open('a') as f:f.write('changed')
    elif defect=='file_bytes':
        with (o.folder/index['chunks'][0]['artifact']['file']).open('ab') as f:f.write(b'changed')
    if defect in ('missing_chunk','source_hash'):
        write_record(o.folder/'index.json',index,replace=True);expected=digest(o.folder/'index.json')
    with pytest.raises(ValueError):audit_observation(o.folder,expected,dict(zip(('q','k','v','cotangent'),[t.numpy() for t in inputs])))


def test_exception_restores_existing_trace_state_and_keeps_partial_index(tmp_path,monkeypatch):
    import models.oracle_trace as module
    saved=module.save_arrays
    def failure(path,values):
        if Path(path).name.startswith('chunk'):raise RuntimeError('injected capture failure')
        return saved(path,values)
    monkeypatch.setattr(module,'save_arrays',failure)
    x=torch.ones(1,1,3,6);o=Observation(tmp_path/'failure');previous=sys.gettrace()
    with pytest.raises(RuntimeError,match='injected'):
        with o.active():fp64_oracle(x,x,x,x)
    assert sys.gettrace() is previous
    index=read_record(o.folder/'index.json');assert index['hook_restored'] and not index['complete']
    assert (o.folder/'converted.npz').is_file()


def test_existing_trace_hook_is_never_replaced(tmp_path):
    o=Observation(tmp_path/'already-traced');previous=sys.gettrace()
    def hook(frame,event,arg):return hook
    try:
        sys.settrace(hook)
        with pytest.raises(ValueError,match='Existing'):
            with o.active():pass
        assert sys.gettrace() is hook
    finally:sys.settrace(previous)


def test_nonfinite_derived_values_are_saved_before_numerical_refusal(tmp_path):
    item=save_arrays(tmp_path/'nan.npz',{'output':np.array([[float('nan')]])})
    values=load_arrays(tmp_path,item,{'output':[1,1]})
    assert np.isnan(values['output']).all() and item['arrays']['output']['finite'] is False
