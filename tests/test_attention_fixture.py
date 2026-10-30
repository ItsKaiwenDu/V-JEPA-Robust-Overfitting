"""Exact portable inputs, bounded NumPy loading and retained early evidence."""
from pathlib import Path

import numpy as np
import pytest
import torch

from models.attention_component import CASES,operands
from models.attention_fixture import BOOKKEEPING,INPUTS,KEYS,freeze,load,snapshot
from models.cuda_profile import tensor_signature
from models.pilot_bundle import digest
from models.study_artifacts import read_record,write_record


@pytest.fixture
def fixture(tmp_path,upstream):
    tensors,shape=operands(upstream,'encoder-0',{},True);shape=dict(shape)
    shape['position_signature']=tensor_signature(shape.pop('positions'))
    folder=tmp_path/'fixture';item=freeze(folder,'encoder-0',tensors,shape,{'kind':'explicit-test-only-capture'},tiny=True)
    return folder,item,tensors


def test_loader_preserves_values_and_non_dense_layout_without_regeneration(fixture,monkeypatch):
    folder,item,original=fixture
    def forbidden(*args,**kwargs):raise AssertionError('Runtime regeneration must never run')
    monkeypatch.setattr('models.attention_component.operands',forbidden)
    a,record=load(folder,item['sha256'],case='encoder-0',tiny=True)
    b,_=load(folder,item['sha256'],case='encoder-0',tiny=True)
    assert set(a)==set(KEYS) and not record['runtime_generation_allowed'] and not record['generator_replay_qualified']
    for name,t in zip(INPUTS,original):
        assert tensor_signature(a[name])==tensor_signature(t)==tensor_signature(b[name])
        assert a[name].stride()==t.stride()
    assert all(tensor_signature(a[n])==tensor_signature(b[n]) for n in KEYS)
    assert record['bookkeeping']==BOOKKEEPING


@pytest.mark.parametrize('key',['q','cotangent','oracle_dv'])
def test_changed_or_missing_array_never_regenerates(fixture,key):
    folder,item,_=fixture;path=folder/(key+'.npy')
    with path.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError):load(folder,item['sha256'],case='encoder-0',tiny=True)
    path.unlink()
    with pytest.raises(ValueError):load(folder,item['sha256'],case='encoder-0',tiny=True)


@pytest.mark.parametrize('defect',['shape','object','fortran','double','huge_header'])
def test_resigned_bad_numpy_header_rejects_before_tensor_allocation(fixture,defect):
    folder,item,_=fixture;path=folder/'q.npy';r=read_record(folder/'fixture.json')
    with path.open('wb') as f:
        if defect=='huge_header':
            np.lib.format.write_array_header_1_0(f,{'descr':'<f4','fortran_order':False,'shape':(1,2,10**12,12)})
        else:
            array=(np.zeros((1,2,11,12),dtype=np.float32) if defect=='shape' else np.zeros((1,2,12,12),dtype=object if defect=='object' else np.float64 if defect=='double' else np.float32))
            if defect=='fortran':array=np.asfortranarray(array)
            np.save(f,array,allow_pickle=defect=='object')
    r['files']['q'].update(bytes=path.stat().st_size,sha256=digest(path));write_record(folder/'fixture.json',r,replace=True)
    with pytest.raises(ValueError):load(folder,digest(folder/'fixture.json'),case='encoder-0',tiny=True)


@pytest.mark.parametrize('field,value',[('runtime_generation_allowed',True),('tiny_fixture',False),('tolerances',{}),('generator_replay_qualified',True)])
def test_fixture_domain_and_bounds_cannot_be_relabelled(fixture,field,value):
    folder,item,_=fixture;r=read_record(folder/'fixture.json');r[field]=value;write_record(folder/'fixture.json',r,replace=True)
    with pytest.raises(ValueError):load(folder,digest(folder/'fixture.json'),case='encoder-0',tiny=True)


def test_atomic_input_snapshot_is_immutable_and_logically_complete(fixture,tmp_path):
    folder,item,_=fixture;values,_=load(folder,item['sha256'],case='encoder-0',tiny=True)
    meta=snapshot(tmp_path,'inputs',{n:values[n] for n in INPUTS})
    saved=torch.load(tmp_path/'inputs.pt',weights_only=True)
    assert meta['sha256']==digest(tmp_path/'inputs.pt') and set(saved)==set(INPUTS)
    assert meta['tensor_signatures']=={n:tensor_signature(values[n]) for n in INPUTS}
    assert not (tmp_path/'inputs.pt.partial').exists()
    with pytest.raises(FileExistsError):snapshot(tmp_path,'inputs',saved)
