"""New tiny-array and saved-metadata checks; no model or actual cohort reruns."""
import ast
import copy
import io
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys

import numpy as np
import pytest
from models import predictor_pgd_component_returns_v1 as review

ROOT=Path(__file__).resolve().parents[1]
SHAPE=(1,1,2,2)
ORIGIN='EXPLICIT TINY SYNTHETIC ARRAYS OR MODIFIED SAVED METADATA; NO EXECUTION ADMISSION'


@pytest.fixture(scope='module',autouse=True)
def no_live_operations():
    patch=pytest.MonkeyPatch()
    def forbidden(*a,**k):pytest.fail('Saved-return checks cannot start process/GPU/signal/timer/network work or use pickle loaders.')
    for mod,name in [(subprocess,'Popen'),(subprocess,'check_output'),(os,'system'),(os,'fork'),(os,'posix_spawn'),(os,'kill'),(os,'killpg'),(os,'pidfd_open'),(signal,'setitimer'),(signal,'alarm'),(signal,'signal'),(socket,'socket'),(np,'load')]:
        if hasattr(mod,name):patch.setattr(mod,name,forbidden)
    yield
    assert 'torch' not in sys.modules
    patch.undo()


def record(folder,name,value):
    with (folder/name).open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False)


def envelope(value,*,header=None):
    if header is None:
        output=io.BytesIO();np.save(output,value,allow_pickle=False);return output.getvalue()
    text=repr(header).encode('ascii');length=len(text)+1;length+=(-10-length)%64
    return b'\x93NUMPY\x01\x00'+length.to_bytes(2,'little')+text+b' '*(length-len(text)-1)+b'\n'+value.tobytes()


def parse(raw):
    length=int.from_bytes(raw[8:10],'little');return review.decode_npy(raw,shape=SHAPE,payload_sha256=review.digest(raw[10+length:]))


@pytest.fixture(scope='module')
def context():
    c=json.loads((ROOT/review.INPUTS).read_bytes());raws={name:review.read_bound(ROOT/name,c['files'][name]) for name in c['metadata'].values()}
    assert not any(name.endswith(('.npy','.pt')) for name in raws)
    return c,raws


def test_frozen_math_and_tolerance_AST():
    own=ast.parse((ROOT/'models/predictor_pgd_component_returns_v1.py').read_bytes());gold=ast.parse((ROOT/review.AGREEMENT_SOURCE).read_bytes());bound=ast.parse((ROOT/review.BOUNDS_SOURCE).read_bytes())
    find=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='agreement')
    pick=lambda tree:next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BOUNDS' for t in n.targets))
    assert ast.dump(find(own),include_attributes=False)==ast.dump(find(gold),include_attributes=False)
    assert ast.dump(pick(own),include_attributes=False)==ast.dump(pick(bound),include_attributes=False)


def test_saved_metadata_join_preserves_failed_worker(context,tmp_path):
    c,raws=context;before=copy.deepcopy(raws);result=review.validate_context(c,raws)
    assert result['native_exact_historical_logical_returns'] and result['old_failed_worker_still_failed']
    assert result['identities']['native']['historical_worker_complete'] is True and result['identities']['forward-preserving']['historical_worker_complete'] is False
    assert raws==before
    record(tmp_path,'CONTEXT.json',{'origin':'SAVED METADATA ONLY; NO ARRAY VALUES DECODED','result':result})


@pytest.mark.parametrize('case',['source-control','candidate-role','lost-failure','wrong-input','foreign-output','metadata-promotes-failure'])
def test_mismatched_metadata_refuses(context,tmp_path,case):
    c,raws=context;values=dict(raws)
    alias={'source-control':'manifest','candidate-role':'forward_preserving_worker','lost-failure':'candidate_failed','wrong-input':'forward_preserving_ownership','foreign-output':'native_ownership','metadata-promotes-failure':'candidate_assessment'}[case]
    path=c['metadata'][alias];v=json.loads(values[path])
    if case=='source-control':v['source_control_sha256']='f'*64
    elif case=='candidate-role':v['role']='native'
    elif case=='lost-failure':v['complete']=True
    elif case=='wrong-input':v['calls'][0]['inputs'][0]['sha256']='f'*64
    elif case=='foreign-output':v['calls'][0]['returned_component'][0]['sha256']='f'*64
    else:v['proof']['worker_completion_inferred']=True
    values[path]=review.encoded(v)
    with pytest.raises(ValueError) as error:review.validate_context(c,values)
    record(tmp_path,'REFUSAL.json',{'origin':ORIGIN,'case':case,'error':str(error.value)})


def test_private_file_pin(tmp_path):
    p=tmp_path/'inert.bin';p.write_bytes(b'INERT BYTES');d={'bytes':p.stat().st_size,'sha256':review.digest(p.read_bytes())}
    assert review.read_bound(p,d)==b'INERT BYTES'


@pytest.mark.parametrize('case',['symlink','changed-bytes','fake-FIFO','fake-multiple-links'])
def test_file_identity_refuses_before_unsafe_read(tmp_path,monkeypatch,case):
    p=tmp_path/'inert.bin';p.write_bytes(b'INERT BYTES');d={'bytes':p.stat().st_size,'sha256':review.digest(p.read_bytes())}
    if case=='symlink':
        q=tmp_path/'link.bin';q.symlink_to(p);p=q
    elif case=='changed-bytes':d['sha256']='f'*64
    else:
        previous=Path.lstat
        def lstat(path,*a,**k):
            value=previous(path,*a,**k)
            if path==p:
                fields=list(value)
                if case=='fake-FIFO':fields[0]=stat.S_IFIFO|0o600
                else:fields[3]=2
                return os.stat_result(fields)
            return value
        monkeypatch.setattr(Path,'lstat',lstat)
        monkeypatch.setattr(review.os,'open',lambda *a,**k:pytest.fail('Unsafe special/multilink file must never open.'))
    with pytest.raises(ValueError) as error:review.read_bound(p,d)
    record(tmp_path,'REFUSAL.json',{'origin':ORIGIN,'case':case,'error':str(error.value)})


def test_safe_npy_roundtrip_readonly(tmp_path):
    value=np.arange(1,5,dtype='<f4').reshape(SHAPE);raw=envelope(value);actual,meta=parse(raw)
    assert np.array_equal(value,actual) and not actual.flags.writeable and meta['shape']==list(SHAPE)
    record(tmp_path,'NPY-SUMMARY.json',{'origin':ORIGIN,'metadata':meta})


@pytest.mark.parametrize('case',['wrong-dtype','fortran','wrong-shape','boolean-shape','object-dtype','trailing','truncated','duplicate-header'])
def test_malformed_npy_refuses(tmp_path,case):
    value=np.ones(SHAPE,dtype='<f4');header={'descr':'<f4','fortran_order':False,'shape':SHAPE}
    if case=='wrong-dtype':header['descr']='<f8'
    elif case=='fortran':header['fortran_order']=True
    elif case=='wrong-shape':header['shape']=(1,2,1,2)
    elif case=='boolean-shape':header['shape']=(True,1,2,2)
    elif case=='object-dtype':header['descr']='|O'
    raw=envelope(value,header=header)
    if case=='trailing':raw+=b'EXTRA'
    elif case=='truncated':raw=raw[:-1]
    elif case=='duplicate-header':
        text=b"{'descr':'<f4','fortran_order':False,'shape':(1,1,2,2),'descr':'<f4'}";length=len(text)+1;length+=(-10-length)%64
        raw=b'\x93NUMPY\x01\x00'+length.to_bytes(2,'little')+text+b' '*(length-len(text)-1)+b'\n'+value.tobytes()
    with pytest.raises(ValueError) as error:parse(raw)
    record(tmp_path,'REFUSAL.json',{'origin':ORIGIN,'case':case,'error':str(error.value)})


@pytest.mark.parametrize('value',[float('nan'),float('inf'),0.])
def test_nonfinite_or_all_zero_npy_refuses(tmp_path,value):
    raw=envelope(np.full(SHAPE,value,dtype='<f4'))
    with pytest.raises(ValueError) as error:parse(raw)
    record(tmp_path,'REFUSAL.json',{'origin':ORIGIN,'error':str(error.value)})


@pytest.mark.parametrize('value,passed',[(1.,True),(1.001,True),(1.004,False)])
def test_original_elementwise_gate_is_retained(tmp_path,value,passed):
    expected=np.ones(SHAPE,dtype='<f4');actual=np.full(SHAPE,value,dtype='<f4');result=review.agreement(actual,expected,review.BOUNDS['dq'])
    assert result['passed'] is passed and (result['failed_elements']==0) is passed
    assert (result['max_tolerance_ratio']<=1.) is passed and result['atol']==2e-7 and result['rtol']==.002
    record(tmp_path,'NUMERICAL-FIXTURE.json',{'origin':ORIGIN,'result':result})
