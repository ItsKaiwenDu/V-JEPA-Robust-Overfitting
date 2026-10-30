"""Immutable synthetic SDPA operands and fixed FP64 references, never weights."""
import io
import json
import os
from pathlib import Path

import numpy as np
import torch

from models.attention_component import CASES,TOLERANCES
from models.attention_saved_evidence import oracle as numpy_oracle
from models.cuda_profile import tensor_signature
from models.pilot_bundle import digest,safe_name
from models.study_artifacts import read_record,write_record

KIND='immutable-attention-fixture-v1'
DIAGNOSTIC_KIND='fixed-input-fp32-attention-diagnostic-v1'
BOOKKEEPING={'rtol':1e-12,'atol':1e-14}
INPUTS=('q','k','v','cotangent')
OUTPUTS=('output','dq','dk','dv')
KEYS=INPUTS+tuple('oracle_'+n for n in OUTPUTS)
MAX_FILE=64*2**20


def geometry(case,tiny):
    if case not in CASES:raise ValueError('Unknown fixed fixture case.')
    heads,dim=(2,12) if tiny else (12,32) if case=='predictor' else (16,64)
    tokens=(16 if case=='predictor' else 12 if case=='encoder-0' else 8) if tiny else (2048 if case=='predictor' else 656 if case=='encoder-0' else 400)
    contiguous=[heads*tokens*dim,tokens*dim,dim,1]
    return {'shape':[1,heads,tokens,dim], 'strides':{'q':contiguous,'k':contiguous,
            'v':[3*heads*tokens*dim,dim,3*heads*dim,1]}}


def freeze(folder,case,tensors,shape,origin,*,tiny):
    """Capture once; no runtime regeneration or fallback is part of this format."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    expected=geometry(case,tiny)
    if len(tensors)!=4 or any(t.dtype!=torch.float32 or t.device.type!='cpu' or list(t.shape)!=expected['shape'] for t in tensors):
        raise ValueError('Wrong fixed synthetic operand shape or dtype.')
    if shape['strides']!=expected['strides']:raise ValueError('Unexpected post-RoPE operand strides.')
    values=dict(zip(INPUTS,tensors));reference=numpy_oracle(*[t.numpy() for t in tensors])
    values.update({'oracle_'+n:torch.from_numpy(reference[n]) for n in OUTPUTS})
    files={}
    for name,t in values.items():
        if not torch.isfinite(t).all():raise ValueError('Nonfinite fixed fixture.')
        # NumPy writes logical values, not a platform-specific Tensor storage.
        array=np.asarray(t.numpy(),dtype='<f4' if name in INPUTS else '<f8',order='C')
        path=folder/(name+'.npy')
        with path.open('xb') as f:np.save(f,array,allow_pickle=False)
        files[name]={'file':path.name,'bytes':path.stat().st_size,'sha256':digest(path),'tensor':tensor_signature(t)}
    record={'kind':KIND,'schema_version':1,'case':case,'tiny_fixture':tiny,'shape_metadata':shape,'origin':origin,
        'synthetic_only':True,'released_weights_or_pixels':False,'research_admitted':False,
        'generator_replay_qualified':False,'runtime_generation_allowed':False,'tolerances':TOLERANCES,
        'bookkeeping':BOOKKEEPING,'oracle':'fixed-numpy-fp64-query-chunked-softmax-analytical-vjp-v1','files':files}
    write_record(folder/'fixture.json',record)
    return {'descriptor':'fixture.json','sha256':digest(folder/'fixture.json')}


def load(folder,sha,*,case,tiny):
    folder=Path(folder).resolve();record=read_record(folder/'fixture.json',expected_sha256=sha)
    if (record.get('kind')!=KIND or record.get('schema_version')!=1 or record.get('case')!=case or record.get('tiny_fixture')!=tiny
            or record.get('synthetic_only') is not True or record.get('released_weights_or_pixels') is not False
            or record.get('research_admitted') is not False or record.get('runtime_generation_allowed') is not False
            or record.get('generator_replay_qualified') is not False or record.get('tolerances')!=TOLERANCES
            or record.get('bookkeeping')!=BOOKKEEPING
            or record.get('oracle')!='fixed-numpy-fp64-query-chunked-softmax-analytical-vjp-v1'
            or set(record.get('files',{}))!=set(KEYS)):
        raise ValueError('Wrong immutable fixture domain, keys or fixed numerical contract.')
    expected=geometry(case,tiny);shape=record['shape_metadata']
    if (shape.get('case')!=case or shape.get('tiny_fixture')!=tiny or shape.get('strides')!=expected['strides']
            or [1,shape.get('heads'),shape.get('tokens'),shape.get('head_dim')]!=expected['shape']):
        raise ValueError('Fixed fixture geometry/layout changed.')
    result={}
    for name,item in record['files'].items():
        path=folder/safe_name(item['file'])
        if (item['file']!=name+'.npy' or path.is_symlink() or not path.is_file()
                or not 0<item['bytes']<=MAX_FILE or path.stat().st_size!=item['bytes']):
            raise ValueError('Require bounded regular immutable NumPy arrays.')
        # Hash exactly the bytes parsed, eliminating a hash/load file race.
        data=path.read_bytes()
        import hashlib
        if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Immutable operand/reference checksum changed: '+name)
        header=io.BytesIO(data);version=np.lib.format.read_magic(header)
        if version not in ((1,0),(2,0)):raise ValueError('Unsupported immutable NumPy header.')
        declared,fortran,dtype=(np.lib.format.read_array_header_1_0(header) if version==(1,0) else np.lib.format.read_array_header_2_0(header))
        wanted=np.dtype('<f4' if name in INPUTS else '<f8')
        if list(declared)!=expected['shape'] or fortran or dtype!=wanted or len(data)-header.tell()!=int(np.prod(declared))*wanted.itemsize:
            raise ValueError('Immutable NumPy header/extent differs from the declared shape/dtype.')
        array=np.load(io.BytesIO(data),allow_pickle=False)
        if (array.dtype!=np.dtype('<f4' if name in INPUTS else '<f8') or list(array.shape)!=expected['shape']
                or not array.flags.c_contiguous or not np.isfinite(array).all()):raise ValueError('Invalid immutable array dtype/shape/value.')
        tensor=torch.from_numpy(array.copy())
        if tensor_signature(tensor)!=item['tensor']:raise ValueError('Immutable logical tensor signature changed.')
        if name in ('q','k','v'):
            stride=shape['strides'][name];count=1+sum((n-1)*s for n,s in zip(expected['shape'],stride))
            storage=torch.zeros(count,dtype=torch.float32);view=storage.as_strided(expected['shape'],stride);view.copy_(tensor);tensor=view
        result[name]=tensor
    return result,record


def snapshot(folder,name,values):
    """Commit a CPU tensor snapshot before checks that may reject the worker."""
    folder=Path(folder);path=folder/(name+'.pt');temporary=folder/(name+'.pt.partial')
    if path.exists() or temporary.exists():raise FileExistsError(path)
    cpu={n:t.detach().cpu() for n,t in values.items()}
    with temporary.open('xb') as f:
        torch.save(cpu,f);f.flush();os.fsync(f.fileno())
    os.link(temporary,path);temporary.unlink()
    return {'file':path.name,'sha256':digest(path),'tensor_signatures':{n:tensor_signature(t) for n,t in cpu.items()}}
