"""Observe the frozen CPU oracle at Python boundaries without editing its math."""
from contextlib import contextmanager
import ctypes
import hashlib
import inspect
import io
import math
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

from models.attention_component import fp64_oracle
from models.attention_fixture import BOOKKEEPING
from models.attention_saved_evidence import agreement
from models.pilot_bundle import digest
from models.study_artifacts import read_record,write_record

KIND='cpu-oracle-line-observation-v1'
STAGES=('scores_centered','probability','output','dprobability','dscores','dq','dk_accumulated','dv_accumulated')
MAX_ARRAY_FILE=32*2**20


def signature(array):
    array=np.ascontiguousarray(array,dtype='<f8');finite=bool(np.isfinite(array).all())
    return {'shape':list(array.shape),'dtype':array.dtype.str,'sha256':hashlib.sha256(array.tobytes()).hexdigest(),
            'finite':finite,'max_abs':float(np.abs(array).max()) if finite and array.size else None}


def save_arrays(path,values):
    """Persist computed values before any finite/numerical acceptance gate."""
    path=Path(path);temporary=path.with_name(path.name+'.partial')
    if path.exists() or temporary.exists():raise FileExistsError(path)
    if any(isinstance(v,torch.Tensor) and v.device.type!='cpu' for v in values.values()):raise ValueError('Oracle capture refuses non-CPU tensors.')
    source_dtypes={n:str(v.dtype) for n,v in values.items()}
    arrays={n:np.array(v.detach().numpy() if isinstance(v,torch.Tensor) else v,dtype='<f8',order='C',copy=True) for n,v in values.items()}
    if sum(a.nbytes for a in arrays.values())>MAX_ARRAY_FILE-65536:raise ValueError('Oracle capture size bound exceeded.')
    with temporary.open('xb') as f:
        np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    os.link(temporary,path);temporary.unlink()
    return {'file':path.name,'sha256':digest(path),'bytes':path.stat().st_size,'source_dtypes':source_dtypes,
            'arrays':{n:signature(a) for n,a in arrays.items()}}


def load_arrays(folder,item,shapes):
    path=Path(folder)/item['file']
    if Path(item['file']).name!=item['file'] or path.is_symlink() or not path.is_file() or not 0<item['bytes']<=MAX_ARRAY_FILE:
        raise ValueError('Invalid bounded oracle capture path.')
    data=path.read_bytes()
    if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Oracle capture checksum changed.')
    result={}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries=archive.infolist()
        if len(entries)!=len(shapes) or {i.filename for i in entries}!={n+'.npy' for n in shapes}:raise ValueError('Capture array keys changed.')
        for name,shape in shapes.items():
            entry=archive.getinfo(name+'.npy');size=int(np.prod(shape))*8
            if not size<=entry.file_size<=size+1024:raise ValueError('Unexpected array extent in oracle capture.')
            raw=archive.read(entry);header=io.BytesIO(raw);version=np.lib.format.read_magic(header)
            if version not in ((1,0),(2,0)):raise ValueError('Unsupported captured NumPy version.')
            declared,fortran,dtype=(np.lib.format.read_array_header_1_0(header) if version==(1,0) else np.lib.format.read_array_header_2_0(header))
            if list(declared)!=list(shape) or fortran or dtype!=np.dtype('<f8') or len(raw)-header.tell()!=size:
                raise ValueError('Captured array shape/dtype changed.')
            array=np.load(io.BytesIO(raw),allow_pickle=False)
            if signature(array)!=item['arrays'][name]:raise ValueError('Captured array logical identity changed.')
            result[name]=array
    return result


def cpu_location():
    result={}
    for name in ('sched_getcpu','fegetround'):
        try:
            function=getattr(ctypes.CDLL(None),name);function.argtypes=[];function.restype=ctypes.c_int
            result[name]=int(function())
        except (AttributeError,OSError):result[name]=None
    return result


def _lines():
    lines,first=inspect.getsourcelines(fp64_oracle)
    tokens={'converted':'out=torch.empty_like(q);','scores':'prob=scores.exp();',
            'probability':'out[b,h,start:stop]=','output':'dp=dy@v[b,h].T;',
            'dscores':'dq[b,h,start:stop]=','dq':'dk[b,h].add_',
            'flush':'for start in range('}
    mapping={}
    for name,token in tokens.items():
        matches=[first+i for i,line in enumerate(lines) if line.strip().startswith(token)]
        if len(matches)!=1:raise ValueError('Frozen oracle boundary layout changed: '+name)
        mapping[matches[0]]=name
    return mapping


class Observation:
    def __init__(self,folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.lines=_lines();self.pending={};self.key=None
        self.record={'kind':KIND,'oracle_file_sha256':digest(Path(inspect.getsourcefile(fp64_oracle))),
                     'complete':False,'hook_restored':False,'chunks':[],'CPU_only':True,'observer_may_perturb_execution':True}
    def checkpoint(self):
        write_record(self.folder/'index.json',self.record,replace=(self.folder/'index.json').exists())
    def trace(self,frame,event,arg):
        if frame.f_code is not fp64_oracle.__code__:return None
        if event!='line':return self.trace
        stage=self.lines.get(frame.f_lineno);v=frame.f_locals
        if stage=='converted':
            self.record['converted']=save_arrays(self.folder/'converted.npz',{n:v[n] for n in ('q','k','v','g')});self.checkpoint()
        elif stage=='scores':
            self.key=[v['b'],v['h'],v['start'],v['stop']];self.pending={'scores_centered':v['scores']}
        elif stage=='probability':self.pending['probability']=v['prob']
        elif stage=='output':self.pending['output']=v['out'][v['b'],v['h'],v['start']:v['stop']]
        elif stage=='dscores':self.pending.update(dprobability=v['dp'],dscores=v['ds'])
        elif stage=='dq':self.pending['dq']=v['dq'][v['b'],v['h'],v['start']:v['stop']]
        elif stage=='flush' and self.pending:
            self.pending.update(dk_accumulated=v['dk'][v['b'],v['h']],dv_accumulated=v['dv'][v['b'],v['h']])
            if set(self.pending)!=set(STAGES):raise ValueError('Incomplete actual oracle observation.')
            item=save_arrays(self.folder/('chunk-%03d.npz'%len(self.record['chunks'])),self.pending)
            self.record['chunks'].append({'key':self.key,'location':cpu_location(),'artifact':item});self.pending={};self.key=None;self.checkpoint()
        return self.trace
    @contextmanager
    def active(self):
        previous=sys.gettrace()
        if previous is not None:raise ValueError('Existing Python trace hook must be preserved; refuse observation.')
        try:
            sys.settrace(self.trace);yield self
            if self.pending:raise ValueError('Oracle returned with an unfinished captured chunk.')
            self.record['complete']=True
        finally:
            sys.settrace(previous);self.record['hook_restored']=sys.gettrace() is previous;self.checkpoint()


def safe_agreement(actual,expected):
    try:return agreement(actual,expected,BOOKKEEPING)
    except ValueError as exc:return {'passed':False,'error':str(exc),**BOOKKEEPING}


def stage_shapes(tokens,dim,start,stop):
    return {n:[stop-start,tokens] if n in ('scores_centered','probability','dprobability','dscores')
            else [tokens,dim] if n in ('dk_accumulated','dv_accumulated') else [stop-start,dim] for n in STAGES}


def audit_observation(folder,expected_sha,inputs):
    """Rebuild stage values with NumPy; a divergence is evidence, never waived."""
    folder=Path(folder);index=read_record(folder/'index.json',expected_sha256=expected_sha)
    if (index['kind']!=KIND or not index['complete'] or not index['hook_restored'] or not index['CPU_only']
            or index['oracle_file_sha256']!=digest(Path(inspect.getsourcefile(fp64_oracle)))):
        raise ValueError('Incomplete or unrestored oracle observation.')
    q,k,v,g=[np.asarray(inputs[n],dtype=np.float64) for n in ('q','k','v','cotangent')]
    shape=list(q.shape);converted=load_arrays(folder,index['converted'],{n:shape for n in ('q','k','v','g')})
    if any(signature(converted[n])!=signature(t) for n,t in zip(('q','k','v','g'),(q,k,v,g))):raise ValueError('FP32-to-FP64 conversion differs from fixed inputs.')
    order=[[b,h,start,min(start+128,shape[-2])] for b in range(shape[0]) for h in range(shape[1]) for start in range(0,shape[-2],128)]
    if [r['key'] for r in index['chunks']]!=order:raise ValueError('Missing, duplicate or reordered actual oracle chunks.')
    checks=[];first=None;dk=np.zeros_like(k);dv=np.zeros_like(v);scale=1/math.sqrt(shape[-1])
    for item in index['chunks']:
        b,h,start,stop=item['key'];x=q[b,h,start:stop];dy=g[b,h,start:stop]
        scores=(x@k[b,h].T)*scale;scores=scores-scores.max(axis=-1,keepdims=True)
        prob=np.exp(scores);prob=prob/prob.sum(axis=-1,keepdims=True)
        dp=dy@v[b,h].T;ds=prob*(dp-(dp*prob).sum(axis=-1,keepdims=True))
        dk[b,h]+=(ds.T@x)*scale;dv[b,h]+=prob.T@dy
        expected={'scores_centered':scores,'probability':prob,'output':prob@v[b,h],
                  'dprobability':dp,'dscores':ds,'dq':(ds@k[b,h])*scale,'dk_accumulated':dk[b,h],'dv_accumulated':dv[b,h]}
        actual=load_arrays(folder,item['artifact'],stage_shapes(shape[-2],shape[-1],start,stop))
        result={n:safe_agreement(actual[n],expected[n]) for n in STAGES}
        checks.append({'key':item['key'],'comparisons':result})
        for name in STAGES:
            if first is None and not result[name]['passed']:first={'key':item['key'],'stage':name,'comparison':result[name]}
    return {'kind':KIND,'capture_integrity_verified':True,'converted_inputs_exact':True,'chunks':len(order),
            'numerical_checks_passed':first is None,'first_captured_mismatch':first,'stage_comparisons':checks,
            'first_mismatch_establishes_root_cause':False,'bounds':BOOKKEEPING}
