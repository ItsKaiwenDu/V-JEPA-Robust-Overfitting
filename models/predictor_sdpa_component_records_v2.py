"""Bounded fixed-input references/returns with distinct completion identities."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat

MiB=2**20
RETURNS=('output','dq','dk','dv')
REFERENCE_KIND='native-SDPA-fixed-FP64-reference-v2'
WORKER_KIND='native-SDPA-fixed-input-component-v2'
META={'reference.json':128*1024,'worker.json':256*1024,'complete.json':16*1024,'failure.json':16*1024,'events.jsonl':8*MiB}
ARRAYS={n+'.npy':16*MiB for n in RETURNS}|{phase+'-'+n+'.npy':16*MiB for phase in ('untimed','profiled') for n in RETURNS}
LIMITS={**META,**ARRAYS}

def encoded(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def folder(p):
    p=Path(p);s=p.lstat()
    if not stat.S_ISDIR(s.st_mode) or s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode)!=0o700:
        raise ValueError('Private fixed-input directory required.')
    return p,(s.st_dev,s.st_ino)

def read(p,name):
    if name not in LIMITS:raise ValueError('Fixed bounded filename required.')
    p,identity=folder(p);path=p/name;s=path.lstat()
    if (not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode)!=0o600
            or s.st_dev!=identity[0] or not 0<=s.st_size<=LIMITS[name]):
        raise ValueError('Fixed-input file ownership/type/extent changed.')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        a=os.fstat(f.fileno());data=f.read(LIMITS[name]+1);b=os.fstat(f.fileno())
    if ((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)!=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
            or (a.st_size,a.st_mtime_ns,a.st_nlink,a.st_mode)!=(b.st_size,b.st_mtime_ns,b.st_nlink,b.st_mode) or len(data)!=s.st_size):
        raise ValueError('Fixed-input file changed during read.')
    return data

class Store:
    def __init__(self,path):
        self.path,self.identity=folder(path);self.files={};self.closed=False;self.poisoned=False
        if any(os.path.lexists(self.path/n) or os.path.lexists(self.path/(n+'.partial')) for n in LIMITS):
            raise FileExistsError('Preserve existing component/reference artifacts.')
    def check(self):
        if folder(self.path)[1]!=self.identity:raise ValueError('Fixed-input directory changed.')
        for n,item in self.files.items():
            data=read(self.path,n)
            if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Previous component/reference bytes changed.')
    def write(self,name,writer):
        if self.closed or self.poisoned or name not in LIMITS:raise ValueError('Single-use bounded fixed-input writer required.')
        self.check();dest=self.path/name;temp=self.path/(name+'.partial')
        if name not in self.files and os.path.lexists(dest):raise FileExistsError('Unexpected fixed-input destination exists.')
        try:
            with os.fdopen(os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:
                writer(f)
                if f.tell()>LIMITS[name]:raise ValueError('Fixed-input serialized byte ceiling.')
                f.flush();os.fsync(f.fileno())
            self.check()
            if name not in self.files and os.path.lexists(dest):raise FileExistsError('Fixed-input destination appeared during publication.')
            os.replace(temp,dest);fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
            item={'file':name,'bytes':dest.stat().st_size,'sha256':sha(dest)};self.files[name]=item
            return dict(item)
        except BaseException:self.poisoned=True;raise
    def json(self,name,value):return self.write(name,lambda f:f.write(encoded(value)))
    def array(self,name,value,*,dtype,shape):
        import numpy as np
        if name not in ARRAYS or name in self.files:raise ValueError('Each fixed-input array saved once.')
        a=np.asarray(value)
        if list(a.shape)!=shape or a.dtype!=np.dtype(dtype) or a.size*a.dtype.itemsize+4096>LIMITS[name]:
            raise ValueError('Exact bounded return dtype and shape required.')
        a=np.ascontiguousarray(a);item=self.write(name,lambda f:np.save(f,a,allow_pickle=False))
        item.update(dtype=a.dtype.str,shape=list(a.shape),logical_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            finite=bool(np.isfinite(a).all()),saved_before_numerical_gate=True)
        return item
    def finish(self,header,value,*,complete):
        h=self.json(header,value)
        self.json('complete.json' if complete else 'failure.json',{'kind':value['kind'],'plan_sha256':value['plan_sha256'],'header':h,'complete':complete})
        self.closed=True

def verified(p,item):
    if not isinstance(item,dict) or item.get('file') not in LIMITS:raise ValueError('Bounded artifact descriptor required.')
    data=read(p,item['file'])
    if len(data)!=item.get('bytes') or hashlib.sha256(data).hexdigest()!=item.get('sha256'):raise ValueError('Fixed-input artifact hash/extent mismatch.')
    return data

def array(p,item,*,dtype,shape,finite=True):
    import numpy as np
    if item.get('file') not in ARRAYS or item.get('dtype')!=np.dtype(dtype).str or item.get('shape')!=shape:
        raise ValueError('Exact reference/return array identity required.')
    data=verified(p,item);header=io.BytesIO(data);version=np.lib.format.read_magic(header)
    if version not in ((1,0),(2,0)):raise ValueError('Unsupported fixed-input array format.')
    declared,fortran,declared_dtype=(np.lib.format.read_array_header_1_0(header,max_header_size=4096) if version==(1,0)
        else np.lib.format.read_array_header_2_0(header,max_header_size=4096))
    expected=np.dtype(dtype)
    if list(declared)!=shape or fortran or declared_dtype!=expected or len(data)-header.tell()!=math.prod(shape)*expected.itemsize:
        raise ValueError('Array header/extent changed; no allocation from unverified shape.')
    value=np.load(io.BytesIO(data),allow_pickle=False,max_header_size=4096)
    isfinite=bool(np.isfinite(value).all())
    if (not value.flags.c_contiguous or value.dtype!=expected or hashlib.sha256(value.tobytes()).hexdigest()!=item['logical_sha256']
            or isfinite is not item.get('finite') or finite and not isfinite or item.get('saved_before_numerical_gate') is not True):
        raise ValueError('Actual logical array bytes or finite status changed.')
    return value

def load(p,*,kind,plan_sha256,require_complete=True):
    if kind not in (REFERENCE_KIND,WORKER_KIND):raise ValueError('Known distinct component/reference format required.')
    p=Path(p);name='reference.json' if kind==REFERENCE_KIND else 'worker.json'
    raw=read(p,name);w=json.loads(raw)
    if w.get('kind')!=kind or w.get('plan_sha256')!=plan_sha256 or type(w.get('complete')) is not bool:
        raise ValueError('Separate fixed-input reference/worker identity required.')
    done=w['complete'];marker=json.loads(read(p,'complete.json' if done else 'failure.json'))
    if marker!={'kind':kind,'plan_sha256':plan_sha256,'header':{'file':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},'complete':done}:
        raise ValueError('Fixed-input final marker mismatch.')
    if done and os.path.lexists(p/'failure.json'):raise ValueError('Failed fixed-input record cannot qualify completion.')
    if not done and os.path.lexists(p/'complete.json'):raise ValueError('Partial fixed-input record has an unexpected completion marker.')
    if done and any(os.path.lexists(p/(name+'.partial')) for name in LIMITS):raise ValueError('Unfinished component/reference publication retained; no completion.')
    if any(w.get(k) is not False for k in ('research_admitted','performance_qualified','full_model_candidate_adopted','pure_backward_arithmetic_isolated')):
        raise ValueError('Component/reference cannot adopt full model or claim isolated backward arithmetic.')
    if require_complete and not done:raise ValueError('Complete passing fixed-input record required; partials are not references.')
    return w
