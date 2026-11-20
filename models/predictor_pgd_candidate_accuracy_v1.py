"""Native-relative candidate return check; NumPy loads only inside a future worker.

Parent validation checks saved receipt fields and byte joins, not numerical values.
Execution, canonical FP64, full-model/replay/performance/training admission stay false.
"""
import ast
import hashlib
import math
from pathlib import Path
from models import predictor_pgd_candidate_reference_v1 as reference
from models import predictor_pgd_component_capsule_v1 as capsule

RETURNS=('output','dq','dk','dv')
SHAPE=(1,12,2048,32)
MAX_FILE=4*2**20
BOUNDS = {'output': {'rtol': 1e-4, 'atol': 2e-5},
          **{name: {'rtol': .002, 'atol': 2e-7} for name in ('dq', 'dk', 'dv')}}
FIELDS=frozenset(('passed','failed_elements','shape','max_absolute_error','max_tolerance_ratio','relative_l2_error','rtol','atol'))
KIND='pgd-candidate-native-relative-accuracy-v1'


def digest(raw):return hashlib.sha256(raw).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def validate_receipt(value,binding,outputs,description):
    reference.verify_description(description,binding)
    require(type(value) is dict and set(value)=={'kind','binding','reference','outputs','checks','bounds','forward_logical_bytes_exact','canonical_FP64_accuracy_qualified','worker_completion_or_scientific_admission'},'Exact native-relative accuracy receipt required.')
    require(value['kind']==KIND and value['binding']==binding and value['reference']==description and value['outputs']==outputs
        and value['bounds']==BOUNDS and value['forward_logical_bytes_exact'] is True
        and value['canonical_FP64_accuracy_qualified'] is False and value['worker_completion_or_scientific_admission'] is False,'Fresh candidate/native reference identity and forward exactness required.')
    require(type(outputs) is dict and set(outputs)==set(RETURNS) and type(value['checks']) is dict and set(value['checks'])==set(RETURNS),'All four candidate/native comparisons required.')
    for name,c in value['checks'].items():
        require(type(c) is dict and set(c)==FIELDS and c['passed'] is True and type(c['failed_elements']) is int and c['failed_elements']==0
            and c['shape']==list(SHAPE) and type(c['shape']) is list and all(type(n) is int for n in c['shape']),'Original elementwise acceptance and shape required.')
        require({k:c[k] for k in ('rtol','atol')}==BOUNDS[name] and all(type(c[k]) is float and math.isfinite(c[k]) and c[k]>=0 for k in ('rtol','atol','max_absolute_error','max_tolerance_ratio','relative_l2_error'))
            and c['max_tolerance_ratio']<=1.,'Original finite numerical bounds cannot change.')
        if name=='output':require(c['max_absolute_error']==c['max_tolerance_ratio']==c['relative_l2_error']==0.,'Candidate forward values must remain exact.')
    require(len(reference.encoded(value))<=16*1024,'Accuracy receipt exceeds fixed 16 KiB metadata bound.')
    return value


def check_values(binding,outputs,folder,source):
    global np
    import numpy as np
    require(np.__version__=='2.4.6','Original NumPy runtime required; no fallback.')
    description=reference.expected_description(binding);source.check();native=source.returns();checks={}
    require(type(outputs) is dict and set(outputs)==set(RETURNS),'Four published candidate returns required.')
    forward_exact=False
    for name in RETURNS:
        d=outputs[name];require(type(d) is dict and set(d)=={'file','bytes','sha256'} and d['file']==name+'.npy'
            and type(d['bytes']) is int and 0<d['bytes']<=MAX_FILE,'Fixed bounded candidate return descriptor required.')
        raw=capsule._read(Path(folder)/d['file'],MAX_FILE,d['sha256']);require(len(raw)==d['bytes'],'Candidate return extent changed.')
        logical=reference.logical_npy(raw)
        actual,_=decode_npy(raw,shape=SHAPE,payload_sha256=logical)
        expected,_=decode_npy(native[name],shape=SHAPE,payload_sha256=description['logical_sha256'][name])
        checks[name]=agreement(actual,expected,BOUNDS[name])
        if name=='output':forward_exact=logical==description['logical_sha256'][name]
    source.check()
    receipt={'kind':KIND,'binding':dict(binding),'reference':description,'outputs':outputs,'checks':checks,'bounds':BOUNDS,
        'forward_logical_bytes_exact':forward_exact,'canonical_FP64_accuracy_qualified':False,'worker_completion_or_scientific_admission':False}
    return validate_receipt(receipt,binding,outputs,description)


def decode_npy(raw,*,shape,payload_sha256):
    """Exact NPY v1 little-endian contiguous FP32; no pickle loader or mmap."""
    require(type(raw) is bytes and 10<len(raw)<=MAX_FILE and raw[:8]==b'\x93NUMPY\x01\x00','Exact bounded NPY v1 envelope required.')
    length=int.from_bytes(raw[8:10],'little');end=10+length
    require(0<length<=4096 and end<=len(raw) and end%16==0 and raw[end-1:end]==b'\n','Bounded complete aligned NPY header required.')
    tree=ast.parse(raw[10:end].decode('ascii').strip(),mode='eval')
    require(isinstance(tree.body,ast.Dict) and len(tree.body.keys)==3,'Exact three-field NPY header required.')
    keys=[ast.literal_eval(k) for k in tree.body.keys]
    require(len(set(keys))==3 and set(keys)=={'descr','fortran_order','shape'},'Unique NPY schema fields required.')
    header=ast.literal_eval(tree)
    require(header['descr']=='<f4' and header['fortran_order'] is False
        and type(header['shape']) is tuple and header['shape']==tuple(shape)
        and len(shape)==4 and all(type(n) is int and n>0 for n in shape)
        and all(type(n) is int for n in header['shape']),'Exact FP32 C-order four-dimensional geometry required.')
    payload=raw[end:];require(len(payload)==math.prod(shape)*4 and digest(payload)==payload_sha256,'Complete pinned logical FP32 payload required.')
    value=np.frombuffer(payload,dtype='<f4').reshape(tuple(shape))
    require(np.isfinite(value).all() and np.any(value),'Finite nonzero saved component return required.')
    require(not value.flags.writeable,'Saved bytes must stay read-only.')
    return value,{'header_bytes':end,'payload_sha256':payload_sha256,'shape':list(shape),'dtype':'<f4','finite_nonzero':True,'read_only':True}

def agreement(actual,expected,tolerance):
    actual=np.asarray(actual,dtype=np.float64);expected=np.asarray(expected,dtype=np.float64)
    if (actual.shape!=expected.shape or actual.size==0 or not np.isfinite(actual).all() or not np.isfinite(expected).all()
            or set(tolerance)!={'rtol','atol'} or any(not math.isfinite(v) or v<0 for v in tolerance.values())
            or tolerance['atol']<=0):raise ValueError('Invalid saved-tensor comparison.')
    error=np.abs(actual-expected);allowed=tolerance['atol']+tolerance['rtol']*np.abs(expected)
    return {'passed':bool(np.all(error<=allowed)),'failed_elements':int(np.count_nonzero(error>allowed)),
            'shape':list(actual.shape),'max_absolute_error':float(error.max()),
            'max_tolerance_ratio':float((error/allowed).max()),
            'relative_l2_error':float(np.linalg.norm(error.ravel())/max(float(np.linalg.norm(expected.ravel())),1e-300)),
            **tolerance}

