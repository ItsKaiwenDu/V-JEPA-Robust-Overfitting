"""Assess one bounded saved raw trace; no model, profiler, CUDA or target work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models import predictor_sdpa_launch_evidence_v1 as evidence


def assess(path,output,*,arm,device):
    path=Path(path);output=Path(output)
    if output.exists():raise FileExistsError('Preserve previous assessments.')
    s=path.lstat()
    if path.is_symlink() or not stat.S_ISREG(s.st_mode) or not 0<s.st_size<=evidence.MAX_BYTES:
        raise ValueError('Bounded regular raw trace required.')
    with path.open('rb') as f:data=f.read(evidence.MAX_BYTES+1)
    after=path.lstat()
    if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) or len(data)!=s.st_size:
        raise ValueError('Raw trace changed during read.')
    result={'kind':'saved-SDPA-launch-correlation-assessment-v1','metadata_only':True,
        'input':{'path':str(path.resolve()),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()},
        'selection':arm,'device':device,'passed':False,'worker_completion_inferred':False,
        'actual_CUDA_execution_admitted':False,'research_or_training_admitted':False,
        'new_model_autograd_SDPA_profiler_CUDA_reference_or_university_work':False}
    try:result.update(proof=evidence.verify(json.loads(data),arm=arm,device=device),passed=True)
    except (ValueError,TypeError,KeyError) as e:result.update(error_type=type(e).__name__,error=str(e))
    names=('models/predictor_sdpa_launch_evidence_v1.py','models/predictor_sdpa_component_events_v2.py',
        'models/predictor_sdpa_component_events_v1.py','scripts/check_predictor_sdpa_launch_evidence_v1.py')
    result['source_sha256']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    output.mkdir(mode=0o700,parents=True)
    p=output/'assessment.json'
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f:
        f.write(evidence.encoded(result));f.flush();os.fsync(f.fileno())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--selection',choices=('original','math'),required=True)
    p.add_argument('--device',choices=('cpu','cuda'),required=True);a=p.parse_args()
    r=assess(a.input,a.output,arm=a.selection,device=a.device)
    print(json.dumps({'passed':r['passed'],'metadata_only':True,'actual_CUDA_execution_admitted':False},sort_keys=True))
    sys.exit(0 if r['passed'] else 1)
