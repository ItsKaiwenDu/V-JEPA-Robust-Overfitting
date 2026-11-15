#!/usr/bin/env python3
"""Local saved-metadata assessment only; never launches a worker or connection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models import predictor_pgd_split_backend_v1 as reader


def assess(path,output,*,expected_sha256,role,control_sha256,boundary_sha256):
    output=Path(output).absolute()
    output.mkdir(mode=0o700,exist_ok=False)
    result={'kind':'pgd-split-profile-local-file-assessment-v1','metadata_only':True,'metadata_matches_recipe':False,
        'new_model_gradient_profiler_GPU_SSH_proof_retrieval_or_paid_work':False,
        'worker_completion_inferred':False,'actual_CUDA_execution_admitted':False,'training_admitted':False,
        'input_expected_sha256':expected_sha256,'role':role}
    try:
        data=reader.read_private_trace(path,expected_sha256)
        proof=reader.assess_bytes(data,role=role,binding={'design_sha256':reader.DESIGN_SHA,'control_sha256':control_sha256,
            'boundary_sha256':boundary_sha256,'probe_index':0 if role=='native' else 1})
        result.update(metadata_matches_recipe=True,proof=proof,input_bytes=len(data),input_sha256=hashlib.sha256(data).hexdigest())
    except Exception as error:
        result.update(error_type=type(error).__name__,error=str(error)[:2048])
    data=reader.encoded(result)
    if len(data)>256*1024:raise ValueError('Local assessment file ceiling; preserve source.')
    with (output/'assessment.json').open('xb') as f:
        os.fchmod(f.fileno(),0o600);f.write(data);f.flush();os.fsync(f.fileno())
    fd=os.open(output,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--output',required=True);p.add_argument('--role',choices=('native','forward-preserving'),required=True)
    p.add_argument('--control-sha256',required=True);p.add_argument('--boundary-sha256',required=True)
    a=p.parse_args();result=assess(a.input,a.output,expected_sha256=a.sha256,role=a.role,
        control_sha256=a.control_sha256,boundary_sha256=a.boundary_sha256)
    print(json.dumps({'metadata_matches_recipe':result['metadata_matches_recipe'],'actual_CUDA_execution_admitted':False},sort_keys=True))
    return 0 if result['metadata_matches_recipe'] else 1


if __name__=='__main__':sys.exit(main())
