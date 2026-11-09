"""Audit pinned saved profiles only. No model, capture, target or GPU work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models import predictor_sdpa_component_events_v1 as previous
from models import predictor_sdpa_component_events_v2 as current

REVIEW='report/checks/2026-11-08-predictor-sdpa-continuation-v1-target-review.json'
REVIEW_SHA='6cec024c2a060007629c81ce5fded67bc5d2f67368c76058ac69b545f65b11da'
PRESERVED='runs/pretrained/diagnostic/predictor-sdpa-association-v2-20261108/preserved-inputs.json'
CPU='runs/pretrained/diagnostic/predictor-component-startup-v2-20261107/pytest-final/fixed-SDPA0/cohort'
SOURCES=('models/predictor_sdpa_component_events_v1.py','models/predictor_sdpa_component_events_v2.py',
         'scripts/check_predictor_sdpa_association_v2.py','tests/test_predictor_sdpa_component_events_v2.py')


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read_trace(path):
    path=Path(path)
    if path.is_symlink() or not 0<path.stat().st_size<=8*2**20:raise ValueError('Bounded regular profile required.')
    rows=[json.loads(line) for line in path.read_bytes().splitlines()]
    if not 0<len(rows)<=current.MAX_EVENTS:raise ValueError('Exact complete bounded trace required.')
    return rows


def audit(output):
    output=Path(output)
    if output.exists():raise FileExistsError('Keep any previous offline audit.')
    if digest(ROOT/REVIEW)!=REVIEW_SHA:raise ValueError('Reviewed consumed result changed.')
    review=json.loads((ROOT/REVIEW).read_text());pins=json.loads((ROOT/PRESERVED).read_text())
    if len(pins)!=41489:raise ValueError('Preserved source/evidence coverage changed.')
    for name,want in pins.items():
        if digest(ROOT/name)!=want:raise ValueError('Frozen source/evidence changed: '+name)
    target_pins=json.loads((ROOT/review['target_input_pins']['path']).read_text())
    if digest(ROOT/review['target_input_pins']['path'])!=review['target_input_pins']['sha256']:
        raise ValueError('Reviewed target input manifest changed.')
    for name,item in target_pins.items():
        path=ROOT/name
        if (path.stat().st_size,path.stat().st_mtime_ns,digest(path))!=(item['bytes'],item['mtime_ns'],item['sha256']):
            raise ValueError('Consumed target input changed: '+name)
    saved=ROOT/review['session']/'saved';worker=json.loads((saved/'results/components/0/worker.json').read_text())
    path=saved/'results/components/0/events.jsonl';trace_sha=digest(path)
    if trace_sha!=worker['trace']['sha256'] or worker['complete'] is not False:
        raise ValueError('Original incomplete worker/trace identity required.')
    rows=read_trace(path)
    try:previous.verify(rows,arm='original',device='cuda')
    except ValueError as e:
        if str(e)!=worker['error']:raise
        old_error=str(e)
    else:raise ValueError('The original association refusal must remain reproducible.')
    proof=current.verify(rows,arm='original',device='cuda')
    if digest(path)!=trace_sha:raise ValueError('Saved trace changed.')
    cpu=[]
    for i in range(4):
        item=ROOT/CPU/str(i)/'events.jsonl';r=read_trace(item);arm='original' if i%2==0 else 'math'
        old=previous.verify(r,arm=arm,device='cpu');new=current.verify(r,arm=arm,device='cpu')
        if any(old[k]!=new[k] for k in old if k!='kind'):
            raise ValueError('Existing CPU phase/backend semantics changed.')
        cpu.append({'path':str(item.relative_to(ROOT)),'sha256':digest(item),'rows':len(r),'proof':new})
    state={'kind':'offline-fixed-SDPA-association-assessment-v2','assessment_only':True,'date':'2026-11-08',
        'source_review':{'path':REVIEW,'sha256':REVIEW_SHA},'trace':{'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':trace_sha,'rows':len(rows)},
        'original_worker_complete':False,'original_error_preserved':old_error,'original_result_or_events_modified':False,
        'revised_trace_association_passed':True,'trace_proof':proof,'cached_CPU_traces':cpu,
        'source_sha256':{name:digest(ROOT/name) for name in SOURCES},'preserved_files_reverified':len(pins),
        'target_files_bytes_hashes_nsmtime_reverified':len(target_pins),'new_target_authorization_or_attempt':False,
        'new_model_autograd_SDPA_profiler_CUDA_or_reference_work':False,'new_SSH_upload_or_paid_compute':False,
        'worker_or_full_model_admitted':False,'fresh_replay_or_missing_resource_receipt_inferred':False,
        'candidate_rejected_entries_unchanged':3724,'numerical_bounds_and_financial_state_unchanged':True}
    os.umask(0o077);output.mkdir(mode=0o700,parents=True)
    with (output/'assessment.json').open('xb') as f:
        f.write((json.dumps(state,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.output)
    print(json.dumps({'assessment_only':True,'revised_trace_association_passed':result['revised_trace_association_passed'],
        'old_worker_complete':False,'new_target_work':False},sort_keys=True))
