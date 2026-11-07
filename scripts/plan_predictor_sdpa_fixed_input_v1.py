"""Plan from verified saved metadata only; no target/model/scientific execution."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if platform.node() in ('linux10603','linux10609'):
        raise ValueError('Local saved-metadata planning only; no university operation.')
    def deny(event,details):
        if event.startswith(('subprocess.','socket.','os.fork','os.posix_spawn','os.exec')) or event=='os.system':
            raise RuntimeError('Planning forbids process/network creation.')
        if event=='import' and details[0].split('.')[0] in ('torch','numpy','timm','triton'):
            raise RuntimeError('Planning reads metadata only, no scientific imports.')
    sys.addaudithook(deny)
    from models import predictor_sdpa_fixed_input_v1 as m
    from models import predictor_common_cotangent_record_v1 as records
    def sha(path):
        with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    def pin(path):return {'path':str(Path(path).relative_to(ROOT)),'sha256':sha(path)}
    report=ROOT/'report/checks/2026-11-06-common-cotangent-v1-target-review.json'
    r=json.loads(report.read_text())
    for name in ('raw_review','review_source','sealed_manifest','ordered_decomposition','response'):
        item=r[name];assert sha(ROOT/item['path'])==item['sha256'],name
    manifest=ROOT/r['sealed_manifest']['path'];pins=json.loads(manifest.read_text())
    assert len(pins)==r['sealed_inputs']==11998
    for name,want in pins.items():assert sha(ROOT/name)==want,name
    raw=ROOT/'runs/pretrained/diagnostic/common-cotangent-handoff-v1-20261106'
    handoff=raw/'linux10603-final';claim=handoff/'ATTEMPT.json'
    assert sha(claim)==r['consumed_attempt_sha256']==m.PREDECESSOR
    saved=handoff/'attended-q8txgu3g/saved';folder=saved/'results/0'
    w=records.load(folder,control_sha256=r['control_sha256'])
    plan=m.proposal(r,w,w['call_rows'])
    sources={name:sha(ROOT/name) for name in ('models/predictor_sdpa_fixed_input_v1.py','scripts/plan_predictor_sdpa_fixed_input_v1.py','tests/test_predictor_sdpa_fixed_input_v1.py','models/attention_saved_evidence.py')}
    plan['evidence']={'predecessor_review':pin(report),'raw_predecessor_review':r['raw_review'],
        'prior_manifest':r['sealed_manifest'],'call_record':pin(folder/'attention-calls.jsonl'),
        'native_original_record':pin(folder/'worker.json'),'source_sha256':sources,
        'sealed_prior_inputs_reverified':len(pins)}
    m.validate(plan)
    os.umask(0o077);out=args.output.resolve()
    if not out.is_relative_to(ROOT/'runs/pretrained/diagnostic'):
        raise ValueError('Planning output must be a new local ignored diagnostic folder.')
    out.mkdir(mode=0o700)
    with (out/'proposal.json').open('x') as f:f.write(json.dumps(plan,indent=2,sort_keys=True,allow_nan=False)+'\n')
    with (out/'provenance.json').open('x') as f:f.write(json.dumps({'prior_inputs':len(pins),'predecessor_claim_unchanged':sha(claim)==m.PREDECESSOR,
        'proposal_sha256':sha(out/'proposal.json'),'source_sha256':sources,
        'scientific_imports_model_network_process_blocked':True,'new_university_or_paid_work':False},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'saved_plan':str(out/'proposal.json'),'prior_inputs_reverified':len(pins),
        'selected_forward_call':plan['selection']['forward_row_index'],'authorized':False,'university_work_started':False},sort_keys=True))

if __name__=='__main__':main()
