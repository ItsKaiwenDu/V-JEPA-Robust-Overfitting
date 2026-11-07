#!/usr/bin/env python3
"""Offline inspection of already saved gradients; no target or model launch."""
from pathlib import Path
import argparse, hashlib, json, platform, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    if platform.node() in ('linux10603','linux10609'):
        raise ValueError('This saved-value analysis has no university execution grant.')
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'runs/pretrained/diagnostic') or output.exists():
        raise ValueError('Fresh local diagnostic output directory required; preserve prior analysis.')
    import torch
    def blocked(*a,**kw):raise RuntimeError('Saved analysis cannot execute a model, backward, SDPA or CUDA.')
    for obj,name in ((torch.nn.Module,'__call__'),(torch.autograd,'grad'),(torch.autograd,'backward'),(torch.Tensor,'backward'),(torch.nn.functional,'scaled_dot_product_attention'),(torch.cuda,'init'),(torch.cuda,'_lazy_init')):setattr(obj,name,blocked)
    def deny(event,unused):
        if event.startswith(('subprocess.','socket.','os.fork','os.posix_spawn','os.exec')) or event=='os.system':raise RuntimeError('Saved analysis cannot create a process/network connection.')
    sys.addaudithook(deny)
    from models import predictor_eot8_record_v2 as records
    from models.predictor_gradient_attribution_v1 import compare_saved,gradient_distribution
    def digest(p):
        with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    read=lambda p:json.loads(Path(p).read_text())
    prior_path=ROOT/'report/checks/2026-11-06-predictor-eot8-record-target-v2-target-review.json'
    prior=read(prior_path)
    for field in ('raw_review','sealed_manifest'):
        row=prior[field]
        if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Prior review binding changed.')
    pins=read(ROOT/prior['sealed_manifest']['path'])
    if len(pins)!=7505:raise ValueError('Prior sealed-input coverage changed.')
    for n,want in pins.items():
        if digest(ROOT/n)!=want:raise ValueError('Frozen source/result changed: '+n)
    handoff=ROOT/'runs/pretrained/diagnostic/predictor-record-target-v2-20261106/linux10603-final'
    if digest(handoff/'ATTEMPT.json')!='4dd65b03ea11ff2f67db43008130dbc7a84e75c7b29c01e1c5b2f84e0bd859bd':raise ValueError('Consumed attempt changed.')
    saved=handoff/'attended-yxu0jipw/saved'
    sha=prior['control_sha256']
    original=records.load_worker(saved/'results/0',expected_control=sha)
    candidate=records.load_worker(saved/'results/1',require_passed=False,expected_control=sha)
    if original['baseline']!=candidate['baseline'] or candidate['passed'] or candidate['phase']!='EOT8-before':raise ValueError('Saved comparison domain changed.')
    tensors=[];artifacts={}
    for i,w in enumerate((original,candidate)):
        values={}
        for kind in ('loss','input'):
            item=w['artifacts']['EOT8-before-'+kind];p=saved/'results'/str(i)/item['file']
            if digest(p)!=item['sha256'] or p.stat().st_size!=item['bytes']:raise ValueError('Saved tensor changed.')
            t=torch.load(p,weights_only=True,map_location='cpu')
            if (t.dtype!=torch.float32 or list(t.shape)!=([] if kind=='loss' else [1,3,16,256,256])
                    or not torch.isfinite(t).all() or t.requires_grad):raise ValueError('Actual saved return shape/dtype/domain changed.')
            values[kind]=t;artifacts[str(p.relative_to(ROOT))]=item['sha256']
        tensors.append(values)
    distribution=gradient_distribution(tensors[1]['input'],tensors[0]['input'])
    loss=compare_saved(tensors[1]['loss'],tensors[0]['loss'],kind='loss')
    if not loss['exact'] or distribution['gate']['violations']!=3724:raise ValueError('Prior numerical rejection changed.')
    shape=original['baseline']['masks'];target_tokens=sum(pair[1]['shape'][1] for draw in shape for pair in draw)
    prediction_bytes=target_tokens*1024*4
    result={'kind':'predictor-gradient-saved-analysis-v1','date':'2026-11-06',
        'prior_review':{'path':str(prior_path.relative_to(ROOT)),'sha256':digest(prior_path)},
        'prior_pins_reverified':len(pins),'consumed_attempt_sha256':digest(handoff/'ATTEMPT.json'),
        'artifacts':artifacts,'loss':loss,'gradient_distribution':distribution,
        'all_four_saved_tensors_included':True,'prediction_or_cotangent_values_saved_by_prior_run':False,
        'cause_can_be_inferred_from_equal_loss':False,'new_forward_backward_SDPA_CUDA_or_remote_work':False,
        'new_tests_represent_target_qualification':False,'numerical_bounds_changed':False,'candidate_adopted':False,
        'prediction_boundary_logical_byte_projection':{'outputs_per_arm':16,'target_tokens_sum':target_tokens,
            'features':1024,'bytes_per_fp32_element':4,'prediction_bytes_one_arm':prediction_bytes,
            'two_arms_predictions_and_cotangents_bytes':prediction_bytes*4,
            'plus_two_input_gradients_and_two_full_targets_bytes':prediction_bytes*4+2*3*16*256*256*4+2*2048*1024*4},
        'financial_state_unchanged':prior['financial_state_unchanged'],'research_admitted':False,
        'source_sha256':{n:digest(ROOT/n) for n in ('models/predictor_gradient_attribution_v1.py','scripts/analyze_predictor_gradient_v1.py','tests/test_predictor_gradient_attribution_v1.py')}}
    assert not torch.cuda.is_initialized()
    output.mkdir(mode=0o700)
    (output/'analysis.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'saved_analysis_complete':True,'violations':3724,'sign_disagreements':distribution['sign_disagreements'],
        'opposite_nonzero_signs':distribution['opposite_nonzero_signs'],'one_zero_sign_changes':distribution['one_zero_sign_changes'],
        'pins_reverified':len(pins),'new_model_or_GPU_work':False}))


if __name__=='__main__':main()
