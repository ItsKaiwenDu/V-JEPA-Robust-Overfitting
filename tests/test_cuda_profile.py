"""Profiler correctness, evidence validation, and real subprocess bounds."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch
from torch.nn import functional as F

from models.cuda_profile import (CASES, file_hash, iteration, make_case, masks_and_shape,
                                 run_bounded, summarize, write_json)

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('case', ('encoder-mask-0','encoder-mask-1','predictor-mask-0',
                                  'predictor-mask-1','predictor-sort'))
def test_masking_input_gradients_match_independent_scatter(upstream,case):
    op,leaves,cotangent,metadata=make_case(case,upstream,'cpu',tiny=True)
    y,grads,timing=iteration(op,leaves,cotangent,'cpu')
    pairs,shape=masks_and_shape(upstream,tiny=True)
    family=int(case[-1]) if case.endswith(('-0','-1')) else 0
    context,target=pairs[family]
    ids=(torch.argsort(torch.cat((context,target),dim=1),dim=1) if case=='predictor-sort'
         else target if 'predictor' in case else context)
    expected=torch.zeros_like(leaves[0])
    for b in range(len(ids)):
        for j,index in enumerate(ids[b]):expected[b,index]+=cotangent[b,j]
    assert torch.equal(grads[0],expected)
    assert torch.equal(y,torch.stack([leaves[0][b,row] for b,row in enumerate(ids)]))
    assert metadata['scale']=='tiny_cpu_fixture'
    assert all(t>0 for t in timing.values())


def test_representative_shapes_use_the_declared_primary_mask_seed(upstream):
    pairs,shape=masks_and_shape(upstream)
    assert shape['tokens']==2048 and shape['encoder_dim']==1024 and shape['predictor_dim']==384
    assert [(len(a[0]),len(b[0])) for a,b in pairs]==[(960,1088),(400,1648)]
    for a,b in pairs:
        assert torch.equal(torch.cat((a,b),dim=1).sort().values,torch.arange(2048).unsqueeze(0))


def test_patch_component_matches_independent_unfold_and_linear_gradients(upstream):
    op,leaves,cotangent,_=make_case('patch-conv',upstream,'cpu',tiny=True,parameter_gradients=True)
    actual=op()
    x,weight,bias=leaves
    patches=x.unfold(2,2,2).unfold(3,4,4).unfold(4,4,4)
    patches=patches.permute(0,2,3,4,1,5,6,7).contiguous().flatten(1,3).flatten(2)
    expected=F.linear(patches,weight.flatten(1),bias)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    got=torch.autograd.grad(actual,leaves,grad_outputs=cotangent)
    want=torch.autograd.grad(expected,leaves,grad_outputs=cotangent)
    for a,b in zip(got,want):torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)


@pytest.mark.parametrize('case',('sdpa-encoder-0','sdpa-encoder-1','sdpa-predictor',
                                  'rope-encoder-0','rope-encoder-1','rope-predictor'))
def test_attention_components_have_finite_nonzero_input_gradients(upstream,case):
    op,leaves,cotangent,_=make_case(case,upstream,'cpu',tiny=True)
    _,grads,_=iteration(op,leaves,cotangent,'cpu')
    assert torch.isfinite(grads[0]).all() and grads[0].abs().max()>0


def test_timeout_kills_the_worker_and_its_descendants(tmp_path):
    marker=tmp_path/'descendant-survived'
    child='import time,pathlib;time.sleep(1);pathlib.Path('+repr(str(marker))+').touch()'
    program='import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",'+repr(child)+']);time.sleep(10)'
    result=run_bounded([sys.executable,'-c',program],tmp_path/'timeout.log',.15)
    assert result['status']=='timeout'
    time.sleep(1.1)
    assert not marker.exists()


def evidence_fixture(tmp_path):
    tasks=[]
    for policy in ('default','strict'):
        for replay in (0,1):
            path=tmp_path/f'{policy}-{replay}.json'
            write_json(path,{'diagnostic_only':True,'research_result':False,
                'metadata':{'case':'patch-conv','operands':{'input':'same'}},
                'runtime':{'policy':policy,'matmul_precision':'highest','matmul_tf32':False,'cudnn_tf32':True},
                'provenance':{'source':'same'},'signature':{'gradient':'same'},'within_process_exact':True,
                'median_seconds':{'forward_seconds':.1,'backward_seconds':.2,'total_seconds':.3}})
            tasks.append({'case':'patch-conv','policy':policy,'replay':replay,'status':'ok',
                          'result':path.name,'result_sha256':file_hash(path)})
    return tasks


@pytest.mark.parametrize('defect',('timeout','changed_hash','replay_mismatch','changed_operand','changed_precision'))
def test_partial_corrupt_or_incomparable_records_never_claim_a_pass(tmp_path,defect):
    tasks=evidence_fixture(tmp_path)
    if defect=='timeout':tasks[-1]['status']='timeout'
    else:
        row=tasks[-1] if defect in ('changed_hash','replay_mismatch') else tasks[-2]
        path=tmp_path/row['result'];v=json.loads(path.read_text())
        if defect=='changed_hash':v['signature']={'gradient':'tampered'}
        elif defect=='replay_mismatch':v['signature']={'gradient':'different'}
        elif defect=='changed_operand':v['metadata']['operands']['input']='different'
        else:v['runtime']['cudnn_tf32']=False
        write_json(path,v)
        if defect!='changed_hash':row['result_sha256']=file_hash(path)
    result=summarize(tmp_path,tasks,['patch-conv'])['patch-conv']
    if defect in ('timeout','changed_hash'):assert not result['strict']['complete']
    else:assert not result['strict']['observed_exact_replay']


def test_fresh_process_parent_and_optional_profiler_round_trip(upstream,tmp_path):
    output=tmp_path/'result'
    command=[sys.executable,str(ROOT/'scripts/profile_cuda.py'),'--device','cpu','--tiny',
             '--upstream',str(upstream),'--output',str(output),'--cases','encoder-mask-0',
             '--max-seconds','150','--worker-seconds','30','--warmup','1','--repeats','2']
    # This positive fixture includes fresh interpreter/import/autograd startup.
    # Keep a finite allowance for slow shared filesystems; timeout/kill behavior
    # is exercised separately with deliberately tiny worker/deadline budgets.
    result=subprocess.run(command,capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stderr
    session=json.loads((output/'session.json').read_text())
    assert session['complete'] and session['strict_replay_passed']
    assert len(session['tasks'])==4
    comparison=session['comparisons']['encoder-mask-0']
    assert comparison['strict']['observed_exact_replay'] and comparison['same_operands_between_policies']
    for task in session['tasks']:
        value=json.loads((output/task['result']).read_text())
        assert value['profiler']['requested']
        assert value['provenance']['source_sha256']['models/cuda_profile.py']==file_hash(ROOT/'models/cuda_profile.py')
        if value['profiler'].get('available'):
            assert any('gather' in r['name'] for r in value['profiler']['top_cpu'])


def test_cli_rejects_hardware_fallback_and_reuse_before_execution(tmp_path):
    output=tmp_path/'kept';output.mkdir()
    for args in (['--device','cpu'],['--device','cuda','--tiny'],['--device','cpu','--tiny','--output',str(output)]):
        p=subprocess.run([sys.executable,str(ROOT/'scripts/profile_cuda.py'),'--upstream','missing',*args],capture_output=True,text=True)
        assert p.returncode==2
    assert list(output.iterdir())==[]


def test_worker_source_must_match_the_parent_manifest(tmp_path):
    tasks=evidence_fixture(tmp_path)
    comparison=summarize(tmp_path,tasks,['patch-conv'],{'source':'unexpected'})['patch-conv']
    assert not comparison['default']['complete'] and not comparison['strict']['complete']


def test_cross_policy_precision_change_has_no_speed_ratio(tmp_path):
    tasks=evidence_fixture(tmp_path)
    for row in tasks:
        if row['policy']=='strict':
            path=tmp_path/row['result'];v=json.loads(path.read_text())
            v['runtime']['cudnn_tf32']=False;write_json(path,v)
            row['result_sha256']=file_hash(path)
    comparison=summarize(tmp_path,tasks,['patch-conv'])['patch-conv']
    assert comparison['strict']['observed_exact_replay']
    assert not comparison['same_precision_between_policies']
    assert 'strict_to_default_total_ratio' not in comparison


def test_deadline_keeps_incomplete_session_and_nonzero_exit(upstream,tmp_path):
    output=tmp_path/'deadline'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/profile_cuda.py'),
        '--device','cpu','--tiny','--upstream',str(upstream),'--output',str(output),
        '--cases','encoder-mask-0','--max-seconds','1','--worker-seconds','.05'],
        capture_output=True,text=True,timeout=15)
    assert result.returncode==1,result.stderr
    session=json.loads((output/'session.json').read_text())
    assert session['finished'] and not session['complete'] and not session['strict_replay_passed']
    assert any(t['status'] in ('timeout','not_started_deadline') for t in session['tasks'])
