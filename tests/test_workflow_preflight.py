"""Bounded candidate preflight and evidence export using actual small CPU models."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest
import torch

from models.diagnostics import state_digest
from models.execution import apply_model_policy, configure_execution, model_execution_contract
from models.pilot_measurement import Timings, compare_recovery
from models.study_artifacts import read_record
from models.training import TrainingSettings, rng_state
from models.workflow_preflight import matched_backward, run_preflight
from test_evaluation import tiny_masks
from test_native_execution import restore_flags
from test_training import assert_tree_equal
from test_vjepa2 import model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def context(model):
    configure_execution('shared-fp64-native', device='cpu')
    apply_model_policy(model, 'shared-fp64-native')
    model.encoder.use_activation_checkpointing = model.predictor.use_activation_checkpointing = True
    pixels = .2 + .6 * torch.rand(8, 3, 4, 16, 16, generator=torch.Generator().manual_seed(51))
    rows = [dict(video_id=f'v{i}', source_group_id=f'g{i}', class_id=str(i % 2)) for i in range(8)]
    training = SimpleNamespace(panel='train', identity={'data':'generated-fixture'}, rows=rows,
                               pixels=lambda i: pixels[i:i+1])
    return SimpleNamespace(model=model, training=training, sampler=tiny_masks,
        monitor=SimpleNamespace(panel='train_monitor', identity={'data':'generated-fixture'}, rows=rows[:1],
                                pixels=lambda i: pixels[i:i+1]),
        settings=TrainingSettings(), plan={'evaluation':{'primary_epochs':[1,15,16,23,24,30],'mask_seed':20261020}},
        checkpoint=dict(epoch=0,condition='pretrained',seed=None,training_contract_sha256=None,file_sha256='original'),
        provenance={'checkpoint_sha256':'original','fixture':'shared-preflight-v1'})


def test_three_actual_training_clips_pass_original_oracles_without_state_rng_or_mode_mutation(context, tmp_path):
    context.model.train()
    state, rng = state_digest(context.model), rng_state('cpu')
    result = run_preflight(context, tmp_path/'preflight', device='cpu', timings=Timings(tmp_path/'timings.jsonl','cpu'))
    assert result['passed'] and result['complete'] and result['indices'] == [0,4,7]
    assert not result['native_backward_probe']['actual_cuda_kernel_evidence']
    assert state_digest(context.model) == state and context.model.training
    assert_tree_equal(rng, rng_state('cpu'))
    assert model_execution_contract(context.model)['policy'] == 'shared-fp64-native'
    saved=read_record(tmp_path/'preflight/preflight.json')
    assert saved == result and len(list((tmp_path/'preflight').glob('*.pt'))) == 6
    for clip in saved['clips']:
        assert clip['comparison']['tokens_bitwise_equal'] and clip['comparison']['input_gradient']['passed']
        for arm in clip['arms'].values():
            assert all(v['passed'] for v in arm['oracle_validation'].values())
            assert arm['passed'] and all(v['passed'] for v in arm['attacks'].values())


@pytest.mark.parametrize('defect', ['oracle', 'cross_gradient', 'probe', 'pixels', 'no_ascent'])
def test_failed_preflight_retains_partial_evidence_and_restores_declared_runtime(context, tmp_path, monkeypatch, defect):
    import models.workflow_preflight as module
    state, rng = state_digest(context.model),rng_state('cpu')
    if defect == 'oracle':
        actual=module.projection_check
        def reject(*args):
            gates,tensors=actual(*args)
            gates['weight']['passed']=False
            return gates,tensors
        monkeypatch.setattr(module,'projection_check',reject)
    elif defect == 'cross_gradient':
        actual=module.agreement
        calls=[]
        def reject(*args):
            value=actual(*args)
            calls.append(True)
            # Two chain reconstructions, loss, then the cross-backend gradient.
            if len(calls)==4: value['passed']=False
            return value
        monkeypatch.setattr(module,'agreement',reject)
    elif defect == 'probe':
        actual=module.native_probe
        def reject(device,retain):
            actual(device,retain)
            raise RuntimeError('Retained probe failed')
        monkeypatch.setattr(module,'native_probe',reject)
    elif defect == 'pixels':
        context.training.pixels=lambda i: (_ for _ in ()).throw(ValueError('pixel checksum differs'))
    else:
        from attacks.pgd import AttackResult
        def unchanged(objective,clean,**kwargs):
            return AttackResult(clean.detach().clone(),objective(clean).detach())
        monkeypatch.setattr(module,'fgsm',unchanged)
        monkeypatch.setattr(module,'pgd_linf',unchanged)
    with pytest.raises((RuntimeError,ValueError)):
        run_preflight(context,tmp_path/'preflight',device='cpu',timings=Timings(tmp_path/'timings.jsonl','cpu'))
    saved=read_record(tmp_path/'preflight/preflight.json')
    assert not saved['passed'] and not saved['complete'] and 'error' in saved
    assert 'native_backward_probe' in saved
    assert model_execution_contract(context.model)['policy']=='shared-fp64-native'
    assert state_digest(context.model)==state
    assert_tree_equal(rng,rng_state('cpu'))
    if defect in ('oracle','cross_gradient','no_ascent'): assert list((tmp_path/'preflight').glob('*.pt'))


def test_declared_backward_scope_restores_after_exception_and_refuses_weight_mutation(context):
    original=model_execution_contract(context.model)
    with pytest.raises(RuntimeError):
        with matched_backward(context.model,'cudnn','cpu'):
            assert model_execution_contract(context.model)['policy']=='shared-fp64-cudnn'
            raise RuntimeError('interruption')
    assert model_execution_contract(context.model)==original
    # Target mutation must remain a failure even while cleaning up an exception.
    with pytest.raises(ValueError,match='fixed target changed'):
        with matched_backward(context.model,'cudnn','cpu'):
            with torch.no_grad(): next(context.model.target_encoder.parameters()).add_(.01)


def test_shared_runner_phases_gate_calibration_and_complete_exact_recovery(context, tmp_path, monkeypatch):
    from models.workflow_pilot import run_phase
    spec=importlib.util.spec_from_file_location('new_pilot_runner',ROOT/'scripts/run_workflow_pilot.py')
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.phase_sequence('original')==runner.PHASES
    phases=runner.phase_sequence('shared-fp64-native')
    assert phases[0]=='preflight' and phases[1]=='calibrate'
    monkeypatch.setattr('models.workflow_pilot.prepare_context',lambda root,args:copy.deepcopy(context))
    args=SimpleNamespace(device='cpu',output=tmp_path/'pilot',phase='preflight')
    for phase in phases:
        args.phase=phase
        result=run_phase(ROOT,args)
        assert not result['research_result']
    for condition in ('clean-finetune','adversarial-finetune'):
        comparison=compare_recovery(args.output/condition/'reference',args.output/condition/'split')
        assert comparison['exact_match'],comparison
    assert read_record(args.output/'preflight/preflight.json')['passed']


@pytest.mark.parametrize('include',[False,True])
def test_export_includes_only_opted_in_preflight_tensors_not_optimizer_states(tmp_path, include):
    run=tmp_path/'run'
    (run/'preflight').mkdir(parents=True)
    (run/'checkpoints').mkdir()
    (run/'session.json').write_text('{}')
    torch.save({'input_gradient':torch.ones(3)},run/'preflight/clip-0-native.pt')
    torch.save({'optimizer':'private-large-state'},run/'checkpoints/resume.pt')
    output=tmp_path/'evidence.tar.gz'
    command=[sys.executable,str(ROOT/'scripts/export_workflow_pilot.py'),'--run',str(run),'--output',str(output)]
    if include: command.append('--include-preflight-tensors')
    result=subprocess.run(command,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    with tarfile.open(output) as archive: names=archive.getnames()
    assert ('pilot/preflight/clip-0-native.pt' in names)==include
    assert not any('resume.pt' in n for n in names)


def test_export_rejects_oversized_or_excessive_preflight_before_creating_archive(tmp_path):
    run=tmp_path/'run'
    (run/'preflight').mkdir(parents=True)
    (run/'session.json').write_text('{}')
    with (run/'preflight/clip-0-native.pt').open('wb') as handle: handle.truncate(256*2**20+1)
    output=tmp_path/'not-created.tar.gz'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/export_workflow_pilot.py'),'--run',str(run),
                           '--output',str(output),'--include-preflight-tensors'],capture_output=True,text=True,timeout=10)
    assert result.returncode!=0 and 'Unexpected preflight' in result.stderr and not output.exists()


@pytest.mark.parametrize('deadline',['bad','0','9999999999'])
def test_launcher_rejects_missing_or_unreviewable_deadline_before_starting_job(deadline):
    result=subprocess.run(['bash',str(ROOT/'scripts/start_shared_workflow.sh'),'--deadline-epoch',deadline],
                          capture_output=True,text=True,timeout=5)
    assert result.returncode==2 and ('Usage:' in result.stderr or 'remaining' in result.stderr)
