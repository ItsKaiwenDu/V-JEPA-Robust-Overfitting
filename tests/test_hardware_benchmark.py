"""Hardware identity, finite economics and real seven-phase small-model coverage."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from models.hardware_benchmark import GPU_MIN_GIB, SCREEN_PHASES, economic_screen, validate_gpu, validate_launch
from models.study_artifacts import read_record
from models.workflow_pilot import run_phase
from scripts.run_hardware_benchmark import collect_observations
from test_workflow_preflight import context
from test_native_execution import restore_flags
from test_vjepa2 import model

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('gpu',tuple(GPU_MIN_GIB))
def test_exact_hardware_contract(gpu):
    validate_gpu(gpu,gpu,GPU_MIN_GIB[gpu]*2**30)
    with pytest.raises(ValueError): validate_gpu(gpu,gpu,(GPU_MIN_GIB[gpu]-1)*2**30)
    with pytest.raises(ValueError): validate_gpu(gpu,gpu,80*2**30,2)
    with pytest.raises(ValueError): validate_gpu(gpu,'unreviewed replacement',80*2**30)


@pytest.mark.parametrize('rate,seconds',[(0,1200),(.71,1200),(float('nan'),1200),(.3,1199),(.3,7201)])
def test_quote_deadline_contract_rejects_before_launch(rate,seconds):
    with pytest.raises(ValueError): validate_launch('NVIDIA RTX A5000',rate,100000,100000+seconds)


def test_quote_contract_keeps_budget_reservation_and_no_auto_billing_control():
    r=validate_launch('NVIDIA RTX A5000',.29,100000,107200)
    assert r['maximum_billed_hours']==2 and r['maximum_inclusive_session_usd']==3
    assert r['proposed_session_hold_usd']<=r['prior_unreserved_pilot_usd']
    assert not r['cloud_billing_stopped_by_runner'] and not r['research_admitted']


def evidence_units():
    original=json.loads((ROOT/'report/checks/2026-10-22-shared-forward-cuda-workflow.json').read_text())
    def values(s): return [s['minimum_seconds'],s['median_seconds'],s['maximum_seconds']]
    result={k:values(v) for k,v in original['native_evaluation_seconds'].items()}
    result.update({k:values(v) for k,v in original['engine_update_seconds_by_condition'].items()})
    for k in ('engine.state','checkpoint.save','calibration_three_primary_candidates','load_anchor'):
        result[k]=values(original['successful_span_timings'][k])
    return result


def test_economic_screen_matches_independent_existing_review_without_claiming_recovery():
    ref=json.loads((ROOT/'report/checks/2026-10-22-measured-budget-review.json').read_text())
    value=economic_screen(ref,evidence_units(),.60)
    for stat,row in value['candidate_economic_screen'].items():
        expected=ref['variants']['focused_20x5']['scenarios'][stat]
        assert row['timed_unit_projection_hours']==pytest.approx(expected['timed_unit_extrapolation_hours'])
        assert row['with_historical_allowances_and_reserves_usd']==pytest.approx((expected['timed_unit_extrapolation_hours']+8)*1.25*.60+50)
    assert not value['exact_optimizer_recovery_checked'] and not value['budget_approved']


@pytest.mark.parametrize('defect',['missing','zero','nonfinite'])
def test_missing_or_bad_fresh_units_never_become_zero_cost(defect):
    obs=evidence_units()
    obs['primary']={'missing':[],'zero':[0],'nonfinite':[float('nan')]}[defect]
    with pytest.raises(ValueError):
        economic_screen(json.loads((ROOT/'report/checks/2026-10-22-measured-budget-review.json').read_text()),obs,.29)


def test_cpu_screen_runs_actual_phases_and_reference_anchor_without_fake_exact_recovery(context,tmp_path,monkeypatch):
    monkeypatch.setattr('models.workflow_pilot.prepare_context',lambda root,args:copy.deepcopy(context))
    args=SimpleNamespace(device='cpu',output=tmp_path/'screen',hardware_screen_only=True,
                         feature_anchor_variant='reference')
    for phase in SCREEN_PHASES:
        args.phase=phase
        result=run_phase(ROOT,args)
        assert result['diagnostic_only'] and not result['research_result']
    assert not (args.output/'adversarial-finetune'/'split').exists()
    assert read_record(args.output/'phase-results/features-trained.json')['training_resubstitution']['examples']==8
    obs=collect_observations(args.output)
    assert len(obs['clean-finetune'])==len(obs['adversarial-finetune'])==2
    assert len(obs['convergence'])==1 and len(obs['load_anchor'])==1
    result=economic_screen(json.loads((ROOT/'report/checks/2026-10-22-measured-budget-review.json').read_text()),obs,.29)
    assert not result['exact_optimizer_recovery_checked']


def test_reference_anchor_variant_cannot_leak_into_original_pilot(context,tmp_path,monkeypatch):
    monkeypatch.setattr('models.workflow_pilot.prepare_context',lambda root,args:copy.deepcopy(context))
    args=SimpleNamespace(device='cpu',output=tmp_path/'reject',phase='features-trained',feature_anchor_variant='reference')
    with pytest.raises(ValueError,match='hardware screen'):
        run_phase(ROOT,args)


def test_non_diagnostic_cli_rejects_before_importing_cuda_or_creating_output(tmp_path):
    output=tmp_path/'must-not-exist'
    args=[sys.executable,str(ROOT/'scripts/run_hardware_benchmark.py'),'--expected-gpu','NVIDIA RTX A5000',
          '--hourly-rate-usd','.29','--deadline-epoch',str(int(time.time())+1800),
          '--checkpoint','missing','--clips','missing','--manifests','missing','--output',str(output)]
    r=subprocess.run(args,capture_output=True,text=True,timeout=10)
    assert r.returncode==2 and '--diagnostic' in r.stderr and not output.exists()


def test_invalid_launcher_quote_rejects_before_a_job_or_handoff(tmp_path):
    r=subprocess.run(['bash',str(ROOT/'scripts/start_hardware_benchmark.sh'),
        '--expected-gpu','NVIDIA RTX A5000','--hourly-rate-usd','.89',
        '--deadline-epoch',str(int(time.time())+1800)],capture_output=True,text=True,timeout=10)
    assert r.returncode!=0 and '$0.70' in r.stderr
