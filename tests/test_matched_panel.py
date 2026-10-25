"""Actual tiny upstream cases plus independent journal/aggregation/control tests."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from models.evaluation import EvaluationSettings,model_identity
from models.evaluation_store import json_hash
from models.execution import apply_model_policy,configure_execution
from models.fixture_state import save_fixture_state
from models.matched_panel import PanelJournal,aggregate_scores,first_acceptance
from models.matched_panel_case import KIND,pixel_digest,run_worker,verify_case,verify_pair,verify_worker
from models.matched_panel_io import RUNNER_SOURCES,export_case,review_export,verify_archive,verify_export_sources
from models.matched_panel_plan import compile_plan,plan_digest
from models.matched_panel_runner import case_deadline,case_session,next_attempt
from models.native_diagnostic import tiny_fixture
from models.pilot_bundle import digest
from models.pilot_measurement import compare_trees
from models.study_artifacts import read_record,write_record
from models.training import rng_state,runtime_contract

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('change',[None,'handoff','source','missing'])
def test_exported_source_and_manifest_bind_to_external_trust(tmp_path,change):
    # Metadata-only provenance fixture; this does not assert a CUDA case pass.
    (tmp_path/'PANEL-HANDOFF.json').write_text('generated fixture handoff metadata\n')
    expected={'handoff_manifest_sha256':digest(tmp_path/'PANEL-HANDOFF.json'),'source_sha256':{}}
    for name in RUNNER_SOURCES:
        path=tmp_path/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,path);expected['source_sha256'][name]=digest(path)
    if change=='handoff':(tmp_path/'PANEL-HANDOFF.json').write_text('changed\n')
    if change=='source':(tmp_path/'source'/RUNNER_SOURCES[0]).write_text('changed\n')
    if change=='missing':(tmp_path/'source'/RUNNER_SOURCES[0]).unlink()
    if change is None:verify_export_sources(tmp_path,expected)
    else:
        with pytest.raises((ValueError,FileNotFoundError)):verify_export_sources(tmp_path,expected)


def fixture_plan(pixels):
    train=[{'video_id':f'fixture-{c}-{i}','relative_path':f'fixture-{c}/fixture-{c}-{i}.avi',
            'class_id':str(c),'source_group_id':f'fixture-group-{c}-{i}'} for c in range(20) for i in range(5)]
    monitor=[train[c*5] for c in range(20)]
    entries=[{**r,'file':r['video_id']+'.pt','pixel_sha256':pixel_digest(pixels[0])} for r in train]
    return compile_plan(train,monitor,entries,identity={'generated_CPU_fixture':True},
                        reference={'session_seconds':2457.378,'seeds':{'attack':[2**31-2],'start':2**63-1}})


@pytest.fixture(scope='module')
def checked(tmp_path_factory,upstream):
    configure_execution('shared-fp64-native',device='cpu')
    torch.set_num_threads(2);torch.manual_seed(20261024)
    model,pixels,sampler=tiny_fixture(upstream)
    base=tmp_path_factory.mktemp('panel-cases')
    initial=save_fixture_state(model,base/'initial.pt')
    apply_model_policy(model,'shared-fp64-native')
    plan=fixture_plan(pixels)
    execution={'domain':'generated-cpu-fixture','plan_sha256':plan_digest(plan),
        'expected_runtime':runtime_contract('cpu',model=model),'expected_weights':model_identity(model),
        'fixture_initial':initial,'research_admitted':False}
    (base/'plan.json').write_text(json.dumps(plan));(base/'execution.json').write_text(json.dumps(execution))
    cases={}
    for case in plan['cases'][:2]:
        case_id=case['case_id'];out=base/case_id;out.mkdir()
        before,rng,modes=model_identity(model),rng_state('cpu'),[m.training for m in model.modules()]
        started=time.monotonic()
        run_worker(model,pixels,lambda s:sampler(s%(2**31-1)),plan=plan,case_id=case_id,
            execution=execution,output=out/'matrix',role='matrix',allow_fixture=True)
        assert model_identity(model)==before and compare_trees(rng,rng_state('cpu'))['exact_match']
        assert [m.training for m in model.modules()]==modes and all(p.grad is None for p in model.parameters())
        subprocess.run([sys.executable,str(ROOT/'tests/fixtures/matched_panel_worker.py'),str(ROOT),
            str(out/'replay'),str(base/'plan.json'),str(base/'execution.json'),case_id,
            str(base/'initial.pt'),initial['sha256']],check=True,capture_output=True,timeout=180)
        verify_pair(out,plan=plan,case_id=case_id,execution=execution,allow_fixture=True)
        write_record(out/'session.json',{'kind':KIND,'case_id':case_id,'plan_sha256':plan_digest(plan),
            'execution_sha256':json_hash(execution),'complete':True,'passed':True,'research_admitted':False,
            'elapsed_seconds':time.monotonic()-started,'tasks':{role:{'complete':True,'exit_status':0,'timed_out':False}
                                                              for role in ('matrix','replay')}})
        receipt=verify_case(out,plan=plan,case_id=case_id,execution=execution,allow_fixture=True)
        write_record(out/'receipt.json',receipt)
        archive=export_case(ROOT,out,plan,execution)
        cases[case_id]={'directory':out,'receipt':receipt,'manifest':archive}
    return SimpleNamespace(root=base,plan=plan,execution=execution,model=model,pixels=pixels,
                           masks=lambda s:sampler(s%(2**31-1)),cases=cases)


def cloned_journal(checked,tmp_path):
    journal=PanelJournal(tmp_path/'panel',plan=checked.plan,execution=checked.execution,allow_fixture=True)
    for case_id,record in checked.cases.items():
        destination=journal.root/'cases'/case_id/'attempt-01'
        shutil.copytree(record['directory'],destination)
    return journal


def test_actual_small_cases_pair_banks_targets_starts_scores_and_fresh_replay(checked):
    a,b=[r['receipt']['contract'] for r in checked.cases.values()]
    assert a['case']['video_id']==b['case']['video_id']=='fixture-0-0'
    assert a['case']['attack_seeds']!=b['case']['attack_seeds']
    for key in ('weights','runtime','targets_sha256','clean_sha256','starts_sha256'):
        assert a[key]==b[key]
    for name,value in a['mask_tensors_sha256'].items():
        if not name.startswith('attack_'):assert b['mask_tensors_sha256'][name]==value
    assert all(r['receipt']['fresh_A_replay_exact'] for r in checked.cases.values())


def test_CPU_fixture_evidence_never_qualifies_as_production(checked):
    case_id=checked.plan['cases'][0]['case_id']
    with pytest.raises(ValueError,match='Generated CPU'):
        verify_case(checked.cases[case_id]['directory'],plan=checked.plan,case_id=case_id,execution=checked.execution)


def test_scoring_only_changes_cannot_select_different_adversaries(checked,tmp_path):
    plan=copy.deepcopy(checked.plan)
    for values in plan['cases'][0]['scoring_seeds'].values():
        for i in range(len(values)):values[i]+=71
    execution={**checked.execution,'plan_sha256':plan_digest(plan)}
    run_worker(checked.model,checked.pixels,checked.masks,plan=plan,case_id='clip-00-bank-0',
        execution=execution,output=tmp_path/'changed',role='replay',allow_fixture=True)
    old=read_record(checked.cases['clip-00-bank-0']['directory']/'matrix/A.json')
    new=read_record(tmp_path/'changed/A.json')
    assert old['adversarial_sha256']==new['adversarial_sha256'] and old['trace']==new['trace']
    assert old['scores']!=new['scores']


@pytest.mark.parametrize('item',['inputs.pt','A.pt','worker.json','contract.json','session.json'])
def test_tampered_or_incomplete_case_rejects(checked,tmp_path,item):
    source=checked.cases['clip-00-bank-0']['directory'];copy_path=tmp_path/'bad'
    shutil.copytree(source,copy_path)
    path=copy_path/item if item=='session.json' else copy_path/'matrix'/item
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises((ValueError,json.JSONDecodeError)):
        verify_case(copy_path,plan=checked.plan,case_id='clip-00-bank-0',execution=checked.execution,allow_fixture=True)


def test_other_case_or_execution_identity_rejects_before_acceptance(checked):
    source=checked.cases['clip-00-bank-0']['directory']
    with pytest.raises(ValueError,match='Mixed plan'):
        verify_case(source,plan=checked.plan,case_id='clip-00-bank-1',execution=checked.execution,allow_fixture=True)
    changed=copy.deepcopy(checked.execution);changed['fixture_initial']['sha256']='0'*64
    with pytest.raises(ValueError,match='Mixed plan'):
        verify_case(source,plan=checked.plan,case_id='clip-00-bank-0',execution=changed,allow_fixture=True)


def test_missing_fresh_replay_never_passes(checked,tmp_path):
    target=tmp_path/'partial';shutil.copytree(checked.cases['clip-00-bank-0']['directory'],target)
    shutil.rmtree(target/'replay')
    with pytest.raises(FileNotFoundError):
        verify_case(target,plan=checked.plan,case_id='clip-00-bank-0',execution=checked.execution,allow_fixture=True)


def test_actual_archive_external_review_then_first_case_gate(checked,tmp_path):
    source=checked.cases['clip-00-bank-0'];archive=source['directory']/'evidence.tar.gz'
    receipt=review_export(archive,source['manifest'],source['manifest']['sha256'],tmp_path/'reviewed',
        plan=checked.plan,execution=checked.execution,allow_fixture=True)
    journal=cloned_journal(checked,tmp_path)
    journal.commit('clip-00-bank-0',journal.root/'cases/clip-00-bank-0/attempt-01')
    with pytest.raises(ValueError,match='Review the first'):
        journal.commit('clip-00-bank-1',journal.root/'cases/clip-00-bank-1/attempt-01')
    acceptance=first_acceptance(checked.plan,checked.execution,receipt,digest(archive))
    journal.accept_first(acceptance)
    journal.commit('clip-00-bank-1',journal.root/'cases/clip-00-bank-1/attempt-01')
    assert journal.progress()['completed_case_ids']==['clip-00-bank-0','clip-00-bank-1']
    assert journal.aggregate()['scientific_aggregate'] is None


def test_wrong_external_hash_or_acceptance_cannot_release_queue(checked,tmp_path):
    source=checked.cases['clip-00-bank-0']
    with pytest.raises(ValueError,match='checksum'):
        review_export(source['directory']/'evidence.tar.gz',source['manifest'],'0'*64,tmp_path/'bad',
            plan=checked.plan,execution=checked.execution,allow_fixture=True)
    journal=cloned_journal(checked,tmp_path)
    receipt=journal.commit('clip-00-bank-0',journal.root/'cases/clip-00-bank-0/attempt-01')
    acceptance=first_acceptance(checked.plan,checked.execution,receipt,'0'*64)
    with pytest.raises(ValueError,match='different evidence'):
        journal.accept_first(acceptance)


def test_atomic_commit_failure_recovers_verified_whole_case_without_execution(checked,tmp_path,monkeypatch):
    journal=cloned_journal(checked,tmp_path)
    real=__import__('models.matched_panel',fromlist=['write_record']).write_record
    def interrupt(path,payload,**kwargs):
        if Path(path).name=='journal.json':raise RuntimeError('interrupted before pointer commit')
        return real(path,payload,**kwargs)
    monkeypatch.setattr('models.matched_panel.write_record',interrupt)
    with pytest.raises(RuntimeError,match='before pointer'):
        journal.commit('clip-00-bank-0',journal.root/'cases/clip-00-bank-0/attempt-01')
    assert journal.read()['completed']=={}
    monkeypatch.setattr('models.matched_panel.write_record',real)
    # next_attempt uses production unique export paths, so move this real fixture archive there.
    src=journal.root/'cases/clip-00-bank-0/attempt-01/evidence.tar.gz'
    archive=journal.root/'exports/clip-00-bank-0-attempt-01.tar.gz';archive.parent.mkdir()
    shutil.copyfile(src,archive);shutil.copyfile(str(src)+'.manifest.json',str(archive)+'.manifest.json')
    assert next_attempt(journal,'clip-00-bank-0')==(None,None,None)
    assert 'clip-00-bank-0' in journal.read()['completed']


def test_damaged_saved_case_is_not_reused(checked,tmp_path):
    journal=cloned_journal(checked,tmp_path)
    attempt=journal.root/'cases/clip-00-bank-0/attempt-01'
    journal.commit('clip-00-bank-0',attempt)
    (attempt/'matrix/A.pt').write_bytes(b'broken')
    with pytest.raises(ValueError,match='checksum'):
        next_attempt(journal,'clip-00-bank-0')


def test_partial_attempt_requires_explicit_retry_and_preserves_time_allowance(checked,tmp_path):
    journal=PanelJournal(tmp_path/'panel',plan=checked.plan,execution=checked.execution,allow_fixture=True)
    attempt=journal.root/'cases/clip-00-bank-0/attempt-01';attempt.mkdir(parents=True)
    write_record(attempt/'session.json',{'elapsed_seconds':100.})
    with pytest.raises(ValueError,match='explicit retry'):
        next_attempt(journal,'clip-00-bank-0')
    path,archive,seconds=next_attempt(journal,'clip-00-bank-0',explicit_retry=True)
    assert path.name=='attempt-02' and archive.name=='clip-00-bank-0-attempt-02.tar.gz'
    assert seconds==4340 and (attempt/'session.json').exists()
    write_record(attempt/'session.json',{'elapsed_seconds':4700.},replace=True)
    with pytest.raises(ValueError,match='exhausted'):
        next_attempt(journal,'clip-00-bank-0',explicit_retry=True)


def test_case_worker_failure_exports_without_committing_success(checked,tmp_path,monkeypatch):
    monkeypatch.setattr('models.matched_panel_runner.bounded_child',lambda *_a:{'complete':False,'exit_status':1,'timed_out':False})
    directory=tmp_path/'failed';archive=tmp_path/'failed.tar.gz'
    with pytest.raises(RuntimeError,match='Worker failed'):
        case_session(ROOT,checked.plan,checked.execution,'clip-00-bank-0',directory,Path('unused.pt'),archive)
    assert read_record(directory/'session.json')['passed'] is False
    assert archive.is_file() and not (directory/'receipt.json').exists() and not (directory/'replay').exists()


def test_whole_case_deadline_includes_failed_worker_cleanup_export(checked,tmp_path,monkeypatch):
    import signal
    prior_handler=signal.getsignal(signal.SIGALRM)
    monkeypatch.setattr('models.matched_panel_runner.bounded_child',lambda *_a:time.sleep(10))
    directory=tmp_path/'timed-out';archive=tmp_path/'timed-out.tar.gz'
    with pytest.raises(RuntimeError,match='Whole-case outer deadline'):
        with case_deadline(.1):
            case_session(ROOT,checked.plan,checked.execution,'clip-00-bank-0',directory,Path('unused.pt'),archive)
    session=read_record(directory/'session.json')
    assert not session['passed'] and not session['complete'] and archive.is_file()
    assert not (directory/'receipt.json').exists()
    assert signal.getsignal(signal.SIGALRM)==prior_handler and signal.getitimer(signal.ITIMER_REAL)==(0.,0.)


def test_whole_case_deadline_restores_alarm_after_success():
    import signal
    prior_handler=signal.getsignal(signal.SIGALRM)
    with case_deadline(10):pass
    assert signal.getsignal(signal.SIGALRM)==prior_handler and signal.getitimer(signal.ITIMER_REAL)==(0.,0.)


def test_whole_case_deadline_hard_stops_stalled_cleanup():
    code='''import time
from models.matched_panel_runner import case_deadline
with case_deadline(.1,kill_grace_seconds=.1):
    try: time.sleep(10)
    except TimeoutError: time.sleep(10)
'''
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,timeout=30)
    assert result.returncode==124


def supervisor_fixture(checked,tmp_path,monkeypatch):
    from models import matched_panel_runner as runner
    journal=cloned_journal(checked,tmp_path)
    # Later cases must not already exist when checking supervisor dispatch.
    shutil.rmtree(journal.root/'cases/clip-00-bank-1')
    archive=journal.root/'exports/clip-00-bank-0-attempt-01.tar.gz';archive.parent.mkdir()
    src=journal.root/'cases/clip-00-bank-0/attempt-01/evidence.tar.gz'
    shutil.copyfile(src,archive);shutil.copyfile(str(src)+'.manifest.json',str(archive)+'.manifest.json')
    monkeypatch.setattr(runner,'load_plan',lambda _root:checked.plan)
    monkeypatch.setattr(runner,'execution_identity',lambda *_a:checked.execution)
    monkeypatch.setattr(runner,'hardware_gate',lambda *_a,**_k:'mocked supervisor hardware; no device query')
    monkeypatch.setattr(runner,'PanelJournal',lambda *_a,**_k:journal)
    return runner,journal,archive


def test_supervisor_recovers_completed_first_case_without_dispatch(checked,tmp_path,monkeypatch):
    runner,journal,_=supervisor_fixture(checked,tmp_path,monkeypatch)
    def forbidden(*_a,**_k):raise AssertionError('No GPU dispatch for a verified whole case.')
    monkeypatch.setattr(runner,'case_session',forbidden)
    result=runner.run_panel(ROOT,SimpleNamespace(mode='first',output=journal.root,checkpoint=Path('unused'),acceptance=None))
    assert not result['complete'] and result['scientific_aggregate'] is None
    assert journal.read()['completed'].keys()=={'clip-00-bank-0'}


def test_supervisor_missing_first_acceptance_cannot_dispatch(checked,tmp_path,monkeypatch):
    runner,journal,_=supervisor_fixture(checked,tmp_path,monkeypatch)
    called=[];monkeypatch.setattr(runner,'case_session',lambda *_a,**_k:called.append(True))
    with pytest.raises(ValueError,match='independent first-case'):
        runner.run_panel(ROOT,SimpleNamespace(mode='remaining',output=journal.root,checkpoint=Path('unused'),acceptance=None))
    assert not called


def test_remaining_supervisor_stops_on_next_failure_without_skip_or_retry(checked,tmp_path,monkeypatch):
    runner,journal,archive=supervisor_fixture(checked,tmp_path,monkeypatch)
    receipt=journal.commit('clip-00-bank-0',journal.root/'cases/clip-00-bank-0/attempt-01',archive=archive)
    acceptance=tmp_path/'acceptance.json'
    write_record(acceptance,first_acceptance(checked.plan,checked.execution,receipt,digest(archive)))
    called=[]
    def fail(_root,_plan,_execution,case_id,directory,*_a,**_k):
        called.append(case_id);Path(directory).mkdir(parents=True)
        write_record(Path(directory)/'session.json',{'elapsed_seconds':1.,'passed':False})
        raise RuntimeError('deliberate case failure')
    monkeypatch.setattr(runner,'case_session',fail)
    with pytest.raises(RuntimeError,match='deliberate case failure'):
        runner.run_panel(ROOT,SimpleNamespace(mode='remaining',output=journal.root,checkpoint=Path('unused'),acceptance=acceptance))
    assert called==['clip-00-bank-1'] and set(journal.read()['completed'])=={'clip-00-bank-0'}
    assert read_record(journal.root/'panel.json')['scientific_aggregate'] is None


def test_retry_cannot_skip_a_declared_case(checked,tmp_path,monkeypatch):
    runner,journal,_=supervisor_fixture(checked,tmp_path,monkeypatch)
    called=[];monkeypatch.setattr(runner,'case_session',lambda *_a,**_k:called.append(True))
    with pytest.raises(ValueError,match='next declared'):
        runner.run_panel(ROOT,SimpleNamespace(mode='retry',case_id='clip-02-bank-0',output=journal.root,
            checkpoint=Path('unused'),acceptance=None))
    assert not called


def test_changed_journal_execution_or_saved_review_rejects(checked,tmp_path):
    journal=cloned_journal(checked,tmp_path)
    attempt=journal.root/'cases/clip-00-bank-0/attempt-01'
    receipt=journal.commit('clip-00-bank-0',attempt)
    marker=first_acceptance(checked.plan,checked.execution,receipt,digest(attempt/'evidence.tar.gz'))
    journal.accept_first(marker)
    state=journal.read();state['first_acceptance']['archive_sha256']='0'*64
    write_record(journal.root/'journal.json',state,replace=True)
    with pytest.raises(ValueError,match='acceptance identity'):
        journal.read()
    changed=copy.deepcopy(checked.execution);changed['fixture_initial']['sha256']='0'*64
    with pytest.raises(ValueError,match='journal plan/source'):
        PanelJournal(journal.root,plan=checked.plan,execution=changed,allow_fixture=True)


def test_case_supervisor_lease_excludes_another_process(checked,tmp_path):
    journal=PanelJournal(tmp_path/'panel',plan=checked.plan,execution=checked.execution,allow_fixture=True)
    code='import fcntl,sys; h=open(sys.argv[1],"a+")\ntry: fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(42)'
    with journal.lease():
        result=subprocess.run([sys.executable,'-c',code,str(journal.root/'.panel-running.lock')],timeout=15)
        assert result.returncode==42


def test_complete_worker_evidence_is_not_recomputed_to_repair_export(checked,tmp_path):
    journal=cloned_journal(checked,tmp_path)
    with pytest.raises(ValueError,match='do not repeat GPU work'):
        next_attempt(journal,'clip-00-bank-0',explicit_retry=True)


def loss_table(plan):
    """Synthetic loss data for aggregation arithmetic only; no scientific-pass receipts."""
    result={}
    for case in plan['cases']:
        i,b=case['clip_index'],case['attack_bank']
        gap=(.02 if i%2==0 else -.01)+(-.03 if b==0 else .03)
        values={'A':1.,'B':1.+gap,'C':.9,'D':.94}
        result[case['case_id']]={a:{bank:{'clean':.5,'adversarial':value*(1 if bank=='main' else 2 if bank=='audit_0' else 3)}
                                      for bank in ('main','audit_0','audit_1')} for a,value in values.items()}
    return result


def test_aggregate_weights_clips_equally_and_keeps_audits_separate(checked):
    result=aggregate_scores(checked.plan,loss_table(checked.plan))
    main=result['summary']['B-A']['main']
    assert main['mean']==pytest.approx(.005) and main['median']==pytest.approx(.005)
    assert main['positive_clips']==main['negative_clips']==5 and main['zero_clips']==0
    assert result['summary']['B-A']['audit_0']['mean']==pytest.approx(.01)
    assert result['summary']['D-C']['main']['mean']==pytest.approx(.04)
    assert result['analysis_unit']=='clip' and result['clips']==10 and not result['score_banks_pooled']
    assert not result['automatic_primary_choice'] and not result['automatic_convergence_pass']


@pytest.mark.parametrize('change',['missing_case','missing_arm','missing_bank','nonfinite','negative','changed_clean'])
def test_incomplete_or_invalid_scores_never_generate_panel_mean(checked,change):
    values=loss_table(checked.plan)
    if change=='missing_case':values.pop('clip-09-bank-1')
    if change=='missing_arm':values['clip-00-bank-0'].pop('D')
    if change=='missing_bank':values['clip-00-bank-0']['A'].pop('audit_1')
    if change=='nonfinite':values['clip-00-bank-0']['A']['main']['adversarial']=float('nan')
    if change=='negative':values['clip-00-bank-0']['A']['main']['adversarial']=-1
    if change=='changed_clean':values['clip-00-bank-0']['D']['main']['clean']=.6
    with pytest.raises(ValueError):aggregate_scores(checked.plan,values)


def test_zero_denominator_is_undefined_not_silently_dropped(checked):
    scores=loss_table(checked.plan)
    scores['clip-00-bank-0']['A']['main']['adversarial']=0.
    result=aggregate_scores(checked.plan,scores)
    assert result['per_case']['clip-00-bank-0']['B-A']['main']['relative_percent'] is None
    assert result['summary']['B-A']['main']['relative_percent_mean'] is None
    assert result['summary']['B-A']['main']['mean']>0


def test_failure_restores_rng_modes_and_weights(checked,tmp_path):
    model=checked.model;model.train()
    rng,before,modes=rng_state('cpu'),model_identity(model),[m.training for m in model.modules()]
    def fail(_seed):torch.rand(4);raise RuntimeError('interrupted bank construction')
    with pytest.raises(RuntimeError,match='bank construction'):
        run_worker(model,checked.pixels,fail,plan=checked.plan,case_id='clip-00-bank-0',execution=checked.execution,
            output=tmp_path/'failed',role='matrix',allow_fixture=True)
    assert compare_trees(rng,rng_state('cpu'))['exact_match'] and model_identity(model)==before
    assert modes==[m.training for m in model.modules()] and all(p.grad is None for p in model.parameters())


def test_canonical_methods_and_production_fallback_remain_guarded(tmp_path):
    with pytest.raises(ValueError):EvaluationSettings('primary',20,2,8,4)
    result=subprocess.run([sys.executable,str(ROOT/'scripts/check_matched_panel.py'),'worker','--diagnostic',
                          '--output',str(tmp_path/'none')],capture_output=True,timeout=30)
    assert result.returncode==2 and b'existing released checkpoint' in result.stderr
