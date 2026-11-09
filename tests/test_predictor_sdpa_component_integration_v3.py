"""Actual small CPU workers plus simulated target startup/resource boundaries."""
import ast
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import pytest
from models import predictor_sdpa_component_v3 as core
from models import predictor_sdpa_component_records_v1 as rec
from models import predictor_sdpa_component_startup_v2 as startup
from models import predictor_sdpa_component_target_v3 as target
ROOT=Path(__file__).resolve().parents[1]


def write(p,value):p.write_bytes(rec.encoded(value));p.chmod(0o600)

@pytest.fixture(scope='module')
def cohort(tmp_path_factory):
    root=tmp_path_factory.mktemp('components-v3')/'cohort'
    result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/check_predictor_sdpa_components_v3.py'),'--output',str(root)],
        cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (root.parent/'cohort.log').write_bytes(result.stdout)
    assert result.returncode==0,result.stdout.decode()[-7000:]
    return root


def test_four_real_fresh_CPU_workers_finish_with_v2_association_and_one_reference(cohort):
    p=core.validate_plan(core.read(cohort/'component-plan.json'));ps=rec.sha(cohort/'component-plan.json');identity=core.capture(p)[3]
    ref,gold=core.load_reference(cohort/'reference',p,ps,identity)
    workers=[core.load_worker(cohort/str(i),p,ps,identity,i) for i in range(4)]
    assert p['kind']=='native-SDPA-fixed-input-plan-v3' and len({w['pid'] for w in workers})==4
    assert [w['oracle_build_calls'] for w in workers]==[1,0,0,0] and ref['oracle_calls']==1
    assert ref['builder_pid']==workers[0]['pid'] and not ref['reference_is_passing_worker']
    for i,w in enumerate(workers):
        assert w['complete'] and w['flags_restored'] and w['inputs_unchanged'] and w['profile_replay_exact']
        assert w['startup']['configure_calls']==1 and w['startup']['device']=='cpu'
        assert w['backend_evidence']['kind']=='one-fixed-SDPA-associated-backend-v2'
        assert w['backend_evidence']['CPU_only'] and not w['backend_evidence']['CUDA_forward_and_backward_qualified']
        assert not w['backend_evidence']['worker_completion_inferred'] and not w['performance_qualified'] and not w['research_admitted']
        for n in rec.RETURNS:
            assert w['outputs']['untimed'][n]['logical_sha256']==w['outputs']['profiled'][n]['logical_sha256']
            if i>=2:assert w['outputs']['untimed'][n]['logical_sha256']==workers[i-2]['outputs']['untimed'][n]['logical_sha256']
    state=core.read(cohort/'session.json');assert state['complete'] and state['CPU_only'] and not state['CUDA_qualified']


def test_verifier_change_does_not_change_math_collector_startup_or_numerical_gates():
    def functions(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    before=functions((ROOT/'models/predictor_sdpa_component_v2.py').read_text())
    source=(ROOT/'models/predictor_sdpa_component_v3.py').read_text()
    source=source.replace('from models import predictor_sdpa_component_events_v2 as events','from models import predictor_sdpa_component_events_v1 as events')
    source=source.replace('    from models import predictor_sdpa_component_events_v1 as capture_events\n','')
    source=source.replace('capture_events.capture(profile)','events.capture(profile)')
    assert functions(source)==before
    from models import predictor_sdpa_component_v2 as frozen
    assert core.BOUNDS==frozen.BOUNDS and core.BOOK==frozen.BOOK and core.CELLS==frozen.CELLS
    assert 'models/predictor_sdpa_component_events_v1.py' in core.SOURCE_NAMES
    assert 'models/predictor_sdpa_component_events_v2.py' in core.SOURCE_NAMES


@pytest.mark.parametrize('field,value',(('numerical_passed',False),('flags_restored',False),('inputs_unchanged',False),
 ('oracle_build_calls',2),('full_model_candidate_adopted',True)))
def test_rehashed_completion_claims_cannot_bypass_actual_receipts(cohort,tmp_path,field,value):
    p=core.read(cohort/'component-plan.json');ps=rec.sha(cohort/'component-plan.json');identity=core.capture(p)[3]
    path=tmp_path/'copy';shutil.copytree(cohort/'0',path);w=core.read(path/'worker.json');w[field]=value
    write(path/'worker.json',w);write(path/'complete.json',{'kind':w['kind'],'plan_sha256':ps,'complete':True,
      'header':{'file':'worker.json','bytes':(path/'worker.json').stat().st_size,'sha256':rec.sha(path/'worker.json')}})
    with pytest.raises(ValueError):core.load_worker(path,p,ps,identity,0)


def test_real_worker_keeps_saved_returns_and_failure_marker_on_association_refusal(tmp_path):
    p=core.make_local_plan(ROOT,tmp_path/'reference');path=tmp_path/'plan.json';write(path,p);ps=rec.sha(path)
    program='''import os,sys,time
from pathlib import Path
from types import SimpleNamespace
os.umask(0o077)
from models import predictor_sdpa_component_events_v2 as events
from models import predictor_sdpa_component_v3 as core
def refuse(*a,**k):raise ValueError('injected-association-refusal-v3')
events.verify=refuse
core.run(SimpleNamespace(plan=Path(sys.argv[1]),plan_sha256=sys.argv[2],index=0,output=Path(sys.argv[3]),device='cpu',deadline_unix=time.time()+120))'''
    result=subprocess.run([sys.executable,'-B','-c',program,str(path),ps,str(tmp_path/'0')],cwd=ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (tmp_path/'worker.log').write_bytes(result.stdout);assert result.returncode==1
    w=rec.load(tmp_path/'0',kind=rec.WORKER_KIND,plan_sha256=ps,require_complete=False)
    assert w['numerical_passed'] and w['profile_replay_exact'] and w['error']=='injected-association-refusal-v3'
    assert len(w['outputs']['untimed'])==len(w['outputs']['profiled'])==4
    assert (tmp_path/'reference/complete.json').exists() and (tmp_path/'0/events.jsonl').exists()
    assert not (tmp_path/'0/complete.json').exists() and not (tmp_path/'0/component-resource.json').exists()


def target_fixture(tmp_path,monkeypatch,*,peak=1024,complete=True,foreign_pid=False,bad_startup=False,expired=False):
    import torch
    monkeypatch.setattr(target.platform,'node',lambda:'linux10603');monkeypatch.setattr(target.platform,'python_version',lambda:'3.11.17')
    path=tmp_path/'plan.json';p={'local_tiny':False,'reference_dir':str(tmp_path/'reference')};write(path,p)
    monkeypatch.setattr(core,'validate_plan',lambda value:value);monkeypatch.setattr(core,'capture',lambda p:(None,None,None,None))
    monkeypatch.setattr(core,'previous',lambda *a:[]);monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:peak)
    pid=os.getpid();receipt={'kind':startup.KIND,'pid':pid,'device':'cuda','policy':'shared-fp64-native','configure_calls':1,
        'cuda_initialized_before':False,'cuda_initialized_after':True,'allocator_cap_bytes':7*2**30,
        'device_total_bytes':8*2**30,'allocator_fraction':7/8,'precision':{'deterministic_algorithms':True,
        'deterministic_warn_only':False,'matmul_precision':'highest','matmul_tf32':False,'cudnn_enabled':True,
        'cudnn_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True,'cublas_workspace_config':':4096:8'}}
    if bad_startup:receipt['configure_calls']=2
    def completed(args,*a):
        # Synthetic wrapper boundary only. Real CPU scientific completion is
        # separately tested above; this is not measured CUDA/resource evidence.
        args.output.mkdir(mode=0o700)
        w={'complete':complete,'pid':pid+1 if foreign_pid else pid,'startup':receipt}
        write(args.output/'worker.json',w);return w
    monkeypatch.setattr(core,'_execute',completed)
    args=SimpleNamespace(device='cuda',plan=path,plan_sha256=rec.sha(path),index=0,output=tmp_path/'0',deadline_unix=time.time()+(-1 if expired else 120))
    return args


def test_target_wrapper_publishes_bound_resource_receipt_only_after_own_completion(tmp_path,monkeypatch):
    args=target_fixture(tmp_path,monkeypatch);w=target.run_component(args);r=core.read(args.output/'component-resource.json')
    assert r['kind']=='native-fixed-SDPA-component-resource-v3' and r['pid']==w['pid']==os.getpid()
    assert r['startup']==w['startup'] and r['worker_sha256']==rec.sha(args.output/'worker.json')
    assert r['plan_sha256']==args.plan_sha256 and r['peak_allocator_bytes']==1024
    assert not r['research_admitted'] and not r['performance_qualified']


@pytest.mark.parametrize('fault',('zero-peak','over-cap','incomplete','foreign-pid','bad-startup','expired'))
def test_target_wrapper_refuses_invalid_completion_or_resources_before_receipt(tmp_path,monkeypatch,fault):
    args=target_fixture(tmp_path,monkeypatch,peak=0 if fault=='zero-peak' else 8*2**30 if fault=='over-cap' else 1024,
        complete=fault!='incomplete',foreign_pid=fault=='foreign-pid',bad_startup=fault=='bad-startup',expired=fault=='expired')
    with pytest.raises(ValueError):target.run_component(args)
    assert (args.output/'worker.json').exists() and not (args.output/'component-resource.json').exists()


def test_real_wrapper_and_new_executor_keep_single_CUDA_configuration_owner(tmp_path,monkeypatch):
    import torch
    from models import execution,training
    monkeypatch.setattr(target.platform,'node',lambda:'linux10603');monkeypatch.setattr(target.platform,'python_version',lambda:'3.11.17')
    path=tmp_path/'plan.json';p={'local_tiny':False,'reference_dir':str(tmp_path/'reference'),'worker_seconds':120};write(path,p)
    monkeypatch.setattr(core,'validate_plan',lambda value:value);monkeypatch.setattr(core,'capture',lambda p:(None,None,None,None));monkeypatch.setattr(core,'previous',lambda *a:[])
    torch.use_deterministic_algorithms(True,warn_only=False)
    state={'initialized':False};events=[];configure=execution.configure_execution
    def configured(*a,**k):events.append('configure');return configure(*a,**k)
    monkeypatch.setattr(startup,'configure_execution',configured);monkeypatch.setattr(torch,'__version__','2.8.0+cu128');monkeypatch.setattr(torch.version,'cuda','12.8')
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:state['initialized']);monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    def properties(i):state['initialized']=True;events.append('properties');return SimpleNamespace(total_memory=8*2**30)
    monkeypatch.setattr(torch.cuda,'get_device_properties',properties)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda value:events.append(('allocator',value)))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:events.append('reset'))
    class AfterStartup(Exception):pass
    def stop(device):raise AfterStartup('stop-before-operands')
    monkeypatch.setattr(training,'runtime_contract',stop)
    args=SimpleNamespace(device='cuda',plan=path,plan_sha256=rec.sha(path),index=0,output=tmp_path/'0',deadline_unix=time.time()+120)
    with pytest.raises(AfterStartup):target.run_component(args)
    assert events==['configure','properties',('allocator',7/8),'reset']
    w=core.read(args.output/'worker.json');assert w['startup']['configure_calls']==1 and w['outputs']=={}
    assert not (args.output/'component-resource.json').exists()
