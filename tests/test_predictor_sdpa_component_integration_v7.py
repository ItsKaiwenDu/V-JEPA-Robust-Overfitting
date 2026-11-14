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
from models import predictor_sdpa_component_v7 as core
from models import predictor_sdpa_component_records_v2 as rec
from models import predictor_sdpa_component_startup_v2 as startup
from models import predictor_sdpa_component_target_v7 as target
ROOT=Path(__file__).resolve().parents[1]


def write(p,value):p.write_bytes(rec.encoded(value));p.chmod(0o600)

@pytest.fixture(scope='module')
def cohort(tmp_path_factory):
    root=tmp_path_factory.mktemp('components-v7')/'cohort'
    result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/check_predictor_sdpa_components_v7.py'),'--output',str(root)],
        cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (root.parent/'cohort.log').write_bytes(result.stdout)
    assert result.returncode==0,result.stdout.decode()[-7000:]
    return root


def test_four_real_fresh_CPU_workers_finish_with_v2_association_and_one_reference(cohort):
    p=core.validate_plan(core.read(cohort/'component-plan.json'));ps=rec.sha(cohort/'component-plan.json');identity=core.capture(p)[3]
    ref,gold=core.load_reference(cohort/'reference',p,ps,identity)
    workers=[core.load_worker(cohort/str(i),p,ps,identity,i) for i in range(4)]
    assert p['kind']=='native-SDPA-fixed-input-plan-v7' and len({w['pid'] for w in workers})==4
    assert [w['oracle_build_calls'] for w in workers]==[1,0,0,0] and ref['oracle_calls']==1
    assert ref['builder_pid']==workers[0]['pid'] and not ref['reference_is_passing_worker']
    for i,w in enumerate(workers):
        assert w['complete'] and w['flags_restored'] and w['inputs_unchanged'] and w['profile_replay_exact']
        assert w['startup']['configure_calls']==1 and w['startup']['device']=='cpu'
        assert w['backend_evidence']['kind']=='one-fixed-SDPA-correlated-backend-with-overhead-v1'
        assert w['backend_evidence']['CPU_only'] and not w['backend_evidence']['CUDA_forward_and_backward_qualified']
        raw=core.load_profile(cohort/str(i),w['trace'],w['runtime'])
        assert raw['kind']==p['trace_format'] and raw['torch_version']==w['runtime']['torch']
        assert w['trace']['encoding']==p['trace_encoding']=='single-JSON-envelope'
        assert w['trace']['metadata_policy']==p['metadata_policy']==core.METADATA_POLICY
        assert w['backend_evidence']['metadata_policy']==core.METADATA_POLICY
        assert len(raw['records'])==w['trace']['records'] and all(r['device']=='cpu' for r in raw['records'])
        assert w['backend_evidence']['recorded_host_correlation_links_available']
        assert not w['backend_evidence']['worker_completion_inferred'] and not w['performance_qualified'] and not w['research_admitted']
        for n in rec.RETURNS:
            assert w['outputs']['untimed'][n]['logical_sha256']==w['outputs']['profiled'][n]['logical_sha256']
            if i>=2:assert w['outputs']['untimed'][n]['logical_sha256']==workers[i-2]['outputs']['untimed'][n]['logical_sha256']
    state=core.read(cohort/'session.json');assert state['complete'] and state['CPU_only'] and not state['CUDA_qualified']


def test_only_version_policy_bindings_and_numerical_comparisons_change_frozen_ASTs():
    from models import predictor_sdpa_component_v6 as old
    from models import predictor_sdpa_component_records_v1 as old_records
    from models import predictor_sdpa_numerical_receipts_v1 as numerical
    source=(ROOT/'models/predictor_sdpa_component_v7.py').read_text()
    source=source.replace('_v7','_v6').replace('-v7','-v6').replace('predictor_sdpa_component_records_v2','predictor_sdpa_component_records_v1')
    source=source.replace("\n            or p.get('numerical_receipt_policy')!=NUMERICAL_POLICY",'')
    source=source.replace(" or w.get('numerical_receipt_policy')!=NUMERICAL_POLICY",'')
    source=source.replace(",'numerical_receipt_policy':NUMERICAL_POLICY",'')
    source=source.replace("numerical.compare_checks(checks,w.get('oracle_checks'),context='canonical-FP64')",
        "if checks!=w.get('oracle_checks') or not all(c['passed'] for c in checks.values()):raise ValueError('Actual saved component/reference numerical bound rejected.')")
    source=source.replace("numerical.compare_checks(actual_checks,w.get('passing_original_checks'),context='passing-original')",
        "if actual_checks!=w.get('passing_original_checks') or not all(c['passed'] for c in actual_checks.values()):raise ValueError('Saved candidate did not pass original component bounds.')")
    def functions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert functions(source)==functions((ROOT/'models/predictor_sdpa_component_v6.py').read_text())
    assert core.BOUNDS==old.BOUNDS==numerical.BOUNDS and core.BOOK==old.BOOK==numerical.BOOK and core.CELLS==old.CELLS
    assert core.METADATA_POLICY==old.METADATA_POLICY and core.NUMERICAL_POLICY==numerical.POLICY
    expected={n.replace('_v6','_v7').replace('predictor_sdpa_component_records_v1','predictor_sdpa_component_records_v2') for n in old.SOURCE_NAMES}
    assert set(core.SOURCE_NAMES)==expected|{'models/predictor_sdpa_numerical_receipts_v1.py'}
    assert functions((ROOT/'models/predictor_sdpa_component_records_v2.py').read_text())==functions((ROOT/'models/predictor_sdpa_component_records_v1.py').read_text())
    assert (rec.REFERENCE_KIND,rec.WORKER_KIND)==('native-SDPA-fixed-FP64-reference-v2','native-SDPA-fixed-input-component-v2')
    assert rec.LIMITS==old_records.LIMITS and rec.RETURNS==old_records.RETURNS
    wrapper=(ROOT/'models/predictor_sdpa_component_target_v7.py').read_text().replace('_v7','_v6').replace('-v7','-v6').replace('predictor_sdpa_component_records_v2','predictor_sdpa_component_records_v1')
    wrapper=wrapper.replace('\nfrom models.predictor_sdpa_numerical_receipts_v1 import POLICY as NUMERICAL_POLICY','')
    wrapper=wrapper.replace(" or w.get('numerical_receipt_policy')!=NUMERICAL_POLICY",'')
    wrapper=wrapper.replace(",'numerical_receipt_policy':NUMERICAL_POLICY",'')
    wrapper=wrapper.replace(",'numerical_receipt_policy'",'')
    wrapper=wrapper.replace("\n            or r['numerical_receipt_policy']!=NUMERICAL_POLICY or worker.get('numerical_receipt_policy')!=NUMERICAL_POLICY",'')
    assert ast.dump(ast.parse(wrapper),include_attributes=False)==ast.dump(ast.parse((ROOT/'models/predictor_sdpa_component_target_v6.py').read_text()),include_attributes=False)
    cli=(ROOT/'scripts/check_predictor_sdpa_components_v7.py').read_text().replace('_v7','_v6').replace('-v7','-v6').replace('predictor_sdpa_component_records_v2','predictor_sdpa_component_records_v1')
    cli=cli.replace(",'numerical_receipt_policy':component.NUMERICAL_POLICY",'')
    assert ast.dump(ast.parse(cli),include_attributes=False)==ast.dump(ast.parse((ROOT/'scripts/check_predictor_sdpa_components_v6.py').read_text()),include_attributes=False)



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
from models import predictor_sdpa_launch_evidence_v2 as events
from models import predictor_sdpa_component_v7 as core
def refuse(*a,**k):raise ValueError('injected-association-refusal-v7')
events.verify=refuse
core.run(SimpleNamespace(plan=Path(sys.argv[1]),plan_sha256=sys.argv[2],index=0,output=Path(sys.argv[3]),device='cpu',deadline_unix=time.time()+120))'''
    result=subprocess.run([sys.executable,'-B','-c',program,str(path),ps,str(tmp_path/'0')],cwd=ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (tmp_path/'worker.log').write_bytes(result.stdout);assert result.returncode==1
    w=rec.load(tmp_path/'0',kind=rec.WORKER_KIND,plan_sha256=ps,require_complete=False)
    assert w['numerical_passed'] and w['profile_replay_exact'] and w['error']=='injected-association-refusal-v7'
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
        w={'numerical_receipt_policy':core.NUMERICAL_POLICY,'complete':complete,'index':0,'plan_sha256':args.plan_sha256,'pid':pid+1 if foreign_pid else pid,'startup':receipt}
        write(args.output/'worker.json',w);return w
    monkeypatch.setattr(core,'_execute',completed)
    args=SimpleNamespace(device='cuda',plan=path,plan_sha256=rec.sha(path),index=0,output=tmp_path/'0',deadline_unix=time.time()+(-1 if expired else 120))
    return args


def test_target_wrapper_publishes_bound_resource_receipt_only_after_own_completion(tmp_path,monkeypatch):
    args=target_fixture(tmp_path,monkeypatch);w=target.run_component(args);r=core.read(args.output/'component-resource.json')
    assert r['kind']=='native-fixed-SDPA-component-resource-v7' and r['pid']==w['pid']==os.getpid()
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


def resign(folder,w):
    write(folder/'worker.json',w)
    write(folder/'complete.json',{'kind':w['kind'],'plan_sha256':w['plan_sha256'],'complete':True,
        'header':{'file':'worker.json','bytes':(folder/'worker.json').stat().st_size,'sha256':rec.sha(folder/'worker.json')}})


@pytest.mark.parametrize('fault',('old-format','wrong-encoding','wrong-version','wrong-count','bad-ID','wrong-envelope-version','wrong-publication','false-proof'))
def test_rehashed_trace_metadata_cannot_bypass_raw_runtime_or_backend_checks(cohort,tmp_path,fault):
    p=core.read(cohort/'component-plan.json');ps=rec.sha(cohort/'component-plan.json');identity=core.capture(p)[3]
    folder=tmp_path/'copy';shutil.copytree(cohort/'0',folder);w=core.read(folder/'worker.json')
    if fault=='old-format':w['trace']['format']='one-fixed-SDPA-associated-backend-v2'
    if fault=='wrong-encoding':w['trace']['encoding']='legacy-JSONL'
    if fault=='wrong-version':w['trace']['torch_version']='2.8.0+cu128'
    if fault=='wrong-count':w['trace']['records']+=1
    if fault in ('bad-ID','wrong-envelope-version'):
        t=core.read(folder/'events.jsonl')
        if fault=='bad-ID':t['records'][0]['correlation_id']=0
        else:t['torch_version']='invented runtime'
        write(folder/'events.jsonl',t);w['trace'].update(bytes=(folder/'events.jsonl').stat().st_size,sha256=rec.sha(folder/'events.jsonl'))
    if fault=='wrong-publication':w['trace']['saved_before_semantic_validation']=False
    if fault=='false-proof':w['backend_evidence']['kernel_name_or_duration_used_as_identity']=True
    resign(folder,w)
    with pytest.raises(ValueError):core.load_worker(folder,p,ps,identity,0)


def test_plan_refuses_legacy_trace_contract_before_execution(tmp_path):
    p=core.make_local_plan(ROOT,tmp_path/'reference')
    for field in ('trace_format','trace_encoding'):
        q=copy.deepcopy(p);q.pop(field)
        with pytest.raises(ValueError):core.validate_plan(q)


@pytest.mark.parametrize('field,value',(('kind','old-resource'),('index',True),('pid',0),('plan_sha256','other'),
 ('worker_sha256','other'),('peak_allocator_bytes',True),('peak_allocator_bytes',8*2**30),
 ('post_entry_wall_seconds',float('inf')),('research_admitted',True),('performance_qualified',True)))
def test_resource_readback_refuses_rewritten_or_mistyped_fields(tmp_path,monkeypatch,field,value):
    args=target_fixture(tmp_path,monkeypatch);w=target.run_component(args)
    r=core.read(args.output/'component-resource.json');r[field]=value
    # Deliberately bypass the canonical writer to test a malformed external number.
    (args.output/'component-resource.json').write_text(json.dumps(r,allow_nan=True))
    with pytest.raises(ValueError):target.load_resource(args.output,plan_sha256=args.plan_sha256,index=0,worker=w)


def test_resource_readback_binds_actual_saved_worker_hash(tmp_path,monkeypatch):
    args=target_fixture(tmp_path,monkeypatch);w=target.run_component(args)
    assert target.load_resource(args.output,plan_sha256=args.plan_sha256,index=0,worker=w)['pid']==w['pid']
    write(args.output/'worker.json',{**w,'changed_after_receipt':True})
    with pytest.raises(ValueError):target.load_resource(args.output,plan_sha256=args.plan_sha256,index=0,worker=w)


@pytest.mark.parametrize('fault',('symlink','hardlink','wrong-mode','oversize'))
def test_resource_readback_refuses_unsafe_file_metadata(tmp_path,monkeypatch,fault):
    args=target_fixture(tmp_path,monkeypatch);w=target.run_component(args);p=args.output/'component-resource.json'
    if fault=='symlink':
        moved=args.output/'original-receipt';p.rename(moved);p.symlink_to(moved)
    if fault=='hardlink':os.link(p,args.output/'second-receipt')
    if fault=='wrong-mode':p.chmod(0o644)
    if fault=='oversize':p.write_bytes(b'x'*(16*1024+1))
    with pytest.raises(ValueError):target.load_resource(args.output,plan_sha256=args.plan_sha256,index=0,worker=w)


def test_actual_raw_CPU_artifacts_pass_frozen_storage_and_collection(cohort,tmp_path):
    import io,stat
    from models import predictor_sdpa_continuation_storage_v2 as policy
    from models import predictor_sdpa_continuation_client_v2 as codec
    scratch=tmp_path/'scratch';scratch.mkdir(mode=0o700);(scratch/'results').mkdir(mode=0o700)
    shutil.copytree(cohort,scratch/'results/components')
    parent=tmp_path.stat();identity={'device':parent.st_dev,'inode':parent.st_ino}
    usage=policy.storage(scratch,identity,parent=tmp_path)
    assert usage['categories']['metadata']<64*2**20 and usage['categories']['raw']<512*2**20
    assert policy.typed_file_limit('results/components/0/events.jsonl')==8*2**20
    # Local codec rehearsal only. This fixed fixture ID cannot be a target grant.
    scope='local-CPU-raw-worker-v7-storage-rehearsal';out=io.BytesIO();policy.stream(scratch,scope,out)
    response=tmp_path/'local-codec-response';response.write_bytes(out.getvalue());saved=tmp_path/'saved'
    codec.unpack(response,saved,{},scope,collection_only=True)
    header=json.loads(response.read_bytes().split(b'\n',1)[0]);assert header['control_sha256']==scope
    for name,item in header['files'].items():
        p=saved/name;s=p.stat()
        assert rec.sha(p)==item['sha256'] and s.st_mtime_ns==item['mtime_ns'] and stat.S_IMODE(s.st_mode)==item['mode']
    for i in range(4):
        folder=saved/'results/components'/str(i);w=core.read(folder/'worker.json');trace=core.load_profile(folder,w['trace'],w['runtime'])
        from models import predictor_sdpa_launch_evidence_v2 as events
        assert events.verify(trace,arm=w['arm'],device='cpu')==w['backend_evidence']
        assert not (folder/'component-resource.json').exists()  # CPU results never claim a CUDA receipt.


def test_raw_trace_writer_preserves_overlimit_partial_without_completion(tmp_path):
    folder=tmp_path/'bounded';folder.mkdir(mode=0o700);store=rec.Store(folder)
    with pytest.raises(ValueError,match='serialized byte ceiling'):
        store.write('events.jsonl',lambda f:f.write(b'x'*(8*2**20+1)))
    assert (folder/'events.jsonl.partial').stat().st_size==8*2**20+1
    assert not (folder/'events.jsonl').exists() and not (folder/'complete.json').exists()


def test_missing_raw_API_refusal_retains_available_numerical_artifacts(tmp_path):
    p=core.make_local_plan(ROOT,tmp_path/'reference');plan=tmp_path/'plan.json';write(plan,p);ps=rec.sha(plan)
    program='''import os,sys,time
from pathlib import Path
from types import SimpleNamespace
os.umask(0o077)
from models import predictor_sdpa_raw_capture_v1 as capture
from models import predictor_sdpa_component_v7 as core
def refuse(*a,**k):raise ValueError('injected-missing-raw-API')
capture.capture=refuse
core.run(SimpleNamespace(plan=Path(sys.argv[1]),plan_sha256=sys.argv[2],index=0,output=Path(sys.argv[3]),device='cpu',deadline_unix=time.time()+120))'''
    result=subprocess.run([sys.executable,'-B','-c',program,str(plan),ps,str(tmp_path/'0')],cwd=ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (tmp_path/'worker.log').write_bytes(result.stdout);assert result.returncode==1
    w=rec.load(tmp_path/'0',kind=rec.WORKER_KIND,plan_sha256=ps,require_complete=False)
    assert w['error']=='injected-missing-raw-API' and w['numerical_passed'] and not w['complete']
    assert len(w['outputs']['untimed'])==len(w['outputs']['profiled'])==4 and 'trace' not in w
    assert (tmp_path/'reference/complete.json').exists() and not (tmp_path/'0/component-resource.json').exists()


def test_local_CLI_refuses_university_hosts_and_CUDA_without_work(monkeypatch):
    from scripts import check_predictor_sdpa_components_v7 as cli
    with pytest.raises(ValueError):cli.local_only('cuda')
    monkeypatch.setattr(cli.platform,'node',lambda:'linux10603')
    with pytest.raises(ValueError):cli.local_only('cpu')


def test_resource_publisher_is_private_under_permissive_caller_umask(tmp_path,monkeypatch):
    import stat
    previous=os.umask(0o022)
    try:
        args=target_fixture(tmp_path,monkeypatch);w=target.run_component(args)
        observed=os.umask(0o022)
        assert observed==0o022 and stat.S_IMODE((args.output/'component-resource.json').stat().st_mode)==0o600
        assert target.load_resource(args.output,plan_sha256=args.plan_sha256,index=0,worker=w)['kind']=='native-fixed-SDPA-component-resource-v7'
    finally:os.umask(previous)


def test_resource_publisher_restores_caller_umask_after_write_failure(tmp_path,monkeypatch):
    from models import cpu_oracle_scratch
    def refuse(*a,**k):raise OSError('injected-receipt-publication-failure')
    previous=os.umask(0o022)
    try:
        args=target_fixture(tmp_path,monkeypatch);monkeypatch.setattr(cpu_oracle_scratch,'write_json',refuse)
        with pytest.raises(OSError,match='publication-failure'):target.run_component(args)
        observed=os.umask(0o022)
        assert observed==0o022 and (args.output/'worker.json').exists() and not (args.output/'component-resource.json').exists()
    finally:os.umask(previous)


def test_actual_worker_saves_invalid_raw_field_before_strict_refusal(tmp_path):
    # The sentinel is injected metadata, not a recovered record from the failed GPU.
    p=core.make_local_plan(ROOT,tmp_path/'reference');path=tmp_path/'plan.json';write(path,p);ps=rec.sha(path)
    program="""import json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
os.umask(0o077)
from models import predictor_sdpa_raw_capture_v1 as capture
from models import predictor_sdpa_launch_evidence_v2 as frozen
from models import predictor_sdpa_component_v7 as core
from models import predictor_sdpa_component_records_v2 as rec
original=capture.capture;project=frozen.project
folder=Path(sys.argv[3])
def malformed(profile,**kwargs):
    t=original(profile,**kwargs);t['records'][0]['device_index']=-1;return t
capture.capture=malformed
def verify_published(t):
    saved=json.loads((folder/'events.jsonl').read_text());w=json.loads((folder/'worker.json').read_text())
    assert saved==t and saved['records'][0]['device_index']==-1
    assert w['trace']['saved_before_semantic_validation'] is True
    assert rec.sha(folder/'events.jsonl')==w['trace']['sha256']
    print('Raw saved bytes and descriptor verified before strict semantic validation.',flush=True)
    return project(t)
frozen.project=verify_published
core.run(SimpleNamespace(plan=Path(sys.argv[1]),plan_sha256=sys.argv[2],index=0,output=folder,device='cpu',deadline_unix=time.time()+120))"""
    result=subprocess.run([sys.executable,'-B','-c',program,str(path),ps,str(tmp_path/'0')],cwd=ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (tmp_path/'worker.log').write_bytes(result.stdout)
    assert result.returncode==1 and b'Raw saved bytes and descriptor verified before strict semantic validation.' in result.stdout
    w=rec.load(tmp_path/'0',kind=rec.WORKER_KIND,plan_sha256=ps,require_complete=False)
    assert w['error']=='Unknown or malformed negative-device profiler record; no name-only exclusion.' and w['numerical_passed'] and w['native_baseline_exact']
    assert not w['complete'] and not w['profile_replay_exact'] and not w['flags_restored'] and not w['inputs_unchanged']
    assert 'backend_evidence' not in w and w['trace']['saved_before_semantic_validation'] is True
    t=core.load_profile(tmp_path/'0',w['trace'],w['runtime']);assert t['records'][0]['device_index']==-1
    from models import predictor_sdpa_launch_evidence_v2 as frozen
    with pytest.raises(ValueError,match='Unknown or malformed negative-device'):frozen.project(t)
    shape=core.capture(p)[3]['layouts']['q']['shape']
    for phase in ('untimed','profiled'):
        for n,item in w['outputs'][phase].items():
            rec.array(tmp_path/'0',item,dtype='<f4',shape=shape)
            assert item['logical_sha256']==w['outputs']['untimed'][n]['logical_sha256']
    assert (tmp_path/'0/failure.json').exists() and (tmp_path/'reference/complete.json').exists()
    assert not (tmp_path/'0/complete.json').exists() and not (tmp_path/'0/component-resource.json').exists()


@pytest.mark.parametrize('value',(None,False,'frozen-raw-v1','fixed-SDPA-profiler-overhead-signature-v0',{}))
def test_explicit_plan_metadata_policy_cannot_be_omitted_or_replaced(tmp_path,value):
    p=core.make_local_plan(ROOT,tmp_path/'reference');p['metadata_policy']=value
    with pytest.raises(ValueError):core.validate_plan(p)


@pytest.mark.parametrize('fault',('missing','old-policy','wrong-type','proof-policy','proof-admission','missing-role-list'))
def test_rehashed_policy_descriptor_or_proof_cannot_bypass_version_binding(cohort,tmp_path,fault):
    p=core.read(cohort/'component-plan.json');ps=rec.sha(cohort/'component-plan.json');identity=core.capture(p)[3]
    path=tmp_path/'copy';shutil.copytree(cohort/'0',path);w=core.read(path/'worker.json')
    if fault=='missing':w['trace'].pop('metadata_policy')
    if fault=='old-policy':w['trace']['metadata_policy']='frozen-raw-v1'
    if fault=='wrong-type':w['trace']['metadata_policy']=True
    if fault=='proof-policy':w['backend_evidence']['metadata_policy']='wrong'
    if fault=='proof-admission':w['backend_evidence']['worker_completion_inferred']=True
    if fault=='missing-role-list':w['backend_evidence'].pop('retained_overhead_indices')
    resign(path,w)
    with pytest.raises(ValueError):core.load_worker(path,p,ps,identity,0)


def test_old_plans_and_readers_cannot_reuse_new_completion(cohort,tmp_path):
    from models import predictor_sdpa_component_v6 as old
    p=core.read(cohort/'component-plan.json');ps=rec.sha(cohort/'component-plan.json');identity=core.capture(p)[3]
    with pytest.raises(ValueError):old.validate_plan(p)
    with pytest.raises(ValueError):core.validate_plan(old.make_local_plan(ROOT,tmp_path/'old-reference'))
    with pytest.raises(ValueError):old.load_worker(cohort/'0',p,ps,identity,0)


def test_actual_saved_GPU_metadata_passes_new_provider_without_closing_historical_worker(tmp_path):
    from models import predictor_sdpa_launch_evidence_v2 as policy
    from models import predictor_sdpa_launch_evidence_v1 as frozen
    final=ROOT/'runs/pretrained/diagnostic/predictor-sdpa-continuation-v4-20261113/linux10603-final'
    original=final/'attended-6ub5e6wx/saved/results/components/0';before=rec.sha(original/'worker.json')
    claim_before=rec.sha(final/'ATTEMPT.json');mtime=(final/'ATTEMPT.json').stat().st_mtime_ns
    local=tmp_path/'saved-GPU-metadata-fixture';local.mkdir(mode=0o700)
    data=(original/'events.jsonl').read_bytes();assert __import__('hashlib').sha256(data).hexdigest()=='d1b2006c583c6cfc4d8dc0bb0b12e823293222079c740d6d78acb78a72382686'
    store=rec.Store(local);item=store.write('events.jsonl',lambda f:f.write(data))
    item.update(records=106,format=policy.KIND,encoding='single-JSON-envelope',torch_version='2.8.0+cu128',
        saved_before_semantic_validation=True,metadata_policy=core.METADATA_POLICY)
    trace=core.load_profile(local,item,{'torch':'2.8.0+cu128'});proof=policy.verify(trace,arm='original',device='cuda')
    assert proof['retained_overhead_indices']==[10] and proof['retained_GPU_annotation_indices']==[20]
    assert proof['raw_records']==proof['projected_records']==106 and len(proof['kernel_witnesses'])==12
    assert not proof['worker_completion_inferred'] and not proof['actual_CUDA_execution_admitted']
    assert not proof['CUDA_forward_and_backward_qualified']
    with pytest.raises(ValueError):core.load_profile(local,item,{'torch':'2.14.1'})
    with pytest.raises(ValueError):frozen.project(trace)
    w=core.read(original/'worker.json')
    assert not w['complete'] and not w['flags_restored'] and not w['inputs_unchanged'] and not w['profile_replay_exact']
    assert not (original/'complete.json').exists() and not (original/'component-resource.json').exists()
    assert rec.sha(original/'worker.json')==before and rec.sha(final/'ATTEMPT.json')==claim_before
    assert (final/'ATTEMPT.json').stat().st_mtime_ns==mtime
    write(local/'fixture.json',{'local_saved_metadata_fixture_only':True,'historical_worker_changed':False,
        'raw_trace_descriptor':item,'metadata_proof':proof,'model_or_CUDA_work_started':False})


def test_old_resource_kind_cannot_be_published_as_v7(tmp_path,monkeypatch):
    args=target_fixture(tmp_path,monkeypatch);w=target.run_component(args)
    r=core.read(args.output/'component-resource.json');r['kind']='native-fixed-SDPA-component-resource-v6'
    write(args.output/'component-resource.json',r)
    with pytest.raises(ValueError):target.load_resource(args.output,plan_sha256=args.plan_sha256,index=0,worker=w)


@pytest.mark.parametrize('value',(None,False,'old-policy','fixed-SDPA-numerical-receipt-bookkeeping-v0',{}))
def test_new_numerical_plan_policy_is_explicit_and_exact(tmp_path,value):
    p=core.make_local_plan(ROOT,tmp_path/'reference');p['numerical_receipt_policy']=value
    with pytest.raises(ValueError):core.validate_plan(p)


def test_plan_cannot_omit_new_numerical_policy_or_its_source_pin(tmp_path):
    for fault in ('policy','source'):
        p=core.make_local_plan(ROOT,tmp_path/fault/'reference')
        if fault=='policy':p.pop('numerical_receipt_policy')
        else:p['source_sha256'].pop('models/predictor_sdpa_numerical_receipts_v1.py')
        with pytest.raises(ValueError):core.validate_plan(p)


def test_actual_cohort_binds_policy_and_distinct_reference_worker_markers(cohort):
    p=core.read(cohort/'component-plan.json');ps=rec.sha(cohort/'component-plan.json')
    assert p['numerical_receipt_policy']==core.NUMERICAL_POLICY
    ref=rec.load(cohort/'reference',kind=rec.REFERENCE_KIND,plan_sha256=ps)
    assert ref['numerical_receipt_policy']==core.NUMERICAL_POLICY and ref['kind'].endswith('-v2')
    for index in range(4):
        w=rec.load(cohort/str(index),kind=rec.WORKER_KIND,plan_sha256=ps)
        marker=core.read(cohort/str(index)/'complete.json')
        assert w['numerical_receipt_policy']==core.NUMERICAL_POLICY and w['kind']==marker['kind']==rec.WORKER_KIND
    state=core.read(cohort/'session.json')
    assert state['numerical_receipt_policy']==core.NUMERICAL_POLICY


@pytest.mark.parametrize('fault',('missing','wrong','wrong-type','old-kind'))
def test_rehashed_worker_policy_and_record_version_are_not_optional(cohort,tmp_path,fault):
    p=core.read(cohort/'component-plan.json');ps=rec.sha(cohort/'component-plan.json');identity=core.capture(p)[3]
    path=tmp_path/'copy';shutil.copytree(cohort/'0',path);w=core.read(path/'worker.json')
    if fault=='missing':w.pop('numerical_receipt_policy')
    elif fault=='wrong':w['numerical_receipt_policy']='foreign'
    elif fault=='wrong-type':w['numerical_receipt_policy']=True
    else:w['kind']='native-SDPA-fixed-input-component-v1'
    resign(path,w)
    with pytest.raises(ValueError):core.load_worker(path,p,ps,identity,0)


@pytest.mark.parametrize('fault',('missing-policy','wrong-policy','old-kind'))
def test_rehashed_reference_cannot_change_numerical_policy_or_version(cohort,tmp_path,fault):
    folder=tmp_path/'reference';shutil.copytree(cohort/'reference',folder)
    p=core.read(cohort/'component-plan.json');ps=rec.sha(cohort/'component-plan.json');identity=core.capture(p)[3]
    w=core.read(folder/'reference.json')
    if fault=='missing-policy':w.pop('numerical_receipt_policy')
    elif fault=='wrong-policy':w['numerical_receipt_policy']='foreign'
    else:w['kind']='native-SDPA-fixed-FP64-reference-v1'
    write(folder/'reference.json',w);write(folder/'complete.json',{'kind':w['kind'],'plan_sha256':ps,'complete':True,
        'header':{'file':'reference.json','bytes':(folder/'reference.json').stat().st_size,'sha256':rec.sha(folder/'reference.json')}})
    with pytest.raises(ValueError):core.load_reference(folder,p,ps,identity)


@pytest.mark.parametrize('field,index',(('oracle_checks',0),('passing_original_checks',1)))
def test_generated_values_accept_last_bit_diagnostic_receipt_rounding(cohort,tmp_path,field,index):
    import math
    folder=tmp_path/'copy';shutil.copytree(cohort,folder)
    p=core.read(folder/'component-plan.json');ps=rec.sha(folder/'component-plan.json');p['reference_dir']=str(folder/'reference')
    identity=core.capture(p)[3];path=folder/str(index);w=core.read(path/'worker.json')
    for name in rec.RETURNS:
        if w[field][name]['max_absolute_error']!=0.0:
            w[field][name]['relative_l2_error']=math.nextafter(w[field][name]['relative_l2_error'],math.inf)
    resign(path,w)
    assert core.load_worker(path,p,ps,identity,index)['complete']


@pytest.mark.parametrize('field,index',(('oracle_checks',0),('passing_original_checks',1)))
@pytest.mark.parametrize('fault',('outside-bookkeeping','max-error','false-pass','nonfinite-L2'))
def test_generated_values_still_refuse_wrong_receipt_at_both_comparison_sites(cohort,tmp_path,field,index,fault):
    folder=tmp_path/'copy';shutil.copytree(cohort,folder)
    p=core.read(folder/'component-plan.json');ps=rec.sha(folder/'component-plan.json');p['reference_dir']=str(folder/'reference')
    identity=core.capture(p)[3];path=folder/str(index);w=core.read(path/'worker.json');check=w[field]['output']
    if fault=='outside-bookkeeping':check['relative_l2_error']+=1e-5
    elif fault=='max-error':check['max_absolute_error']+=1e-20
    elif fault=='false-pass':check['passed']=False
    else:check['relative_l2_error']='nan'
    resign(path,w)
    with pytest.raises(ValueError):core.load_worker(path,p,ps,identity,index)


def test_forged_passing_header_cannot_hide_an_actual_tensor_bound_violation(cohort,tmp_path):
    import hashlib,numpy as np
    folder=tmp_path/'copy';shutil.copytree(cohort,folder)
    p=core.read(folder/'component-plan.json');ps=rec.sha(folder/'component-plan.json');p['reference_dir']=str(folder/'reference')
    identity=core.capture(p)[3];path=folder/'0';w=core.read(path/'worker.json')
    for phase in ('untimed','profiled'):
        item=w['outputs'][phase]['dq'];file=path/item['file'];array=np.load(file,allow_pickle=False);array.flat[0]+=1.0
        with file.open('wb') as out:np.save(out,array,allow_pickle=False)
        item.update(bytes=file.stat().st_size,sha256=rec.sha(file),logical_sha256=hashlib.sha256(array.tobytes()).hexdigest())
    resign(path,w)
    with pytest.raises(ValueError,match='elementwise acceptance'):core.load_worker(path,p,ps,identity,0)


@pytest.mark.parametrize('value',(None,False,'old-policy',{}))
def test_resource_receipt_requires_exact_numerical_policy(tmp_path,monkeypatch,value):
    args=target_fixture(tmp_path,monkeypatch);w=target.run_component(args)
    r=core.read(args.output/'component-resource.json');assert r['numerical_receipt_policy']==core.NUMERICAL_POLICY
    r['numerical_receipt_policy']=value;write(args.output/'component-resource.json',r)
    with pytest.raises(ValueError):target.load_resource(args.output,plan_sha256=args.plan_sha256,index=0,worker=w)


def test_resource_cannot_bind_a_worker_with_foreign_policy(tmp_path,monkeypatch):
    args=target_fixture(tmp_path,monkeypatch);w=target.run_component(args);w['numerical_receipt_policy']='foreign'
    write(args.output/'worker.json',w);r=core.read(args.output/'component-resource.json')
    r['worker_sha256']=rec.sha(args.output/'worker.json');write(args.output/'component-resource.json',r)
    with pytest.raises(ValueError):target.load_resource(args.output,plan_sha256=args.plan_sha256,index=0,worker=w)


def test_old_record_reader_cannot_load_new_worker_or_reference(cohort):
    from models import predictor_sdpa_component_records_v1 as old
    ps=rec.sha(cohort/'component-plan.json')
    with pytest.raises(ValueError):old.load(cohort/'0',kind=old.WORKER_KIND,plan_sha256=ps)
    with pytest.raises(ValueError):old.load(cohort/'reference',kind=old.REFERENCE_KIND,plan_sha256=ps)


def test_historical_target_completion_stays_in_its_old_failed_domain():
    from models import predictor_sdpa_component_records_v1 as old
    folder=ROOT/'runs/pretrained/diagnostic/predictor-sdpa-continuation-v5-20261113/linux10603-final'
    saved=folder/'attended-vlbwjaqq/saved';before=rec.sha(folder/'ATTEMPT.json')
    ps=rec.sha(saved/'results/components/component-plan.json')
    assert old.load(saved/'results/components/0',kind=old.WORKER_KIND,plan_sha256=ps)['complete']
    with pytest.raises(ValueError):rec.load(saved/'results/components/0',kind=rec.WORKER_KIND,plan_sha256=ps)
    with pytest.raises(ValueError):core.validate_plan(core.read(saved/'results/components/component-plan.json'))
    assert not core.read(saved/'session.json')['complete'] and rec.sha(folder/'ATTEMPT.json')==before


def test_four_fresh_processes_continue_when_serialized_diagnostics_change_one_last_bit(tmp_path,monkeypatch):
    # Controlled final-summary serialization only. Original attention arithmetic,
    # elementwise checks, input arrays, raw profiling and replay stay unchanged.
    from scripts import check_predictor_sdpa_components_v7 as cli
    program=r"""import copy,math,runpy,sys
from models import predictor_sdpa_component_records_v2 as rec
original=rec.Store.json
def rounded(self,name,value):
    if name=='worker.json' and value.get('complete') is True:
        value=copy.deepcopy(value)
        for key in ('oracle_checks','passing_original_checks'):
            for check in value.get(key,{}).values():
                if check['max_absolute_error']!=0.0:
                    check['relative_l2_error']=math.nextafter(check['relative_l2_error'],math.inf)
    return original(self,name,value)
rec.Store.json=rounded
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    original_run=subprocess.run
    def run(cmd,**kwargs):
        assert cmd[:2]==[sys.executable,'-B'] and '--worker' in cmd
        return original_run([sys.executable,'-B','-c',program,*cmd[2:]],**kwargs)
    monkeypatch.setattr(cli.subprocess,'run',run)
    folder=tmp_path/'rounded-cohort';state=cli.cohort(folder)
    assert state['complete'] and state['numerical_receipt_policy']==core.NUMERICAL_POLICY
    p=core.read(folder/'component-plan.json');ps=rec.sha(folder/'component-plan.json');identity=core.capture(p)[3]
    workers=[core.load_worker(folder/str(i),p,ps,identity,i) for i in range(4)]
    assert len({w['pid'] for w in workers})==4 and [w['oracle_build_calls'] for w in workers]==[1,0,0,0]
    from models import attention_saved_evidence as arithmetic
    _,gold=core.load_reference(folder/'reference',p,ps,identity)
    for w in workers:
        for name in rec.RETURNS:
            actual=rec.array(folder/str(w['index']),w['outputs']['untimed'][name],dtype='<f4',shape=identity['layouts']['q']['shape'])
            recomputed=arithmetic.agreement(actual,gold[name],core.BOUNDS[name])
            assert recomputed['passed'] and recomputed['failed_elements']==0
            if recomputed['max_absolute_error']!=0.0:
                assert recomputed['relative_l2_error']!=w['oracle_checks'][name]['relative_l2_error']
            else:assert w['oracle_checks'][name]['relative_l2_error']==0.0
    write(folder/'controlled-rounding-fixture.json',{'actual_fresh_CPU_workers':4,'diagnostic_serialization_injected':True,
        'original_attention_arithmetic_changed':False,'new_target_work':False})
