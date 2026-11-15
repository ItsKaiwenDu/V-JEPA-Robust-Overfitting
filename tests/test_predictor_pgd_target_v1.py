"""Cold preparations and explicit simulations only; no new native/model objectives."""
import ast
import base64
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest
import torch

from attacks import pgd_pretrained_v1 as attack
from models import predictor_pgd_target_records_v1 as records
from models import predictor_pgd_target_operator_v1 as op
from models import predictor_pgd_target_client_v1 as client
from models import predictor_pgd_target_worker_v1 as worker

ROOT=Path(__file__).resolve().parents[1]


def script(name):
    p=ROOT/'scripts'/name;spec=importlib.util.spec_from_file_location(name.removesuffix('.py'),p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def control(folder):
    text=(ROOT/op.CONTRACT).read_text();p=json.loads(text)
    archive=p['reference']['existing_packet_archive'];spec=json.loads((ROOT/Path(archive).parent/'vjepa-eot-profile.tar.gz.manifest.json').read_bytes())
    key=(ROOT/'runs/pretrained/diagnostic/attention-replay1-v1-20261105/linux10603/known-hosts').read_text()
    sources={n:op.VIRTUAL[n] if n in op.VIRTUAL else (ROOT/n).read_bytes() for n in op.REMOTE_NAMES}
    return {'kind':op.KIND,'host':'linux10603','parent':'/var/tmp','expected_device':66307,'expected_mount':'/',
       'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f','expected_driver':'580.178.04','expected_ssh_fingerprint':op.base.SSH_FINGERPRINT,
       'contract_sha256':op.CONTRACT_SHA,'contract_text':text,'contract':p,'resource':p['resource'],'paid_compute':False,'research_admitted':False,
       'authorization_present_at_preparation':False,'automatic_retries':0,'repository':str(ROOT),'local_handoff_directory':str(folder),
       'archive_path':archive,'archive_manifest':spec,'known_hosts_sha256':hashlib.sha256(key.encode()).hexdigest(),
       'remote_source_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in sources.items()},
       'local_source_sha256':{n:op.old.digest(ROOT/n) for n in sorted((set(op.REMOTE_NAMES)-set(op.VIRTUAL))|set(op.LOCAL_EXTRA))},
       'prior_inputs_manifest':p['prior_inputs_manifest'],'prior_inputs_manifest_sha256':op.old.digest(ROOT/p['prior_inputs_manifest']),
       'host_binding':{'prior_consumed_files':p['prior_consumed_files']}}


@pytest.fixture
def private(tmp_path):
    previous=os.umask(0o077)
    try:yield tmp_path
    finally:os.umask(previous)


def test_actual_gpu_PGD_AST_changes_only_identity_and_declared_guard():
    def function(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    a=function(ROOT/'attacks/pgd_context_v1.py','pgd_linf_context_v1')
    b=function(ROOT/'attacks/pgd_pretrained_v1.py','pgd_linf_pretrained_v1')
    b.name=a.name
    guard=next(n for n in b.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='target_domain')
    guard.value.func.id='tiny_domain'
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    assert op.old.digest(ROOT/'attacks/pgd_context_v1.py')==attack.CPU_PGD_SHA256


@pytest.mark.parametrize('host,system', [('linux10609','Linux'),('linux10603','Darwin'),('local','Linux')])
def test_gpu_domain_refuses_before_tensor_access(monkeypatch,host,system):
    monkeypatch.setattr(attack.platform,'node',lambda:host);monkeypatch.setattr(attack.platform,'system',lambda:system)
    with pytest.raises(ValueError,match='requires linux10603'):
        attack.target_domain(None,None,None,None,None,None,None)


@pytest.mark.parametrize('mode',['inference','autocast','cpu','steps','restarts'])
def test_unsupported_gpu_domain_modes_counts_before_model_work(monkeypatch,mode):
    monkeypatch.setattr(attack.platform,'node',lambda:'linux10603');monkeypatch.setattr(attack.platform,'system',lambda:'Linux')
    c=torch.full((1,3,4,16,16),.5)
    ctx=torch.inference_mode() if mode=='inference' else torch.autocast('cpu') if mode=='autocast' else __import__('contextlib').nullcontext()
    with ctx,pytest.raises(ValueError,match='Only declared'):
        attack.target_domain(c,4/255,1/255,3 if mode=='steps' else 2,3 if mode=='restarts' else 2,attack.START_SEED,True)


def simulated_algorithm(*,fail=False):
    # Execute the actual new PGD body, replacing ONLY the GPU admission guard in
    # a private namespace. Analytic zeros supply gradients; no autograd runs.
    run=FunctionType(attack.pgd_linf_pretrained_v1.__code__,dict(attack.pgd_linf_pretrained_v1.__globals__,target_domain=lambda *a:None))
    seen=[];active=[False]
    def objective(value):
        assert not torch.is_grad_enabled() and not active[0]
        seen.append('score');return torch.ones(1)
    def context(value):
        class Scope:
            def __enter__(self):active[0]=True;self.losses=torch.ones(1);seen.append('forward');return self
            def note_consider(self,current,losses):assert current is value and losses is self.losses;seen.append('consider')
            def gradient(self):
                assert seen[-1]=='consider';seen.append('backward')
                if fail:raise RuntimeError('SIMULATED failure')
                return torch.zeros_like(value)
            def __exit__(self,*a):active[0]=False;seen.append('exit')
        return Scope()
    c=torch.full((1,3,4,16,16),.5)
    return run,objective,context,c,seen,active


def test_actual_gpu_body_simulation_retains_ties_scores_and_order():
    run,obj,ctx,c,seen,active=simulated_algorithm()
    r=run(obj,c,epsilon=4/255,step_size=1/255,steps=2,restarts=2,seed=attack.START_SEED,random_start=True,gradient_context=ctx)
    assert torch.equal(r.adversarial,c) and torch.equal(r.losses,torch.ones(1))
    assert seen.count('score')==3 and seen.count('forward')==seen.count('consider')==seen.count('backward')==seen.count('exit')==4
    assert not active[0]


def test_actual_gpu_body_simulation_failure_stops_later_steps():
    run,obj,ctx,c,seen,active=simulated_algorithm(fail=True)
    with pytest.raises(RuntimeError,match='SIMULATED'):
        run(obj,c,epsilon=4/255,step_size=1/255,steps=2,restarts=2,seed=attack.START_SEED,random_start=True,gradient_context=ctx)
    assert seen==['score','forward','consider','backward','exit'] and not active[0]


def test_scientific_worker_refuses_local_before_Torch_imports(monkeypatch):
    monkeypatch.setattr(worker.platform,'node',lambda:'local')
    with pytest.raises(ValueError,match='Only the separately'):
        worker.run(None)


def test_new_identity_rejects_old_clean_worker_kind():
    with pytest.raises(ValueError,match='new perturbed'):
        records.identity({'kind':records.frozen.WORKER_KIND})
    s=dict(kind=records.OBJECTIVE,index=7,step=3,mode='candidate',execution_host='linux10603',local_tiny=False,pid=123,control_sha256='1'*64)
    assert records.identity(s)['index']==7


def test_frozen_scope_and_writer_code_objects_are_retained():
    assert records._native_scope.__code__ is records.native_record.verify_scope.__code__
    assert records._scope.__code__ is records.frozen.check_scope.__code__
    assert records.write_final.__code__ is records.frozen.write_final.__code__


def test_storage_predicate_body_matches_reviewed_logical_allocated_scope():
    source=(ROOT/'models/predictor_missing_replay_operator_v1.py').read_text()
    a=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='storage')
    b=next(n for n in ast.parse((ROOT/'models/predictor_pgd_target_operator_v1.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='storage')
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


def test_plan_and_full_raw_partial_proof_bounds(private):
    c=control(private);op.validate_control(c)
    report=script('prepare_predictor_pgd_target_v1.py').resource_bound(c['contract'])
    assert report['eight_objectives_plus_rollout_and_partial_bytes']<1792*op.MiB
    assert report['all_capture_category_caps_bytes']<9*op.GiB//4
    assert report['proof_phase_category_bound_bytes']<3*op.GiB
    assert c['contract']['workload']['seeds']['start']==attack.START_SEED
    assert c['contract']['workload']['total_new_gradients_max']==8


@pytest.mark.parametrize('field,value',[('host','linux10609'),('automatic_retries',1),('paid_compute',True),('expected_driver','other'),('expected_gpu_uuid','other'),('expected_device',55)])
def test_control_changes_refused(private,field,value):
    c=control(private);c[field]=value
    with pytest.raises(ValueError):op.validate_control(c)


@pytest.mark.parametrize('field,value',[('maximum_new_attempts',2),('maximum_new_gradient_objectives',9),('additional_profile_objectives',1),('cleanup',True),('reset_old_claims',True)])
def test_exact_new_approval_cannot_expand_scope(private,field,value):
    c=control(private);a=op.authorization_template(c,'1'*64,'SIMULATION only')
    op.validate_authorization(a,'1'*64,c);a[field]=value
    with pytest.raises(ValueError,match='exact scoped'):op.validate_authorization(a,'1'*64,c)


def test_unarmed_client_refuses_before_source_scan_claim_or_connection(private,monkeypatch):
    monkeypatch.setattr(client,'check_local',lambda *a:pytest.fail('must not scan without approval'))
    with pytest.raises(ValueError,match='Unarmed'):
        client.run(private,popen=lambda *a,**k:pytest.fail('No SSH in preparation tests'))
    assert not list(private.iterdir())


def test_download_mode_and_payload_collect_refused(private):
    with pytest.raises(ValueError,match='Separate retrieval'):client.run(private,download_only=True)
    with pytest.raises(ValueError,match='separate retrieval'):client.payload({},collect='/tmp/old')
    with pytest.raises(ValueError,match='same-session'):client.unpack(None,None,{},'1'*64,collect=True)


def test_empty_saved_response_refuses_and_keeps_source(private):
    c=control(private);sha=hashlib.sha256(op.old.json_bytes(c)).hexdigest()
    p=private/'response.bin';p.write_bytes(b'')
    with pytest.raises((ValueError,json.JSONDecodeError)):
        client.unpack(p,private/'saved',c,sha)
    assert p.read_bytes()==b''


def test_actual_inventory_private_files_and_external_hardlink_refusal(private):
    root=private/'scratch';root.mkdir(mode=0o700);(root/'control.json').write_text('{}')
    files,_=op.inventory(root,hashing=True);assert set(files)=={'control.json'}
    outside=private/'outside';outside.write_text('x');os.link(outside,root/'external.log')
    with pytest.raises(ValueError,match='inventory='):
        op.inventory(root)


def test_actual_inventory_typed_npy_cap_and_other_result_root_refused(private):
    root=private/'scratch';root.mkdir(mode=0o700);d=root/'results/0';d.mkdir(parents=True,mode=0o700)
    with (d/'loss.npy').open('wb') as f:f.truncate(65537)
    with pytest.raises(ValueError,match='Typed iterated'):op.inventory(root)
    with pytest.raises(ValueError,match='Only results/0'):op.category('results/1/input.npy')


def test_private_overlay_bootstrap_resolves_dependencies_and_unarmed_refuses(private):
    c=control(private);data=client.payload(c)
    assert len(data)<2*op.MiB
    # Execute the actual in-memory bootstrap up to m.main. Only call its pure
    # validator; no authorization, scratch, process, proof or model is entered.
    text=data.decode();head=text[:text.rindex('sys.exit(m.main(')]
    captured={}
    import subprocess
    payload=head+"\nprint(json.dumps({'kind':m.KIND,'validated':m.validate_control(json.loads(base64.b64decode("+repr(base64.b64encode(op.old.json_bytes(c)).decode())+"))) is None,'torch_loaded':'torch' in sys.modules}))\n"
    r=subprocess.run([sys.executable,'-I','-B','-c',payload],capture_output=True,text=True,timeout=20)
    assert r.returncode==0,r.stderr
    got=json.loads(r.stdout);assert got=={'kind':op.KIND,'validated':True,'torch_loaded':False}


def test_attacks_overlay_imports_new_body_and_original_objective(private):
    root=private/'scratch';overlay=root/'overlay';code=root/'code';overlay.mkdir(parents=True,mode=0o700);code.mkdir(mode=0o700)
    (overlay/'attacks').mkdir(mode=0o700);(code/'attacks').mkdir(mode=0o700)
    (overlay/'attacks/__init__.py').write_bytes(op.ATTACK_INIT)
    (overlay/'attacks/pgd_pretrained_v1.py').write_bytes((ROOT/'attacks/pgd_pretrained_v1.py').read_bytes())
    (code/'attacks/__init__.py').write_bytes(b'')
    (code/'attacks/pgd.py').write_bytes((ROOT/'attacks/pgd.py').read_bytes())
    import subprocess
    program='import sys;sys.path.insert(0,'+repr(str(overlay))+');import attacks.pgd_pretrained_v1 as a;print(a.IDENTITY)'
    r=subprocess.run([sys.executable,'-I','-B','-c',program],capture_output=True,text=True,timeout=20)
    assert r.returncode==0,r.stderr
    assert r.stdout.strip()==records.ALGORITHM


def backend_fixture(role):
    # Explicit synthetic operator/kernel observations for cold validator tests.
    rows=[]
    for r,n in [('encoder',24),('predictor',12)]:
        for i in range(n):rows.append({'kind':'efficient-forward','owner':r+'.'+str(i),'operator':'aten::_scaled_dot_product_efficient_attention','calls':32,'kernels':{'fmha_cutlassF_SIMULATED':{'calls':32,'duration_us':1.}}})
    rows.append({'kind':'efficient-backward','owner':'native-autograd','operator':'aten::_scaled_dot_product_efficient_attention_backward','calls':576 if role=='original' else 384,'kernels':{'fmha_cutlassB_SIMULATED':{'calls':1,'duration_us':1.}}})
    if role=='candidate':rows.append({'kind':'inner-math','owner':'custom-predictor','operator':'aten::_scaled_dot_product_attention_math','calls':192,'kernels':{'SIMULATED_sgemm':{'calls':192,'duration_us':1.},'SIMULATED_softmax':{'calls':192,'duration_us':1.}}})
    return {'kind':'pretrained-pgd-attention-backend-aggregate-v1','role':role,'event_count':1000,'rows':rows,'options':{'record_shapes':False,'with_stack':False,'profile_memory':False},'extra_objectives':0,'warmups':0,'performance_qualified':False}


@pytest.mark.parametrize('role',['original','candidate'])
def test_synthetic_backend_contract(role):
    assert records.validate_backend(backend_fixture(role),role)['passed']


@pytest.mark.parametrize('fault',['owner','forward-count','backward-count','math-count','math-kernel','warmup','performance'])
def test_synthetic_backend_missing_or_expanded_evidence_rejected(fault):
    value=backend_fixture('candidate')
    if fault=='owner':value['rows'][0]['owner']='wrong'
    elif fault=='forward-count':value['rows'][0]['calls']=31
    elif fault=='backward-count':value['rows'][-2]['calls']=576
    elif fault=='math-count':value['rows'][-1]['calls']=191
    elif fault=='math-kernel':value['rows'][-1]['kernels']={}
    elif fault=='warmup':value['warmups']=1
    else:value['performance_qualified']=True
    with pytest.raises(ValueError):records.validate_backend(value,'candidate')


def test_profiler_summary_reads_original_native_ancestry_and_kernels():
    parent=SimpleNamespace(name='predictor-eot8/call/12/predictor.2',cpu_parent=None)
    e=SimpleNamespace(name='aten::_scaled_dot_product_efficient_attention',cpu_parent=parent,device_type='DeviceType.CPU',kernels=[SimpleNamespace(name='SIMULATED_fmha_cutlassF',duration=2.)])
    got=records.profile_summary(SimpleNamespace(events=lambda:[e]),'candidate')
    assert got['rows'][0]['owner']=='predictor.2' and got['rows'][0]['kernels']['SIMULATED_fmha_cutlassF']['duration_us']==2.
    # This aggregate has only one event and is deliberately NOT a passing trace.
    with pytest.raises(ValueError):records.validate_backend(got,'candidate')


def test_profiler_summary_rejects_math_outside_real_custom_backward():
    e=SimpleNamespace(name='aten::_scaled_dot_product_attention_math',cpu_parent=None,device_type='DeviceType.CPU',kernels=[])
    with pytest.raises(ValueError,match='outside actual candidate'):
        records.profile_summary(SimpleNamespace(events=lambda:[e]),'candidate')


def test_full_input_comparison_keeps_original_bounds_and_exact_native_values():
    reference={'input':np.array([1.],dtype=np.float32),'loss':np.array(1.,dtype=np.float32),'prediction-0':np.array([.5],dtype=np.float32)}
    actual={n:a.copy() for n,a in reference.items()};actual['input'][0]+=.001
    assert records.compare(actual,reference)['input']['passed']
    actual['input'][0]=1.01
    with pytest.raises(ValueError,match='numerical rejection'):records.compare(actual,reference)
    actual['input'][0]=1.;actual['prediction-0'][0]+=.000001
    with pytest.raises(ValueError,match='Original forward'):records.compare(actual,reference)


def test_scientific_capture_has_required_argument_scope_and_no_clean_derivative():
    tree=ast.parse((ROOT/'scripts/capture_predictor_pgd_target_v1.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='SimpleNamespace']
    assert len(calls)==1
    names={n.arg for n in calls[0].keywords}
    assert names=={'output','tiny','device','upstream','checkpoint','packet','proposal','control_sha256'}
    worker_source=(ROOT/'models/predictor_pgd_target_worker_v1.py').read_text()
    assert 'context(clean)' not in worker_source and 'prepare_backward(losses.sum()' not in worker_source
    assert 'with reference:reference.gradient()' in worker_source
    assert 'self.step==0' in worker_source


@pytest.mark.parametrize('outcome',['proof-refusal','worker-refusal','completion-refusal'])
def test_actual_supervisor_simulated_refusals_collect_in_original_session(private,monkeypatch,outcome):
    c=control(private);sha=hashlib.sha256(op.old.json_bytes(c)).hexdigest()
    sources={n:op.VIRTUAL[n] if n in op.VIRTUAL else (ROOT/n).read_bytes() for n in op.REMOTE_NAMES}
    monkeypatch.setattr(op,'readiness',lambda c:{'parent':str(private),'device':private.stat().st_dev,'inode':private.stat().st_ino})
    monkeypatch.setattr(op.base,'gpu_gate',lambda *a,**k:{'SIMULATED':True})
    monkeypatch.setattr(op.base.primitives,'validate_proof',lambda *a:None)  # No storage proof actually runs.
    # Keep the actual scanner, category accounting and framed collection. Replace
    # only the process runner with an explicit simulated result.
    roles=[]
    def supervise(command,folder,*,role,gate,**kwargs):
        roles.append(role)
        gate()
        op.old.write_json(folder/(role+'-SIMULATION.json'),{'SIMULATION':True,'no_process_model_GPU_or_proof_ran':True,'role':role})
        return {'exit_status':1 if role=='proof' and outcome=='proof-refusal' or role=='worker-0' and outcome=='worker-refusal' else 0}
    monkeypatch.setattr(op.base.primitives,'supervise',supervise)
    monkeypatch.setattr(op,'storage',lambda folder,identity,**kwargs:{'files':len(op.inventory(folder,running=True)[0]),'SIMULATED_identity':True})
    monkeypatch.setattr(op,'check_worker',lambda *a,**k:(_ for _ in ()).throw(ValueError('SIMULATED incomplete worker rejected')))
    archive=(ROOT/c['archive_path']).read_bytes();out=io.BytesIO()
    import time
    start=time.monotonic()
    status=op._execute(c,op.authorization_template(c,sha,'SIMULATION ONLY'),io.BytesIO(archive),out,sources,sha,start,start+2400,lambda *a:None)
    assert status==1 and roles==(['proof'] if outcome=='proof-refusal' else ['proof','worker-0'])
    response=private/'response.bin';response.write_bytes(out.getvalue())
    first=client.unpack(response,private/'saved',c,sha)
    assert first['kind']==op.KIND
    state=json.loads((private/'saved/session.json').read_bytes())
    assert not state['passed'] and not state['complete'] and state['tasks']==[]
    assert (private/'saved/proof-SIMULATION.json').is_file()
    assert all(stat.S_IMODE(p.stat().st_mode)==0o700 for p in [private/'saved',*filter(Path.is_dir,(private/'saved').rglob('*'))])


def test_armed_transport_simulation_consumes_once_and_never_retries(private,monkeypatch):
    c=control(private);sha=hashlib.sha256(op.old.json_bytes(c)).hexdigest()
    op.old.write_json(private/'CONTROL.json',c);op.old.write_json(private/'AUTHORIZATION.json',op.authorization_template(c,sha,'SIMULATED transport only'))
    monkeypatch.setattr(client,'check_local',lambda folder:(c,sha));monkeypatch.setattr(client,'payload',lambda c,**kw:b'SIMULATED no remote command')
    # An empty response mimics disconnect before scratch creation. It must keep
    # the consumed claim/session and refuse any second launch.
    class Input(io.BytesIO):
        def close(self):pass
    calls=[]
    class Process:
        def __init__(self,command,stdin,stdout):self.stdin=Input();self.returncode=1;calls.append(command)
        def poll(self):return self.returncode
    with pytest.raises(json.JSONDecodeError):client.run(private,popen=Process)
    assert len(calls)==1 and json.loads((private/'ATTEMPT.json').read_bytes())['consumed'] is True
    assert len(list(private.glob('attended-*')))==1
    with pytest.raises(FileExistsError,match='consumed'):client.run(private,popen=Process)
    assert len(calls)==1


def test_profile_array_file_caps_and_actual_nonfinite_values_preserved(private):
    state={'artifacts':{}}
    with pytest.raises(ValueError,match='nonfinite'):
        records.save_tensor(private,state,'input',torch.full((2,),float('nan')))
    assert (private/'input.npy').is_file() and 'input' in state['artifacts']
    with pytest.raises(ValueError,match='Fresh bounded'):
        records.save_tensor(private,state,'input',torch.zeros(2))


def test_capture_script_passes_proposal_before_worker_and_preserves_alarm_order():
    source=(ROOT/'scripts/capture_predictor_pgd_target_v1.py').read_text()
    assert source.index('op.validate_authorization')<source.index('from models.predictor_pgd_target_worker_v1 import run')
    assert source.index('signal.setitimer')<source.index('from models.predictor_pgd_target_worker_v1 import run')
    assert 'proposal=overlay/op.CONTRACT' in source
    assert 'remaining<=1920' in source


@pytest.mark.parametrize('fault',['allocator','rng','restoration','model','baseline','count','wall'])
def test_saved_worker_header_rejects_missing_resource_or_restoration_evidence(fault):
    base={'SIMULATION':True};sha=hashlib.sha256(records.frozen.atomic._json(base)).hexdigest()
    s={'kind':records.KIND,'algorithm':records.ALGORITHM,'control_sha256':'1'*64,'execution_host':'linux10603',
       'complete':True,'passed':True,'research_admitted':False,'performance_qualified':False,'gradient_objectives_started':8,
       'candidate_points':4,'wall_seconds':100.,'model_unchanged':True,'global_rng_unchanged':True,
       'all_methods_hooks_flags_restored':True,'peak_allocator_bytes':5*2**30,'baseline':base,'baseline_sha256':sha}
    assert records.worker_header(s,'1'*64)
    if fault=='allocator':s['peak_allocator_bytes']=7*2**30+1
    elif fault=='rng':s['global_rng_unchanged']=False
    elif fault=='restoration':s['all_methods_hooks_flags_restored']=False
    elif fault=='model':s['model_unchanged']=False
    elif fault=='baseline':s['baseline_sha256']='2'*64
    elif fault=='count':s['gradient_objectives_started']=9
    else:s['wall_seconds']=1921.
    with pytest.raises(ValueError):records.worker_header(s,'1'*64)
