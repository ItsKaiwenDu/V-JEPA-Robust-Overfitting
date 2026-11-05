"""Real cold CPU analogues and fail-closed replay-1 provenance checks."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models import attention_replay1_contract_v1 as refs
from models import attention_replay1_runtime_v1 as runtime
from models import attention_replay1_operator_v1 as op
from models import attention_replay1_client_v1 as client
from models import attention_replay1_cohort_v1 as science
from models import original_attention_pair_operator_v1 as oldop
from models import original_attention_pair_client_v1 as oldclient
from models.attention_return_capture_v1 import load_return
from models.fixed_attention_packet import build
from models.oracle_boundary_observer_v1 import record,read_record,digest

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'runs/pretrained/diagnostic/original-attention-pair-v1-20261104/linux10603'
PROPOSAL=ROOT/'runs/pretrained/diagnostic/original-attention-replay1-proposal-20261104/verified-design'


def env():
    return {**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1',
        'PYTHONPATH':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}


def overlay(root,names=op.REMOTE_NAMES):
    for name in names:
        p=root/'overlay'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(op.OVERLAY_INIT if name=='models/__init__.py' else (ROOT/name).read_bytes())


def tiny_prepare(root,inputs):
    op.base.extract_packet(inputs/'build/vjepa-fixed-attention.tar.gz',json.loads((inputs/'spec.json').read_text()),root/'code')
    overlay(root);(root/'neutral').mkdir()


def launch(root,inputs):
    return subprocess.run([sys.executable,'-I','-B',str(root/'overlay/scripts/capture_attention_replay1_v1.py'),
        '--packet',str(root/'code'),'--scratch',str(root),'--upstream',str(ROOT/'external/vjepa2'),
        '--local-tiny-analogue','--reference-archive',str(inputs/'reference-subset.tar'),
        '--reference-spec',str(inputs/'reference-spec.json'),'--reference-manifest',str(inputs/'reference-manifest.json')],
        cwd=root/'neutral',env=env(),capture_output=True,text=True,timeout=125)


@pytest.fixture(scope='module')
def inputs(tmp_path_factory,record_testsuite_property):
    root=tmp_path_factory.mktemp('replay1-original-tiny');_,spec=build(ROOT,root/'build',tiny=True)
    (root/'spec.json').write_text(json.dumps(spec));baseline=root/'baseline';baseline.mkdir()
    op.base.extract_packet(root/'build/vjepa-fixed-attention.tar.gz',spec,baseline/'code');overlay(baseline,oldop.REMOTE_NAMES)
    done=subprocess.run([sys.executable,'-I','-B',str(baseline/'overlay/scripts/capture_original_attention_pair_v1.py'),
        '--packet',str(baseline/'code'),'--scratch',str(baseline),'--upstream',str(ROOT/'external/vjepa2'),
        '--local-tiny-rehearsal'],cwd=baseline,env=env(),capture_output=True,text=True,timeout=125)
    (baseline/'cold.stdout').write_text(done.stdout);(baseline/'cold.stderr').write_text(done.stderr)
    assert done.returncode==0,done.stdout+done.stderr
    files={}
    for name in refs.expected_files(True):
        p=baseline/('results/pair.json' if name=='provenance/local-pair.json' else 'results/'+name)
        files[name]={'source':str(p.relative_to(baseline)),'bytes':p.stat().st_size,'sha256':digest(p),'mtime_ns':p.stat().st_mtime_ns}
    manifest={'kind':'local-tiny-CPU-reference-subset-v1','files':files}
    spec=refs.create_archive(baseline,manifest,root/'reference-subset.tar',local=True)
    (root/'reference-manifest.json').write_bytes(refs.canonical(manifest));(root/'reference-spec.json').write_bytes(refs.canonical(spec))
    record_testsuite_property('genuine_CPU_reference_inputs',str(root));return root


@pytest.fixture(scope='module')
def positive(tmp_path_factory,inputs,record_testsuite_property):
    root=tmp_path_factory.mktemp('replay1-cold-positive');tiny_prepare(root,inputs);done=launch(root,inputs)
    (root/'cold.stdout').write_text(done.stdout);(root/'cold.stderr').write_text(done.stderr)
    record_testsuite_property('cold_replay1_positive',str(root))
    assert done.returncode==0,done.stdout+done.stderr+'\n'+str(root)+'\n'+'\n'.join(p.read_text() for p in root.glob('local-*.log'))
    return root


def test_real_two_new_workers_and_one_fresh_CPU_verifier(positive):
    state=read_record(positive/'results/replay1.json')
    assert state['run_finished'] and state['cross_attempt_component_cohort_checks_passed']
    assert state['imported_workers']==list(refs.imported(True))
    assert len(state['new_workers_completed'])==2 and state['verifier_completed']
    pids=[]
    for i in (0,1):
        folder=positive/'results/workers'/refs.task(i,True)['worker']
        receipt=runtime.worker_status(folder,local=True);cap=read_record(folder/'capture.json');w=read_record(folder/'worker.json')
        assert state['new_workers_completed'][i]==receipt and cap['outcome']=='original-worker-passed'
        assert w['device']=='cpu' and w['arm']=='math' and w['replay']==1
        assert cap['CPU_oracle_calls']==cap['CPU_oracle_returns']==1
        assert cap['attention_calls']==cap['SDPA_calls']==len(cap['attention_returns'])==6
        assert cap['post_worker_context']['CUDA_initialized'] is False
        assert [x['phase'] for x in cap['attention_returns']]==list(refs.PHASES)
        pids.append(cap['process_id'])
    verifier=read_record(positive/'results/cohort-verifier/verifier.json');pids.append(verifier['process_id'])
    assert len(set(pids))==3 and verifier['NumPy_oracle_calls']==verifier['NumPy_oracle_returns']==1
    assert verifier['return_saved_before_gate'] and not verifier['CUDA_initialized_before'] and not verifier['CUDA_initialized_after']
    assert runtime.verifier_status(positive,local=True)==state['verifier']
    assert all(state[k] is v for k,v in runtime.FLAGS.items())
    assert not (positive/'results/session.json').exists()


def test_saved_FP32_returns_and_NumPy_FP64_return_keep_actual_dtype(positive):
    for i in (0,1):
        folder=positive/'results/workers'/refs.task(i,True)['worker'];cap=read_record(folder/'capture.json')
        values=torch.load(folder/'values.pt',map_location='cpu',weights_only=True);shape=list(values['q'].shape)
        cpu=load_return(folder,cap['CPU_return'],dtype='float64',shape=shape,limit=16*2**20)
        assert all(a.dtype==np.float64 for a in cpu.values())
        for entry in cap['attention_returns']:
            actual=load_return(folder,entry['artifact'],dtype='float32',shape=shape,limit=8*2**20)
            for n,a in actual.items():assert a.dtype==np.float32 and a.tobytes()==values['actual_'+n].numpy().tobytes()
    folder=positive/'results/cohort-verifier';v=read_record(folder/'verifier.json')
    returns=load_return(folder,v['NumPy_return'],dtype='float64',shape=shape,limit=16*2**20)
    assert all(a.dtype==np.float64 for a in returns.values())


def test_imports_stay_read_only_and_old_grants_are_only_provenance(positive,inputs):
    manifest=json.loads((inputs/'reference-manifest.json').read_text())
    receipt=refs.verify_imports(positive/'results',manifest,local=True)
    assert receipt['old_grants_renewed'] is False and receipt['new_workers_executed_by_import']==0
    assert not receipt['old_NPZ_files_present'] and receipt['files_verified']==13
    for name,item in manifest['files'].items():
        p=positive/'results'/name
        assert p.stat().st_mtime_ns==item['mtime_ns'] and stat.S_IMODE(p.stat().st_mode)==0o400
        assert p.stat().st_nlink==1 and digest(p)==item['sha256']


def test_actual_new_reference_is_first_new_worker_not_old_pair(positive):
    for i in (0,1):
        folder=positive/'results/workers'/refs.task(i,True)['worker'];a=read_record(folder/'authorization.json')
        assert a['reference']['worker']==(refs.imported(True)[0] if i==0 else refs.task(0,True)['worker'])
        assert a['reference']['values_sha256']==digest(folder.parent/a['reference']['worker']/'values.pt')


def test_independent_saved_review_can_reread_all_old_captures(positive,inputs):
    folder=positive/'results/cohort-verifier';v=read_record(folder/'verifier.json')
    values=load_return(folder,v['NumPy_return'],dtype='float64',shape=v['NumPy_return']['arrays']['output']['shape'],limit=16*2**20)
    old=[inputs/'baseline/results/workers'/name for name in refs.imported(True)]
    result=science.audit_cohort(positive/'results',positive/'code',local=True,numpy_values=values,full_old_roots=old)
    assert result['old_captures_directly_reread'] and not result['old_subset_attestation_only']
    assert result['cross_attempt_component_cohort_checks_passed'] and not result['original_parent_session_qualified']


def live_control():
    c=json.loads((OLD/'CONTROL.json').read_text())
    c.update(kind=op.KIND,role='encoder-1-replay1-and-CPU-cohort-verifier',policy=op.POLICY,prerequisite=op.PREREQUISITE,
        tasks=[refs.task(i,False) for i in (0,1)],original_parent_session_qualified=False,CPU_only_verifier_allowed=True)
    c['reference_manifest']=json.loads((PROPOSAL/'reference-subset-manifest.json').read_text())
    c['reference_archive_path']='runs/pretrained/diagnostic/attention-replay1-v1-20261105/packet/reference-subset.tar'
    c['reference_spec']=json.loads((ROOT/Path(c['reference_archive_path']).with_name('reference-spec.json')).read_text())
    c['prior_bindings_sha256']=json.loads((PROPOSAL/'sealed-inputs.json').read_text())
    c['prior_bindings_sha256'].update({n:digest(ROOT/n) for n in (
        'configs/original_attention_replay1_proposal_v1.json','report/checks/2026-11-04-original-attention-pair-v1-target-review.json',
        'runs/pretrained/diagnostic/original-attention-pair-v1-20261104/linux10603/ATTEMPT.json')})
    c['local_source_sha256']={n:digest(ROOT/n) for n in op.LOCAL_NAMES}
    c['remote_source_sha256']={n:hashlib.sha256(op.OVERLAY_INIT if n=='models/__init__.py' else (ROOT/n).read_bytes()).hexdigest() for n in op.REMOTE_NAMES}
    return c


def authorization(sha):
    return {'kind':op.AUTH_KIND,'approved':True,'control_sha256':sha,'host':'linux10603','maximum_new_attempts':1,
        'max_seconds':600,'worker_seconds':120,'max_fresh_workers':2,'max_CPU_only_verifiers':1,'CPU_verifier_seconds':90,
        'max_CPU_oracle_calls':3,'max_Torch_CPU_oracle_calls':2,'max_NumPy_CPU_oracle_calls':1,'max_attention_iterations':12,
        'GPU_input_transfer_allowed':True,'GPU_attention_allowed':True,'write_test_bytes':2684354560,
        'setup_ceiling':3*2**30,'capture_ceiling':9*2**30//4,'raw_capture_ceiling':400*2**20,'input_ceiling':256*2**20,
        'reference_archive_ceiling':64*2**20,'collection_seconds':600,'CPU_only_verifier_allowed':True,
        'paid_compute':False,'reset_old_claims':False,'automatic_retries':0,'user_basis':'LOCAL TEST ONLY; no university grant.'}


def client_base(tmp_path,armed=False):
    base=tmp_path/'client';base.mkdir();c=live_control();(base/'known-hosts').write_bytes((OLD/'known-hosts').read_bytes())
    c['known_hosts_sha256']=digest(base/'known-hosts');op.base.old.write_json(base/'CONTROL.json',c);sha=digest(base/'CONTROL.json')
    if armed:op.base.old.write_json(base/'AUTHORIZATION.json',authorization(sha))
    return base,c,sha


@pytest.mark.parametrize('key,value',[
    ('host','linux10609'),('kind',oldop.KIND),('tasks',[refs.task(1,False),refs.task(0,False)]),
    ('reuse_old_grant',True),('old_refusals_waived',True),('paid_compute',True),('research_admitted',True),
    ('original_parent_session_qualified',True),('CPU_only_verifier_allowed',False),('timings_usable_for_speed_or_budget',True)])
def test_domain_changes_refuse(key,value):
    c=live_control();c[key]=value
    with pytest.raises(ValueError):op.validate_control(c)


@pytest.mark.parametrize('key',['max_seconds','worker_seconds','CPU_verifier_seconds','max_input_bytes','max_raw_bytes',
    'max_CPU_verifiers','max_CPU_oracle_calls','max_Torch_CPU_oracle_calls','max_NumPy_CPU_oracle_calls',
    'max_attention_iterations','max_reference_archive_bytes','write_test_bytes','setup_ceiling','capture_ceiling','collection_seconds'])
def test_policy_cannot_expand(key):
    c=copy.deepcopy(live_control());c['policy'][key]+=1
    with pytest.raises(ValueError):op.validate_control(c)


@pytest.mark.parametrize('key',['max_seconds','worker_seconds','max_CPU_only_verifiers','CPU_verifier_seconds','max_CPU_oracle_calls',
    'max_Torch_CPU_oracle_calls','max_NumPy_CPU_oracle_calls','max_attention_iterations','raw_capture_ceiling','input_ceiling',
    'reference_archive_ceiling','maximum_new_attempts'])
def test_grant_cannot_expand(key):
    a=authorization('a'*64);a[key]+=1
    with pytest.raises(ValueError):op.validate_authorization(a,'a'*64)


def test_expired_pair_grant_does_not_authorize_replay1():
    a=json.loads((OLD/'AUTHORIZATION.json').read_text())
    with pytest.raises(ValueError):op.validate_authorization(a,a['control_sha256'])


def test_unarmed_refuses_before_claim_session_process_or_SSH(tmp_path,monkeypatch):
    base,c,sha=client_base(tmp_path)
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('Unarmed means no process.'))
    with pytest.raises(ValueError,match='UNARMED'):client.run_client(base,sha)
    assert sorted(p.name for p in base.iterdir())==['CONTROL.json','known-hosts']


def test_exact_payload_verifies_and_consumed_claim_refuses_without_connection(tmp_path,monkeypatch):
    base,c,sha=client_base(tmp_path,True)
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('No process in local verification.'))
    assert client.verify_local(base,sha)==c
    assert len(client.sources(c))==16 and len(c['prior_bindings_sha256'])>=2252
    op.base.old.write_json(base/'ATTEMPT.json',{'consumed':True})
    with pytest.raises(FileExistsError):client.run_client(base,sha)


@pytest.mark.parametrize('defect',['hash','mtime','mode','symlink','hardlink'])
def test_imported_reference_changes_refuse(tmp_path,positive,inputs,defect):
    shutil.copytree(positive/'results',tmp_path/'results');p=tmp_path/'results/workers'/refs.imported(True)[0]/'values.pt'
    manifest=json.loads((inputs/'reference-manifest.json').read_text())
    if defect=='hash':p.chmod(0o600);p.write_bytes(p.read_bytes()+b'bad')
    elif defect=='mtime':os.utime(p,ns=(1,1))
    elif defect=='mode':p.chmod(0o600)
    else:
        p.rename(tmp_path/'kept')
        if defect=='symlink':p.symlink_to(tmp_path/'kept')
        else:os.link(tmp_path/'kept',p)
    with pytest.raises(ValueError):refs.verify_imports(tmp_path/'results',manifest,local=True)


@pytest.mark.parametrize('defect',['hash','bytes','linked_member','extra_member','unsafe_member','duplicate','existing'])
def test_reference_extraction_is_exact_and_never_overwrites(tmp_path,inputs,defect):
    manifest=json.loads((inputs/'reference-manifest.json').read_text());spec=json.loads((inputs/'reference-spec.json').read_text())
    target=tmp_path/'results';target.mkdir();archive=inputs/'reference-subset.tar'
    if defect=='hash':spec['sha256']='0'*64
    elif defect=='bytes':spec['bytes']+=1
    elif defect=='existing':
        p=target/next(iter(manifest['files']));p.parent.mkdir(parents=True);p.write_bytes(b'preserved')
    else:
        path=tmp_path/'changed.tar'
        with tarfile.open(archive,'r:') as old,tarfile.open(path,'w',format=tarfile.USTAR_FORMAT) as new:
            for i,m in enumerate(old):
                data=old.extractfile(m).read()
                if i==0 and defect=='linked_member':m.type=tarfile.LNKTYPE;m.linkname='elsewhere';m.size=0
                if i==0 and defect=='unsafe_member':m.name='../outside'
                new.addfile(m,io.BytesIO(data) if m.isfile() else None)
                if i==0 and defect=='duplicate':new.addfile(m,io.BytesIO(data))
            if defect=='extra_member':m=tarfile.TarInfo('extra');m.size=1;new.addfile(m,io.BytesIO(b'x'))
        archive=path;spec.update(bytes=path.stat().st_size,sha256=digest(path))
    with pytest.raises((ValueError,FileExistsError)):refs.extract_archive(archive,spec,manifest,target,local=True)
    assert not (tmp_path/'outside').exists()
    if defect=='existing':assert p.read_bytes()==b'preserved'


@pytest.mark.parametrize('field',['kind','passed','complete','device','replay','common','strides','oracle','bookkeeping',
    'backend','replays','timings','median','wall','memory','returns','hook'])
def test_saved_scientific_refusals_are_preserved(tmp_path,positive,field):
    folder=tmp_path/refs.task(0,True)['worker'];shutil.copytree(positive/'results/workers'/folder.name,folder)
    w=read_record(folder/'worker.json');cap=read_record(folder/'capture.json')
    if field=='kind':w['kind']='wrong'
    elif field in ('passed','complete'):w[field]=False
    elif field=='device':w['device']='cuda'
    elif field=='replay':w['replay']=0
    elif field=='common':w['common']['oracle_query_chunk']=64
    elif field=='strides':w['device_input_strides']['v'][-1]+=1
    elif field=='oracle':w['oracle_checks']['dq']['passed']=False
    elif field=='bookkeeping':w['oracle_bookkeeping']['dv']['atol']=1e-10
    elif field=='backend':w['backend_evidence']['passed']=False
    elif field=='replays':w['replay_signatures'].pop()
    elif field=='timings':w['timings'][0]['total_seconds']=-1
    elif field=='median':w['median_seconds']['total_seconds']+=1
    elif field=='wall':w['wall_seconds']=31
    elif field=='memory':w['peak_cuda_allocated_bytes']=123
    elif field=='returns':cap['attention_returns'].pop()
    elif field=='hook':cap['profile_restored']=False
    record(folder/'worker.json',w,replace=True);cap['original_worker_sha256']=digest(folder/'worker.json');record(folder/'capture.json',cap,replace=True)
    with pytest.raises((ValueError,KeyError)):science.saved_worker(folder,positive/'code',local=True,expected=refs.task(0,True),preliminary=True)


@pytest.mark.parametrize('dtype',[np.float32,np.int64])
def test_NumPy_writer_refuses_casts(tmp_path,dtype):
    values={n:np.ones((1,2,3,4),dtype=dtype) for n in science.OUTPUTS}
    with pytest.raises(ValueError,match='never silently cast'):science.save_numpy_return(tmp_path/'r.npz',values)
    assert not (tmp_path/'r.npz').exists()


def test_NumPy_writer_keeps_nonfinite_and_original_return_values(tmp_path):
    values={n:np.ones((1,2,3,4),dtype=np.float64).transpose(1,2,0,3) for n in science.OUTPUTS};values['dq'][0,0,0,0]=np.nan
    item=science.save_numpy_return(tmp_path/'r.npz',values)
    saved=load_return(tmp_path,item,dtype='float64',shape=list(values['output'].shape),limit=16*2**20)
    assert not item['arrays']['dq']['finite'] and np.isnan(saved['dq'][0,0,0,0])
    assert all(a.tobytes(order='C')==saved[n].tobytes() for n,a in values.items())
    with pytest.raises(FileExistsError):science.save_numpy_return(tmp_path/'r.npz',values)


def numpy_refusal_harness(tmp_path,positive,body):
    shutil.copytree(positive/'results',tmp_path/'results');shutil.rmtree(tmp_path/'results/cohort-verifier')
    script=tmp_path/'refusal.py';script.write_text("""
import sys
from pathlib import Path
root=Path(sys.argv[1]);packet=Path(sys.argv[2]);sys.path.insert(0,str(packet.parent/'overlay'));sys.path.insert(1,str(packet))
from models import attention_replay1_cohort_v1 as science
from models.oracle_boundary_observer_v1 import read_record
"""+body)
    done=subprocess.run([sys.executable,'-I','-B',str(script),str(tmp_path),str(positive/'code')],
        env=env(),capture_output=True,text=True,timeout=30)
    assert done.returncode==0,done.stdout+done.stderr


def test_NumPy_actual_return_is_saved_before_original_reconstruction_refusal(tmp_path,positive):
    numpy_refusal_harness(tmp_path,positive,"""
original_audit=science.audit_cohort
def negative(*a,numpy_values,**kw):
    # The new reader's test input changes; original oracle and bounds remain bound and unchanged.
    changed={n:v.copy() for n,v in numpy_values.items()};changed['output']+=1e-3
    return original_audit(*a,numpy_values=changed,**kw)
science.audit_cohort=negative
assert science.verify_once(root/'results',packet,local=True)==1
v=read_record(root/'results/cohort-verifier/verifier.json')
assert v['return_saved_before_gate'] and v['NumPy_oracle_calls']==v['NumPy_oracle_returns']==1
assert v['error']=='Fixed FP64 reference failed independent NumPy reconstruction.'
assert (root/'results/cohort-verifier/NumPy-return.npz').exists() and not v['complete']
""")


def test_NumPy_capture_IO_error_refuses_and_restores_hooks(tmp_path,positive):
    numpy_refusal_harness(tmp_path,positive,"""
def failure(*a,**kw):raise OSError('LOCAL TEST: full capture disk')
science.save_numpy_return=failure
assert science.verify_once(root/'results',packet,local=True)==1
v=read_record(root/'results/cohort-verifier/verifier.json')
assert v['error_type']=='OSError' and not v['complete'] and not v['passed']
assert v['profile_restored'] and v['trace_restored'] and v['NumPy_oracle_returns']==1
""")


def worker_harness(tmp_path,inputs,body):
    tiny_prepare(tmp_path,inputs)
    script=tmp_path/'worker-harness.py';script.write_text('''
import sys,json,time,signal
from pathlib import Path
from types import SimpleNamespace
root=Path(sys.argv[1]);source=Path(sys.argv[2]);sys.path.insert(0,str(root/'overlay'));sys.path.insert(1,str(root/'code'))
from models import attention_replay1_runtime_v1 as runtime,attention_replay1_contract_v1 as refs
manifest=json.loads((source/'reference-manifest.json').read_text());spec=json.loads((source/'reference-spec.json').read_text())
runtime.initialize(root,root/'code',manifest,local=True,control_sha=None,started=time.time())
(root/'local-reference-manifest.json').write_bytes(refs.canonical(manifest))
refs.extract_archive(source/'reference-subset.tar',spec,manifest,root/'results',local=True)
worker,plan,path=runtime.issue(root,root/'code',manifest,'worker',local=True,control_sha=None,index=0)
args=SimpleNamespace(packet=root/'code',scratch=root,upstream=Path(sys.argv[3]),local_tiny_analogue=True,
    phase='worker',control_sha256=None,task_index=0,worker_plan_sha256=runtime.digest(path))
'''+body)
    done=subprocess.run([sys.executable,'-I','-B',str(script),str(tmp_path),str(inputs),str(ROOT/'external/vjepa2')],
        env=env(),capture_output=True,text=True,timeout=40)
    (tmp_path/'harness.stdout').write_text(done.stdout);(tmp_path/'harness.stderr').write_text(done.stderr)
    assert done.returncode==0,done.stdout+done.stderr


@pytest.mark.parametrize('hook',['profile','trace'])
def test_existing_observer_hook_is_not_replaced(tmp_path,inputs,hook):
    worker_harness(tmp_path,inputs,'''
existing=lambda *a:None
getattr(sys,'set'''+hook+'''')(existing)
try:runtime.dispatch(args,root/'overlay')
except ValueError:pass
else:raise AssertionError('An existing hook must refuse')
assert getattr(sys,'get'''+hook+'''')() is existing
getattr(sys,'set'''+hook+'''')(None)
assert not (worker/'CPU-return.npz').exists()
''')


@pytest.mark.parametrize('name',['iteration','worker'])
def test_changed_original_code_refuses_before_oracle(tmp_path,inputs,name):
    worker_harness(tmp_path,inputs,'''
from models import fixed_attention_diagnostic as original
def replacement(*a,**k):raise AssertionError('Replacement must never run')
original.'''+name+'''=replacement
try:runtime.dispatch(args,root/'overlay')
except ValueError:pass
else:raise AssertionError('Changed original source/function must refuse')
assert not (worker/'CPU-return.npz').exists()
''')


def test_actual_attention_capture_failure_stops_and_restores_remaining_alarm(tmp_path,inputs):
    worker_harness(tmp_path,inputs,'''
from models import attention_return_capture_v1 as capture
writer=capture.save_return
def failure(*a,**kw):
    if kw['dtype']=='float32':raise OSError('LOCAL TEST: FP32 capture failure')
    return writer(*a,**kw)
capture.save_return=failure
def old_alarm(*a):raise AssertionError('Test timeout')
signal.signal(signal.SIGALRM,old_alarm);signal.setitimer(signal.ITIMER_REAL,25);began=time.monotonic()
assert runtime.dispatch(args,root/'overlay')==1
remaining=signal.getitimer(signal.ITIMER_REAL)[0];signal.setitimer(signal.ITIMER_REAL,0)
assert signal.getsignal(signal.SIGALRM) is old_alarm and 0<remaining<=25-(time.monotonic()-began)+.05
cap=runtime.read_record(worker/'capture.json');w=runtime.read_record(worker/'worker.json')
assert cap['outcome']=='technical-failure' and cap['CPU_oracle_returns']==cap['attention_calls']==cap['SDPA_calls']==1
assert len(cap['attention_returns'])==0 and w['error_type']=='OSError' and not w['passed'] and not w['complete']
assert sys.getprofile() is None and sys.gettrace() is None
assert not (worker.parent/refs.task(1,True)['worker']).exists()
''')


def test_original_CPU_bookkeeping_refusal_keeps_actual_return_and_stops(tmp_path,inputs):
    worker_harness(tmp_path,inputs,'''
import numpy as np,torch
from models.cuda_profile import tensor_signature
packet=root/'code';p=packet/'fixtures/encoder-1/oracle_output.npy';value=np.load(p,allow_pickle=False)+.01
with p.open('wb') as f:np.save(f,value,allow_pickle=False)
fp=p.parent/'fixture.json';f=runtime.read_record(fp)
f['files']['oracle_output'].update(sha256=runtime.digest(p),bytes=p.stat().st_size,tensor=tensor_signature(torch.from_numpy(value)))
runtime.record(fp,f,replace=True);mp=packet/'FIXED-ATTENTION-HANDOFF.json';m=json.loads(mp.read_text())
m['fixtures']['encoder-1']['sha256']=runtime.digest(fp)
for changed in (p,fp):m['files'][str(changed.relative_to(packet))].update(sha256=runtime.digest(changed),bytes=changed.stat().st_size)
mp.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
# Rebind this explicitly local negative fixture; no live control or scientific function is changed.
a=runtime.read_record(worker/'authorization.json');a['handoff_sha256']=runtime.digest(mp);runtime.record(worker/'authorization.json',a,replace=True)
plan['original_authorization_sha256']=runtime.digest(worker/'authorization.json');runtime.record(path,plan,replace=True);args.worker_plan_sha256=runtime.digest(path)
assert runtime.dispatch(args,root/'overlay')==1
cap=runtime.read_record(worker/'capture.json');w=runtime.read_record(worker/'worker.json')
assert cap['CPU_oracle_returns']==1 and cap['attention_calls']==cap['SDPA_calls']==0 and cap['CPU_return']
assert w['error']=='Fixed oracle bookkeeping failed; preserve input snapshot.' and not w['passed']
assert not (worker/'device-inputs.pt').exists()
'''.replace("+'\n')","+'\\n')"))


@pytest.mark.parametrize('defect',['incomplete','return_hash','values_hash'])
def test_second_new_worker_requires_first_complete_passing_capture(tmp_path,positive,inputs,defect):
    shutil.copytree(positive/'results',tmp_path/'results');manifest=json.loads((inputs/'reference-manifest.json').read_text())
    state=read_record(tmp_path/'results/replay1.json');state.update(run_finished=False,verifier_completed=False,
        new_workers_completed=state['new_workers_completed'][:1],cross_attempt_component_cohort_checks_passed=False)
    record(tmp_path/'results/replay1.json',state,replace=True)
    first=tmp_path/'results/workers'/refs.task(0,True)['worker'];second=first.parent/refs.task(1,True)['worker'];shutil.rmtree(second)
    if defect=='incomplete':
        cap=read_record(first/'capture.json');cap['capture_coverage_complete']=False;record(first/'capture.json',cap,replace=True)
    else:
        p=first/('return-5-timed-3.npz' if defect=='return_hash' else 'values.pt');p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError):runtime.issue(tmp_path,positive/'code',manifest,'worker',local=True,control_sha=None,index=1)
    assert not second.exists()


def test_late_worker_or_verifier_cannot_reset_the_outer_clock(tmp_path,positive,inputs):
    shutil.copytree(positive/'results',tmp_path/'results');manifest=json.loads((inputs/'reference-manifest.json').read_text())
    state=read_record(tmp_path/'results/replay1.json');state.update(run_finished=False,verifier_completed=False,
        started_unix=time.time()-118,deadline_unix=time.time()+2)
    record(tmp_path/'results/replay1.json',state,replace=True);(tmp_path/'results/verifier-plan.json').unlink()
    with pytest.raises(TimeoutError):runtime.issue(tmp_path,positive/'code',manifest,'cohort',local=True,control_sha=None)
    assert not (tmp_path/'results/verifier-plan.json').exists()
    state['started_unix']-=121;state['deadline_unix']-=121;record(tmp_path/'results/replay1.json',state,replace=True)
    with pytest.raises(TimeoutError):runtime.finish(tmp_path,local=True)


def test_reference_shortcut_to_old_worker_is_refused(tmp_path,positive):
    from models import attention_replay1_cohort_v1 as science
    shutil.copytree(positive/'results/workers',tmp_path/'workers');folder=tmp_path/'workers'/refs.task(1,True)['worker']
    a=read_record(folder/'authorization.json');old=tmp_path/'workers'/refs.imported(True)[0]
    a['reference']={'worker':old.name,'worker_sha256':digest(old/'worker.json'),'values_sha256':digest(old/'values.pt')}
    record(folder/'authorization.json',a,replace=True)
    with pytest.raises(ValueError):science.reference_check(folder,tmp_path/'workers'/refs.task(0,True)['worker'],local=True)
