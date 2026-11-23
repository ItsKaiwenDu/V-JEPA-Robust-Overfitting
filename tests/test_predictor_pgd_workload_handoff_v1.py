"""Exact primary package/CLI/preflight checks; opaque bytes and FAKE grants only."""
import base64
import builtins
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_workload_handoff_v1 as handoff
from models import predictor_pgd_workload_handoff_client_v1 as client
from models import predictor_pgd_workload_handoff_outer_v1 as outer

ROOT=Path(__file__).resolve().parents[1]
ORIGIN='EXPLICIT-FAKE-WHOLE-PGD-HANDOFF-PREFLIGHT; NO USER GRANT OR EXECUTION'
EVIDENCE=Path(os.environ['VJEPA_WORKLOAD_HANDOFF_EVIDENCE']).resolve()


@pytest.fixture(scope='module',autouse=True)
def forbid_execution():
    patch=pytest.MonkeyPatch()
    def forbidden(*a,**k): pytest.fail('No process, timer, proof, GPU, science or network API allowed.')
    for obj,name in [(subprocess,'Popen'),(subprocess,'check_output'),(os,'system'),(os,'fork'),
                     (os,'posix_spawn'),(os,'kill'),(os,'killpg'),(os,'pidfd_open'),(os,'pipe2'),
                     (os,'posix_fallocate'),(signal,'alarm'),(signal,'signal'),(signal,'setitimer'),
                     (signal,'pidfd_send_signal'),(socket,'socket'),(threading.Thread,'start')]:
        if hasattr(obj,name): patch.setattr(obj,name,forbidden)
    original=builtins.__import__
    def load(name,*a,**k):
        if name.split('.')[0] in ('torch','numpy'): forbidden()
        return original(name,*a,**k)
    patch.setattr(builtins,'__import__',load)
    yield
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    patch.undo()


def put(path,data):
    missing=[]; parent=path.parent
    while not parent.exists(): missing.append(parent); parent=parent.parent
    for directory in reversed(missing): directory.mkdir(mode=0o700)
    data=data if type(data) is bytes else handoff.enc(data)
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as out: out.write(data)


@pytest.fixture
def area(request):
    path=EVIDENCE/request.node.name
    path.mkdir(mode=0o700)
    return path


def test_three_local_source_edits_leave_all_execution_bytes_exact():
    generated,d,c,f=handoff.assemble(ROOT)
    for name,row in d['edits'].items():
        raw=generated[name]; start=row['start']; after=row['after'].encode()
        assert raw[start:start+len(after)]==after
        original=raw[:start]+row['before'].encode()+raw[start+len(after):]
        assert handoff.digest(original)==row['original_sha256']
        alias={handoff.OUTER:'outer',handoff.CLIENT:'client',handoff.CLI:'cli_scaffold'}[name]
        assert original==f[c['pins'][alias]] and raw==(ROOT/name).read_bytes()
    assert len(outer.LOCAL_SOURCES)==13 and handoff.CLI in outer.LOCAL_SOURCES
    assert not d['uploaded_sources_changed'] and not d['supervised_control_changed']


@pytest.fixture(scope='module')
def prepared():
    parent=EVIDENCE/'unarmed-base'; parent.mkdir(mode=0o700); folder=parent/'unarmed'
    result=handoff.prepare_unarmed(ROOT,folder)
    return folder,result


def test_exact_unarmed_package_and_readonly_review(prepared):
    folder,result=prepared
    assert {p.name for p in folder.iterdir()}==handoff.FILES and result['unarmed_bytes_verified']
    assert not result['armed'] and result['roles']==['proof','original-unit','candidate-unit','candidate-replay']
    before={p.name:(p.lstat().st_ino,p.lstat().st_mtime_ns,p.read_bytes()) for p in folder.iterdir()}
    assert handoff.review_unarmed(ROOT,folder)==result
    assert before=={p.name:(p.lstat().st_ino,p.lstat().st_mtime_ns,p.read_bytes()) for p in folder.iterdir()}
    c=json.loads((folder/'CONTROL-WORKLOAD.json').read_bytes()); payload=json.loads((folder/'SOURCE-PAYLOAD.json').read_bytes())
    assert len(c['local_source_sha256'])==13 and len(payload)==55
    assert result['source_bytes']==895657 and result['packet']['bytes']==17192095
    assert result['full_model_input_gradient_objectives_maximum']==120
    assert result['candidate_inner_math_VJPs_maximum']==15360 and not result['research_calibration_selected']
    assert c['supervised_control_sha256']=='bc53eda6e44e8cfae82bf58e2b567461ede9c5a395d742d00f20944d6005df1c'
    assert all(stat.S_IMODE(p.lstat().st_mode)==0o600 and p.lstat().st_nlink==1 for p in folder.iterdir())


@pytest.mark.parametrize('fault',['reuse','parent-mode','extra-file','copied-path','file-mode'])
def test_unarmed_package_guards_preserve_prior_bytes(fault,prepared,area):
    original,result=prepared; folder=area/'unarmed'
    if fault=='reuse':
        with pytest.raises(ValueError): handoff.prepare_unarmed(ROOT,original)
        return
    if fault=='parent-mode':
        area.chmod(0o755)
        with pytest.raises(ValueError): handoff.prepare_unarmed(ROOT,folder)
        assert not folder.exists(); return
    shutil.copytree(original,folder)
    if fault=='extra-file': put(folder/'EXTRA.json',{'origin':ORIGIN})
    elif fault=='file-mode': (folder/'PACKAGE.json').chmod(0o644)
    with pytest.raises(ValueError): handoff.review_unarmed(ROOT,folder)


def test_unarmed_client_stops_before_alarm_or_claim(prepared,monkeypatch):
    folder,result=prepared; events=[]
    monkeypatch.setattr(client,'time',NS(monotonic=lambda:events.append('fake-clock') or 100.))
    with pytest.raises(ValueError,match='Unarmed'): client.run(folder)
    with pytest.raises(ValueError,match='Unarmed'): client.check_local(folder)
    assert events==['fake-clock'] and {p.name for p in folder.iterdir()}==handoff.FILES


@pytest.mark.parametrize('fake_armed',[False,True])
def test_isolated_cli_requires_authorization_before_project_import(fake_armed,area):
    folder=area/'fake'; folder.mkdir(mode=0o700)
    if fake_armed: put(folder/'AUTHORIZATION.json',{'origin':ORIGIN,'actual_user_approval':False})
    events=[]; fake_sys=NS(path=[],exit=lambda result:events.append(('exit',result)))
    original=builtins.__import__
    def load(name,globals=None,locals=None,fromlist=(),level=0):
        if name=='sys': return fake_sys
        if name=='models.predictor_pgd_workload_handoff_client_v1':
            events.append(('project-import',name)); assert fake_sys.path==[str(ROOT)]
            return NS(run=lambda p:events.append(('fake-run',str(p))) or 0)
        assert not name.startswith(('models','torch','numpy'))
        return original(name,globals,locals,fromlist,level)
    ns={'__name__':'_UNINVOKED_FAKE_ISOLATED_CLI','__file__':str(ROOT/handoff.CLI),
        '__builtins__':{**vars(builtins),'__import__':load}}
    exec(compile((ROOT/handoff.CLI).read_bytes(),str(ROOT/handoff.CLI),'exec'),ns)
    if fake_armed:
        assert ns['main']([str(folder)])==0
        assert events[-1]==('fake-run',str(folder))
    else:
        with pytest.raises(SystemExit,match='Unarmed'): ns['main']([str(folder)])
        assert events==[] and fake_sys.path==[]


@pytest.fixture(scope='module')
def fake_preflight_baseline(prepared):
    original,result=prepared; parent=EVIDENCE/'fake-preflight-baseline'; parent.mkdir(mode=0o700)
    project=parent/'project'; project.mkdir(mode=0o700)
    c=json.loads((original/'CONTROL-WORKLOAD.json').read_bytes())
    for name in [*outer.LOCAL_SOURCES,c['packet_path']]: put(project/name,(ROOT/name).read_bytes())
    shutil.copytree(original,parent/'package')
    put(parent/'FAKE-ORIGIN.json',{'origin':ORIGIN,'actual_user_authorization':False,'actual_target_work':False})
    return parent


def preflight_case(baseline,area):
    base=area/'case'; shutil.copytree(baseline,base); project=base/'project'; folder=base/'package'
    c=json.loads((folder/'CONTROL-WORKLOAD.json').read_bytes()); c.update(repository=str(project),local_handoff=str(folder))
    raw=handoff.enc(c); (folder/'CONTROL-WORKLOAD.json').write_bytes(raw)
    put(folder/'AUTHORIZATION.json',outer.authorization(c,handoff.digest(raw),ORIGIN))
    return base,project,folder,c


def test_real_local_preflight_reads_one_original_archive_with_fake_grant(fake_preflight_baseline,area):
    base,project,folder,c=preflight_case(fake_preflight_baseline,area)
    checked=client.check_local(folder); assert checked[0]==c
    assert checked[-1]==(project/c['packet_path'],)
    assert checked[6]['user_basis']==ORIGIN and checked[6]['extra_component_VJPs']==0
    assert checked[6]['full_model_gradients']==120
    assert not (folder/'ATTEMPT.json').exists() and not (folder/'attended-session').exists()
    put(base/'PREFLIGHT.json',{'origin':ORIGIN,'local_preflight_returned':True,'actual_authorization':False,
        'source_pins':len(c['local_source_sha256']),'archives':[str(p.relative_to(base)) for p in checked[-1]],
        'actual_process_timer_GPU_proof_network_or_scientific_work':False})


@pytest.mark.parametrize('fault',['CLI','packet','host-key','payload','old-grant','copy-path','method','budget'])
def test_preflight_refuses_drift_without_consuming_attempt(fault,fake_preflight_baseline,area):
    base,project,folder,c=preflight_case(fake_preflight_baseline,area)
    if fault=='CLI': (project/handoff.CLI).write_bytes(b'EXPLICIT-FAKE source drift\n')
    elif fault=='packet': (project/c['packet_path']).write_bytes(b'EXPLICIT-FAKE packet drift')
    elif fault=='host-key': (folder/'known-hosts').write_bytes(b'EXPLICIT-FAKE key drift')
    elif fault=='payload':
        payload=json.loads((folder/'SOURCE-PAYLOAD.json').read_bytes())
        name='overlay/models/_pgd_workload_entry_v1.py'; assert name in payload
        payload[name]=base64.b64encode(b'EXPLICIT-FAKE source drift').decode()
        (folder/'SOURCE-PAYLOAD.json').write_bytes(handoff.enc(payload))
    elif fault=='old-grant':
        a=json.loads((folder/'AUTHORIZATION.json').read_bytes()); a['kind']='pgd-candidate-outer-university-exception-v1'
        (folder/'AUTHORIZATION.json').write_bytes(handoff.enc(a))
    elif fault=='copy-path':
        copied=base/'copied'; shutil.copytree(folder,copied); folder=copied
    elif fault=='method':
        b=json.loads(c['supervised_control_text']); p=json.loads(b['policy_text']); p['scope']['method']='convergence'
        b['policy_text']=handoff.enc(p).decode(); b['policy_sha256']=handoff.digest(b['policy_text'].encode())
        c['supervised_control_text']=handoff.enc(b).decode(); c['supervised_control_sha256']=handoff.digest(c['supervised_control_text'].encode())
        raw=handoff.enc(c); (folder/'CONTROL-WORKLOAD.json').write_bytes(raw)
        (folder/'AUTHORIZATION.json').write_bytes(handoff.enc(outer.authorization(c,handoff.digest(raw),ORIGIN)))
    else:
        a=json.loads((folder/'AUTHORIZATION.json').read_bytes()); a['full_model_gradients']=121
        (folder/'AUTHORIZATION.json').write_bytes(handoff.enc(a))
    with pytest.raises((ValueError,ImportError)): client.check_local(folder)
    assert not (folder/'ATTEMPT.json').exists() and not (folder/'attended-session').exists()
    put(base/'REFUSAL.json',{'origin':ORIGIN,'fault':fault,'attempt_or_target_work':False})


def test_requested_authorization_is_only_primary_three_units_and_retains_caps(prepared):
    folder,result=prepared; raw=(folder/'CONTROL-WORKLOAD.json').read_bytes(); c=json.loads(raw)
    a=outer.authorization(c,handoff.digest(raw),ORIGIN); b=json.loads(c['supervised_control_text'])
    policy=json.loads(b['policy_text']); scope=policy['scope']; cat=json.loads(policy['scratch_contract_text'])
    operator=json.loads((ROOT/'configs/predictor_pgd_workload_operator_v1.json').read_bytes())
    assert a['maximum_new_attempts']==1 and a['automatic_retries']==0 and a['extra_component_VJPs']==0
    assert a['full_model_gradients']==120 and scope['method']=='primary' and scope['step_size_255']==1 and scope['sensitivity_banks']==0
    assert result['steps']==20 and result['restarts']==2 and result['input_gradients_per_unit']==40
    assert result['native_component_or_closed_cohort_reruns']==0 and not result['new_FP64_reference_or_full_model_trace']
    assert (a['work_seconds'],a['collection_seconds'],a['connection_including_login_seconds'])==(2400,600,3000)
    clocks=operator['clocks_seconds']
    assert (clocks['setup'],clocks['proof'],clocks['unit_maximum'],clocks['objective_maximum'])==(180,60,720,120)
    assert (clocks['outer_work_soft_deadline'],clocks['work_hard_deadline'],clocks['finalization_reserve'])==(2280,2400,120)
    assert operator['shared_storage_RSS_allocator_GPU_source_guards_unchanged']
    assert cat['caps']['metadata']==64*2**20 and cat['caps']['raw']==1792*2**20 and cat['caps']['trace']==8*2**20
    assert not a['paid_compute'] and not a['reset_old_claims'] and not a['private_key_upload'] and not a['held_out_pixels']
