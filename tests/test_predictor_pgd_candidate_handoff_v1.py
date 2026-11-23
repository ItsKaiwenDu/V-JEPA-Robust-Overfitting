"""New handoff/CLI/preflight checks. Opaque bytes and fake grants only; no work."""
import ast
import base64
import builtins
import copy
import hashlib
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
from models import predictor_pgd_candidate_handoff_v1 as handoff
from models import predictor_pgd_candidate_handoff_client_v1 as client
from models import predictor_pgd_candidate_handoff_outer_v1 as outer

ROOT=Path(__file__).resolve().parents[1]
ORIGIN='EXPLICIT-FAKE-HANDOFF-PREFLIGHT; NO USER GRANT OR EXECUTION'


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


def test_three_local_source_edits_leave_all_execution_bytes_exact():
    generated,d,c,f=handoff.assemble(ROOT)
    for name,row in d['edits'].items():
        raw=generated[name]; start=row['start']; after=row['after'].encode()
        assert raw[start:start+len(after)]==after
        original=raw[:start]+row['before'].encode()+raw[start+len(after):]
        assert handoff.digest(original)==row['original_sha256']
        alias={handoff.OUTER:'outer',handoff.CLIENT:'client',handoff.CLI:'cli_scaffold'}[name]
        assert original==f[c['pins'][alias]] and raw==(ROOT/name).read_bytes()
    assert len(outer.LOCAL_SOURCES)==12 and handoff.CLI in outer.LOCAL_SOURCES
    assert not d['uploaded_sources_changed'] and not d['supervised_control_changed']


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    parent=tmp_path_factory.mktemp('exact-unarmed-packet'); parent.chmod(0o700); folder=parent/'unarmed'
    result=handoff.prepare_unarmed(ROOT,folder)
    return folder,result


def test_exact_unarmed_package_and_readonly_review(prepared):
    folder,result=prepared
    assert {p.name for p in folder.iterdir()}==handoff.FILES and result['unarmed_bytes_verified']
    assert not result['armed'] and result['roles']==['proof','forward-preserving']
    before={p.name:(p.lstat().st_ino,p.lstat().st_mtime_ns,p.read_bytes()) for p in folder.iterdir()}
    assert handoff.review_unarmed(ROOT,folder)==result
    assert before=={p.name:(p.lstat().st_ino,p.lstat().st_mtime_ns,p.read_bytes()) for p in folder.iterdir()}
    c=json.loads((folder/'CONTROL-CANDIDATE.json').read_bytes()); payload=json.loads((folder/'SOURCE-PAYLOAD.json').read_bytes())
    assert len(c['local_source_sha256'])==12 and len(payload)==45
    assert result['source_bytes']==754567 and result['capsule']['bytes']==27279360 and result['native_reference']['bytes']==13281280
    assert c['supervised_control_sha256']=='0be5267678bbd45a63649b1dc39429ee9a2543e6cf305e1355260b7a0925f99c'
    assert c['reference_manifest_sha256']=='3dc2484c07c04bb72593c2ee386824edf52b8d80ec0fe3bb803e645714e5d932'
    assert all(stat.S_IMODE(p.lstat().st_mode)==0o600 and p.lstat().st_nlink==1 for p in folder.iterdir())


@pytest.mark.parametrize('fault',['reuse','parent-mode','extra-file','copied-path','file-mode'])
def test_unarmed_package_guards_preserve_prior_bytes(fault,prepared,tmp_path):
    original,result=prepared; tmp_path.chmod(0o700); folder=tmp_path/'unarmed'
    if fault=='reuse':
        with pytest.raises(ValueError): handoff.prepare_unarmed(ROOT,original)
        return
    if fault=='parent-mode':
        tmp_path.chmod(0o755)
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
def test_isolated_cli_requires_authorization_before_project_import(fake_armed,tmp_path):
    folder=tmp_path/'fake'; folder.mkdir(mode=0o700)
    if fake_armed: put(folder/'AUTHORIZATION.json',{'origin':ORIGIN,'actual_user_approval':False})
    events=[]; fake_sys=NS(path=[],exit=lambda result:events.append(('exit',result)))
    original=builtins.__import__
    def load(name,globals=None,locals=None,fromlist=(),level=0):
        if name=='sys': return fake_sys
        if name=='models.predictor_pgd_candidate_handoff_client_v1':
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
def fake_preflight_baseline(prepared,tmp_path_factory):
    original,result=prepared; parent=tmp_path_factory.mktemp('fake-preflight-baseline'); parent.chmod(0o700)
    project=parent/'project'; project.mkdir(mode=0o700)
    c=json.loads((original/'CONTROL-CANDIDATE.json').read_bytes())
    for name in [*outer.LOCAL_SOURCES,c['packet_path'],c['reference_path']]: put(project/name,(ROOT/name).read_bytes())
    shutil.copytree(original,parent/'package')
    put(parent/'FAKE-ORIGIN.json',{'origin':ORIGIN,'actual_user_authorization':False,'actual_target_work':False})
    return parent


def preflight_case(baseline,tmp_path):
    base=tmp_path/'case'; shutil.copytree(baseline,base); project=base/'project'; folder=base/'package'
    c=json.loads((folder/'CONTROL-CANDIDATE.json').read_bytes()); c.update(repository=str(project),local_handoff=str(folder))
    raw=handoff.enc(c); (folder/'CONTROL-CANDIDATE.json').write_bytes(raw)
    put(folder/'AUTHORIZATION.json',outer.authorization(c,handoff.digest(raw),ORIGIN))
    return base,project,folder,c


def test_real_local_preflight_reads_both_archives_with_fake_grant(fake_preflight_baseline,tmp_path):
    base,project,folder,c=preflight_case(fake_preflight_baseline,tmp_path)
    checked=client.check_local(folder); assert checked[0]==c
    assert checked[-1]==(project/c['packet_path'],project/c['reference_path'])
    assert checked[6]['user_basis']==ORIGIN and checked[6]['extra_component_VJPs']==1
    assert not (folder/'ATTEMPT.json').exists() and not (folder/'attended-session').exists()
    put(base/'PREFLIGHT.json',{'origin':ORIGIN,'local_preflight_returned':True,'actual_authorization':False,
        'source_pins':len(c['local_source_sha256']),'archives':[str(p.relative_to(base)) for p in checked[-1]],
        'actual_process_timer_GPU_proof_network_or_scientific_work':False})


@pytest.mark.parametrize('fault',['CLI','reference','capsule','host-key','payload','old-grant','copy-path','reference-manifest'])
def test_preflight_refuses_drift_without_consuming_attempt(fault,fake_preflight_baseline,tmp_path):
    base,project,folder,c=preflight_case(fake_preflight_baseline,tmp_path)
    if fault=='CLI': (project/handoff.CLI).write_bytes(b'EXPLICIT-FAKE source drift\n')
    elif fault=='reference': (project/c['reference_path']).write_bytes(b'EXPLICIT-FAKE reference drift')
    elif fault=='capsule': (project/c['packet_path']).write_bytes(b'EXPLICIT-FAKE capsule drift')
    elif fault=='host-key': (folder/'known-hosts').write_bytes(b'EXPLICIT-FAKE key drift')
    elif fault=='payload':
        payload=json.loads((folder/'SOURCE-PAYLOAD.json').read_bytes())
        name='overlay/models/predictor_pgd_candidate_accuracy_v1.py'; payload[name]=base64.b64encode(b'EXPLICIT-FAKE source drift').decode()
        (folder/'SOURCE-PAYLOAD.json').write_bytes(handoff.enc(payload))
    elif fault=='old-grant':
        a=json.loads((folder/'AUTHORIZATION.json').read_bytes()); a['kind']='pgd-component-outer-university-exception-v1'
        (folder/'AUTHORIZATION.json').write_bytes(handoff.enc(a))
    elif fault=='copy-path':
        copied=base/'copied'; shutil.copytree(folder,copied); folder=copied
    else:
        c['reference_manifest_sha256']='0'*64; raw=handoff.enc(c); (folder/'CONTROL-CANDIDATE.json').write_bytes(raw)
        (folder/'AUTHORIZATION.json').write_bytes(handoff.enc(outer.authorization(c,handoff.digest(raw),ORIGIN)))
    with pytest.raises((ValueError,ImportError)): client.check_local(folder)
    assert not (folder/'ATTEMPT.json').exists() and not (folder/'attended-session').exists()
    put(base/'REFUSAL.json',{'origin':ORIGIN,'fault':fault,'attempt_or_target_work':False})


def test_requested_authorization_is_only_one_candidate_and_retains_caps(prepared):
    folder,result=prepared; raw=(folder/'CONTROL-CANDIDATE.json').read_bytes(); c=json.loads(raw)
    a=outer.authorization(c,handoff.digest(raw),ORIGIN); b=json.loads(c['supervised_control_text'])
    policy=json.loads(b['policy_text']); scope=policy['scope']; cat=json.loads(policy['scratch_contract_text'])
    assert a['maximum_new_attempts']==1 and a['automatic_retries']==0 and a['extra_component_VJPs']==1
    assert a['full_model_gradients']==0 and scope['components']==['forward-preserving'] and scope['native_component_reruns']==0
    assert (a['work_seconds'],a['collection_seconds'],a['connection_including_login_seconds'])==(2400,600,3000)
    assert (scope['proof_seconds'],scope['component_seconds'])==(60,120)
    assert (scope['setup_bytes'],scope['capture_bytes'],scope['response_bytes'])==(3*2**30,9*2**30//4,3*2**30)
    assert (scope['GPU_bytes'],scope['component_RSS_bytes'])==(7*2**30,4*2**30)
    assert cat['caps']['metadata']==64*2**20 and not a['paid_compute'] and not a['reset_old_claims']
