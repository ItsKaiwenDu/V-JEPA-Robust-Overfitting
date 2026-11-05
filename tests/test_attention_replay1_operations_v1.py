"""Offline scope, full-packet dispatch and transport tests for the new operator."""
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest

from test_attention_replay1_v1 import ROOT,OLD,PROPOSAL,env,overlay,live_control,authorization,client_base
from models import attention_replay1_contract_v1 as refs
from models import attention_replay1_operator_v1 as op
from models import attention_replay1_runtime_v1 as runtime
from models import attention_replay1_client_v1 as client
from models.oracle_boundary_observer_v1 import record,read_record,digest


@pytest.fixture(autouse=True)
def private_umask():
    previous=os.umask(0o077)
    yield
    os.umask(previous)


def wire(c,sha,*,defect=None):
    data={'operator-control.json':op.base.old.json_bytes(c),'operator-result.json':op.base.old.json_bytes({
        'kind':op.KIND,'host':c['host'],'control_sha256':sha,'work_finished':True,'GPU_attention_allowed':True,
        'cross_attempt_component_cohort_checks_passed':False,'new_workers_completed':0,'verifier_completed':False,
        **runtime.FLAGS,'scientific_review_complete':False,'error':'LOCAL TEST: proof refused before computation.'}),
        'results/retained-return.npz':b'LOCAL transport fixture; actual array interpretation belongs to independent review.',
        'results/provenance/read-only-reference.json':b'LOCAL read-only reference fixture.'}
    files={n:{'bytes':len(b),'mtime_ns':1234567890123456789,'sha256':hashlib.sha256(b).hexdigest(),
        'mode':0o400 if '/provenance/' in n else 0o600} for n,b in data.items()}
    if defect=='hash':files['results/retained-return.npz']['sha256']='0'*64
    elif defect=='bytes':files['results/retained-return.npz']['bytes']+=1
    elif defect=='unsafe':files['../outside']=files.pop('results/retained-return.npz')
    elif defect=='mode':files['results/retained-return.npz']['mode']=0o777
    out=io.BytesIO();out.write(op.base.old.json_bytes({'kind':op.KIND,'phase':'scratch-created','host':c['host'],
        'control_sha256':sha,'scratch_directory':'/var/tmp/'+op.PREFIX+'linux10603-local-transport-fixture'}))
    out.write(op.base.old.json_bytes({'kind':op.KIND,'phase':'collection','control_sha256':sha,'files':files,
        'logical_bytes':sum(x['bytes'] for x in files.values()),'scientific_review_complete':False,'model_restarted_for_collection':False}))
    with tarfile.open(mode='w',fileobj=out) as archive:
        for i,(name,raw) in enumerate(data.items()):
            m=tarfile.TarInfo(name);m.size=len(raw);m.mode=files.get(name,{}).get('mode',0o600)
            if defect=='link' and i==2:m.type=tarfile.LNKTYPE;m.linkname='operator-control.json';m.size=0
            archive.addfile(m,io.BytesIO(raw) if m.isfile() else None)
            if defect=='duplicate' and i==1:archive.addfile(m,io.BytesIO(raw))
    return out.getvalue()


def test_transport_preserves_NPZ_exact_mtime_and_read_only_mode(tmp_path):
    c=live_control();sha=hashlib.sha256(op.base.old.json_bytes(c)).hexdigest();p=tmp_path/'response.bin';p.write_bytes(wire(c,sha))
    checked=client.verify_response(p,tmp_path/'saved',c,sha)
    assert checked['transport_verified'] and checked['members_verified']==4
    assert not checked['scientific_review_complete'] and not checked['cross_attempt_component_cohort_checks_passed']
    assert (tmp_path/'saved/results/retained-return.npz').stat().st_mtime_ns==1234567890123456789
    assert (tmp_path/'saved/results/provenance/read-only-reference.json').stat().st_mode&0o777==0o400
    with pytest.raises(FileExistsError):client.verify_response(p,tmp_path/'saved',c,sha)


@pytest.mark.parametrize('defect',['hash','bytes','unsafe','mode','link','duplicate','truncated','relabel'])
def test_transport_defects_refuse_without_overwrite_or_new_model(tmp_path,defect):
    c=live_control();sha=hashlib.sha256(op.base.old.json_bytes(c)).hexdigest();data=wire(c,sha,defect=defect)
    if defect=='truncated':data=data[:data.index(b'operator-control.json')+200]
    elif defect=='relabel':data=data.replace(op.KIND.encode(),op.base.KIND.encode(),1)
    p=tmp_path/'response.bin';p.write_bytes(data)
    with pytest.raises((ValueError,tarfile.ReadError,EOFError)):client.verify_response(p,tmp_path/'saved',c,sha)
    assert p.exists() and not (tmp_path/'outside').exists()


def test_fake_connection_sends_both_bound_archives_once(tmp_path):
    base,c,sha=client_base(tmp_path,True);response=tmp_path/'wire.bin';response.write_bytes(wire(c,sha))
    script=tmp_path/'fake.py';script.write_text('''
import sys,json,hashlib
from pathlib import Path
bundle=json.loads(sys.stdin.buffer.readline());a=json.loads(sys.stdin.buffer.readline());c=bundle['control']
assert len(bundle['sources'])==16 and a['kind']=='attention-replay1-resource-exception-v1'
for key,n in [('archive_sha256',c['archive_bytes']),('reference_spec',c['reference_spec']['bytes'])]:
    data=sys.stdin.buffer.read(n);assert len(data)==n
    want=c[key] if key=='archive_sha256' else c[key]['sha256'];assert hashlib.sha256(data).hexdigest()==want
assert not sys.stdin.buffer.read()
sys.stdout.buffer.write(Path(sys.argv[1]).read_bytes())
''')
    checked=client.run_client(base,sha,ssh_override=[sys.executable,'-I','-B',str(script),str(response)])
    assert checked['transport_verified'] and not checked['cross_attempt_component_cohort_checks_passed']
    assert not list(base.rglob('upload-stream.bin')) and (base/'ATTEMPT.json').exists()
    with pytest.raises(FileExistsError):client.run_client(base,sha,ssh_override=['must-not-run'])


def test_full_stdin_bootstrap_imports_stdlib_only_and_keeps_password_connection(tmp_path):
    base,c,sha=client_base(tmp_path);command=client.command_for(base,c)
    assert command[0]=='ssh' and command[-2]=='kdu@linux10603.dc.engr.scu.edu'
    assert all(flag in command for flag in ('ForwardAgent=no','StrictHostKeyChecking=yes','ControlMaster=no','ClearAllForwardings=yes'))
    code=shlex.split(command[-1])[-1]
    harness='''
import sys
code=sys.argv[1];code=code[:code.rindex('sys.modules["models.attention_replay1_operator_v1"].remote_main')]
exec(code)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
from models import attention_replay1_operator_v1 as op
op.validate_control(control)
print('stdlib bootstrap only; no oracle/CUDA/SSH')
'''
    done=subprocess.run([sys.executable,'-I','-B','-c',harness,code],input=client.bootstrap_payload(c),capture_output=True,timeout=15)
    assert done.returncode==0,done.stderr.decode()
    assert len(command[-1])<8000 and len(client.bootstrap_payload(c))<2**20


def test_full_original_packet_cold_import_and_reference_import_before_blocked_dispatch(tmp_path,record_testsuite_property):
    c=live_control();overlay(tmp_path);op.base.extract_packet(ROOT/c['original_archive_path'],c['archive_manifest'],tmp_path/'code')
    op.base.old.write_json(tmp_path/'operator-control.json',c);sha=digest(tmp_path/'operator-control.json')
    op.base.old.write_json(tmp_path/'resource-authorization.json',authorization(sha))
    script=tmp_path/'blocked.py';script.write_text('''
import sys,time,json
from pathlib import Path
from types import SimpleNamespace
root=Path(sys.argv[1]);sha=sys.argv[2];sys.path.insert(0,str(root/'overlay'))
from models import attention_replay1_runtime_v1 as runtime,attention_replay1_contract_v1 as refs
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
c=json.loads((root/'operator-control.json').read_text());runtime.initialize(root,root/'code',c['reference_manifest'],local=False,control_sha=sha,started=time.time())
refs.extract_archive(Path(sys.argv[3]),c['reference_spec'],c['reference_manifest'],root/'results',local=False)
worker,plan,path=runtime.issue(root,root/'code',c['reference_manifest'],'worker',local=False,control_sha=sha,index=0)
from models import attention_replay1_observer_v1 as observer
class Halt(BaseException):pass
calls=[]
def blocked(original,args,packet,folder,**kw):
    import torch
    assert not torch.cuda.is_initialized() and args.arm=='efficient' and args.replay==1 and args.device=='cuda'
    assert original.worker.__name__=='worker';calls.append(1);raise Halt()
observer.observe_worker=blocked
args=SimpleNamespace(packet=root/'code',scratch=root,upstream=root/'no-upstream-needed',control_sha256=sha,
    phase='worker',task_index=0,local_tiny_analogue=False,worker_plan_sha256=runtime.digest(path))
try:runtime.dispatch(args,root/'overlay')
except Halt:pass
else:raise AssertionError('Dispatch must be blocked')
assert calls==[1] and not (worker/'worker.json').exists()
print('Full original packet/imported subset verified; computation blocked.')
''')
    done=subprocess.run([sys.executable,'-I','-B',str(script),str(tmp_path),sha,str(ROOT/c['reference_archive_path'])],
        env=env(),capture_output=True,text=True,timeout=30)
    (tmp_path/'cold.stdout').write_text(done.stdout);(tmp_path/'cold.stderr').write_text(done.stderr)
    record_testsuite_property('full_original_blocked_dispatch',str(tmp_path))
    assert done.returncode==0,done.stdout+done.stderr


@pytest.mark.parametrize('foreign',['123, /usr/NX/bin/nxnode.bin, 175','456, foreign, 10','999, self, 5\n456, foreign, 10'])
def test_foreign_compute_refuses_without_killing_anything(monkeypatch,foreign):
    c=live_control();lines=iter([f"{c['expected_gpu']}, {c['expected_gpu_uuid']}, 8192, 94, 580.178.04",foreign])
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:next(lines))
    monkeypatch.setattr(os,'kill',lambda *a:pytest.fail('Foreign processes must stay untouched.'))
    with pytest.raises(ValueError,match='Another CUDA'):op.base.gpu_gate(c,allowed_pid=999)


def test_rejected_proof_stops_before_either_archive_or_science(tmp_path,monkeypatch):
    c=live_control();sha=hashlib.sha256(op.base.old.json_bytes(c)).hexdigest();folder=tmp_path/'scratch';folder.mkdir(mode=0o700);calls=[]
    monkeypatch.setattr(op,'readiness',lambda c:{'device':folder.stat().st_dev,'inode':folder.stat().st_ino})
    monkeypatch.setattr(op.base,'gpu_gate',lambda *a,**k:{'LOCAL_TEST':True})
    monkeypatch.setattr(op.tempfile,'mkdtemp',lambda **k:str(folder))
    def fail(command,root,**kw):
        calls.append(kw['role']);assert kw['role']=='proof' and 0<kw['deadline']-time.monotonic()<=60
        assert kw['file_limit']==3*2**30;return {'exit_status':1}
    monkeypatch.setattr(op.base.primitives,'supervise',fail)
    class NoRead:
        def read(self,*a):pytest.fail('Proof failure must prevent archive receipt.')
    state=op.run_remote(c,authorization(sha),NoRead(),io.BytesIO(),client.sources(c))
    assert calls==['proof'] and not state['cross_attempt_component_cohort_checks_passed'] and state['work_finished']
    assert state['deadline_unix']-state['started_unix']==600 and 'workers_started' not in state
    assert not (folder/'results').exists() and not (folder/'emergency-space.bin').exists()


@pytest.mark.parametrize('visibility',[None,'0'])
def test_rejected_first_worker_never_starts_second_or_verifier(tmp_path,monkeypatch,visibility):
    c=live_control();sha=hashlib.sha256(op.base.old.json_bytes(c)).hexdigest();folder=tmp_path/'scratch';folder.mkdir(mode=0o700);calls=[];seen=[]
    if visibility is None:monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    else:monkeypatch.setenv('CUDA_VISIBLE_DEVICES',visibility)
    monkeypatch.setattr(op,'readiness',lambda c:{'device':folder.stat().st_dev,'inode':folder.stat().st_ino})
    monkeypatch.setattr(op.tempfile,'mkdtemp',lambda **k:str(folder))
    monkeypatch.setattr(op.base,'gpu_gate',lambda *a,**k:{'LOCAL_TEST':True})
    monkeypatch.setattr(op,'storage_gate',lambda *a,**k:{'LOCAL_TEST':True})
    monkeypatch.setattr(op.base.primitives,'validate_proof',lambda *a:{'LOCAL_TEST':True})
    def fake(command,root,**kw):
        calls.append(kw['role'])
        if kw['role']=='proof':return {'exit_status':0}
        seen.append((command,kw));assert kw['role']=='worker-0' and 0<kw['deadline']-time.monotonic()<=120
        return {'exit_status':1}
    monkeypatch.setattr(op.base.primitives,'supervise',fake)
    with (ROOT/c['original_archive_path']).open('rb') as a,(ROOT/c['reference_archive_path']).open('rb') as b:
        incoming=io.BytesIO(a.read()+b.read())
    state=op.run_remote(c,authorization(sha),incoming,io.BytesIO(),client.sources(c))
    assert calls==['proof','worker-0'] and state['new_workers_completed']==0 and state['references_imported']
    assert not state['cross_attempt_component_cohort_checks_passed'] and not state['verifier_completed']
    command,kw=seen[0];assert kw['file_limit']==32*2**20
    if visibility is None:assert command[:3]==['/usr/bin/env','-u','CUDA_VISIBLE_DEVICES'] and 'CUDA_VISIBLE_DEVICES' not in kw['env']
    else:assert kw['env']['CUDA_VISIBLE_DEVICES']==visibility
    assert not (folder/'results/workers/encoder-1-math-1').exists()


@pytest.mark.parametrize('defect',['tensor_limit','raw_category','input_category','external_hardlink','symlink'])
def test_storage_bounds_and_aliases_refuse(tmp_path,defect):
    folder=tmp_path/'scratch';folder.mkdir(mode=0o700);(folder/'results').mkdir()
    parent=Path('/var/tmp').stat();identity={'device':folder.stat().st_dev,'inode':parent.st_ino}
    if defect=='tensor_limit':
        with (folder/'results/too-large.npz').open('wb') as f:f.truncate(32*2**20+1)
    elif defect=='raw_category':
        for i in range(13):
            with (folder/f'results/return-{i}.npz').open('wb') as f:f.truncate(31*2**20)
    elif defect=='input_category':
        with (folder/'reference-subset.tar').open('wb') as f:f.truncate(256*2**20+1)
    else:
        (tmp_path/'kept').write_bytes(b'preserved')
        if defect=='external_hardlink':os.link(tmp_path/'kept',folder/'results/alias.npz')
        else:(folder/'results/alias.npz').symlink_to(tmp_path/'kept')
    with pytest.raises(ValueError):op.storage_gate(folder,identity)


def test_22_original_refusals_are_mapped_without_parent_session_claim():
    coverage=json.loads((PROPOSAL/'cohort-coverage-map.json').read_text())['explicit_original_refusals']
    code=(ROOT/'models/attention_replay1_cohort_v1.py').read_text()
    for condition in coverage:
        if condition['original_refusal']=='Incomplete or relabelled immutable-input session.':continue
        assert repr(condition['original_refusal']) in code
    assert len(coverage)==22 and 'original_parent_session_qualified\':False' in code


@pytest.mark.parametrize('defect',['mtime','bytes','extra','symlink'])
def test_accepted_full_capture_recheck_refuses_tampering_before_upload(tmp_path,defect):
    session=tmp_path/'old-session';saved=session/'saved';saved.mkdir(parents=True);p=saved/'capture.npz';p.write_bytes(b'unchanged old bytes')
    st=p.stat();files={'capture.npz':{'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':digest(p)}}
    response=session/'response.bin';response.write_bytes(b'{}\n'+op.base.old.json_bytes({'files':files}));sha=digest(response)
    assert client.verify_old_files(session,sha,expected_count=1)==1
    if defect=='mtime':os.utime(p,ns=(1,1))
    elif defect=='bytes':p.write_bytes(p.read_bytes()+b'bad')
    elif defect=='extra':(saved/'unreviewed').write_bytes(b'bad')
    else:p.rename(tmp_path/'preserved');p.symlink_to(tmp_path/'preserved')
    with pytest.raises(ValueError):client.verify_old_files(session,sha,expected_count=1)
