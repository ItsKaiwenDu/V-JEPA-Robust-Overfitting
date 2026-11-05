"""Scope, resource and transport controls; no SSH/CUDA/full allocation proof."""
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

import pytest
from models import oracle_boundary_operator_v1 as op
from models import oracle_boundary_client_v1 as client

ROOT=Path(__file__).resolve().parents[1]
PACKET=ROOT/'runs/pretrained/diagnostic/fixed-attention-university-20261030/packet'
OLD=ROOT/'runs/pretrained/diagnostic/cpu-oracle-control-v1-20261104/linux10609'
SEALED=ROOT/'runs/pretrained/diagnostic/cpu-oracle-context-comparison-20261104/verified-review/sealed-inputs.json'


def control():
    c=json.loads((OLD/'CONTROL.json').read_text())
    c.update(kind=op.KIND,host='linux10603',role='pre-SDPA-original-math-label-0',policy=op.POLICY,
        expected_gpu='NVIDIA GeForce RTX 3070',expected_gpu_uuid='GPU-e54fae90-feeb-8106-d8c0-1974fe48555f',
        expected_driver='580.178.04',expected_ssh_fingerprint=op.SSH_FINGERPRINT,prerequisite=dict(op.PREREQUISITE),CUDA_setup_allowed=True,GPU_SDPA_allowed=False,
        paid_compute=False,research_admitted=False,original_refusals_waived=False,reuse_old_grant=False,
        authorization_present_at_preparation=False,
        tasks=[{'case':'encoder-1','arm_label':'math','replay':0,'original_oracle_calls':1}],
        archive_sha256=op.ARCHIVE_SHA,archive_bytes=85622679,handoff_sha256=op.HANDOFF_SHA,
        original_archive_path=str((PACKET/'vjepa-fixed-attention.tar.gz').relative_to(ROOT)),
        archive_manifest=json.loads((PACKET/'vjepa-fixed-attention.tar.gz.manifest.json').read_text()))
    for name in ('CPU_only','GPU_work','target_prerequisite','prior_claims_sha256'):
        c.pop(name,None)
    c['prior_bindings_sha256']=json.loads(SEALED.read_text())
    for p in (SEALED,ROOT/'report/checks/2026-11-04-cpu-oracle-context-comparison.json',
            ROOT/'report/checks/2026-11-04-cpu-oracle-control-v1-review.json',OLD/'ATTEMPT.json'):
        c['prior_bindings_sha256'][str(p.relative_to(ROOT))]=op.old.digest(p)
    c['local_source_sha256']={n:op.old.digest(ROOT/n) for n in op.LOCAL_NAMES}
    c['remote_source_sha256']={n:hashlib.sha256(op.OVERLAY_INIT).hexdigest() if n=='models/__init__.py'
        else op.old.digest(ROOT/n) for n in op.REMOTE_NAMES}
    return c


def authorization(sha):
    return {'kind':op.AUTH_KIND,'approved':True,'control_sha256':sha,'host':'linux10603',
        'maximum_new_attempts':1,'max_seconds':600,'worker_seconds':120,'original_oracle_calls':1,
        'CUDA_setup_allowed':True,'GPU_SDPA_allowed':False,'write_test_bytes':2684354560,
        'setup_ceiling':3*2**30,'capture_ceiling':9*2**30//4,'collection_seconds':600,
        'paid_compute':False,'reset_old_claims':False,'automatic_retries':0,
        'user_basis':'LOCAL TEST FIXTURE ONLY; grants no university resources.'}


def base(tmp_path,armed=False):
    b=tmp_path/'client';b.mkdir();(b/'known-hosts').write_bytes((ROOT/'runs/pretrained/diagnostic/cpu-oracle-capture-v3-20261104/linux10603/known-hosts').read_bytes())
    c=control();c['known_hosts_sha256']=op.old.digest(b/'known-hosts')
    op.old.write_json(b/'CONTROL.json',c);sha=op.old.digest(b/'CONTROL.json')
    if armed:op.old.write_json(b/'AUTHORIZATION.json',authorization(sha))
    return b,c,sha


@pytest.mark.parametrize('field,value',[
    ('kind',op.primitives.KIND),('host','linux10609'),('role','control'),('CUDA_setup_allowed',False),
    ('GPU_SDPA_allowed',True),('reuse_old_grant',True),('paid_compute',True),('research_admitted',True),
    ('original_refusals_waived',True),('archive_bytes',19042363),('archive_sha256',op.old.ARCHIVE_SHA),
    ('tasks',[{'case':'encoder-0','arm_label':'math','replay':0,'original_oracle_calls':1}])])
def test_wrong_scope_cannot_be_relabelled_as_boundary(field,value):
    c=control();c[field]=value
    with pytest.raises(ValueError):op.validate_control(c)


@pytest.mark.parametrize('field',['max_seconds','worker_seconds','write_test_bytes','setup_ceiling','capture_ceiling',
    'collection_seconds','max_raw_bytes','max_code_bytes','max_input_bytes'])
def test_changed_policy_refuses(field):
    c=control();c['policy']=dict(c['policy']);c['policy'][field]+=1
    with pytest.raises(ValueError):op.validate_control(c)


@pytest.mark.parametrize('field,value',[('host','linux10609'),('maximum_new_attempts',2),('CUDA_setup_allowed',False),
    ('GPU_SDPA_allowed',True),('worker_seconds',121),('original_oracle_calls',2),('max_seconds',601),
    ('reset_old_claims',True),('automatic_retries',1),('paid_compute',True)])
def test_new_authorization_is_exact_and_cannot_extend_old_grants(field,value):
    a=authorization('a'*64);a[field]=value
    with pytest.raises(ValueError):op.validate_authorization(a,'a'*64)


def test_consumed_cpu_only_grant_cannot_authorize_cuda_domain():
    old=json.loads((OLD/'AUTHORIZATION.json').read_text())
    with pytest.raises(ValueError):op.validate_authorization(old,old['control_sha256'])


def test_unarmed_client_refuses_before_claim_session_or_process(tmp_path,monkeypatch):
    b,c,sha=base(tmp_path)
    def denied(*a,**k):pytest.fail('Unarmed helper must not start a process.')
    monkeypatch.setattr(subprocess,'Popen',denied)
    with pytest.raises(ValueError,match='UNARMED'):client.run_client(b,sha)
    assert sorted(p.name for p in b.iterdir())==['CONTROL.json','known-hosts']


def test_new_source_maps_and_historical_prerequisites_reverify_without_processes(tmp_path,monkeypatch):
    b,c,sha=base(tmp_path,True)
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('No process in verification.'))
    assert client.verify_local(b,sha)==c
    assert len(c['prior_bindings_sha256'])>=935
    data=client.sources(c)
    assert len(data)==9 and sum(map(len,data.values()))<op.POLICY['max_overlay_bytes']


def test_consumed_new_claim_refuses_before_rehash_or_ssh(tmp_path,monkeypatch):
    b,c,sha=base(tmp_path,True);op.old.write_json(b/'ATTEMPT.json',{'consumed':True})
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('No process.'))
    with pytest.raises(FileExistsError):client.run_client(b,sha)


def test_compressed_bootstrap_is_host_pinned_bounded_and_stdlib_only(tmp_path):
    b,c,sha=base(tmp_path);command=client.command_for(b,c)
    assert command[0]=='ssh' and command[-2]=='kdu@linux10603.dc.engr.scu.edu'
    assert 'StrictHostKeyChecking=yes' in command and 'ControlMaster=no' in command
    assert len(command[-1].encode())<120000
    code=shlex.split(command[-1])[-1]
    harness='''
import sys,json
code=sys.argv[1];code=code[:code.rindex('sys.modules["models.oracle_boundary_operator_v1"].remote_main')]
exec(code)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print(json.dumps({'scientific_modules_loaded':False,'decoded':True}))
'''
    result=subprocess.run([sys.executable,'-I','-B','-c',harness,code],input=client.bootstrap_payload(c).decode(),capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['scientific_modules_loaded'] is False


@pytest.mark.parametrize('foreign',['123, /usr/NX/bin/nxnode.bin, 175','456, other, 800','123, unknown, 1\n456, other, 1'])
def test_gpu_guard_rejects_foreign_compute_without_killing_it(monkeypatch,foreign):
    c=control();answers=iter([f"{c['expected_gpu']}, {c['expected_gpu_uuid']}, 8192, 94, 580.178.04",foreign])
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:next(answers))
    with pytest.raises(ValueError,match='Another CUDA'):op.gpu_gate(c,allowed_pid=999)


@pytest.mark.parametrize('row',['wrong, GPU-wrong, 8192, 94, 580.178.04',
    'NVIDIA GeForce RTX 3070, GPU-e54fae90-feeb-8106-d8c0-1974fe48555f, 8192, 900, 580.178.04',
    'NVIDIA GeForce RTX 3070, GPU-e54fae90-feeb-8106-d8c0-1974fe48555f, 8192, 94, wrong'])
def test_gpu_identity_memory_driver_refusal(monkeypatch,row):
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:row)
    with pytest.raises(ValueError):op.gpu_gate(control())


def test_gpu_guard_allows_only_observed_owned_worker(monkeypatch):
    c=control();answers=iter([f"{c['expected_gpu']}, {c['expected_gpu_uuid']}, 8192, 250, 580.178.04",'321, python, 120'])
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:next(answers))
    value=op.gpu_gate(c,allowed_pid=321)
    assert value['allowed_compute_pid']==321 and value['sampling_is_not_continuous_exclusivity']


def wire(c,sha,*,mutate=None):
    root={'operator-control.json':op.old.json_bytes(c),'operator-result.json':op.old.json_bytes({
        'kind':op.KIND,'host':c['host'],'control_sha256':sha,'work_finished':True,'CUDA_setup_allowed':True,
        'GPU_SDPA_allowed':False,'paid_compute':False,'original_full_worker_accepted':False,
        'old_refusals_waived':False,'scientific_review_complete':False,'scope_capture_complete':False,
        'capture_outcome':'technical-failure','original_gate_passed':None,'error':'Local fixture: proof refused before worker.'})}
    files={n:{'bytes':len(b),'mtime_ns':1234567890123456789,'sha256':hashlib.sha256(b).hexdigest()} for n,b in root.items()}
    if mutate:mutate(files)
    out=io.BytesIO();out.write(op.old.json_bytes({'kind':op.KIND,'phase':'scratch-created','host':c['host'],
        'control_sha256':sha,'scratch_directory':'/var/tmp/'+op.PREFIX+c['host']+'-local'}))
    out.write(op.old.json_bytes({'kind':op.KIND,'phase':'collection','control_sha256':sha,'files':files,
        'logical_bytes':sum(x['bytes'] for x in files.values()),'scientific_review_complete':False,'model_restarted_for_collection':False}))
    with tarfile.open(mode='w|',fileobj=out) as tar:
        for name,data in root.items():
            member=tarfile.TarInfo(name);member.size=len(data);tar.addfile(member,io.BytesIO(data))
    return out.getvalue()


def test_finished_failure_transport_preserves_hashes_and_exact_mtime(tmp_path):
    b,c,sha=base(tmp_path);p=tmp_path/'response.bin';p.write_bytes(wire(c,sha));saved=tmp_path/'saved'
    receipt=client.verify_response(p,saved,c,sha)
    assert receipt['transport_verified'] and receipt['capture_outcome']=='technical-failure'
    assert not receipt['scope_capture_complete'] and not receipt['original_full_worker_accepted']
    assert (saved/'operator-result.json').stat().st_mtime_ns==1234567890123456789


@pytest.mark.parametrize('defect',['truncated','checksum','traversal','wrong_host','missing_manifest'])
def test_bad_transport_never_becomes_verified(tmp_path,defect):
    b,c,sha=base(tmp_path);data=wire(c,sha)
    if defect=='truncated':data=data[:len(data)//3]
    elif defect=='checksum':data=wire(c,sha,mutate=lambda f:f['operator-result.json'].update(sha256='0'*64))
    elif defect=='traversal':data=wire(c,sha,mutate=lambda f:f.update({'../escape':{'bytes':1,'mtime_ns':0,'sha256':'0'*64}}))
    elif defect=='wrong_host':data=data.replace(b'"host":"linux10603"',b'"host":"linux10609"',1)
    elif defect=='missing_manifest':data=data.split(b'\n',1)[0]+b'\n'
    p=tmp_path/'response.bin';p.write_bytes(data)
    with pytest.raises((ValueError,OSError,KeyError,tarfile.TarError)):client.verify_response(p,tmp_path/'saved',c,sha)
    assert not (tmp_path/'transport-verification.json').exists()


def test_failed_allocation_collects_before_any_original_worker(tmp_path,monkeypatch):
    c=control();sha=hashlib.sha256(op.old.json_bytes(c)).hexdigest();folder=tmp_path/'scratch'
    folder.mkdir(mode=0o700);calls=[]
    monkeypatch.setattr(op,'readiness',lambda c:{'host':c['host']})
    monkeypatch.setattr(op,'gpu_gate',lambda *a,**k:{})
    monkeypatch.setattr(op.tempfile,'mkdtemp',lambda **k:str(folder))
    def fail_proof(*a,**k):calls.append(k['role']);return {'exit_status':1}
    monkeypatch.setattr(op.primitives,'supervise',fail_proof)
    state=op.run_remote(c,authorization(sha),io.BytesIO(),io.BytesIO(),client.sources(c))
    assert calls==['proof'] and state['work_finished'] and state['capture_outcome']=='technical-failure'
    assert not state.get('worker_process_started',False) and not (folder/'code').exists()
    assert not (folder/'emergency-space.bin').exists()


def test_download_only_constructs_no_input_archive_or_worker(tmp_path,monkeypatch):
    b,c,sha=base(tmp_path);s=b/'attended-local';s.mkdir()
    (s/'response.bin').write_bytes(wire(c,sha))
    op.old.write_json(b/'ATTEMPT.json',{'kind':op.KIND,'control_sha256':sha,'local_session':s.name})
    # Verification is read-only and download ignores a consumed/absent run grant.
    monkeypatch.setattr(client,'verify_prerequisites',lambda c:None)
    source=tmp_path/'fixture.bin';source.write_bytes(wire(c,sha))
    result=client.run_client(b,sha,download_only=True,ssh_override=[sys.executable,'-I','-c',
        'import sys;sys.stdin.buffer.read();sys.stdout.buffer.write(open(sys.argv[1],"rb").read())',str(source)])
    assert result['transport_verified'] and result['model_restarted_for_collection'] is False
    assert not any(b.rglob('upload-stream.bin')) and not (b/'AUTHORIZATION.json').exists()


@pytest.mark.parametrize('suffix',['.json','.pt','.npz'])
def test_inventory_accepts_only_exact_atomic_writer_pairs(tmp_path,suffix):
    folder=tmp_path/'capture';folder.mkdir();a=folder/('result'+suffix);a.write_bytes(b'local evidence')
    b=a.with_name(a.name+'.partial');os.link(a,b)
    files,allocated=op.inventory(folder,hashing=True)
    assert set(files)=={a.name,b.name} and allocated==a.stat().st_blocks*512
    assert files[a.name]['sha256']==files[b.name]['sha256']
    os.link(a,tmp_path/'outside')
    with pytest.raises(ValueError,match='external hardlink'):op.inventory(folder)


@pytest.mark.parametrize('bad',['symlink','external_hardlink','file_limit','byte_limit'])
def test_inventory_refuses_aliases_and_extent_overflow(tmp_path,bad):
    folder=tmp_path/'capture';folder.mkdir();outside=tmp_path/'outside';outside.write_bytes(b'bound')
    if bad=='symlink':(folder/'inside').symlink_to(outside)
    elif bad=='external_hardlink':os.link(outside,folder/'inside')
    elif bad=='file_limit':
        for i in range(op.POLICY['max_files']+1):(folder/str(i)).touch()
    else:(folder/'inside').write_bytes(b'12345')
    with pytest.raises(ValueError):op.inventory(folder,max_bytes=4 if bad=='byte_limit' else None)


def test_one_local_transport_sends_bound_bootstrap_grant_and_full_packet_once(tmp_path):
    b,c,sha=base(tmp_path,True);fixture=tmp_path/'response.fixture';fixture.write_bytes(wire(c,sha))
    peer=tmp_path/'fake-ssh.py'
    peer.write_text('''
import hashlib,json,sys
bundle=json.loads(sys.stdin.buffer.readline(1048577));control=bundle['control']
assert control['host']=='linux10603' and len(bundle['sources'])==9
auth=json.loads(sys.stdin.buffer.readline(1048577))
assert auth['kind']=='oracle-boundary-resource-exception-v1' and auth['original_oracle_calls']==1
data=sys.stdin.buffer.read(control['archive_bytes']+1)
assert len(data)==85622679 and hashlib.sha256(data).hexdigest()==control['archive_sha256']
sys.stderr.write('LOCAL FAKE SSH: one stream verified, no model dispatched.\\n')
sys.stdout.buffer.write(open(sys.argv[1],'rb').read())
''')
    result=client.run_client(b,sha,ssh_override=[sys.executable,'-I','-B',str(peer),str(fixture)])
    assert result['transport_verified'] and not result['scope_capture_complete']
    sessions=list(b.glob('attended-*'));assert len(sessions)==1
    assert 'one stream verified' in (sessions[0]/'ssh.stderr').read_text()
    claim=json.loads((b/'ATTEMPT.json').read_text());assert claim['single_attempt']
    before=op.old.digest(b/'ATTEMPT.json')
    with pytest.raises(FileExistsError):client.run_client(b,sha,ssh_override=['must-not-run'])
    assert op.old.digest(b/'ATTEMPT.json')==before and not any(b.rglob('upload-stream.bin'))


@pytest.mark.parametrize('bad',['other_host','fingerprint'])
def test_public_key_target_and_fingerprint_checked_before_claim(tmp_path,bad):
    b,c,sha=base(tmp_path,True)
    if bad=='other_host':
        (b/'known-hosts').write_bytes((OLD/'known-hosts').read_bytes())
        c['known_hosts_sha256']=op.old.digest(b/'known-hosts')
        (b/'CONTROL.json').unlink();op.old.write_json(b/'CONTROL.json',c);sha=op.old.digest(b/'CONTROL.json')
        (b/'AUTHORIZATION.json').unlink();op.old.write_json(b/'AUTHORIZATION.json',authorization(sha))
    else:
        c['expected_ssh_fingerprint']='SHA256:wrong'
        (b/'CONTROL.json').unlink();op.old.write_json(b/'CONTROL.json',c);sha=op.old.digest(b/'CONTROL.json')
    with pytest.raises(ValueError):client.verify_local(b,sha)
    assert not (b/'ATTEMPT.json').exists()
