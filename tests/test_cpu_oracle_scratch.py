"""Operational storage/transport tests; no university reference or GPU execution."""
import base64
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import pytest

import models.cpu_oracle_scratch as remote
import models.cpu_oracle_scratch_client as client

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'runs/pretrained/diagnostic/cpu-oracle-university-20261030/packet'


def control(host='linux10603'):
    spec = json.loads((PACKET / 'vjepa-cpu-oracle.tar.gz.manifest.json').read_text())
    return {'kind': remote.KIND, 'host': host, 'policy': remote.POLICY,
        'parent': '/var/tmp', 'expected_filesystem': 'xfs', 'expected_device': 66307,
        'expected_mount': '/', 'CPU_only': True, 'GPU_work': False,
        'paid_compute': False, 'research_admitted': False,
        'archive_sha256': remote.ARCHIVE_SHA, 'archive_bytes': spec['bytes'],
        'handoff_sha256': remote.HANDOFF_SHA, 'tasks': remote.TASKS,
        'archive_manifest': spec}


def authorization(sha, host='linux10603'):
    return {'kind': 'cpu-oracle-scratch-resource-exception-v1', 'approved': True,
        'control_sha256': sha, 'host': host, 'maximum_new_attempts': 1,
        'max_seconds': 600, 'write_test_bytes': remote.POLICY['write_test_bytes'],
        'CPU_only': True, 'paid_compute': False, 'reset_old_claims': False,
        'user_basis': 'OFFLINE TEST FIXTURE ONLY; not a real resource authorization.'}


@pytest.mark.parametrize('key,value', [('parent','/tmp'), ('expected_filesystem','tmpfs'),
    ('expected_device',55), ('expected_mount','/DCNFS/users'), ('CPU_only',False),
    ('GPU_work',True), ('paid_compute',True), ('research_admitted',True),
    ('tasks',[[4,0]]), ('archive_sha256','0'*64)])
def test_changed_storage_source_or_resource_scope_refuses(key, value):
    value_control = control()
    value_control[key] = value
    with pytest.raises(ValueError):
        remote.validate_control(value_control)


@pytest.mark.parametrize('key,value', [('approved',False), ('control_sha256','0'*64),
    ('host','linux10609'), ('maximum_new_attempts',2), ('max_seconds',601),
    ('write_test_bytes',1), ('CPU_only',False), ('paid_compute',True), ('reset_old_claims',True)])
def test_resource_exception_cannot_extend_an_old_or_different_attempt(key, value):
    auth = authorization('1'*64)
    auth[key] = value
    with pytest.raises(ValueError):
        remote.validate_authorization(auth, '1'*64, 'linux10603')


def ready_identity():
    return {'host':'linux10603', 'parent':'/var/tmp', 'device':66307, 'home_device':54,
        'mount':'/', 'filesystem':'xfs', 'options':['rw'], 'owner_uid':0, 'mode':0o1777,
        'available_bytes':170*2**30, 'available_inodes':104479322,
        'writable_for_current_user':True}


@pytest.mark.parametrize('key,value', [('device',54), ('home_device',66307),
    ('filesystem','nfs4'), ('options',['ro']), ('mode',0o777), ('owner_uid',1000),
    ('available_bytes',2**30), ('available_inodes',100), ('writable_for_current_user',False),
    ('host','linux10609'), ('parent','/tmp')])
def test_readiness_rejects_shared_quota_mount_or_insufficient_capacity(key, value):
    observation = ready_identity()
    observation[key] = value
    with pytest.raises(ValueError):
        remote.validate_readiness(observation, control())


def test_user_reported_local_xfs_identity_meets_planning_gate():
    remote.validate_readiness(ready_identity(), control())
    assert remote.mount_for('/var/tmp', '1 0 259:3 / / rw,relatime - xfs /dev/nvme0n1p3 rw\n')['filesystem']=='xfs'
    text = ('1 0 259:3 / / rw - xfs /dev/nvme0n1p3 rw\n'
            '2 1 0:5 / /var/tmp rw - tmpfs tmpfs rw\n')
    assert remote.mount_for('/var/tmp', text)['filesystem']=='tmpfs'


def test_actual_small_write_fsync_readback_and_own_file_cleanup(tmp_path):
    witness = tmp_path/'keep-me'; witness.write_bytes(b'keep')
    result = remote.full_write_test(tmp_path, 16384, time.monotonic()+10)
    assert result['fsync_succeeded'] and result['bytes_written']==16384 and result['bytes_read_back']==16384
    assert result['allocated_bytes'] >=16384 and not result['reservation']
    assert not (tmp_path/'space-proof.bin.partial').exists() and witness.read_bytes()==b'keep'


def test_disk_quota_error_retains_failed_proof_and_cannot_start_a_child(tmp_path, monkeypatch):
    def quota_error(*args):
        raise OSError(errno.EDQUOT, 'Disk quota exceeded')
    monkeypatch.setattr(remote.os,'fsync',quota_error)
    with pytest.raises(OSError, match='quota'):
        remote.full_write_test(tmp_path,16384,time.monotonic()+10)
    assert (tmp_path/'space-proof.bin.partial').stat().st_size==16384
    assert not (tmp_path/'launcher.log').exists()


def test_expired_storage_write_test_leaves_partial_without_removing_existing_files(tmp_path):
    witness=tmp_path/'witness';witness.write_text('preserve')
    with pytest.raises(TimeoutError):
        remote.full_write_test(tmp_path,16384,time.monotonic()-1)
    assert (tmp_path/'space-proof.bin.partial').is_file() and witness.read_text()=='preserve'


@pytest.mark.parametrize('data,sha', [(b'truncated','0'*64),(b'changed bytes','1'*64)])
def test_truncated_or_changed_upload_preserves_partial(tmp_path,data,sha):
    with pytest.raises(ValueError):
        remote.receive_archive(io.BytesIO(data),tmp_path/'packet.tar.gz',20,sha,time.monotonic()+10)
    assert (tmp_path/'packet.tar.gz.partial').exists() and not (tmp_path/'packet.tar.gz').exists()


@pytest.mark.parametrize('defect',['traversal','symlink','duplicate','checksum'])
def test_extractor_refuses_untrusted_archive_before_creating_code(tmp_path,defect):
    path=tmp_path/'packet.tar.gz'
    with tarfile.open(path,'w:gz') as archive:
        name='../escape' if defect=='traversal' else 'file'
        item=tarfile.TarInfo(name);item.size=4
        if defect=='symlink':item.type=tarfile.SYMTYPE;item.linkname='../escape';item.size=0
        archive.addfile(item,io.BytesIO(b'data'))
        if defect=='duplicate':archive.addfile(item,io.BytesIO(b'data'))
    spec={'sha256':remote.digest(path),'files':{name:{'bytes':4,'sha256':hashlib.sha256(b'bad!' if defect=='checksum' else b'data').hexdigest()}}}
    with pytest.raises(ValueError):remote.extract_packet(path,spec,tmp_path/'code')
    assert not (tmp_path/'code').exists() and not (tmp_path.parent/'escape').exists()


def test_stream_inventory_refuses_symlinks_and_unrelated_hardlinks(tmp_path):
    folder=tmp_path/'scratch';folder.mkdir()
    outside=tmp_path/'outside';outside.write_bytes(b'private')
    link=folder/'link';link.symlink_to(outside)
    with pytest.raises(ValueError):remote.tree_inventory(folder)
    link.unlink();os.link(outside,link)
    with pytest.raises(ValueError):remote.tree_inventory(folder)


def test_interrupted_frozen_npz_link_pair_is_preserved_as_regular_evidence(tmp_path):
    partial=tmp_path/'chunk.npz.partial';partial.write_bytes(b'captured')
    os.link(partial,tmp_path/'chunk.npz')
    files,allocated=remote.tree_inventory(tmp_path,hashing=True)
    assert set(files)=={'chunk.npz','chunk.npz.partial'}
    assert files['chunk.npz']['sha256']==files['chunk.npz.partial']['sha256']
    assert allocated==partial.stat().st_blocks*512


def scratch_fixture(tmp_path):
    folder=tmp_path/'scratch';folder.mkdir(mode=0o700)
    parent=tmp_path.stat()
    return folder,{'device':parent.st_dev,'inode':parent.st_ino}


@pytest.mark.parametrize('category,size', [('array',33*2**20),('archive',2**30+9*2**20),('metadata',65*2**20)])
def test_runtime_gate_catches_array_archive_and_metadata_growth(tmp_path,category,size):
    folder,identity=scratch_fixture(tmp_path)
    name={'array':'results/chunk.npz.partial','archive':'results/evidence.tar.gz','metadata':'diagnostic.log'}[category]
    path=folder/name;path.parent.mkdir(exist_ok=True)
    with path.open('wb') as handle:handle.truncate(size) # sparse test fixture, no large real allocation
    with pytest.raises(ValueError):remote.storage_gate(folder,identity,parent_path=tmp_path)


def test_changed_parent_identity_stops_capture(tmp_path):
    folder,identity=scratch_fixture(tmp_path);identity['inode']+=1
    with pytest.raises(ValueError):remote.storage_gate(folder,identity,parent_path=tmp_path)


def test_foreign_pid_is_never_in_descendant_identity_set(tmp_path):
    def proc(pid,start,children):
        p=tmp_path/str(pid);(p/'task'/str(pid)).mkdir(parents=True)
        (p/'stat').write_text(str(pid)+' (python worker) '+' '.join(['S']+['0']*18+[str(start)]+['0']*5))
        (p/'task'/str(pid)/'children').write_text(children)
    proc(10,100,'20');proc(20,200,'');proc(999,900,'')
    assert remote.descendants(10,tmp_path)=={20:200}
    assert 999 not in remote.descendants(10,tmp_path)


def test_supervisor_finishes_a_small_stub_and_never_initializes_torch(tmp_path):
    report=remote.supervise([sys.executable,'-B','-c','print("operational stub only")'],tmp_path,time.monotonic()+120,lambda:{'logical_bytes':0,'allocated_bytes':0})
    assert report['exit_status']==0 and not report['polling_is_a_hard_kernel_quota']
    assert b'operational stub only' in (tmp_path/'launcher.log').read_bytes()


def test_expired_clock_refuses_before_popen(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No child may start')
    monkeypatch.setattr(remote.subprocess,'Popen',forbidden)
    with pytest.raises(TimeoutError):remote.supervise(['unused'],tmp_path,time.monotonic()-1,lambda:{})
    assert not (tmp_path/'launcher.log').exists()


def test_supervisor_stops_its_child_on_storage_refusal(tmp_path):
    def gate():
        if (tmp_path/'started').exists():raise ValueError('storage floor')
        return {'logical_bytes':0,'allocated_bytes':0}
    code='from pathlib import Path;import time,os;Path('+repr(str(tmp_path/'started'))+').write_text(str(os.getpid()));time.sleep(30)'
    before=time.monotonic()
    with pytest.raises(ValueError,match='storage floor'):
        remote.supervise([sys.executable,'-B','-c',code],tmp_path,time.monotonic()+120,gate)
    assert time.monotonic()-before<5
    pid=int((tmp_path/'started').read_text())
    with pytest.raises(ProcessLookupError):os.kill(pid,0)


@pytest.fixture
def local_operator(tmp_path,monkeypatch):
    base=tmp_path/'operator';base.mkdir()
    value=control();value['local_source_sha256']={'models/cpu_oracle_scratch.py':remote.digest(ROOT/'models/cpu_oracle_scratch.py')}
    value['prior_claims_sha256']={}
    known=base/'known-hosts';known.write_text('test fixture, no network use\n')
    value['known_hosts_sha256']=remote.digest(known)
    value['original_archive_path']=str((PACKET/'vjepa-cpu-oracle.tar.gz').relative_to(ROOT))
    (base/'CONTROL.json').write_bytes(remote.json_bytes(value))
    sha=remote.digest(base/'CONTROL.json')
    monkeypatch.setattr(client.shutil,'disk_usage',lambda path:shutil._ntuple_diskusage(100*2**30,0,100*2**30))
    return base,value,sha


def arm_fixture(base,sha):
    (base/'AUTHORIZATION.json').write_bytes(remote.json_bytes(authorization(sha)))


def test_unarmed_client_refuses_before_claim_or_network(local_operator,monkeypatch):
    base,_,sha=local_operator
    def forbidden(*args,**kwargs):raise AssertionError('Network use forbidden')
    monkeypatch.setattr(client.subprocess,'Popen',forbidden)
    with pytest.raises(ValueError,match='UNARMED'):client.run_client(base,sha)
    assert not (base/'ATTEMPT.json').exists()


def test_consumed_new_claim_cannot_repeat(local_operator,monkeypatch):
    base,_,sha=local_operator;arm_fixture(base,sha);(base/'ATTEMPT.json').write_text('{}')
    def forbidden(*args,**kwargs):raise AssertionError('Must not reconnect')
    monkeypatch.setattr(client.subprocess,'Popen',forbidden)
    with pytest.raises(FileExistsError):client.run_client(base,sha)


def response_fixture(folder,control_value,sha,*,error=None):
    folder.mkdir()
    (folder/'operator-control.json').write_bytes(remote.json_bytes(control_value))
    state={'kind':remote.KIND,'host':'linux10603','control_sha256':sha,'run_phase_finished':True,
        'CPU_only':True,'GPU_work':False,'paid_compute':False,'scientific_review_complete':False,
        'reference_process_started':False,'error':error}
    (folder/'operator-result.json').write_bytes(remote.json_bytes(state))
    (folder/'saved.partial').write_bytes(b'preserved partial bytes')
    return folder


def write_response(path,folder,sha):
    with path.open('wb') as stream:
        stream.write(remote.json_bytes({'phase':'scratch-created','kind':remote.KIND,'host':'linux10603',
            'control_sha256':sha,'scratch_directory':'/var/tmp/vjepa-cpu-oracle-20261104-linux10603-fixture'}))
        remote.result_stream(folder,sha,stream)


def test_complete_partial_evidence_transport_is_not_a_scientific_pass(tmp_path,local_operator):
    _,value,sha=local_operator
    folder=response_fixture(tmp_path/'remote',value,sha,error='simulated quota failure; no model')
    path=tmp_path/'response.bin';write_response(path,folder,sha)
    receipt=client.verify_response(path,tmp_path/'saved',value,sha)
    assert receipt['transport_verified'] and not receipt['scientific_review_complete']
    assert not receipt['reference_process_started'] and receipt['technical_error']
    assert (tmp_path/'saved/saved.partial').read_bytes()==b'preserved partial bytes'
    assert not receipt['remote_scratch_deleted']


def test_changed_stream_member_is_rejected_and_partial_output_retained(tmp_path,local_operator):
    _,value,sha=local_operator
    folder=response_fixture(tmp_path/'remote',value,sha);path=tmp_path/'response.bin';write_response(path,folder,sha)
    data=path.read_bytes();assert b'preserved partial bytes' in data
    path.write_bytes(data.replace(b'preserved partial bytes',b'corrupted partial byte!'))
    with pytest.raises(ValueError):client.verify_response(path,tmp_path/'saved',value,sha)
    assert path.exists() and (tmp_path/'saved').exists()
    assert not (tmp_path/'transport-verification.json').exists()


def test_one_connection_mock_and_copy_only_recovery_do_not_repeat_model(tmp_path,local_operator):
    base,value,sha=local_operator;arm_fixture(base,sha)
    fixture=response_fixture(tmp_path/'remote',value,sha)
    complete=tmp_path/'complete.bin';write_response(complete,fixture,sha)
    stub=tmp_path/'ssh_stub.py';counter=tmp_path/'calls';mode=tmp_path/'mode';mode.write_text('partial')
    stub.write_text('from pathlib import Path\nimport sys\n'
        'sys.stdin.buffer.read()\n'
        'counter=Path('+repr(str(counter))+')\n'
        'counter.write_text(counter.read_text()+"1" if counter.exists() else "1")\n'
        'data=Path('+repr(str(complete))+').read_bytes()\n'
        'if Path('+repr(str(mode))+').read_text()=="partial":data=data.split(b"\\n",1)[0]+b"\\n"\n'
        'sys.stdout.buffer.write(data)\n')
    command=[sys.executable,'-B',str(stub)]
    with pytest.raises(ValueError):client.run_client(base,sha,ssh_override=command)
    assert counter.read_text()=='1' and (base/'ATTEMPT.json').exists()
    mode.write_text('complete')
    receipt=client.run_client(base,sha,download_only=True,ssh_override=command)
    assert counter.read_text()=='11' and receipt['transport_verified']
    assert not receipt['reference_process_started'] and not receipt['model_restarted_for_collection']
    assert len(list(base.glob('attended-*')))==1 and len(list(base.glob('download-only-*')))==1


def test_copy_only_path_refuses_unfinished_or_unrelated_scratch(tmp_path):
    with pytest.raises(ValueError):remote.collect_remote(control(),str(tmp_path),io.BytesIO())


def test_remote_quota_refusal_streams_evidence_without_loading_or_starting_probe(tmp_path,monkeypatch):
    value=control();sha=hashlib.sha256(remote.json_bytes(value)).hexdigest()
    folder=tmp_path/'new-scratch';folder.mkdir(mode=0o700)
    observation={**ready_identity(),'inode':tmp_path.stat().st_ino}
    monkeypatch.setattr(remote,'readiness',lambda unused:observation)
    monkeypatch.setattr(remote.tempfile,'mkdtemp',lambda **unused:str(folder))
    monkeypatch.setattr(remote.signal,'signal',lambda *args:None)
    monkeypatch.setattr(remote.signal,'setitimer',lambda *args:None)
    def quota(*args,**kwargs):
        (folder/'space-proof.bin.partial').write_bytes(b'partial allocation proof')
        raise OSError(errno.EDQUOT,'quota refusal fixture')
    monkeypatch.setattr(remote,'full_write_test',quota)
    def forbidden(*args,**kwargs):raise AssertionError('No reference child may start')
    monkeypatch.setattr(remote.subprocess,'Popen',forbidden)
    stream=io.BytesIO()
    result=remote.run_remote(value,authorization(sha),io.BytesIO(),stream)
    assert result['run_phase_finished'] and not result['reference_process_started']
    assert result['error_type']=='OSError' and b'partial allocation proof' in stream.getvalue()
    assert not (folder/'emergency-space.bin').exists()
    assert not (folder/'code').exists() and not (folder/'new-attended-plan.json').exists()


def test_control_host_cannot_start_without_target_review(local_operator,monkeypatch):
    base,value,_=local_operator;value['host']='linux10609';value['target_control_sha256']='a'*64
    (base/'CONTROL.json').write_bytes(remote.json_bytes(value));sha=remote.digest(base/'CONTROL.json')
    (base/'AUTHORIZATION.json').write_bytes(remote.json_bytes(authorization(sha,'linux10609')))
    def forbidden(*args,**kwargs):raise AssertionError('Must not connect before target review')
    monkeypatch.setattr(client.subprocess,'Popen',forbidden)
    with pytest.raises(FileNotFoundError):client.run_client(base,sha)
    assert not (base/'ATTEMPT.json').exists()


def test_original_complete_packet_restores_without_importing_numerical_code(tmp_path):
    spec=control()['archive_manifest']
    remote.extract_packet(PACKET/'vjepa-cpu-oracle.tar.gz',spec,tmp_path/'code')
    assert len(spec['files'])==129
    for name,item in spec['files'].items():
        path=tmp_path/'code'/name
        assert path.stat().st_size==item['bytes'] and remote.digest(path)==item['sha256']
    assert remote.digest(tmp_path/'code/CPU-ORACLE-HANDOFF.json')==remote.HANDOFF_SHA
    assert not (tmp_path/'code/cpu-oracle-claims').exists()


def test_individual_file_limit_is_enforced_by_child_kernel_limit(tmp_path):
    path=tmp_path/'bounded-file'
    code=('import errno,sys\n'
        'try:\n'
        ' with open('+repr(str(path))+',"wb") as f:f.truncate('+str(remote.POLICY['max_archive_bytes']+1)+')\n'
        'except OSError as exc:sys.exit(0 if exc.errno==errno.EFBIG else 2)\n'
        'sys.exit(3)\n')
    result=remote.supervise([sys.executable,'-B','-c',code],tmp_path,time.monotonic()+120,lambda:{})
    assert result['exit_status']==0 and path.stat().st_size<=remote.POLICY['max_archive_bytes']
