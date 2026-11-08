"""No-network preparation, clocks/ceilings, cold authorization and same-stream tests."""
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace
import pytest
from models import predictor_sdpa_handoff_operator_v2 as op
from models import predictor_sdpa_handoff_client_v2 as client
from models import predictor_sdpa_target_v1 as target
from scripts.prepare_predictor_sdpa_handoff_v2 import prepare
from test_predictor_sdpa_handoff_v1 import cohort,CONTROL,ROOT,write,copy_component


def authorization(sha='fixture-control'):
    return copy.deepcopy(op.authorization_template(sha,'Local simulation fixture only; no user or university resource grant.'))


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    folder=tmp_path_factory.mktemp('SDPA-prepared')/'handoff';c,info=prepare(ROOT,folder)
    return folder,c,info


def test_prepared_payload_is_new_exact_unarmed_and_has_no_candidate_or_extra_data(prepared):
    folder,c,info=prepared;actual,sha=client.check_local(folder)
    assert actual==c and sha==info['control_sha256'] and info['packet_bytes']==17192095 and info['packet_members']==99
    assert info['prior_pins_verified']==21507 and info['overlay_files']==len(op.REMOTE_NAMES)
    assert info['overlay_bytes']<2*op.MiB and info['stdin_bootstrap_bytes']<2*op.MiB
    assert not (folder/'AUTHORIZATION.json').exists() and not (folder/'ATTEMPT.json').exists()
    assert set(op.COMPONENT_SOURCES)<=set(op.REMOTE_NAMES)
    assert op.RESOURCE['static_reserved_seconds']==180+420+4*120+120<=1500
    assert c['contract']['candidate_remains_rejected'] and c['contract']['candidate_failed_input_entries']==3724
    op.validate_authorization(authorization(sha),sha)


def test_unarmed_copy_and_retrieval_refuse_before_claim_or_network(prepared,tmp_path):
    folder,c,info=prepared
    def forbidden(*a,**k):raise AssertionError('No SSH expected')
    with pytest.raises(ValueError,match='Unarmed'):client.run(folder,popen=forbidden)
    with pytest.raises(ValueError,match='Separate retrieval'):client.run(folder,download_only=True,popen=forbidden)
    copied=tmp_path/'copied';shutil.copytree(folder,copied)
    with pytest.raises(ValueError,match='Exact local handoff'):client.check_local(copied)
    assert not (folder/'ATTEMPT.json').exists() and not (copied/'ATTEMPT.json').exists()


@pytest.mark.parametrize('field,value',[
 ('kind','predictor-common-cotangent-university-exception-v1'),('host','linux10609'),('maximum_new_attempts',2),
 ('approved',1),('full_model_backwards',2),('FP64_reference_builds',2),('component_profiled_iterations',5),
 ('component_warmup_or_timed_iterations',1),('bootstrap_bytes',2*2**20+1),('automatic_retries',1),
 ('paid_compute',True),('cleanup',True),('reset_old_claims',True),('prior_consumed_claim_sha256','old'),('control_sha256','old')])
def test_old_expanded_or_mistyped_authorizations_refuse(field,value):
    a=authorization();a[field]=value
    with pytest.raises(ValueError):op.validate_authorization(a,'fixture-control')


@pytest.mark.parametrize('field',('work_seconds','setup_seconds','capture_worker_seconds','component_worker_seconds',
 'collection_seconds','collection_bytes','raw_bytes','metadata_bytes','allocator_bytes','tensor_file_bytes','write_proof_bytes','mask_indices_file_bytes'))
def test_every_resource_increase_needs_new_scope(field):
    a=authorization();a['resource'][field]+=1
    with pytest.raises(ValueError):op.validate_authorization(a,'fixture-control')


def test_cold_bootstrap_blocks_scientific_imports_and_any_target_write_before_auth(prepared,tmp_path):
    _,c,_=prepared;path=tmp_path/'bootstrap.py';path.write_bytes(client.payload(c))
    script='''import importlib.abc,io,sys,os
from pathlib import Path
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','numpy','timm'):raise AssertionError('Scientific import before authorization')
sys.meta_path.insert(0,Deny())
source=Path(sys.argv[1]).read_bytes()
def forbidden(*a,**k):raise AssertionError('Target write/process/network before authorization')
os.mkdir=os.makedirs=os.system=forbidden
sys.stdin=io.TextIOWrapper(io.BytesIO(b'{}\\n'))
exec(compile(source,'<cold-native-SDPA-bootstrap>','exec'))'''
    p=subprocess.run([sys.executable,'-I','-B','-c',script,str(path)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
    assert p.returncode==1 and b'One new exact fixed-input university authorization required' in p.stderr,p.stderr.decode()[-5000:]
    assert not p.stdout


@pytest.mark.parametrize('name,limit',[
 ('results/capture/boundary/v.pt.partial',16*2**20),('results/components/reference/output.npy.partial',16*2**20),
 ('results/components/0/profiled-dq.npy.partial',16*2**20),('results/capture/boundary/boundary.json.partial',128*2**10),
 ('results/capture/worker.json.partial',256*2**10),('results/components/0/events.jsonl.partial',8*2**20),
 ('results/capture/boundary/attention-scopes.json.partial',64*2**10),('results/components/0/complete.json.partial',16*2**10)])
def test_typed_partial_file_bounds_are_enforced_before_growth_is_accepted(tmp_path,monkeypatch,name,limit):
    monkeypatch.setattr(op.base,'inventory',lambda *a,**k:({name:{'bytes':limit+1}},0))
    with pytest.raises(ValueError,match='typed-file'):op.inventory(tmp_path,running=True)


def test_NPY_arrays_count_as_raw_and_proof_is_only_allowed_during_setup(tmp_path,monkeypatch):
    folder=tmp_path/'scratch';folder.mkdir(mode=0o700);s=tmp_path.stat();identity={'device':s.st_dev,'inode':s.st_ino}
    data={'results/components/reference/output.npy':{'bytes':512*op.MiB+1}}
    monkeypatch.setattr(op.policy,'inventory',lambda *a,**k:(data,0))
    with pytest.raises(ValueError,match='category'):op.storage(folder,identity,parent=tmp_path)
    data['results/components/reference/output.npy']['bytes']=1024
    row=op.storage(folder,identity,parent=tmp_path);assert row['categories']['raw']==1024 and row['categories']['metadata']==0
    data['space-proof-v2.bin.partial']={'bytes':op.RESOURCE['write_proof_bytes']}
    with pytest.raises(ValueError,match='category'):op.storage(folder,identity,parent=tmp_path)
    assert op.storage(folder,identity,parent=tmp_path,setup=True)['categories']['proof']==5*op.GiB//2


def test_complete_original_packet_and_tamper_refusal_before_destination(prepared,tmp_path):
    _,c,_=prepared;source=ROOT/c['archive_path'];op.extract(source,c['archive_manifest'],tmp_path/'code')
    assert len([f for f in (tmp_path/'code').rglob('*') if f.is_file()])==99
    changed=tmp_path/'changed.tar.gz';changed.write_bytes(source.read_bytes()[:-4])
    with pytest.raises(ValueError):op.extract(changed,c['archive_manifest'],tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_actual_packet_prior_values_match_independently_reviewed_original_native_anchors(prepared):
    import torch
    from models.predictor_eot8_capture_v1 import tensor_info
    _,c,_=prepared
    with tarfile.open(ROOT/c['archive_path']) as tar:
        data=tar.extractfile('input/prior-EOT8-derivatives.pt').read()
    values=torch.load(io.BytesIO(data),weights_only=True,map_location='cpu')
    assert set(values)=={'loss','input'}
    for n,t in values.items():
        assert t.dtype==torch.float32 and torch.isfinite(t).all()
        assert tensor_info(t)['sha256']==c['contract']['full_native_anchor_logical_sha256'][n]


def test_collection_preserves_failure_bytes_modes_nsmtime_and_refuses_truncation(tmp_path):
    folder=tmp_path/'source';folder.mkdir(mode=0o700);path=folder/'failed.npy.partial';path.write_bytes(b'actual-failure-bytes');path.chmod(0o600)
    output=io.BytesIO();op.stream(folder,'sha',output);p=tmp_path/'response';p.write_bytes(output.getvalue())
    client.unpack(p,tmp_path/'saved',{},'sha',collection_only=True)
    saved=tmp_path/'saved/failed.npy.partial'
    assert saved.read_bytes()==path.read_bytes() and saved.stat().st_mtime_ns==path.stat().st_mtime_ns and saved.stat().st_mode&0o777==0o600
    bad=tmp_path/'bad';bad.write_bytes(p.read_bytes()[:-512])
    with pytest.raises(ValueError):client.unpack(bad,tmp_path/'incomplete',{},'sha',collection_only=True)


def simulation(prepared,cohort,tmp_path,monkeypatch,*,fail_role=None,corrupt_prior=False,setup_overrun=False):
    _,c,_=prepared;events=[];clock=[0.];sha=CONTROL;st=tmp_path.stat()
    monkeypatch.setattr(op,'time',SimpleNamespace(monotonic=lambda:clock[0],time=time.time))
    monkeypatch.setattr(op,'readiness',lambda c:{'parent':str(tmp_path),'device':st.st_dev,'inode':st.st_ino})
    monkeypatch.setattr(op.base,'gpu_gate',lambda *a,**k:{'CPU_simulation_only':True})
    monkeypatch.setattr(op,'storage',lambda *a,**k:{'allocated_bytes':0,'logical_bytes':0,'categories':{},'ceiling':9*op.GiB//4})
    monkeypatch.setattr(op.signal,'signal',lambda *a,**k:None);monkeypatch.setattr(op.signal,'setitimer',lambda *a,**k:None)
    monkeypatch.setattr(op,'activate_sources',lambda *a:None)
    def receive(incoming,path,size,digest,deadline):events.append(('packet',clock[0],deadline));path.write_bytes(b'local-transport-simulation-only')
    monkeypatch.setattr(op.old,'receive_archive',receive)
    def extract(a,spec,destination):
        destination.mkdir(mode=0o700)
        if setup_overrun:clock[0]=181.
    monkeypatch.setattr(op,'extract',extract)
    monkeypatch.setattr(op.base.primitives,'validate_proof',lambda folder:events.append(('proof-verified',clock[0])))
    verify=op.verify_task;seen=[]
    def check(folder,role,sha,c,**kw):
        seen.append(role)
        if corrupt_prior and seen.count('component-0')==2:raise ValueError('changed-prior-component-local-fixture')
        return verify(folder,role,sha,cohort.context,tiny=True)
    monkeypatch.setattr(op,'verify_task',check)
    def supervise(command,folder,*,role,deadline,**kw):
        events.append((role,clock[0],deadline));clock[0]+=10 if role=='proof' else 60 if role=='capture' else 12
        if role=='proof':return {'exit_status':0}
        if role=='capture':
            shutil.copytree(cohort.root/'results/capture',folder/'results/capture')
            plan=target.component_plan(ROOT,folder/'results/capture',folder/'results/components',control_sha256=sha,local_tiny=True)
            write(folder/'results/components/component-plan.json',plan)
        else:copy_component(cohort,folder,int(role[-1]))
        return {'exit_status':1 if role==fail_role else 0}
    monkeypatch.setattr(op.base.primitives,'supervise',supervise)
    output=io.BytesIO();sources={n:op.INIT if n=='models/__init__.py' else (ROOT/n).read_bytes() for n in op.REMOTE_NAMES}
    before=os.umask(0o077)
    try:exit_status=op._execute(c,authorization(sha),io.BytesIO(),output,sources,sha,0.,1500.,lambda *a:None)
    finally:os.umask(before)
    response=tmp_path/'response';response.write_bytes(output.getvalue());client.unpack(response,tmp_path/'saved',c,sha)
    return exit_status,events,tmp_path/'saved',response


def test_same_connection_simulation_proof_precedes_packet_then_five_serial_workers(prepared,cohort,tmp_path,monkeypatch):
    result,events,saved,response=simulation(prepared,cohort,tmp_path,monkeypatch)
    assert result==0 and [e[0] for e in events]==['proof','proof-verified','packet',*op.ROLES]
    assert events[0][2]==60. and events[2][2]==180.
    for role,started,deadline in [e for e in events if e[0] in op.ROLES]:
        assert deadline<=started+(420 if role=='capture' else 120) and deadline<=1375.
    state=json.loads((saved/'session.json').read_text())
    assert state['complete'] and state['component_comparison_complete'] and len(state['tasks'])==5
    assert not state['scientific_passed'] and not state['candidate_native_numerical_passed'] and not state['research_admitted']
    assert len({t['pid'] for t in state['tasks']})==5


@pytest.mark.parametrize('role',('capture','component-0','component-1','component-3'))
def test_failure_stops_sequence_collects_actual_available_evidence_without_retry(prepared,cohort,tmp_path,monkeypatch,role):
    result,events,saved,_=simulation(prepared,cohort,tmp_path,monkeypatch,fail_role=role)
    ran=[e[0] for e in events if e[0] in op.ROLES];assert result==1 and ran==list(op.ROLES[:op.ROLES.index(role)+1])
    state=json.loads((saved/'session.json').read_text());assert not state['complete'] and len(state['tasks'])==op.ROLES.index(role)
    assert 'no retry' in state['error'] and (saved/'results/capture/worker.json').exists()


def test_changed_prior_stops_before_next_component_and_preserves_stream(prepared,cohort,tmp_path,monkeypatch):
    result,events,saved,_=simulation(prepared,cohort,tmp_path,monkeypatch,corrupt_prior=True)
    assert result==1 and [e[0] for e in events if e[0] in op.ROLES]==['capture','component-0']
    assert 'changed-prior-component' in json.loads((saved/'session.json').read_text())['error']


def test_setup_and_collection_clocks_are_separate_without_worker_reset(prepared,cohort,tmp_path,monkeypatch):
    result,events,saved,_=simulation(prepared,cohort,tmp_path,monkeypatch,setup_overrun=True)
    assert result==1 and not any(e[0] in op.ROLES for e in events)
    state=json.loads((saved/'session.json').read_text());assert state['wall_seconds']==181 and 'setup allowance' in state['error']


def test_client_claims_once_and_collects_a_failure_through_one_mocked_SSH(prepared,cohort,tmp_path,monkeypatch):
    result,events,saved,response=simulation(prepared,cohort,tmp_path,monkeypatch,fail_role='component-1')
    original,c,_=prepared;folder=tmp_path/'mock-client';shutil.copytree(original,folder)
    c=copy.deepcopy(c);c['local_handoff_directory']=str(folder);write(folder/'CONTROL.json',c);sha=op.old.digest(folder/'CONTROL.json')
    write(folder/'AUTHORIZATION.json',authorization(sha))
    # The mock stream's identity changes, not its saved numerical evidence.
    data=response.read_bytes();first,header,archive=data.split(b'\n',2)
    first=json.loads(first);first['control_sha256']=sha;header=json.loads(header);header['control_sha256']=sha
    data=op.old.json_bytes(first)+op.old.json_bytes(header)+archive
    calls=[]
    class Pipe(io.BytesIO):
        def close(self):self.sent_bytes=self.tell();super().close()
    class Fake:
        def __init__(self,cmd,stdin,stdout,stderr):
            calls.append(cmd);assert (folder/'ATTEMPT.json').exists()
            self.stdin=Pipe();self.returncode=1;stdout.write(data);stdout.flush()
            stderr.write(b'local-mock-retained-worker-refusal\n');stderr.flush()
        def poll(self):return self.returncode if self.stdin.closed else None
        def terminate(self):self.stdin.close()
        def wait(self,timeout=None):return self.returncode
    assert client.run(folder,popen=Fake)==1 and len(calls)==1
    assert calls[0][0]=='ssh' and 'StrictHostKeyChecking=yes' in calls[0]
    claim=json.loads((folder/'ATTEMPT.json').read_text());assert claim['consumed'] and claim['scratch_directory']
    assert len(list(folder.glob('attended-*/saved/session.json')))==1
    assert next(folder.glob('attended-*/ssh.stderr.log')).read_bytes()==b'local-mock-retained-worker-refusal\n'
    with pytest.raises(FileExistsError):client.run(folder,popen=Fake)
    assert len(calls)==1


def test_preflight_failure_before_scratch_keeps_stderr_and_consumed_claim(prepared,tmp_path):
    original,c,_=prepared;folder=tmp_path/'preflight-client';shutil.copytree(original,folder)
    c=copy.deepcopy(c);c['local_handoff_directory']=str(folder);write(folder/'CONTROL.json',c);sha=op.old.digest(folder/'CONTROL.json')
    write(folder/'AUTHORIZATION.json',authorization(sha))
    class ClosedInput:
        def write(self,*a):raise BrokenPipeError('mock preflight refused')
    class Refused:
        def __init__(self,cmd,stdin,stdout,stderr):
            assert (folder/'ATTEMPT.json').exists();self.stdin=ClosedInput();self.returncode=1
            stderr.write(b'local-fixture: foreign CUDA process refused before scratch creation\n');stderr.flush()
        def poll(self):return 1
    with pytest.raises(ValueError,match='Missing bound scratch receipt'):client.run(folder,popen=Refused)
    session=next(folder.glob('attended-*'));assert session.is_dir()
    assert b'foreign CUDA process' in (session/'ssh.stderr.log').read_bytes()
    assert (session/'response.bin').read_bytes()==b'' and json.loads((session/'transport.json').read_text())['exit_status']==1
    assert json.loads((folder/'ATTEMPT.json').read_text())['consumed']


def test_frozen_scientific_gates_and_execution_bodies_are_unchanged():
    import ast
    old = ast.parse((ROOT/'models/predictor_sdpa_handoff_operator_v1.py').read_text())
    new = ast.parse((ROOT/'models/predictor_sdpa_handoff_operator_v2.py').read_text().replace('capture_predictor_sdpa_handoff_v2.py','capture_predictor_sdpa_handoff_v1.py'))
    methods = lambda tree: {n.name: ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    previous, current = methods(old), methods(new)
    for name in ('verify_capture','verify_task','readiness','remote','activate_sources','_execute','child','main'):
        assert current[name] == previous[name], name
    assert op.inventory is op.policy.inventory and op.storage is op.policy.storage and op.stream is op.policy.stream
    assert op.RESOURCE == {**op.policy.RESOURCE,'mask_indices_file_bytes':262144}


@pytest.mark.parametrize('field,value', [('kind','native-fixed-SDPA-university-exception-v1'),
    ('prior_consumed_claim_sha256','9066c0afb97cb9e92f02e047806124aab73fff36cc9cdb46b4c31e28d167e94c'),
    ('prior_metadata_probe_claim_sha256','0'*64)])
def test_consumed_predecessor_grants_cannot_authorize_this_replacement(field,value):
    auth=authorization();auth[field]=value
    with pytest.raises(ValueError):op.validate_authorization(auth,'fixture-control')


def test_repair_evidence_tamper_is_refused_before_any_claim(prepared,monkeypatch):
    folder,c,_=prepared;victim=ROOT/c['contract']['repair_bindings']['0']['path'];digest=client.old.digest
    monkeypatch.setattr(client.old,'digest',lambda path:'0'*64 if Path(path)==victim else digest(path))
    with pytest.raises(ValueError,match='Frozen prior binding changed'):
        client.check_local(folder)
    assert not (folder/'ATTEMPT.json').exists()


def test_missing_collection_header_has_primary_diagnostic_and_preserves_response(tmp_path):
    response=tmp_path/'response'
    data=op.old.json_bytes({'kind':op.KIND,'phase':'scratch-created','control_sha256':'fixture','scratch_directory':'local-fixture'})
    response.write_bytes(data)
    with pytest.raises(ValueError,match='Collection header missing'):
        client.unpack(response,tmp_path/'saved',{},'fixture')
    assert response.read_bytes()==data


def test_real_v2_storage_gate_and_collection_accept_full_masks_on_capture_refusal(prepared,tmp_path,monkeypatch):
    _,c,_=prepared;sha=CONTROL;clock=[0.];parent=tmp_path.stat();events=[]
    maskpath=ROOT/'runs/pretrained/diagnostic/predictor-sdpa-storage-v2-20261107/pytest-final/test_full_original_EOT8_masks_0/results/capture/mask-indices.json'
    maskbytes=maskpath.read_bytes();assert len(maskbytes)==146562
    monkeypatch.setattr(op,'time',SimpleNamespace(monotonic=lambda:clock[0],time=time.time))
    monkeypatch.setattr(op,'readiness',lambda c:{'parent':str(tmp_path),'device':parent.st_dev,'inode':parent.st_ino})
    monkeypatch.setattr(op.base,'gpu_gate',lambda *a,**k:{'mock_GPU_not_a_qualification':True})
    real_storage=op.policy.storage
    monkeypatch.setattr(op,'storage',lambda folder,identity,setup=False:real_storage(folder,identity,setup=setup,parent=tmp_path))
    monkeypatch.setattr(op.signal,'signal',lambda *a,**k:None);monkeypatch.setattr(op.signal,'setitimer',lambda *a,**k:None)
    monkeypatch.setattr(op,'activate_sources',lambda *a:None)
    monkeypatch.setattr(op.base.primitives,'validate_proof',lambda *a:None)
    def receive(incoming,path,*a):path.write_bytes((ROOT/c['archive_path']).read_bytes())
    monkeypatch.setattr(op.old,'receive_archive',receive)
    def supervise(command,folder,*,role,gate,**kwargs):
        events.append(role);clock[0]+=5
        gate()
        if role=='proof':return {'exit_status':0}
        assert role=='capture'
        path=folder/op.policy.MASK_PATH;path.parent.mkdir(mode=0o700)
        partial=path.with_name(path.name+'.partial');partial.write_bytes(maskbytes);partial.chmod(0o600)
        partial_row=gate();assert partial_row['categories']['metadata']>=len(maskbytes)
        os.link(partial,path);gate();partial.unlink();gate()
        write(path.parent/'worker.json',{'complete':False,'mock_worker':True})
        return {'exit_status':1}
    monkeypatch.setattr(op.base.primitives,'supervise',supervise)
    output=io.BytesIO();sources={n:op.INIT if n=='models/__init__.py' else (ROOT/n).read_bytes() for n in op.REMOTE_NAMES}
    before=os.umask(0o077)
    try:status=op._execute(c,authorization(sha),io.BytesIO(),output,sources,sha,0.,1500.,lambda *a:None)
    finally:os.umask(before)
    response=tmp_path/'response';response.write_bytes(output.getvalue())
    client.unpack(response,tmp_path/'saved',c,sha)
    assert status==1 and events==['proof','capture']
    assert (tmp_path/'saved'/op.policy.MASK_PATH).read_bytes()==maskbytes
    session=json.loads((tmp_path/'saved/session.json').read_text())
    assert not session['complete'] and 'no retry' in session['error']
    assert not (tmp_path/'saved/results/components/0').exists()
