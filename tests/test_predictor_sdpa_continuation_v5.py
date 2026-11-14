"""Frozen-input continuation: cold guards, actual packaged CPU workers, collection."""
import copy
import hashlib
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
from models import predictor_sdpa_continuation_operator_v5 as op
from models import predictor_sdpa_continuation_client_v5 as client
from models import predictor_sdpa_component_v6 as core
from scripts.prepare_predictor_sdpa_continuation_v5 import prepare

ROOT=Path(__file__).resolve().parents[1]
CPU_REFERENCE=ROOT/'runs/pretrained/diagnostic/predictor-sdpa-observer-v1-20261106/pytest-third/native-sdpa-observer0/creator-0'


def write(path,value):
    path.write_bytes(op.old.json_bytes(value));path.chmod(0o600)

def authorization(sha='fixture'):
    return copy.deepcopy(op.authorization_template(sha,'Local simulation only; no university resource grant.'))

@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    folder=tmp_path_factory.mktemp('component-only-prepared')/'handoff'
    c,info=prepare(ROOT,folder);return folder,c,info

@pytest.fixture(scope='module')
def cohort(tmp_path_factory):
    folder=tmp_path_factory.mktemp('components')/'cohort'
    result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/check_predictor_sdpa_components_v6.py'),'--output',str(folder)],
        cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (folder.parent/'cohort.log').write_bytes(result.stdout)
    assert result.returncode==0,result.stdout.decode()[-6000:]
    return folder


def test_exact_capsule_contract_keeps_original_bounds_and_unarmed_state(prepared):
    folder,c,info=prepared;actual,sha=client.check_local(folder)
    assert actual==c and sha==info['control_sha256']
    assert info['capsule_bytes']==36042589 and info['capsule_members']==118
    assert info['capsule_uncompressed_bytes']==48627833
    assert not (folder/'AUTHORIZATION.json').exists() and not (folder/'ATTEMPT.json').exists()
    assert op.RESOURCE['static_reserved_seconds']==180+4*120+120<=900
    assert op.RESOURCE['capture_worker_seconds']==0 and c['contract']['full_model_backwards']==0
    assert c['contract']['new_native_captures']==0 and c['contract']['candidate_failed_input_entries']==3724
    assert c['contract']['numerical_bounds']==core.BOUNDS and c['contract']['bookkeeping']==core.BOOK
    assert info['overlay_files']==len(op.REMOTE_NAMES) and info['stdin_bootstrap_bytes']<2*op.MiB
    op.validate_authorization(authorization(sha),sha)


def test_unarmed_copy_retrieval_and_consumed_claim_cannot_launch(prepared,tmp_path):
    folder,c,_=prepared
    def forbidden(*a,**k):raise AssertionError('No SSH expected')
    with pytest.raises(ValueError,match='Unarmed'):client.run(folder,popen=forbidden)
    with pytest.raises(ValueError,match='Separate retrieval'):client.run(folder,download_only=True,popen=forbidden)
    copied=tmp_path/'copy';shutil.copytree(folder,copied)
    with pytest.raises(ValueError,match='Exact local handoff'):client.check_local(copied)
    assert not (folder/'ATTEMPT.json').exists() and not (copied/'ATTEMPT.json').exists()


@pytest.mark.parametrize('field,value',[
 ('kind','native-fixed-SDPA-university-exception-v2'),('host','linux10609'),('maximum_new_attempts',2),
 ('approved',1),('full_model_backwards',1),('new_native_captures',1),('FP64_reference_builds',2),
 ('component_profiled_iterations',5),('component_warmup_or_timed_iterations',1),('automatic_retries',1),
 ('paid_compute',True),('cleanup',True),('reset_old_claims',True),('prior_consumed_claim_sha256','old')])
def test_old_expanded_or_mistyped_grant_refuses(field,value):
    auth=authorization();auth[field]=value
    with pytest.raises(ValueError):op.validate_authorization(auth,'fixture')


@pytest.mark.parametrize('field',('work_seconds','setup_seconds','component_worker_seconds','write_proof_bytes',
 'collection_seconds','collection_bytes','allocator_bytes','raw_bytes','metadata_bytes','mask_indices_file_bytes'))
def test_resource_expansion_requires_separate_scope(field):
    auth=authorization();auth['resource'][field]+=1
    with pytest.raises(ValueError):op.validate_authorization(auth,'fixture')


def test_actual_cold_bootstrap_refuses_before_science_write_or_network(prepared,tmp_path):
    _,c,_=prepared;path=tmp_path/'bootstrap.py';path.write_bytes(client.payload(c))
    program='''import importlib.abc,io,sys,os,socket,subprocess
from pathlib import Path
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','numpy','timm'):raise AssertionError('Scientific import before authorization')
sys.meta_path.insert(0,Deny())
source=Path(sys.argv[1]).read_bytes()
def forbidden(*a,**k):raise AssertionError('Target write/process/network before authorization')
os.mkdir=os.makedirs=os.system=subprocess.Popen=socket.socket=forbidden
sys.stdin=io.TextIOWrapper(io.BytesIO(b'{}\\n'))
exec(compile(source,'<cold-component-only-bootstrap>','exec'))'''
    result=subprocess.run([sys.executable,'-I','-B','-c',program,str(path)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
    assert result.returncode==1 and b'One new exact component-only university authorization required' in result.stderr,result.stderr.decode()[-4000:]
    assert not result.stdout


def unpack_fixture(prepared,tmp_path):
    _,c,_=prepared;scratch=tmp_path/'scratch';scratch.mkdir(mode=0o700)
    previous=os.umask(0o077)
    try:op.extract(ROOT/c['archive_path'],c['archive_manifest'],scratch)
    finally:os.umask(previous)
    return c,scratch


def test_actual_capsule_extracts_without_pixels_and_reverifies_native_parent(prepared,tmp_path):
    c,scratch=unpack_fixture(prepared,tmp_path);receipt=op.verify_frozen_capture(scratch,c)
    assert receipt['worker_sha256']==op.NATIVE_WORKER_SHA and receipt['boundary_sha256']==op.BOUNDARY_SHA
    assert not (scratch/'code/input/training-clip.pt').exists()
    assert not (scratch/'code/input/prior-EOT8-derivatives.pt').exists()
    assert not (scratch/'results/components').exists()
    with pytest.raises(ValueError,match='collision'):op.extract(ROOT/c['archive_path'],c['archive_manifest'],scratch)


def test_changed_native_member_or_ns_time_refuses(prepared,tmp_path):
    c,scratch=unpack_fixture(prepared,tmp_path)
    victim=scratch/'results/capture/mask-indices.json';st=victim.stat()
    os.utime(victim,ns=(st.st_atime_ns,st.st_mtime_ns+1))
    with pytest.raises(ValueError,match='Frozen input changed'):op.verify_frozen_capture(scratch,c)


def test_capsule_receive_partials_are_inputs_and_other_limits_stay_frozen(tmp_path,monkeypatch):
    folder=tmp_path/'scratch';folder.mkdir(mode=0o700);st=tmp_path.stat();identity={'device':st.st_dev,'inode':st.st_ino}
    data={'native-input-capsule.tar.gz.partial':{'bytes':36_042_589}}
    monkeypatch.setattr(op.policy,'inventory',lambda *a,**k:(data,0))
    row=op.storage(folder,identity,parent=tmp_path)
    assert row['categories']['input']==36_042_589 and row['categories']['metadata']==0
    assert op.policy.typed_file_limit('native-input-capsule.tar.gz.partial')==128*op.MiB
    data['results/components/reference/output.npy']={'bytes':512*op.MiB+1}
    with pytest.raises(ValueError,match='category'):op.storage(folder,identity,parent=tmp_path)
    data.clear();data['space-proof-v2.bin.partial']={'bytes':op.RESOURCE['write_proof_bytes']}
    with pytest.raises(ValueError,match='category'):op.storage(folder,identity,parent=tmp_path)
    assert op.storage(folder,identity,setup=True,parent=tmp_path)['categories']['proof']==op.RESOURCE['write_proof_bytes']
    for n in ('results/capture/mask-indices.json','results/capture/boundary/attention-scopes.json',
              'results/components/0/events.jsonl','results/components/0/profiled-dq.npy.partial'):
        assert op.policy.typed_file_limit(n)==op.prior.policy.typed_file_limit(n)


def copy_component(cohort,folder,index):
    destination=folder/'results/components';destination.mkdir(mode=0o700,parents=True,exist_ok=True)
    p=json.loads((cohort/'component-plan.json').read_text());p['reference_dir']=str(destination/'reference')
    write(destination/'component-plan.json',p);ps=op.old.digest(destination/'component-plan.json')
    if index==0:shutil.copytree(cohort/'reference',destination/'reference')
    shutil.copytree(cohort/str(index),destination/str(index))
    def resign(path,name,value):
        write(path/name,value);write(path/'complete.json',{'kind':value['kind'],'plan_sha256':ps,
           'header':{'file':name,'bytes':(path/name).stat().st_size,'sha256':op.old.digest(path/name)},'complete':True})
    ref=json.loads((destination/'reference/reference.json').read_text());ref['plan_sha256']=ps;resign(destination/'reference','reference.json',ref)
    for i in range(index+1):
        path=destination/str(i);w=json.loads((path/'worker.json').read_text());w['plan_sha256']=ps
        w['reference_manifest_sha256']=op.old.digest(destination/'reference/reference.json')
        w['reference_marker_sha256']=op.old.digest(destination/'reference/complete.json')
        w['previous_receipts']=[{'index':j,'worker_sha256':op.old.digest(destination/str(j)/'worker.json'),
            'complete_marker_sha256':op.old.digest(destination/str(j)/'complete.json')} for j in range(i)]
        if i>=2:w['fresh_replay_reference_sha256']=op.old.digest(destination/str(i-2)/'worker.json')
        resign(path,'worker.json',w)


def simulation(prepared,cohort,tmp_path,monkeypatch,*,fail_role=None,setup_overrun=False,corrupt_prior=False,proof_failure=False,expired_after_first=False):
    _,c,_=prepared;sha='fixture';clock=[0.];events=[];parent=tmp_path.stat()
    monkeypatch.setattr(op,'time',SimpleNamespace(monotonic=lambda:clock[0],time=time.time))
    monkeypatch.setattr(op,'readiness',lambda c:{'parent':str(tmp_path),'device':parent.st_dev,'inode':parent.st_ino})
    monkeypatch.setattr(op.base,'gpu_gate',lambda *a,**k:{'simulated_GPU':True})
    monkeypatch.setattr(op.signal,'signal',lambda *a,**k:None);monkeypatch.setattr(op.signal,'setitimer',lambda *a,**k:None)
    real_storage=op.storage
    monkeypatch.setattr(op,'storage',lambda f,i,setup=False:real_storage(f,i,setup=setup,parent=tmp_path))
    monkeypatch.setattr(op,'activate_sources',lambda *a:None)
    def receive(incoming,path,*a):
        events.append(('capsule',clock[0],a[-1]));path.write_bytes((ROOT/c['archive_path']).read_bytes())
    monkeypatch.setattr(op.old,'receive_archive',receive)
    real_extract=op.extract
    def extract(*a):
        real_extract(*a)
        if setup_overrun:clock[0]=181.
    monkeypatch.setattr(op,'extract',extract)
    monkeypatch.setattr(op.base.primitives,'validate_proof',lambda f:events.append(('proof-verified',clock[0])))
    # The full frozen target capsule is really extracted/verified. Only later
    # scientific tasks use the small saved CPU inputs and prior fresh CPU returns.
    make=op.component_plan
    monkeypatch.setattr(op,'component_plan',lambda overlay,capture,components:make(ROOT,CPU_REFERENCE,components,local_tiny=True,control_sha256='d'*64))
    verifier=op.verify_task;seen=[]
    def verify(folder,role,sha,c,**kw):
        seen.append(role)
        if corrupt_prior and seen.count('component-0')==2:raise ValueError('changed-prior-local-fixture')
        return verifier(folder,role,sha,c,tiny=True)
    monkeypatch.setattr(op,'verify_task',verify)
    def supervise(command,folder,*,role,deadline,gate,**kwargs):
        events.append((role,clock[0],deadline));clock[0]+=5 if role=='proof' else 12
        gate()
        if role=='proof':return {'exit_status':1 if proof_failure else 0}
        copy_component(cohort,folder,int(role[-1]))
        if expired_after_first and role=='component-0':clock[0]=800.
        return {'exit_status':1 if role==fail_role else 0}
    monkeypatch.setattr(op.base.primitives,'supervise',supervise)
    sources={n:op.INIT if n=='models/__init__.py' else (ROOT/n).read_bytes() for n in op.REMOTE_NAMES}
    output=io.BytesIO();mask=os.umask(0o077)
    try:status=op._execute(c,authorization(sha),io.BytesIO(),output,sources,sha,0.,900.,lambda *a:None)
    finally:os.umask(mask)
    response=tmp_path/'response';response.write_bytes(output.getvalue());client.unpack(response,tmp_path/'saved',c,sha)
    return status,events,tmp_path/'saved',response


def test_proof_capsule_four_serial_components_and_complete_same_connection(prepared,cohort,tmp_path,monkeypatch):
    status,events,saved,response=simulation(prepared,cohort,tmp_path,monkeypatch)
    assert status==0 and [e[0] for e in events]==['proof','proof-verified','capsule',*op.ROLES]
    assert events[0][2]==60 and events[2][2]==180
    for role,start,end in [e for e in events if e[0] in op.ROLES]:assert end<=start+120 and end<=775
    state=json.loads((saved/'session.json').read_text())
    assert state['complete'] and state['component_comparison_complete'] and len(state['tasks'])==4
    assert state['full_model_backwards']==state['new_native_captures']==0 and not state['research_admitted']
    assert len({t['pid'] for t in state['tasks']})==4


@pytest.mark.parametrize('role',op.ROLES)
def test_failed_component_stops_and_collects_available_evidence_without_retry(prepared,cohort,tmp_path,monkeypatch,role):
    status,events,saved,_=simulation(prepared,cohort,tmp_path,monkeypatch,fail_role=role)
    assert status==1 and [e[0] for e in events if e[0] in op.ROLES]==list(op.ROLES[:op.ROLES.index(role)+1])
    state=json.loads((saved/'session.json').read_text());assert not state['complete'] and len(state['tasks'])==op.ROLES.index(role)
    assert 'no retry' in state['error'] and (saved/'results/capture/worker.json').exists()


def test_changed_predecessor_stops_before_next_worker(prepared,cohort,tmp_path,monkeypatch):
    status,events,saved,_=simulation(prepared,cohort,tmp_path,monkeypatch,corrupt_prior=True)
    assert status==1 and [e[0] for e in events if e[0] in op.ROLES]==['component-0']
    assert 'changed-prior' in json.loads((saved/'session.json').read_text())['error']


def test_setup_deadline_is_not_reset_before_workers(prepared,cohort,tmp_path,monkeypatch):
    status,events,saved,_=simulation(prepared,cohort,tmp_path,monkeypatch,setup_overrun=True)
    assert status==1 and not any(e[0] in op.ROLES for e in events)
    assert 'Setup allowance' in json.loads((saved/'session.json').read_text())['error']


def test_client_single_mock_connection_claims_once_and_preserves_failure(prepared,cohort,tmp_path,monkeypatch):
    _,_,_,response=simulation(prepared,cohort,tmp_path,monkeypatch,fail_role='component-1')
    original,c,_=prepared;folder=tmp_path/'client';shutil.copytree(original,folder)
    c=copy.deepcopy(c);c['local_handoff_directory']=str(folder);write(folder/'CONTROL.json',c);sha=op.old.digest(folder/'CONTROL.json')
    write(folder/'AUTHORIZATION.json',authorization(sha));data=response.read_bytes();first,header,archive=data.split(b'\n',2)
    first=json.loads(first);header=json.loads(header);first['control_sha256']=header['control_sha256']=sha
    data=op.old.json_bytes(first)+op.old.json_bytes(header)+archive;calls=[]
    class Pipe(io.BytesIO):
        def close(self):self.extent=self.tell();super().close()
    class Fake:
        def __init__(self,cmd,stdin,stdout,stderr):
            calls.append(cmd);assert (folder/'ATTEMPT.json').exists();self.stdin=Pipe();self.returncode=1
            stdout.write(data);stdout.flush();stderr.write(b'simulated-worker-refusal\n');stderr.flush()
        def poll(self):return self.returncode if self.stdin.closed else None
        def terminate(self):self.stdin.close()
        def wait(self,timeout=None):return self.returncode
    assert client.run(folder,popen=Fake)==1 and len(calls)==1
    with pytest.raises(FileExistsError):client.run(folder,popen=Fake)
    assert len(calls)==1 and calls[0][0]=='ssh' and 'StrictHostKeyChecking=yes' in calls[0]
    assert json.loads((folder/'ATTEMPT.json').read_text())['consumed']
    assert next(folder.glob('attended-*/ssh.stderr.log')).read_bytes()==b'simulated-worker-refusal\n'


def test_missing_or_truncated_collection_preserves_response(tmp_path):
    response=tmp_path/'response';data=op.old.json_bytes({'kind':op.KIND,'phase':'scratch-created','control_sha256':'fixture'})
    response.write_bytes(data)
    with pytest.raises(ValueError,match='Collection header missing'):client.unpack(response,tmp_path/'saved',{},'fixture')
    assert response.read_bytes()==data


def test_actual_payload_import_paths_run_four_fresh_small_CPU_components(prepared,tmp_path):
    c,scratch=unpack_fixture(prepared,tmp_path);overlay=scratch/'overlay';overlay.mkdir(mode=0o700)
    for name in op.REMOTE_NAMES:
        path=overlay/name;path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        path.write_bytes(op.INIT if name=='models/__init__.py' else (ROOT/name).read_bytes());path.chmod(0o600)
    out=tmp_path/'packaged-CPU';out.mkdir(mode=0o700)
    program='''import os,sys,json,hashlib,subprocess,time
from pathlib import Path
os.umask(0o077)
overlay=Path(sys.argv[1]);code=Path(sys.argv[2]);reference=Path(sys.argv[3]);out=Path(sys.argv[4])
sys.path[:0]=[str(overlay),str(code)]
from models import predictor_sdpa_component_v6 as core
from models import predictor_sdpa_component_records_v1 as rec
from models.predictor_sdpa_continuation_operator_v5 import component_plan
p=component_plan(overlay,reference,out,local_tiny=True,control_sha256='d'*64)
path=out/'component-plan.json';path.write_bytes(rec.encoded(p));ps=rec.sha(path)
for i in range(4):
 cmd=[sys.executable,'-I','-B',str(overlay/'scripts/check_predictor_sdpa_components_v6.py'),'--output',str(out/str(i)),
  '--worker','--plan',str(path),'--plan-sha256',ps,'--index',str(i),'--deadline-unix',str(time.time()+120)]
 with (out/(str(i)+'.log')).open('xb') as log:
  # Match child()'s explicit code search path before entering the unchanged CLI.
  wrapper="import runpy,sys;sys.path.insert(1,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name='__main__')"
  result=subprocess.run([sys.executable,'-I','-B','-c',wrapper,str(code),*cmd[3:]],cwd=out,stdout=log,stderr=subprocess.STDOUT,timeout=120)
 if result.returncode:raise RuntimeError((out/(str(i)+'.log')).read_text()[-4000:])
_,_,_,identity=core.capture(p);workers=[core.load_worker(out/str(i),p,ps,identity,i) for i in range(4)]
assert len({w['pid'] for w in workers})==4
(out/'CPU-summary.json').write_bytes(rec.encoded({'complete':True,'CPU_only':True,'CUDA_qualified':False,
 'full_model_backwards':0,'new_native_captures':0,'fresh_processes':4,'canonical_FP64_builds':1,
 'overlay_source_verified':True,'trace_format':p['trace_format'],'trace_publication':p['trace_publication'],'metadata_policy':p['metadata_policy'],'raw_record_counts':[w['trace']['records'] for w in workers],'workers':[w['pid'] for w in workers]}))'''
    result=subprocess.run([sys.executable,'-I','-B','-c',program,str(overlay),str(scratch/'code'),str(CPU_REFERENCE),str(out)],
        cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (tmp_path/'packaged-CPU.log').write_bytes(result.stdout)
    assert result.returncode==0,result.stdout.decode()[-6000:]
    summary=json.loads((out/'CPU-summary.json').read_text())
    assert summary['complete'] and summary['CPU_only'] and not summary['CUDA_qualified']
    assert summary['trace_format']=='fixed-SDPA-raw-Kineto-correlation-v1'
    assert summary['raw_record_counts']==[75,249,75,249]
    assert summary['trace_publication']=='before-semantic-validation'
    assert summary['metadata_policy']==op.METADATA_POLICY


def test_old_frozen_math_and_startup_sources_are_pinned(prepared):
    _,c,_=prepared
    report=json.loads((ROOT/c['contract']['local_startup_review']['path']).read_text())
    for name,item in report['source'].items():
        assert op.old.digest(ROOT/name)==item['sha256']
    assert set(core.SOURCE_NAMES)==set(op.COMPONENT_SOURCES)


def test_native_verification_requires_no_scientific_import(prepared,tmp_path):
    c,scratch=unpack_fixture(prepared,tmp_path)
    source="""import importlib.abc,sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','numpy','timm'):raise AssertionError('Science while validating frozen inputs')
sys.meta_path.insert(0,Deny())
from models.predictor_sdpa_continuation_operator_v5 import verify_frozen_capture
control=json.loads(Path(sys.argv[2]).read_text());verify_frozen_capture(Path(sys.argv[3]),control)
"""
    folder=prepared[0]
    result=subprocess.run([sys.executable,'-I','-B','-c',source,str(ROOT),str(folder/'CONTROL.json'),str(scratch)],
        cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
    assert result.returncode==0,result.stderr.decode()[-4000:]


@pytest.mark.parametrize('fault',('hash','mode','mtime','size','duplicate','truncated'))
def test_collection_checks_actual_stream_and_preserves_on_refusal(tmp_path,fault):
    folder=tmp_path/'scratch';folder.mkdir(mode=0o700);file=folder/'session.json';file.write_bytes(b'{}\n');file.chmod(0o600)
    output=io.BytesIO();op.stream(folder,'fixture',output);data=output.getvalue();header,archive=data.split(b'\n',1);header=json.loads(header)
    if fault=='hash':header['files']['session.json']['sha256']='0'*64
    if fault=='mode':header['files']['session.json']['mode']=0o400
    if fault=='mtime':header['files']['session.json']['mtime_ns']+=10**9
    if fault=='size':header['logical_bytes']+=1
    if fault=='duplicate':header['files']['other.json']=dict(header['files']['session.json']);header['logical_bytes']*=2
    if fault=='truncated':archive=archive[:-512]
    first=op.old.json_bytes({'kind':op.KIND,'control_sha256':'fixture','phase':'scratch-created'})
    data=first+op.old.json_bytes(header)+archive;response=tmp_path/'response';response.write_bytes(data)
    with pytest.raises(ValueError):client.unpack(response,tmp_path/'saved',{},'fixture')
    assert response.read_bytes()==data


def test_real_target_child_adds_frozen_code_path_before_target_import():
    source=(ROOT/'models/predictor_sdpa_continuation_operator_v5.py').read_text().split('def child(',1)[1]
    assert source.index("sys.path.insert(1,str(scratch/'code'))")<source.index('from models import predictor_sdpa_component_target_v6 as target')
    assert 'run_capture(' not in source and 'run_component(' in source


def test_frozen_association_and_consumed_parent_are_bound(prepared):
    _,c,_=prepared
    contract=c['contract'];item=contract['local_association_review']
    report=json.loads((ROOT/item['path']).read_text())
    assert op.old.digest(ROOT/item['path'])==item['sha256'] and report['passed']
    assert contract['association_proof_kind']=='one-fixed-SDPA-correlated-backend-with-overhead-v1'
    assert contract['component_plan_kind']==core.PLAN_KIND
    assert op.PRIOR_CLAIM_SHA=='3829d1b634b32053f73560610f2e78d5a835e7193eaea885b55211e6c2536c32'
    assert op.PRIOR_CLAIM_SHA!=contract['origin_native_claim']['sha256']
    assert contract['old_incomplete_worker_remains_incomplete'] and not contract['backend_adoption']
    for name,item in report['new_source_bindings'].items():assert op.old.digest(ROOT/name)==item['sha256']


@pytest.mark.parametrize('version',(1,2,3,4))
def test_old_valid_authorization_cannot_authorize_new_handoff(version):
    import importlib
    consumed=importlib.import_module('models.predictor_sdpa_continuation_operator_v'+str(version))
    with pytest.raises(ValueError):op.validate_authorization(consumed.authorization_template('fixture','Old consumed grant.'),'fixture')


def test_actual_raw_envelopes_fit_existing_trace_worker_and_collection_limits(cohort):
    from models import predictor_sdpa_launch_evidence_v1 as events
    for i in range(4):
        path=cohort/str(i);w=json.loads((path/'worker.json').read_text())
        raw=core.load_profile(path,w['trace'],w['runtime'])
        assert raw['kind']==events.KIND and raw['torch_version']==w['runtime']['torch']
        assert len(raw['records'])==w['trace']['records']
        assert (path/'events.jsonl').stat().st_size<=op.policy.typed_file_limit('results/components/0/events.jsonl')==8*op.MiB
        assert (path/'worker.json').stat().st_size<=256*1024
        assert w['backend_evidence']['metadata_only'] and not w['backend_evidence']['actual_CUDA_execution_admitted']




def test_new_worker_and_raw_correlation_reviews_and_consumed_claim_are_bound(prepared):
    _,c,_=prepared;contract=c['contract']
    for key in ('local_worker_review','local_launch_evidence_review'):
        item=contract[key];assert op.old.digest(ROOT/item['path'])==item['sha256']
        report=json.loads((ROOT/item['path']).read_text());assert report['passed']
        for name,value in report.get('source',{}).items():assert op.old.digest(ROOT/name)==value['sha256']
        for name,want in report.get('tested_source_sha256',{}).items():assert op.old.digest(ROOT/name)==want
    assert contract['trace_format']=='fixed-SDPA-raw-Kineto-correlation-v1'
    assert contract['trace_encoding']=='single-JSON-envelope' and contract['recorded_host_correlation_links_available']
    assert not contract['per_kernel_global_unique_correlation_id_assumed']
    assert contract['resource_receipt_kind']=='native-fixed-SDPA-component-resource-v6'
    assert contract['parent_consumed_claim']['sha256']!=contract['preceding_component_claim']['sha256']
    assert op.PRIOR_CLAIM_SHA=='3829d1b634b32053f73560610f2e78d5a835e7193eaea885b55211e6c2536c32'


def test_previous_consumed_control_cannot_become_the_new_handoff():
    prior=ROOT/'runs/pretrained/diagnostic/predictor-sdpa-continuation-v4-20261113/linux10603-final/CONTROL.json'
    with pytest.raises(ValueError):op.validate_control(json.loads(prior.read_text()))


@pytest.mark.parametrize('case',('idle','owned-child','busy-unowned','busy-other-child','wrong-uuid','wrong-driver','initial-memory','two-GPUs'))
def test_real_GPU_gate_uses_expected_identity_and_allows_only_the_owned_child(prepared,monkeypatch,case):
    c=prepared[1];name=c['expected_gpu'];uuid=c['expected_gpu_uuid'];driver=c['expected_driver']
    allowed=123 if case in ('owned-child','busy-other-child') else None
    active='123, owned-python, 5400' if case=='owned-child' else '999, /usr/NX/bin/nxnode.bin, 183' if case.startswith('busy') else ''
    used=5451 if allowed else 513 if case=='initial-memory' else 94
    if case=='wrong-uuid':uuid='wrong'
    if case=='wrong-driver':driver='wrong'
    row=f'{name}, {uuid}, 8192, {used}, {driver}'
    if case=='two-GPUs':row+='\n'+row
    commands=[]
    def query(command,**kwargs):
        commands.append(command);assert command[0]=='nvidia-smi' and kwargs=={'text':True,'timeout':10}
        return active if any('--query-compute-apps=' in x for x in command) else row
    monkeypatch.setattr(op.base.subprocess,'check_output',query)
    if case in ('idle','owned-child'):
        observed=op.base.gpu_gate(c,allowed);assert observed['allowed_compute_pid']==allowed and observed['sampling_is_not_continuous_exclusivity']
    else:
        with pytest.raises(ValueError):op.base.gpu_gate(c,allowed)
    assert commands


@pytest.mark.parametrize('case',('wrong-host','busy'))
def test_live_guard_refusal_precedes_scratch_and_proof(prepared,tmp_path,monkeypatch,case):
    c=prepared[1];out=io.BytesIO()
    def no_scratch(*a,**k):raise AssertionError('No scratch or proof after failed live guard')
    monkeypatch.setattr(op.tempfile,'mkdtemp',no_scratch)
    if case=='wrong-host':
        platform=op.readiness.__globals__['platform'];monkeypatch.setattr(platform,'node',lambda:'not-the-approved-host')
    else:
        monkeypatch.setattr(op,'readiness',lambda c:{'parent':str(tmp_path)})
        def query(command,**kwargs):
            if any('--query-compute-apps=' in x for x in command):return '999, /usr/NX/bin/nxnode.bin, 183'
            return f"{c['expected_gpu']}, {c['expected_gpu_uuid']}, 8192, 94, {c['expected_driver']}"
        monkeypatch.setattr(op.base.subprocess,'check_output',query)
    with pytest.raises(ValueError):op._execute(c,authorization(),io.BytesIO(),out,{},'fixture',0.,900.,lambda *a:None)
    assert out.getvalue()==b''


def test_storage_proof_failure_stops_before_capsule_receive(prepared,cohort,tmp_path,monkeypatch):
    status,events,saved,_=simulation(prepared,cohort,tmp_path,monkeypatch,proof_failure=True)
    assert status==1 and [e[0] for e in events]==['proof']
    state=json.loads((saved/'session.json').read_text());assert not state['complete'] and state['tasks']==[]
    assert 'Full storage proof failed' in state['error'] and not (saved/'native-input-capsule.tar.gz').exists()


def test_consumed_work_allowance_is_not_reset_before_next_component(prepared,cohort,tmp_path,monkeypatch):
    status,events,saved,_=simulation(prepared,cohort,tmp_path,monkeypatch,expired_after_first=True)
    assert status==1 and [e[0] for e in events if e[0] in op.ROLES]==['component-0']
    state=json.loads((saved/'session.json').read_text());assert len(state['tasks'])==1 and state['wall_seconds']==800.
    assert 'allowance exhausted' in state['error'] and not (saved/'worker-plan-component-1.json').exists()


@pytest.fixture
def shallow_resource(prepared,cohort,tmp_path):
    # Explicitly synthetic metadata boundary; the deep scientific checker MUST refuse it.
    c,scratch=unpack_fixture(prepared,tmp_path);copy_component(cohort,scratch,0)
    components=scratch/'results/components';p=op.component_plan(ROOT,scratch/'results/capture',components)
    write(components/'component-plan.json',p);ps=op.old.digest(components/'component-plan.json')
    folder=components/'0';w=json.loads((folder/'worker.json').read_text());w['plan_sha256']=ps
    prior=ROOT/'runs/pretrained/diagnostic/predictor-sdpa-integration-v3-20261108/linux10603-final/attended-1o9_nbog/saved/results/components/0/worker.json'
    startup=copy.deepcopy(json.loads(prior.read_text())['startup']);startup['pid']=w['pid'];w['startup']=startup
    write(folder/'worker.json',w)
    write(folder/'complete.json',{'kind':w['kind'],'plan_sha256':ps,'complete':True,
        'header':{'file':'worker.json','bytes':(folder/'worker.json').stat().st_size,'sha256':op.old.digest(folder/'worker.json')}})
    write(folder/'component-resource.json',{'kind':'native-fixed-SDPA-component-resource-v6','plan_sha256':ps,
        'worker_sha256':op.old.digest(folder/'worker.json'),'pid':w['pid'],'index':0,'peak_allocator_bytes':1024,
        'startup':startup,'post_entry_wall_seconds':0.1,'research_admitted':False,'performance_qualified':False})
    write(tmp_path/'control.json',c)
    return c,scratch,tmp_path/'control.json'


def test_preimport_resource_read_is_science_free_and_deep_domain_check_still_refuses(shallow_resource,tmp_path):
    c,scratch,control=shallow_resource
    code='''import importlib.abc,sys,json,socket,subprocess,os
from pathlib import Path
sys.path.insert(0,sys.argv[1])
class NoScience(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','numpy','timm'):raise RuntimeError('Science before prior receipt validation')
sys.meta_path.insert(0,NoScience())
def no(*a,**k):raise RuntimeError('No process/network in receipt metadata read')
subprocess.Popen=subprocess.run=socket.socket=socket.create_connection=os.system=os.fork=no
from models import predictor_sdpa_continuation_operator_v5 as op
c=json.loads(Path(sys.argv[2]).read_text());r=op.verify_task(Path(sys.argv[3]),'component-0','fixture',c,deep=False)
print(json.dumps({'metadata_only':True,'role':r['role'],'resource_sha256':r['resource_sha256']}))'''
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(ROOT),str(control),str(scratch)],cwd=ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
    (tmp_path/'cold-resource-read.log').write_bytes(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr.decode()[-4000:]
    assert json.loads(result.stdout)['resource_sha256']==op.old.digest(scratch/'results/components/0/component-resource.json')
    with pytest.raises(ValueError):op.verify_task(scratch,'component-0','fixture',c,deep=True)


def test_handoff_uses_v5_resource_validator_and_refuses_stale_worker_binding(shallow_resource):
    c,scratch,_=shallow_resource;file=scratch/'results/components/0/component-resource.json';r=json.loads(file.read_text())
    r['worker_sha256']='0'*64;write(file,r)
    with pytest.raises(ValueError,match='resource receipt'):op.verify_task(scratch,'component-0','fixture',c,deep=False)


def test_preimport_adapter_validation_bodies_match_frozen_startup_and_v5_reader():
    import ast
    def function(path,name):
        return next(n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    pure='models/predictor_sdpa_continuation_receipts_v3.py'
    assert ast.dump(function(pure,'validate'),include_attributes=False)==ast.dump(function('models/predictor_sdpa_component_startup_v2.py','validate'),include_attributes=False)
    original=function('models/predictor_sdpa_component_target_v6.py','load_resource')
    original.body=[n for n in original.body if not (isinstance(n,ast.ImportFrom) and n.module=='models.predictor_sdpa_component_startup_v2')]
    assert ast.dump(function(pure,'load_resource'),include_attributes=False)==ast.dump(original,include_attributes=False)
    from models import predictor_sdpa_continuation_receipts_v3 as adapter
    from models import predictor_sdpa_component_startup_v2 as frozen
    assert adapter.GPU_BYTES==frozen.GPU_BYTES==7*op.GiB
    assert adapter.KIND==frozen.KIND=='native-SDPA-component-startup-v2'


@pytest.mark.parametrize('fault',('valid','configure','fraction','precision','pid','device','cap','extra'))
def test_metadata_startup_validation_keeps_exact_frozen_acceptance(fault):
    from models import predictor_sdpa_continuation_receipts_v3 as adapter
    from models import predictor_sdpa_component_startup_v2 as frozen
    old=ROOT/'runs/pretrained/diagnostic/predictor-sdpa-integration-v3-20261108/linux10603-final/attended-1o9_nbog/saved/results/components/0/worker.json'
    value=copy.deepcopy(json.loads(old.read_text())['startup']);pid=value['pid']
    if fault=='configure':value['configure_calls']=2
    if fault=='fraction':value['allocator_fraction']=0.5
    if fault=='precision':value['precision']['matmul_precision']='high'
    if fault=='pid':value['pid']+=1
    if fault=='device':value['device']='cpu'
    if fault=='cap':value['allocator_cap_bytes']=8*op.GiB
    if fault=='extra':value['extra']='not in the receipt schema'
    for validate in (adapter.validate,frozen.validate):
        if fault=='valid':validate(value,pid=pid,device='cuda')
        else:
            with pytest.raises(ValueError):validate(value,pid=pid,device='cuda')


def test_new_plan_and_contract_require_exact_before_validation_publication(prepared,tmp_path):
    _,c,_=prepared;contract=c['contract'];p=op.component_plan(ROOT,CPU_REFERENCE,tmp_path/'components',local_tiny=True,control_sha256='d'*64)
    assert contract['trace_publication']==p['trace_publication']=='before-semantic-validation'
    assert contract['raw_descriptor_publication_marker_required'] is True
    assert contract['old_missing_raw_trace_remains_missing'] and contract['old_unknown_raw_field_remains_unknown']
    assert 'models/predictor_sdpa_raw_capture_v1.py' in op.REMOTE_NAMES and set(core.SOURCE_NAMES)==set(op.COMPONENT_SOURCES)
    assert len({contract[k]['sha256'] for k in ('parent_consumed_claim','missing_trace_component_claim','preceding_component_claim','earlier_component_claim','origin_native_claim')})==5
    q=copy.deepcopy(p);q.pop('trace_publication')
    with pytest.raises(ValueError):core.validate_plan(q)
    q=copy.deepcopy(p);q['trace_publication']='after-validation'
    with pytest.raises(ValueError):core.validate_plan(q)


def test_retained_invalid_CPU_trace_collects_exactly_but_cannot_complete(prepared,tmp_path):
    # Reuse byte-verified LOCAL CPU failure evidence; no model or target work here.
    source=ROOT/'runs/pretrained/diagnostic/predictor-sdpa-component-v6-20261113/pytest-first/test_actual_worker_saves_inval0'
    scratch=tmp_path/'scratch';scratch.mkdir(mode=0o700);components=scratch/'results/components';components.mkdir(mode=0o700,parents=True)
    shutil.copytree(source/'0',components/'0');shutil.copytree(source/'reference',components/'reference')
    shutil.copy2(source/'plan.json',components/'component-plan.json')
    from models import predictor_sdpa_component_records_v1 as rec
    from models import predictor_sdpa_launch_evidence_v2 as frozen
    ps=op.old.digest(components/'component-plan.json');w=rec.load(components/'0',kind=rec.WORKER_KIND,plan_sha256=ps,require_complete=False)
    trace=core.load_profile(components/'0',w['trace'],w['runtime'])
    assert w['trace']['saved_before_semantic_validation'] is True and trace['records'][0]['device_index']==-1
    with pytest.raises(ValueError,match='Unknown or malformed negative-device'):frozen.project(trace)
    with pytest.raises(ValueError):op.verify_task(scratch,'component-0','local-fixture',prepared[1],tiny=True)
    scope='local-CPU-retained-refusal-collection-v5';stream=io.BytesIO();op.stream(scratch,scope,stream)
    response=tmp_path/'response';response.write_bytes(stream.getvalue());saved=tmp_path/'saved'
    client.unpack(response,saved,{},scope,collection_only=True)
    copied=saved/'results/components/0';ww=rec.load(copied,kind=rec.WORKER_KIND,plan_sha256=ps,require_complete=False)
    assert not ww['complete'] and not ww['profile_replay_exact'] and ww['numerical_passed']
    assert (copied/'events.jsonl').read_bytes()==(components/'0/events.jsonl').read_bytes()
    assert core.load_profile(copied,ww['trace'],ww['runtime'])==trace
    with pytest.raises(ValueError):frozen.project(core.load_profile(copied,ww['trace'],ww['runtime']))
    assert not (copied/'complete.json').exists() and not (copied/'component-resource.json').exists()


def test_old_v5_resource_kind_cannot_substitute_for_the_v6_receipt(shallow_resource):
    c,scratch,_=shallow_resource;file=scratch/'results/components/0/component-resource.json';r=json.loads(file.read_text())
    r['kind']='native-fixed-SDPA-component-resource-v5';write(file,r)
    with pytest.raises(ValueError,match='resource receipt'):op.verify_task(scratch,'component-0','fixture',c,deep=False)



def test_explicit_policy_and_five_consumed_claims_are_bound(prepared,tmp_path):
    _,c,_=prepared;contract=c['contract']
    assert contract['metadata_policy']==op.METADATA_POLICY==core.METADATA_POLICY
    assert contract['parent_raw_trace_retained'] and contract['parent_incomplete_worker_stays_incomplete']
    assert contract['parent_consumed_claim']['sha256']=='3829d1b634b32053f73560610f2e78d5a835e7193eaea885b55211e6c2536c32'
    assert contract['missing_trace_component_claim']['sha256']=='95a15a08e35721a7840e26915544b5599d3656635b2527162b3ec595236770b0'
    for key in ('local_retention_review','parent_retained_trace_review'):
        item=contract[key];assert op.old.digest(ROOT/item['path'])==item['sha256'] and json.loads((ROOT/item['path']).read_text())['passed']
    p=op.component_plan(ROOT,CPU_REFERENCE,tmp_path/'components',local_tiny=True,control_sha256='d'*64)
    op.metadata_binding(p,c)
    assert p['metadata_policy']==contract['metadata_policy']
    assert 'models/predictor_sdpa_launch_evidence_v2.py' in op.REMOTE_NAMES
    assert contract['resource_receipt_kind']=='native-fixed-SDPA-component-resource-v6'


@pytest.mark.parametrize('field,value',(('metadata_policy','old-policy'),('component_plan_kind','old-plan'),
 ('resource_receipt_kind','native-fixed-SDPA-component-resource-v5')))
def test_grant_cannot_change_explicit_policy_or_receipt_scope(field,value):
    auth=authorization();auth[field]=value
    with pytest.raises(ValueError):op.validate_authorization(auth,'fixture')


@pytest.mark.parametrize('field,value',(('metadata_policy',None),('metadata_policy',True),('metadata_policy','old-policy'),
 ('kind','native-SDPA-fixed-input-plan-v5'),('trace_publication','after-validation'),('trace_format','old-format'),
 ('trace_encoding','old-encoding'),('source_sha256',{})))
def test_cold_plan_policy_binding_refuses_before_scientific_checker(shallow_resource,monkeypatch,field,value):
    c,scratch,_=shallow_resource;path=scratch/'results/components/component-plan.json';p=json.loads(path.read_text());p[field]=value;write(path,p)
    def no(*a,**k):raise AssertionError('No scientific checker after malformed metadata plan')
    monkeypatch.setattr(core,'load_worker',no)
    with pytest.raises(ValueError,match='before scientific imports'):op.verify_task(scratch,'component-0','fixture',c,deep=True)


@pytest.mark.parametrize('fault',('trace-policy','trace-publication','trace-count','trace-version','proof-policy','proof-kind','proof-admission'))
def test_preimport_worker_policy_guards_rehashed_trace_or_proof(shallow_resource,fault):
    c,scratch,_=shallow_resource;folder=scratch/'results/components/0';w=json.loads((folder/'worker.json').read_text())
    if fault=='trace-policy':w['trace']['metadata_policy']='old-policy'
    if fault=='trace-publication':w['trace']['saved_before_semantic_validation']=False
    if fault=='trace-count':w['trace']['records']=True
    if fault=='trace-version':w['trace']['torch_version']='invented'
    if fault=='proof-policy':w['backend_evidence']['metadata_policy']='old-policy'
    if fault=='proof-kind':w['backend_evidence']['kind']='one-fixed-SDPA-correlated-backend-v3'
    if fault=='proof-admission':w['backend_evidence']['actual_CUDA_execution_admitted']=True
    write(folder/'worker.json',w);ps=w['plan_sha256']
    write(folder/'complete.json',{'kind':w['kind'],'plan_sha256':ps,'complete':True,
      'header':{'file':'worker.json','bytes':(folder/'worker.json').stat().st_size,'sha256':op.old.digest(folder/'worker.json')}})
    resource=json.loads((folder/'component-resource.json').read_text());resource['worker_sha256']=op.old.digest(folder/'worker.json');write(folder/'component-resource.json',resource)
    with pytest.raises(ValueError,match='before scientific imports'):op.verify_task(scratch,'component-0','fixture',c,deep=False)
