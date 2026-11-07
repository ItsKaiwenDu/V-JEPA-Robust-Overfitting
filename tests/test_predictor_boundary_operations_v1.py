"""No-network handoff, continuing budgets, stop paths and byte-exact collection."""
from pathlib import Path
from types import SimpleNamespace
import base64,io,json,shutil,subprocess,sys,time
import pytest
from models import predictor_boundary_operator_v1 as op
from models import predictor_boundary_client_v1 as client
from models import predictor_boundary_record_v1 as rec
from scripts.prepare_predictor_boundary_v1 import prepare
from test_predictor_boundary_v1 import captures,CONTROL,resign_header

ROOT=Path(__file__).resolve().parents[1]


def authorization(sha='new'):
    return {'kind':'predictor-boundary-university-exception-v1','approved':True,'control_sha256':sha,'host':'linux10603',
        'maximum_new_attempts':1,'work_seconds':1200,'workers':2,'worker_seconds':420,'collection_seconds':600,'collection_bytes':3*op.GiB,
        'write_proof_bytes':5*op.GiB//2,'setup_bytes':3*op.GiB,'capture_bytes':9*op.GiB//4,'raw_bytes':512*op.MiB,'metadata_bytes':64*op.MiB,
        'allocator_bytes':7*op.GiB,'bootstrap_bytes':2*op.MiB,'automatic_retries':0,'paid_compute':False,'reset_old_claims':False,
        'prior_consumed_claim_sha256':op.PRIOR_CLAIM_SHA,'user_basis':'Local test fixture only; no target resource grant.'}


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    folder=tmp_path_factory.mktemp('boundary-prepared')/'handoff'
    c,info=prepare(ROOT,folder);return folder,c,info


def test_unarmed_final_payload_binds_source_history_value_only_capsule_and_host(prepared):
    folder,c,info=prepared
    actual,sha=client.check_local(folder)
    assert actual==c and sha==info['control_sha256']
    assert info['old_packet_bytes']==17192095 and info['anchor_members']==5 and info['prior_pins_verified']==7526
    assert info['overlay_files']==len(op.REMOTE_NAMES) and info['overlay_bytes']<2*op.MiB and info['stdin_bootstrap_bytes']<2*op.MiB
    assert not (folder/'AUTHORIZATION.json').exists() and not (folder/'ATTEMPT.json').exists()
    assert c['contract']['anchors']['0']['source_worker_passed'] and not c['contract']['anchors']['1']['source_worker_passed']
    assert all(a['value_only_not_executable_reference'] for a in c['contract']['anchors'].values())
    op.validate_control(c);op.validate_authorization(authorization(sha),sha)


def test_unarmed_and_copied_helpers_refuse_before_claim_or_SSH(prepared,tmp_path):
    folder,c,info=prepared
    def forbidden(*a,**k):raise AssertionError('SSH should not start.')
    with pytest.raises(ValueError,match='Unarmed'):client.run(folder,popen=forbidden)
    copy=tmp_path/'copied';shutil.copytree(folder,copy)
    with pytest.raises(ValueError,match='Exact local handoff'):client.check_local(copy)
    assert not (folder/'ATTEMPT.json').exists() and not (copy/'ATTEMPT.json').exists()
    with pytest.raises(ValueError):client.run(folder,download_only=True,popen=forbidden)


@pytest.mark.parametrize('field,value',[
 ('kind','predictor-clean-eot8-record-university-exception-v2'),('host','linux10609'),('maximum_new_attempts',2),
 ('work_seconds',1201),('worker_seconds',421),('workers',3),('collection_seconds',601),('raw_bytes',513*2**20),
 ('metadata_bytes',65*2**20),('paid_compute',True),('reset_old_claims',True),('prior_consumed_claim_sha256','old'),('control_sha256','old')])
def test_old_or_expanded_scope_refused(field,value):
    a=authorization();a[field]=value
    with pytest.raises(ValueError):op.validate_authorization(a,'new')


def test_cold_bootstrap_has_no_scientific_import_or_scratch_write_before_authorization(prepared,tmp_path):
    _,c,_=prepared;body=client.payload(c)
    body_path=tmp_path/'bootstrap.py';body_path.write_bytes(body)
    script='''import importlib.abc,sys,io,base64
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','numpy','timm'):raise AssertionError('Scientific import before authorization: '+name)
sys.meta_path.insert(0,Deny())
sys.stdin=io.TextIOWrapper(io.BytesIO(b'{}\\n'))
exec(compile(open(sys.argv[1],'rb').read(),'<cold-boundary-bootstrap>','exec'))
'''
    p=subprocess.run([sys.executable,'-I','-B','-c',script,str(body_path)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
    assert p.returncode!=0 and b'New exact boundary university authorization required' in p.stderr,p.stderr.decode()[-3000:]
    assert not p.stdout


def test_live_overlay_whole_packet_and_anchor_headers_are_verified_before_output(prepared,tmp_path):
    _,c,_=prepared
    source=ROOT/c['archive_path'];op.extract(source,c['archive_manifest'],tmp_path/'code')
    assert len([p for p in (tmp_path/'code').rglob('*') if p.is_file()])==99
    op.extract(Path(c['anchors_path']),c['anchors_manifest'],tmp_path/'anchors')
    meta=json.loads((tmp_path/'anchors/ANCHORS.json').read_text());assert meta==op.anchor_metadata(c['contract'])
    assert not meta['anchors']['1']['source_worker_complete'] and not meta['anchors']['1']['source_worker_passed']
    broken=tmp_path/'broken.tar.gz';broken.write_bytes(source.read_bytes()[:-5])
    with pytest.raises(ValueError):op.extract(broken,c['archive_manifest'],tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_collection_preserves_hash_mode_exact_mtime_and_refuses_truncation(tmp_path):
    source=tmp_path/'source';source.mkdir();p=source/'result.json';p.write_text('{"failed":true}\n');p.chmod(0o600)
    stream=io.BytesIO();op.stream(source,'sha',stream);response=tmp_path/'response';response.write_bytes(stream.getvalue())
    client.unpack(response,tmp_path/'saved',{},'sha',collect=True)
    actual=tmp_path/'saved/result.json';assert actual.read_bytes()==p.read_bytes() and actual.stat().st_mtime_ns==p.stat().st_mtime_ns and actual.stat().st_mode&0o777==0o600
    truncated=tmp_path/'truncated';truncated.write_bytes(response.read_bytes()[:-512])
    with pytest.raises(ValueError):client.unpack(truncated,tmp_path/'partial',{},'sha',collect=True)


@pytest.mark.parametrize('name,limit',[
 ('results/0/draw-00-family-0-prediction.pt',16*2**20),('results/0/draw-00-family-0-cotangent.pt.partial',16*2**20),
 ('results/0/worker.json.partial',256*2**10),('results/0/attention-calls.jsonl.partial',8*2**20),
 ('results/0/attention-scopes.json.partial',64*2**10),('results/0/worker-failure.json.partial',16*2**10)])
def test_new_typed_partial_limits_are_enforced(tmp_path,monkeypatch,name,limit):
    # Metadata-only fake inventory: not an allocation or filesystem capacity proof.
    monkeypatch.setattr(op.base,'inventory',lambda *a,**k:({name:{'bytes':limit+1}},0))
    with pytest.raises(ValueError,match='typed-file'):op.inventory(tmp_path,running=True)


def test_observation_tensors_are_raw_not_metadata_and_raw_cap_stays_512_MiB(tmp_path,monkeypatch):
    folder=tmp_path/'scratch';folder.mkdir(mode=0o700);st=tmp_path.stat();identity={'device':st.st_dev,'inode':st.st_ino}
    names={'results/0/draw-00-family-0-prediction.pt':{'bytes':512*op.MiB+1}}
    monkeypatch.setattr(op,'inventory',lambda *a,**k:(names,0))
    with pytest.raises(ValueError,match='category'):op.storage(folder,identity,parent=tmp_path)
    names['results/0/draw-00-family-0-prediction.pt']['bytes']=1024
    got=op.storage(folder,identity,parent=tmp_path)
    assert got['categories']['raw']==1024 and got['categories']['metadata']==0


def bind_test_control(folder,sha):
    # Test-owned copies only, never captured research or target files.
    p=folder/rec.atomic.SCOPES;scope=json.loads(p.read_text());scope['identity']['control_sha256']=sha;p.write_bytes(rec.atomic._json(scope))
    hp=folder/rec.atomic.HEADER;w=json.loads(hp.read_text());w['control_sha256']=sha
    w['evidence']['scopes']={'file':p.name,'bytes':p.stat().st_size,'sha256':op.old.digest(p)};hp.write_bytes(rec.atomic._json(w))
    m=folder/(rec.atomic.COMPLETE if w['passed'] else rec.atomic.FAILURE);marker=json.loads(m.read_text());marker['identity']['control_sha256']=sha
    marker['header']={'file':hp.name,'bytes':hp.stat().st_size,'sha256':op.old.digest(hp)};m.write_bytes(rec.atomic._json(marker))


def simulation(prepared,captures,tmp_path,monkeypatch,*,failure=False,corrupt_prior=False):
    _,c,_=prepared;sha=op.hashlib.sha256(op.old.json_bytes(c)).hexdigest();events=[];clock=[0.]
    monkeypatch.setattr(op,'time',SimpleNamespace(monotonic=lambda:clock[0],time=time.time))
    st=tmp_path.stat()
    monkeypatch.setattr(op,'readiness',lambda c:{'parent':str(tmp_path),'device':st.st_dev,'inode':st.st_ino})
    monkeypatch.setattr(op.base,'gpu_gate',lambda *a,**k:{'CPU_simulation_only':True})
    monkeypatch.setattr(op,'storage',lambda *a,**k:{'allocated_bytes':0,'categories':{},'ceiling':9*op.GiB//4})
    monkeypatch.setattr(op.signal,'signal',lambda *a,**k:None);monkeypatch.setattr(op.signal,'setitimer',lambda *a,**k:None)
    def receive(incoming,path,size,digest,deadline):
        events.append(('archive',path.name,clock[0]));path.write_bytes(b'CPU-simulation-only')
    monkeypatch.setattr(op.old,'receive_archive',receive)
    monkeypatch.setattr(op,'extract',lambda archive,spec,destination:destination.mkdir(mode=0o700))
    monkeypatch.setattr(op.base.primitives,'validate_proof',lambda folder:events.append(('proof-verified',clock[0])))
    verify=op.verify_capture;calls=[0]
    def check(folder,index,**kw):
        calls[0]+=1
        if corrupt_prior and calls[0]==2:raise ValueError('prior-capture-tamper-test')
        return verify(folder,index,control_sha256=kw['control_sha256'],live=False,require_passed=kw.get('require_passed',True))
    monkeypatch.setattr(op,'verify_capture',check)
    def supervise(command,folder,*,role,deadline,**kw):
        if role=='proof':
            events.append(('proof',deadline));clock[0]+=10;return {'exit_status':0}
        i=int(role[-1]);events.append(('worker',i,deadline,clock[0]));clock[0]+=30
        source=captures/('failed-candidate' if failure and i==1 else str(i));target=folder/'results'/str(i)
        shutil.copytree(source,target);bind_test_control(target,sha)
        return {'exit_status':1 if failure and i==1 else 0}
    monkeypatch.setattr(op.base.primitives,'supervise',supervise)
    output=io.BytesIO();sources={n:op.INIT if n=='models/__init__.py' else (ROOT/n).read_bytes() for n in op.REMOTE_NAMES}
    prior_umask=op.os.umask(0o077)
    try:result=op._execute(c,authorization(sha),io.BytesIO(),output,sources,sha,0.,1200.,lambda *a:None)
    finally:op.os.umask(prior_umask)
    response=tmp_path/'response';response.write_bytes(output.getvalue());client.unpack(response,tmp_path/'saved',c,sha)
    return result,events,tmp_path/'saved'


def test_continuing_clock_two_serial_workers_proof_before_both_inputs_and_same_connection(prepared,captures,tmp_path,monkeypatch):
    result,events,saved=simulation(prepared,captures,tmp_path,monkeypatch)
    assert result==0 and events[0]==('proof',60.) and events[1][0]=='proof-verified'
    assert [e[1] for e in events if e[0]=='archive']==['original-packet.tar.gz','anchors.tar.gz']
    workers=[e for e in events if e[0]=='worker'];assert [e[1] for e in workers]==[0,1]
    assert all(e[2]<=e[3]+420 and e[2]<=1075 for e in workers)
    session=json.loads((saved/'session.json').read_text());assert session['capture_complete'] and session['passed'] and len(session['tasks'])==2
    assert all(not x['CUDA_backend_qualified'] for x in session['tasks'])


def test_numerical_failure_collects_complete_observations_without_passing_candidate_or_retry(prepared,captures,tmp_path,monkeypatch):
    result,events,saved=simulation(prepared,captures,tmp_path,monkeypatch,failure=True)
    assert result==1 and [e[1] for e in events if e[0]=='worker']==[0,1]
    session=json.loads((saved/'session.json').read_text());assert session['capture_complete'] and not session['passed'] and not session['complete']
    assert len(session['tasks'])==2 and not session['tasks'][1]['numerical_passed']
    candidate=rec.load(saved/'results/1',control_sha256=op.hashlib.sha256(op.old.json_bytes(prepared[1])).hexdigest(),require_passed=False)
    assert len(candidate['observations'])==16 and candidate['capture_complete'] and not candidate['passed']


def test_changed_original_receipt_stops_before_candidate_and_still_collects(prepared,captures,tmp_path,monkeypatch):
    result,events,saved=simulation(prepared,captures,tmp_path,monkeypatch,corrupt_prior=True)
    assert result==1 and [e[1] for e in events if e[0]=='worker']==[0]
    session=json.loads((saved/'session.json').read_text());assert not session['passed'] and 'prior-capture-tamper-test' in session['error']
    assert not (saved/'results/1').exists()
