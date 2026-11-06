"""Cold payload, source integrity, transport and supervision boundaries."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

import pytest
from models import predictor_eot8_record_operator_v2 as op
from models import predictor_eot8_record_client_v2 as client

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    spec=importlib.util.spec_from_file_location('prepare_predictor',ROOT/'scripts/prepare_predictor_eot8_record_v2.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    folder=tmp_path_factory.mktemp('predictor-preparation')/'unarmed'
    c,info=mod.prepare(ROOT,folder)
    return folder,c,info


def test_exact_unarmed_payload_and_bounded_stdin(prepared):
    folder,c,info=prepared;checked,sha=client.check_local(folder)
    assert checked==c and sha==info['control_sha256']
    assert info['stdin_bootstrap_bytes']<2*2**20 and info['remote_overlay_bytes']<2*2**20
    assert info['old_packet_bytes']==17192095 and info['prior_pins_verified']==6759
    assert not (folder/'AUTHORIZATION.json').exists() and not (folder/'ATTEMPT.json').exists()
    with pytest.raises(ValueError,match='Unarmed'):client.run(folder)
    assert not (folder/'ATTEMPT.json').exists()


@pytest.mark.parametrize('field,value',[('host','linux10609'),('parent','/tmp'),('paid_compute',True),('old_grants_consumed',False),('automatic_retries',1),('expected_gpu_uuid','GPU-other')])
def test_changed_target_or_domain_refused(prepared,field,value):
    c=copy.deepcopy(prepared[1]);c[field]=value
    with pytest.raises(ValueError):op.validate_control(c)


def test_bootstrap_cold_import_denies_scientific_or_network(prepared,tmp_path):
    body=client.payload(prepared[1])
    # Running this bootstrap with no authorization must load stdlib guards only,
    # refuse the empty auth input, and never reach readiness, mkdir or calculation.
    prefix="""import sys
forbid={'torch','numpy','models.attention_component','models.oracle_trace'}
def audit(event,args):
 if event=='import' and args[0].split('.')[0] in {'torch','numpy'}:raise AssertionError('Scientific bootstrap import')
 if event.startswith(('socket.','subprocess.')):raise AssertionError('Unexpected process/network')
sys.addaudithook(audit)
"""
    p=subprocess.run([sys.executable,'-I','-B','-c',prefix+'\nexec(compile(sys.stdin.buffer.read('+str(len(body))+'),"bootstrap","exec"))'],input=body+b'{}\n',stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=tmp_path,timeout=30)
    assert p.returncode!=0 and b'New exact university resource authorization required' in p.stderr,p.stderr.decode()[-2500:]
    assert b'AssertionError' not in p.stderr and list(tmp_path.iterdir())==[]


def test_full_packet_cold_restore_import_without_dispatch(prepared,tmp_path):
    c=prepared[1];packet=tmp_path/'code';op.extract(ROOT/c['archive_path'],c['archive_manifest'],packet)
    for name,item in c['archive_manifest']['files'].items():assert op.old.digest(packet/name)==item['sha256']
    overlay=tmp_path/'overlay';(overlay/'models').mkdir(parents=True)
    for name in op.REMOTE_NAMES:
        p=overlay/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(op.INIT if name=='models/__init__.py' else (ROOT/name).read_bytes())
    program='''import importlib,sys
sys.path.insert(0,sys.argv[1]);sys.path.insert(1,str(__import__('pathlib').Path(sys.argv[1]).parent/'code'))
import torch
blocked=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Scientific dispatch blocked'))
torch.nn.Module.__call__=blocked;torch.autograd.grad=blocked;torch.nn.functional.scaled_dot_product_attention=blocked
for name in sys.argv[2:]:importlib.import_module(name)
print('Cold imports passed with scientific dispatch blocked')'''
    modules=sorted({name[:-3].replace('/','.') for name in c['archive_manifest']['files'] if name.startswith(('models/','attacks/','datasets/')) and name.endswith('.py') and not name.endswith('__init__.py')})
    modules+=['models.predictor_eot8_record_v2','models.predictor_eot8_owned_v1','models.predictor_eot8_owned_receipt_v1','models.predictor_eot8_adapter_v1','models.predictor_eot8_record_worker_v2','models.predictor_eot8_events_v1','models.predictor_eot8_record_operator_v2']
    p=subprocess.run([sys.executable,'-I','-B','-c',program,str(overlay),*modules],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
    assert p.returncode==0,p.stderr.decode()[-3000:]


def test_corrupt_archive_refuses_before_extraction(prepared,tmp_path):
    p=tmp_path/'corrupt.tar';p.write_bytes(b'not the original archive')
    with pytest.raises(ValueError):op.extract(p,prepared[1]['archive_manifest'],tmp_path/'code')
    assert not (tmp_path/'code').exists()


def test_same_connection_collection_hash_mode_mtime_and_no_model(tmp_path):
    source=tmp_path/'source';source.mkdir();p=source/'saved.json';p.write_text('{"result": 1}\n');p.chmod(0o600)
    response=io.BytesIO();op.stream(source,'sha',response)
    path=tmp_path/'response';path.write_bytes(response.getvalue())
    client.unpack(path,tmp_path/'collected',{},'sha',collect=True)
    q=tmp_path/'collected/saved.json'
    assert q.read_bytes()==p.read_bytes() and q.stat().st_mtime_ns==p.stat().st_mtime_ns and q.stat().st_mode&0o777==0o600
    assert not list(source.glob('ATTEMPT*'))
    broken=tmp_path/'truncated';broken.write_bytes(response.getvalue()[:len(response.getvalue())//4])
    with pytest.raises((ValueError,EOFError,tarfile.ReadError)):client.unpack(broken,tmp_path/'partial',{},'sha',collect=True)


@pytest.mark.parametrize('fault',['symlink','external_hardlink','oversized_loss','oversized_gradient','oversized_trace','oversized_log'])
def test_inventory_file_scope_refusals(tmp_path,fault):
    folder=tmp_path/'scratch';folder.mkdir(mode=0o700)
    if fault=='symlink':(folder/'link').symlink_to(tmp_path)
    elif fault=='external_hardlink':
        p=tmp_path/'external';p.write_text('x');os.link(p,folder/'link')
    else:
        name,size={'oversized_loss':('EOT8-before-loss.pt',65537),'oversized_gradient':('EOT8-before-input.pt',16*2**20+1),'oversized_trace':('trace.jsonl',256*2**20+1),'oversized_log':('worker.log',16*2**20+1)}[fault]
        with (folder/name).open('wb') as f:f.truncate(size)
    with pytest.raises(ValueError):op.inventory(folder)


def test_proof_refusal_precedes_archive_and_science(prepared,tmp_path,monkeypatch):
    from test_predictor_eot8_record_target_v2 import authorization
    c=prepared[1];sha=op.hashlib.sha256(op.old.json_bytes(c)).hexdigest();auth=authorization();auth['control_sha256']=sha
    parent=tmp_path/'parent';parent.mkdir()
    monkeypatch.setattr(op,'readiness',lambda c:{'parent':str(parent),'device':parent.stat().st_dev,'inode':parent.stat().st_ino})
    monkeypatch.setattr(op.base,'gpu_gate',lambda *a:None)
    roles=[]
    def proof(*a,**k):roles.append(k['role']);return {'exit_status':1}
    monkeypatch.setattr(op.base.primitives,'supervise',proof)
    monkeypatch.setattr(op.old,'receive_archive',lambda *a:(_ for _ in ()).throw(AssertionError('Archive arrived before passing proof')))
    sources={n:op.INIT if n=='models/__init__.py' else (ROOT/n).read_bytes() for n in op.REMOTE_NAMES}
    response=io.BytesIO();result=op.remote(c,auth,io.BytesIO(),response,sources)
    path=tmp_path/'response';path.write_bytes(response.getvalue());client.unpack(path,tmp_path/'saved',c,sha)
    assert result==1 and roles==['proof']
    scratch=next(parent.iterdir());assert not (scratch/'original-packet.tar.gz').exists() and not (scratch/'results').exists()
    session=json.loads((scratch/'session.json').read_text());assert not session['passed'] and not session['complete']
    assert 0<session['deadline_unix']-session['started_unix']<=2400
    assert signal.getitimer(signal.ITIMER_REAL)[0]==0


def test_expired_supervisor_cannot_start_a_process(tmp_path):
    def gate():return {'logical_bytes':0,'allocated_bytes':0}
    with pytest.raises(TimeoutError):op.base.primitives.supervise([sys.executable,'-c','raise AssertionError("must not run")'],tmp_path,role='expired',deadline=time.monotonic()-1,gate=gate,file_limit=1024)
    data=json.loads((tmp_path/'supervisor-expired.json').read_text());assert data['exit_status'] is None and not data['complete']


def test_standalone_collection_never_reuses_old_clock_or_claim(prepared):
    folder=prepared[0]
    with pytest.raises(ValueError,match='Separate retrieval'):client.run(folder,download_only=True)
    assert not (folder/'ATTEMPT.json').exists()


def test_public_key_pin_change_refused(prepared,tmp_path):
    import shutil
    source,c,info=prepared;folder=tmp_path/'copy';shutil.copytree(source,folder)
    (folder/'known-hosts').write_text('linux10603.dc.engr.scu.edu ssh-ed25519 bad-key\n')
    with pytest.raises(ValueError,match='public host key'):client.check_local(folder)


@pytest.mark.parametrize('fail_worker',[None,1])
def test_serial_clock_and_stop_without_retry(prepared,tmp_path,monkeypatch,fail_worker):
    from test_predictor_eot8_record_target_v2 import authorization
    c=prepared[1];sha=op.hashlib.sha256(op.old.json_bytes(c)).hexdigest();auth=authorization();auth['control_sha256']=sha
    parent=tmp_path/'parent';parent.mkdir();roles=[];proof_passed=[False];archive_received=[False]
    monkeypatch.setattr(op,'readiness',lambda c:{'parent':str(parent),'device':parent.stat().st_dev,'inode':parent.stat().st_ino})
    monkeypatch.setattr(op.base,'gpu_gate',lambda *a:None)
    monkeypatch.setattr(op,'storage',lambda *a,**k:{'logical_bytes':0,'allocated_bytes':0})
    def supervise(command,folder,*,role,deadline,**kwargs):
        roles.append(role)
        assert deadline>time.monotonic()
        if role=='proof':return {'exit_status':0}
        i=int(role.split('-')[1]);assert proof_passed[0] and archive_received[0]
        state=json.loads((folder/'session.json').read_text());plan=json.loads((folder/('worker-plan-'+str(i)+'.json')).read_text())
        assert len(state['tasks'])==i and plan['deadline_unix']<=state['deadline_unix']-120 and plan['deadline_unix']-time.time()<=480
        return {'exit_status':1 if i==fail_worker else 0}
    monkeypatch.setattr(op.base.primitives,'supervise',supervise)
    monkeypatch.setattr(op.base.primitives,'validate_proof',lambda folder:proof_passed.__setitem__(0,True))
    def receive(*a):assert proof_passed[0];archive_received[0]=True
    monkeypatch.setattr(op.old,'receive_archive',receive)
    monkeypatch.setattr(op,'extract',lambda a,b,d:d.mkdir())
    monkeypatch.setattr(op,'check_worker',lambda folder,index,**kwargs:{'worker_sha256':'test','pid':10000+index,'index':index})
    response=io.BytesIO();status=op.remote(c,auth,io.BytesIO(),response,{n:op.INIT if n=='models/__init__.py' else (ROOT/n).read_bytes() for n in op.REMOTE_NAMES})
    assert roles==['proof','worker-0','worker-1'] if fail_worker==1 else roles==['proof','worker-0','worker-1','worker-2','worker-3']
    state=json.loads((next(parent.iterdir())/'session.json').read_text())
    assert state['passed'] is (fail_worker is None) and status==(0 if fail_worker is None else 1)
    assert signal.getitimer(signal.ITIMER_REAL)[0]==0


def test_preflight_is_inside_timer_and_restores_handlers(prepared,monkeypatch):
    from test_predictor_eot8_record_target_v2 import authorization
    c=prepared[1];sha=op.hashlib.sha256(op.old.json_bytes(c)).hexdigest();auth=authorization();auth['control_sha256']=sha
    previous=signal.getsignal(signal.SIGALRM)
    def fail(c):
        assert 0<signal.getitimer(signal.ITIMER_REAL)[0]<=2400
        raise RuntimeError('preflight refusal')
    monkeypatch.setattr(op,'readiness',fail)
    with pytest.raises(RuntimeError,match='preflight refusal'):op.remote(c,auth,io.BytesIO(),io.BytesIO(),{})
    assert signal.getsignal(signal.SIGALRM)==previous and signal.getitimer(signal.ITIMER_REAL)[0]==0


def test_copied_handoff_cannot_reset_local_claim(prepared,tmp_path):
    import shutil
    source,c,info=prepared;copy=tmp_path/'copied-handoff';shutil.copytree(source,copy)
    with pytest.raises(ValueError,match='Exact local handoff directory'):client.check_local(copy)
    assert not (copy/'ATTEMPT.json').exists()


@pytest.mark.parametrize('leaf,limit',[
 ('attention-calls.jsonl',8*2**20),('attention-calls.jsonl.partial',8*2**20),
 ('attention-scopes.json',64*2**10),('attention-scopes.json.partial',64*2**10),
 ('worker.json',256*2**10),('worker.json.partial',256*2**10),
 ('worker-complete.json',16*2**10),('worker-complete.json.partial',16*2**10),
 ('worker-failure.json',16*2**10),('worker-failure.json.partial',16*2**10)])
def test_new_record_files_and_partials_have_typed_byte_limits(tmp_path,leaf,limit):
    folder=tmp_path/'scratch';(folder/'results/0').mkdir(parents=True)
    with (folder/'results/0'/leaf).open('wb') as f:f.truncate(limit+1)
    with pytest.raises(ValueError,match='Typed file limit'):op.inventory(folder)


def test_call_records_count_as_metadata_with_one_atomic_copy(tmp_path):
    folder=tmp_path/'scratch';folder.mkdir(mode=0o700)
    for i in range(4):
        d=folder/'results'/str(i);d.mkdir(parents=True)
        with (d/'attention-calls.jsonl').open('wb') as f:f.truncate(8*2**20)
    with (folder/'results/3/attention-calls.jsonl.partial').open('wb') as f:f.truncate(8*2**20)
    p=tmp_path.stat();sample=op.storage(folder,{'device':p.st_dev,'inode':p.st_ino},parent=tmp_path)
    assert sample['categories']['metadata']==40*2**20 and sample['categories']['raw']==0


def test_failed_full_volume_worker_collects_without_retry_or_next_worker(prepared,tmp_path,monkeypatch):
    from test_predictor_eot8_record_target_v2 import authorization
    from test_predictor_eot8_record_v2 import capacity_fixture
    c=prepared[1];sha=op.hashlib.sha256(op.old.json_bytes(c)).hexdigest();auth=authorization();auth['control_sha256']=sha
    parent=tmp_path/'parent';parent.mkdir();roles=[]
    monkeypatch.setattr(op,'readiness',lambda c:{'parent':str(parent),'device':parent.stat().st_dev,'inode':parent.stat().st_ino})
    monkeypatch.setattr(op.base,'gpu_gate',lambda *a:None)
    monkeypatch.setattr(op,'storage',lambda *a,**k:{'logical_bytes':0,'allocated_bytes':0})
    monkeypatch.setattr(op.base.primitives,'validate_proof',lambda *a:None)
    monkeypatch.setattr(op.old,'receive_archive',lambda *a:None)
    monkeypatch.setattr(op,'extract',lambda a,b,d:d.mkdir())
    def supervise(command,folder,*,role,**kwargs):
        roles.append(role)
        if role=='proof':return {'exit_status':0}
        assert role=='worker-0'
        d=folder/'results/0';d.mkdir(mode=0o700);state,route=capacity_fixture()
        op.records.WorkerRecord(d).write(state,route=route,final=True)
        return {'exit_status':1}
    monkeypatch.setattr(op.base.primitives,'supervise',supervise)
    monkeypatch.setattr(op,'check_worker',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Failed worker was qualified')))
    response=io.BytesIO();status=op.remote(c,auth,io.BytesIO(),response,{n:op.INIT if n=='models/__init__.py' else (ROOT/n).read_bytes() for n in op.REMOTE_NAMES})
    assert status==1 and roles==['proof','worker-0']
    p=tmp_path/'response.bin';p.write_bytes(response.getvalue());client.unpack(p,tmp_path/'saved',c,sha)
    saved=op.records.load_worker(tmp_path/'saved/results/0',require_passed=False)
    assert len(saved['call_rows'])==2736 and not saved['passed']
    with pytest.raises(ValueError):op.records.load_worker(tmp_path/'saved/results/0')
    assert json.loads((tmp_path/'saved/session.json').read_text())['tasks']==[]
    assert signal.getitimer(signal.ITIMER_REAL)[0]==0
