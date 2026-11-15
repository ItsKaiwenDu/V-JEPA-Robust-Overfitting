"""Cold wiring/guard/scheduler/transport tests; no new scientific objectives.

All proof, GPU, clock and accepted-worker scheduler fixtures below are explicit
simulations. No simulated receipt is a CUDA numerical qualification.
"""
import ast
import base64
import builtins
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import FunctionType, SimpleNamespace
import pytest
from models import predictor_missing_replay_operator_v1 as op
from models import predictor_missing_replay_client_v1 as client
from models import predictor_missing_replay_worker_v1 as worker
from models import predictor_missing_replay_records_v1 as records

ROOT=Path(__file__).resolve().parents[1]
RAW=Path(os.environ['VJEPA_MISSING_REPLAY_EVIDENCE'])
HANDOFF=Path(os.environ['VJEPA_MISSING_REPLAY_HANDOFF'])
OLD=ROOT/'runs/pretrained/diagnostic/predictor-forward-preserving-handoff-v2-20261114/linux10603-final/attended-yo4uf0wc/saved'
REFERENCE=ROOT/'runs/pretrained/diagnostic/predictor-replay-capsule-v1-20261114/build-first/capsule'


class BeforeScience(Exception):pass


def control():return json.loads((HANDOFF/'CONTROL.json').read_bytes())


def sha(c):return op.hashlib.sha256(op.old.json_bytes(c)).hexdigest()


def synthetic_control(tmp_path):
    c=control();c['local_handoff_directory']=str(tmp_path/'SYNTHETIC-TEST-ONLY')
    assert sha(c)!=sha(control()) and sha(c)!=op.capsule.SOURCE_CONTROL
    return c


def test_scientific_AST_changes_only_binding_state_field():
    original=next(n for n in ast.parse(Path(worker.frozen.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    actual=worker.body_ast().body[0]
    added=[i for i,n in enumerate(actual.body) if ast.dump(n,include_attributes=False)==ast.dump(ast.parse('state["reference_binding"] = args.reference_binding').body[0],include_attributes=False)]
    assert len(added)==1
    del actual.body[added[0]]
    assert ast.dump(actual,include_attributes=False)==ast.dump(original,include_attributes=False)
    assert records.write_final.__code__ is records.frozen.write_final.__code__
    assert records._load.__code__ is records.frozen.load.__code__
    assert records.check_scope.__code__ is records.frozen.check_scope.__code__
    assert records._native_scope.__code__ is records.frozen._native_scope.__code__


def test_actual_worker_cold_prelude_routes_old_refs_separately_then_stops_before_imports(tmp_path):
    c=synthetic_control(tmp_path);h=sha(c)
    with op.capsule.Capsule(REFERENCE,manifest_sha256=records.MANIFEST_SHA) as reader:
        fn,tokens,binding=worker.bound_body(reader,h)
        assert binding==records.binding(h)
        seen=[];real_import=builtins.__import__
        def stop_import(name,*args,**kwargs):
            if name in ('numpy','torch'):
                seen.append(name);raise BeforeScience('Intentional stop before scientific imports.')
            return real_import(name,*args,**kwargs)
        namespace=dict(fn.__globals__,platform=SimpleNamespace(node=lambda:'linux10603',python_version=lambda:'3.11.17'),
                       __builtins__={**vars(builtins),'__import__':stop_import})
        run=FunctionType(fn.__code__,namespace)
        args=SimpleNamespace(index=3,tiny=False,device='cuda',control_sha256=h,reference=tokens['original'],replay_reference=tokens['replay'],reference_binding=binding)
        with pytest.raises(BeforeScience):run(args)
        assert seen==['numpy'] and not (tmp_path/'results').exists()
        load=fn.__globals__['records'].load
        for fault in [('original',1,h),('replay',2,h),('original',2,op.capsule.SOURCE_CONTROL)]:
            with pytest.raises(ValueError):load(tokens[fault[0]],control_sha256=fault[2],index=fault[1],live=True)


@pytest.mark.parametrize('fault',['old-kind','old-control','wrong-index','tiny','missing-binding','altered-binding'])
def test_new_identity_rejects_old_grants_roles_and_missing_lineage(fault):
    h='ab'*32
    state={'kind':records.WORKER_KIND,'execution_host':'linux10603','local_tiny':False,'mode':'candidate',
           'replay':1,'index':3,'pid':999999,'control_sha256':h,'reference_binding':records.binding(h)}
    if fault=='old-kind':state['kind']=records.frozen.WORKER_KIND
    elif fault=='old-control':state['control_sha256']=op.capsule.SOURCE_CONTROL
    elif fault=='wrong-index':state.update(index=1,replay=0)
    elif fault=='tiny':state.update(local_tiny=True,execution_host='TEST-CPU')
    elif fault=='missing-binding':del state['reference_binding']
    else:state['reference_binding']['references']['same_replay_original']['source_index']=3
    with pytest.raises(ValueError):records.identity(state)


@pytest.mark.parametrize('fault',['old-authorization','four-workers','changed-contract','changed-manifest','changed-archive','different-files'])
def test_exact_new_control_and_authorization_refuse_changed_scope(tmp_path,fault):
    c=synthetic_control(tmp_path);auth=op.authorization_template(c,sha(c),'Synthetic local cold test only.')
    if fault=='old-authorization':auth=json.loads((OLD/'authorization.json').read_bytes())
    elif fault=='four-workers':c['resource']['workers']=4
    elif fault=='changed-contract':c['contract_text']+=' '
    elif fault=='changed-manifest':c['reference_manifest']['execution_permitted']=True
    elif fault=='changed-archive':c['archive_manifest']['bytes']+=1
    else:c['remote_source_sha256'].pop(next(iter(c['remote_source_sha256'])))
    with pytest.raises(ValueError):
        op.validate_control(c);op.validate_authorization(auth,sha(c),c)


def test_final_helper_unarmed_refuses_before_claim_session_or_SSH():
    def forbidden(*a,**k):raise AssertionError('Unarmed helper reached SSH.')
    before=set(HANDOFF.iterdir())
    with pytest.raises(ValueError,match='Unarmed'):client.run(HANDOFF,popen=forbidden)
    assert set(HANDOFF.iterdir())==before
    assert not (HANDOFF/'AUTHORIZATION.json').exists() and not (HANDOFF/'ATTEMPT.json').exists()
    with pytest.raises(ValueError):client.run(HANDOFF,download_only=True,popen=forbidden)


def test_transport_AST_changes_only_second_archive_and_total_clock():
    original=next(n for n in ast.parse(Path(client.transport.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    adapted=client.transport_ast().body[0];counts={'archives':0,'deadline':0}
    class Reverse(ast.NodeTransformer):
        def visit_For(self,n):
            if isinstance(n.target,ast.Name) and n.target.id=='archive_key':
                counts['archives']+=1
                old=n.body[0];old.items[0].context_expr=ast.parse("(Path(c['repository'])/c['archive_path']).open('rb')",mode='eval').body
                return old
            return self.generic_visit(n)
        def visit_Constant(self,n):
            if n.value==1500:counts['deadline']+=1;return ast.copy_location(ast.Constant(3000),n)
            return n
    assert ast.dump(Reverse().visit(adapted),include_attributes=False)==ast.dump(original,include_attributes=False)
    assert counts=={'archives':1,'deadline':1}


@pytest.mark.parametrize('valid_auth',[False,True])
def test_actual_memory_bootstrap_stops_at_authorization_or_host_gate_without_writes(tmp_path,valid_auth):
    c=synthetic_control(tmp_path);body=client.payload(c)
    code="""
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','numpy'}:raise AssertionError('Scientific import')
sys.meta_path.insert(0,Block())
def audit(event,args):
 if event in {'os.mkdir','subprocess.Popen','os.system','socket.__new__','socket.connect'}:raise AssertionError('Target operation '+event)
sys.addaudithook(audit)
try:
 exec(compile(sys.stdin.buffer.read(int(sys.argv[1])),'<actual-single-replay-bootstrap>','exec'))
except ValueError as error:
 print('COLD-REFUSAL:',str(error))
else:raise AssertionError('Bootstrap unexpectedly executed')
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
"""
    auth=op.authorization_template(c,sha(c),'Synthetic cold bootstrap only.') if valid_auth else {}
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(len(body))],input=body+op.old.json_bytes(auth),capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr.decode()
    assert (b'Wrong target host/OS' if valid_auth else b'new exact scoped') in result.stdout
    (RAW/('bootstrap-host-gate.txt' if valid_auth else 'bootstrap-authorization.txt')).write_bytes(result.stdout)


def setup_child(tmp_path,monkeypatch):
    c=synthetic_control(tmp_path);h=sha(c);scratch=tmp_path/(op.PREFIX+'cold-fixture');overlay=scratch/'overlay'
    scratch.mkdir(mode=0o700);overlay.mkdir(mode=0o700)
    for name in op.REMOTE_NAMES:
        data=op.INIT if name=='models/__init__.py' else (ROOT/name).read_bytes()
        p=overlay/name;p.parent.mkdir(mode=0o700,parents=True,exist_ok=True);p.write_bytes(data)
    shutil.copytree(OLD/'code',scratch/'code')
    op.old.write_json(scratch/'control.json',c)
    op.old.write_json(scratch/'authorization.json',op.authorization_template(c,h,'Synthetic cold entry only.'))
    op.old.write_json(scratch/'session.json',{'complete':False,'tasks':[],'deadline_unix':1900.,'reference_binding':records.binding(h)})
    op.old.write_json(scratch/'worker-plan-3.json',{'index':3,'control_sha256':h,'deadline_unix':1480.})
    source=ROOT/'scripts/capture_predictor_missing_replay_v1.py'
    spec=importlib.util.spec_from_file_location('single_replay_cold_entry_fixture',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.overlay=overlay
    module.Path=lambda *args:tmp_path if args==('/var/tmp',) else Path(*args)
    module.time=SimpleNamespace(time=lambda:1000.)
    timers=[];module.signal=SimpleNamespace(SIGALRM=1,SIG_DFL=0,ITIMER_REAL=0,signal=lambda *a:None,setitimer=lambda *a:timers.append(a))
    seen=[]
    def mapped_reader(path,*,manifest_sha256):
        assert path==scratch/'reference' and manifest_sha256==records.MANIFEST_SHA
        seen.append('actual-reference-reader-with-explicit-fixture-path-mapping')
        return op.capsule.Capsule(REFERENCE,manifest_sha256=manifest_sha256)
    def stop(*a):seen.append('live-GPU-boundary');raise BeforeScience()
    view=SimpleNamespace(**{k:getattr(op,k) for k in ['PREFIX','old','validate_control','validate_authorization','records']})
    view.capsule=SimpleNamespace(Capsule=mapped_reader);view.base=SimpleNamespace(gpu_gate=stop);module.op=view
    monkeypatch.setattr(sys,'argv',[str(source),'--scratch',str(scratch),'--control-sha256',h,'--index','3'])
    return module,scratch,c,h,seen,timers


def test_actual_child_entry_validates_bound_control_refs_and_reaches_GPU_boundary(tmp_path,monkeypatch):
    entry,scratch,c,h,seen,timers=setup_child(tmp_path,monkeypatch)
    with pytest.raises(BeforeScience):entry.main()
    assert seen==['actual-reference-reader-with-explicit-fixture-path-mapping','live-GPU-boundary']
    assert timers==[(0,480.)] and not (scratch/'results').exists()
    (RAW/'actual-child-check.json').write_text(json.dumps({'passed':True,'source_and_packet_pins_verified':True,'actual_capsule_reader_used':True,'synthetic_control_distinct':True,'GPU_clock_parent_and_reference_paths_explicit_fixtures':True,'scientific_or_GPU_work':False})+'\n')


@pytest.mark.parametrize('fault',['old-auth','source','packet','clock','already-started','reference-binding'])
def test_actual_child_refusals_precede_science(tmp_path,monkeypatch,fault):
    entry,scratch,c,h,seen,timers=setup_child(tmp_path,monkeypatch)
    if fault=='old-auth':(scratch/'authorization.json').write_bytes((OLD/'authorization.json').read_bytes())
    elif fault=='source':
        with (scratch/'overlay/models/predictor_missing_replay_worker_v1.py').open('ab') as f:f.write(b' ')
    elif fault=='packet':
        with (scratch/'code/PROFILE-HANDOFF.json').open('ab') as f:f.write(b' ')
    elif fault=='clock':
        x=json.loads((scratch/'worker-plan-3.json').read_bytes());x['deadline_unix']=999.;(scratch/'worker-plan-3.json').write_bytes(op.old.json_bytes(x))
    else:
        x=json.loads((scratch/'session.json').read_bytes())
        if fault=='already-started':x['tasks']=[{'simulation_only':True}]
        else:x['reference_binding']['new_worker']['index']=1
        (scratch/'session.json').write_bytes(op.old.json_bytes(x))
    with pytest.raises(ValueError):entry.main()
    assert not seen and not (scratch/'results').exists()


@pytest.fixture(scope='module')
def extracted_tree():
    folder=RAW/'extracted-corrected';folder.mkdir(mode=0o700)
    c=control()
    op.extract(ROOT/c['archive_path'],c['archive_manifest'],folder/'code')
    op.extract_reference(ROOT/c['reference_archive_path'],c['reference_manifest'],folder/'reference')
    return folder


def test_actual_extraction_inventory_categories_and_stat_footprint(extracted_tree):
    files,allocated=op.inventory(extracted_tree,hashing=True)
    assert len(files)==154
    categories=dict.fromkeys(['raw','input','code','metadata','cache','emergency','proof'],0)
    for name,item in files.items():categories[op.category(name)]+=item['bytes']
    assert categories['raw']==214635140 and categories['metadata']==2978204
    assert categories['code']==26446958 and categories['input']==0
    info=extracted_tree.parent.stat();row=op.storage(extracted_tree,{'device':info.st_dev,'inode':info.st_ino},parent=extracted_tree.parent)
    assert row['categories']==categories and row['logical_bytes']==244060302
    (RAW/'actual-extraction-footprint.json').write_bytes(op.old.json_bytes({'passed':True,'files':len(files),**row,'new_proof_or_scientific_work':False}))


def test_new_archive_and_shared_category_caps_are_exact(tmp_path):
    c=control();assert len(client.payload(c))<=2*op.MiB
    assert c['archive_manifest']['bytes']+op.REFERENCE_BYTES==134095272<128*op.MiB
    assert op.category('reference-capsule.tar.gz.partial')=='input'
    assert op.category(next('reference/'+n for n in op.REFERENCE_EXTENTS if n.startswith('arrays/')))=='raw'
    with pytest.raises(ValueError):op.category('reference/arrays/unpinned.npy')
    with pytest.raises(ValueError):op.category('results/0/input.npy')
    fake={'reference-capsule.tar.gz':{'bytes':op.REFERENCE_BYTES+1}}
    namespace=dict(op.inventory.__globals__,observed=SimpleNamespace(inventory=lambda *a,**k:(fake,0)))
    with pytest.raises(ValueError):FunctionType(op.inventory.__code__,namespace)(tmp_path,hashing=False,running=False)


def simulate_supervisor(tmp_path,fault):
    c=synthetic_control(tmp_path);h=sha(c);calls=[];clock=[1000.];saved=[]
    sources={n:op.INIT if n=='models/__init__.py' else (ROOT/n).read_bytes() for n in op.REMOTE_NAMES}
    def readiness(_):return {'parent':str(tmp_path),'device':tmp_path.stat().st_dev,'inode':tmp_path.stat().st_ino}
    def gpu(*a):
        calls.append(('GPU-SIMULATION',None))
        if fault=='foreign-GPU' and sum(x[0]=='GPU-SIMULATION' for x in calls)==2:raise ValueError('SIMULATED foreign CUDA process')
        return {'SIMULATION_ONLY':True}
    def supervise(command,folder,*,role,deadline,gate,file_limit,env=None):
        calls.append(('SUPERVISION-SIMULATION',role));assert deadline<=1900. and file_limit==(3*op.GiB if role=='proof' else 16*op.MiB)
        if role=='worker-3':
            assert command[-1]=='3' and 'capture_predictor_missing_replay_v1.py' in str(command)
            assert env['CUDA_VISIBLE_DEVICES']!=''
            if fault=='deadline':clock[0]+=901.
        return {'exit_status':1 if fault in ('proof','child') and role==('proof' if fault=='proof' else 'worker-3') else 0}
    def receive(stream,path,size,expected,deadline):
        calls.append(('ARCHIVE-RECEIVE-SIMULATION',path.name));path.write_bytes(b'SIMULATION-ONLY')
    def extract(archive,spec,path):path.mkdir(mode=0o700)
    def extract_ref(archive,spec,path):
        if fault=='reference':raise ValueError('SIMULATED reference refusal')
        path.mkdir(mode=0o700)
    class Reader:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def execution_binding(self,h):return records.binding(h)
    def check(folder,*,control_sha256):
        if fault=='receipt':raise ValueError('SIMULATED result refusal')
        # Explicit scheduler stub: cannot satisfy records.load and never scientific qualification.
        return {'SIMULATION_ONLY':True,'pid':999999,'index':3,'numerical_passed':False}
    def stream(folder,sha,output):
        calls.append(('COLLECTION-SIMULATION',None));saved.append(json.loads((folder/'session.json').read_bytes()))
        output.write(b'SIMULATION-ONLY; no scientific result')
    old_view=SimpleNamespace(**{k:getattr(op.old,k) for k in ['json_bytes','write_json','safe_name']},receive_archive=receive,descendants=lambda p:{})
    base_view=SimpleNamespace(gpu_gate=gpu,primitives=SimpleNamespace(supervise=supervise,validate_proof=lambda p:calls.append(('PROOF-RECEIPT-SIMULATION',None))))
    namespace=dict(op._execute.__globals__,readiness=readiness,base=base_view,old=old_view,extract=extract,extract_reference=extract_ref,
                   capsule=SimpleNamespace(Capsule=lambda *a,**k:Reader()),check_worker=check,stream=stream,
                   signal=SimpleNamespace(ITIMER_REAL=0,SIGALRM=1,signal=lambda *a:None,setitimer=lambda *a:None),
                   storage=lambda *a,**k:{'SIMULATION_ONLY':True},time=SimpleNamespace(time=lambda:clock[0],monotonic=lambda:clock[0]))
    execute=FunctionType(op._execute.__code__,namespace)
    result=execute(c,{},io.BytesIO(),io.BytesIO(),sources,h,1000.,1900.,lambda *a:None)
    return result,calls,saved


@pytest.mark.parametrize('fault',['none','proof','reference','foreign-GPU','child','receipt','deadline'])
def test_one_worker_stop_and_same_session_collection_simulations(tmp_path,fault):
    result,calls,saved=simulate_supervisor(tmp_path,fault)
    roles=[x[1] for x in calls if x[0]=='SUPERVISION-SIMULATION']
    assert roles[0]=='proof' and roles.count('worker-3')==(0 if fault in ('proof','reference','foreign-GPU') else 1)
    assert set(roles)<= {'proof','worker-3'} and len([x for x in calls if x[0]=='COLLECTION-SIMULATION'])==1
    assert len(saved)==1 and saved[0]['passed'] is (fault=='none') and result==(0 if fault=='none' else 1)
    if fault!='proof':assert [x[1] for x in calls if x[0]=='ARCHIVE-RECEIVE-SIMULATION']==['original-packet.tar.gz','reference-capsule.tar.gz']
    assert saved[0]['old_work_seconds_not_reset']==477.3622766463086
    (RAW/('supervisor-'+fault+'.json')).write_bytes(op.old.json_bytes({'explicit_simulation_only':True,'calls':calls,'state':saved[0],'result':result,'no_scientific_or_GPU_objective':True}))


def test_strict_inventory_observer_survives_live_operator_integration(tmp_path):
    folder=tmp_path/'scratch';folder.mkdir(mode=0o700);p=folder/'unsafe.bin';p.write_bytes(b'x');os.link(p,tmp_path/'outside-link')
    with pytest.raises(ValueError,match='external hardlink') as error:op.inventory(folder)
    assert error.value.__cause__ is not None
    info=error.value.inventory_observation
    assert info['refusal_preserved'] is True and info['scan_retried'] is False
    assert info['relative_path']=='unsafe.bin'
    (RAW/'integrated-inventory-refusal.json').write_bytes(op.old.json_bytes(info))


def test_memory_bootstrap_rebases_sources_and_collects_real_guarded_proof_refusal(tmp_path):
    c=synthetic_control(tmp_path);h=sha(c);body=client.payload(c)
    fixture=tmp_path/'bootstrap-local-fixture';fixture.mkdir(mode=0o700)
    cpath=RAW/'synthetic-bootstrap-final-control.json';cpath.write_bytes(op.old.json_bytes(c))
    response=RAW/'bootstrap-final-proof-refusal-response.bin'
    code=r'''
import io,json,sys,importlib.abc
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','numpy'}:raise AssertionError('Scientific import')
sys.meta_path.insert(0,Block())
def audit(event,args):
 if event in {'subprocess.Popen','os.system','socket.__new__','socket.connect'}:raise AssertionError('External action '+event)
sys.addaudithook(audit)
body=sys.stdin.buffer.read().decode();prefix=body.rsplit('sys.exit(m.main(',1)[0]
exec(compile(prefix,'<actual-bootstrap-prefix>','exec'))
m=sys.modules['models.predictor_missing_replay_operator_v1']
c=json.loads(Path(sys.argv[2]).read_bytes());h=m.hashlib.sha256(m.old.json_bytes(c)).hexdigest()
fixture=Path(sys.argv[1]);info=fixture.stat()
m.readiness=lambda _:dict(parent=str(fixture),device=info.st_dev,inode=info.st_ino)
actual_storage=m.storage
m.storage=lambda folder,identity,**kwargs:actual_storage(folder,identity,parent=fixture,**kwargs)
m.base.gpu_gate=lambda *a:dict(EXPLICIT_GPU_SIMULATION=True)
seen=[]
def before_proof(command,folder,**kwargs):
 assert kwargs['role']=='proof' and kwargs['file_limit']==3*m.GiB
 row=kwargs['gate']()  # actual new storage -> observer -> frozen inventory/source checks
 assert row['logical_bytes']<m.GiB
 seen.append('actual-bootstrap-sources-and-strict-guard-before-SIMULATED-proof')
 raise RuntimeError('EXPLICIT-PROOF-BOUNDARY-SIMULATION; no allocation or science')
m.base.primitives.supervise=before_proof
sources={n:m.INIT if n=='models/__init__.py' else (Path(c['repository'])/n).read_bytes() for n in m.REMOTE_NAMES}
result=m.remote(c,m.authorization_template(c,h,'Synthetic local bootstrap fixture only.'),io.BytesIO(),sys.stdout.buffer,sources)
assert result==1 and len(seen)==1
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
'''
    with response.open('xb') as out:
        result=subprocess.run([sys.executable,'-I','-B','-c',code,str(fixture),str(cpath)],input=body,stdout=out,stderr=subprocess.PIPE,timeout=30)
    (RAW/'bootstrap-final-proof-refusal-stderr.txt').write_bytes(result.stderr)
    assert result.returncode==0,result.stderr.decode()
    dest=RAW/'bootstrap-final-proof-refusal-saved';client.unpack(response,dest,c,h)
    state=json.loads((dest/'session.json').read_bytes())
    assert state['complete'] is False and state['passed'] is False and state['tasks']==[]
    assert 'EXPLICIT-PROOF-BOUNDARY-SIMULATION' in state['error']
    assert not (dest/'results').exists() and not (dest/'space-proof-v2.bin.partial').exists()
    assert json.loads((dest/'control.json').read_bytes())==c
    (RAW/'bootstrap-rebased-collection-check.json').write_bytes(op.old.json_bytes({'passed':True,'actual_memory_bootstrap_and_strict_inventory_and_collection_used':True,'explicit_GPU_parent_and_proof_boundary_fixtures':True,'new_proof_model_or_GPU_work':False,'collected_failed_session_tasks':0,'response_bytes':response.stat().st_size}))
