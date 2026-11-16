"""Whole v4 source/lifecycle checks; real metadata IO, no scientific/OS workers."""
import ast
import base64
import builtins
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
from types import FunctionType,SimpleNamespace as NS
import pytest
from models import predictor_pgd_split_outer_v4 as outer
from models import predictor_pgd_split_preclaim_refusal_v4 as recipe
from models import predictor_pgd_split_bootstrap_v1 as boot
from models import predictor_pgd_split_coordinator_v1 as coordinator
from models import predictor_pgd_split_scratch_v1 as scratch
from models import predictor_pgd_split_collection_v1 as collection
from models import predictor_pgd_split_source_bundle_v1 as source

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'runs/pretrained/diagnostic/predictor-pgd-startup-v3-20261116/handoff-parent/linux10603-unarmed'

@pytest.fixture(scope='module')
def package(tmp_path_factory):
    parent=tmp_path_factory.mktemp('EXPLICIT-UNARMED-V4');parent.chmod(0o700);folder=parent/'candidate'
    result=outer.prepare_package(ROOT,folder);raw=(folder/'CONTROL-CANDIDATE.json').read_bytes()
    code=(ROOT/'models/predictor_pgd_split_outer_v4.py').read_bytes();c,b,p,sha=outer.validate_control(raw,outer.digest(code))
    payload=json.loads((folder/'SOURCE-PAYLOAD.json').read_bytes());sources=outer.decode_sources(payload,c,b)
    a=outer.authorization(c,sha,'EXPLICIT INERT V4 LOCAL TEST METADATA ONLY; NOT A LIVE USER GRANT')
    return NS(folder=folder,result=result,raw=raw,code=code,c=c,b=b,p=p,sha=sha,payload=payload,sources=sources,auth=outer.encoded(a))


def test_new_source_package_preserves_every_v3_upload_and_remains_unarmed(package):
    x=package;oldc=json.loads((OLD/'CONTROL-CANDIDATE.json').read_bytes())
    assert (x.folder/'SOURCE-PAYLOAD.json').read_bytes()==(OLD/'SOURCE-PAYLOAD.json').read_bytes()
    assert x.c['supervised_control_text']==oldc['supervised_control_text'] and x.c['runtime_files']==oldc['runtime_files']
    assert len(x.c['local_source_sha256'])==14 and x.c['kind']=='pgd-split-outer-control-v4'
    assert x.sha!=oldc['outer_source_sha256'] and x.sha!=json.loads((OLD/'ATTEMPT.json').read_bytes())['outer_control_sha256']
    assert x.result['preclaim_refusal_preserves_primary_error'] is True
    assert x.result['preclaim_refusal_policy_sha256']==outer.REFUSAL_POLICY_SHA
    assert set(p.name for p in x.folder.iterdir())=={'CONTROL-CANDIDATE.json','SOURCE-PAYLOAD.json','PACKAGE.json','known-hosts'}
    assert x.p==json.loads((ROOT/'configs/predictor_pgd_split_launch_v1.json').read_bytes())
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


@pytest.mark.parametrize('fault',['old-control','old-auth','missing-recipe','recipe-sha','frozen-source'])
def test_v4_scope_cannot_reuse_consumed_authority_or_change_recipe_bindings(package,fault):
    x=package;c=copy.deepcopy(x.c)
    if fault=='old-auth':
        with pytest.raises(ValueError):outer.validate_authorization((OLD/'AUTHORIZATION.json').read_bytes(),c,x.sha)
        return
    if fault=='old-control':c['kind']='pgd-split-outer-control-v3'
    elif fault=='missing-recipe':c['local_source_sha256'].pop('models/predictor_pgd_split_preclaim_refusal_v4.py')
    elif fault=='recipe-sha':c['local_source_sha256']['models/predictor_pgd_split_preclaim_refusal_v4.py']='f'*64
    else:c['local_source_sha256']['models/predictor_pgd_split_outer_v3.py']='f'*64
    with pytest.raises(ValueError):outer.validate_control(outer.encoded(c),c['outer_source_sha256'])


def test_whole_remote_function_recipe_verification_refuses_drift():
    outer.verify_refusal_policy_sources(ROOT,source)
    original=(ROOT/'models/predictor_pgd_split_outer_v4.py').read_bytes()
    assert b'if not owner.claimed or owner.driver.guard is None:' in original
    changed=original.replace(b'if not owner.claimed or owner.driver.guard is None:',b'if False:',1)
    def read(path,**kw):return changed if path.name=='predictor_pgd_split_outer_v4.py' else source.read_source(path,**kw)
    with pytest.raises(ValueError,match='exact reviewed whole-function'):outer.verify_refusal_policy_sources(ROOT,NS(read_source=read))


def lifecycle(x,tmp_path,monkeypatch,mode):
    clock=lambda:1000.0;events=[];ownerbox=[];finders=[];receipts=[]
    root=tmp_path/'EXPLICIT-FAKE-REMOTE-ROOT'
    observed=json.loads((OLD/'attended-session/REFUSAL.json').read_bytes())['last_guard_refusal']
    queries=[q['text'] for q in observed['queries']]
    clear=queries[0].replace(', 293,',', 94,')
    def query(args,**kw):
        assert args[0]=='nvidia-smi';events.append('FAKE-GPU-QUERY')
        compute='--query-compute-apps=' in args[1]
        return (queries[1] if mode=='busy' else '') if compute else (queries[0] if mode=='busy' else clear)
    osproxy=NS(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')})
    osproxy.getpid=lambda:424242;osproxy.urandom=lambda n:b'0'*n
    osproxy.posix_fallocate=lambda *a:events.append('FAKE-RESERVE-NO-ALLOCATION')
    def fstat(fd):
        s=os.fstat(fd);p=root/'emergency-space.bin'
        if p.exists() and s.st_ino==p.stat().st_ino:return NS(st_size=16*outer.MiB,st_blocks=16*outer.MiB//512)
        return s
    osproxy.fstat=fstat
    monkeypatch.setattr(outer,'os',osproxy)
    monkeypatch.setattr(coordinator,'target_gate',lambda *a:events.append('FAKE-SETUP-HOST'))
    def verify(self,**kw):
        self.scan(phase='setup',settled=True);events.append('FAKE-ASSET-CONTENTS-REAL-SCAN')
        assert len(self.layout.assets)==172 and len(self.layout.result)==419
        assert len(outer._setup_directory_plan(self.layout)['setup_directories'])==26
        return {'source_only':True}
    monkeypatch.setattr(scratch.Guard,'verify_assets',verify)
    monkeypatch.setattr(scratch.Guard,'proof_readiness',lambda *a:{'proof_launch_authorized':False})
    monkeypatch.setattr(outer,'receive_packet',lambda *a:events.append('FAKE-PACKET-RECEIVE'))
    monkeypatch.setattr(outer,'extract_packet',lambda *a:events.append('FAKE-PACKET-EXTRACT'))
    realboot=outer.boot_namespace
    def cached_only(sources):
        b=realboot(sources)
        cls=b.PinnedImporter
        class CacheOnly:
            def __init__(self,*a):
                assert not (root/'code').exists()
                finder=cls(*a);assert len(finder.files)==69 and len(finder.packet_imports)==2;finders.append(finder)
            def install(self):return finders[-1]
        b.PinnedImporter=CacheOnly
        return b
    # Actual generated coordinator definitions; only stdlib supervisors import.
    ns={'__name__':'EXPLICIT_STDLIB_GENERATED_PARENT','__file__':'<CHECKED-GENERATED-PARENT>'}
    exec(compile(x.sources['supervisor/predictor_pgd_split_coordinator_v1.py'],ns['__file__'],'exec'),ns)
    bootproxy=NS(**boot.__dict__);bootproxy.require_namespace=lambda *a:events.append('FAKE-NAMESPACE-ACTIVATION')
    ns.update(boot=bootproxy,os=osproxy,_outer_threading=NS(get_ident=lambda:1),
        subprocess=NS(check_output=query,PIPE=object()),signal=NS(getsignal=lambda *a:signal.SIG_DFL,SIGCHLD=signal.SIGCHLD,SIG_DFL=signal.SIG_DFL),
        target_gate=lambda *a:events.append('FAKE-PARENT-HOST'))
    def api():
        events.append('FAKE-PIDFD-API-CHECK')
        if mode=='guard-ready-refusal':raise RuntimeError('INERT PRECLAIM PIDFD API REFUSAL')
        return NS()
    ns['frozen_watch']=NS(system_pidfd_api=api)
    cls=ns['OuterCoordinator']
    def owner(*a,**kw):
        value=cls(*a,**kw,clock=clock);ownerbox.append(value)
        def no_worker(role,plan):
            events.append(('INERT-WORKER-REFUSAL',role));assert role=='proof'
            raise RuntimeError('INERT CLAIMED PROOF-WORKER REFUSAL; NO PROOF EXECUTED')
        value.driver.launch=no_worker
        return value
    realprepare=outer.prepare_tree
    def prepare(*a):
        _,s,g,b,derived=realprepare(*a)
        return NS(OuterCoordinator=owner),s,g,b,derived
    class Alarm:
        def arm(self,deadline,*,hard_deadline,phase):events.append(('FAKE-ALARM',phase,deadline,hard_deadline))
        def check(self):pass
        def close(self):events.append('FAKE-ALARM-CLOSED')
    class Collector:
        def __init__(self,guard,control,auth,*,connection_started):
            events.append('INERT-COLLECTOR-CONSTRUCTOR')
            assert ownerbox[0].claimed and guard is ownerbox[0].driver.guard and guard is not None
            self.guard=guard;self.control=control;self.auth=auth
            self.window=NS(connection_started=connection_started,started=1001.0,deadline=1601.0)
        def prepare(self):
            events.append('INERT-COLLECTOR-PREPARE')
            # Original saved ended-work predicate, with the parent's fake PID.
            original=FunctionType(collection.work_receipt.__code__,dict(collection.work_receipt.__globals__,os=osproxy))
            raw=self.guard.read('session.json',128*1024)
            receipt=original(raw,outer.digest(self.control),outer.digest(self.auth),self.guard.layout.installation_sha256,self.window)
            assert receipt['all_saved_role_checks_returned'] is False;receipts.append(receipt)
        def stream(self,out):events.append('INERT-COLLECTOR-STREAM');out.write(b'EXPLICIT-INERT-COLLECTION-NO-TENSORS')
    def importing(name,globals=None,locals=None,fromlist=(),level=0):
        assert name=='models' and fromlist==('predictor_pgd_split_collection_v1',)
        events.append('INERT-COLLECTOR-IMPORT');return NS(predictor_pgd_split_collection_v1=NS(Collector=Collector))
    g=dict(outer.__dict__);g.update(time=NS(monotonic=clock),os=osproxy,Alarms=Alarm,boot_namespace=cached_only,
        preflight_parent=lambda:events.append('FAKE-OUTER-HOST'),remote_root=lambda *a:root,prepare_tree=prepare,
        __builtins__=dict(vars(builtins),__import__=importing))
    fn=next(n for n in ast.parse(x.code).body if isinstance(n,ast.FunctionDef) and n.name=='remote_session')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<ACTUAL-V4-SERVER-FAKE-OS>','exec'),g)
    message={'nonce':(b'0'*32).hex(),'client_elapsed':1.0,'control_text':x.raw.decode(),'authorization_text':x.auth.decode(),'sources':x.payload}
    incoming=io.BytesIO();outer.send_frame(incoming,message,lambda:None,cap=2*outer.MiB);incoming.seek(0);outgoing=io.BytesIO()
    status=g['remote_session'](incoming,outgoing,code_sha256=outer.digest(x.code))
    stream=io.BytesIO(outgoing.getvalue());frames=[outer.frame(stream,lambda:None) for _ in range(3)];tail=stream.read()
    assert status==1 and frames[0]['kind']=='pgd-split-hello-v1' and frames[1]['outer_control_sha256']==x.sha
    assert len(finders)==1 and not finders[0].loaded and not finders[0].installed
    assert (root/'emergency-space.bin').stat().st_size==0
    assert len(ownerbox)==1 and ownerbox[0].driver.outer_journal.closed and ownerbox[0].driver.outer_journal.fd is None
    assert ownerbox[0].driver.outer_child is None and ownerbox[0].driver.outer_lease is None
    return NS(root=root,events=events,frames=frames,tail=tail,owner=ownerbox[0],receipts=receipts,mode=mode)


def test_complete_v4_busy_GPU_route_preserves_actual_journal_and_skips_uninitialized_collection(package,tmp_path,monkeypatch):
    x=lifecycle(package,tmp_path,monkeypatch,'busy');reply=x.frames[-1]
    assert reply['kind']=='pgd-split-outer-refusal-v1' and reply['error_type']=='ValueError'
    assert reply['message']=='Another CUDA compute process is active; leave it alone and refuse.'
    assert reply['last_guard_refusal']==json.loads((OLD/'attended-session/REFUSAL.json').read_bytes())['last_guard_refusal']
    assert x.owner.claimed is False and x.owner.driver.guard is None and not (x.root/'session.json').exists()
    assert not x.tail and not any('COLLECTOR' in str(e) or 'WORKER' in str(e) for e in x.events)
    journal=[json.loads(v) for v in (x.root/'gpu-samples.jsonl').read_bytes().splitlines()]
    assert len(journal)==1 and journal[0]['refusal']==reply['last_guard_refusal']
    (tmp_path/'WHOLE-V4-BUSY-GPU-REFUSAL.json').write_bytes(outer.encoded({'real_source_catalogue_guard_journal_coordinator_and_codec':True,
        'reply':reply,'events':x.events,'parent_claimed':False,'proof_or_scientific_worker_executed':False,'actual_process_timer_GPU_network':False}))


def test_complete_v4_refuses_after_guard_creation_but_before_session_claim(package,tmp_path,monkeypatch):
    x=lifecycle(package,tmp_path,monkeypatch,'guard-ready-refusal');reply=x.frames[-1]
    assert reply['kind']=='pgd-split-outer-refusal-v1' and reply['message']=='INERT PRECLAIM PIDFD API REFUSAL'
    assert x.owner.driver.guard is not None and x.owner.claimed is False and not (x.root/'session.json').exists()
    assert reply['last_guard_refusal'] is None and not x.tail
    assert not any('COLLECTOR' in str(e) or 'WORKER' in str(e) for e in x.events)


def test_complete_v4_keeps_claimed_failure_collection_and_original_saved_receipt_gates(package,tmp_path,monkeypatch):
    x=lifecycle(package,tmp_path,monkeypatch,'claimed-proof-refusal');ready=x.frames[-1]
    assert x.owner.claimed is True and x.owner.driver.guard is not None
    assert ready['kind']=='pgd-split-collection-ready-v1' and ready['collection_deadline']==1601.0
    assert x.tail==b'EXPLICIT-INERT-COLLECTION-NO-TENSORS' and len(x.receipts)==1
    assert x.receipts[0]['work_phase']=='failed' and x.receipts[0]['roles']==[]
    state=json.loads((x.root/'session.json').read_bytes())
    assert state['complete'] is False and state['completed']==[] and state['current']['role']=='proof'
    assert state['error']=='INERT CLAIMED PROOF-WORKER REFUSAL; NO PROOF EXECUTED'
    assert 'parent_error_diagnostic' in state
    assert len([e for e in x.events if isinstance(e,tuple) and e[0]=='INERT-WORKER-REFUSAL'])==1


def test_v4_remote_recipe_and_unaffected_function_client_and_CLI_sources_are_exact():
    original=(ROOT/'models/predictor_pgd_split_outer_v3.py').read_bytes();expected,_=recipe.build(original)
    new=ast.parse((ROOT/'models/predictor_pgd_split_outer_v4.py').read_bytes())
    functions=lambda t:{n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=functions(ast.parse(original)),functions(new)
    assert ast.dump(b['remote_session'],include_attributes=False)==ast.dump(ast.parse(expected).body[0],include_attributes=False)
    changed={'remote_session','validate_control','authorization','verify_directory_policy_sources','prepare_package'}
    for name in set(a)-changed:assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False),name
    assert set(b)-set(a)=={'verify_refusal_policy_sources'}
    for old,new,before,after in [('models/predictor_pgd_split_outer_client_v3.py','models/predictor_pgd_split_outer_client_v4.py',
      b'predictor_pgd_split_outer_v3 as outer',b'predictor_pgd_split_outer_v4 as outer'),
      ('scripts/run_predictor_pgd_split_outer_v3.py','scripts/run_predictor_pgd_split_outer_v4.py',
      b'predictor_pgd_split_outer_client_v3 import run',b'predictor_pgd_split_outer_client_v4 import run')]:
        assert (ROOT/new).read_bytes().replace(after,before)==(ROOT/old).read_bytes()


def test_unarmed_v4_entry_refuses_before_client_timer_attempt_or_SSH(package):
    spec=importlib.util.spec_from_file_location('EXPLICIT_UNARMED_V4',ROOT/'scripts/run_predictor_pgd_split_outer_v4.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(SystemExit,match='No timer, claim, SSH or model work'):module.main([str(package.folder)])
    assert not (package.folder/'AUTHORIZATION.json').exists() and not (package.folder/'ATTEMPT.json').exists()
