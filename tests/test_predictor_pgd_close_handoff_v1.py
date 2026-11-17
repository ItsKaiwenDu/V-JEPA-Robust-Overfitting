"""Focused repaired handoff bindings; target/worker/clock/transport stay inert."""
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
from models import predictor_pgd_close_handoff_v1 as binding
from models import predictor_pgd_close_outer_v1 as outer
from models import predictor_pgd_close_client_v1 as client
from models import predictor_pgd_close_bootstrap_v1 as boot
from models import predictor_pgd_close_scratch_v1 as scratch
from models import predictor_pgd_close_collection_v1 as collection
from models import predictor_pgd_split_coordinator_v1 as coordinator

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def package(tmp_path_factory):
    parent=tmp_path_factory.mktemp('EXPLICIT-UNARMED-CLOSE-HANDOFF');parent.chmod(0o700)
    folder=parent/'unarmed';result=outer.prepare_package(ROOT,folder)
    raw=(folder/'CONTROL-CANDIDATE.json').read_bytes();code=(ROOT/binding.NEW['outer']).read_bytes()
    c,b,p,sha=outer.validate_control(raw,outer.digest(code));mapping=outer.parsed((folder/'SOURCE-PAYLOAD.json').read_bytes())
    sources=outer.decode_sources(mapping,c,b)
    auth=outer.encoded(outer.authorization(c,sha,'EXPLICIT INERT LOCAL HANDOFF TEST ONLY; NOT A UNIVERSITY GRANT'))
    return NS(folder=folder,result=result,raw=raw,code=code,c=c,b=b,p=p,sha=sha,payload=mapping,sources=sources,auth=auth)


def test_cold_derivation_exact_remote_session_and_import_inverses(package):
    generated,meta=binding.build_local(ROOT)
    for name,data in generated.items():assert (ROOT/name).read_bytes()==data
    original=(ROOT/binding.OLD['outer']).read_bytes();new=generated[binding.NEW['outer']]
    functions=lambda src:{n.name:n for n in ast.parse(src).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=functions(original),functions(new)
    assert a.keys()==b.keys()
    for name in a.keys()-{'validate_control','authorization','verify_directory_policy_sources','verify_refusal_policy_sources','prepare_package'}:
        assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False),name
    assert ast.dump(a['remote_session'],include_attributes=False)==ast.dump(b['remote_session'],include_attributes=False)
    restored=new.replace(b'from models import predictor_pgd_close_handoff_v1 as binding',b'from models import predictor_pgd_host10609_v1 as binding',1)
    change=meta['local_source_assignment'];restored=restored[:change['begin']]+change['old_value'].encode()+restored[change['new_end']:]
    pieces=[];cursor=0
    for e in meta['outer_literal_edits']:
        pieces.extend((restored[cursor:e['new_begin']],e['old_literal'].encode()));cursor=e['new_end']
    pieces.append(restored[cursor:]);assert b''.join(pieces)==original
    oldclient=(ROOT/binding.OLD['client']).read_bytes();newclient=generated[binding.NEW['client']]
    for before,after in [(b'predictor_pgd_split_outer_10609_v1 as outer',b'predictor_pgd_close_outer_v1 as outer'),
                         (b'predictor_pgd_split_bootstrap_10609_v1 as boot',b'predictor_pgd_close_bootstrap_v1 as boot'),
                         (b'predictor_pgd_split_collection_10609_v1 as collection',b'predictor_pgd_close_collection_v1 as collection')]:newclient=newclient.replace(after,before,1)
    assert newclient==oldclient and meta['local_source_files']==34


def test_complete_package_binds_repairs_and_preserves_original_scientific_limits(package):
    x=package;reviewed=binding.integration.assemble(ROOT)
    assert x.c['supervised_control_text'].encode()==reviewed[0]
    assert x.sources==reviewed[1] and len(x.sources)==69 and sum(map(len,x.sources.values()))==878631
    assert x.c['kind']==binding.CONTROL_KIND and len(x.c['local_source_sha256'])==34
    old=binding.integration.load_frozen(ROOT)[2]
    assert x.p['scope']==old['scope'] and x.p['base_text']==old['base_text'] and x.p['design_text']==old['design_text']
    assert set(p.name for p in x.folder.iterdir())=={'CONTROL-CANDIDATE.json','SOURCE-PAYLOAD.json','known-hosts','PACKAGE.json'}
    assert not x.result['authorization_created'] and not x.result['target_work_or_SSH_executed']
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


@pytest.mark.parametrize('fault',['old-control','old-auth','old-bootstrap','missing-new-pin','refusal-pin','source-store','source-watch'])
def test_new_handoff_rejects_stale_scope_sources_or_local_coverage(package,fault):
    x=package;c=copy.deepcopy(x.c)
    if fault=='old-auth':
        old=ROOT/binding.integration.FOLDER/'AUTHORIZATION.json'
        with pytest.raises(ValueError):outer.validate_authorization(old.read_bytes(),c,x.sha)
        return
    if fault.startswith('source-'):
        mapping=dict(x.payload);name=binding.integration.STORE if fault=='source-store' else binding.integration.WATCH
        mapping[name]=base64.b64encode(b'EXPLICIT INVALID SOURCE').decode()
        with pytest.raises(ValueError,match='source bytes/extent'):outer.decode_sources(mapping,c,x.b)
        return
    if fault=='old-control':c['kind']='pgd-split-outer-control-host10609-v1'
    elif fault=='old-bootstrap':
        b=json.loads(c['supervised_control_text']);b['supervisor_sources']['bootstrap.py']['sha256']='d0b7933dc001bb603c38bf9b7aaa9488080cb3d0f2661145886a570f19a870fb'
        c['supervised_control_text']=outer.encoded(b).decode();c['supervised_control_sha256']=outer.digest(outer.encoded(b))
    elif fault=='missing-new-pin':c['local_source_sha256'].pop('models/predictor_pgd_exit_race_v1.py')
    elif fault=='refusal-pin':c['local_source_sha256']['models/predictor_pgd_split_preclaim_refusal_v4.py']='f'*64
    with pytest.raises(ValueError):outer.validate_control(outer.encoded(c),c['outer_source_sha256'])


def test_unarmed_cli_and_client_refuse_before_source_or_process_work(package,monkeypatch):
    spec=importlib.util.spec_from_file_location('EXPLICIT-CLOSE-CLI',ROOT/binding.NEW['cli']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def forbidden(*a,**kw):raise AssertionError('Unarmed cannot scan/hash packet, create timer or launch process')
    monkeypatch.setattr(client,'file_bytes',forbidden);monkeypatch.setattr(outer,'Alarms',forbidden)
    with pytest.raises(SystemExit,match='Unarmed'):module.main([str(package.folder)])
    with pytest.raises(ValueError,match='Unarmed'):client.check_local(package.folder)
    with pytest.raises(ValueError,match='Unarmed'):client.run(package.folder)
    assert not (package.folder/'ATTEMPT.json').exists()


def test_local_preflight_exact_new_sources_with_inert_grant_and_packet_hash_stub(tmp_path,monkeypatch):
    parent=tmp_path/'parent';parent.mkdir(mode=0o700);folder=parent/'candidate';outer.prepare_package(ROOT,folder)
    raw=(folder/'CONTROL-CANDIDATE.json').read_bytes();c,b,p,sha=outer.validate_control(raw,outer.digest((ROOT/binding.NEW['outer']).read_bytes()))
    auth=outer.encoded(outer.authorization(c,sha,'EXPLICIT INERT LOCAL GRANT ONLY'))
    outer.private_write(folder,'AUTHORIZATION.json',auth,lambda:None)
    real=client.file_bytes;seen=[];packet=ROOT/outer.PACKET_PATH
    def file_bytes(path,cap,sha256=None):
        if Path(path)==packet:
            seen.append((cap,sha256));return b'EXPLICIT PACKET BYTE-HASH STUB; NO PIXELS READ'
        return real(path,cap,sha256)
    monkeypatch.setattr(client,'file_bytes',file_bytes)
    result=client.check_local(folder);assert result[3]==sha and len(seen)==1
    assert seen[0]==(17192095,'2278bce2b519943c0f83a49c556bb73dd928fe880fba0f6c56de2a4c3c2ec0da')
    import shutil
    target=parent/'copied';shutil.copytree(folder,target)
    with pytest.raises(ValueError,match='Copied helpers'):client.check_local(target)
    assert len(seen)==1 and not (folder/'ATTEMPT.json').exists()


def full_header(x):
    c,p,sha=boot.control(x.c['supervised_control_text'].encode());ib,isha=boot.installation(c,sha)
    derived=boot.encoded({'kind':'pgd-split-supervised-university-exception-v1','approved':True,'control_sha256':sha,'maximum_new_attempts':1,'user_basis':outer.parsed(x.auth)['user_basis']})
    layout=scratch.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),ib,isha)
    extras={'control.json':x.c['supervised_control_text'].encode(),'authorization.json':derived,'session.json':b'EXPLICIT INERT SESSION',
            collection.RECEIPT:b'EXPLICIT INERT ENDED-WORK RECEIPT',collection.STATUS:b'EXPLICIT INERT STATUS'}
    pins=dict(layout.assets);pins.update({n:{'bytes':len(v),'sha256':boot.digest(v)} for n,v in extras.items()})
    uid=os.getuid();files={n:dict(d,mode=0o600,mtime_ns=1000001+i) for i,(n,d) in enumerate(sorted(pins.items()))}
    inventory={n:{'identity':[500,100+i,uid,0o600,1],'bytes':v['bytes'],'allocated_bytes':((v['bytes']+4095)//4096)*4096,'mtime_ns':v['mtime_ns']} for i,(n,v) in enumerate(files.items())}
    inventory['emergency-space.bin']={'identity':[500,900,uid,0o600,1],'bytes':16*scratch.MiB,'allocated_bytes':16*scratch.MiB,'mtime_ns':100}
    names={'.'}
    for n in inventory:names.update(str(q) for q in Path(n).parents)
    dirs={n:{'identity':[500,2000+i,uid,0o700],'allocated_bytes':4096} for i,n in enumerate(sorted(names))}
    storage=layout.assess(inventory,dirs,{'available_bytes':8*scratch.GiB,'available_inodes':2048},phase='collection')
    h={'kind':'pgd-split-evidence-stream-v1','control_sha256':sha,'authorization_sha256':boot.digest(derived),
       'root_binding':{'root':'/var/tmp/vjepa-pgd-split-v1-2170-'+x.sha,'root_identity':list(dirs['.']['identity']),'parent':'/var/tmp','parent_identity':[500,1,uid,0o1777]},
       'files':files,'inventory':inventory,'directories':dirs,'storage':storage,'final_receipt_sha256':boot.digest(extras[collection.RECEIPT]),
       'session_sha256':boot.digest(extras['session.json']),'collection_started':100.,'collection_deadline':700.,'connection_started':0.,'connection_deadline':3000.,
       'logical_bytes':sum(v['bytes'] for v in files.values()),'omitted_content':['emergency-space.bin'],'scientific_qualified':False,'training_admitted':False}
    claimed={'scratch':h['root_binding']['root'],'connection_started':0.}
    ready={'collection_started':100.,'collection_deadline':700.}
    return h,derived,claimed,ready


@pytest.mark.parametrize('fault',['none','root','clock','store-pin','authority'])
def test_new_client_collection_header_binds_exact_root_clock_and_repaired_installation(package,tmp_path,fault):
    h,derived,claimed,ready=full_header(package)
    if fault=='root':claimed['scratch']+='-other'
    elif fault=='clock':ready['collection_started']=101.
    elif fault=='store-pin':h['files']['overlay/models/predictor_pgd_split_store_v1.py']['sha256']=binding.integration.store_recipe.FROZEN_SHA
    elif fault=='authority':h['control_sha256']='f'*64
    raw=outer.encoded(h);wire=collection.MAGIC+len(raw).to_bytes(4,'big')+raw+b'EXPLICIT TAIL'
    window=NS(check=lambda:None)
    if fault=='none':
        source=client.bound_collection_source(io.BytesIO(wire),window,claimed,ready,package.c,derived)
        assert source.read(len(wire))==wire[:-len(b'EXPLICIT TAIL')]
        assert source.read(100)==b'EXPLICIT TAIL'
        (tmp_path/'INERT-CLOSE-HANDOFF-HEADER.json').write_bytes(raw)
    else:
        with pytest.raises(ValueError):client.bound_collection_source(io.BytesIO(wire),window,claimed,ready,package.c,derived)


def test_packet_and_remote_source_are_checked_without_loading_scientific_modules(package):
    contract=outer.parsed(package.p['scratch_contract_text'])
    assert contract['packet']['archive']['bytes']==17192095 and outer.PACKET_PATH==package.c['packet_path']
    namespace=outer.boot_namespace(package.sources)
    checked,p,sha=namespace.control(package.c['supervised_control_text'].encode())
    assert sha==package.c['supervised_control_sha256'] and len(checked['supervisor_sources'])==6
    ib,isha=namespace.installation(checked,sha)
    layout=scratch.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),ib,isha)
    assert len(layout.assets)==172 and len(layout.result)==419 and len(outer._setup_directory_plan(layout)['setup_directories'])==26
    assert not any(n in sys.modules for n in ('torch','numpy'))


def test_cold_policy_verifiers_refuse_remote_session_or_worker_directory_drift():
    from models import predictor_pgd_split_source_bundle_v1 as source
    outer.verify_directory_policy_sources(ROOT,source);outer.verify_refusal_policy_sources(ROOT,source)
    raw=(ROOT/binding.NEW['outer']).read_bytes()
    changed=raw.replace(b'if not owner.claimed or owner.driver.guard is None:',b'if False:',1)
    fake=NS(read_source=lambda path,**kw:changed if Path(path)==ROOT/binding.NEW['outer'] else source.read_source(path,**kw))
    with pytest.raises(ValueError,match='exact reviewed whole-function'):outer.verify_refusal_policy_sources(ROOT,fake)


def lifecycle(x,tmp_path,monkeypatch,mode):
    clock=lambda:1000.0;events=[];ownerbox=[];finders=[];receipts=[]
    root=tmp_path/'EXPLICIT-FAKE-REMOTE-ROOT'
    clear='NVIDIA GeForce RTX 3070, GPU-4e2a7c77-19bd-689f-6c08-60eb774c0d52, 8192, 94, 580.178.04\n'
    queries=[clear,'777, /EXPLICIT-FAKE-COMPUTE, 180\n']
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
    ns.update(boot=bootproxy,scratch=scratch,os=osproxy,_outer_threading=NS(get_ident=lambda:1),
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
    def prepare_import(name,globals=None,locals=None,fromlist=(),level=0):
        if name=='models' and fromlist==('predictor_pgd_split_scratch_v1',):return NS(predictor_pgd_split_scratch_v1=scratch)
        if name=='models' and fromlist==('predictor_pgd_split_coordinator_v1',):return NS(predictor_pgd_split_coordinator_v1=coordinator)
        return builtins.__import__(name,globals,locals,fromlist,level)
    realprepare=FunctionType(outer.prepare_tree.__code__,dict(outer.prepare_tree.__globals__,__builtins__=dict(vars(builtins),__import__=prepare_import)))
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
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<ACTUAL-10609-SERVER-FAKE-OS>','exec'),g)
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



@pytest.mark.parametrize('mode',['busy','claimed-proof-refusal'])
def test_repaired_handoff_actual_setup_cache_claim_and_failed_collection_with_inert_worker(package,tmp_path,monkeypatch,mode):
    x=lifecycle(package,tmp_path,monkeypatch,mode);reply=x.frames[-1]
    if mode=='busy':
        assert reply['kind']=='pgd-split-outer-refusal-v1' and 'Another CUDA compute process' in reply['message']
        assert not x.owner.claimed and not x.tail and not x.receipts
        assert not any('COLLECTOR' in str(e) or 'WORKER' in str(e) for e in x.events)
    else:
        assert reply['kind']=='pgd-split-collection-ready-v1' and x.owner.claimed
        assert x.receipts[0]['work_phase']=='failed' and x.receipts[0]['all_saved_role_checks_returned'] is False
        assert x.tail==b'EXPLICIT-INERT-COLLECTION-NO-TENSORS'
        assert x.owner.driver.outer_child is None and x.owner.driver.outer_lease is None
    assert (x.root/'overlay/models/predictor_pgd_split_store_v1.py').read_bytes()==package.sources[binding.integration.STORE]
    assert (x.root/'supervisor/predictor_pgd_split_watch_v1.py').read_bytes()==package.sources[binding.integration.WATCH]
    (tmp_path/'INERT-REPAIRED-HANDOFF-LIFECYCLE.json').write_bytes(outer.encoded({'mode':mode,'reply':reply,'events':x.events,
       'source_cache_setup_directory_journal_and_claim_real':True,'proof_worker_process_timer_GPU_and_packet_bytes_are_fakes':True,
       'new_model_or_gradient_or_numerical_work':False}))
