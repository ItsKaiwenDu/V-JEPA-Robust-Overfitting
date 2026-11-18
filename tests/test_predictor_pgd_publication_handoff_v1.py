"""New scanner handoff bindings; all target/worker/transport operations inert."""
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
from models import predictor_pgd_publication_handoff_v1 as binding
from models import predictor_pgd_publication_outer_v1 as outer
from models import predictor_pgd_publication_client_v1 as client
from models import predictor_pgd_publication_bootstrap_v1 as boot
from models import predictor_pgd_publication_bound_scratch_v1 as scratch
from models import predictor_pgd_publication_collection_v1 as collection
from models import predictor_pgd_split_coordinator_v1 as coordinator
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def package(tmp_path_factory):
 parent=tmp_path_factory.mktemp('EXPLICIT-UNARMED-PUBLICATION-HANDOFF');parent.chmod(0o700);folder=parent/'unarmed'
 result=outer.prepare_package(ROOT,folder);raw=(folder/'CONTROL-CANDIDATE.json').read_bytes();code=(ROOT/binding.NEW['outer']).read_bytes()
 c,b,p,sha=outer.validate_control(raw,outer.digest(code));payload=outer.parsed((folder/'SOURCE-PAYLOAD.json').read_bytes());sources=outer.decode_sources(payload,c,b)
 auth=outer.encoded(outer.authorization(c,sha,'EXPLICIT INERT LOCAL TEST ONLY; NOT A UNIVERSITY APPROVAL'))
 return NS(folder=folder,result=result,raw=raw,code=code,c=c,b=b,p=p,sha=sha,payload=payload,sources=sources,auth=auth)


def test_new_operators_have_exact_byte_inverses_and_unchanged_full_execution_logic(package):
 generated,meta=binding.build_local(ROOT)
 for n,data in generated.items():assert (ROOT/n).read_bytes()==data
 old=(ROOT/binding.OLD['outer']).read_bytes();new=generated[binding.NEW['outer']]
 functions=lambda v:{n.name:n for n in ast.parse(v).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 a,b=functions(old),functions(new);assert a.keys()==b.keys()
 for n in a.keys()-{'validate_control','authorization','verify_directory_policy_sources','verify_refusal_policy_sources','prepare_package'}:
  assert ast.dump(a[n],include_attributes=False)==ast.dump(b[n],include_attributes=False),n
 restored=new.replace(b'from models import predictor_pgd_publication_handoff_v1 as binding',b'from models import predictor_pgd_host10614_v1 as binding',1)
 e=meta['local_source_assignment'];restored=restored[:e['begin']]+e['old_value'].encode()+restored[e['new_end']:]
 parts=[];cursor=0
 for e in meta['outer_literal_edits']:parts.extend((restored[cursor:e['new_begin']],e['old_literal'].encode()));cursor=e['new_end']
 assert b''.join(parts+[restored[cursor:]])==old
 data=generated[binding.NEW['client']]
 for a,b in [(b'predictor_pgd_publication_outer_v1 as outer',b'predictor_pgd_host10614_outer_v1 as outer'),(b'predictor_pgd_publication_bootstrap_v1 as boot',b'predictor_pgd_host10614_bootstrap_v1 as boot'),(b'predictor_pgd_publication_collection_v1 as collection',b'predictor_pgd_host10614_collection_v1 as collection')]:data=data.replace(a,b,1)
 assert data==(ROOT/binding.OLD['client']).read_bytes()
 assert generated[binding.NEW['cli']].replace(b'from models.predictor_pgd_publication_client_v1 import run',b'from models.predictor_pgd_host10614_client_v1 import run',1)==(ROOT/binding.OLD['cli']).read_bytes()
 assert meta['local_source_files']==56


def test_complete_handoff_is_exact_reviewed_bundle_with_unmodified_scope_and_no_grant(package):
 x=package;control,sources,readers,manifest,meta=binding.integration.assemble(ROOT)
 assert x.sources==sources and x.c['supervised_control_text'].encode()==control
 assert len(sources)==69 and sum(map(len,sources.values()))==888096 and len(x.c['local_source_sha256'])==56
 oldp=binding.integration.load_frozen(ROOT)[2]
 for name in ('scope','base_text','design_text','host10614_binding','runtime_repair'):assert x.p[name]==oldp[name]
 assert x.p['publication_scan_repair']['bound_scanner_sha256']==outer.digest(sources[binding.integration.SCRATCH])
 assert set(p.name for p in x.folder.iterdir())=={'CONTROL-CANDIDATE.json','SOURCE-PAYLOAD.json','PACKAGE.json','known-hosts'}
 assert not x.result['authorization_created'] and not x.result['target_work_or_SSH_executed']
 assert not any(n in sys.modules for n in ('torch','numpy'))

@pytest.mark.parametrize('fault',['old-control','old-auth','old-bootstrap','missing-pin','refusal-pin','source-scanner','source-bootstrap'])
def test_old_authority_or_incomplete_scanner_source_coverage_refuses(package,fault):
 x=package;c=copy.deepcopy(x.c)
 if fault=='old-auth':
  old=ROOT/binding.integration.FOLDER/'AUTHORIZATION.json'
  with pytest.raises(ValueError):outer.validate_authorization(old.read_bytes(),c,x.sha)
  return
 if fault.startswith('source-'):
  name=binding.integration.SCRATCH if fault=='source-scanner' else binding.integration.BOOT
  payload=dict(x.payload);data=binding.integration.load_frozen(ROOT)[3][name];payload[name]=base64.b64encode(data).decode()
  with pytest.raises(ValueError,match='source bytes/extent'):outer.decode_sources(payload,c,x.b)
  return
 if fault=='old-control':c['kind']='pgd-host10614-outer-control-v1'
 elif fault=='old-bootstrap':
  b=json.loads(c['supervised_control_text']);b['supervisor_sources']['bootstrap.py']['sha256']='9d865b43e4f841213963005d85c4cac4e261f62d726353fb78cd52ad48589b2c'
  c['supervised_control_text']=outer.encoded(b).decode();c['supervised_control_sha256']=outer.digest(outer.encoded(b))
 elif fault=='missing-pin':c['local_source_sha256'].pop('models/predictor_pgd_publication_scan_v1.py')
 else:c['local_source_sha256']['models/predictor_pgd_split_preclaim_refusal_v4.py']='f'*64
 with pytest.raises(ValueError):outer.validate_control(outer.encoded(c),c['outer_source_sha256'])


def test_missing_approval_refuses_cli_and_client_before_clock_claim_timer_or_SSH(package,monkeypatch):
 spec=importlib.util.spec_from_file_location('EXPLICIT-PUBLICATION-CLI',ROOT/binding.NEW['cli']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 def forbidden(*a,**kw):raise AssertionError('Unarmed cannot hash packet, create timer/claim or launch process')
 monkeypatch.setattr(client,'file_bytes',forbidden);monkeypatch.setattr(outer,'Alarms',forbidden)
 monkeypatch.setattr(client,'time',NS(monotonic=lambda:1000.))
 with pytest.raises(SystemExit,match='Unarmed'):module.main([str(package.folder)])
 with pytest.raises(ValueError,match='Unarmed'):client.check_local(package.folder)
 with pytest.raises(ValueError,match='Unarmed'):client.run(package.folder)
 assert not (package.folder/'ATTEMPT.json').exists()


def local_case(tmp_path,monkeypatch):
 parent=tmp_path/'parent';parent.mkdir(mode=0o700);folder=parent/'candidate';outer.prepare_package(ROOT,folder)
 raw=(folder/'CONTROL-CANDIDATE.json').read_bytes();c,b,p,sha=outer.validate_control(raw,outer.digest((ROOT/binding.NEW['outer']).read_bytes()))
 auth=outer.encoded(outer.authorization(c,sha,'EXPLICIT INERT TEST FIXTURE ONLY'))
 outer.private_write(folder,'AUTHORIZATION.json',auth,lambda:None)
 real=client.file_bytes;seen=[];packet=ROOT/outer.PACKET_PATH
 def file_bytes(path,cap,sha256=None):
  if Path(path)==packet:seen.append((cap,sha256));return b'EXPLICIT PACKET BYTE-HASH STUB; NO PIXELS READ'
  return real(path,cap,sha256)
 monkeypatch.setattr(client,'file_bytes',file_bytes)
 return folder,sha,seen


def test_real_local_source_preflight_uses56pins_and_refuses_copied_folder_before_packet_rehash(tmp_path,monkeypatch):
 folder,sha,seen=local_case(tmp_path,monkeypatch);checked=client.check_local(folder)
 assert checked[3]==sha and len(checked[0]['local_source_sha256'])==56 and len(seen)==1
 assert seen[0]==(17192095,'2278bce2b519943c0f83a49c556bb73dd928fe880fba0f6c56de2a4c3c2ec0da')
 import shutil
 copied=folder.parent/'copied';shutil.copytree(folder,copied)
 with pytest.raises(ValueError,match='Copied helpers'):client.check_local(copied)
 assert len(seen)==1 and not (folder/'ATTEMPT.json').exists()


def test_already_consumed_counter_cannot_create_second_SSH_or_change_old_attempt(tmp_path,monkeypatch):
 folder,sha,seen=local_case(tmp_path,monkeypatch);old=b'EXPLICIT ALREADY CONSUMED LOCAL FIXTURE\n';outer.private_write(folder,'ATTEMPT.json',old,lambda:None)
 class Alarm:
  closed=False
  def arm(self,*a,**kw):pass
  def check(self):return 1000.
  def close(self):self.closed=True
 alarm=Alarm();monkeypatch.setattr(outer,'Alarms',lambda:alarm);monkeypatch.setattr(client,'time',NS(monotonic=lambda:1000.))
 def forbidden(*a,**kw):raise AssertionError('Consumed attempt must not start SSH')
 monkeypatch.setattr(client,'subprocess',NS(Popen=forbidden))
 with pytest.raises(FileExistsError):client.run(folder)
 assert alarm.closed and (folder/'ATTEMPT.json').read_bytes()==old and not (folder/'attended-session').exists()



# Fresh declarations of pure/inert helpers; no old test functions are imported or executed.
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

@pytest.mark.parametrize('fault',['none','root','clock','scanner-pin','authority','reserve-shape'])
def test_new_client_header_pins_root_clock_scanner_and_strict_collection_report(package,tmp_path,fault):
 h,derived,claimed,ready=full_header(package)
 if fault=='root':claimed['scratch']+='-other'
 elif fault=='clock':ready['collection_started']=101.
 elif fault=='scanner-pin':h['files'][binding.integration.SCRATCH]['sha256']=binding.integration.scanner.FROZEN_SHA
 elif fault=='authority':h['control_sha256']='f'*64
 elif fault=='reserve-shape':h['storage']['publication_accounting']={'EXPLICIT':'capture-only must not enter collection'}
 raw=outer.encoded(h);wire=collection.MAGIC+len(raw).to_bytes(4,'big')+raw+b'EXPLICIT TAIL';window=NS(check=lambda:None)
 if fault=='none':
  source=client.bound_collection_source(io.BytesIO(wire),window,claimed,ready,package.c,derived)
  assert source.read(len(wire))==wire[:-len(b'EXPLICIT TAIL')] and source.read(100)==b'EXPLICIT TAIL'
  (tmp_path/'INERT-PUBLICATION-HANDOFF-HEADER.json').write_bytes(raw)
 else:
  with pytest.raises(ValueError):client.bound_collection_source(io.BytesIO(wire),window,claimed,ready,package.c,derived)


def test_source_only_bootstrap_and_directory_plan_bind_full172assets419results(package):
 namespace=outer.boot_namespace(package.sources);c,p,sha=namespace.control(package.c['supervised_control_text'].encode());ib,isha=namespace.installation(c,sha)
 layout=scratch.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),ib,isha)
 assert sha==package.c['supervised_control_sha256'] and len(layout.assets)==172 and len(layout.result)==419
 assert len(outer._setup_directory_plan(layout)['setup_directories'])==26
 assert p['publication_scan_repair']['rules']['maximum_resolutions_per_scan']==2
 assert not any(n in sys.modules for n in ('torch','numpy'))


def test_cold_inline_directory_and_refusal_policy_verifiers_reject_whole_function_drift():
 from models import predictor_pgd_split_source_bundle_v1 as source
 outer.verify_directory_policy_sources(ROOT,source);outer.verify_refusal_policy_sources(ROOT,source)
 raw=(ROOT/binding.NEW['outer']).read_bytes();changed=raw.replace(b'if not owner.claimed or owner.driver.guard is None:',b'if False:',1)
 fake=NS(read_source=lambda path,**kw:changed if Path(path)==ROOT/binding.NEW['outer'] else source.read_source(path,**kw))
 with pytest.raises(ValueError,match='exact reviewed whole-function'):outer.verify_refusal_policy_sources(ROOT,fake)


def lifecycle(x,tmp_path,monkeypatch,mode):
    clock=lambda:1000.0;events=[];ownerbox=[];finders=[];receipts=[]
    root=tmp_path/'EXPLICIT-FAKE-REMOTE-ROOT'
    clear='NVIDIA GeForce RTX 3070, GPU-c7a30a6f-08fb-371f-41b3-86a13a1a632d, 8192, 94, 580.178.04\n'
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
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<ACTUAL-PUBLICATION-SERVER-FAKE-OS>','exec'),g)
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
def test_integrated_full_setup_cache_claim_and_collection_refusal_paths_with_inert_worker(package,tmp_path,monkeypatch,mode):
 x=lifecycle(package,tmp_path,monkeypatch,mode);reply=x.frames[-1]
 if mode=='busy':
  assert reply['kind']=='pgd-split-outer-refusal-v1' and 'Another CUDA compute process' in reply['message']
  assert not x.owner.claimed and not x.tail and not x.receipts
  assert not any('COLLECTOR' in str(e) or 'WORKER' in str(e) for e in x.events)
 else:
  assert reply['kind']=='pgd-split-collection-ready-v1' and x.owner.claimed and x.receipts[0]['work_phase']=='failed'
  assert x.receipts[0]['all_saved_role_checks_returned'] is False and x.tail==b'EXPLICIT-INERT-COLLECTION-NO-TENSORS'
 for name in (binding.integration.SCRATCH,'overlay/models/predictor_pgd_split_store_v1.py','supervisor/predictor_pgd_split_watch_v1.py'):
  assert (x.root/name).read_bytes()==package.sources[name]
 assert x.owner.driver.outer_child is None and x.owner.driver.outer_lease is None
 (tmp_path/'INERT-PUBLICATION-HANDOFF-LIFECYCLE.json').write_bytes(outer.encoded({'mode':mode,'reply':reply,'events':x.events,
  'source_cache_setup_directory_journal_and_claim_real':True,'proof_worker_process_timer_GPU_and_packet_bytes_are_fakes':True,
  'new_model_or_gradient_or_numerical_work':False,'bound_scanner_sha256':outer.digest(package.sources[binding.integration.SCRATCH])}))
