"""Fresh full binding checks; actual byte/metadata paths, explicit inert target APIs."""
import ast,base64,builtins,copy,hashlib,io,json,os,signal,stat,sys
from pathlib import Path
from types import FunctionType,ModuleType,SimpleNamespace as NS
import pytest
from models import predictor_pgd_joint_integration_v1 as integration
from models import predictor_pgd_joint_handoff_v1 as binding
from models import predictor_pgd_joint_bootstrap_v1 as boot
from models import predictor_pgd_joint_scratch_v1 as scratch
from models import predictor_pgd_joint_collection_v1 as collection
from models import predictor_pgd_joint_outer_v1 as outer
from models import predictor_pgd_joint_client_v1 as client
from models import predictor_pgd_split_coordinator_v1 as coordinator
ROOT=Path(__file__).resolve().parents[1]
RAW='capture/results/numerical/objective-5/cotangent-14.npy'
DATA=b'INERT BINDING BYTE FIXTURE; NOT AN ARRAY OR EXPERIMENT.\n'


def write(root,name,data):
 p=root/name;p.parent.mkdir(mode=0o700,parents=True,exist_ok=True);q=p.parent
 while q!=root:q.chmod(0o700);q=q.parent
 with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 return p

@pytest.fixture(scope='module')
def package(tmp_path_factory):
 parent=tmp_path_factory.mktemp('INERT-UNARMED-JOINT-PACKAGE');parent.chmod(0o700);folder=parent/'unarmed';result=outer.prepare_package(ROOT,folder)
 raw=(folder/'CONTROL-CANDIDATE.json').read_bytes();code=(ROOT/binding.NEW['outer']).read_bytes();c,b,p,sha=outer.validate_control(raw,outer.digest(code))
 payload=json.loads((folder/'SOURCE-PAYLOAD.json').read_bytes());sources=outer.decode_sources(payload,c,b)
 control=c['supervised_control_text'].encode();bc,policy,bsha=boot.control(control);ib,isha=boot.installation(bc,bsha)
 layout=scratch.Layout(policy['scratch_contract_text'].encode(),policy['base_text'].encode(),policy['design_text'].encode(),ib,isha)
 auth=outer.encoded(outer.authorization(c,sha,'EXPLICIT INERT LOCAL FIXTURE ONLY; NOT TARGET APPROVAL'))
 derived=boot.encoded({'kind':'pgd-split-supervised-university-exception-v1','approved':True,'control_sha256':bsha,'maximum_new_attempts':1,'user_basis':outer.parsed(auth)['user_basis']})
 return NS(folder=folder,result=result,raw=raw,code=code,c=c,b=b,p=p,sha=sha,payload=payload,sources=sources,control=control,bc=bc,bsha=bsha,ib=ib,isha=isha,layout=layout,auth=auth,derived=derived)


def test_complete69_source_bundle_changes_only_three_supervisors_and_preserves_all_scope(package):
 x=package;prior=integration.load_frozen(ROOT);oldp,old=prior[2],prior[3]
 assert len(x.sources)==69 and sum(map(len,x.sources.values()))==923094 and len(x.c['local_source_sha256'])==69
 assert {n for n in old if old[n]!=x.sources[n]}=={integration.BOOT,integration.SCRATCH,integration.COLLECTION}
 for name in ('scope','base_text','design_text','runtime_repair','host10614_binding'):assert x.p[name]==oldp[name]
 ct=boot.parsed(x.p['scratch_contract_text']);oldct=boot.parsed(oldp['scratch_contract_text'])
 assert ct.pop('immutable_raw_publication_observation')==integration.RAW_RULE
 protocol=ct.pop('priority_collection_protocol');assert protocol['source_sha256']==outer.digest(x.sources[integration.COLLECTION]) and ct==oldct
 assert len(x.layout.assets)==172 and len(x.layout.result)==419 and len(outer._setup_directory_plan(x.layout)['setup_directories'])==26
 assert x.p['publication_scan_repair']['journal_and_collector_bytes_unchanged'] is False
 assert x.p['publication_scan_repair']['historical_JSON_only_repair']==oldp['publication_scan_repair']
 assert not x.result['authorization_created'] and set(p.name for p in x.folder.iterdir())=={'CONTROL-CANDIDATE.json','SOURCE-PAYLOAD.json','PACKAGE.json','known-hosts'}
 assert not any(n in sys.modules for n in ('torch','numpy'))


def test_legacy_closure_and_priority_code_have_exact_AST_and_byte_inverses(package):
 control,sources,readers,manifest,meta=integration.assemble(ROOT);old=integration.load_frozen(ROOT)[3];raw=sources[integration.COLLECTION];tree=ast.parse(raw)
 factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_frozen_legacy_scope');legacy=ast.parse(old[integration.COLLECTION]);count=meta['collector_derivation']['legacy_body_nodes']
 assert ast.dump(ast.Module(body=factory.body[:count],type_ignores=[]),include_attributes=False)==ast.dump(legacy,include_attributes=False)
 assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('exec','compile') for n in ast.walk(tree))
 assert meta['collector_derivation']['prototype_inverse_verified'] and meta['collector_derivation']['legacy_import_resolved_lexically_no_extra_project_source']
 for name,data in readers.items():assert (ROOT/name).read_bytes()==data
 local=readers[integration.READERS['collection']]
 for a,b in ((b'predictor_pgd_joint_bootstrap_v1 as boot',b'predictor_pgd_split_bootstrap_v1 as boot'),(b'predictor_pgd_joint_scratch_v1 as scratch',b'predictor_pgd_split_scratch_v1 as scratch')):local=local.replace(a,b,1)
 assert local==raw and collection.legacy.boot is boot and collection.legacy.scratch is scratch
 assert collection.WireReader is collection.legacy.WireReader and collection.Window is collection.legacy.Window
 assert boot.POLICY_SHA==meta['policy_sha256'] and scratch.CONTRACT_SHA==meta['scratch_contract_sha256']


def test_outer_and_client_only_change_reviewed_bindings_nested_header_and_ready_record(package):
 generated,meta=binding.build_local(ROOT)
 for n,data in generated.items():assert (ROOT/n).read_bytes()==data
 old=(ROOT/binding.OLD['outer']).read_bytes();new=generated[binding.NEW['outer']]
 defs=lambda raw:{n.name:n for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 a,b=defs(old),defs(new)
 for name in a.keys()-{'validate_control','authorization','verify_directory_policy_sources','verify_refusal_policy_sources','prepare_package'}:assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False),name
 normalized=new.replace(b'from models import predictor_pgd_joint_handoff_v1 as binding',b'from models import predictor_pgd_publication_handoff_v1 as binding',1)
 e=meta['local_source_assignment'];normalized=normalized[:e['begin']]+e['old_value'].encode()+normalized[e['new_end']:]
 chunks=[];cursor=0
 for e in meta['outer_literal_edits']:chunks.extend((normalized[cursor:e['new_begin']],e['old_literal'].encode()));cursor=e['new_end']
 assert b''.join(chunks+[normalized[cursor:]])==old
 candidate=generated[binding.NEW['client']];e=meta['client_ready_receipt_edit'];candidate=candidate.replace(e['new'].encode(),e['old'].encode(),1)
 e=meta['client_bound_header_edit'];candidate=candidate[:e['begin']]+e['old_method'].encode()+candidate[e['new_end']:]
 for a,b in ((b'predictor_pgd_joint_outer_v1 as outer',b'predictor_pgd_publication_outer_v1 as outer'),(b'predictor_pgd_joint_bootstrap_v1 as boot',b'predictor_pgd_publication_bootstrap_v1 as boot'),(b'predictor_pgd_joint_collection_v1 as collection',b'predictor_pgd_publication_collection_v1 as collection')):candidate=candidate.replace(a,b,1)
 assert candidate==(ROOT/binding.OLD['client']).read_bytes()
 assert generated[binding.NEW['cli']].replace(b'from models.predictor_pgd_joint_client_v1 import run',b'from models.predictor_pgd_publication_client_v1 import run',1)==(ROOT/binding.OLD['cli']).read_bytes()

@pytest.mark.parametrize('fault',['old-control','old-auth','missing-local-pin','bad-refusal-pin','old-collector','old-scanner','old-bootstrap','old-runtime'])
def test_stale_authority_or_any_source_binding_refuses(package,fault):
 x=package;c=copy.deepcopy(x.c)
 if fault=='old-auth':
  old=ROOT/integration.FOLDER/'AUTHORIZATION.json'
  with pytest.raises(ValueError):outer.validate_authorization(old.read_bytes(),c,x.sha)
  return
 if fault.startswith('old-') and fault not in ('old-control',):
  source=integration.load_frozen(ROOT)[3];name={'old-collector':integration.COLLECTION,'old-scanner':integration.SCRATCH,'old-bootstrap':integration.BOOT,'old-runtime':'overlay/models/predictor_pgd_split_store_v1.py'}[fault]
  payload=dict(x.payload);data=source[name]+(b'# DRIFT' if fault=='old-runtime' else b'');payload[name]=base64.b64encode(data).decode()
  with pytest.raises(ValueError):outer.decode_sources(payload,c,x.b)
  return
 if fault=='old-control':c['kind']='pgd-publication-outer-control-v1'
 elif fault=='missing-local-pin':c['local_source_sha256'].pop(integration.PROTOCOL)
 else:c['local_source_sha256']['models/predictor_pgd_split_preclaim_refusal_v4.py']='0'*64
 with pytest.raises(ValueError):outer.validate_control(outer.encoded(c),c['outer_source_sha256'])


class ColdCache:
 """Actual pinned exec_module with an isolated registered-only import namespace."""
 def __init__(self,finder):self.finder=finder;self.modules={};self.dependencies=[]
 def module(self,name):
  if name in self.modules:return self.modules[name]
  if name not in self.finder.files:raise ModuleNotFoundError('No registered project dependency: '+name)
  spec=self.finder.find_spec(name);module=ModuleType(name);module.__spec__=spec;module.__package__=name if self.finder.files[name][2] else name.rpartition('.')[0]
  module.__dict__['__builtins__']=dict(vars(builtins),__import__=self.importing);self.modules[name]=module
  self.finder.exec_module(module);return module
 def importing(self,name,globals=None,locals=None,fromlist=(),level=0):
  if name.split('.')[0] in ('models','attacks','datasets'):
   if level:raise ImportError('No implicit project-relative fallback in this fixture.')
   self.dependencies.append((name,tuple(fromlist)));module=self.module(name)
   for item in fromlist:
    if not hasattr(module,item):setattr(module,item,self.module(name+'.'+item))
   return module if fromlist else self.module(name.split('.')[0])
  return builtins.__import__(name,globals,locals,fromlist,level)

@pytest.mark.parametrize('fault',['none','collector-drift','missing-source'])
def test_actual_source_cache_executes_supervisors_and_all_legacy_dependencies_without_checkout_fallback(package,tmp_path,fault):
 root=tmp_path/'pinned';root.mkdir(mode=0o700)
 for name,data in package.sources.items():write(root,name,data)
 if fault=='collector-drift':
  (root/integration.COLLECTION).write_bytes(package.sources[integration.COLLECTION]+b'# drift')
  with pytest.raises(ValueError):boot.PinnedImporter(root,scratch.identity_binding(root),package.bc,package.p)
  return
 if fault=='missing-source':
  (root/integration.COLLECTION).unlink()
  with pytest.raises(FileNotFoundError):boot.PinnedImporter(root,scratch.identity_binding(root),package.bc,package.p)
  return
 finder=boot.PinnedImporter(root,scratch.identity_binding(root),package.bc,package.p);cache=ColdCache(finder)
 for name in ('models.predictor_pgd_split_bootstrap_v1','models.predictor_pgd_split_scratch_v1','models.predictor_pgd_split_collection_v1','models.predictor_pgd_split_coordinator_v1'):cache.module(name)
 col=cache.modules['models.predictor_pgd_split_collection_v1'];sc=cache.modules['models.predictor_pgd_split_scratch_v1'];bc=cache.modules['models.predictor_pgd_split_bootstrap_v1']
 assert col.boot is bc and col.scratch is sc and col.legacy.boot is bc and col.legacy.scratch is sc
 assert col.legacy.Collector.__init__.__closure__ and col.legacy.validate_header.__closure__
 assert finder.loaded==set(cache.modules) and len(finder.loaded)==7 and not finder.installed
 with pytest.raises(ModuleNotFoundError):cache.importing('models',fromlist=('predictor_pgd_publication_collection_v1',))
 assert not any(n in sys.modules for n in ('torch','numpy'))
 (tmp_path/'ACTUAL-JOINT-COLD-CACHE.json').write_bytes(boot.encoded({'loaded':sorted(finder.loaded),'registered_sources':len(finder.files),'legacy_closure_matches_registered_bootstrap_and_scanner':True,'dependency_calls':cache.dependencies,'isolated_import_shim_not_global_activation':True,'new_model_or_worker_work':False}))


def full_header(x):
 extras={'control.json':x.control,'authorization.json':x.derived,'session.json':b'INERT SESSION METADATA',collection.RECEIPT:b'INERT ENDED-WORK RECEIPT',collection.STATUS:b'INERT STATUS'}
 pins=dict(x.layout.assets);pins.update({n:{'bytes':len(v),'sha256':boot.digest(v)} for n,v in extras.items()})
 files={n:dict(d,mode=0o600,mtime_ns=1000001+i) for i,(n,d) in enumerate(sorted(pins.items()))}
 inventory={n:{'identity':[500,100+i,os.getuid(),0o600,1],'bytes':v['bytes'],'allocated_bytes':((v['bytes']+4095)//4096)*4096,'mtime_ns':v['mtime_ns']} for i,(n,v) in enumerate(files.items())}
 dirs={'.'}
 for n in inventory:dirs.update(str(p) for p in Path(n).parents)
 directories={n:{'identity':[500,2000+i,os.getuid(),0o700],'allocated_bytes':4096} for i,n in enumerate(sorted(dirs))}
 storage=x.layout.assess(inventory,directories,{'available_bytes':8*scratch.GiB,'available_inodes':2048},phase='collection')
 base={'kind':'pgd-split-evidence-stream-v1','control_sha256':x.bsha,'authorization_sha256':boot.digest(x.derived),
  'root_binding':{'root':'/var/tmp/vjepa-pgd-split-v1-2170-'+x.sha,'root_identity':list(directories['.']['identity']),'parent':'/var/tmp','parent_identity':[500,1,0,0o1777]},
  'files':files,'inventory':inventory,'directories':directories,'storage':storage,'final_receipt_sha256':boot.digest(extras[collection.RECEIPT]),'session_sha256':boot.digest(extras['session.json']),
  'connection_started':0.,'collection_started':100.,'collection_deadline':700.,'connection_deadline':3000.,'logical_bytes':sum(v['bytes'] for v in files.values()),'omitted_content':[],**collection.FALSE_FLAGS}
 return collection.make_header(base,x.control,x.derived)

@pytest.mark.parametrize('fault',['none','root','clock','old-magic','source-pin','capture-reserve','reordered','old-authority'])
def test_client_prefix_accepts_only_new_nested_header_root_clock_and_source_binding(package,tmp_path,fault):
 h=full_header(package);claimed={'scratch':h['base']['root_binding']['root'],'connection_started':0.};ready={'collection_started':100.,'collection_deadline':700.}
 if fault=='root':claimed['scratch']+='-wrong'
 elif fault=='clock':ready['collection_started']=101.
 elif fault=='source-pin':h['base']['files'][integration.COLLECTION]['sha256']='0'*64
 elif fault=='capture-reserve':h['base']['storage']['publication_accounting']={'capture_only':True}
 elif fault=='reordered':h['records'][0],h['records'][1]=h['records'][1],h['records'][0]
 elif fault=='old-authority':h['base']['control_sha256']=integration.load_frozen(ROOT)[0]['supervised_control_sha256']
 raw=boot.encoded(h);magic=b'VJEPA-PGD-SPLIT-EVIDENCE-1\n' if fault=='old-magic' else collection.MAGIC
 wire=magic+len(raw).to_bytes(4,'big')+raw+b'INERT TAIL';window=NS(check=lambda:None)
 if fault=='none':
  stream=client.bound_collection_source(io.BytesIO(wire),window,claimed,ready,package.c,package.derived)
  assert stream.read(len(wire))==wire[:-len(b'INERT TAIL')] and stream.read(100)==b'INERT TAIL'
  (tmp_path/'JOINT-NESTED-CLIENT-HEADER.json').write_bytes(raw)
 else:
  with pytest.raises(ValueError):client.bound_collection_source(io.BytesIO(wire),window,claimed,ready,package.c,package.derived)


def fake_inputs(root,layout,p):
 packet={'original-packet.tar.gz':b'INERT PACKET STUB; NO ARCHIVE READ'}
 packet.update({'code/'+name:b'# INERT\n'[:d['bytes']] for name,d in boot.parsed(p['scratch_contract_text'])['packet']['files'].items()})
 for name,data in packet.items():
  if not (root/name).exists():write(root,name,data)
  layout.assets[name]={'bytes':len(data),'sha256':boot.digest(data)}
 return packet


def ended_session(x):return {'kind':'pgd-split-parent-session-v1','control_sha256':x.bsha,'parent_pid':os.getpid(),'connection_started':0.,'work_deadline':2400.,'connection_deadline':3000.,'phase':'failed','complete':False,'current':None,'completed':[],'receipts':[],'error':'INERT WORK FAILURE; no process ran',**collection.FALSE_FLAGS}

@pytest.mark.parametrize('mode',['complete','interrupted'])
def test_bound_paired_codec_uses_actual69_source_bytes_and_preserves_failed_work_prefix(package,tmp_path,monkeypatch,mode):
 x=package;root=tmp_path/'source';root.mkdir(mode=0o700);layout=copy.deepcopy(x.layout)
 for name,data in x.sources.items():write(root,name,data)
 fake_inputs(root,layout,x.p)
 for name,data in [('configs/base.json',x.p['base_text'].encode()),('configs/design.json',x.p['design_text'].encode()),('installation.json',x.ib),('control.json',x.control),('authorization.json',x.derived),('session.json',boot.encoded(ended_session(x))),('emergency-space.bin',b'INERT RESERVE STUB')]:
  if (root/name).exists():(root/name).write_bytes(data)
  else:write(root,name,data)
 for index,name in enumerate((RAW,'capture/results/numerical/objective-6/cotangent-14.npy')):
  p=write(root,name,DATA);os.utime(p,ns=(1234000001+index,1234000001+index))
 monkeypatch.setattr(scratch,'Layout',lambda *a,**kw:layout)
 guard=scratch.Guard(root,scratch.identity_binding(root),layout);sender=collection.Collector(guard,x.control,x.derived,connection_started=0.,clock=lambda:100.)
 assert sender.window is sender.original.window and sender.window.deadline==700.
 h=sender.prepare();wire=io.BytesIO();remote=sender.stream(wire);data=wire.getvalue()
 parent=tmp_path/'receiver';parent.mkdir(mode=0o700)
 if mode=='interrupted':
  first=next(r for r in h['records'] if r['name']==RAW);prefix=collection.RECORD+len(boot.encoded(first)).to_bytes(4,'big')+boot.encoded(first);cut=data.index(prefix)+len(prefix)+7
  with pytest.raises(ValueError):collection.receive(io.BytesIO(data[:cut]),parent/'saved',x.control,x.derived,window=collection.Window(0.,100.,clock=lambda:100.))
  assert (parent/'saved/session.json').read_bytes()==boot.encoded(ended_session(x))
  assert (parent/'saved-transport/HEADER.json').is_file() and not boot.parsed((parent/'saved-transport/PARTIAL-RECEIPT.json').read_bytes())['transport_verified']
  receipt={'transport_verified':False,'interrupted_before_raw_complete':True}
 else:
  receipt=collection.receive(io.BytesIO(data),parent/'saved',x.control,x.derived,window=collection.Window(0.,100.,clock=lambda:100.))
  assert receipt['transport_verified'] and receipt['wire_sha256']==remote['wire_sha256']==boot.digest(data)
  for name,original in x.sources.items():assert (parent/'saved'/name).read_bytes()==original
  assert (parent/'saved'/RAW).stat().st_ino!=(parent/'saved/capture/results/numerical/objective-6/cotangent-14.npy').stat().st_ino
  assert boot.parsed((parent/'saved'/collection.RECEIPT).read_bytes())['work_phase']=='failed'
 (tmp_path/'BOUND-69-SOURCE-PRIORITY-ROUNDTRIP.json').write_bytes(boot.encoded({'mode':mode,'header':h,'receipt':receipt,'actual_uploaded_source_files':69,'only_packet_inputs_and_work_role_metadata_are_fakes':True,'model_GPU_process_proof_work':False}))


def local_auth_case(tmp_path,monkeypatch):
 parent=tmp_path/'parent';parent.mkdir(mode=0o700);folder=parent/'unarmed';outer.prepare_package(ROOT,folder)
 raw=(folder/'CONTROL-CANDIDATE.json').read_bytes();c,b,p,sha=outer.validate_control(raw,outer.digest((ROOT/binding.NEW['outer']).read_bytes()));auth=outer.encoded(outer.authorization(c,sha,'INERT LOCAL TEST ONLY'))
 outer.private_write(folder,'AUTHORIZATION.json',auth,lambda:None);real=client.file_bytes;seen=[]
 def checked(path,cap,sha256=None):
  if Path(path)==ROOT/outer.PACKET_PATH:seen.append((cap,sha256));return b'FAKE PACKET HASH RESPONSE; NO PACKET READ'
  return real(path,cap,sha256)
 monkeypatch.setattr(client,'file_bytes',checked);return folder,seen


def test_local69_pin_preflight_then_copied_folder_refusal_without_real_packet_read(tmp_path,monkeypatch):
 folder,seen=local_auth_case(tmp_path,monkeypatch);result=client.check_local(folder)
 assert len(result[0]['local_source_sha256'])==69 and len(seen)==1
 assert seen==[(17192095,'2278bce2b519943c0f83a49c556bb73dd928fe880fba0f6c56de2a4c3c2ec0da')]
 import shutil
 copied=folder.parent/'copy';shutil.copytree(folder,copied)
 with pytest.raises(ValueError,match='Copied helpers'):client.check_local(copied)
 assert len(seen)==1 and not (folder/'ATTEMPT.json').exists()


def test_consumed_local_attempt_refuses_before_fake_SSH_and_preserves_counter(tmp_path,monkeypatch):
 folder,seen=local_auth_case(tmp_path,monkeypatch);old=b'INERT ALREADY CONSUMED COUNTER\n';outer.private_write(folder,'ATTEMPT.json',old,lambda:None)
 class Alarm:
  def arm(self,*a,**kw):pass
  def check(self):return 1000.
  def close(self):pass
 monkeypatch.setattr(outer,'Alarms',Alarm);monkeypatch.setattr(client,'time',NS(monotonic=lambda:1000.))
 monkeypatch.setattr(client,'subprocess',NS(Popen=lambda *a,**kw:pytest.fail('No SSH from consumed counter')))
 with pytest.raises(FileExistsError):client.run(folder)
 assert (folder/'ATTEMPT.json').read_bytes()==old and not (folder/'attended-session').exists()


def test_unarmed_final_and_thin_CLI_refuse_before_source_packet_or_timer_work(package,monkeypatch):
 import importlib.util
 spec=importlib.util.spec_from_file_location('INERT-JOINT-CLI',ROOT/binding.NEW['cli']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 forbidden=lambda *a,**kw:pytest.fail('Unarmed must not preflight/start a timer or SSH')
 monkeypatch.setattr(client,'file_bytes',forbidden);monkeypatch.setattr(outer,'Alarms',forbidden);monkeypatch.setattr(client,'time',NS(monotonic=lambda:1000.))
 with pytest.raises(SystemExit,match='Unarmed'):module.main([str(package.folder)])
 with pytest.raises(ValueError,match='Unarmed'):client.check_local(package.folder)
 with pytest.raises(ValueError,match='Unarmed'):client.run(package.folder)
 assert not (package.folder/'ATTEMPT.json').exists()


def test_unchanged_inline_setup_and_refusal_policies_still_verify_full_sources():
 from models import predictor_pgd_split_source_bundle_v1 as source
 outer.verify_directory_policy_sources(ROOT,source);outer.verify_refusal_policy_sources(ROOT,source)
 raw=(ROOT/binding.NEW['outer']).read_bytes();bad=raw.replace(b'if not owner.claimed or owner.driver.guard is None:',b'if False:',1)
 fake=NS(read_source=lambda p,**kw:bad if Path(p)==ROOT/binding.NEW['outer'] else source.read_source(p,**kw))
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
    osproxy.getpid=os.getpid;osproxy.urandom=lambda n:b'0'*n
    osproxy.posix_fallocate=lambda *a:events.append('FAKE-RESERVE-NO-ALLOCATION')
    def fstat(fd):
        s=os.fstat(fd);p=root/'emergency-space.bin'
        if p.exists() and s.st_ino==p.stat().st_ino:return NS(st_size=16*outer.MiB,st_blocks=16*outer.MiB//512)
        return s
    osproxy.fstat=fstat
    monkeypatch.setattr(outer,'os',osproxy)
    monkeypatch.setattr(coordinator,'target_gate',lambda *a:events.append('FAKE-SETUP-HOST'))
    def verify(self,**kw):
        fake_inputs(self.root,self.layout,x.p)
        self.scan(phase='setup',settled=True);events.append('FAKE-PACKET-CONTENTS-REAL-SCAN')
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
    def Collector(guard,control,auth,*,connection_started):
        events.append('ACTUAL-JOINT-COLLECTOR')
        assert ownerbox[0].claimed and guard is ownerbox[0].driver.guard
        monkeypatch.setattr(scratch,'Layout',lambda *a,**kw:guard.layout)
        value=collection.Collector(guard,control,auth,connection_started=connection_started,clock=clock)
        assert value.window is value.original.window
        return value
    def importing(name,globals=None,locals=None,fromlist=(),level=0):
        assert name=='models' and fromlist==('predictor_pgd_split_collection_v1',)
        events.append('ACTUAL-COLLECTOR-IMPORT');return NS(predictor_pgd_split_collection_v1=NS(Collector=Collector))
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
    return NS(root=root,events=events,frames=frames,tail=tail,owner=ownerbox[0],receipts=receipts,mode=mode,clock=clock)


@pytest.mark.parametrize('mode',['busy','claimed-proof-refusal','guard-ready-refusal'])
def test_complete_outer_actual_source_setup_claim_and_real_priority_failure_collection(package,tmp_path,monkeypatch,mode):
 x=lifecycle(package,tmp_path,monkeypatch,mode);reply=x.frames[-1]
 if mode!='claimed-proof-refusal':
  assert reply['kind']=='pgd-split-outer-refusal-v1' and not x.owner.claimed and not x.tail
  assert ('Another CUDA compute process' if mode=='busy' else 'INERT PRECLAIM PIDFD') in reply['message']
  assert not any('COLLECTOR' in str(e) or 'WORKER' in str(e) for e in x.events)
  h=None;receipt=None
 else:
  assert reply['kind']=='pgd-split-collection-ready-v1' and x.owner.claimed and x.tail.startswith(collection.MAGIC)
  wire=io.BytesIO(x.tail);read=collection.WireReader(wire,collection.Window(999.,1000.,clock=x.clock));_,h=read.frame(collection.MAGIC,collection.HEADER_CAP)
  assert h['base']['connection_started']==999. and h['base']['collection_started']==reply['collection_started']==1000.
  assert h['base']['collection_deadline']==reply['collection_deadline']==1600. and h['records'][0]['name']=='session.json'
  parent=tmp_path/'real-receiver';parent.mkdir(mode=0o700)
  receipt=collection.receive(io.BytesIO(x.tail),parent/'saved',package.control,package.derived,window=collection.Window(999.,1000.,clock=x.clock))
  assert receipt['transport_verified'] and receipt['wire_sha256']==boot.digest(x.tail)
  final=boot.parsed((parent/'saved'/collection.RECEIPT).read_bytes())
  assert final['work_phase']=='failed' and final['all_saved_role_checks_returned'] is False
  assert boot.parsed((parent/'saved/session.json').read_bytes())['phase']=='failed'
  for name,raw in package.sources.items():assert (parent/'saved'/name).read_bytes()==raw
  (tmp_path/'ACTUAL-PRIORITY-WIRE.bin').write_bytes(x.tail)
 assert x.owner.driver.outer_child is None and x.owner.driver.outer_lease is None and x.owner.driver.outer_journal.closed
 for name in (integration.SCRATCH,'overlay/models/predictor_pgd_split_store_v1.py','supervisor/predictor_pgd_split_watch_v1.py'):assert (x.root/name).read_bytes()==package.sources[name]
 (tmp_path/'COMPLETE-JOINT-OUTER-LIFECYCLE.json').write_bytes(boot.encoded({'mode':mode,'reply':reply,'events':x.events,'header':h,'receipt':receipt,
  'actual_source_cache_setup_directory_journal_claim_and_priority_codec':True,'packet_asset_hashes_and_GPU_host_PIDs_proof_worker_timer_clock_are_explicit_fakes':True,
  'model_array_GPU_network_proof_or_child_work':False,'actual_uploaded_sources':69}))


def test_real_journal_observes_raw_writer_transition_and_persists_new_reserve(package,tmp_path):
 x=package;root=tmp_path/'raw-journal';root.mkdir(mode=0o700);(root/'capture').mkdir(mode=0o700)
 ns={'__name__':'INERT-JOINT-SCANNER','__file__':'<VERIFIED-SCANNER>'};exec(compile(x.sources[integration.SCRATCH],ns['__file__'],'exec'),ns)
 guard=ns['Guard'](root,ns['identity_binding'](root),copy.deepcopy(x.layout));final=root/RAW;state=NS(fired=False,writerfd=None,queries=[])
 proxy=NS(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')})
 def named(path,*a,**kw):
  if state.writerfd is not None and str(path)==final.name+'.partial' and not state.fired:
   fd=kw.get('dir_fd');assert fd is not None and os.fstat(fd).st_ino==os.fstat(state.writerfd).st_ino
   os.link(final.name+'.partial',final.name,src_dir_fd=state.writerfd,dst_dir_fd=state.writerfd,follow_symlinks=False)
   os.unlink(final.name+'.partial',dir_fd=state.writerfd);state.fired=True
  return os.stat(path,*a,**kw)
 proxy.stat=named;proxy.read=lambda *a:pytest.fail('Scanner must never read contents');ns['os']=proxy
 coord={'__name__':'INERT-JOINT-JOURNAL','__file__':'<VERIFIED-PARENT>'};exec(compile(x.sources[integration.COORDINATOR],coord['__file__'],'exec'),coord)
 coord.update(boot=boot,scratch=scratch)
 def query(args,**kw):
  state.queries.append(args);assert args[0]=='nvidia-smi'
  return '' if '--query-compute-apps=' in args[1] else 'NVIDIA GeForce RTX 3070, GPU-c7a30a6f-08fb-371f-41b3-86a13a1a632d, 8192, 43, 580.178.04\n'
 coord['subprocess']=NS(check_output=query,PIPE=object());journal=coord['OuterJournal'](guard,lambda:1000.)
 store={'__name__':'ACTUAL-PINNED-BYTE-WRITER','__file__':'<VERIFIED-BYTE-STORE>'};exec(compile(x.sources['overlay/models/predictor_pgd_split_store_v1.py'],store['__file__'],'exec'),store)
 wp=NS(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')})
 def linked(src,dst,**kw):state.writerfd=kw['src_dir_fd'];journal.checkpoint('numerical');assert state.fired
 def unlinked(*a,**kw):assert state.fired
 wp.link=linked;wp.unlink=unlinked;store['os']=wp
 inventory=store['Inventory'](root/'capture',base_bytes=x.p['base_text'].encode(),design_bytes=x.p['design_text'].encode());writer=store['Store'](inventory)
 try:
  writer.write_bytes(Path(RAW).relative_to('capture').as_posix(),DATA)
  status=boot.parsed((root/'storage-status.json').read_bytes());a=status['last_storage']['publication_accounting'];cap=x.layout.classify(RAW)['max_bytes']
  assert state.fired and journal.calls==1 and len(state.queries)==2 and journal.refusals==0
  assert a['raw_partial_slots_reserved']==1 and a['raw_reserve_bytes']==cap and a['raw_observation']['state']=='single-final'
  assert status['last_storage']['logical_bytes']==a['observed_logical_bytes']+a['logical_reserve_bytes']
  assert not status['scientific_qualified'] and not status['training_admitted'] and final.read_bytes()==DATA
  (tmp_path/'JOINT-RAW-JOURNAL-RESERVE.json').write_bytes((root/'storage-status.json').read_bytes())
 finally:journal.close()
 assert journal.fd is None and journal.closed


@pytest.mark.parametrize('fault',['deadline','rollback','nonfinite'])
def test_bound_server_and_client_clocks_keep_original_origins_and_refuse_changes(fault):
 moment=[1000.];window=collection.Window(999.,1000.,clock=lambda:moment[0])
 receiver=client.ReceiverClock(50.,51.,1000.,1600.,clock=lambda:51.)
 assert window.deadline==1600. and receiver.deadline==651.
 moment[0]={'deadline':1600.,'rollback':999.,'nonfinite':float('nan')}[fault]
 with pytest.raises(TimeoutError):window.check()
 receiver.clock=lambda:{'deadline':651.,'rollback':50.,'nonfinite':float('nan')}[fault]
 with pytest.raises(outer.Deadline):receiver.check()
 assert window.connection_started==999. and window.started==1000. and window.deadline==1600. and receiver.deadline==651.

@pytest.mark.parametrize('fault',['bad-prefix','ready-clock','root-claim'])
def test_actual_client_persists_ready_only_after_claim_and_clock_before_paired_decode(tmp_path,monkeypatch,fault):
 folder,seen=local_auth_case(tmp_path,monkeypatch);checked=client.check_local(folder)
 c,b,p,sha,raw,araw,auth,mapping,code,_=checked;packet=tmp_path/'INERT-PACKET';packet.write_bytes(b'INERT PACKET; NEVER AN ARCHIVE')
 monkeypatch.setattr(client,'check_local',lambda path:(c,b,p,sha,raw,araw,auth,mapping,code,packet))
 phases=[];writes=[];clock=lambda:100.
 class Alarm:
  def arm(self,deadline,*,hard_deadline,phase):phases.append((phase,deadline,hard_deadline))
  def check(self):return 100.
  def close(self):phases.append(('closed',None,None))
 class Thread:
  def __init__(self,*,target,daemon):self.target=target
  def start(self):self.target()
  def join(self,timeout):pass
 class Input(io.BytesIO):
  def fileno(self):return 77777
 class Process:
  def __init__(self,wire):self.stdin=Input();self.stdout=io.BytesIO(wire);self.stderr=io.BytesIO();self.signalled=False
  def poll(self):return 1
  def wait(self,timeout):return 1
  def terminate(self):self.signalled=True
  def kill(self):self.signalled=True
 root='/var/tmp/vjepa-pgd-split-v1-2170-'+sha
 hello={'kind':'pgd-split-hello-v1','nonce':'0'*64,'hello_monotonic':1000.,'code_sha256':c['outer_source_sha256']}
 claimed={'kind':'pgd-split-root-claimed-v1','outer_control_sha256':sha,'scratch':root,'hello_monotonic':1000.,'handshake_received':1000.,'connection_started':1000.,'work_deadline':3400.,'connection_deadline':4000.,**outer.FLAGS}
 ready={'kind':'pgd-split-collection-ready-v1','outer_control_sha256':sha,'connection_started':1000.,'handshake_received':1000.,'collection_started':1100.,'collection_deadline':1700.,**outer.FLAGS}
 if fault=='ready-clock':ready['collection_deadline']+=1
 if fault=='root-claim':claimed['scratch']+='-wrong'
 stream=io.BytesIO()
 for frame in (hello,claimed,ready):outer.send_frame(stream,frame,lambda:None)
 stream.write(b'WRONG PROTOCOL PREFIX; NO RETRY OR LEGACY FALLBACK');proc=Process(stream.getvalue())
 def popen(command,**kw):
  assert command[0]=='ssh' and 'ForwardAgent=no' in command and command[-2]=='kdu@linux10614.dc.engr.scu.edu'
  writes.append('FAKE-SSH-ONLY');return proc
 osproxy=NS(**vars(os));osproxy.write=lambda fd,data:(writes.append({'bootstrap_bytes':len(data)}) or len(data)) if fd==77777 else pytest.fail('Unexpected fake pipe write')
 realclock=client.ReceiverClock
 monkeypatch.setattr(client,'ReceiverClock',lambda *a:realclock(*a,clock=clock))
 monkeypatch.setattr(client,'os',osproxy);monkeypatch.setattr(client,'threading',NS(Thread=Thread))
 monkeypatch.setattr(client,'subprocess',NS(Popen=popen,PIPE=object(),TimeoutExpired=RuntimeError))
 monkeypatch.setattr(client,'time',NS(monotonic=clock));monkeypatch.setattr(outer,'Alarms',Alarm)
 with pytest.raises(ValueError):client.run(folder)
 session=folder/'attended-session';assert (folder/'ATTEMPT.json').is_file() and (session/'CLIENT-REFUSAL.json').is_file()
 assert not proc.signalled and not (session/'saved').exists() and not (session/'COLLECTION-RECEIPT.json').exists()
 if fault=='bad-prefix':
  assert boot.parsed((session/'COLLECTION-READY.json').read_bytes())==ready and boot.parsed((session/'ROOT-RECEIPT.json').read_bytes())==claimed
  assert ('client-collection',800.,3100.) in phases
 else:assert not (session/'COLLECTION-READY.json').exists()
 (tmp_path/'ACTUAL-CLIENT-CONTINUING-CLOCK-REFUSAL.json').write_bytes(boot.encoded({'fault':fault,'phases':phases,'writes':writes,'source_preflight_claim_root_receipt_ready_receipt_and_prefix_parser_real':True,'only_packet_process_pipe_thread_timer_host_and_clock_APIs_fake':True,'model_GPU_child_proof_or_network_work':False}))
