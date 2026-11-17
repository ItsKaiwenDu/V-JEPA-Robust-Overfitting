"""Fresh 10609 migration checks; metadata IO only, all target/process bindings inert."""
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
from models import predictor_pgd_host10609_v1 as bind
from models import predictor_pgd_split_outer_10609_v1 as outer
from models import predictor_pgd_split_outer_client_10609_v1 as client
from models import predictor_pgd_split_bootstrap_10609_v1 as boot
from models import predictor_pgd_split_scratch_10609_v1 as scratch
from models import predictor_pgd_split_collection_10609_v1 as collection
from models import predictor_pgd_split_coordinator_v1 as coordinator
b,s,m=boot,scratch,collection
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'runs/pretrained/diagnostic/predictor-pgd-startup-v3-20261116/handoff-parent/linux10603-unarmed'

@pytest.fixture(scope='module')
def package(tmp_path_factory):
 parent=tmp_path_factory.mktemp('EXPLICIT-UNARMED-HOST10609');parent.chmod(0o700);folder=parent/'candidate'
 result=outer.prepare_package(ROOT,folder);raw=(folder/'CONTROL-CANDIDATE.json').read_bytes();code=(ROOT/bind.NAMES['outer']).read_bytes()
 c,bc,p,sha=outer.validate_control(raw,outer.digest(code));payload=json.loads((folder/'SOURCE-PAYLOAD.json').read_bytes());sources=outer.decode_sources(payload,c,bc)
 auth=outer.encoded(outer.authorization(c,sha,'EXPLICIT INERT LOCAL HOST10609 TEST ONLY; NOT A LIVE USER GRANT'))
 return NS(folder=folder,result=result,raw=raw,code=code,c=c,b=bc,p=p,sha=sha,payload=payload,sources=sources,auth=auth)


def test_source_inverse_host_surface_reference_values_and_resource_bounds(package):
 x=package;oldc=json.loads((ROOT/bind.ROOT_PACKAGE/'CONTROL-CANDIDATE.json').read_bytes());oldb=json.loads(oldc['supervised_control_text']);oldp=json.loads(oldb['policy_text'])
 oldmap={k:base64.b64decode(v) for k,v in json.loads((ROOT/bind.ROOT_PACKAGE/'SOURCE-PAYLOAD.json').read_bytes()).items()}
 meta=x.result['host_change_metadata'];assert meta['changed_source_files']==11 and meta['unchanged_source_files']==58
 assert x.result['source_bytes']==875628 and x.result['source_files']==69
 assert x.p['base_text']==oldp['base_text'] and x.p['design_text']==oldp['design_text'] and x.p['scope']['resource']==oldp['scope']['resource']
 assert json.loads(x.p['base_text'])['host']=='linux10603'
 assert x.p['scope']['host']==dict(oldp['scope']['host'],hostname='linux10609',gpu_uuid=bind.UUID)
 for name,data in x.sources.items():
  edits=meta['source_edits'].get(name,[]);parts=[];cursor=0
  for e in edits:
   assert type(ast.literal_eval(e['old_literal'])) is str and type(ast.literal_eval(e['new_literal'])) is str
   parts.extend((data[cursor:e['new_begin']],e['old_literal'].encode()));cursor=e['new_end']
  parts.append(data[cursor:]);assert b''.join(parts)==oldmap[name],name
 assert x.sources['overlay/models/_pgd_split_io_v1.py']==oldmap['overlay/models/_pgd_split_io_v1.py']
 for name in ('overlay/models/_pgd_split_observer_v1.py','overlay/models/predictor_forward_preserving_target_v1.py'):
  assert b"('linux10603', 'linux10609')" in x.sources[name] or b"('linux10603','linux10609')" in x.sources[name]
 assert set(p.name for p in x.folder.iterdir())=={'CONTROL-CANDIDATE.json','SOURCE-PAYLOAD.json','known-hosts','PACKAGE.json'}
 assert 'torch' not in sys.modules and 'numpy' not in sys.modules


@pytest.mark.parametrize('fault',['old-control','old-auth','host-key','policy','source-coverage'])
def test_new_host_scope_refuses_consumed_grants_or_changed_bindings(package,fault):
 x=package;c=copy.deepcopy(x.c)
 if fault=='old-auth':
  with pytest.raises(ValueError):outer.validate_authorization((ROOT/bind.ROOT_PACKAGE/'AUTHORIZATION.json').read_bytes(),c,x.sha)
  return
 if fault=='old-control':c['kind']='pgd-split-outer-control-v4'
 elif fault=='host-key':c['known_hosts_text']=c['known_hosts_text'].replace('10609','10603')
 elif fault=='policy':
  v=json.loads(c['supervised_control_text']);v['policy_text']=v['policy_text'].replace(bind.UUID,bind.OLD_UUID);c['supervised_control_text']=outer.encoded(v).decode();c['supervised_control_sha256']=outer.digest(c['supervised_control_text'].encode())
 else:c['local_source_sha256'].pop(bind.NAMES['scratch'])
 with pytest.raises(ValueError):outer.validate_control(outer.encoded(c),c['outer_source_sha256'])


def test_unarmed_entry_and_client_refuse_without_timers_sources_or_transport(package,monkeypatch):
 spec=importlib.util.spec_from_file_location('EXPLICIT_UNARMED_HOST10609',ROOT/bind.NAMES['cli']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 with pytest.raises(SystemExit,match='No timer, claim, SSH or model work'):module.main([str(package.folder)])
 with pytest.raises(ValueError,match='Unarmed'):client.check_local(package.folder)
 assert not (package.folder/'ATTEMPT.json').exists() and not (package.folder/'AUTHORIZATION.json').exists()


def test_source_only_host_bridge_uses_exact_server_bootstrap_and_local_catalogue(package):
 raw=package.c['supervised_control_text'].encode();bc,p,sha=boot.control(raw);ib,isha=boot.installation(bc,sha)
 layout=scratch.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),ib,isha)
 assert len(layout.assets)==172 and len(layout.result)==419 and len(outer._setup_directory_plan(layout)['setup_directories'])==26
 assert (ROOT/bind.NAMES['boot']).read_bytes()==package.sources['supervisor/bootstrap.py']
 assert (ROOT/bind.NAMES['scratch']).read_bytes()==package.sources['supervisor/predictor_pgd_split_scratch_v1.py']
 assert boot.POLICY_SHA==outer.POLICY_SHA and scratch.CONTRACT_SHA==p['scratch_contract_sha256']


def test_real_local_preflight_with_inert_grant_and_packet_binding_then_copy_refusal(package,tmp_path,monkeypatch):
 import shutil
 target=tmp_path/'candidate';target.mkdir(mode=0o700)
 # New bound handoff, never arm the production candidate.
 outer.prepare_package(ROOT,tmp_path/'prepared');target.rmdir();target=tmp_path/'prepared'
 raw=(target/'CONTROL-CANDIDATE.json').read_bytes();c,bc,p,sha=outer.validate_control(raw,outer.digest(package.code))
 auth=outer.encoded(outer.authorization(c,sha,'EXPLICIT LOCAL TEST GRANT ONLY; NO TARGET OR MODEL'))
 outer.private_write(target,'AUTHORIZATION.json',auth,lambda:None)
 original=client.file_bytes;packet=ROOT/outer.PACKET_PATH;seen=[]
 def file_bytes(path,cap,sha256=None):
  if Path(path)==packet:
   seen.append((cap,sha256));return b'EXPLICIT-INERT-PACKET-BYTE-HASH-BINDING; NOT PIXELS'
  return original(path,cap,sha256)
 monkeypatch.setattr(client,'file_bytes',file_bytes)
 checked=client.check_local(target);assert checked[3]==sha and len(seen)==1 and seen[0][0]==17192095
 copyto=tmp_path/'copied';shutil.copytree(target,copyto);copyto.chmod(0o700)
 with pytest.raises(ValueError,match='Copied helpers'):client.check_local(copyto)
 assert len(seen)==1 and not (target/'ATTEMPT.json').exists()


def coordinator_namespace(package):
 ns={'__name__':'EXPLICIT-GENERATED-10609-COORDINATOR','__file__':'<STDLIB-ONLY-HOST10609>'}
 exec(compile(package.sources['supervisor/predictor_pgd_split_coordinator_v1.py'],ns['__file__'],'exec'),ns)
 ns['boot']=boot;ns['scratch']=scratch
 return ns


@pytest.mark.parametrize('mode',['clear','old-UUID','foreign-compute','memory-busy'])
def test_actual_generated_gpu_guard_is_bound_to10609_and_remains_strict(package,mode):
 ns=coordinator_namespace(package);queries=[]
 def query(argv,**kw):
  queries.append(argv);assert kw['timeout']==10
  if '--query-compute-apps=' in argv[1]:return '77, /usr/NX/bin/nxnode.bin, 191\n' if mode=='foreign-compute' else ''
  return 'NVIDIA GeForce RTX 3070, '+(bind.OLD_UUID if mode=='old-UUID' else bind.UUID)+', 8192, '+('513' if mode=='memory-busy' else '96')+', 580.178.04\n'
 ns['subprocess']=NS(check_output=query,PIPE=object())
 if mode=='clear':
  result=ns['gpu_gate']();assert bind.UUID in result['GPU_row'] and result['compute_rows']==[] and len(queries)==2
 else:
  with pytest.raises(ValueError):ns['gpu_gate']()


@pytest.mark.parametrize('host',['linux10603','local-Mac'])
def test_new_parent_and_worker_host_guards_refuse_before_scratch_or_science(package,host):
 ns=coordinator_namespace(package);ns['platform']=NS(node=lambda:host,system=lambda:'Linux',python_version=lambda:'3.11.17')
 def fail(*a):raise AssertionError('Wrong host touched scratch before refusal')
 ns['Path']=fail
 with pytest.raises(ValueError,match='Exact authorized university'):ns['target_gate']('/inert',{})
 # Exercise the actual generated numerical entry; its first host gate precedes scientific imports.
 tree=ast.parse(package.sources['overlay/models/_pgd_split_numerical_v1.py']);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='unbound_numerical_worker')
 g={'platform':ns['platform']};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<GENERATED-NUMERICAL-EARLY-HOST-GATE>','exec'),g)
 with pytest.raises(ValueError,match='10609'):g['unbound_numerical_worker'](NS(device='cuda',tiny=False))


def test_outer_runtime_and_protocol_functions_preserve_v4_code_objects():
 a=ast.parse((ROOT/'models/predictor_pgd_split_outer_v4.py').read_bytes());b=ast.parse((ROOT/bind.NAMES['outer']).read_bytes())
 funcs=lambda t:{n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 x,y=funcs(a),funcs(b);changed={'preflight_parent','validate_control','authorization','verify_directory_policy_sources','verify_refusal_policy_sources','prepare_package'}
 assert set(x)==set(y)
 for n in x.keys()-changed:assert ast.dump(x[n],include_attributes=False)==ast.dump(y[n],include_attributes=False),n
 old=(ROOT/'models/predictor_pgd_split_outer_client_v4.py').read_bytes();new=(ROOT/bind.NAMES['client']).read_bytes()
 for before,after in [(b'predictor_pgd_split_outer_v4 as outer',b'predictor_pgd_split_outer_10609_v1 as outer'),(b'predictor_pgd_split_bootstrap_v1 as boot',b'predictor_pgd_split_bootstrap_10609_v1 as boot'),(b'predictor_pgd_split_collection_v1 as collection',b'predictor_pgd_split_collection_10609_v1 as collection')]:new=new.replace(after,before)
 assert new==old


def session(control, *, failed=False, prefix=4):
    roles = list(b.ROLES[:prefix])
    return {'kind':'pgd-split-parent-session-v1','control_sha256':control.sha,'parent_pid':os.getpid(),
        'connection_started':0.,'work_deadline':2400.,'connection_deadline':3000.,
        'phase':'failed' if failed else 'await-collection','complete':False,'collection_pending':True,
        'current':None,'completed':roles,'receipts':[
            {'role':role,'pid':900+i,'start_ticks':400+i,'saved_gates_returned':True,
             'receipt':{'origin':'EXPLICIT FAKE ROLE RECEIPT; NO WORK'}, 'source_hashes':{},
             'scientific_qualified':False,'training_admitted':False} for i, role in enumerate(roles)],
        'scientific_qualified':False,'training_admitted':False}

def metadata_header(control):
    layout = control.layout
    st = session(control)
    win = m.Window(0.,100.,clock=lambda:100.)
    receipt = m.work_receipt(b.encoded(st),control.sha,b.digest(control.auth),control.isha,win)
    extras = {'control.json':control.raw,'authorization.json':control.auth,'session.json':b.encoded(st),
              m.RECEIPT:b.encoded(receipt),m.STATUS:b.encoded({'EXPLICIT':'SYNTHETIC FULL INVENTORY METADATA ONLY'})}
    pins = dict(layout.assets)
    pins.update({n:{'bytes':len(v),'sha256':b.digest(v)} for n,v in extras.items()})
    names = sorted(pins)
    uid = os.getuid()
    files = {n:dict(pins[n],mode=0o600,mtime_ns=1000000001+i) for i,n in enumerate(names)}
    inventory = {n:{'identity':[500,100+i,uid,0o600,1],'bytes':v['bytes'],
                   'allocated_bytes':((v['bytes']+4095)//4096)*4096,'mtime_ns':v['mtime_ns']} for i,(n,v) in enumerate(files.items())}
    inventory['emergency-space.bin'] = {'identity':[500,900,uid,0o600,1], 'bytes':16*s.MiB,'allocated_bytes':16*s.MiB,'mtime_ns':100}
    dirs = {'.'}
    for name in inventory:
        dirs.update(str(q) for q in Path(name).parents)
    directories = {n:{'identity':[500,2000+i,uid,0o700],'allocated_bytes':4096} for i,n in enumerate(sorted(dirs))}
    storage = layout.assess(inventory,directories,{'available_bytes':8*s.GiB,'available_inodes':2048},phase='collection')
    return {'kind':'pgd-split-evidence-stream-v1','control_sha256':control.sha,'authorization_sha256':b.digest(control.auth),
        'root_binding':{'root':'/EXPLICIT-SYNTHETIC/scratch','root_identity':list(directories['.']['identity']),
                        'parent':'/EXPLICIT-SYNTHETIC','parent_identity':[500,1,uid,0o700]},
        'files':files,'inventory':inventory,'directories':directories,'storage':storage,
        'final_receipt_sha256':b.digest(extras[m.RECEIPT]),'session_sha256':b.digest(extras['session.json']),
        'collection_started':100.,'collection_deadline':700.,'connection_started':0.,'connection_deadline':3000.,
        'logical_bytes':sum(r['bytes'] for r in files.values()),'omitted_content':['emergency-space.bin'],
        'scientific_qualified':False,'training_admitted':False}

def lifecycle(x,tmp_path,monkeypatch,mode):
    clock=lambda:1000.0;events=[];ownerbox=[];finders=[];receipts=[]
    root=tmp_path/'EXPLICIT-FAKE-REMOTE-ROOT'
    observed=json.loads((OLD/'attended-session/REFUSAL.json').read_bytes())['last_guard_refusal']
    queries=[q['text'].replace(bind.OLD_UUID,bind.UUID) for q in observed['queries']]
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


def test_new_collection_reader_accepts_new_host_contract_and_rejects_cross_source_header(package):
 raw=package.c['supervised_control_text'].encode();bc,p,sha=boot.control(raw);ib,isha=boot.installation(bc,sha)
 auth=boot.encoded({'kind':'pgd-split-supervised-university-exception-v1','approved':True,'control_sha256':sha,'maximum_new_attempts':1,'user_basis':'EXPLICIT INERT HEADER ONLY'})
 x=NS(raw=raw,c=bc,p=p,sha=sha,auth=auth,ib=ib,isha=isha,layout=scratch.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),ib,isha))
 h=metadata_header(x);layout=collection.validate_header(h,raw,auth);assert len(layout.assets)==172 and len(layout.result)==419
 changed=copy.deepcopy(h);changed['files']['supervisor/predictor_pgd_split_coordinator_v1.py']['sha256']='f'*64
 with pytest.raises(ValueError,match='pinned installation'):collection.validate_header(changed,raw,auth)
 # The consumed v4 policy is refused by the new bootstrap before any content collection.
 old=json.loads((ROOT/bind.ROOT_PACKAGE/'CONTROL-CANDIDATE.json').read_bytes())['supervised_control_text'].encode()
 with pytest.raises(ValueError):collection.validate_header(h,old,auth)


@pytest.mark.parametrize('mode',['busy','claimed-proof-refusal'])
def test_complete_host10609_setup_source_cache_journal_claim_and_failure_collection(package,tmp_path,monkeypatch,mode):
 x=lifecycle(package,tmp_path,monkeypatch,mode);reply=x.frames[-1]
 if mode=='busy':
  assert not x.owner.claimed and x.owner.driver.guard is None and not x.tail
  assert reply['kind']=='pgd-split-outer-refusal-v1' and reply['message']=='Another CUDA compute process is active; leave it alone and refuse.'
  assert bind.UUID in reply['last_guard_refusal']['queries'][0]['text']
  assert not any('COLLECTOR' in str(e) or 'WORKER' in str(e) for e in x.events)
 else:
  assert x.owner.claimed and x.owner.driver.guard is not None and len(x.receipts)==1
  assert reply['kind']=='pgd-split-collection-ready-v1' and x.receipts[0]['work_phase']=='failed'
  assert x.tail==b'EXPLICIT-INERT-COLLECTION-NO-TENSORS'
 (tmp_path/'INERT-HOST10609-LIFECYCLE.json').write_bytes(outer.encoded({'mode':mode,'reply':reply,'events':x.events,
  'real_catalogue_cache_coordinator_journal_claim_and_codec':True,'scientific_worker_proof_or_tensor_executed':False,'actual_process_timer_GPU_or_network':False}))


def test_correct_numerical_host_reaches_only_inert_science_import_boundary(package):
 tree=ast.parse(package.sources['overlay/models/_pgd_split_numerical_v1.py']);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='unbound_numerical_worker')
 class BeforeScience(RuntimeError):pass
 seen=[]
 def importing(name,*args,**kwargs):
  seen.append(name);assert name=='numpy';raise BeforeScience('Stopped before loading any scientific library')
 g={'platform':NS(node=lambda:'linux10609',system=lambda:'Linux',python_version=lambda:'3.11.17'), '__builtins__':dict(vars(builtins),__import__=importing)}
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'<INERT-HOST10609-SCIENCE-BOUNDARY>','exec'),g)
 with pytest.raises(BeforeScience):g['unbound_numerical_worker'](NS(device='cuda',tiny=False))
 assert seen==['numpy'] and 'numpy' not in sys.modules and 'torch' not in sys.modules
