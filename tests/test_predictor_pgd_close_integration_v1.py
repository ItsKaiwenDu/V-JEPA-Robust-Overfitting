"""Focused new integration/exit-race checks; all children, clocks and signals fake."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_close_integration_v1 as integration
from models import predictor_pgd_exit_race_v1 as recipe
from models import predictor_pgd_close_bootstrap_v1 as boot
from models import predictor_pgd_close_scratch_v1 as scratch
from models import predictor_pgd_close_collection_v1 as collection

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'runs/pretrained/diagnostic/pgd-close-integration-v1-20261117'


@pytest.fixture(scope='module')
def package():
    control,sources,readers,manifest,metadata=integration.assemble(ROOT)
    c,p,sha=boot.control(control)
    ib,isha=boot.installation(c,sha)
    layout=scratch.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),ib,isha)
    return NS(control=control,sources=sources,readers=readers,manifest=manifest,metadata=metadata,c=c,p=p,sha=sha,ib=ib,isha=isha,layout=layout)


def monitor_namespace(source):
    ns={'__name__':'EXPLICIT-FAKE-CHILD-MONITOR','__file__':str(RAW/'UNEXECUTED-watch.py')}
    exec(compile(source,'<LOCAL-FAKE-PROCESS-MONITOR>','exec'),ns)
    return ns


class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
    def pause(self,seconds):self.now+=seconds


class Child:
    pid=900001
    def __init__(self,clock,status=1,*,wait_error=None,delay=.01):
        self.clock=clock;self.status=status;self.wait_error=wait_error;self.delay=delay
        self.exit_visible=False;self.waits=[];self.polls=0;self.lease=None
    def poll(self):
        self.polls+=1
        return self.status if self.exit_visible else None
    def wait(self,timeout):
        self.waits.append(timeout)
        if self.wait_error is not None:
            error,self.wait_error=self.wait_error,None
            raise error
        self.clock.now+=self.delay;self.exit_visible=True
        if self.lease is not None:self.lease.exited=True
        return self.status


class Lease:
    def __init__(self,pid=Child.pid,birth=91):
        self.pid,self.birth=pid,birth;self.closed=False;self.exited=False;self.signals=[]
    def dead(self):
        assert not self.closed
        return self.exited
    def send(self,number):
        assert not self.closed and number in (signal.SIGTERM,signal.SIGKILL)
        if not self.exited:self.signals.append(number);self.exited=True
    def close(self):self.closed=True


def row(pid=Child.pid,birth=91,rss=1024):
    return {'pid':pid,'start_ticks':birth,'parent_pid':os.getpid(),'group_pid':pid,'session_pid':pid,'state':'R',
            'rss_bytes':rss,'children':[]}


def simulate(ns,*,status=1,wait_error=None,delay=.01,phase=120.,work=150.,sample_error=None,first_good=True,descendant=False):
    clock=Clock();lease=Lease();child=Child(clock,status,wait_error=wait_error,delay=delay);child.lease=lease
    calls=[];desc=Lease(900002,92);factory=[]
    missing=ProcessLookupError(3,'EXPLICIT exiting process metadata') if sample_error is None else sample_error
    def sample(pid,birth):
        calls.append((pid,birth))
        if first_good and len(calls)==1:
            rows={pid:row()}
            if descendant:
                d=row(desc.pid,desc.birth);d['parent_pid']=pid;d['session_pid']=pid;d['group_pid']=pid
                rows[desc.pid]=d
            return {'processes':rows,'rss_bytes':1024*len(rows)}
        raise missing
    def make(pid,birth,parent,session):
        factory.append((pid,birth,parent,session));assert pid==desc.pid and birth==desc.birth and parent==child.pid
        return desc
    state=NS(clock=clock,lease=lease,child=child,calls=calls,desc=desc,factory=factory,missing=missing,ns=ns)
    def run():return ns['watch'](child,lease,sample=sample,lease_factory=make,gate=lambda:None,
                                 phase_deadline=phase,work_deadline=work,launch_started=0.,clock=clock,pause=clock.pause)
    return state,run


def test_frozen_monitor_reproduces_saved_exit_race_with_fake_owned_child(tmp_path):
    frozen=(ROOT/'models/predictor_pgd_split_watch_v1.py').read_bytes()
    ns=monitor_namespace(frozen);state,run=simulate(ns)
    with pytest.raises(ProcessLookupError):run()
    assert len(state.calls)==2 and state.child.polls==3
    assert state.lease.closed and state.child.exit_visible and state.child.status==1
    (tmp_path/'FROZEN-EXIT-RACE-EXPLICIT-FAKES.json').write_text(json.dumps({'primary':'ProcessLookupError','owned_exit':1,'sample_calls':2,'real_process':False}))


@pytest.mark.parametrize('status',[1,7])
def test_repaired_race_reports_confirmed_failed_child_not_lookup_error(package,tmp_path,status):
    ns=monitor_namespace(package.sources[integration.WATCH]);state,run=simulate(ns,status=status)
    with pytest.raises(RuntimeError,match='child failed') as raised:run()
    assert raised.value is not state.missing and state.child.waits[0]==.1
    assert len(state.calls)==2 and state.lease.closed and not state.lease.signals
    (tmp_path/'REPAIRED-EXIT-RACE-EXPLICIT-FAKES.json').write_text(json.dumps({'primary':type(raised.value).__name__,'owned_exit':status,'first_wait':state.child.waits[0],'real_process':False}))


def test_repaired_race_can_finish_only_with_prior_RSS_and_held_exit(package):
    state,run=simulate(monitor_namespace(package.sources[integration.WATCH]),status=0)
    result=run()
    assert result['parent_observation_passed'] and result['RSS_samples']==1 and result['exit_status']==0
    assert result['scientific_qualified'] is False and state.lease.closed and not state.lease.signals
    assert state.child.waits[0]==.1 and len(state.calls)==2


@pytest.mark.parametrize('missing',[ProcessLookupError(3,'EXPLICIT process metadata'),FileNotFoundError('EXPLICIT proc path')])
def test_exit_not_confirmed_retains_original_metadata_error_and_stops(package,missing):
    ns=monitor_namespace(package.sources[integration.WATCH])
    state,run=simulate(ns,wait_error=subprocess.TimeoutExpired('EXPLICIT-OWNED-CHILD',.1),sample_error=missing)
    with pytest.raises(type(missing)) as raised:run()
    assert raised.value is missing and any('unconfirmed' in n for n in raised.value.__notes__)
    assert state.lease.closed and state.lease.signals==[signal.SIGTERM]
    assert len(state.calls)==2 and len(state.child.waits)==2


def test_other_wait_error_does_not_replace_primary_metadata_error(package):
    state,run=simulate(monitor_namespace(package.sources[integration.WATCH]),wait_error=OSError('EXPLICIT wait failure'))
    with pytest.raises(ProcessLookupError) as raised:run()
    assert raised.value is state.missing and any('also failed: OSError' in n for n in raised.value.__notes__)
    assert state.lease.closed


def test_wait_timeout_is_clamped_to_existing_phase_clock(package):
    state,run=simulate(monitor_namespace(package.sources[integration.WATCH]),status=0,phase=.15,work=.2,delay=.01)
    result=run()
    assert result['parent_observation_passed']
    assert state.child.waits[0]==pytest.approx(.05) and state.clock.now<.15


def test_wait_cannot_turn_late_exit_into_success(package):
    state,run=simulate(monitor_namespace(package.sources[integration.WATCH]),status=0,phase=.15,work=.2,delay=.06)
    with pytest.raises(TimeoutError,match='after owned exit'):run()
    assert state.lease.closed and state.clock.now>=.15


def test_no_prior_RSS_sample_still_refuses_even_confirmed_clean_exit(package):
    state,run=simulate(monitor_namespace(package.sources[integration.WATCH]),status=0,first_good=False)
    with pytest.raises(ValueError,match='At least one RSS'):run()
    assert len(state.calls)==1 and state.lease.closed


def test_surviving_held_descendant_prevents_clean_exit_admission(package):
    state,run=simulate(monitor_namespace(package.sources[integration.WATCH]),status=0,descendant=True)
    with pytest.raises(ValueError,match='no surviving owned task'):run()
    assert state.factory==[(900002,92,900001,900001)]
    assert state.lease.closed and state.desc.closed and state.desc.signals==[signal.SIGTERM]
    assert not state.lease.signals


def test_unrelated_sampling_error_is_not_treated_as_exit_race(package):
    state,run=simulate(monitor_namespace(package.sources[integration.WATCH]),sample_error=ValueError('EXPLICIT identity changed'))
    with pytest.raises(ValueError,match='identity changed'):run()
    assert len(state.child.waits)==1 # Original cleanup only; no confirmation wait.
    assert state.lease.closed and len(state.calls)==2


def test_repair_only_changes_sampling_boundary_and_one_stdlib_import(package):
    frozen=(ROOT/'models/predictor_pgd_split_watch_v1.py').read_bytes();fixed=package.sources[integration.WATCH]
    assert fixed.decode().replace(recipe.NEW,recipe.OLD,1).replace('import stat\nimport subprocess\n','import stat\n',1).encode()==frozen
    old,new=(ast.parse(v) for v in (frozen,fixed))
    defs=lambda tree:{n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=defs(old),defs(new)
    for n in a.keys()-{'watch'}:assert ast.dump(a[n],include_attributes=False)==ast.dump(b[n],include_attributes=False)
    before,after=a['watch'],b['watch']
    assert ast.dump(before.body[-1].finalbody[0],include_attributes=False)==ast.dump(after.body[-1].finalbody[0],include_attributes=False)
    assert 'os.kill(' not in fixed.decode() and 'os.killpg(' not in fixed.decode()


def test_unknown_monitor_source_is_refused():
    with pytest.raises(ValueError,match='Exact consumed'):recipe.build(b'unknown')


def test_actual_full_payload_bindings_and_exact_five_file_surface(package):
    oldc,oldb,oldp,old=integration.load_frozen(ROOT)
    assert package.p['scope']==oldp['scope'] and package.p['base_text']==oldp['base_text'] and package.p['design_text']==oldp['design_text']
    assert package.p['host_change']==oldp['host_change']
    assert len(package.sources)==69 and package.metadata['unchanged_uploaded_sources']==64
    changed={n for n in old if old[n]!=package.sources[n]}
    assert changed=={integration.STORE,integration.WATCH,integration.COORDINATOR,integration.SCRATCH,integration.BOOT}
    assert package.metadata['uploaded_source_bytes']==878631
    assert integration.digest(package.sources[integration.STORE])=='7bcbd863eab7c23940fa1d32b194448b88353202c2100b66a9a30f59e4ef6022'
    for n in old.keys()-changed:assert old[n]==package.sources[n]
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


def test_every_new_runtime_supervisor_and_contract_pin_joins(package):
    ct=json.loads(package.p['scratch_contract_text'])
    assert package.c['policy_sha256']==boot.POLICY_SHA==integration.digest(package.c['policy_text'].encode())
    assert scratch.CONTRACT_SHA==integration.digest(package.p['scratch_contract_text'].encode())
    assert ct['runtime_files']==package.manifest['runtime_files'] and len(ct['runtime_files'])==63
    assert ct['runtime_manifest_sha256']==integration.digest(integration.encoded(package.manifest))
    watch=integration.digest(package.sources[integration.WATCH])
    assert scratch.PINNED_SUPPORT['predictor_pgd_split_watch_v1.py']==package.p['fixed_support']['predictor_pgd_split_watch_v1.py']==ct['sources']['models/predictor_pgd_split_watch_v1.py']==watch
    for n,d in package.c['supervisor_sources'].items():assert d=={'bytes':len(package.sources['supervisor/'+n]),'sha256':integration.digest(package.sources['supervisor/'+n])}
    assert package.p['runtime_repair']['runtime_manifest_text'].encode()==integration.encoded(package.manifest)


def test_local_reader_bridge_joins_real_full_catalogue(package):
    result=integration.verify_readers(ROOT,package.readers)
    assert result['source_identity_verified'] and result['reader_files']==3
    assert len(package.layout.assets)==172 and len(package.layout.result)==419
    assert collection.boot is boot and collection.scratch is scratch
    for name,raw in package.readers.items():assert (ROOT/name).read_bytes()==raw
    server=package.sources[integration.COLLECTION]
    local=package.readers[integration.READERS['collection']]
    assert local.replace(b'predictor_pgd_close_bootstrap_v1 as boot',b'predictor_pgd_split_bootstrap_v1 as boot').replace(b'predictor_pgd_close_scratch_v1 as scratch',b'predictor_pgd_split_scratch_v1 as scratch')==server


@pytest.mark.parametrize('fault',['source','contract','old-control'])
def test_stale_local_or_old_control_bindings_refuse(package,tmp_path,fault):
    if fault=='source':
        for n,data in package.readers.items():
            p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data+(b'\n' if n==integration.READERS['scratch'] else b''))
        with pytest.raises(ValueError):integration.verify_readers(tmp_path,package.readers)
    elif fault=='contract':
        oldp=integration.load_frozen(ROOT)[2]
        with pytest.raises(ValueError,match='Exact source/storage'):scratch.Layout(oldp['scratch_contract_text'].encode(),package.p['base_text'].encode(),package.p['design_text'].encode(),package.ib,package.isha)
    else:
        oldb=integration.load_frozen(ROOT)[1]
        with pytest.raises(ValueError,match='Exact new supervised'):boot.control(integration.encoded(oldb))


def coordinator_namespace(package):
    ns={'__name__':'EXPLICIT-UNEXECUTED-COORDINATOR-DEFINITIONS','__file__':'<LOCAL-CANDIDATE-COORDINATOR>'}
    exec(compile(package.sources[integration.COORDINATOR],'<LOCAL-CANDIDATE-COORDINATOR>','exec'),ns)
    watch=monitor_namespace(package.sources[integration.WATCH]);watch['__file__']=str(RAW/'UNEXECUTED-watch.py')
    ns.update(boot=boot,scratch=scratch,frozen_watch=NS(**watch))
    return ns


@pytest.mark.parametrize('role',['proof','numerical'])
def test_actual_coordinator_stage_adapter_uses_repaired_monitor_without_clock_reset(package,role):
    ns=coordinator_namespace(package);fn=ns['monitor_for'](role)
    monitor=monitor_namespace(package.sources[integration.WATCH]);state,unused=simulate(monitor,status=0)
    def sample(pid,birth):
        if not state.calls:
            state.calls.append('sample');return {'processes':{pid:row(rss=5*2**30)},'rss_bytes':5*2**30}
        state.calls.append('race');raise ProcessLookupError('EXPLICIT exit')
    result=fn(state.child,state.lease,sample=sample,lease_factory=lambda *a:None,gate=lambda:None,
              phase_deadline=60 if role=='proof' else 1680,work_deadline=2400,launch_started=0,
              clock=state.clock,pause=state.clock.pause)
    assert result['parent_observation_passed'] and result['role']==role and result['RSS_ceiling_enforced'] is False
    assert result['peak_observed_RSS_bytes']==5*2**30 and state.lease.closed


def test_component_monitor_still_enforces_original_RSS_ceiling(package):
    ns=coordinator_namespace(package);fn=ns['monitor_for']('native')
    state,unused=simulate(monitor_namespace(package.sources[integration.WATCH]),status=0)
    with pytest.raises(MemoryError,match='4 GiB'):
        fn(state.child,state.lease,sample=lambda pid,birth:{'processes':{pid:row(rss=4*2**30+1)},'rss_bytes':4*2**30+1},
           lease_factory=lambda *a:None,gate=lambda:None,phase_deadline=120,work_deadline=150,launch_started=0,
           clock=state.clock,pause=state.clock.pause)
    assert state.lease.closed


def test_full_419_result_store_real_byte_publication_and_later_foreign_change_refusal(package,tmp_path):
    ns={'__name__':'LOCAL-FULL-CATALOGUE-STORE'}
    exec(compile(package.sources[integration.STORE],'<ACTUAL-CANDIDATE-FULL-STORE>','exec'),ns)
    root=tmp_path/'capture';root.mkdir(mode=0o700)
    inv=ns['Inventory'](root,base_bytes=package.p['base_text'].encode(),design_bytes=package.p['design_text'].encode())
    assert len(inv.spec)==419
    store=ns['Store'](inv);key='results/numerical/objective-0/pixels.npy';data=b'INERT-BYTE-ONLY\n'*100
    result=store.write_bytes(key,data);assert result['sha256']==hashlib.sha256(data).hexdigest()
    store.check();assert (root/key).read_bytes()==data
    # Explicit foreign metadata modification; never a fabricated native close.
    st=(root/key).stat();os.utime(root/key,ns=(st.st_atime_ns,st.st_mtime_ns+1000000))
    with pytest.raises(ValueError,match='Previously committed') as raised:store.check()
    assert 'mtime_ns' in raised.value.__notes__[0]


def test_numerical_sources_and_guard_functions_unchanged(package):
    _,_,_,old=integration.load_frozen(ROOT)
    assert all(package.sources[n]==data for n,data in old.items() if n.startswith('overlay/') and n!=integration.STORE)
    funcs=lambda src:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(src).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    before,after=funcs(old[integration.COORDINATOR]),funcs(package.sources[integration.COORDINATOR])
    assert before==after # Only WATCH_SHA literal changes, no coordinator function.
    assert package.sources['supervisor/scratch_allocation_v2.py']==old['supervisor/scratch_allocation_v2.py']
    assert not package.metadata['runnable_outer_client_or_target_handoff_prepared'] and not package.metadata['authorized']


@pytest.mark.parametrize('interrupt',[KeyboardInterrupt('EXPLICIT cancellation'),SystemExit('EXPLICIT exit')])
def test_wait_cancellation_propagates_and_cleans_owned_handles(package,interrupt):
    state,run=simulate(monitor_namespace(package.sources[integration.WATCH]),wait_error=interrupt)
    with pytest.raises(type(interrupt)) as raised:run()
    assert raised.value is interrupt and state.lease.closed and state.lease.signals==[signal.SIGTERM]
    assert len(state.calls)==2


def header_fixture(x):
    auth=boot.encoded({'kind':'pgd-split-supervised-university-exception-v1','approved':True,'control_sha256':x.sha,
                       'maximum_new_attempts':1,'user_basis':'EXPLICIT INERT HEADER FIXTURE, NOT A TARGET GRANT'})
    session={'kind':'pgd-split-parent-session-v1','control_sha256':x.sha,'parent_pid':os.getpid(),
             'connection_started':0.,'work_deadline':2400.,'connection_deadline':3000.,'phase':'failed',
             'complete':False,'completed':[],'receipts':[],'scientific_qualified':False,'training_admitted':False}
    win=collection.Window(0.,100.,clock=lambda:100.)
    receipt=collection.work_receipt(boot.encoded(session),x.sha,boot.digest(auth),x.isha,win)
    extras={'control.json':x.control,'authorization.json':auth,'session.json':boot.encoded(session),
            collection.RECEIPT:boot.encoded(receipt),collection.STATUS:boot.encoded({'EXPLICIT':'INERT SYNTHETIC METADATA'})}
    pins=dict(x.layout.assets);pins.update({n:{'bytes':len(v),'sha256':boot.digest(v)} for n,v in extras.items()})
    uid=os.getuid();files={n:dict(d,mode=0o600,mtime_ns=1000000000+i) for i,(n,d) in enumerate(sorted(pins.items()))}
    inventory={n:{'identity':[500,1000+i,uid,0o600,1],'bytes':v['bytes'],'allocated_bytes':((v['bytes']+4095)//4096)*4096,
                  'mtime_ns':v['mtime_ns']} for i,(n,v) in enumerate(files.items())}
    inventory['emergency-space.bin']={'identity':[500,900,uid,0o600,1],'bytes':16*scratch.MiB,'allocated_bytes':16*scratch.MiB,'mtime_ns':100}
    dirs={'.'}
    for name in inventory:dirs.update(str(p) for p in Path(name).parents)
    directories={n:{'identity':[500,2000+i,uid,0o700],'allocated_bytes':4096} for i,n in enumerate(sorted(dirs))}
    storage=x.layout.assess(inventory,directories,{'available_bytes':8*scratch.GiB,'available_inodes':2048},phase='collection')
    h={'kind':'pgd-split-evidence-stream-v1','control_sha256':x.sha,'authorization_sha256':boot.digest(auth),
       'root_binding':{'root':'/EXPLICIT-INERT/scratch','root_identity':list(directories['.']['identity']),
                       'parent':'/EXPLICIT-INERT','parent_identity':[500,1,uid,0o700]},
       'files':files,'inventory':inventory,'directories':directories,'storage':storage,
       'final_receipt_sha256':boot.digest(extras[collection.RECEIPT]),'session_sha256':boot.digest(extras['session.json']),
       'collection_started':100.,'collection_deadline':700.,'connection_started':0.,'connection_deadline':3000.,
       'logical_bytes':sum(v['bytes'] for v in files.values()),'omitted_content':['emergency-space.bin'],
       'scientific_qualified':False,'training_admitted':False}
    return h,auth


@pytest.mark.parametrize('fault',['none','old-store','old-monitor','old-control','admission'])
def test_full_catalogue_collection_reader_accepts_only_repaired_pins(package,tmp_path,fault):
    h,auth=header_fixture(package)
    if fault=='old-store':
        h['files']['overlay/models/predictor_pgd_split_store_v1.py']['sha256']=integration.store_recipe.FROZEN_SHA
    elif fault=='old-monitor':
        h['files']['supervisor/predictor_pgd_split_watch_v1.py']['sha256']=recipe.FROZEN_SHA
    elif fault=='old-control':
        h['control_sha256']=integration.load_frozen(ROOT)[0]['supervised_control_sha256']
    elif fault=='admission':
        h['scientific_qualified']=True
    if fault=='none':
        layout=collection.validate_header(h,package.control,auth)
        assert len(layout.assets)==172 and len(layout.result)==419
        (tmp_path/'INERT-FULL-CATALOGUE-HEADER.json').write_bytes(boot.encoded(h))
    else:
        with pytest.raises(ValueError):collection.validate_header(h,package.control,auth)


@pytest.mark.parametrize('fault',['none','store','watch'])
def test_pinned_importer_actual_69_source_reads_keep_repaired_bytes_cold(package,tmp_path,fault):
    root=tmp_path/'source-only';root.mkdir(mode=0o700)
    for name,data in package.sources.items():
        p=root/name;p.parent.mkdir(parents=True,mode=0o700,exist_ok=True)
        # Parents made with parents=True have ambient modes; all are made private.
        parent=p.parent
        while parent!=root:parent.chmod(0o700);parent=parent.parent
        p.write_bytes(data);p.chmod(0o600)
    binding=scratch.identity_binding(root)
    if fault!='none':
        name=integration.STORE if fault=='store' else integration.WATCH
        with (root/name).open('ab') as f:f.write(b'EXPLICIT TAMPER')
        with pytest.raises(ValueError):boot.PinnedImporter(root,binding,package.c,package.p)
    else:
        finder=boot.PinnedImporter(root,binding,package.c,package.p)
        assert len(finder.files)==69 and not finder.loaded and not finder.installed
        assert finder.files['models.predictor_pgd_split_store_v1'][1]==package.sources[integration.STORE]
        assert finder.files['models.predictor_pgd_split_watch_v1'][1]==package.sources[integration.WATCH]
        assert len(finder.packet_imports)==2
        assert finder.find_spec('models.predictor_pgd_split_store_v1').loader is finder
        assert 'torch' not in sys.modules and 'numpy' not in sys.modules
