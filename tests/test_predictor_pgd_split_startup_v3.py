"""Actual loader/catalogue, AST imports only, fake owned children; no science."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import sys
from types import ModuleType,SimpleNamespace as NS
import pytest
from models import predictor_pgd_split_startup_sources_v3 as source
from models import predictor_pgd_split_bootstrap_v1 as oldboot
from models import predictor_pgd_split_coordinator_v1 as oldcoord
from models import predictor_pgd_split_watch_v1 as watch

ROOT=Path(__file__).resolve().parents[1]
SAVED=ROOT/'runs/pretrained/diagnostic/predictor-pgd-split-outer-v2-20261116/handoff-parent/linux10603-unarmed/attended-session/saved'

def write(root,name,data):
    p=root/name;p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    q=p.parent
    while q!=root:q.chmod(0o700);q=q.parent
    p.write_bytes(data);p.chmod(0o600);return p


def body(path):return (ROOT/path).read_bytes()


def test_source_builders_retain_both_complete_consumed_prefixes_and_frozen_science():
    boot=body('models/predictor_pgd_split_bootstrap_v1.py');coordinator=body('models/predictor_pgd_split_coordinator_v1.py')
    newboot=source.build_bootstrap(boot);newcoord,edits=source.build_coordinator(coordinator)
    assert newboot.startswith(boot+b'\n')
    assert newcoord.startswith((SAVED/'supervisor/predictor_pgd_split_coordinator_v1.py').read_bytes()+b'\n')
    assert edits=={'child':1,'guard_rebind':1,'launch_gate':1,'lease':1,'prepare_gpu':1,'parent_error_wrappers':2}
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    with pytest.raises(ValueError):source.build_bootstrap(boot+b'changed')
    with pytest.raises(ValueError):source.build_coordinator(coordinator+b'changed')


@pytest.fixture
def installed(tmp_path):
    root=tmp_path/'private-root';root.mkdir(mode=0o700)
    c=json.loads((SAVED/'control.json').read_bytes());p=json.loads(c['policy_text']);contract=json.loads(p['scratch_contract_text'])
    raw={}
    for path,d in contract['runtime_files'].items():raw['overlay/'+path]=(SAVED/'overlay'/path).read_bytes()
    for path,d in c['supervisor_sources'].items():raw['supervisor/'+path]=(SAVED/'supervisor'/path).read_bytes()
    newboot=source.build_bootstrap(body('models/predictor_pgd_split_bootstrap_v1.py'))
    newcoord,_=source.build_coordinator(body('models/predictor_pgd_split_coordinator_v1.py'))
    raw['supervisor/bootstrap.py']=newboot;raw['supervisor/predictor_pgd_split_coordinator_v1.py']=newcoord
    for path,data in raw.items():
        write(root,path,data)
        if path.startswith('supervisor/'):
            c['supervisor_sources'][path.removeprefix('supervisor/')]={'bytes':len(data),'sha256':oldboot.digest(data)}
    binding={'root':str(root),'root_identity':list(oldboot.dir_id(root.lstat())),'parent':str(root.parent),'parent_identity':list(oldboot.dir_id(root.parent.lstat()))}
    ns={'__name__':'EXPLICIT_STDLIB_BOOTSTRAP_DEFINITIONS_ONLY','__file__':'<CHECKED-STDLIB-BOOTSTRAP>'}
    exec(compile(newboot,ns['__file__'],'exec'),ns) # ONLY stdlib supervisor definitions.
    proxy=NS(modules={},meta_path=[]);ns['sys']=proxy
    finder=ns['PinnedImporter'](root,binding,c,p);assert len(finder.files)==69 and finder.loaded==set()
    assert not (root/'code').exists() # Constructor must not require unreceived packet.
    finder.install();assert proxy.meta_path==[finder]
    for name,(path,d,package) in source.PACKET_IMPORTS.items():
        data=(SAVED/path).read_bytes();assert len(data)==d['bytes'] and oldboot.digest(data)==d['sha256'];write(root,path,data)
    return NS(root=root,c=c,p=p,contract=contract,raw=raw,ns=ns,proxy=proxy,finder=finder,tmp=tmp_path)


def load_inert(x,name):
    parent=name.rpartition('.')[0]
    if parent and parent not in x.proxy.modules:load_inert(x,parent)
    path=x.proxy.modules[parent].__path__ if parent else None
    spec=x.finder.find_spec(name,path)
    assert spec is not None
    module=importlib.util.module_from_spec(spec);x.proxy.modules[name]=module
    calls=[]
    def inert(code,namespace):
        # Compiled actual source is deliberately NOT executed here.
        calls.append((code.co_filename,namespace['__name__']))
    x.ns['exec']=inert;x.finder.exec_module(module)
    assert calls and module.__file__==str(x.root/x.finder.files[name][0])
    return module


def test_packet_names_resolve_only_after_extraction_and_use_real_private_reader(installed):
    x=installed;root=load_inert(x,'datasets');child=load_inert(x,'datasets.prepared')
    assert root.__path__==[str(x.root/'code/datasets')]
    assert child.__loader__ is root.__loader__ is x.finder
    assert x.ns['require_namespace'](x.root,x.finder.binding,oldboot.digest(oldboot.encoded(x.c))) is x.finder
    assert len(x.finder.files)==71


def test_missing_packet_member_refuses_instead_of_ambient_fallback(installed):
    x=installed;p=x.root/'code/datasets/__init__.py';p.rename(p.with_name('PRESERVED-PACKET-SOURCE'))
    with pytest.raises(FileNotFoundError):x.finder.find_spec('datasets')
    assert not x.finder.loaded


@pytest.mark.parametrize('fault',['bytes','symlink','public-mode','hardlink'])
def test_packet_dependency_changed_or_unowned_input_is_refused(installed,fault):
    x=installed;p=x.root/'code/datasets/__init__.py'
    if fault=='bytes':p.write_bytes(b'CHANGED')
    elif fault=='public-mode':p.chmod(0o644)
    elif fault=='hardlink':os.link(p,p.with_name('INERT-HARDLINK'))
    else:
        saved=p.with_name('PRESERVED');p.rename(saved);p.symlink_to(saved)
    with pytest.raises((ValueError,OSError)):x.finder.find_spec('datasets')


def test_changed_after_spec_is_rechecked_before_any_execution(installed):
    x=installed;spec=x.finder.find_spec('datasets');module=importlib.util.module_from_spec(spec)
    (x.root/'code/datasets/__init__.py').write_bytes(b'CHANGED AFTER SPEC')
    x.ns['exec']=lambda *a:pytest.fail('Changed source executed')
    with pytest.raises(ValueError):x.finder.exec_module(module)
    assert not x.finder.loaded


@pytest.mark.parametrize('fault',['preloaded','unknown','reload','path','foreign-parent'])
def test_namespace_pollution_unlisted_names_reloads_and_paths_fail_closed(installed,fault):
    x=installed
    if fault=='preloaded':
        x.proxy.modules['datasets']=ModuleType('datasets')
        other=x.ns['PinnedImporter'](x.root,x.finder.binding,x.c,x.p)
        with pytest.raises(ValueError):other.install()
    elif fault=='unknown':
        with pytest.raises(ModuleNotFoundError):x.finder.find_spec('datasets.UNREVIEWED')
        with pytest.raises(ModuleNotFoundError):x.finder.find_spec('models.UNREVIEWED')
    else:
        parent=load_inert(x,'datasets')
        if fault=='reload':
            with pytest.raises(ImportError):x.finder.find_spec('datasets',target=parent)
        else:
            if fault=='path':parent.__path__=['/ambient/NOT-ALLOWED']
            else:parent.__loader__=object()
            with pytest.raises(ImportError):x.finder.find_spec('datasets.prepared',parent.__path__)
            with pytest.raises(ImportError):x.ns['require_namespace'](x.root,x.finder.binding,oldboot.digest(oldboot.encoded(x.c)))


def test_descriptor_drift_and_packet_model_shadow_cannot_change_providers(installed):
    x=installed;p=copy.deepcopy(x.p);c=json.loads(p['scratch_contract_text']);c['packet']['files']['datasets/prepared.py']['sha256']='f'*64;p['scratch_contract_text']=oldboot.encoded(c).decode()
    with pytest.raises(ValueError):x.ns['PinnedImporter'](x.root,x.finder.binding,x.c,p)
    old_packet=(SAVED/'code/models/native_diagnostic.py').read_bytes();write(x.root,'code/models/native_diagnostic.py',old_packet)
    assert x.finder.files['models.native_diagnostic'][0]=='overlay/models/native_diagnostic.py'
    assert x.finder.find_spec('json') is None # stdlib remains external to project finder.


def test_complete_actual_catalogue_import_resolution_with_no_scientific_execution(installed):
    x=installed;visited=set();edges=[];inactive=[];dynamic=[]
    declared=set(x.finder.files)|set(x.finder.packet_imports)
    def resolve(name):
        if name in visited:return
        parent=name.rpartition('.')[0]
        if parent:resolve(parent)
        if name in visited:return
        module_path=x.proxy.modules[parent].__path__ if parent else None
        spec=x.finder.find_spec(name,module_path);assert spec is not None,name
        module=importlib.util.module_from_spec(spec);x.proxy.modules[name]=module;visited.add(name)
        x.finder.exec_module(module) # exec boundary below only visits AST imports.
    def ast_imports_only(code,namespace):
        name=namespace['__name__'];data=x.finder.files[name][1]
        assert code.co_filename==str(x.root/x.finder.files[name][0])
        imports,ignored,sites=source.active_import_nodes(ast.parse(data))
        inactive.extend({'module':name,'line':n.lineno,'import':ast.unparse(n)} for n in ignored)
        for n in sites:
            assert len(n.args)>=1 and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str)
            value=n.args[0].value;assert value=='sys' or value.startswith('src.'),value
            dynamic.append({'module':name,'line':n.lineno,'target':value})
        for n in imports:
            targets=[]
            if isinstance(n,ast.Import):targets=[a.name for a in n.names]
            elif n.level:pytest.fail('Relative project import needs explicit source classification')
            elif n.module in ('models','attacks','datasets'):targets=[n.module+'.'+a.name for a in n.names]
            elif n.module:targets=[n.module]
            for target in targets:
                if target.split('.')[0] not in ('models','attacks','datasets'):continue
                edges.append({'from':name,'line':n.lineno,'to':target});assert target in declared,target
                resolve(target)
    x.ns['exec']=ast_imports_only
    for name in sorted(declared):resolve(name)
    assert visited==declared and len(visited)==71 and x.finder.loaded==declared
    assert len(inactive)==7 and len(dynamic)==14
    inactive_project=[n for n in inactive if 'from models import' in n['import']]
    assert len(inactive_project)==4
    assert x.ns['require_namespace'](x.root,x.finder.binding,oldboot.digest(oldboot.encoded(x.c))) is x.finder
    assert all(x.finder.files[n][0].startswith('overlay/') for n in ('models.native_diagnostic','models.vjepa2'))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    write(x.tmp,'ACTUAL-CATALOGUE-IMPORT-RESOLUTION.json',oldboot.encoded({'kind':'actual-pinned-loader-AST-import-resolution-v3',
        'modules':sorted(visited),'edges':edges,'inactive_literal_false_imports':inactive,'dynamic_imports':dynamic,
        'actual_scientific_source_compiled_but_never_executed':True,'external_library_imports_executed':False,
        'not_an_actual_runtime_import_or_scientific_test':True,'authorized':False,'training_admitted':False}))


class Child:
    pid=901
    def __init__(self,values):self.values=list(values);self.status=None;self.waits=[]
    def poll(self):
        if self.values:self.status=self.values.pop(0)
        return self.status
    def wait(self,timeout):self.waits.append(timeout);self.status=1;return 1
class Lease:
    pid=901;birth=111
    def __init__(self):self.closed=False;self.ended=False;self.sent=[]
    def dead(self):return self.ended
    def send(self,sig):assert sig in (signal.SIGTERM,signal.SIGKILL);self.sent.append(sig);self.ended=True
    def close(self):self.closed=True


def diagnostic_namespace(base_driver,base_coordinator):
    ns={'OuterDriver':base_driver,'OuterCoordinator':base_coordinator,'boot':oldboot}
    exec(compile(ast.parse(source.COORDINATOR_EXTENSION),'<STDLIB-DIAGNOSTICS-ONLY>','exec'),ns)
    return ns


@pytest.mark.parametrize('role',['native','numerical'])
@pytest.mark.parametrize('when',['already-ended','gone-after-poll','still-unconfirmed'])
def test_original_monitor_early_exit_keeps_failure_and_closes_only_owned_fake_handle(tmp_path,role,when):
    lease=Lease();values=[1] if when=='already-ended' else [None,1] if when=='gone-after-poll' else [None,None]
    child=Child(values);clock=lambda:10.0
    def missing(*a):raise ProcessLookupError(3,'No such process')
    monitor=oldcoord.monitor_for(role)
    exc=ProcessLookupError if when=='still-unconfirmed' else RuntimeError
    with pytest.raises(exc) as info:
        monitor(child,lease,sample=missing,lease_factory=lambda *a:pytest.fail('No descendant should be bound'),gate=lambda:None,
            phase_deadline=100.0,work_deadline=200.0,launch_started=9.0,clock=clock,pause=lambda *a:pytest.fail('No real pause'))
    assert lease.closed and child.status==1 and child.waits and max(child.waits)<=.1
    assert all(sig in (signal.SIGTERM,signal.SIGKILL) for sig in lease.sent)
    ns=diagnostic_namespace(type('BaseDriver',(),{}),type('BaseCoordinator',(),{}))
    record=ns['_startup_error_record'](info.value)
    assert record['error_type']==exc.__name__ and record['frames'] and len(oldboot.encoded(record))<=16*1024
    write(tmp_path,'INERT-EXIT-ERROR.json',oldboot.encoded({'role':role,'when':when,'diagnostic':record,
        'owned_handle_closed':lease.closed,'child_exit_status':child.status,'actual_process_or_signal':False}))


def test_new_wrappers_retain_original_error_and_attach_child_status_before_bounded_save():
    error=ProcessLookupError(3,'No such process');saved=[];child=Child([1])
    class BaseDriver:
        outer_child=child
        def launch(self,*a):raise error
        def save(self,state,replace):assert replace;saved.append(copy.deepcopy(state))
    class BaseCoordinator:
        claimed=True;state={'phase':'failed','complete':False}
        def run_work(self):return self.driver.launch('numerical',{})
    ns=diagnostic_namespace(BaseDriver,BaseCoordinator);owner=ns['OuterCoordinator']();owner.driver=ns['OuterDriver']()
    with pytest.raises(ProcessLookupError) as caught:owner.run_work()
    assert caught.value is error and saved[0]['phase']=='failed' and saved[0]['complete'] is False
    detail=saved[0]['parent_error_diagnostic']
    assert any('child exit status: 1' in n for n in detail['notes']) and detail['error_type']=='ProcessLookupError'
    assert any(n['function']=='launch' for n in detail['frames']) and 'locals' not in detail


@pytest.mark.parametrize('case',['unclaimed','save-fails','huge-message'])
def test_diagnostic_failure_never_promotes_or_replaces_primary_error(case):
    error=RuntimeError('界'*20000 if case=='huge-message' else 'PRIMARY');saved=[]
    class Driver:
        def save(self,state,replace):
            if case=='save-fails':raise OSError('INERT SAVE FAILURE')
            saved.append(state)
    class Base:
        claimed=case!='unclaimed';state={'phase':'failed'};driver=Driver()
        def run_work(self):raise error
    ns=diagnostic_namespace(type('DriverBase',(),{}),Base)
    with pytest.raises(RuntimeError) as caught:ns['OuterCoordinator']().run_work()
    assert caught.value is error
    if case=='unclaimed':assert not saved
    elif case=='save-fails':assert any('could not be saved' in n for n in error.__notes__)
    else:assert len(oldboot.encoded(saved[0]['parent_error_diagnostic']))<=16*1024


def test_same_pinned_loader_is_accepted_by_the_separately_imported_bootstrap_copy(installed):
    x=installed
    load_inert(x,'models');load_inert(x,'datasets');load_inert(x,'datasets.prepared')
    second={'__name__':'SECOND_CHECKED_BOOTSTRAP_NAMESPACE','__file__':'<SECOND-STDLIB-BOOTSTRAP-COPY>'}
    data=source.build_bootstrap(body('models/predictor_pgd_split_bootstrap_v1.py'))
    exec(compile(data,second['__file__'],'exec'),second)
    second['sys']=x.proxy
    assert second['PinnedImporter'] is not x.ns['PinnedImporter']
    assert second['require_namespace'](x.root,x.finder.binding,oldboot.digest(oldboot.encoded(x.c))) is x.finder
