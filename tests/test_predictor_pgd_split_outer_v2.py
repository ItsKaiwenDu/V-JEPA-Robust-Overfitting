"""New v2 integration checks with the REAL catalogue and tiny ordinary storage.

Host, installer activation, preallocation, packet contents and proof admission are
substituted explicitly. Real directory planning/scanning/source caching executes;
only single frozen worker mkdir expressions run, never scientific workers.
"""
import ast
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_split_outer_v2 as o
from models import predictor_pgd_split_outer_client_v2 as client
from models import predictor_pgd_split_directories_v2 as policy
from models import predictor_pgd_split_bootstrap_v1 as boot
from models import predictor_pgd_split_coordinator_v1 as coordinator
from models import predictor_pgd_split_scratch_v1 as scratch
from models import predictor_pgd_split_store_v1 as store
from models import predictor_pgd_split_source_bundle_v1 as sources

ROOT=Path(__file__).resolve().parents[1]
FAILED=ROOT/'runs/pretrained/diagnostic/predictor-pgd-split-outer-v1-20261115/handoff-parent/linux10603-final-unarmed'
SAVED=FAILED/'attended-session/saved'
BASIS='EXPLICIT LOCAL INERT TEST METADATA ONLY; NOT A LIVE USER GRANT'


def dump(n):return ast.dump(n,include_attributes=False)
def functions(t):return {n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def write(root,name,data):
    p=root/name;p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    q=p.parent
    while q!=root:q.chmod(0o700);q=q.parent
    p.write_bytes(data);p.chmod(0o600);return p

@pytest.fixture(scope='module')
def package(tmp_path_factory):
    parent=tmp_path_factory.mktemp('UNARMED-V2-SOURCE-PACKAGE');parent.chmod(0o700)
    folder=parent/'candidate'
    result=o.prepare_package(ROOT,folder)
    raw=(folder/'CONTROL-CANDIDATE.json').read_bytes()
    c,b,p,sha=o.validate_control(raw,o.digest((ROOT/'models/predictor_pgd_split_outer_v2.py').read_bytes()))
    mapping=o.parsed((folder/'SOURCE-PAYLOAD.json').read_bytes());raw_sources=o.decode_sources(mapping,c,b)
    auth=o.encoded(o.authorization(c,sha,BASIS)) # MEMORY ONLY; never an armed test handoff
    return NS(folder=folder,result=result,raw=raw,c=c,b=b,p=p,sha=sha,mapping=mapping,sources=raw_sources,auth=auth)


def test_new_package_keeps_every_uploaded_worker_and_supervisor_byte_exact_and_unarmed(package):
    x=package
    assert x.c['kind']=='pgd-split-outer-control-v2' and len(x.sources)==69
    assert x.result['directory_policy_sha256']==o.DIRECTORY_POLICY_SHA
    assert x.result['setup_policy_preserves_worker_claims']
    assert x.c['supervised_control_text'].encode()==(SAVED/'control.json').read_bytes()
    for name,data in x.sources.items():assert data==(SAVED/name).read_bytes()
    assert x.p==json.loads((ROOT/'configs/predictor_pgd_split_launch_v1.json').read_bytes())
    assert not (x.folder/'AUTHORIZATION.json').exists() and not (x.folder/'ATTEMPT.json').exists()
    assert o.remote_root(x.sha)!=o.remote_root(json.loads((FAILED/'ATTEMPT.json').read_bytes())['outer_control_sha256'])
    assert o.remote_root(x.sha).name.startswith('vjepa-pgd-split-v1-') # original frozen host gate namespace
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


def test_inline_directory_function_and_constants_are_exact_reviewed_pure_AST():
    o.verify_directory_policy_sources(ROOT,sources)
    helper=ast.parse((ROOT/'models/predictor_pgd_split_directories_v2.py').read_bytes())
    module=ast.parse((ROOT/'models/predictor_pgd_split_outer_v2.py').read_bytes())
    assert set(functions(helper))=={'plan'}
    fn=copy.deepcopy(functions(module)['_setup_directory_plan']);fn.name='plan'
    class Restore(ast.NodeTransformer):
        def visit_Name(self,n):
            n.id={'_DIRECTORY_SHELL':'SHELL','_DIRECTORY_WORKER_CLAIMS':'WORKER_CLAIMS'}.get(n.id,n.id);return n
    assert dump(Restore().visit(fn))==dump(functions(helper)['plan'])


@pytest.mark.parametrize('fault',['old-control','old-authority','missing-policy','policy-sha','extra-local-source','changed-runtime','paid','second-attempt'])
def test_v2_source_and_scope_identity_refuses_v1_reuse_or_broader_permission(package,fault):
    c=copy.deepcopy(package.c);a=o.parsed(package.auth)
    if fault=='old-control':c['kind']='pgd-split-outer-control-v1'
    elif fault=='missing-policy':c['local_source_sha256'].pop('models/predictor_pgd_split_directories_v2.py')
    elif fault=='policy-sha':c['local_source_sha256']['models/predictor_pgd_split_directories_v2.py']='f'*64
    elif fault=='extra-local-source':c['local_source_sha256']['models/UNREVIEWED.py']='e'*64
    elif fault=='changed-runtime':c['runtime_files'][next(iter(c['runtime_files']))]['sha256']='f'*64
    else:
        if fault=='old-authority':a=o.parsed((FAILED/'AUTHORIZATION.json').read_bytes())
        elif fault=='paid':a['paid_compute']=True
        else:a['maximum_new_attempts']=2
        with pytest.raises(ValueError):o.validate_authorization(o.encoded(a),c,package.sha)
        return
    with pytest.raises((ValueError,KeyError)):o.validate_control(o.encoded(c),c['outer_source_sha256'])


def test_cold_source_check_rejects_inline_policy_drift_without_executing_it():
    original=(ROOT/'models/predictor_pgd_split_outer_v2.py').read_bytes()
    assert b'len(layout.assets) != 172' in original
    changed=original.replace(b'len(layout.assets) != 172',b'len(layout.assets) != 171',1)
    def read(path,**kw):
        return changed if path.name=='predictor_pgd_split_outer_v2.py' else sources.read_source(path,**kw)
    with pytest.raises(ValueError,match='differs'):o.verify_directory_policy_sources(ROOT,NS(read_source=read))


@pytest.fixture
def prepared(package,tmp_path,monkeypatch):
    root=tmp_path/'REAL-CATALOGUE-TINY-SETUP';root.mkdir(mode=0o700)
    events=[];cache=[];real_scans=[];holder={}
    class CacheOnly:
        def __init__(self,*a):
            finder=boot.PinnedImporter(*a);cache.append(finder)
            assert len(finder.files)==69 and not finder.loaded
        def install(self):
            events.append('SOURCE-CACHE-ONLY-NO-EXECUTION');return cache[-1]
    b=NS(control=boot.control,authorization=boot.authorization,installation=boot.installation,encoded=boot.encoded,PinnedImporter=CacheOnly)
    real_scan=scratch.Guard.scan
    def scan(self,**kw):
        value=real_scan(self,**kw);real_scans.append(value[0]);return value
    def verify(self,**kw):
        # Retain the REAL whole-tree scan and catalogue. Pixel/source-member
        # contents and full installation proof are not exercised by this test.
        self.scan(phase=kw.get('phase','setup'),settled=True)
        observed=o._setup_directory_plan(self.layout)
        assert len(observed['setup_directories'])==26
        for name in self.layout.assets:assert (root/str(PurePosixPath(name).parent)).is_dir()
        for name in observed['worker_claim_directories'].values():assert not (root/name).exists()
        holder['guard']=self;events.append('ASSET-CONTENTS-SUBSTITUTED-DIRECTORIES-REAL')
        return {'scientific_qualified':False}
    def readiness(self):events.append('PROOF-READINESS-SUBSTITUTED-NOT-EXECUTED');return {'proof_launch_authorized':False}
    monkeypatch.setattr(scratch.Guard,'scan',scan)
    monkeypatch.setattr(scratch.Guard,'verify_assets',verify)
    monkeypatch.setattr(scratch.Guard,'proof_readiness',readiness)
    monkeypatch.setattr(coordinator,'target_gate',lambda *a:events.append('HOST-GATE-SUBSTITUTED'))
    monkeypatch.setattr(o,'receive_packet',lambda *a:events.append('PACKET-RECEIVE-SUBSTITUTED-NO-PIXELS'))
    monkeypatch.setattr(o,'extract_packet',lambda *a:events.append('PACKET-EXTRACT-SUBSTITUTED-NO-PIXELS'))
    def allocate(fd,start,size):events.append(('EMERGENCY-ALLOCATION-SUBSTITUTED',start,size))
    def fstat(fd):
        info=os.fstat(fd);reserve=root/'emergency-space.bin'
        if reserve.exists() and reserve.stat().st_ino==info.st_ino:
            return NS(st_size=16*o.MiB,st_blocks=16*o.MiB//512)
        return info
    fake_os=NS(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')})
    fake_os.posix_fallocate=allocate;fake_os.fstat=fstat
    monkeypatch.setattr(o,'os',fake_os)
    result=o.prepare_tree(root,package.sources,package.c,package.b,package.p,o.parsed(package.auth),
        package.raw,package.auth,b,io.BytesIO(),lambda:None)
    guard=holder['guard'];layout=guard.layout
    assert type(layout) is scratch.Layout and len(layout.assets)==172 and len(layout.result)==419 and len(layout.directories)==39
    assert len(guard.scan(phase='setup')[2])==26
    assert len(cache)==1 and len(cache[0].files)==69 and not cache[0].installed and not cache[0].loaded
    assert (root/'emergency-space.bin').stat().st_size==0 # NO real reserve or proof allocation
    record=o.parsed((root/'runtime-source-manifest.json').read_bytes())
    assert record['setup_directory_plan']==o._setup_directory_plan(layout)==policy.plan(layout)
    assert record['files']=={n:{'bytes':len(data),'sha256':o.digest(data)} for n,data in package.sources.items()}
    assert len((root/'runtime-source-manifest.json').read_bytes())<128*1024
    return NS(root=root,guard=guard,layout=layout,result=result,record=record,events=events,cache=cache,
              fake_os=fake_os,tmp=tmp_path,real_scans=real_scans,package=package,boot=b)


def mkdir_only(role):
    if role=='numerical' or role.startswith('objective-'):
        path=SAVED/'overlay/models/_pgd_split_numerical_v1.py';name='folder.mkdir' if role=='numerical' else 'self.folder.mkdir'
    elif role=='boundary':path=SAVED/'overlay/models/_pgd_split_observer_v1.py';name='folder.mkdir'
    else:path=SAVED/'overlay/models/predictor_pgd_split_probe_job_v1.py';name='self.folder.mkdir'
    tree=ast.parse(path.read_bytes())
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)==name]
    assert len(nodes)==1 and nodes[0].value.func.attr=='mkdir'
    return compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(nodes[0])],type_ignores=[])),'<EXACT-FROZEN-SINGLE-MKDIR-ONLY>','exec')


def claim(prepared,role):
    p=prepared.root/policy.WORKER_CLAIMS[role]
    if role in ('native','forward-preserving'):
        for parent in (prepared.root/'capture/results',prepared.root/'capture/results/components'):
            try:parent.mkdir(mode=0o700)
            except FileExistsError:pass
    exec(mkdir_only(role),{'folder':p,'self':NS(folder=p)})
    return p


def test_integrated_setup_records_the_actual_minimal_plan_and_leaves_all_workers_fresh(prepared):
    x=prepared
    assert len(x.record['setup_directory_plan']['setup_directories'])==26
    assert len(x.record['setup_directory_plan']['allowed_but_not_precreated'])==13
    assert len(x.real_scans)>=3
    assert x.events.index('SOURCE-CACHE-ONLY-NO-EXECUTION')<x.events.index('HOST-GATE-SUBSTITUTED')<x.events.index('PACKET-RECEIVE-SUBSTITUTED-NO-PIXELS')
    assert x.events[-1]=='PROOF-READINESS-SUBSTITUTED-NOT-EXECUTED'
    write(x.tmp,'REAL-CATALOGUE-SETUP-EVENTS.json',o.encoded({'origin':'ACTUAL V2 SETUP/REAL CATALOGUE/SCANS; HOST/ALLOCATION/CONTENTS/PROOF SUBSTITUTED',
        'events':x.events,'source_modules_cached_without_execution':69,'setup_directories':26,
        'worker_claims_not_precreated':12,'physical_reserve_bytes':0,'scientific_qualified':False,'training_admitted':False}))


@pytest.mark.parametrize('role',['numerical','boundary','native','forward-preserving'])
def test_each_frozen_entry_claim_succeeds_after_actual_v2_setup(prepared,role):
    p=claim(prepared,role)
    assert p.is_dir() and stat.S_IMODE(p.stat().st_mode)==0o700
    assert all(not (prepared.root/name).exists() for r,name in policy.WORKER_CLAIMS.items()
               if r!=role and not (role=='numerical' and r.startswith('objective-')))


def test_all_twelve_frozen_claims_and_typed_store_remain_compatible_with_integrated_setup(prepared):
    x=prepared
    for role in policy.WORKER_CLAIMS:assert not (x.root/policy.WORKER_CLAIMS[role]).exists();claim(x,role)
    inv=store.Inventory(x.root/'capture',base_bytes=x.package.p['base_text'].encode(),design_bytes=x.package.p['design_text'].encode())
    assert inv.scan()['files']=={}
    x.guard.scan(phase='setup')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


def test_existing_worker_evidence_and_completed_setup_cannot_be_reused_or_deleted(prepared):
    x=prepared;folder=claim(x,'numerical');file=write(folder,'PRESERVED.txt',b'INERT OLD WORKER EVIDENCE')
    with pytest.raises(FileExistsError):claim(x,'numerical')
    before=(x.root/'runtime-source-manifest.json').read_bytes()
    with pytest.raises(FileExistsError):o.prepare_tree(x.root,x.package.sources,x.package.c,x.package.b,x.package.p,o.parsed(x.package.auth),
        x.package.raw,x.package.auth,x.boot,io.BytesIO(),lambda:None)
    assert file.read_bytes()==b'INERT OLD WORKER EVIDENCE' and (x.root/'runtime-source-manifest.json').read_bytes()==before


def test_unarmed_v2_entry_refuses_before_any_client_activity(package):
    import importlib.util
    spec=importlib.util.spec_from_file_location('EXPLICIT_UNARMED_V2_ENTRY',ROOT/'scripts/run_predictor_pgd_split_outer_v2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(SystemExit,match='No timer, claim, SSH or model work'):module.main([str(package.folder)])
    assert not (package.folder/'ATTEMPT.json').exists()


def test_v2_client_and_entry_only_change_the_bound_module_version():
    old=(ROOT/'models/predictor_pgd_split_outer_client_v1.py').read_bytes()
    new=(ROOT/'models/predictor_pgd_split_outer_client_v2.py').read_bytes()
    assert new.replace(b'predictor_pgd_split_outer_v2 as outer',b'predictor_pgd_split_outer_v1 as outer')==old
    old=(ROOT/'scripts/run_predictor_pgd_split_outer_v1.py').read_bytes();new=(ROOT/'scripts/run_predictor_pgd_split_outer_v2.py').read_bytes()
    assert new.replace(b'predictor_pgd_split_outer_client_v2 import run',b'predictor_pgd_split_outer_client_v1 import run')==old


def test_v2_timers_transport_hardware_and_claim_bodies_remain_exact_v1_AST():
    old=ast.parse((ROOT/'models/predictor_pgd_split_outer_v1.py').read_bytes())
    new=ast.parse((ROOT/'models/predictor_pgd_split_outer_v2.py').read_bytes())
    a,b=functions(old),functions(new)
    changed={'validate_control','authorization','prepare_tree','prepare_package'}
    for name in set(a)-changed:assert dump(a[name])==dump(b[name]),name
    assert set(b)-set(a)=={'_setup_directory_plan','verify_directory_policy_sources'}


@pytest.mark.parametrize('fault',['existing','public-parent','symlink-parent'])
def test_source_preparation_cannot_reset_or_escape_a_previous_handoff(tmp_path,fault):
    parent=tmp_path/'parent';parent.mkdir(mode=0o700);folder=parent/'candidate'
    if fault=='existing':folder.mkdir(mode=0o700);write(folder,'PRESERVED.txt',b'OLD')
    elif fault=='public-parent':parent.chmod(0o755)
    else:
        link=tmp_path/'LINK';link.symlink_to(parent,target_is_directory=True);folder=link/'candidate'
    with pytest.raises(ValueError):o.prepare_package(ROOT,folder)
    if fault=='existing':assert (folder/'PRESERVED.txt').read_bytes()==b'OLD'
