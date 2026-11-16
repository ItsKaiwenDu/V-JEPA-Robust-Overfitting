"""New v3 whole-source/setup checks; runtime/host/proof operations are substituted."""
import ast
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_split_outer_v3 as outer
from models import predictor_pgd_split_bootstrap_v1 as boot
from models import predictor_pgd_split_scratch_v1 as scratch
from models import predictor_pgd_split_coordinator_v1 as coordinator
from models import predictor_pgd_split_startup_sources_v3 as startup
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'runs/pretrained/diagnostic/predictor-pgd-split-outer-v2-20261116/handoff-parent/linux10603-unarmed'
SAVED=OLD/'attended-session/saved'

@pytest.fixture(scope='module')
def package(tmp_path_factory):
    parent=tmp_path_factory.mktemp('EXPLICIT-UNARMED-V3');parent.chmod(0o700);folder=parent/'handoff'
    result=outer.prepare_package(ROOT,folder);raw=(folder/'CONTROL-CANDIDATE.json').read_bytes()
    c,b,p,sha=outer.validate_control(raw,outer.digest((ROOT/'models/predictor_pgd_split_outer_v3.py').read_bytes()))
    sources=outer.decode_sources(json.loads((folder/'SOURCE-PAYLOAD.json').read_bytes()),c,b)
    auth=outer.encoded(outer.authorization(c,sha,'EXPLICIT INERT LOCAL TEST BASIS ONLY; NOT A LIVE GRANT'))
    return NS(folder=folder,result=result,raw=raw,c=c,b=b,p=p,sha=sha,sources=sources,auth=auth)


def test_v3_source_package_changes_only_two_supervisors_and_preserves_packet_science_and_limits(package):
    x=package
    assert x.c['kind']=='pgd-split-outer-control-v3' and len(x.c['local_source_sha256'])==12
    assert set(p.name for p in x.folder.iterdir())=={'CONTROL-CANDIDATE.json','SOURCE-PAYLOAD.json','PACKAGE.json','known-hosts'}
    changed=[]
    for name,data in x.sources.items():
        if data!=(SAVED/name).read_bytes():changed.append(name)
    assert set(changed)=={'supervisor/bootstrap.py','supervisor/predictor_pgd_split_coordinator_v1.py'}
    assert x.sources['supervisor/bootstrap.py'].startswith((SAVED/'supervisor/bootstrap.py').read_bytes()+b'\n')
    assert x.sources['supervisor/predictor_pgd_split_coordinator_v1.py'].startswith((SAVED/'supervisor/predictor_pgd_split_coordinator_v1.py').read_bytes()+b'\n')
    assert x.p==json.loads((ROOT/'configs/predictor_pgd_split_launch_v1.json').read_bytes())
    assert x.result['parent_error_diagnostic_cap_bytes']==16*1024
    assert outer.digest(x.sources['supervisor/bootstrap.py'])==outer.BOOT_SHA
    assert x.sha!=json.loads((OLD/'ATTEMPT.json').read_bytes())['outer_control_sha256']
    assert not (x.folder/'AUTHORIZATION.json').exists() and not (x.folder/'ATTEMPT.json').exists()


@pytest.mark.parametrize('fault',['old-control-kind','old-authorization','missing-startup-pin'])
def test_consumed_v2_authority_or_missing_v3_sources_cannot_authorize_new_package(package,fault):
    x=package;c=copy.deepcopy(x.c)
    if fault=='old-authorization':
        with pytest.raises(ValueError):outer.validate_authorization((OLD/'AUTHORIZATION.json').read_bytes(),c,x.sha)
    else:
        if fault=='old-control-kind':c['kind']='pgd-split-outer-control-v2'
        else:c['local_source_sha256'].pop('models/predictor_pgd_split_startup_sources_v3.py')
        with pytest.raises(ValueError):outer.validate_control(outer.encoded(c),c['outer_source_sha256'])


def test_actual_v3_setup_caches_checked_supervisors_before_packet_and_leaves_workers_fresh(package,tmp_path,monkeypatch):
    x=package;root=tmp_path/'private';root.mkdir(mode=0o700);cache=[];records=[]
    ns={'__name__':'STDLIB_BOOTSTRAP_DEFINITIONS_ONLY','__file__':'<V3-BOOTSTRAP-DEFINITIONS>'}
    exec(compile(x.sources['supervisor/bootstrap.py'],ns['__file__'],'exec'),ns)
    class CacheOnly:
        def __init__(self,*args):
            assert not (root/'code').exists()
            finder=ns['PinnedImporter'](*args);assert len(finder.files)==69 and len(finder.packet_imports)==2
            cache.append(finder)
        def install(self):return cache[-1] # No scientific imports or meta_path mutation.
    facade=NS(control=ns['control'],authorization=ns['authorization'],installation=ns['installation'],encoded=ns['encoded'],PinnedImporter=CacheOnly)
    real_scan=scratch.Guard.scan
    def verify(self,**kw):
        real_scan(self,phase='setup',settled=True);records.append(self)
        assert len(self.layout.assets)==172 and len(self.layout.result)==419 and len(self.layout.directories)==39
        plan=outer._setup_directory_plan(self.layout)
        assert len(plan['setup_directories'])==26
        assert all(not (root/p).exists() for p in plan['worker_claim_directories'].values())
        return {'scientific_qualified':False}
    monkeypatch.setattr(scratch.Guard,'verify_assets',verify)
    monkeypatch.setattr(scratch.Guard,'proof_readiness',lambda *a:{'proof_launch_authorized':False})
    monkeypatch.setattr(coordinator,'target_gate',lambda *a:None)
    monkeypatch.setattr(outer,'receive_packet',lambda *a:None)
    monkeypatch.setattr(outer,'extract_packet',lambda *a:None)
    fakeos=NS(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')});allocations=[]
    fakeos.posix_fallocate=lambda *a:allocations.append('EXPLICIT FAKE ZERO BYTE RESERVE')
    def fstat(fd):
        st=os.fstat(fd);p=root/'emergency-space.bin'
        if p.exists() and st.st_ino==p.stat().st_ino:return NS(st_size=16*outer.MiB,st_blocks=16*outer.MiB//512)
        return st
    fakeos.fstat=fstat;monkeypatch.setattr(outer,'os',fakeos)
    result=outer.prepare_tree(root,x.sources,x.c,x.b,x.p,json.loads(x.auth),x.raw,x.auth,facade,io.BytesIO(),lambda:None)
    assert len(records)==1 and len(cache)==1 and not cache[0].loaded and not cache[0].installed
    assert allocations and (root/'emergency-space.bin').stat().st_size==0
    manifest=json.loads((root/'runtime-source-manifest.json').read_bytes())
    assert len(manifest['setup_directory_plan']['setup_directories'])==26
    assert all(not (root/path).exists() for path,d,pkg in startup.PACKET_IMPORTS.values())
    assert len(records[0].scan(phase='setup')[2])==26
    (tmp_path/'ACTUAL-V3-SETUP.json').write_bytes(outer.encoded({'actual_outer_setup_and_catalogue':True,'assets':172,'result_files':419,
        'setup_directories':26,'cached_sources':69,'lazy_packet_dependencies':2,'physical_reserve_bytes':0,'host_proof_packet_contents_substituted':True,
        'scientific_source_executed':False,'authorized':False,'training_admitted':False}))


def test_v3_transport_timers_claims_and_setup_remain_exact_v2_AST():
    old=ast.parse((ROOT/'models/predictor_pgd_split_outer_v2.py').read_bytes());new=ast.parse((ROOT/'models/predictor_pgd_split_outer_v3.py').read_bytes())
    def functions(t):return {q.name:q for q in t.body if isinstance(q,(ast.FunctionDef,ast.ClassDef))}
    a,b=functions(old),functions(new)
    assert set(a)==set(b)
    for name in set(a)-{'validate_control','authorization','verify_directory_policy_sources','prepare_package'}:
        assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False),name
    oldclient=(ROOT/'models/predictor_pgd_split_outer_client_v2.py').read_bytes();newclient=(ROOT/'models/predictor_pgd_split_outer_client_v3.py').read_bytes()
    assert newclient.replace(b'predictor_pgd_split_outer_v3 as outer',b'predictor_pgd_split_outer_v2 as outer')==oldclient
    oldcli=(ROOT/'scripts/run_predictor_pgd_split_outer_v2.py').read_bytes();newcli=(ROOT/'scripts/run_predictor_pgd_split_outer_v3.py').read_bytes()
    assert newcli.replace(b'predictor_pgd_split_outer_client_v3 import run',b'predictor_pgd_split_outer_client_v2 import run')==oldcli


def test_unarmed_v3_cli_refuses_before_any_timer_claim_or_SSH(package):
    spec=importlib.util.spec_from_file_location('EXPLICIT_UNARMED_V3_ENTRY',ROOT/'scripts/run_predictor_pgd_split_outer_v3.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(SystemExit,match='No timer, claim, SSH or model work'):module.main([str(package.folder)])
    assert not (package.folder/'ATTEMPT.json').exists()
