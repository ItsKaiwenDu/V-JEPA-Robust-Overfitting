"""CPU resource/provenance/failure guards. No fixture qualifies a CUDA case."""
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('contended_recovery_tests',ROOT/'scripts/recover_contended_panel_case.py')
new=importlib.util.module_from_spec(s);s.loader.exec_module(new)
OLD=Path(os.environ.get('CONTENDED_TEST_ORIGINAL',str(ROOT)))
s=importlib.util.spec_from_file_location('original_recovery_guards',OLD/'models/two_host_panel.py')
ops=importlib.util.module_from_spec(s);s.loader.exec_module(ops)
FIX=json.loads((ROOT/'tests/fixtures/contended_panel_metadata.json').read_text())


def spec():
    return {'kind':new.KIND,'case_id':new.CASE,'host':new.HOST,'attempt':new.ATTEMPT,
        'reference_revision_sha256':new.RP,'reference_implementation_sha256':new.MP,'reference_operation_sha256':new.OP,
        'failed_STOP_sha256':new.STOP,'extra_case_seconds':4800,'case_seconds':new.CASE_LIMIT,'kill_grace_seconds':15,
        'parent_seconds':4620,'matrix_seconds':3600,'replay_seconds':900,'additional_attempts':1,
        'reuse_old_tensors':False,'old_remainder_added':False,'automatic_GPU_retries':0,'automatic_queue':False,
        'strict_GPU_guard_unchanged':True,'preserve_global_STOP':True,'one_case_only':True,
        'paid_compute':False,'research_admitted':False,'numeric_settings_changed':False,
        'sources_sha256':{n:'1'*64 for n in new.SOURCES},'peer_review_file_sha256':'2'*64,
        'old_budget':new.budget_assessment(2065.4081135171,2086.5223807520233,2113),
        'preserved_coordinator_files_sha256':{n:v['observed_sha256'] for n,v in FIX['startup_manifest']['coordinator_file_hashes'].items()},
        'failed_attempt_files_sha256':{},'original_proposal_sha256':new.ORIGINAL_PROPOSAL,
        'original_authorization_file_sha256':new.ORIGINAL_AUTH,'original_grant_id':new.ORIGINAL_GRANT,
        'prior_startup_transport_sha256':new.STARTUP_TRANSPORT,'prior_startup_manifest_sha256':new.STARTUP_MANIFEST,
        'prior_startup_charge_seconds':new.STARTUP_CHARGE,
        'prior_startup_files_sha256':{n:v['sha256'] for n,v in FIX['startup_manifest']['files'].items()},
        'preserved_failed_startup_files_sha256':{
            (n.removeprefix('packet/') if n.startswith('packet/') else n):v['sha256']
            for n,v in FIX['startup_manifest']['files'].items() if n.startswith(('packet/','launcher/'))}}


def authorization(q):
    return json.loads(FIX['original_authorization_text'])


def fixture(tmp_path):
    canonical=tmp_path/'shared-home';canonical.mkdir();home=tmp_path/'home-alias';home.symlink_to(canonical,target_is_directory=True)
    root=home/new.BASE/'two-host-panel-20261026';old=root/'cases'/new.CASE/'attempt-01';old.mkdir(parents=True)
    (root/'STOP.json').write_text(FIX['failed_STOP_text']);(old/'session.json').write_text(FIX['failed_session_text'])
    (old/'matrix').mkdir();(old/'matrix/worker.json').write_text('historical incomplete numerical metadata')
    q=spec();q['failed_attempt_files_sha256']={str(p.relative_to(old)):new.digest(p) for p in old.rglob('*') if p.is_file()}
    q['preserved_coordinator_files_sha256']={'STOP.json':new.STOP}
    q['preserved_failed_startup_files_sha256']={} # Generated CPU-only transaction fixture has no deployed startup.
    return root.resolve(),q


def test_genuine_failure_and_separate_new_allowance():
    q=spec();new.validate_spec(q,new.canonical(q));a=authorization(q);new.validate_authorization(q,a)
    assert new.canonical(json.loads(FIX['failed_STOP_text'])['payload'])==FIX['audit']['STOP_payload_sha256']
    assert FIX['audit']['case_accepted'] is False and FIX['audit']['qualified_CUDA_worker'] is False
    assert q['old_budget']=={'old_charge_seconds':2246,'old_remaining_seconds':2554,'old_remaining_parent_seconds':2374}
    assert q['case_seconds']==4764 and q['prior_startup_charge_seconds']==36 and not q['old_remainder_added']
    assert q['case_seconds']+q['prior_startup_charge_seconds']==4800 and a['grant_id']==new.ORIGINAL_GRANT


@pytest.mark.parametrize('key,value',[
    ('case_id','clip-05-bank-1'),('host','linux10609'),('attempt','attempt-03'),
    ('reference_revision_sha256','0'*64),('reference_implementation_sha256','0'*64),
    ('failed_STOP_sha256','0'*64),('extra_case_seconds',4801),('case_seconds',7354),('case_seconds',4800),
    ('kill_grace_seconds',16),('parent_seconds',4621),('matrix_seconds',3601),('replay_seconds',901),
    ('additional_attempts',2),('reuse_old_tensors',True),('old_remainder_added',True),
    ('automatic_GPU_retries',1),('automatic_queue',True),('strict_GPU_guard_unchanged',False),
    ('preserve_global_STOP',False),('one_case_only',False),('paid_compute',True),('research_admitted',True),
    ('numeric_settings_changed',True),('original_grant_id','1'*32),('prior_startup_charge_seconds',0),
    ('prior_startup_manifest_sha256','0'*64),('prior_startup_transport_sha256','0'*64)])
def test_scope_or_resource_changes_rejected(key,value):
    q=spec();q[key]=value
    with pytest.raises(ValueError):new.validate_spec(q,new.canonical(q))


def test_external_pin_source_inventory_and_old_time_cannot_drift():
    q=spec();pin=new.canonical(q);q['note']='changed'
    with pytest.raises(ValueError):new.validate_spec(q,pin)
    q=spec();q['sources_sha256'].pop(new.SOURCES[-1])
    with pytest.raises(ValueError):new.validate_spec(q,new.canonical(q))
    q=spec();q['old_budget']['old_charge_seconds']=0
    with pytest.raises(ValueError):new.validate_spec(q,new.canonical(q))


@pytest.mark.parametrize('key,value',[
    ('approved',False),('proposal_sha256','0'*64),('host','linux10609'),('attempt','attempt-03'),
    ('extra_case_seconds',7354),('additional_attempts',2),('fresh_matrix_and_replay',False),
    ('paid_compute',True),('research_admitted',True),('grant_id','a60abb0ffd3b46079ea055cc24bd57e7'),('grant_id','bad')])
def test_mismatched_or_reused_authorization_rejected(key,value):
    q=spec();a=authorization(q);a[key]=value
    with pytest.raises(ValueError):new.validate_authorization(q,a)


@pytest.mark.parametrize('values',[(float('nan'),0,0),(1,-1,1),(True,1,1),(float('inf'),0,0)])
def test_unknown_or_invalid_time_is_not_treated_as_zero(values):
    with pytest.raises(ValueError):new.budget_assessment(*values)


def test_failed_files_and_new_stop_preserved_and_symlink_alias_allowed(tmp_path):
    root,q=fixture(tmp_path);before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    new.check_preserved(root,q,ops)
    assert {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}==before


@pytest.mark.parametrize('change',['stop','session','extra-file','fabricated-observation','replay-directory','file-symlink','directory-symlink'])
def test_partial_mutation_or_new_observation_is_not_repaired(tmp_path,change):
    root,q=fixture(tmp_path);old=root/'cases'/new.CASE/'attempt-01'
    if change=='stop':(root/'STOP.json').write_text('other failure')
    elif change=='session':(old/'session.json').write_text('changed runtime')
    elif change=='extra-file':(old/'extra.txt').write_text('unreviewed')
    elif change=='fabricated-observation':(old/'matrix/observation.json').write_text('{}')
    elif change=='replay-directory':(old/'replay').mkdir()
    elif change=='file-symlink':
        (old/'session.json').unlink();(old/'session.json').symlink_to(root/'STOP.json')
    else:
        (old/'matrix/worker.json').unlink();(old/'matrix').rmdir();(old/'matrix').symlink_to(root,target_is_directory=True)
    with pytest.raises(ValueError):new.check_preserved(root,q,ops)


def test_grant_is_single_use_and_case_namespace_prevents_second_grant(tmp_path):
    root,q=fixture(tmp_path);a=authorization(q);stop=(root/'STOP.json').read_bytes()
    directory=new.consume(root,q,a,ops);assert ops.read_record(directory/'claim.json')['grant_id']==a['grant_id']
    with pytest.raises(FileExistsError):new.consume(root,q,a,ops)
    a['grant_id']='6'*32
    with pytest.raises(FileExistsError):new.consume(root,q,a,ops)
    assert (root/'STOP.json').read_bytes()==stop and not (root/'cases'/new.CASE/new.ATTEMPT).exists()


def test_worker_requires_live_role_and_cannot_reuse_old_artifacts(tmp_path,monkeypatch):
    root,q=fixture(tmp_path);a=authorization(q);d=new.consume(root,q,a,ops)
    case=root/'cases'/new.CASE/new.ATTEMPT;case.mkdir()
    control={'proposal_sha256':new.canonical(q),'complete':False,'active_role':'matrix'}
    ops.write_record(case/'case-envelope.json',control)
    context=(q,a,a['grant_id'],ops,None,None,None,None,None,None,root)
    calls=[];monkeypatch.setattr(new.platform,'node',lambda:new.HOST)
    monkeypatch.setattr(new,'strict_gate',lambda *a,**k:calls.append((a,k)))
    assert new.live_worker_gate(context,'matrix')==case and len(calls)==1
    with pytest.raises(FileExistsError):new.live_worker_gate(context,'matrix')
    with pytest.raises(ValueError):new.live_worker_gate(context,'replay')
    assert not (case/'matrix').exists() and not (d/'workers/replay').exists()


def test_strict_original_PID_guard_allows_only_this_worker():
    ops.validate_hardware(new.HOST,ops.HOSTS[new.HOST],'123\n',allowed_pid=123)
    for active,allowed in [('1162024\n',None),('123\n1162024\n',123)]:
        with pytest.raises(ValueError,match='Another CUDA compute process'):ops.validate_hardware(new.HOST,ops.HOSTS[new.HOST],active,allowed_pid=allowed)


def test_rejection_is_retained_and_never_fabricates_observation(tmp_path,monkeypatch):
    def reject(*a,**k):raise ValueError('Another CUDA compute process is active on this host.')
    delegate=SimpleNamespace(hardware_gate=reject,write_record=ops.write_record)
    monkeypatch.setattr(new.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='1162024\n',stderr=''))
    with pytest.raises(ValueError,match='Another CUDA'):new.strict_gate(delegate,tmp_path,'matrix-after',allow_self=True)
    event=ops.read_record(tmp_path/'hardware-matrix-after.json')
    assert event['passed'] is False and event['compute_pids']['stdout']=='1162024\n'
    assert not (tmp_path/'observation.json').exists()


def test_recovery_identity_is_distinct_and_binds_grant_without_relabelling_reference():
    q=spec();a=authorization(q);reference={'domain':'university-cuda','implementation_sha256':new.MP,'source_sha256':{'scientific':'2'*64}}
    delegate=SimpleNamespace(execution_identity=lambda *a:copy.deepcopy(reference))
    context=(q,a,a['grant_id'],delegate,{},{});result=new.execution_identity(context)
    assert new.canonical(result)!=new.canonical(reference) and result['reference_execution_sha256']==new.canonical(reference)
    assert result['source_sha256']==reference['source_sha256'] and result['implementation_sha256']==new.MP
    assert result['recovery_sources_sha256']==q['sources_sha256'] and result['recovery_grant_id']==a['grant_id']


def test_serial_commands_have_fresh_roles_and_declared_time_caps(tmp_path):
    commands=[new.child_command(tmp_path,'1'*64,'2'*64,role) for role in ('matrix','replay')]
    assert commands[0][-1]=='matrix' and commands[1][-1]=='replay'
    assert all('--proposal-sha256' in c and '--authorization-sha256' in c and 'attempt-01' not in ' '.join(c) for c in commands)
    assert new.worker_limit('matrix',4620,0)==3600
    assert new.worker_limit('replay',4620,3500)==900
    assert new.worker_limit('replay',2374,2100)==244
    with pytest.raises(TimeoutError):new.worker_limit('replay',4620,4590)
    with pytest.raises(ValueError):new.worker_limit('matrix',4801,0)


@pytest.mark.parametrize('name',['../escape','/absolute','matrix/../../escape','matrix\\escape'])
def test_packet_paths_cannot_escape(tmp_path,name):
    with pytest.raises(ValueError):new.regular_bound(tmp_path,name)


@pytest.mark.parametrize('ending',['success','worker-failure','deadline'])
def test_one_case_parent_retains_failures_and_stop_and_never_accepts_or_queues(tmp_path,monkeypatch,ending):
    root,q=fixture(tmp_path);a=authorization(q);before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    delegate=SimpleNamespace(file_lock=ops.file_lock,hardware_gate=lambda *a,**k:None,bound=ops.bound,
        write_record=ops.write_record,read_record=ops.read_record)
    context=(q,a,a['grant_id'],delegate,None,None,None,None,None,None,root)
    status={'CPU_fixture_only':True}
    monkeypatch.setattr(new,'preflight',lambda *a,**k:(SimpleNamespace(status=lambda:status),status))
    monkeypatch.setitem(sys.modules,'models.matched_panel_runner',SimpleNamespace(case_deadline=lambda *a,**k:nullcontext()))
    invoked=[]
    def numerical(*args):
        directory,budget=args[-2:];assert 0<budget<=4620;invoked.append(directory)
        ops.write_record(directory/'session.json',{'CPU_fixture_only':True,'research_admitted':False})
        if ending=='worker-failure':raise RuntimeError('CPU fixture rejected')
        if ending=='deadline':raise TimeoutError('CPU fixture deadline')
        return {'execution_sha256':'1'*64},{'sha256':'2'*64}
    monkeypatch.setattr(new,'run_session',numerical)
    exported=[];monkeypatch.setattr(new,'export_recovery',lambda *args:exported.append(args))
    args=(context,tmp_path,new.canonical(q),'3'*64,new.time.monotonic())
    if ending=='success':new.run(*args)
    else:
        with pytest.raises((RuntimeError,TimeoutError)):new.run(*args)
    assert len(invoked)==len(exported)==1 and all(Path(n).read_bytes()==b for n,b in before.items())
    outcome=ops.read_record(root/'recovery-grants'/new.CASE/'outcome.json')
    assert outcome['complete']==(ending=='success') and not outcome['independently_reviewed']
    assert ops.read_record(root/'cases'/new.CASE/new.ATTEMPT/'case-envelope.json')['active_role'] is None
    assert not (root/'reviews').exists() and not (root/'exports').exists()
    with pytest.raises(FileExistsError):new.run(*args)
    assert len(invoked)==1


def test_preflight_refuses_a_consumed_grant_before_any_model_or_verifier(tmp_path):
    root,q=fixture(tmp_path);a=authorization(q);new.consume(root,q,a,ops)
    context=(q,a,a['grant_id'],ops,None,None,None,None,None,None,root)
    with pytest.raises(ValueError,match='already claimed'):new.preflight(context,new.HOST)
    with pytest.raises(ValueError,match='assigned GPU host'):new.preflight(context,'linux10609')


def test_copied_recovery_root_cannot_reset_a_grant(tmp_path):
    with pytest.raises(ValueError,match='fixed recovery packet'):new.load_context(tmp_path/'copied','1'*64,'2'*64,tmp_path)


def test_run_can_construct_real_registry_without_reentering_its_lock(tmp_path,monkeypatch):
    """Real flock + real Registry constructor; replace only numerical work."""
    root,q=fixture(tmp_path);a=authorization(q)
    revision=json.loads((OLD/'configs/matched_panel_two_host_v1.json').read_text())
    imports={cid:{'CPU_fixture_only':True} for cid in ops.CASE_ORDER[:7]}
    ops.Registry(root,revision,new.MP,imports)
    real_lock=ops.file_lock
    # Fail immediately on a real conflicting descriptor instead of waiting 10s.
    def immediate(path,*,wait_seconds=10):return real_lock(path,wait_seconds=0)
    monkeypatch.setattr(ops,'file_lock',immediate)
    delegate=SimpleNamespace(file_lock=immediate,hardware_gate=lambda *a,**k:None,bound=ops.bound,
        write_record=ops.write_record,read_record=ops.read_record)
    context=(q,a,a['grant_id'],delegate,None,None,None,None,None,None,root)
    observed=[]
    def real_registry_preflight(context,host,*,return_registry=False):
        with pytest.raises(BlockingIOError):
            with immediate(root/'leases'/new.HOST):pass
        registry=ops.Registry(root,revision,new.MP,imports)
        status=registry.status();observed.append(status)
        return (registry,status) if return_registry else status
    monkeypatch.setattr(new,'preflight',real_registry_preflight)
    monkeypatch.setitem(sys.modules,'models.matched_panel_runner',SimpleNamespace(case_deadline=lambda *a,**k:nullcontext()))
    monkeypatch.setattr(new,'run_session',lambda *a:({'execution_sha256':'1'*64},{'sha256':'2'*64}))
    monkeypatch.setattr(new,'export_recovery',lambda *a:None)
    new.run(context,tmp_path,new.canonical(q),'3'*64,new.time.monotonic())
    assert len(observed)==1 and ops.read_record(root/'recovery-grants'/new.CASE/'outcome.json')['complete']


@pytest.mark.parametrize('change',['registry-status','preserved-STOP'])
def test_change_between_preflight_and_locked_claim_stops_before_grant_or_GPU(tmp_path,monkeypatch,change):
    root,q=fixture(tmp_path);a=authorization(q);calls=[]
    delegate=SimpleNamespace(file_lock=ops.file_lock,hardware_gate=lambda *a,**k:calls.append('GPU'),
        bound=ops.bound,write_record=ops.write_record,read_record=ops.read_record)
    context=(q,a,a['grant_id'],delegate,None,None,None,None,None,None,root)
    def preflight(*a,**k):
        if change=='preserved-STOP':(root/'STOP.json').write_text('New CPU-fixture failure; never overwrite it.\n')
        return SimpleNamespace(status=lambda:{'generation':2}),{'generation':1}
    monkeypatch.setattr(new,'preflight',preflight)
    monkeypatch.setitem(sys.modules,'models.matched_panel_runner',SimpleNamespace(case_deadline=lambda *a,**k:nullcontext()))
    with pytest.raises(ValueError):new.run(context,tmp_path,new.canonical(q),'3'*64,new.time.monotonic())
    assert not calls and not (root/'recovery-grants'/new.CASE).exists() and not (root/'cases'/new.CASE/new.ATTEMPT).exists()
    with ops.file_lock(root/'.registry.lock',wait_seconds=0):pass
    with ops.file_lock(root/'leases'/new.HOST,wait_seconds=0):pass


def test_unrelated_real_registry_lock_is_still_respected(tmp_path,monkeypatch):
    root,q=fixture(tmp_path);a=authorization(q);revision=json.loads((OLD/'configs/matched_panel_two_host_v1.json').read_text())
    imports={cid:{'CPU_fixture_only':True} for cid in ops.CASE_ORDER[:7]};ops.Registry(root,revision,new.MP,imports)
    real_lock=ops.file_lock
    def immediate(path,*,wait_seconds=10):return real_lock(path,wait_seconds=0)
    monkeypatch.setattr(ops,'file_lock',immediate)
    delegate=SimpleNamespace(file_lock=immediate,bound=ops.bound)
    context=(q,a,a['grant_id'],delegate,None,None,None,None,None,None,root)
    def preflight(*a,**k):return ops.Registry(root,revision,new.MP,imports),{}
    monkeypatch.setattr(new,'preflight',preflight)
    monkeypatch.setitem(sys.modules,'models.matched_panel_runner',SimpleNamespace(case_deadline=lambda *a,**k:nullcontext()))
    with real_lock(root/'.registry.lock',wait_seconds=0):
        with pytest.raises(BlockingIOError):new.run(context,tmp_path,new.canonical(q),'3'*64,new.time.monotonic())
    assert not (root/'recovery-grants'/new.CASE).exists()


def test_original_authorization_is_unchanged_and_only_its_unused_time_is_carried():
    q=spec();a=authorization(q);new.validate_authorization(q,a)
    assert a['proposal_sha256']==new.ORIGINAL_PROPOSAL and a['proposal_sha256']!=new.canonical(q)
    assert new.canonical(a)==new.canonical(json.loads(FIX['original_authorization_text']))
    assert new.STARTUP_CHARGE+new.CASE_LIMIT==a['extra_case_seconds']==4800
    assert new.validate_authorization(q,a)==FIX['startup_readback']['original_grant_id']


def test_authentic_startup_manifest_preserves_unused_boundary_and_all_file_pins():
    q=spec();new.validate_startup_manifest(FIX['startup_manifest'],q)
    assert len(q['prior_startup_files_sha256'])==31 and len(q['preserved_failed_startup_files_sha256'])==22
    assert FIX['startup_readback']['whole_startup_elapsed_seconds']+1==new.STARTUP_CHARGE


@pytest.mark.parametrize('change',['grant-exists','attempt-symlink','wrong-host','wrong-case','changed-preserved-file',
    'trimmed-sources','relabelled-proposal','remote-writes','started-model','source-hash'])
def test_failed_startup_must_not_hide_a_grant_model_launch_or_evidence_change(change):
    q=spec();m=copy.deepcopy(FIX['startup_manifest'])
    if change=='grant-exists':m['recovery_locations']['recovery_grant']['exists']=True
    elif change=='attempt-symlink':m['recovery_locations']['attempt_02']['is_symlink']=True
    elif change=='wrong-host':m['host']='linux10609'
    elif change=='wrong-case':m['case_id']='clip-05-bank-1'
    elif change=='changed-preserved-file':m['coordinator_file_hashes']['STOP.json']['observed_sha256']='0'*64
    elif change=='trimmed-sources':m['files'].pop(next(iter(m['files'])))
    elif change=='relabelled-proposal':m['proposal_sha256']=new.canonical(q)
    elif change=='remote-writes':m['university_files_changed']=True
    elif change=='started-model':m['model_or_test_job_started']=True
    else:m['files']['packet/scripts/recover_contended_panel_case.py']['sha256']='0'*64
    with pytest.raises(ValueError):new.validate_startup_manifest(m,q)


def test_failed_startup_preservation_cannot_be_dropped():
    q=spec();q['preserved_failed_startup_files_sha256']={}
    with pytest.raises(ValueError):new.validate_spec(q,new.canonical(q))
