"""CPU metadata transaction guards. These fixtures qualify no CUDA case."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('reviewed_metadata_tests',ROOT/'models/reviewed_panel.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
DATA=json.loads((ROOT/'tests/fixtures/reviewed_panel_metadata.json').read_text())


def operation(data=None):
    data=DATA if data is None else data
    return {'kind':m.KIND,'host':'linux10609','target_namespace':m.TARGET,'legacy_namespace':m.LEGACY,
        'legacy_revision_sha256':m.RP,'legacy_implementation_sha256':m.MP,'recovery_proposal_sha256':m.RECOVERY_PROPOSAL,
        'original_authorization_file_sha256':m.AUTH,'original_recovery_grant_id':m.GRANT,'legacy_STOP_sha256':m.STOP,
        'accepted_case_ids':m.ACCEPTED,'metadata_only':True,'max_seconds':300,'kill_grace_seconds':15,
        'legacy_records_changed':False,'STOP_to_remain_active':True,'new_GPU_dispatch_authorized':False,
        'models_started':False,'test_suite_started':False,'automatic_queue':False,'research_admitted':False,
        'sources_sha256':{n:'1'*64 for n in m.SOURCES},'inputs_sha256':hashlib.sha256(m.json_bytes(data)).hexdigest()}


def legacy_status():
    return {'accepted_case_ids':m.ORDER[:9],'computed_new_case_ids':list(m.NORMALS),
        'claimed_case_ids':[c for c in m.ORDER if c in set(m.NORMALS)|{m.RECOVERED}],
        'failed_case_ids':[m.RECOVERED],'stop_dispatch':True,'complete':False,'scientific_aggregate':None,'research_admitted':False}


def receipt(op,data):
    targets=m.records(op,data);status=legacy_status()
    return {'kind':m.KIND+'-receipt','operation_sha256':m.canonical(op),'inputs_sha256':op['inputs_sha256'],'host':'linux10609',
        'created':list(targets),'preexisting_identical':[],'records_readback':{n:b.decode() for n,b in targets.items()},
        'records_file_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in targets.items()},
        'fresh_preserved_files_sha256':copy.deepcopy(data['preserved_files_sha256']),
        'legacy_status_before':copy.deepcopy(status),'legacy_status_after':copy.deepcopy(status),
        'original_imports_reverified':7,'newer_saved_cases_reverified':4,
        'effective_progress':m.record_text(targets['progress.json'].decode()),'legacy_STOP_readback':data['legacy_STOP_text'],
        'legacy_records_changed':False,'models_started':False,'test_suite_started':False,'new_GPU_dispatch_authorized':False,'research_admitted':False}


def test_genuine_eleven_origins_keep_original_sources_and_failure_history():
    op=operation();m.validate_operation(op,m.canonical(op));m.validate_input(DATA)
    targets=m.records(op,DATA);assert set(targets)==m.TARGET_FILES
    progress=m.record_text(targets['progress.json'].decode());imports=m.record_text(targets['imports.json'].decode())
    assert progress['accepted_case_ids']==m.ACCEPTED and len(progress['pending_case_ids'])==9 and progress['complete_clip_pairs']==5
    assert progress['stop_dispatch'] and not progress['new_GPU_dispatch_authorized'] and progress['scientific_aggregate'] is None
    assert imports['case_origins']==DATA['case_index']
    assert len([r for r in imports['case_origins'].values() if r['origin']=='frozen-v1'])==7
    assert len([r for r in imports['case_origins'].values() if r['origin']=='two-host-revision'])==3
    assert imports['case_origins'][m.RECOVERED]['origin']=='explicit-attended-recovery'
    assert targets['STOP.json']==DATA['legacy_STOP_text'].encode()
    assert targets['reviews/'+m.RECOVERED+'.json']==DATA['recovery_review']['text'].encode()


@pytest.mark.parametrize('key,value',[
    ('host','linux10603'),('target_namespace',m.LEGACY),('legacy_namespace','other'),
    ('metadata_only',False),('legacy_records_changed',True),('STOP_to_remain_active',False),
    ('new_GPU_dispatch_authorized',True),('models_started',True),('test_suite_started',True),
    ('automatic_queue',True),('research_admitted',True),('max_seconds',4800),('kill_grace_seconds',60),
    ('legacy_implementation_sha256','0'*64),('recovery_proposal_sha256','0'*64),('original_recovery_grant_id','0'*32),
    ('accepted_case_ids',m.ACCEPTED[:-1])])
def test_operation_cannot_switch_source_scope_or_allow_a_model(key,value):
    op=operation();op[key]=value
    with pytest.raises(ValueError):m.validate_operation(op,m.canonical(op))


def test_external_operation_pin_and_full_source_inventory_required():
    op=operation();pin=m.canonical(op);op['note']='drift'
    with pytest.raises(ValueError):m.validate_operation(op,pin)
    op=operation();op['sources_sha256'].pop(m.SOURCES[0])
    with pytest.raises(ValueError):m.validate_operation(op,m.canonical(op))


@pytest.mark.parametrize('change',['drop-case','extra-case','alias','relabel-recovery','wrong-attempt','wrong-host',
    'rewrite-execution','archive-escape','missing-archive-pin','missing-receipt-pin','new-grant','edited-marker',
    'changed-journal','changed-STOP','permit-model','precision-change'])
def test_case_origins_and_independent_markers_cannot_be_relabelled(change):
    data=copy.deepcopy(DATA);row=data['case_index'][m.RECOVERED]
    if change=='drop-case':data['case_index'].pop(m.ACCEPTED[0])
    elif change=='extra-case':data['case_index']['clip-05-bank-1']=copy.deepcopy(row)
    elif change=='alias':row['case_id']='clip-05-bank-0'
    elif change=='relabel-recovery':row['origin']='two-host-revision'
    elif change=='wrong-attempt':row['directory']=row['directory'].replace('attempt-02','attempt-01')
    elif change=='wrong-host':row['host']='linux10609'
    elif change=='rewrite-execution':row['execution_sha256']=m.V1_EXEC
    elif change=='archive-escape':row['archive_path']='../outside'
    elif change=='missing-archive-pin':data['preserved_files_sha256'].pop(row['archive_path'])
    elif change=='missing-receipt-pin':data['preserved_files_sha256'].pop(row['directory']+'/receipt.json')
    elif change=='new-grant':row['grant_id']='0'*32
    elif change=='edited-marker':data['recovery_review']['text']+=' '
    elif change=='changed-journal':data['legacy_journal_text']+=' '
    elif change=='changed-STOP':data['legacy_STOP_text']+=' '
    elif change=='permit-model':data['models_started']=True
    else:data['numerical_settings_changed']=True
    with pytest.raises(ValueError):m.validate_input(data)


def test_first_publish_and_idempotent_repeat_keep_old_files_and_STOP(tmp_path):
    old=tmp_path/'old';old.mkdir();f=old/'evidence';f.write_bytes(b'unchanged original evidence')
    verify=lambda:m.check_preserved(old,{'evidence':hashlib.sha256(f.read_bytes()).hexdigest()})
    root=tmp_path/'new';targets=m.records(operation(),DATA)
    created,existing=m.publish(root,targets,verify)
    assert len(created)==6 and created[-1]=='genesis.json' and not existing
    assert (root/'STOP.json').read_bytes()==DATA['legacy_STOP_text'].encode()
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    created,existing=m.publish(root,targets,verify)
    assert not created and len(existing)==6 and {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}==before
    assert f.read_bytes()==b'unchanged original evidence'


@pytest.mark.parametrize('cut',range(6))
def test_interrupted_publication_only_fills_identical_missing_metadata(tmp_path,monkeypatch,cut):
    root=tmp_path/'new';targets=m.records(operation(),DATA);original=m.exclusive_bytes;calls=[]
    def interrupted(path,data):
        if len(calls)==cut:raise OSError('CPU publication fixture interrupted')
        calls.append(path);original(path,data)
    monkeypatch.setattr(m,'exclusive_bytes',interrupted)
    with pytest.raises(OSError):m.publish(root,targets,lambda:None)
    assert not (root/'genesis.json').exists()
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(m,'exclusive_bytes',original)
    created,existing=m.publish(root,targets,lambda:None)
    assert len(created)==6-cut and len(existing)==cut and all(Path(p).read_bytes()==b for p,b in before.items())
    assert (root/'genesis.json').is_file()


@pytest.mark.parametrize('which',sorted(m.TARGET_FILES))
def test_conflicting_record_is_never_overwritten_or_partially_published(tmp_path,which):
    root=tmp_path/'new';path=root/which;path.parent.mkdir(parents=True);path.write_bytes(b'conflicting existing record')
    targets=m.records(operation(),DATA)
    with pytest.raises(ValueError):m.publish(root,targets,lambda:None)
    assert path.read_bytes()==b'conflicting existing record' and len([p for p in root.rglob('*') if p.is_file()])==1


@pytest.mark.parametrize('which',['root-symlink','reviews-symlink','extra-file','future-cases'])
def test_unknown_or_redirected_namespace_is_not_reinitialized(tmp_path,which):
    root=tmp_path/'new';outside=tmp_path/'outside';outside.mkdir()
    if which=='root-symlink':root.symlink_to(outside,target_is_directory=True)
    else:
        root.mkdir()
        if which=='reviews-symlink':(root/'reviews').symlink_to(outside,target_is_directory=True)
        elif which=='extra-file':(root/'unexpected').write_text('unreviewed')
        else:(root/'cases').mkdir()
    with pytest.raises(ValueError):m.publish(root,m.records(operation(),DATA),lambda:None)
    assert not list(outside.iterdir())


def test_old_pin_drift_stops_before_any_metadata_write(tmp_path):
    root=tmp_path/'new';old=tmp_path/'old';old.mkdir();(old/'case').write_text('changed')
    with pytest.raises(ValueError):m.publish(root,m.records(operation(),DATA),lambda:m.check_preserved(old,{'case':'0'*64}))
    assert not root.exists()


def test_receipt_distinguishes_effective_eleven_from_immutable_legacy_nine():
    op=operation();r=receipt(op,DATA);result=m.verify_receipt(r,op,DATA)
    assert result['verified'] and result['effective_accepted_cases']==11 and result['legacy_accepted_cases']==9 and result['STOP_still_active']
    r['preexisting_identical']=r['created'];r['created']=[];assert m.verify_receipt(r,op,DATA)['verified']


@pytest.mark.parametrize('change',['drop-readback','overlap-created','fresh-pin','legacy-progress','effective-progress','STOP',
    'source-operation','model','GPU-authority','case-recheck-count'])
def test_receipt_requires_exact_readback_and_fresh_preserved_evidence(change):
    op=operation();r=receipt(op,DATA)
    if change=='drop-readback':r['records_readback'].pop('imports.json')
    elif change=='overlap-created':r['preexisting_identical']=[r['created'][0]]
    elif change=='fresh-pin':r['fresh_preserved_files_sha256'].pop(next(iter(r['fresh_preserved_files_sha256'])))
    elif change=='legacy-progress':r['legacy_status_after']['accepted_case_ids']=m.ACCEPTED
    elif change=='effective-progress':r['effective_progress']['stop_dispatch']=False
    elif change=='STOP':r['legacy_STOP_readback']='replacement stop'
    elif change=='source-operation':r['operation_sha256']='0'*64
    elif change=='model':r['models_started']=True
    elif change=='GPU-authority':r['new_GPU_dispatch_authorized']=True
    else:r['newer_saved_cases_reverified']=3
    with pytest.raises(ValueError):m.verify_receipt(r,op,DATA)


@pytest.mark.parametrize('name',['../escape','/absolute','reviews/../../escape','reviews\\escape'])
def test_receipt_paths_cannot_escape_the_declared_namespace(tmp_path,name):
    with pytest.raises(ValueError):m.bound(tmp_path,name)
