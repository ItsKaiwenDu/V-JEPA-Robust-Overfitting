"""Strict refusal observation and controlled real filesystem writer interleaving."""
import ast
import hashlib
import json
import os
from pathlib import Path
import socket
import stat

import pytest
from models import scratch_inventory_observer_v1 as observer

ROOT=Path(__file__).resolve().parents[1]


def publish(name, value):
    destination=os.environ.get('VJEPA_INVENTORY_EVIDENCE')
    if destination:
        path=Path(destination)/(name+'.json')
        with path.open('x') as f:
            json.dump(value,f,sort_keys=True,indent=2);f.write('\n')


@pytest.mark.parametrize('hashing',(False,True))
def test_successful_inventory_exactly_matches_frozen_scan(tmp_path,hashing):
    folder=tmp_path/'private';folder.mkdir(mode=0o700)
    (folder/'record.json').write_bytes(b'{"ok":true}\n')
    (folder/'tensor.npy').write_bytes(b'fixture bytes; not a numerical tensor')
    expected=observer.frozen.inventory(folder,hashing=hashing,max_bytes=4096,running=True)
    assert observer.inventory(folder,hashing=hashing,max_bytes=4096,running=True)==expected


def test_valid_legacy_atomic_pair_and_allocation_accounting_unchanged(tmp_path):
    folder=tmp_path/'private';folder.mkdir(mode=0o700)
    partial=folder/'state.json.partial';partial.write_bytes(b'{}\n');os.link(partial,folder/'state.json')
    expected=observer.frozen.inventory(folder,hashing=True,running=True)
    actual=observer.inventory(folder,hashing=True,running=True)
    assert actual==expected and len(actual[0])==2
    assert actual[1]==partial.stat().st_blocks*512


@pytest.mark.parametrize('kind',('external-json-hardlink','external-bin-hardlink','symlink','fifo','socket'))
def test_unsafe_entries_stay_refused_with_bounded_path_stat_metadata(tmp_path,monkeypatch,kind):
    folder=tmp_path/'private';folder.mkdir(mode=0o700);outside=tmp_path/'outside';outside.write_bytes(b'never read')
    channel=None
    if kind.startswith('external'):
        name='bad.json' if 'json' in kind else 'bad.bin';path=folder/name;os.link(outside,path)
    elif kind=='symlink':path=folder/'bad.link';path.symlink_to(outside)
    elif kind=='fifo':path=folder/'bad.fifo';os.mkfifo(path,0o600)
    else:
        path=folder/'bad.sock';channel=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        with monkeypatch.context() as patch:
            patch.chdir(folder);channel.bind(path.name)
    try:
        with pytest.raises(ValueError) as before:
            observer.frozen.inventory(folder,running=True)
        with pytest.raises(ValueError) as after:
            observer.inventory(folder,running=True)
        assert str(after.value).startswith(str(before.value)+' | inventory=')
        context=after.value.inventory_observation
        assert context['relative_path']==path.name and context['capture_status']=='captured'
        assert context['observed_stat']['inode']==path.lstat().st_ino
        assert context['observed_stat']['mode']==stat.S_IMODE(path.lstat().st_mode)
        assert context['observed_stat']['links']==path.lstat().st_nlink
        assert context['observed_stat']['uid']==os.getuid()
        assert context['scan_retried'] is False and context['refusal_preserved'] is True
        assert len(observer._json(context).encode('ascii'))<=observer.MAX_CONTEXT_BYTES
        assert len(str(after.value).encode('ascii'))<2048
        assert type(after.value.__cause__) is ValueError
        publish(kind,{'expected_refusal':True,'original_reason':str(before.value),'observed':context,'new_target_cause_established':False})
    finally:
        if channel:channel.close()


@pytest.mark.parametrize('observed',(False,True))
def test_controlled_legitimate_writer_interleaving_can_trigger_frozen_refusal(tmp_path,monkeypatch,observed):
    # Real inodes/link counts; only the scheduling point is deliberately controlled.
    # This proves a local mechanism, not the missing target pathname or XFS cause.
    folder=tmp_path/'private';folder.mkdir(mode=0o700)
    final=folder/'state.json';partial=folder/'state.json.partial'
    final.write_bytes(b'{"generation":0}\n');os.link(final,partial)
    retained_inode=final.open('rb')  # Prevent inode reuse while testing the real interleaving.
    actual_lstat=Path.lstat;changed=[]
    def interleave(path):
        info=actual_lstat(path)
        if not changed and path in (final,partial) and info.st_nlink==2:
            changed.append(True)
            partial.unlink()
            observer.frozen.old.write_json(final,{'generation':1},replace=True)
            partial.write_bytes(b'{"generation":2}\n')
        return info
    with monkeypatch.context() as patch:
        patch.setattr(Path,'lstat',interleave)
        scan=observer.inventory if observed else observer.frozen.inventory
        with pytest.raises(ValueError,match='Unexpected special file or external hardlink') as refused:
            scan(folder,running=True)
    retained_inode.close()
    assert changed==[True]
    assert final.stat().st_nlink==partial.stat().st_nlink==1
    # Separate test-only scan demonstrates the end state is accepted. The observer
    # itself neither retries nor changes the first scan's refusal.
    stable=observer.frozen.inventory(folder,running=True)
    assert set(stable[0])=={'state.json','state.json.partial'}
    if observed:
        c=refused.value.inventory_observation
        assert c['observed_stat']['links']==2 and c['entry_now']['stat']['links']==1
        assert c['observed_stat']['inode']!=c['entry_now']['stat']['inode']
        assert c['partner_now']['stat']['links']==1 and c['guard_paired'] is False
        assert c['scan_retried'] is False
        publish('controlled-writer-interleaving',{'local_controlled_real_filesystem_interleaving':True,
                 'schedule_injected_at_lstat':True,'stat_results_fabricated':False,'final_entries_accepted_by_frozen_scan':True,
                 'observed':c,'target_filename_or_cause_confirmed':False})


def test_observation_rechecks_read_no_file_contents(tmp_path,monkeypatch):
    folder=tmp_path/'private';folder.mkdir(mode=0o700);outside=tmp_path/'outside';outside.write_bytes(b'private fixture')
    os.link(outside,folder/'bad.json')
    actual_open=Path.open
    def no_data_reads(path,*args,**kwargs):
        if path==outside or path.is_relative_to(folder):raise AssertionError('Unexpected data read.')
        return actual_open(path,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'open',no_data_reads)
        with pytest.raises(ValueError) as e:observer.inventory(folder,running=True)
    assert e.value.inventory_observation['capture_status']=='captured'
    assert (folder/'bad.json').read_bytes()==b'private fixture'


def test_metadata_failure_preserves_original_refusal(tmp_path,monkeypatch):
    folder=tmp_path/'private';folder.mkdir(mode=0o700);os.mkfifo(folder/'pipe')
    def fail(*args,**kwargs):raise RuntimeError('Synthetic diagnostic failure.')
    monkeypatch.setattr(observer,'_observation',fail)
    with pytest.raises(ValueError,match='Unexpected special file or external hardlink') as e:
        observer.inventory(folder,running=True)
    assert e.value.inventory_observation['capture_status']=='diagnostic-unavailable'
    assert e.value.inventory_observation['refusal_preserved'] is True
    assert str(e.value.__cause__)=='Unexpected special file or external hardlink.'


def test_disappeared_recheck_does_not_admit_scan(tmp_path,monkeypatch):
    folder=tmp_path/'private';folder.mkdir(mode=0o700);outside=tmp_path/'outside';outside.write_bytes(b'x');os.link(outside,folder/'bad.json')
    def missing(path):raise FileNotFoundError(2,'Synthetic recheck disappearance.')
    monkeypatch.setattr(observer.os,'lstat',missing)
    with pytest.raises(ValueError) as e:observer.inventory(folder,running=True)
    c=e.value.inventory_observation
    assert c['observed_stat']['links']==2 and c['entry_now']=={'error_type':'FileNotFoundError','errno':2}
    assert c['refusal_preserved'] is True


def test_unicode_deep_path_metadata_bounded_and_digested(tmp_path):
    folder=tmp_path/'private';folder.mkdir(mode=0o700);directory=folder
    for i in range(8):directory=directory/(('深'*15)+str(i))
    directory.mkdir(parents=True)
    outside=tmp_path/'outside';outside.write_bytes(b'no content capture');path=directory/'external.json';os.link(outside,path)
    with pytest.raises(ValueError) as e:observer.inventory(folder,running=True)
    c=e.value.inventory_observation;relative=str(path.relative_to(folder))
    assert c['relative_path_truncated'] is True
    assert c['relative_path_sha256']==hashlib.sha256(os.fsencode(relative)).hexdigest()
    assert c['observed_stat']['inode']==path.stat().st_ino
    assert len(observer._json(c).encode('ascii'))<=1536 and len(str(e.value).encode('ascii'))<2048
    publish('bounded-unicode-path',c)


def test_aggregate_limit_failure_still_fails(tmp_path):
    folder=tmp_path/'private';folder.mkdir(mode=0o700);(folder/'a').write_bytes(b'xx')
    with pytest.raises(ValueError,match='Scratch file/byte bound exceeded') as e:
        observer.inventory(folder,max_bytes=1,running=True)
    assert e.value.inventory_observation['observed_stat']['bytes']==2


def test_missing_root_keeps_original_exception_type(tmp_path):
    with pytest.raises(FileNotFoundError):observer.inventory(tmp_path/'absent',running=True)


def test_frozen_source_change_refused_before_scan(tmp_path,monkeypatch):
    monkeypatch.setattr(observer,'SOURCE_SHA','0'*64)
    with pytest.raises(ValueError,match='source/identity changed'):
        observer.inventory(tmp_path/'absent',running=True)


def test_wrapper_delegates_once_and_never_returns_after_refusal():
    assert observer._FROZEN_SCAN is observer.frozen.inventory and observer._FROZEN_SCAN.__code__ is observer._FROZEN_CODE
    module=ast.parse((ROOT/'models/scratch_inventory_observer_v1.py').read_text())
    body=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='inventory')
    assert len([n for n in ast.walk(body) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_FROZEN_SCAN'])==1
    outer=next(n for n in body.body if isinstance(n,ast.Try))
    assert len(outer.handlers)==1 and not any(isinstance(n,ast.Return) for n in ast.walk(outer.handlers[0]))
    assert isinstance(outer.handlers[0].body[-1],ast.Raise)


def test_actual_collected_snapshot_has_identical_inventory_without_admitting_partial_replay():
    saved=ROOT/'runs/pretrained/diagnostic/predictor-forward-preserving-handoff-v2-20261114/linux10603-final/attended-yo4uf0wc/saved'
    before=observer.frozen.inventory(saved,hashing=True,max_bytes=3*2**30,running=False)
    actual=observer.inventory(saved,hashing=True,max_bytes=3*2**30,running=False)
    assert actual==before and len(actual[0])==280
    state=json.loads((saved/'results/3/worker.json').read_text())
    assert state['complete'] is False and state['passed'] is False and not (saved/'results/3/success.json').exists()
    publish('actual-collected-inventory',{'passed':True,'files':len(actual[0]),'logical_bytes':sum(x['bytes'] for x in actual[0].values()),
             'inventory_matches_frozen':True,'historical_failure_resolved':False,'partial_replay_still_unadmitted':True,
             'new_scientific_or_remote_work':False})
