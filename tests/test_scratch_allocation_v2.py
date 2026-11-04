"""Small I/O plus simulated XFS block accounting; no full-size/live host test."""
import errno
import json
import os
from pathlib import Path
import time

import pytest

import models.scratch_allocation_v2 as module

MIB=2**20


def policy():
    return {'size':16384,'ceiling':4*MIB,'minimum_free':4*MIB,'free_floor':0,
        'deadline':time.monotonic()+10,'block_bytes':4096}


def test_allocator(fd,offset,length):
    # Explicit injected test backend on the Mac. Production must require the
    # OS preallocator; successful small writes here do not qualify XFS.
    assert offset==0
    os.ftruncate(fd,length)


test_allocator.__test__=False


def test_real_small_write_fsync_hash_and_only_own_proof_cleanup(tmp_path):
    witness=tmp_path/'keep';witness.write_bytes(b'keep original')
    result=module.bounded_write_proof(tmp_path,allocator=test_allocator,**policy())
    assert result['passed'] and result['complete'] and result['proof_removed']
    assert result['bytes_written']==result['bytes_read_back']==16384
    assert result['preallocator_injected_for_test'] and not result['live_XFS_qualified']
    assert not result['kernel_aggregate_quota'] and not result['reservation_against_other_writers']
    assert not (tmp_path/'space-proof-v2.bin.partial').exists() and witness.read_bytes()==b'keep original'
    lines=[json.loads(line) for line in (tmp_path/'allocation-events-v2.jsonl').read_text().splitlines()]
    stages=[line['stage'] for line in lines]
    for stage in ('after-preallocate','after-data-fsync','before-close','after-close','after-own-proof-removal'):
        assert stage in stages
    assert all(line['footprint']['allocated_bytes']<=4*MIB for line in lines)
    stored=json.loads((tmp_path/'allocation-result-v2.json').read_text())
    assert stored==result and result['peak_observed_allocated_bytes']<=4*MIB


def test_recorded_four_gib_allocation_is_rejected_under_same_three_gib_ceiling():
    observed={'logical_bytes':2684354560,'allocated_bytes':4294901760,'available_bytes':8*2**30}
    with pytest.raises(module.AllocationRefusal,match='ceiling') as error:
        module.check_footprint(observed,ceiling=3*2**30,free_floor=2**30)
    assert error.value.observation==observed


@pytest.mark.parametrize('stage',['preallocate','first-write','close'])
def test_overallocation_stops_before_success_and_preserves_partial(tmp_path,monkeypatch,stage):
    original_inventory=module.inventory;original_write=os.write;original_close=os.close
    state={'trigger':False,'writes':0}
    def allocate(fd,offset,length):
        test_allocator(fd,offset,length)
        if stage=='preallocate':state['trigger']=True
    def write(fd,data):
        count=original_write(fd,data);state['writes']+=1
        if stage=='first-write':state['trigger']=True
        return count
    def close(fd):
        original_close(fd)
        if stage=='close':state['trigger']=True
    def inventory(folder,identity):
        result=original_inventory(folder,identity)
        if state['trigger']:result['allocated_bytes']=4294901760
        return result
    monkeypatch.setattr(module,'inventory',inventory)
    monkeypatch.setattr(module.os,'write',write);monkeypatch.setattr(module.os,'close',close)
    with pytest.raises(module.AllocationRefusal,match='ceiling'):
        module.bounded_write_proof(tmp_path,allocator=allocate,**policy())
    result=json.loads((tmp_path/'allocation-result-v2.json').read_text())
    assert not result['passed'] and not result['complete'] and not result['proof_removed']
    assert result['peak_observed_allocated_bytes']==4294901760
    assert result['refusal_observation']['allocated_bytes']==4294901760
    assert (tmp_path/'space-proof-v2.bin.partial').exists()
    if stage=='preallocate':assert result['bytes_written']==0 and state['writes']==0
    if stage=='first-write':assert result['bytes_written']==4096 and state['writes']==1


@pytest.mark.parametrize('number',[errno.EDQUOT,errno.ENOSPC,errno.EOPNOTSUPP])
def test_preallocation_error_has_no_extending_write_fallback(tmp_path,number):
    calls=[]
    def unsupported(fd,offset,size):
        calls.append(size);raise OSError(number,'injected allocator refusal')
    with pytest.raises(OSError):module.bounded_write_proof(tmp_path,allocator=unsupported,**policy())
    result=json.loads((tmp_path/'allocation-result-v2.json').read_text())
    assert len(calls)==1 and not result['passed'] and result['bytes_written']==0
    assert (tmp_path/'space-proof-v2.bin.partial').stat().st_size==0


def test_missing_os_preallocator_refuses_before_file_creation(tmp_path,monkeypatch):
    monkeypatch.delattr(module.os,'posix_fallocate',raising=False)
    with pytest.raises(module.AllocationRefusal,match='unavailable'):
        module.bounded_write_proof(tmp_path,**policy())
    assert not list(tmp_path.iterdir())


def test_existing_proof_is_not_overwritten(tmp_path):
    path=tmp_path/'space-proof-v2.bin.partial';path.write_bytes(b'preserve failure')
    with pytest.raises(FileExistsError):module.bounded_write_proof(tmp_path,allocator=test_allocator,**policy())
    assert path.read_bytes()==b'preserve failure' and not (tmp_path/'allocation-events-v2.jsonl').exists()


def test_readback_corruption_keeps_proof(tmp_path,monkeypatch):
    original_read=os.read;first=True
    def read(fd,count):
        nonlocal first
        data=original_read(fd,count)
        if first and data:
            first=False;return bytes([data[0]^1])+data[1:]
        return data
    monkeypatch.setattr(module.os,'read',read)
    with pytest.raises(module.AllocationRefusal,match='hash differs'):
        module.bounded_write_proof(tmp_path,allocator=test_allocator,**policy())
    assert (tmp_path/'space-proof-v2.bin.partial').exists()
    assert json.loads((tmp_path/'allocation-result-v2.json').read_text())['passed'] is False


def test_short_writes_are_completed_within_preallocated_length(tmp_path,monkeypatch):
    original=os.write
    monkeypatch.setattr(module.os,'write',lambda fd,data:original(fd,data[:127]))
    result=module.bounded_write_proof(tmp_path,allocator=test_allocator,**policy())
    assert result['passed'] and result['bytes_written']==16384


def test_setup_deadline_not_reset_by_preallocation(tmp_path):
    def slow(fd,offset,size):
        os.ftruncate(fd,size);time.sleep(.03)
    p=policy();p['deadline']=time.monotonic()+.01
    with pytest.raises(TimeoutError):module.bounded_write_proof(tmp_path,allocator=slow,**p)
    result=json.loads((tmp_path/'allocation-result-v2.json').read_text())
    assert not result['passed'] and result['bytes_written']==0


@pytest.mark.parametrize('kind',['symlink','hardlink'])
def test_setup_aliases_refuse_before_proof(tmp_path,kind):
    folder=tmp_path/'private';folder.mkdir(mode=0o700)
    source=tmp_path/'source';source.write_bytes(b'leave alone')
    if kind=='symlink':(folder/'alias').symlink_to(source)
    else:os.link(source,folder/'alias')
    with pytest.raises(module.AllocationRefusal):module.bounded_write_proof(folder,allocator=test_allocator,**policy())
    assert source.read_bytes()==b'leave alone' and not (folder/'space-proof-v2.bin.partial').exists()


def test_insufficient_free_space_refuses_before_preallocation(tmp_path,monkeypatch):
    original=module.inventory
    def low(folder,identity):
        result=original(folder,identity);result['available_bytes']=0;return result
    monkeypatch.setattr(module,'inventory',low)
    with pytest.raises(module.AllocationRefusal,match='space floor'):
        module.bounded_write_proof(tmp_path,allocator=test_allocator,**policy())
    assert not list(tmp_path.iterdir())


def test_growth_of_other_setup_files_is_counted(tmp_path,monkeypatch):
    original_write=os.write
    def write(fd,data):
        with (tmp_path/'other-setup-file').open('wb') as handle:handle.truncate(5*MIB)
        return original_write(fd,data)
    monkeypatch.setattr(module.os,'write',write)
    with pytest.raises(module.AllocationRefusal,match='ceiling'):
        module.bounded_write_proof(tmp_path,allocator=test_allocator,**policy())
    assert (tmp_path/'other-setup-file').is_file() and (tmp_path/'space-proof-v2.bin.partial').exists()


@pytest.mark.parametrize('key,value',[('size',0),('ceiling',1),('deadline',float('nan')),('block_bytes',0),('minimum_free',-1),('free_floor',-1)])
def test_invalid_policy_refuses_without_io(tmp_path,key,value):
    p=policy();p[key]=value
    with pytest.raises(ValueError):module.bounded_write_proof(tmp_path,allocator=test_allocator,**p)
    assert not list(tmp_path.iterdir())
