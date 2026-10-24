"""Checkpoint locks retain reader/writer protection on POSIX and NFS-style filesystems."""

import errno
import fcntl
import json
import subprocess
import sys

import pytest
import torch

from models.checkpointing import CheckpointStore


LOCK_CHILD = """
import errno, fcntl, json, pathlib, sys
with pathlib.Path(sys.argv[1]).open('a+') as handle:
    try:
        fcntl.flock(handle, int(sys.argv[2]) | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
            raise
        print(json.dumps({'acquired': False, 'errno': error.errno}))
    else:
        print(json.dumps({'acquired': True}))
        fcntl.flock(handle, fcntl.LOCK_UN)
"""


def contender(root, exclusive):
    result = subprocess.run([sys.executable, '-c', LOCK_CHILD,
                             str(root / '.writer.lock'), str(fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)['acquired']


def test_checkpoint_round_trip_with_nfs_record_lock_access_requirements(tmp_path, monkeypatch):
    # Linux NFS implements flock with record locks. Use real POSIX record locks
    # here to exercise their access requirements even without a mounted NFS share.
    # This is a test control only; production continues to use flock.
    monkeypatch.setattr('models.checkpointing.fcntl.flock', fcntl.lockf)
    store = CheckpointStore(tmp_path)
    lock_file = tmp_path / '.writer.lock'
    lock_file.write_bytes(b'preexisting lock-file contents\n')
    payload = {'step': 0, 'contract_sha256': 'lock-fixture',
               'encoder': {'weight': torch.arange(4, dtype=torch.float32)}}
    first_ref = store.save(payload)
    loaded, ref = store.load()
    assert ref == first_ref and loaded['step'] == 0
    assert torch.equal(loaded['encoder']['weight'], payload['encoder']['weight'])
    store.save({**payload, 'step': 1})
    assert store.load()[0]['step'] == 1 and store.load('previous')[0]['step'] == 0
    assert lock_file.read_bytes() == b'preexisting lock-file contents\n'


@pytest.mark.parametrize('holder_exclusive,request_exclusive,allowed', [
    (False, False, True), (False, True, False),
    (True, False, False), (True, True, False),
])
def test_independent_processes_share_reads_and_reject_conflicting_locks(
        tmp_path, holder_exclusive, request_exclusive, allowed):
    store = CheckpointStore(tmp_path)
    with store._lock(exclusive=holder_exclusive):
        assert contender(tmp_path, request_exclusive) is allowed
    assert contender(tmp_path, request_exclusive)


@pytest.mark.parametrize('error_number', [errno.EBADF, errno.ENOLCK])
def test_lock_errors_abort_checkpoint_read_and_write_without_fallback(tmp_path, monkeypatch, error_number):
    store = CheckpointStore(tmp_path)
    payload = {'step': 0, 'contract_sha256': 'lock-fixture'}
    store.save(payload)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    def unavailable(*args):
        raise OSError(error_number, 'deliberate unavailable checkpoint lock')

    monkeypatch.setattr('models.checkpointing.fcntl.flock', unavailable)
    for operation in (store.load, lambda: store.save({**payload, 'step': 1})):
        with pytest.raises(OSError) as caught:
            operation()
        assert caught.value.errno == error_number
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_checkpoint_lock_released_after_protected_operation_raises(tmp_path):
    store = CheckpointStore(tmp_path)
    with pytest.raises(RuntimeError, match='deliberate protected failure'):
        with store._lock(exclusive=True):
            assert not contender(tmp_path, exclusive=True)
            raise RuntimeError('deliberate protected failure')
    assert contender(tmp_path, exclusive=True)
