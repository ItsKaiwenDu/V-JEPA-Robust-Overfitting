"""Allocated-block checks for a future scratch operator; no launch entry point.

This does not install a filesystem quota. It refuses an observed excess and
preserves its receipt before any caller can start numerical work. Unsupported
preallocation is a refusal, never a fallback to extending writes.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import time

KIND = 'scratch-allocation-proof-v2'
GIB = 2**30
DEFAULT_SIZE = 5 * GIB // 2
DEFAULT_CEILING = 3 * GIB


class AllocationRefusal(RuntimeError):
    def __init__(self, message, observation=None):
        super().__init__(message)
        self.observation = observation


def check_footprint(observation, *, ceiling, free_floor):
    """Both logical length and actual allocated blocks must fit independently."""
    for key in ('logical_bytes', 'allocated_bytes', 'available_bytes'):
        if type(observation.get(key)) is not int or observation[key] < 0:
            raise AllocationRefusal('Invalid filesystem-accounting observation.', observation)
    if observation['logical_bytes'] > ceiling or observation['allocated_bytes'] > ceiling:
        raise AllocationRefusal('Setup logical/allocated-byte ceiling exceeded.', observation)
    if observation['available_bytes'] < free_floor:
        raise AllocationRefusal('Setup available-space floor reached.', observation)


def inventory(folder, identity):
    folder = Path(folder)
    root = folder.lstat()
    if (not stat.S_ISDIR(root.st_mode) or (root.st_dev, root.st_ino, root.st_uid) != identity
            or stat.S_IMODE(root.st_mode) != 0o700):
        raise AllocationRefusal('Private proof directory identity changed.')
    allocated = root.st_blocks * 512
    logical = 0
    count = 0
    seen = {(root.st_dev, root.st_ino)}
    for current, dirs, files in os.walk(folder, followlinks=False):
        for name in dirs + files:
            path = Path(current) / name
            info = path.lstat()
            if info.st_dev != root.st_dev or stat.S_ISLNK(info.st_mode):
                raise AllocationRefusal('Proof path changed filesystem or became a symlink.')
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise AllocationRefusal('Non-regular setup member.')
            if stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise AllocationRefusal('Setup refuses multiply linked files.')
                logical += info.st_size
                count += 1
            key = (info.st_dev, info.st_ino)
            if key not in seen:
                allocated += info.st_blocks * 512
                seen.add(key)
            if count > 1024:
                raise AllocationRefusal('Setup file-count bound exceeded.')
    capacity = os.statvfs(folder)
    return {'logical_bytes': logical, 'allocated_bytes': allocated,
            'available_bytes': capacity.f_bavail * capacity.f_frsize,
            'available_inodes': capacity.f_favail, 'files': count}


def bounded_write_proof(folder, *, size=DEFAULT_SIZE, ceiling=DEFAULT_CEILING,
                        minimum_free=8*GIB, free_floor=GIB, deadline,
                        allocator=None, block_bytes=4*2**20):
    """Preallocate one fixed extent, write within it, and account through cleanup.

    Only a verified newly created proof file is removed. Refusals before that
    cleanup retain partial bytes; all stages retain their event trail. A caller must
    include its own other setup files and emergency receipt space in this root.
    """
    if (type(size) is not int or size <= 0 or type(ceiling) is not int or ceiling < size
            or type(block_bytes) is not int or not 0 < block_bytes <= 4*2**20
            or type(minimum_free) is not int or type(free_floor) is not int or free_floor < 0
            or minimum_free < ceiling + free_floor or not 0 < deadline-time.monotonic() <= 60):
        raise ValueError('Invalid bounded allocation policy or original deadline.')
    folder = Path(folder)
    first = folder.lstat()
    identity = (first.st_dev, first.st_ino, first.st_uid)
    if first.st_uid != os.getuid():
        raise AllocationRefusal('Require an owned private scratch directory.')
    proof = folder / 'space-proof-v2.bin.partial'
    events_path = folder / 'allocation-events-v2.jsonl'
    result_path = folder / 'allocation-result-v2.json'
    if any(p.exists() or p.is_symlink() for p in (proof, events_path, result_path)):
        raise FileExistsError('A fresh v2 allocation proof is required; preserve existing evidence.')
    allocate = allocator if allocator is not None else getattr(os, 'posix_fallocate', None)
    if allocate is None:
        raise AllocationRefusal('posix_fallocate is unavailable; no extending-write fallback.')
    initial = inventory(folder, identity)
    check_footprint(initial, ceiling=ceiling, free_floor=minimum_free)
    if initial['allocated_bytes'] + size + 2*2**20 > ceiling:
        raise AllocationRefusal('No setup allowance for proof plus operational metadata.', initial)
    if initial['available_inodes'] < 2048:
        raise AllocationRefusal('Insufficient setup inodes.')
    started = time.monotonic()
    result = {'kind': KIND, 'complete': False, 'passed': False, 'proof_removed': False,
        'requested_bytes': size, 'setup_ceiling_bytes': ceiling, 'bytes_written': 0,
        'bytes_read_back': 0, 'peak_observed_allocated_bytes': initial['allocated_bytes'],
        'kernel_aggregate_quota': False, 'reservation_against_other_writers': False,
        'live_XFS_qualified': False, 'preallocator_injected_for_test': allocator is not None}
    fd = None
    with events_path.open('xb', buffering=0) as events:
        def event(stage, extra=None):
            if time.monotonic() >= deadline:
                raise TimeoutError('Original allocation-proof deadline reached.')
            observation = inventory(folder, identity)
            result['peak_observed_allocated_bytes'] = max(result['peak_observed_allocated_bytes'], observation['allocated_bytes'])
            line = json.dumps({'stage': stage, 'elapsed_seconds': time.monotonic()-started,
                'footprint': observation, **(extra or {})}, sort_keys=True).encode() + b'\n'
            if events.tell() + len(line) > 2**20:
                raise AllocationRefusal('Allocation event-log bound exceeded.', observation)
            events.write(line)
            check_footprint(observation, ceiling=ceiling, free_floor=free_floor)
            # Logging can allocate another extent too; count it before returning.
            after_log=inventory(folder, identity)
            result['peak_observed_allocated_bytes']=max(result['peak_observed_allocated_bytes'],after_log['allocated_bytes'])
            check_footprint(after_log, ceiling=ceiling, free_floor=free_floor)
            if observation['available_inodes'] < 1024:
                raise AllocationRefusal('Setup inode floor reached.', observation)
            return observation
        try:
            event('before-create')
            fd = os.open(proof, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            own = os.fstat(fd)
            if own.st_dev != identity[0] or own.st_nlink != 1 or own.st_uid != os.getuid():
                raise AllocationRefusal('New proof-file ownership/device changed.')
            allocate(fd, 0, size)
            os.fsync(fd)
            if os.fstat(fd).st_size != size:
                raise AllocationRefusal('Preallocation did not establish the exact file length.')
            event('after-preallocate')
            os.lseek(fd, 0, os.SEEK_SET)
            hashed = hashlib.sha256()
            while result['bytes_written'] < size:
                event('before-write')
                data = os.urandom(min(block_bytes, size-result['bytes_written']))
                view = memoryview(data)
                while view:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Proof write deadline reached.')
                    count = os.write(fd, view)
                    if not 0 < count <= len(view):
                        raise AllocationRefusal('Invalid/zero proof write progress.')
                    result['bytes_written'] += count
                    view = view[count:]
                hashed.update(data)
                if os.fstat(fd).st_size != size:
                    raise AllocationRefusal('Writing escaped the preallocated extent.')
                event('after-write')
            os.fsync(fd)
            result['fsync_succeeded'] = True
            event('after-data-fsync')
            os.lseek(fd, 0, os.SEEK_SET)
            actual = hashlib.sha256()
            while result['bytes_read_back'] < size:
                event('before-read')
                data = os.read(fd, min(block_bytes, size-result['bytes_read_back']))
                if not data:
                    raise AllocationRefusal('Proof readback ended early.')
                actual.update(data)
                result['bytes_read_back'] += len(data)
            if actual.hexdigest() != hashed.hexdigest():
                raise AllocationRefusal('Proof readback hash differs from bytes actually written.')
            result['sha256'] = actual.hexdigest()
            own = os.fstat(fd)
            if own.st_blocks * 512 < size:
                raise AllocationRefusal('Sparse/compressed proof does not demonstrate full allocation.')
            event('before-close', {'proof_allocated_bytes': own.st_blocks*512})
            os.close(fd); fd = None
            event('after-close')
            current = proof.lstat()
            if (current.st_dev, current.st_ino, current.st_nlink) != (own.st_dev, own.st_ino, 1):
                raise AllocationRefusal('Named proof changed before own-file cleanup.')
            proof.unlink()
            result['proof_removed'] = True
            event('after-own-proof-removal')
            result.update(complete=True, passed=True)
        except BaseException as exc:
            result.update(error_type=type(exc).__name__, error=str(exc))
            if isinstance(exc, AllocationRefusal):
                result['refusal_observation'] = exc.observation
            raise
        finally:
            if fd is not None:
                os.close(fd)
            result['wall_seconds'] = time.monotonic()-started
            os.fsync(events.fileno())
            # Caller-provided emergency space remains necessary if the device
            # stops accepting even this bounded failure record.
            with result_path.open('x') as handle:
                json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
                handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    return result
