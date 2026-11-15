"""Local-only private reconstruction of an already saved same-SSH response.

No connection, claim, authorization, model or target operation is available here.
The frozen framing/hash/mode/nsmtime decoder executes its unchanged code object.
"""
import hashlib
import os
from pathlib import Path
import re
import stat
from types import FunctionType, SimpleNamespace
from models import predictor_eot8_record_client_v2 as frozen

FROZEN_SHA = 'd0d12ef34b731906b12ac964416eba3a48c1d8a2e80cd412bb3054f18c69aede'
_UNPACK = frozen.unpack
_CODE = _UNPACK.__code__
MiB, GiB = 2**20, 2**30


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _private_directory(path):
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700
            or info.st_uid != os.getuid()):
        raise ValueError('Owned private collection directory required.')
    return info


class _PinnedSource:
    """Give the old decoder the same open file, never a newly resolved pathname."""
    def __init__(self, fd):
        self.fd = fd

    def open(self, mode):
        if mode != 'rb':
            raise ValueError('Collected response is read-only.')
        os.lseek(self.fd, 0, os.SEEK_SET)
        return os.fdopen(os.dup(self.fd), 'rb')

    def stat(self):
        return os.fstat(self.fd)


def unpack(response, destination, control, control_sha256, *, response_sha256):
    """Decode locally into a new private root; preserve every old file and refusal.

    The caller supplies the recorded response hash and control identity. Directory
    modes are an explicit local 0700 reconstruction policy, not transported target
    metadata. The original read-only input may be 0644 inside its private session
    directory; it is never chmodded. Output remains private. The process umask is
    restored on success and every failure.
    """
    if (frozen.unpack is not _UNPACK or _UNPACK.__code__ is not _CODE
            or hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest() != FROZEN_SHA):
        raise ValueError('Frozen collection decoder source/identity changed.')
    if (hashlib.sha256(frozen.old.json_bytes(control)).hexdigest() != control_sha256
            or not isinstance(control.get('kind'), str) or not control['kind']
            or not isinstance(response_sha256, str) or not re.fullmatch('[0-9a-f]{64}', response_sha256)):
        raise ValueError('Exact recorded control and response identities required.')
    source, dest = Path(response).absolute(), Path(destination).absolute()
    if source.resolve() != source or dest.parent.resolve() != dest.parent:
        raise ValueError('Canonical source and destination parent required; no path aliases.')
    if os.path.lexists(dest):
        raise FileExistsError('Use a fresh local destination; preserve all prior output.')
    parent = _private_directory(dest.parent)
    _private_directory(source.parent)
    before = source.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) not in (0o400, 0o600, 0o644) or not 0 < before.st_size <= 3*GiB):
        raise ValueError('Owned single-link bounded saved response required.')
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if _signature(os.fstat(fd)) != _signature(before):
            raise ValueError('Response changed during open.')
        with os.fdopen(os.dup(fd), 'rb') as stream:
            actual_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual_sha != response_sha256 or _signature(os.fstat(fd)) != _signature(before):
            raise ValueError('Recorded response changed before local reconstruction.')
        pinned = _PinnedSource(fd)
        def paths(value):
            return pinned if value is pinned else Path(value)
        view = SimpleNamespace(KIND=control['kind'], MiB=MiB, GiB=GiB)
        decoder = FunctionType(_CODE, dict(_UNPACK.__globals__, Path=paths, op=view),
                               argdefs=_UNPACK.__defaults__)
        decoder.__kwdefaults__ = _UNPACK.__kwdefaults__
        previous = os.umask(0o077)
        try:
            receipt = decoder(pinned, dest, control, control_sha256, collect=False)
            root = _private_directory(dest)
            if root.st_dev != parent.st_dev:
                raise ValueError('Collection destination changed filesystem.')
            for path in dest.rglob('*'):
                info = path.lstat()
                if info.st_dev != root.st_dev or info.st_uid != os.getuid():
                    raise ValueError('Local collection ownership/device changed.')
                if stat.S_ISDIR(info.st_mode):
                    _private_directory(path)
                elif (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                      or stat.S_IMODE(info.st_mode) not in (0o400, 0o600)):
                    raise ValueError('Only private single-link regular collection members allowed.')
            if (_signature(os.fstat(fd)) != _signature(before)
                    or _signature(source.lstat()) != _signature(before)):
                raise ValueError('Response changed during local reconstruction; preserve partial output.')
            return receipt
        finally:
            os.umask(previous)
    finally:
        os.close(fd)
