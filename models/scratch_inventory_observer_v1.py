"""Add bounded refusal metadata after the unchanged scratch inventory has failed.

No acceptance decision, filesystem walk or live operator is replaced. A refusal
stays a refusal; optional metadata rechecks never retry or admit a scan.
"""
import hashlib
import json
import os
from pathlib import Path
import stat

from models import oracle_boundary_operator_v1 as frozen

FORMAT = 'scratch-inventory-refusal-metadata-v1'
SOURCE_SHA = 'dde0d903986a5e83d02d9e7cb5fe4b6869d02f3731c8d4f30dd749a20f742b73'
MAX_CONTEXT_BYTES = 1536
MAX_PATH_JSON_BYTES = 384
_FROZEN_SCAN = frozen.inventory
_FROZEN_CODE = _FROZEN_SCAN.__code__


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def _snapshot(value):
    if not isinstance(value, os.stat_result):
        return None
    mode = value.st_mode
    kind = ('regular' if stat.S_ISREG(mode) else 'directory' if stat.S_ISDIR(mode)
            else 'symlink' if stat.S_ISLNK(mode) else 'fifo' if stat.S_ISFIFO(mode)
            else 'socket' if stat.S_ISSOCK(mode) else 'device' if stat.S_ISCHR(mode) or stat.S_ISBLK(mode)
            else 'other')
    return {'device': value.st_dev, 'inode': value.st_ino, 'mode': stat.S_IMODE(mode),
            'kind': kind, 'links': value.st_nlink, 'uid': value.st_uid,
            'bytes': value.st_size, 'allocated_bytes': value.st_blocks*512,
            'mtime_ns': value.st_mtime_ns}


def _recheck(path):
    # lstat observes the link itself. Never follow/read a symlink target or file.
    try:
        return {'stat': _snapshot(os.lstat(path))}
    except OSError as error:
        return {'error_type': type(error).__name__, 'errno': error.errno}


def _bounded(context):
    display = context.get('relative_path', '')[:MAX_PATH_JSON_BYTES]
    while len(_json(display).encode('ascii')) > MAX_PATH_JSON_BYTES:
        display = display[:-1]
    if 'relative_path' in context:
        context['relative_path_truncated'] = display != context['relative_path']
        context['relative_path'] = display
    # The original observed inode/mode/link count and path digest are retained.
    # Follow-up snapshots are optional diagnostic evidence, never a new verdict.
    for optional in ('partner_now', 'entry_now'):
        if len(_json(context).encode('ascii')) <= MAX_CONTEXT_BYTES:
            break
        context.pop(optional, None)
    if len(_json(context).encode('ascii')) > MAX_CONTEXT_BYTES:
        context['relative_path'] = context.get('relative_path', '')[:64]
        context['relative_path_truncated'] = True
    if len(_json(context).encode('ascii')) > MAX_CONTEXT_BYTES:
        raise ValueError('Refusal context byte cap.')
    return context


def _observation(error, *, hashing, running):
    tb = error.__traceback__
    frame = None
    for _ in range(32):
        if tb is None:
            break
        if tb.tb_frame.f_code is _FROZEN_CODE:
            frame = tb.tb_frame
            break
        tb = tb.tb_next
    context = {'format': FORMAT, 'metadata_only_addition': True, 'scan_retried': False,
               'refusal_preserved': True, 'hashing': hashing, 'running': running}
    if frame is None:
        context['capture_status'] = 'inventory-frame-unavailable'
        return _bounded(context)
    values = frame.f_locals
    folder, path, info = values.get('folder'), values.get('path'), values.get('info')
    context.update(guard_line=tb.tb_lineno, observed_stat=_snapshot(info),
                   root_device=getattr(values.get('root'), 'st_dev', None))
    if not isinstance(folder, Path) or not isinstance(path, Path):
        context['capture_status'] = 'entry-unavailable'
        return _bounded(context)
    relative = str(path.relative_to(folder))
    context.update(capture_status='captured', relative_path=relative,
                   relative_path_sha256=hashlib.sha256(os.fsencode(relative)).hexdigest())
    context['entry_now'] = _recheck(path)
    if isinstance(info, os.stat_result) and info.st_nlink == 2:
        name = path.name[:-8] if path.name.endswith('.partial') else path.name
        if name.endswith(('.json', '.npz', '.pt')):
            partner = path.with_name(name if path.name.endswith('.partial') else name+'.partial')
            context['partner_now'] = _recheck(partner)
            context['guard_paired'] = bool(values.get('paired', False))
    return _bounded(context)


def inventory(folder, *, hashing=False, max_bytes=None, running=False):
    """Delegate exactly once; only decorate a ValueError after its failed decision.

    Successful scans and all other exception types propagate unchanged. The
    frozen scan's requested hashing still occurs; added diagnostics read no data.
    """
    if (frozen.inventory is not _FROZEN_SCAN or _FROZEN_SCAN.__code__ is not _FROZEN_CODE
            or hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest() != SOURCE_SHA):
        raise ValueError('Frozen scratch inventory source/identity changed.')
    try:
        return _FROZEN_SCAN(folder, hashing=hashing, max_bytes=max_bytes, running=running)
    except ValueError as original:
        try:
            context = _observation(original, hashing=hashing, running=running)
        except Exception as diagnostic_error:
            # A logging failure cannot admit an invalid scan or hide its reason.
            context = {'format': FORMAT, 'capture_status': 'diagnostic-unavailable',
                       'diagnostic_error_type': type(diagnostic_error).__name__[:80],
                       'scan_retried': False, 'refusal_preserved': True}
        encoded = _json(context)
        refusal = ValueError(str(original)+' | inventory='+encoded)
        refusal.inventory_observation = context
        raise refusal from original
