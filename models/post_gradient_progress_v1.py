"""Bounded durable diagnostic metadata; never a scientific completion receipt.

No model, tensor, profiler, SSH, process runner or training policy is imported.
Before/after records bound the last observed interval; they do not prove a cause.
"""
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time

FORMAT = 'post-gradient-progress-v1'
FILE = 'post-gradient-progress-v1.jsonl'
MAX_BYTES = 64 * 1024
MAX_ROWS = 64
MAX_ROW = 1024
ZERO = '0' * 64
PHASES = frozenset(('gradient_checks', 'partial_snapshot', 'hook_removal', 'native_exit',
    'profiler_stop', 'restoration_checks', 'input_restore_checks', 'native_proof_validation',
    'events_materialize', 'events_aggregate', 'backend_publish', 'backend_validate', 'record_publish'))
CLEANUP = frozenset(('hook_removal', 'native_exit', 'profiler_stop'))
REQUIRED = frozenset(('gradient_checks', 'hook_removal', 'native_exit', 'restoration_checks',
                      'input_restore_checks', 'native_proof_validation'))
PROFILE_REQUIRED = frozenset(('partial_snapshot', 'profiler_stop', 'events_materialize',
                             'events_aggregate', 'backend_publish', 'backend_validate'))


class ProgressError(ValueError):
    pass


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def _plain(value, depth=0):
    """Reject tensor-like/arbitrary objects before invoking serialization methods."""
    if depth > 12:
        raise ProgressError('Metadata nesting bound.')
    if value is None or type(value) is bool:
        return
    if type(value) is int and abs(value) < 2**128:
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is str and len(value) <= 4096:
        return
    if type(value) in (list, tuple) and len(value) <= 4096:
        for item in value:
            _plain(item, depth+1)
        return
    if type(value) is dict and len(value) <= 128 and all(type(k) is str and len(k) <= 128 for k in value):
        for item in value.values():
            _plain(item, depth+1)
        return
    raise ProgressError('Only bounded plain diagnostic metadata is permitted.')


def _private_folder(folder):
    p = Path(folder).absolute()
    if p.resolve(strict=True) != p:
        raise ProgressError('Canonical existing private folder required.')
    s = p.lstat()
    if not stat.S_ISDIR(s.st_mode) or s.st_uid != os.getuid() or stat.S_IMODE(s.st_mode) != 0o700:
        raise ProgressError('Owned 0700 diagnostic directory required.')
    return p, s


def _file_id(s):
    return (s.st_dev, s.st_ino, s.st_uid, stat.S_IMODE(s.st_mode), s.st_nlink)


class Journal:
    """One append-only, fsynced 64 KiB log in a new private capture directory."""
    def __init__(self, folder, *, control_sha256, objective_index, profiled, clock=time.monotonic_ns):
        if (not isinstance(control_sha256, str) or not re.fullmatch('[0-9a-f]{64}', control_sha256)
                or type(objective_index) is not int or not 0 <= objective_index < 8
                or type(profiled) is not bool or not callable(clock)):
            raise ProgressError('Exact diagnostic identity required.')
        self.folder, parent = _private_folder(folder)
        self.identity = {'control_sha256': control_sha256, 'objective_index': objective_index,
                         'pid': os.getpid(), 'thread': threading.get_ident(), 'profiled': profiled}
        self.clock = clock
        self.started = clock()
        if type(self.started) is not int or self.started < 0:
            raise ProgressError('Monotonic integer clock required.')
        self.last_clock = self.started
        self.prefix, self.head, self.rows = b'', ZERO, 0
        self.pending = None
        self.begun, self.returned, self.raised = set(), set(), set()
        self.gradient_saved = False
        self.poisoned = self.closed = False
        self.parent_fd = os.open(self.folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.fd = None
        try:
            if _file_id(os.fstat(self.parent_fd)) != _file_id(parent):
                raise ProgressError('Diagnostic parent changed while opening.')
            self.fd = os.open(FILE, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                              0o600, dir_fd=self.parent_fd)
            self.file_identity = _file_id(os.fstat(self.fd))
            os.fsync(self.parent_fd)  # Make the new name durable before callbacks.
            self._write('observer', 'opened', {})
        except BaseException:
            self.close()
            raise

    def owner(self):
        if (os.getpid(), threading.get_ident()) != (self.identity['pid'], self.identity['thread']):
            raise ProgressError('Diagnostic writer belongs to its creator process/thread.')

    def _check(self):
        self.owner()
        if self.closed or self.poisoned:
            raise ProgressError('Diagnostic writer closed or poisoned; preserve partial file.')
        parent = self.folder.lstat()
        if (_file_id(parent) != _file_id(os.fstat(self.parent_fd)) or stat.S_IMODE(parent.st_mode) != 0o700):
            raise ProgressError('Diagnostic directory identity/mode changed.')
        held = os.fstat(self.fd)
        named = os.stat(FILE, dir_fd=self.parent_fd, follow_symlinks=False)
        if (not stat.S_ISREG(named.st_mode) or _file_id(held) != self.file_identity
                or _file_id(named) != self.file_identity or held.st_size != len(self.prefix)
                or os.pread(self.fd, len(self.prefix)+1, 0) != self.prefix):
            raise ProgressError('Diagnostic inode, mode, links or committed prefix changed.')

    def _write(self, phase, event, detail):
        try:
            self._check()
            _plain(detail)
            tick = self.clock()
            if type(tick) is not int or tick < self.last_clock:
                raise ProgressError('Monotonic diagnostic clock changed.')
            body = {'kind': FORMAT, 'sequence': self.rows, 'identity': self.identity,
                    'phase': phase, 'event': event, 'elapsed_ns': tick-self.started,
                    'previous_sha256': self.head, 'detail': detail, 'scientific_qualified': False}
            checksum = hashlib.sha256(encoded(body)).hexdigest()
            data = encoded(dict(body, row_sha256=checksum))
            if self.rows >= MAX_ROWS or len(data) > MAX_ROW or len(self.prefix)+len(data) > MAX_BYTES:
                raise ProgressError('Diagnostic row/file/count ceiling; no callback retry.')
            remaining = memoryview(data)
            while remaining:
                n = os.write(self.fd, remaining)
                if n <= 0:
                    raise OSError('Incomplete diagnostic append.')
                remaining = remaining[n:]
            os.fsync(self.fd)  # Before a callback starts, not only at final close.
            self.prefix += data
            self.head, self.rows, self.last_clock = checksum, self.rows+1, tick
        except BaseException:
            self.poisoned = True
            raise

    def observed_gradient(self, descriptor):
        if self.gradient_saved or self.pending is not None or self.begun:
            raise ProgressError('One gradient observation before finalization only.')
        # A descriptor is a provenance observation, not a numerical/proof gate.
        if (type(descriptor) is not dict or descriptor.get('file') != 'input.npy'
                or type(descriptor.get('bytes')) is not int or not 0 < descriptor['bytes'] <= 16*2**20
                or not isinstance(descriptor.get('sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', descriptor['sha256'])):
            raise ProgressError('Actual saved input-gradient descriptor required.')
        self._write('gradient_saved', 'observed', descriptor)
        self.gradient_saved = True

    def begin(self, phase):
        if phase not in PHASES or phase in self.begun or self.pending is not None:
            raise ProgressError('Fresh non-overlapping named diagnostic phase required.')
        self._write(phase, 'begin', {})
        self.begun.add(phase)
        self.pending = phase

    def end(self, phase, error=None):
        if self.pending != phase:
            raise ProgressError('End must name the active diagnostic phase.')
        detail = {} if error is None else {'error_type': type(error).__name__[:80]}
        self._write(phase, 'returned' if error is None else 'raised', detail)
        self.pending = None
        (self.returned if error is None else self.raised).add(phase)

    def require_publication_gates(self):
        self._check()
        needed = REQUIRED | (PROFILE_REQUIRED if self.identity['profiled'] else frozenset())
        if not self.gradient_saved or not needed <= self.returned or self.raised or self.pending is not None:
            raise ProgressError('No passing publication before all required completion gates return.')

    def close(self):
        # Closing never publishes a completion/success marker or repairs a tail.
        if getattr(self, 'fd', None) is not None:
            os.close(self.fd)
            self.fd = None
        if getattr(self, 'parent_fd', None) is not None:
            os.close(self.parent_fd)
            self.parent_fd = None
        self.closed = True


class Observer:
    def __init__(self, journal):
        self.journal = journal
        self.cleanup_attempted = set()
        self.publication_attempted = False

    def note_gradient(self, descriptor):
        return self.journal.observed_gradient(descriptor)

    def call(self, phase, action, *args, **kwargs):
        self.journal.begin(phase)
        try:
            result = action(*args, **kwargs)
        except BaseException as error:
            try:
                self.journal.end(phase, error)
            except BaseException as recording:
                error.add_note('Diagnostic end record also failed: '+type(recording).__name__)
            raise
        self.journal.end(phase)
        return result  # Never inspect, cast, copy or serialize the callback result.

    def cleanup(self, phase, action, *args, **kwargs):
        """Even a failed begin/end write may not skip mandatory resource cleanup."""
        self.journal.owner()
        if phase not in CLEANUP or phase in self.cleanup_attempted:
            raise ProgressError('One declared cleanup callback only; no retries.')
        self.cleanup_attempted.add(phase)
        before = None
        try:
            self.journal.begin(phase)
        except BaseException as recording:
            before = recording
        try:
            result = action(*args, **kwargs)
        except BaseException as error:
            if before is None:
                try:
                    self.journal.end(phase, error)
                except BaseException as recording:
                    error.add_note('Cleanup diagnostic end failed: '+type(recording).__name__)
            else:
                error.add_note('Cleanup diagnostic begin failed: '+type(before).__name__)
            raise
        if before is not None:
            raise before
        self.journal.end(phase)
        return result

    def publish(self, action, *, success, args=(), kwargs=None):
        self.journal.owner()
        if type(success) is not bool or self.publication_attempted:
            raise ProgressError('Explicit single-use publication outcome required.')
        self.publication_attempted = True
        kwargs = {} if kwargs is None else kwargs
        if success:
            self.journal.require_publication_gates()
            return self.call('record_publish', action, *args, **kwargs)
        # A failed journal must not prevent the authoritative writer from saving
        # an INCOMPLETE/FAILED record. Never use fallback for success publication.
        before = None
        try:
            self.journal.begin('record_publish')
        except BaseException as recording:
            before = recording
        try:
            result = action(*args, **kwargs)
        except BaseException as error:
            if before is None:
                try:
                    self.journal.end('record_publish', error)
                except BaseException as recording:
                    error.add_note('Failure-publication diagnostic failed: '+type(recording).__name__)
            raise
        if before is not None:
            raise before
        self.journal.end('record_publish')
        return result


class ObservedExitStack(ExitStack):
    """The same LIFO/suppression semantics, with named durable exit boundaries."""
    def __init__(self, observer):
        super().__init__()
        self.observer = observer
        self.registered = set()
        self.finished = False

    def _register(self, phase):
        self.observer.journal.owner()
        if phase not in CLEANUP or phase in self.registered or self.finished:
            raise ProgressError('One named cleanup registration before exit only.')
        self.registered.add(phase)

    def named_enter_context(self, phase, cm):
        self._register(phase)
        enter, exit = type(cm).__enter__, type(cm).__exit__
        result = enter(cm)
        def observed(*exc):
            return self.observer.cleanup(phase, exit, cm, *exc)
        ExitStack.push(self, observed)
        return result

    def named_callback(self, phase, callback, *args, **kwargs):
        self._register(phase)
        def original(*unused):
            callback(*args, **kwargs)
            return None
        def observed(*exc):
            return self.observer.cleanup(phase, original, *exc)
        ExitStack.push(self, observed)
        return callback

    def __exit__(self, *exc):
        self.observer.journal.owner()
        if self.finished:
            raise ProgressError('Observed exit stack is single use.')
        self.finished = True
        return super().__exit__(*exc)


SNAPSHOT_NAMES = ('provisional-native-calls-v1.jsonl', 'provisional-VJP-calls-v1.jsonl',
                  'provisional-scopes-v1.json', 'provisional-capture-v1.json')


def provisional_snapshot(journal, *, calls, scopes, refusals, vjps, borrowed):
    """Bounded metadata-only copy of actual observations; no proof is admitted."""
    journal._check()
    if (journal.identity['objective_index'] not in (0, 1) or not journal.identity['profiled']
            or journal.pending != 'partial_snapshot' or 'gradient_checks' not in journal.returned):
        raise ProgressError('Only the first original/candidate pair gets provisional snapshots.')
    if (type(calls) is not list or len(calls) > 4096 or type(vjps) is not list or len(vjps) > 384
            or type(scopes) is not list or len(scopes) > 1 or type(refusals) is not list or len(refusals) > 32):
        raise ProgressError('Original per-objective metadata row caps remain unchanged.')
    payloads = {}
    for name, rows, row_cap, file_cap in ((SNAPSHOT_NAMES[0], calls, 2048, 8*2**20),
                                          (SNAPSHOT_NAMES[1], vjps, 8192, 8*2**20)):
        pieces = []
        for row in rows:
            _plain(row)
            if type(row) is not dict:
                raise ProgressError('Actual native/VJP row dictionaries required.')
            data = encoded(row)
            if len(data) > row_cap:
                raise ProgressError('Original per-row byte ceiling.')
            pieces.append(data)
        data = b''.join(pieces)
        if len(data) > file_cap:
            raise ProgressError('Original per-file proof ceiling.')
        payloads[name] = data
    scope = {'scopes': scopes, 'refusals': refusals, 'borrowed': borrowed}
    _plain(scope)
    payloads[SNAPSHOT_NAMES[2]] = encoded(scope)
    if len(payloads[SNAPSHOT_NAMES[2]]) > 64*1024:
        raise ProgressError('Original scope byte ceiling.')
    header = {'kind': 'provisional-native-capture-v1', 'identity': journal.identity,
              'scientific_qualified': False, 'completion_or_restoration_admitted': False,
              'restoration_state': 'not_yet_checked', 'files': {
                  name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                  for name, data in payloads.items()}}
    payloads[SNAPSHOT_NAMES[3]] = encoded(header)
    if len(payloads[SNAPSHOT_NAMES[3]]) > 16*1024:
        raise ProgressError('Provisional header bound.')
    # Check the complete namespace before writing; never overwrite/repair a partial.
    for name in payloads:
        try:
            os.stat(name, dir_fd=journal.parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError('Preserve previous provisional capture: '+name)
    for name, data in payloads.items():
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=journal.parent_fd)
        try:
            remaining = memoryview(data)
            while remaining:
                n = os.write(fd, remaining)
                if n <= 0:
                    raise OSError('Incomplete provisional write.')
                remaining = remaining[n:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(journal.parent_fd)
    return header


def read_progress(path):
    """Read once; retain an incomplete tail, never repair or certify old results."""
    p = Path(path).absolute()
    _private_folder(p.parent)
    if p.name != FILE:
        raise ProgressError('Exact new progress filename required.')
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o600 or before.st_size > MAX_BYTES):
            raise ProgressError('Private single-link bounded progress file required.')
        data = os.read(fd, MAX_BYTES+1)
        after = os.fstat(fd)
        if _file_id(p.lstat()) != _file_id(after):
            raise ProgressError('Named progress file replaced during read.')
        if (len(data) != before.st_size or _file_id(before) != _file_id(after)
                or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
            raise ProgressError('Progress file changed during the read.')
    finally:
        os.close(fd)
    lines = data.splitlines(keepends=True)
    tail = b'' if not lines or lines[-1].endswith(b'\n') else lines.pop()
    records, head, identity, pending = [], ZERO, None, None
    begun, gradient_seen, elapsed = set(), False, -1
    fields = {'kind','sequence','identity','phase','event','elapsed_ns','previous_sha256',
              'detail','scientific_qualified','row_sha256'}
    def strict_pairs(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ProgressError('Duplicate JSON key refused.')
            out[key] = value
        return out
    for i, line in enumerate(lines):
        if len(line) > MAX_ROW or i >= MAX_ROWS:
            raise ProgressError('Progress row/count bound.')
        row = json.loads(line, object_pairs_hook=strict_pairs)
        _plain(row)
        if type(row) is not dict or set(row) != fields:
            raise ProgressError('Exact diagnostic row schema required.')
        checksum = row.pop('row_sha256')
        if (checksum != hashlib.sha256(encoded(row)).hexdigest() or row['previous_sha256'] != head
                or row['kind'] != FORMAT or type(row['sequence']) is not int or row['sequence'] != i
                or row['scientific_qualified'] is not False or type(row['elapsed_ns']) is not int
                or row['elapsed_ns'] < elapsed or type(row['detail']) is not dict):
            raise ProgressError('Committed diagnostic chain changed.')
        if identity is None:
            identity = row['identity']
            if (type(identity) is not dict or set(identity) != {'control_sha256','objective_index','pid','thread','profiled'}
                    or not isinstance(identity['control_sha256'],str)
                    or not re.fullmatch('[0-9a-f]{64}', identity['control_sha256'])
                    or type(identity['objective_index']) is not int or not 0 <= identity['objective_index'] < 8
                    or type(identity['profiled']) is not bool
                    or any(type(identity[k]) is not int or identity[k] <= 0 for k in ('pid','thread'))
                    or i != 0 or row['phase'] != 'observer' or row['event'] != 'opened' or row['detail']):
                raise ProgressError('Original typed opening record required.')
        elif row['identity'] != identity:
            raise ProgressError('Diagnostic identity changed.')
        phase, event, detail = row['phase'], row['event'], row['detail']
        if event == 'begin':
            if pending is not None or phase not in PHASES or phase in begun or detail:
                raise ProgressError('Diagnostic phase ordering changed.')
            begun.add(phase)
            pending = phase
        elif event in ('returned', 'raised'):
            if (pending != phase or (event == 'returned' and detail)
                    or (event == 'raised' and (set(detail) != {'error_type'}
                        or type(detail['error_type']) is not str or not 0 < len(detail['error_type']) <= 80))):
                raise ProgressError('Diagnostic phase ending changed.')
            pending = None
        elif (phase, event) == ('gradient_saved', 'observed'):
            if (gradient_seen or begun or detail.get('file') != 'input.npy'
                    or type(detail.get('bytes')) is not int or not 0 < detail['bytes'] <= 16*2**20
                    or type(detail.get('sha256')) is not str or not re.fullmatch('[0-9a-f]{64}',detail['sha256'])):
                raise ProgressError('One typed gradient descriptor before finalization only.')
            gradient_seen = True
        elif i != 0:
            raise ProgressError('Unknown diagnostic event.')
        records.append(dict(row, row_sha256=checksum))
        head, elapsed = checksum, row['elapsed_ns']
    return {'kind': 'post-gradient-progress-read-v1', 'identity': identity, 'rows': records,
            'unreturned_observed_phase': pending, 'incomplete_tail_bytes': len(tail),
            'last_committed_sha256': head, 'scientific_qualified': False,
            'exact_stall_cause_established': False, 'bytes': len(data)}
