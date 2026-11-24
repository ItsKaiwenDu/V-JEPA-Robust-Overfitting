"""Guarded timing metadata only; never authorizes science, OS work or a launch."""
import hashlib
import json
import math
import os
from pathlib import PurePosixPath
import stat
import sys
import threading
import time

MAX_BYTES = 512*1024
MAX_ROWS = 384
MAX_SPANS = 192
MAX_DEPTH = 16
NAMES = frozenset(('child_cache_dispatch', 'parent_prepare', 'worker_body',
    'model_preparation', 'prepared_unit', 'post_unit_guards', 'unit_setup',
    'evaluation', 'evaluation_checks', 'unit_publication', 'owned_context',
    'score', 'selection', 'rng_mode_restore', 'failure_checks',
    'worker_validation', 'worker_publication', 'proof', 'original-unit',
    'candidate-unit', 'candidate-replay'))


def enc(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def error_record(error):
    try:
        message = str(error)[:256]
    except BaseException:
        message = '<unavailable>'
    return {'type': type(error).__name__[:64], 'message': message}


def note(primary, extra):
    primary.add_note('Phase accounting also refused: '+type(extra).__name__[:64])


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class FileSink:
    """One new declared metadata file, anchored ancestors, held bytes and fsync."""
    def __init__(self, guard, name):
        descriptor = guard.layout.classify(name)
        if (descriptor['category'] != 'metadata' or not descriptor['mutable']
                or descriptor['max_bytes'] != MAX_BYTES):
            raise ValueError('Declared bounded append-only phase journal required.')
        self.guard, self.name = guard, name
        self.fd = self.parent = None
        self.parents = []
        self.bytes = 0
        self.digest = hashlib.sha256()
        self.closed = False
        try:
            self.parent = guard.open_root()
            parts = PurePosixPath(name).parts
            for part in parts[:-1]:
                before = os.stat(part, dir_fd=self.parent, follow_symlinks=False)
                if (not stat.S_ISDIR(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o700
                        or before.st_uid != os.getuid() or before.st_dev != guard.binding['root_identity'][0]):
                    raise ValueError('Private phase-journal ancestry required.')
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.parent)
                if _signature(os.fstat(child)) != _signature(before):
                    os.close(child)
                    raise ValueError('Phase ancestor changed while opening.')
                self.parents.append((self.parent, part, before))
                self.parent = child
            self.leaf = parts[-1]
            self.fd = os.open(self.leaf, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                              0o600, dir_fd=self.parent)
            self.observed = _signature(os.fstat(self.fd))
            os.fsync(self.fd)
            os.fsync(self.parent)
            self.check()
        except BaseException as primary:
            try:
                self.close()
            except BaseException as extra:
                note(primary, extra)
            raise

    def check(self):
        if self.closed or self.fd is None:
            raise ValueError('One held phase file required; no reuse.')
        self.guard.check_root()
        info = os.fstat(self.fd)
        named = os.stat(self.leaf, dir_fd=self.parent, follow_symlinks=False)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_size != self.bytes
                or _signature(info) != self.observed or _signature(named) != self.observed):
            raise ValueError('Held phase-file identity, extent or metadata changed.')
        for parent, part, before in self.parents:
            got = os.stat(part, dir_fd=parent, follow_symlinks=False)
            if (got.st_dev, got.st_ino, got.st_uid, stat.S_IMODE(got.st_mode)) != (
                    before.st_dev, before.st_ino, before.st_uid, stat.S_IMODE(before.st_mode)):
                raise ValueError('Named phase ancestor changed.')
        raw = os.pread(self.fd, self.bytes+1, 0)
        if len(raw) != self.bytes or hashlib.sha256(raw).digest() != self.digest.digest():
            raise ValueError('Owned phase journal bytes changed.')

    def append(self, raw):
        self.check()
        if type(raw) is not bytes or not raw or self.bytes+len(raw) > MAX_BYTES:
            raise ValueError('Phase journal byte ceiling exhausted.')
        remaining = memoryview(raw)
        while remaining:
            count = os.write(self.fd, remaining)
            if not 0 < count <= len(remaining):
                raise OSError('Incomplete phase row; retain partial journal.')
            remaining = remaining[count:]
        os.fsync(self.fd)
        self.bytes += len(raw)
        self.digest.update(raw)
        self.observed = _signature(os.fstat(self.fd))
        self.check()

    def receipt(self):
        self.check()
        return {'bytes': self.bytes, 'sha256': self.digest.hexdigest()}

    def close(self):
        if self.closed:
            return
        self.closed = True
        errors = []
        for fd in [self.fd, self.parent, *(row[0] for row in reversed(self.parents))]:
            if fd is not None:
                try:
                    os.close(fd)
                except BaseException as error:
                    errors.append(error)
        self.fd = self.parent = None
        if errors:
            raise errors[0]


class _Token:
    def __init__(self, ordinal, name, index, parent, started, retrospective, deadline):
        self.ordinal, self.name, self.index, self.parent = ordinal, name, index, parent
        self.started, self.retrospective = started, retrospective
        self.deadline = deadline


class PhaseLedger:
    """One process/thread and monotonic domain; intervals can nest, never sum them."""
    def __init__(self, sink, *, role, control_sha256, plan_sha256, deadline,
                 scope_started=None, clock=time.monotonic):
        if (role not in ('parent', 'original-unit', 'candidate-unit', 'candidate-replay')
                or not finite(deadline)
                or any(type(value) is not str or len(value) != 64
                       or any(c not in '0123456789abcdef' for c in value)
                       for value in (control_sha256, plan_sha256))):
            raise ValueError('Exact phase role/source-plan identity required.')
        self.sink, self.clock, self.deadline = sink, clock, deadline
        self.role, self.control, self.plan = role, control_sha256, plan_sha256
        self.owner = (os.getpid(), threading.get_ident())
        self.domain = 'process-monotonic/'+str(self.owner[0])
        self.last = None
        self.stack = []
        self.spans = self.rows = 0
        self.previous = '0'*64
        self.counts = {}
        self.poisoned = self.finalized = self.closed = False
        self.problem = None
        self.scope_started = scope_started
        try:
            now = self.moment()
            if scope_started is not None and (not finite(scope_started) or scope_started > now or scope_started < 0):
                raise ValueError('Same-child live entry sample required.')
        except BaseException as primary:
            try:
                sink.close()
            except BaseException as extra:
                note(primary, extra)
            self.closed = True
            raise

    def owner_check(self):
        if self.owner != (os.getpid(), threading.get_ident()) or self.closed:
            raise ValueError('One owned process/thread phase ledger required.')

    def moment(self, *, allow_expired=False):
        self.owner_check()
        now = self.clock()
        if not finite(now) or self.last is not None and now < self.last:
            raise ValueError('Continuing monotonic phase clock changed; no reset.')
        self.last = now
        if not allow_expired and now >= self.deadline:
            raise TimeoutError('Original phase/unit/work deadline exhausted by accounting.')
        return now

    def fail(self, error):
        self.poisoned = True
        if self.problem is None:
            self.problem = error_record(error)

    def record(self, event, token, at, **fields):
        if self.rows >= MAX_ROWS:
            raise ValueError('Bounded phase row count exhausted.')
        row = {'kind': 'whole-pgd-phase-event-v1', 'sequence': self.rows,
            'previous_sha256': self.previous, 'event': event, 'span': token.ordinal,
            'parent_span': token.parent, 'name': token.name, 'index': token.index,
            'start': token.started, 'at': at, 'retrospective_start': token.retrospective,
            'deadline': token.deadline,
            'role': self.role, 'pid': self.owner[0], 'clock_domain': self.domain,
            'control_sha256': self.control, 'plan_sha256': self.plan,
            'scientific_qualified': False, 'training_admitted': False,
            'performance_qualified': False, **fields}
        row['row_sha256'] = hashlib.sha256(enc(row)).hexdigest()
        raw = enc(row)
        if len(raw) > 4096:
            raise ValueError('Bounded phase event extent required.')
        self.sink.append(raw)
        self.rows += 1
        self.previous = row['row_sha256']

    def begin(self, name, *, index=None, retrospective=None, deadline=None):
        self.owner_check()
        if (self.poisoned or self.finalized or name not in NAMES
                or self.spans >= MAX_SPANS or len(self.stack) >= MAX_DEPTH
                or index is not None and (type(index) is not int or not 0 <= index < 128)):
            raise ValueError('One bounded declared phase start required.')
        now = self.moment()
        limit = self.deadline if deadline is None else deadline
        if not finite(limit) or not now < limit <= self.deadline:
            raise TimeoutError('An original unexpired absolute phase deadline is required.')
        started = now if retrospective is None else retrospective
        if (not finite(started) or not 0 <= started <= now
                or retrospective is not None and (self.stack or name not in ('child_cache_dispatch', 'parent_prepare'))):
            raise ValueError('Retrospective startup sample must be an unnested live interval.')
        token = _Token(self.spans, name, index,
                       self.stack[-1].ordinal if self.stack else None, started, retrospective is not None, limit)
        self.spans += 1
        self.stack.append(token)
        try:
            self.record('begin', token, now)
            if self.moment() >= limit:
                raise TimeoutError('Phase start publication exhausted its original deadline.')
        except BaseException as error:
            self.fail(error)
            raise
        return token

    def end(self, token, primary=None):
        try:
            self.owner_check()
            if not self.stack or self.stack[-1] is not token:
                raise ValueError('Exact LIFO phase token required; no replay or cross-owner end.')
            now = self.moment(allow_expired=True)
            expired = now >= token.deadline
            failure = primary
            if failure is None and expired:
                failure = TimeoutError('Original deadline reached before phase completion.')
            self.record('end', token, now, end=now, seconds=now-token.started,
                completed=failure is None, error=None if failure is None else error_record(failure),
                own_end_row_publication_included=False)
            self.stack.pop()
            if failure is not None:
                self.fail(failure)
            after = self.moment(allow_expired=True)
            if primary is None and (failure is not None or after >= token.deadline):
                raise failure if failure is not None else TimeoutError('Phase publication exhausted the original deadline.')
        except BaseException as error:
            self.fail(error)
            if primary is not None:
                note(primary, error)
                return
            raise
        if primary is None:
            self.counts[token.name] = self.counts.get(token.name, 0)+1

    def span(self, name, *, index=None, deadline=None):
        return _Span(self, name, index, deadline=deadline)

    def restoring(self, name='rng_mode_restore'):
        # Capture a pending work exception before trying any observation.
        return _Span(self, name, None, critical=True, pending=sys.exception())

    def context(self, create, *, index):
        token = self.begin('owned_context', index=index)
        try:
            original = create()
        except BaseException as primary:
            self.end(token, primary)
            raise
        return _Context(self, token, original)

    def adopt_startup(self, name, started, *, deadline=None):
        token = self.begin(name, retrospective=started, deadline=deadline)
        self.end(token)

    def finalize(self, publish, name, *, primary=None, require_counts=False):
        if self.finalized:
            if primary is not None:
                primary.add_note('Phase finalization was already attempted; no overwrite/retry.')
                return
            raise ValueError('One immutable phase summary attempt required.')
        self.finalized = True
        error = None
        try:
            self.owner_check()
            end = self.moment(allow_expired=primary is not None)
            complete = primary is None and not self.poisoned and not self.stack
            if require_counts and complete and any(self.counts.get(k) != n for k, n in (
                    ('owned_context', 40), ('selection', 42), ('score', 6))):
                raise ValueError('Exact primary full-context/selection/no-grad scoring phase counts required.')
            if primary is not None:
                self.fail(primary)
            receipt = self.sink.receipt()
            row = {'kind': 'whole-pgd-phase-summary-v1', 'role': self.role,
                'pid': self.owner[0], 'clock_domain': self.domain, 'control_sha256': self.control,
                'plan_sha256': self.plan, 'scope_started': self.scope_started, 'scope_ended': end,
                'deadline': self.deadline,
                'scope_seconds': None if self.scope_started is None else end-self.scope_started,
                'journal': receipt, 'journal_rows': self.rows, 'final_row_sha256': self.previous,
                'completed_spans': dict(self.counts), 'pending_spans': [t.ordinal for t in self.stack],
                'measurements_complete': complete, 'problem': self.problem,
                'overlapping_nested_intervals_must_not_be_summed': True,
                'scope_excludes_own_summary_publication_close_and_return': True,
                'intervals_exclude_own_end_row_publication_and_return': True,
                'overhead_subtracted': False, 'observation_authenticates_execution': False,
                'scientific_qualified': False, 'training_admitted': False, 'performance_qualified': False}
            publish(name, row)
            if not complete and primary is None:
                raise ValueError('Incomplete/poisoned phase accounting cannot complete a unit.')
        except BaseException as extra:
            error = extra
        finally:
            try:
                self.sink.close()
            except BaseException as extra:
                if error is None:
                    error = extra
            self.closed = True
        if error is not None:
            self.fail(error)
            if primary is not None:
                note(primary, error)
            else:
                raise error


class _Span:
    def __init__(self, ledger, name, index, *, critical=False, pending=None, deadline=None):
        self.ledger, self.name, self.index = ledger, name, index
        self.critical, self.pending = critical, pending
        self.deadline = deadline
        self.token = self.observation_error = None

    def __enter__(self):
        try:
            self.token = self.ledger.begin(self.name, index=self.index, deadline=self.deadline)
        except BaseException as error:
            if not self.critical:
                raise
            # Restoration is mandatory even if its start observation failed.
            self.observation_error = error
            self.ledger.fail(error)
        return self

    def __exit__(self, kind, error, tb):
        primary = error if error is not None else self.pending
        if self.token is not None:
            self.ledger.end(self.token, primary)
        if self.observation_error is not None:
            if primary is not None:
                note(primary, self.observation_error)
            else:
                raise self.observation_error
        return False


class _Context:
    def __init__(self, ledger, token, original):
        self.ledger, self.token, self.original = ledger, token, original

    def __enter__(self):
        try:
            return self.original.__enter__()  # Return the exact original OwnedStep.
        except BaseException as primary:
            self.ledger.end(self.token, primary)
            raise

    def __exit__(self, kind, error, tb):
        try:
            result = self.original.__exit__(kind, error, tb)  # Always before observation.
        except BaseException as primary:
            self.ledger.end(self.token, primary)
            raise
        self.ledger.end(self.token, None if result else error)
        return result


def require_ledger(value, args):
    if (type(value) is not PhaseLedger or value.poisoned or value.finalized
            or value.role != args.role or value.control != args.control_sha256
            or value.plan != args.plan_sha256 or value.deadline != args.deadline):
        raise ValueError('A live owned source-bound phase object is required; metadata cannot bind it.')
    value.owner_check()
    value.moment()
    return value


def validate_saved(raw, summary, *, role, pid, control_sha256, plan_sha256, deadline):
    """Metadata/byte joins only, never execution or performance authentication."""
    if type(raw) is not bytes or len(raw) > MAX_BYTES:
        raise ValueError('Bounded saved phase journal required.')
    lines = raw.splitlines(keepends=True)
    if not 0 < len(lines) <= MAX_ROWS:
        raise ValueError('Bounded nonempty saved phase rows required.')
    previous = '0'*64
    stack = []
    spans = {}
    counts = {}
    last = None
    earliest = None
    for sequence, line in enumerate(lines):
        if len(line) > 4096:
            raise ValueError('Saved phase row exceeds bound.')
        row = json.loads(line)
        row_sha = row.pop('row_sha256')
        if enc(dict(row, row_sha256=row_sha)) != line or hashlib.sha256(enc(row)).hexdigest() != row_sha:
            raise ValueError('Canonical exact phase row hash required.')
        if (row['kind'] != 'whole-pgd-phase-event-v1'
                or row['sequence'] != sequence or row['previous_sha256'] != previous or row['role'] != role
                or row['pid'] != pid or row['clock_domain'] != 'process-monotonic/'+str(pid)
                or row['control_sha256'] != control_sha256 or row['plan_sha256'] != plan_sha256
                or row['name'] not in NAMES or not finite(row['at'])
                or not finite(row['deadline']) or row['deadline'] > deadline
                or last is not None and row['at'] < last
                or any(row[k] is not False for k in ('scientific_qualified', 'training_admitted', 'performance_qualified'))):
            raise ValueError('Exact continuing phase owner/source/clock identity required.')
        last = row['at']
        if row['event'] == 'begin':
            if (row['span'] in spans or row['parent_span'] != (stack[-1] if stack else None)
                    or not finite(row['start']) or not 0 <= row['start'] <= row['at'] < row['deadline']):
                raise ValueError('Unique nested phase start required.')
            if (type(row['span']) is not int or not 0 <= row['span'] < MAX_SPANS
                    or type(row['retrospective_start']) is not bool
                    or row['retrospective_start'] and (stack or row['name'] not in ('child_cache_dispatch', 'parent_prepare'))
                    or not row['retrospective_start'] and row['start'] != row['at']):
                raise ValueError('Typed live or explicit retrospective startup phase required.')
            earliest = row['start'] if earliest is None else min(earliest, row['start'])
            spans[row['span']] = row
            stack.append(row['span'])
        elif row['event'] == 'end':
            if not stack or stack.pop() != row['span']:
                raise ValueError('Exact nested phase end required.')
            begin = spans[row['span']]
            if (any(row[k] != begin[k] for k in ('start', 'name', 'index', 'parent_span', 'retrospective_start', 'deadline'))
                    or row['end'] != row['at'] or row['seconds'] != row['end']-row['start']
                    or not finite(row['seconds']) or row['seconds'] < 0
                    or row['completed'] is not True or row['error'] is not None
                    or row['end'] >= row['deadline'] or row['own_end_row_publication_included'] is not False):
                raise ValueError('Matched same-clock interval required.')
            if row['completed']:
                counts[row['name']] = counts.get(row['name'], 0)+1
        else:
            raise ValueError('Closed phase event kinds required.')
        previous = row_sha
    if (summary['kind'] != 'whole-pgd-phase-summary-v1' or summary['role'] != role or summary['pid'] != pid
            or summary['control_sha256'] != control_sha256 or summary['plan_sha256'] != plan_sha256
            or summary['clock_domain'] != 'process-monotonic/'+str(pid)
            or summary['overlapping_nested_intervals_must_not_be_summed'] is not True
            or summary['scope_excludes_own_summary_publication_close_and_return'] is not True
            or summary['intervals_exclude_own_end_row_publication_and_return'] is not True
            or summary['journal'] != {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            or summary['journal_rows'] != len(lines) or summary['final_row_sha256'] != previous
            or summary['completed_spans'] != counts or summary['pending_spans'] != stack
            or summary['measurements_complete'] is not True or stack or summary['problem'] is not None
            or any(summary[k] is not False for k in ('scientific_qualified', 'training_admitted',
                    'performance_qualified', 'observation_authenticates_execution', 'overhead_subtracted'))):
        raise ValueError('Complete phase metadata and exact journal/summary joins required.')
    if (not finite(summary['scope_started']) or not finite(summary['scope_ended'])
            or not 0 <= summary['scope_started'] <= summary['scope_ended'] < deadline
            or earliest is None or summary['scope_started'] > earliest
            or summary['scope_ended'] < last or summary['deadline'] != deadline
            or summary['scope_seconds'] != summary['scope_ended']-summary['scope_started']):
        raise ValueError('Exact same-process envelope and original absolute deadline required.')
    return {'phase_metadata_joined': True, 'performance_or_execution_authenticated': False}
