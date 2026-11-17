"""Inert stdlib definitions for a two-file publication probe, never a model job.

The builder appends the exact frozen/candidate Store classes and private helpers.
Only the main guard executes remote work. The inventory is deliberately a new,
closed one-file byte namespace, not the scientific result catalogue.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import sys
import threading
import time
import types

MiB = 2**20
FILE_SIZE = 12 * MiB + 128
DATA_SHA = 'bfcd6fd60e8da4bae4eef2eac7655a718753b77226a10cf8d8a157b691fcb658'
KEY = 'results/numerical/objective-0/pixels.npy'
EXPECTED_HOST = {'hostname': 'linux10609', 'system': 'Linux', 'python': '3.11.17'}
FLAGS = {'byte_only': True, 'read_only': False, 'model_work_authorized': False,
         'scientific_qualified': False, 'training_admitted': False}
FIELDS = ('bytes', 'allocated_bytes', 'identity', 'mtime_ns')
LIMITS = {'work_seconds': 60, 'finalization_seconds': 5, 'remote_timeout_seconds': 65,
          'kill_grace_seconds': 2, 'client_seconds_including_login': 180,
          'logical_bytes': 32*MiB, 'allocated_bytes': 64*MiB, 'metadata_bytes': 65536,
          'response_bytes': 65536, 'SSH_stderr_bytes': 16384, 'files': 5, 'directories': 9,
          'connections': 1, 'automatic_retries': 0, 'data_file_bytes': FILE_SIZE}


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def is_sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def row(s):
    return {'bytes': s.st_size, 'allocated_bytes': s.st_blocks*512,
            'identity': identity(s), 'mtime_ns': s.st_mtime_ns}


def delta(want, got):
    return sorted(k for k in FIELDS if got is None or want is None or encoded(want[k]) != encoded(got[k]))


def problem(error):
    return {'type': type(error).__name__, 'message': str(error)[:512],
            'notes': [str(v)[:2200] for v in getattr(error, '__notes__', ())[:2]]}


def preflight():
    host = {'hostname': platform.node().split('.')[0], 'system': platform.system(), 'python': platform.python_version()}
    if host != EXPECTED_HOST:
        raise ValueError('Exact linux10609 Linux/Python runtime required before writes.')
    parent = Path('/var/tmp')
    s = parent.lstat()
    home = Path.home().stat()
    if (parent.resolve(strict=True) != parent or not stat.S_ISDIR(s.st_mode)
            or s.st_uid != 0 or stat.S_IMODE(s.st_mode) != 0o1777 or s.st_dev != 66307
            or home.st_dev == s.st_dev):
        raise ValueError('Observed separate local private-scratch parent required.')
    with open('/proc/self/mountinfo', 'rb') as f:
        raw = f.read(65537)
    if len(raw) > 65536:
        raise ValueError('Mount metadata exceeds bound.')
    mounts = []
    for line in raw.decode('ascii').splitlines():
        left, right = line.split(' - ', 1)
        parts, fs = left.split(), right.split()
        if parts[4] == '/' and parts[2] == str(os.major(s.st_dev))+':'+str(os.minor(s.st_dev)):
            mounts.append((fs[0], 'rw' in parts[5].split(',')))
    if mounts != [('xfs', True)]:
        raise ValueError('Expected writable local XFS root mount required.')
    free = os.statvfs(parent)
    if free.f_bavail*free.f_frsize < 64*MiB or free.f_favail < 64:
        raise ValueError('Small byte-probe free-space/inode floor unavailable.')
    return {'host': host, 'uid': os.getuid(), 'device': s.st_dev, 'filesystem': 'xfs',
            'parent_identity': identity(s)[:4], 'free_bytes_observed': free.f_bavail*free.f_frsize,
            'free_inodes_observed': free.f_favail}, parent


class Budget:
    """App checks only; no quota, allocation reservation, or atomic cap claim."""
    def __init__(self, root, deadline, clock):
        self.root, self.deadline, self.clock = root, deadline, clock
        self.root_id = identity(private(root, directory=True))[:4]
        self.device = root.lstat().st_dev
        self.dirs = {'.'}
        self.files = {'RESULT.json'}
        for case in ('frozen', 'candidate'):
            p = Path(case)/KEY
            self.files.update((str(p), str(p)+'.partial'))
            self.dirs.update(str(v) for v in p.parents)

    def check(self):
        if self.clock() >= self.deadline:
            raise TimeoutError('Original byte-probe deadline reached.')
        if identity(private(self.root, directory=True))[:4] != self.root_id:
            raise ValueError('Byte-probe root identity changed.')
        logical = allocated = count = directories = 0
        for current, dirs, names in os.walk(self.root, followlinks=False):
            here = Path(current)
            rel = str(here.relative_to(self.root))
            if rel not in self.dirs:
                raise ValueError('Unknown byte-probe directory.')
            before = private(here, directory=True, device=self.device)
            directories += 1
            allocated += before.st_blocks*512
            for name in dirs:
                private(here/name, directory=True, device=self.device)
            for name in names:
                p = here/name
                key = str(p.relative_to(self.root))
                if key not in self.files:
                    raise ValueError('Unknown byte-probe file.')
                s = private(p, device=self.device)
                if not 0 <= s.st_size <= (65536 if key == 'RESULT.json' else FILE_SIZE):
                    raise ValueError('Byte-probe file extent exceeded.')
                logical += s.st_size
                allocated += s.st_blocks*512
                count += 1
            if identity(private(here, directory=True, device=self.device))[:4] != identity(before)[:4]:
                raise ValueError('Byte-probe directory changed during scan.')
        if (logical > LIMITS['logical_bytes'] or allocated > LIMITS['allocated_bytes']
                or count > LIMITS['files'] or directories > LIMITS['directories']):
            raise ValueError('Byte-probe namespace/logical/allocation bound exceeded.')
        if identity(private(self.root, directory=True))[:4] != self.root_id:
            raise ValueError('Byte-probe root changed during scan.')
        return {'logical_bytes': logical, 'allocated_bytes': allocated, 'files': count, 'directories': directories}


class BytePlan:
    GiB = 2**30

    @staticmethod
    def classify(name, spec):
        plain = name[:-8] if name.endswith('.partial') else name
        if plain != KEY or plain not in spec:
            raise ValueError('Only the one inert byte publication is allowed.')
        return spec[plain], plain, name != plain

    @staticmethod
    def result_totals(files, spec):
        for name, size in files.items():
            BytePlan.classify(name, spec)
            if type(size) is not int or not 0 <= size <= FILE_SIZE:
                raise ValueError('Inert byte extent exceeded.')
        if len(files) > 2 or sum(files.values()) > 2*FILE_SIZE:
            raise ValueError('One byte publication plus its original partial only.')
        return {'byte_probe': sum(files.values())}


plan = BytePlan


class OneFileInventory:
    def __init__(self, root, budget):
        self.root, self.budget = root, budget
        self.device = budget.device
        self.root_id = identity(private(root, directory=True, device=self.device))[:4]
        self.spec = {KEY: {'max_bytes': FILE_SIZE, 'mutable': False, 'partial_allowed': True}}
        self.dirs = {'.', 'results', 'results/numerical', 'results/numerical/objective-0'}

    def scan(self):
        self.budget.check()
        if identity(private(self.root, directory=True, device=self.device))[:4] != self.root_id:
            raise ValueError('One-file inventory root changed.')
        files = {}
        allocated = 0
        for current, dirs, names in os.walk(self.root, followlinks=False):
            here = Path(current)
            if str(here.relative_to(self.root)) not in self.dirs:
                raise ValueError('Unknown one-file directory.')
            before = private(here, directory=True, device=self.device)
            allocated += before.st_blocks*512
            for name in dirs:
                private(here/name, directory=True, device=self.device)
            for name in names:
                p = here/name
                key = str(p.relative_to(self.root))
                BytePlan.classify(key, self.spec)
                s = private(p, device=self.device)
                if not 0 <= s.st_size <= FILE_SIZE:
                    raise ValueError('One-file extent exceeded.')
                files[key] = row(s)
                allocated += s.st_blocks*512
            if identity(private(here, directory=True, device=self.device))[:4] != identity(before)[:4]:
                raise ValueError('One-file directory changed during scan.')
        totals = BytePlan.result_totals({k: v['bytes'] for k, v in files.items()}, self.spec)
        if identity(private(self.root, directory=True, device=self.device))[:4] != self.root_id:
            raise ValueError('One-file root changed during scan.')
        return {'files': files, 'totals': totals, 'allocated_bytes': allocated, 'scientific_qualified': False}


class RecordingOS:
    """Observe real metadata around this writer's owned close, never substitute it."""
    def __init__(self, store):
        self.store, self.owned, self.observations = store, set(), []

    def __getattr__(self, name):
        return getattr(os, name)

    def open(self, *args, **kwargs):
        fd = os.open(*args, **kwargs)
        try:
            if args[1] & os.O_WRONLY and stat.S_ISREG(os.fstat(fd).st_mode):
                self.owned.add(fd)
        except BaseException:
            os.close(fd)  # An unsuccessful observer setup still owns this fd.
            raise
        return fd

    def close(self, fd):
        if fd not in self.owned:
            return os.close(fd)
        self.owned.remove(fd)  # Descriptor is never closed twice, including errors.
        observation = {'before_held': None, 'before_named': None, 'after_named': None,
                       'committed_at_close': self.store.written.get(KEY), 'observation_errors': [], 'close_error': None}
        try:
            observation['before_held'] = row(os.fstat(fd))
            observation['before_named'] = row((self.store.root/KEY).lstat())
        except Exception as error:
            observation['observation_errors'].append(problem(error))
        try:
            os.close(fd)
        except BaseException as error:
            observation['close_error'] = problem(error)
            raise
        finally:
            try:
                observation['after_named'] = row((self.store.root/KEY).lstat())
            except Exception as error:
                observation['observation_errors'].append(problem(error))
            self.observations.append(observation)


def bind_observer(store):
    proxy = RecordingOS(store)
    original = store.write_bytes.__func__
    namespace = dict(original.__globals__, os=proxy)
    function = types.FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    store.write_bytes = types.MethodType(function, store)
    return proxy


def readback(path, budget):
    budget.check()
    before = row(private(path, device=budget.device))
    if before['bytes'] != FILE_SIZE:
        raise ValueError('Complete inert byte file required.')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if delta(before, row(os.fstat(fd))):
            raise ValueError('Read-back file changed on open.')
        sha, count = hashlib.sha256(), 0
        while True:
            budget.check()
            chunk = os.read(fd, min(MiB, FILE_SIZE-count+1))
            if not chunk:
                break
            count += len(chunk)
            if count > FILE_SIZE:
                raise ValueError('Read-back extent exceeded.')
            sha.update(chunk)
        after = row(os.fstat(fd))
    finally:
        os.close(fd)
    named = row(private(path, device=budget.device))
    if delta(before, after) or delta(after, named) or count != FILE_SIZE or sha.hexdigest() != DATA_SHA:
        raise ValueError('Inert byte read-back hash or metadata changed.')
    budget.check()
    return {'bytes': count, 'sha256': sha.hexdigest(), 'before': before, 'after': named}


def classify_case(case):
    """Recomputable only from retained byte/metadata facts; never scientific admission."""
    try:
        if case['write_error'] is not None or case['receipt'] != {'file': 'pixels.npy', 'bytes': FILE_SIZE, 'sha256': DATA_SHA}:
            return 'inconclusive'
        if case['read_error'] is not None or case['readback']['bytes'] != FILE_SIZE or case['readback']['sha256'] != DATA_SHA:
            return 'inconclusive'
        observations = case['close_observations']
        if len(observations) != 1:
            return 'inconclusive'
        o = observations[0]
        if o['observation_errors'] or o['close_error'] is not None or delta(o['before_held'], o['before_named']):
            return 'inconclusive'
        if any(v != 'allocated_bytes' for v in delta(o['before_named'], o['after_named'])):
            return 'inconclusive'
        expected, before, after = case['committed'], case['before_read'], case['readback']['after']
        if any(v != 'allocated_bytes' for v in delta(expected, before)) or delta(before, after):
            return 'inconclusive'
        changed = delta(expected, before)
        if case['kind'] == 'candidate' and (o['committed_at_close'] is not None or delta(o['after_named'], expected)):
            return 'inconclusive'
        if case['kind'] == 'frozen' and delta(o['committed_at_close'], expected):
            return 'inconclusive'
        errors = (case['check_before_error'], case['check_after_error'])
        if not changed and all(v is None for v in errors):
            return 'consistent'
        exact = 'Previously committed result metadata changed: '+KEY
        if (case['kind'] == 'frozen' and changed == ['allocated_bytes']
                and delta(o['before_named'], expected) == [] and delta(o['after_named'], before) == []
                and all(type(v) is dict and v['type'] == 'ValueError' and v['message'] == exact for v in errors)):
            return 'allocation_only_refusal'
    except (KeyError, TypeError, ValueError):
        pass
    return 'inconclusive'


def one_case(kind, cls, root, data, budget):
    budget.check()
    case_root = root/kind
    case_root.mkdir(mode=0o700)
    store = cls(OneFileInventory(case_root, budget))
    proxy = bind_observer(store)
    result = {'kind': kind, 'receipt': None, 'write_error': None, 'committed': None,
              'close_observations': proxy.observations, 'before_read': None, 'check_before_error': None,
              'readback': None, 'read_error': None, 'check_after_error': None}
    try:
        result['receipt'] = store.write_bytes(KEY, data)
        result['committed'] = store.written.get(KEY)
        result['before_read'] = row(private(case_root/KEY, device=budget.device))
    except Exception as error:
        result['write_error'] = problem(error)
    if result['write_error'] is None:
        try:
            store.check()
        except Exception as error:
            result['check_before_error'] = problem(error)
        try:
            result['readback'] = readback(case_root/KEY, budget)
        except Exception as error:
            result['read_error'] = problem(error)
        try:
            store.check()
        except Exception as error:
            result['check_after_error'] = problem(error)
    result['classification'] = classify_case(result)
    return result


def assessment(cases):
    if len(cases) == 2 and [v['kind'] for v in cases] == ['frozen', 'candidate']:
        kinds = [classify_case(v) for v in cases]
        if kinds[1] == 'consistent' and kinds[0] in ('consistent', 'allocation_only_refusal'):
            return 'allocation_transition_observed' if kinds[0] == 'allocation_only_refusal' else 'failure_not_reproduced'
    return 'inconclusive'


def run_probe(nonce, csha, psha, *, clock=None, inspect=None):
    if not all(is_sha(v) for v in (nonce, csha, psha)):
        raise ValueError('Three exact public bindings required.')
    clock = time.monotonic if clock is None else clock
    inspect = preflight if inspect is None else inspect
    started = clock()
    result = {'kind': 'pgd-byte-close-observation-v1', 'request_nonce': nonce, 'control_sha256': csha,
              'program_sha256': psha, 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'preflight': None, 'root': None, 'cases': [], 'footprint': None, 'error': None,
              'assessment': 'inconclusive', 'elapsed_seconds': None, **FLAGS}
    root = budget = None
    try:
        observed, parent = inspect()
        result['preflight'] = observed
        root = parent/('vjepa-byte-close-v1-'+str(os.getuid())+'-'+csha)
        if clock() >= started+60:
            raise TimeoutError('Preflight consumed the work window.')
        # One exclusive root; never reopen any prior or partially completed attempt.
        root.mkdir(mode=0o700)
        result['root'] = str(root)
        if identity(parent.lstat())[:4] != tuple(observed['parent_identity']):
            raise ValueError('Scratch parent changed before exclusive root creation.')
        budget = Budget(root, started+60, clock)
        data = b'\xa5'*FILE_SIZE
        if digest(data) != DATA_SHA:
            raise ValueError('Fixed inert bytes differ.')
        for kind, cls in (('frozen', FrozenStore), ('candidate', CandidateStore)):
            case = one_case(kind, cls, root, data, budget)
            result['cases'].append(case)
            if case['classification'] not in ('consistent', 'allocation_only_refusal'):
                break
        result['footprint'] = budget.check()
        result['assessment'] = assessment(result['cases'])
    except Exception as error:
        result['error'] = problem(error)
    # A separate five-second finalization allowance does not reset the work clock.
    if budget is not None:
        budget.deadline = started+65
        try:
            budget.check()
            result['elapsed_seconds'] = clock()-started
            raw = encoded(result)
            if len(raw) > 65536:
                raise ValueError('Metadata response bound exceeded.')
            fd = os.open(root/'RESULT.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                view = memoryview(raw)
                while view:
                    n = os.write(fd, view)
                    if n <= 0:
                        raise OSError('Incomplete byte-probe status write.')
                    view = view[n:]
                os.fsync(fd)
            finally:
                os.close(fd)
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            result['footprint'] = budget.check()
        except Exception as error:
            result['assessment'] = 'inconclusive'
            result['error'] = result['error'] or problem(error)
    result['elapsed_seconds'] = clock()-started
    if result['elapsed_seconds'] >= 65:
        result['assessment'] = 'inconclusive'
        result['error'] = result['error'] or problem(TimeoutError('Finalization deadline reached.'))
    if len(encoded(result)) > 65536:
        raise ValueError('Response bound exceeded; preserve own scratch files.')
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        raise SystemExit('Three public binding hashes required.')
    result = run_probe(*argv)
    sys.stdout.buffer.write(encoded(result))
    sys.stdout.buffer.flush()
    return 0 if result['error'] is None and result['assessment'] != 'inconclusive' else 1
