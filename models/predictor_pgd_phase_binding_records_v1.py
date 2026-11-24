"""Read-only joins of saved phase metadata; never execution or budget admission."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

from models import predictor_pgd_phase_accounting_v1 as phases


def require(value, message):
    if not value:
        raise ValueError(message)


def signature(info):
    return (info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class SavedTree:
    """Held/named private ancestors and bounded files joined to transport bytes."""
    def __init__(self, destination, records, window):
        self.path = Path(destination)
        self.records, self.window = records, window
        self.parent = self.root = None
        try:
            require(self.path.is_absolute() and self.path.parent.resolve(strict=True) == self.path.parent,
                    'Canonical saved phase destination required.')
            self.parent = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self.parent_id = signature(os.fstat(self.parent))[:5]
            self.root = os.open(self.path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.parent)
            self.root_id = signature(os.fstat(self.root))[:5]
            self.device = self.root_id[0]
            for fd in (self.parent, self.root):
                info = os.fstat(fd)
                require(stat.S_IMODE(info.st_mode) == 0o700 and info.st_uid == os.getuid()
                        and info.st_dev == self.device, 'Private same-device saved phase ancestry required.')
            self.check()
        except BaseException:
            self.close()
            raise

    def check(self):
        self.window.check()
        require(signature(os.fstat(self.parent))[:5] == self.parent_id
                and signature(self.path.parent.lstat())[:5] == self.parent_id
                and self.path.parent.resolve(strict=True) == self.path.parent
                and signature(os.fstat(self.root))[:5] == self.root_id
                and signature(os.stat(self.path.name, dir_fd=self.parent, follow_symlinks=False))[:5] == self.root_id,
                'Named saved phase root or parent changed.')

    def read(self, name, cap):
        self.check()
        path = PurePosixPath(name)
        item = self.records.get(name)
        require(not path.is_absolute() and path.as_posix() == name and '..' not in path.parts
                and item is not None and type(item['bytes']) is int and 0 < item['bytes'] <= cap,
                'Required bounded saved phase member missing.')
        held = [os.dup(self.root)]
        ancestors = []
        datafd = None
        try:
            for part in path.parts[:-1]:
                parent = held[-1]
                before = os.stat(part, dir_fd=parent, follow_symlinks=False)
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                held.append(child)
                require(stat.S_IMODE(before.st_mode) == 0o700 and before.st_uid == os.getuid()
                        and before.st_dev == self.device and signature(os.fstat(child)) == signature(before),
                        'Saved phase ancestor changed.')
                ancestors.append((parent, part, before))
            before = os.stat(path.name, dir_fd=held[-1], follow_symlinks=False)
            require(stat.S_ISREG(before.st_mode) and stat.S_IMODE(before.st_mode) == 0o600
                    and before.st_uid == os.getuid() and before.st_dev == self.device and before.st_nlink == 1
                    and before.st_size == item['bytes'] and before.st_mtime_ns == item['mtime_ns'],
                    'Private exact saved phase member required.')
            datafd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=held[-1])
            require(signature(os.fstat(datafd)) == signature(before), 'Held saved phase member changed.')
            chunks, left = [], item['bytes']
            while left:
                self.window.check()
                block = os.read(datafd, min(65536, left))
                require(bool(block), 'Truncated saved phase metadata.')
                chunks.append(block)
                left -= len(block)
            raw = b''.join(chunks)
            require(not os.read(datafd, 1) and hashlib.sha256(raw).hexdigest() == item['sha256']
                    and signature(os.fstat(datafd)) == signature(before)
                    and signature(os.stat(path.name, dir_fd=held[-1], follow_symlinks=False)) == signature(before),
                    'Saved phase bytes or held/named identity changed.')
            for parent, part, info in ancestors:
                require(signature(os.stat(part, dir_fd=parent, follow_symlinks=False)) == signature(info),
                        'Named saved phase ancestor changed.')
            self.check()
            return raw
        finally:
            if datafd is not None:
                os.close(datafd)
            for fd in reversed(held):
                os.close(fd)

    def json(self, name, cap=65536):
        value = json.loads(self.read(name, cap))
        require(type(value) is dict, 'Saved phase metadata object required.')
        return value

    def close(self):
        for name in ('root', 'parent'):
            fd = getattr(self, name, None)
            if fd is not None:
                setattr(self, name, None)
                os.close(fd)


def validate_collection(destination, receipt, control_bytes, *, window):
    """Check completed roles only; failed/incomplete records are preserved as-is."""
    csha = hashlib.sha256(control_bytes).hexdigest()
    require(receipt['control_sha256'] == csha and receipt['transport_verified'] is True,
            'Original verified transport/source identity required before phase joins.')
    tree = SavedTree(destination, receipt['files'], window)
    try:
        session = tree.json('session.json')
        roles = ('proof', 'original-unit', 'candidate-unit', 'candidate-replay')
        completed = session.get('completed')
        require(session.get('kind') == 'whole-pgd-parent-session-v1' and session.get('control_sha256') == csha
                and session.get('phase') in ('failed', 'await-collection') and session.get('complete') is False
                and type(completed) is list and completed == list(roles[:len(completed)])
                and len(completed) <= 4 and len(session.get('receipts', [])) == len(completed)
                and all(session.get(k) is False for k in ('scientific_qualified','training_admitted')),
                'Saved phase session/prefix must retain incomplete scientific status.')
        passed = session['phase'] == 'await-collection'
        require(not passed or completed == list(roles), 'All original roles required for ended-work phase accounting.')
        parent_raw = None
        if passed or completed[1:]:
            parent_raw = tree.read('phase-parent-events.jsonl', phases.MAX_BYTES)
        if passed:
            parent = tree.json('phase-parent-summary.json')
            phases.validate_saved(parent_raw, parent, role='parent', pid=session['parent_pid'],
                control_sha256=csha, plan_sha256=csha, deadline=session['work_deadline'])
            require(all(parent['completed_spans'].get(name) == 1 for name in ('parent_prepare', *roles)),
                    'Complete parent phase counts required.')
        starts = {} if parent_raw is None else {row['name']: row for row in
            (json.loads(line) for line in parent_raw.splitlines()) if row['event'] == 'begin' and row['name'] in roles}
        for role, owned in zip(completed, session['receipts']):
            require(owned.get('role') == role and owned.get('saved_gates_returned') is True
                    and all(owned.get(k) is False for k in ('scientific_qualified','training_admitted')),
                    'Original saved role receipt required; no partial promotion.')
            if role == 'proof':
                continue
            prefix = 'capture/results/workload/'+role+'/'
            summary = tree.json(prefix+'phase-summary.json')
            marker = tree.json(prefix+'complete.json')
            work = tree.json(prefix+'unit-work.json', 2*1024*1024)
            observed = tree.json('parent-'+role+'.json')
            require(role in starts and starts[role]['clock_domain'] == 'process-monotonic/'+str(session['parent_pid'])
                    and starts[role]['control_sha256'] == csha
                    and marker.get('complete') is True and observed.get('exit_status') == 0
                    and observed.get('parent_observation_passed') is True
                    and observed.get('pid') == owned['pid'] and observed.get('start_ticks') == owned['start_ticks']
                    and marker.get('work') == {k:receipt['files'][prefix+'unit-work.json'][k] for k in ('bytes','sha256')}
                    and all(v.get('role') == role and v.get('pid') == owned['pid'] and v.get('control_sha256') == csha
                            and v.get('plan_sha256') == marker.get('plan_sha256') for v in (summary, work, marker)),
                    'Saved phase/plan/owned-parent/completion joins changed.')
            phases.validate_saved(tree.read(prefix+'phase-events.jsonl', phases.MAX_BYTES), summary,
                role=role, pid=owned['pid'], control_sha256=csha,
                plan_sha256=marker['plan_sha256'], deadline=starts[role]['deadline'])
            require(all(summary['completed_spans'].get(name) == count for name,count in
                        (('owned_context',40),('selection',42),('score',6))), 'Complete primary phase counts required.')
        if passed:
            outer = tree.json('phase-outer-setup.json', 16384)
            require(outer.get('kind') == 'whole-pgd-outer-setup-interval-v1'
                    and outer.get('pid') == session['parent_pid'] and outer.get('control_sha256') == csha
                    and outer.get('clock_domain') == 'process-monotonic/'+str(session['parent_pid'])
                    and all(phases.finite(outer.get(k)) for k in ('started','ended','seconds'))
                    and session['connection_started'] <= outer['started'] <= outer['ended'] <= session['work_deadline']
                    and outer['seconds'] == outer['ended']-outer['started']
                    and all(outer.get(k) is False for k in ('scientific_qualified','training_admitted',
                        'performance_qualified','observation_authenticates_execution')),
                    'Saved outer setup must retain its same-parent caller boundary.')
        tree.check()
        return {'kind':'whole-pgd-local-phase-byte-joins-v1','completed_worker_roles':completed[1:],
            'parent_measurements_complete':passed,'failed_incomplete_records_preserved':True,
            'metadata_authenticates_execution':False,'performance_qualified':False,
            'scientific_qualified':False,'training_admitted':False}
    finally:
        tree.close()
