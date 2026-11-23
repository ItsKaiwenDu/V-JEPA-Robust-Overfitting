"""Concrete owned whole-unit child support; no launch or timer runs on import."""
from argparse import Namespace
import _thread
import math
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import threading
import time

# Exact Alarms/Deadline/Stopped definitions are appended by the cold source setup.


class UnitTimers:
    """Take over only the authenticated release timer; restore absolute unit time."""
    def __init__(self, deadline, *, clock=time.monotonic):
        if (os.getpid() == os.getppid() or threading.current_thread() is not threading.main_thread()
                or type(deadline) not in (int,float) or not math.isfinite(deadline)):
            raise ValueError('One owned main-thread child deadline required.')
        now=clock();remaining,interval=signal.getitimer(signal.ITIMER_REAL)
        if (not 0 < deadline-now <= 720 or interval != 0
                or not 0 < remaining <= deadline-now+.05
                or signal.getsignal(signal.SIGALRM) is not signal.SIG_DFL):
            raise ValueError('Only the still-active original child release timer may be replaced.')
        signal.setitimer(signal.ITIMER_REAL,0)
        self.alarm=Alarms(clock=clock)
        self.deadline=deadline;self.clock=clock;self.active=False;self.entered=False;self.poisoned=False
        self.owner=(os.getpid(),threading.get_ident())

    def __enter__(self):
        if self.entered:raise ValueError('One unit timer scope only.')
        self.entered=True
        self.alarm.arm(self.deadline,hard_deadline=self.deadline,phase='whole-unit-child')
        return self

    def objective(self,deadline,*,hard_deadline):
        outer=self
        class Scope:
            def __enter__(self):
                now=outer.alarm.moment()
                if (outer.owner!=(os.getpid(),threading.get_ident()) or not outer.entered or outer.active or outer.poisoned
                        or hard_deadline!=outer.deadline or not now<deadline<=min(now+120,outer.deadline)):
                    raise ValueError('One continuing owned objective clock within the unit required.')
                outer.active=True
                # Hard expiration is the objective deadline, never the later unit deadline.
                outer.alarm.arm(deadline,hard_deadline=deadline,phase='owned-input-gradient')
                return self
            def __exit__(self,*error):
                outer.active=False
                if outer.alarm.moment()>=deadline:
                    outer.poisoned=True
                    raise Deadline('Owned objective expired; never extend it to the unit deadline.')
                if error[0] is not None:outer.poisoned=True
                outer.alarm.arm(outer.deadline,hard_deadline=outer.deadline,phase='whole-unit-child')
                return False
        return Scope()

    def __exit__(self,*error):
        try:
            if self.active or self.alarm.moment()>=self.deadline or (self.poisoned and error[0] is None):
                raise Deadline('Whole-unit timer expired or objective scope remains open.')
        finally:self.alarm.close()
        return False


def work_plan(plan,scope):
    from models.predictor_pgd_split_bootstrap_v1 import digest,encoded
    return {'role':plan['role'],'method':scope['method'],'step_size_255':scope['step_size_255'],
        'sensitivity_banks':scope['sensitivity_banks'],'control_sha256':plan['control_sha256'],
        'source_manifest_sha256':scope['runtime_manifest_sha256'],'plan_sha256':digest(encoded(plan))}


def _compute(root,plan,scope,timers):
    from models import _pgd_workload_entry_v1 as entry
    from models import predictor_pgd_split_coordinator_v1 as coordinator
    if entry.EXECUTION_BINDING is not None:raise ValueError('One new null worker binding required.')
    w=work_plan(plan,scope)
    output=str(Path(root)/'capture/results/workload'/plan['role']/'unit')
    entry.EXECUTION_BINDING={'kind':'whole-pgd-reviewed-child-binding-v1',
        'host':coordinator.HOST['hostname'],'control_sha256':w['control_sha256'],
        'manifest_sha256':w['source_manifest_sha256'],'result_root':output,
        'deadline':plan['deadline'],'role':plan['role']}
    existing=Path.home()/'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting'
    args=Namespace(output=output,host=coordinator.HOST['hostname'],deadline=plan['deadline'],
        role=plan['role'],control_sha256=w['control_sha256'],manifest_sha256=w['source_manifest_sha256'],
        plan_sha256=w['plan_sha256'],method=w['method'],step_size_255=w['step_size_255'],
        sensitivity_banks=w['sensitivity_banks'],packet=Path(root)/'code',
        upstream=existing/'external/vjepa2',checkpoint=existing/'checkpoints/vitl.pt')
    result,_=entry.entry(args,objective_timer=timers.objective)
    return result


def publish(guard,name,value):
    """Append only declared immutable metadata through anchored private ancestors."""
    from models.predictor_pgd_split_bootstrap_v1 import encoded,digest
    d=guard.layout.classify(name);raw=encoded(value)
    if d['category']!='metadata' or d['mutable'] or len(raw)>d['max_bytes']:
        raise ValueError('Closed bounded immutable worker metadata required.')
    fd=guard.open_root();parents=[]
    try:
        parts=PurePosixPath(name).parts
        for part in parts[:-1]:
            before=os.stat(part,dir_fd=fd,follow_symlinks=False)
            if (not stat.S_ISDIR(before.st_mode) or stat.S_IMODE(before.st_mode)!=0o700
                    or before.st_uid!=os.getuid() or before.st_dev!=guard.binding['root_identity'][0]):
                raise ValueError('Private same-device worker ancestors required.')
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            if (os.fstat(child).st_dev,os.fstat(child).st_ino)!=(before.st_dev,before.st_ino):
                os.close(child);raise ValueError('Worker ancestry changed before open.')
            parents.append((fd,part,before));fd=child
        leaf=parts[-1]
        with os.fdopen(os.open(leaf+'.partial',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd),'wb') as out:
            out.write(raw);out.flush();os.fsync(out.fileno())
        os.link(leaf+'.partial',leaf,src_dir_fd=fd,dst_dir_fd=fd,follow_symlinks=False)
        os.unlink(leaf+'.partial',dir_fd=fd);os.fsync(fd)
        for parent,part,want in parents:
            got=os.stat(part,dir_fd=parent,follow_symlinks=False)
            if (got.st_dev,got.st_ino,got.st_uid,stat.S_IMODE(got.st_mode))!=(want.st_dev,want.st_ino,want.st_uid,stat.S_IMODE(want.st_mode)):
                raise ValueError('Named worker ancestry changed.')
        guard.check_root()
    finally:
        os.close(fd)
        for parent,_,_ in reversed(parents):os.close(parent)
    return {'bytes':len(raw),'sha256':digest(raw)}


def _unit_files(guard,prefix,row):
    from models.predictor_pgd_split_bootstrap_v1 import parsed
    if parsed(guard.read(prefix+'unit/unit-complete.json',512*1024))!=row['unit']:
        raise ValueError('Exact saved complete unit bytes must join returned metadata.')
    start=parsed(guard.read(prefix+'unit/unit-started.json',16384))
    if (start.get('pid')!=row['pid'] or start.get('gradient_budget')!=len(row['unit']['steps'])
            or start.get('settings')!=row['unit']['metrics']['settings']
            or start.get('video_id')!=row['baseline']['video_id']):
        raise ValueError('Exact durable whole-unit start required.')
    for step in row['unit']['steps']:
        part=prefix+f"unit/step-{step['index']:03}"
        begun=parsed(guard.read(part+'-started.json',4096))
        if (begun.get('index')!=step['index'] or begun.get('pid')!=row['pid']
                or begun.get('pixels')!=step['pixels'] or begun.get('factory')!=row['unit']['route']
                or parsed(guard.read(part+'-complete.json',16384))!=step):
            raise ValueError('Exact immutable start/complete input-gradient joins required.')


def run_unit(root,guard,plan,scope):
    from models import predictor_pgd_workload_runtime_records_v1 as records
    started=time.monotonic();prefix='capture/results/workload/'+plan['role']+'/'
    # The parent/setup must not precreate this worker-owned directory.
    (Path(root)/prefix).mkdir(mode=0o700)
    with UnitTimers(plan['deadline']) as timers:
        row=_compute(root,plan,scope,timers)
        records.validate_work(row,work_plan(plan,scope))
        if row['pid']!=os.getpid():raise ValueError('This exact newly released child must own the result.')
        _unit_files(guard,prefix,row);guard.scan(phase='capture',settled=True)
        work=publish(guard,prefix+'unit-work.json',row)
        wall=time.monotonic()-started
        if not row['wall_seconds']<=wall<720:raise ValueError('Own complete child wall accounting required.')
        resource={'kind':'whole-pgd-os-resource-v1','role':plan['role'],'pid':os.getpid(),
            'control_sha256':row['control_sha256'],'plan_sha256':row['plan_sha256'],
            'wall_seconds':wall,'peak_allocator_bytes':row['peak_allocator_bytes'],
            'work_sha256':work['sha256'],'actual_CUDA_execution_admitted':False,
            'scientific_qualified':False,'training_admitted':False}
        r=publish(guard,prefix+'resource.json',resource)
        marker={'kind':'whole-pgd-os-complete-v1','role':plan['role'],'pid':os.getpid(),
            'control_sha256':row['control_sha256'],'plan_sha256':row['plan_sha256'],
            'complete':True,'work':work,'resource':r,'scientific_qualified':False,'training_admitted':False}
        publish(guard,prefix+'complete.json',marker)
        guard.scan(phase='capture',settled=True)
    return 0


def read_complete(guard,plan,scope,observed):
    from models import predictor_pgd_workload_runtime_records_v1 as records
    from models.predictor_pgd_split_bootstrap_v1 import parsed,digest
    prefix='capture/results/workload/'+plan['role']+'/'
    marker=parsed(guard.read(prefix+'complete.json',16384))
    raw=guard.read(prefix+'unit-work.json',768*1024);row=parsed(raw)
    resource_bytes=guard.read(prefix+'resource.json',16384);resource=parsed(resource_bytes)
    records.validate_work(row,work_plan(plan,scope));_unit_files(guard,prefix,row)
    if (marker.get('kind')!='whole-pgd-os-complete-v1' or marker.get('complete') is not True
            or marker.get('work')!={'bytes':len(raw),'sha256':digest(raw)}
            or marker.get('resource')!={'bytes':len(resource_bytes),'sha256':digest(resource_bytes)}
            or resource.get('kind')!='whole-pgd-os-resource-v1'
            or resource.get('work_sha256')!=digest(raw)
            or any(v.get('pid')!=observed['pid'] or v.get('role')!=plan['role']
                   or v.get('control_sha256')!=row['control_sha256']
                   or v.get('plan_sha256')!=row['plan_sha256'] for v in (resource,marker))
            or resource.get('peak_allocator_bytes')!=row['peak_allocator_bytes']
            or observed.get('kind')!='whole-pgd-parent-observation-v1'
            or observed.get('exit_status')!=0 or observed.get('parent_observation_passed') is not True
            or not row['wall_seconds']<=resource['wall_seconds']<=observed['wall_seconds']<=720
            or type(observed.get('RSS_samples')) is not int or not 0<observed['RSS_samples']<=2400
            or type(observed.get('peak_observed_RSS_bytes')) is not int
            or not 0<observed['peak_observed_RSS_bytes']<=4*2**30
            or any(v.get(k) is not False for v in (marker,resource) for k in ('scientific_qualified','training_admitted'))):
        raise ValueError('Whole-unit source/resource/completion/parent identity or bound failed.')
    return row



def finite(v):
    return type(v) in (int, float) and math.isfinite(v)

class Deadline(TimeoutError):
    pass

class Stopped(TimeoutError):
    pass

class Alarms:
    """Default kernel hard stop plus a best-effort main-thread soft refusal.

    Python handlers are not an instantaneous kernel/C-extension stop guarantee.
    Tests exercise this class with fake signal/time/exit APIs only.
    """

    def __init__(self, *, clock=time.monotonic):
        if threading.current_thread() is not threading.main_thread():
            raise ValueError('Outer timers require the main thread.')
        self.clock = clock
        self.owner = os.getpid()
        self.phase = None
        self.hard = None
        self.soft_deadline = None
        self.last = clock()
        self.closed = False
        self.waker = None
        self.generation = 0
        self.previous = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGHUP, signal.SIGTERM, signal.SIGUSR1, signal.SIGINT)}
        self.previous_timer = signal.getitimer(signal.ITIMER_REAL)
        if self.previous_timer != (0.0, 0.0):
            raise ValueError('Fresh outer process must not replace an existing timer.')

    def moment(self):
        now = self.clock()
        if self.closed or os.getpid() != self.owner or (not finite(now)) or (now < self.last):
            raise ValueError('Owned continuing outer clock required.')
        self.last = now
        return now

    def check(self):
        now = self.moment()
        if self.soft_deadline is not None and now >= self.soft_deadline:
            self.handler(signal.SIGALRM, None)
        return now

    def cancel_wake(self):
        self.generation += 1
        if self.waker is not None:
            self.waker.cancel()
            self.waker = None

    def handler(self, number, frame):
        now = self.clock()
        if number == signal.SIGUSR1 and now < self.soft_deadline:
            return
        self.cancel_wake()
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        remaining = min(2, self.hard - now)
        if remaining <= 0:
            os._exit(124)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        if number in (signal.SIGALRM, signal.SIGUSR1):
            raise Deadline('Original ' + str(self.phase) + ' deadline reached; no reset.')
        raise Stopped('Attended connection was interrupted; stop owned work and preserve evidence.')

    def arm(self, deadline, *, hard_deadline, phase):
        now = self.moment()
        if not finite(deadline) or not finite(hard_deadline) or (not now < deadline <= hard_deadline):
            raise Deadline('Original outer deadline is already exhausted.')
        self.cancel_wake()
        self.phase = phase
        self.hard = hard_deadline
        self.soft_deadline = deadline
        for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(s, self.handler)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        current = self.moment()
        if current >= deadline:
            raise Deadline('Original phase exhausted while binding its timers.')
        signal.setitimer(signal.ITIMER_REAL, hard_deadline - current)
        token = self.generation

        def wake():
            if not self.closed and token == self.generation:
                _thread.interrupt_main(signal.SIGUSR1)
        self.waker = threading.Timer(deadline - current, wake)
        self.waker.daemon = True
        self.waker.start()

    def close(self):
        if self.closed:
            return
        self.cancel_wake()
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, h in self.previous.items():
            signal.signal(s, h)
        self.closed = True
