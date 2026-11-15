"""Parent wall/RSS supervision core; current checks use fixtures/fake handles.

No process is spawned here. A future authorized launcher must supply its own
unreaped child and immediately bind the root handle before any polling/reaping.
Proc sampling is racy and not an instantaneous address-space/kernel quota.
"""
import math
import os
from pathlib import Path
import select
import signal
import stat
import time
from types import SimpleNamespace

MAX_PROCESSES=64
MAX_THREADS=512
RSS_BYTES=4*2**30


def integer(v):return type(v) is int and v>0


class ProcReader:
    def __init__(self,root=Path('/proc')):self.root=Path(root)
    def text(self,path,cap):
        with path.open('rb') as f:data=f.read(cap+1)
        if len(data)>cap:raise ValueError('Bounded process metadata required.')
        return data.decode('ascii')
    def identity(self,pid):
        if not integer(pid):raise ValueError('Positive process identity required.')
        folder=self.root/str(pid);s=folder.stat()
        if not stat.S_ISDIR(s.st_mode) or s.st_uid!=os.getuid():raise ValueError('Only same-user launched process metadata is permitted.')
        raw=self.text(folder/'stat',16384);end=raw.rfind(')');start=raw.find('(')
        if start<1 or end<=start or raw[:start].strip()!=str(pid):raise ValueError('Exact /proc process stat identity required.')
        fields=raw[end+1:].split()
        if len(fields)<22:raise ValueError('Complete bounded process stat required.')
        parent,group,session,birth=(int(fields[i]) for i in (1,2,3,19))
        if min(parent,group,session)<0 or birth<=0:raise ValueError('Typed process birth/parent/session required.')
        return {'pid':pid,'start_ticks':birth,'parent_pid':parent,'group_pid':group,'session_pid':session,'state':fields[0]}
    def process(self,pid):
        before=self.identity(pid);folder=self.root/str(pid)
        # smaps_rollup supplies one process sum. Shared resident pages in two
        # processes are deliberately counted twice in the conservative tree sum.
        lines=self.text(folder/'smaps_rollup',16384).splitlines();rss=[]
        for line in lines:
            if line.startswith('Rss:'):
                fields=line.split()
                if len(fields)!=3 or fields[2]!='kB' or not fields[1].isdigit():raise ValueError('Typed resident-byte observation required.')
                rss.append(int(fields[1])*1024)
        if len(rss)!=1:raise ValueError('One observed smaps_rollup Rss field required; no guessed fallback.')
        threads=list((folder/'task').iterdir())
        if not 0<len(threads)<=MAX_THREADS:raise ValueError('Bounded task enumeration required.')
        children=set()
        for thread in threads:
            if not thread.name.isdigit():raise ValueError('Typed task identity required.')
            try:values=self.text(thread/'children',16384).split()
            except FileNotFoundError:
                if thread.exists():raise
                continue # A vanished task contributes no live child list; enumeration remains explicitly non-exhaustive.
            if any(not v.isdigit() or int(v)<=0 for v in values):raise ValueError('Typed child PID list required.')
            children.update(int(v) for v in values)
            if len(children)>MAX_PROCESSES:raise ValueError('Unexpected descendant count; no unbounded discovery.')
        after=self.identity(pid)
        if any(before[k]!=after[k] for k in before if k!='state'):raise ValueError('Process identity changed while sampling.')
        return dict(before,rss_bytes=rss[0],children=sorted(children))
    def sample(self,pid,start_ticks):
        queue=[(pid,None)];seen={};total=0
        while queue:
            current,parent=queue.pop(0)
            try:row=self.process(current)
            except FileNotFoundError:
                if parent is None:raise
                continue # An already vanished child cannot add a current RSS sample.
            if current in seen:raise ValueError('Repeated process/cyclic child metadata.')
            if ((parent is None and row['start_ticks']!=start_ticks)
                    or parent is not None and row['parent_pid']!=parent
                    or parent is None and (row['session_pid']!=pid or row['group_pid']!=pid)):
                raise ValueError('Launched tree birth/parent/session identity changed.')
            seen[current]=row;total+=row['rss_bytes']
            if len(seen)>MAX_PROCESSES:raise ValueError('Observed process count ceiling.')
            queue.extend((child,current) for child in row['children'])
        return {'processes':seen,'rss_bytes':total}


def system_pidfd_api():
    opener=getattr(os,'pidfd_open',None);sender=getattr(signal,'pidfd_send_signal',None)
    if not callable(opener) or not callable(sender):raise ValueError('Linux PID-handle APIs required; never fall back to numeric-PID signals.')
    def open_handle(pid):
        fd=opener(pid);os.set_inheritable(fd,False);return fd
    def dead(fd):
        poll=select.poll();poll.register(fd,select.POLLIN);return bool(poll.poll(0))
    return SimpleNamespace(open=open_handle,dead=dead,send=lambda fd,sig:sender(fd,sig,None,0),close=os.close)


class PidfdLease:
    """A handle to a verified owned birth; signals never use a numeric PID."""
    def __init__(self,pid,birth,parent,session,reader,*,api=None):
        if not all(integer(v) for v in (pid,birth,parent,session)) or pid==os.getpid():raise ValueError('Owned launched process identity required.')
        self.pid,self.birth,self.parent,self.session=pid,birth,parent,session;self.api=system_pidfd_api() if api is None else api
        self.fd=None;self.closed=False
        before=reader.identity(pid);expected_group=before['group_pid']
        def check():
            value=reader.identity(pid)
            if (value['start_ticks']!=birth or value['parent_pid']!=parent or value['session_pid']!=session
                    or value['group_pid']!=expected_group):raise ValueError('Owned process birth/parent/session changed before handle binding.')
        check()
        try:self.fd=self.api.open(pid);check()
        except BaseException:
            if self.fd is not None:self.api.close(self.fd)
            self.closed=True;raise
    def dead(self):
        if self.closed:raise ValueError('Closed process handle cannot be reused.')
        return self.api.dead(self.fd)
    def send(self,sig):
        if self.closed or sig not in (signal.SIGTERM,signal.SIGKILL):raise ValueError('Bound owned termination handle required.')
        if not self.dead():
            try:self.api.send(self.fd,sig)
            except ProcessLookupError:pass
    def close(self):
        if not self.closed:self.api.close(self.fd);self.closed=True


def watch(child,root_lease,*,sample,lease_factory,gate,phase_deadline,work_deadline,launch_started,clock=time.monotonic,pause=time.sleep):
    """Observe an already-launched child, then stop only held owned handles.

    The same phase deadline covers startup, saved-value checks, CUDA work,
    profiler stop, raw extraction, validation and publication. Up to 2 seconds
    termination grace comes from the existing overall overhead, never an
    extension of the fixed work deadline. This module does not authorize launch.
    """
    start=launch_started
    leases={(root_lease.pid,root_lease.birth):root_lease};count=0;peak=0;last=start;primary=None;stop_errors=[]
    try:
        if (not integer(child.pid) or child.pid!=root_lease.pid or not all(type(v) in (int,float) and math.isfinite(v) for v in (start,phase_deadline,work_deadline))
                or not start<phase_deadline<=min(start+120,work_deadline)):
            raise ValueError('Original bounded launch/phase/work deadlines required.')
        gate()
        while True:
            now=clock()
            if type(now) not in (int,float) or not math.isfinite(now) or now<last or now>=phase_deadline or now>=work_deadline:
                raise TimeoutError('Original component parent deadline reached; no clock reset.')
            last=now;status=child.poll()
            if status is not None:break
            try:observation=sample(child.pid,root_lease.birth)
            except (FileNotFoundError,ProcessLookupError):
                status=child.poll()
                if status is not None:break
                raise
            processes=observation['processes']
            if type(processes) is not dict or not 0<len(processes)<=MAX_PROCESSES:raise ValueError('Bounded observed process tree required.')
            if child.pid not in processes or processes[child.pid]['start_ticks']!=root_lease.birth:raise ValueError('Owned root missing from RSS observation.')
            for pid,row in processes.items():
                if (type(pid) is not int or row.get('pid')!=pid or not integer(row.get('start_ticks'))
                        or type(row.get('rss_bytes')) is not int or row['rss_bytes']<0):raise ValueError('Typed observed process identity required.')
                key=(pid,row['start_ticks'])
                if key not in leases:
                    parents=[value for (p,b),value in leases.items() if p==row['parent_pid']]
                    if len(parents)!=1 or parents[0].dead() or any(p==pid for p,b in leases):
                        raise ValueError('Unowned/reused observed descendant; never signal it.')
                    owned=lease_factory(pid,row['start_ticks'],row['parent_pid'],row['session_pid'])
                    if parents[0].dead():owned.close();raise ValueError('Parent exited while binding descendant; no guessed ownership.')
                    leases[key]=owned
            rss=observation.get('rss_bytes')
            if type(rss) is not int or rss<0 or rss!=sum(r['rss_bytes'] for r in processes.values()):raise ValueError('Complete observed tree RSS sum required.')
            peak=max(peak,rss);count+=1
            if rss>RSS_BYTES:raise MemoryError('Observed component RSS exceeds 4 GiB; stop this launch only.')
            if count>2400:raise ValueError('Bounded supervision samples; never reset the phase.')
            gate()
            now=clock()
            if type(now) not in (int,float) or not math.isfinite(now) or now<last or now>=phase_deadline or now>=work_deadline:raise TimeoutError('Original component parent deadline reached after observation.')
            last=now;pause(min(.1,phase_deadline-now,work_deadline-now))
        if type(status) is not int or status!=0:raise RuntimeError('Component child failed; no retry or next component.')
        if not count or any(not value.dead() for value in leases.values()):raise ValueError('At least one RSS observation and no surviving owned task required.')
        gate();end=clock()
        if type(end) not in (int,float) or not math.isfinite(end) or end<last or end>=phase_deadline or end>=work_deadline:raise TimeoutError('Component final gates exhausted the original deadline.')
        return {'kind':'pgd-split-component-parent-observation-v1','pid':child.pid,'start_ticks':root_lease.birth,
            'exit_status':status,'wall_seconds':end-start,'RSS_samples':count,'peak_observed_RSS_bytes':peak,
            'observed_process_identities':len(leases),'parent_observation_passed':True,
            'sampling_is_instantaneous_kernel_or_address_space_quota':False,'proc_child_enumeration_is_exhaustive':False,
            'scientific_qualified':False,'actual_CUDA_execution_admitted':False}
    except BaseException as error:primary=error;raise
    finally:
        try:
            active=[]
            for lease in leases.values():
                try:
                    if not lease.dead():active.append(lease);lease.send(signal.SIGTERM)
                except BaseException as e:stop_errors.append(e)
            bound=work_deadline if type(work_deadline) in (int,float) and math.isfinite(work_deadline) else clock()
            stop_end=min(bound,clock()+2)
            grace_samples=0
            while active and clock()<stop_end and grace_samples<20:
                grace_samples+=1
                try:active=[value for value in active if not value.dead()]
                except BaseException as e:stop_errors.append(e);break
                if active:pause(min(.1,max(0,stop_end-clock())))
            for lease in active:
                try:lease.send(signal.SIGKILL)
                except BaseException as e:stop_errors.append(e)
            # Reaping a child already exited is bounded by the Popen interface;
            # on failure never block beyond the existing termination allowance.
            try:child.wait(timeout=max(0,min(.1,bound-clock())))
            except BaseException as e:stop_errors.append(e)
        finally:
            for lease in leases.values():
                try:lease.close()
                except BaseException as e:stop_errors.append(e)
        if stop_errors:
            if primary is not None:primary.add_note('Owned process cleanup also failed: '+type(stop_errors[0]).__name__)
            else:raise stop_errors[0]
