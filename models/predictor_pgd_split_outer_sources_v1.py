"""Cold-only source adapter: retain the entire frozen coordinator prefix.

The appended driver adds bounded observations and outer deadline handoff. No
process, signal, allocation, network or scientific operation occurs on import.
"""
import ast
import copy
import hashlib

COORDINATOR_SHA='956f64be47bfae62bde77c04f44ad6f4d544895ee0fcadc9baadc06b5bcc0afb'

EXTENSION='''
# BEGIN EXACT OUTER JOURNAL/OWNERSHIP ADAPTER V1 (old prefix above unchanged)
import threading as _outer_threading
from types import SimpleNamespace as _OuterNamespace

class OuterJournal:
    def __init__(self,guard,clock):
        self.guard=guard;self.clock=clock;self.owner=(os.getpid(),_outer_threading.get_ident())
        self.calls=0;self.refusals=0;self.rows=0;self.bytes=0;self.previous=None;self.last=None
        self.status_sha=None;self.closed=False;self.last_refusal=None;self.fd=None
    def check(self):
        if self.closed or self.owner!=(os.getpid(),_outer_threading.get_ident()):raise ValueError('One live owned main-thread journal required.')
        self.guard.check_root()
    def checkpoint(self,role,allowed_pid=None):
        self.check()
        if role not in ('setup','proof','numerical','native','forward-preserving','finalization'):raise ValueError('Fixed observation role required.')
        phase='proof' if role=='proof' else 'capture'
        self.calls+=1
        if self.calls>25000:raise ValueError('Bounded guard observations exhausted; no reset.')
        captured=[];storage=None
        try:
            storage,_,_=self.guard.scan(phase=phase)
            def query(*args,**kw):
                value=subprocess.check_output(*args,**kw)
                if type(value) is not str:raise ValueError('Text GPU query required.')
                raw=value.encode()
                captured.append({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                    'text':raw[:1024].decode('utf-8','replace'),'diagnostic_text_truncated':len(raw)>1024})
                return value
            proxy=_OuterNamespace(check_output=query,PIPE=subprocess.PIPE)
            function=FunctionType(gpu_gate.__code__,dict(gpu_gate.__globals__,subprocess=proxy))
            gpu=function(allowed_pid)
        except BaseException as error:
            self.refusals+=1
            self.last_refusal={'role':role,'error_type':type(error).__name__,'message':str(error)[:512],
                'queries':captured,'storage_observation_available':storage is not None}
            # Do not add disk bytes to a tree whose storage gate has refused.
            # The outer same-connection refusal frame preserves this diagnostic.
            if storage is not None:self.persist(role,storage,captured,refusal=self.last_refusal)
            raise
        now=self.clock()
        if not boot.finite(now) or self.last is not None and now<self.last:raise ValueError('Journal continuing clock changed.')
        if self.last is None or now-self.last>=10 or self.previous!=role or role=='finalization':
            self.persist(role,storage,captured)
            self.last=now;self.previous=role
        return gpu
    def persist(self,role,storage,queries,refusal=None):
        self.check();phase='proof' if role=='proof' else 'capture'
        row={'kind':'pgd-split-guard-sample-v1','sequence':self.rows,'role':role,'monotonic':self.clock(),
            'guard_calls':self.calls,'refusals':self.refusals,'GPU_queries':queries,'refusal':refusal,
            'sampling_is_continuous_exclusivity':False,'scientific_qualified':False}
        data=boot.encoded(row)
        if len(data)>8192 or self.bytes+len(data)>512*1024-8192 or self.rows>=384:
            raise ValueError('Bounded guard journal reached; stop this attempt.')
        self.guard.scan(phase=phase)
        if self.fd is None:
            rootfd=self.guard.open_root()
            try:self.fd=os.open('gpu-samples.jsonl',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=rootfd)
            finally:os.close(rootfd)
        held=os.fstat(self.fd);named=(self.guard.root/'gpu-samples.jsonl').lstat()
        if (not stat.S_ISREG(held.st_mode) or held.st_nlink!=1 or held.st_uid!=os.getuid()
                or stat.S_IMODE(held.st_mode)!=0o600 or held.st_size!=self.bytes
                or scratch.file_id(held)!=scratch.file_id(named)):raise ValueError('Owned journal extent/identity changed.')
        with os.fdopen(os.dup(self.fd),'ab',buffering=0) as out:
            if out.write(data)!=len(data):raise OSError('Incomplete bounded journal write.')
        os.fsync(self.fd);self.bytes+=len(data);self.rows+=1
        state={'kind':'pgd-split-storage-status-v1','role':role,'guard_calls':self.calls,'refusals':self.refusals,
            'last_storage':storage,'last_refusal':self.last_refusal,'GPU_journal_bytes':self.bytes,
            'GPU_journal_rows':self.rows,'scientific_qualified':False,'training_admitted':False}
        if self.status_sha is not None and boot.digest(self.guard.read('storage-status.json',64*1024))!=self.status_sha:
            raise ValueError('Only this journal may replace its own status.')
        self.status_sha=_metadata_write(self.guard,'storage-status.json',state,replace=self.status_sha is not None,phase=phase)
        self.guard.scan(phase=phase)
    def close(self):
        if self.closed:return
        self.closed=True
        if self.fd is not None:
            try:os.fsync(self.fd)
            finally:os.close(self.fd);self.fd=None

class OuterDriver(SystemDriver):
    def __init__(self,owner):
        super().__init__(owner)
        self.outer_journal=OuterJournal(owner.outer_guard,owner.clock)
        self.outer_child=None;self.outer_lease=None
    def assert_idle(self):
        if self.outer_child is not None and self.outer_child.poll() is None:
            raise RuntimeError('Owned child exit is unconfirmed; no finalization or collection.')
        if self.outer_lease is not None and not self.outer_lease.closed:
            raise RuntimeError('Owned process handle cleanup is incomplete; no clock handoff.')
    def launch(self,role,plan):
        self.owner.outer_alarm.arm(plan['deadline'],hard_deadline=self.owner.work_deadline,phase=role)
        try:return self._outer_owned_launch(role,plan)
        except BaseException:
            self.assert_idle()
            self.owner.outer_alarm.arm(min(self.owner.work_deadline,self.owner.clock()+120),
                hard_deadline=self.owner.work_deadline,phase='finalization')
            raise
    # Cold builder inserts two copies of frozen methods here. Only observation
    # callbacks and retention of owned child/lease references differ.

class OuterCoordinator(Coordinator):
    def __init__(self,*args,outer_guard,outer_alarm,**kw):
        self.outer_guard=outer_guard;self.outer_alarm=outer_alarm
        super().__init__(*args,**kw)
        self.driver=OuterDriver(self)
# END EXACT OUTER JOURNAL/OWNERSHIP ADAPTER V1
'''


def build(original):
    if type(original) is not bytes or hashlib.sha256(original).hexdigest()!=COORDINATOR_SHA:
        raise ValueError('Exact frozen coordinator source required.')
    tree=ast.parse(original)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SystemDriver')
    methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    prepare=copy.deepcopy(methods['prepare']);launch=copy.deepcopy(methods['launch']);launch.name='_outer_owned_launch'
    changes={'prepare_gpu':0,'guard_rebind':0,'launch_gate':0,'child':0,'lease':0}
    class Edit(ast.NodeTransformer):
        def visit_Expr(self,n):
            if ast.unparse(n)=='gpu_gate()':
                changes['prepare_gpu']+=1
                return ast.copy_location(ast.parse("self.outer_journal.checkpoint('setup')").body[0],n)
            return self.generic_visit(n)
        def visit_Assign(self,n):
            text=ast.unparse(n)
            if text=='self.guard = scratch.Guard(o.root, o.binding, layout)':
                changes['guard_rebind']+=1
                return [n,*ast.parse('self.outer_journal.guard=self.guard').body]
            if text.startswith('child = subprocess.Popen('):
                changes['child']+=1
                return [n,*ast.parse('self.outer_child=child').body]
            if text.startswith('lease = frozen_watch.PidfdLease('):
                changes['lease']+=1
                return [n,*ast.parse('self.outer_lease=lease').body]
            return self.generic_visit(n)
        def visit_FunctionDef(self,n):
            if n.name=='gate':
                assert len(n.body)==2 and ast.unparse(n.body[1]).startswith('gpu_gate(')
                changes['launch_gate']+=1
                n.body=ast.parse('self.outer_journal.checkpoint(role,child.pid if child is not None else None)').body
                return n
            return self.generic_visit(n)
    prepare=Edit().visit(prepare);launch=Edit().visit(launch)
    assert changes=={'prepare_gpu':1,'guard_rebind':1,'launch_gate':1,'child':1,'lease':1},changes
    extension=ast.parse(EXTENSION)
    new=next(n for n in extension.body if isinstance(n,ast.ClassDef) and n.name=='OuterDriver')
    new.body.extend([prepare,launch])
    data=original+b'\n'+ast.unparse(ast.fix_missing_locations(extension)).encode()+b'\n'
    compile(data,'<UNEXECUTED-JOURNALED-COORDINATOR>','exec')
    return data,changes
