"""Bounded component lifecycle with explicitly supplied UNBOUND runtime ports.

No scientific imports, launch, signal, network or training entry exists here.
Current executions use fake engines/profiles and tiny labeled byte fixtures only.
"""
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import threading
import time
from models import predictor_pgd_split_backend_v1 as backend
from models import predictor_pgd_split_lineage_v1 as lineage

KIND='pgd-split-component-worker-v1'
PHASES=frozenset(('startup','source_verify','operands','route_start','profiler_start','forward','backward',
    'cuda_sync','profiler_stop','route_close','ownership_publish','returns_publish','raw_extract','raw_publish',
    'backend_validate','backend_publish','restoration','worker_publish','resource_publish','complete_publish'))
CLEANUP=frozenset(('profiler_stop','route_close'))
PORTS=frozenset(('startup','source','operands','route','profile','compute','raw','serialize','restore','allocator'))


def encoded(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def directory_id(s):return (s.st_dev,s.st_ino,s.st_uid,stat.S_IMODE(s.st_mode))
def file_id(s):return (*directory_id(s),s.st_nlink,s.st_size,s.st_mtime_ns)


def private_bytes(path,cap,want):
    path=Path(path).absolute();parent=path.parent
    if not lineage.sha(want) or type(cap) is not int or not 0<cap<=4*2**20:raise ValueError('Bound source extent/hash required.')
    if parent.resolve(strict=True)!=parent:raise ValueError('Canonical private source parent required.')
    p=parent.lstat();s=path.lstat()
    if (not stat.S_ISDIR(p.st_mode) or p.st_uid!=os.getuid() or stat.S_IMODE(p.st_mode)!=0o700
            or not stat.S_ISREG(s.st_mode) or s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode)!=0o600
            or s.st_nlink!=1 or s.st_dev!=p.st_dev or not 0<=s.st_size<=cap):raise ValueError('Private bounded source required.')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        before=os.fstat(f.fileno());data=f.read(cap+1);after=os.fstat(f.fileno())
    if (file_id(s)!=file_id(before) or file_id(before)!=file_id(after) or file_id(path.lstat())!=file_id(after)
            or directory_id(parent.lstat())!=directory_id(p) or len(data)!=s.st_size or hashlib.sha256(data).hexdigest()!=want):
        raise ValueError('Held source/parent bytes or identity changed.')
    return data


class Stages:
    """One exclusive 64 KiB fsynced progress chain; no success semantics alone."""
    def __init__(self,folder,identity,clock,guard):
        self.folder=Path(folder);self.identity=dict(identity);self.clock=clock;self.guard=guard
        self.owner=(os.getpid(),threading.get_ident());self.rows=[];self.head='0'*64;self.pending=None
        self.begun=set();self.returned=set();self.raised=set();self.cleanup_attempted=set()
        self.closed=self.poisoned=False;self.fd=self.parent_fd=None
        s=self.folder.lstat();self.parent_id=directory_id(s)
        if not stat.S_ISDIR(s.st_mode) or s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode)!=0o700:raise ValueError('Private component progress directory required.')
        try:
            self.parent_fd=os.open(self.folder,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            if directory_id(os.fstat(self.parent_fd))!=self.parent_id:raise ValueError('Progress parent changed.')
            self.fd=os.open('progress.jsonl',os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=self.parent_fd)
            self.file_identity=directory_id(os.fstat(self.fd));self.prefix=b''
            os.fsync(self.parent_fd);self.write('observer','opened')
        except BaseException:self.close();raise
    def check(self):
        if self.closed or self.poisoned or (os.getpid(),threading.get_ident())!=self.owner:raise ValueError('Owned active component journal required.')
        if directory_id(self.folder.lstat())!=self.parent_id or directory_id(os.fstat(self.parent_fd))!=self.parent_id:raise ValueError('Component journal parent changed.')
        a=os.fstat(self.fd);b=os.stat('progress.jsonl',dir_fd=self.parent_fd,follow_symlinks=False)
        if (not stat.S_ISREG(b.st_mode) or directory_id(a)!=self.file_identity or file_id(a)!=file_id(b)
                or a.st_nlink!=1 or a.st_size!=len(self.prefix) or os.pread(self.fd,len(self.prefix)+1,0)!=self.prefix):
            raise ValueError('Component journal prefix/identity changed.')
    def write(self,phase,event,error=None):
        try:
            self.check();tick=self.clock()
            if type(tick) not in (int,float) or not math.isfinite(tick) or self.rows and tick<self.rows[-1]['monotonic']:
                raise ValueError('Monotonic component clock required.')
            body={'kind':'pgd-split-component-progress-v1','sequence':len(self.rows),'identity':self.identity,
                'phase':phase,'event':event,'monotonic':tick,'previous_sha256':self.head,
                'error_type':None if error is None else type(error).__name__[:80],'scientific_qualified':False}
            checksum=hashlib.sha256(encoded(body)).hexdigest();raw=encoded(dict(body,row_sha256=checksum))
            if len(raw)>1024 or len(self.rows)>=64 or len(self.prefix)+len(raw)>64*1024:raise ValueError('Component journal bound; no truncation.')
            view=memoryview(raw)
            while view:
                n=os.write(self.fd,view)
                if n<=0:raise OSError('Incomplete progress append.')
                view=view[n:]
            os.fsync(self.fd);self.rows.append(body);self.prefix+=raw;self.head=checksum
        except BaseException:self.poisoned=True;raise
    def begin(self,phase):
        if phase not in PHASES or phase in self.begun or self.pending is not None:raise ValueError('One nonoverlapping component phase required.')
        self.write(phase,'begin');self.pending=phase;self.begun.add(phase)
    def end(self,phase,error=None):
        if self.pending!=phase:raise ValueError('Expected active component phase ending.')
        self.write(phase,'returned' if error is None else 'raised',error);self.pending=None
        (self.returned if error is None else self.raised).add(phase)
    def call(self,phase,action,*args,**kwargs):
        self.guard();self.begin(phase)
        try:
            self.guard();value=action(*args,**kwargs);self.guard()
        except BaseException as error:
            try:self.end(phase,error)
            except BaseException as extra:error.add_note('Component end record failed: '+type(extra).__name__)
            raise
        self.end(phase);self.guard();return value
    def cleanup(self,phase,action,*args):
        if phase not in CLEANUP or phase in self.cleanup_attempted:raise ValueError('One cleanup attempt required.')
        self.cleanup_attempted.add(phase);before=None
        try:self.begin(phase)
        except BaseException as e:before=e
        try:value=action(*args)
        except BaseException as error:
            if before is None:
                try:self.end(phase,error)
                except BaseException as extra:error.add_note('Cleanup end record failed: '+type(extra).__name__)
            else:error.add_note('Cleanup begin record failed: '+type(before).__name__)
            raise
        if before is not None:raise before
        self.end(phase);return value
    def enter(self,stack,phase,cm):
        value=type(cm).__enter__(cm)
        stack.push(lambda *exc:self.cleanup(phase,type(cm).__exit__,cm,*exc))
        return value
    def require(self):
        self.check()
        if self.pending is not None or self.raised or not PHASES-{'complete_publish'}<=self.returned:
            raise ValueError('All component phases must return before completion publication.')
    def close(self):
        if self.fd is not None:os.close(self.fd);self.fd=None
        if self.parent_fd is not None:os.close(self.parent_fd);self.parent_fd=None
        self.closed=True


class ComponentJob:
    """One fresh role, no warmup/reference/replay; runtime ports stay unbound."""
    def __init__(self,role,binding,store,ports,*,deadline,clock=time.monotonic,evidence_origin):
        if role not in ('native','forward-preserving') or type(ports) is not dict or set(ports)!=PORTS or any(not callable(v) for v in ports.values()):
            raise ValueError('Exact component role/runtime port set required.')
        expected={'design_sha256','control_sha256','boundary_sha256','probe_index'}
        if (type(binding) is not dict or set(binding)!=expected or binding['design_sha256']!=lineage.DESIGN_SHA
                or not all(lineage.sha(binding[k]) for k in expected-{'probe_index'})
                or type(binding['probe_index']) is not int or binding['probe_index']!=(role=='forward-preserving')):
            raise ValueError('Exact source/control/probe binding required.')
        if type(evidence_origin) is not str or not 0<len(evidence_origin)<=80:raise ValueError('Explicit evidence origin required.')
        self.role,self.binding,self.store,self.ports=role,dict(binding),store,dict(ports)
        self.clock=clock;self.started=clock()
        if type(deadline) not in (int,float) or not math.isfinite(deadline) or not deadline>self.started:raise ValueError('Continuing component deadline required.')
        self.deadline=min(deadline,self.started+120);self.origin=evidence_origin;self.used=False;self.log=None
        self.prefix='results/components/'+role+'/';self.folder=store.root/self.prefix
        self.state={'kind':KIND,'role':role,'binding':dict(binding),'pid':os.getpid(),'complete':False,
            'evidence_origin':evidence_origin,'scientific_qualified':False,'actual_CUDA_execution_admitted':False,
            'component_accuracy_qualified':False,'representative_backend_qualified':False,'full_EOT8_backend_qualified':False,
            'performance_qualified':False,'replay_qualified':False,'training_admitted':False,'outputs':{}}
    def time_gate(self):
        now=self.clock()
        if type(now) not in (int,float) or not math.isfinite(now) or now<self.started or now>=self.deadline:
            raise TimeoutError('Continuing component deadline exhausted; no retry.')
    def save(self,name,value):return self.store.write_bytes(self.prefix+name,encoded(value))
    def execute(self):
        if self.used:raise ValueError('One new component attempt; no retry.')
        self.used=True;primary=None
        try:
            self.time_gate()
            for parent in (self.store.root/'results',self.store.root/'results/components'):
                try:parent.mkdir(mode=0o700)
                except FileExistsError:pass
            self.folder.mkdir(mode=0o700,exist_ok=False);self.store.check()
            self.log=Stages(self.folder,{'role':self.role,'binding':self.binding,'pid':os.getpid(),'origin':self.origin},self.clock,self.time_gate)
            self.save('worker.json',self.state)
            startup=self.log.call('startup',self.ports['startup'])
            if (startup.get('pid')!=os.getpid() or startup.get('kind')!='native-SDPA-component-startup-v2'
                    or type(startup.get('configure_calls')) is not int or startup['configure_calls']!=1 or startup.get('cuda_initialized_before') is not False
                    or startup.get('cuda_initialized_after') is not True or startup.get('allocator_cap_bytes')!=7*2**30):
                raise ValueError('Original one-owner capped startup required.')
            self.state['startup']=startup;self.time_gate()
            cap,joined=self.log.call('source_verify',self.ports['source'])
            if (joined.get('control_sha256')!=self.binding['control_sha256'] or joined.get('boundary_sha256')!=self.binding['boundary_sha256']
                    or joined.get('design_sha256')!=lineage.DESIGN_SHA or joined.get('source_objective_index')!=0
                    or joined.get('lineage_metadata_consistent') is not True):raise ValueError('New canonical source binding changed.')
            self.state['source']=joined
            values,incoming,metadata=self.log.call('operands',self.ports['operands'],cap)
            expected=lineage.operands(cap)
            if joined.get('layout')!=expected or type(metadata) is not dict or set(metadata)!=set(lineage.NAMES):
                raise ValueError('Reconstructed operand metadata differs from the canonical source plan.')
            for name,want in expected['operands'].items():
                actual_meta=metadata[name]
                matching={'shape':want['shape'],'stride':want['stride'],'sha256':want['logical_sha256'],
                    'dtype':'torch.float32','device':'cuda:0','storage_offset_elements':want['storage_offset_elements'],
                    'storage_bytes':want['storage_bytes'],'alias_group':want['alias_group'],'requires_grad':want['requires_grad'],
                    'unknown_padding_filled_zero':True,'native_physical_alignment_identity_claimed':False}
                if any(type(actual_meta.get(k)) is not type(v) or actual_meta[k]!=v for k,v in matching.items()):
                    raise ValueError('Canonical reconstructed bytes/layout/aliases/flags changed: '+name)
            self.state['reconstructed_inputs']=metadata;self.time_gate()
            with ExitStack() as stack:
                route=self.log.call('route_start',lambda:self.log.enter(stack,'route_close',self.ports['route'](self.role,values,incoming,self.binding)))
                profile=self.log.call('profiler_start',lambda:self.log.enter(stack,'profiler_stop',self.ports['profile']()))
                actual=self.ports['compute'](route,self.log.call)
                self.time_gate()
            ownership=route.snapshot()
            if ownership.get('route_complete') is not True:raise ValueError('Closed original root/leaf/flags/restoration proof required.')
            self.state['ownership']=self.log.call('ownership_publish',self.save,'ownership.json',ownership)
            def returns():
                if type(actual) is not dict or set(actual)!={'output','dq','dk','dv'}:raise ValueError('Four actual component returns required.')
                for name in ('output','dq','dk','dv'):
                    data=self.ports['serialize'](actual[name]);self.state['outputs'][name]=self.store.write_bytes(self.prefix+name+'.npy',data)
            self.log.call('returns_publish',returns);self.time_gate()
            data=self.log.call('raw_extract',self.ports['raw'],profile)
            if type(data) is not bytes or not 0<len(data)<=4*2**20:raise ValueError('Whole 4 MiB raw envelope required; no truncation.')
            trace=self.log.call('raw_publish',self.store.write_bytes,self.prefix+'events.json',data)
            self.state['trace']=trace
            self.state['trace_manifest']=self.save('trace-manifest.json',dict(trace,kind='pgd-split-raw-trace-manifest-v1',saved_before_semantic_validation=True,binding=self.binding))
            observed=self.log.call('backend_validate',lambda:backend.assess_bytes(
                backend.read_private_trace(self.folder/'events.json',trace['sha256']),role=self.role,binding=self.binding))
            self.state['backend']=self.log.call('backend_publish',self.save,'backend.json',observed)
            self.log.call('restoration',self.ports['restore']);self.time_gate()
            peak=self.ports['allocator']()
            if type(peak) is not int or not 0<peak<=7*2**30:raise ValueError('Component allocator cap/receipt required.')
            self.state.update(complete=True,wall_seconds=self.clock()-self.started)
            worker=self.log.call('worker_publish',self.save,'worker.json',self.state)
            resource=self.log.call('resource_publish',self.save,'resource.json',{
                'kind':'pgd-split-component-resource-v1','binding':self.binding,'role':self.role,'pid':os.getpid(),
                'worker_sha256':worker['sha256'],'startup':startup,'peak_allocator_bytes':peak,
                'wall_seconds':self.clock()-self.started,'requires_parent_RSS_supervision':True,'actual_parent_RSS_receipt_admitted':False,
                'scientific_qualified':False,'performance_qualified':False})
            self.log.require();self.time_gate()
            marker={'kind':'pgd-split-component-complete-v1','role':self.role,'binding':self.binding,
                'worker':worker,'resource':resource,'complete':True,'scientific_qualified':False}
            self.log.call('complete_publish',self.save,'complete.json',marker)
            self.time_gate();return dict(marker)
        except BaseException as error:
            primary=error;self.state.update(complete=False,error_type=type(error).__name__,error=str(error)[:2048])
            if self.folder.exists():
                try:self.save('failed.json',{'kind':'pgd-split-component-failed-v1','role':self.role,'binding':self.binding,
                    'complete':False,'error_type':type(error).__name__,'scientific_qualified':False})
                except BaseException as publication:error.add_note('Failed component publication also failed: '+type(publication).__name__)
            raise
        finally:
            if self.log is not None:self.log.close()


def read_completion(folder,*,role,binding,marker_sha256,progress_sha256):
    """Saved metadata/byte checks only; no CUDA/lineage/accuracy admission."""
    folder=Path(folder);p=folder/'complete.json'
    if (folder/'failed.json').exists():raise ValueError('Failed component cannot qualify completion.')
    marker=json.loads(private_bytes(p,16*1024,marker_sha256))
    if (marker.get('kind')!='pgd-split-component-complete-v1' or marker.get('role')!=role
            or marker.get('binding')!=binding or marker.get('complete') is not True or marker.get('scientific_qualified') is not False):
        raise ValueError('New component metadata completion identity required.')
    def verified(d,name,cap):
        if (type(d) is not dict or set(d)!={'file','bytes','sha256'} or d.get('file')!=name
                or type(d.get('bytes')) is not int or not 0<=d['bytes']<=cap):raise ValueError('Fixed bounded component descriptor required.')
        data=private_bytes(folder/name,cap,d['sha256'])
        if len(data)!=d['bytes']:raise ValueError('Committed descriptor extent changed.')
        return data
    worker=json.loads(verified(marker['worker'],'worker.json',256*1024))
    resource=json.loads(verified(marker['resource'],'resource.json',16*1024))
    if (worker.get('kind')!=KIND or worker.get('binding')!=binding or worker.get('role')!=role or worker.get('complete') is not True
            or type(worker.get('pid')) is not int or worker['pid']<=0
            or any(worker.get(k) is not False for k in ('scientific_qualified','actual_CUDA_execution_admitted','component_accuracy_qualified',
                'representative_backend_qualified','full_EOT8_backend_qualified','performance_qualified','replay_qualified','training_admitted'))):
        raise ValueError('New unadmitted component worker metadata required.')
    if (resource.get('kind')!='pgd-split-component-resource-v1' or resource.get('binding')!=binding or resource.get('role')!=role
            or resource.get('pid')!=worker['pid'] or resource.get('worker_sha256')!=marker['worker']['sha256']
            or resource.get('startup')!=worker.get('startup') or resource.get('requires_parent_RSS_supervision') is not True
            or resource.get('actual_parent_RSS_receipt_admitted') is not False or type(resource.get('peak_allocator_bytes')) is not int
            or not 0<resource['peak_allocator_bytes']<=7*2**30 or type(resource.get('wall_seconds')) not in (int,float)
            or not 0<=resource['wall_seconds']<=120):raise ValueError('Bound component resource metadata required.')
    if type(worker.get('outputs')) is not dict or set(worker['outputs'])!={'output','dq','dk','dv'}:raise ValueError('Four saved component returns required.')
    for name in ('output','dq','dk','dv'):verified(worker['outputs'][name],name+'.npy',4*2**20)
    ownership=json.loads(verified(worker['ownership'],'ownership.json',64*1024))
    if (ownership.get('route_complete') is not True or ownership.get('role')!=role or ownership.get('binding')!=binding
            or ownership.get('scientific_qualified') is not False):raise ValueError('Closed source-bound component route metadata required.')
    trace=verified(worker['trace'],'events.json',4*2**20)
    manifest=json.loads(verified(worker['trace_manifest'],'trace-manifest.json',16*1024))
    if (manifest.get('sha256')!=worker['trace']['sha256'] or manifest.get('binding')!=binding
            or manifest.get('saved_before_semantic_validation') is not True):raise ValueError('Raw trace manifest binding changed.')
    observed=json.loads(verified(worker['backend'],'backend.json',128*1024))
    if observed!=backend.assess_bytes(trace,role=role,binding=binding):raise ValueError('Saved backend metadata differs from its immutable raw source.')
    raw=private_bytes(folder/'progress.jsonl',64*1024,progress_sha256)
    if len(raw)>64*1024 or not raw.endswith(b'\n'):raise ValueError('Complete bounded component progress chain required.')
    previous='0'*64;returned=set();pending=None;seen=set();identity=None;last_tick=-1
    for i,line in enumerate(raw.splitlines()):
        if i>=64 or len(line)+1>1024:raise ValueError('Progress row ceiling.')
        row=json.loads(line)
        if set(row)!={'kind','sequence','identity','phase','event','monotonic','previous_sha256','error_type','scientific_qualified','row_sha256'}:
            raise ValueError('Exact component progress row schema required.')
        checksum=row.pop('row_sha256')
        if (row.get('kind')!='pgd-split-component-progress-v1' or type(row.get('sequence')) is not int or row.get('sequence')!=i or row.get('previous_sha256')!=previous or checksum!=hashlib.sha256(encoded(row)).hexdigest()
                or row.get('scientific_qualified') is not False or row.get('error_type') is not None
                or type(row.get('monotonic')) not in (int,float) or not math.isfinite(row['monotonic']) or row['monotonic']<last_tick):raise ValueError('Component progress chain changed.')
        ident=row['identity']
        if identity is None:identity=ident
        if (type(ident) is not dict or set(ident)!={'role','binding','pid','origin'} or ident!=identity or ident.get('role')!=role
                or ident.get('pid')!=worker['pid'] or ident.get('origin')!=worker.get('evidence_origin') or ident.get('binding')!=binding):raise ValueError('Component progress identity changed.')
        phase,event=row['phase'],row['event']
        if i==0:
            if (phase,event)!=('observer','opened'):raise ValueError('Component opening record required.')
        elif event=='begin':
            if pending is not None or phase in seen or phase not in PHASES:raise ValueError('Component progress ordering changed.')
            pending=phase;seen.add(phase)
        elif event=='returned':
            if pending!=phase:raise ValueError('Component return lacks a unique begin.')
            returned.add(phase);pending=None
        else:raise ValueError('Raised or unknown component stage cannot qualify completion.')
        previous=checksum;last_tick=row['monotonic']
    if pending is not None or returned!=PHASES or (phase,event)!=('complete_publish','returned'):
        raise ValueError('All component gates and terminal publication return required.')
    return {'metadata_complete':True,'scientific_qualified':False,'actual_CUDA_execution_admitted':False}


def bind_parent_metadata(marker,worker,resource,observed,*,role,binding):
    """Join already-verified receipts; no dict alone authenticates execution."""
    if (marker.get('kind')!='pgd-split-component-complete-v1' or marker.get('role')!=role or marker.get('binding')!=binding
            or marker.get('complete') is not True or worker.get('kind')!=KIND or worker.get('role')!=role
            or worker.get('binding')!=binding or worker.get('complete') is not True or type(worker.get('pid')) is not int
            or worker['pid']<=0 or resource.get('pid')!=worker['pid'] or resource.get('binding')!=binding
            or resource.get('role')!=role or resource.get('worker_sha256')!=marker.get('worker',{}).get('sha256')
            or observed.get('kind')!='pgd-split-component-parent-observation-v1' or observed.get('pid')!=worker['pid']
            or type(observed.get('start_ticks')) is not int or observed['start_ticks']<=0
            or observed.get('exit_status')!=0 or type(observed['exit_status']) is not int
            or observed.get('parent_observation_passed') is not True):
        raise ValueError('Same new completed child/process/resource identity required.')
    for key in ('RSS_samples','peak_observed_RSS_bytes','observed_process_identities'):
        if type(observed.get(key)) is not int or observed[key]<=0:raise ValueError('Actual bounded parent observations required.')
    if (observed['RSS_samples']>2400 or observed['peak_observed_RSS_bytes']>4*2**30 or observed['observed_process_identities']>64
            or type(resource.get('peak_allocator_bytes')) is not int or not 0<resource['peak_allocator_bytes']<=7*2**30
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in
                (worker.get('wall_seconds'),resource.get('wall_seconds'),observed.get('wall_seconds')))
            or not 0<=worker['wall_seconds']<=resource['wall_seconds']<=observed['wall_seconds']<=120
            or any(observed.get(k) is not False for k in ('sampling_is_instantaneous_kernel_or_address_space_quota','proc_child_enumeration_is_exhaustive',
                'scientific_qualified','actual_CUDA_execution_admitted'))):
        raise ValueError('Original parent/component resource/clock bounds changed.')
    return {'kind':'pgd-split-component-parent-binding-v1','role':role,'binding':dict(binding),'pid':worker['pid'],
        'start_ticks':observed['start_ticks'],'metadata_bindings_consistent':True,'canonical_lineage_authenticated':False,
        'actual_CUDA_execution_admitted':False,'scientific_qualified':False,'performance_qualified':False,'training_admitted':False}
