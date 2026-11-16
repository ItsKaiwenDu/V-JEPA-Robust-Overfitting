"""Local-reviewed trusted launch definitions; current tests use fake children only.

No CLI, upload or work occurs on import. This is not the complete remote/helper
package: the outer claim/setup/collection driver and independent scope review remain.
"""
from argparse import Namespace
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import stat
import subprocess
import sys
import time
from types import FunctionType
from models import predictor_pgd_split_bootstrap_v1 as boot
from models import predictor_pgd_split_scratch_v1 as scratch
from models import predictor_pgd_split_watch_v1 as frozen_watch

ROLES=boot.ROLES
HOST={'hostname':'linux10603','system':'Linux','python':'3.11.17','parent':'/var/tmp','filesystem':'xfs','mount':'/',
 'device':66307,'parent_uid':0,'parent_mode':0o1777,'gpu':'NVIDIA GeForce RTX 3070',
 'gpu_uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f','driver':'580.178.04','total_MiB':8192}
WATCH_SHA='ab23159993939318b6d28946a25fe3c4be0427ed6e9732e1530cb8246d6d9cdb'


def make_control(policy_bytes,sources):
    """Concrete source-only control preparation; not authorization or arming."""
    if type(policy_bytes) is not bytes or boot.digest(policy_bytes)!=boot.POLICY_SHA or type(sources) is not dict or set(sources)!=set(boot.SUPERVISORS.values()):
        raise ValueError('Exact policy and all six prepared supervisor source files required.')
    pins={}
    for name,data in sources.items():
        if type(data) is not bytes or not data or len(data)>2*2**20:raise ValueError('Bounded supervisor source bytes required.')
        compile(data,'<UNEXECUTED-SUPERVISOR-'+name+'>','exec')
        pins[name]={'bytes':len(data),'sha256':boot.digest(data)}
    data=boot.encoded({'kind':'pgd-split-supervised-control-v1','policy_text':policy_bytes.decode(),
        'policy_sha256':boot.POLICY_SHA,'supervisor_sources':pins})
    boot.control(data)
    return data


def mount_row(root,text):
    """Exact longest mountpoint, without touching proc during cold checks."""
    rows=[];root=str(root)
    for line in text.splitlines():
        left,sep,right=line.partition(' - ')
        if not sep:continue
        a,b=left.split(),right.split()
        if len(a)<6 or len(b)<3:continue
        point=a[4]
        for old,new in (('\\040',' '),('\\011','\t'),('\\134','\\')):point=point.replace(old,new)
        if root==point or root.startswith(point.rstrip('/')+'/'):
            rows.append({'mount':point,'filesystem':b[0],'options':a[5].split(',')+b[2].split(',')})
    if not rows:raise ValueError('No exact local mount identity.')
    return max(rows,key=lambda r:len(r['mount']))


def target_gate(root,binding):
    # Refuse on the local Mac BEFORE reading /proc or querying any GPU.
    if (platform.node().split('.')[0],platform.system(),platform.python_version())!=(HOST['hostname'],HOST['system'],HOST['python']):
        raise ValueError('Exact authorized university host/runtime required; no local work analogue.')
    root=Path(root)
    if root.parent!=Path('/var/tmp') or not root.name.startswith('vjepa-pgd-split-v1-') or root.resolve(strict=True)!=root:
        raise ValueError('Exact new private scratch namespace required.')
    parent=root.parent.lstat()
    if (parent.st_dev,parent.st_uid,stat.S_IMODE(parent.st_mode))!=(HOST['device'],HOST['parent_uid'],HOST['parent_mode']):
        raise ValueError('Original local scratch parent device/owner/mode changed.')
    if Path.home().stat().st_dev==parent.st_dev:raise ValueError('Shared home cannot substitute for local scratch.')
    with Path('/proc/self/mountinfo').open('rb') as f:text=f.read(2*2**20+1)
    if len(text)>2*2**20:raise ValueError('Bounded mount metadata required.')
    mount=mount_row(root,text.decode())
    if mount['filesystem']!='xfs' or mount['mount']!='/' or 'rw' not in mount['options']:raise ValueError('Original writable local XFS mount required.')
    if list(boot.dir_id(root.lstat()))!=binding['root_identity'] or list(boot.dir_id(parent))!=binding['parent_identity']:
        raise ValueError('Bound root/parent identity changed.')
    return {'kind':'pgd-split-target-readiness-v1','host':HOST['hostname'],'filesystem':'xfs','device':parent.st_dev,
        'scientific_qualified':False,'GPU_exclusivity_claimed':False}


def gpu_gate(allowed_pid=None):
    def query(fields,kind):
        result=subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],stderr=subprocess.PIPE,text=True,timeout=10)
        if len(result)>16*1024:raise ValueError('Bounded GPU status output required.')
        return [r for r in result.splitlines() if r.strip()]
    rows=query('name,uuid,memory.total,memory.used,driver_version','gpu')
    if len(rows)!=1:raise ValueError('Exactly one expected GPU required.')
    parts=[v.strip() for v in rows[0].split(',')]
    if len(parts)!=5:raise ValueError('Exact GPU status columns required.')
    name,uuid,total,used,driver=parts
    if ((name,uuid,driver)!=(HOST['gpu'],HOST['gpu_uuid'],HOST['driver']) or float(total)!=8192
            or not math.isfinite(float(used)) or float(used)<0 or allowed_pid is None and float(used)>512):raise ValueError('GPU/driver/initial-memory guard refused.')
    if allowed_pid is not None and not boot.positive(allowed_pid):raise ValueError('Exact owned allowed process required.')
    active=query('pid,process_name,used_memory','compute-apps')
    for row in active:
        values=[v.strip() for v in row.split(',')]
        if len(values)!=3 or not values[0].isdigit() or int(values[0])!=allowed_pid:
            raise ValueError('Another CUDA compute process is active; leave it alone and refuse.')
    return {'kind':'pgd-split-GPU-observation-v1','allowed_pid':allowed_pid,'GPU_row':rows[0],
        'compute_rows':active,'sampling_is_continuous_exclusivity':False}


def watch_source(data):
    """One exact frozen monitor adapted ONLY for proof/numerical role clocks.

    Component callers keep the original monitor. Numeric/proof RSS is observed,
    not assigned the component-only 4 GiB ceiling. No function is executed here.
    """
    if type(data) is not bytes or boot.digest(data)!=WATCH_SHA:raise ValueError('Frozen PID-handle parent source required.')
    module=ast.parse(data);f=copy.deepcopy(next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='watch'))
    f.name='watch_stage';f.args.kwonlyargs.append(ast.arg(arg='role'));f.args.kw_defaults.append(None)
    changed={'time':0,'samples':0,'rss':0,'kind':0}
    class Edit(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='pgd-split-component-parent-observation-v1':changed['kind']+=1;return ast.copy_location(ast.Constant('pgd-split-stage-parent-observation-v1'),n)
            return n
        def visit_BinOp(self,n):
            if ast.dump(n,include_attributes=False)==ast.dump(ast.parse('start+120',mode='eval').body,include_attributes=False):
                changed['time']+=1;return ast.copy_location(ast.parse('start+role_seconds',mode='eval').body,n)
            return self.generic_visit(n)
        def visit_Compare(self,n):
            text=ast.unparse(n)
            if text=='count > 2400':changed['samples']+=1;return ast.copy_location(ast.parse('count > role_seconds*10',mode='eval').body,n)
            if text=='rss > RSS_BYTES':changed['rss']+=1;return ast.copy_location(ast.parse("role in ('native','forward-preserving') and rss > RSS_BYTES",mode='eval').body,n)
            return self.generic_visit(n)
    f=Edit().visit(f)
    assert changed=={'time':1,'samples':1,'rss':1,'kind':1}
    f.body[1:1]=ast.parse("if role not in ('proof','numerical'):raise ValueError('Only the two explicit non-component roles.')\nrole_seconds={'proof':60,'numerical':1680}[role]").body
    for n in ast.walk(f):
        if isinstance(n,ast.Return) and isinstance(n.value,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='parent_observation_passed' for k in n.value.keys):
            n.value.keys.extend([ast.Constant('role'),ast.Constant('RSS_ceiling_enforced')]);n.value.values.extend([ast.Name('role',ast.Load()),ast.Constant(False)])
    return ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),changed


def monitor_for(role):
    # Prepare the exact monitor BEFORE creating/releasing any child. A source or
    # compilation refusal must not leave released work outside a monitor finally.
    if role in ('native','forward-preserving'):return frozen_watch.watch
    if role not in ('proof','numerical'):raise ValueError('Exact monitor role required.')
    source=Path(frozen_watch.__file__).read_bytes();tree,_=watch_source(source)
    namespace=dict(frozen_watch.__dict__)
    exec(compile(tree,'<PINNED-PROOF-NUMERICAL-PARENT-MONITOR>','exec'),namespace)
    function=namespace['watch_stage']
    return lambda child,lease,**kw:function(child,lease,role=role,**kw)


# Python reads and checks the bootstrap bytes before executing those exact bytes.
# The bootstrap imports only stdlib until its source-pinned finder is installed.
CHILD_DRIVER='''
import base64,hashlib,json,os,stat,sys
path,want,count,plan64,gate=sys.argv[1:]
limit=int(count)
if not 0<limit<=2*2**20:raise ValueError('Bootstrap extent refused.')
s=os.lstat(path)
if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode)!=0o600 or s.st_size!=limit:raise ValueError('Private exact bootstrap required.')
with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
    before=os.fstat(f.fileno());data=f.read(limit+1);after=os.fstat(f.fileno())
if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(s.st_dev,s.st_ino,after.st_size,after.st_mtime_ns) or len(data)!=limit or hashlib.sha256(data).hexdigest()!=want:raise ValueError('Bootstrap hash/identity changed.')
raw=base64.b64decode(plan64,validate=True)
if len(raw)>16*1024:raise ValueError('Bounded launch envelope required.')
plan=json.loads(raw)
ns={'__name__':'_PINNED_SPLIT_BOOTSTRAP_','__file__':path}
exec(compile(data,path,'exec'),ns)
ns['wait_release'](int(gate),plan['parent_pid'],plan['deadline'])
sys.exit(ns['child_entry'](raw))
'''


def _metadata_write(guard,name,value,*,replace,phase='capture'):
    d=guard.layout.classify(name);data=boot.encoded(value)
    if d['category']!='metadata' or len(data)>d['max_bytes'] or replace and not d['mutable']:raise ValueError('Bounded declared supervisor metadata publication required.')
    guard.scan(phase=phase)
    path=guard.root/name;part=path.with_name(path.name+'.partial')
    with os.fdopen(os.open(part,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    if replace:os.replace(part,path)
    else:os.link(part,path);os.unlink(part)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
    return boot.digest(data)


def environment(root):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    if visible not in ('0',HOST['gpu_uuid']):raise ValueError('Explicit original single GPU visibility required.')
    keep={k:v for k,v in os.environ.items() if k in ('PATH','HOME','USER','LOGNAME','LANG','LC_ALL','LD_LIBRARY_PATH','VIRTUAL_ENV')}
    keep.update(CUDA_VISIBLE_DEVICES=visible,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),CUDA_CACHE_PATH=str(root/'cache/cuda'),CUDA_CACHE_MAXSIZE=str(32*2**20))
    return keep


class SystemDriver:
    """Concrete future OS bindings. None of these launch APIs runs in local checks."""
    def __init__(self,owner):self.owner=owner;self.guard=None;self.sources={};self.current=None;self.session_sha256=None
    def prepare(self):
        o=self.owner;target_gate(o.root,o.binding);boot.require_namespace(o.root,o.binding,o.csha);gpu_gate()
        if signal.getsignal(signal.SIGCHLD) is not signal.SIG_DFL:raise ValueError('Default non-reaping SIGCHLD state required before gated launch.')
        ib,isha=boot.installation(o.c,o.csha)
        if boot.private_read(o.root,o.binding,'control.json',256*1024)!=o.cbytes or boot.private_read(o.root,o.binding,'authorization.json',64*1024)!=o.abytes:
            raise ValueError('Scope files differ from parent-reviewed bytes.')
        if boot.private_read(o.root,o.binding,'installation.json',64*1024)!=ib:raise ValueError('Source installation record changed.')
        p=o.policy
        layout=scratch.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),ib,isha)
        self.guard=scratch.Guard(o.root,o.binding,layout)
        self.guard.verify_assets(phase='setup');self.guard.proof_readiness()
        # Require APIs before launch, without opening a process handle yet.
        frozen_watch.system_pidfd_api()
        return {'source_installation_metadata_verified':True,'proof_execution_admitted':False}
    def save(self,state,*,replace):
        if replace:
            if self.session_sha256 is None or boot.digest(self.guard.read('session.json',128*1024))!=self.session_sha256:
                raise ValueError('Only this parent may replace its own previously committed session.')
        phase='proof' if state.get('current') is not None and state['current']['role']=='proof' else 'capture'
        result=_metadata_write(self.guard,'session.json',state,replace=replace,phase=phase)
        self.session_sha256=result
        return result
    def launch(self,role,plan):
        import base64
        if role not in ROLES or plan.get('role')!=role:raise ValueError('Exact parent/child role before launch required.')
        monitor=monitor_for(role)
        o=self.owner;self.current=role;child=None;lease=None;readfd=writefd=logfd=None;monitored=False
        start=plan['launch_started'];deadline=plan['deadline'];work=o.work_deadline
        def gate():
            self.guard.scan(phase='proof' if role=='proof' else 'capture')
            gpu_gate(child.pid if child is not None else None)
        try:
            if o.clock()>=deadline:raise TimeoutError('Original launch clock exhausted.')
            readfd,writefd=os.pipe2(os.O_CLOEXEC)
            log=o.root/'capture/logs'/(role+'.log');logfd=os.open(log,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            desc=o.c['supervisor_sources']['bootstrap.py']
            command=[sys.executable,'-I','-B','-u','-c',CHILD_DRIVER,str(o.root/'supervisor/bootstrap.py'),desc['sha256'],str(desc['bytes']),base64.b64encode(boot.encoded(plan)).decode(),str(readfd)]
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=logfd,stderr=logfd,cwd=str(o.root),env=environment(o.root),
                start_new_session=True,close_fds=True,pass_fds=(readfd,),shell=False,umask=0o077)
            os.close(readfd);readfd=None;os.close(logfd);logfd=None
            reader=frozen_watch.ProcReader();row=reader.identity(child.pid)
            if row['parent_pid']!=os.getpid() or row['session_pid']!=child.pid or row['group_pid']!=child.pid:raise ValueError('New root child identity refused before release.')
            lease=frozen_watch.PidfdLease(child.pid,row['start_ticks'],os.getpid(),child.pid,reader)
            gate()
            if o.clock()>=deadline:raise TimeoutError('Handle-binding/gates consumed the original phase.')
            if os.write(writefd,b'G')!=1:raise OSError('Incomplete child release.')
            os.close(writefd);writefd=None
            kwargs={'sample':reader.sample,'lease_factory':lambda pid,birth,parent,session:frozen_watch.PidfdLease(pid,birth,parent,session,reader),
                'gate':gate,'phase_deadline':deadline,'work_deadline':work,'launch_started':start}
            monitored=True
            observed=monitor(child,lease,**kwargs)
            if not lease.closed:raise ValueError('Monitor returned without completing owned handle cleanup.')
            return observed
        finally:
            # Before release, closing the only parent writer gives the waiting
            # child EOF. Never fall back to a numeric-PID signal on binding failure.
            for fd in (readfd,writefd,logfd):
                if fd is not None:os.close(fd)
            needs_owned_cleanup=lease is not None and not lease.closed
            if needs_owned_cleanup:
                try:lease.send(signal.SIGKILL)
                finally:lease.close()
            if child is not None and (not monitored or needs_owned_cleanup):
                try:child.wait(timeout=max(0,min(2,work-o.clock())))
                except subprocess.TimeoutExpired:pass # Child has EOF plus its original deadline; no model was released without a held handle.
            self.current=None
    def verify(self,role,plan,observed):
        o=self.owner;g=self.guard
        if observed.get('pid') is None or observed.get('exit_status')!=0 or observed.get('parent_observation_passed') is not True:raise ValueError('Successful owned child observation required.')
        if role=='proof':
            raw=g.read('allocation-result-v2.json',16*1024);receipt=g.verify_proof(boot.digest(raw))
        else:
            # Imported only through the already-installed pinned finder in the
            # future parent. Numerical readers use saved values, no new objective.
            from models import _pgd_split_io_v1 as io
            if getattr(io,'_STORE') is None:io.bind(o.root/'capture')
            if role=='numerical':
                from models import _pgd_split_records_v1 as records
                receipt=records.verify_worker(o.root/'capture/results/numerical',control_sha256=o.csha)
                header=boot.parsed(g.read('capture/results/run-status.json',256*1024))
                if header['pid']!=observed['pid']:raise ValueError('Numerical header belongs to another child.')
                self.sources={key:boot.digest(g.read(name,cap)) for key,name,cap in (
                    ('numerical_sha256','capture/results/run-status.json',256*1024),
                    ('original_sha256','capture/results/numerical/objective-0/numerical.json',256*1024),
                    ('boundary_sha256','capture/results/boundary/boundary.json',128*1024))}
                if receipt['worker_sha256']!=self.sources['numerical_sha256']:raise ValueError('Numerical receipt source changed.')
            else:
                from models import predictor_pgd_split_probe_job_v1 as job
                prefix='capture/results/components/'+role+'/'
                marker_bytes=g.read(prefix+'complete.json',16*1024);marker=boot.parsed(marker_bytes)
                progress=g.read(prefix+'progress.jsonl',64*1024)
                binding={'design_sha256':self.owner.policy['scope']['design_sha256'],'control_sha256':o.csha,
                    'boundary_sha256':self.sources['boundary_sha256'],'probe_index':int(role=='forward-preserving')}
                job.read_completion(o.root/prefix,role=role,binding=binding,marker_sha256=boot.digest(marker_bytes),progress_sha256=boot.digest(progress))
                worker=boot.parsed(g.read(prefix+'worker.json',256*1024));resource=boot.parsed(g.read(prefix+'resource.json',16*1024))
                receipt=job.bind_parent_metadata(marker,worker,resource,observed,role=role,binding=binding)
            _metadata_write(g,'parent-'+role+'.json',observed,replace=False)
        g.scan(phase='capture',settled=True)
        return {'role':role,'pid':observed['pid'],'start_ticks':observed['start_ticks'],'saved_gates_returned':True,
            'receipt':receipt,'source_hashes':dict(self.sources),'scientific_qualified':False,'training_admitted':False}


class Coordinator:
    """One fixed role chain. Tests replace SystemDriver only in a fake namespace."""
    def __init__(self,root,binding,control_bytes,authorization_bytes,*,connection_started,clock=time.monotonic):
        self.c,self.policy,self.csha=boot.control(control_bytes);boot.authorization(authorization_bytes,self.csha)
        self.cbytes,self.abytes=control_bytes,authorization_bytes;self.root=Path(root);self.binding=binding;self.clock=clock
        if not boot.finite(connection_started) or not boot.finite(clock()) or clock()<connection_started:raise ValueError('Original continuing connection clock required.')
        self.started=connection_started;self.work_deadline=connection_started+2400;self.connection_deadline=connection_started+3000
        self.last=connection_started;self.used=False;self.claimed=False;self.driver=SystemDriver(self);self.receipts=[];self.state=None
    def check(self,deadline=None):
        now=self.clock()
        if not boot.finite(now) or now<self.last or now>=min(self.work_deadline,self.connection_deadline,deadline if deadline is not None else self.work_deadline):
            raise TimeoutError('Original continuing clock exhausted or changed; no reset.')
        self.last=now;return now
    def run_work(self):
        if self.used:raise ValueError('One coordinator work attempt; no retries.')
        self.used=True
        self.state={'kind':'pgd-split-parent-session-v1','control_sha256':self.csha,'parent_pid':os.getpid(),
            'connection_started':self.started,'work_deadline':self.work_deadline,'connection_deadline':self.connection_deadline,
            'phase':'preparing','completed':[],'current':None,'receipts':[],'complete':False,'scientific_qualified':False,'training_admitted':False}
        try:
            self.check(self.started+180);self.driver.prepare();self.check(self.started+180)
            # Exclusive session creation is the durable per-root no-retry claim.
            self.driver.save(self.state,replace=False);self.claimed=True
            for index,role in enumerate(ROLES):
                start=self.check();later=sum(boot.SECONDS[r] for r in ROLES[index+1:]);deadline=min(start+boot.SECONDS[role],self.work_deadline-240-later)
                self.check(deadline)
                sources=dict(self.driver.sources) if role in ('native','forward-preserving') else {}
                current={'role':role,'nonce':os.urandom(32).hex(),'launch_started':start,'deadline':deadline,'source_hashes':sources}
                self.state.update(phase='running',current=current)
                session_sha=self.driver.save(self.state,replace=True)
                plan=dict(current,kind='pgd-split-child-launch-v1',control_sha256=self.csha,authorization_sha256=boot.digest(self.abytes),
                    parent_pid=os.getpid(),root_binding=self.binding,work_deadline=self.work_deadline,connection_deadline=self.connection_deadline,session_sha256=session_sha)
                boot.validate_plan(plan,self.csha,self.state,self.check(deadline))
                observed=self.driver.launch(role,plan);self.check(deadline)
                if any(old['pid']==observed.get('pid') for old in self.receipts):raise ValueError('Distinct fresh child PID required; no reuse.')
                receipt=self.driver.verify(role,plan,observed);self.check(deadline)
                if (receipt.get('role')!=role or receipt.get('pid')!=observed.get('pid') or receipt.get('start_ticks')!=observed.get('start_ticks')
                        or receipt.get('saved_gates_returned') is not True):raise ValueError('Same newly completed role/receipt required.')
                self.receipts.append(receipt);self.state['receipts']=list(self.receipts)
                self.state['completed'].append(role);self.state.update(current=None,phase='between-roles')
                self.driver.save(self.state,replace=True);self.check(deadline)
            self.state.update(phase='await-collection',collection_pending=True)
            self.driver.save(self.state,replace=True);self.check()
            return dict(self.state)
        except BaseException as error:
            self.state.update(phase='failed',complete=False,error_type=type(error).__name__,error=str(error)[:1024])
            if self.claimed:
                try:self.driver.save(self.state,replace=True)
                except BaseException as extra:error.add_note('Parent failure receipt also failed: '+type(extra).__name__)
            raise


def dispatch_child(root,binding,cbytes,abytes,plan):
    """Concrete future dispatch AFTER the pipe release and authenticated source setup."""
    import resource
    c,policy,csha=boot.control(cbytes);boot.authorization(abytes,csha);target_gate(root,binding);gpu_gate()
    ib,isha=boot.installation(c,csha)
    if boot.private_read(root,binding,'installation.json',64*1024)!=ib:raise ValueError('Child installation bytes changed.')
    layout=scratch.Layout(policy['scratch_contract_text'].encode(),policy['base_text'].encode(),policy['design_text'].encode(),ib,isha)
    guard=scratch.Guard(root,binding,layout);guard.verify_assets(phase='setup' if plan['role']=='proof' else 'capture')
    if not boot.finite(time.monotonic()) or time.monotonic()>=plan['deadline']:raise TimeoutError('Original child phase exhausted before work.')
    limit=3*2**30 if plan['role']=='proof' else 16*2**20
    resource.setrlimit(resource.RLIMIT_FSIZE,(limit,limit))
    if plan['role']=='proof':
        guard.proof_readiness()
        from models.scratch_allocation_v2 import bounded_write_proof
        bounded_write_proof(root,deadline=plan['deadline'])
        return 0
    from models import _pgd_split_entry_v1 as runtime
    if plan['role']=='numerical':
        existing=Path.home()/'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting'
        result,receipt=runtime.numerical_entry(Namespace(result_root=str(root/'capture'),control_sha256=csha,packet=root/'code',
            upstream=existing/'external/vjepa2',checkpoint=existing/'checkpoints/vitl.pt',proposal=root/'configs/base.json',
            output=root/'capture/results/numerical',device='cuda',tiny=False))
        if result.get('passed') is not True:raise ValueError('Numerical child did not pass.')
    else:
        runtime.component_entry(Namespace(result_root=str(root/'capture'),control_sha256=csha,role=plan['role'],deadline=plan['deadline'],**plan['source_hashes']))
    return 0
