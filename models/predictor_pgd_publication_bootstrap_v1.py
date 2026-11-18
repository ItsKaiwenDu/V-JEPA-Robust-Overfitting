"""LOCAL-reviewed bootstrap definitions; no CLI or work runs on import.

Future authorized parent/child namespaces may use these concrete standard-library
bindings. Current checks use metadata fixtures, fake pipes and inert source only.
"""
from argparse import Namespace
import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path,PurePosixPath
import platform
import select
import signal
import stat
import sys
import time

POLICY_SHA='9f1d10b01fc48e09711084be515f1193fa5140244e1d07eb1848b0fde5164920'
ROLES=('proof','numerical','native','forward-preserving')
SECONDS={'proof':60,'numerical':1680,'native':120,'forward-preserving':120}
SUPERVISORS={'models.predictor_pgd_split_bootstrap_v1':'bootstrap.py',
 'models.predictor_pgd_split_coordinator_v1':'predictor_pgd_split_coordinator_v1.py',
 'models.predictor_pgd_split_scratch_v1':'predictor_pgd_split_scratch_v1.py',
 'models.predictor_pgd_split_collection_v1':'predictor_pgd_split_collection_v1.py',
 'models.predictor_pgd_split_watch_v1':'predictor_pgd_split_watch_v1.py',
 'models.scratch_allocation_v2':'scratch_allocation_v2.py'}
_CHILD_USED=False


def encoded(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(data):return hashlib.sha256(data).hexdigest()
def sha(v):return type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v)
def finite(v):return type(v) in (int,float) and math.isfinite(v)
def positive(v):return type(v) is int and v>0

def parsed(data):
    def pairs(rows):
        d={}
        for k,v in rows:
            if k in d:raise ValueError('Duplicate launch metadata key.')
            d[k]=v
        return d
    def bad(v):raise ValueError('Nonfinite launch metadata: '+v)
    return json.loads(data,object_pairs_hook=pairs,parse_constant=bad)

def safe(name):
    if type(name) is not str:raise ValueError('Canonical private source path required.')
    p=PurePosixPath(name)
    if (not name or len(name)>768 or p.is_absolute() or '..' in p.parts or '.' in p.parts
            or p.as_posix()!=name or '\\' in name):raise ValueError('Canonical private source path required.')
    return p.parts

def dir_id(s):return (s.st_dev,s.st_ino,s.st_uid,stat.S_IMODE(s.st_mode))
def file_id(s):return (*dir_id(s),s.st_nlink,s.st_size,s.st_mtime_ns)

def private_read(root,binding,name,cap):
    """Read one exact anchored private file; never follow a source symlink."""
    root=Path(root);parts=safe(name)
    if (not root.is_absolute() or root.resolve(strict=True)!=root or str(root)!=binding['root']
            or str(root.parent)!=binding['parent'] or list(dir_id(root.lstat()))!=binding['root_identity']
            or list(dir_id(root.parent.lstat()))!=binding['parent_identity']
            or binding['root_identity'][2:]!=[os.getuid(),0o700]
            or type(cap) is not int or not 0<=cap<=2*2**20):raise ValueError('Bound private source root/cap required.')
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);parents=[]
    try:
        if list(dir_id(os.fstat(fd)))!=binding['root_identity']:raise ValueError('Held source root changed.')
        for part in parts[:-1]:
            before=os.stat(part,dir_fd=fd,follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode) or before.st_dev!=binding['root_identity'][0] or before.st_uid!=os.getuid() or stat.S_IMODE(before.st_mode)!=0o700:
                raise ValueError('Private source ancestor required.')
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            if dir_id(os.fstat(child))!=dir_id(before):os.close(child);raise ValueError('Source ancestor changed.')
            parents.append((fd,part,before));fd=child
        before=os.stat(parts[-1],dir_fd=fd,follow_symlinks=False)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_uid!=os.getuid()
                or before.st_dev!=binding['root_identity'][0] or stat.S_IMODE(before.st_mode)!=0o600 or not 0<=before.st_size<=cap):
            raise ValueError('Private bounded single-link source file required.')
        filefd=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
        try:
            if file_id(os.fstat(filefd))!=file_id(before):raise ValueError('Source changed before read.')
            chunks=[];size=0
            while size<=cap:
                data=os.read(filefd,min(2**20,cap+1-size))
                if not data:break
                chunks.append(data);size+=len(data)
            if file_id(os.fstat(filefd))!=file_id(before):raise ValueError('Source changed during read.')
        finally:os.close(filefd)
        if size!=before.st_size or file_id(os.stat(parts[-1],dir_fd=fd,follow_symlinks=False))!=file_id(before):raise ValueError('Named source changed during read.')
        for parent,part,want in parents:
            if dir_id(os.stat(part,dir_fd=parent,follow_symlinks=False))!=dir_id(want):raise ValueError('Source ancestry changed during read.')
        if root.resolve(strict=True)!=root or list(dir_id(root.lstat()))!=binding['root_identity']:raise ValueError('Source root changed after read.')
        return b''.join(chunks)
    finally:
        os.close(fd)
        for parent,_,_ in reversed(parents):os.close(parent)


def control(data):
    if type(data) is not bytes or len(data)>256*1024:raise ValueError('Bounded concrete control bytes required.')
    c=parsed(data)
    if (type(c) is not dict or set(c)!={'kind','policy_text','policy_sha256','supervisor_sources'}
            or c['kind']!='pgd-split-supervised-control-v1' or c['policy_sha256']!=POLICY_SHA
            or type(c['policy_text']) is not str or digest(c['policy_text'].encode())!=POLICY_SHA):raise ValueError('Exact new supervised split control required.')
    if encoded(c)!=data:raise ValueError('Canonical immutable control bytes required.')
    policy=parsed(c['policy_text']);scope=policy['scope'];s=parsed(policy['scratch_contract_text'])
    if (type(c['supervisor_sources']) is not dict or set(c['supervisor_sources'])!=set(SUPERVISORS.values())
            or digest(policy['scratch_contract_text'].encode())!=policy['scratch_contract_sha256']
            or digest(policy['design_text'].encode())!=scope['design_sha256']
            or digest(policy['base_text'].encode())!=policy['base_sha256']):
        raise ValueError('Fixed source scope/coverage changed.')
    total=0
    for name,d in c['supervisor_sources'].items():
        if (type(d) is not dict or set(d)!={'bytes','sha256'} or not positive(d['bytes']) or d['bytes']>2*2**20 or not sha(d['sha256'])):
            raise ValueError('Bounded supervisor source descriptor required.')
        if name in policy['fixed_support'] and d['sha256']!=policy['fixed_support'][name]:raise ValueError('Frozen supervisor source changed.')
        total+=d['bytes']
    if total>2*2**20:raise ValueError('Supervisor source total exceeds existing code bound.')
    return c,policy,digest(data)


def authorization(data,control_sha256):
    if type(data) is not bytes or len(data)>64*1024:raise ValueError('One bounded scoped authorization required.')
    a=parsed(data)
    if (type(a) is not dict or set(a)!={'kind','approved','control_sha256','maximum_new_attempts','user_basis'}
            or a['kind']!='pgd-split-supervised-university-exception-v1' or a['approved'] is not True
            or a['control_sha256']!=control_sha256 or type(a['maximum_new_attempts']) is not int or a['maximum_new_attempts']!=1
            or type(a['user_basis']) is not str or not 1<=len(a['user_basis'])<=8192):raise ValueError('One new exact supervised split grant required; no old scope reuse.')
    return a


def installation(c,sha256):
    # Derive AFTER control hashing to avoid a self-referential control/install hash.
    data=encoded({'kind':'pgd-split-scratch-installation-v1','control_sha256':sha256,'files':c['supervisor_sources']})
    return data,digest(data)


class PinnedImporter(importlib.abc.MetaPathFinder,importlib.abc.Loader):
    """Only fixed project names from held, verified bytes; no cwd/package fallback."""
    def __init__(self,root,binding,c,policy):
        self.kind='pgd-split-pinned-importer-v1';self.root=Path(root);self.binding=binding;self.control_sha256=digest(encoded(c));self.files={};self.loaded=set();self.installed=False
        mapping={}
        for path,d in parsed(policy['scratch_contract_text'])['runtime_files'].items():
            if not path.endswith('.py'):raise ValueError('Only declared Python runtime source.')
            name=path[:-3].replace('/','.')
            if name.endswith('.__init__'):name=name[:-len('.__init__')]
            if name in mapping:raise ValueError('Duplicate installed project module name.')
            mapping[name]=('overlay/'+path,d)
        for name,path in SUPERVISORS.items():
            if name in mapping:raise ValueError('Supervisor/runtime namespace collision.')
            mapping[name]=('supervisor/'+path,c['supervisor_sources'][path])
        if len(mapping)!=69:raise ValueError('Exact 63 runtime and six supervisor names required.')
        for name,(path,d) in mapping.items():
            data=private_read(root,binding,path,d['bytes'])
            if len(data)!=d['bytes'] or digest(data)!=d['sha256']:raise ValueError('Source hash/extent changed before import: '+path)
            self.files[name]=(path,data,path.endswith('/__init__.py'))
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] not in ('models','attacks'):return None
        if fullname not in self.files:raise ModuleNotFoundError('Unpinned project import refused: '+fullname)
        if fullname in self.loaded:raise ImportError('Project module reload refused: '+fullname)
        relative,data,package=self.files[fullname]
        return importlib.util.spec_from_loader(fullname,self,origin=str(self.root/relative),is_package=package)
    def create_module(self,spec):return None
    def exec_module(self,module):
        name=module.__name__
        if name not in self.files or name in self.loaded:raise ImportError('One pinned module execution only.')
        self.loaded.add(name);relative,data,package=self.files[name]
        module.__file__=str(self.root/relative);module.__cached__=None
        if package:module.__path__=[str((self.root/relative).parent)]
        exec(compile(data,module.__file__,'exec'),module.__dict__)
    def install(self):
        if self.installed or any(n.split('.')[0] in ('models','attacks') for n in sys.modules):raise ValueError('Fresh project module namespace required; no reuse.')
        self.installed=True;sys.meta_path.insert(0,self)
        return self


def validate_plan(plan,csha,session,now):
    keys={'kind','control_sha256','authorization_sha256','role','nonce','parent_pid','root_binding','launch_started','deadline','work_deadline','connection_deadline','session_sha256','source_hashes'}
    if (type(plan) is not dict or set(plan)!=keys or plan['kind']!='pgd-split-child-launch-v1' or plan['control_sha256']!=csha
            or plan['role'] not in ROLES or not all(sha(plan[k]) for k in ('control_sha256','authorization_sha256','nonce','session_sha256'))
            or not positive(plan['parent_pid']) or not all(finite(plan[k]) for k in ('launch_started','deadline','work_deadline','connection_deadline'))
            or not finite(now) or not plan['launch_started']<=now<plan['deadline']<=min(plan['launch_started']+SECONDS[plan['role']],plan['work_deadline'])
            or plan['work_deadline']>plan['connection_deadline']):raise ValueError('Original role/launch-origin/deadline binding required.')
    if (type(plan['source_hashes']) is not dict or set(plan['source_hashes'])!=(set(('numerical_sha256','original_sha256','boundary_sha256')) if plan['role'] in ('native','forward-preserving') else set())
            or any(not sha(v) for v in plan['source_hashes'].values())):raise ValueError('Exact immutable numerical source hashes required.')
    if (session.get('kind')!='pgd-split-parent-session-v1' or session.get('control_sha256')!=csha
            or session.get('parent_pid')!=plan['parent_pid'] or session.get('current')!={k:plan[k] for k in ('role','nonce','launch_started','deadline','source_hashes')}
            or session.get('work_deadline')!=plan['work_deadline'] or session.get('connection_deadline')!=plan['connection_deadline']
            or session.get('completed')!=list(ROLES[:ROLES.index(plan['role'])]) or session.get('phase')!='running' or session.get('complete') is not False or not finite(session.get('connection_started'))
            or session['work_deadline']!=session['connection_started']+2400 or session['connection_deadline']!=session['connection_started']+3000):
        raise ValueError('Exact fresh serial parent/session role required.')
    return plan


def wait_release(fd,parent_pid,deadline):
    """Future child waits before bootstrap/model imports; tests use fake fd APIs."""
    if not positive(fd) or not positive(parent_pid) or os.getppid()!=parent_pid or not finite(deadline):raise ValueError('Owned inherited release descriptor required.')
    if not stat.S_ISFIFO(os.fstat(fd).st_mode):raise ValueError('Anonymous release pipe required.')
    remaining=deadline-time.monotonic()
    if not 0<remaining<=1680:raise TimeoutError('Original child deadline exhausted before release.')
    signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,remaining)
    try:
        ready,_,_=select.select([fd],[],[],remaining)
        if ready!=[fd] or os.read(fd,2)!=b'G' or os.getppid()!=parent_pid or time.monotonic()>=deadline:
            raise ValueError('Parent did not release this exact still-owned child.')
    finally:os.close(fd)


def child_entry(plan_bytes):
    """Uninvoked concrete child entry; future reviewed driver performs release first."""
    global _CHILD_USED
    if _CHILD_USED:raise ValueError('One child entry; no retries.')
    _CHILD_USED=True
    if type(plan_bytes) is not bytes or len(plan_bytes)>16*1024:raise ValueError('Bounded child launch metadata required.')
    plan=parsed(plan_bytes);root=Path(plan['root_binding']['root']);binding=plan['root_binding']
    cbytes=private_read(root,binding,'control.json',256*1024);c,p,csha=control(cbytes)
    abytes=private_read(root,binding,'authorization.json',64*1024);authorization(abytes,csha)
    if digest(abytes)!=plan['authorization_sha256']:raise ValueError('Authorization bytes changed.')
    sbytes=private_read(root,binding,'session.json',128*1024)
    if digest(sbytes)!=plan['session_sha256']:raise ValueError('Parent session changed before dispatch.')
    validate_plan(plan,csha,parsed(sbytes),time.monotonic())
    if os.getppid()!=plan['parent_pid'] or os.getsid(0)!=os.getpid() or os.getpgrp()!=os.getpid():raise ValueError('Fresh still-owned child session required.')
    PinnedImporter(root,binding,c,p).install()
    from models import predictor_pgd_split_coordinator_v1 as coordinator
    if os.getppid()!=plan['parent_pid'] or time.monotonic()>=plan['deadline']:
        raise TimeoutError('Original owner/phase changed during source installation.')
    return coordinator.dispatch_child(root,binding,cbytes,abytes,plan)


def require_namespace(root,binding,csha):
    candidates=[v for v in sys.meta_path if getattr(v,'kind',None)=='pgd-split-pinned-importer-v1'
        and getattr(v,'root',None)==Path(root) and getattr(v,'binding',None)==binding
        and getattr(v,'control_sha256',None)==csha and getattr(v,'installed',False)]
    if len(candidates)!=1:raise ValueError('One trusted fresh pinned project importer required.')
    finder=candidates[0]
    for name,module in sys.modules.items():
        if name.split('.')[0] not in ('models','attacks'):continue
        if (name not in finder.files or getattr(getattr(module,'__spec__',None),'loader',None) is not finder
                or getattr(module,'__file__',None)!=str(Path(root)/finder.files[name][0])):
            raise ValueError('Preloaded/foreign project namespace refused: '+name)
    return finder

_STARTUP_PACKET_IMPORTS = {'datasets': ('code/datasets/__init__.py', {'bytes': 67, 'sha256': 'd81c61d6627e4e28075736950978c3cca23843d93fb213454458bd32f5357cf1'}, True), 'datasets.prepared': ('code/datasets/prepared.py', {'bytes': 3784, 'sha256': '531dcb4e4e69e3a59890e020940a6a8d0459d47100154730eabc6d267c5e2bb5'}, False)}
_StartupBaseImporter = PinnedImporter
_StartupBaseRequireNamespace = require_namespace

class PinnedImporter(_StartupBaseImporter):

    def __init__(self, root, binding, c, policy):
        super().__init__(root, binding, c, policy)
        packet = parsed(policy['scratch_contract_text'])['packet']['files']
        self.packet_imports = {}
        self.packet_import_policy = 'pgd-split-existing-packet-imports-v3'
        for name, (path, descriptor, package) in _STARTUP_PACKET_IMPORTS.items():
            if packet.get(path.removeprefix('code/')) != descriptor or name in self.files:
                raise ValueError('Exact existing packet dependency descriptor required.')
            self.packet_imports[name] = (path, dict(descriptor), package)

    def _packet_source(self, name):
        path, descriptor, package = self.packet_imports[name]
        data = private_read(self.root, self.binding, path, descriptor['bytes'])
        if len(data) != descriptor['bytes'] or digest(data) != descriptor['sha256']:
            raise ValueError('Packet dependency hash/extent changed: ' + name)
        return (path, data, package)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] != 'datasets':
            return super().find_spec(fullname, path, target)
        if fullname not in self.packet_imports:
            raise ModuleNotFoundError('Unpinned project import refused: ' + fullname)
        if fullname in self.loaded or target is not None:
            raise ImportError('Project module reload refused: ' + fullname)
        if fullname != 'datasets':
            parent = sys.modules.get('datasets')
            _startup_check_module(self, 'datasets', parent)
            expected = [str(self.root / 'code/datasets')]
            if path != expected:
                raise ImportError('Exact packet package search path required.')
        elif path is not None:
            raise ImportError('Top-level packet package cannot use an ambient path.')
        self.files[fullname] = self._packet_source(fullname)
        relative, data, package = self.files[fullname]
        return importlib.util.spec_from_loader(fullname, self, origin=str(self.root / relative), is_package=package)

    def exec_module(self, module):
        name = module.__name__
        if name in self.packet_imports:
            current = self._packet_source(name)
            if name not in self.files or current != self.files[name]:
                raise ImportError('Packet source changed after resolution.')
        return super().exec_module(module)

    def install(self):
        if any((n.split('.')[0] == 'datasets' for n in sys.modules)):
            raise ValueError('Fresh pinned datasets namespace required.')
        return super().install()

def _startup_check_module(finder, name, module):
    if module is None or name not in finder.files or name not in finder.loaded:
        raise ImportError('One already-loaded pinned project parent required: ' + name)
    path, data, package = finder.files[name]
    expected = str(finder.root / path)
    if getattr(module, '__file__', None) != expected or getattr(module, '__loader__', None) is not finder or getattr(getattr(module, '__spec__', None), 'loader', None) is not finder or (getattr(getattr(module, '__spec__', None), 'origin', None) != expected) or (package and getattr(module, '__path__', None) != [str((finder.root / path).parent)]):
        raise ImportError('Project module origin/loader/package path changed: ' + name)

def require_namespace(root, binding, csha):
    finder = _StartupBaseRequireNamespace(root, binding, csha)
    if getattr(finder, 'packet_import_policy', None) != 'pgd-split-existing-packet-imports-v3' or getattr(finder, 'packet_imports', None) != _STARTUP_PACKET_IMPORTS:
        raise ValueError('Reviewed packet-aware importer required.')
    for name, module in sys.modules.items():
        if name.split('.')[0] in ('models', 'attacks', 'datasets'):
            _startup_check_module(finder, name, module)
    return finder
