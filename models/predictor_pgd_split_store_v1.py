"""Typed local result store/inventory, without array decoding or process execution.

This covers only the planned results/logs namespace. A future complete scratch
supervisor must ALSO account for inputs/code/cache/proof/free-space/inodes.
"""
import hashlib
import os
from pathlib import Path
import stat
import threading
from models import predictor_pgd_split_plan_v1 as plan

DESIGN_SHA='c5daae30cbe4043cf4d5265de2a2a7420d015ab1e244cc3d74fcc93b5f0186af'


def identity(s):
    return (s.st_dev,s.st_ino,s.st_uid,stat.S_IMODE(s.st_mode),s.st_nlink)


def private(path,*,directory=False,device=None):
    s=path.lstat()
    if ((not stat.S_ISDIR(s.st_mode) if directory else not stat.S_ISREG(s.st_mode))
            or s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode)!=(0o700 if directory else 0o600)
            or not directory and s.st_nlink!=1 or device is not None and s.st_dev!=device):
        raise ValueError('Owned private same-device regular result namespace required: '+str(path))
    return s


class Inventory:
    def __init__(self,root,*,base_bytes,design_bytes):
        if type(design_bytes) is not bytes or hashlib.sha256(design_bytes).hexdigest()!=DESIGN_SHA:
            raise ValueError('Frozen split design bytes required.')
        self.spec=plan.result_specs(plan.base_contract(base_bytes))
        self.root=Path(root).absolute()
        if self.root.resolve(strict=True)!=self.root:raise ValueError('Canonical result root required.')
        s=private(self.root,directory=True);self.root_id=identity(s)[:4];self.device=s.st_dev
        self.directories={'.'}
        for name in self.spec:
            self.directories.update(str(p) for p in Path(name).parents)

    def scan(self):
        if identity(private(self.root,directory=True))[:4]!=self.root_id:raise ValueError('Result root changed.')
        files={};allocated=0
        for current,dirs,names in os.walk(self.root,followlinks=False):
            here=Path(current);rel=str(here.relative_to(self.root))
            if rel not in self.directories:raise ValueError('Unknown result directory: '+rel)
            before=private(here,directory=True,device=self.device);allocated+=before.st_blocks*512
            for name in dirs:private(here/name,directory=True,device=self.device)
            for name in names:
                p=here/name;key=p.relative_to(self.root).as_posix();s=private(p,device=self.device)
                item,plain,partial=plan.classify(key,self.spec)
                if not 0<=s.st_size<=item['max_bytes']:raise ValueError('Typed result extent exceeded: '+key)
                files[key]={'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'identity':identity(s),'mtime_ns':s.st_mtime_ns}
                allocated+=s.st_blocks*512
                if len(files)>1024:raise ValueError('Result file count ceiling.')
            if identity(private(here,directory=True,device=self.device))[:4]!=identity(before)[:4]:raise ValueError('Result directory changed during scan.')
        totals=plan.result_totals({k:v['bytes'] for k,v in files.items()},self.spec)
        if allocated>9*plan.GiB//4:raise ValueError('Result allocated-byte ceiling.')
        if identity(private(self.root,directory=True))[:4]!=self.root_id:raise ValueError('Result root changed during scan.')
        return {'files':files,'totals':totals,'allocated_bytes':allocated,'scientific_qualified':False}


class Store:
    """Serial byte publication only; no tensor serializer or scientific receipt."""
    def __init__(self,inventory):
        self.inventory=inventory;self.root=inventory.root;self.owner=(os.getpid(),threading.get_ident())
        self.poisoned=False;self.busy=False;self.written={}

    def check(self):
        if self.poisoned or (os.getpid(),threading.get_ident())!=self.owner:raise ValueError('Owned unpoisoned serial result writer required.')
        report=self.inventory.scan()
        for name,want in self.written.items():
            if report['files'].get(name)!=want:raise ValueError('Previously committed result metadata changed: '+name)
        return report

    def write_bytes(self,name,data):
        if type(data) is not bytes:raise ValueError('Prebounded plain bytes required; no array/tensor conversion.')
        item,plain,is_partial=plan.classify(name,self.inventory.spec)
        if is_partial or self.busy or len(data)>item['max_bytes']:raise ValueError('One bounded complete result publication required.')
        self.busy=True
        fd=None;parent_fd=None
        try:
            before=self.check();dest=self.root/name
            if name in before['files'] and (not item['mutable'] or name not in self.written):
                raise FileExistsError('Preserve exclusive or foreign result: '+name)
            partial=name+'.partial' if item['partial_allowed'] else name
            if partial in before['files'] and partial!=name:raise FileExistsError('Preserve existing result partial.')
            projected={n:d['bytes'] for n,d in before['files'].items()};projected[partial]=len(data)
            plan.result_totals(projected,self.inventory.spec)
            parent=self.root
            for part in Path(name).parts[:-1]:
                parent=parent/part
                try:
                    parent.mkdir(mode=0o700)
                    ancestor_fd=os.open(parent.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
                    try:os.fsync(ancestor_fd)
                    finally:os.close(ancestor_fd)
                except FileExistsError:pass
                private(parent,directory=True,device=self.inventory.device)
            pst=private(parent,directory=True,device=self.inventory.device)
            parent_fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            if identity(os.fstat(parent_fd))[:4]!=identity(pst)[:4]:raise ValueError('Result parent changed.')
            temp=Path(partial).name
            fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent_fd)
            view=memoryview(data)
            while view:
                n=os.write(fd,view[:1024*1024])
                if n<=0:raise OSError('Incomplete result write.')
                view=view[n:]
            os.fsync(fd);held=os.fstat(fd)
            if held.st_size!=len(data) or held.st_nlink!=1 or stat.S_IMODE(held.st_mode)!=0o600:raise ValueError('Held result extent/mode changed.')
            current=self.inventory.scan()
            for key,want in self.written.items():
                if current['files'].get(key)!=want:raise ValueError('Committed result changed during write: '+key)
            if identity(private(parent,directory=True,device=self.inventory.device))[:4]!=identity(pst)[:4]:raise ValueError('Result parent replaced during write.')
            named=os.stat(temp,dir_fd=parent_fd,follow_symlinks=False)
            if identity(named)!=identity(held) or named.st_size!=held.st_size or named.st_mtime_ns!=held.st_mtime_ns:
                raise ValueError('Named and held result changed.')
            if item['partial_allowed']:
                if name in before['files']:
                    now=os.stat(dest.name,dir_fd=parent_fd,follow_symlinks=False)
                    want=before['files'][name]
                    if identity(now)!=want['identity'] or now.st_size!=want['bytes'] or now.st_mtime_ns!=want['mtime_ns']:
                        raise ValueError('Mutable destination replaced before commit.')
                    os.replace(temp,dest.name,src_dir_fd=parent_fd,dst_dir_fd=parent_fd)
                else:
                    # link refuses an existing destination; never overwrite an
                    # immutable file that appeared between checking and publish.
                    os.link(temp,dest.name,src_dir_fd=parent_fd,dst_dir_fd=parent_fd,follow_symlinks=False)
                    os.unlink(temp,dir_fd=parent_fd)
            os.fsync(parent_fd)
            self.written[name]=self.inventory.scan()['files'][name]
            return {'file':Path(name).name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        except BaseException:
            self.poisoned=True
            raise
        finally:
            if fd is not None:os.close(fd)
            if parent_fd is not None:os.close(parent_fd)
            self.busy=False
