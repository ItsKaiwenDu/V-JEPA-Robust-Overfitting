"""LOCAL scratch metadata/byte guard; no creation, proof, launch or cleanup API.

Callers still need a separately reviewed source bootstrap, phase coordinator and
exact authorization. Scans are observations, not quotas or atomic snapshots.
"""
import hashlib
import json
import math
import os
from pathlib import Path,PurePosixPath
import re
import stat
import threading
from models import predictor_pgd_split_plan_v1 as plan

MiB=2**20;GiB=2**30
CONTRACT_SHA='6abefa0d40eb571fea1385086dde79eb418511db5e4e7ad73da0c8d65a6c2869'
DESIGN_SHA='c5daae30cbe4043cf4d5265de2a2a7420d015ab1e244cc3d74fcc93b5f0186af'
SUPERVISOR_NAMES=('bootstrap.py','predictor_pgd_split_scratch_v1.py','predictor_pgd_split_coordinator_v1.py',
 'predictor_pgd_split_collection_v1.py','predictor_pgd_split_watch_v1.py','scratch_allocation_v2.py')
PINNED_SUPPORT={'predictor_pgd_split_watch_v1.py':'07237d77806aba1c129e0e93520a2ceedd3ff2b06001f391d1cdaa46e55fae06',
 'scratch_allocation_v2.py':'6b0503b8ab2c88a44a4d95c8fafbad52d1c0414293379406520a22f67700e874'}
PHASES=('setup','proof','capture','collection')
ROOT_META={
 'control.json':(256*1024,False,True),'authorization.json':(64*1024,False,True),
 'proposal.json':(256*1024,False,True),'storage-before.json':(16*1024,False,True),
 'session.json':(128*1024,True,True),'runtime-source-manifest.json':(128*1024,False,True),
 'installation.json':(64*1024,False,True),'storage-status.json':(64*1024,True,True),
 'allocation-events-v2.jsonl':(MiB,True,False),'allocation-result-v2.json':(16*1024,False,False),
 'worker-plan-numerical.json':(16*1024,False,True),'worker-plan-native.json':(16*1024,False,True),
 'worker-plan-forward-preserving.json':(16*1024,False,True),
 'parent-numerical.json':(128*1024,False,True),'parent-native.json':(128*1024,False,True),
 'parent-forward-preserving.json':(128*1024,False,True),'gpu-samples.jsonl':(512*1024,True,False)}
CAPS={'raw':1792*MiB,'input':128*MiB,'code':128*MiB,'cache':32*MiB,'emergency':16*MiB,'metadata':64*MiB,'trace':8*MiB,'proof':5*GiB//2}


def encoded(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(v):return type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v)
def natural(v):return type(v) is int and v>=0

def parse_json(data):
    def pairs(rows):
        out={}
        for key,value in rows:
            if key in out:raise ValueError('Duplicate storage metadata key.')
            out[key]=value
        return out
    def bad(value):raise ValueError('Nonfinite storage metadata: '+value)
    return json.loads(data,object_pairs_hook=pairs,parse_constant=bad)

def safe(name):
    if type(name) is not str or not name or len(name)>768 or '\\' in name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/' for c in name):
        raise ValueError('Bounded canonical ASCII scratch path required.')
    p=PurePosixPath(name)
    if p.is_absolute() or p.as_posix()!=name or any(v in ('.','..') or len(v)>128 for v in p.parts):raise ValueError('Canonical scratch path required.')
    return name

def directory_id(s):return (s.st_dev,s.st_ino,s.st_uid,stat.S_IMODE(s.st_mode))
def file_id(s):return (*directory_id(s),s.st_nlink,s.st_size,s.st_mtime_ns,s.st_blocks*512)

def identity_binding(root):
    """Read-only metadata for the future setup owner; not host authorization."""
    root=Path(root)
    if not root.is_absolute() or root.resolve(strict=True)!=root:raise ValueError('Canonical absolute scratch root required.')
    return {'root':str(root),'root_identity':list(directory_id(root.lstat())),
        'parent':str(root.parent),'parent_identity':list(directory_id(root.parent.lstat()))}


class Layout:
    def __init__(self,contract_bytes,base_bytes,design_bytes,installation_bytes,installation_sha256):
        if (type(contract_bytes) is not bytes or hashlib.sha256(contract_bytes).hexdigest()!=CONTRACT_SHA
                or type(design_bytes) is not bytes or hashlib.sha256(design_bytes).hexdigest()!=DESIGN_SHA
                or type(installation_bytes) is not bytes or not sha(installation_sha256)
                or len(installation_bytes)>64*1024 or hashlib.sha256(installation_bytes).hexdigest()!=installation_sha256):
            raise ValueError('Exact source/storage/installation bytes required.')
        self.contract=parse_json(contract_bytes);self.result=plan.result_specs(plan.base_contract(base_bytes))
        design=parse_json(design_bytes)
        if (len(self.result)!=419 or hashlib.sha256(plan.encoded(self.result)).hexdigest()!=design['inventory']['canonical_result_sha256']
                or encoded(self.contract['category_caps'])!=encoded(CAPS) or encoded(self.contract['root_metadata'])!=encoded(ROOT_META)):
            raise ValueError('Frozen result namespace or whole-scratch policy changed.')
        self.installation=parse_json(installation_bytes);self.installation_sha256=installation_sha256
        i=self.installation
        if (type(i) is not dict or set(i)!={'kind','control_sha256','files'} or i['kind']!='pgd-split-scratch-installation-v1'
                or not sha(i['control_sha256']) or type(i['files']) is not dict or set(i['files'])!=set(SUPERVISOR_NAMES)):
            raise ValueError('Exact future source installation descriptor required; no caller callbacks.')
        for name,d in i['files'].items():
            if (type(d) is not dict or set(d)!={'bytes','sha256'} or not natural(d['bytes']) or not 0<d['bytes']<=2*MiB
                    or not sha(d['sha256']) or name in PINNED_SUPPORT and d['sha256']!=PINNED_SUPPORT[name]):
                raise ValueError('Bounded supervisor source descriptor changed: '+name)
        if sum(d['bytes'] for d in i['files'].values())>2*MiB:raise ValueError('Shared supervisor code ceiling.')
        self.assets={'original-packet.tar.gz':self.contract['packet']['archive']}
        self.assets.update({'code/'+p:d for p,d in self.contract['packet']['files'].items()})
        self.assets.update({'overlay/'+p:d for p,d in self.contract['runtime_files'].items()})
        self.assets.update({'supervisor/'+p:d for p,d in i['files'].items()})
        self.assets.update({'configs/base.json':{'bytes':len(base_bytes),'sha256':hashlib.sha256(base_bytes).hexdigest()},
            'configs/design.json':{'bytes':len(design_bytes),'sha256':DESIGN_SHA}})
        self.assets['installation.json']={'bytes':len(installation_bytes),'sha256':installation_sha256}
        self.directories={'.','capture','cache','tmp','cache/cuda'}
        for name in (*self.assets,*ROOT_META,*(('capture/'+p) for p in self.result)):
            safe(name)
            self.directories.update(str(p) for p in PurePosixPath(name).parents)

    def classify(self,name):
        safe(name)
        if name.startswith('capture/'):
            d,plain,partial=plan.classify(name[len('capture/'):],self.result)
            return dict(d,plain='capture/'+plain,partial=partial,space='result')
        if name=='space-proof-v2.bin.partial':return dict(category='proof',max_bytes=5*GiB//2,plain=name,partial=False,mutable=False,partial_allowed=False,space='proof')
        if name=='emergency-space.bin':return dict(category='emergency',max_bytes=16*MiB,plain=name,partial=False,mutable=False,partial_allowed=False,space='emergency')
        if name.startswith(('cache/','tmp/')):
            if len(PurePosixPath(name).parts)>8:raise ValueError('Cache directory depth ceiling.')
            return dict(category='cache',max_bytes=32*MiB,plain=name,partial=False,mutable=True,partial_allowed=False,space='cache')
        partial=name.endswith('.partial');plain=name.removesuffix('.partial')
        if plain in ROOT_META:
            limit,mutable,allow=ROOT_META[plain]
            if partial and not allow:raise ValueError('Disallowed supervisor metadata partial.')
            return dict(category='metadata',max_bytes=limit,plain=plain,partial=partial,mutable=mutable,partial_allowed=allow,space='metadata')
        if name in self.assets or partial and plain=='original-packet.tar.gz':
            d=self.assets[plain]
            return dict(category='input' if plain=='original-packet.tar.gz' else 'code',max_bytes=d['bytes'],plain=plain,
                partial=partial,mutable=False,partial_allowed=plain=='original-packet.tar.gz',space='asset')
        raise ValueError('Unknown scratch path: '+name)

    def directory(self,name):
        safe(name) if name!='.' else None
        if name in self.directories:return
        if name.startswith(('cache/','tmp/')) and len(PurePosixPath(name).parts)<8:return
        raise ValueError('Unknown scratch directory: '+name)

    def assess(self,files,directories,capacity,*,phase,initial=False,settled=False):
        """Pure metadata assessment; fake observations cannot authenticate execution."""
        if phase not in PHASES or type(initial) is not bool or type(settled) is not bool or initial and phase!='setup':raise ValueError('Exact scratch phase required.')
        if type(files) is not dict or type(directories) is not dict or len(files)>1024 or len(directories)>256:raise ValueError('Scratch entry-count ceiling.')
        if (set(capacity)!={'available_bytes','available_inodes'} or not all(natural(v) for v in capacity.values())
                or capacity['available_bytes']<(8*GiB if initial else GiB) or capacity['available_inodes']<(2048 if initial else 1024)):
            raise ValueError('Observed free-space/inode floor reached; no reservation or retry.')
        root=directories.get('.')
        if (type(root) is not dict or type(root.get('identity')) not in (list,tuple) or len(root['identity'])!=4
                or any(not natural(v) for v in root['identity'])):raise ValueError('Bounded root identity required.')
        device,_,uid,_=root['identity']
        info={};inodes={};pairs=[]
        for name,row in files.items():
            d=self.classify(name);info[name]=d
            if (type(row) is not dict or set(row)!={'identity','bytes','allocated_bytes','mtime_ns'} or not natural(row['bytes'])
                    or not natural(row['allocated_bytes']) or row['bytes']>d['max_bytes'] or type(row['identity']) not in (list,tuple)
                    or len(row['identity'])!=5 or any(not natural(v) for v in row['identity']) or row['identity'][3]!=0o600
                    or row['identity'][4] not in (1,2) or row['identity'][0]!=device or row['identity'][2]!=uid or type(row['mtime_ns']) is not int):raise ValueError('Typed scratch file observation refused: '+name)
            if phase!='proof' and d['category']=='proof':raise ValueError('Proof file outside its one proof phase.')
            if phase in ('setup','proof') and d['category'] in ('raw','trace'):raise ValueError('No capture arrays/trace during setup or proof.')
            if settled and d['partial']:raise ValueError('Unfinished publication in settled scratch.')
            inode=tuple(row['identity'][:2]);inodes.setdefault(inode,[]).append(name)
        # The result store uses link-then-unlink to publish an immutable name.
        # Only its exact same-directory final/partial alias may be observed live.
        canonical=dict(files)
        for inode,names in inodes.items():
            row=files[names[0]];links=row['identity'][4]
            if links==1:
                if len(names)!=1:raise ValueError('Repeated single-link inode observation.')
                continue
            if settled or len(names)!=2:raise ValueError('External/incomplete hardlink pair refused.')
            a,b=sorted(names,key=len);d=info[a];other=info[b]
            if (b!=a+'.partial' or other['plain']!=a or not other['partial'] or d['partial'] or not d['partial_allowed']
                    or d['space'] not in ('result','metadata') or files[a]!=files[b]):raise ValueError('Only one exact owned publication inode pair is permitted.')
            # Count the pending entity once (still partial, never complete), not
            # two references to the same logical/allocated extent. Both names
            # remain in the unmodified observation and named-byte diagnostic.
            canonical.pop(a);pairs.append([a,b])
        if sum(d['space']=='cache' for d in info.values())>48:raise ValueError('Combined cache/tmp file-count ceiling.')
        if sum(not n.startswith('capture/') for n in files)>256:raise ValueError('Reserved non-result file-count ceiling.')
        result_entries={n[len('capture/'):]:v['bytes'] for n,v in canonical.items() if n.startswith('capture/')}
        result_totals=plan.result_totals(result_entries,self.result)
        counts={'raw':0,'metadata':0,'input':0};extra_metadata=0;categories=dict.fromkeys(CAPS,0)
        for name,row in canonical.items():
            d=info[name];categories[d['category']]+=row['bytes']
            if d['partial']:
                label='raw' if d['category']=='raw' else 'input' if d['category']=='input' else 'metadata';counts[label]+=1
                if d['plain'] in canonical:
                    if not d['mutable']:raise ValueError('Exclusive publication cannot replace completed bytes.')
                    extra_metadata+=row['bytes']
        if counts['raw']>1 or counts['metadata']>2 or counts['input']>1 or extra_metadata>512*1024:raise ValueError('Shared scratch partial-count/extra-metadata ceiling.')
        for name in (*files,*directories):
            if name!='.' and str(PurePosixPath(name).parent) not in directories:
                raise ValueError('Every observed scratch parent must be included.')
        logical=sum(categories.values());allocated=0
        for name,row in canonical.items():allocated+=row['allocated_bytes']
        seen_dirs=set()
        for name,row in directories.items():
            self.directory(name)
            if (type(row) is not dict or set(row)!={'identity','allocated_bytes'} or not natural(row['allocated_bytes'])
                    or type(row['identity']) not in (list,tuple) or len(row['identity'])!=4
                    or any(not natural(v) for v in row['identity']) or row['identity'][3]!=0o700 or row['identity'][0]!=device or row['identity'][2]!=uid):raise ValueError('Private directory accounting required.')
            inode=tuple(row['identity'][:2])
            if inode in seen_dirs or inode in inodes:raise ValueError('Directory inode repeated or aliased.')
            seen_dirs.add(inode);allocated+=row['allocated_bytes']
        if '.' not in directories:raise ValueError('Root allocation must be counted.')
        limit=3*GiB if phase in ('setup','proof') else 9*GiB//4
        if (logical>limit or allocated>limit or categories['metadata']+categories['trace']>64*MiB
                or any(categories[k]>v for k,v in CAPS.items())):raise ValueError('Shared logical/allocated/category ceiling reached.')
        return {'kind':'pgd-split-whole-scratch-observation-v1','phase':phase,'initial':initial,'settled':settled,
            'files':len(files),'directories':len(directories),'logical_bytes':logical,'named_bytes':sum(v['bytes'] for v in files.values()),
            'allocated_bytes':allocated,'categories':categories,'result_categories':result_totals,'ceiling_bytes':limit,
            'capacity':dict(capacity),'publication_pairs':pairs,'partials':counts,'extra_metadata_bytes':extra_metadata,
            'sampling_is_not_atomic_or_kernel_quota':True,'scientific_qualified':False,'proof_execution_admitted':False,'training_admitted':False}


class Guard:
    def __init__(self,root,binding,layout):
        self.root=Path(root);self.binding=json.loads(json.dumps(binding));self.layout=layout
        if (type(binding) is not dict or set(binding)!={'root','root_identity','parent','parent_identity'}
                or binding['root']!=str(self.root) or binding['parent']!=str(self.root.parent)
                or not self.root.is_absolute() or self.root.resolve(strict=True)!=self.root):raise ValueError('Explicit canonical scratch identity binding required.')
        for key in ('root_identity','parent_identity'):
            if type(binding[key]) is not list or len(binding[key])!=4 or not all(natural(v) for v in binding[key]):raise ValueError('Typed directory identity binding required.')
        if binding['root_identity'][2:]!=[os.getuid(),0o700] or binding['root_identity'][0]!=binding['parent_identity'][0]:raise ValueError('Owned private same-device scratch binding required.')
        self.owner=(os.getpid(),threading.get_ident());self.pins={};self.check_root()
    def check_root(self):
        if ((os.getpid(),threading.get_ident())!=self.owner or self.root.resolve(strict=True)!=self.root or list(directory_id(self.root.lstat()))!=self.binding['root_identity']
                or list(directory_id(self.root.parent.lstat()))!=self.binding['parent_identity']):raise ValueError('Scratch creator/root/parent identity changed.')
    def open_root(self):
        self.check_root();fd=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        if list(directory_id(os.fstat(fd)))!=self.binding['root_identity']:
            os.close(fd);raise ValueError('Held scratch root changed.')
        return fd
    def scan(self,*,phase,initial=False,settled=False):
        """Read-only anchored descriptor walk; never follows symlinks or retries."""
        files={};dirs={};rootfd=self.open_root();device=self.binding['root_identity'][0];uid=os.getuid();nodes=0
        def walk(fd,prefix):
            nonlocal nodes
            before=os.fstat(fd);name=prefix or '.';self.layout.directory(name)
            if (not stat.S_ISDIR(before.st_mode) or before.st_uid!=uid or before.st_dev!=device or stat.S_IMODE(before.st_mode)!=0o700):raise ValueError('Private same-device scratch directory required.')
            dirs[name]={'identity':directory_id(before),'allocated_bytes':before.st_blocks*512}
            if len(dirs)>256:raise ValueError('Directory-count ceiling.')
            with os.scandir(fd) as entries:
                for entry in entries:
                    nodes+=1
                    if nodes>1280:raise ValueError('Bounded scratch traversal required.')
                    path=(prefix+'/' if prefix else '')+entry.name;safe(path)
                    observed=os.stat(entry.name,dir_fd=fd,follow_symlinks=False)
                    if observed.st_dev!=device or observed.st_uid!=uid:raise ValueError('Scratch device/owner changed.')
                    if stat.S_ISDIR(observed.st_mode):
                        self.layout.directory(path);child=os.open(entry.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                        try:
                            if directory_id(os.fstat(child))!=directory_id(observed):raise ValueError('Named/held scratch directory changed.')
                            walk(child,path)
                            if directory_id(os.stat(entry.name,dir_fd=fd,follow_symlinks=False))!=directory_id(observed):raise ValueError('Directory replaced during walk.')
                        finally:os.close(child)
                    else:
                        self.layout.classify(path)
                        if not stat.S_ISREG(observed.st_mode) or stat.S_IMODE(observed.st_mode)!=0o600:raise ValueError('Private regular scratch file required; no symlink/special path.')
                        files[path]={'identity':(*directory_id(observed),observed.st_nlink),'bytes':observed.st_size,
                            'allocated_bytes':observed.st_blocks*512,'mtime_ns':observed.st_mtime_ns}
                        if len(files)>1024:raise ValueError('File-count ceiling.')
            if directory_id(os.fstat(fd))!=directory_id(before):raise ValueError('Directory identity changed during scan.')
        try:
            walk(rootfd,'');fs=os.fstatvfs(rootfd)
            capacity={'available_bytes':fs.f_bavail*fs.f_frsize,'available_inodes':fs.f_favail}
            report=self.layout.assess(files,dirs,capacity,phase=phase,initial=initial,settled=settled)
            for path,want in self.pins.items():
                if files.get(path)!=want:raise ValueError('Verified source asset changed after sealing: '+path)
            self.check_root()
            return report,files,dirs
        finally:os.close(rootfd)
    def read(self,name,cap):
        safe(name);d=self.layout.classify(name)
        if type(cap) is not int or cap<0 or cap>d['max_bytes']:raise ValueError('Bounded source read cap required.')
        parts=PurePosixPath(name).parts;fd=self.open_root();held=[]
        try:
            for part in parts[:-1]:
                expected=os.stat(part,dir_fd=fd,follow_symlinks=False)
                if not stat.S_ISDIR(expected.st_mode) or expected.st_dev!=self.binding['root_identity'][0] or expected.st_uid!=os.getuid() or stat.S_IMODE(expected.st_mode)!=0o700:
                    raise ValueError('Private source parent required.')
                child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                if directory_id(os.fstat(child))!=directory_id(expected):os.close(child);raise ValueError('Source parent changed before open.')
                held.append((fd,part,expected));fd=child
            expected=os.stat(parts[-1],dir_fd=fd,follow_symlinks=False)
            if (not stat.S_ISREG(expected.st_mode) or expected.st_nlink!=1 or expected.st_uid!=os.getuid() or stat.S_IMODE(expected.st_mode)!=0o600
                    or expected.st_dev!=self.binding['root_identity'][0] or not 0<=expected.st_size<=cap):raise ValueError('Private bounded single-link source required.')
            data=bytearray();filefd=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
            try:
                if file_id(os.fstat(filefd))!=file_id(expected):raise ValueError('Held source changed before read.')
                while len(data)<=cap:
                    chunk=os.read(filefd,min(MiB,cap+1-len(data)))
                    if not chunk:break
                    data.extend(chunk)
                if file_id(os.fstat(filefd))!=file_id(expected):raise ValueError('Source changed during read.')
            finally:os.close(filefd)
            if len(data)!=expected.st_size or file_id(os.stat(parts[-1],dir_fd=fd,follow_symlinks=False))!=file_id(expected):raise ValueError('Named source changed during read.')
            for parent,part,want in held:
                if directory_id(os.stat(part,dir_fd=parent,follow_symlinks=False))!=directory_id(want):raise ValueError('Source ancestry replaced during read.')
            self.check_root();return bytes(data)
        finally:
            os.close(fd)
            for parent,_,_ in reversed(held):os.close(parent)
    @staticmethod
    def require_emergency(files):
        item=files.get('emergency-space.bin')
        if (type(item) is not dict or item.get('bytes')!=16*MiB or type(item.get('allocated_bytes')) is not int or item['allocated_bytes']<16*MiB):
            raise ValueError('Full allocated emergency extent required; sparse length is insufficient.')
    def proof_readiness(self):
        report,files,_=self.scan(phase='setup',initial=True,settled=True)
        if any(n in files for n in ('space-proof-v2.bin.partial','allocation-events-v2.jsonl','allocation-result-v2.json')):
            raise ValueError('Preserve previous proof artifacts; no retry.')
        self.require_emergency(files)
        if (any(not natural(report.get(k)) for k in ('logical_bytes','allocated_bytes'))
                or max(report['logical_bytes'],report['allocated_bytes'])+5*GiB//2+32*MiB>3*GiB):
            raise ValueError('Full proof plus original 32 MiB allocation reserve does not fit setup.')
        return dict(report,proof_can_be_proposed_only=True,proof_launch_authorized=False)
    def verify_assets(self,*,phase='capture'):
        before,files,_=self.scan(phase=phase,settled=True)
        self.require_emergency(files)
        for name,want in self.layout.assets.items():
            if name not in files or files[name]['bytes']!=want['bytes']:raise ValueError('Missing/exact-extent source asset required: '+name)
            data=self.read(name,want['bytes'])
            if hashlib.sha256(data).hexdigest()!=want['sha256']:raise ValueError('Source asset digest changed: '+name)
        after,current,_=self.scan(phase=phase,settled=True)
        selected={p:current[p] for p in self.layout.assets}
        if any(files[p]!=row for p,row in selected.items()):raise ValueError('Source asset changed around verification.')
        self.pins=selected
        return {'kind':'pgd-split-scratch-source-metadata-v1','control_sha256':self.layout.installation['control_sha256'],
            'assets':len(selected),'installation_sha256':self.layout.installation_sha256,
            'source_names_and_hashes_verified':True,'execution_or_authorization_authenticated':False,'scientific_qualified':False}
    def verify_proof(self,expected_sha256):
        if not sha(expected_sha256):raise ValueError('Parent-bound proof receipt digest required.')
        data=self.read('allocation-result-v2.json',16*1024)
        if hashlib.sha256(data).hexdigest()!=expected_sha256:raise ValueError('Proof receipt digest changed.')
        p=parse_json(data)
        if (type(p) is not dict or p.get('kind')!='scratch-allocation-proof-v2'
                or any(p.get(k) is not True for k in ('passed','complete','proof_removed','fsync_succeeded'))
                or p.get('preallocator_injected_for_test') is not False
                or any(type(p.get(k)) is not int or p[k]!=5*GiB//2 for k in ('requested_bytes','bytes_written','bytes_read_back'))
                or type(p.get('setup_ceiling_bytes')) is not int or p['setup_ceiling_bytes']!=3*GiB
                or type(p.get('peak_observed_allocated_bytes')) is not int or not 5*GiB//2<=p['peak_observed_allocated_bytes']<=3*GiB
                or type(p.get('wall_seconds')) not in (int,float) or not math.isfinite(p['wall_seconds']) or not 0<p['wall_seconds']<=60
                or not sha(p.get('sha256'))):raise ValueError('Original full fixed-extent proof receipt did not pass; no fallback.')
        self.scan(phase='capture',settled=True)
        return {'kind':'pgd-split-storage-proof-metadata-v1','receipt_sha256':expected_sha256,'requested_bytes':5*GiB//2,
            'original_proof_predicates_retained':True,'proof_execution_authenticated':False,'scientific_qualified':False}
