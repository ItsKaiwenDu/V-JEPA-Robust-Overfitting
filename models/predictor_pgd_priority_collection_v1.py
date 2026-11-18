"""Local priority/deduplicated evidence codec; no CLI, connection or work API.

The legacy source/ownership/clock validators remain mandatory. This module is a
source-only protocol candidate, not a new bound runtime or experiment grant.
"""
import hashlib,os,stat,struct,time
from pathlib import Path,PurePosixPath
from models import predictor_pgd_publication_collection_v1 as legacy
boot,scratch=legacy.boot,legacy.scratch
MiB,GiB=2**20,2**30
MAGIC=b'VJEPA-PGD-PRIORITY-EVIDENCE-1\n';RECORD=b'VJEPA-PGD-PRIORITY-FILE-1\n';END=b'VJEPA-PGD-PRIORITY-END-1\n'
HEADER_CAP,RECORD_CAP,TRAILER_CAP,WIRE_CAP=MiB,4096,16*1024,3*GiB
JOURNAL_CAP,PROGRESS_CAP=8*MiB,4*MiB
KIND='pgd-priority-evidence-stream-v1'
FALSE_FLAGS={'scientific_qualified':False,'training_admitted':False}
CRITICAL=('session.json',legacy.RECEIPT,legacy.STATUS,'control.json','authorization.json','installation.json','proposal.json')


def priority_order(files,layout):
 rank={name:i for i,name in enumerate(CRITICAL)}
 groups={'metadata':10,'trace':11,'code':12,'input':13,'raw':14}
 return sorted(files,key=lambda name:(rank[name] if name in rank else groups[layout.classify(name)['category']],name))

def _planned(base,layout):
 records=[];first={};payload=0
 for name in priority_order(base['files'],layout):
  item=base['files'][name];key=(item['sha256'],item['bytes'])
  if key in first:record={'name':name,'kind':'reference','source':first[key]}
  else:record={'name':name,'kind':'data','source':None};first[key]=name;payload+=item['bytes']
  records.append(record)
 return {'kind':KIND,'base':base,'records':records,'payload_bytes':payload,'expanded_bytes':base['logical_bytes'],
  'deduplicated_bytes':base['logical_bytes']-payload,**FALSE_FLAGS}

def _bounds(h):
 raw=boot.encoded(h)
 if len(raw)>HEADER_CAP:raise ValueError('Priority header exceeds original 1 MiB cap.')
 frames=sum(len(RECORD)+4+len(boot.encoded(r)) for r in h['records'])
 if any(len(boot.encoded(r))>RECORD_CAP for r in h['records']):raise ValueError('Bounded file record required.')
 # The no-dedup case must fit too; repeated bytes cannot authorize expansion.
 if h['expanded_bytes']+JOURNAL_CAP>WIRE_CAP or h['expanded_bytes']+len(raw)+len(MAGIC)+len(END)+8+frames+TRAILER_CAP>WIRE_CAP:
  raise ValueError('Original 3 GiB expansion/disk/response envelope exceeded.')
 categories=h['base']['storage']['categories']
 if categories['metadata']+categories['trace']+JOURNAL_CAP>64*MiB:raise ValueError('Journal reserve exceeds original shared metadata cap.')

def make_header(base,control_bytes,authorization_bytes):
 layout=legacy.validate_header(base,control_bytes,authorization_bytes)
 h=_planned(base,layout);_bounds(h);return h

def validate_header(h,control_bytes,authorization_bytes):
 if type(h) is not dict or set(h)!={'kind','base','records','payload_bytes','expanded_bytes','deduplicated_bytes',*FALSE_FLAGS} or h['kind']!=KIND:
  raise ValueError('Exact priority protocol header required; legacy streams cannot be guessed.')
 layout=legacy.validate_header(h['base'],control_bytes,authorization_bytes)
 if type(h['records']) is not list or len(h['records'])>1024:raise ValueError('Bounded deterministic file plan required.')
 if any(type(h[k]) is not int for k in ('payload_bytes','expanded_bytes','deduplicated_bytes')):raise ValueError('Integer byte totals required.')
 if any(h[k] is not False for k in FALSE_FLAGS) or h!=_planned(h['base'],layout):raise ValueError('Priority order/reference identity or exact totals changed.')
 _bounds(h);return layout


class Collector:
 """One legacy-validated snapshot followed by one new priority stream."""
 def __init__(self,guard,control_bytes,authorization_bytes,*,connection_started,clock=time.monotonic):
  self.original=legacy.Collector(guard,control_bytes,authorization_bytes,connection_started=connection_started,clock=clock)
  self.prepared=self.streamed=False;self.header=None
 def prepare(self):
  if self.prepared:raise ValueError('One priority preparation; no reset or retry.')
  self.prepared=True
  base=self.original.prepare();self.header=make_header(base,self.original.control,self.original.auth)
  self.original.window.check(min(self.original.window.connection_started+2400,self.original.window.started+120));return self.header
 def stream(self,out):
  if not self.prepared or self.header is None or self.streamed:raise ValueError('One prepared priority stream required.')
  self.streamed=True;c=self.original;window=c.window;window.check()
  validate_header(self.header,c.control,c.auth)
  header=boot.encoded(self.header);writer=legacy.WireWriter(out,window);writer.write(MAGIC+struct.pack('!I',len(header))+header)
  for record in self.header['records']:
   name=record['name'];item=self.header['base']['files'][name];digest=hashlib.sha256();encoded=boot.encoded(record)
   if record['kind']=='data':writer.write(RECORD+struct.pack('!I',len(encoded))+encoded)
   # References still read/hash their own exact source. They omit wire bytes,
   # never source identity checks or a post-snapshot mutation check.
   for data in legacy.FileReader(c.guard,name,c.inventory[name],window).chunks():
    digest.update(data)
    if record['kind']=='data':writer.write(data)
   if digest.hexdigest()!=item['sha256']:raise ValueError('Source hash changed; no priority terminal receipt.')
   if record['kind']=='reference':writer.write(RECORD+struct.pack('!I',len(encoded))+encoded)
  _,current,dirs=c.guard.scan(phase='collection')
  if current!=c.inventory or dirs!=c.directories:raise ValueError('Scratch changed; no priority terminal receipt.')
  terminal=_terminal(self.header,boot.digest(header),writer.hasher.hexdigest(),writer.count)
  data=boot.encoded(terminal)
  if len(data)>TRAILER_CAP:raise ValueError('Priority terminal cap.')
  writer.write(END+struct.pack('!I',len(data))+data);out.flush();window.check()
  return dict(terminal,wire_bytes=writer.count,wire_sha256=writer.hasher.hexdigest(),receiver_verified=False,independent_review_pending=True)


def _terminal(h,hsha,prefix_sha,prefix_bytes):
 b=h['base']
 return {'kind':'pgd-priority-stream-terminal-v1','control_sha256':b['control_sha256'],'authorization_sha256':b['authorization_sha256'],
  'header_sha256':hsha,'prefix_sha256':prefix_sha,'prefix_bytes':prefix_bytes,'files':len(h['records']),
  'expanded_bytes':h['expanded_bytes'],'payload_bytes':h['payload_bytes'],'deduplicated_bytes':h['deduplicated_bytes'],
  'final_receipt_sha256':b['final_receipt_sha256'],'stream_complete':True,**FALSE_FLAGS}

def _private(info,device):
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_dev!=device or info.st_nlink!=1 or stat.S_IMODE(info.st_mode)!=0o600:
  raise ValueError('Private owned single-link local file required.')

def _write(fd,data,window):
 offset=0
 while offset<len(data):
  window.check();n=os.write(fd,data[offset:])
  if type(n) is not int or not 0<n<=len(data)-offset:raise OSError('Short local byte reconstruction.')
  offset+=n
 window.check()

def _root_check(parentfd,name,rootfd,want):
 if scratch.directory_id(os.fstat(rootfd))!=want or scratch.directory_id(os.stat(name,dir_fd=parentfd,follow_symlinks=False))!=want:
  raise ValueError('Named/held local root changed.')

def _prior_chunks(rootfd,name,want,window):
 """Read a fully verified prior DATA file; never follow links or ref chains."""
 fd=os.dup(rootfd);parents=[];sourcefd=None;parts=PurePosixPath(name).parts;device=os.fstat(rootfd).st_dev
 try:
  for part in parts[:-1]:
   before=os.stat(part,dir_fd=fd,follow_symlinks=False)
   if not stat.S_ISDIR(before.st_mode) or before.st_uid!=os.getuid() or before.st_dev!=device or stat.S_IMODE(before.st_mode)!=0o700:raise ValueError('Private reference ancestry required.')
   child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
   if scratch.directory_id(os.fstat(child))!=scratch.directory_id(before):os.close(child);raise ValueError('Reference ancestor changed.')
   parents.append((fd,part,before));fd=child
  before=os.stat(parts[-1],dir_fd=fd,follow_symlinks=False);_private(before,device)
  if legacy._row(before)!=want:raise ValueError('Verified reference source identity changed.')
  sourcefd=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
  if legacy._row(os.fstat(sourcefd))!=want:raise ValueError('Held reference source changed.')
  left=before.st_size
  while left:
   window.check();data=os.read(sourcefd,min(MiB,left))
   if not data:raise ValueError('Reference source truncated.')
   left-=len(data);yield data;window.check()
  if os.read(sourcefd,1) or legacy._row(os.fstat(sourcefd))!=want or legacy._row(os.stat(parts[-1],dir_fd=fd,follow_symlinks=False))!=want:
   raise ValueError('Reference source grew or changed.')
  for parent,part,w in parents:
   if scratch.directory_id(os.stat(part,dir_fd=parent,follow_symlinks=False))!=scratch.directory_id(w):raise ValueError('Named reference ancestry changed.')
 finally:
  if sourcefd is not None:os.close(sourcefd)
  os.close(fd)
  for parent,_,_ in reversed(parents):os.close(parent)


class Journal:
 """Bounded local metadata only. Its receipts ALWAYS remain incomplete.

 Only receive's return value can describe a terminal-verified full transfer.
 An interrupted journal preserves a manifest plus a verified per-file prefix.
 """
 def __init__(self,parentfd,name,window,header_bytes,h):
  scratch.safe(name);self.parentfd,self.name,self.window=parentfd,name,window;self.rootfd=None;self.progressfd=None
  self.records={};self.sequence=0;self.previous='0'*64;self.progresshash=hashlib.sha256();self.progress_bytes=0;self.bytes=0
  try:
   os.mkdir(name,0o700,dir_fd=parentfd);self.rootfd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parentfd)
   self.root_id=scratch.directory_id(os.fstat(self.rootfd))
   if self.root_id[0]!=os.fstat(parentfd).st_dev or self.root_id[2:]!=(os.getuid(),0o700):raise ValueError('Private same-device journal required.')
   self.meta('HEADER.json',header_bytes)
   self.meta('PARTIAL-RECEIPT.json',boot.encoded({'kind':'pgd-priority-partial-receipt-v1','header_sha256':boot.digest(header_bytes),
    'control_sha256':h['base']['control_sha256'],'authorization_sha256':h['base']['authorization_sha256'],'transport_verified':False,
    'stream_complete':False,'per_file_journal':'VERIFIED-FILES.jsonl','independent_review_pending':True,**FALSE_FLAGS}))
   self.progressfd=os.open('VERIFIED-FILES.jsonl',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=self.rootfd)
   self.snapshot=legacy._row(os.fstat(self.progressfd));_private(os.fstat(self.progressfd),self.root_id[0]);os.fsync(self.progressfd);os.fsync(self.rootfd)
  except BaseException:self.close();raise
 def meta(self,name,data):
  self.window.check();_root_check(self.parentfd,self.name,self.rootfd,self.root_id)
  if self.bytes+len(data)>JOURNAL_CAP:raise ValueError('Bounded journal metadata ceiling.')
  fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=self.rootfd)
  try:_write(fd,data,self.window);os.fsync(fd)
  finally:os.close(fd)
  s=os.stat(name,dir_fd=self.rootfd,follow_symlinks=False);_private(s,self.root_id[0])
  if s.st_size!=len(data):raise ValueError('Journal file extent changed.')
  self.records[name]={'bytes':s.st_size,'sha256':boot.digest(data),'mode':0o600,'mtime_ns':s.st_mtime_ns};self.bytes+=len(data);os.fsync(self.rootfd)
 def append(self,name,item,record,wire_bytes):
  self.window.check();_root_check(self.parentfd,self.name,self.rootfd,self.root_id)
  current=os.fstat(self.progressfd);_private(current,self.root_id[0])
  if legacy._row(current)!=self.snapshot or legacy._row(os.stat('VERIFIED-FILES.jsonl',dir_fd=self.rootfd,follow_symlinks=False))!=self.snapshot:
   raise ValueError('Owned progress journal changed.')
  row={'kind':'pgd-priority-verified-file-v1','sequence':self.sequence,'previous_sha256':self.previous,'file':name,
   'descriptor':item,'record':record,'wire_bytes_observed':wire_bytes,'transport_verified':False,'stream_complete':False,**FALSE_FLAGS}
  row['row_sha256']=boot.digest(boot.encoded(row));data=boot.encoded(row)
  if len(data)>RECORD_CAP or self.progress_bytes+len(data)>PROGRESS_CAP or self.bytes+self.progress_bytes+len(data)>JOURNAL_CAP:
   raise ValueError('Progress/journal byte cap.')
  _write(self.progressfd,data,self.window);os.fsync(self.progressfd)
  s=os.fstat(self.progressfd);_private(s,self.root_id[0]);self.snapshot=legacy._row(s)
  if s.st_size!=self.progress_bytes+len(data) or legacy._row(os.stat('VERIFIED-FILES.jsonl',dir_fd=self.rootfd,follow_symlinks=False))!=self.snapshot:raise ValueError('Progress append identity changed.')
  self.progresshash.update(data);self.progress_bytes+=len(data);self.sequence+=1;self.previous=row['row_sha256']
 def finish(self):
  self.window.check();_root_check(self.parentfd,self.name,self.rootfd,self.root_id)
  before=os.fstat(self.progressfd);_private(before,self.root_id[0])
  if legacy._row(before)!=self.snapshot:raise ValueError('Progress changed before close.')
  fd,self.progressfd=self.progressfd,None;os.close(fd)
  after=os.stat('VERIFIED-FILES.jsonl',dir_fd=self.rootfd,follow_symlinks=False);_private(after,self.root_id[0])
  if scratch.file_id(after)[:-1]!=scratch.file_id(before)[:-1]:raise ValueError('Progress changed at owned close.')
  self.records['VERIFIED-FILES.jsonl']={'bytes':after.st_size,'sha256':self.progresshash.hexdigest(),'mode':0o600,'mtime_ns':after.st_mtime_ns}
  legacy.verify_local(self.rootfd,self.parentfd,self.name,self.root_id,self.records,self.window);os.fsync(self.rootfd)
  return {'header_sha256':self.records['HEADER.json']['sha256'],'progress_sha256':self.progresshash.hexdigest(),
   'verified_files':self.sequence,'journal_bytes':self.bytes+self.progress_bytes,'last_row_sha256':self.previous,'journal_is_not_full_transfer_receipt':True}
 def close(self):
  if self.progressfd is not None:fd,self.progressfd=self.progressfd,None;os.close(fd)
  if self.rootfd is not None:fd,self.rootfd=self.rootfd,None;os.close(fd)


def _disk_sample(rootfd,records,window):
 """Final metadata-only allocated-byte sample; not a kernel quota."""
 nodes=0;allocated=0;seen=set();dirs={'.'}
 for name in records:dirs.update(str(p) for p in PurePosixPath(name).parents)
 def walk(fd,prefix):
  nonlocal nodes,allocated
  window.check();info=os.fstat(fd);allocated+=info.st_blocks*512
  with os.scandir(fd) as entries:
   for entry in entries:
    nodes+=1
    if nodes>1280:raise ValueError('Bounded final allocation sample.')
    name=(prefix+'/' if prefix else '')+entry.name;s=os.stat(entry.name,dir_fd=fd,follow_symlinks=False)
    if stat.S_ISDIR(s.st_mode):
     if name not in dirs or s.st_uid!=os.getuid() or s.st_dev!=info.st_dev or stat.S_IMODE(s.st_mode)!=0o700:raise ValueError('Allocation sample ancestry changed.')
     child=os.open(entry.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
     try:
      if scratch.directory_id(os.fstat(child))!=scratch.directory_id(s):raise ValueError('Held allocation sample directory changed.')
      walk(child,name)
     finally:os.close(child)
    else:
     _private(s,info.st_dev);wanted=records.get(name)
     if wanted is None or (s.st_size,s.st_mtime_ns)!=(wanted['bytes'],wanted['mtime_ns']):raise ValueError('Allocation sample member changed.')
     allocated+=s.st_blocks*512;seen.add(name)
 walk(rootfd,'')
 if seen!=set(records):raise ValueError('Allocation sample incomplete.')
 return allocated


def receive(source,destination,control_bytes,authorization_bytes,*,window):
 """One fresh private byte reconstruction; no SSH, retries, or clock reset."""
 reader=legacy.WireReader(source,window);raw,h=reader.frame(MAGIC,HEADER_CAP);validate_header(h,control_bytes,authorization_bytes)
 dest=Path(destination);journal_name=dest.name+'-transport';scratch.safe(dest.name);scratch.safe(journal_name)
 if not dest.is_absolute() or dest.parent.resolve(strict=True)!=dest.parent or os.path.lexists(dest) or os.path.lexists(dest.parent/journal_name):
  raise ValueError('Fresh canonical evidence and journal directories required; preserve existing files.')
 parent=os.open(dest.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);rootfd=None;journal=None
 try:
  ps=os.fstat(parent)
  if not stat.S_ISDIR(ps.st_mode) or ps.st_uid!=os.getuid() or stat.S_IMODE(ps.st_mode)!=0o700 or scratch.directory_id(dest.parent.lstat())!=scratch.directory_id(ps):raise ValueError('Private destination parent required.')
  os.mkdir(dest.name,0o700,dir_fd=parent);rootfd=os.open(dest.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent);root_id=scratch.directory_id(os.fstat(rootfd))
  if root_id[0]!=ps.st_dev or root_id[2:]!=(os.getuid(),0o700):raise ValueError('Private same-device destination required.')
  journal=Journal(parent,journal_name,window,raw,h);records={};snapshots={};data_sources=set();expanded=0
  for planned in h['records']:
   window.check();_,record=reader.frame(RECORD,RECORD_CAP)
   if record!=planned:raise ValueError('Unexpected, duplicate, stale or forward file reference.')
   name=record['name'];item=h['base']['files'][name];expanded+=item['bytes']
   if expanded+JOURNAL_CAP>WIRE_CAP:raise ValueError('Expanded reconstruction disk ceiling.')
   if record['kind']=='reference':
    ref=record['source']
    if ref not in data_sources or ref not in records or any(records[ref][k]!=item[k] for k in ('bytes','sha256')):raise ValueError('Reference requires an earlier fully verified DATA file.')
   fd=os.dup(rootfd);outfd=None
   try:
    for part in PurePosixPath(name).parts[:-1]:
     child=legacy._directory(fd,part,ps.st_dev);os.close(fd);fd=child
    leaf=PurePosixPath(name).name;outfd=os.open(leaf,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd);digest=hashlib.sha256()
    if record['kind']=='data':
     left=item['bytes']
     while left:
      data=reader.read(min(MiB,left));_write(outfd,data,window);digest.update(data);left-=len(data)
    else:
     for data in _prior_chunks(rootfd,record['source'],snapshots[record['source']],window):_write(outfd,data,window);digest.update(data)
    os.fsync(outfd);os.utime(outfd,ns=(item['mtime_ns'],item['mtime_ns']));s=os.fstat(outfd);_private(s,ps.st_dev)
    if (s.st_size,s.st_mtime_ns,digest.hexdigest())!=(item['bytes'],item['mtime_ns'],item['sha256']):raise ValueError('Reconstructed member size/hash/mtime changed.')
    # Snapshot only after owned close, so legitimate allocation retirement does
    # not turn a completed local copy into a stale reference identity.
    closing,outfd=outfd,None;os.close(closing);named=os.stat(leaf,dir_fd=fd,follow_symlinks=False);_private(named,ps.st_dev)
    if scratch.file_id(named)[:-1]!=scratch.file_id(s)[:-1]:raise ValueError('Local member changed at owned close.')
    records[name]={'bytes':named.st_size,'sha256':digest.hexdigest(),'mode':0o600,'mtime_ns':named.st_mtime_ns};snapshots[name]=legacy._row(named)
    if record['kind']=='data':data_sources.add(name)
    os.fsync(fd)
   finally:
    if outfd is not None:os.close(outfd)
    os.close(fd)
   _root_check(parent,dest.name,rootfd,root_id);journal.append(name,records[name],record,reader.count)
  prefix_sha,prefix_bytes=reader.hasher.hexdigest(),reader.count
  _,terminal=reader.frame(END,TRAILER_CAP)
  if terminal!=_terminal(h,boot.digest(raw),prefix_sha,prefix_bytes):raise ValueError('Missing or mismatched priority terminal; no completion admission.')
  window.check()
  if source.read(1)!=b'':raise ValueError('Trailing priority stream bytes.')
  window.check()
  if dest.parent.resolve(strict=True)!=dest.parent or scratch.directory_id(dest.parent.lstat())!=scratch.directory_id(ps):raise ValueError('Named local parent changed.')
  legacy.verify_local(rootfd,parent,dest.name,root_id,records,window);progress=journal.finish()
  allocated=_disk_sample(rootfd,records,window)+_disk_sample(journal.rootfd,journal.records,window)
  if allocated>WIRE_CAP:raise ValueError('Final local allocated disk ceiling.')
  _root_check(parent,dest.name,rootfd,root_id);_root_check(parent,journal_name,journal.rootfd,journal.root_id)
  if scratch.directory_id(dest.parent.lstat())!=scratch.directory_id(ps):raise ValueError('Parent changed during final verification.')
  os.fsync(rootfd);os.fsync(parent);window.check()
  return {'kind':'pgd-priority-local-collection-receipt-v1','control_sha256':h['base']['control_sha256'],
   'authorization_sha256':h['base']['authorization_sha256'],'header_sha256':boot.digest(raw),'final_receipt_sha256':h['base']['final_receipt_sha256'],
   'wire_bytes':reader.count,'wire_sha256':reader.hasher.hexdigest(),'expanded_bytes':expanded,'payload_bytes':h['payload_bytes'],
   'deduplicated_bytes':h['deduplicated_bytes'],'files':records,'journal':progress,'sampled_allocated_bytes':allocated,
   'transport_verified':True,'independent_review_pending':True,'allocation_sample_is_not_kernel_quota':True,**FALSE_FLAGS}
 finally:
  if journal is not None:journal.close()
  if rootfd is not None:os.close(rootfd)
  os.close(parent)
