"""Read two exact saved metadata files; no write, inventory, arrays or compute."""
import base64,hashlib,json,os,platform,signal,stat,sys
ROOT='/var/tmp/vjepa-pgd-split-v1-2170-3f7b6ee518807ee36e330805082f28ea44a069bdf6cd59593804ea745a3ceb16'
CONTROL='03ddfc3fe3ec900213b7161251b445b436d6378783a18c591ac0722a729e0d0d'
SESSION='055f34ea7eadeb6f880838d2c20fb5bec47cd7d29025f6e44a8d19723c80aede'
HOST='linux10614.dc.engr.scu.edu';UID=2170;DEVICE=66307;LIMIT=2**20
KIND='pgd-publication-metadata-read-v1'
def encoded(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(v):return hashlib.sha256(v).hexdigest()
def authorization(plan,basis):
 return {'kind':KIND+'-authorization','approved':True,'plan_sha256':sha(encoded(plan)),'maximum_new_reads':1,'user_basis':basis}
def validate(plan,auth):
 expected={'kind':KIND+'-plan','host':HOST,'scratch':ROOT,'original_control_sha256':CONTROL,'session_sha256':SESSION,'read_seconds':20,'kill_grace_seconds':1,'transport_seconds':180,'response_bytes':LIMIT,'stderr_bytes':16384,'maximum_new_reads':1,'remote_writes':False,'model_GPU_proof_or_paid_work':False}
 if not isinstance(plan,dict) or any(plan.get(k)!=v for k,v in expected.items()):raise ValueError('Exact metadata-only host/root/hash/bounds required.')
 if type(auth) is not dict or type(auth.get('user_basis')) is not str or not 1<=len(auth['user_basis'])<=8192 or auth!=authorization(plan,auth['user_basis']):raise ValueError('One new exact metadata-only approval required.')
def identity(s):return (s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def read_file(fd,name,cap,uid,device,want):
 before=os.stat(name,dir_fd=fd,follow_symlinks=False)
 if not stat.S_ISREG(before.st_mode) or before.st_uid!=uid or before.st_dev!=device or before.st_nlink!=1 or stat.S_IMODE(before.st_mode)!=0o600:raise ValueError('Saved metadata type/owner/link/mode/device changed.')
 if before.st_size>cap:raise ValueError('Saved metadata exceeds read cap.')
 handle=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 try:
  if identity(os.fstat(handle))!=identity(before):raise ValueError('Saved metadata changed before read.')
  pieces=[];left=cap+1
  while left:
   chunk=os.read(handle,min(left,65536))
   if not chunk:break
   pieces.append(chunk);left-=len(chunk)
  data=b''.join(pieces)
  if identity(os.fstat(handle))!=identity(before) or identity(os.stat(name,dir_fd=fd,follow_symlinks=False))!=identity(before):raise ValueError('Saved metadata changed during read.')
 finally:os.close(handle)
 if len(data)!=before.st_size or len(data)>cap or sha(data)!=want:raise ValueError('Saved metadata size/hash mismatch; preserve old evidence.')
 return data

def inspect(root,uid,device,control_sha,session_sha):
 parent=os.open('/var/tmp',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW) if os.path.dirname(root)=='/var/tmp' else os.open(os.path.dirname(root),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 fd=None
 try:
  leaf=os.path.basename(root);before=os.stat(leaf,dir_fd=parent,follow_symlinks=False)
  if not stat.S_ISDIR(before.st_mode) or before.st_uid!=uid or before.st_dev!=device or stat.S_IMODE(before.st_mode)!=0o700:raise ValueError('Exact owned private scratch identity required.')
  fd=os.open(leaf,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
  if identity(os.fstat(fd))!=identity(before):raise ValueError('Scratch changed before read.')
  read_file(fd,'control.json',128*1024,uid,device,control_sha)
  session=read_file(fd,'session.json',256*1024,uid,device,session_sha)
  if identity(os.fstat(fd))!=identity(before) or identity(os.stat(leaf,dir_fd=parent,follow_symlinks=False))!=identity(before):raise ValueError('Scratch changed during read.')
 finally:
  if fd is not None:os.close(fd)
  os.close(parent)
 value=json.loads(session)
 if value.get('control_sha256')!=control_sha:raise ValueError('Saved session control mismatch.')
 return {'kind':KIND+'-result','scratch':root,'control_sha256':control_sha,'session_sha256':session_sha,'session_bytes':len(session),'session_base64':base64.b64encode(session).decode(),'server_file_writes':False,'array_contents_read':False,'model_GPU_proof_or_paid_work':False,'scientific_qualified':False,'training_admitted':False}

def readonly(event,args):
 if event=='open' and (len(args)>2 and isinstance(args[2],int) and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND) or len(args)>1 and isinstance(args[1],str) and any(x in args[1] for x in 'wax+')):raise PermissionError('Inspector writes forbidden.')
 if event in {'os.remove','os.rename','os.rmdir','os.mkdir','os.chmod','os.chown','os.link','os.symlink','os.truncate','os.utime','os.kill','os.killpg','subprocess.Popen','os.system','os.fork','os.posix_spawn'} or event.startswith('socket.'):raise PermissionError('Only two exact metadata reads permitted.')
 if event=='import' and args[0].split('.')[0] not in sys.stdlib_module_names:raise PermissionError('Scientific/external imports forbidden.')

def main():
 request=json.loads(sys.stdin.buffer.readline(65537));plan=request['plan'];validate(plan,request['authorization'])
 if globals().get('SOURCE_SHA256')!=plan['remote_source_sha256']:raise ValueError('Reviewed inspector source changed.')
 if sys.platform!='linux' or platform.node()!='linux10614' or os.getuid()!=UID:raise ValueError('Exact target Linux/host/user required.')
 def expired(*args):raise TimeoutError('Original metadata read deadline reached; no reset.')
 signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,20);sys.addaudithook(readonly)
 try:result=inspect(ROOT,UID,DEVICE,CONTROL,SESSION)
 except Exception as error:result={'kind':KIND+'-refusal','scratch':ROOT,'control_sha256':CONTROL,'session_sha256':SESSION,'error_type':type(error).__name__,'error':str(error)[:2048],'server_file_writes':False,'array_contents_read':False,'model_GPU_proof_or_paid_work':False,'scientific_qualified':False,'training_admitted':False}
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 data=encoded(result)
 if len(data)>LIMIT:raise ValueError('Metadata response ceiling.')
 sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
if __name__=='__main__':main()
