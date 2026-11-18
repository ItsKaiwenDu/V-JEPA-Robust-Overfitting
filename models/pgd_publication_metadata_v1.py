"""One scoped read-only metadata connection, with a new local one-use claim."""
import base64,hashlib,importlib.util,json,os,selectors,shlex,stat,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REMOTE=ROOT/'scripts/pgd_publication_metadata_remote_v1.py'
SPEC=importlib.util.spec_from_file_location('publication_metadata_remote_v1',REMOTE)
remote=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(remote)
DEFAULT=ROOT/'runs/pretrained/diagnostic/pgd-publication-metadata-v1-20261117/linux10614-unarmed'
HOST_KEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAELc+tu7f+683dQbOPg5lndZ9pGDeODvo8r1Sp8d7ri'

EVIDENCE_SHA256={'runs/pretrained/diagnostic/pgd-publication-handoff-v1-20261117/handoff-parent/linux10614-final-unarmed/ATTEMPT.json': '2eb0dc0ad3058a09f069c0b3abebf14fc90bbe5843371c33516e5f9328727424', 'runs/pretrained/diagnostic/pgd-publication-handoff-v1-20261117/handoff-parent/linux10614-final-unarmed/attended-session/ROOT-RECEIPT.json': '8ffbe347e28fb971039ac79c37ea74e4085f97c06f7588decdb93728d7acda18', 'runs/pretrained/diagnostic/pgd-publication-handoff-v1-20261117/handoff-parent/linux10614-final-unarmed/attended-session/saved/capture/results/final-receipt.json': '90deb755563a6f14de74d5005f132cda778dca2c7619c3725480ae2a48e4c744', 'runs/pretrained/diagnostic/pgd-publication-handoff-v1-20261117/collection-timeout-review-20261117/independent-saved-review.json': '29d5c70c4709125875b353830026ae88c5081a4f3cc6050094b3546db17d424a', 'runs/pretrained/diagnostic/pgd-publication-handoff-v1-20261117/collection-timeout-review-20261117/fixture-layout-preserved.json': '588fff30c4ea745663e60a5d5cb6791a09c12453e4c9d2e7bac2262c3b233fbc'}

def digest(p):
 s=p.lstat()
 if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:raise ValueError('Regular single-link local input required.')
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def private_write(path,data):
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def claim(folder,plan_sha,clock):
 private_write(folder/'READ-ATTEMPT.json',remote.encoded({'kind':remote.KIND+'-attempt','plan_sha256':plan_sha,'consumed':True,'started_monotonic':clock(),'scientific_qualified':False,'training_admitted':False}))

def check_local(folder):
 folder=Path(folder)
 if folder!=DEFAULT or folder.resolve()!=DEFAULT or folder.is_symlink() or stat.S_IMODE(folder.lstat().st_mode)!=0o700:raise ValueError('Exact reviewed private handoff required.')
 if (folder/'READ-ATTEMPT.json').exists():raise ValueError('Metadata read consumed; never repeat or reset.')
 plan=json.loads((folder/'PLAN.json').read_bytes());auth=json.loads((folder/'AUTHORIZATION.json').read_bytes())
 remote.validate(plan,auth)
 if plan['local_directory']!=str(folder) or plan['repository']!=str(ROOT):raise ValueError('Copied handoff refused.')
 required={'scripts/pgd_publication_metadata_remote_v1.py','models/pgd_publication_metadata_v1.py','scripts/run_pgd_publication_metadata_v1.py'}
 if set(plan['source_pins'])!=required or plan['remote_source_sha256']!=plan['source_pins'][str(REMOTE.relative_to(ROOT))]:raise ValueError('Exact three reviewed sources required.')
 for name,want in plan['source_pins'].items():
  if digest(ROOT/name)!=want:raise ValueError('Reviewed source changed: '+name)
 if {n:v['sha256'] for n,v in plan['evidence_pins'].items()}!=EVIDENCE_SHA256:raise ValueError('Exact five original incident pins required.')
 for name,row in plan['evidence_pins'].items():
  if not name.startswith('runs/pretrained/diagnostic/pgd-publication-handoff-v1-20261117/') or '..' in Path(name).parts:raise ValueError('Evidence pin outside exact local incident.')
  p=ROOT/name;s=p.lstat()
  if digest(p)!=row['sha256'] or (s.st_size,s.st_mtime_ns,stat.S_IMODE(s.st_mode))!=(row['bytes'],row['mtime_ns'],row['mode']):raise ValueError('Original incident evidence changed.')
 known=(folder/'known-hosts').read_bytes()
 if known!=(remote.HOST+' '+HOST_KEY+'\n').encode() or remote.sha(known)!=plan['known_hosts_sha256']:raise ValueError('Exact public SSH host pin required.')
 return plan,auth

def command(folder,plan):
 source=REMOTE.read_bytes();want=plan['remote_source_sha256']
 if remote.sha(source)!=want:raise ValueError('Inspector source drift before command.')
 wrapper='import sys,hashlib;p=sys.stdin.buffer.read('+str(len(source))+');SOURCE_SHA256=hashlib.sha256(p).hexdigest();assert SOURCE_SHA256=='+repr(want)+';exec(compile(p,"<bounded-metadata-read>","exec"))'
 argv=['ssh','-T','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(folder/'known-hosts'),'-o','GlobalKnownHostsFile=/dev/null','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-o','ForwardAgent=no','-o','ClearAllForwardings=yes','kdu@'+remote.HOST,'exec timeout -k 1s 20s "$HOME/.venvs/vjepa/bin/python" -I -S -B -u -c '+shlex.quote(wrapper)]
 return argv,source

def run_ssh(argv,payload,folder,deadline,clock):
 # Only our SSH child is ever terminated. No agent forwarding, remote writes, or retries.
 outputs={};p=None;reader=selectors.DefaultSelector()
 try:
  for name in ('response.json','ssh-stderr.log'):
   fd=os.open(folder/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);outputs[name]=os.fdopen(fd,'wb')
  p=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  # Payload is below 64 KiB; nonblocking writes share the original connection clock.
  for stream,name,limit in ((p.stdout,'response.json',remote.LIMIT),(p.stderr,'ssh-stderr.log',16384)):
   os.set_blocking(stream.fileno(),False);reader.register(stream,selectors.EVENT_READ,('read',name,limit))
  os.set_blocking(p.stdin.fileno(),False);reader.register(p.stdin,selectors.EVENT_WRITE,('write',None,None));sent=0;counts={name:0 for name in outputs}
  while reader.get_map():
   left=deadline-clock()
   if left<=0:raise TimeoutError('Original three-minute metadata connection deadline; preserve partial files, no retry.')
   for key,_ in reader.select(min(.25,left)):
    action,name,limit=key.data
    if action=='write':
     try:sent+=os.write(key.fd,payload[sent:sent+65536])
     except BrokenPipeError:reader.unregister(key.fileobj);key.fileobj.close();continue
     if sent==len(payload):reader.unregister(key.fileobj);key.fileobj.close()
    else:
     data=os.read(key.fd,min(65536,limit-counts[name]+1))
     if not data:reader.unregister(key.fileobj);key.fileobj.close();continue
     remain=limit-counts[name];outputs[name].write(data[:remain]);counts[name]+=min(len(data),remain)
     if len(data)>remain:raise ValueError('Metadata transport byte bound; preserve partial files, no retry.')
  left=deadline-clock()
  if left<=0:raise TimeoutError('Original metadata connection deadline; no reset.')
  return p.wait(timeout=left)
 finally:
  reader.close()
  if p is not None:
   if p.poll() is None:
    p.terminate()
    try:p.wait(timeout=1)
    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=1)
   for stream in (p.stdin,p.stdout,p.stderr):
    if stream is not None:stream.close()
  for f in outputs.values():f.flush();os.fsync(f.fileno());f.close()

def validate_response(data):
 if len(data)>remote.LIMIT:raise ValueError('Metadata response cap.')
 value=json.loads(data)
 if value.get('kind') not in (remote.KIND+'-result',remote.KIND+'-refusal') or value.get('scratch')!=remote.ROOT or value.get('control_sha256')!=remote.CONTROL or value.get('session_sha256')!=remote.SESSION:raise ValueError('Metadata response binding changed.')
 for k in ('server_file_writes','array_contents_read','model_GPU_proof_or_paid_work','scientific_qualified','training_admitted'):
  if value.get(k) is not False:raise ValueError('Read-only response flags changed.')
 if value['kind'].endswith('-result'):
  session=base64.b64decode(value['session_base64'],validate=True)
  if len(session)>256*1024 or len(session)!=value['session_bytes'] or remote.sha(session)!=remote.SESSION:raise ValueError('Session response hash/extent mismatch.')
 return value

def run(folder=DEFAULT,*,transport=run_ssh,clock=time.monotonic):
 folder=Path(folder);plan,auth=check_local(folder);argv,source=command(folder,plan)
 payload=source+remote.encoded({'plan':plan,'authorization':auth})
 if len(payload)>65536 or len(remote.encoded({'plan':plan,'authorization':auth}))>65536:raise ValueError('Metadata request cap.')
 started=clock();claim(folder,remote.sha(remote.encoded(plan)),lambda:started)
 try:
  code=transport(argv,payload,folder,started+180,clock)
  if code:raise RuntimeError('Metadata SSH returned '+str(code)+'; preserve files, no retry.')
  value=validate_response((folder/'response.json').read_bytes())
  private_write(folder/'READ-RECEIPT.json',remote.encoded({'kind':remote.KIND+'-receipt','plan_sha256':remote.sha(remote.encoded(plan)),'response_sha256':digest(folder/'response.json'),'SSH_exit_status':code,'read_refused':value['kind'].endswith('-refusal'),'scientific_qualified':False,'training_admitted':False}))
 except BaseException as error:
  private_write(folder/'READ-CLIENT-REFUSAL.json',remote.encoded({'kind':remote.KIND+'-client-refusal','error_type':type(error).__name__,'message':str(error)[:2048],'consumed':True,'scientific_qualified':False,'training_admitted':False}));raise
 print('Metadata-only response saved at '+str(folder)+'. Do not repeat. No model work ran.')
 return 1 if value['kind'].endswith('-refusal') else 0
