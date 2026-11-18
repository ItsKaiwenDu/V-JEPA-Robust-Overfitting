"""New metadata-read checks: inert JSON/files, explicit fake host/process/clock I/O."""
import base64,copy,json,os,stat
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from models import pgd_publication_metadata_v1 as client
remote=client.remote

def inert_saved(tmp_path):
 root=tmp_path/'owned';root.mkdir(mode=0o700)
 control=remote.encoded({'inert_control':True});csha=remote.sha(control)
 session=remote.encoded({'control_sha256':csha,'phase':'failed','error_type':'InertFixture','error':'No experiment ran.'})
 for name,data in [('control.json',control),('session.json',session)]:client.private_write(root/name,data)
 return root,root.stat().st_uid,root.stat().st_dev,csha,remote.sha(session)

def test_metadata_read_returns_only_hash_bound_session_and_never_opens_other_files(tmp_path):
 args=inert_saved(tmp_path);root=args[0]
 (root/'do-not-read.npy').symlink_to('/nonexistent-inert-target')
 before={p.name:(p.lstat().st_size,p.lstat().st_mtime_ns,p.lstat().st_mode) for p in root.iterdir()}
 result=remote.inspect(str(root),*args[1:])
 assert base64.b64decode(result['session_base64'])==(root/'session.json').read_bytes()
 assert not any(result[k] for k in ('server_file_writes','array_contents_read','model_GPU_proof_or_paid_work','scientific_qualified','training_admitted'))
 assert {p.name:(p.lstat().st_size,p.lstat().st_mtime_ns,p.lstat().st_mode) for p in root.iterdir()}==before

@pytest.mark.parametrize('change',['symlink','hardlink','mode','wrong-hash','oversize'])
def test_saved_session_identity_and_size_refuse_before_promotion(tmp_path,change):
 args=inert_saved(tmp_path);root=args[0];p=root/'session.json'
 if change=='symlink':
  p.rename(root/'inert-backup.json');p.symlink_to(root/'inert-backup.json')
 elif change=='hardlink':os.link(p,root/'inert-link.json')
 elif change=='mode':p.chmod(0o644)
 elif change=='wrong-hash':p.write_bytes(b'changed inert metadata')
 else:p.write_bytes(b'0'*(256*1024+1))
 with pytest.raises(ValueError):remote.inspect(str(root),*args[1:])

@pytest.mark.parametrize('change',['mode','device','owner','control'])
def test_root_binding_or_original_control_refuses(tmp_path,change):
 args=list(inert_saved(tmp_path));root=args[0]
 if change=='mode':root.chmod(0o755)
 elif change=='device':args[2]+=1
 elif change=='owner':args[1]+=1
 else:args[3]='0'*64
 with pytest.raises(ValueError):remote.inspect(str(root),*args[1:])

def test_named_metadata_swap_during_read_refuses_and_closes_owned_descriptors(tmp_path,monkeypatch):
 args=inert_saved(tmp_path);root=args[0];original=os.read;triggered=[]
 def fake_read(fd,size):
  data=original(fd,size)
  if not triggered:
   triggered.append(True);p=root/'control.json';p.rename(root/'preserved-control.json');client.private_write(p,b'inert swapped metadata')
  return data
 with monkeypatch.context() as m:
  m.setattr(remote.os,'read',fake_read)
  with pytest.raises(ValueError,match='changed during read'):remote.inspect(str(root),*args[1:])
 # A new ordinary descriptor can read the preserved file after refusal; no fd reuse is relied on.
 assert (root/'preserved-control.json').is_file()

def test_audit_hook_blocks_mutation_process_network_and_science_but_allows_read():
 for event,args in [('open',('x','wb',os.O_WRONLY)),('os.rename',()),('os.kill',()),('subprocess.Popen',()),('socket.connect',()),('import',('torch',))]:
  with pytest.raises(PermissionError):remote.readonly(event,args)
 remote.readonly('open',('session.json','rb',os.O_RDONLY));remote.readonly('import',('json',))

def reviewed_plan(folder):
 plan=json.loads((client.DEFAULT/'PLAN.json').read_bytes());plan['local_directory']=str(folder)
 return plan

def fixture_handoff(tmp_path,monkeypatch,armed=True):
 folder=tmp_path/'handoff';folder.mkdir(mode=0o700);plan=reviewed_plan(folder)
 client.private_write(folder/'PLAN.json',remote.encoded(plan))
 client.private_write(folder/'known-hosts',(remote.HOST+' '+client.HOST_KEY+'\n').encode())
 if armed:client.private_write(folder/'AUTHORIZATION.json',remote.encoded(remote.authorization(plan,'INERT LOCAL TEST; no target approval or work.')))
 monkeypatch.setattr(client,'DEFAULT',folder)
 return folder,plan

@pytest.mark.parametrize('field,value',[('host','linux10603.dc.engr.scu.edu'),('scratch','/var/tmp/other'),('session_sha256','0'*64),('read_seconds',21),('transport_seconds',181),('remote_writes',True)])
def test_new_authorization_cannot_widen_exact_read_scope(tmp_path,field,value):
 plan=reviewed_plan(tmp_path);plan[field]=value
 with pytest.raises(ValueError):remote.validate(plan,remote.authorization(plan,'inert scope drift'))

def test_unarmed_client_creates_no_claim_or_transport(tmp_path,monkeypatch):
 folder,plan=fixture_handoff(tmp_path,monkeypatch,armed=False)
 with pytest.raises(FileNotFoundError):client.run(folder,transport=lambda *a:pytest.fail('No transport'),clock=lambda:1.)
 assert not (folder/'READ-ATTEMPT.json').exists()

@pytest.mark.parametrize('drift',['host-key','source-pin','evidence-pin','copied-directory'])
def test_client_refuses_stale_bindings_before_claim(tmp_path,monkeypatch,drift):
 folder,plan=fixture_handoff(tmp_path,monkeypatch)
 if drift=='host-key':(folder/'known-hosts').write_bytes(b'inert wrong public key')
 elif drift=='copied-directory':plan['local_directory']=str(tmp_path/'copy')
 elif drift=='source-pin':plan['source_pins']['models/pgd_publication_metadata_v1.py']='0'*64
 else:plan['evidence_pins']={}
 if drift!='host-key':
  (folder/'PLAN.json').write_bytes(remote.encoded(plan));(folder/'AUTHORIZATION.json').write_bytes(remote.encoded(remote.authorization(plan,'inert binding drift')))
 with pytest.raises(ValueError):client.check_local(folder)
 assert not (folder/'READ-ATTEMPT.json').exists()

def test_pinned_command_has_independent_remote_timeout_and_no_forwarded_agent(tmp_path):
 plan=reviewed_plan(tmp_path);argv,source=client.command(tmp_path,plan)
 assert argv[0]=='ssh' and 'ForwardAgent=no' in argv and 'ClearAllForwardings=yes' in argv and 'ConnectionAttempts=1' in argv
 assert argv[-2]=='kdu@linux10614.dc.engr.scu.edu'
 assert argv[-1].startswith('exec timeout -k 1s 20s "$HOME/.venvs/vjepa/bin/python" -I -S -B -u -c ')
 assert remote.sha(source)==plan['remote_source_sha256'] and len(source)<65536

class FakeStream:
 def __init__(self,fd):self.fd=fd;self.closed=False
 def fileno(self):return self.fd
 def close(self):self.closed=True
class FakeSelector:
 def __init__(self):self.items={}
 def register(self,f,event,data):self.items[f.fileno()]=NS(fd=f.fileno(),fileobj=f,data=data)
 def unregister(self,f):del self.items[f.fileno()]
 def get_map(self):return self.items
 def select(self,timeout):return [(next(iter(self.items.values())),1)] if self.items else []
 def close(self):self.items.clear()
class FakeProcess:
 def __init__(self):self.stdin=FakeStream(801);self.stdout=FakeStream(802);self.stderr=FakeStream(803);self.returncode=None;self.terminated=0;self.killed=0
 def poll(self):return self.returncode
 def wait(self,timeout):self.returncode=0;return 0
 def terminate(self):self.terminated+=1
 def kill(self):self.killed+=1

@pytest.mark.parametrize('mode',['complete','response-cap','stderr-cap','deadline'])
def test_transport_bounds_retain_bytes_and_only_clean_owned_fake_SSH(tmp_path,monkeypatch,mode):
 folder=tmp_path/'transport';folder.mkdir(mode=0o700);process=FakeProcess();responses={802:bytearray(b'{}' if mode!='response-cap' else b'0'*(remote.LIMIT+1)),803:bytearray(b'' if mode!='stderr-cap' else b'e'*16385)};sent=[]
 def fake_read(fd,size):
  data=bytes(responses[fd][:size]);del responses[fd][:size];return data
 def fake_write(fd,data):sent.append(data);return len(data)
 with monkeypatch.context() as m:
  m.setattr(client.subprocess,'Popen',lambda *a,**kw:process);m.setattr(client.selectors,'DefaultSelector',FakeSelector)
  m.setattr(client.os,'set_blocking',lambda *a:None);m.setattr(client.os,'read',fake_read);m.setattr(client.os,'write',fake_write)
  if mode=='complete':assert client.run_ssh(['INERT FAKE SSH'],b'inert request',folder,180.,lambda:1.)==0
  else:
   with pytest.raises((TimeoutError,ValueError)):client.run_ssh(['INERT FAKE SSH'],b'inert request',folder,0. if mode=='deadline' else 180.,lambda:1.)
 assert process.stdin.closed and process.stdout.closed and process.stderr.closed
 assert process.terminated==(0 if mode=='complete' else 1) and process.killed==0
 assert (folder/'response.json').stat().st_size<=remote.LIMIT and (folder/'ssh-stderr.log').stat().st_size<=16384
 assert all(stat.S_IMODE(p.stat().st_mode)==0o600 for p in folder.iterdir())

def test_inert_refusal_lifecycle_consumes_once_and_never_retries(tmp_path,monkeypatch):
 folder,plan=fixture_handoff(tmp_path,monkeypatch);calls=[]
 def fake(argv,payload,directory,deadline,clock):
  calls.append((argv,payload,deadline));assert deadline==190. and directory==folder
  value={'kind':remote.KIND+'-refusal','scratch':remote.ROOT,'control_sha256':remote.CONTROL,'session_sha256':remote.SESSION,'error_type':'InertFixture','error':'No real SSH.'}
  value.update({k:False for k in ('server_file_writes','array_contents_read','model_GPU_proof_or_paid_work','scientific_qualified','training_admitted')})
  client.private_write(folder/'response.json',remote.encoded(value));client.private_write(folder/'ssh-stderr.log',b'');return 0
 assert client.run(folder,transport=fake,clock=lambda:10.)==1
 claim=(folder/'READ-ATTEMPT.json').read_bytes();receipt=json.loads((folder/'READ-RECEIPT.json').read_bytes())
 assert receipt['read_refused'] and not receipt['scientific_qualified'] and len(calls)==1
 with pytest.raises(ValueError,match='consumed'):client.run(folder,transport=fake,clock=lambda:20.)
 assert (folder/'READ-ATTEMPT.json').read_bytes()==claim and len(calls)==1

def test_client_transport_failure_preserves_one_use_claim_and_partial_response(tmp_path,monkeypatch):
 folder,plan=fixture_handoff(tmp_path,monkeypatch)
 def fake(*args):client.private_write(folder/'response.json',b'partial inert response');raise TimeoutError('fake deadline')
 with pytest.raises(TimeoutError):client.run(folder,transport=fake,clock=lambda:5.)
 assert json.loads((folder/'READ-ATTEMPT.json').read_bytes())['consumed']
 assert json.loads((folder/'READ-CLIENT-REFUSAL.json').read_bytes())['error_type']=='TimeoutError'
 assert (folder/'response.json').read_bytes()==b'partial inert response' and not (folder/'READ-RECEIPT.json').exists()

def test_response_rejects_wrong_hash_and_false_qualification():
 value={'kind':remote.KIND+'-result','scratch':remote.ROOT,'control_sha256':remote.CONTROL,'session_sha256':remote.SESSION,'session_bytes':2,'session_base64':base64.b64encode(b'{}').decode()}
 value.update({k:False for k in ('server_file_writes','array_contents_read','model_GPU_proof_or_paid_work','scientific_qualified','training_admitted')})
 with pytest.raises(ValueError,match='hash/extent'):client.validate_response(remote.encoded(value))
 value['scientific_qualified']=True
 with pytest.raises(ValueError,match='flags'):client.validate_response(remote.encoded(value))
