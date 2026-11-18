"""New priority codec tests: inert byte files and explicit fake payload/clock APIs.

The real full-header case uses unchanged source pins with synthetic metadata.
Round trips replace asset descriptors with tiny inert files, not model inputs.
"""
import base64,copy,hashlib,io,json,os,stat,struct,sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_priority_collection_v1 as m
b,s,old=m.boot,m.scratch,m.legacy
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'runs/pretrained/diagnostic/pgd-publication-handoff-v1-20261117/handoff-parent/linux10614-final-unarmed'
A='capture/results/numerical/objective-0/input.npy';B='capture/results/numerical/objective-1/input.npy';C='capture/results/numerical/objective-2/input.npy'
DATA=b'INERT REPEATED BYTE FIXTURE; NEVER AN ARRAY OR MODEL INPUT.\n'*11


def write(root,name,data):
 p=root/name;p.parent.mkdir(mode=0o700,parents=True,exist_ok=True);q=p.parent
 while q!=root:q.chmod(0o700);q=q.parent
 with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 return p

@pytest.fixture(scope='module')
def control():
 outer=json.loads((FOLDER/'CONTROL-CANDIDATE.json').read_bytes());raw=outer['supervised_control_text'].encode();c,p,sha=b.control(raw)
 auth=b.encoded({'kind':'pgd-split-supervised-university-exception-v1','approved':True,'control_sha256':sha,'maximum_new_attempts':1,
  'user_basis':'EXPLICIT LOCAL INERT TRANSPORT FIXTURE; NOT TARGET APPROVAL'})
 ib,isha=b.installation(c,sha);layout=s.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),ib,isha)
 return NS(raw=raw,c=c,p=p,sha=sha,auth=auth,ib=ib,isha=isha,layout=layout)

def fake_session(c):
 return {'kind':'pgd-split-parent-session-v1','control_sha256':c.sha,'parent_pid':os.getpid(),'connection_started':0.,'work_deadline':2400.,'connection_deadline':3000.,
  'phase':'failed','complete':False,'collection_pending':True,'current':None,'completed':[],'receipts':[],
  'error_type':'INERT-FAKE-WORK-STATUS','error':'No computation was performed.',**m.FALSE_FLAGS}

def full_metadata(c):
 st=b.encoded(fake_session(c));window=old.Window(0.,100.,clock=lambda:100.)
 receipt=old.work_receipt(st,c.sha,b.digest(c.auth),c.isha,window)
 extra={'control.json':c.raw,'authorization.json':c.auth,'session.json':st,old.RECEIPT:b.encoded(receipt),old.STATUS:b.encoded({'EXPLICIT':'FAKE FULL METADATA INVENTORY ONLY'})}
 pins=dict(c.layout.assets);pins.update({n:{'bytes':len(v),'sha256':b.digest(v)} for n,v in extra.items()})
 files={n:dict(row,mode=0o600,mtime_ns=1000000001+i) for i,(n,row) in enumerate(sorted(pins.items()))}
 inv={n:{'identity':[500,100+i,os.getuid(),0o600,1],'bytes':v['bytes'],'allocated_bytes':((v['bytes']+4095)//4096)*4096,'mtime_ns':v['mtime_ns']} for i,(n,v) in enumerate(files.items())}
 dirs={'.'}
 for name in inv:dirs.update(str(p) for p in Path(name).parents)
 directories={n:{'identity':[500,2000+i,os.getuid(),0o700],'allocated_bytes':4096} for i,n in enumerate(sorted(dirs))}
 storage=c.layout.assess(inv,directories,{'available_bytes':8*s.GiB,'available_inodes':2048},phase='collection')
 return {'kind':'pgd-split-evidence-stream-v1','control_sha256':c.sha,'authorization_sha256':b.digest(c.auth),
  'root_binding':{'root':'/EXPLICIT-FAKE/scratch','root_identity':list(directories['.']['identity']),'parent':'/EXPLICIT-FAKE','parent_identity':[500,1,os.getuid(),0o700]},
  'files':files,'inventory':inv,'directories':directories,'storage':storage,'final_receipt_sha256':b.digest(extra[old.RECEIPT]),'session_sha256':b.digest(st),
  'connection_started':0.,'collection_started':100.,'collection_deadline':700.,'connection_deadline':3000.,'logical_bytes':sum(v['bytes'] for v in files.values()),'omitted_content':[],**m.FALSE_FLAGS}


def test_full_unchanged_172_asset_metadata_validator_is_mandatory_before_new_envelope(control,tmp_path):
 base=full_metadata(control);h=m.make_header(base,control.raw,control.auth);layout=m.validate_header(h,control.raw,control.auth)
 assert len(layout.assets)==172 and len(base['files'])==177 and h['records'][0]['name']=='session.json'
 assert h['base']==base and h['expanded_bytes']==base['logical_bytes'] and b.encoded(h)==b.encoded(json.loads(b.encoded(h)))
 assert m.WIRE_CAP==old.WIRE_CAP==3*s.GiB and m.HEADER_CAP==old.HEADER_CAP and m.JOURNAL_CAP<64*s.MiB
 assert not h['scientific_qualified'] and 'torch' not in sys.modules and 'numpy' not in sys.modules
 (tmp_path/'FULL-172-ASSET-PRIORITY-HEADER-METADATA-ONLY.json').write_bytes(b.encoded(h))

@pytest.mark.parametrize('fault',['missing-source','source-hash','clock-reset','training','inventory-link','unknown-path'])
def test_new_envelope_does_not_weaken_full_legacy_source_clock_inventory_or_admission_gates(control,fault):
 base=full_metadata(control)
 if fault=='missing-source':base['files'].pop('original-packet.tar.gz')
 elif fault=='source-hash':base['files']['original-packet.tar.gz']['sha256']='f'*64
 elif fault=='clock-reset':base['collection_deadline']+=1
 elif fault=='training':base['training_admitted']=True
 elif fault=='inventory-link':base['inventory']['original-packet.tar.gz']['identity'][4]=3
 else:base['files']['unknown.npy']={'bytes':1,'sha256':'a'*64,'mode':0o600,'mtime_ns':1}
 with pytest.raises((ValueError,KeyError,TypeError)):m.make_header(base,control.raw,control.auth)

@pytest.fixture
def tiny(tmp_path,control,monkeypatch):
 root=tmp_path/'INERT-SUBSTITUTED-SCRATCH';root.mkdir(mode=0o700);layout=copy.deepcopy(control.layout)
 selected=['original-packet.tar.gz','code/models/__init__.py','overlay/models/__init__.py','configs/base.json','configs/design.json',*('supervisor/'+n for n in b.SUPERVISORS.values())]
 data={n:b'INERT DUPLICATED SOURCE BYTES; NEVER EXECUTED.\n' for n in selected};data['installation.json']=control.ib
 layout.assets={n:{'bytes':len(v),'sha256':b.digest(v)} for n,v in data.items()}
 data.update({'control.json':control.raw,'authorization.json':control.auth,'session.json':b.encoded(fake_session(control)),
  'emergency-space.bin':b'FAKE TINY RESERVE; NO ALLOCATION PROOF','cache/ignored.bin':b'NEVER-TRANSFER-CACHE','tmp/ignored.bin':b'NEVER-TRANSFER-TMP',
  A:DATA,B:DATA,C:b'OTHER INERT BYTE DATA\n','capture/logs/numerical.log':b'EXPLICIT INERT FAILED WORK; NO CHILD OR GPU WAS STARTED\n'})
 for index,(name,value) in enumerate(data.items()):
  p=write(root,name,value);os.utime(p,ns=(1000000001+index,1000000001+index))
 (root/'capture/results').mkdir(mode=0o700,parents=True,exist_ok=True)
 # Only declared asset bytes are tiny fakes; actual guard/read/stream/copy/journal
 # ownership and hash paths execute. Full-header tests above do not substitute.
 monkeypatch.setattr(s,'Layout',lambda *a,**kw:layout)
 guard=s.Guard(root,s.identity_binding(root),layout);now=[100.]
 collector=m.Collector(guard,control.raw,control.auth,connection_started=0.,clock=lambda:now[0])
 return NS(root=root,layout=layout,guard=guard,now=now,collector=collector,data=data,control=control)


def encode(tiny):
 h=tiny.collector.prepare();out=io.BytesIO();receipt=tiny.collector.stream(out);return h,out.getvalue(),receipt

def scan_wire(data):
 n=struct.unpack('!I',data[len(m.MAGIC):len(m.MAGIC)+4])[0];start=len(m.MAGIC)+4;h=b.parsed(data[start:start+n]);pos=start+n;frames=[]
 for planned in h['records']:
  first=pos;assert data[pos:pos+len(m.RECORD)]==m.RECORD;pos+=len(m.RECORD);n=struct.unpack('!I',data[pos:pos+4])[0];pos+=4
  record=b.parsed(data[pos:pos+n]);pos+=n;payload=pos;pos+=h['base']['files'][record['name']]['bytes'] if record['kind']=='data' else 0
  frames.append(NS(first=first,payload=payload,end=pos,record=record))
 return h,frames,pos

def parent_for(tmp_path):p=tmp_path/'RECEIVER';p.mkdir(mode=0o700);return p

def receive(data,tiny,parent,*,source=None,window=None):
 return m.receive(io.BytesIO(data) if source is None else source,parent/'saved',tiny.control.raw,tiny.control.auth,
  window=window or old.Window(0.,100.,clock=lambda:100.))

def journal(parent):return parent/'saved-transport'

def assert_partial(parent):
 p=journal(parent)
 if (p/'PARTIAL-RECEIPT.json').exists():
  v=b.parsed((p/'PARTIAL-RECEIPT.json').read_bytes());assert not v['transport_verified'] and not v['stream_complete'] and not v['scientific_qualified']
 assert not any(p.name in ('COMPLETE-RECEIPT.json','COLLECTION-RECEIPT.json') for p in parent.rglob('*'))


def test_byte_roundtrip_priorities_dedup_independent_inodes_modes_mtimes_and_hash_chained_journal(tiny,tmp_path):
 h,data,remote=encode(tiny);decoded,frames,end=scan_wire(data);parent=parent_for(tmp_path);receipt=receive(data,tiny,parent)
 assert b.encoded(decoded)==b.encoded(h) and frames[0].record['name']=='session.json' and frames[1].record['name']==old.RECEIPT
 assert h['deduplicated_bytes']>=len(DATA)>0 and h['expanded_bytes']==sum(v['bytes'] for v in h['base']['files'].values())
 assert receipt['transport_verified'] and not receipt['scientific_qualified'] and not remote['receiver_verified']
 assert receipt['wire_sha256']==remote['wire_sha256']==b.digest(data) and receipt['wire_bytes']==len(data)
 for name,item in h['base']['files'].items():
  actual=parent/'saved'/name;assert actual.read_bytes()==(tiny.root/name).read_bytes()
  assert actual.stat().st_nlink==1 and stat.S_IMODE(actual.stat().st_mode)==0o600 and actual.stat().st_mtime_ns==item['mtime_ns']
 assert (parent/'saved'/A).stat().st_ino!=(parent/'saved'/B).stat().st_ino and (parent/'saved'/A).stat().st_mtime_ns!=(parent/'saved'/B).stat().st_mtime_ns
 assert b'NEVER-TRANSFER-CACHE' not in data and not (parent/'saved/cache').exists()
 assert (journal(parent)/'HEADER.json').read_bytes()==b.encoded(h)
 rows=[b.parsed(line) for line in (journal(parent)/'VERIFIED-FILES.jsonl').read_bytes().splitlines()];previous='0'*64
 for i,row in enumerate(rows):
  sha=row['row_sha256'];content=dict(row);del content['row_sha256']
  assert sha==b.digest(b.encoded(content)) and row['previous_sha256']==previous and row['sequence']==i
  assert row['descriptor']==receipt['files'][row['file']] and not row['transport_verified'];previous=sha
 assert len(rows)==len(h['records']) and receipt['journal']['last_row_sha256']==previous
 assert_partial(parent)
 (tmp_path/'INERT-PRIORITY-ROUNDTRIP.json').write_bytes(b.encoded({'header':h,'receipt':receipt,'fake_asset_bytes_only':True,'model_GPU_proof_or_child_work':False}))


def test_failure_status_and_validated_header_survive_cutoff_before_first_large_frame(tiny,tmp_path):
 h,data,_=encode(tiny);_,frames,_=scan_wire(data);first=next(v for v in frames if v.record['name']==A);parent=parent_for(tmp_path)
 with pytest.raises(ValueError):receive(data[:first.payload+7],tiny,parent)
 assert (parent/'saved/session.json').read_bytes()==tiny.data['session.json']
 assert (parent/'saved'/old.RECEIPT).is_file() and (journal(parent)/'HEADER.json').read_bytes()==b.encoded(h)
 rows=[b.parsed(v) for v in (journal(parent)/'VERIFIED-FILES.jsonl').read_bytes().splitlines()]
 # The unchanged exact reader buffers the unverified short tail before EOF.
 assert rows and all(v['file']!=A for v in rows) and (parent/'saved'/A).stat().st_size==0
 assert_partial(parent)
 (tmp_path/'INERT-INTERRUPTED-PRIORITY-PREFIX.json').write_bytes(b.encoded({'verified_prefix':[v['file'] for v in rows],'interrupted_raw_file':A,'whole_transfer_complete':False,'fake_asset_bytes_only':True}))

@pytest.mark.parametrize('fault',['empty','legacy-magic','header-cap','truncated-header','bad-record-magic','data-byte','missing-terminal','terminal-hash','terminal-truncated','trailing','noncanonical','duplicate-json'])
def test_damaged_priority_stream_never_returns_completion_and_preserves_partials(tiny,tmp_path,fault):
 h,data,_=encode(tiny);_,frames,end=scan_wire(data);header_start=len(m.MAGIC)+4;header_end=frames[0].first
 if fault=='empty':data=b''
 elif fault=='legacy-magic':data=old.MAGIC+data[len(m.MAGIC):]
 elif fault=='header-cap':data=m.MAGIC+struct.pack('!I',m.HEADER_CAP+1)
 elif fault=='truncated-header':data=data[:header_start+5]
 elif fault=='bad-record-magic':data=data[:header_end]+b'!'+data[header_end+1:]
 elif fault=='data-byte':
  at=frames[0].payload;data=data[:at]+bytes([data[at]^1])+data[at+1:]
 elif fault=='missing-terminal':data=data[:end]
 elif fault=='terminal-truncated':data=data[:-10]
 elif fault=='trailing':data+=b'EXTRA'
 elif fault=='terminal-hash':
  start=end+len(m.END)+4;value=b.parsed(data[start:]);value['prefix_sha256']='0'*64;raw=b.encoded(value);data=data[:end]+m.END+struct.pack('!I',len(raw))+raw
 else:
  raw=data[header_start:header_end]
  raw=raw+b'\n' if fault=='noncanonical' else raw.rstrip()[:-1]+b',"scientific_qualified":false}\n'
  data=m.MAGIC+struct.pack('!I',len(raw))+raw+data[header_end:]
 parent=parent_for(tmp_path)
 with pytest.raises((ValueError,OSError)):receive(data,tiny,parent)
 assert_partial(parent)

@pytest.mark.parametrize('fault',['order','forward-ref','ref-chain','wrong-hash-ref','wrong-payload','expansion','admission','extra-key','duplicate-record'])
def test_manifest_plan_rejects_changed_order_forward_chained_or_wrong_content_references(tiny,fault):
 h=tiny.collector.prepare();v=copy.deepcopy(h);refs=[i for i,r in enumerate(v['records']) if r['kind']=='reference'];assert len(refs)>=2
 if fault=='order':v['records'][0],v['records'][1]=v['records'][1],v['records'][0]
 elif fault=='forward-ref':v['records'][refs[0]]['source']=v['records'][refs[-1]]['name']
 elif fault=='ref-chain':v['records'][refs[-1]]['source']=v['records'][refs[0]]['name']
 elif fault=='wrong-hash-ref':v['records'][refs[0]]['source']='session.json'
 elif fault=='wrong-payload':v['payload_bytes']+=1
 elif fault=='expansion':v['expanded_bytes']+=1
 elif fault=='admission':v['scientific_qualified']=True
 elif fault=='extra-key':v['new_grant']=True
 else:v['records'].append(copy.deepcopy(v['records'][0]))
 with pytest.raises(ValueError):m.validate_header(v,tiny.control.raw,tiny.control.auth)


def test_receiver_rejects_reference_frame_that_disagrees_with_manifest(tiny,tmp_path):
 h,data,_=encode(tiny);_,frames,_=scan_wire(data);frame=next(v for v in frames if v.record['name']==B);bad=dict(frame.record,source=B);encoded=b.encoded(bad)
 data=data[:frame.first]+m.RECORD+struct.pack('!I',len(encoded))+encoded+data[frame.payload:]
 parent=parent_for(tmp_path)
 with pytest.raises(ValueError,match='Unexpected, duplicate, stale or forward'):receive(data,tiny,parent)
 assert_partial(parent)

class HookStream(io.BytesIO):
 def __init__(self,data,at,callback):super().__init__(data);self.at=at;self.callback=callback;self.called=False
 def read(self,size=-1):
  if not self.called and self.tell()>=self.at:self.called=True;self.callback()
  return super().read(size)

@pytest.mark.parametrize('fault',['inode','symlink','hardlink','bytes','mtime','ancestor'])
def test_reference_reconstruction_refuses_changed_previously_verified_local_source(tiny,tmp_path,fault):
 h,data,_=encode(tiny);_,frames,_=scan_wire(data);at=next(v.first for v in frames if v.record['name']==B);parent=parent_for(tmp_path)
 def mutate():
  p=parent/'saved'/A;before=p.stat()
  if fault=='inode':
   q=write(parent,'replacement',DATA);os.utime(q,ns=(before.st_mtime_ns,before.st_mtime_ns));os.replace(q,p)
  elif fault=='symlink':p.unlink();q=write(parent,'outside-inert-bytes',DATA);p.symlink_to(q)
  elif fault=='hardlink':os.link(p,parent/'external-alias')
  elif fault=='bytes':p.write_bytes(bytes([DATA[0]^1])+DATA[1:]);os.utime(p,ns=(before.st_mtime_ns,before.st_mtime_ns))
  elif fault=='mtime':os.utime(p,ns=(1,2))
  else:p.parent.rename(p.parent.with_name('moved-inert-ancestor'))
 stream=HookStream(data,at,mutate)
 with pytest.raises((ValueError,OSError)):receive(data,tiny,parent,source=stream)
 assert stream.called;assert_partial(parent)


def test_sender_rehashes_duplicate_source_and_refuses_same_extent_changed_bytes(tiny):
 h=tiny.collector.prepare();assert next(v for v in h['records'] if v['name']==B)['kind']=='reference'
 p=tiny.root/B;before=p.stat();p.write_bytes(bytes([DATA[0]^1])+DATA[1:]);os.utime(p,ns=(before.st_mtime_ns,before.st_mtime_ns));out=io.BytesIO()
 with pytest.raises(ValueError,match='Source hash changed'):tiny.collector.stream(out)
 assert m.END not in out.getvalue()
 with pytest.raises(ValueError,match='One prepared priority stream'):tiny.collector.stream(io.BytesIO())


def test_additional_plan_work_cannot_extend_original_120_second_preparation(tiny,monkeypatch):
 original=m.make_header
 def delayed(*args):
  h=original(*args);tiny.now[0]=220.;return h
 monkeypatch.setattr(m,'make_header',delayed)
 with pytest.raises(TimeoutError):tiny.collector.prepare()
 with pytest.raises(ValueError,match='One priority preparation'):tiny.collector.prepare()


def test_original_receiver_deadline_applies_during_reference_copy_without_reset(tiny,tmp_path,monkeypatch):
 h,data,_=encode(tiny);parent=parent_for(tmp_path);now=[100.];window=old.Window(0.,100.,clock=lambda:now[0]);original=m._prior_chunks
 def expire(*args):
  now[0]=700.;yield from original(*args)
 monkeypatch.setattr(m,'_prior_chunks',expire)
 with pytest.raises(TimeoutError):receive(data,tiny,parent,window=window)
 assert window.deadline==700.;assert_partial(parent)


def test_expansion_budget_cannot_be_replaced_by_smaller_unique_payload(tiny,monkeypatch):
 base=tiny.collector.original.prepare();h=m.make_header(base,tiny.control.raw,tiny.control.auth);assert h['payload_bytes']<h['expanded_bytes']
 monkeypatch.setattr(m,'WIRE_CAP',h['expanded_bytes']+m.JOURNAL_CAP-1)
 with pytest.raises(ValueError,match='expansion/disk/response'):m.make_header(base,tiny.control.raw,tiny.control.auth)


def test_progress_metadata_cap_refuses_without_discarding_header_or_marking_complete(tiny,tmp_path,monkeypatch):
 h,data,_=encode(tiny);parent=parent_for(tmp_path);monkeypatch.setattr(m,'PROGRESS_CAP',16)
 with pytest.raises(ValueError,match='Progress/journal byte cap'):receive(data,tiny,parent)
 assert (journal(parent)/'HEADER.json').is_file() and (journal(parent)/'VERIFIED-FILES.jsonl').stat().st_size==0
 assert_partial(parent)


def test_final_allocated_disk_sample_stays_inside_original_disk_envelope(tiny,tmp_path,monkeypatch):
 h,data,_=encode(tiny);parent=parent_for(tmp_path);sample=m._disk_sample
 def fake(*args):return sample(*args)+m.WIRE_CAP
 monkeypatch.setattr(m,'_disk_sample',fake)
 with pytest.raises(ValueError,match='allocated disk ceiling'):receive(data,tiny,parent)
 assert_partial(parent)


def test_journal_content_tampering_is_detected_even_if_extent_and_mtime_restored(tiny,tmp_path):
 h,data,_=encode(tiny);_,frames,_=scan_wire(data);parent=parent_for(tmp_path);at=next(v.first for v in frames if v.record['name']==B)
 def mutate():
  p=journal(parent)/'VERIFIED-FILES.jsonl';st=p.stat();raw=p.read_bytes();p.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(p,ns=(st.st_mtime_ns,st.st_mtime_ns))
 with pytest.raises((ValueError,OSError)):receive(data,tiny,parent,source=HookStream(data,at,mutate))
 assert_partial(parent)

@pytest.mark.parametrize('fault',['destination','journal','parent-mode'])
def test_existing_output_or_nonprivate_parent_never_overwritten(tiny,tmp_path,fault):
 h,data,_=encode(tiny);parent=parent_for(tmp_path)
 if fault=='destination':(parent/'saved').mkdir(mode=0o700);write(parent/'saved','KEEP',b'preserve')
 elif fault=='journal':(parent/'saved-transport').mkdir(mode=0o700);write(parent/'saved-transport','KEEP',b'preserve')
 else:parent.chmod(0o755)
 with pytest.raises(ValueError):receive(data,tiny,parent)
 if fault!='parent-mode':assert (parent/('saved' if fault=='destination' else 'saved-transport')/'KEEP').read_bytes()==b'preserve'


class FragmentedOutput(io.BytesIO):
 def write(self,data):return super().write(data[:13])
class FragmentedInput(io.BytesIO):
 def read(self,size=-1):return super().read(min(size,17) if size>=0 else 17)

def test_additional_fragmented_reads_and_short_writes_keep_exact_wire_hash_and_copy_bytes(tiny,tmp_path):
 h=tiny.collector.prepare();out=FragmentedOutput();remote=tiny.collector.stream(out);data=out.getvalue();parent=parent_for(tmp_path)
 receipt=receive(data,tiny,parent,source=FragmentedInput(data))
 assert receipt['transport_verified'] and receipt['wire_sha256']==remote['wire_sha256']==b.digest(data)
 assert (parent/'saved'/A).read_bytes()==(parent/'saved'/B).read_bytes()==DATA
 assert receipt['expanded_bytes']==h['expanded_bytes'] and (parent/'saved'/A).stat().st_ino!=(parent/'saved'/B).stat().st_ino


def test_additional_zero_length_duplicates_reconstruct_as_distinct_empty_regular_files(tiny,tmp_path):
 for name in (A,B):(tiny.root/name).write_bytes(b'')
 h,data,_=encode(tiny);parent=parent_for(tmp_path);receipt=receive(data,tiny,parent)
 assert next(v for v in h['records'] if v['name']==B)=={'name':B,'kind':'reference','source':A}
 assert (parent/'saved'/A).stat().st_size==(parent/'saved'/B).stat().st_size==0
 assert (parent/'saved'/A).stat().st_ino!=(parent/'saved'/B).stat().st_ino and receipt['transport_verified']


def test_additional_zero_local_write_refuses_and_retains_verified_status_prefix(tiny,tmp_path,monkeypatch):
 h,data,_=encode(tiny);parent=parent_for(tmp_path);real=os.write;triggered=[]
 def broken(fd,block):
  p=parent/'saved'/A
  if p.exists() and os.fstat(fd).st_ino==p.stat().st_ino:triggered.append(True);return 0
  return real(fd,block)
 with monkeypatch.context() as patch:
  patch.setattr(m.os,'write',broken)
  with pytest.raises(OSError,match='Short local byte reconstruction'):receive(data,tiny,parent)
 assert triggered and (parent/'saved/session.json').is_file();assert_partial(parent)

@pytest.mark.parametrize('damaged',[False,True])
def test_additional_receive_success_and_failure_close_all_owned_descriptors(tiny,tmp_path,monkeypatch,damaged):
 h,data,_=encode(tiny);parent=parent_for(tmp_path)
 if damaged:
  _,frames,_=scan_wire(data);first=next(v for v in frames if v.record['name']==A);data=data[:first.payload+7]
 tracked=set();real_open,real_dup,real_close=os.open,os.dup,os.close
 def opened(*a,**kw):fd=real_open(*a,**kw);tracked.add(fd);return fd
 def duplicated(fd):new=real_dup(fd);tracked.add(new);return new
 def closed(fd):assert fd in tracked;tracked.remove(fd);real_close(fd)
 with monkeypatch.context() as patch:
  patch.setattr(m.os,'open',opened);patch.setattr(m.os,'dup',duplicated);patch.setattr(m.os,'close',closed)
  if damaged:
   with pytest.raises(ValueError):receive(data,tiny,parent)
  else:assert receive(data,tiny,parent)['transport_verified']
 assert not tracked
 (tmp_path/'OWNED-PRIORITY-DESCRIPTORS-CLOSED.json').write_bytes(b.encoded({'damaged':damaged,'remaining_owned_descriptors':len(tracked),'only_tiny_inert_bytes':True,'no_process_GPU_or_proof':True}))


@pytest.mark.parametrize('name',['HEADER.json','PARTIAL-RECEIPT.json'])
def test_extra_journal_metadata_hash_is_rechecked_before_transport_success(tiny,tmp_path,name):
 h,data,_=encode(tiny);_,frames,_=scan_wire(data);parent=parent_for(tmp_path);at=next(v.first for v in frames if v.record['name']==B);changed=[]
 def mutate():
  p=journal(parent)/name;st=p.stat();raw=p.read_bytes();p.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(p,ns=(st.st_mtime_ns,st.st_mtime_ns));changed.append(name)
 with pytest.raises(ValueError,match='hash or held identity changed'):receive(data,tiny,parent,source=HookStream(data,at,mutate))
 assert changed==[name] and not any(p.name in ('COMPLETE-RECEIPT.json','COLLECTION-RECEIPT.json') for p in parent.rglob('*'))
 (tmp_path/'JOURNAL-METADATA-HASH-REFUSAL.json').write_bytes(b.encoded({'forged_inert_journal_file':name,'receiver_returned_success':False,'no_array_or_model_work':True}))


def test_extra_file_record_cap_is_checked_before_attempting_oversize_body_read(tiny,tmp_path):
 h,data,_=encode(tiny);_,frames,_=scan_wire(data);start=frames[0].first;wire=data[:start]+m.RECORD+struct.pack('!I',m.RECORD_CAP+1);stream=io.BytesIO(wire);parent=parent_for(tmp_path)
 with pytest.raises(ValueError,match='Bounded canonical JSON frame required'):receive(wire,tiny,parent,source=stream)
 assert stream.tell()==len(wire) and (journal(parent)/'HEADER.json').is_file() and not (parent/'saved/session.json').exists()
 assert_partial(parent)
