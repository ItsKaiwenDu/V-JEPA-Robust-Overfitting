"""New inert byte/metadata transport checks; no scientific/OS launch work.

Full 172-asset manifests below are metadata-only. Round trips explicitly replace
asset descriptors with tiny inert files; they are not a target installation test.
"""
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import sys
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_split_bootstrap_v1 as b
from models import predictor_pgd_split_coordinator_v1 as coordinator
from models import predictor_pgd_split_scratch_v1 as s
from models import predictor_pgd_split_collection_v1 as m

ROOT = Path(__file__).resolve().parents[1]


def write(root, name, data):
    p = root/name
    p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    q = p.parent
    while q != root:
        q.chmod(0o700)
        q = q.parent
    p.write_bytes(data)
    p.chmod(0o600)
    return p


@pytest.fixture(scope='module')
def control():
    sources = {name: (ROOT/'models'/('predictor_pgd_split_bootstrap_v1.py' if name == 'bootstrap.py' else name)).read_bytes()
               for name in b.SUPERVISORS.values()}
    raw = coordinator.make_control((ROOT/'configs/predictor_pgd_split_launch_v1.json').read_bytes(), sources)
    c, p, sha = b.control(raw)
    auth = b.encoded({'kind':'pgd-split-supervised-university-exception-v1', 'approved':True, 'control_sha256':sha,
        'maximum_new_attempts':1, 'user_basis':'EXPLICIT INERT LOCAL FIXTURE ONLY; NOT A USER GRANT OR ARMED HANDOFF'})
    ib, isha = b.installation(c, sha)
    layout = s.Layout(p['scratch_contract_text'].encode(), p['base_text'].encode(), p['design_text'].encode(), ib, isha)
    return NS(sources=sources, raw=raw, c=c, p=p, sha=sha, auth=auth, ib=ib, isha=isha, layout=layout)


def session(control, *, failed=False, prefix=4):
    roles = list(b.ROLES[:prefix])
    return {'kind':'pgd-split-parent-session-v1','control_sha256':control.sha,'parent_pid':os.getpid(),
        'connection_started':0.,'work_deadline':2400.,'connection_deadline':3000.,
        'phase':'failed' if failed else 'await-collection','complete':False,'collection_pending':True,
        'current':None,'completed':roles,'receipts':[
            {'role':role,'pid':900+i,'start_ticks':400+i,'saved_gates_returned':True,
             'receipt':{'origin':'EXPLICIT FAKE ROLE RECEIPT; NO WORK'}, 'source_hashes':{},
             'scientific_qualified':False,'training_admitted':False} for i, role in enumerate(roles)],
        'scientific_qualified':False,'training_admitted':False}


def metadata_header(control):
    layout = control.layout
    st = session(control)
    win = m.Window(0.,100.,clock=lambda:100.)
    receipt = m.work_receipt(b.encoded(st),control.sha,b.digest(control.auth),control.isha,win)
    extras = {'control.json':control.raw,'authorization.json':control.auth,'session.json':b.encoded(st),
              m.RECEIPT:b.encoded(receipt),m.STATUS:b.encoded({'EXPLICIT':'SYNTHETIC FULL INVENTORY METADATA ONLY'})}
    pins = dict(layout.assets)
    pins.update({n:{'bytes':len(v),'sha256':b.digest(v)} for n,v in extras.items()})
    names = sorted(pins)
    uid = os.getuid()
    files = {n:dict(pins[n],mode=0o600,mtime_ns=1000000001+i) for i,n in enumerate(names)}
    inventory = {n:{'identity':[500,100+i,uid,0o600,1],'bytes':v['bytes'],
                   'allocated_bytes':((v['bytes']+4095)//4096)*4096,'mtime_ns':v['mtime_ns']} for i,(n,v) in enumerate(files.items())}
    inventory['emergency-space.bin'] = {'identity':[500,900,uid,0o600,1], 'bytes':16*s.MiB,'allocated_bytes':16*s.MiB,'mtime_ns':100}
    dirs = {'.'}
    for name in inventory:
        dirs.update(str(q) for q in Path(name).parents)
    directories = {n:{'identity':[500,2000+i,uid,0o700],'allocated_bytes':4096} for i,n in enumerate(sorted(dirs))}
    storage = layout.assess(inventory,directories,{'available_bytes':8*s.GiB,'available_inodes':2048},phase='collection')
    return {'kind':'pgd-split-evidence-stream-v1','control_sha256':control.sha,'authorization_sha256':b.digest(control.auth),
        'root_binding':{'root':'/EXPLICIT-SYNTHETIC/scratch','root_identity':list(directories['.']['identity']),
                        'parent':'/EXPLICIT-SYNTHETIC','parent_identity':[500,1,uid,0o700]},
        'files':files,'inventory':inventory,'directories':directories,'storage':storage,
        'final_receipt_sha256':b.digest(extras[m.RECEIPT]),'session_sha256':b.digest(extras['session.json']),
        'collection_started':100.,'collection_deadline':700.,'connection_started':0.,'connection_deadline':3000.,
        'logical_bytes':sum(r['bytes'] for r in files.values()),'omitted_content':['emergency-space.bin'],
        'scientific_qualified':False,'training_admitted':False}


def test_actual_six_source_control_and_172_asset_metadata_keep_original_scope(control, tmp_path):
    h = metadata_header(control)
    assert len(control.sources)==6 and len(control.layout.assets)==172
    assert len(h['files'])==177 and len(b.encoded(h))<m.HEADER_CAP
    assert m.validate_header(h,control.raw,control.auth).installation_sha256==control.isha
    (tmp_path/'SYNTHETIC-FULL-HEADER-NO-PAYLOAD.json').write_bytes(b.encoded(h))
    assert control.c['supervisor_sources']['predictor_pgd_split_collection_v1.py']['sha256']==b.digest((ROOT/'models/predictor_pgd_split_collection_v1.py').read_bytes())
    assert control.p['scope']['resource']['collection_seconds']==600 and m.WIRE_CAP==3*s.GiB
    assert not control.p['authorized'] and not control.p['target_helper_ready']
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


@pytest.mark.parametrize('fault',['old-control','old-auth','admission','training','extra-key','clock-reset','clock-nan','whole-clock',
    'unknown-name','missing-source','changed-asset-hash','changed-source-extent','mode','mtime-bool','hash','manifest-missing','omitted',
    'logical-bool','logical-size','receipt-hash','session-hash','root','capacity','inventory-extra','raw-cap','special-link'])
def test_real_contract_header_refuses_changed_identity_geometry_sources_or_caps(control,fault):
    h = metadata_header(control)
    first = next(iter(h['files']))
    if fault=='old-control':h['control_sha256']='f'*64
    elif fault=='old-auth':h['authorization_sha256']='f'*64
    elif fault=='admission':h['scientific_qualified']=True
    elif fault=='training':h['training_admitted']=True
    elif fault=='extra-key':h['unreviewed-port']='NO'
    elif fault=='clock-reset':h['collection_deadline']+=1
    elif fault=='clock-nan':h['collection_started']=float('nan')
    elif fault=='whole-clock':h['connection_deadline']+=1
    elif fault=='unknown-name':h['inventory']['unknown.bin']=copy.deepcopy(h['inventory'][first])
    elif fault=='missing-source':
        h['files'].pop('original-packet.tar.gz');h['inventory'].pop('original-packet.tar.gz')
    elif fault=='changed-asset-hash':h['files']['original-packet.tar.gz']['sha256']='e'*64
    elif fault=='changed-source-extent':h['files']['original-packet.tar.gz']['bytes']-=1
    elif fault=='mode':h['files'][first]['mode']=0o644
    elif fault=='mtime-bool':h['files'][first]['mtime_ns']=True
    elif fault=='hash':h['files'][first]['sha256']='BAD'
    elif fault=='manifest-missing':h['files'].pop(first)
    elif fault=='omitted':h['omitted_content']=[]
    elif fault=='logical-bool':h['logical_bytes']=True
    elif fault=='logical-size':h['logical_bytes']+=1
    elif fault=='receipt-hash':h['final_receipt_sha256']='e'*64
    elif fault=='session-hash':h['session_sha256']='e'*64
    elif fault=='root':h['root_binding']['root_identity'][0]+=1
    elif fault=='capacity':h['storage']['capacity']['available_bytes']=s.GiB-1
    elif fault=='inventory-extra':h['inventory'][first]['untyped']=0
    elif fault=='raw-cap':
        row=copy.deepcopy(h['inventory'][first]);row['bytes']=1792*s.MiB+1
        h['inventory']['capture/results/numerical/clean.npy']=row
    else:h['inventory'][first]['identity'][4]=3
    with pytest.raises((ValueError,KeyError,TypeError)):
        m.validate_header(h,control.raw,control.auth)


@pytest.fixture
def tiny(tmp_path,control,monkeypatch):
    root = tmp_path/'EXPLICIT-TINY-SUBSTITUTED-SCRATCH'
    root.mkdir(mode=0o700)
    layout = copy.deepcopy(control.layout)
    selected = ['original-packet.tar.gz','code/models/__init__.py','overlay/models/__init__.py',
                'configs/base.json','configs/design.json',*('supervisor/'+n for n in b.SUPERVISORS.values())]
    data = {n:b'"""EXPLICIT INERT TRANSPORT TEST BYTES; NEVER EXECUTED."""\n' for n in selected}
    data['installation.json']=control.ib
    layout.assets={n:{'bytes':len(v),'sha256':b.digest(v)} for n,v in data.items()}
    data.update({'control.json':control.raw,'authorization.json':control.auth,'session.json':b.encoded(session(control)),
        'emergency-space.bin':b'EXPLICIT-TINY-RESERVE-STUB; NOT AN ALLOCATION PROOF',
        'cache/ignored.bin':b'NEVER-COLLECT-ME','tmp/ignored.bin':b'NEVER-COLLECT-ME',
        'capture/results/numerical/clean.npy':b'INERT-FAKE-NPY-NOT-AN-ARRAY\x00\xff',
        'capture/logs/numerical.log':b'EXPLICIT FAKE WORK; NO MODEL OR CHILD WAS RUN\n'})
    for n,v in data.items():write(root,n,v)
    (root/'capture/results').mkdir(mode=0o700,parents=True,exist_ok=True)
    guard=s.Guard(root,s.identity_binding(root),layout)
    # Only this explicit tiny fixture uses substituted descriptors. Actual-header
    # tests above use the unmodified 172-asset contract and metadata sizes only.
    monkeypatch.setattr(s,'Layout',lambda *a,**kw:layout)
    now=[100.]
    collector=m.Collector(guard,control.raw,control.auth,connection_started=0.,clock=lambda:now[0])
    return NS(root=root,layout=layout,guard=guard,clock=now,collector=collector,data=data,control=control)


def encode_stream(tiny):
    h=tiny.collector.prepare()
    stream=io.BytesIO()
    remote=tiny.collector.stream(stream)
    return h,stream.getvalue(),remote


def test_tiny_byte_round_trip_private_modes_nsmtime_exclusions_and_distinct_admission(tiny,tmp_path):
    h,data,remote=encode_stream(tiny)
    output=tmp_path/'PRIVATE-RECEIVER';output.mkdir(mode=0o700)
    receipt=m.receive(io.BytesIO(data),output/'saved',tiny.control.raw,tiny.control.auth,
                      window=m.Window(0.,100.,clock=lambda:100.))
    assert receipt['transport_verified'] and not receipt['scientific_qualified'] and not receipt['training_admitted']
    assert not remote['receiver_verified'] and not b.parsed((tiny.root/m.RECEIPT).read_bytes())['collection_complete']
    assert receipt['wire_sha256']==remote['wire_sha256']==b.digest(data)
    assert receipt['wire_bytes']==len(data) and set(receipt['files'])==set(h['files'])
    for name,row in h['files'].items():
        before=tiny.root/name;after=output/'saved'/name
        assert before.read_bytes()==after.read_bytes()
        assert stat.S_IMODE(after.stat().st_mode)==0o600 and after.stat().st_mtime_ns==before.stat().st_mtime_ns
    assert b'NEVER-COLLECT-ME' not in data
    assert not (output/'saved/cache').exists() and not (output/'saved/emergency-space.bin').exists()
    write(output,'INERT-LOCAL-ROUND-TRIP-RECEIPT.json',b.encoded(dict(receipt,origin='EXPLICIT TINY SUBSTITUTIONS; NOT TARGET EVIDENCE')))


@pytest.mark.parametrize('fault',['empty','magic','header-length','truncated-header','body-byte','missing-terminal','truncated-terminal','terminal-hash','trailing','noncanonical','duplicate-json'])
def test_stream_damage_never_yields_a_complete_receipt_and_preserves_local_partials(tiny,tmp_path,fault):
    h,data,_=encode_stream(tiny)
    prefix=len(m.MAGIC)+4
    header_size=struct.unpack('!I',data[len(m.MAGIC):prefix])[0]
    start=prefix+header_size
    terminal=data.index(m.END,start)
    if fault=='empty':data=b''
    elif fault=='magic':data=b'!'+data[1:]
    elif fault=='header-length':data=m.MAGIC+struct.pack('!I',m.HEADER_CAP+1)
    elif fault=='truncated-header':data=data[:prefix+2]
    elif fault=='body-byte':data=data[:start]+bytes([data[start]^1])+data[start+1:]
    elif fault=='missing-terminal':data=data[:terminal]
    elif fault=='truncated-terminal':data=data[:-10]
    elif fault=='terminal-hash':
        n=terminal+len(m.END)+4;obj=b.parsed(data[n:]);obj['prefix_sha256']='0'*64
        encoded=b.encoded(obj);data=data[:terminal]+m.END+struct.pack('!I',len(encoded))+encoded
    elif fault=='trailing':data+=b'UNEXPECTED'
    else:
        raw=data[prefix:start]
        if fault=='noncanonical':raw+=b'\n'
        else:raw=raw.rstrip()[:-1]+b',"scientific_qualified":false}\n'
        data=m.MAGIC+struct.pack('!I',len(raw))+raw+data[start:]
    parent=tmp_path/'receiver';parent.mkdir(mode=0o700)
    with pytest.raises((ValueError,OSError)):
        m.receive(io.BytesIO(data),parent/'saved',tiny.control.raw,tiny.control.auth,window=m.Window(0.,100.,clock=lambda:100.))
    if fault in ('body-byte','missing-terminal','truncated-terminal','terminal-hash','trailing'):
        assert (parent/'saved').is_dir()
        assert any(p.is_file() for p in (parent/'saved').rglob('*'))


@pytest.mark.parametrize('fault',['hash-after-snapshot','new-path','source-mode','source-symlink','source-hardlink','source-growth','deadline','rollback','broken-write','zero-write','None-write','flush-error'])
def test_producer_refusal_omits_terminal_and_preserves_failed_evidence(tiny,fault):
    tiny.collector.prepare()
    p=tiny.root/'capture/results/numerical/clean.npy'
    if fault=='hash-after-snapshot':
        info=p.stat();value=p.read_bytes();p.write_bytes(b'!'+value[1:]);os.utime(p,ns=(info.st_atime_ns,info.st_mtime_ns))
    elif fault=='new-path':write(tiny.root,'UNDECLARED.bin',b'PRESERVE')
    elif fault=='source-mode':p.chmod(0o644)
    elif fault=='source-symlink':p.rename(p.with_name('PRESERVED-RAW'));p.symlink_to('PRESERVED-RAW')
    elif fault=='source-hardlink':os.link(p,p.with_name('PRESERVED-LINK'))
    elif fault=='source-growth':p.write_bytes(p.read_bytes()+b'GROWTH')
    elif fault=='deadline':tiny.clock[0]=700.
    elif fault=='rollback':tiny.clock[0]=99.
    class Output(io.BytesIO):
        def write(self,data):
            if fault=='broken-write':raise BrokenPipeError('EXPLICIT FAKE')
            if fault=='zero-write':return 0
            if fault=='None-write':return None
            return super().write(data)
        def flush(self):
            if fault=='flush-error':raise OSError('EXPLICIT FAKE FLUSH')
            return super().flush()
    stream=Output()
    with pytest.raises((ValueError,TimeoutError,OSError)):
        tiny.collector.stream(stream)
    # A flush error can leave all bytes written but the caller must still treat
    # transport as failed. All earlier failures omit a valid terminal receipt.
    if fault!='flush-error':assert m.END not in stream.getvalue()
    assert (tiny.root/m.RECEIPT).is_file()
    with pytest.raises(ValueError):tiny.collector.stream(io.BytesIO())


def test_partial_writes_and_reads_are_completed_without_reset_or_extra_frames(tiny,tmp_path):
    tiny.collector.prepare()
    class ShortOut(io.BytesIO):
        def write(self,value):return super().write(value[:7])
    out=ShortOut();tiny.collector.stream(out)
    class ShortIn(io.BytesIO):
        def read(self,count):return super().read(min(count,11))
    parent=tmp_path/'receiver';parent.mkdir(mode=0o700)
    got=m.receive(ShortIn(out.getvalue()),parent/'saved',tiny.control.raw,tiny.control.auth,window=m.Window(0.,100.,clock=lambda:100.))
    assert got['transport_verified'] and got['wire_bytes']==len(out.getvalue())


@pytest.mark.parametrize('fault',['existing-destination','public-parent','symlink-parent','expired','clock-rollback','oversized-read'])
def test_local_destination_and_original_client_clock_refusals_preserve_outputs(tiny,tmp_path,fault):
    _,data,_=encode_stream(tiny)
    parent=tmp_path/'receiver';parent.mkdir(mode=0o700)
    dest=parent/'saved'
    if fault=='existing-destination':dest.mkdir(mode=0o700);write(dest,'PRESERVED.txt',b'OLD')
    elif fault=='public-parent':parent.chmod(0o755)
    elif fault=='symlink-parent':
        alias=tmp_path/'alias';alias.symlink_to(parent,target_is_directory=True);dest=alias/'saved'
    now=[100.];window=m.Window(0.,100.,clock=lambda:now[0])
    if fault=='expired':now[0]=700.
    elif fault=='clock-rollback':now[0]=99.
    source=io.BytesIO(data)
    if fault=='oversized-read':source=NS(read=lambda size:b'X'*(size+1))
    with pytest.raises((ValueError,TimeoutError,OSError)):
        m.receive(source,dest,tiny.control.raw,tiny.control.auth,window=window)
    if fault=='existing-destination':assert (dest/'PRESERVED.txt').read_bytes()==b'OLD'


@pytest.mark.parametrize('fault',['phase','wrong-parent','old-control','old-start','extended-work','admitted','partial-prefix','receipt-false','duplicate-pid','missing-role','complete-flag','pending','current'])
def test_combined_work_receipts_cannot_promote_failed_or_unfinished_roles(control,fault):
    st=session(control)
    if fault=='phase':st['phase']='running'
    elif fault=='wrong-parent':st['parent_pid']+=1
    elif fault=='old-control':st['control_sha256']='f'*64
    elif fault=='old-start':st['connection_started']=1.
    elif fault=='extended-work':st['work_deadline']+=1
    elif fault=='admitted':st['scientific_qualified']=True
    elif fault=='partial-prefix':st['completed']=['proof','native']
    elif fault=='receipt-false':st['receipts'][0]['saved_gates_returned']=False
    elif fault=='duplicate-pid':st['receipts'][1]['pid']=st['receipts'][0]['pid']
    elif fault=='missing-role':st['completed'].pop();st['receipts'].pop()
    elif fault=='complete-flag':st['complete']=True
    elif fault=='pending':st['collection_pending']=False
    else:st['current']={'role':'forward-preserving'}
    with pytest.raises(ValueError):
        m.work_receipt(b.encoded(st),control.sha,b.digest(control.auth),control.isha,m.Window(0.,100.,clock=lambda:100.))


def test_failed_prefix_remains_failed_and_no_second_preparation_may_replace_receipts(tiny):
    st=session(tiny.control,failed=True,prefix=2)
    (tiny.root/'session.json').write_bytes(b.encoded(st))
    h=tiny.collector.prepare()
    r=b.parsed((tiny.root/m.RECEIPT).read_bytes())
    assert not r['all_saved_role_checks_returned'] and r['work_phase']=='failed'
    assert not r['collection_complete'] and not r['scientific_qualified']
    before=(tiny.root/m.RECEIPT).read_bytes()
    with pytest.raises(ValueError):tiny.collector.prepare()
    assert (tiny.root/m.RECEIPT).read_bytes()==before and h['session_sha256']==b.digest(b.encoded(st))


def test_live_final_partial_publication_pair_is_copied_as_two_preserved_names_not_completion(tiny,tmp_path):
    p=write(tiny.root,'capture/results/numerical/adversarial.npy.partial',b'INERT-INCOMPLETE')
    # A mutable metadata final/partial pair is legal, while an immutable raw
    # completed name may not alias a partial. Use a distinct declared metadata.
    final=write(tiny.root,'capture/results/run-status.json',b'{"EXPLICIT":"INCOMPLETE"}\n')
    os.link(final,final.with_name(final.name+'.partial'))
    (tiny.root/'session.json').write_bytes(b.encoded(session(tiny.control,failed=True,prefix=2)))
    h,data,_=encode_stream(tiny)
    parent=tmp_path/'receiver';parent.mkdir(mode=0o700)
    r=m.receive(io.BytesIO(data),parent/'saved',tiny.control.raw,tiny.control.auth,window=m.Window(0.,100.,clock=lambda:100.))
    assert r['transport_verified'] and not r['scientific_qualified']
    assert 'capture/results/numerical/adversarial.npy.partial' in r['files']
    assert len(h['storage']['publication_pairs'])==1
    assert (parent/'saved/capture/results/run-status.json.partial').read_bytes()==final.read_bytes()
    assert (parent/'saved/capture/results/run-status.json').stat().st_nlink==1


def test_definition_only_source_has_no_transport_process_scientific_or_timer_entry():
    source=(ROOT/'models/predictor_pgd_split_collection_v1.py').read_bytes()
    tree=ast.parse(source);compile(tree,'<UNEXECUTED-COLLECTOR>','exec')
    imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    names={alias.name.split('.')[0] for n in imports if isinstance(n,ast.Import) for alias in n.names}
    assert names <= {'hashlib','os','pathlib','stat','struct','time'}
    assert not any(isinstance(n,ast.If) and '__name__' in ast.unparse(n.test) for n in tree.body)
    calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    for forbidden in ('subprocess.Popen','os.kill','os.pipe','signal.setitimer','torch.load','numpy.load','socket.socket'):
        assert forbidden not in calls
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


@pytest.mark.parametrize('fault',['file-bytes','file-mode','extra-name','root-replaced','parent-replaced'])
def test_final_local_seal_refuses_changes_while_receiving_terminal_EOF(tiny,tmp_path,fault):
    _,data,_=encode_stream(tiny)
    parent=tmp_path/'late-receiver';parent.mkdir(mode=0o700)
    dest=parent/'saved'
    class LateSource(io.BytesIO):
        changed=False
        def read(self,count):
            data=super().read(count)
            if count==1 and data==b'' and not self.changed:
                self.changed=True
                p=dest/'capture/results/numerical/clean.npy'
                if fault=='file-bytes':
                    st=p.stat();old=p.read_bytes();p.write_bytes(b'!'+old[1:]);os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns))
                elif fault=='file-mode':p.chmod(0o644)
                elif fault=='extra-name':write(dest,'EXTRA-UNVERIFIED.txt',b'PRESERVE')
                elif fault=='root-replaced':
                    dest.rename(parent/'PRESERVED-saved');dest.mkdir(mode=0o700)
                else:
                    parent.rename(tmp_path/'PRESERVED-parent');parent.mkdir(mode=0o700);dest.mkdir(mode=0o700)
            return data
    with pytest.raises(ValueError):
        m.receive(LateSource(data),dest,tiny.control.raw,tiny.control.auth,window=m.Window(0.,100.,clock=lambda:100.))
    assert dest.is_dir()


def test_concrete_source_only_bundle_pins_all_69_modules_without_executing_them(tmp_path,control):
    from models import predictor_pgd_split_source_bundle_v1 as assembler
    parent=tmp_path/'private-source-parent';parent.mkdir(mode=0o700)
    dest=parent/'UNEXECUTED-sources'
    manifest=assembler.assemble(ROOT,dest)
    assert manifest['source_files']==69 and len(manifest['files'])==73
    assert not manifest['authorized'] and not manifest['target_helper_ready']
    assert not manifest['outer_attempt_setup_timer_and_guard_journals_integrated']
    raw=(dest/'CONTROL-CANDIDATE.json').read_bytes();c,p,sha=b.control(raw)
    finder=b.PinnedImporter(dest,s.identity_binding(dest),c,p) # hash/cache only, NOT installed/executed
    assert len(finder.files)==69 and not finder.loaded and not finder.installed
    assert sha==control.sha and (dest/'INSTALLATION-CANDIDATE.json').read_bytes()==control.ib
    for name,row in manifest['files'].items():
        data=(dest/name).read_bytes()
        assert len(data)==row['bytes'] and b.digest(data)==row['sha256']
        if name.endswith('.py'):compile(data,'<UNEXECUTED-SOURCE-ONLY>','exec')
    assert not list(dest.rglob('AUTHORIZATION*')) and not list(dest.rglob('ATTEMPT*'))
    write(parent,'source-bundle-manifest.json',b.encoded(manifest))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


@pytest.mark.parametrize('fault',['old-destination','public-parent','symlink-parent'])
def test_offline_bundle_cannot_overwrite_old_source_or_escape_private_destination(tmp_path,fault):
    from models import predictor_pgd_split_source_bundle_v1 as assembler
    parent=tmp_path/'private-source-parent';parent.mkdir(mode=0o700)
    dest=parent/'PRESERVED'
    if fault=='old-destination':dest.mkdir(mode=0o700);write(dest,'OLD.txt',b'OLD')
    elif fault=='public-parent':parent.chmod(0o755)
    else:
        alias=tmp_path/'alias';alias.symlink_to(parent,target_is_directory=True);dest=alias/'PRESERVED'
    with pytest.raises(ValueError):assembler.assemble(ROOT,dest)
    if fault=='old-destination':assert (dest/'OLD.txt').read_bytes()==b'OLD'


@pytest.mark.parametrize('fault',['symlink','hardlink','mode','size','hash'])
def test_bundle_source_reader_refuses_nonregular_or_changed_untrusted_source_before_output(tmp_path,fault):
    from models import predictor_pgd_split_source_bundle_v1 as assembler
    p=write(tmp_path,'source.py',b'"""INERT"""\n')
    want=b.digest(p.read_bytes());cap=p.stat().st_size
    if fault=='symlink':
        p.rename(tmp_path/'PRESERVED.py');p.symlink_to('PRESERVED.py')
    elif fault=='hardlink':os.link(p,tmp_path/'PRESERVED.py')
    elif fault=='mode':p.chmod(0o666)
    elif fault=='size':cap-=1
    else:want='0'*64
    with pytest.raises(ValueError):assembler.read_source(p,cap=cap,sha256=want)


@pytest.mark.parametrize('origin,start,now,expected',[(0.,100.,100.,700.),(0.,2900.,2900.,3000.)])
def test_collection_window_clips_to_original_login_inclusive_origin(origin,start,now,expected):
    clock=[now];w=m.Window(origin,start,clock=lambda:clock[0]);assert w.deadline==expected
    clock[0]=expected
    with pytest.raises(TimeoutError):w.check()


@pytest.mark.parametrize('fault',['bool-origin','nan-origin','later-origin','late-start','expired-now'])
def test_collection_window_rejects_invalid_or_reset_clock_metadata(fault):
    origin,start,now=0.,100.,100.
    if fault=='bool-origin':origin=False
    elif fault=='nan-origin':origin=float('nan')
    elif fault=='later-origin':origin=101.
    elif fault=='late-start':start=101.
    else:now=700.
    with pytest.raises((ValueError,TimeoutError)):m.Window(origin,start,clock=lambda:now)


def test_terminal_EOF_and_local_verification_remain_inside_original_collection_deadline(tiny,tmp_path):
    _,data,_=encode_stream(tiny)
    parent=tmp_path/'deadline-receiver';parent.mkdir(mode=0o700)
    now=[100.]
    class Source(io.BytesIO):
        def read(self,count):
            data=super().read(count)
            if count==1 and data==b'':now[0]=700.
            return data
    with pytest.raises(TimeoutError):
        m.receive(Source(data),parent/'saved',tiny.control.raw,tiny.control.auth,window=m.Window(0.,100.,clock=lambda:now[0]))
    assert (parent/'saved/capture/results/numerical/clean.npy').read_bytes()==tiny.data['capture/results/numerical/clean.npy']


def test_local_write_failure_retains_partial_file_and_never_returns_verified(tiny,tmp_path,monkeypatch):
    _,data,_=encode_stream(tiny)
    parent=tmp_path/'disk-refusal';parent.mkdir(mode=0o700)
    real=m.os.write;count=[0]
    def fail(fd,block):
        count[0]+=1
        if count[0]>1:raise OSError(28,'EXPLICIT FAKE local disk full')
        return real(fd,block[:1])
    monkeypatch.setattr(m.os,'write',fail)
    with pytest.raises(OSError,match='EXPLICIT FAKE'):
        m.receive(io.BytesIO(data),parent/'saved',tiny.control.raw,tiny.control.auth,window=m.Window(0.,100.,clock=lambda:100.))
    saved=[p for p in (parent/'saved').rglob('*') if p.is_file()]
    assert len(saved)==1 and saved[0].stat().st_size==1
