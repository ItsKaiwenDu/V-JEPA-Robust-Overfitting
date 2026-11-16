"""New whole-scratch source/metadata/tiny ordinary filesystem checks only.

All GiB values and allocation-proof records are synthetic observations. No proof,
preallocation, real arrays/science, child/PID/signal, GPU or remote work is run.
"""
import ast
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path,PurePosixPath
import stat
import sys
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_split_scratch_v1 as m

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=(ROOT/'configs/predictor_pgd_split_scratch_v1.json').read_bytes()
BASE=(ROOT/'configs/predictor_pgd_target_v1.json').read_bytes();DESIGN=(ROOT/'configs/predictor_pgd_split_profile_v1.json').read_bytes()
CTRL='7'*64


def install():
    items={}
    for n in m.SUPERVISOR_NAMES:
        p=ROOT/'models'/n
        data=p.read_bytes() if p.exists() else ('OPAQUE FUTURE SOURCE LABEL: '+n).encode()
        items[n]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    return m.encoded({'kind':'pgd-split-scratch-installation-v1','control_sha256':CTRL,'files':items})

@pytest.fixture(scope='module')
def layout():
    b=install();return m.Layout(CONTRACT,BASE,DESIGN,b,hashlib.sha256(b).hexdigest())


class Observation:
    """Exact fake stat records; no sparse files or large allocations are created."""
    def __init__(self):
        self.files={};self.dirs={};self.ids=itertools.count(10);self.capacity={'available_bytes':16*m.GiB,'available_inodes':4096};self.directory('.')
    def directory(self,name):
        if name not in self.dirs:self.dirs[name]={'identity':(55,next(self.ids),os.getuid(),0o700),'allocated_bytes':4096}
        if name!='.':self.directory(str(PurePosixPath(name).parent))
    def file(self,name,size=1,allocated=None):
        self.directory(str(PurePosixPath(name).parent))
        self.files[name]={'identity':(55,next(self.ids),os.getuid(),0o600,1),'bytes':size,'allocated_bytes':size if allocated is None else allocated,'mtime_ns':1234}
        return self.files[name]
    def pair(self,name,size=1):
        row=self.file(name,size);row['identity']=(*row['identity'][:4],2);self.files[name+'.partial']=copy.deepcopy(row)
        return row
    def assess(self,layout,**kw):return layout.assess(self.files,self.dirs,self.capacity,**kw)


def test_contract_covers_exact_packet_runtime_namespaces_and_original_budgets(layout):
    c=json.loads(CONTRACT);assert hashlib.sha256(CONTRACT).hexdigest()==m.CONTRACT_SHA
    assert len(layout.result)==419 and len(layout.assets)==172
    assert len(c['packet']['files'])==99 and sum(d['bytes'] for d in c['packet']['files'].values())==26446958
    assert len(c['runtime_files'])==63 and sum(d['bytes'] for d in c['runtime_files'].values())==732104
    assert c['packet']['archive']=={'bytes':17192095,'sha256':'2278bce2b519943c0f83a49c556bb73dd928fe880fba0f6c56de2a4c3c2ec0da'}
    metadata=json.loads(DESIGN)['resource']['metadata_plus_mutable_partials_bytes']
    assert metadata+sum(d[0] for d in m.ROOT_META.values())<64*m.MiB
    assert set(m.CAPS)=={'raw','input','code','cache','emergency','metadata','trace','proof'}
    assert not c['authorized'] and not c['target_runner_available']
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules

@pytest.mark.parametrize('fault',['contract','base','design','installation','source-set','source-size','support-hash'])
def test_changed_contract_or_unbound_source_descriptor_refuses(fault):
    contract,base,design,b=CONTRACT,BASE,DESIGN,install();i=json.loads(b)
    if fault=='contract':contract+=b'\n'
    elif fault=='base':base+=b'\n'
    elif fault=='design':design+=b'\n'
    elif fault=='installation':
        with pytest.raises(ValueError,match='Exact source/storage/installation'):m.Layout(contract,base,design,b,'f'*64)
        return
    elif fault=='source-set':i['files'].pop('bootstrap.py')
    elif fault=='source-size':i['files']['bootstrap.py']['bytes']=3*m.MiB
    else:i['files']['scratch_allocation_v2.py']['sha256']='e'*64
    if fault in ('source-set','source-size','support-hash'):b=m.encoded(i)
    with pytest.raises(ValueError):m.Layout(contract,base,design,b,hashlib.sha256(b).hexdigest())

@pytest.mark.parametrize('name',['/absolute','../escape','capture//results/plan.json','capture/./results/plan.json','capture/results/../x',
 'capture/results/unknown.npy','code/not-in-packet.py','overlay/models/not-in-package.py','cache/'+'a'*129,'cache/space name','tmp/a\\b','UNKNOWN','space-proof-v2.bin','emergency-space.bin.partial',
 'allocation-result-v2.json.partial','capture/results/numerical/objective-0/post-gradient-progress-v2.jsonl.partial'])
def test_unknown_aliased_or_disallowed_partial_paths_are_never_admitted(layout,name):
    with pytest.raises(ValueError):layout.classify(name)


def test_worst_completed_result_table_and_root_metadata_fit_shared_envelope(layout):
    o=Observation()
    for name,d in layout.result.items():o.file('capture/'+name,d['max_bytes'])
    for name,(size,_,_) in m.ROOT_META.items():o.file(name,size)
    row=o.assess(layout,phase='capture')
    assert row['categories']['metadata']+row['categories']['trace']<64*m.MiB
    assert row['logical_bytes']<9*m.GiB//4 and row['allocated_bytes']<9*m.GiB//4
    assert row['categories']['trace']==8*m.MiB and row['files']==419+len(m.ROOT_META)
    assert not row['scientific_qualified'] and row['sampling_is_not_atomic_or_kernel_quota']


def test_full_proof_extent_uses_setup_ceiling_and_forbids_capture(layout):
    o=Observation();o.file('space-proof-v2.bin.partial',5*m.GiB//2);o.file('emergency-space.bin',16*m.MiB)
    row=o.assess(layout,phase='proof');assert row['ceiling_bytes']==3*m.GiB and row['categories']['proof']==5*m.GiB//2
    with pytest.raises(ValueError,match='outside its one proof phase'):o.assess(layout,phase='capture')
    o.file('capture/results/numerical/objective-0/input.npy')
    with pytest.raises(ValueError,match='No capture arrays/trace'):o.assess(layout,phase='proof')

@pytest.mark.parametrize('fault',['file-allocated','directory-allocated','initial-space','initial-inodes','work-space','work-inodes','capacity-bool',
 'cache-total','cache-count','file-count','directory-count','raw-partial','meta-partial','mixed-meta-partial','input-partial','private-mode','foreign-device','foreign-uid','special-links','missing-parent','duplicate-dir','typed-size'])
def test_independent_capacity_counts_and_typed_ceilings_refuse(layout,fault):
    o=Observation();phase='capture';initial=False
    if fault=='file-allocated':o.file('capture/results/plan.json',1,3*m.GiB)
    elif fault=='directory-allocated':o.dirs['.']['allocated_bytes']=3*m.GiB
    elif fault.startswith('initial'):
        phase='setup';initial=True;o.capacity['available_bytes' if fault.endswith('space') else 'available_inodes']=(8*m.GiB-1 if fault.endswith('space') else 2047)
    elif fault.startswith('work'):
        o.capacity['available_bytes' if fault.endswith('space') else 'available_inodes']=(m.GiB-1 if fault.endswith('space') else 1023)
    elif fault=='capacity-bool':o.capacity['available_bytes']=True
    elif fault=='cache-total':
        o.file('cache/cuda/a',32*m.MiB);o.file('tmp/b',1)
    elif fault=='cache-count':
        for i in range(49):o.file('cache/cuda/f'+str(i))
    elif fault=='file-count':o.files={str(i):{} for i in range(1025)}
    elif fault=='directory-count':o.dirs={str(i):{} for i in range(257)}
    elif fault=='raw-partial':
        for i in range(2):o.file(f'capture/results/numerical/objective-{i}/input.npy.partial')
    elif fault=='meta-partial':
        for n in ('control.json.partial','proposal.json.partial','storage-before.json.partial'):o.file(n)
    elif fault=='mixed-meta-partial':
        for n in ('capture/results/run-status.json','capture/results/numerical/objective-0/numerical.json'):
            o.file(n,256*1024);o.file(n+'.partial',256*1024)
        o.file('session.json',1);o.file('session.json.partial',1)
    elif fault=='input-partial':
        o.file('original-packet.tar.gz');o.file('original-packet.tar.gz.partial')
    elif fault=='private-mode':o.file('control.json')['identity']=(55,100,os.getuid(),0o644,1)
    elif fault=='foreign-device':o.file('control.json')['identity']=(66,100,os.getuid(),0o600,1)
    elif fault=='foreign-uid':o.file('control.json')['identity']=(55,100,os.getuid()+1,0o600,1)
    elif fault=='special-links':o.file('control.json')['identity']=(55,100,os.getuid(),0o600,3)
    elif fault=='missing-parent':o.file('capture/results/plan.json');o.dirs.pop('capture/results')
    elif fault=='duplicate-dir':o.directory('cache');o.dirs['cache']['identity']=o.dirs['.']['identity']
    else:o.file('control.json',256*1024+1)
    with pytest.raises(ValueError):o.assess(layout,phase=phase,initial=initial)

@pytest.mark.parametrize('category',['raw','metadata','trace','root-metadata'])
def test_exact_temporary_publication_inode_pair_counts_once_and_stays_pending(layout,category):
    name={'raw':'capture/results/numerical/objective-0/input.npy','metadata':'capture/results/numerical/objective-0/native-calls.jsonl',
        'trace':'capture/results/components/native/events.json','root-metadata':'control.json'}[category]
    o=Observation();o.pair(name,1024);row=o.assess(layout,phase='capture')
    assert row['logical_bytes']==1024 and row['named_bytes']==2048 and row['publication_pairs']==[[name,name+'.partial']]
    assert row['allocated_bytes']==1024+sum(d['allocated_bytes'] for d in o.dirs.values())
    assert row['partials']['raw' if category=='raw' else 'metadata']==1
    with pytest.raises(ValueError):o.assess(layout,phase='collection',settled=True)

@pytest.mark.parametrize('fault',['outside-pair','mismatch','foreign-pair','input-pair','third-link','single-duplicate','two-raw-pairs'])
def test_unrelated_or_ambiguous_hardlinks_cannot_use_publication_exception(layout,fault):
    o=Observation();name='capture/results/numerical/objective-0/input.npy';o.pair(name)
    if fault=='outside-pair':o.files.pop(name+'.partial')
    elif fault=='mismatch':o.files[name+'.partial']['bytes']+=1
    elif fault=='foreign-pair':o.files['capture/results/numerical/objective-0/pixels.npy']=o.files.pop(name+'.partial')
    elif fault=='input-pair':
        o=Observation();o.pair('original-packet.tar.gz')
    elif fault=='third-link':o.files[name]['identity']=(*o.files[name]['identity'][:4],3)
    elif fault=='single-duplicate':
        for row in o.files.values():row['identity']=(*row['identity'][:4],1)
    else:o.pair('capture/results/numerical/objective-1/input.npy')
    with pytest.raises(ValueError):o.assess(layout,phase='capture')


def make_guard(layout,tmp_path):
    tmp_path.chmod(0o700);root=tmp_path/'scratch';root.mkdir(mode=0o700)
    return m.Guard(root,m.identity_binding(root),layout)

def write(root,name,data=b'EXPLICIT SMALL ORDINARY BYTE FIXTURE'):
    p=root/name;p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    # Every intermediate fixture parent is private before observation.
    q=p.parent
    while q!=root:q.chmod(0o700);q=q.parent
    with p.open('xb') as f:f.write(data)
    p.chmod(0o600);return p


def test_real_descriptor_walk_keeps_private_namespace_and_counts_directory_blocks(layout,tmp_path):
    g=make_guard(layout,tmp_path);root=g.root
    write(root,'control.json',b'{}\n');write(root,'cache/cuda/a',b'CACHE-LABEL');write(root,'capture/logs/numerical.log',b'LOG-LABEL')
    report,files,dirs=g.scan(phase='setup')
    assert set(files)=={'control.json','cache/cuda/a','capture/logs/numerical.log'}
    assert report['allocated_bytes']==sum(p.lstat().st_blocks*512 for p in [root,*root.rglob('*')])
    assert report['logical_bytes']==sum(row['bytes'] for row in files.values())
    assert g.read('control.json',256*1024)==b'{}\n' and dirs['.']['identity']==tuple(g.binding['root_identity'])

@pytest.mark.parametrize('fault',['symlink','directory-symlink','hardlink','mode','directory-mode','fifo','unknown-file','unknown-dir','root-mode','root-replace','parent-mode','owner','read-cap','read-link','read-parent'])
def test_real_fs_refusals_never_follow_or_rewrite_invalid_paths(layout,tmp_path,fault):
    g=make_guard(layout,tmp_path);r=g.root;p=write(r,'control.json',b'PRESERVE')
    if fault=='symlink':p.rename(r/'proposal.json');p.symlink_to('proposal.json')
    elif fault=='directory-symlink':(r/'cache').symlink_to(tmp_path,target_is_directory=True)
    elif fault=='hardlink':os.link(p,r/'proposal.json')
    elif fault=='mode':p.chmod(0o644)
    elif fault=='directory-mode':
        (r/'cache').mkdir(mode=0o700);(r/'cache').chmod(0o755)
    elif fault=='fifo':os.mkfifo(r/'proposal.json',0o600)
    elif fault=='unknown-file':write(r,'unexpected.bin')
    elif fault=='unknown-dir':(r/'unlisted').mkdir(mode=0o700)
    elif fault=='root-mode':r.chmod(0o755)
    elif fault=='root-replace':r.rename(tmp_path/'preserved');r.mkdir(mode=0o700)
    elif fault=='parent-mode':tmp_path.chmod(0o755)
    elif fault=='owner':g.owner=(-1,g.owner[1])
    elif fault=='read-cap':
        with pytest.raises(ValueError):g.read('control.json',1)
        assert p.read_bytes()==b'PRESERVE';return
    elif fault=='read-link':
        os.link(p,r/'proposal.json')
        with pytest.raises(ValueError,match='single-link'):g.read('control.json',256*1024)
        return
    else:
        (r/'configs').symlink_to(tmp_path,target_is_directory=True)
        with pytest.raises(ValueError,match='source parent'):g.read('configs/base.json',len(BASE))
        return
    with pytest.raises((ValueError,OSError)):g.scan(phase='capture')
    assert not (r/'REFUSAL-WRITTEN-BY-GUARD').exists()


def test_real_owned_result_pair_is_accepted_only_by_live_inventory(layout,tmp_path):
    g=make_guard(layout,tmp_path);name='capture/results/numerical/objective-0/input.npy'
    p=write(g.root,name+'.partial',b'OPAQUE LABEL; NOT AN ARRAY');os.link(p,g.root/name)
    report,files,_=g.scan(phase='capture')
    assert report['publication_pairs']==[[name,name+'.partial']] and len(files)==2
    assert report['named_bytes']==2*report['logical_bytes']
    with pytest.raises(ValueError):g.scan(phase='collection',settled=True)
    assert p.lstat().st_nlink==2


def test_pinned_real_code_leaf_reads_without_loading_it(layout,tmp_path):
    g=make_guard(layout,tmp_path);key='models/execution.py';data=(ROOT/key).read_bytes()
    write(g.root,'overlay/'+key,data)
    actual=g.read('overlay/'+key,layout.assets['overlay/'+key]['bytes'])
    assert hashlib.sha256(actual).hexdigest()==layout.assets['overlay/'+key]['sha256']
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules

@pytest.mark.parametrize('fault',[None,'missing','sparse','short'])
def test_emergency_extent_requires_real_allocated_bytes_not_only_length(fault):
    files={}
    if fault!='missing':files['emergency-space.bin']={'bytes':16*m.MiB if fault!='short' else 1,'allocated_bytes':16*m.MiB if fault!='sparse' else 0}
    if fault is None:m.Guard.require_emergency(files)
    else:
        with pytest.raises(ValueError,match='Full allocated emergency'):m.Guard.require_emergency(files)


def proof():
    return {'kind':'scratch-allocation-proof-v2','passed':True,'complete':True,'proof_removed':True,'fsync_succeeded':True,
        'preallocator_injected_for_test':False,'requested_bytes':5*m.GiB//2,'bytes_written':5*m.GiB//2,
        'bytes_read_back':5*m.GiB//2,'setup_ceiling_bytes':3*m.GiB,'peak_observed_allocated_bytes':5*m.GiB//2+4096,
        'wall_seconds':42.,'sha256':'9'*64,'origin':'SYNTHETIC METADATA ONLY; NO PROOF EXECUTED'}

@pytest.mark.parametrize('fault',[None,'wrong-digest','short-read','injected','incomplete','sparse-peak-bool','peak-short','time-nan','time-bool','time-limit','bad-digest','leftover-proof','duplicate-json'])
def test_saved_proof_receipt_checks_retained_without_performing_a_proof(layout,tmp_path,fault):
    g=make_guard(layout,tmp_path);p=proof()
    if fault=='short-read':p['bytes_read_back']-=1
    elif fault=='injected':p['preallocator_injected_for_test']=True
    elif fault=='incomplete':p['complete']=False
    elif fault=='sparse-peak-bool':p['peak_observed_allocated_bytes']=True
    elif fault=='peak-short':p['peak_observed_allocated_bytes']=5*m.GiB//2-1
    elif fault=='time-nan':p['wall_seconds']=float('nan')
    elif fault=='time-bool':p['wall_seconds']=True
    elif fault=='time-limit':p['wall_seconds']=61.
    elif fault=='bad-digest':p['sha256']='WRONG'
    data=(json.dumps(p,allow_nan=True)+'\n').encode()
    if fault=='duplicate-json':data=data.rstrip()[:-1]+b',"complete": true}\n'
    write(g.root,'allocation-result-v2.json',data)
    if fault=='leftover-proof':write(g.root,'space-proof-v2.bin.partial',b'PRESERVED SMALL PROOF LABEL; NOT A PROOF')
    want='e'*64 if fault=='wrong-digest' else hashlib.sha256(data).hexdigest()
    if fault is None:
        got=g.verify_proof(want)
        assert not got['proof_execution_authenticated'] and not got['scientific_qualified']
    else:
        with pytest.raises(ValueError):g.verify_proof(want)
    assert (g.root/'allocation-result-v2.json').read_bytes()==data


def test_original_proof_predicates_and_receipt_scope_are_not_weakened():
    # Cold AST comparison of the old predicate components; no old function call.
    old=ast.parse((ROOT/'models/cpu_oracle_control_v1.py').read_bytes())
    f=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='validate_proof')
    text=ast.unparse(f)
    for name in ('passed','complete','proof_removed','fsync_succeeded','preallocator_injected_for_test','requested_bytes','bytes_written','bytes_read_back',
        'setup_ceiling_bytes','peak_observed_allocated_bytes','wall_seconds','space-proof-v2.bin.partial'):
        assert name in text
    source=ast.parse((ROOT/'models/predictor_pgd_split_scratch_v1.py').read_bytes())
    method=next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='verify_proof')
    for name in ('passed','complete','proof_removed','fsync_succeeded','preallocator_injected_for_test','requested_bytes','bytes_written','bytes_read_back',
        'setup_ceiling_bytes','peak_observed_allocated_bytes','wall_seconds'):
        assert name in ast.unparse(method)
    assert 'posix_fallocate' not in ast.unparse(source) and 'subprocess' not in ast.unparse(source)


class SourceMetadataHarness:
    """Exercise full verification order with explicitly substituted tiny descriptors.

    The production constructor has no callback/descriptor replacement parameter;
    this test-only namespace cannot qualify a real installation or proof.
    """
    def __init__(self,g,*,fault=None):
        self.events=[];self.g=g;self.fault=fault
        a=b'EXPLICIT SOURCE BYTE STUB A';b=b'EXPLICIT SOURCE BYTE STUB B'
        self.data={'control.json':a,'proposal.json':b}
        self.spec={n:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for n,v in self.data.items()}
        self.rows={n:{'identity':(55,i,os.getuid(),0o600,1),'bytes':len(v),'allocated_bytes':4096,'mtime_ns':1} for i,(n,v) in enumerate(self.data.items(),100)}
        self.rows['emergency-space.bin']={'bytes':16*m.MiB,'allocated_bytes':16*m.MiB}
        self.calls=0
    def scan(self,**kw):
        self.events.append(('scan',kw));self.calls+=1;rows=copy.deepcopy(self.rows)
        if self.fault=='missing':rows.pop('proposal.json')
        if self.fault=='race' and self.calls==2:rows['control.json']['mtime_ns']+=1
        return {},rows,{}
    def read(self,name,size):
        self.events.append(('read',name,size))
        return self.data[name]+(b'CHANGED' if self.fault=='digest' and name=='proposal.json' else b'')

@pytest.mark.parametrize('fault',[None,'missing','digest','race'])
def test_complete_source_verification_order_requires_all_hashes_and_stable_snapshots(layout,tmp_path,fault):
    g=make_guard(layout,tmp_path);h=SourceMetadataHarness(g,fault=fault)
    # An explicit test double, not mutation of the shared production layout.
    g.layout=NS(assets=h.spec,installation=layout.installation,installation_sha256=layout.installation_sha256)
    g.scan=h.scan;g.read=h.read
    if fault is None:
        result=g.verify_assets()
        assert result['assets']==2 and not result['execution_or_authorization_authenticated']
        assert set(g.pins)==set(h.data) and [e[0] for e in h.events]==['scan','read','read','scan']
    else:
        with pytest.raises(ValueError):g.verify_assets()
        assert not g.pins

@pytest.mark.parametrize('fault',[None,'previous','reserve'])
def test_proof_readiness_consumes_no_work_and_refuses_previous_artifacts(layout,tmp_path,fault):
    g=make_guard(layout,tmp_path);events=[]
    files={'emergency-space.bin':{'bytes':16*m.MiB,'allocated_bytes':16*m.MiB}}
    if fault=='previous':files['allocation-result-v2.json']={'bytes':1}
    if fault=='reserve':files.clear()
    def scan(**kw):events.append(kw);return {'logical_bytes':1024,'allocated_bytes':4096,'opaque':'OBSERVED-METADATA-STUB'},files,{}
    g.scan=scan
    if fault is None:assert g.proof_readiness()['proof_launch_authorized'] is False
    else:
        with pytest.raises(ValueError):g.proof_readiness()
    assert events==[{'phase':'setup','initial':True,'settled':True}]


def test_reserved_other_file_budget_and_largest_legal_mutable_partials_fit(layout):
    # Exact finite assets/root names plus 48 cache entries and all live partials
    # cannot exhaust the reserved 256 paths; actual global counts remain checked.
    other=set(layout.assets)|set(m.ROOT_META)|{'emergency-space.bin','space-proof-v2.bin.partial'}
    assert len(other)+48+3<=256
    o=Observation()
    for name in ('capture/results/run-status.json','capture/results/numerical/objective-0/numerical.json'):
        o.file(name,256*1024);o.file(name+'.partial',256*1024)
    row=o.assess(layout,phase='capture')
    assert row['partials']['metadata']==2 and row['extra_metadata_bytes']==512*1024


@pytest.mark.parametrize('metric',['logical_bytes','allocated_bytes'])
def test_proof_readiness_projects_full_extent_and_original_reserve_before_any_work(layout,tmp_path,metric):
    g=make_guard(layout,tmp_path);rows={'emergency-space.bin':{'bytes':16*m.MiB,'allocated_bytes':16*m.MiB}}
    report={'logical_bytes':16*m.MiB,'allocated_bytes':16*m.MiB}
    report[metric]=3*m.GiB-5*m.GiB//2-32*m.MiB+1
    calls=[]
    def scan(**kw):calls.append(kw);return report,rows,{}
    g.scan=scan
    with pytest.raises(ValueError,match='Full proof plus original 32 MiB'):g.proof_readiness()
    assert len(calls)==1


@pytest.mark.parametrize('fault',['changed','missing','replaced'])
def test_later_scans_refuse_changed_previously_verified_asset_metadata(layout,tmp_path,fault):
    g=make_guard(layout,tmp_path);name='overlay/models/execution.py';data=(ROOT/'models/execution.py').read_bytes();p=write(g.root,name,data)
    _,files,_=g.scan(phase='capture');g.pins={name:files[name]}
    if fault=='missing':p.rename(g.root/'overlay/models/predictor_eot8_capture_v1.py')
    elif fault=='replaced':
        p.rename(g.root/'overlay/models/predictor_eot8_capture_v1.py');write(g.root,name,data)
    else:
        # Keep shape/extent; only owned fixture bytes and mtime change.
        p.write_bytes(b'#'+data[1:]);os.utime(p,ns=(p.stat().st_atime_ns,files[name]['mtime_ns']+1))
    # Preserve the renamed fixture without introducing an unpinned-path refusal.
    if fault in ('missing','replaced'):
        # The destination is a real declared runtime source with its own larger
        # cap; keep its small content within that cap to isolate the pin check.
        alias=g.root/'overlay/models/predictor_eot8_capture_v1.py'
        alias.write_bytes(b'PRESERVED ORIGINAL SOURCE LABEL');alias.chmod(0o600)
    with pytest.raises(ValueError,match='Verified source asset changed'):g.scan(phase='capture')
