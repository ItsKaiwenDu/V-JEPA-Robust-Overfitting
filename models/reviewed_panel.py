"""Append-only review reconciliation. No numerical dispatch or model execution."""
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import re
import tempfile
import uuid

KIND='reviewed-panel-reconciliation-v1'
BASE='Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024'
PREFIX='reviewed-panel-reconciliation-20261026'
TARGET='reviewed-panel-20261026'
LEGACY='two-host-panel-20261026'
PACKAGE='unpacked/matched-panel-20261024'
REFERENCE='two-host-execution-20261026/unpacked/two-host-execution-20261026'
RECOVERY='contended-panel-recovery-lockfix-20261026/unpacked/contended-panel-recovery-lockfix-20261026'
RP='5ca7596fe95ecb8f6b6c6752426e56bb354b1f019d3576c6618f7ede98ae4116'
MP='da5aeb93bd6d3e46040b158cc14f0d6feb1284a83b2b2778a3639b420db0acf9'
RECOVERY_PROPOSAL='4e496e51f93545301d6335aa33f1b03ac062de667717617e7d0056a7c4c374d8'
RECOVERY_MODULE='0f9b8a4a09c54088b660741476abca3d51787e7e4ae643834f332509dc2d513c'
AUTH='721bbf9e77804b7acc7bdf542c63e2d29cda3f56b148dec7b24fea810c0f9d2d'
GRANT='814965108dd544eba53ebc5e34496d27'
STOP='776e80c041599c1c1a35ee31348312b4f696eec03f0e1e8787cc27dd231b11f5'
JOURNAL='cf851847721b40a24b45563ee4b06f589024269c94d4cbf73ce96af79a3e6814'
V1_REVIEW='b08361f743f71b2066cb3ac63f4179891c4b1f3b06f293e5199760f346a8acff'
V1_EXEC='1e59a735e2f0b6f5485cf0867851c638dcd027715d19270422f390f7745aa70e'
NORMALS=('clip-03-bank-1','clip-04-bank-0','clip-05-bank-0')
RECOVERED='clip-04-bank-1'
ORDER=[f'clip-{i:02d}-bank-{b}' for i in range(10) for b in (0,1)]
ACCEPTED=ORDER[:11]
SOURCES=('models/reviewed_panel.py','scripts/reconcile_reviewed_panel.py','scripts/prepare_reviewed_panel.py',
         'tests/test_reviewed_panel.py','tests/fixtures/reviewed_panel_metadata.json')
TARGET_FILES={'genesis.json','imports.json','progress.json','STOP.json',
              'reviews/clip-04-bank-1.json','reviews/clip-05-bank-0.json'}


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def bound(root,name):
    root=Path(root).resolve();p=PurePosixPath(name);path=root/name
    if (not name or name=='.' or p.is_absolute() or '..' in p.parts or '\\' in name or str(p)!=name
        or not path.resolve().is_relative_to(root)
        or any(q.is_symlink() for q in [path,*list(path.parents)[:len(p.parts)-1]])):
        raise ValueError('Unsafe reconciliation path: '+name)
    return path


def json_bytes(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def record_bytes(payload):return json_bytes({'payload':payload,'sha256':canonical(payload)})


def record_text(text,pin=None):
    if pin is not None and hashlib.sha256(text.encode()).hexdigest()!=pin:raise ValueError('Review text differs from its external file pin.')
    value=json.loads(text)
    if set(value)!={'payload','sha256'} or canonical(value['payload'])!=value['sha256']:raise ValueError('Invalid checksummed review record.')
    return value['payload']


def validate_operation(op,pin):
    policy={'kind':KIND,'host':'linux10609','target_namespace':TARGET,'legacy_namespace':LEGACY,
        'legacy_revision_sha256':RP,'legacy_implementation_sha256':MP,'recovery_proposal_sha256':RECOVERY_PROPOSAL,
        'original_authorization_file_sha256':AUTH,'original_recovery_grant_id':GRANT,'legacy_STOP_sha256':STOP,
        'accepted_case_ids':ACCEPTED,'metadata_only':True,'max_seconds':300,'kill_grace_seconds':15,
        'legacy_records_changed':False,'STOP_to_remain_active':True,'new_GPU_dispatch_authorized':False,
        'models_started':False,'test_suite_started':False,'automatic_queue':False,'research_admitted':False}
    if canonical(op)!=pin or any(k not in op or canonical(op[k])!=canonical(v) for k,v in policy.items()):
        raise ValueError('Use only the pinned eleven-case metadata reconciliation.')
    if (set(op['sources_sha256'])!=set(SOURCES) or re.fullmatch('[a-f0-9]{64}',op['inputs_sha256']) is None
        or any(re.fullmatch('[a-f0-9]{64}',p) is None for p in op['sources_sha256'].values())):
        raise ValueError('Reconciliation source inventory differs.')


def validate_input(data):
    if (data['kind']!=KIND+'-input' or set(data['case_index'])!=set(ACCEPTED) or set(data['normal_reviews'])!=set(NORMALS)
        or data['recovery_proposal_sha256']!=RECOVERY_PROPOSAL or data['original_authorization_file_sha256']!=AUTH
        or data['recovery_grant_id']!=GRANT or data['numerical_settings_changed'] is not False
        or data['models_started'] is not False or data['STOP_to_remain_active'] is not True or data['research_admitted'] is not False):
        raise ValueError('Wrong review reconciliation scope.')
    journal=record_text(data['legacy_journal_text'],JOURNAL);record_text(data['legacy_review_text'],V1_REVIEW)
    if set(journal['completed'])!=set(ORDER[:7]) or journal['execution_sha256']!=V1_EXEC:raise ValueError('Seven original origins changed.')
    stop=record_text(data['legacy_STOP_text'],STOP)
    if stop['case_id']!=RECOVERED or stop['complete'] is not False:raise ValueError('Preserve the original failed outcome.')
    markers={cid:record_text(row['text'],row['sha256']) for cid,row in data['normal_reviews'].items()}
    recovered=record_text(data['recovery_review']['text'],data['recovery_review']['sha256'])
    if (data['normal_reviews']['clip-05-bank-0']['sha256']!='653856812c19671e25c68dcb666b18707c3b2a7dcbd012d83957397ba9342ed0'
        or data['recovery_review']['sha256']!='fadc158372c2abc7574f7d33ded5d5930c3b8230d297d6605f20e192c71e179c'):
        raise ValueError('Pending reviews differ from the independently accepted exports.')
    if (recovered['kind']!='matched-panel-contended-recovery-lockfix-v1-independent-review' or recovered['case_id']!=RECOVERED
        or recovered['host']!='linux10603' or recovered['attempt']!='attempt-02' or recovered['grant_id']!=GRANT
        or recovered['proposal_sha256']!=RECOVERY_PROPOSAL or recovered['authorization_file_sha256']!=AUTH
        or recovered['reviewed_export'] is not True or recovered['research_admitted'] is not False):raise ValueError('Wrong recovery origin.')
    base=PACKAGE+'/runs/panel.qIXeDn'
    for cid,row in data['case_index'].items():
        if row['case_id']!=cid:raise ValueError('Aliased case identity.')
        for name in ('directory','archive_path','archive_manifest_path','review_path'):bound(Path('/fixed'),row[name])
        if cid in ORDER[:7]:
            p=journal['completed'][cid]
            expected={'origin':'frozen-v1','host':'linux10609','directory':base+'/'+p['directory'],
                'archive_path':base+'/'+p['archive_path'],'archive_sha256':p['archive_sha256'],
                'archive_manifest_path':base+'/'+p['archive_manifest_path'],'archive_manifest_sha256':p['archive_manifest_sha256'],
                'receipt_file_sha256':p['receipt_file_sha256'],'execution_sha256':V1_EXEC,
                'review_path':REFERENCE+'/bootstrap-review.json','review_file_sha256':V1_REVIEW}
        else:
            m=recovered if cid==RECOVERED else markers[cid];is_recovery=cid==RECOVERED
            if not is_recovery and (m['kind']!='matched-panel-two-host-case-review-v1' or m['revision_sha256']!=RP
                or m['implementation_sha256']!=MP or m['case_id']!=cid or not m['reviewed_export'] or m['research_admitted']):
                raise ValueError('Wrong original two-host case review.')
            directory=LEGACY+'/cases/'+cid+('/attempt-02' if is_recovery else '/attempt-01')
            review_root=TARGET if cid in (RECOVERED,'clip-05-bank-0') else LEGACY
            expected={'origin':'explicit-attended-recovery' if is_recovery else 'two-host-revision','host':m['host'],
                'directory':directory,'archive_path':directory+('/recovery-evidence.tar.gz' if is_recovery else '/revision-evidence.tar.gz'),
                'archive_sha256':m['archive_sha256'],'execution_sha256':m['execution_sha256'],
                'review_path':review_root+'/reviews/'+cid+'.json','review_file_sha256':data['recovery_review']['sha256'] if is_recovery else data['normal_reviews'][cid]['sha256']}
            if row['archive_manifest_path']!=row['archive_path']+'.manifest.json':raise ValueError('Wrong saved archive sidecar path.')
            if is_recovery:
                expected.update(grant_id=GRANT,original_failed_outcome_path=LEGACY+'/outcomes/'+cid+'.json',original_failed_outcome_sha256=STOP)
        if any(k not in row or canonical(row[k])!=canonical(v) for k,v in expected.items()):raise ValueError('Original execution identity or artifact origin changed: '+cid)
        for path_key,pin_key in [('archive_path','archive_sha256'),('archive_manifest_path','archive_manifest_sha256')]:
            if data['preserved_files_sha256'].get(row[path_key])!=row[pin_key]:raise ValueError('Unpinned saved archive.')
        if data['preserved_files_sha256'].get(row['directory']+'/receipt.json')!=row['receipt_file_sha256']:raise ValueError('Unpinned original receipt.')
    if data['preserved_files_sha256'].get(LEGACY+'/STOP.json')!=STOP:raise ValueError('The legacy stop must remain byte-identical.')
    for name in data['preserved_files_sha256']:bound(Path('/fixed'),name)
    return journal,markers,recovered


def records(op,data):
    validate_operation(op,canonical(op));validate_input(data)
    imports={'kind':KIND+'-imports','operation_sha256':canonical(op),'case_origins':data['case_index'],
        'original_execution_identities_preserved':True,'research_admitted':False}
    progress={'kind':KIND+'-progress','operation_sha256':canonical(op),'imports_sha256':canonical(imports),
        'accepted_case_ids':ACCEPTED,'pending_case_ids':ORDER[11:],'complete_clip_pairs':5,
        'legacy_accepted_case_ids':ORDER[:9],'historical_failed_case_ids':[RECOVERED],
        'independently_reviewed_recovery_case_ids':[RECOVERED],'stop_dispatch':True,
        'new_GPU_dispatch_authorized':False,'complete':False,'scientific_aggregate':None,'research_admitted':False}
    genesis={'kind':KIND+'-genesis','operation_sha256':canonical(op),'inputs_file_sha256':op['inputs_sha256'],
        'imports_sha256':canonical(imports),'legacy_root':LEGACY,'legacy_STOP_sha256':STOP,
        'source_sha256':op['sources_sha256'],'metadata_only':True,'research_admitted':False}
    result={'imports.json':record_bytes(imports),'progress.json':record_bytes(progress),'genesis.json':record_bytes(genesis),
        'STOP.json':data['legacy_STOP_text'].encode(),
        'reviews/'+RECOVERED+'.json':data['recovery_review']['text'].encode(),
        'reviews/clip-05-bank-0.json':data['normal_reviews']['clip-05-bank-0']['text'].encode()}
    if set(result)!=TARGET_FILES:raise ValueError('Wrong append-only target inventory.')
    return result


def check_preserved(transfer,pins):
    observations={}
    for name,pin in pins.items():
        path=bound(transfer,name)
        if not path.is_file() or digest(path)!=pin:raise ValueError('A preserved source or artifact changed: '+name)
        observations[name]=pin
    return observations


def check_targets(root,targets):
    root=Path(root)
    if root.is_symlink():raise ValueError('Reconciled namespace must not be a symlink.')
    if root.exists():
        for path in root.rglob('*'):
            name=str(path.relative_to(root))
            temporary=re.fullmatch(r'\.(genesis|imports|progress|STOP|clip-04-bank-1|clip-05-bank-0)\.json\.[a-f0-9]{32}\.tmp',path.name)
            if (path.is_symlink() or (path.is_dir() and name!='reviews')
                or (path.is_file() and name not in TARGET_FILES|{'.reconciliation.lock'} and temporary is None)):
                raise ValueError('Unexpected state in the metadata-only namespace: '+name)
    existing=[]
    for name,data in targets.items():
        path=bound(root,name)
        if path.exists():
            if not path.is_file() or path.read_bytes()!=data:raise ValueError('Conflicting reconciled record: '+name)
            existing.append(name)
    return existing


def exclusive_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(temporary,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temporary.unlink(missing_ok=True)


def publish(root,targets,verify_old):
    """Publish the genesis commit last. Exact partial publication is resumable."""
    verify_old();existing=check_targets(root,targets);created=[]
    for name in sorted(TARGET_FILES-{'genesis.json'})+['genesis.json']:
        if name not in existing:exclusive_bytes(bound(root,name),targets[name]);created.append(name)
        if bound(root,name).read_bytes()!=targets[name]:raise ValueError('Reconciliation readback mismatch.')
    verify_old()
    return created,existing


def load(packet,pin,user_home):
    packet=Path(packet).resolve();transfer=(Path(user_home)/BASE).resolve()
    if packet!=(transfer/PREFIX/'unpacked'/PREFIX).resolve():raise ValueError('Wrong fixed reconciliation packet.')
    op=json.loads((packet/'RECONCILIATION.json').read_text());validate_operation(op,pin)
    for name,p in op['sources_sha256'].items():
        if digest(bound(packet,name))!=p:raise ValueError('Reconciliation source changed: '+name)
    if digest(packet/'INPUT.json')!=op['inputs_sha256']:raise ValueError('Externally pinned reconciliation input changed.')
    data=json.loads((packet/'INPUT.json').read_text());validate_input(data)
    reference=transfer/REFERENCE;manifest=json.loads((reference/'IMPLEMENTATION.json').read_text())
    if canonical(manifest)!=MP:raise ValueError('Original implementation changed.')
    for name,p in manifest['source_sha256'].items():
        if digest(bound(reference,name))!=p:raise ValueError('Original source changed before import.')
    module=importlib.util.spec_from_file_location('reviewed_original_operator',reference/'models/two_host_panel.py')
    ops=importlib.util.module_from_spec(module);module.loader.exec_module(ops)
    revision=json.loads((reference/'configs/matched_panel_two_host_v1.json').read_text());package=transfer/PACKAGE
    numeric,_,old=ops.check_sources(package,reference,revision,manifest,RP,MP)
    recovery_packet=transfer/RECOVERY;recovery_path=recovery_packet/'scripts/recover_contended_panel_case.py'
    if digest(recovery_path)!=RECOVERY_MODULE:raise ValueError('Qualified recovery verifier changed before import.')
    module=importlib.util.spec_from_file_location('reviewed_qualified_recovery',recovery_path)
    recovery=importlib.util.module_from_spec(module);module.loader.exec_module(recovery)
    recovery_context=recovery.load_context(recovery_packet,RECOVERY_PROPOSAL,AUTH,user_home)
    if recovery_context[0]['sources_sha256']!=data['recovery_source_sha256']:raise ValueError('Qualified recovery source inventory changed.')
    return op,data,transfer,package,reference,ops,revision,manifest,numeric,old,recovery,recovery_packet,recovery_context


def check_legacy(status):
    expected={'accepted_case_ids':ORDER[:9],'computed_new_case_ids':list(NORMALS),
        'claimed_case_ids':[c for c in ORDER if c in set(NORMALS)|{RECOVERED}],
        'failed_case_ids':[RECOVERED],'stop_dispatch':True,'complete':False,'scientific_aggregate':None,'research_admitted':False}
    if any(status.get(k)!=v for k,v in expected.items()):raise ValueError('Legacy coordinator changed; preserve it for independent review.')


def verify_live(context,temporary):
    op,data,transfer,package,reference,ops,revision,manifest,numeric,old,recovery,recovery_packet,rc=context
    from models.matched_panel_case import verify_case
    # Seven original cases were read by verify_imports before Registry construction.
    for cid in NORMALS:
        row=data['case_index'][cid];marker=record_text(data['normal_reviews'][cid]['text'],data['normal_reviews'][cid]['sha256'])
        archive=bound(transfer,row['archive_path']);sidecar=json.loads(bound(transfer,row['archive_manifest_path']).read_text())
        rebuilt=ops.review_case(package,reference,RP,MP,archive,sidecar,row['archive_sha256'],Path(temporary)/cid)
        if rebuilt!=marker:raise ValueError('Saved normal export differs from its independent marker.')
        receipt=verify_case(bound(transfer,row['directory']),plan=numeric,case_id=cid,execution=ops.execution_identity(revision,manifest,row['host']))
        if canonical(receipt)!=marker['numeric_receipt_sha256']:raise ValueError('Live normal case differs from its reviewed archive.')
    row=data['case_index'][RECOVERED];marker=record_text(data['recovery_review']['text'],data['recovery_review']['sha256'])
    archive=bound(transfer,row['archive_path']);sidecar=json.loads(bound(transfer,row['archive_manifest_path']).read_text())
    rebuilt=recovery.review_case(recovery_packet,reference,package,RECOVERY_PROPOSAL,AUTH,archive,sidecar,row['archive_sha256'],Path(temporary)/RECOVERED)
    if rebuilt!=marker:raise ValueError('Saved recovery differs from the original accepted marker.')
    receipt=verify_case(bound(transfer,row['directory']),plan=numeric,case_id=RECOVERED,execution=recovery.execution_identity(rc))
    if canonical(receipt)!=marker['numeric_receipt_sha256']:raise ValueError('Live recovery differs from its reviewed archive.')
    recovery.check_preserved(transfer/LEGACY,rc[0],rc[3])
    for cid,row in data['case_index'].items():
        if digest(bound(transfer,row['directory']+'/receipt.json'))!=row['receipt_file_sha256']:raise ValueError('Case receipt changed.')
        if cid in ORDER[:7]:
            p=old['completed'][cid]
            if row['directory']!=PACKAGE+'/runs/panel.qIXeDn/'+p['directory']:raise ValueError('Original case locator changed.')
    return 11


def apply(packet,pin,user_home,host):
    if host!='linux10609':raise ValueError('Use the declared metadata-only host.')
    context=load(packet,pin,user_home)
    op,data,transfer,package,reference,ops,revision,manifest,numeric,old,recovery,recovery_packet,rc=context
    legacy=transfer/LEGACY;target=transfer/TARGET;targets=records(op,data)
    import torch
    torch.set_num_threads(4)
    with ExitStack() as stack:
        for assigned in ('linux10609','linux10603'):stack.enter_context(ops.file_lock(legacy/'leases'/assigned,wait_seconds=0))
        # Constructor locks internally. Never put it inside the registry transaction.
        registry=ops.Registry(legacy,revision,MP,ops.verify_imports(package,old))
        with ops.file_lock(legacy/'.registry.lock'):
            before=registry.status();check_legacy(before);check_preserved(transfer,data['preserved_files_sha256'])
            with tempfile.TemporaryDirectory(prefix='reviewed-panel-CPU-') as temporary:verified=verify_live(context,temporary)
            if verified!=11:raise ValueError('Incomplete source/evidence review.')
            if registry.status()!=before:raise ValueError('Legacy state changed during CPU recheck.')
            if target.is_symlink():raise ValueError('Wrong new namespace.')
            target.mkdir(exist_ok=True)
            with ops.file_lock(target/'.reconciliation.lock'):
                created,existing=publish(target,targets,lambda:check_preserved(transfer,data['preserved_files_sha256']))
                observed=check_preserved(transfer,data['preserved_files_sha256'])
            after=registry.status();check_legacy(after)
            if before!=after:raise ValueError('Metadata reconciliation altered legacy state.')
    receipt={'kind':KIND+'-receipt','operation_sha256':pin,'inputs_sha256':op['inputs_sha256'],'host':host,
        'created':created,'preexisting_identical':existing,'records_readback':{n:bound(target,n).read_text() for n in targets},
        'records_file_sha256':{n:digest(bound(target,n)) for n in targets},'fresh_preserved_files_sha256':observed,
        'legacy_status_before':before,'legacy_status_after':after,'original_imports_reverified':7,'newer_saved_cases_reverified':4,
        'effective_progress':record_text(targets['progress.json'].decode()),'legacy_STOP_readback':(legacy/'STOP.json').read_text(),
        'legacy_records_changed':False,'models_started':False,'test_suite_started':False,'new_GPU_dispatch_authorized':False,
        'research_admitted':False}
    verify_receipt(receipt,op,data)
    return receipt


def verify_receipt(receipt,op,data):
    targets=records(op,data);created=receipt['created'];existing=receipt['preexisting_identical'];progress=record_text(targets['progress.json'].decode())
    if (receipt['kind']!=KIND+'-receipt' or receipt['operation_sha256']!=canonical(op) or receipt['inputs_sha256']!=op['inputs_sha256']
        or receipt['host']!='linux10609' or set(created)&set(existing) or set(created)|set(existing)!=TARGET_FILES
        or len(created)+len(existing)!=len(TARGET_FILES) or receipt['fresh_preserved_files_sha256']!=data['preserved_files_sha256']
        or receipt['original_imports_reverified']!=7 or receipt['newer_saved_cases_reverified']!=4
        or receipt['legacy_status_before']!=receipt['legacy_status_after'] or receipt['effective_progress']!=progress
        or receipt['legacy_STOP_readback']!=data['legacy_STOP_text']
        or any(receipt[k] is not False for k in ('legacy_records_changed','models_started','test_suite_started','new_GPU_dispatch_authorized','research_admitted'))):
        raise ValueError('Reconciliation receipt does not match the bounded operation.')
    check_legacy(receipt['legacy_status_after'])
    if set(receipt['records_readback'])!=TARGET_FILES or set(receipt['records_file_sha256'])!=TARGET_FILES:raise ValueError('Incomplete committed readback.')
    for name,expected in targets.items():
        if receipt['records_readback'][name].encode()!=expected or receipt['records_file_sha256'][name]!=hashlib.sha256(expected).hexdigest():
            raise ValueError('Published reconciliation differs from expected bytes.')
    return {'kind':KIND+'-receipt-verification','verified':True,'effective_accepted_cases':11,'legacy_accepted_cases':9,
        'complete_clip_pairs':5,'preserved_files_verified':len(data['preserved_files_sha256']),'STOP_still_active':True,
        'models_started':False,'new_GPU_dispatch_authorized':False,'research_admitted':False}
