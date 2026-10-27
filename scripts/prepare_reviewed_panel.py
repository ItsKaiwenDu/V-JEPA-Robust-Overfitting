#!/usr/bin/env python3
"""Build the metadata-only eleven-case reconciliation handoff; no remote work."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())['payload']


def inputs():
    recovery_raw=ROOT/'runs/pretrained/diagnostic/contended-panel-recovery-lockfix-20261026'
    qualified=recovery_raw/'extracted/contended-panel-recovery-lockfix-20261026'
    proposal=json.loads((qualified/'RECOVERY.json').read_text())
    accepted=json.loads((ROOT/'report/checks/2026-10-26-floor-recovery-verification.json').read_text())
    assert accepted['verified'] and accepted['independently_verified_case_count']==11
    reference=ROOT/'runs/pretrained/diagnostic/two-host-panel-20261026/extracted/two-host-execution-20261026'
    journal=read(reference/'bootstrap-journal.json');entries={};normal_reviews={}
    legacy='two-host-panel-20261026';base='unpacked/matched-panel-20261024/runs/panel.qIXeDn';target='reviewed-panel-20261026'
    for cid,p in journal['completed'].items():
        entries[cid]={'case_id':cid,'host':'linux10609','origin':'frozen-v1','directory':base+'/'+p['directory'],
            'archive_path':base+'/'+p['archive_path'],'archive_sha256':p['archive_sha256'],
            'archive_manifest_path':base+'/'+p['archive_manifest_path'],'archive_manifest_sha256':p['archive_manifest_sha256'],
            'receipt_file_sha256':p['receipt_file_sha256'],'execution_sha256':'1e59a735e2f0b6f5485cf0867851c638dcd027715d19270422f390f7745aa70e',
            'review_path':'two-host-execution-20261026/unpacked/two-host-execution-20261026/bootstrap-review.json',
            'review_file_sha256':digest(reference/'bootstrap-review.json')}
    first=ROOT/'runs/pretrained/diagnostic/two-host-export-incident-20261026/evidence-download-20261026T185802Z.H23uD0'
    repair=read(first/'ledger-repair-plan.json')
    for cid in ('clip-03-bank-1','clip-04-bank-0','clip-05-bank-0'):
        if cid=='clip-05-bank-0':
            d=Path((ROOT/'runs/pretrained/diagnostic/panel-continuation-20261026/linux10609/latest-download.txt').read_text().strip())
            v=d/'independently-reviewed';event=read(d/'downloaded/ledger/export.json')
        else:
            v=first/'independently-reviewed'/cid;event=repair['proposed_records']['exports/'+cid+'.json']['payload']
        marker=v/'independent-review.json';m=read(marker);normal_reviews[cid]={'text':marker.read_text(),'sha256':digest(marker)}
        entries[cid]={'case_id':cid,'host':m['host'],'origin':'two-host-revision','directory':legacy+'/cases/'+cid+'/attempt-01',
            'archive_path':legacy+'/'+event['archive_path'],'archive_sha256':event['archive_sha256'],
            'archive_manifest_path':legacy+'/'+event['archive_manifest_path'],'archive_manifest_sha256':event['archive_manifest_sha256'],
            'receipt_file_sha256':digest(v/'numeric-reviewed/receipt.json'),'execution_sha256':m['execution_sha256'],
            'review_path':(target if cid=='clip-05-bank-0' else legacy)+'/reviews/'+cid+'.json','review_file_sha256':digest(marker)}
    d=Path((recovery_raw/'latest-download.txt').read_text().strip());v=d/'independently-reviewed';marker=v/'independent-review.json';m=read(marker);cid='clip-04-bank-1'
    entries[cid]={'case_id':cid,'host':'linux10603','origin':'explicit-attended-recovery','directory':legacy+'/cases/'+cid+'/attempt-02',
        'archive_path':legacy+'/cases/'+cid+'/attempt-02/recovery-evidence.tar.gz','archive_sha256':m['archive_sha256'],
        'archive_manifest_path':legacy+'/cases/'+cid+'/attempt-02/recovery-evidence.tar.gz.manifest.json',
        'archive_manifest_sha256':digest(d/'downloaded/evidence/recovery-evidence.tar.gz.manifest.json'),
        'receipt_file_sha256':digest(v/'numeric-reviewed/receipt.json'),'execution_sha256':m['execution_sha256'],
        'review_path':target+'/reviews/'+cid+'.json','review_file_sha256':digest(marker),'grant_id':m['grant_id'],
        'original_failed_outcome_path':legacy+'/outcomes/'+cid+'.json',
        'original_failed_outcome_sha256':proposal['preserved_coordinator_files_sha256']['outcomes/'+cid+'.json']}
    preserved={legacy+'/'+n:p for n,p in proposal['preserved_coordinator_files_sha256'].items()}
    previous='contended-panel-recovery-20261026/unpacked/contended-panel-recovery-20261026'
    preserved.update({previous+'/'+n:p for n,p in proposal['preserved_failed_startup_files_sha256'].items()})
    for row in entries.values():
        for name,pin in [('archive_path','archive_sha256'),('archive_manifest_path','archive_manifest_sha256')]:preserved[row[name]]=row[pin]
        preserved[row['directory']+'/receipt.json']=row['receipt_file_sha256']
    preserved[base+'/journal.json']=digest(reference/'bootstrap-journal.json')
    for name,local in [('claim.json','recovery-claim.json'),('outcome.json','recovery-outcome.json')]:
        preserved[legacy+'/recovery-grants/'+cid+'/'+name]=digest(d/'downloaded/ledger'/local)
    directory=entries[cid]['directory'];preserved[directory+'/case-envelope.json']=digest(v/'case-envelope.json')
    inner=json.loads((v/'numeric/numeric-evidence.tar.gz.manifest.json').read_text())
    for name,row in inner['files'].items():
        if name!='PANEL-HANDOFF.json' and not name.startswith('source/'):preserved[directory+'/'+name]=row['sha256']
    preserved[directory+'/numeric-evidence.tar.gz']=inner['sha256'];preserved[directory+'/numeric-evidence.tar.gz.manifest.json']=digest(v/'numeric/numeric-evidence.tar.gz.manifest.json')
    for path in (v/'hardware').glob('*.json'):preserved[directory+'/'+path.name]=digest(path)
    return {'kind':'reviewed-panel-reconciliation-v1-input','date_Pacific':'2026-10-26','case_index':entries,
        'normal_reviews':normal_reviews,'recovery_review':{'text':marker.read_text(),'sha256':digest(marker)},
        'legacy_journal_text':(reference/'bootstrap-journal.json').read_text(),'legacy_review_text':(reference/'bootstrap-review.json').read_text(),
        'preserved_files_sha256':preserved,'legacy_STOP_text':(d/'downloaded/ledger/STOP.json').read_text(),
        'recovery_source_sha256':proposal['sources_sha256'],'recovery_proposal_sha256':m['proposal_sha256'],
        'original_authorization_file_sha256':m['authorization_file_sha256'],'recovery_grant_id':m['grant_id'],
        'numerical_settings_changed':False,'models_started':False,'STOP_to_remain_active':True,'research_admitted':False}


def build():
    s=importlib.util.spec_from_file_location('reviewed_prepare',ROOT/'models/reviewed_panel.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    data=inputs();m.validate_input(data)
    if data!=json.loads((ROOT/'tests/fixtures/reviewed_panel_metadata.json').read_text()):raise ValueError('Genuine metadata fixture differs from reviewed inputs.')
    raw=ROOT/'runs/pretrained/diagnostic'/m.PREFIX;raw.mkdir(exist_ok=False);(raw/'validation').mkdir()
    packet=raw/'staged'/m.PREFIX;packet.mkdir(parents=True)
    for name in m.SOURCES:
        p=packet/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    (packet/'INPUT.json').write_bytes(m.json_bytes(data))
    op={'kind':m.KIND,'date_Pacific':'2026-10-26','host':'linux10609','target_namespace':m.TARGET,'legacy_namespace':m.LEGACY,
        'legacy_revision_sha256':m.RP,'legacy_implementation_sha256':m.MP,'recovery_proposal_sha256':m.RECOVERY_PROPOSAL,
        'original_authorization_file_sha256':m.AUTH,'original_recovery_grant_id':m.GRANT,'legacy_STOP_sha256':m.STOP,
        'accepted_case_ids':m.ACCEPTED,'metadata_only':True,'max_seconds':300,'kill_grace_seconds':15,
        'legacy_records_changed':False,'STOP_to_remain_active':True,'new_GPU_dispatch_authorized':False,
        'models_started':False,'test_suite_started':False,'automatic_queue':False,'research_admitted':False,
        'sources_sha256':{n:digest(packet/n) for n in m.SOURCES},'inputs_sha256':digest(packet/'INPUT.json')}
    pin=m.canonical(op);m.validate_operation(op,pin);(packet/'RECONCILIATION.json').write_bytes(m.json_bytes(op))
    expected=m.records(op,data);(raw/'expected-records.json').write_bytes(m.json_bytes({n:{'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()} for n,b in expected.items()}))
    sys.path.insert(0,str(ROOT));from models.pilot_bundle import make_archive,extract_verified
    archive=raw/'reviewed-operator.tar.gz';manifest=make_archive(archive,{m.PREFIX+'/'+str(p.relative_to(packet)):p for p in packet.rglob('*') if p.is_file()})
    (raw/'operator.manifest.json').write_bytes(m.json_bytes(manifest));extract_verified(archive,manifest,raw/'extracted')
    payload={'kind':'reviewed-panel-reconciliation-upload-v1','operation_sha256':pin,'archive_base64':base64.b64encode(archive.read_bytes()).decode(),'manifest':manifest}
    (raw/'upload-payload.json').write_bytes(m.json_bytes(payload))
    installer=(ROOT/'scripts/install_two_host_packet.py').read_text()
    installer=installer.replace("socket.gethostname() not in ('linux10609','linux10603')","socket.gethostname() != 'linux10609'")
    installer=installer.replace('matched-panel-two-host-source-upload-v1','reviewed-panel-reconciliation-upload-v1').replace('two-host-execution-20261026',m.PREFIX)
    installer=installer.replace('check=True,timeout=30)','check=True,timeout=30,stdout=sys.stderr)')
    installer=installer.replace("print('Shared two-host packet installed/verified. No GPU job started.',flush=True)",
        "print('Metadata packet installed/verified; rechecking saved cases on CPU.',file=sys.stderr,flush=True)\n    packet=target/'unpacked/"+m.PREFIX+"'\n    os.execv(sys.executable,[sys.executable,'-B',str(packet/'scripts/reconcile_reviewed_panel.py'),'--operation-sha256',payload['operation_sha256']])")
    (raw/'install-and-reconcile.py').write_text(installer)
    encoded=base64.b64encode(installer.encode()).decode();body='exec(__import__("base64").b64decode('+repr(encoded)+'))'
    command=shlex.join(['timeout','--signal=TERM','--kill-after=15s','300','env','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',
        '/home/kdu/.venvs/vjepa/bin/python','-B','-c',body,digest(raw/'upload-payload.json')])
    (raw/'reconcile-command.txt').write_text(command+'\n')
    shutil.copyfile(ROOT/'runs/pretrained/diagnostic/two-host-panel-20261026/known_hosts',raw/'known_hosts')
    verifier='''from pathlib import Path
import hashlib,importlib.util,json,sys
base=Path(__file__).resolve().parent;output=Path(sys.argv[1]);packet=base/'extracted/'''+m.PREFIX+''''
s=importlib.util.spec_from_file_location('receipt_metadata_verifier',packet/'models/reviewed_panel.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
op=json.loads((packet/'RECONCILIATION.json').read_text());m.validate_operation(op,'''+repr(pin)+''')
for n,p in op['sources_sha256'].items():assert hashlib.sha256((packet/n).read_bytes()).hexdigest()==p,n
assert hashlib.sha256((packet/'INPUT.json').read_bytes()).hexdigest()==op['inputs_sha256']
data=json.loads((packet/'INPUT.json').read_text());receipt=json.loads((output/'receipt.json.part').read_text())
result=m.verify_receipt(receipt,op,data);result['receipt_file_sha256']=hashlib.sha256((output/'receipt.json.part').read_bytes()).hexdigest()
with (output/'receipt-verification.json').open('x') as f:f.write(json.dumps(result,sort_keys=True,indent=2)+'\\n')
(output/'receipt.json.part').rename(output/'receipt.json')
print(json.dumps(result,sort_keys=True,indent=2))
'''
    (raw/'verify-receipt.py').write_text(verifier)
    pins={n:digest(raw/n) for n in ['upload-payload.json','reconcile-command.txt','verify-receipt.py','known_hosts']}
    pins.update({'extracted/'+m.PREFIX+'/'+n:digest(packet/n) for n in (*m.SOURCES,'INPUT.json','RECONCILIATION.json')})
    helper='''#!/usr/bin/env bash
set -euo pipefail
base='''+shlex.quote(str(raw))+'''
if [[ "${REVIEWED_PANEL_AWAKE:-0}" != 1 ]]; then
 exec /usr/bin/caffeinate -i env REVIEWED_PANEL_AWAKE=1 bash "$0"
fi
py='''+shlex.quote(str(ROOT/'.venv/bin/python'))+'''
PYTHONDONTWRITEBYTECODE=1 "$py" - "$base" <<'PIN_CHECK'
from pathlib import Path
import hashlib,sys
p=Path(sys.argv[1]);expected='''+repr(pins)+'''
for n,s in expected.items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==s,n
print('Metadata only: recheck eleven saved cases, append a paused progress view, retain all legacy records and STOP.')
PIN_CHECK
output=$(mktemp -d "$base/reconcile-session-$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
printf '%s\\n' "$output" > "$base/latest-session.txt"
command=$(cat "$base/reconcile-command.txt")
set +e
ssh -T -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$base/known_hosts" -o GlobalKnownHostsFile=/dev/null \\
 -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 kdu@linux10609.dc.engr.scu.edu "$command" \\
 < "$base/upload-payload.json" > "$output/receipt.json.part" 2> "$output/reconcile.stderr"
code=$?
set -e
printf '%s\\n' "$code" > "$output/connection-exit-status.txt"
cat "$output/reconcile.stderr"
if [[ "$code" -ne 0 ]]; then
 echo "Metadata reconciliation did not confirm success. Preserve $output and report the ending before repeating." >&2
 exit "$code"
fi
PYTHONDONTWRITEBYTECODE=1 "$py" "$base/verify-receipt.py" "$output"
echo "Receipt saved at $output. Reply Done for independent review. No GPU case started."
'''
    (raw/'reconcile-reviewed.sh').write_text(helper)
    prepared={'kind':m.KIND+'-preparation','operation_sha256':pin,'inputs_sha256':op['inputs_sha256'],'source_sha256':op['sources_sha256'],
        'archive':manifest,'payload_sha256':digest(raw/'upload-payload.json'),'payload_bytes':(raw/'upload-payload.json').stat().st_size,
        'helper_sha256':digest(raw/'reconcile-reviewed.sh'),'preserved_files':len(data['preserved_files_sha256']),
        'new_metadata_records':len(expected),'effective_accepted_case_count':11,'legacy_accepted_case_count':9,
        'STOP_to_remain_active':True,'remote_operation_performed':False,'models_started':False,'research_admitted':False}
    (raw/'preparation.json').write_bytes(m.json_bytes(prepared))
    for p in raw.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
    for p in raw.rglob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
    return raw,prepared


if __name__=='__main__':
    raw,result=build();print(json.dumps({k:v for k,v in result.items() if k!='archive'},sort_keys=True,indent=2))
