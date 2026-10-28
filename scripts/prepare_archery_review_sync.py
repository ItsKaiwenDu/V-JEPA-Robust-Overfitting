#!/usr/bin/env python3
"""Build a small source/metadata-only handoff for the original approved Archery recovery review."""
import ast
import base64
import importlib.util
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def derive(repo):
    raw=repo/'runs/pretrained/diagnostic/archery-recovery-20261027';packet=raw/'extracted/archery-recovery-20261027'
    core=module('archery_sync_derived_core',packet/'models/archery_recovery.py')
    assert core.digest(packet/'models/archery_recovery.py')=='8e29c1931c1ecb9734fa31a969be0524efe2545f08df15127bc7eb73f7db4215'
    c=core.trusted(packet,'1757892c83921e0cee8391b2468989680bcb44c57b6fa686f3528de9142eb4b5',repo/'runs/pretrained/diagnostic/matched-panel-handoff-20261024/snapshot',repo/'runs/pretrained/diagnostic/two-host-panel-20261026/extracted/two-host-execution-20261026')
    activation=Path((raw/'latest-activation.txt').read_text().strip());receipt=activation/'receipt.json'
    assert core.digest(receipt)=='5417f0b5db0cff2306a646a8c40e4d3e8642134abca315ec8aba7da856ec1a52'
    activation_texts=json.loads(receipt.read_text())['record_texts']
    d=Path((raw/'linux10603/latest-download.txt').read_text().strip());down=d/'downloaded';outer=d/'independently-reviewed'
    audit=json.loads((d/'independent-verification.json').read_text());assert audit['verified'] and audit['case_id']=='clip-06-bank-1'
    assert audit['actual_campus_CPU_JUnit']=={'tests':172,'failures':0,'errors':0,'skipped':0}
    pins=core.ancestor_pins(c);sources={};metadata={};cid='clip-06-bank-1';case_root=core.TARGET+'/cases/'+cid+'/recovery-01/'
    def add(source,name):
        assert source.is_file() and not source.is_symlink();h=core.digest(source)
        if name in pins:assert pins[name]==h
        pins[name]=h;sources[name]=str(source)
    for n,text in activation_texts.items():
        name=core.TARGET+'/'+n;value=hashlib.sha256(text.encode()).hexdigest()
        if name in pins:assert pins[name]==value
        pins[name]=value;metadata[n]=text
    for p in packet.rglob('*'):
        if p.is_file():add(p,core.PREFIX+'/unpacked/'+core.PREFIX+'/'+str(p.relative_to(packet)))
    for p in (outer/'numeric-reviewed').rglob('*'):
        name=str(p.relative_to(outer/'numeric-reviewed'))
        if p.is_file() and name!='PANEL-HANDOFF.json' and not name.startswith('source/'):add(p,case_root+name)
    for p in (outer/'hardware').glob('*.json'):add(p,case_root+p.name)
    for n in ('case-envelope.json','preservation-before-export.json','preservation-after-export.json'):add(down/'case'/n,case_root+n)
    for n in ('numeric-evidence.tar.gz','numeric-evidence.tar.gz.manifest.json'):add(outer/'numeric'/n,case_root+n)
    for n in ('continuation-evidence.tar.gz','continuation-evidence.tar.gz.manifest.json'):add(down/'evidence'/n,case_root+n)
    for src,n in [('claim','claims/'+cid+'/claim.json'),('outcome','outcomes/'+cid+'.json'),('export','exports/'+cid+'.json')]:
        p=down/'ledger'/(src+'.json');add(p,core.TARGET+'/'+n);metadata[n]=p.read_text()
    launcher_files={}
    for p in (down/'launcher').iterdir():
        name=core.PREFIX+'/unpacked/'+core.PREFIX+'/launchers/linux10603-'+cid+'/'+p.name
        add(p,name);launcher_files[p.name]=p.read_text()
    marker=outer/'independent-review.json';assert core.digest(marker)==audit['review_marker_file_sha256']=='ba9adb7e31588dd85a697436c637a3c602997fb4f6c701957ee775af0c9f401a'
    account=d/'attempt-accounting.json';assert json.loads(account.read_text())==audit['accounting']
    data={'kind':'matched-panel-archery-recovery-review-sync-v1-input','core_operation_sha256':core.canonical(c['op']),
        'actual_activation_receipt_file_sha256':core.digest(receipt),'review':{'text':marker.read_text(),'sha256':core.digest(marker)},
        'preserved_files_sha256':pins,'accounting':{'text':account.read_text(),'sha256':core.digest(account)},
        'models_started':False,'research_admitted':False}
    fixture={'case_envelope':json.loads((down/'case/case-envelope.json').read_text())['payload'],'input':data,'core_operation':c['op'],'core_data':c['data'],'activation_record_texts':activation_texts,'case_metadata':metadata,'launcher_files':launcher_files}
    return data,fixture,raw,packet,activation

def build():
    sys.path.insert(0, str(ROOT))
    from models.pilot_bundle import make_archive, extract_verified
    m = module('sync_prepared_module', ROOT / 'models/archery_review_sync.py')
    data, fixture, core_raw, core_packet, activation = derive(ROOT)
    m.validate_input(data)
    assert json.loads((ROOT / 'tests/fixtures/archery_review_sync_metadata.json').read_text()) == fixture
    raw = ROOT / 'runs/pretrained/diagnostic' / m.PREFIX
    raw.mkdir(exist_ok=False)
    (raw / 'validation').mkdir()
    packet = raw / 'staged' / m.PREFIX
    packet.mkdir(parents=True)
    for name in m.SOURCES:
        source = ROOT / name if not name.startswith('support/') else core_packet / 'models/archery_recovery.py'
        target = packet / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(packet / 'INPUT.json', data)
    shutil.copyfile(activation / 'receipt.json', packet / 'activation-receipt.json')
    op = dict(m.policy(), date_Pacific='2026-10-27', inputs_file_sha256=m.digest(packet / 'INPUT.json'),
        sources_sha256={name: m.digest(packet / name) for name in m.SOURCES})
    pin = m.canonical(op)
    m.validate_operation(op, pin)
    write_json(packet / 'SYNC.json', op)
    archive = raw / 'sync-operator.tar.gz'
    files = {m.PREFIX + '/' + str(p.relative_to(packet)): p for p in packet.rglob('*') if p.is_file()}
    manifest = make_archive(archive, files)
    write_json(raw / 'operator.manifest.json', manifest)
    extract_verified(archive, manifest, raw / 'extracted')
    payload = {'kind': 'archery-recovery-review-sync-upload-v1', 'operation_sha256': pin,
        'archive_base64': base64.b64encode(archive.read_bytes()).decode(), 'manifest': manifest}
    write_json(raw / 'upload-payload.json', payload)
    installer = (ROOT / 'runs/pretrained/diagnostic/yoyo-review-sync-20261027/install-and-sync.py').read_text()
    installer = installer.replace('yoyo-review-sync-20261027', m.PREFIX)
    installer = installer.replace('yoyo-single-review-sync-upload-v1', 'archery-recovery-review-sync-upload-v1')
    installer = installer.replace('scripts/sync_yoyo_review.py', 'scripts/sync_archery_review.py')
    installer = installer.replace('YoYo review-sync packet verified; recheck saved evidence on CPU. Archery remains unclaimed; no model or test suite starts.',
        'Archery recovery review packet verified; recheck saved evidence on CPU and append its original review. No model or test suite starts.')
    (raw / 'install-and-sync.py').write_text(installer)
    code = 'exec(__import__("base64").b64decode(' + repr(base64.b64encode(installer.encode()).decode()) + '))'
    command = shlex.join(['timeout', '--signal=TERM', '--kill-after=15s', '300', 'env',
        'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
        '/home/kdu/.venvs/vjepa/bin/python', '-B', '-c', code, m.digest(raw / 'upload-payload.json')])
    (raw / 'sync-command.txt').write_text(command + '\n')
    shutil.copyfile(core_raw / 'known_hosts', raw / 'known_hosts')
    verify = '''from pathlib import Path
import hashlib,importlib.util,json,sys
base=Path(__file__).resolve().parent;output=Path(sys.argv[1]);packet=base/'extracted/'''+m.PREFIX+"'\n"+'''
s=importlib.util.spec_from_file_location('actual_sync_receipt_check',packet/'models/archery_review_sync.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
op=json.loads((packet/'SYNC.json').read_text());m.validate_operation(op,'''+repr(pin)+''')
for n,p in op['sources_sha256'].items():assert hashlib.sha256((packet/n).read_bytes()).hexdigest()==p,n
assert hashlib.sha256((packet/'INPUT.json').read_bytes()).hexdigest()==op['inputs_file_sha256']
data=json.loads((packet/'INPUT.json').read_text());m.validate_input(data)
r=json.loads((output/'receipt.json.part').read_text());v=m.verify_receipt(r,op,data)
v['receipt_file_sha256']=hashlib.sha256((output/'receipt.json.part').read_bytes()).hexdigest()
with (output/'receipt-verification.json').open('x') as f:f.write(json.dumps(v,sort_keys=True,indent=2)+'\\n')
(output/'receipt.json.part').rename(output/'receipt.json')
print(json.dumps(v,sort_keys=True,indent=2))
'''
    (raw / 'verify-receipt.py').write_text(verify)
    pins = {name: m.digest(raw / name) for name in ('upload-payload.json', 'sync-command.txt', 'verify-receipt.py', 'known_hosts')}
    pins.update({'extracted/' + m.PREFIX + '/' + str(p.relative_to(packet)): m.digest(p) for p in packet.rglob('*') if p.is_file()})
    helper = '''#!/usr/bin/env bash
set -euo pipefail
base='''+shlex.quote(str(raw))+'''
if [[ "${REVIEWED_SYNC_AWAKE:-0}" != 1 ]]; then
 exec /usr/bin/caffeinate -i env REVIEWED_SYNC_AWAKE=1 bash "$0"
fi
py='''+shlex.quote(str(ROOT / '.venv/bin/python'))+'''
PYTHONDONTWRITEBYTECODE=1 "$py" - "$base" <<'PIN_CHECK'
from pathlib import Path
import hashlib,sys
p=Path(sys.argv[1]);expected='''+repr(pins)+'''
for n,s in expected.items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==s,n
print('Register the original Archery recovery review after CPU verification. Preserve578 files, failed attempts and consumed grant; no model or test suite starts.')
PIN_CHECK
output=$(mktemp -d "$base/sync-session-$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
printf '%s\\n' "$output" > "$base/latest-session.txt"
command=$(cat "$base/sync-command.txt")
set +e
ssh -T -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$base/known_hosts" -o GlobalKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \\
 kdu@linux10609.dc.engr.scu.edu "$command" < "$base/upload-payload.json" > "$output/receipt.json.part" 2> "$output/sync.stderr"
code=$?
set -e
printf '%s\\n' "$code" > "$output/connection-exit-status.txt"
cat "$output/sync.stderr"
if [[ "$code" -ne 0 ]]; then
 echo "Review sync did not confirm success. Preserve $output and report the ending before repeating." >&2
 exit "$code"
fi
PYTHONDONTWRITEBYTECODE=1 "$py" "$base/verify-receipt.py" "$output"
echo "Fifteen-case receipt saved at $output. Reply Done for independent review. No next GPU case starts."
'''
    (raw / 'sync-reviews.sh').write_text(helper)
    prepared = {'kind': m.KIND + '-preparation', 'operation_sha256': pin,
        'source_sha256': op['sources_sha256'], 'inputs_file_sha256': op['inputs_file_sha256'],
        'archive': manifest, 'payload_sha256': m.digest(raw / 'upload-payload.json'),
        'payload_bytes': (raw / 'upload-payload.json').stat().st_size,
        'helper_sha256': m.digest(raw / 'sync-reviews.sh'), 'records_to_append': [m.TARGET],
        'preserved_files': m.PRESERVED, 'accepted_after_sync': 15, 'complete_clip_pairs': 7,
        'consumed_grant_id':m.GRANT,'previous_charged_seconds':2348,'recovery_charged_seconds':2599,'total_charged_seconds':4947,
        'remote_operation_performed': False, 'new_GPU_jobs_started': 0}
    write_json(raw / 'preparation.json', prepared)
    for p in raw.rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    subprocess.run(['bash', '-n', str(raw / 'sync-reviews.sh')], check=True)
    return raw, prepared


if __name__ == '__main__':
    raw, result = build()
    print(json.dumps({k: v for k, v in result.items() if k != 'archive'}, sort_keys=True, indent=2))
