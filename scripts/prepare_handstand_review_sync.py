#!/usr/bin/env python3
"""Build a small source/metadata-only handoff for the one original HandStandPushups reviews."""
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


from datetime import datetime
from decimal import Decimal, ROUND_CEILING

def derive(repo):
 raw=repo/'runs/pretrained/diagnostic/bowling-handstand-pair-20261028';packet=raw/'extracted/bowling-handstand-pair-20261028'
 core=module('derive_completed_handstand',packet/'models/reviewed_panel_final_pair.py')
 assert core.digest(packet/'models/reviewed_panel_final_pair.py')=='6d435b42a87c51c3992ea022ebdb4e366cb3cb7c021e1213c2b15cfa2b8372ad'
 c=core.trusted(packet,'b899e6d9b2e6824804a035807ff61ea3517182fbbaabf7910d2dada5a34c82e1',repo/'runs/pretrained/diagnostic/matched-panel-handoff-20261024/snapshot',repo/'runs/pretrained/diagnostic/two-host-panel-20261026/extracted/two-host-execution-20261026')
 activation=Path((raw/'latest-activation.txt').read_text().strip());receipt=activation/'receipt.json'
 assert core.digest(receipt)=='c78105922962b3e4aff228bf135ef7c5df1514a6c35d38572eaa0fb20ad43d83'
 activation_texts=json.loads(receipt.read_text())['record_texts']
 audit=json.loads((repo/'report/checks/2026-10-28-handstand-bank0-verification.json').read_text())
 assert audit['verified'] and audit['independently_verified_case_count']==18
 pins=core.ancestor_pins(c);sources={};metadata={};clocks={};reviews={};launchers={};envelopes={}
 def add(source,name):
  assert source.is_file() and not source.is_symlink();h=core.digest(source)
  if name in pins:assert pins[name]==h,name
  pins[name]=h;sources[name]=str(source)
 for n,text in activation_texts.items():
  name=core.TARGET+'/'+n;h=hashlib.sha256(text.encode()).hexdigest()
  if name in pins:assert pins[name]==h
  pins[name]=h;metadata[n]=text
 for n in (*core.SOURCES,'CONTINUATION.json','INPUT.json','prior-INPUT.json','prior-OPERATION.json','prior-receipt.json'):
  add(packet/n,core.PREFIX+'/unpacked/'+core.PREFIX+'/'+n)
 host,cid='linux10609','clip-09-bank-0'
 assert audit['case_id']==cid and audit['host']==host and audit['campus_CPU_tests']=={'tests':177,'failures':0,'errors':0,'skipped':0}
 d=repo/audit['download_directory'];down=d/'downloaded';outer=d/'independently-reviewed'
 case_root=core.TARGET+'/cases/'+cid+'/attempt-01/'
 for p in (outer/'numeric-reviewed').rglob('*'):
  n=str(p.relative_to(outer/'numeric-reviewed'))
  if p.is_file() and n!='PANEL-HANDOFF.json' and not n.startswith('source/'):add(p,case_root+n)
 for p in (outer/'hardware').glob('*.json'):add(p,case_root+p.name)
 for n in ('case-envelope.json','preservation-before-export.json','preservation-after-export.json'):add(down/'case'/n,case_root+n)
 for n in ('numeric-evidence.tar.gz','numeric-evidence.tar.gz.manifest.json'):add(outer/'numeric'/n,case_root+n)
 for n in ('continuation-evidence.tar.gz','continuation-evidence.tar.gz.manifest.json'):add(down/'evidence'/n,case_root+n)
 for src,n in [('claim','claims/'+cid+'/claim.json'),('outcome','outcomes/'+cid+'.json'),('export','exports/'+cid+'.json')]:
  p=down/'ledger'/(src+'.json');add(p,core.TARGET+'/'+n);metadata[n]=p.read_text()
 launchers[host]={}
 for p in (down/'launcher').iterdir():
  n=core.PREFIX+'/unpacked/'+core.PREFIX+'/launchers/'+host+'-'+cid+'/'+p.name
  add(p,n);launchers[host][p.name]=p.read_text()
 marker=outer/'independent-review.json';assert core.digest(marker)==audit['review_marker_file_sha256']
 reviews[cid]={'text':marker.read_text(),'sha256':core.digest(marker)}
 from datetime import datetime
 from decimal import Decimal,ROUND_CEILING
 start=Decimal(launchers[host]['started-epoch.txt'].strip())
 shown=datetime.fromisoformat(launchers[host]['started-utc.txt'].strip().replace('Z','+00:00'))
 end=datetime.fromisoformat(launchers[host]['ended-utc.txt'].strip().replace('Z','+00:00'))
 charge=int((Decimal(int(end.timestamp()))+1-start).to_integral_value(rounding=ROUND_CEILING))
 assert charge==audit['whole_launcher_seconds_conservative_upper_bound']==2951
 clocks[cid]={'host':host,'attempt':'attempt-01','started_epoch_decimal':str(start),'started_UTC_floor':shown.isoformat(),
  'ended_UTC_floor':end.isoformat(),'conservative_charge_upper_seconds':charge,'original_allowance_seconds':4800,
  'kill_grace_seconds':15,'additional_attempt_authorized':False,'unused_time_authorizes_no_retry':True}
 envelopes[cid]=json.loads((down/'case/case-envelope.json').read_text())['payload']
 stopped=json.loads((repo/'report/checks/2026-10-28-bowling-startup-verification.json').read_text())
 assert stopped['verified'] and stopped['conservative_startup_charge_seconds']==206 and stopped['remaining_original_allowance_seconds']==4594
 startup=Path((repo/'runs/pretrained/diagnostic/bowling-guard-refusal-20261028/latest-download.txt').read_text().strip())
 assert core.digest(startup/'transport.tar')==stopped['transport']['sha256']=='b8011463210c3764a960e6fbaf0130d58b8f707316c43774c94122140d60aaa0'
 launchers['linux10603']={}
 for p in (startup/'downloaded/launcher').iterdir():
  n=core.PREFIX+'/unpacked/'+core.PREFIX+'/launchers/linux10603-clip-08-bank-1/'+p.name
  add(p,n);launchers['linux10603'][p.name]=p.read_text()
 data={'kind':'matched-panel-handstand-review-sync-v1-input','core_operation_sha256':core.canonical(c['op']),
  'actual_activation_receipt_file_sha256':core.digest(receipt),'reviews':reviews,'clocks':clocks,
  'previous_consumed_history':c['sync_receipt']['time_after'],'bowling_startup_accounting':stopped['accounting'],
  'bowling_capture_transport_sha256':stopped['transport']['sha256'],'preserved_files_sha256':pins,
  'models_started':False,'research_admitted':False}
 fixture={'input':data,'core_operation':c['op'],'core_data':c['data'],
  'activation_record_texts':activation_texts,'case_metadata':metadata,'launcher_files':launchers,'case_envelopes':envelopes}
 return data,fixture,raw,packet,activation,sources


def build():
    sys.path.insert(0, str(ROOT))
    from models.pilot_bundle import make_archive, extract_verified
    m = module('sync_prepared_module', ROOT / 'models/handstand_review_sync.py')
    data, fixture, core_raw, core_packet, activation, sources = derive(ROOT)
    m.validate_input(data)
    assert json.loads((ROOT / 'tests/fixtures/handstand_review_sync_metadata.json').read_text()) == fixture
    raw = ROOT / 'runs/pretrained/diagnostic' / m.PREFIX
    raw.mkdir(exist_ok=False)
    (raw / 'validation').mkdir()
    write_json(raw / 'local-source-map.json', sources)
    packet = raw / 'staged' / m.PREFIX
    packet.mkdir(parents=True)
    for name in m.SOURCES:
        source = ROOT / name if not name.startswith('support/') else core_packet / 'models/reviewed_panel_final_pair.py'
        target = packet / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(packet / 'INPUT.json', data)
    shutil.copyfile(activation / 'receipt.json', packet / 'activation-receipt.json')
    op = dict(m.policy(), date_Pacific='2026-10-28', inputs_file_sha256=m.digest(packet / 'INPUT.json'),
        sources_sha256={name: m.digest(packet / name) for name in m.SOURCES})
    pin = m.canonical(op)
    m.validate_operation(op, pin)
    write_json(packet / 'SYNC.json', op)
    archive = raw / 'sync-operator.tar.gz'
    files = {m.PREFIX + '/' + str(p.relative_to(packet)): p for p in packet.rglob('*') if p.is_file()}
    manifest = make_archive(archive, files)
    write_json(raw / 'operator.manifest.json', manifest)
    extract_verified(archive, manifest, raw / 'extracted')
    payload = {'kind': 'handstand-review-sync-upload-v1', 'operation_sha256': pin,
        'archive_base64': base64.b64encode(archive.read_bytes()).decode(), 'manifest': manifest}
    write_json(raw / 'upload-payload.json', payload)
    installer = '"""Idempotently install one verified shared-home packet; no job starts here."""\nimport base64\nimport hashlib\nimport json\nimport os\nfrom pathlib import Path\nimport socket\nimport subprocess\nimport sys\nimport time\n\ndef digest(path):\n    with path.open(\'rb\') as stream:return hashlib.file_digest(stream,\'sha256\').hexdigest()\n\ndef main():\n    expected=sys.argv[1]\n    if socket.gethostname() != \'linux10609\':raise ValueError(\'Wrong university host.\')\n    body=sys.stdin.buffer.read(1024*1024+1)\n    if len(body)>1024*1024 or hashlib.sha256(body).hexdigest()!=expected:raise ValueError(\'Upload changed.\')\n    payload=json.loads(body)\n    if payload[\'kind\']!=\'archery-recovery-review-sync-upload-v1\':raise ValueError(\'Wrong source upload.\')\n    archive=base64.b64decode(payload[\'archive_base64\'],validate=True);manifest=payload[\'manifest\']\n    if len(archive)!=manifest[\'bytes\'] or hashlib.sha256(archive).hexdigest()!=manifest[\'sha256\']:raise ValueError(\'Archive changed.\')\n    transfer=Path.home()/\'Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024\'\n    extractor=transfer/\'verify_extract.py\'\n    if digest(extractor)!=\'efe7ae73e955eca1660e66920f1cc78ccad7ef42ede71933dddbce661008bda1\':raise ValueError(\'Extractor changed.\')\n    os.umask(0o077);target=transfer/\'archery-review-sync-20261027\'\n    try:target.mkdir(exist_ok=False);owner=True\n    except FileExistsError:owner=False\n    if owner:\n        (target/\'operator.tar.gz\').write_bytes(archive)\n        (target/\'operator.manifest.json\').write_text(json.dumps(manifest,sort_keys=True,indent=2)+\'\\n\')\n        subprocess.run([sys.executable,\'-B\',str(extractor),\'--archive\',str(target/\'operator.tar.gz\'),\n            \'--manifest\',str(target/\'operator.manifest.json\'),\'--destination\',str(target/\'unpacked\')],check=True,timeout=30,stdout=sys.stderr)\n        with (target/\'.READY.tmp\').open(\'x\') as stream:\n            stream.write(manifest[\'sha256\']+\'\\n\');stream.flush();os.fsync(stream.fileno())\n        os.replace(target/\'.READY.tmp\',target/\'READY\')\n    else:\n        for _ in range(300):\n            if (target/\'READY\').is_file():break\n            time.sleep(.1)\n        else:raise ValueError(\'An incomplete shared installation exists; preserve it for inspection.\')\n    if target.is_symlink() or (target/\'READY\').read_text().strip()!=manifest[\'sha256\']:\n        raise ValueError(\'Existing installation identity differs.\')\n    if digest(target/\'operator.tar.gz\')!=manifest[\'sha256\']:raise ValueError(\'Stored upload changed.\')\n    for name,entry in manifest[\'files\'].items():\n        relative=Path(name);path=target/\'unpacked\'/relative\n        if (relative.is_absolute() or \'..\' in relative.parts or path.is_symlink()\n            or not path.resolve().is_relative_to((target/\'unpacked\').resolve())\n            or path.stat().st_size!=entry[\'bytes\'] or digest(path)!=entry[\'sha256\']):\n            raise ValueError(\'Shared packet file changed: \'+name)\n    print(\'Archery recovery review packet verified; recheck saved evidence on CPU and append its original review. No model or test suite starts.\',file=sys.stderr,flush=True)\n    packet=target/\'unpacked/archery-review-sync-20261027\'\n    os.execv(sys.executable,[sys.executable,\'-B\',str(packet/\'scripts/sync_archery_review.py\'),\'--operation-sha256\',payload[\'operation_sha256\']])\n\nif __name__==\'__main__\':main()\n'
    installer = installer.replace('archery-review-sync-20261027', m.PREFIX)
    installer = installer.replace('archery-recovery-review-sync-upload-v1', 'handstand-review-sync-upload-v1')
    installer = installer.replace('scripts/sync_archery_review.py', 'scripts/sync_handstand_review.py')
    installer = installer.replace('Archery recovery review packet verified; recheck saved evidence on CPU and append its original review. No model or test suite starts.',
        'HandStandPushups review packet verified; recheck the saved case on CPU and append only its original review. No model or test suite starts.')
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
s=importlib.util.spec_from_file_location('actual_sync_receipt_check',packet/'models/handstand_review_sync.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
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
print('Register the original review after CPU verification. Preserve918 files and consumed attempts; no model or test suite starts.')
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
echo "Eighteen-case receipt saved at $output. Reply Done for independent review. No next GPU case starts."
'''
    (raw / 'sync-reviews.sh').write_text(helper)
    prepared = {'kind': m.KIND + '-preparation', 'operation_sha256': pin,
        'source_sha256': op['sources_sha256'], 'inputs_file_sha256': op['inputs_file_sha256'],
        'archive': manifest, 'payload_sha256': m.digest(raw / 'upload-payload.json'),
        'payload_bytes': (raw / 'upload-payload.json').stat().st_size,
        'helper_sha256': m.digest(raw / 'sync-reviews.sh'), 'records_to_append': sorted(m.TARGETS),
        'preserved_files': m.PRESERVED, 'accepted_after_sync': 18, 'complete_clip_pairs': 8,
        'original_attempt_charge_upper_seconds':m.CHARGES,'previous_consumed_history_seconds':4947,'bowling_startup_charge_seconds':206,'bowling_remaining_original_allowance_seconds':4594,
        'remote_operation_performed': False, 'new_GPU_jobs_started': 0}
    write_json(raw / 'preparation.json', prepared)
    for p in raw.rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    subprocess.run(['bash', '-n', str(raw / 'sync-reviews.sh')], check=True)
    return raw, prepared


if __name__ == '__main__':
    raw, result = build()
    print(json.dumps({k: v for k, v in result.items() if k != 'archive'}, sort_keys=True, indent=2))
