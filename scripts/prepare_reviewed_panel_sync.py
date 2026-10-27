#!/usr/bin/env python3
"""Build a small source/metadata-only handoff for the two original review markers."""
import ast
import base64
import importlib.util
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
    repo = Path(repo)
    raw = repo / 'runs/pretrained/diagnostic/reviewed-continuation-20261026'
    packet = raw / 'extracted/reviewed-continuation-20261026'
    core = module('sync_prepared_core', packet / 'models/reviewed_panel_continuation.py')
    assert core.digest(packet / 'models/reviewed_panel_continuation.py') == '51f78c2b4ff4904f4492e4ff055d58eb7f219a4a3b1f440dd502c43725c71b96'
    scientific = repo / 'runs/pretrained/diagnostic/matched-panel-handoff-20261024/snapshot'
    reference = repo / 'runs/pretrained/diagnostic/two-host-panel-20261026/extracted/two-host-execution-20261026'
    context = core.trusted(packet, '0714e035422d4e21ba41c37734997138aa3249abb0966cf8b8324b6c96ac7300', scientific, reference)
    activation = Path((raw / 'latest-activation.txt').read_text().strip())
    assert core.digest(activation / 'receipt.json') == '32acded6836409b294caf3d57f562e1b1d7055c27574d4ad99d9ba612cce0dce'
    activation_receipt = json.loads((activation / 'receipt.json').read_text())
    pins = core.ancestor_pins(context)
    for name, text in activation_receipt['record_texts'].items():
        pins[core.TARGET + '/' + name] = __import__('hashlib').sha256(text.encode()).hexdigest()
    for path in packet.rglob('*'):
        if path.is_file():
            pins[core.PREFIX + '/unpacked/' + core.PREFIX + '/' + str(path.relative_to(packet))] = core.digest(path)
    reviews = {}
    case_metadata = {}
    for host, cid in core.TARGETS.items():
        d = Path((raw / host / 'latest-download.txt').read_text().strip())
        down = d / 'downloaded'
        outer = d / 'independently-reviewed'
        audit = json.loads((d / 'independent-verification.json').read_text())
        assert audit['verified'] and audit['case_id'] == cid and audit['campus_CPU_tests']['tests'] == 143
        case_root = core.TARGET + '/cases/' + cid + '/attempt-01/'
        for p in (outer / 'numeric-reviewed').rglob('*'):
            if not p.is_file():
                continue
            name = str(p.relative_to(outer / 'numeric-reviewed'))
            if name != 'PANEL-HANDOFF.json' and not name.startswith('source/'):
                pins[case_root + name] = core.digest(p)
        pins[case_root + 'case-envelope.json'] = core.digest(down / 'case/case-envelope.json')
        for path in (outer / 'hardware').glob('*.json'):
            pins[case_root + path.name] = core.digest(path)
        for name in ('preservation-before-export.json', 'preservation-after-export.json'):
            pins[case_root + name] = core.digest(down / 'case' / name)
        for name in ('numeric-evidence.tar.gz', 'numeric-evidence.tar.gz.manifest.json'):
            pins[case_root + name] = core.digest(outer / 'numeric' / name)
        for name in ('continuation-evidence.tar.gz', 'continuation-evidence.tar.gz.manifest.json'):
            pins[case_root + name] = core.digest(down / 'evidence' / name)
        live_names = {'claim': 'claims/' + cid + '/claim.json', 'outcome': 'outcomes/' + cid + '.json',
                      'export': 'exports/' + cid + '.json'}
        case_metadata[cid] = {}
        for name, relative in live_names.items():
            source = down / 'ledger' / (name + '.json')
            pins[core.TARGET + '/' + relative] = core.digest(source)
            case_metadata[cid][relative] = source.read_text()
        for path in (down / 'launcher').iterdir():
            assert path.is_file() and not path.is_symlink()
            pins[core.PREFIX + '/unpacked/' + core.PREFIX + '/launchers/' + host + '-' + cid + '/' + path.name] = core.digest(path)
        marker = outer / 'independent-review.json'
        assert core.digest(marker) == audit['review_marker_file_sha256']
        reviews[cid] = {'text': marker.read_text(), 'sha256': core.digest(marker)}
    assert len(pins) == 275, len(pins)
    data = {'kind': 'matched-panel-reviewed-case-sync-v1-input',
        'core_operation_sha256': core.canonical(context['op']),
        'actual_activation_receipt_file_sha256': core.digest(activation / 'receipt.json'),
        'reviews': reviews, 'preserved_files_sha256': pins, 'models_started': False, 'research_admitted': False}
    fixture = {'input': data, 'core_operation': context['op'], 'core_data': context['data'],
        'activation_record_texts': activation_receipt['record_texts'], 'case_metadata': case_metadata}
    return data, fixture, raw, packet, activation


def build():
    sys.path.insert(0, str(ROOT))
    from models.pilot_bundle import make_archive, extract_verified
    m = module('sync_prepared_module', ROOT / 'models/reviewed_panel_sync.py')
    data, fixture, core_raw, core_packet, activation = derive(ROOT)
    m.validate_input(data)
    assert json.loads((ROOT / 'tests/fixtures/reviewed_panel_sync_metadata.json').read_text()) == fixture
    raw = ROOT / 'runs/pretrained/diagnostic' / m.PREFIX
    raw.mkdir(exist_ok=False)
    (raw / 'validation').mkdir()
    packet = raw / 'staged' / m.PREFIX
    packet.mkdir(parents=True)
    for name in m.SOURCES:
        source = ROOT / name if not name.startswith('support/') else core_packet / 'models/reviewed_panel_continuation.py'
        target = packet / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(packet / 'INPUT.json', data)
    shutil.copyfile(activation / 'receipt.json', packet / 'activation-receipt.json')
    op = {'kind': m.KIND, 'host': 'linux10609', 'date_Pacific': '2026-10-27',
        'core_operation_sha256': m.CORE_PIN, 'core_module_sha256': m.CORE_SOURCE,
        'actual_activation_receipt_file_sha256': m.ACTIVATION_RECEIPT, 'coordinator_namespace': m.CORE_ROOT,
        'case_ids': list(m.CASES.values()), 'accepted_after_sync': m.ORDER[:13],
        'preserved_file_count': 275, 'max_seconds': 300, 'kill_grace_seconds': 15,
        'metadata_only': True, 'models_started': False, 'test_suite_started': False,
        'new_GPU_dispatch_authorized': False, 'historical_STOP_records_unchanged': True,
        'automatic_queue': False, 'research_admitted': False,
        'inputs_file_sha256': m.digest(packet / 'INPUT.json'),
        'sources_sha256': {name: m.digest(packet / name) for name in m.SOURCES}}
    pin = m.canonical(op)
    m.validate_operation(op, pin)
    write_json(packet / 'SYNC.json', op)
    archive = raw / 'sync-operator.tar.gz'
    files = {m.PREFIX + '/' + str(p.relative_to(packet)): p for p in packet.rglob('*') if p.is_file()}
    manifest = make_archive(archive, files)
    write_json(raw / 'operator.manifest.json', manifest)
    extract_verified(archive, manifest, raw / 'extracted')
    payload = {'kind': 'reviewed-pair-sync-upload-v1', 'operation_sha256': pin,
        'archive_base64': base64.b64encode(archive.read_bytes()).decode(), 'manifest': manifest}
    write_json(raw / 'upload-payload.json', payload)
    installer = (core_raw / 'install-and-activate.py').read_text()
    installer = installer.replace(m.CORE_PREFIX, m.PREFIX).replace('reviewed-continuation-upload-v1', 'reviewed-pair-sync-upload-v1')
    installer = installer.replace('scripts/continue_reviewed_panel.py', 'scripts/sync_reviewed_panel.py')
    installer = installer.replace("'activate','--operation-sha256'", "'--operation-sha256'")
    installer = installer.replace('Continuation packet verified; activate two metadata requests only. No GPU case starts.',
        'Review-sync packet verified; rechecking saved cases on CPU. No model or test suite starts.')
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
s=importlib.util.spec_from_file_location('actual_sync_receipt_check',packet/'models/reviewed_panel_sync.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
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
print('Append the two original independent reviews after CPU rechecks. Preserve all 275 files; no GPU case starts.')
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
echo "Thirteen-case receipt saved at $output. Reply Done for independent review. No next GPU case starts."
'''
    (raw / 'sync-reviews.sh').write_text(helper)
    prepared = {'kind': m.KIND + '-preparation', 'operation_sha256': pin,
        'source_sha256': op['sources_sha256'], 'inputs_file_sha256': op['inputs_file_sha256'],
        'archive': manifest, 'payload_sha256': m.digest(raw / 'upload-payload.json'),
        'payload_bytes': (raw / 'upload-payload.json').stat().st_size,
        'helper_sha256': m.digest(raw / 'sync-reviews.sh'), 'records_to_append': sorted(m.TARGETS),
        'preserved_files': 275, 'accepted_after_sync': 13, 'complete_clip_pairs': 6,
        'remote_operation_performed': False, 'new_GPU_jobs_started': 0}
    write_json(raw / 'preparation.json', prepared)
    for p in raw.rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    subprocess.run(['bash', '-n', str(raw / 'sync-reviews.sh')], check=True)
    return raw, prepared


if __name__ == '__main__':
    raw, result = build()
    print(json.dumps({k: v for k, v in result.items() if k != 'archive'}, sort_keys=True, indent=2))
