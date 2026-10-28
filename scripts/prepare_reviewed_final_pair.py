#!/usr/bin/env python3
"""Prepare a source-only activation and two bounded attended helpers; connect to no host."""
import argparse
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.pilot_bundle import make_archive, extract_verified


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def build(test_results):
    module = importlib.util.spec_from_file_location('prepared_reviewed_control', ROOT / 'models/reviewed_panel_final_pair.py')
    m = importlib.util.module_from_spec(module)
    module.loader.exec_module(m)
    x = ET.parse(test_results).getroot()
    counts = {k: sum(int(s.get(k, 0)) for s in x.iter('testsuite')) for k in ('tests', 'failures', 'errors', 'skipped')}
    assert counts['tests'] > 0 and all(counts[k] == 0 for k in ('failures', 'errors', 'skipped'))
    prior_raw = ROOT / 'runs/pretrained/diagnostic' / m.SYNC_PREFIX
    prior_packet = prior_raw / 'extracted' / m.SYNC_PREFIX
    session = Path((prior_raw / 'latest-session.txt').read_text().strip())
    audit = json.loads((session / 'independent-verification.json').read_text())
    assert audit['verified'] and audit['receipt_file_sha256'] == sha(session / 'receipt.json') == m.PRIOR_RECEIPT
    raw = ROOT / 'runs/pretrained/diagnostic' / m.PREFIX
    raw.mkdir(exist_ok=False)
    (raw / 'validation').mkdir()
    packet = raw / 'staged' / m.PREFIX
    packet.mkdir(parents=True)
    core_raw = ROOT / 'runs/pretrained/diagnostic' / m.CORE_PREFIX
    core_packet = core_raw / 'extracted' / m.CORE_PREFIX
    for name in m.SOURCES:
        if name.startswith('support/core-packet/'):
            source = core_packet / name.removeprefix('support/core-packet/')
        elif name == 'support/yoyo_bowling_review_sync.py':
            source = prior_packet / 'models/yoyo_bowling_review_sync.py'
        else:
            source = ROOT / name
        path = packet / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
    for source, name in [(prior_packet / 'SYNC.json', 'prior-OPERATION.json'),
                         (prior_packet / 'INPUT.json', 'prior-INPUT.json'), (session / 'receipt.json', 'prior-receipt.json')]:
        shutil.copyfile(source, packet / name)
    fixture = json.loads((packet / 'tests/fixtures/reviewed_final_pair_metadata.json').read_text())
    sync = m.load_module('builder_prior_sync', prior_packet / 'models/yoyo_bowling_review_sync.py')
    sop = json.loads((prior_packet / 'SYNC.json').read_text())
    si = json.loads((prior_packet / 'INPUT.json').read_text())
    sr = json.loads((session / 'receipt.json').read_text())
    sync.verify_receipt(sr, sop, si)
    old_data = json.loads((core_packet / 'INPUT.json').read_text())
    data = m.expected_input(sync, sop, si, old_data, sha(prior_packet / 'SYNC.json'))
    assert fixture['inputs'] == data and fixture['receipt'] == sr
    write_json(packet / 'INPUT.json', data)
    op = {'kind': m.KIND, 'date_Pacific': '2026-10-28', 'sync_operation_sha256': m.SYNC_OP,
        'sync_receipt_file_sha256': m.PRIOR_RECEIPT, 'reference_revision_sha256': m.RP,
        'reference_implementation_sha256': m.MP, 'imported_case_ids': m.ACCEPTED,
        'first_requests': m.TARGETS, 'coordinator_namespace': m.TARGET, 'historical_STOP_sha256': m.STOP,
        'preserve_all_three_historical_STOP_records': True, 'numeric_settings_changed': False,
        'case_seconds': 4800, 'kill_grace_seconds': 15, 'matrix_seconds': 3600, 'replay_seconds': 900,
        'parent_seconds': 4620, 'max_attempts': 1, 'additional_allowance_seconds': 0,
        'automatic_GPU_retries': 0, 'max_parallel_cases': 2, 'one_case_per_host': True,
        'automatic_queue': False, 'paid_compute': False, 'research_admitted': False,
        'sources_sha256': {name: sha(packet / name) for name in m.SOURCES},
        'focused_CPU_guards': counts['tests'], 'preserved_file_count': m.PRESERVED,
        'inputs_file_sha256': sha(packet / 'INPUT.json')}
    pin = m.canonical(op)
    m.validate_operation(op, pin)
    write_json(packet / 'CONTINUATION.json', op)
    archive = raw / 'continuation-operator.tar.gz'
    entries = {m.PREFIX + '/' + str(p.relative_to(packet)): p for p in packet.rglob('*') if p.is_file()}
    manifest = make_archive(archive, entries)
    write_json(raw / 'operator.manifest.json', manifest)
    extract_verified(archive, manifest, raw / 'extracted')
    payload = {'kind': 'reviewed-seventeen-pair-upload-v1', 'operation_sha256': pin,
        'archive_base64': base64.b64encode(archive.read_bytes()).decode(), 'manifest': manifest}
    write_json(raw / 'upload-payload.json', payload)
    installer = (ROOT / 'runs/pretrained/diagnostic/archery-yoyo-pair-20261027/install-and-activate.py').read_text()
    installer = installer.replace('archery-yoyo-pair-20261027', m.PREFIX)
    installer = installer.replace('reviewed-batch-upload-v1', 'reviewed-seventeen-pair-upload-v1')
    installer = installer.replace('scripts/continue_reviewed_batch.py', 'scripts/continue_reviewed_final_pair.py')
    installer = installer.replace('1024*1024', '2*1024*1024')
    assert (raw / 'upload-payload.json').stat().st_size <= 2 * 1024**2
    (raw / 'install-and-activate.py').write_text(installer)
    body = 'exec(__import__("base64").b64decode(' + repr(base64.b64encode(installer.encode()).decode()) + '))'
    command = shlex.join(['timeout', '--signal=TERM', '--kill-after=15s', '300', 'env',
        'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
        '/home/kdu/.venvs/vjepa/bin/python', '-B', '-c', body, sha(raw / 'upload-payload.json')])
    (raw / 'activation-command.txt').write_text(command + '\n')
    shutil.copyfile(prior_raw / 'known_hosts', raw / 'known_hosts')
    verify = '''from pathlib import Path
import hashlib,importlib.util,json,sys
base=Path(__file__).resolve().parent;output=Path(sys.argv[1]);packet=base/'extracted/'''+m.PREFIX+"'\n"+'''
s=importlib.util.spec_from_file_location('actual_activation_verifier',packet/'models/reviewed_panel_final_pair.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
op=json.loads((packet/'CONTINUATION.json').read_text());m.validate_operation(op,'''+repr(pin)+''')
for n,p in op['sources_sha256'].items():assert hashlib.sha256((packet/n).read_bytes()).hexdigest()==p,n
c=m.boundary(packet,'''+repr(pin)+''')
r=json.loads((output/'receipt.json.part').read_text());result=m.verify_activation(r,op,c['data'],c['prior'] if 'prior' in c else m.load_module('record_bytes_support',packet/'support/core-packet/support/core-packet/support/core-packet/support/core-packet/support/reviewed_panel.py'))
result['receipt_file_sha256']=hashlib.sha256((output/'receipt.json.part').read_bytes()).hexdigest()
with (output/'receipt-verification.json').open('x') as f:f.write(json.dumps(result,sort_keys=True,indent=2)+'\\n')
(output/'receipt.json.part').rename(output/'receipt.json')
print(json.dumps(result,sort_keys=True,indent=2))
'''
    (raw / 'verify-activation.py').write_text(verify)
    pins = {n: sha(raw / n) for n in ('upload-payload.json', 'activation-command.txt', 'verify-activation.py', 'known_hosts')}
    pins.update({'extracted/' + m.PREFIX + '/' + str(p.relative_to(packet)): sha(p) for p in packet.rglob('*') if p.is_file()})
    helper = '''#!/usr/bin/env bash
set -euo pipefail
base='''+shlex.quote(str(raw))+'''
if [[ "${REVIEWED_ACTIVATION_AWAKE:-0}" != 1 ]]; then
 exec /usr/bin/caffeinate -i env REVIEWED_ACTIVATION_AWAKE=1 bash "$0"
fi
py='''+shlex.quote(str(ROOT / '.venv/bin/python'))+'''
PYTHONDONTWRITEBYTECODE=1 "$py" - "$base" <<'PIN_CHECK'
from pathlib import Path
import hashlib,sys
p=Path(sys.argv[1]);expected='''+repr(pins)+'''
for n,s in expected.items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==s,n
print('Create two fresh requests in a separate namespace; retain all seventeen imports and all three historical STOP records. No model starts.')
PIN_CHECK
output=$(mktemp -d "$base/activation-session-$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
printf '%s\\n' "$output" > "$base/latest-activation.txt"
command=$(cat "$base/activation-command.txt")
set +e
ssh -T -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$base/known_hosts" -o GlobalKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \\
 kdu@linux10609.dc.engr.scu.edu "$command" < "$base/upload-payload.json" > "$output/receipt.json.part" 2> "$output/activation.stderr"
code=$?
set -e
printf '%s\\n' "$code" > "$output/connection-exit-status.txt"
cat "$output/activation.stderr"
if [[ "$code" -ne 0 ]]; then
 echo "Activation did not confirm success. Preserve $output and report the ending before repeating." >&2
 exit "$code"
fi
PYTHONDONTWRITEBYTECODE=1 "$py" "$base/verify-activation.py" "$output"
echo "Activation receipt saved at $output. Reply Done for independent review; no GPU helper has started."
'''
    (raw / 'activate-continuation.sh').write_text(helper)
    launches = build_launches(raw, packet, m, op, pin, counts['tests'])
    prepared = {'kind': m.KIND + '-preparation', 'operation_sha256': pin, 'source_sha256': op['sources_sha256'],
        'archive': manifest, 'payload_sha256': sha(raw / 'upload-payload.json'),
        'payload_bytes': (raw / 'upload-payload.json').stat().st_size,
        'activation_helper_sha256': sha(raw / 'activate-continuation.sh'), 'case_helpers': launches,
        'focused_CPU_tests': counts, 'old_CPU_guards': 76, 'campus_CPU_guards': 76 + counts['tests'],
        'activation_is_metadata_only': True, 'ancestor_files_preserved': m.PRESERVED,
        'all_three_historical_STOP_records_preserved': True, 'remote_activation_performed': False, 'GPU_jobs_started': 0}
    write_json(raw / 'preparation.json', prepared)
    for p in raw.rglob('*.sh'):
        subprocess.run(['bash', '-n', str(p)], check=True)
    for p in raw.rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    return raw, prepared


def build_launches(raw, packet, m, op, pin, new_count):
    templates = ROOT / 'runs/pretrained/diagnostic/panel-continuation-20261026/linux10609'
    remote_packet = m.BASE + '/' + m.PREFIX + '/unpacked/' + m.PREFIX
    check = """import hashlib,json,sys;from pathlib import Path
p=Path(sys.argv[1]);q=json.loads((p/'CONTINUATION.json').read_text())
assert hashlib.sha256(json.dumps(q,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()=="""+repr(pin)+"\n"+"""for n,s in q['sources_sha256'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==s,n
"""
    merge = """import json,sys,xml.etree.ElementTree as E;from pathlib import Path
p=Path(sys.argv[1]);root=E.Element('testsuites');counts=[]
for name,expected in [('old',76),('new',"""+str(new_count)+""")]:
 x=E.parse(p/('cpu-'+name+'.xml')).getroot();c={k:sum(int(s.get(k,0)) for s in x.iter('testsuite')) for k in ['tests','failures','errors','skipped']}
 assert c=={'tests':expected,'failures':0,'errors':0,'skipped':0},c
 counts.append(c)
 for s in x.iter('testsuite'):root.append(s)
E.ElementTree(root).write(p/'cpu-results.xml',encoding='unicode')
(p/'cpu.log').write_text((p/'cpu-old.log').read_text()+'\\n'+(p/'cpu-new.log').read_text())
(p/'cpu-summary.log').write_text(json.dumps({k:sum(c[k] for c in counts) for k in counts[0]},sort_keys=True)+'\\n')
print((p/'cpu-summary.log').read_text(),end='')
"""
    result = []
    for host, cid in m.TARGETS.items():
        local = raw / host
        local.mkdir()
        pipeline = '''#!/usr/bin/env bash
set -euo pipefail
output=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
base=$(cd -- "$output/../.." && pwd)
py="$HOME/.venvs/vjepa/bin/python"
original="$HOME/'''+m.BASE+'/'+m.REFERENCE+'''"
package="$HOME/'''+m.BASE+'/'+m.PACKAGE+'''"
export REVIEWED_TEST_REFERENCE="$original" TWO_HOST_TEST_PACKAGE="$package" PYTHONPATH="$package"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$py" -B -c '''+shlex.quote(check)+''' "$base"
"$py" -B -m pytest -q -p no:cacheprovider --rootdir="$original" "$original/tests/test_two_host_panel.py" \\
 --basetemp="$output/cpu-old-fixtures" --junitxml="$output/cpu-old.xml" > "$output/cpu-old.log" 2>&1
cat "$output/cpu-old.log"
"$py" -B -m pytest -q -p no:cacheprovider --rootdir="$base" "$base/tests/test_reviewed_final_pair.py" \\
 --basetemp="$output/cpu-new-fixtures" --junitxml="$output/cpu-new.xml" > "$output/cpu-new.log" 2>&1
cat "$output/cpu-new.log"
"$py" -B -c '''+shlex.quote(merge)+''' "$output"
exec "$py" -B -u "$base/scripts/continue_reviewed_final_pair.py" run --operation-sha256 '''+pin+''' --host '''+host+''' --launcher-started "${REVIEWED_LAUNCHER_STARTED:?}"
'''
        launch = (templates / 'remote-launch.sh').read_text().replace('linux10609', host).replace('clip-05-bank-0', cid)
        launch = launch.replace("date -u +%Y-%m-%dT%H:%M:%SZ > \"$output/started-utc.txt\"", ": \"${REVIEWED_LAUNCHER_STARTED:?}\"\nexport REVIEWED_LAUNCHER_STARTED\nprintf '%s\\n' \"$REVIEWED_LAUNCHER_STARTED\" > \"$output/started-epoch.txt\"\ndate -u +%Y-%m-%dT%H:%M:%SZ > \"$output/started-utc.txt\"")
        remaining = "remaining=$(\"$HOME/.venvs/vjepa/bin/python\" -B -c 'import math,os,time;v=4800-(time.time()-float(os.environ[\"REVIEWED_LAUNCHER_STARTED\"]));assert 0<v<=4800;print(max(1,math.floor(v)))')\n"
        launch = launch.replace('timeout --verbose --signal=TERM --kill-after=15s 4800', remaining + 'timeout --verbose --signal=TERM --kill-after=15s "$remaining"')
        (local / 'remote-launch.sh').write_text(launch)
        (local / 'remote-pipeline.sh').write_text(pipeline)
        body = {'host': host, 'case_id': cid, 'sources': {n: base64.b64encode(text.encode()).decode()
            for n, text in [('source-launch.sh', launch), ('source-pipeline.sh', pipeline)]}}
        encoded = base64.b64encode(json.dumps(body, sort_keys=True).encode()).decode()
        bootstrap = """import base64,hashlib,json,os,socket,time;from pathlib import Path
started=time.time();q=json.loads(base64.b64decode("""+repr(encoded)+"""))
assert socket.gethostname()==q['host']=="""+repr(host)+""";assert q['case_id']=="""+repr(cid)+"""
p=Path.home()/"""+repr(remote_packet)+"""
op=json.loads((p/'CONTINUATION.json').read_text());assert hashlib.sha256(json.dumps(op,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()=="""+repr(pin)+"""
for n,s in op['sources_sha256'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==s,n
assert not (p/'launchers').is_symlink();out=p/'launchers'/(q['host']+'-'+q['case_id']);assert out.resolve().is_relative_to(p.resolve())
os.umask(0o077);out.mkdir(parents=True,exist_ok=False);assert set(q['sources'])=={'source-launch.sh','source-pipeline.sh'}
for n,b in q['sources'].items():
 with (out/n).open('xb') as f:f.write(base64.b64decode(b,validate=True))
os.environ['REVIEWED_LAUNCHER_STARTED']=str(started)
os.execv('/bin/bash',['bash',str(out/'source-launch.sh')])
"""
        (local / 'bootstrap.py').write_text(bootstrap)
        (local / 'launch-command.txt').write_text(shlex.join(['/home/kdu/.venvs/vjepa/bin/python', '-B', '-c', bootstrap]) + '\n')
        collector = (templates / 'stream-evidence.py').read_text().replace('linux10609', host).replace('clip-05-bank-0', cid)
        collector = collector.replace("CASE_ARCHIVE = 'revision-evidence.tar.gz'", "CASE_ARCHIVE = 'continuation-evidence.tar.gz'")
        start = collector.index('    packet = Path.home()')
        end = collector.index('    inventory = {}', start)
        files = {'evidence/continuation-evidence.tar.gz': "output/'continuation-evidence.tar.gz'",
            'evidence/continuation-evidence.tar.gz.manifest.json': "output/'continuation-evidence.tar.gz.manifest.json'"}
        logs = ('started-utc.txt', 'started-epoch.txt', 'ended-utc.txt', 'exit-status.txt', 'case.log',
            'cpu.log', 'cpu-results.xml', 'cpu-summary.log', 'cpu-old.log', 'cpu-new.log', 'cpu-old.xml', 'cpu-new.xml',
            'source-launch.sh', 'source-pipeline.sh')
        files.update({'launcher/' + name: 'launcher/' + repr(name) for name in logs})
        files.update({'operation/' + name: 'packet/' + repr(name) for name in
            ('CONTINUATION.json', 'INPUT.json', 'prior-OPERATION.json', 'prior-INPUT.json', 'prior-receipt.json')})
        files.update({'operation/source/' + name: 'packet/' + repr(name) for name in m.SOURCES})
        for name, expression in {'request': "root/'requests'/(CASE_ID+'.json')", 'claim': "root/'claims'/CASE_ID/'claim.json'",
            'outcome': "root/'outcomes'/(CASE_ID+'.json')", 'export': "root/'exports'/(CASE_ID+'.json')",
            'genesis': "root/'genesis.json'", 'imports': "root/'imports.json'", 'resolution': "root/'resolution.json'"}.items():
            files['ledger/' + name + '.json'] = expression
        files.update({'case/' + name: 'output/' + repr(name) for name in ('case-envelope.json', 'preservation-before-export.json', 'preservation-after-export.json')})
        replacements = '    packet = Path.home() / ' + repr(remote_packet) + '\n'
        replacements += '    root = Path.home() / ' + repr(m.BASE + '/' + m.TARGET) + '\n'
        replacements += "    output=root/'cases'/CASE_ID/'attempt-01'\n    launcher=packet/'launchers'/(HOST+'-'+CASE_ID)\n    files={\n"
        replacements += ''.join('        ' + repr(name) + ':' + expression + ',\n' for name, expression in files.items()) + '    }\n'
        collector = collector[:start] + replacements + collector[end:]
        collector = collector.replace('read-only-canonical-continuation-download-v1', 'read-only-reviewed-seventeen-pair-download-v1')
        collector = collector.replace('b5501e3fbc4c76b2d66d49ddc063fe64dd70f22885ef3d506ce51c4bcf391a05', pin)
        (local / 'stream-evidence.py').write_text(collector)
        receiver = (templates / 'receive-evidence.py').read_text().replace('linux10609', host).replace('clip-05-bank-0', cid)
        receiver = receiver.replace('revision-evidence.tar.gz', 'continuation-evidence.tar.gz')
        receiver = receiver.replace('read-only-canonical-continuation-download-v1', 'read-only-reviewed-seventeen-pair-download-v1')
        receiver = receiver.replace('b5501e3fbc4c76b2d66d49ddc063fe64dd70f22885ef3d506ce51c4bcf391a05', pin)
        a = receiver.index('EXPECTED ='); b = receiver.index('\n\n\ndef digest', a)
        receiver = receiver[:a] + 'EXPECTED = ' + repr(set(files)) + receiver[b:]
        (local / 'receive-evidence.py').write_text(receiver)
        download = (templates / 'download-evidence.sh').read_text().replace(str(templates), str(local)).replace('linux10609.dc.', host + '.dc.')
        (local / 'download-evidence.sh').write_text(download)
        pins = {host + '/' + n: sha(local / n) for n in ('bootstrap.py', 'launch-command.txt', 'stream-evidence.py', 'receive-evidence.py', 'download-evidence.sh')}
        pins['known_hosts'] = sha(raw / 'known_hosts')
        helper = (templates / 'run-case.sh').read_text()
        a = helper.index('PYTHONDONTWRITEBYTECODE=1'); b = helper.index('output=$(mktemp', a)
        guard = '''PYTHONDONTWRITEBYTECODE=1 '''+shlex.quote(str(ROOT / '.venv/bin/python'))+''' - "$base/.." <<'PIN_CHECK'
from pathlib import Path
import hashlib,json,sys
p=Path(sys.argv[1]);expected='''+repr(pins)+'''
for n,s in expected.items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==s,n
review=json.loads((p/'activation-reviewed.json').read_text())
assert review['verified'] is True and review['operation_sha256']=='''+repr(pin)+''' and review['accepted_cases']==17
print('Reviewed activation, bounded case and copy-only helpers verified.')
PIN_CHECK
'''
        helper = helper[:a] + guard + helper[b:]
        helper = helper.replace(str(templates), str(local)).replace('linux10609.dc.', host + '.dc.').replace('clip-05-bank-0', cid)
        helper = helper.replace('runs on linux10609', 'runs on ' + host)
        (local / 'run-case.sh').write_text(helper)
        result.append({'host': host, 'case_id': cid, 'launch_helper_sha256': sha(local / 'run-case.sh'),
            'bootstrap_sha256': sha(local / 'bootstrap.py'), 'source_launcher_sha256': sha(local / 'remote-launch.sh'),
            'pipeline_sha256': sha(local / 'remote-pipeline.sh'), 'download_expected_files': len(files),
            'activation_independent_review_required': True})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-results', type=Path, required=True)
    args = parser.parse_args()
    raw, prepared = build(args.test_results)
    print(json.dumps({k: v for k, v in prepared.items() if k != 'archive'}, sort_keys=True, indent=2))
