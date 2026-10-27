#!/usr/bin/env python3
"""Build two attended, bounded launch/download helpers after verified activation."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    raw=ROOT/'runs/pretrained/diagnostic/panel-continuation-20261026'
    packet=raw/'extracted/two-host-continuation-20261026';operation=json.loads((packet/'OPERATION.json').read_text())
    pin=json.loads((raw/'preparation.json').read_text())['operation_sha256']
    confirmation=json.loads((ROOT/'report/checks/2026-10-26-panel-continuation-activation-verification.json').read_text())
    assert confirmation['verified'] and confirmation['operation_sha256']==pin and not confirmation['GPU_or_model_jobs_started']
    old=ROOT/'runs/pretrained/diagnostic/two-host-panel-20261026'
    base='Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024'
    remote_packet=base+'/two-host-continuation-20261026/unpacked/two-host-continuation-20261026'
    extra_logs=('cpu-old.log','cpu-new.log','cpu-old.xml','cpu-new.xml')
    staged=[]
    for host,cid in operation['targets'].items():
        local=raw/host;local.mkdir(exist_ok=False)
        check='''import hashlib,json,sys;from pathlib import Path
p=Path(sys.argv[1]);q=json.loads((p/'OPERATION.json').read_text())
assert hashlib.sha256(json.dumps(q,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()=='''+repr(pin)+'''
for n,s in q['sources_sha256'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==s,n
'''
        merge='''import json,sys,xml.etree.ElementTree as E;from pathlib import Path
p=Path(sys.argv[1]);root=E.Element('testsuites');counts=[]
for name,expected in [('old',76),('new',46)]:
 x=E.parse(p/('cpu-'+name+'.xml')).getroot();c={k:sum(int(s.get(k,0)) for s in x.iter('testsuite')) for k in ['tests','failures','errors','skipped']}
 assert c=={'tests':expected,'failures':0,'errors':0,'skipped':0},c
 counts.append(c)
 for s in x.iter('testsuite'):root.append(s)
E.ElementTree(root).write(p/'cpu-results.xml',encoding='unicode')
(p/'cpu.log').write_text((p/'cpu-old.log').read_text()+'\\n'+(p/'cpu-new.log').read_text())
(p/'cpu-summary.log').write_text(json.dumps({k:sum(c[k] for c in counts) for k in counts[0]},sort_keys=True)+'\\n')
print((p/'cpu-summary.log').read_text(),end='')
'''
        pipeline='''#!/usr/bin/env bash
set -euo pipefail
output=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
base=$(cd -- "$output/../.." && pwd)
py="$HOME/.venvs/vjepa/bin/python"
original="$HOME/'''+base+'''/two-host-execution-20261026/unpacked/two-host-execution-20261026"
package="$HOME/'''+base+'''/unpacked/matched-panel-20261024"
export CONTINUATION_TEST_ORIGINAL="$original" TWO_HOST_TEST_PACKAGE="$package" PYTHONPATH="$package"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$py" -B -c '''+shlex.quote(check)+''' "$base"
echo 'Checking the original 76 CPU guards and 46 continuation guards before any model.'
"$py" -B -m pytest -q -p no:cacheprovider --rootdir="$original" "$original/tests/test_two_host_panel.py" \
 --basetemp="$output/cpu-old-fixtures" --junitxml="$output/cpu-old.xml" > "$output/cpu-old.log" 2>&1
cat "$output/cpu-old.log"
"$py" -B -m pytest -q -p no:cacheprovider --rootdir="$base" "$base/tests/test_panel_continuation.py" \
 --basetemp="$output/cpu-new-fixtures" --junitxml="$output/cpu-new.xml" > "$output/cpu-new.log" 2>&1
cat "$output/cpu-new.log"
"$py" -B -c '''+shlex.quote(merge)+''' "$output"
exec "$py" -B -u "$base/scripts/continue_two_host_panel.py" run --operation-sha256 '''+pin+'''\n'''
        launch=(old/'extracted/two-host-execution-20261026/launch-linux10609.sh').read_text()
        a=launch.index('base=$(cd');b=launch.index("child_pid=''",a)
        launch=launch[:a]+'''output=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
base=$(cd -- "$output/../.." && pwd)
'''+launch[b:]
        launch=launch.replace('linux10609 clip-03-bank-1',host+' '+cid).replace('on linux10609','on '+host)
        launch=launch.replace('bash "$base/pipeline-linux10609.sh"','bash "$output/source-pipeline.sh"')
        (local/'remote-launch.sh').write_text(launch);(local/'remote-pipeline.sh').write_text(pipeline)
        payload={'host':host,'case_id':cid,'sources':{n:base64.b64encode(s.encode()).decode() for n,s in [('source-launch.sh',launch),('source-pipeline.sh',pipeline)]}}
        body=(json.dumps(payload,sort_keys=True)+'\n').encode();encoded=base64.b64encode(body).decode()
        bootstrap='''import base64,hashlib,json,os;from pathlib import Path;import socket
body=base64.b64decode('''+repr(encoded)+''');assert hashlib.sha256(body).hexdigest()=='''+repr(hashlib.sha256(body).hexdigest())+'''
q=json.loads(body);assert socket.gethostname()==q['host']=='''+repr(host)+''';assert q['case_id']=='''+repr(cid)+'''
p=Path.home()/'''+repr(remote_packet)+'''
op=json.loads((p/'OPERATION.json').read_text());assert hashlib.sha256(json.dumps(op,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()=='''+repr(pin)+'''
for n,s in op['sources_sha256'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==s,n
assert not (p/'launchers').is_symlink();out=p/'launchers'/(q['host']+'-'+q['case_id'])
assert out.resolve().is_relative_to(p.resolve());os.umask(0o077);out.mkdir(parents=True,exist_ok=False)
assert set(q['sources'])=={'source-launch.sh','source-pipeline.sh'}
for n,b in q['sources'].items():
 with (out/n).open('xb') as f:f.write(base64.b64decode(b,validate=True))
os.execv('/bin/bash',['bash',str(out/'source-launch.sh')])
'''
        (local/'bootstrap.py').write_text(bootstrap)
        command=shlex.join(['/home/kdu/.venvs/vjepa/bin/python','-B','-c',bootstrap]);(local/'launch-command.txt').write_text(command+'\n')
        # Extend the existing byte-checked copy path with the operational sources/history.
        stream=(old/host/'stream-evidence.py').read_text()
        prior='clip-03-bank-1' if host=='linux10609' else 'clip-04-bank-0'
        stream=stream.replace(prior,cid).replace('two-host-execution-20261026/unpacked/two-host-execution-20261026','two-host-continuation-20261026/unpacked/two-host-continuation-20261026')
        stream=stream.replace("'cpu-summary.log'):","'cpu-summary.log',"+','.join(repr(n) for n in extra_logs)+"):")
        stream=stream.replace("packet/('launch-'+HOST+'.sh')","packet/'launchers'/(HOST+'-'+CASE_ID)/'source-launch.sh'")
        stream=stream.replace("packet/('pipeline-'+HOST+'.sh')","packet/'launchers'/(HOST+'-'+CASE_ID)/'source-pipeline.sh'")
        at=stream.index('    inventory = {}')
        additions="    files['operation/OPERATION.json']=packet/'OPERATION.json'\n    files['operation/prior-sync-receipt.json']=packet/'prior-sync-receipt.json'\n    files['operation/resolution.json']=root/'stop-resolutions/export-alias-20261026.json'\n    files['operation/archived-STOP.json']=root/'archived-stops/b42a023f4b7e3bf0c9183f7af384d4abe3cccecaf05d727495b9782107c0f892.json'\n"
        for name in operation['sources_sha256']:additions+="    files["+repr('operation/source/'+name)+"]=packet/"+repr(name)+"\n"
        stream=stream[:at]+additions+stream[at:]
        stream=stream.replace('read-only-two-host-case-download-v1','read-only-canonical-continuation-download-v1')
        stream=stream.replace("'revision_sha256':REVISION_SHA,","'revision_sha256':REVISION_SHA,'operation_sha256':"+repr(pin)+",")
        (local/'stream-evidence.py').write_text(stream)
        receiver=(old/host/'receive-evidence.py').read_text().replace(prior,cid)
        a=receiver.index('EXPECTED =');b=receiver.index('\n\n\ndef digest',a)
        old_expected=ast.literal_eval(receiver[a+len('EXPECTED ='):b].strip())
        expected=set(old_expected)|{'launcher/'+n for n in extra_logs}|{'operation/'+n for n in ['OPERATION.json','prior-sync-receipt.json','resolution.json','archived-STOP.json']}|{'operation/source/'+n for n in operation['sources_sha256']}
        receiver=receiver[:a]+'EXPECTED = '+repr(expected)+receiver[b:]
        receiver=receiver.replace('read-only-two-host-case-download-v1','read-only-canonical-continuation-download-v1')
        receiver=receiver.replace("manifest['attempt'] != 'attempt-01'", "manifest['attempt'] != 'attempt-01' or manifest['operation_sha256'] != "+repr(pin))
        (local/'receive-evidence.py').write_text(receiver)
        download=(old/host/'download-evidence.sh').read_text().replace(str(old/host),str(local))
        (local/'download-evidence.sh').write_text(download)
        pins={host+'/'+name:digest(local/name) for name in ['launch-command.txt','bootstrap.py','stream-evidence.py','receive-evidence.py','download-evidence.sh']}
        pins['known_hosts']=digest(raw/'known_hosts')
        helper=(old/host/'run-case.sh').read_text()
        a=helper.index('PYTHONDONTWRITEBYTECODE=1');b=helper.index('output=$(mktemp',a)
        check_local='''PYTHONDONTWRITEBYTECODE=1 '''+shlex.quote(str(ROOT/'.venv/bin/python'))+''' - "$base/.." <<'PIN_CHECK'
from pathlib import Path
import hashlib,sys
root=Path(sys.argv[1]);expected='''+repr(pins)+'''
for name,pin in expected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==pin,name
print('Bounded continuation launch and copy-only helpers verified.')
PIN_CHECK
'''
        helper=helper[:a]+check_local+helper[b:]
        helper=helper.replace(str(old/host),str(local))
        a=helper.index('install_command=');b=helper.index('set +e',a)
        helper=helper[:a]+"echo 'Keep this Terminal and Mac lid open. Only "+cid+" runs on "+host+"; then stop for independent review.'\nlaunch_command=$(cat \"$base/launch-command.txt\")\n"+helper[b:]
        (local/'run-case.sh').write_text(helper)
        staged.append({'host':host,'case_id':cid,'bootstrap_sha256':digest(local/'bootstrap.py'),'download_expected_regular_files':len(expected),
            'launch_helper_sha256':digest(local/'run-case.sh'),'source_launcher_sha256':digest(local/'remote-launch.sh'),'pipeline_sha256':digest(local/'remote-pipeline.sh')})
    prepared={'kind':'canonical-continuation-bounded-launch-preparation-v1','operation_sha256':pin,'activation_receipt_sha256':confirmation['receipt_sha256'],
        'case_helpers':staged,'case_wall_seconds':4800,'cleanup_seconds':15,'CPU_guards_per_host':122,
        'old_guards':76,'continuation_guards':46,'automatic_GPU_retry':False,'copy_only_retry_count':1,'GPU_jobs_started':0}
    (raw/'launch-preparation.json').write_text(json.dumps(prepared,sort_keys=True,indent=2)+'\n')
    return raw,prepared


if __name__=='__main__':
    raw,p=build()
    for f in raw.rglob('*.sh'):subprocess.run(['bash','-n',str(f)],check=True)
    for f in raw.rglob('*.py'):ast.parse(f.read_text(),filename=str(f))
    print(json.dumps(p,indent=2))
