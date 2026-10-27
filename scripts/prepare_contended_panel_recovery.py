#!/usr/bin/env python3
"""Build one separately authorized recovery handoff; perform no remote work."""
import ast
import base64
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
NEW_GUARDS=85


def build():
    m=importlib.util.spec_from_file_location('contended_prepare',ROOT/'scripts/recover_contended_panel_case.py')
    ops=importlib.util.module_from_spec(m);m.loader.exec_module(ops)
    readback=json.loads((ROOT/'report/checks/2026-10-26-recovery-lock-startup-readback.json').read_text())
    assert readback['verified'] and readback['recovery_grant_absent'] and readback['attempt_02_absent']
    assert readback['startup_charge_seconds']==ops.STARTUP_CHARGE and readback['remaining_case_seconds']==ops.CASE_LIMIT
    assert readback['original_grant_id']==ops.ORIGINAL_GRANT and readback['transport_manifest_sha256']==ops.STARTUP_MANIFEST
    startup=ROOT/readback['download_directory']/'downloaded'
    previous=json.loads((startup/'packet/RECOVERY.json').read_text())
    assert ops.canonical(previous)==ops.ORIGINAL_PROPOSAL
    raw=ROOT/'runs/pretrained/diagnostic'/ops.PREFIX;raw.mkdir(exist_ok=False)
    (raw/'validation').mkdir();packet=raw/'staged'/ops.PREFIX;packet.mkdir(parents=True)
    audit=json.loads((ROOT/'report/checks/2026-10-26-floor-partial-audit.json').read_text())
    assert audit['copy_verified'] and not audit['case_accepted'] and audit['budget']['extra_budget_approved']
    assert audit['STOP_file_sha256']==ops.STOP and audit['partial_archive_members']==24
    d=ROOT/audit['download_directory']/'downloaded'
    peerraw=ROOT/'runs/pretrained/diagnostic/panel-continuation-20261026/linux10609'
    peerd=Path((peerraw/'latest-download.txt').read_text().strip())
    peer=peerd/'independently-reviewed/independent-review.json'
    prior=ROOT/'runs/pretrained/diagnostic/two-host-export-incident-20261026/evidence-download-20261026T185802Z.H23uD0'
    repair=json.loads((prior/'ledger-repair-plan.json').read_text())['payload']
    preserved={}
    for name,pin in repair['preserve_remote_files_sha256'].items():
        if name.startswith('cases/'):
            _,cid,tail=name.split('/',2);preserved['cases/'+cid+'/attempt-01/'+tail]=pin
        elif name.startswith('ledger/'):
            name=name.removeprefix('ledger/')
            if name=='STOP.json':preserved['archived-stops/'+pin+'.json']=pin
            elif name.startswith('claims/'):preserved[name.removesuffix('.json')+'/claim.json']=pin
            else:preserved[name]=pin
    for name,row in repair['proposed_records'].items():preserved[name]=row['file_sha256']
    for name,local in {'STOP.json':'ledger/STOP.json','outcomes/'+ops.CASE+'.json':'ledger/outcome.json',
        'claims/'+ops.CASE+'/claim.json':'ledger/claim.json','genesis.json':'ledger/genesis.json','imports.json':'ledger/imports.json'}.items():
        preserved[name]=ops.digest(d/local)
    for name,pin in audit['failed_attempt_preserve_files_sha256'].items():preserved['cases/'+ops.CASE+'/attempt-01/'+name]=pin
    for name,local in {'outcomes/clip-05-bank-0.json':'ledger/outcome.json','exports/clip-05-bank-0.json':'ledger/export.json',
                      'claims/clip-05-bank-0/claim.json':'ledger/claim.json'}.items():preserved[name]=ops.digest(peerd/'downloaded'/local)
    for name in ops.SOURCES:
        target=packet/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    shutil.copyfile(peer,packet/'biking-independent-review.json')
    for name in (*ops.PRIOR_FILES,'transport-manifest.json'):
        target=packet/'prior-startup'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(startup/name,target)
    assert preserved==previous['preserved_coordinator_files_sha256']
    startup_manifest=json.loads((startup/'transport-manifest.json').read_text())
    startup_files={n:v['sha256'] for n,v in startup_manifest['files'].items()}
    startup_preserved={n.removeprefix('packet/'):p for n,p in startup_files.items() if n.startswith('packet/')}
    startup_preserved.update({n:p for n,p in startup_files.items() if n.startswith('launcher/')})
    spec={'kind':ops.KIND,'date_Pacific':'2026-10-26','case_id':ops.CASE,'host':ops.HOST,'attempt':ops.ATTEMPT,
        'reference_revision_sha256':ops.RP,'reference_implementation_sha256':ops.MP,'reference_operation_sha256':ops.OP,
        'failed_STOP_sha256':ops.STOP,'extra_case_seconds':4800,'case_seconds':ops.CASE_LIMIT,'kill_grace_seconds':15,
        'parent_seconds':4620,'matrix_seconds':3600,'replay_seconds':900,'additional_attempts':1,
        'reuse_old_tensors':False,'old_remainder_added':False,'automatic_GPU_retries':0,'automatic_queue':False,
        'strict_GPU_guard_unchanged':True,'preserve_global_STOP':True,'one_case_only':True,'paid_compute':False,
        'research_admitted':False,'numeric_settings_changed':False,
        'old_budget':ops.budget_assessment(2065.4081135171,2086.5223807520233,2113),
        'failed_partial_archive_sha256':audit['partial_archive_sha256'],
        'independent_partial_audit_file_sha256':ops.digest(ROOT/'report/checks/2026-10-26-floor-partial-audit.json'),
        'failed_attempt_files_sha256':audit['failed_attempt_preserve_files_sha256'],
        'preserved_coordinator_files_sha256':preserved,'peer_review_file_sha256':ops.digest(peer),
        'original_proposal_sha256':ops.ORIGINAL_PROPOSAL,'original_authorization_file_sha256':ops.ORIGINAL_AUTH,
        'original_grant_id':ops.ORIGINAL_GRANT,'prior_startup_transport_sha256':ops.STARTUP_TRANSPORT,
        'prior_startup_manifest_sha256':ops.STARTUP_MANIFEST,'prior_startup_charge_seconds':ops.STARTUP_CHARGE,
        'prior_startup_files_sha256':startup_files,'preserved_failed_startup_files_sha256':startup_preserved,
        'sources_sha256':{n:ops.digest(packet/n) for n in ops.SOURCES},
        'scope':'Correct the preflight lock order and carry the same unused grant. Charge 36 seconds for the failed startup; only 4764 seconds remain. One fresh matrix/replay attempt, same host and case. Preserve the old packet, claim/outcome/STOP and all evidence; independent review precedes reconciliation.'}
    pin=ops.canonical(spec);ops.validate_spec(spec,pin)
    (packet/'RECOVERY.json').write_text(json.dumps(spec,sort_keys=True,indent=2)+'\n')
    authorization=json.loads((startup/'packet/authorization.json').read_text())
    ops.validate_authorization(spec,authorization)
    shutil.copyfile(startup/'packet/authorization.json',packet/'authorization.json')
    authpin=ops.digest(packet/'authorization.json')
    assert authpin==ops.ORIGINAL_AUTH
    ops.verify_prior_startup(packet/'prior-startup',spec,authorization)
    old=ROOT/'runs/pretrained/diagnostic/two-host-panel-20261026'
    py='/home/kdu/.venvs/vjepa/bin/python'
    check='''import hashlib,json,sys;from pathlib import Path
p=Path(sys.argv[1]);q=json.loads((p/'RECOVERY.json').read_text())
assert hashlib.sha256(json.dumps(q,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()=='''+repr(pin)+'''
assert hashlib.sha256((p/'authorization.json').read_bytes()).hexdigest()=='''+repr(authpin)+'''
for n,s in q['sources_sha256'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==s,n
'''
    merge='''import json,sys,xml.etree.ElementTree as E;from pathlib import Path
p=Path(sys.argv[1]);r=E.Element('testsuites');counts=[]
for name,expected in [('old',76),('new',__NEW_GUARDS__)]:
 x=E.parse(p/('cpu-'+name+'.xml')).getroot();c={k:sum(int(s.get(k,0)) for s in x.iter('testsuite')) for k in ['tests','failures','errors','skipped']}
 assert c=={'tests':expected,'failures':0,'errors':0,'skipped':0},c
 counts.append(c)
 for s in x.iter('testsuite'):r.append(s)
E.ElementTree(r).write(p/'cpu-results.xml',encoding='unicode')
(p/'cpu.log').write_text((p/'cpu-old.log').read_text()+'\\n'+(p/'cpu-new.log').read_text())
(p/'cpu-summary.log').write_text(json.dumps({k:sum(c[k] for c in counts) for k in counts[0]},sort_keys=True)+'\\n')
print((p/'cpu-summary.log').read_text(),end='')
'''.replace('__NEW_GUARDS__',str(NEW_GUARDS))
    pipeline='''#!/usr/bin/env bash
set -euo pipefail
output=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
base=$(cd -- "$output/.." && pwd)
py="$HOME/.venvs/vjepa/bin/python"
started=$("$py" -B -c 'import time;print(time.monotonic())')
original="$HOME/'''+ops.BASE+'''/two-host-execution-20261026/unpacked/two-host-execution-20261026"
package="$HOME/'''+ops.BASE+'''/unpacked/matched-panel-20261024"
export CONTENDED_TEST_ORIGINAL="$original" TWO_HOST_TEST_PACKAGE="$package" PYTHONPATH="$package"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$py" -B -c '''+shlex.quote(check)+''' "$base"
echo 'Checking original 76 CPU guards and __NEW_GUARDS__ corrected recovery guards before any GPU model.'
"$py" -B -m pytest -q -p no:cacheprovider --rootdir="$original" "$original/tests/test_two_host_panel.py" \
 --basetemp="$output/cpu-old-fixtures" --junitxml="$output/cpu-old.xml" > "$output/cpu-old.log" 2>&1
cat "$output/cpu-old.log"
"$py" -B -m pytest -q -p no:cacheprovider --rootdir="$base" "$base/tests/test_contended_panel_recovery.py" \
 --basetemp="$output/cpu-new-fixtures" --junitxml="$output/cpu-new.xml" > "$output/cpu-new.log" 2>&1
cat "$output/cpu-new.log"
"$py" -B -c '''+shlex.quote(merge)+''' "$output"
exec "$py" -B -u "$base/scripts/recover_contended_panel_case.py" run --proposal-sha256 '''+pin+''' --authorization-sha256 '''+authpin+''' --launcher-started "$started"
'''.replace('__NEW_GUARDS__',str(NEW_GUARDS))
    launch=(ROOT/'runs/pretrained/diagnostic/panel-continuation-20261026/linux10603/remote-launch.sh').read_text()
    launch=launch.replace('"$output/../.."','"$output/.."').replace('clip-04-bank-1 launcher','clip-04-bank-1 recovery launcher')
    launch=launch.replace('4800 bash',str(ops.CASE_LIMIT)+' bash').replace('80-minute-limit','remaining-79m24s-limit')
    (packet/'remote-launch.sh').write_text(launch);(packet/'remote-pipeline.sh').write_text(pipeline)
    sys.path.insert(0,str(ROOT))
    from models.pilot_bundle import make_archive,extract_verified
    files={ops.PREFIX+'/'+str(p.relative_to(packet)):p for p in packet.rglob('*') if p.is_file()}
    archive=raw/'recovery-operator.tar.gz';manifest=make_archive(archive,files)
    (raw/'operator.manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    extract_verified(archive,manifest,raw/'extracted')
    payload={'kind':'matched-panel-contended-lockfix-upload-v1','archive_base64':base64.b64encode(archive.read_bytes()).decode(),'manifest':manifest}
    (raw/'upload-payload.json').write_text(json.dumps(payload,sort_keys=True)+'\n')
    installer=(ROOT/'scripts/install_two_host_packet.py').read_text()
    installer=installer.replace("socket.gethostname() not in ('linux10609','linux10603')","socket.gethostname() != 'linux10603'")
    installer=installer.replace('matched-panel-two-host-source-upload-v1','matched-panel-contended-lockfix-upload-v1')
    installer=installer.replace('two-host-execution-20261026',ops.PREFIX)
    (raw/'install-only.py').write_text(installer)
    encoded=base64.b64encode(installer.encode()).decode();command=shlex.join([py,'-B','-c','exec(__import__("base64").b64decode('+repr(encoded)+'))',ops.digest(raw/'upload-payload.json')])
    (raw/'install-command.txt').write_text(command+'\n');shutil.copyfile(old/'known_hosts',raw/'known_hosts')
    remote=ops.BASE+'/'+ops.PREFIX+'/unpacked/'+ops.PREFIX
    bootstrap='''import hashlib,json,os,socket;from pathlib import Path
assert socket.gethostname()=='linux10603'
p=Path.home()/'''+repr(remote)+'''
expected='''+repr({n:ops.digest(packet/n) for n in ['RECOVERY.json','authorization.json','remote-launch.sh','remote-pipeline.sh',*ops.SOURCES]})+'''
for n,s in expected.items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==s,n
out=p/'launcher';assert not out.is_symlink();os.umask(0o077);out.mkdir(exist_ok=False)
for n,source in [('source-launch.sh','remote-launch.sh'),('source-pipeline.sh','remote-pipeline.sh')]:
 with (out/n).open('xb') as f:f.write((p/source).read_bytes())
os.execv('/bin/bash',['bash',str(out/'source-launch.sh')])
'''
    (raw/'bootstrap.py').write_text(bootstrap)
    (raw/'launch-command.txt').write_text(shlex.join([py,'-B','-c',bootstrap])+'\n')
    stream=(old/'linux10603/stream-evidence.py').read_text().replace('clip-04-bank-0',ops.CASE).replace('revision-evidence.tar.gz','recovery-evidence.tar.gz')
    stream=stream.replace('two-host-execution-20261026',ops.PREFIX).replace("'attempt-01'","'attempt-02'")
    a=stream.index('    files =');b=stream.index('    inventory = {}',a)
    filemap={'evidence/recovery-evidence.tar.gz':"output/'recovery-evidence.tar.gz'",
        'evidence/recovery-evidence.tar.gz.manifest.json':"output/'recovery-evidence.tar.gz.manifest.json'",
        'ledger/STOP.json':"root/'STOP.json'",'ledger/recovery-claim.json':"root/'recovery-grants'/CASE_ID/'claim.json'",
        'ledger/recovery-outcome.json':"root/'recovery-grants'/CASE_ID/'outcome.json'",
        'ledger/failed-outcome.json':"root/'outcomes'/(CASE_ID+'.json')",'ledger/failed-session.json':"root/'cases'/CASE_ID/'attempt-01/session.json'",
        'operation/RECOVERY.json':"packet/'RECOVERY.json'",'operation/authorization.json':"packet/'authorization.json'",
        'operation/biking-independent-review.json':"packet/'biking-independent-review.json'"}
    for name in ['started-utc.txt','ended-utc.txt','exit-status.txt','case.log','cpu.log','cpu-results.xml','cpu-summary.log','cpu-old.log','cpu-new.log','cpu-old.xml','cpu-new.xml','source-launch.sh','source-pipeline.sh']:
        filemap['launcher/'+name]="packet/'launcher'/"+repr(name)
    for name in ops.SOURCES:filemap['operation/source/'+name]='packet/'+repr(name)
    for name in (*ops.PRIOR_FILES,'transport-manifest.json'):filemap['operation/prior-startup/'+name]='packet/'+repr('prior-startup/'+name)
    files='    files = {\n'+''.join('        '+repr(n)+':'+expr+',\n' for n,expr in filemap.items())+'    }\n'
    stream=stream[:a]+files+stream[b:]
    stream=stream.replace('read-only-two-host-case-download-v1','read-only-contended-recovery-download-v1')
    stream=stream.replace('revision_sha256','proposal_sha256').replace(ops.RP,pin)
    stream=stream.replace("'remote_files_changed':False,","'authorization_file_sha256':"+repr(authpin)+",'remote_files_changed':False,")
    (raw/'stream-evidence.py').write_text(stream)
    receiver=(old/'linux10603/receive-evidence.py').read_text().replace('clip-04-bank-0',ops.CASE).replace('revision-evidence.tar.gz','recovery-evidence.tar.gz')
    a=receiver.index('EXPECTED =');b=receiver.index('\n\n\ndef digest',a)
    receiver=receiver[:a]+'EXPECTED = '+repr(set(filemap))+receiver[b:]
    receiver=receiver.replace('read-only-two-host-case-download-v1','read-only-contended-recovery-download-v1').replace('two-host-case-download-byte-verification-v1','contended-recovery-download-byte-verification-v1')
    receiver=receiver.replace('revision_sha256','proposal_sha256').replace(ops.RP,pin).replace("manifest['attempt'] != 'attempt-01'","manifest['attempt'] != 'attempt-02' or manifest['authorization_file_sha256'] != "+repr(authpin))
    (raw/'receive-evidence.py').write_text(receiver)
    download=(ROOT/'runs/pretrained/diagnostic/panel-continuation-20261026/linux10603/download-evidence.sh').read_text()
    download=download.replace(str(ROOT/'runs/pretrained/diagnostic/panel-continuation-20261026/linux10603'),str(raw)).replace('$base/../known_hosts','$base/known_hosts')
    (raw/'download-evidence.sh').write_text(download)
    pins={n:ops.digest(raw/n) for n in ['upload-payload.json','install-command.txt','launch-command.txt','known_hosts','stream-evidence.py','receive-evidence.py','download-evidence.sh']}
    helper='''#!/usr/bin/env bash
set -euo pipefail
base='''+shlex.quote(str(raw))+'''
if [[ "${CONTENDED_PANEL_AWAKE:-0}" != 1 ]]; then
 exec /usr/bin/caffeinate -i env CONTENDED_PANEL_AWAKE=1 bash "$0"
fi
PYTHONDONTWRITEBYTECODE=1 '''+shlex.quote(str(ROOT/'.venv/bin/python'))+''' - "$base" <<'PIN_CHECK'
from pathlib import Path
import hashlib,sys
p=Path(sys.argv[1]);expected='''+repr(pins)+'''
for n,s in expected.items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==s,n
print('Same original grant: 79m24s remain on linux10603; old startup, evidence and global STOP are preserved.')
PIN_CHECK
output=$(mktemp -d "$base/attended-session-$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
printf '%s\\n' "$output" > "$base/latest-session.txt"
install_command=$(cat "$base/install-command.txt")
ssh -T -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$base/known_hosts" -o GlobalKnownHostsFile=/dev/null \\
 -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 kdu@linux10603.dc.engr.scu.edu "$install_command" < "$base/upload-payload.json"
echo 'Keep this Terminal and your Mac lid open until the one recovery ends. There is no automatic next case.'
launch_command=$(cat "$base/launch-command.txt")
set +e
ssh -t -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$base/known_hosts" -o GlobalKnownHostsFile=/dev/null \\
 -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 kdu@linux10603.dc.engr.scu.edu "$launch_command" 2>&1 | tee "$output/terminal.log"
codes=("${PIPESTATUS[@]}")
set -e
printf '%s\\n' "${codes[0]}" > "$output/job-connection-exit-status.txt"
if [[ "${codes[0]}" -ne 0 || "${codes[1]}" -ne 0 ]]; then
 echo "The recovery returned ${codes[0]}. Preserve $output and report the result; never repeat this launch." >&2
 exit 1
fi
if ! bash "$base/download-evidence.sh" "$output"; then
 echo 'Retrying only the download once into a fresh folder. The model will not restart.'
 sleep 3
 bash "$base/download-evidence.sh"
fi
'''
    (raw/'run-recovery.sh').write_text(helper)
    prepared={'kind':ops.KIND+'-preparation','proposal_sha256':pin,'authorization_file_sha256':authpin,'grant_id':authorization['grant_id'],
        'packet_archive':manifest,'source_sha256':spec['sources_sha256'],'helper_sha256':ops.digest(raw/'run-recovery.sh'),
        'preserved_coordinator_files':len(preserved),'failed_attempt_files':len(spec['failed_attempt_files_sha256']),
        'CPU_guards_expected':76+NEW_GUARDS,'new_CPU_guards_expected':NEW_GUARDS,'case_seconds':ops.CASE_LIMIT,'kill_grace_seconds':15,
        'original_approved_seconds':4800,'prior_startup_charge_seconds':ops.STARTUP_CHARGE,
        'same_original_authorization_and_grant':True,'new_grant_issued':False,
        'prior_startup_regular_files':len(ops.PRIOR_FILES)+1,
        'parent_seconds':4620,'matrix_seconds':3600,'replay_seconds':900,'GPU_jobs_started':0,'remote_upload_performed':False,
        'global_STOP_to_remain_active':True,'automatic_queue':False,'automatic_GPU_retry':False,'paid_compute':False,'research_admitted':False}
    (raw/'preparation.json').write_text(json.dumps(prepared,sort_keys=True,indent=2)+'\n')
    for p in raw.rglob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
    for p in raw.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
    return raw,prepared


if __name__=='__main__':
    raw,result=build();print(json.dumps({k:v for k,v in result.items() if k!='packet_archive'},sort_keys=True,indent=2))
