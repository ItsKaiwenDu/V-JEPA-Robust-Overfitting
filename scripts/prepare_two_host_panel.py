"""Build the separately identified two-host operator and manual Mac helpers."""
import base64
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from models.pilot_bundle import make_archive,extract_verified,digest
spec=importlib.util.spec_from_file_location('two_host_build',REPO/'models/two_host_panel.py')
ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
raw=REPO/'runs/pretrained/diagnostic/two-host-panel-20261026'
raw.mkdir(exist_ok=False);(raw/'validation').mkdir()
packet=raw/'staged'/ops.PREFIX;packet.mkdir(parents=True)
for name in ops.SOURCES:
    path=packet/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,path)
panel=REPO/'runs/pretrained/diagnostic/matched-panel-attended-clip-03-bank-0-20261025/case-download-20261026T055507Z.4WxZ3I'
bridge=REPO/'runs/pretrained/diagnostic/cross-host-replay-10603-20261025/case-download-20261026T061851Z.ZycTiC'
for src,name in [(panel/'verified-panel-mirror/journal.json','bootstrap-journal.json'),
    (panel/'attended-review-acceptance.json','bootstrap-review.json'),(bridge/'shadow-review-acceptance.json','bridge-review.json')]:
    shutil.copyfile(src,packet/name)
revision=json.loads((packet/'configs/matched_panel_two_host_v1.json').read_text());rp=ops.canonical(revision)
fixture=json.loads((packet/'tests/fixtures/two_host_bootstrap_metadata.json').read_text())
manifest={'kind':ops.MANIFEST_KIND,'revision_sha256':rp,'reference_execution':fixture['reference_execution'],
          'source_sha256':{n:digest(packet/n) for n in ops.SOURCES}}
mp=ops.canonical(manifest);(packet/'IMPLEMENTATION.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
(packet/'requests').mkdir();request_pins={}
shadow=REPO/'runs/pretrained/diagnostic/cross-host-replay-10603-20261025'
for host,cid in ops.FIRST_BATCH.items():
    request={'kind':ops.REQUEST_KIND,'revision_sha256':rp,'implementation_sha256':mp,'host':host,'case_id':cid,
        'case_seconds':4800,'kill_grace_seconds':15,'max_attempts':1,'one_case_only':True,'additional_allowance_seconds':0,
        'automatic_queue':False,'paid_compute':False,'research_admitted':False}
    qp=ops.canonical(request);request_pins[cid]=qp;ops.write_record(packet/'requests'/(cid+'.json'),request)
    common=f'--package "$package" --root "$root" --revision-sha256 {rp} --implementation-sha256 {mp} --request "$base/requests/{cid}.json" --request-sha256 {qp}'
    pipeline='''#!/usr/bin/env bash
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
py="$HOME/.venvs/vjepa/bin/python"
package="$HOME/'''+ops.PACKAGE+'''"
root="$HOME/'''+ops.COORDINATOR+'''"
output="$base/launchers/'''+host+'-'+cid+'''"
export TWO_HOST_TEST_PACKAGE="$package" PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo 'Phase: verify fixed paths, unchanged sources and actual idle assigned GPU.'
"$py" -B -u "$base/scripts/check_two_host_panel.py" preflight '''+common+'''
echo 'Phase: CPU coordinator guards. These tests qualify no CUDA case.'
"$py" -B -m pytest -q -rs -p no:cacheprovider --rootdir="$base" "$base/tests/test_two_host_panel.py" \
  --basetemp="$output/cpu-fixtures" --junitxml="$output/cpu-results.xml" > "$output/cpu.log" 2>&1
cat "$output/cpu.log"
"$py" -B -c 'import json,sys,xml.etree.ElementTree as E;r=E.parse(sys.argv[1]).getroot();c={k:sum(int(s.get(k,0)) for s in r.iter("testsuite")) for k in ("tests","failures","errors","skipped")};assert c=={"tests":76,"failures":0,"errors":0,"skipped":0},c;print(json.dumps(c,sort_keys=True))' "$output/cpu-results.xml" > "$output/cpu-summary.log"
cat "$output/cpu-summary.log"
echo 'Phase: one original-limit case, A/B/C/D then a fresh A-only process, then stop for review.'
exec "$py" -B -u "$base/scripts/check_two_host_panel.py" run '''+common+'''\n'''
    (packet/('pipeline-'+host+'.sh')).write_text(pipeline)
    launch=(shadow/'staged/cross-host-replay-10603-20261025/launch-shadow.sh').read_text()
    launch=launch.replace('output="$base/launcher"',f'output="$base/launchers/{host}-{cid}"\nmkdir -p "$base/launchers"')
    launch=launch.replace('Shadow launcher ended with exit %s. No panel cases admitted.',f'{host} {cid} launcher ended with exit %s. Stop for independent review.')
    launch=launch.replace('Running only the separate 20-minute candidate reference replay.',f'Running one attended 80-minute-limit case on {host}.')
    launch=launch.replace('1200 bash "$base/run-pipeline.sh"',f'4800 bash "$base/pipeline-{host}.sh"')
    launch=launch.replace('shadow.log','case.log')
    (packet/('launch-'+host+'.sh')).write_text(launch)

files={str(p.relative_to(packet)):p for p in packet.rglob('*') if p.is_file()}
archive=raw/'two-host-operator.tar.gz';archive_manifest=make_archive(archive,{ops.PREFIX+'/'+n:p for n,p in files.items()})
(raw/'operator.manifest.json').write_text(json.dumps(archive_manifest,sort_keys=True,indent=2)+'\n')
extract_verified(archive,archive_manifest,raw/'extracted')
payload={'kind':'matched-panel-two-host-source-upload-v1','archive_base64':base64.b64encode(archive.read_bytes()).decode(),'manifest':archive_manifest}
(raw/'upload-payload.json').write_text(json.dumps(payload,sort_keys=True)+'\n')
shutil.copyfile(REPO/'scripts/install_two_host_packet.py',raw/'install-shared-packet.py')
encoded=base64.b64encode((raw/'install-shared-packet.py').read_bytes()).decode()
installer='exec(__import__("base64").b64decode('+repr(encoded)+'))'
command=shlex.join(['/home/kdu/.venvs/vjepa/bin/python','-B','-c',installer,digest(raw/'upload-payload.json')])
(raw/'install-command.txt').write_text(command+'\n')
key=(shadow/'known_hosts').read_text().split(' ',1)[1].strip()
(raw/'known_hosts').write_text(''.join(host+'.dc.engr.scu.edu '+key+'\n' for host in ops.HOSTS))

# Reuse the independently checked transport structure, with the exact new inventory.
for host,cid in ops.FIRST_BATCH.items():
    local=raw/host;local.mkdir()
    expected={'evidence/revision-evidence.tar.gz','evidence/revision-evidence.tar.gz.manifest.json'}
    expected|={'launcher/'+n for n in ('started-utc.txt','ended-utc.txt','exit-status.txt','case.log','cpu.log','cpu-results.xml','cpu-summary.log')}
    expected|={'ledger/'+n+'.json' for n in ('claim','outcome','export','genesis','imports')}
    expected|={'launcher/source-launch.sh','launcher/source-pipeline.sh'}
    stream=(shadow/'stream-shadow-evidence.py').read_text()
    start=stream.index("PLAN_SHA =");end=stream.index('\ndef digest',start)
    stream=stream[:start]+f"REVISION_SHA = {rp!r}\nHOST = {host!r}\nCASE_ID = {cid!r}\nCASE_ARCHIVE = 'revision-evidence.tar.gz'\nLIMIT = 300 * 2**20\n\n"+stream[end:]
    start=stream.index("    if socket.gethostname()");end=stream.index('    inventory = {}',start)
    stream=stream[:start]+'''    if socket.gethostname() != HOST:raise ValueError('Wrong assigned host for evidence copy.')
    packet = Path.home() / '''+repr(ops.PACKET)+'''
    root = Path.home() / '''+repr(ops.COORDINATOR)+'''
    output = root / 'cases' / CASE_ID / 'attempt-01'
    files = {'evidence/'+CASE_ARCHIVE:output/CASE_ARCHIVE,
             'evidence/'+CASE_ARCHIVE+'.manifest.json':output/(CASE_ARCHIVE+'.manifest.json')}
    for name in ('started-utc.txt','ended-utc.txt','exit-status.txt','case.log','cpu.log','cpu-results.xml','cpu-summary.log'):
        files['launcher/'+name]=packet/'launchers'/(HOST+'-'+CASE_ID)/name
    for name,path in {'claim':root/'claims'/CASE_ID/'claim.json','outcome':root/'outcomes'/(CASE_ID+'.json'),
        'export':root/'exports'/(CASE_ID+'.json'),'genesis':root/'genesis.json','imports':root/'imports.json'}.items():
        files['ledger/'+name+'.json']=path
    files['launcher/source-launch.sh']=packet/('launch-'+HOST+'.sh')
    files['launcher/source-pipeline.sh']=packet/('pipeline-'+HOST+'.sh')
'''+stream[end:]
    stream=stream.replace('not path.resolve().is_relative_to(packet.resolve())','not any(path.resolve().is_relative_to(r.resolve()) for r in (packet,root))')
    stream=stream.replace("'read-only-cross-host-shadow-download-v1'","'read-only-two-host-case-download-v1'")
    stream=stream.replace("'shadow_plan_sha256':PLAN_SHA","'revision_sha256':REVISION_SHA,'case_id':CASE_ID")
    stream=stream.replace('SHADOW_REPLAY_BUNDLE ','TWO_HOST_CASE_BUNDLE ')
    stream=stream.replace('Streaming completed shadow evidence','Streaming completed two-host case evidence')
    (local/'stream-evidence.py').write_text(stream)
    receive=(shadow/'receive-shadow-evidence.py').read_text().replace("CASE_ARCHIVE = 'evidence.tar.gz'","CASE_ARCHIVE = 'revision-evidence.tar.gz'")
    start=receive.index('EXPECTED =');end=receive.index('\n\n\ndef digest',start)
    receive=receive[:start]+'EXPECTED = '+repr(expected)+receive[end:]
    receive=receive.replace('SHADOW_REPLAY_BUNDLE ','TWO_HOST_CASE_BUNDLE ')
    receive=receive.replace('read-only-cross-host-shadow-download-v1','read-only-two-host-case-download-v1')
    receive=receive.replace("manifest['host'] != 'linux10603'",f"manifest['host'] != {host!r}")
    receive=receive.replace("manifest['shadow_plan_sha256'] != '66f8595088397e1ca0b56fd69c5bc9b82842feec9c3fe39c6f83a369ec40d1c4'",f"manifest['revision_sha256'] != {rp!r} or manifest['case_id'] != {cid!r}")
    receive=receive.replace('cross-host-shadow-download-byte-verification-v1','two-host-case-download-byte-verification-v1')
    receive=receive.replace("'independent_shadow_review_complete'","'independent_case_review_complete'")
    (local/'receive-evidence.py').write_text(receive)
    download=(shadow/'download-shadow-evidence.sh').read_text()
    download=download.replace(str(shadow),str(local)).replace('download-shadow-evidence.sh','download-evidence.sh')
    download=download.replace('stream-shadow-evidence.py','stream-evidence.py').replace('receive-shadow-evidence.py','receive-evidence.py')
    download=download.replace('completed shadow replay','completed case').replace('linux10603.dc.engr.scu.edu',host+'.dc.engr.scu.edu')
    download=download.replace('"$base/known_hosts"','"$base/../known_hosts"')
    (local/'download-evidence.sh').write_text(download)
    helper=(shadow/'run-shadow-replay.sh').read_text()
    start=helper.index('PYTHONDONTWRITEBYTECODE=1');end=helper.index('output=$(mktemp',start)
    pins={n:digest(raw/n) for n in ('upload-payload.json','install-command.txt','known_hosts')}
    pins.update({host+'/'+n:digest(local/n) for n in ('stream-evidence.py','receive-evidence.py','download-evidence.sh')})
    check='''PYTHONDONTWRITEBYTECODE=1 '''+shlex.quote(str(REPO/'.venv/bin/python'))+''' - "$base/.." <<'PIN_CHECK'
import hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1]);expected='''+repr(pins)+'''
for name,pin in expected.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==pin,name
print('Two-host source packet and case-specific helpers verified.')
PIN_CHECK
'''
    helper=helper[:start]+check+helper[end:]
    helper=helper.replace(str(shadow),str(local)).replace('"$base/install-command.txt"','"$base/../install-command.txt"')
    helper=helper.replace('"$base/upload-payload.json"','"$base/../upload-payload.json"').replace('"$base/known_hosts"','"$base/../known_hosts"')
    helper=helper.replace('linux10603.dc.engr.scu.edu',host+'.dc.engr.scu.edu')
    helper=helper.replace("Only two shadow reference workers will run, then stop; no panel cases start.",f'Only {cid} will run on {host}, then stop for independent review.')
    helper=helper.replace('launch_command=$(cat "$base/launch-command.txt")',
        'launch_command='+shlex.quote('bash /home/kdu/'+ops.PACKET+'/launch-'+host+'.sh'))
    helper=helper.replace('download-shadow-evidence.sh','download-evidence.sh')
    (local/'run-case.sh').write_text(helper)
    (local/'review-command.txt').write_text(shlex.join([str(REPO/'.venv/bin/python'),'-B',str(raw/'extracted'/ops.PREFIX/'scripts/check_two_host_panel.py'),
        'review','--package',str(REPO/'runs/pretrained/diagnostic/matched-panel-handoff-20261024/snapshot'),
        '--revision-sha256',rp,'--implementation-sha256',mp])+'\n')

preparation={'kind':'two-host-panel-offline-preparation-v1','date_Pacific':'2026-10-26','revision_sha256':rp,'implementation_sha256':mp,
    'archive':archive_manifest,'requests_sha256':request_pins,'first_batch':ops.FIRST_BATCH,
    'fresh_user_readonly_checks':{'linux10609':{'GPU_identity':ops.HOSTS['linux10609'],'memory_used_MiB':74,'compute_processes':[]},
        'linux10603':{'GPU_identity':ops.HOSTS['linux10603'],'memory_used_MiB':94,'compute_processes':[]}},
    'accepted_cases_before_batch':7,'new_GPU_jobs_started':0,'uploads_performed':0,'paid_compute':False,'research_admitted':False}
(raw/'preparation.json').write_text(json.dumps(preparation,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in preparation.items() if k!='archive'},indent=2))
print('Packet:',archive_manifest['bytes'],'bytes;',len(archive_manifest['files']),'members;',archive_manifest['sha256'])
