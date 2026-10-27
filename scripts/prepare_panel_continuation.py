#!/usr/bin/env python3
"""Prepare a bounded stop resolution and two requests; start no remote work."""
import base64
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.pilot_bundle import make_archive,extract_verified,digest


def build():
    spec=importlib.util.spec_from_file_location('canonical_ops',ROOT/'scripts/continue_two_host_panel.py')
    ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
    raw=ROOT/'runs/pretrained/diagnostic/panel-continuation-20261026';raw.mkdir(exist_ok=False)
    (raw/'validation').mkdir();packet=raw/'staged/two-host-continuation-20261026';packet.mkdir(parents=True)
    for name in ops.SOURCES:
        source=ROOT/('scripts/sync_two_host_bookkeeping.py' if name.startswith('support/') else name)
        target=packet/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    prior=ROOT/'runs/pretrained/diagnostic/two-host-bookkeeping-sync-20261026'
    session=Path((prior/'latest-session.txt').read_text().strip())
    assert digest(session/'receipt.json')==ops.PRIOR
    shutil.copyfile(session/'receipt.json',packet/'prior-sync-receipt.json')
    shutil.copyfile(prior/'upload-payload.json',packet/'reviewed-sync-upload.json')
    assert digest(packet/'reviewed-sync-upload.json')==ops.UPLOAD
    operation={'kind':ops.KIND,'date_Pacific':'2026-10-26','numeric_plan_changed':False,
        'reference_revision_sha256':ops.RP,'reference_implementation_sha256':ops.MP,'prior_sync_receipt_sha256':ops.PRIOR,
        'resolved_stop_sha256':ops.STOP,'accepted_before':ops.ACCEPTED,'targets':ops.TARGETS,'root_path_mode':'canonical',
        'case_seconds':4800,'kill_grace_seconds':15,'matrix_seconds':3600,'replay_seconds':900,'parent_seconds':4620,
        'max_attempts':1,'additional_allowance_seconds':0,'automatic_GPU_retries':0,'automatic_queue':False,
        'one_case_per_host':True,'paid_compute':False,'research_admitted':False,
        'sources_sha256':{name:digest(packet/name) for name in ops.SOURCES},
        'scope':'Archive and resolve only the reviewed export alias stop; append the two fixed requests. No model runs during activation. Run each new case once through the unchanged numerical runner with a canonical root.'}
    pin=ops.canonical(operation);ops.validate_operation(operation,pin)
    (packet/'OPERATION.json').write_text(json.dumps(operation,sort_keys=True,indent=2)+'\n')
    archive=raw/'continuation-operator.tar.gz'
    entries={'two-host-continuation-20261026/'+str(p.relative_to(packet)):p for p in packet.rglob('*') if p.is_file()}
    manifest=make_archive(archive,entries);(raw/'operator.manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    extract_verified(archive,manifest,raw/'extracted')
    payload={'kind':'matched-panel-canonical-continuation-upload-v1','operation_sha256':pin,
        'archive_base64':base64.b64encode(archive.read_bytes()).decode(),'manifest':manifest}
    (raw/'upload-payload.json').write_text(json.dumps(payload,sort_keys=True)+'\n')
    installer=(ROOT/'scripts/install_two_host_packet.py').read_text()
    installer=installer.replace("socket.gethostname() not in ('linux10609','linux10603')","socket.gethostname() != 'linux10609'")
    installer=installer.replace('matched-panel-two-host-source-upload-v1','matched-panel-canonical-continuation-upload-v1')
    installer=installer.replace('two-host-execution-20261026','two-host-continuation-20261026')
    installer=installer.replace('check=True,timeout=30)','check=True,timeout=30,stdout=sys.stderr)')
    installer=installer.replace("print('Shared two-host packet installed/verified. No GPU job started.',flush=True)",
        "print('Operational packet installed/verified; activating metadata only.',file=sys.stderr,flush=True)\n    packet=target/'unpacked/two-host-continuation-20261026'\n    os.execv(sys.executable,[sys.executable,'-B',str(packet/'scripts/continue_two_host_panel.py'),'activate','--operation-sha256',payload['operation_sha256']])")
    assert 'os.execv' in installer
    (raw/'install-and-activate.py').write_text(installer)
    encoded=base64.b64encode(installer.encode()).decode();body='exec(__import__("base64").b64decode('+repr(encoded)+'))'
    command=shlex.join(['timeout','--signal=TERM','--kill-after=15s','300','env','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',
        '/home/kdu/.venvs/vjepa/bin/python','-B','-c',body,digest(raw/'upload-payload.json')])
    (raw/'activation-command.txt').write_text(command+'\n');shutil.copyfile(prior/'known_hosts',raw/'known_hosts')
    verify='''"""Check actual activation record bytes without model execution."""
from pathlib import Path
import importlib.util,json,hashlib,sys
base=Path(__file__).resolve().parent;output=Path(sys.argv[1]);packet=base/'extracted/two-host-continuation-20261026'
s=importlib.util.spec_from_file_location('activation_check',packet/'scripts/continue_two_host_panel.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
operation=json.loads((packet/'OPERATION.json').read_text());receipt=json.loads((output/'receipt.json.part').read_text())
result=m.verify_activation(receipt,operation,'''+repr(pin)+''')
result['receipt_sha256']=hashlib.sha256((output/'receipt.json.part').read_bytes()).hexdigest()
with (output/'receipt-verification.json').open('x') as stream:stream.write(json.dumps(result,sort_keys=True,indent=2)+'\\n')
(output/'receipt.json.part').rename(output/'receipt.json')
print(json.dumps(result,sort_keys=True,indent=2))
'''
    (raw/'verify-activation.py').write_text(verify)
    pins={n:digest(raw/n) for n in ['upload-payload.json','activation-command.txt','verify-activation.py','known_hosts']}
    pins.update({'extracted/two-host-continuation-20261026/'+name:digest(packet/name) for name in ops.SOURCES})
    pins['extracted/two-host-continuation-20261026/OPERATION.json']=digest(packet/'OPERATION.json')
    helper='''#!/usr/bin/env bash
set -euo pipefail
base='''+shlex.quote(str(raw))+'''
if [[ "${PANEL_CONTINUATION_AWAKE:-0}" != 1 ]]; then
  exec /usr/bin/caffeinate -i env PANEL_CONTINUATION_AWAKE=1 bash "$0"
fi
py='''+shlex.quote(str(ROOT/'.venv/bin/python'))+'''
PYTHONDONTWRITEBYTECODE=1 "$py" - "$base" <<'PIN_CHECK'
from pathlib import Path
import hashlib,sys
root=Path(sys.argv[1]);expected='''+repr(pins)+'''
for name,pin in expected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==pin,name
print('Verified: archive only the reviewed stop and prepare two requests. No GPU model starts.')
PIN_CHECK
output=$(mktemp -d "$base/activation-session-$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
printf '%s\\n' "$output" > "$base/latest-activation.txt"
command=$(cat "$base/activation-command.txt")
set +e
ssh -T -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$base/known_hosts" -o GlobalKnownHostsFile=/dev/null \\
 -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \\
 kdu@linux10609.dc.engr.scu.edu "$command" < "$base/upload-payload.json" > "$output/receipt.json.part" 2> "$output/activation.stderr"
code=$?
set -e
printf '%s\\n' "$code" > "$output/connection-exit-status.txt"
cat "$output/activation.stderr"
if [[ "$code" -ne 0 ]]; then
 echo "Activation did not confirm success. Report the saved result at $output before repeating; no GPU launcher ran." >&2
 exit "$code"
fi
PYTHONDONTWRITEBYTECODE=1 "$py" "$base/verify-activation.py" "$output"
echo "Metadata activation verified at $output. Both GPU cases still await separate manual launch. Reply Done."
'''
    (raw/'activate-continuation.sh').write_text(helper)
    prepared={'kind':'canonical-continuation-preparation-v1','date_Pacific':'2026-10-26','operation_sha256':pin,
        'source_sha256':operation['sources_sha256'],'archive':manifest,'payload_sha256':digest(raw/'upload-payload.json'),
        'payload_bytes':(raw/'upload-payload.json').stat().st_size,'helper_sha256':digest(raw/'activate-continuation.sh'),
        'next_cases':ops.TARGETS,'activation_is_metadata_only':True,'remote_activation_performed':False,'GPU_jobs_started':0}
    (raw/'preparation.json').write_text(json.dumps(prepared,sort_keys=True,indent=2)+'\n')
    return raw,prepared


if __name__=='__main__':
    raw,prepared=build();print(json.dumps({k:v for k,v in prepared.items() if k!='archive'},indent=2))
