#!/usr/bin/env bash
# Detached finite diagnostic on an already selected Pod; no cloud API calls.
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

finish() {
  status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$run/exit_status.txt"
  if [[ ! -f "$run/pilot/session.json" && -f "$base/runs/pretrained/diagnostic/workflow-setup/setup.log" ]]; then
    python3 - "$run" "$status" "${stage:-setup}" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])/'pilot'
root.mkdir(exist_ok=True)
payload={'schema_version':1,'diagnostic_only':True,'research_result':False,'budget_approved':False,
         'complete':False,'execution_policy':'shared-fp64-native','phases':[],
         'error':sys.argv[3]+' ended before a pilot session was created',
         'pre_session_stage':sys.argv[3],'worker_exit_status':int(sys.argv[2])}
checksum=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
(root/'session.json').write_text(json.dumps({'payload':payload,'sha256':checksum},indent=2)+'\n')
PY
  fi
  if [[ -f "$run/pilot/session.json" ]]; then
    cp "$run/launcher-metadata.json" "$run/pilot/launcher-metadata.json"
    cp "$run/exit_status.txt" "$run/pilot/worker_exit_status.txt"
    cp "$run/launcher.log" "$run/pilot/supervisor.log"
    export_py="$base/.venv/bin/python"
    [[ -x "$export_py" ]] || export_py=$(command -v python3)
    timeout --signal=TERM --kill-after=10s 180 "$export_py" \
      "$base/scripts/export_workflow_pilot.py" --run "$run/pilot" \
      --output "$run/evidence.tar.gz" --include-preflight-tensors \
      || printf '%s\n' 'Export incomplete: retain the Pod volume and inspect before termination.'
  fi
  date -u
  echo "Shared-forward diagnostic exit status: $status"
  echo 'Python has stopped. Stop the Pod in Runpod to stop GPU billing; verify copied evidence before termination.'
  exit "$status"
}

if [[ ${1:-} == --worker ]]; then
  run=$2
  deadline=$3
  trap finish EXIT
  cd "$base"
  export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  stage=setup
  remaining=$((deadline-$(date +%s)))
  setup_seconds=$((remaining-600))
  (( setup_seconds > 1500 )) && setup_seconds=1500
  (( setup_seconds > 0 )) || { echo 'No setup allowance remains.' >&2; exit 1; }
  timeout --signal=TERM --kill-after=10s "$setup_seconds" bash scripts/setup_workflow_pilot.sh
  stage=pilot
  remaining=$((deadline-$(date +%s)))
  pilot_seconds=$((remaining-240))
  (( pilot_seconds > 5400 )) && pilot_seconds=5400
  (( pilot_seconds > 0 )) || { echo 'No pilot allowance remains; setup evidence is retained.' >&2; exit 1; }
  timeout --signal=TERM --kill-after=10s "$((pilot_seconds+15))" .venv/bin/python -u \
    scripts/run_workflow_pilot.py --diagnostic --execution-policy shared-fp64-native \
    --checkpoint checkpoints/vitl.pt --upstream external/vjepa2 \
    --manifests data/manifests/ucf101-staged-20x5-v1 \
    --clips data/processed/staged-exploration-v1 --output "$run/pilot" --max-seconds "$pilot_seconds"
  exit
fi

[[ $# == 2 && $1 == --deadline-epoch && $2 =~ ^[0-9]{10}$ ]] || {
  echo 'Usage: bash scripts/start_shared_workflow.sh --deadline-epoch ABSOLUTE_UTC_EPOCH_SECONDS' >&2
  echo 'The deadline must be based on deployment time and the approved session allowance.' >&2
  exit 2
}
deadline=$2
now=$(date +%s)
remaining=$((deadline-now))
(( remaining >= 1200 && remaining <= 7200 )) || {
  echo 'Need 20 minutes to 2 hours remaining; refresh the deployment-based deadline/budget.' >&2
  exit 2
}
for tool in timeout nohup python3; do command -v "$tool" >/dev/null; done
python3 - "$base" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
manifest=json.loads((root/'handoff-manifest.json').read_text())
assert manifest['diagnostic_only'] and not manifest['research_admitted']
for name,expected in manifest['archives']['source']['files'].items():
    path=(root/name).resolve()
    if (not path.is_relative_to(root) or path.is_symlink() or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']):
        raise SystemExit('Restored source checksum differs: '+name)
print('Verified restored source files:',len(manifest['archives']['source']['files']))
PY
mkdir -p "$base/runs/pretrained/diagnostic"
run=$(mktemp -d "$base/runs/pretrained/diagnostic/shared-workflow.XXXXXX")
python3 - "$run" "$now" "$deadline" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
(root/'launcher-metadata.json').write_text(json.dumps({
    'schema_version':1,'diagnostic_only':True,'execution_policy':'shared-fp64-native',
    'launched_epoch':int(sys.argv[2]),'deadline_epoch':int(sys.argv[3]),
    'setup_cap_seconds':1500,'pilot_cap_seconds':5400,'preflight_cap_seconds':1200,
    'export_reserve_seconds':240,'deadline_origin':'user-reviewed deployment-based session allowance',
    'cloud_billing_stopped_by_launcher':False},indent=2)+'\n')
PY
nohup timeout --signal=TERM --kill-after=15s "$remaining" \
  bash "$base/scripts/start_shared_workflow.sh" --worker "$run" "$deadline" \
  > "$run/launcher.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run/launcher.pid"
echo "Started detached shared-forward pilot, supervisor PID $pid."
echo "Run directory: $run"
echo "Watch: tail -f '$run/launcher.log'"
echo "Result: $run/pilot/session.json"
echo "Export after completion/failure: $run/evidence.tar.gz"
echo 'SSH can be closed; GPU billing continues until you stop the Pod.'
