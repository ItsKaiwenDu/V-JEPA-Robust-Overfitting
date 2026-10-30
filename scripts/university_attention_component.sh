#!/usr/bin/env bash
# Foreground only. Keep the SSH connection and local Mac awake.
set -euo pipefail
[[ $# == 5 ]] || { echo 'Expected HANDOFF_SHA PLAN_FILE PLAN_SHA EXISTING_UPSTREAM ROLE' >&2; exit 2; }
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
handoff_sha=$1
plan=$2
plan_sha=$3
upstream=$4
role=$5
py="$HOME/.venvs/vjepa/bin/python"
[[ -x "$py" && -f "$plan" && -d "$upstream/.git" ]]
command -v timeout >/dev/null
started=${ATTENTION_LAUNCH_STARTED:-$(date +%s)}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$py" - "$base" "$handoff_sha" "$plan" "$plan_sha" "$role" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=root/'ATTENTION-HANDOFF.json';plan=pathlib.Path(sys.argv[3])
if sha(manifest)!=sys.argv[2] or sha(plan)!=sys.argv[4]:raise SystemExit('Attention manifest or plan changed.')
for name,item in json.loads(manifest.read_text())['files'].items():
    p=root/name
    if p.is_symlink() or not p.resolve().is_relative_to(root) or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
        raise SystemExit('Attention source changed: '+name)
v=json.loads(plan.read_text())
if (v.get('authorized') is not True or v.get('paid_compute') is not False or v.get('attended') is not True
        or v.get('max_seconds')!=1200 or v.get('role')!=sys.argv[5]):
    raise SystemExit('Unarmed or wrong-role preparation is not a launch plan.')
print('Verified attention sources and plan. Keep SSH open.',flush=True)
PY
mkdir -p "$base/attention-runs"
run=$(mktemp -d "$base/attention-runs/attention.XXXXXX")
printf '%s\n' "$run" > "$base/last-attention-run-location.txt"
trap 'status=$?; printf "%s\n" "$status" > "$run/exit_status.txt"; echo "Attended components exited $status; preserve $run"' EXIT
remaining=$((started + 1200 - $(date +%s)))
[[ $remaining -gt 0 ]]
echo "Run directory: $run"
echo 'Independent fresh component workers; total 20 minutes plus five-second kill grace. No retry.'
timeout --signal=TERM --kill-after=5s "$remaining" "$py" -B -u "$base/scripts/check_attention_component.py" \
  --diagnostic --device cuda --role "$role" --handoff-sha256 "$handoff_sha" --upstream "$upstream" \
  --plan "$plan" --plan-sha256 "$plan_sha" --launcher-started "$started" \
  --max-seconds 1200 --worker-seconds 120 --output "$run/results" \
  > >(tee "$run/launcher.log") 2>&1
