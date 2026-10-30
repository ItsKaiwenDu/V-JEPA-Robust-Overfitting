#!/usr/bin/env bash
# Foreground only: keep this SSH connection and the Mac awake. No detach/retry.
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# == 4 ]] || { echo 'Usage: bash scripts/university_projection_reuse.sh HANDOFF_SHA PLAN_FILE PLAN_SHA EXISTING_REPOSITORY' >&2; exit 2; }
handoff_sha=$1
plan=$2
plan_sha=$3
existing=$4
py="$HOME/.venvs/vjepa/bin/python"
[[ -x "$py" && -f "$plan" && -f "$existing/checkpoints/vitl.pt" && -d "$existing/external/vjepa2/.git" ]] || { echo 'Existing environment/assets and a reviewed plan are required.' >&2; exit 1; }
command -v timeout >/dev/null
started=${REUSE_LAUNCH_STARTED:-$(date +%s)}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$py" - "$base" "$handoff_sha" "$plan" "$plan_sha" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=root/'REUSE-HANDOFF.json';plan=pathlib.Path(sys.argv[3])
if digest(manifest)!=sys.argv[2] or digest(plan)!=sys.argv[4]:raise SystemExit('Handoff or launch plan checksum mismatch.')
for name,item in json.loads(manifest.read_text())['files'].items():
    p=root/name
    if p.is_symlink() or not p.resolve().is_relative_to(root) or p.stat().st_size!=item['bytes'] or digest(p)!=item['sha256']:
        raise SystemExit('Handoff member changed: '+name)
v=json.loads(plan.read_text())
if v.get('authorized') is not True or v.get('attended') is not True or v.get('paid_compute') is not False or v.get('max_seconds')!=4800:
    raise SystemExit('Require the reviewed attended university plan; preparation is not authorization.')
print('Verified handoff and launch plan. Keep this SSH connection open.',flush=True)
PY
mkdir -p "$base/runs"
run=$(mktemp -d "$base/runs/projection-reuse.XXXXXX")
printf '%s\n' "$run" > "$base/last-run-location.txt"
trap 'status=$?; printf "%s\n" "$status" > "$run/exit_status.txt"; echo "Attended projection diagnostic exited $status; preserve $run"' EXIT
echo "Run directory: $run"
echo 'One attended attempt; 80 minutes inclusive plus five-second outer kill grace. No optimizer or automatic retry.'
remaining=$((started + 4800 - $(date +%s)))
[[ $remaining -gt 0 ]] || { echo 'Launcher deadline already elapsed.' >&2; exit 1; }
timeout --signal=TERM --kill-after=5s "$remaining" "$py" -B -u "$base/scripts/check_projection_reuse.py" \
  --diagnostic --device cuda --handoff-sha256 "$handoff_sha" \
  --upstream "$existing/external/vjepa2" --checkpoint "$existing/checkpoints/vitl.pt" \
  --plan "$plan" --plan-sha256 "$plan_sha" --launcher-started "$started" \
  --max-seconds 4800 --worker-seconds 1080 --output "$run/results" \
  > >(tee "$run/launcher.log") 2>&1
