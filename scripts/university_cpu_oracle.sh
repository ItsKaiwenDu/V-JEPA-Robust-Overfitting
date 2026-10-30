#!/usr/bin/env bash
# Foreground CPU-only capture. Keep the attended SSH connection open.
set -euo pipefail
[[ $# == 4 ]] || { echo 'Expected HANDOFF_SHA PLAN_FILE PLAN_SHA EXISTING_UPSTREAM' >&2; exit 2; }
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
handoff_sha=$1
plan=$2
plan_sha=$3
upstream=$4
py="$HOME/.venvs/vjepa/bin/python"
[[ -x "$py" && -f "$plan" && -d "$upstream/.git" ]]
command -v timeout >/dev/null
started=${CPU_ORACLE_LAUNCH_STARTED:-$(date +%s)}
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$py" - "$base" "$handoff_sha" "$plan" "$plan_sha" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();plan=pathlib.Path(sys.argv[3])
manifest=root/'CPU-ORACLE-HANDOFF.json'
if sha(manifest)!=sys.argv[2] or sha(plan)!=sys.argv[4]:raise SystemExit('CPU handoff or plan changed.')
m=json.loads(manifest.read_text());v=json.loads(plan.read_text())
for n,item in m['files'].items():
 p=root/n
 if p.is_symlink() or not p.resolve().is_relative_to(root) or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise SystemExit('CPU source/fixture changed: '+n)
if (v.get('authorized') is not True or v.get('CPU_only') is not True or v.get('GPU_work_allowed') is not False
    or v.get('max_seconds')!=600 or v.get('paid_compute') is not False):raise SystemExit('A new armed CPU-only plan is required.')
print('CPU-only source/fixture and plan verified. Keep SSH open.',flush=True)
PY
mkdir -p "$base/cpu-oracle-runs"
run=$(mktemp -d "$base/cpu-oracle-runs/cpu-oracle.XXXXXX")
printf '%s\n' "$run" > "$base/last-cpu-oracle-run-location.txt"
trap 'status=$?; printf "%s\n" "$status" > "$run/exit_status.txt"; echo "CPU oracle exited $status; preserve $run"' EXIT
remaining=$((started + 600 - $(date +%s)))
[[ $remaining -gt 0 ]]
timeout --signal=TERM --kill-after=5s "$remaining" "$py" -B -u "$base/scripts/check_cpu_oracle.py" \
 --diagnostic --handoff-sha256 "$handoff_sha" --upstream "$upstream" --plan "$plan" --plan-sha256 "$plan_sha" \
 --launcher-started "$started" --output "$run/results" > >(tee "$run/launcher.log") 2>&1
