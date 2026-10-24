#!/usr/bin/env bash
# Existing university assets only. No installs, downloads, optimizer or rental.
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
py="$HOME/.venvs/vjepa/bin/python"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

verify() {
  "$py" - "$base" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
manifest=json.loads((root/'MATCHED-HANDOFF.json').read_text())
assert manifest['kind']=='matched-training-attack-handoff-v1'
assert manifest['diagnostic_only'] and not manifest['research_admitted']
for name,record in manifest['files'].items():
    path=root/name
    if (path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file()
            or path.stat().st_size!=record['bytes']
            or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']):
        raise SystemExit('Handoff identity mismatch: '+name)
print('Verified matched diagnostic handoff:',len(manifest['files']),'files')
PY
}

if [[ ${1:-} == --worker ]]; then
  [[ $# == 5 ]] || exit 2
  existing=$2; run=$3; expected_host=$4; expected_gpu=$5
  trap 'status=$?; printf "%s\n" "$status" > "$run/exit_status.txt"; date -u; echo "Matched diagnostic exit status: $status"' EXIT
  [[ $(hostname) == "$expected_host" ]]
  [[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == "$expected_gpu" ]]
  verify
  hostname
  date -u
  nvidia-smi
  "$py" -u "$base/scripts/check_matched_attack.py" --diagnostic \
    --checkpoint "$existing/checkpoints/vitl.pt" --upstream "$base/external/vjepa2" \
    --manifests "$base/data/manifests/ucf101-staged-20x5-v1" \
    --clips "$base/data/processed/matched-monitor-v1" --output "$run/results" --max-seconds 4620
  verify
  exit
fi
[[ $# == 2 || $# == 3 ]] || { echo 'Usage: bash scripts/university_matched_attack.sh EXPECTED_HOST "EXPECTED_GPU" [existing-repository]' >&2; exit 2; }
expected_host=$1; expected_gpu=$2
existing=${3:-"$HOME/Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting"}
[[ $(uname -s) == Linux && -x "$py" ]] || { echo 'Use the existing university Linux Python environment.' >&2; exit 1; }
[[ $(hostname) == "$expected_host" ]] || { echo 'Host differs from the reviewed SSH check.' >&2; exit 1; }
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == "$expected_gpu" ]] || { echo 'GPU differs from the reviewed SSH check, or multiple GPUs are exposed.' >&2; exit 1; }
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || { echo 'Another CUDA compute process is active; do not start this diagnostic yet.' >&2; exit 1; }
[[ -f "$existing/checkpoints/vitl.pt" ]] || { echo "Existing checkpoint missing: $existing/checkpoints/vitl.pt" >&2; exit 1; }
for tool in git flock timeout nohup; do command -v "$tool" >/dev/null; done
verify
if [[ ! -e "$base/external/vjepa2" ]]; then
  mkdir -p "$base/external"
  git clone --quiet "$base/upstream.bundle" "$base/external/vjepa2"
  git -C "$base/external/vjepa2" checkout --quiet --detach 204698b45b3712590f06245fbfba32d3be539812
fi
[[ $(git -C "$base/external/vjepa2" rev-parse HEAD) == 204698b45b3712590f06245fbfba32d3be539812 ]]
mkdir -p "$base/runs"
exec 9>"$base/runs/matched.lock"
flock --nonblock 9 || { echo 'A matched diagnostic already holds this launcher lock.' >&2; exit 1; }
run=$(mktemp -d "$base/runs/matched.XXXXXX")
nohup timeout --signal=TERM --kill-after=15s 4800 \
  bash "$base/scripts/university_matched_attack.sh" --worker "$existing" "$run" "$expected_host" "$expected_gpu" \
  > "$run/launcher.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$run/launcher.pid"
echo "Started detached matched attack diagnostic on $(hostname)."
echo "Run directory: $run"
echo "Watch: tail -f '$run/launcher.log'"
echo "Result: $run/results/session.json"
echo "Export: $run/results/evidence.tar.gz"
echo 'Maximum 60 minutes for the four arms + 15 minutes for fresh A-only replay; outer limit 80 minutes + 15s kill grace.'
echo 'CUDA allocator at most 7 GiB (or 90% of VRAM); 8 GB fit/runtime is unproved. Failure/timeout/missing replay is not a pass.'
echo 'Only the first declared training monitor; no optimizer or research evaluation. SSH can be closed.'
