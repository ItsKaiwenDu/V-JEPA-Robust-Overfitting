#!/usr/bin/env bash
# Existing assets only. No install, downloads, optimizer, or paid compute.
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
py="$HOME/.venvs/vjepa/bin/python"
if [[ ${1:-} == --worker ]]; then
  existing=$2
  observed_run=$3
  run=$4
  trap 'status=$?; printf "%s\n" "$status" > "$run/exit_status.txt"; date -u; echo "Derivative control exit status: $status"' EXIT
  export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  hostname
  date -u
  nvidia-smi
  timeout --signal=TERM --kill-after=5s 330 "$py" -u "$base/scripts/check_gradient_attribution.py" \
    --diagnostic --upstream "$existing/external/vjepa2" --checkpoint "$existing/checkpoints/vitl.pt" \
    --clips "$existing/data/processed/pilot-train-v1" --observed "$base/OBSERVED.json" \
    --reference-run "$observed_run" --output "$run/results" --max-seconds 300
  exit
fi
[[ $# -le 2 ]] || { echo 'Usage: bash scripts/university_gradient.sh [existing-repository] [original-failed-run]' >&2; exit 2; }
[[ -x "$py" ]] || { echo "Missing existing environment: $py" >&2; exit 1; }
existing=${1:-"$(dirname "$base")/V-JEPA-Robust-Overfitting"}
observed_run=${2:-"$(dirname "$base")/cuda-native-model-fix-20261021/runs/native.5S6yHj"}
for path in external/vjepa2/.git checkpoints/vitl.pt data/processed/pilot-train-v1/index.json; do
  [[ -e "$existing/$path" ]] || { echo "Missing existing input: $existing/$path" >&2; exit 1; }
done
[[ -f "$observed_run/results/session.json" ]] || { echo "Missing original failed run: $observed_run" >&2; exit 1; }
for tool in timeout nohup; do command -v "$tool" >/dev/null; done
"$py" - "$base" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
manifest=root/'MANIFEST.json'
if not manifest.is_file():raise SystemExit('Use the prepared source handoff with MANIFEST.json.')
for name,expected in json.loads(manifest.read_text())['files_sha256'].items():
    path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit('Handoff checksum mismatch: '+name)
print('Verified derivative-control source and observed-evidence manifest.')
PY
mkdir -p "$base/runs"
run=$(mktemp -d "$base/runs/derivative.XXXXXX")
nohup bash "$base/scripts/university_gradient.sh" --worker "$existing" "$observed_run" "$run" > "$run/launcher.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run/launcher.pid"
echo "Started detached derivative control on $(hostname), launcher PID $pid."
echo "Run directory: $run"
echo "Watch: tail -f '$run/launcher.log'"
echo "Detailed progress: tail -f '$run/results/control.log'"
echo "Result: $run/results/control.json"
echo 'Worker deadline: 300 seconds; outer deadline: 330 seconds plus 5-second kill grace.'
echo '7 GiB CUDA allocator cap; one existing training clip; no optimizer or held-out pixels.'
echo 'You can exit SSH. Reconnect to the same host; incomplete/rejected controls are not passes.'
