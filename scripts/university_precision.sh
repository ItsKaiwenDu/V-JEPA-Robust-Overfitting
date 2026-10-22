#!/usr/bin/env bash
# Matched forward control; existing assets only; no optimizer or rental.
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
py="$HOME/.venvs/vjepa/bin/python"
if [[ ${1:-} == --worker ]]; then
  existing=$2
  reference=$3
  run=$4
  trap 'status=$?; printf "%s\n" "$status" > "$run/exit_status.txt"; date -u; echo "Shared-forward control exit status: $status"' EXIT
  export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  hostname
  date -u
  nvidia-smi
  timeout --signal=TERM --kill-after=5s 900 "$py" -u "$base/scripts/check_precision.py" \
    --diagnostic --upstream "$existing/external/vjepa2" --checkpoint "$existing/checkpoints/vitl.pt" \
    --clips "$existing/data/processed/pilot-train-v1" --observed "$base/OBSERVED.json" \
    --reference-run "$reference" --output "$run/results" --worker-seconds 180 --max-seconds 840
  exit
fi
[[ $# -le 2 ]] || { echo 'Usage: bash scripts/university_precision.sh [existing-repository] [original-failed-native-run]' >&2; exit 2; }
[[ -x "$py" ]] || { echo "Missing existing environment: $py" >&2; exit 1; }
existing=${1:-"$(dirname "$base")/V-JEPA-Robust-Overfitting"}
reference=${2:-"$(dirname "$base")/cuda-native-model-fix-20261021/runs/native.5S6yHj"}
for path in external/vjepa2/.git checkpoints/vitl.pt data/processed/pilot-train-v1/index.json; do
  [[ -e "$existing/$path" ]] || { echo "Missing existing input: $existing/$path" >&2; exit 1; }
done
[[ -f "$reference/results/session.json" ]] || { echo "Missing preserved reference: $reference" >&2; exit 1; }
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
print('Verified shared-forward handoff source and observed-evidence manifest.')
PY
mkdir -p "$base/runs"
run=$(mktemp -d "$base/runs/precision.XXXXXX")
nohup bash "$base/scripts/university_precision.sh" --worker "$existing" "$reference" "$run" > "$run/launcher.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run/launcher.pid"
echo "Started detached shared-forward control on $(hostname), launcher PID $pid."
echo "Run directory: $run"
echo "Watch: tail -f '$run/launcher.log'"
echo "Result: $run/results/session.json"
echo 'Four fresh workers, 180 seconds each; scheduling deadline 840s; outer timeout 900s + 5s kill grace.'
echo 'At most 7 GiB CUDA allocator; original training clip and clean target only; no optimizer.'
echo 'You can exit SSH. Reconnect to the same host. Partial/rejected controls are not passes.'
