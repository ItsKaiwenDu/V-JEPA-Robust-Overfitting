#!/usr/bin/env bash
# Reuse existing environment, pinned upstream, checkpoint and training pixels.
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
py="$HOME/.venvs/vjepa/bin/python"
if [[ ${1:-} == --worker ]]; then
  existing=$2
  run=$3
  trap 'status=$?; printf "%s\n" "$status" > "$run/exit_status.txt"; date -u; echo "Native full-model input diagnostic exit status: $status"' EXIT
  export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  hostname
  date -u
  nvidia-smi
  timeout --signal=TERM --kill-after=5s 900 "$py" -u "$base/scripts/check_native_model.py" \
    --diagnostic --upstream "$existing/external/vjepa2" --checkpoint "$existing/checkpoints/vitl.pt" \
    --clips "$existing/data/processed/pilot-train-v1" --output "$run/results" \
    --max-seconds 840 --worker-seconds 180 --max-memory-gib 7
  exit
fi
[[ $# -le 1 ]] || { echo 'Usage: bash scripts/university_native.sh [existing-university-repository]' >&2; exit 2; }
[[ -x "$py" ]] || { echo "Missing existing environment: $py" >&2; exit 1; }
existing=${1:-"$(dirname "$base")/V-JEPA-Robust-Overfitting"}
for path in external/vjepa2/.git checkpoints/vitl.pt data/processed/pilot-train-v1/index.json; do
  [[ -e "$existing/$path" ]] || { echo "Missing existing input: $existing/$path" >&2; exit 1; }
done
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
print('Verified native full-model diagnostic handoff source files.')
PY
mkdir -p "$base/runs"
run=$(mktemp -d "$base/runs/native.XXXXXX")
nohup bash "$base/scripts/university_native.sh" --worker "$existing" "$run" > "$run/launcher.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run/launcher.pid"
echo "Started detached training-only input/attack diagnostic on $(hostname), launcher PID $pid."
echo "Run directory: $run"
echo "Watch: tail -f '$run/launcher.log'"
echo "Result: $run/results/session.json"
echo 'You can exit SSH. Reconnect to the same host; timeout or partial results are not passes.'
