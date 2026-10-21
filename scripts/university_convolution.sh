#!/usr/bin/env bash
# Source-only diagnostic; uses the existing university environment and upstream.
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
py="$HOME/.venvs/vjepa/bin/python"
if [[ ${1:-} == --worker ]]; then
  upstream=$2
  run=$3
  tf32=$4
  trap 'status=$?; printf "%s\n" "$status" > "$run/exit_status.txt"; date -u; echo "Convolution comparison exit status: $status"' EXIT
  export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  hostname
  date -u
  echo "Declared diagnostic cuDNN TF32 policy: $tf32"
  nvidia-smi
  timeout --signal=TERM --kill-after=5s 540 "$py" -u "$base/scripts/compare_cuda_convolution.py" \
    --upstream "$upstream" --output "$run/results" \
    --cudnn-tf32 "$tf32" \
    --max-seconds 480 --worker-seconds 45 --max-memory-gib 4
  exit
fi
tf32=on
if [[ ${1:-} == --cudnn-tf32 ]]; then
  [[ $# -ge 2 && ( $2 == on || $2 == off ) ]] || { echo 'Use --cudnn-tf32 on|off.' >&2; exit 2; }
  tf32=$2
  shift 2
fi
[[ $# -le 1 ]] || { echo 'Usage: bash scripts/university_convolution.sh [--cudnn-tf32 on|off] [existing-upstream-path]' >&2; exit 2; }
[[ -x "$py" ]] || { echo "Missing existing environment: $py" >&2; exit 1; }
upstream=${1:-"$(dirname "$base")/V-JEPA-Robust-Overfitting/external/vjepa2"}
[[ -d "$upstream/.git" ]] || { echo "Cannot find existing upstream: $upstream" >&2; exit 1; }
for tool in timeout nohup; do command -v "$tool" >/dev/null; done
"$py" - "$base" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
manifest=root/'MANIFEST.json'
if not manifest.is_file():raise SystemExit('Use the prepared handoff, which includes MANIFEST.json.')
for name,expected in json.loads(manifest.read_text())['files_sha256'].items():
    path=(root/name).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit('Handoff checksum mismatch: '+name)
print('Verified convolution handoff source files.')
PY
mkdir -p "$base/runs"
run=$(mktemp -d "$base/runs/convolution.XXXXXX")
nohup bash "$base/scripts/university_convolution.sh" --worker "$upstream" "$run" "$tf32" > "$run/launcher.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run/launcher.pid"
echo "Started detached convolution comparison on $(hostname), launcher PID $pid."
echo "Run directory: $run"
echo "Watch: tail -f '$run/launcher.log'"
echo "Result: $run/results/session.json"
echo 'You can exit SSH. Reconnect to this same host to inspect the job.'
