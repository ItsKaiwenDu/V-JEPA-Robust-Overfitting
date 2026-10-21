#!/usr/bin/env bash
# Standalone profiler handoff; uses the existing university environment/upstream.
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
py="$HOME/.venvs/vjepa/bin/python"
if [[ ${1:-} == --worker ]]; then
  upstream=$2
  run=$3
  trap 'status=$?; printf "%s\n" "$status" > "$run/exit_status.txt"; date -u; echo "Profile exit status: $status"' EXIT
  export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  hostname
  date -u
  "$py" -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available())'
  nvidia-smi
  # Overall supervisor: 8 minutes plus startup/finalization; no optimizer/run starts.
  timeout --signal=TERM --kill-after=5s 540 "$py" -u "$base/scripts/profile_cuda.py" \
    --upstream "$upstream" --output "$run/results" \
    --max-seconds 480 --worker-seconds 20 --max-memory-gib 4
  exit
fi
[[ $# -le 1 ]] || { echo 'Usage: bash scripts/university_profile.sh [existing-upstream-path]' >&2; exit 2; }
[[ -x "$py" ]] || { echo "Missing existing environment: $py" >&2; exit 1; }
upstream=${1:-"$(dirname "$base")/V-JEPA-Robust-Overfitting/external/vjepa2"}
[[ -d "$upstream/.git" ]] || { echo "Cannot find existing upstream: $upstream" >&2; exit 1; }
for tool in timeout nohup; do command -v "$tool" >/dev/null; done
"$py" - "$base" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
manifest=root/'MANIFEST.json'
if not manifest.is_file():raise SystemExit('Run this launcher from the prepared handoff, which includes MANIFEST.json.')
v=json.loads(manifest.read_text())
for name,expected in v['files_sha256'].items():
    p=(root/name).resolve()
    if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
        raise SystemExit('Handoff checksum mismatch: '+name)
print('Verified handoff source files:',len(v['files_sha256']))
PY
mkdir -p "$base/runs"
run=$(mktemp -d "$base/runs/components.XXXXXX")
nohup bash "$base/scripts/university_profile.sh" --worker "$upstream" "$run" > "$run/launcher.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run/launcher.pid"
echo "Started detached component diagnostic on $(hostname), launcher PID $pid."
echo "Run directory: $run"
echo "Watch: tail -f '$run/launcher.log'"
echo "Result: $run/results/session.json"
echo 'You can exit SSH. Reconnect to this same host to inspect the job; its files live in your shared home.'
