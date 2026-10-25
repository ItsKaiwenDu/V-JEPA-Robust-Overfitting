#!/usr/bin/env bash
# Detached bounded university execution; no installs, downloads or cloud API.
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
py="$HOME/.venvs/vjepa/bin/python"
existing="$HOME/Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

verify_handoff() {
  "$py" - "$base" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
m=json.loads((root/'PANEL-HANDOFF.json').read_text())
assert m['kind']=='matched-panel-handoff-v1' and not m['research_admitted'] and m['training_pixel_tensors']==10
for name,v in m['files'].items():
    p=root/name
    if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file() or p.stat().st_size!=v['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=v['sha256']:
        raise SystemExit('Handoff mismatch: '+name)
print('Panel handoff verified:',len(m['files']),'files')
PY
}

if [[ ${1:-} == --worker ]]; then
  mode=$2; run=$3; acceptance=$4; case_id=$5
  trap 'status=$?; printf "%s\n" "$status" > "$run/exit_status.txt"; date -u; echo "Panel command exit status: $status"' EXIT
  verify_handoff
  hostname
  date -u
  nvidia-smi
  options=("$mode" --diagnostic --checkpoint "$existing/checkpoints/vitl.pt" --output "$run")
  [[ "$acceptance" == - ]] || options+=(--acceptance "$acceptance")
  [[ "$case_id" == - ]] || options+=(--case-id "$case_id")
  "$py" -u "$base/scripts/check_matched_panel.py" "${options[@]}"
  verify_handoff
  exit
fi
[[ $(uname -s) == Linux && -x "$py" ]] || { echo 'Use the existing university Linux environment.' >&2; exit 1; }
for tool in git flock timeout nohup nvidia-smi; do command -v "$tool" >/dev/null; done
[[ -f "$existing/checkpoints/vitl.pt" ]] || { echo 'Existing released checkpoint is missing.' >&2; exit 1; }
verify_handoff
if [[ ! -e "$base/external/vjepa2" ]]; then
  mkdir -p "$base/external"
  git clone --quiet "$base/upstream.bundle" "$base/external/vjepa2"
  git -C "$base/external/vjepa2" checkout --quiet --detach 204698b45b3712590f06245fbfba32d3be539812
fi
mode=${1:-}
acceptance=-; case_id=-
mkdir -p "$base/runs"
exec 9>"$base/runs/panel-launch.lock"
flock --nonblock 9 || { echo 'Another panel command is running.' >&2; exit 1; }
case "$mode" in
  first)
    [[ $# == 3 ]] || { echo 'Usage: bash scripts/university_matched_panel.sh first EXPECTED_HOST "EXPECTED_GPU"' >&2; exit 2; }
    [[ $(hostname) == "$2" && $(nvidia-smi --query-gpu=name --format=csv,noheader) == "$3" ]] || { echo 'Host or GPU differs from the current reviewed check.' >&2; exit 1; }
    run=$(mktemp -d "$base/runs/panel.XXXXXX")
    seconds=4800
    ;;
  remaining)
    [[ $# == 3 && -d "$2" && -f "$3" ]] || { echo 'Usage: bash scripts/university_matched_panel.sh remaining PANEL_DIRECTORY FIRST_CASE_ACCEPTANCE' >&2; exit 2; }
    run=$(cd -- "$2" && pwd); acceptance=$(realpath "$3")
    # 19 remaining80m cases + all kill graces + bounded coordinator reserve.
    seconds=91785
    ;;
  retry)
    [[ $# == 4 && -d "$2" ]] || { echo 'Usage: bash scripts/university_matched_panel.sh retry PANEL_DIRECTORY CASE_ID ACCEPTANCE_OR_DASH' >&2; exit 2; }
    run=$(cd -- "$2" && pwd); case_id=$3
    [[ "$4" == - ]] || acceptance=$(realpath "$4")
    seconds=4800
    ;;
  *) echo 'Choose first, remaining or an explicitly reviewed retry.' >&2; exit 2 ;;
esac
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || { echo 'Another CUDA compute process is active.' >&2; exit 1; }
log="$run/launcher-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup timeout --signal=TERM --kill-after=15s "$seconds" bash "$base/scripts/university_matched_panel.sh" \
  --worker "$mode" "$run" "$acceptance" "$case_id" > "$log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$run/launcher.pid"
echo "Detached panel command: $mode"
echo "Panel directory: $run"
echo "Watch: tail -f '$log'"
echo "Case exports: $run/exports/"
echo 'First mode runs one case and stops for independent export review.'
echo 'Each case keeps its80-minute allowance; retries do not reset time already spent.'
echo 'Remaining mode is serial; any failure stops it. SSH can be closed.'
