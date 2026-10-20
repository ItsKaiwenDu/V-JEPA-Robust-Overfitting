#!/usr/bin/env bash
# Run on the university Linux host, beside the two verified handoff archives.
set -euo pipefail
phase=${1:-}
case "$phase" in setup|smoke) ;; *) echo 'Usage: bash university_check.sh setup|smoke' >&2; exit 2 ;; esac
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo="$base/V-JEPA-Robust-Overfitting"
py="$HOME/.venvs/vjepa/bin/python"
uv="$HOME/.local/bin/uv"
pin=204698b45b3712590f06245fbfba32d3be539812
weight_hash=5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012
[[ -x "$py" ]] || { echo "Missing environment: $py" >&2; exit 1; }
mkdir -p "$base/logs"
log=$(mktemp "$base/logs/$phase.XXXXXX")
exec > >(tee "$log") 2>&1
trap 'status=$?; echo "Exit status: $status; log: $log"' EXIT
hostname
date -u
"$py" - <<'PY'
import torch, torchvision
print('PyTorch:', torch.__version__, 'torchvision:', torchvision.__version__)
assert torch.__version__ == '2.8.0+cu128', 'Unexpected PyTorch version'
assert torchvision.__version__ == '0.23.0+cu128', 'Unexpected torchvision version'
assert torch.cuda.is_available(), 'CUDA unavailable'
print('GPU:', torch.cuda.get_device_name(0))
PY
cd "$base"
if [[ "$phase" == setup ]]; then
  for tool in git curl sha256sum tar; do command -v "$tool" >/dev/null; done
  sha256sum -c - <<'HASHES'
3dae83f7adb9cefebf01eb3e46bf5104ffa037c3a7f9377fdb33efd664a86e6b  vjepa-pilot-source.tar.gz
93b7b5d9b1b85d8e65e4ce186a4cb6c0e2444a106c4f5589297553c16adc9af6  vjepa-pilot-clips.tar.gz
HASHES
  if [[ ! -d "$repo" ]]; then
    tar -xzf vjepa-pilot-source.tar.gz
    git clone vjepa-pilot-source/repository.bundle "$repo"
    cp -R vjepa-pilot-source/working-tree/. "$repo/"
    tar -xzf vjepa-pilot-clips.tar.gz -C "$repo"
  fi
  "$py" - "$base/manifest.json" "$repo" <<'PY'
import hashlib, json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2])
for group in ('source_files_sha256', 'prepared_clip_files_sha256'):
    for name, expected in manifest[group].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f'Hash mismatch: {name}; preserve this folder and use a new handoff directory.')
    print(f'Verified {len(manifest[group])} files: {group}')
PY
  cd "$repo"
  if [[ ! -d external/vjepa2 ]]; then
    git clone https://github.com/facebookresearch/vjepa2.git external/vjepa2
    git -C external/vjepa2 checkout --detach "$pin"
  fi
  [[ "$(git -C external/vjepa2 rev-parse HEAD)" == "$pin" ]]
  [[ -z "$(git -C external/vjepa2 status --porcelain)" ]]
  constraints=$(mktemp "$base/torch-constraints.XXXXXX")
  printf '%s\n' 'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128' > "$constraints"
  "$uv" pip install --python "$py" --no-cache --link-mode copy \
    --constraint "$constraints" -r requirements-smoke.txt
  "$py" scripts/verify_setup.py --require-cuda
  "$py" -m pytest -q -rs tests --upstream external/vjepa2
  mkdir -p checkpoints
  if [[ ! -f checkpoints/vitl.pt ]]; then
    curl --fail --location --retry 3 --connect-timeout 30 --max-time 3600 \
      --output checkpoints/vitl.pt.part https://dl.fbaipublicfiles.com/vjepa2/vitl.pt
    printf '%s  %s\n' "$weight_hash" checkpoints/vitl.pt.part | sha256sum -c -
    mv checkpoints/vitl.pt.part checkpoints/vitl.pt
  else
    printf '%s  %s\n' "$weight_hash" checkpoints/vitl.pt | sha256sum -c -
  fi
  echo 'SETUP PASS. Model CUDA smoke test has NOT run. Next: bash university_check.sh smoke'
else
  command -v timeout >/dev/null
  cd "$repo"
  nvidia-smi
  mkdir -p runs/pretrained/diagnostic
  output=$(mktemp -d runs/pretrained/diagnostic/university-cuda.XXXXXX)
  cp "$base/university_check.sh" "$output/launcher.sh"
  "$uv" pip freeze --python "$py" > "$output/packages.txt"
  # Bound this correctness check; no optimizer, attack pilot, or training run starts here.
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout --signal=TERM --kill-after=30s 20m \
    "$py" -u scripts/smoke_test.py --checkpoint checkpoints/vitl.pt \
    --device cuda --patch-embedding conv3d --expected-sha256 "$weight_hash" \
    --output "$output/smoke.json"
  echo "MODEL SMOKE PASS: $repo/$output/smoke.json"
  echo 'Optimizer/attack memory and throughput are still unmeasured.'
fi
