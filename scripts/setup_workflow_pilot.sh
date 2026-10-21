#!/usr/bin/env bash
# Run only inside a restored handoff on the explicitly selected existing Pod.
set -euo pipefail
cd "$(dirname "$0")/.."
test -f handoff-manifest.json
test -d offline-wheels
mkdir -p runs/pretrained/diagnostic/workflow-setup
test ! -e runs/pretrained/diagnostic/workflow-setup/setup.log
exec > >(tee runs/pretrained/diagnostic/workflow-setup/setup.log) 2>&1
started_seconds=$SECONDS
trap 'status=$?; date -u; echo "SETUP STATUS: $status; wall seconds: $((SECONDS-started_seconds))"' EXIT
date -u
python3 - <<'PY'
import platform, sys, torch, torchvision
assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
assert sys.version_info[:2] == (3, 12), 'Use the verified Python 3.12 Runpod template.'
assert torch.__version__ == '2.8.0+cu128' and torchvision.__version__ == '0.23.0+cu128'
assert torch.cuda.is_available() and '5090' in torch.cuda.get_device_name()
print('Template CUDA stack:', torch.__version__, torchvision.__version__)
PY
test ! -e .venv
python3 -m venv --system-site-packages .venv
printf 'torch==2.8.0+cu128\ntorchvision==0.23.0+cu128\n' > runs/pretrained/diagnostic/workflow-setup/torch-constraints.txt
.venv/bin/python -m pip install --no-index --find-links offline-wheels \
  -c runs/pretrained/diagnostic/workflow-setup/torch-constraints.txt \
  -r requirements-pilot-offline.txt -r requirements-smoke.txt
.venv/bin/python -m pip freeze > runs/pretrained/diagnostic/workflow-setup/packages.txt
nvidia-smi > runs/pretrained/diagnostic/workflow-setup/nvidia-smi.txt
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
.venv/bin/python scripts/verify_setup.py --require-cuda
.venv/bin/python -m pytest -q -rs -p no:cacheprovider tests \
  --junitxml=runs/pretrained/diagnostic/workflow-setup/tests.xml
if test ! -f checkpoints/vitl.pt; then
  curl --fail --location --retry 1 --connect-timeout 20 --max-time 600 \
    --output checkpoints/vitl.pt.part https://dl.fbaipublicfiles.com/vjepa2/vitl.pt
  printf '%s  %s\n' 5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012 checkpoints/vitl.pt.part | sha256sum -c -
  mv checkpoints/vitl.pt.part checkpoints/vitl.pt
fi
printf '%s  %s\n' 5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012 checkpoints/vitl.pt | sha256sum -c -
echo 'Setup complete. The bounded workflow pilot has not started.'
