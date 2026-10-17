#!/usr/bin/env python3
"""Check the repository scaffold; optionally inspect an installed PyTorch runtime.

No downloads, file writes, model loads, or training are performed.
"""

from __future__ import annotations

import argparse
import importlib
import platform
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "README.md", "LICENSE", ".gitignore", "AGENTS.md", "requirements.txt", "requirements-smoke.txt",
    "goals.md", "progress.md", "setup_lambda_labs.md",
    "configs/study.yaml", "configs/local.example.yaml",
    "docs/experimental_protocol.md", "docs/setup.md",
    "docs/repository_structure.md", "docs/reproducibility.md", "docs/decisions.md",
    "models/__init__.py", "models/README.md", "models/vjepa2.py", "models/diagnostics.py",
    "scripts/README.md", "scripts/smoke_test.py", "tests/test_vjepa2.py", "docs/smoke_test.md",
    "attacks/__init__.py", "attacks/pgd.py", "datasets/__init__.py", "datasets/ucf101.py",
    "datasets/video.py", "models/pilot.py", "scripts/prepare_data.py", "scripts/prepare_clips.py",
    "scripts/check_attacks.py", "scripts/pilot.py", "docs/data_and_attacks.md", "docs/gpu_pilot.md",
    "tests/test_attacks.py", "tests/test_ucf101.py", "tests/test_video.py",
    "notes/README.md", "notes/2026-10-16_novelty-audit.md",
    "report/main.tex", "report/references.bib", "report/README.md",
    "data/manifests/README.md", "external/README.md", "tests/README.md",
    ".github/workflows/repository-checks.yml",
)
REQUIRED_DIRS = (
    "assets", "data/raw", "data/processed", "data/features",
    "checkpoints", "runs", "run_logs", "report/pretrained",
    "report/clean-finetune", "report/adversarial-finetune",
    "report/checks", "report/figures", "report/tables",
)


def check_scaffold() -> list[str]:
    errors = []
    for name in REQUIRED_FILES:
        path = ROOT / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty file: {name}")
    for name in REQUIRED_DIRS:
        if not (ROOT / name).is_dir():
            errors.append(f"Missing directory: {name}")
    try:
        yaml = importlib.import_module("yaml")
    except ImportError:
        errors.append("PyYAML is missing. Install requirements.txt in your environment.")
        return errors

    for name in ("configs/study.yaml", "configs/local.example.yaml"):
        try:
            value = yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                errors.append(f"{name} must contain a YAML mapping.")
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"Cannot read {name}: {exc}")
    return errors


def check_torch(require_cuda: bool) -> list[str]:
    try:
        torch = importlib.import_module("torch")
        print(f"PyTorch: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())
        print(f"CUDA available: {cuda_available}")
        print(f"MPS available: {mps_available}")
        print(f"PyTorch CUDA build: {torch.version.cuda}")
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                print(f"GPU {index}: {props.name}, {props.total_memory / 2**30:.1f} GiB")
        if require_cuda and not cuda_available:
            return ["CUDA is required for this check but is unavailable."]
    except Exception as exc:
        return [f"PyTorch inspection failed: {type(exc).__name__}: {exc}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-torch", action="store_true", help="Inspect installed PyTorch.")
    parser.add_argument("--require-cuda", action="store_true", help="Inspect PyTorch and require CUDA.")
    args = parser.parse_args()

    print(f"Repository: {ROOT}")
    print(f"Python: {platform.python_version()} ({platform.machine()})")
    if sys.version_info < (3, 11):
        print("FAIL: Use Python 3.11 or newer for the scaffold checker.")
        return 1
    errors = check_scaffold()
    if args.check_torch or args.require_cuda:
        errors.extend(check_torch(args.require_cuda))
    else:
        print("PyTorch: not checked; use --check-torch or --require-cuda explicitly.")
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        return 1
    print("PASS: Scaffold files and YAML syntax are valid.")
    print("This scaffold check does not certify GPU training readiness; run the CUDA feasibility pilot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
