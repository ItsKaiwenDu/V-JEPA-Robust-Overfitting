#!/usr/bin/env python3
"""Check a local official V-JEPA 2 checkpoint on one synthetic FP32 batch.

No network requests, dataset downloads, training jobs, or cloud actions.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=ROOT / "external/vjepa2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", help="Optional independently recorded checkpoint SHA-256.")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; existing files are never overwritten.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The output report already exists; choose a new run name.")
    if not args.checkpoint.is_file():
        parser.error("Checkpoint not found. Download it explicitly using docs/smoke_test.md.")

    import torch
    from models.diagnostics import run_smoke_checks
    from models.vjepa2 import UPSTREAM_COMMIT, build_vitl, load_checkpoint, sample_masks, use_upstream

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this environment.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this environment.")
    torch.manual_seed(args.seed)
    upstream = use_upstream(args.upstream)
    print("Building the ViT-L/16 encoder, fixed target, and released predictor...", flush=True)
    model = build_vitl(upstream)
    print("Hashing and strictly auditing the local checkpoint...", flush=True)
    checkpoint = load_checkpoint(model, args.checkpoint, args.expected_sha256)
    model.to(args.device)
    clean = torch.rand(1, 3, 16, 256, 256, generator=torch.Generator().manual_seed(args.seed))
    masks = [pair.to(args.device) for pair in sample_masks(upstream, batch_size=1, seed=args.seed)]
    print(f"Checking one synthetic FP32 batch on {args.device}...", flush=True)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    checks = run_smoke_checks(model, clean.to(args.device), masks, upstream)
    report = {"kind": "synthetic_pretrained_smoke_test", "research_result": False,
              "upstream_commit": UPSTREAM_COMMIT, "checkpoint": checkpoint,
              "device": args.device, "seed": args.seed, "python": platform.python_version(),
              "packages": {name: importlib.metadata.version(name)
                           for name in ("torch", "torchvision", "timm", "einops", "numpy", "PyYAML")},
              "project_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "project_dirty": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()),
              "mask_shapes": [{"context": list(p.context.shape), "target": list(p.target.shape)} for p in masks],
              "checks": checks}
    if args.device == "cuda":
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["gpu"] = torch.cuda.get_device_name()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"PASS: checkpoint/native-loss smoke test. Report: {args.output}")
    print("This is a synthetic correctness check, not training readiness or a feasibility measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
