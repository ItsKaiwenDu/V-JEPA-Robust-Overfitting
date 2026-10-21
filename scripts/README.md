# Command-line scripts

Keep this interface familiar to the previous repository. Reusable model and
attack logic should be importable and shared, not duplicated across scripts.

## Available now

```bash
python scripts/verify_setup.py
python scripts/verify_setup.py --check-torch
python scripts/verify_setup.py --require-cuda
python scripts/verify_setup.py --require-mps
```

The first command checks scaffold files, directories, and configuration YAML.
The optional modes inspect an already installed PyTorch runtime. A missing
dependency or a required unavailable device produces a nonzero exit
code. These checks are read-only and do not establish model correctness.

`smoke_test.py` implements strict local V-JEPA 2 checkpoint loading and a
one-batch synthetic loss/gradient check. It requires an upstream checkout,
model dependencies, and an explicitly downloaded checkpoint. It writes a
new JSON report and refuses to overwrite an existing one. Full pretrained
CPU and MPS execution have passed. Run `python scripts/smoke_test.py --help` and
follow [the model-check guide](../docs/smoke_test.md).

`prepare_data.py`, `prepare_clips.py`, and `check_attacks.py` implement
grouped manifests, deterministic training/validation clips, and a one-clip
pretrained attack sanity check. See [data and attacks](../docs/data_and_attacks.md).
`check_mps.py` compares CPU and Apple GPU losses, targets, input gradients,
and selected parameter gradients on one real clip. `check_patch_embedding.py`
isolates the original and equivalent linear projection on MPS. `pilot.py` supports
explicit `--device cuda` (default) and `--device mps` timing. The October 19 RTX 5090 pilot passed; the new recovery/evaluation commands
still need full-model GPU validation. See [the CUDA guide](../docs/gpu_pilot.md) and
[Mac guide](../docs/mac_pilot.md).

`estimate_compute.py` reads the separate compute proposal and existing Mac
reports to reproduce workload counts and partial runtime estimates. It does
not launch models or cloud jobs. Missing costs stay unknown, stronger-attack
timing proxies stay labeled, and no estimate authorizes spending. See the
[planning document](../docs/evaluation_compute_plan.md).

## Upcoming entry points

| Name | Responsibility |
| --- | --- |
| `run_report_checks.py` | Gradient, projection, masking, and attack-strength diagnostics |
| `plot_results.py` | Per-seed curves, aggregate figures, and gap summaries |

The planned scripts are not created as nonfunctional placeholders. Add them
with implementation and meaningful checks. Preserve familiar condition,
seed, run-name, and diagnostic-mode concepts, with resolved configurations
saved alongside every run. Document actual command syntax when implemented.

`estimate_staged_study.py` reproduces the October 20 budget alternatives and
metadata-only proposed subset audit without a GPU or network. See the
[staged plan](../docs/budget_study_plan.md). It never admits a run.

`train.py` implements a bounded diagnostic over the FP32 training/resume core.
It requires verified training clips and a fresh output directory. See
[training and recovery](../docs/training_runner.md); full research execution
remains gated on evaluation, data validation, and measured cost.

`export_staged_manifests.py` replays and exports the proposed identities.
`prepare_staged_clips.py` extracts only training/exploration sources and verifies
two decodes. `evaluate.py` supports bounded native-attack checks, saved anchors,
and atomic per-clip recovery. Full research orchestration and downstream probes
remain pending. See [staged data and evaluation](../docs/staged_data_and_evaluation.md).
