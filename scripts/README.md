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
and atomic per-clip recovery. `check_workflow.py` adds bounded clean feature/probe,
training-only calibration, and scheduled training/recovery diagnostics. Research
execution and full-model CUDA workflow validation remain pending. See
[staged data and evaluation](../docs/staged_data_and_evaluation.md).

Use `python scripts/check_workflow.py --help` and the
[workflow guide](../docs/workflow_diagnostics.md) for exact commands. Its clean
feature command passed with the released model on eight training clips on CPU.
Calibration and scheduled recovery have small-model CPU integration tests;
full-model GPU execution still needs a bounded pilot.

`profile_cuda.py` measures 12 synthetic upstream component cases with fresh
processes, default/strict policies, forward/backward timing, gradient signatures,
and bounded optional operator profiling. `university_profile.sh` verifies the
source-only handoff and starts it detached using the existing university Python
and upstream checkout. See the [component guide](../docs/cuda_component_profile.md).
No checkpoint, video, optimizer, installation, or cloud launch is involved.
The university RTX 3070 component export is verified: all 48 workers completed
and strict replay passes all 12 cases. This is isolated evidence, not full-model
recovery. See the [outcome](../report/checks/2026-10-21-university-cuda-components.md).

`compare_cuda_convolution.py` compares the same original Conv3d under contiguous,
channels-last-3D, and explicit native-backend execution. A CPU FP64 analytic
reference checks outputs and input/weight/bias gradients at fixed tolerances
before timing. All modes retain strict determinism. `university_convolution.sh`
launches the checksummed source handoff detached with worker/overall bounds.
The original TF32-on CUDA run rejected all options at the accuracy gate; its
cuDNN reference itself fails the fixed FP64 comparison. The separately tagged
`--cudnn-tf32 off` control passed all 12 workers and observed replay on the
university GPU. Native execution is a much faster isolated component candidate;
full-model integration/recovery and study adoption remain pending. See the
[comparison guide](../docs/cuda_convolution_comparison.md).

`check_native_model.py` and the detached `university_native.sh` implement the
next bounded native-Conv3d integration check, using the existing university
checkpoint and training cache. `check_workflow.py` and `run_workflow_pilot.py`
now accept an explicit `--execution-policy`; `original` remains the default.
See [native execution](../docs/native_execution.md) for backend identity,
first-order limits, the university handoff and pending full-model recovery.

`check_gradient_attribution.py` runs the derivative control with explicit
`--diagnostic`, a fresh output directory and a 300-second maximum worker.
`university_gradient.sh` verifies the source handoff and starts a detached
330-second supervisor using existing university assets and the saved failed
run. See [the guide](../docs/native_gradient_control.md) for transfer/launch.

`check_precision.py` and `university_precision.sh` implement the separate
matched-FP64-forward control: explicit diagnostic scope, first training clip,
original pinned assets, four fresh workers, numerical/replay gates, saved raw
references and a detached bounded launcher. See
[the guide](../docs/shared_forward_control.md) before university execution.

`check_workflow.py` and `run_workflow_pilot.py` also accept explicit diagnostic
`shared-fp64-cudnn` and `shared-fp64-native` policies. Defaults remain `original`.
See [the training integration](../docs/shared_forward_training.md); accepting
an option does not establish a new offline GPU handoff or deployment approval.

`start_shared_workflow.sh` verifies the restored source and launches bounded
setup/preflight/recovery/export on an already selected Pod with a reviewed
deployment-based deadline. It never stops GPU billing. `export_workflow_pilot.py`
can opt into six bounded preflight tensor artifacts while excluding optimizer
states. See [the prepared pilot](../docs/shared_forward_cuda_pilot.md).
