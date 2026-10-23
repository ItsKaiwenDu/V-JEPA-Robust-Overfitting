# Project guidance

## Current stage

Latest October 22 outcome: the pretrained `shared-fp64-native` CUDA workflow
completed all eleven phases. Read
report/checks/2026-10-22-shared-forward-cuda-workflow.md and its JSON. CUDA
suite: 436 passed, no skips; local suite remains 434 passed/two CUDA skips.
Three representative inputs and six retained tensor files are independently
verified. Both clean/adversarial two-update comparisons match all 5,351
logical entries exactly. Diagnostic calibration, features and PGD-50/FGSM/
random controls completed. All export members/records/receipts verify.
The user confirmed full Pod termination. The $1.688 combined summary minus
the prior $0.603 gives a provisional $1.085 increment / $2.280 cumulative.
Prior confirmed reports total $1.195; retain the $3 reservation until
Pod-specific billing settles ($5.805 unreserved shared pilot allowance).
Next: reconcile billing, rebuild measured costs, review data and freeze
precision/protocol before research admission. No study YAML or thresholds
changed; held-out pixels stayed closed. Earlier FP32 failures remain rejected.

**Historical milestone snapshots follow; their pending actions describe
those earlier checkpoints. Use the latest verified outcome above.**

Read README.md, progress.md, docs/experimental_protocol.md, and
docs/workflow_diagnostics.md before adding research code. Do not describe
planned entry points, unrun experiments, or proposed results as implemented
or verified. Keep progress.md factual.

Pretrained hardware feasibility and staged data preparation passed. The repaired
suite passes 225 tests on the CUDA host, including small independent-process
CUDA replay. The full-model workflow pilot failed exact clean recovery. Its
strict deterministic retry was stopped after a primary attack took 16x longer;
full-model recovery is still unvalidated. Read
report/checks/2026-10-21-cuda-workflow.md and docs/cuda_workflow_pilot.md.
Both evidence exports are verified locally; the user confirmed Pod termination.
Reported spending is $1.195 after the corrected $0.603 session charge.
The university component profile is verified: all 48 workers completed; all 12
strict cases replay exactly. Original Conv3d input backward rose from 1.102 ms
to 3.400 s under strict policy; default attention gradients fail exact replay.
Read report/checks/2026-10-21-university-cuda-components.md and
docs/cuda_component_profile.md. This is isolated RTX 3070 evidence, not a
full-model RTX 5090 recovery or performance result. The bounded same-Conv3d
comparison completed on CUDA, but all 12 workers rejected before timing:
even the original TF32-permitted cuDNN reference fails the FP64 accuracy gate.
Native matches FP64 but differs from that reference. Read
report/checks/2026-10-21-university-cuda-convolution.md for that failed attempt.
The explicit TF32-off control now passes all 12 workers, numerical gates, and
observed exact replay in both gradient modes. Original cuDNN forward/backward
is 3.410 / 3.431 seconds; native is 5.897 / 7.043 milliseconds (input / parameter
gradient modes). Channels-last provides no useful improvement. Read
report/checks/2026-10-21-university-cuda-convolution-precision.md and
docs/cuda_convolution_comparison.md. This resolves the isolated precision gate
and qualifies native as a component candidate; it does not establish RTX 5090
behavior or full-model correctness/speed. No study backend/precision is adopted.
Opt-in integration now preserves Conv3d parameters/state keys and scopes native
forward/backward explicitly; it supports first-order diagnostics only. Read
docs/native_execution.md and report/checks/2026-10-21-native-integration.md.
Caches, features, evaluation, calibration, anchors and recovery bind actual
backend/geometry/checkpointing/precision; incompatible old artifacts reject.
The first university integration attempt stopped at the small native probe,
before full pretrained native execution. Its observer incorrectly searched for
col2vol; the earlier verified CUDA profile uses at::native::vol2im_kernel.
Read docs/native_probe_fix.md and report/checks/2026-10-21-native-probe-fix.md.
The corrected schema-2 observer retains device-typed events before rejecting,
requires actual native CUDA launch/backward evidence, and recomputes the gate
on export. A separate native-first retry handoff passes four fresh CPU workers
after extraction. The first failed export is now copied/verified: 16 archived
source files, 32 upstream files and worker/tensor hashes match; the two cuDNN
full-model input-gradient/FGSM/PGD-5 workers replay exactly. The first native
worker failed at the old observer before full pretrained execution.
The corrected retry export native.5S6yHj is now fully verified: 17 source files,
32 upstream files, all worker/tensor hashes, both native probes, and exact
within-backend input-gradient/attack replay pass. Clean-loss absolute difference
is 1.192e-7 and passes. Input gradients reject the unchanged pointwise gate:
relative L2 0.3583%, max abs error 6.820e-4, 242,613 failed elements, 464 sign
changes. Cosine is 0.9999937; PGD outputs/independent scores differ. The entire
attempt remains rejected. Read
report/checks/2026-10-21-university-native-model-comparison.md and its JSON.
The derivative control in models/gradient_attribution.py now has a verified
CUDA outcome: derivative.aIsykm completed in 66.915s and remains rejected.
All archive/source/upstream/result/nine-artifact hashes, prior evidence and
runtime/data/weight/target/mask identities match. Both reconstructed losses and
input gradients are bitwise equal to native.5S6yHj; identical-token tail replay
is exact. All four fixed-cotangent backwards match FP64; the sole failed final
gate is FP64 forward accuracy (12 cuDNN / 35 native token values). Forward-token
changes alter incoming derivatives; 4+7 residual signs cross, but those crossings
alone are not proved to explain the whole discrepancy. Independent CPU FP64
conv/transpose-conv using the rehashed original trained projection reproduce
the saved oracle within 1.56e-13 forward / 1.25e-16 VJP. Read
report/checks/2026-10-21-university-gradient-attribution.md and its JSON.
The separate shared-forward precision control is now implemented in
models/precision_patch.py and models/precision_diagnostic.py. Both arms use
FP64 context patch accumulation cast to FP32 tokens, with original FP32
parameters/normalization/targets/tail. Only backward dispatch differs. The
reference is shared FP64 forward / FP32 tokens / cuDNN FP32 backward, not the
historical FP32 reference. A distinct temporary policy is rejected by existing
training/cache APIs; separate guards and failure restoration are tested.
The CUDA shared-forward export precision.RvqoMB is now independently verified:
source/archive/upstream/session/worker/twelve-tensor hashes, actual new native
CUDA probe in all four workers, original identity pins, all oracle gates and
reconstructions pass. Shared forward tokens and both FGSM/PGD-5 outputs are
bitwise equal across backwards; fresh-process replay is exact within each.
Full input-gradient max difference is 7.451e-8 (no failed elements/sign changes).
Median two-single-attempt PGD-5+score is 6.324s native / 40.745s cuDNN (6.44x);
clean loss/input backward 1.108s / 7.923s (7.15x). Peak allocator 2.838 GiB.
This is one training clip/mask bank on RTX 3070, not study throughput or full
AdamW/update/recovery evidence. Read
report/checks/2026-10-21-university-shared-forward.md and its JSON.
October 22 explicit diagnostic integration now passes CPU checks. Named
shared-fp64-cudnn/native policies bind schema-2 forward/precision/backward and
actual model/normalization/geometry identities into existing caches, features,
calibration, evaluation, anchors and resume. The old temporary scope remains
unknown to training APIs; old policy defaults/schema-1 identities are unchanged.
Forty-seven new small-upstream CPU cases cover full clean/adversarial AdamW
updates/exact recovery at three cuts, twelve fresh CPU processes, scheduled
evaluation recovery, failure boundaries, and rejection before weight/optimizer
restoration. Target/parameter/state-key invariants remain. Read
docs/shared_forward_training.md and report/checks/2026-10-22-shared-forward-integration.md.
The fresh shared-workflow-handoff-20261022 package is now built/restored and
verified. Read docs/shared_forward_cuda_pilot.md and
report/checks/2026-10-22-shared-workflow-handoff.md before any launch. A new
preflight checks training indices 0/4/7 under both backwards against unchanged
oracles/full-input gates, required nonzero gradients/ascent, frozen targets,
and actual new native CUDA evidence before calibration. Existing exact recovery,
evaluation, features and convergence phases follow. The detached launcher
requires a deployment-based reviewed absolute deadline and preserves failure
exports; Python exit does not stop billing. No GPU job launched. The user
reported a $1.21/hour console quote on October 22 and confirmed keeping
30 GB container plus 80 GB persistent storage at /workspace. Availability
is not explicitly confirmed. Next user action is a manual deployment when
available and ready, then current SSH endpoint/deployment time; bind the
absolute deadline to deployment. Limit two billed hours/$3 inclusive
($2.42 at the quote, $0.58 margin for unquoted charges). No automatic rental.
Full pretrained CUDA optimizer recovery is still unvalidated; original 8 GB
AdamW OOM remains. No study/YAML/tolerance adoption, paid launch or held-out pixels.
Preserve both historical FP32 failures; reported spend $1.195 / reserve $8.805.
Do not widen thresholds or adopt the backend based on cosine similarity or
speed alone. Full-model optimizer recovery, RTX 5090 behavior, research
calibration and budget admission remain pending.
The source-only derivative handoff and extracted fresh-process CPU fixture are
verified; 18 new checks cover chain-rule reconstruction, FP64 normalization,
hook/RNG restoration, and failed source/runtime/data identities. Read
report/checks/2026-10-21-gradient-attribution.md before the manual university launch.
Worker limit 300 seconds, outer limit 330 seconds + five-second kill grace,
7 GiB allocator cap. No optimizer or paid launch.
Latest local suite: 434 passed, two actual-CUDA skips. Research calibration,
measured budget admission, and research results remain pending. Confirmation/test pixels remain closed.
No cloud rental is automatic.

## Research scope

- V-JEPA 2 ViT-L/16, fixed clean target encoder, native masked L1 prediction.
- Update the context encoder and predictor; do not silently introduce EMA,
  a reinitialized predictor, a classifier training loss, or V-JEPA 2.1.
- Record protocol changes in docs/decisions.md and configs/study.yaml.
- Preserve group-separated train/validation/test manifests. Never select
  checkpoints or tune attacks on the final test set.
- Generate fresh adversaries per checkpoint. Measure epsilon in raw [0, 1]
  pixel space, and keep target inputs clean and target gradients disabled.
- Treat robust overfitting as an empirical question. Report failed checks,
  negative findings, and insufficient training budgets plainly.

## Repository conventions

- Keep the familiar root models/, scripts/, notes/, and report/ layout.
- Put reusable implementation in modules; keep command-line scripts thin.
- Use condition/seed run directories and unique names for reruns.
- Track small curated reports and relative-path manifests; keep weights,
  videos, feature caches, credentials, and raw logs out of Git.
- Pin upstream code and checkpoint hashes before any experiment claims.
- Preserve upstream licenses and attribution; our MIT license covers only
  original project code.

## Verification

Run python scripts/verify_setup.py for scaffold changes. This is not a
scientific correctness test. When implementing attacks or training, add
meaningful gradient, projection, masking, target-freezing, split-integrity,
and checkpoint-resume checks, as described in tests/README.md.

## Compute

The total project budget is at most $200, with $160 planned for compute.
Estimate the remaining experiment from measured throughput before launching
full runs. Count evaluation, idle instance time, storage, and taxes. Keep
the university GPU and local development paths usable.
