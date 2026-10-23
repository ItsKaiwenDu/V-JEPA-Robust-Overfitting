# Scientific acceptance tests

The last full local suite run recorded **434 passes with two actual-CUDA skips**.
The [October 22 CUDA workflow](../report/checks/2026-10-22-shared-forward-cuda-workflow.md)
runs all **436 tests** with zero failures/errors/skips on the selected host.
Both bounded pretrained clean/adversarial two-update recovery comparisons
also pass. The [earlier integration milestone](../report/checks/2026-10-22-shared-forward-integration.md)
records the previous 420-test local suite.
The earlier 283-test milestone below records component coverage. The
35 convolution-comparison tests check analytic
FP64 derivatives against convolution/autograd, original Conv3d layouts and
gradient modes, fixed elementwise error gates, fresh processes, optional
profiling, deadlines, and rejection of corrupted source/runtime/tolerance/
gradient/timing evidence. Both declared TF32 policies propagate to fresh
workers; native execution preserves the precision flag, and mixed-policy
evidence is rejected. All 12 workers pass the restored TF32-off tiny CPU
roundtrip. The original TF32-on CUDA run rejected all options before timing;
the separately tagged TF32-off CUDA control now passes all 12 workers and
observed replay. These are component checks, without a full transformer,
optimizer update, or full-model recovery acceptance.

The 24 component-profiler checks retain independent gradients, representative
masks, real subprocess/descendant timeouts, changed precision/source rejection,
and fixture replay. Its separate university RTX 3070 export is verified with
all 48 workers complete and strict replay on 12 cases; this is isolated evidence.

The repaired suite has **225 passing tests on the CUDA host**, with no
failures/skips and 82 upstream FutureWarnings. The original local CPU suite had
219 passing tests; the targeted local runtime/training recheck passed 28 with
one actual-CUDA skip. The preceding 53 additions check fixed ridge probes
against an independent primal solution, training-only standardization, raw
feature diagnostics, calibration evidence/replay, and scheduled recovery.
Small pinned-upstream clean/adversarial models exactly match uninterrupted
training after recovery, including weights, optimizer, history, and RNG.

Interruption tests cover per-clip evaluation, a saved epoch before evaluation,
and completed evaluation before its workflow-record commit. Stale source/runtime,
changed schedules, missing/corrupt receipts, and mixed-run records are rejected.
The previous 144 checks retain their coverage of native losses/gradients,
attacks, group splits, decoding, checkpoint state, primary selection, and costs.

The preceding 22 additions exercise the bounded pilot phases, exact recovery and
optimizer-tampering rejection, incomplete/corrupt evidence, fresh-versus-reused
evaluation timings, worker deadlines, and reproducible/archive-safe handoffs.
These use small real upstream models. Six additional runtime checks cover
strict configuration, conflict/late-init rejection, incompatible resume policy,
and actual independent-process CUDA gradient/update replay (skipped without
CUDA). The full-model pilot failed replay under the original runtime, and its
strict retry was stopped for runtime cost. A passing small CUDA check does not
establish full-model correctness or feasibility. See the
[CUDA outcome](../report/checks/2026-10-21-cuda-workflow.md).

Run `python -m pytest -q tests --upstream external/vjepa2` after
[local setup](../docs/smoke_test.md). GitHub CI currently checks scaffold and
syntax only. These tests do not require downloading weights or renting a GPU.

The separate full pretrained CPU feature command passed on eight real training
clips; its training-set probe score is not held-out performance. Full-model
CUDA recovery/calibration, convergence validation, near-duplicate review, and
research admission remain pending. See [workflow diagnostics](../docs/workflow_diagnostics.md)
and the [latest evidence](../report/checks/2026-10-20-workflow-checks.json).

Add tests alongside the
corresponding implementation, including:

- Pixel-space perturbation bounds, clipping, and zero-epsilon equivalence.
- Finite, nonzero input gradients and a small FP32 finite-difference check.
- Frozen target weights, clean target inputs, and correct detached features.
- Correct context/target token alignment and independent mask evaluation.
- Agreement with the upstream clean native loss on the same batch and masks.
- Dataset source-group separation, manifest replay, and cache invalidation.
- Parameter updates restricted to context encoder and predictor.
- Checkpoint resume preserving optimizer, scheduler, scaler, and RNG state.
- Stronger-attack/restart checks and per-example candidate selection.

Use small fixtures for correctness tests; do not require a full dataset or
large checkpoint in ordinary CI. GPU integration checks should be explicit,
with provenance and measured runtime. Do not label skipped scientific tests
as evidence of a validated pipeline.

`test_native_execution.py` covers the opt-in Conv3d parameter/state identities,
FP64 numerical derivatives, scoped forward/backward flags, full small-model
input/parameter gradients, PGD bounds, frozen targets, exact clean/adversarial
recovery, and rejected cache/calibration/evaluation/anchor policy mismatches.
Its actual CUDA kernel test skips without CUDA. `test_native_diagnostic.py`
exercises four fresh CPU workers and verifies saved gradient/attack evidence,
exact within-backend replay and numerical agreement; incomplete/mixed source,
precision, geometry, checkpointing or scale cannot pass. These are fixtures;
full pretrained CUDA execution and optimizer recovery remain pending.

`test_native_probe.py` covers the corrected device-typed observer: the recorded
`vol2im_kernel` name stem, rejected CPU-only/zero-duration/unrelated/cuDNN
records, invalid timings, restored flags, recomputation of retained evidence,
and persistence of rejected probe/worker JSON. These parser fixtures do not
claim CUDA execution; the shared actual-CUDA regression still skips locally.
See [the observer correction](../docs/native_probe_fix.md).

`test_gradient_attribution.py` adds 18 checks for actual upstream chain-rule
reconstruction, CPU FP64 normalization/family derivatives, reliable hook/RNG
restoration on exceptions, fixed targets/parameter identities/modes, residual
sign accounting, rejected reconstruction and source/session/worker/tensor hash
or runtime/mask/weight mismatches. The extracted CPU CLI fixture also passes.
The verified CUDA attribution rejects both forward-oracle gates; fixed-cotangent backwards and exact reconstruction pass. Latest project suite: 373 passed, two
CUDA-only skips (`python -m pytest -q tests`; exclude vendored upstream tests).

`test_precision_patch.py` adds 28 checks for the separate shared-forward
precision: FP64 numeric derivatives, actual small-model full input/parameter
derivatives against independent double-convolution autograd, original tensor
identities and target freezing, exception restoration, strict scoped backwards,
rejected old/mixed/runtime/false-gate identities, bias-free parameter gradients,
rejected higher-order derivatives, explicit bounds and four fresh upstream CPU
workers. Local trained-projection gates and the extracted CLI also pass; this
is not new CUDA evidence. Project suite: 373 passed, two CUDA-only skips.

The university `precision.RvqoMB` export now independently verifies four fresh
CUDA workers, forward/input/weight/bias numerical gates, actual new native
probe, exact within-backend replay and matched forward/attack outputs. These
are separate recorded CUDA diagnostics; they do not turn the two locally
skipped pytest regressions into pytest passes. Full-model optimizer recovery
and study admission remain pending. See
[the verified result](../report/checks/2026-10-21-university-shared-forward.md).

`test_shared_execution.py` adds 47 integration cases for both named shared-
forward policies: clean/adversarial full-small-model AdamW updates and exact
recovery, twelve fresh CPU worker processes, scheduled evaluations and three
failure boundaries, original parameter/state-key/target invariants, matching
anchor loads and stale/absent precision rejected before restoration, cache/
module/normalization/geometry/dtype/checkpointing drift and incompatible
calibration/evaluation/features. The fresh process worker lives in
`tests/fixtures/shared_recovery_worker.py`. Current suite: 420 passed,
two actual-CUDA skips. Full pretrained CUDA recovery is still pending.

`test_workflow_preflight.py` adds 14 cases for three actual upstream input
checks, rejected oracles/gradients/no-ascent/probe/data, state/RNG/mode restoration,
complete shared-forward phases and exact recovery, bounded tensor exports and
invalid deadline refusal. Current local suite: 434 passed, two actual-CUDA
skips. The extracted handoff separately passes six selected cases. No new CUDA
result is claimed. [Evidence](../report/checks/2026-10-22-shared-workflow-handoff.md).

`test_measured_budget.py` adds 14 offline accounting cases. Together with the
10 existing staged-budget cases, all 24 targeted tests pass. They check schedule
counts, unknown costs, single charging of pilot/scoring work, input drift,
invalid timing/policy evidence, economic thresholds and no automatic admission.
This is additional local budget coverage, not another CUDA scientific run.

`test_hardware_benchmark.py` adds 20 cases for GPU identity/VRAM/device count,
quote/deadline rejection, budget accounting, missing/nonfinite observations,
and the complete seven-phase path on actual small upstream CPU models. The
20 new plus 52 related workflow/preflight/budget cases pass (72 total).
Three selected cases also pass from the fresh restored package. These are
local/extracted checks, not new CUDA performance or full recovery evidence.
