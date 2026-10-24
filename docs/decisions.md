# Research decisions

## 2026-10-16 — Initial scope

| Decision | Reason and consequence |
| --- | --- |
| Focused research report | A useful first deliverable within three weeks plus a short review buffer; publication quality depends on evidence |
| V-JEPA 2 ViT-L first | Released encoder/predictor; avoids the smaller V-JEPA 2.1 distillation mismatch risk |
| Fixed target encoder | Stable loss reference across checkpoints; conclusions exclude the original moving-target protocol |
| One small video dataset | Preserve budget for repeated training and strong evaluation |
| Three fine-tuning seeds | Expose variation; data split is shared across seeds |
| Up to $200 total | Use university/local resources first and measure rental feasibility |
| 10-20 human hours weekly | Reserve human review for access, unexpected results, and scientific interpretation |

These choices were confirmed in the research planning discussion. UCF101,
the 20/5/5 per-class subset, and numerical training defaults are the proposed
implementation of that scope, subject to the documented feasibility gate.

## 2026-10-16 — Repository setup

Keep the previous project's root `models/`, `scripts/`, `notes/`, `report/`,
`goals.md`, `progress.md`, and `setup_lambda_labs.md`. Add configurations,
protocol documentation, split manifests, and lightweight GitHub checks.
Continue the previous repository's MIT license for original code.

No previous experimental results, model code, collaborator list, or private
conversation exports are copied into this project. Only the organization
and useful documentation conventions are carried over.

## 2026-10-17 — Local model integration (protocol 0.2)

Pin Meta's source at `204698b45b3712590f06245fbfba32d3be539812`. Construct the
upstream V-JEPA 2 architectures directly instead of using the current Hub
loader's localhost URL and non-strict loading. Preserve all ten predictor
mask tokens observed in the official checkpoint metadata. A single sequence
group uses mask token 0, matching the upstream wrapper. Record upstream
pixel-normalization constants in the study configuration.

These resolve implementation details without changing the fixed-target
question, 16-frame input, dataset subset, seeds, or training settings.
Checkpoint metadata inspection is not a successful pretrained load. The
full file hash, full-model execution, CUDA compatibility, and feasibility
are still unverified. See [the smoke-test guide](smoke_test.md).

## Decisions still requiring pilot evidence

- CUDA execution and memory/runtime on the actual experiment hardware.
- Full-dataset decoding, corruption/duplicate checks, and final preprocessing lock.
- GPU microbatch, accumulation, precision, and optimizer implementation details.
- Evaluation step sizes, fixed subsets, attack/scoring mask banks, and probes.
- Measured end-to-end cost and whether 20 or 10 training videos per class fit.

Resolve these before locking the protocol, without using final test outcomes.

## 2026-10-17 — Pretrained/data/attack checks (protocol 0.3)

The official checkpoint passed strict CPU loading, native-loss agreement,
input/parameter gradients, target freezing, and replay. Its SHA-256 is now
recorded. Official split-1 metadata yields 20/5/5 videos per class with
source-group separation and manifest hash recorded in the study config.

Choose the documented centered 4 FPS deterministic clip recipe for the
pilot. Real AVI inspection revealed a missing final timestamp; blindly
enabling FFmpeg's `genpts` also duplicated an existing timestamp. Preserve
observed timestamps and fill only missing entries when all observations
agree with the declared constant frame rate. Record repairs, decoder version,
source bytes, and exact processed-pixel hashes. This is a pilot-tested
proposal; complete dataset integrity and preprocessing lock remain pending.

The PGD implementation keeps the strongest attack-bank candidate per example
over clean input, starts, and all iterates. Independent masks are reserved
for scoring. The feasibility pilot uses FP32, microbatch 1, accumulation 8,
and explicit AdamW defaults; these do not silently fix the final precision
or monitoring/evaluation configuration. No cloud spending is authorized by
a pilot result alone; retain the existing measured cost gate.

## 2026-10-19 — Mac GPU checks and equivalent patch projection

The M4 Max exposes MPS with normal host GPU access. A restricted process
reported it unavailable; that observation did not establish a hardware
limitation. Keep CPU fallback disabled in GPU feasibility checks. The
CPU/MPS loss, target-feature, input-gradient, and selected parameter-gradient
comparison passed on one real training clip. This does not establish
cross-device equality of full training trajectories.

The original context Conv3d projection has an expensive input backward pass
on this MPS runtime. Add an explicit `--patch-embedding linear` option that
reshapes nonoverlapping patches and applies the same projection parameters.
It preserves state keys and parameter identity; the fixed target encoder
retains its original projection. CPU output/gradient tests and full-model
MPS diagnostics passed. PGD-5 on the checked clip fell from 121.29 seconds
to 4.84 seconds, with matching reported MPS attack losses and bounds.

Keep the default projection `conv3d`. The Mac commands explicitly select
`linear`, and reports record that choice. Any experiment using this option
must use it consistently across clean and adversarial conditions. No
upstream source, checkpoint, masking objective, or trainable component was
changed. Numerical reduction order may differ; do not describe this as
bitwise equivalence for arbitrary inputs or platforms.

The FP32, microbatch-one, accumulation-eight pilot is a hardware measurement.
It does not freeze the confirmatory backend, precision, validation subset,
or schedule. The [feasibility record](../report/checks/2026-10-19-mac-feasibility.md)
separates training from evaluation cost. Choose the monitoring schedule
before examining research outcomes; full training and resume support are
still pending. Protocol/configuration version 0.3 remains a draft.

## 2026-10-19 — Evaluation and compute proposal, not a protocol lock

Created a separate [compute plan](evaluation_compute_plan.md) with one
training/validation clip per class for monitoring, stronger primary attacks
at nine declared epochs plus initialization, and full 505-video final testing
at the pretrained and per-run selected/final checkpoints. Three seeds and
both conditions remain. The proposed best checkpoint is selected only by
the primary validation metric among those declared epochs; sparse primary
measurements cannot establish the best of all epochs. A denser alternative
is costed separately. Neither schedule is frozen or used for training.

The calculator retains 15 missing timing categories as unknown and labels
stronger-attack/epsilon estimates as proxies. It never approves a run. The
20-video-per-class proposal projects to 1,209.66 Mac hours from already-timed
operations, plus 483.60 hours of proxies and additional unknown work. A cloud
speedup must be measured; published hourly prices are insufficient evidence.

Reserve up to $10 for pilots within the existing $160 compute envelope,
retain the $40 noncompute reserve, and propose 25% runtime headroom when
pricing the measured full workload. These are planning allowances, not
authorization to launch an instance. Kaiwen confirmed an existing Lambda
account and chose it for the first pilot. The candidate is one 48 GB A6000;
its live quote and availability remain to be checked. Runpod's advertised
rate remains a comparison. No provider account was accessed and no CUDA
throughput was measured in this step.
The draft study file remains unchanged so prior measurement hashes remain
valid. Review scope and evidence before a new protocol version is locked.

## 2026-10-20 — Proposed staged scope, separate from protocol 0.3

The RTX 5090 measurements rule out the original larger schedule within $200.
Prepare a separate [staged proposal](budget_study_plan.md) and
[configuration](../configs/budget_study_plan.yaml), preserving the original
study hash and prior evidence. Recommend 20 classes with five actual training
videos each, one discovery pair, and three new confirmation pairs conditional
on signal validity and measured budget. Preserve the model, loss, 16-frame
256-pixel input, FP32 baseline, and primary PGD-20/R2/EOT4 attack. Narrow
claims to native-loss dynamics; defer direct classifier adversaries, embedding
attacks, and epsilon sweeps. Thirty small-subset epochs are not equivalent to
the original update exposure.

Use disjoint source groups for exploratory and confirmation validation.
The existing validation pool contains only one or two groups per class;
withhold one unused group per class from the parent training pool before
selecting actual training videos. Select classes/groups by identifiers only.
Keep source-group uncertainty and avoid treating three seeds on the same
videos as independent data replicates. Sparse strong-evaluation checkpoints
limit the selection claim to the declared anchors.

Account for the user-reported $0.592 Runpod bill inside the existing $10 pilot
reserve, not in addition to it. The $193.91 candidate envelope includes 25%
headroom, eight untimed hours of allowance, and $40 noncompute reserve. Unknown
costs and proxy attacks still block admission; no unmeasured speedup is assumed.
The proposal does not launch work, lock protocol 0.4, or admit confirmation.

## 2026-10-20 — Training recovery implementation, without study admission

Implement a reusable FP32 native training core and bounded diagnostic CLI.
Reuse the pilot update and preserve its attack settings. Define linear warmup
by optimizer-update position, apply LR drops in the epoch after each declared
boundary, and divide partial-batch loss by the actual number of examples.
Cache the upstream mask sampler once while testing parity against the old
sampler; use explicit epoch/video seeds shared by matched conditions.

Persist AdamW/RNG state and source/config/data identity, with a checksummed
atomic manifest and two rolling resume states. Keep fixed-target identity
without duplicating its weights. Preserve prior attempts when resuming and
require an explicit choice to recover an older generation. CPU tests verify
exact trajectory recovery; GPU equivalence remains untested. See the
[runner guide](training_runner.md). No protocol/configuration freeze or new
cost evidence follows from these implementation checks. Primary research
evaluation and total-cost admission must precede full trajectories.

## Change record template

```text
Date:
Protocol/config version:
Change:
Reason and evidence:
Affected runs:
Effect on comparison/claims:
```

## 2026-10-20 — Staged exports and native evaluation contract

Export the proposed 20-class identities separately, retaining the original
protocol/config and parent manifests. Prepare only the 100 actual training and
20 exploration clips now; confirmation/test pixels remain closed. Two decodes
and saved-cache reloads passed for all 120 clips, without exact hash duplicates.
Near-duplicate review and full protocol freezing remain pending.

Fix evaluation seed serialization explicitly: compact JSON, SHA-256 first
eight bytes modulo 2**63, then modulo 2**31-1 for the upstream signed counter.
Use one fixed attack bank across restarts, independent scoring banks, and fresh
perturbations at every checkpoint. Preserve training modes/RNG around evaluation.
Primary selection requires all declared validation anchors including initialization,
with earliest exact ties; routine PGD and test results cannot choose a checkpoint.

These are implementation details of the staged proposal, not a new admitted
study or changes to the historical pilot. The new CLI remains bounded and
diagnostic. Record them in the eventual frozen configuration after calibration,
full-model validation, and measured budget checks. See
[staged data and evaluation](staged_data_and_evaluation.md).

## 2026-10-20 — Probe convention and epoch evaluation barriers

Implement mean pooling of all clean context tokens and a separate ridge probe
for each representation checkpoint. Fix training-only population normalization,
FP64 solution, unpenalized intercept, and lambda one under **sum** squared error.
Measure variance and covariance-spectrum effective rank before normalization;
report their sample-size limit without declaring a collapse threshold.

Declare the first ten training-monitor manifest rows for eventual research
step calibration. Compare 0.5/1/2 per 255 using the common independent primary
scoring bank, highest mean then smaller exact tie. Keep training PGD-5 step one.
The current CLI's one/two-clip diagnostic is separately marked and cannot
admit or replace ten-clip research calibration.

Require evaluations to complete at saved epoch boundaries before further
training. Checksum-bound in-place continuation is explicit for the new
workflow command; reruns still require fresh directories, and the older
training CLI preserves its fresh-attempt behavior. Reuse shared initialization
only under identical evaluation contracts. Retain primary anchor weights and
reject altered schedules, calibration, source/runtime, data, or mixed receipts.

These details are implemented and locally tested, with a full pretrained CPU
feature check; full-model CUDA recovery/calibration remains pending. Record
these conventions in the eventual frozen configuration after validation and
cost admission. Preserve existing configuration hashes and prior evidence.
See [workflow diagnostics](workflow_diagnostics.md).

## 2026-10-21 — Strict CUDA replay after a failed recovery pilot

The first full-model workflow pilot stopped at the clean recovery comparison.
Its run contracts, RNG states, and fixed target matched, but learned weights,
optimizer moments, and evaluation losses did not. Independent attacks already
differed at the identical pretrained checkpoint, before any restoration.
The recorded runtime had deterministic algorithms and cuDNN determinism disabled.
This supports a kernel nondeterminism diagnosis; a repaired full-model replay
must establish the outcome rather than relaxing comparison tolerances.

The CUDA workflow runner now enables strict deterministic algorithms (errors,
not warnings), disables cuDNN benchmarking, enables cuDNN determinism, and sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization in every process.
It rejects conflicting configuration or late initialization. Precision flags,
model architecture, attacks, masks, and optimizer settings remain unchanged.
The runtime contract also binds deterministic warning mode and cuBLAS workspace
configuration. This follows [PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/2.8/notes/randomness.html).

Preserve the failed run and original handoff. Transfer repairs as a separate
checksummed overlay, use a fresh output and calibration, and measure the new
runtime separately. No previous calibration, checkpoint, or timing is silently
promoted into this retry. Full-study configuration and admission remain pending.

## 2026-10-21 — Explicit diagnostic precision control

The first strict same-Conv3d comparison rejected its unchanged cuDNN reference
against the fixed CPU FP64 accuracy gate. cuDNN TF32 was permitted; native
output and gradients pass FP64 but fail compatibility with that reference.
This supports a precision hypothesis, not a measured speed or replay result.
See the [preserved outcome](../report/checks/2026-10-21-university-cuda-convolution.md).

Prepare a separately tagged `--cudnn-tf32 off` diagnostic, retain strict
execution and all tolerances, and bind the declared policy to actual worker
runtime. Preserve the original TF32-on failure/archive. Validation signatures
are saved even after rejection. This control changes only the isolated
experiment factor; no study precision/backend, optimizer, model, or protocol
configuration is adopted. CUDA causation, performance, and full-model recovery
remain pending. Any eventual study precision change needs an explicit new
contract and a fresh acceptance pilot.

## 2026-10-21 — Native convolution qualifies as a component candidate

The explicit TF32-off control completed all 12 university CUDA workers with
fixed-tolerance accuracy and observed exact replay. Matching operands/oracles
and shared runtime resolve the earlier cuDNN precision rejection. Native
forward/backward takes 5.897 ms for input gradients and 7.043 ms for input plus
parameter gradients, versus 3.410 / 3.431 seconds for strict contiguous cuDNN.
Channels-last does not help. See the
[verified outcome](../report/checks/2026-10-21-university-cuda-convolution-precision.md).

Use native original-Conv3d execution as the next full-model diagnostic candidate.
This is not adoption of a study backend/precision change. Prepare an opt-in
integration that preserves default behavior, parameter identities/state keys,
clean frozen targets, and explicitly scoped backend behavior through backward.
Bind the chosen policy to runtime/target-cache/anchor/resume contracts and use
fresh targets/output directories. Require full-model gradient/attack checks,
exact recovery, representative GPU timing/memory and measured budget admission
before research runs. Component speedups are not whole-model speedups.

## 2026-10-21 — First-order native dispatch remains opt-in

Implement `native-fp32` as a separately named diagnostic policy, with a
matched `cudnn-fp32` reference and the existing `original` default. Preserve
the original Conv3d object, parameter identity, geometry and state keys;
leave the target patch module original and frozen. This diagnostic sets
cuDNN TF32 off explicitly and requires fresh targets, calibration and outputs.
The study/proposal YAML remain unchanged and unadmitted.

PyTorch 2.8 re-queries cuDNN flags in convolution backward. A forward-only
context would not establish the selected backward backend. Scope PyTorch's
own convolution calls in both directions through an explicitly first-order
autograd delegator. Higher-order differentiation and concurrent Python model
execution are not supported. See [implementation and limits](native_execution.md).

Bind actual patch execution, activation checkpointing and precision to cached
targets, features, evaluations, calibration loading, anchors and recovery.
Reject old or incompatible identities before loading saved trainable weights;
retain exact recovery equality. The next university check uses only an existing
training clip and compares full-model input gradients/FGSM/PGD-5 in four fresh
workers, with hard limits and no optimizer step. It cannot establish full-model
AdamW recovery, RTX 5090 behavior, study feasibility, or robust overfitting.
Those gates and the existing budget remain in force; no rental is automatic.

## 2026-10-21 — Correct the observer and retain rejected evidence

The first native integration attempt rejected the small backward probe before
full pretrained native execution. Its name check expected `col2vol`; the
already verified university export records `at::native::vol2im_kernel`.
This expectation was incorrect; the failed attempt's profiler evidence was
not saved before raising, so a new CUDA pass cannot be inferred from its logs.

Use schema-2 device-typed raw events, require a positive-duration native CUDA
reconstruction kernel plus convolution backward, reject cuDNN execution, and
persist flags/events before validation raises. Recompute the observer gate
on export. Preserve the original failure/source archive. Use a separate fresh
retry with native workers first; repeat reference workers under the same new
source identity. Mathematical code, numerical and exact-replay gates, resource
bounds, and study configuration remain unchanged. No research backend adopted.
See [the fix and handoff](native_probe_fix.md).

## 2026-10-21 — Diagnose the numerical rejection at fixed incoming derivatives

Preserve the verified full-model native/cuDNN failure and its original bounds.
Use a separate bounded training-only diagnostic: retain each forward token
source, intervene with distinct preserve-format patch leaves per mask family,
compare both backwards with identical incoming derivatives and independent
CPU FP64 raw-pixel VJPs, and reconstruct the saved failed-run gradients. Bind
source/runtime/weights/data/targets/masks and rejected-evidence hashes. Record
residual sign changes without automatically claiming L1 sensitivity as cause.
No study, loss, original mathematical implementation, acceptance threshold,
backend adoption or spending change. Pretrained CUDA is pending; local tests
and the source-only handoff are ready. See [the control](native_gradient_control.md).

## 2026-10-21 — Preserve the failed forward gate after verified attribution

The complete exported control verifies fixed-cotangent backwards against FP64,
exact identical-token tail replay and bitwise reconstruction of the saved
failed gradients. Both FP32 forward paths fail the existing output gate
(12 cuDNN / 35 native values); that rejection remains. Forward-token changes
alter incoming derivatives, including eleven residual sign crossings, without
proving those crossings alone explain the entire effect. Independent CPU FP64
library convolutions reproduce the trained-weight analytical references.

Specify a shared, more accurate forward before the next backward comparison;
a candidate is explicit FP64 patch accumulation cast to FP32 tokens. This is
not implemented/adopted. Declare precision/reference identity before execution,
keep gates unchanged, and preserve historical FP32 failures. A hybrid retaining
cuDNN forward cannot bypass its failed forward-oracle gate. No study/YAML,
mathematical implementation, budget, or backend adoption change. See the
[verified outcome](../report/checks/2026-10-21-university-gradient-attribution.md).

## 2026-10-21 — Declare a separate shared-forward precision/reference control

Implement FP64 accumulation only for the original learned context patch
projection, with explicit FP32 output/layout and unchanged original FP32
parameters, normalization, clean frozen target and transformer/predictor/L1.
Compare cuDNN/native FP32 backward against a shared-FP64-forward reference;
do not use the historical FP32 forward as an undeclared new reference. Keep
all prior pointwise gates and rejected FP32 artifacts. FP32 convolution VJPs
are a first-order affine derivative approximation checked against CPU FP64.

Use a distinct temporary identity and guards; existing study/cache/training
APIs reject it. Both arms run the same sanity probe before resetting model
construction RNG. Save raw reference/actual tensors and recompute gates. Local
trained-projection/CPU worker checks pass; pretrained CUDA and full optimizer
recovery remain pending. No study policy/YAML adoption or budget change. See
[the declared control](shared_forward_control.md).

## 2026-10-21 — Qualify the bounded shared-forward candidate; preserve admission gates

Accept the independently verified `precision.RvqoMB` result for its declared
input-only precision control: all numerical gates, reconstructed gradients,
actual native dispatch and exact within-backend replay pass. Shared tokens
and both attacks match bit-for-bit across backwards on one clip; native
PGD-5+score is 6.44x faster in medians of two single attempts. This qualifies
an integration candidate, not research adoption or full optimizer recovery.

Keep the separate shared-FP64-forward reference and both historical FP32
failures. Plan deliberate opt-in runtime/cache/calibration/anchor/checkpoint
integration and CPU exact optimizer recovery before a bounded sufficient-VRAM
CUDA update/resume pilot. Preserve all original thresholds, target/data rules
and budget gates. Current scientific implementation and study/proposal YAML
remain unchanged by this verification. See
[verified outcome](../report/checks/2026-10-21-university-shared-forward.md) and
[next plan](shared_forward_integration_plan.md).

## 2026-10-22 — Explicit shared-forward diagnostic training identity

Promote the verified shared-forward algorithm to separately named
`shared-fp64-cudnn` / `shared-fp64-native` diagnostic options. FP64 context-patch
accumulation is explicit; parameters, tokens, VJPs, normalization, original clean
target and tail stay FP32. Schema-2 identity and unchanged equality checks bind
all artifact consumers, including rejection before resume/anchor restoration.
Keep the earlier temporary scope separate and retain old default/schema-1
identities. CUDA candidates require the exercised PyTorch 2.8.0 boundary.

Local small-upstream clean/adversarial AdamW recovery, including fresh processes
and scheduled evaluations, now passes. This is implementation acceptance only;
no research precision/protocol is adopted or study YAML changed. Full pretrained
CUDA recovery, representative/convergence checks and measured budget admission
remain pending. Preserve historical FP32 rejections and the university AdamW OOM.
See [the integration record](../report/checks/2026-10-22-shared-forward-integration.md).

## 2026-10-22 — Bounded representative gate before another CUDA workflow

Add the fixed first/middle/last training-input comparison before calibration
for the explicitly named shared-forward policies. Retain the existing pointwise
numerical and exact recovery limits, nonzero gradients/loss ascent and frozen
targets. Export bounded oracle/gradient/attack tensors for independent inspection;
keep full optimizer/model state out of the small evidence export. Old failure
records and source archives are preserved.

Use a new offline working-tree snapshot, existing verified training/wheel bytes,
exact upstream bundle and checkpoint pin. A detached supervisor requires a
reviewed deployment-based absolute deadline; it never stops cloud billing.
Prepare/verify before asking for current quote/deployment action. Local source
and CPU fixture success is not new Linux/CUDA acceptance or study admission.
See [the prepared handoff](shared_forward_cuda_pilot.md).

## 2026-10-22 — Retain pilot storage and review the current console quote

The user reported $1.21/hour for one RTX 5090 with 30 GB container disk and
80 GB persistent volume at `/workspace`, and explicitly retained both disk
capacities. Keep the existing 55 GiB free-after-setup gate, checkpoint retention
and atomic writes. The verified source package remains frozen; this update
changes planning/status documentation only.

At that rate two running hours cost $2.42. Keep the existing two-billed-hour/$3
inclusive session limit and account for any unquoted charges in the $0.58
margin. This is a user-reported quote, not a billed charge or independent
availability verification. Manual deployment/current SSH details and the
deployment-based absolute deadline remain next. No cloud lifecycle action,
new research execution or change to study admission follows from this quote.

## 2026-10-22 — Accept bounded shared-forward CUDA recovery; retain study gates

The verified RTX 5090 diagnostic completes all eleven phases. Both native-policy
clean/adversarial two-update recovery comparisons are exact; three representative
training inputs pass unchanged numerical limits and independently recomputed
saved evidence. Cross-backward FGSM/PGD bitwise equality is descriptive and
varies on two clip/method combinations. Keep historical FP32 rejections.

One-clip diagnostic calibration, feature/probe work and pretrained-monitor
PGD-50/FGSM/random controls are measured. These do not admit a research
precision policy, replace ten-clip study calibration, or establish robust
overfitting. Freeze/rebudget deliberately before an exploratory trajectory.
Study/proposal YAML remains unchanged. The user terminated the Pod after
verified export; the $1.085 billing increment is provisional and derived from
the reported combined summary. See
[the outcome](../report/checks/2026-10-22-shared-forward-cuda-workflow.md).

## 2026-10-22 — Reprice the explicit candidate; preserve budget and study gates

The measured shared-forward candidate costs exceed the reviewed caps for all
existing variants: the 20-class design is about $304–$307 and the ten-class
alternative $219–$221 with retained allowances/reserves. Unknown full-study
costs remain. Do not use the historical $193.91 envelope or cheaper default
runtime to admit execution. No YAML, dataset, attack, epoch or seed change.

Prefer screening a lower-cost hardware measurement before sacrificing class
coverage or attack strength. Any narrower/discovery-only study is an explicit
new scope decision made before trajectories. The $200 total/$160 compute caps
remain. See [the review](../report/checks/2026-10-22-measured-budget-review.md).

## 2026-10-22 — Preserve 20 classes and screen lower-cost hardware

Kaiwen chose the cheaper-hardware route and reported an A5000 quote of
$0.29/hour for the retained 30/80 GB disks. Prepare a seven-phase screen before
any new rental. It uses original numerical limits and fresh unit timings,
without split/resume trajectories; full candidate recovery remains a later
gate if the economics are promising. Preserve the $200/$160 caps and existing
two-hour/$3 pilot ceiling. No automatic rental or scope reduction.
See [the preparation](../report/checks/2026-10-22-lower-cost-benchmark-preparation.md).


## 2026-10-24 — Accept the one-clip matched control; retain study attack gates

Exact start/prefix pairing, frozen-source/data/checkpoint identities, retained
native CUDA kernels and fresh A-only replay pass on university RTX 3070. This
accepts the diagnostic mechanism and observed 8 GB fit for input-gradient
attacks; it does not accept full training or optimizer recovery.

At four attack draws, 20-to-50 independent main loss is nearly flat and audits
have mixed directions. At eight draws all three banks increase; eight draws
are not uniformly higher than four. Keep the proposed canonical methods and
study YAML unchanged. Predeclare a broader training-only clip/bank check and
finish research calibration before choosing the primary attack or revising
costs. Do not promote instrumented RTX 3070 times to A40 observations. No paid
launch or held-out pixels follow from this diagnostic acceptance. See
[the verified result](../report/checks/2026-10-24-university-matched.md).
