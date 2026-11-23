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


## 2026-10-24 — Freeze a training-only panel before further attack comparisons

Keep the original first ten training-monitor rows and two distinct attack-bank
replicates per clip. Share starts, fixed clean targets and scoring banks within
clip; retain all four20/50-step×4/8-draw diagnostic arms and fresh A-only replay
per case. Analyze bank replicates within clip, then weight clips equally. No
partial-panel aggregate, score-bank selection, new strength tolerance or automatic
research attack choice. Prior one-clip data remain excluded from panel means.

The specification is metadata-tested, with67 focused passes and a CPU mask dry
run. It is not a runnable handoff, research calibration or study admission.
Use only a reviewed university host under serial bounded cases after separate
implementation/qualification; no paid rental or changed project caps. See
[the predeclaration](matched_panel_diagnostic_plan.md).

## 2026-10-25 — One documented attended recovery exception completes case three

The user explicitly approved one additional 80-minute university attempt after
the interrupted case left an unfinalized initial elapsed record. Charge the
unknown original allowance in full; retain the separately identified single-use
authorization rather than infer unused time from zero. Keep the qualified
scientific execution and both previously accepted cases unchanged.

That fresh whole-case recovery now independently verifies, including all four
matched arms and a fresh A-only replay. Its 40-minute-41-second attended window
fits the added allowance. Preserve the failed partial and operational failure
history; a successful attended run does not establish the prior stopping cause
or qualify unattended logout behavior. The consumed grant covers no further
cases. Three of twenty cases are accepted; the next continuation is logged and
attended, with original case limits. No study setting, tolerance, held-out data,
precision adoption or budget admission changes. See the
[verification](../report/checks/2026-10-25-matched-panel-recovery-verification.md).

## 2026-10-25 — Qualify a second host in an isolated shadow experiment

After the user requested the next step, prepare one separately identified
linux10603 reference replay with a twenty-minute total bound, two fresh
arm-A workers and no automatic GPU retry. Fix the latest independently
accepted reference before candidate execution, and require exact retained
numerical/gradient/tensor agreement and observed native CUDA evidence.
Record actual new hardware and source identity. No hostname/UUID spoofing,
old-plan edits, old receipt relabeling, panel-journal writes or source-pin
replacement. Original v1 single-host/serial requirements and all historical
FP32 failures remain. A pass is scoped to this reference, not automatic
multi-host panel admission, study precision adoption or research results.
See [the separate replay guide](cross_host_replay.md).

## 2026-10-25 — Accept the bounded reference bridge and stop for the night

Both independently verified candidate arm-A replays match the fixed reference
bitwise, with actual native CUDA evidence and unchanged source/runtime/weight
contracts. Accept only this one-clip/one-arm shadow result, retaining the
original v1 single-host plan and all old artifact identities. A two-host panel
needs an explicit execution revision and tested coordination; no automatic
dispatch or research adoption follows. Both jobs ended. At the user's request,
stop after verification and resume further work only when they return.
See [the verification](../report/checks/2026-10-25-cross-host-shadow-verification.md).

## 2026-10-26 — Explicit two-host execution revision

User resumed and requested using both university workstation GPUs after the
scoped linux10603 reference replay passed. Preserve the frozen numerical plan
and original seven v1 receipts. New cases use truthful per-host identities in
a separate execution namespace and shared append-only coordinator. This
replaces serial-only execution for the separately identified revision; it
does not relabel v1 evidence or generalize the one-clip/A-only bridge.

Keep clip pairs co-located, including pending Tennis bank1 on10609. First
run only Tennis bank1 and Floor bank0, one case/GPU, under their original
80-minute limits. Review both exports independently before later dispatch.
Preserve all historical failures, budgets and scientific settings. See
[two-host operator guide](two_host_panel.md).

## 2026-10-28 — Propose eight-draw primary evaluation; measure feasibility before calibration

The complete20-case/10-clip matched panel shows positive independent-scoring
gains from four to eight attack draws on every clip in all three scoring banks.
Propose PGD20/R2/EOT8/score4 as the next primary candidate, retaining a matched
PGD50/R2/EOT8 stronger control and audit banks. This is a reviewed proposal,
not protocol adoption; current evaluator/calibration guards and study/proposal
YAML remain unchanged. No strength threshold or convergence pass is introduced.

Preserve the staged20-class/100-clip/eight-run design and original training
PGD5/step1/255. Specify all30training-only0.5/1/2step calibration evaluations
on the original first ten monitor rows, shared candidates and new pinned masks;
main independent mean chooses the step with smaller exact ties. Audit/validation/
test scores do not choose it, and no step or other-GPU calibration is adopted.

Reconcile accepted RTX5090 bounded exact recovery, untested A40 exact resume,
and the unresolved8GB full-optimizer qualification separately. New EOT8 primary
and calibration timing units remain null. Keep both$200/$160caps and the full
workload; do not invent a mask-ratio speedup, transfer university timings, or
price a new profile with old four-draw units. Conditional ceilings are necessary
conditions with explicit optimistic calibration-free algebra, never approval.

Next implement/test an opt-in versioned path and prepare the bounded research-
hardware benchmark before quote/deployment or full calibration. No new paid
allocation, old recovery-time reuse, held-out access or research run follows.
See [the candidate](research_attack_candidate.md) and its full decision packet.


## October 28: implement EOT8 only through a separate diagnostic contract

The proposed PGD20/R2/EOT8 primary now has a tested opt-in evaluator, calibration
journal and artifact consumers. The original evaluator/calibrator and study YAML
stay frozen. PGD20/PGD50 share attack banks and starts; independent scoring does
not select perturbations. New artifacts bind settings, 61 source/metadata inputs,
data, weights and runtime, including CUDA UUID/driver. No old artifact is relabeled.

The portable benchmark measures bounded recovery and representative EOT8 units.
Its independent worker deadline survives loss of the parent; it does not stop
billing. No quote, launch allowance or full research calibration is approved by
this implementation. Require selected-hardware numerical/recovery and economic
qualification before the declared thirty research calibration evaluations and
any protocol adoption. [Implementation](candidate_eot8_implementation.md).

## 2026-11-19 — Consolidate scoped evidence before training

Retain the accepted clean EOT8 replay, earlier fixed-input FP64/native/math
control, completed eight-gradient/four-pair PGD numerical result, native component,
corrected saved backend metadata and native-relative component value checks.
These receipts retain their own input, control, runtime and admission scope.
The old candidate/session failure and earlier rejected derivative remain unchanged.

Next implement a candidate-only completion adapter locally using backend v4,
the unchanged component capsule and immutable PGD/native receipts. No completed
native, clean replay or numerical cohort needs a repeat. The declared split
mechanism diagnostic does not add a new FP64 oracle or full-model trace; those
broader claims remain unestablished at the new perturbed point. A future separately
approved candidate must produce its own complete runtime/resource/parent/collection
chain before that diagnostic closes.

Keep the input-gradient candidate separate from training PGD5/EOT1, routine EOT1,
optimizer backward and the research policy registry. Full workloads/performance,
chosen-hardware memory/recovery, complete cost and finances, research calibration,
data/protocol lock and explicit adoption still precede research trajectories.
Historical unfavorable cost screens, the single $10 diagnostic cap, $200/$160
project/compute caps and unresolved financial holds remain. No live quote, new
scope, upload, university connection, paid work or research run is authorized.
See [the dated readiness assessment](training_readiness_20261119.md).

## 2026-11-23 — Candidate-only supervisor integration stays unbound

[The local operator review](../report/checks/2026-11-23-pgd-candidate-operator-v1-local-review.md)
connects 39 unchanged candidate runtime files to six supervisors, a closed native
reference catalogue, private source cache, proof/candidate OS interfaces and bounded
same-connection collection. The older three-role operator and all historical
scientific results and failures remain unchanged. The existing input contract v1
is retained; the separate v2 contract pins 109 inputs. All derived supervisor/client
bytes have inverses. Original resource, source, ownership, numerical and continuing
clock predicates remain, with one new private candidate namespace.

All 28 distinct new checks pass across 39 executions. Only setup-error/failed
nodes reran after fake-label/PID and terminal-row assertion corrections. Production
was unchanged after first checks. Two complete lifecycles are fake; real metadata
readers and a 130-file priority transfer verify joins without actual arrays,
Torch/NumPy, model/autograd/profiler, process/PIDFD, signals/timers, GPU, proof or
network activity. This establishes local integration, not actual OS/scientific
execution. The independent stdlib audit v2 preserves 26,395 hashes and 28,596
fixture entries and independently reconstructs the wire. Its first namespace-only
comparison refusal is preserved. The local device change 31→33 has an explicit
APFS observation/review; cause and historical physical identity stay unproven.
Original records and target device/storage/host/GPU checks are untouched.

No exact target scope, handoff, approval request, authorization, attempt or live
binding is created. The next separate stage prepares/reviews that candidate-only
packet before requesting fresh authorization. A future target still needs its own
source/reference/route/backend/value/restoration/resource/completion/parent wall-RSS/
collection chain. Then close this mechanism diagnostic and qualify whole workloads.
No completed native, clean or eight-gradient/four-pair cohort repeats; no new FP64
oracle or full-model trace is introduced for lifecycle repair. Broader research and
training gates, parameter backward, attack policies, financial caps and holds stay.

## 2026-11-23 — Exact candidate-only attended scope stays unarmed

[The candidate-only package/scope](../report/checks/2026-11-23-pgd-candidate-handoff-v1-scope.md)
now binds the reviewed worker/supervisor closure, both unchanged archives, twelve
local source pins and isolated attended CLI. Three local edits have whole-byte
inverses; uploaded source and supervised control bytes remain identical. The new
outer control is 24045e50d1db73c39925dbda25ee4a3f5b57eb1132e7a428eb4d3d78be7963db.
Twenty focused package/CLI/preflight cases pass once; the preceding 28 operator
checks are inherited. Positive preflight uses only a fake grant in fixture copies.
No process/timer/thread/network, proof, GPU, Torch/NumPy, array values or science
runs. Independent stdlib review preserves 30,080 hashes / 33,227 fixture entries,
checks package/source/archive/scope joins, and leaves the old FIFO lstat-only.

Request one new exact approval before arming. The proposed attended scope is one
full 2.5 GiB proof within 60 seconds followed by one candidate component within
120 seconds, forty minutes work and ten same-SSH collection within fifty including
login. Original resource/numerical/backend/source/ownership/continuing-clock gates
remain. No native or completed-cohort rerun, extra full-model objective, warmup,
new FP64/full-model trace, retry, separate retrieval, installation, old cleanup,
private-key/checkpoint/held-out upload, paid work or financial hold release.

The exact four-file package has no authorization or attempt. After approval, append
its authorization exclusively, record the exact authorized directory metadata change,
run real local preflight and independent arming review, then give one user-attended
Mac command; the agent never runs SSH. No repeated manual GPU/key paste. Preserve
all historical claims/failures and inspect bound saved evidence on Done/error.
Actual candidate completion and broader research/training remain unqualified.

## 2026-11-23 — Candidate-only one-attempt approval accepted and armed

The user explicitly approved the reviewed 24045e50d1db73c39925dbda25ee4a3f5b57eb1132e7a428eb4d3d78be7963db
outer scope. The exact authorization was appended once with exclusive creation;
its 1,281 bytes hash to c1ebed1a4c6d6a51979f8e31bf19ff8c9769d8537c9780eda8d7caeb442f158a.
No control, source, package path or prior file changed. Real local preflight passes
twelve source pins, 45 uploaded sources and both archives. Standalone saved review
v1 passes 30,361 prior hashes / 33,640 fixture entries with exactly one authorized
package-directory append exception: inode, device, ownership, mode and old file
stats/hashes fixed; APFS nlink 6→7 and mtime changed. The older exception remains.
No old checks/science rerun, process/timer/SSH/GPU/proof or target work started.

The grant is armed and unspent. No further approval is needed for this exact
attempt. Next provide one user-attended Mac command after VPN; retain fifty-minute
login-inclusive connection, proof/candidate-only scope and all original caps/gates.
The agent never runs SSH. On Done/error, inspect bound local attempt/session/error
records first, never repeat or reset the command/claim. Historical native/PGD/clean
cohorts, candidate failure, broader research/training gates and finances stay.

## 2026-11-23 — Candidate component completes; split mechanism diagnostic closes

The user-attended candidate-only grant is consumed. SSH exits 0 and the
[independent saved review](../report/checks/2026-11-23-pgd-candidate-handoff-v1-target-review.md)
passes first run: full 2.5 GiB proof, candidate PID 1645944 with 23 phases/47 rows,
source/reference/route/backend/fresh-value/restoration/resource/completion/parent
and terminal-verified collection all join. All four target native-relative checks
pass unchanged bounds with zero failed elements; forward files are exact. Backend
v4 matches 2,236 raw rows, four typed associations and nine genuine correlated CUDA
witnesses, preserving all overhead/annotations. Candidate parent wall is 9.535349 s,
RSS 1,193,832,448 bytes, allocator 873,463,808 bytes; these component observations
are not complete attack throughput or instantaneous quotas.

The declared split mechanism diagnostic is closed using the accepted eight-gradient/
four-pair PGD result, completed native component and this new candidate chain. The
old failed candidate/session statuses and all source/test/auditor/fixture/failure
versions remain unchanged. No native or completed-cohort rerun, new FP64 reference,
full-model trace or production admission-flag rewrite is added. The perturbed
check remains native-relative; the earlier clean FP64 scope is different.

All 134 collected files / 95,918,135 expanded bytes and the 91,865,380-byte wire
verify independently, with all journal rows and exact controls/sources. Local
review uses stdlib hashes, metadata, source text and NPY headers/payload byte hashes,
without array value decoding/recomparison or project/test/scientific callbacks.
30,388 prior hashes / 33,663 fixture entries remain, old FIFO lstat-only. The exact
approved package AUTH/attempt/session lifecycle changes nlink 6→7→9 and mtime while
inode/device/ownership/mode and old five file hashes remain. Old exceptions remain.

Next is a separate local opt-in input-gradient whole-PGD workload/replay/performance
qualification path with original starts, scoring/selection/projection/restoration,
matched complete units and explicit overhead. Hardware memory/exact recovery,
complete study economics and financial reconciliation, training-only calibration,
data/protocol and explicit adoption remain. Training PGD5/EOT1, routine EOT1,
parameter-update backward, registry/YAML and financial caps/holds remain unchanged.
No new university action, scope/approval request, retry, separate retrieval or paid
work follows from this completed review. Research training remains unadmitted.


## 2026-11-23 — Keep whole-PGD qualification separate from mechanism completion

The [local whole-unit adapter](predictor_pgd_workload_v1.md) preserves the frozen
attack/evaluation flow with fresh per-iterate input-gradient contexts. Complete
analytic primary/convergence controls, exact fresh-process analytic replay, one
new tiny frozen CPU bridge objective and caller-level timing controls pass.
The 24 distinct checks required 26 executions: two final-mode failures were
corrected and only those failed nodes repeated; two timing cases were new.
Old failed records and every source/test version remain preserved.

Keep this entry small and CPU-only with execution binding null. Analytic receipts
must deny native ownership; the one actual frozen tiny objective does not qualify
a complete native workload or target replay/performance. Caller timing includes
restoration and observed guards/journals, with no instrumentation subtraction or
throughput extrapolation. The independent stdlib audit preserves all 30,552 prior
hashes and 33,850 prior fixture entries without repeating numerical work.

Next derive a separate target-sized whole-unit runtime and test ownership,
continuing clocks, resource/parent, replay and collection with explicit fake ports
before any new attended scope. Do not reopen the closed mechanism diagnostic,
repeat completed scientific cohorts, widen old guards, register a training policy
or change study/calibration settings. Target workload/replay/performance, hardware
memory/recovery, economics, calibration and protocol/adoption gates remain. All
remote grants are consumed; financial caps and holds remain unchanged.


## 2026-11-23 — Keep target whole-unit OS binding separate from cold derivation

The [target-sized whole-PGD runtime](predictor_pgd_workload_runtime_v1.md) emits
47 source files with a 46-module cache, preserving the reviewed attack/evaluator
flow and frozen scientific ownership/math methods. All 26 new checks pass once.
Separate proof/original/candidate/fresh-replay roles join through explicit fake
OS/compute ports, the original scratch Guard and bounded priority collection.
Two complete transports are independently reconstructed; failed/partial records
and every source version remain unchanged. The successful earlier CPU gradient
and completed scientific cohorts were not repeated.

Use a new 720-second whole-unit design bound while retaining the 120-second owned
objective bound, original RSS/allocator/call caps and continuing2400/3000-second
work/connection ceilings. The derived monitor changes only deadline, cadence and
schema; ownership and cleanup predicates retain exact inverses. Fake clocks,
processes, proof and numerical receipts establish no physical enforcement or
scientific/performance admission. Real proof verification refuses the false local
proof receipt. No Torch/NumPy/model/gradient, GPU, real process/signal/timer, storage
proof, SSH or paid work ran in this stage.

Next derive the concrete OS/bootstrap/executable-cache/client adapter with hard
unit/objective timers, proof/live GPU ownership, parent resource and collection
joins, and test locally before any new scope, handoff or approval request. Keep
all live bindings null meanwhile. Target workload/replay/performance, hardware
recovery/memory, economics, calibration and data/protocol/adoption gates remain.
Mechanism completion stays closed; study/training policies and financial caps/
holds are unchanged. User stages/commits/pushes.
