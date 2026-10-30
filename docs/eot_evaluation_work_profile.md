# EOT8 evaluation work profile

**October 30 outcome:** the completed university run passes exact observer and
fresh-worker replay. Backward is about 79% of unprofiled EOT8 time; attention
backward is about 54% of corrected direct device duration in the traced draw.
[Independent review](../report/checks/2026-10-30-eot-work-profile-review.md).
The original aggregation counted CUDA annotation ranges as kernels. Use the
separate offline reader `models/eot_trace_accounting.py` for corrected device
shares; captured sources and summaries remain frozen. The original launch
claim exists: never rerun the command below. Earlier preparation instructions
describe that completed one-attempt diagnostic.

The next diagnostic times full clean EOT8 loss/input-gradient controls and
attributes one original attack draw with both mask families. The previous projection-reuse result passed correctness but
improved its broader one-clip timing by only 1.021×. This profile asks where the
remaining time goes before choosing another optimization. It changes no tensor
operation, attack policy, numerical tolerance or study runner.

## Run once from the Mac

The handoff is prepared and unarmed. When ready to attend for up to **20 minutes**,
run this once in a **local Mac Terminal**, outside university SSH:

```bash
bash /Users/kaiwendu/Documents/GitHub/V-JEPA-Robust-Overfitting/runs/pretrained/diagnostic/eot-work-profile-university-20261029/run-attended.sh
```

Enter the university password locally at each prompt. Keep the Terminal and Mac
lid open until the command ends. The helper keeps the Mac awake through its
execution and downloads. Explicit invocation activates one free university
attempt on linux10609; it does not start or reserve any paid resource.

The total limit is 20 minutes, including remote extraction and verification,
with a five-second outer kill grace. Two fresh sequential workers have up to
eight minutes each within that total. The PyTorch allocator ceiling is 7 GiB.
Source, checkpoint, host/GPU UUID, driver and existing PyTorch 2.8/CUDA 12.8 verify
at runtime. The parent polls the strict foreign-CUDA-process gate every five
seconds. Failures stop without a model retry and only the diagnostic's own child
process group can be terminated. No installation or server setting changes.

The expected host is linux10609 and GPU UUID is
`GPU-4e2a7c77-19bd-689f-6c08-60eb774c0d52`. The user's preparation check shows
8,192 MiB total, 94 MiB used, driver 580.178.04 and no CUDA compute processes.
The launcher rechecks rather than treating that observation as a reservation.

linux10603 is a separate GPU usable for independently prepared parallel work.
This profile uses one GPU for both workers so its observations remain directly
comparable. No second-host job is prepared or started in this step.

## Measurement and evidence

The reference is the existing shared-FP64-forward/native-FP32-backward model.
Each worker loads the released checkpoint and the same single training tensor,
reconstructs the original eight attack draws with both mask families, computes
clean targets once, and verifies the prior model/target/pixel/mask/runtime
identities. No full PGD attack, scoring evaluation, optimizer or trajectory runs.
The research step size remains unselected.

After an EOT8 warm-up, each worker performs two unprofiled EOT8 controls, then
an unprofiled first-draw control, the same draw under profiling, the draw again
after profiler removal, and a final EOT8 control. EOT8 controls must agree
exactly with each other; the three first-draw controls must agree exactly with
each other. The clean EOT8 loss/input derivative also must
pass the unchanged original numerical bounds against the previously reviewed
university reference. Fresh-worker replay must be exact.

Temporary `record_function` wrappers label pixel validation/normalization, mask
validation, each draw's loss, patch projection, context encoder, predictor and
their block forward/recomputation. They remain active through backward, including
checkpoint recomputation, then restore the original methods even after errors.
Existing source bytes and scientific execution policies remain unchanged.

The profile records CPU and actual CUDA events. Shape, stack and allocation
profiling are disabled to limit observation overhead; those options can affect
tensor retention and execution cost according to the
[PyTorch 2.8 profiler documentation](https://docs.pytorch.org/docs/2.8/profiler.html).
The separately timed boundary guards are outside the objective timing. A CUDA
synchronization splits forward and backward, and is declared in the contract.
These are diagnostic timing choices, not ordinary evaluation cost units.

Raw event records retain CPU containment, intervals, exclusive CPU times and
CPU-linked device activities. The reader recomputes exclusive CPU time from
the retained intervals, rejects cycles/bad containment, and assigns each linked
activity once to its nearest annotation. Inclusive annotation totals are kept
separately and never added together. This follows the distinction in
[PyTorch 2.8 event accounting](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/autograd/profiler_util.py).
CUDA activities may include transfers as well as compute kernels. Raw device
interval unions, duration sums, linkage coverage and top activity names are
separate observations; summed durations are not wall time or a speedup forecast.

Retained profile data are bounded at 800,000 events and 256 MiB per worker.
Exceeding either bound fails without truncating a trace into a passing result.
The export caps uncompressed evidence at 1 GiB. Actual native CUDA backward
activity and both original mask families of draw0 must be visible in the detailed
trace before a CUDA pass can be accepted. All eight draws remain in the EOT8
controls, but the detailed first-draw activity mix cannot be claimed for all
eight draws or the full attack. CPU fixtures cannot be relabelled as CUDA.

The 17.2 MB packet has 98 source/input/reference files plus its manifest: 99
regular members. It preserves all 61 original scientific pins and exactly one
training clip. A prior diagnostic input derivative and scalar loss are included
as the numerical reference. Released weights, dependencies and upstream code
are reused and verified on the host. No private SSH keys, financial ledger or
confirmation/test pixels are uploaded.

## Download and review

The Mac helper downloads the existing evidence after the foreground launcher
ends. It verifies the archive and every regular member in a fresh local folder.
The initial login lookup now permits ten minutes for password entry. Reply
**Done**, or report an error and the saved folder. Never repeat the launch.
Upload retry is allowed only before model launch; a connection failure may retry
only the download once into a fresh folder. No model work is repeated.

After review, this command can retrieve existing completed output separately:

```bash
bash /Users/kaiwendu/Documents/GitHub/V-JEPA-Robust-Overfitting/runs/pretrained/diagnostic/eot-work-profile-university-20261029/download-only.sh
```

Keep the Mac awake during a standalone download. Both local and remote launch
claims remain exclusive. Older projection-reuse launch claims and evidence are
untouched. Independent review will combine the saved events/pass tensors with
the verified local preparation packet; the result export need not duplicate its
training/reference tensors.

## Preparation outcome

All **52 focused CPU checks** pass with zero skips: 43 profile/packet checks and
nine existing adapter checks. Two actual fresh tiny upstream workers preserve
all pass tensors exactly through profiler observation and removal. Tests reject
altered event timing, containment, trace summaries, gradient tensors, runtime,
seeds, fixture identities, resource records and artifact domains.

The final packet equals the final tested packet byte for byte. Four simulated
transport scenarios verify normal operation, upload/download retry boundaries
and refusal of an existing remote directory, without a real network connection.
An initial unlaunched packet is preserved after aligning the new scalar loss
shape with the prior reference. A second unlaunched preparation is also preserved: its CPU full-EOT8 trace had
200,832 events, motivating the bounded first-draw detailed capture. The final
controls retain all eight draws; no research setting is reduced. The corrected
sources and bounded packet are revalidated.

Real CUDA attribution and its observation overhead remain unknown. Profiled
timings cannot replace study-cost inputs. Full optimizer integration/recovery,
complete measured affordability, research calibration and trajectories remain
pending. No paid allocation or financial cap changes in this preparation.
