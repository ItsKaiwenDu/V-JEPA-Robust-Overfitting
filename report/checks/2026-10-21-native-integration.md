# Native Conv3d integration and university handoff — October 21

The opt-in diagnostic integration is implemented and locally checked. The
native candidate has **not** been accepted on the full pretrained CUDA model.
The study/proposal YAML remain unchanged and unadmitted.

## What changed

`original` remains the default. The separate `cudnn-fp32` and `native-fp32`
diagnostic policies select strict determinism, matmul TF32 off/highest, and
cuDNN TF32 off. Native context execution preserves the original Conv3d object,
parameters, geometry and checkpoint keys. The clean target patch stays original
and frozen. Both convolution directions explicitly scope cuDNN off: PyTorch
2.8 re-queries its flags during backward. The delegator calls PyTorch's own
kernels and explicitly supports first-order differentiation only.

Runtime/artifact identities now bind actual patch geometry/backend, activation
checkpointing, precision and determinism. Fresh target caches detect runtime,
backend and target-version drift. Calibration, saved anchors, evaluation,
features and recovery reject incompatible identities. Training checks execution
and parameter identity before updates/save/restore and after an effective
update. Exact recovery equality is retained. Old artifacts lacking the new
identity require a fresh diagnostic, not silent migration.

See [the implementation guide](../../docs/native_execution.md) and
[machine-readable evidence](2026-10-21-native-integration.json).

## Local evidence

- Final complete suite: **312 passed, 2 actual-CUDA skips, 95 warnings**, in
  **111.31 seconds**, Python 3.11.16 / PyTorch 2.14.1 on arm64 CPU fixtures.
- Meaningful checks cover FP64 numerical derivatives, original parameter and
  state-key identity, full small-model loss/input/parameter gradient agreement,
  activation checkpointing, PGD bounds, frozen clean targets, exact clean and
  adversarial interrupted recovery, and incompatible cache/calibration,
  evaluation, anchor and resume rejection.
- Four fresh workers from the **extracted archive** passed in **6.720 seconds**
  on generated CPU pixels and small actual upstream models. Gradients and
  FGSM/PGD-5 adversaries replay exactly within each policy. The CPU policies
  also have equal clean loss/input gradients in that fixture. This is not
  evidence of CUDA kernel dispatch or pretrained/full-model performance.
- Partial, duplicate, mixed-source/scale/precision/geometry/checkpointing,
  timed-out and altered-hash evidence is rejected. Only declared backend
  differences are removed from cross-backend contract comparison.
- An initial targeted feature test caught a missing guard initialization;
  it was repaired before the final passing suite. Two CUDA-only checks remain
  skipped locally; they are not counted as passes.
- Scaffold/YAML check, shell syntax and whitespace checks passed. Raw fixture
  exports and test logs are ignored; their checksums are retained in the JSON.

## Source handoff

Archive: `runs/pretrained/diagnostic/cuda-native-model-handoff-20261021/vjepa-native-model.tar.gz`

- Bytes: **102777**.
- SHA-256: `ddac322987ffe8a1883f3fe6fa615dda2acad8ac77afb5831c73726f6e698fb5`.
- **43 regular members**, with **42 manifest-listed files** plus `MANIFEST.json`.
- Base HEAD: `7c3ee216f7d4ba73b9544af087df861e96a232e9`; the source manifest binds the explicit
  working-tree overlay, not an invented committed revision.
- All archive member hashes match the current declared source inputs; member
  types/paths were checked before extraction. The extracted package passed the
  fresh-process CPU fixture, and its comparison was recomputed from saved
  tensor evidence. The source-only archive includes no weights, videos, SSH
  keys, credentials or upstream checkout.

The earlier failed/successful diagnostic archives and CUDA evidence are
preserved. The JSON records intentional existing Python changes and confirms
unchanged adapter, original patch implementation, PGD, effective-update math,
strict-runtime helper, preprocessing and study/proposal configuration.

## Next CUDA gate

Copy and launch using [the university instructions](../../docs/native_execution.md#launch-and-collect).
The fresh sibling folder reads the existing university environment, pinned
upstream, official checkpoint and first verified training clip. It performs
four fresh-process attempts: two cuDNN FP32 and two native FP32 workers.
Each checks full-model clean loss/input gradient, FGSM, PGD-5, independent
scoring masks, unchanged weights/targets and empty parameter gradients.
Native workers additionally profile a small input/parameter backward outside
forward's context to verify native CUDA dispatch and restored flags.

Within-backend replay must be exact. Shared data, target signatures, weights,
masks, precision, checkpointing and runtime must agree. Across backends,
predeclared clean loss tolerances are atol 2e-6 / rtol 1e-5; input-gradient
bounds are atol 2e-7 / rtol 2e-3. The criterion is elementwise absolute plus
relative tolerance. Adversarial perturbations may differ across backends;
they must independently satisfy bounds and replay within their own backend.
These tolerances were chosen before this full-model CUDA attempt.

Limits: 180 seconds per worker, 840 seconds scheduling budget, 900-second outer
supervisor plus 5-second kill grace, at most 7 GiB CUDA allocator memory and
90% of visible VRAM. The allocator cap excludes driver/other-user allocations.
Timeout, OOM, missing native kernel evidence and incomplete replay cannot pass.
All logs/partial records remain for review. No optimizer or rental is started.

## Remaining limits and spending

No full pretrained weights or real video pixels were loaded locally for this
implementation. No new university CUDA check has launched, and no new paid
compute was used. Reported spending remains **$1.195**, with **$8.805** left
in the shared pilot reserve; the overall $200 / compute $160 caps remain.
Confirmation/test pixels remain closed.

The university test cannot validate full ViT-L AdamW recovery on its 8 GB GPU.
Full-model optimizer/recovery acceptance and representative RTX 5090
performance/memory still require a later bounded pilot. Research calibration,
novelty refresh, near-duplicate review, protocol freeze and measured budget
admission remain pending. No overfitting finding or whole-model speedup is
claimed from the component or CPU fixture results.
