# Shared higher-precision patch forward — October 21

**Subsequent outcome:** the four-worker CUDA export is now independently
verified and passes the input-only control, including shared forward accuracy,
gradient agreement and exact within-backend replay. See
[the verified CUDA outcome](2026-10-21-university-shared-forward.md). The
preparation-time record below remains historical.

**Local checks pass; full pretrained CUDA comparison is pending.** This is a
separate diagnostic precision/reference, not study adoption or a retroactive
pass of either failed FP32 run. See [the guide](../../docs/shared_forward_control.md).

Both arms compute the same learned context patch projection through FP64
unfold/matmul and separate FP64 bias addition, then cast to FP32 in the original
token layout. Parameters/state keys and identities remain original FP32.
Normalization, the original frozen FP32 clean target, transformer, predictor
and native masked L1 remain as declared. Only the backward path differs:
strict cuDNN or native FP32 `aten.convolution_backward`, explicitly scoped.
This first-order affine VJP approximation is checked against CPU FP64 math;
higher-order differentiation rejects. The reference is explicitly **shared
FP64 forward / FP32 tokens / cuDNN FP32 backward**.

The new scope installs a temporary wrapper with a distinct policy that existing
training/cache/calibration/evaluation APIs reject. Its separate guard binds
actual forward/backward/precision/geometry/checkpointing/runtime/tensor
identities and versions. Original module/policy restoration runs on failure.
Historical mathematical/execution source modules and study/proposal YAML
remain byte-identical. No general research backend integration was added.

## Verified locally

- **373 passed, 2 CUDA-only skips, 119 warnings, 125.73 seconds**;
  28 new tests. Includes FP64 gradcheck, actual small upstream full-model
  input/parameter gradients against independent double-convolution autograd,
  raw-pixel/weight/bias VJPs, state/parameter/target identities, exception
  restoration, rejected legacy/runtime/mixed-source identities, rejected
  higher-order derivatives and bounded explicit CLI use.
- Rehashed the original pretrained checkpoint and first already-open training
  pixel signature. On its actual learned projection, both retained incoming-
  derivative sources × both CPU backward settings pass all unchanged forward,
  input, weight and bias gates. Maximum forward error **1.9058679541217316e-6**;
  zero failed forward values (the prior FP32 paths failed 12/35 values).
  CPU execution is not cuDNN/CUDA evidence.
- From the actual extracted source archive, four fresh upstream CPU workers
  complete in **7.078s**, with exact within-backend
  tensor/attack replay, bitwise shared forward tokens, and cross-backward
  input/loss agreement. All workers run the same new-delegator native sanity
  probe before resetting construction RNG; initialization order is symmetric.

The forward gate remains rtol 1e-4/atol 2e-5; raw-pixel VJP and full input
comparison retain rtol 0.002/atol 2e-7; weight/bias bounds remain their earlier
component values. No pointwise acceptance tolerance changed. Worker output
retains actual and FP64 reference tensors for every projection VJP; the parent
recomputes numerical gates rather than trusting stored pass flags.

Each worker then reconstructs the full input gradient through its new shared
forward and checks FGSM/PGD-5, bounds/ascent and independent scoring masks.
Exact attack replay is required within each backward. The clean original
training clip is the only pixel input. Original failed-run hashes/identities
bind source/data/weights/targets/masks; historical FP32 losses/gradients are
not redefined as this comparison's reference.

## Handoff and pending checks

Archive: `runs/pretrained/diagnostic/cuda-shared-forward-handoff-20261021/vjepa-shared-forward.tar.gz`.
SHA-256: `e4016ffbdfe5a8d95b7424ac88142af57d493aeae7089f6fa3a71efbd7c1895b`.
127,593 bytes, 53 checksummed files plus MANIFEST.json,
54 regular tar members; no weights, pixels, credentials or bytecode included.
[Source hashes, local results and limits](2026-10-21-shared-forward.json).

Reuse existing university assets and preserved failed run read-only. Four
fresh workers, at most 180 seconds each; scheduling budget 840 seconds; outer
900-second timeout plus five-second kill grace. CUDA allocator at most
min(7 GiB, 90% visible VRAM). No installer/download, optimizer, held-out pixels
or paid instance. Launch instructions are in [the guide](../../docs/shared_forward_control.md).

Actual new-delegator CUDA dispatch, trained-model forward/oracle agreement,
input-gradient comparison, fresh-process replay and timings remain unmeasured
for this precision. A passing input-only control still does not validate full
pretrained parameter updates/optimizer recovery, RTX 5090 behavior,
representative robustness, calibration or measured study budget admission.
Reported spend remains $1.195; pilot reserve $8.805. No commit or push.
