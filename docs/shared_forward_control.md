# Shared FP64 patch-forward diagnostic

**Status: the bounded four-worker pretrained CUDA input-only control is
independently verified and passes. Integration/full optimizer recovery remain pending.** This is a separate diagnostic precision
policy and reference. It is not integrated into research training or adopted
by the study. Both earlier FP32 failures remain rejected. See the
[verified attribution](../report/checks/2026-10-21-university-gradient-attribution.md).

## Verified CUDA outcome

`precision.RvqoMB` passes every oracle/reconstruction gate, shared-token equality
and exact within-backend fresh-process replay. Input-gradient cross-backward
max difference is 7.451e-8 with zero sign disagreements; both attacks also match
bit-for-bit on this clip. Median two-single-attempt PGD-5+score is 6.324s native
versus 40.745s cuDNN. Read
[the verified outcome](../report/checks/2026-10-21-university-shared-forward.md)
and [the next integration plan](shared_forward_integration_plan.md). The launch
commands below document the completed control, not an instruction to rerun it.

## Precision and comparison reference

Both arms use the same nonoverlapping patch geometry and actual learned
projection parameters. The normalized FP32 context input is unfolded, cast to
FP64, multiplied by the FP64 view of the original weight, and then receives
an explicit separate FP64 bias addition. The result is cast to FP32 and given
the original Conv3d flatten/transpose token layout. No trained tensor is
replaced/reinitialized; FP64 temporaries live only within the context patch
forward. Normalization, transformer, predictor, original clean frozen target
encoder and native masked L1 remain as declared. Target encoding stays FP32.

The declared forward ID is
`nonoverlapping-unfold-fp64-mm-separate-bias-cast-fp32-transposed-v1`.
Backward uses first-order FP32 `aten.convolution_backward` at the original
operands, with explicitly scoped strict cuDNN or native dispatch. These VJPs
approximate the real affine derivative; they are checked against independent
CPU FP64 derivatives including the original normalization derivative. Higher
order differentiation is rejected. Both arms therefore have identical forward
arithmetic, differing only in backward execution.

The new comparison reference is **shared FP64 patch forward, FP32 tokens,
cuDNN FP32 backward**. It is not the historical all-FP32 forward reference.
Identity pins to the old failed run establish the same source/weights/data/
normalization/targets/masks/global runtime; they do not claim equality with
its loss/gradient values. Historical failed gates are never relabeled as passed.

`precision_scope` temporarily installs a wrapper sharing the original projection
object, with the same state keys and parameter IDs. Its distinct model policy
is deliberately rejected by existing cache/training/evaluation contracts;
only this separate diagnostic understands it. Its own contract binds forward
algorithm, transient/parameter/token dtypes, backward, geometry, activation
checkpointing, runtime and actual context/target classes. A guard detects
policy/precision/module/tensor/version drift. The original module/policy is
restored on success or failure. No old caches, anchors, recovery or calibration
records can be adopted under this diagnostic identity.

## Gates, evidence and scope

Existing pointwise limits are retained:

| Check | Relative tolerance | Absolute tolerance |
| --- | ---: | ---: |
| Patch forward vs CPU FP64 | 1e-4 | 2e-5 |
| Raw-pixel patch VJP vs FP64 | 0.002 | 2e-7 |
| Patch weight VJP vs FP64 | 1e-4 | 2e-4 |
| Patch bias VJP vs FP64 | 1e-4 | 1e-4 |
| Full input-gradient comparison/reconstruction | 0.002 | 2e-7 |
| Clean loss comparison/reconstruction | 1e-5 | 2e-6 |

A learned-weight projection/independent CPU FP64 gate runs before attacks in
each worker. One separate leaf per family obtains incoming derivatives through
the unchanged tail; original branch order and one shared normalization are
preserved. Oracle output/input/weight/bias tensors, actual derivatives and
full-model loss/gradient reconstruction are retained and recomputed by the
parent from checksummed saved tensors. An aggregate norm-report reproduction
tolerance only accommodates host reduction differences; no pointwise gate changes.

Four fresh workers (two per backward, native first) must replay outputs,
losses, gradients, references and attacks exactly **within** each backward.
Forward tokens must also match bit-for-bit **across** backwards. Cross-backward
full loss/input gradients retain their prior pointwise limits. FGSM and PGD-5
use raw epsilon 4/255, step 1/255, seed 0 and independent scoring masks 100000;
attack bounds, clean-candidate retention and ascent are checked. Exact attack
agreement is required within each backend, not across different rounding paths.

Every worker runs the same small native-backward sanity probe before resetting
construction RNG and loading the model. This keeps probe/initialization order
symmetric across both arms. The probe profiles this new delegator's native
backward after its FP64 forward. Retained device-typed events must show actual native CUDA vol2im plus
convolution backward, no cuDNN execution, finite input/parameter gradients and
restored flags. It cannot be inferred from a CPU fixture or another delegator.

Only the first already-open training clip is loaded. Projection parameter VJPs
and small-model full parameter checks are not a full pretrained parameter-update
or optimizer-recovery test. No optimizer, new data panel, training trajectory,
confirmation/test pixels, install, download or paid compute is started.

## Local evidence

The original checkpoint and first training pixel hashes were reverified.
With the actual trained projection and both saved incoming-derivative sources,
the CPU shared forward rejects zero values at the unchanged output gate;
maximum absolute error is 1.9059e-6. Input/weight/bias VJPs also reject zero
values, and both local arms return bitwise identical tokens. On CPU, both
backward settings execute CPU kernels, so this is not cuDNN/CUDA evidence.

Tests cover FP64 numerical derivatives, full small-model input/parameter
derivatives against independent double-convolution autograd, projection/state
identities, target freezing, guard failures, exception restoration, scoped
backward flags, bias-free parameter-only gradients, rejected second derivatives,
explicit CLI bounds and four actual upstream fresh-process workers. The full
project suite and extracted handoff checks are recorded in the
[local verification record](../report/checks/2026-10-21-shared-forward.md).

## University handoff

Archive:
`runs/pretrained/diagnostic/cuda-shared-forward-handoff-20261021/vjepa-shared-forward.tar.gz`.
Fresh extracted folder: `cuda-shared-forward-20261021`.

On the **Mac**:

```bash
cd /Users/kaiwendu/Documents/GitHub/V-JEPA-Robust-Overfitting
scp runs/pretrained/diagnostic/cuda-shared-forward-handoff-20261021/vjepa-shared-forward.tar.gz \
  kdu@linux10609.dc.engr.scu.edu:Desktop/VJEPA-Robust-Overfitting/
```

Inside **university SSH on linux10609**:

```bash
(
  set -e
  cd "$HOME/Desktop/VJEPA-Robust-Overfitting"
  test ! -e cuda-shared-forward-20261021
  tar -xzf vjepa-shared-forward.tar.gz
  bash cuda-shared-forward-20261021/scripts/university_precision.sh
)
```

Reuse the original environment, pinned upstream, checkpoint, first training
clip and `cuda-native-model-fix-20261021/runs/native.5S6yHj` read-only.
Source/observed-reference hashes are checked before execution. The launcher
prints the fresh `precision.*` run path. Atomic worker JSON records stages,
gates and saved artifacts before any rejection. The parent launcher log records
worker scheduling and the final status; each worker has its own detailed log.

Limits: four workers, at most 180 seconds each; scheduling budget 840 seconds;
outer timeout 900 seconds plus five-second kill grace; CUDA allocator capped
at min(7 GiB, 90% visible VRAM), excluding driver/other-user memory. These are
hard bounds, not a completion-time prediction. SSH can be closed after detached
launch. Reconnect to the same host to inspect progress. Ctrl-C exits a `tail -f`
viewer before typing shell commands; it does not stop the detached job.

Preserve/export the whole run even if it fails. Verify source/session/worker/
tensor/probe hashes, exact replay and numerical gates before interpreting
performance. Successful input-only comparison cannot establish full-model
optimizer recovery, RTX 5090 behavior, calibration, representative robustness
or admission under the measured budget. Reported spending stays $1.195 and the
remaining pilot reserve $8.805.
