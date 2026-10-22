# Opt-in native Conv3d integration

**October 21 retry:** the first full-model integration attempt stopped at the
small native probe because it expected `col2vol`; the verified CUDA trace uses
`vol2im_kernel`. Use the [separate corrected handoff](native_probe_fix.md).
The original attempt remains failed/preserved; the corrected probe and full
pretrained native execution still need CUDA validation.

This is a diagnostic candidate, not a frozen study execution policy. The
original study and staged proposal YAML are unchanged. Earlier full-model
CUDA recovery failed; the native candidate has not yet passed that gate.

## Explicit policies

`scripts/check_workflow.py` and `scripts/run_workflow_pilot.py` accept
`--execution-policy original|cudnn-fp32|native-fp32`. The default is `original`.
The workflow pilot retains its existing strict deterministic default.

- `original`: retains the existing model construction and precision flags.
- `cudnn-fp32`: original context/target Conv3d, strict deterministic execution,
  cuDNN TF32 disabled, matmul TF32 disabled, highest FP32 matmul precision.
- `native-fp32`: the same strict FP32 policy, with the context patch module
  explicitly dispatching both Conv3d forward and backward without cuDNN.
  The clean frozen target keeps its original patch module.

The latter two options support CUDA diagnostics and CPU fixtures only. Configure
them before CUDA initialization. They do not silently fall back to MPS, install
packages, adopt a study backend, change the predictor, or open held-out panels.
They require fresh targets, calibration and output directories.

## Why backward needs its own scope

PyTorch 2.8's convolution derivative queries backend flags again during
backward; it does not retain forward's cuDNN-enabled flag. See the pinned
[autograd derivative definitions](https://github.com/pytorch/pytorch/blob/v2.8.0/tools/autograd/derivatives.yaml)
and [convolution implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/Convolution.cpp).
Wrapping only forward in `cudnn.flags(enabled=False)` would leave the original
slow backward path possible.

`NativePatchEmbed3D` retains the original `nn.Conv3d` object, parameter objects,
geometry, tensor names and checkpoint keys. A scoped autograd delegator calls
PyTorch `F.conv3d` in forward and `aten.convolution_backward` in backward, both
with cuDNN disabled. It does not replace the patch map with a learned or
linear layer and does not supply a hand-derived approximate gradient.

The delegator explicitly supports **first-order gradients only**. Our PGD
adversaries are detached before outer optimization; the implementation does
not differentiate through attack optimization. Higher-order differentiation,
autocast, concurrent Python model execution and other PyTorch/CUDA versions
are outside this diagnostic's acceptance scope. The private ATen boundary
requires validation on the pinned PyTorch 2.8.0 CUDA environment. CPU tests
on the local PyTorch version cannot establish that CUDA boundary.

## Artifact and cache identity

Runtime identities now include actual context/target patch classes and
geometry, the explicit policy, native backward implementation version,
activation checkpointing settings, determinism, cuDNN flags, matmul precision,
and cuBLAS workspace configuration. Evaluation, features, calibration loading,
training, saved-anchor loading and resume bind that identity.

Fresh in-memory target caches check the runtime/backend and target tensor
versions before reuse. Training also checks parameter identity and execution
drift before updates, saves and restores. Full target hashes remain part of
save/restore checks. Tensor-version checks are process-local mutation guards,
not a cryptographic substitute for target hashes or permission to use `.data`
to bypass PyTorch mutation tracking.

Old artifacts lacking the new identity are rejected. Identical weight values
and state keys do not make two execution policies interchangeable. Strict exact
resume equality has not been relaxed.

## Next university check

The prepared source-only handoff is
`runs/pretrained/diagnostic/cuda-native-model-handoff-20261021/vjepa-native-model.tar.gz`.
Its manifest and tracked preparation report record source/archive hashes.
It creates a fresh sibling directory, `cuda-native-model-20261021`, and reads
the existing university repository's upstream checkout, checkpoint and pilot
training cache without modifying that repository. No downloads or installs.

The detached launcher runs four fresh workers: two strict cuDNN FP32 and two
native FP32 attempts. Every worker opens exactly the first verified training
clip, keeps clean targets frozen, and measures clean native loss/input
gradient, FGSM, PGD-5, and losses on an independent scoring-mask draw.
Each draw retains both upstream mask families. Bounds are 4/255 and PGD step
1/255. A small native backward probe checks parameter/input gradients after
the forward context has exited and retains actual CUDA operator/kernel names.

Two workers within each backend must replay exactly. Across backends, clean
loss and input gradient must meet the predeclared elementwise bounds in
`models/native_diagnostic.py`. Data, original weights, clean target signatures,
masks and all shared runtime flags must agree. Attack perturbations need only
replay within their own backend: sign-gradient optimization can amplify small
cross-backend rounding differences. Such differences are retained, not hidden
by requiring equal adversarial pixels across algorithms.

Per-worker wall limit is 180 seconds; parent worker scheduling limit is
840 seconds; the detached outer supervisor stops after 900 seconds plus a
5-second kill grace. Parent setup can exceed its scheduling limit but remains
inside the outer limit. The CUDA allocator cap is 7 GiB (also at most 90% of
visible VRAM); it excludes driver and other users' allocations. OOM, timeout,
missing CUDA kernel evidence, or incomplete replay is a failure, not a pass.
All four workers stop at the first worker failure. Logs and partial JSON remain.

This check performs **no full-model parameter update or optimizer recovery**.
Full AdamW previously exceeded the university GPU's 8 GB. The small backward
probe and CPU optimizer tests do not resolve full ViT-L optimizer memory or
RTX 5090 replay. Those require a later bounded acceptance pilot after this
check passes. Single-attempt timings include warm-up effects and are not
steady-state throughput or a study cost estimate.

## Launch and collect

On the Mac, copy the archive:

```bash
cd /Users/kaiwendu/Documents/GitHub/V-JEPA-Robust-Overfitting
scp runs/pretrained/diagnostic/cuda-native-model-handoff-20261021/vjepa-native-model.tar.gz \
  kdu@linux10609.dc.engr.scu.edu:Desktop/VJEPA-Robust-Overfitting/
```

On the university SSH connection:

```bash
(
  set -e
  cd "$HOME/Desktop/VJEPA-Robust-Overfitting"
  test ! -e cuda-native-model-20261021
  tar -xzf vjepa-native-model.tar.gz
  bash cuda-native-model-20261021/scripts/university_native.sh
)
```

Use the printed `tail -f` command to watch progress. Ctrl-C exits `tail` only;
the detached diagnostic continues. SSH can be closed. Reconnect to
`linux10609.dc.engr.scu.edu` to inspect the process. Do not relaunch into an
existing extracted folder or overwrite a previous run.

When the launcher exits, copy the entire printed run directory to a fresh
local directory under `runs/pretrained/diagnostic/`, including worker JSON,
tensor evidence, session JSON, logs and exit status. An exit status alone is
insufficient: verify all artifact/source hashes and recompute replay/numerical
comparisons before calling the CUDA check accepted.

## Local checks

```bash
.venv/bin/python -m pytest -q tests/test_native_execution.py tests/test_native_diagnostic.py
```

The new checks cover FP64 numerical derivatives, original parameter/state-key
identity, full small-model gradients with activation checkpointing, PGD,
fixed targets, exact small-model clean/adversarial recovery, artifact/backend
mismatch rejection, and four fresh CPU processes. The actual CUDA backward
kernel test is skipped without CUDA. No CPU fixture result is a pretrained
CUDA result. Calibration, novelty refresh, full-model recovery, and measured
budget admission remain required before research training.
