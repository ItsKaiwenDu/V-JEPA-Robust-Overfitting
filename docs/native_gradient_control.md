# Controlled derivative investigation — university handoff

**Status: university export verified; completed control rejected both FP32 patch forward-oracle gates.** The corrected university
comparison remains rejected. This plan diagnoses the cause; it does not change
the native loss, adopt a backend, widen acceptance thresholds, or admit
research training. See [the verified comparison](../report/checks/2026-10-21-university-native-model-comparison.md).

## Verified university outcome

Run `derivative.aIsykm` completed in 66.915 seconds. Both FP32 forward paths
reject the unchanged FP64 output gate: 12 cuDNN and 35 native token values.
All four same-cotangent backward/oracle comparisons pass; both reconstructed
losses/gradients equal the prior failed run bit-for-bit. Identical-token tail
replay is exact. Forward-token changes alter incoming derivatives; eleven L1
residual signs cross, without proving that crossings alone explain all of the
change. The original rejection remains rejected. Read
[the full verified outcome](../report/checks/2026-10-21-university-gradient-attribution.md).

**Next:** the [separate shared-forward control](shared_forward_control.md) is
implemented with passing local checks and a university handoff; CUDA remains
pending. It establishes a shared forward precision policy that
passes the existing FP64 output gate, then compare backward execution using
identical tail inputs. A candidate is FP64 patch-projection accumulation cast
to FP32 tokens, with declared new precision/reference identities and unchanged
parameters/normalization/targets/tail. No pretrained CUDA run of that control has launched.
CuDNN-forward/native-backward alone still inherits the cuDNN forward failures.
The transfer commands below describe the completed control, not a new retry.

## Question

Is the discrepancy caused by the patch backward implementation at the same
incoming derivative, or by slightly different forward token values changing
the transformer/L1 incoming derivative? Both could contribute. The saved
export contains final input gradients and adversaries, but no patch-output
cotangents or prediction residuals, so it cannot resolve that distinction.

For backend b, the chain rule is `g_b = J_patch,b.T @ c_b`, where c_b is the
loss derivative at the patch output. Comparing g values alone changes both
the patch derivative execution and its incoming cotangent. L1 can change its
derivative discontinuously when a prediction residual crosses zero; that is
a hypothesis here, not an established explanation of the observed failure.

## Frozen inputs and environment

Use the same pinned checkpoint, first already-open training clip, target
features, mask draw 0 (both families), strict FP32 execution, TF32 off,
highest matmul precision, activation checkpointing and PyTorch 2.8.0 CUDA.
Bind to the verified `native.5S6yHj` source/data/runtime and tensor hashes.
Keep original weights/target, normalization and prediction loss unchanged.
Confirmation/test pixels and all new dataset panels remain closed.

The university RTX 3070 is sufficient for the previously measured input-only
workload. This investigation must have a detached bounded launcher, fresh
outputs, explicit progress/partial evidence, source hashes, the existing
7 GiB allocator cap and a short hard wall deadline. No optimizer, installation,
download or rental should start. The implemented worker deadline is 300 seconds, supervised by a 330-second
outer timeout with a five-second kill grace. The CUDA allocator cap is
min(7 GiB, 90% of visible VRAM), excluding driver/other-user allocations.
This is a bound, not a promised completion time.

## Controlled measurements

1. Capture the actual original/native patch-output tokens on the same normalized
   clean input. Preserve and record their shapes, strides, dtype and values;
   compare them with the existing independent CPU FP64 patch reference.
2. At an explicitly supplied identical patch-token input, run the unchanged
   transformer, predictor and L1 loss to obtain one incoming cotangent. Any
   temporary leaf-token substitution is a labeled diagnostic intervention,
   removed reliably on success/failure; it is not an adversarial-training path.
3. Apply each original patch backward to that **same** cotangent. Compare both
   raw-pixel VJPs with an independent CPU FP64 analytical patch VJP, including
   the original channel normalization derivative. The existing independently
   tested unfolded-patch analytical reference can supply this oracle.
4. Repeat using the incoming cotangent at each backend's own forward token
   values. Retain prediction residuals and count sign/zero crossings per mask
   family. This separates a derivative-execution discrepancy from changed
   incoming derivatives. Keep algorithm/order/layout controls explicit.
5. Reconstruct each observed full-model input gradient through the chain rule
   and compare with the saved `native.5S6yHj` gradient. If reconstruction is not
   faithful, diagnose instrumentation/layout/runtime effects before attributing
   a cause. Preserve all negative checks and raw evidence.

Do not interpret similar cosine similarity, successful random component tests,
or a small global relative error as a substitute for these controls. Do not
change the L1 objective to a smooth training objective to make this comparison
pass. Any smooth auxiliary control, if later justified, must be explicitly
labeled and cannot replace native-loss acceptance evidence.

## Verification and decision boundaries

The local checks validate the intervention on small actual upstream CPU
models, including exact hook/module restoration, original parameter identity,
fixed targets, preserved RNG/modes, no accumulated parameter gradients, and
comparison with independently computed FP64 derivatives. The actual-CUDA
kernel boundary must remain explicit. Preserve original rejected artifacts.

If shared-cotangent VJPs disagree with the oracle, inspect the backward path
before continuing. If both agree while the incoming cotangents differ, quantify
that forward/L1 sensitivity and its attack implications; do not silently
retrofit acceptance criteria or relabel the old failure as passed. A defensible
numerical-policy decision, representative examples and full-model optimizer
recovery still precede study adoption. No outcome of this one-clip control
alone establishes held-out robustness, overfitting, or budget admission.


## Implemented control and interpretation

`models/gradient_attribution.py` keeps the original context module installed.
A standalone native wrapper shares its original projection parameters. Each
forward-token source is supplied to a temporary patch-output hook, with a
**distinct preserve-format leaf for each mask family**. The unchanged tail
produces separate incoming derivatives. Each backward path then receives the
same ordered family derivatives, with one shared original normalization. This
preserves the original branch accumulation rather than summing derivatives in
FP32 before the patch backward. Hooks and Python/NumPy/Torch RNG are restored
on failure; parameter identity, modes, target versions, weights, target values
and absence of accumulated parameter gradients are checked.

Both actual trained-weight incoming derivatives are compared with the existing
CPU FP64 unfolded-patch analytical reference, including raw-pixel RGB standard
deviation. Its family sum is FP64. Forward tokens retain the existing component
output gate (rtol 1e-4, atol 2e-5); fixed-cotangent raw-pixel VJPs and saved-gradient
reconstruction use the existing full-model derivative gate (rtol 0.002,
atol 2e-7). Saved loss reconstruction retains rtol 1e-5, atol 2e-6. No prior gate
is widened. Token/cotangent cross-source error and sign counts are descriptive;
they are not substitute acceptance criteria.

A repeated identical-token tail must reproduce loss, predictions, residuals and
incoming derivatives exactly within this process. This control does not claim
fresh-process or full-model optimizer replay. Prediction residual sign/zero
changes are counted by family. The four VJPs separate the incoming-derivative
effect at the cuDNN backward from the backward-path effect at the native
incoming derivative; vector norms need not add. Raw token/derivative/prediction/
residual/oracle tensors are saved after each stage, each with a SHA-256 hash.
Atomic progress is retained in `control.json`. A successful control is recorded
as `control_checks_passed`; `backend_adopted` and `research_result` stay false.
Even successful reconstruction does not retroactively pass `native.5S6yHj`.

`OBSERVED.json` pins the rejected session, the two replay-0 worker JSON/tensor
hashes and contracts, and all 17 original project/32 upstream source hashes.
The current runtime, weights, data identity, clean targets and first mask bank
must match that saved run. Only those existing training pixels are opened.
The actual native CUDA kernel observer is repeated and retains raw events
before rejection. All original model/loss/attack modules and study YAML remain
unchanged by this implementation.

## Transfer and detached launch

Archive: `runs/pretrained/diagnostic/cuda-gradient-attribution-handoff-20261021/vjepa-gradient-attribution.tar.gz`.
Extracted folder: `cuda-gradient-attribution-20261021`. Reuse the existing
university environment/assets and `cuda-native-model-fix-20261021/runs/native.5S6yHj`
read-only. No installation, download, optimizer or cloud rental is started.

On the **Mac**:

```bash
cd /Users/kaiwendu/Documents/GitHub/V-JEPA-Robust-Overfitting
scp runs/pretrained/diagnostic/cuda-gradient-attribution-handoff-20261021/vjepa-gradient-attribution.tar.gz \
  kdu@linux10609.dc.engr.scu.edu:Desktop/VJEPA-Robust-Overfitting/
```

Inside **university SSH on linux10609**:

```bash
(
  set -e
  cd "$HOME/Desktop/VJEPA-Robust-Overfitting"
  test ! -e cuda-gradient-attribution-20261021
  tar -xzf vjepa-gradient-attribution.tar.gz
  bash cuda-gradient-attribution-20261021/scripts/university_gradient.sh
)
```

The launcher prints the fresh run directory. Progress stages appear in
`results/control.log`; `launcher.log` records startup and final exit status.
You can exit SSH after launch and reconnect to the same host. Ctrl-C exits a
`tail -f` viewer before entering another shell command; it does not kill the
detached job. A timeout, partial JSON, or rejected control is not a pass.

Preserve/copy the whole printed run directory after completion. Review hashes,
identity checks, both same-cotangent oracles, saved-gradient reconstruction,
residual changes, timing and actual CUDA evidence before attributing a cause.
New numerical-policy work, representative checks and full optimizer recovery
remain required before research adoption. Spending stays $1.195; no paid
compute is needed for this handoff.
