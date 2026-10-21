# Bounded CUDA component profile

Prepared October 21, 2026 after the [full-model workflow outcome](../report/checks/2026-10-21-cuda-workflow.md).
The strict retry measured a 16.011x primary-attack slowdown, but did not identify
the responsible operation. This diagnostic measures synthetic components before
another rental. It does not load a model checkpoint, videos, optimizer, or data
splits. No study setting or scientific implementation is replaced.

**Completed university outcome:** all 48 CUDA workers and all 12 strict replay
cases passed. The original patch Conv3d input backward increased from 1.102 ms
to 3.400 seconds under strict settings on RTX 3070; default attention gradients
failed exact replay. See the [verified outcome](../report/checks/2026-10-21-university-cuda-components.md).
The [first same-Conv3d comparison](../report/checks/2026-10-21-university-cuda-convolution.md)
rejected all options before timing because the TF32-permitted cuDNN reference
fails the fixed FP64 gate. The [completed TF32-off control](../report/checks/2026-10-21-university-cuda-convolution-precision.md)
resolves that isolated rejection and qualifies native original-Conv3d execution
with observed exact replay and much lower component cost. Opt-in integration
and full-model RTX 5090 gradient/recovery/performance checks remain pending.

## What it measures

The [runner](../scripts/profile_cuda.py) delegates to
[cuda_profile.py](../models/cuda_profile.py). It checks the existing upstream
revision and clean scientific source paths, then uses actual upstream
`PatchEmbed3D`, `apply_masks`, and `RoPEAttention`. Separate cases isolate SDPA
with the interleaved qkv layout produced by Linear, and the predictor's original
advanced-index sorting expression. The full RoPE case includes its actual
rotations, SDPA, and output projection; the SDPA-only case is a component
surrogate and does not assert identical post-rotary strides.

| Component | Representative FP32 shape |
| --- | --- |
| Patch embedding | `[1,3,16,256,256]`, original kernel/stride `[2,16,16]`, 1024 outputs |
| Encoder masking, both families | `[1,2048,1024]` to 960 / 400 context tokens |
| Predictor masking, both families | `[1,2048,384]` to 1088 / 1648 target tokens |
| Predictor sorting | `[1,2048,384]`, original concatenation's sort order |
| Encoder SDPA and RoPE, both families | 16 heads, head dimension 64, 960 / 400 tokens |
| Predictor SDPA and RoPE | 12 heads, head dimension 32, 2048 tokens |

Masks use the first primary-attack mask seed from the failed pilot, reduced for
upstream's signed counter: `6221833`. These are one representative draw's
shapes, not all masks encountered by the study. CPU-only fixtures explicitly
use smaller dimensions and cannot provide CUDA evidence.

Each of the 12 cases runs twice in fresh processes for each policy: 48 workers
in total. The default policy reproduces the original deterministic flags;
strict policy uses the repaired configuration. Both retain the existing
precision flags. Each worker warms up once, measures two forwards/backwards,
and optionally profiles one additional iteration. The profiled iteration and
progress-file writes are excluded from timing medians. Inputs, random module
weights, masks, cotangents, outputs, and gradients have finite-value checks and
signatures. Observed exact replay requires matching operands, runtime, source,
and gradient signatures across both workers; two observations are not a
universal reproducibility guarantee.

Default gradient mode profiles **input gradients**, matching the attack path.
`--parameter-gradients` additionally differentiates isolated module parameters;
no optimizer is constructed. This option needs a separate fresh output and is
not enabled by the first university launcher.

## Bounds and evidence

Defaults allow 20 seconds per worker and 480 seconds for the whole diagnostic,
including worker startup. The [university launcher](../scripts/university_profile.sh)
adds a 540-second external supervisor, allowing termination/finalization.
Timeout cleanup terminates the worker process group. Slow/error/OOM cases
remain incomplete; they never count as successful replay. A 4 GiB CUDA allocator
limit is configured per worker; it does not include driver allocations or
prove a case fits on a particular GPU.

`session.json` is committed atomically after each worker and includes the
active task on interruption, statuses, verified result hashes, source/runtime
identity, and same-operand/same-precision timing comparisons. Per-worker progress
records show whether construction, forward, backward, or optional profiling was
active at a timeout. Partial timing records remain available for inspection,
but do not pass coverage checks. Incomplete diagnostics return a nonzero exit.

The optional [PyTorch profiler](https://docs.pytorch.org/docs/2.8/profiler.html)
records the most expensive CPU operators and CUDA activity when available.
Unavailable CUPTI/device events are recorded rather than fabricated. A CPU
operator table can still identify attention backend selection. Event timings
may overlap; do not sum them into wall time. A slow isolated component on an
RTX 3070 is a diagnostic lead, not proof of the same kernel behavior or measured
throughput on the RTX 5090. Full-model replay remains a separate acceptance test.

## Run on the existing university host

The package uses the existing Python environment at
`$HOME/.venvs/vjepa/bin/python` and the existing pinned upstream checkout.
CUDA profiling requires the already validated PyTorch `2.8.0` environment.
The CPU fixtures use the existing local runtime and record its version.
No installation or download is performed.

From the Mac repository directory:

```bash
scp runs/pretrained/diagnostic/cuda-profiler-handoff-20261021/vjepa-cuda-profiler.tar.gz \
  kdu@linux10609.dc.engr.scu.edu:Desktop/VJEPA-Robust-Overfitting/
```

Then in university SSH, on the host with the working RTX 3070:

```bash
cd "$HOME/Desktop/VJEPA-Robust-Overfitting"
test ! -e cuda-profiler-20261021
tar -xzf vjepa-cuda-profiler.tar.gz
bash cuda-profiler-20261021/scripts/university_profile.sh
```

The launcher verifies every packaged source against `MANIFEST.json`. Its default
upstream path is the existing neighboring
`V-JEPA-Robust-Overfitting/external/vjepa2` checkout. If it is elsewhere, pass that
actual path as the launcher's one argument. Preserve existing folders; a rerun
uses a new run directory, not overwritten results.

It prints the actual host, detached job PID, run directory, and watch command.
You may exit SSH after launch; reconnect to the printed host to inspect it.
Shared home files persist, while the job runs on that host's GPU.

A completed attempt writes `exit_status.txt` in the printed run directory and
`results/session.json`. Send the final log/status and result JSON for review.
A timeout is useful evidence; do not repeatedly rerun a slow case or relax the
deterministic guard without inspecting it first. No cloud Pod is required.

## Interpret before optimizing

First distinguish forward from backward cost and identify the actual expensive
operators. Compare only matching operand, source, and precision records. Validate
outputs and both input/parameter gradients of any proposed equivalent operation
against the pinned reference before changing model execution or claiming speed.
This profiler introduces no optimized replacement. A subsequent representative
CUDA check and full-model recovery still precede research budget admission.
