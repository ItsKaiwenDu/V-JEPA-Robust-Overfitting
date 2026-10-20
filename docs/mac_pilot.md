# Apple GPU correctness and feasibility

The October 18–19, 2026 local checks used an Apple M4 Max (32 GPU cores) and
36 GiB unified memory. PyTorch 2.14.1 on Python 3.11.16, arm64 macOS 15.7.9,
reports MPS built and available when run with normal host GPU access.
The restricted execution environment reports MPS unavailable with the same
interpreter. That difference concerns process access, not absent GPU hardware.
The recommended MPS working set is 28,991,029,248 bytes (27 GiB); this is
shared memory, not a separate VRAM allocation or a guarantee that every
workload fits.

The [completed pilot](../report/checks/2026-10-19-mac-feasibility.md) measured
9.09 seconds per clean batch-eight update, 49.33 seconds per adversarial
update, and 161.09 seconds per primary evaluation clip. The Mac can execute
the workload, but training alone projects to 15.4 days across the draft six
runs. Full evaluation and monitoring do not yet fit the four-week goal.

## Reproduce the checks

Use the [model environment instructions](smoke_test.md), the pinned upstream
checkout, `checkpoints/vitl.pt`, and the prepared eight training clips. The
tested local Python environment is temporary; do not assume `.venv` already
contains these packages. Keep `PYTORCH_ENABLE_MPS_FALLBACK` unset or `0` for
these checks so an unsupported model operation fails rather than silently
running on the CPU. Leave allocator limits at their defaults.

From an activated environment and the repository root:

```bash
python scripts/verify_setup.py --require-mps
python scripts/smoke_test.py \
  --checkpoint checkpoints/vitl.pt --device mps --patch-embedding linear \
  --expected-sha256 5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012 \
  --output runs/pretrained/diagnostic/mps-smoke-01/smoke.json
python scripts/check_mps.py \
  --checkpoint checkpoints/vitl.pt \
  --mps-patch-embedding linear \
  --clip data/processed/pilot-train-v1/v_ApplyEyeMakeup_g08_c04.pt \
  --output runs/pretrained/diagnostic/mps-parity-01/parity.json
python scripts/check_attacks.py \
  --checkpoint checkpoints/vitl.pt --device mps --patch-embedding linear \
  --clip data/processed/pilot-train-v1/v_ApplyEyeMakeup_g08_c04.pt \
  --output runs/pretrained/diagnostic/mps-attacks-01/attacks.json
```

The comparison uses identical released weights, raw pixels, and masks on
CPU and MPS. It compares loss, clean targets, full input gradients, and two
selected parameter-gradient tensors. Thresholds are recorded in the output.
It does not establish equality of every parameter gradient or full training
trajectories. The smoke diagnostic reduces gradient norms in float64 on CPU
because MPS does not support that dtype; model forward/backward stays on MPS.

## Optional equivalent context patch projection

`--patch-embedding linear` on the smoke, attack, and pilot commands selects
a patchwise matrix multiplication for the context encoder. The default
remains the upstream `conv3d` implementation. The target encoder keeps its
original projection. Kernel and stride are both `(2,16,16)` with no padding,
so flattening independent patches in `[channel,time,height,width]` order
and applying the reshaped original weight gives the same affine map.
The wrapper reuses the actual parameters and keeps checkpoint state keys;
it does not initialize new weights or change the prediction objective.
Floating-point reduction order can differ and must be checked.

On the checked training clip, the original and linear MPS paths produced
the same reported clean/attack losses and perturbation maxima. PGD-5 took
121.29 seconds with the original projection and 4.84 seconds with the linear
projection. This is a one-clip diagnostic, not a study-wide speedup guarantee.
The commands above explicitly select the faster checked path. Keep the
backend and projection choice consistent across experimental conditions and
record them in every run; CUDA validation remains separate.

Use `--mps-patch-embedding linear` on `scripts/check_mps.py` to compare that
MPS implementation against the original CPU Conv3d path. The separate
`scripts/check_patch_embedding.py --output <new-report.json>` isolates native
and linear projection output/input-gradient correctness and timing at the
full clip shape. That microbenchmark uses random projection weights; only
the full checkpoint checks establish behavior of the actual research model.

## Measure the actual workload

After correctness checks pass:

```bash
python scripts/pilot.py \
  --checkpoint checkpoints/vitl.pt \
  --clips data/processed/pilot-train-v1 \
  --device mps --patch-embedding linear --steps 5 \
  --output runs/pretrained/diagnostic/mps-pilot-01/pilot.json
```

Use a new output path for every invocation. The pilot uses FP32, one clip
per microbatch, gradient accumulation to effective batch eight, activation
checkpointing, and cached clean targets. Each condition starts from the
same released weights and has one warmup update. No pilot-trained weights
are saved as experiment outputs. It then times the primary PGD-20 / two
restart / four-mask attack and independent four-mask scoring on a training
clip. The [GPU guide](gpu_pilot.md) explains optimizer and cost assumptions.

For an initial bounded check, the pilot also accepts `--accumulation 2
--steps 1 --evaluation-steps 2 --evaluation-restarts 1 --evaluation-masks 4`.
This uses two clips per update and a shorter attack. Its training projection
scales the observed update time to effective batch eight, including optimizer
overhead; it is an extrapolation, not a measured batch-eight update. The report
records the actual settings and marks primary evaluation as unmeasured.

The code synchronizes the selected GPU before and after timing. CUDA reports
an allocator peak. MPS reports allocation maxima sampled every 0.1 seconds;
these can miss transients. Driver allocations include caches and are not
equivalent to live tensor bytes or total system RAM usage. The report labels
these separately and never presents a sampled maximum as an exact peak.

Use update times for the six-run training projection and evaluation time
for a separately chosen validation schedule. Evaluation, target-cache
construction, I/O, probes, and interruptions are excluded from the reported
training-only projection. A small warm-cache pilot is not a multi-day
stability test. Local jobs require the Mac to stay powered on and awake;
the pilot does not alter sleep settings or launch a background service.

References: [PyTorch MPS](https://docs.pytorch.org/docs/2.14/notes/mps.html),
[MPS memory/timing APIs](https://docs.pytorch.org/docs/2.14/mps.html),
and [MPS environment variables](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html).
