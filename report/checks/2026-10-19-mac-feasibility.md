# M4 Max feasibility measurement — October 18–19, 2026

The Mac successfully ran the full pretrained V-JEPA 2 ViT-L/16 loss,
input/parameter gradients, clean/adversarial optimizer updates, and the
primary attack workload. It is suitable for development and small pilots.
The measured full study is not a credible under-four-week commitment on
this Mac: training alone projects to 15.4 days, and evaluation is expensive.
These are implementation and hardware checks, not robust-overfitting results.

## Machine and reproducible workload

- Apple M4 Max, 32 GPU cores, 36 GiB unified memory; arm64 macOS 15.7.9.
- Python 3.11.16, PyTorch 2.14.1, torchvision 0.29.1, timm 1.0.30,
  einops 0.8.2, NumPy 2.4.6; temporary isolated environment.
- MPS recommended working set: 27 GiB. CPU fallback and fast math were unset;
  allocator limits were unchanged. Restricted tool execution hid MPS; normal
  host GPU access exposed it with the same interpreter.
- Upstream revision: `204698b45b3712590f06245fbfba32d3be539812`.
- Checkpoint SHA-256:
  `5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012`.
- Eight existing UCF101 training clips, each `[1,3,16,256,256]`, FP32,
  microbatch one, accumulation eight, activation checkpointing, cached clean
  targets. Clip-index hash and source-file hashes are in the JSON report.
- Both conditions restart from the released weights. Each has one warmup
  and five measured updates. Optimizer settings are in the
  [pilot guide](../../docs/gpu_pilot.md). Training PGD uses five steps,
  one restart, radius 4/255, step 1/255, and fixed masks within each attack.
- The full pilot completed in 528.14 seconds. It saved measurements, not
  trained research checkpoints. No final-test clips were used.

## Correctness and the MPS projection bottleneck

The original Conv3d context patch projection had a slow input backward pass
on this runtime. The optional linear implementation reshapes nonoverlapping
patches and applies the original weight and bias. It preserves parameter
identity, state keys, and the mathematical map; the fixed target encoder
retains its original Conv3d. The upstream source and checkpoint are unchanged.
The default remains `conv3d`; all optimized Mac commands explicitly select
`--patch-embedding linear`.

| One-clip measurement | Original MPS projection | Equivalent linear MPS projection |
| --- | ---: | ---: |
| FGSM plus independent-mask scoring | 24.60 s | 1.45 s |
| PGD-5 plus independent-mask scoring | 121.29 s | 4.84 s |

The full-model PGD-5 diagnostic is about 25 times faster. Reported clean,
attack-bank, independent-mask losses, and perturbation maxima match between
these two MPS runs. This is not a study-wide speedup guarantee or a claim of
bitwise equality for arbitrary clips. The isolated projection benchmark
also passed, but its much larger speed ratio is not a full-model result.

CPU/MPS comparison on the real clip passed the thresholds recorded before
comparison. CPU loss was 0.5805838108 and optimized MPS loss was 0.5805839300.
Input-gradient relative L2 error was 0.00684 (0.684%), with cosine similarity
0.999978. Target-feature relative L2 error was 0.000117. Two selected parameter
gradient tensors also passed; this is not a comparison of every parameter
gradient or a full training trajectory.

The full pretrained smoke checks passed native/upstream loss agreement,
finite nonzero gradients, replay, and frozen targets. During the full pilot,
both conditions updated the context encoder and predictor while leaving
the target state unchanged. No memory-sampling errors were recorded.

## Measured throughput and memory

Source: [full optimized pilot JSON](2026-10-19-mps-linear-pilot.json).

| Workload | Mean time | Measured range | Sampled maximum MPS driver allocation |
| --- | ---: | ---: | ---: |
| Clean update, effective batch 8 | 9.09 s | 8.81–9.47 s | 10.18 GiB |
| Adversarial update, effective batch 8 | 49.33 s | 47.46–51.22 s | 10.20 GiB |
| PGD-20, two restarts, four attack-mask draws, four independent scoring draws; one clip | 161.09 s | One measurement | 5.11 GiB |

The largest sampled live-tensor allocation was about 8.00 GiB. MPS memory
was sampled every 0.1 seconds; these are observed maxima, not exact peaks.
Driver allocation includes caches and does not equal total system RAM use
or the CUDA memory requirement. Model construction/loading and full-dataset
target-cache residency are outside these memory windows.

## Runtime projection and decision

For 2,020 training clips and effective batch eight, use 253 updates per epoch.
The draft has 30 epochs, three seeds, and two conditions:

```text
training hours = 253 × 30 × 3 × (9.093182 + 49.330720) / 3600
               = 369.53 hours = 15.40 uninterrupted days
```

This uses measured batch-eight updates, not scaling from the earlier
batch-two diagnostic. Five timed updates per condition are a short sample;
multi-day stability and sustained speed have not been established.

Evaluation must be budgeted separately at the measured 161.09 seconds per
clip. The examples below are projections, not selected protocol changes:

| Evaluation schedule | Additional projected time |
| --- | ---: |
| One pass over 505 clips for one checkpoint | 22.60 hours |
| All 505 validation clips at every epoch of all six runs (180 checkpoints) | 169.49 days |
| Only 32 monitoring clips at those 180 checkpoints, using the same strong attack | 10.74 days |
| Primary test evaluation of 505 clips at up to 13 distinct checkpoints: pretrained plus best/final for six runs | 12.24 days |

Even training plus the 13-checkpoint primary test example totals about
27.64 days before routine monitoring, stronger attacks, epsilon sweeps,
probe fitting/direct probe attacks, data preparation, checkpoint I/O, or
interruptions. Fewer distinct best/final checkpoints could reduce duplicated
test work, but cannot be assumed in advance. A 32-clip monitoring subset
does not cover all 101 classes and is not proposed here as a scientifically
adequate default. Merely shrinking validation does not resolve all costs.

**Recommendation:** Keep this Mac for development, correctness checks, and
bounded pilots. Before committing to the full study, budget a fixed monitoring
schedule plus all required evaluation and compare a measured faster CUDA
option, or explicitly revise scope before confirmatory runs. No rental is
started or approved by this measurement. The university RTX 3070 remains
unbenchmarked because of its reported driver mismatch; MPS measurements
cannot determine its CUDA runtime or exact memory requirement.

During the pilot, `pmset` reported AC power while battery charge fell from
71% to 53%. This does not diagnose the cause, but a long local run needs a
power connection that maintains charge, along with an awake machine. No
power or sleep setting was changed. A shut-down Mac cannot continue a local
job. Resume support is still to be implemented and tested.

## Evidence and verification

- [Original MPS smoke check](2026-10-18-mps-smoke.json) and
  [original CPU/MPS comparison](2026-10-18-cpu-mps-parity.json).
- [Original attack check](2026-10-18-mps-attacks.json) and
  [short original-projection pilot](2026-10-18-mps-conv3d-short-pilot.json).
  That pilot used accumulation two and reduced evaluation; its extrapolation
  is superseded for planning by the measured batch-eight optimized pilot.
- [Isolated patch projection](2026-10-18-mps-patch-projection.json),
  [optimized CPU/MPS comparison](2026-10-18-cpu-mps-linear-parity.json),
  [optimized smoke check](2026-10-18-mps-linear-smoke.json), and
  [optimized attack check](2026-10-18-mps-linear-attacks.json).
- [Full optimized pilot](2026-10-19-mps-linear-pilot.json), with package,
  checkpoint, clip-index, source hashes, per-update timings, and memory samples.
- 51 local CPU tests pass, including projection output, input-gradient, and
  weight/bias-gradient agreement. The separate full-model GPU checks are not
  part of this count. Upstream deprecation warnings remain recorded.

The reports were produced from a temporary worktree based on project commit
`03e87ddf8f78810ccece272b9ad548c1cf03486e`, with source changes not yet committed.
Current optimized report source hashes match the accompanying implementation.
The original-projection reports predate the explicit performance option;
their exact source is preserved by
[the baseline patch](2026-10-18-mps-baseline.patch). Applying that patch to a
clean checkout of the base commit reproduced all source hashes recorded by
the four original-projection reports. Use a separate checkout for replay:

```bash
git checkout --detach 03e87ddf8f78810ccece272b9ad548c1cf03486e
git apply /absolute/path/to/2026-10-18-mps-baseline.patch
```

The [Mac guide](../../docs/mac_pilot.md) contains current commands. No full
experiment, final-test evaluation, Lambda rental, or paid compute was used;
the $200 allowance remains unspent.
