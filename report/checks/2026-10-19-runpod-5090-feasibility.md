# RTX 5090 feasibility pilot — October 19, 2026

The original FP32 CUDA pilot passed on one Runpod RTX 5090. All 56 tests,
the native-loss/gradient smoke check, FGSM/PGD-5 checks, and optimizer invariants
passed. Both context encoder and predictor updated; the clean target stayed
fixed. No full study or held-out test evaluation was run.

## Measurements

| Component | Measurement |
| --- | --- |
| Clean batch-eight update | 4.0262 s |
| Adversarial batch-eight update | 24.4249 s |
| PGD-20, two restarts, EOT-4, four scoring masks | 73.4419 s per clip |
| Largest measured training allocator peak | 7.551 GiB |
| Pilot wall time, excluding setup/smoke/attack check | 273.26 s |

One warmup and five measured updates per condition, microbatch one, effective
batch eight, 16 frames at 256 pixels, original Conv3d path, fixed clean target,
FP32. Allocator peak excludes non-PyTorch memory and is not total device usage.
Only eight cached training clips were used; full dataset throughput may differ.

## Budget implication

The training-only extrapolation for 2,020 videos, 30 epochs, two conditions,
and three seeds is **179.95 GPU-hours**,
or **$215.94** at the displayed $1.20/hour.
Training alone therefore exceeds the $200 total research envelope on this
hardware at this rate. Evaluation adds substantial cost.

The existing proposed evaluation schedule projects
571.14 hours from
measured units plus 220.47
hours from attack-work proxies: a partial total of
791.61 hours, or
$949.93.
Fifteen additional categories remain unmeasured. These are extrapolations,
not a completed cost estimate; they do not authorize the full study.
See [projection](2026-10-19-runpod-5090-projection.json).

## Environment and provenance

Runpod PyTorch 2.8.0 template; actual Python 3.12.3, torch 2.8.0+cu128,
torchvision 0.23.0+cu128; driver 595.91.07. A 50 GB volume is mounted at
/workspace. The source handoff and clip archive checksums passed; 101 source
files and nine clip files matched their manifest. Upstream commit:
204698b45b3712590f06245fbfba32d3be539812. Checkpoint SHA-256:
5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012.

Direct GitHub and PyPI downloads stalled. Setup was recovered using the exact
upstream Git bundle and Linux Python 3.12 wheels transferred from the Mac.
The official checkpoint endpoint worked. Setup failures and recovery logs
are retained; this did not change scientific code. Reported source hashes
were checked against the local repository after export.

[Smoke](2026-10-19-runpod-5090-smoke.json),
[attack check](2026-10-19-runpod-5090-attacks.json),
and [pilot](2026-10-19-runpod-5090-pilot.json) are curated reports.
The full export, including environment, scripts, and setup logs, is locally
saved under runs/pretrained/diagnostic/runpod-export-20261019/pilot-results.tar.gz.
Export SHA-256: 68c877f67200a13e86eeb0d5407a8afc69e48790f29b1580017035191be74c2f.
The checksum matched on both ends. Pilot finished at 19:53:52 PDT.

## University comparison

User-provided terminal evidence from linux10609 showed 55 tests passed and
one FFmpeg test skipped, then successful CUDA smoke and input attack checks.
The RTX 3070 8 GB failed on the backward pass of the next clean update after
the first optimizer update. It requested 20 MiB with 15.56 MiB free; PyTorch
reported 7.30 GiB allocated. Adversarial training was not reached. This is
not a sustained training speed measurement or proof that every optimized
implementation must fail on 8 GB. The university reports have not yet been
copied from that host; this comparison is sourced from the user's transcript.

## Session state

No research job remained after the bounded pilot. The user confirmed termination
after stopping the Pod and verifying local export. Termination is user-reported;
actual billing has not been reconciled.
Do not interpret the $15 account credit as measured spending. No commit or
push was performed. Next: redesign or optimize the workload and re-estimate
cost before a longer rental; the original research protocol remains unchanged.
