# Native backward observer correction — October 21

The first full-model integration attempt stopped at the small native probe,
before full pretrained native model loading. The user-reported failure is
`Native backward CUDA kernel boundary was not observed`, launcher exit 1.
The complete failed export has not yet been copied locally or hash-verified;
no cuDNN exact-replay result or native CUDA pass is accepted from pasted logs.

The probe incorrectly searched for `col2vol`. The prior independently verified
university native Conv3d export records a positive-duration
`at::native::vol2im_kernel<float, float>` GPU launch. Its worker SHA-256 is
`0de375f98ff1c6d1c16091d78e0b5ac78cd7d3d07f3e655a0451236084f03028`.
The failed probe did not persist profiler names when raising, so its exact
observed names remain unavailable. The incorrect matcher is fixed; that does
not retroactively turn the failed attempt into a pass.

The schema-2 observer retains device-typed raw CPU/CUDA events, requires the
native reconstruction CUDA kernel with positive duration plus convolution
backward, rejects cuDNN execution, verifies finite gradients and restored flags,
and writes evidence before rejecting. Export validation recomputes the gate.
The actual-CUDA regression shares the implementation. CPU-only kernel-looking
names, zero-duration launches, unrelated kernels, missing backward, cuDNN
execution and falsified pass summaries reject in parser/regression tests.
Rejected worker JSON retains its probe observations.

**Local verification:** 327 tests passed, 2 actual-CUDA tests skipped, 95 warnings,
115.15 seconds; PyTorch 2.14.1, Python 3.11.16,
arm64 CPU. The extracted retry handoff passed four fresh CPU workers in
5.790 seconds; exact replay and saved-tensor comparisons
were recomputed. No new CUDA execution is established by those fixtures.

**Separate retry:** `runs/pretrained/diagnostic/cuda-native-model-fix-handoff-20261021/vjepa-native-model-fix.tar.gz`; 106,350 bytes;
SHA-256 `0943830e1f7cde5a261df3430b0b88dc5b3b3806f452f7759950c9d4dd1a4f13`. All 45 archive members are regular safe paths;
44 files are manifest-listed. The old archive/folder remain preserved.
The new folder is `cuda-native-model-fix-20261021`. Native workers run first,
then references are repeated under the same fresh source identity. The model,
first-order mathematical wrapper, optimizer, attacks, precision, tolerances,
exact replay and study YAML are unchanged. The new observer and retry order
are explicitly versioned and source-hashed.

Follow [the corrected launch guide](../../docs/native_probe_fix.md).
Limits remain 180 seconds per worker, 840 seconds scheduling budget,
900-second outer timeout plus 5-second cleanup, and 7 GiB allocator memory
(at most 90% of visible VRAM, excluding driver/other users). No optimizer step
or paid rental starts. The retry is prepared, not launched/accepted on CUDA.

Reported spending remains $1.195 and pilot reserve $8.805. Confirmation/test
pixels remain closed. Full pretrained native input gradients/attacks/replay,
full-model AdamW recovery, representative RTX 5090 measurements, calibration
and budget admission remain pending. See [hashes and limits](2026-10-21-native-probe-fix.json).
