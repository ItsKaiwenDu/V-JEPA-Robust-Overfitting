# Saved component values pass the original native-relative bounds

The candidate forward output is byte-exact, and dQ/dK/dV pass the original
component tolerances with **zero failed elements**. Each return has 786,432 FP32
values. All four native payload hashes exactly match the original full-model
boundary. Four saved comparisons ran once; no new model or GPU work ran.

| Return | Maximum absolute error | Maximum error / original allowance | Failed elements |
| --- | ---: | ---: | ---: |
| Output | 0 | 0 | 0 |
| dQ | 8.640199666842818e-12 | 4.141192636807819e-5 | 0 |
| dK | 7.457856554538012e-11 | 3.621051061605147e-4 | 0 |
| dV | 2.523847797419876e-11 | 1.2498138492055534e-4 | 0 |

The elementwise bounds remain output `rtol=1e-4, atol=2e-5` and derivatives
`rtol=0.002, atol=2e-7`. The agreement function and tolerance assignment match
their original ASTs exactly. The fixed 23-file contract joins the collected NPY
bytes, route fingerprints, historical selection, operands/cotangent, separate
component controls and source boundary before decoding. The reader accepts only
bounded, finite, nonzero, read-only FP32 NPY arrays and never unpickles `.pt` data.

All **28 distinct new checks pass once / 28 executions**. They use tiny labeled
synthetic arrays or saved/modified metadata and do not repeat the actual saved
comparison. The independent standard-library audit **passes on its first run**:
**25,968 prior hashes / 27,996 prior fixture entries** preserved, all 23 input pins,
source ASTs and result/provenance joins checked. It does not invoke project/test
callbacks, decode the actual arrays or recompute their metrics. The old FIFO is
lstat-only; the new tests create no FIFO. All source snapshots and logs stay saved.

This closes the saved candidate-to-passing-native value check for the selected
captured input. The original candidate worker and attended session remain failed;
no missing resource/completion/parent-success record is invented. Canonical FP64
accuracy, full-EOT8 coverage, replay, performance, study cost and training remain
gated. The completed eight-gradient/four-pair cohort, original target code,
controls, consumed claims and financial caps/holds are unchanged. No SSH, upload,
GPU query, proof, objective, gradient, profiler, paid work or new scope was used.

Next: consolidate the preserved PGD, saved backend and component value evidence
into a readiness assessment, identifying the remaining qualifications before
full training. No completed numerical cohort or native component needs a repeat
for this saved-data review. Details:
`docs/predictor_pgd_component_returns_v1.md` and both component-return configs.
