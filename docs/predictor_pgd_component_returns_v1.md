# Accuracy of the saved component returns

All four candidate-to-native saved-return comparisons pass the original component
bounds, with zero failed elements. The forward output is byte-exact. The native
returns themselves exactly match the historical full-model boundary's logical
hashes. This closes the saved native-relative value check for this captured input.
The historical candidate worker and attended session remain failed.

## Evidence and comparison scope

The analysis reads the eight previously collected `output.npy`, `dq.npy`,
`dk.npy` and `dv.npy` files from the native and forward-preserving component roles.
Each contains 786,432 FP32 values with shape `[1,12,2048,32]`. The fixed input
contract pins 23 files: eight NPY files, 13 provenance/ownership/review metadata
files, the original agreement function source and the original tolerance source.
All inputs were already in the preceding preservation seal.

The metadata join retains the original `predictor.0`, draw 0, family 0, forward
row 24, `EOT8-before` selection. Both routes use the same saved Q/K/V and arriving
cotangent, source boundary and separate component execution control. Operand
strides, offsets, backing extents and aliases remain linked to the manifest;
V's noncontiguous original layout is not replaced by a new model calculation.
The route fingerprints, collected file/payload hashes and historical boundary
hashes join before numerical decoding. Native, candidate and original source PIDs
remain distinct. Candidate failure and source-supported backend metadata are
separate preserved observations.

The NPY decoder accepts only the bounded version-1 envelope, three unique header
fields, little-endian FP32, exact four-dimensional shape and C-order payload.
It rejects additional/truncated bytes, Fortran/object/foreign dtypes, Boolean
shape values, nonfinite data and all-zero returns. File reads check canonical
paths, owner, single-link regular-file identity, exact size/hash and held/named
identity stability. Arrays are read-only views over verified bytes; no pickle,
Torch deserialization or generic `numpy.load` runs. The serialized `.pt` source
operands remain opaque.

The agreement function is an exact AST copy of
`models/attention_saved_evidence.py::agreement`. The tolerance assignment is
AST-identical to `models/predictor_sdpa_numerical_receipts_v1.py::BOUNDS`.
Each element must satisfy `abs(candidate-native) <= atol + rtol*abs(native)`.
No allowance or threshold changes. Relative L2 remains a diagnostic, not a
replacement for the elementwise gate.

## Actual saved-value result

| Return | rtol | atol | Failed elements | Maximum absolute error | Maximum error / allowance | Relative L2 error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Output | 1e-4 | 2e-5 | 0 | 0 | 0 | 0 |
| dQ | 0.002 | 2e-7 | 0 | 8.640199666842818e-12 | 4.141192636807819e-5 | 3.6052752255697426e-6 |
| dK | 0.002 | 2e-7 | 0 | 7.457856554538012e-11 | 3.621051061605147e-4 | 6.254645797935905e-6 |
| dV | 0.002 | 2e-7 | 0 | 2.523847797419876e-11 | 1.2498138492055534e-4 | 4.096173862563291e-6 |

All four native logical payload hashes equal the original boundary's output and
derivative hashes. Native and candidate forward files are also byte-identical.
Candidate derivatives differ in bytes and pass their original numerical bounds;
they are not relabeled as exact. The actual analysis ran once with local NumPy
2.4.6 and single-thread CPU settings. It decoded eight arrays and made four saved
comparisons. No model objective, gradient, profiler, GPU, independent reference
oracle, replay or paid work was generated. The completed eight-gradient / four-
pair PGD cohort was not rerun.

## Validation and preserved limits

All 28 distinct new checks pass in 28 executions. They cover the exact frozen
arithmetic/bounds, saved metadata joins, six mismatched-context refusals, unsafe
file identities, a read-only NPY roundtrip, eight malformed headers/envelopes,
three nonfinite/zero cases and tiny exact/passing/rejecting numerical fixtures.
Actual saved arrays are decoded only by the one analysis run, not by these tests.
Synthetic arrays and modified metadata are explicitly labeled. Special-file and
multiple-link cases use fabricated stat metadata; no new FIFO is created.

The separate standard-library audit passes on its first run. It checks 25,968
prior hashes, 27,996 prior fixture entries, all 23 input pins, unchanged arithmetic
and tolerance ASTs, provenance/route/payload joins, the recorded elementwise
acceptance fields and every new check result. It invokes no project/test callback
and does not decode the actual arrays or recompute their metrics again. The old
FIFO stays lstat-only. Source snapshots before validation and all logs are retained.

This new receipt qualifies the four saved candidate-to-passing-native comparisons
at one captured input. Canonical FP64 reference accuracy, missing candidate
resource/completion/parent-success records, full-EOT8 backend coverage, fresh
replay, performance, study cost and training admission remain open. The original
failed worker, consumed claims, uploaded sources and controls are unchanged.
There is no new university scope, connection, upload, storage proof or permission
request. Project/compute/diagnostic caps and financial holds remain unchanged.

The reproducible local entry is `scripts/check_pgd_component_returns_v1.py`, with
a fresh output directory under the ignored `pgd-component-returns-v1-20261119`
stage. The existing `saved-comparison-v1/comparison.json` is immutable evidence;
there is no reason to repeat it now. Raw arrays stay in the ignored diagnostic
tree; source/configuration and the curated review are versioned.

Next local work is to join the preserved full-model PGD result, corrected saved
backend metadata and this value check into a readiness assessment that names the
remaining execution/reference/replay/performance gates before full training.
No completed numerical cohort or native component needs a repeat for this review.
