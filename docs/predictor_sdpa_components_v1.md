# Fixed-input attention component workers: local verification

This implements the next local step of the [fixed-input proposal](predictor_sdpa_fixed_input_v1.md),
using the actual saved boundary from the [native observer](predictor_sdpa_observer_v1.md).
It adds separate bounded reference/return storage, a component worker, a
device-associated backend recognizer, a local CPU entry and focused tests.
All predecessor scientific sources, captures, proposals, bounds and consumed
claims remain unchanged. There is no university operator or execution grant.

## One input set, one analytical reference, four processes

The local rehearsal binds the previously reviewed original `creator-0` native
capture by its worker and boundary hashes. It requires the original full
loss/input value controls, complete owned boundary record and exact source/runtime
identity. Q/K/V and native SDPA `grad_output` come only from those saved tensors;
there is no generator, model rerun, recapture or substituted incoming gradient.

The worker reconstructs recorded offsets, strides, backing extents and alias
relationships on zeroed storage, verifies whole logical byte hashes and records
the actual reconstructed layout. Saved contiguous copies do not preserve native
physical allocation, padding or alignment. Unobserved padding is explicitly
zero-filled; physical identity is never claimed. V retains its nonzero offset
and non-dense stride. Input bytes, versions, absent `.grad` values, runtime and
selection flags must remain unchanged throughout work.

The first original worker calls the frozen chunked NumPy FP64 softmax/VJP oracle
once, with query chunk 128. It publishes four typed FP64 arrays and a distinct
reference manifest plus completion marker. A reference publication is not a
passing scientific worker. Later workers require both that completed reference
and the preceding passing original worker. They bind the same reference bytes
and may never regenerate them. A fresh later math worker also passes with the
oracle function explicitly disabled in the test.

The local serial order is original-0, math-0, original-1, math-1. Each runs in
a fresh process with a continuing 120-second parent/worker deadline, no retries,
one untimed iteration and one subsequent profiled iteration. It loads the same
native Q/K/V and incoming gradient. Only the math arm scopes the public math-only
backend selection; the original retains its captured default flags.

Untimed output and dQ/dK/dV are saved before numerical gates. Original output
and derivatives must exactly reproduce the captured native boundary, including
signed-zero bytes. Both arms must pass FP64 component bounds. Math must also pass
those bounds against the passing standalone original. Replay workers must match
their own arm's first process bitwise. The profiled iteration is permitted only
after the untimed gates pass and must return identical bytes. There are no warmups,
speed trials, attacks, scoring, optimizer steps or held-out evaluation.

Output bounds remain rtol 1e-4 / atol 2e-5; dQ/dK/dV remain 0.002 / 2e-7.
Independent analytical bookkeeping remains 1e-12 / 1e-14. Full-model bounds are
unchanged. Readers recompute actual saved numerical checks rather than trusting
passing booleans, and require exact earlier manifests, markers, input identities,
runtime, flags and fresh PIDs. No candidate or research policy is adopted.

## Backend observations

The profiler keeps actual device type/index, CPU parent containment, phase
annotations, sequence/thread fields and CPU-operation kernel links. The original
family is detected from actual forward/backward operations, never assumed to be
efficient attention. The local original is CPU flash attention; the selected math
arm is observed as math. Neither constitutes CUDA evidence.

The CUDA recognizer requires positive non-annotation device events and actual
operator-associated kernels in both phases on device zero. Efficient attention
requires recognized forward/backward efficient kernel families. Math requires
associated matrix and forward/backward softmax CUDA kernels. Flags alone, names
without links, an unrelated kernel, the wrong device/phase or mixed families
refuse. Unrecognized CUDA flash/cuDNN/version-specific evidence refuses and is
preserved; it never silently forces a fallback. Handwritten CUDA rows test this
parser only. No actual CUDA comparison or full-model kernel qualification has
occurred at this milestone.

## Publication and failure evidence

Reference and worker formats are distinct from native observation or passing
full-model records. Private directories, regular mode-0600 files, ownership,
link/device/byte caps, exclusive partial files, fsync, atomic publication and
final marker hashes are checked. Four fixed FP64 and eight per-worker FP32 NPY
files have 16 MiB ceilings; NPY header shape/dtype/order/extent are validated
before allocation, with pickles disabled. Metadata and event streams have
separate caps. A prior partial, changed source/reference/return or failed marker
cannot be relabelled as completion.

Numerical refusals preserve all available untimed returns before profiling.
Profile/backend refusals retain actual profiled returns and event rows. A
nonfinite reference array is saved and rejected. Interrupted writes preserve
their partial bytes, keep the record incomplete and retain the primary exception
in the child log even if final metadata publication also fails. Completion is
never inferred from an exit code or a reference marker alone.

## Verified local evidence

61 focused checks pass, with zero failures, errors or skips. Four fresh local
workers use 1×2×32×32 FP32 saved native tensors and one canonical FP64 reference.
Both original workers exactly reproduce all four native boundary returns;
within-arm fresh replay and all profiled returns are exact. All component bounds
pass. An independent saved-array review reconstructs the canonical reference
under the original bookkeeping bounds and rechecks all 32 saved FP32 returns,
input layouts, actual CPU traces and predecessor bindings without another model,
autograd, SDPA, CUDA, process or network evaluation.

The first run had 59 passes and one test failure: the reference-as-worker check
expected only ValueError, whereas the correctly refused format has no worker
file and raises FileNotFoundError. Only that test expectation changed, and an
actual fresh-worker oracle-prohibition check was added. The second run passes
61 checks. Both exact source snapshots, logs and arrays remain; counts are not
summed. Prior 37/25/68-check suites were not rerun.

The saved reviewer initially refused on pytest's shortened directory names and
then on its duplicate `current` symlink. Only review-file discovery changed;
both reviewer versions/logs are preserved. No worker or test was rerun for those
review fixes. Live review stdout is excluded from the immutable input seal.

These are arm64 Python 3.11.16 / Torch 2.14.1 / NumPy 2.4.6 CPU results. They
are not a released-pretrained CUDA capture, capacity measurement or performance
benchmark. Same Q/K/V and incoming gradient still permit different private
backend forward state; a component comparison cannot isolate backward arithmetic
alone. The native full-model candidate remains rejected at 3,724 input-gradient
entries, and both four-cell decomposition orders and their interaction remain.

## Next stage

Prepare and test a separate bounded same-SSH operator/handoff under the inert
25-minute proposal: one original-only native capture, exact full native value
controls, then the four component processes, continuing time/storage/allocator
guards and preserved collection. Target capture, actual CUDA recognizer coverage,
pretrained replay, peaks and end-to-end acceptance are still pending. This local
entry refuses CUDA and the named university hosts before constructing output or
starting calculations. It provides no upload/helper, authorization, attempt,
storage proof, cleanup, retrieval, university or paid-compute action. Only after
the concrete operator is reviewed should one new exact scope be requested.

Both university sessions may idle or close. The $200 project / $160 compute /
single $10 diagnostic limits, $40 noncompute reserve, $9.8116 encumbrance,
$0.1884 unreserved amount, old $2 hold / $0.0792 gap, last reported $6.56 wallet
and pending reconciliation remain unchanged. Known Runpod Pods stay terminated.

See the [curated local review](../report/checks/2026-11-07-predictor-sdpa-components-v1-local-review.md).
