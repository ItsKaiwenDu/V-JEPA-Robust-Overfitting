# Proposal: one original efficient/math attention pair

**Offline design only.** No new operator, launch helper, authorization or attempt
claim is prepared. Both university hosts can remain idle. The
[verified original CPU-return capture](../report/checks/2026-11-04-oracle-boundary-v1-target-review.md)
passes after CUDA setup and matches the CPU-only controls, but it stops before
GPU input transfer. This proposal advances to that next boundary without
explaining or waiving the historical CPU refusal.

## Question and proposed cohort

For the same frozen 400-token encoder-1 fixture, do the unchanged original
efficient and math workers pass their CPU, GPU numerical, backend and replay
checks when every actual return is preserved before acceptance?

Use **linux10603 only**, with its pinned RTX 3070 UUID/driver and the existing
PyTorch 2.8.0+cu128, Python 3.11.17 and NumPy 2.4.6 environment. Run two fresh
processes sequentially:

| Order | Original worker | Reference |
| --- | --- | --- |
| 1 | encoder-1 / efficient / replay 0 | None, matching the original first worker |
| 2 | encoder-1 / math / replay 0 | Completed, passing efficient worker from this same attempt |

The original reference check requires a complete/passing worker, matching common
contract and bound `worker.json`/`values.pt` hashes. The preceding CPU-boundary
capture is intentionally incomplete and cannot serve as this GPU reference.
The historical efficient result is retained as evidence rather than substituted
for a fresh baseline. Any failure or incomplete capture stops the pair; the
candidate must not start after a rejected baseline.

The original four-worker session also requires replay-1 workers. This stage
contains only its first pair. Never call the frozen parent/summarizer to label
two workers as a complete original session, disable its coverage checks, or
claim fresh-process replay qualification from this pair alone. Linux10609,
the 656-token/predictor fixtures, full-model training and study evaluation are
outside this proposed cohort.

## Preserve the original work and capture the missing evidence

Each original worker calls the CPU FP64 oracle once, then executes six attention
iterations if all gates pass: untimed, profiled, warmup, timed-1, timed-2 and
timed-3. Each iteration includes one SDPA forward and one Q/K/V vector-Jacobian
product. Thus the pair permits at most **two original CPU oracle calls and
twelve GPU attention iterations**, with no extra warmup or capture-only call.

A separately named observer/operator must call the unchanged worker. Preserve
original configuration, operands/layouts, thread requests, forced backend,
precision, checks, exception records and completion flags. Do not edit or extend
the consumed CPU-boundary observer: it must retain its stop before transfer.

Capture the actual CPU return before its original gate. Capture all six actual
GPU returns at their original return boundaries, before the caller can reject
them. Save GPU outputs and gradients as **FP32**, with observed dtype, shape,
stride and device metadata before CPU copying. Do not reuse the FP64-only
capture writer in a way that silently converts these values to float64. Check
logical byte identity independently from comparison-metric conversions.

The frozen worker already saves its inputs, transferred inputs and first
untimed GPU result before its numerical gate. Preserve those snapshots and
verify the new untimed return agrees with the original `values.pt`. Later
returns currently have replay signatures; new captures must retain their actual
arrays too, including a mismatching result before its original gate can reject
it. Do not regenerate operands, use a previous capture as a replacement oracle,
or overwrite a failed/partial artifact.

Keep the CPU gate at `rtol=1e-12, atol=1e-14`. Keep original component/reference
bounds: output `rtol=1e-4, atol=2e-5`; dQ/dK/dV `rtol=2e-3, atol=2e-7`.
Require the original within-process exact replay, finite/nonzero values,
runtime/flag restoration and actual CUDA forward/backward backend evidence.
An arm label alone is insufficient. Preserve all profiler rows; never remove
unfavorable observations to satisfy backend recognition.

New detailed library/thread introspection must follow the original worker's
completion or refusal, not precede its first CPU oracle. Code identity, observer
call coverage and restoration must be checked. A capture-write error is a
technical refusal, not a reason to continue without evidence or retry.

### Timing and qualification limits

Copies and capture I/O can appear inside the original profiler and affect
subsequent synchronization, clock state and cache warmth. Keep original timing
observations and trace events, with these effects disclosed. **Do not use these
instrumented timings for a study-cost projection, speed claim or optimization
adoption.** A later performance experiment needs its own reviewed protocol.

Original worker completion is distinct from new capture coverage and the pair
decision. Both original workers and all planned captures/checks must succeed
before a pair can pass. A successful pair still does not complete the original
replay-1 coverage, qualify the full model or admit research execution. The old
approximately 1e-10 refusal and missing historical computed arrays remain.

## Proposed resource and transport bounds

| Bound | Proposal |
| --- | --- |
| Host / concurrent scientific workers | linux10603 only / one |
| Additional attempts / retries | one / zero |
| Work, including setup and proof | at most 600 seconds, one continuing outer clock |
| Fresh original workers | at most two, each at most 120 seconds within that clock |
| Storage proof | full 2.5 GiB write/fsync/readback, at most 60 seconds |
| Scratch | fresh private mode-0700 directory under `/var/tmp`, on the checked local XFS device |
| Initial capacity / runtime floor | 8 GiB / 1 GiB, plus inode checks |
| Observed setup / capture ceiling | unchanged 3 GiB / 2.25 GiB |
| Raw capture category | **proposed 320 MiB**, increased from the preceding single-return capture's 256 MiB |
| Individual tensor file | unchanged 32 MiB; refuse if an artifact cannot fit |
| Original Torch allocator fraction cap | unchanged 4 GiB per worker; no simultaneous workers |
| Collection | same attended SSH connection, at most 600 seconds / 3 GiB of existing files |
| Paid compute / new financial hold | none |

Static evidence budgeting allows 240 MiB for two workers, an additional 32 MiB
for a transient atomic-save name pair, and 48 MiB remaining under the proposed
320 MiB raw category. This explicit 64 MiB category increase covers additional
return evidence; it changes no numerical tolerance or consumed allowance. The
overall capture ceiling stays 2.25 GiB. The math worker's future PT extents have
not been measured; an oversize result must refuse rather than expand a bound.
Observed accounting and per-process file limits do not establish an aggregate
kernel quota, exclusive reservation or future free capacity.

Keep the unchanged 85,622,679-byte / 139-member synthetic packet. The future
overlay's exact file list, bytes, hashes and control must be finalized and
tested before a resource request. Its current design ceiling is 2 MiB, within
the 128 MiB code category. Include no private keys, credentials, financial
ledger, weights, research pixels or prior raw captures in the upload.

Use one attended Mac SSH connection with the verified public host key and
strict host/UUID/driver/foreign-compute guards. Leave foreign processes alone;
stop only owned children on refusal. Do not promise detached/overnight survival.
Keep the Mac awake and its lid/Terminal open through collection. Collection
reads a finished attempt only; it cannot regenerate values, extend work or
restart a worker. Retain remote scratch; no cleanup is proposed.
The frozen exporter omits NPZ files. The new collection manifest must explicitly
cover every CPU/GPU capture; do not assume an old export contains that evidence.

The original worker validator's 1200-second schema may be retained only in a
clearly bound compatibility plan; the original parent is not invoked and the
new outer 600-second bound controls. Issue each worker's original authorization
with at most 120 seconds remaining, capped by the same outer deadline. Never
reset the outer clock for worker two or count proof/upload time twice as new
allowance. Temp/capture/cache paths and visibility handling must be reviewed
in the new operator; do not inherit the CPU-only supervisor's hidden CUDA mode.

## Implementation gate before any university request

Next implement a distinct unarmed operator and meaningfully test: untouched
original function/source identities and fresh-reference dependency; real tiny
CPU workers covering all six phases; preservation of FP32 returns and failing
arrays; capture errors, existing hooks and restoration; profile/capture event
interactions; partial coverage and blocked candidates; cold extracted full
package imports with dispatch blocked; storage/deadline/foreign-process guards;
one-connection upload/collection, corrupt transfer, consumed grants and unarmed
refusal before SSH or claim creation. Tiny CPU tests cannot qualify CUDA.
Use the original tiny math workers (including a fresh math reference) for local
rehearsals; never label a CPU result as the live efficient CUDA arm.

Independent saved-data review must compare the new captures with original
snapshots, frozen FP64 references, original gate decisions, fresh efficient
reference, backend rows and all replay signatures. Retain post-context and
observational comparisons with the accepted CPU return without making them
replacement references or inventing a causal result.

This proposal has no runnable helper and requests no resources. Existing grants
are consumed. After the concrete operator and payload pass local review, seek
one separate explicit grant for this broader GPU-execution scope; generic
continuation cannot reuse the preceding CUDA-setup-only grant. All project and
compute caps, diagnostic holds and pending billing reconciliation stay unchanged.
