# Encoding candidate cost decision

The [local comparison](../report/checks/2026-11-24-pgd-encoding-cost-v1-local-review.md) rejects integration of the
byte-equivalent candidate on this local synthetic corpus. The candidate is retained
as evidence; the existing runtime and closed budget source remain unchanged.

**Reject this candidate for integration on the observed local corpus.** The
candidate took 1.869 times the baseline's aggregate measured time. Byte equivalence
alone did not produce a cost improvement. Keep the existing baseline and closed
budget gate. No replacement candidate is selected here.

| Arm | Batches | Records | Aggregate ns | ns/record |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 16 | 1808 | 36052540 | 19940.6 |
| Candidate | 16 | 1808 | 67377626 | 37266.4 |

One short local run used113 predeclared accepted synthetic record rows, no warmup,
and eight ABBA groups. All32 batch aggregate digests match the frozen expected
bytes. Candidate time exceeded baseline time in all eight groups. The entire
cooperative stopwatch interval was 0.103468s, with no5s cap stop or failure. No
assumed overhead was subtracted. This is a short observation, not robust
statistical calibration or a claim about other hosts/corpora.

The timed path includes per-row private state/token/kwargs construction, whole
record construction, JSON encoding, SHA256, parser/copies, append-spy bookkeeping
and aggregate verification hashing. Imports/AST extraction, corpus decoding and
artifact writes are excluded. No FileSink, full ledger, fsync, worker or scientific
lifecycle ran. The candidate path had higher observed cost; individual component costs were
not isolated.

Host/interpreter, every source/corpus hash, inclusions/exclusions and the full
ABBA order were saved in PLAN-v1.json before timing. Raw per-batch timestamps,
durations and digests remain in OBSERVATIONS-v1.json. Sources are extracted as
metadata functions only; no project test is imported or rerun. A dedicated local
stopwatch surrounds batches, with all workload clock/process/signal/timer/thread/
network ports blocked and zero attempted calls. Original workload clocks/caps
are untouched. Cooperative cap checks are not OS enforcement.

The independent saved audit passes on70862 prior hashes/77650 fixture entries,
plan/source/corpus pins, exact durations/sums/digests and decision joins. It invokes
no project/test/benchmark/scientific callbacks, subprocess, timer, signal, GPU,
proof or network ports. Ordinary metadata FS/hash/arithmetic work is real. No new
fixtures or scientific tests run. Historical FIFO LSTAT only, never opened;
device16777229 uses the frozen observation, with no probe/remap/waiver. Four prior
directory exceptions and target66307 gates remain unchanged.

The prior uncommitted prototype is preserved. HEAD8416b66 stays unchanged; user
alone stages/commits/pushes. All prior source/test/auditor versions, failures,
claims and actual v3 proof/31-of40 partial failure remain intact. Existing runtime,
client, scientific/source graph and budget sources do not change. Index history
has exact before/after snapshots; live indexes are excluded from the seal.

Local Mac metadata timing establishes no Linux/CUDA/whole-unit timing, memory,
durability, replay or training result. Current budget remains
REJECT_CURRENT_ROUTE_FOR_NEW_SCOPE. No scope/package/grant/approval request/launch
command or remote/paid work follows. The next budget proposal needs a separate
source-level rationale; do not integrate or bind this candidate. No further
candidate is adopted. All original guards/caps/policies/study/financial holds
remain fixed. Mechanism closed; research/training FALSE, no training percentage
or time estimate.
