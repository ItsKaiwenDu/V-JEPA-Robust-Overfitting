# Phase-row encoding: local byte prototype

A separate stdlib prototype reuses the first immutable canonical row encoding
when adding `row_sha256`. All28 new checks pass first run. They establish exact
bytes for the saved synthetic corpus, not faster execution or complete-unit fit.
The active runtime/client/source graph and unconditionally closed budget gate
remain unchanged.

See the [local review](../report/checks/2026-11-24-pgd-phase-encoding-v1-local-review.md)
and [prototype design](../configs/predictor_pgd_phase_encoding_v1.json).

## Baseline and candidate

The existing `PhaseLedger.record` serializes its private row, hashes those bytes,
adds the hash and serializes the row again. The source-only candidate keeps the
first serialization and uses structural offsets to insert or replace the hash
member in those immutable bytes. Both hash inputs include the final newline.

`models/predictor_pgd_phase_encoding_v1.py` contains two stdlib helpers. An
iterative container walker handles nested objects/arrays; the standard JSON
decoder handles individual strings and scalar tokens. Top-level keys are decoded
and compared in their original string order. Key-like text inside escaped or
nested values cannot select a top-level insertion point. There is no substring
key search or recursive container decoding. Values retain their original bytes.

This helper accepts only bytes produced by the unchanged local `enc(row)` for
a private string-keyed row. Producer provenance establishes canonical spelling.
It is not a general canonicalizer or an external-input admission API. Equivalence
is not claimed for concurrent mutation, stateful custom containers/encoders or
arbitrary external JSON. These assumptions need review before any integration.

The cold factory pins eight inputs and emits baseline/candidate/codec source
definitions and a derivation record. The baseline is23801B and candidate28765B;
the standalone codec is5162B. Two source edits have one full-module byte inverse:
replace the two-line encoding seam with three lines, then append the two helpers.
The original `enc`, FileSink, all other ledger methods, tokens, context/restore
bridges and saved-reader methods remain AST-exact. There is no operational source
graph, package, grant or launch API. The actual phase module is not modified.

The candidate still checks the4096B final row ceiling before append. An existing
`row_sha256` value is included in the prehash, then replaced exactly as before.
Even a pre-existing value making the prehash exceed4096B is not prematurely
rejected when the final row fits; the corpus covers that case explicitly.

## What ran

Twenty-eight new stdlib checks ran once, with zero errors or failures. Sources,
tests, runner and initial emission were frozen before checks; nothing changed
after them. No passing or closed scientific case was repeated.

The runner extracts only `enc`, `PhaseLedger.record` and the new byte helpers
into private metadata namespaces. Records use literal synthetic identities and
timestamps, with an append spy that captures requests or raises a declared
synthetic error. No PhaseLedger/FileSink object, journal, fsync, begin/end timer,
whole-unit lifecycle or worker is run. Guard/durability/timer preservation is a
source/AST result, not a new physical-enforcement test.

The suite uses a stdlib result collector without a stopwatch. Workload clock,
process, PIDFD, signal, timer, thread and network ports are explicitly blocked;
the recorded attempted-call list is empty. Ordinary artifact/source filesystem
operations, JSON parsing/encoding, AST work and SHA256 are real local work.
There is no Torch/NumPy/model/tensor/gradient/GPU/proof/target/network or paid work.

The saved corpus contains123 typed synthetic rows and122 final-byte vectors,
including a32-row synthetic hash chain. The remaining typed row is a4097B final
row refusal. Checks cover:

- Hash insertion at empty/first/middle/last positions, replacement, nested
  hash/role-like text, all ASCII controls, Unicode and lone surrogates.
- Nested empty containers, tuples, deep nesting, numeric-key dictionaries,
  signed zero, subnormals, large finite floats and integers.
- A deterministic64-vector metadata matrix, exact4096B acceptance,4097B refusal,
  row-count exhaustion and append-spy failure without committed counters.
- Identical nonfinite, unsupported-value, mixed-key and circular-value refusals;
  twenty saved malformed/truncated/duplicate/order cases; mutable-byte/digest
  refusal and pinned-source drift before emission.

Typed metadata preserves nested key types, tuple order and exact float spelling
for independent readback. The append spy sees identical final bytes; successful
record paths call the original encoder twice and the candidate encoder once.
That count is a mechanical observation, not a net-cost or speed result.

## Guards and limitations

Both full-prefix checks per append, the terminal receipt check, fsync/durable
ordering, anchored identities, file/row/count/depth caps, clocks/endpoints,
mandatory cleanup and primary-error code remain unchanged. The new parser,
ASCII conversion, temporary scalar/string objects, stack and final byte copies
add work and memory use. Any future comparison must include them. The candidate
may be slower or allocate more despite one fewer serialization.

The bound graph remains56 sources/936711B, runtime48 files/481518B, cache47,
nine supervisors and270 primary paths. Control remains
`d997bf9ed74c8ebf4504a5e6214a35e28365f85e9da3d7738a5b58f5edaa9f29`.
Objective120s/unit720s/setup180s/proof60s/soft2280s/hard2400s/final120s,
collection600s/connection3000s, RSS4GiB/allocator7GiB and all storage/process/
thread/sample/cadence caps remain fixed. No timing, peak memory or actual fit
is qualified. The candidate is unbound and is not adopted by the v5 client.

Raw stage: `runs/pretrained/diagnostic/pgd-phase-encoding-v1-20261124`.
The standalone stdlib audit passes first run on69894 prior hashes,76570 prior fixture entries,
1080 new entries, source pins/full inverse/ASTs, first-run history and every saved
typed row's prehash/newline/SHA256/final bytes. It decodes/reencodes synthetic
metadata only, with no project/test/scientific callbacks, tensor/scientific-array
decoding or recalculation, subprocess, clock/timer/signal, GPU/proof/network work.
The historical FIFO is LSTAT only, never opened. Current device16777229 uses the
frozen prior observation; no probe/remap/waiver, with old target66307 gates and
four narrow directory exceptions unchanged.

Starting clean user commit `8416b66`. All prior sources/tests/auditors/fixtures/
failures/results/statuses/consumed claims, including actual v3 proof/31-of40
partial failure, remain unchanged. Seal: `sealed-phase-encoding-v1.json`.
Live mutable indexes are excluded and retain exact before/after snapshots.
User alone stages, commits and pushes.

## Next separate local step

Assess cost before considering integration. A separate bounded, matched stdlib
metadata comparison may use a dedicated local benchmark stopwatch; it must not
invoke or reset any workload clock, timer, FileSink, worker or scientific route.
Compare the whole metadata encoding path on the same private synthetic rows,
including hashing, parsing, copying and bookkeeping. Keep the sources/corpus
frozen and treat order/allocation effects explicitly. Do not rerun this suite or
an old scientific cohort just to obtain timing.

That would measure local metadata cost only. It cannot establish Linux/CUDA
performance, phase-ledger durability, complete-unit timing, memory fit or training
progress. Integration still needs separate source/domain/guard review; actual
complete-unit budget admission stays `REJECT_CURRENT_ROUTE_FOR_NEW_SCOPE`.
No scope/package/grant/approval request/launch command, SSH/query/upload/retry/
resume/retrieval/remote cleanup/target/GPU/proof/paid work follows here.

Mechanism closed. Scientific policies, trainingPGD5/EOT1, routineEOT1, optimizer
backward, registry/study YAML and financial200/160/40/single10 caps/holds stay
fixed. Full units/replay/performance/overhead/memory/recovery/economics/calibration/
data/protocol/training remain gated. Historical financial snapshots are not
current quotes or balances. No training percentage or time-to-training estimate.
