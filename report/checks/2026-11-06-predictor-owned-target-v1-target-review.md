# Graph-owned EOT8 target result — November 6

**All five original-model phases saved their actual losses and input gradients,
but the attempt is incomplete because final result serialization failed.**
The ten actual FP32 return files are intact. Full EOT8 before/after losses and
input gradients exactly match the saved prior reference; four original-bound
checks have zero maximum absolute error. The six within-process comparisons
also match exactly across full EOT8 before/after and draw before/profiled/after.
The candidate and both fresh replay workers never started.

## Failure and limits of the evidence

The supervisor reports worker 0 exit 1 and stops without retry or a subsequent
worker. The only traceback ends at the worker's final `record()` call with
`ValueError: Operational JSON exceeds bound.` The frozen operational writer
has a 1 MiB per-JSON ceiling. The full live result places attention-call rows,
graph scopes and final status into that one JSON. The local tiny-worker checks
did not exercise a record with the full pretrained call volume. This is a
result-writing defect that should have been tested at the full metadata size.

The last successfully saved `worker.json` is 18,656 bytes and records all five
phase completions, ten captures, peak allocations and the earlier profiled
backend summary. Its `complete` and `passed` values remain false. The final
call rows, graph-scope records, refusal list, restoration receipt and final
completion state were not written. Do not infer or recreate those records from
CPU fixtures, source expectations or profiler IDs; do not edit this JSON into
a passing reference. The attempt is consumed.

The raw first-draw profile is retained: 216,865 events / 81,439,573 bytes,
144 CPU ownership labels spanning 36 encoder/predictor owners. Its recorded
live summary reports the original efficient CUDA route for both families, but
the unchanged event verifier requires the missing call rows for an independent
repeat. The review checks the raw event structure and label coverage without
fabricating a substitute receipt. Complete ownership/backend qualification and
candidate qualification remain pending.

The saved tensors establish that original CUDA forward/backward calculation
reached every declared phase and that the reviewed numerical values agree.
They do not complete the four-worker diagnostic, qualify the predictor-math
candidate, prove fresh-process replay or establish performance/cost/research
admission. Historical numerical and broader research gates remain open.

## Download, scope and resource verification

The complete same-SSH response is 214,061,148 bytes. All 143 files verify against
the stream header and every tar member: SHA256, size, mode and exact nanosecond
modification time match. All 99 packet members and 16 approved overlay sources
match the approved hashes. The source and consumed earlier claims remain exact;
5,574 preparation pins were rechecked and 5,734 inputs sealed. Saved review
blocks model, backward, SDPA, CUDA, process and network execution.

The 2.5 GiB write/fsync/readback proof passed in about 9.60 seconds, with 1,926
allocation records on local XFS separate from shared home. The whole work ended
in about 128.89 seconds, and the worker supervisor ran about 117.21 seconds.
The largest worker-reported allocator peak is 5,130,884,096 bytes, below the
7 GiB bound. The 24 GPU samples contain only the owned scientific PID 3225336;
this does not prove continuous exclusivity. No resource-capacity refusal,
foreign-process refusal, timeout or numerical failure is recorded.

Control SHA256:
`90863e16153414e276219ddf6305660e9a1a9f515199137d2216749c68b88aa6`.
Consumed attempt SHA256:
`b018c09881ed58dc166b890af1865b817d495813581ae7c9905692293e087528`.
The captured scratch directory is
`/var/tmp/vjepa-predictor-eot8-owned-v1-20261106-9l7l5zjo`.
All captured scientific/operator/client/test/config sources, claims, values and
logs are now frozen. The old pending linux10609 handoff remains unselected and
unarmed; it is not a fallback under this consumed authorization.

## Next task

Prepare a separately named bounded evidence writer that stores call rows and
scope metadata outside the small operational JSON and leaves a small,
source-bound file receipt in `worker.json`. Exercise full pretrained metadata
volume and the 4,096-row ceiling, including failure, atomic completion,
reference loading and collection paths. Preserve the numerical calculation and
all resource bounds; do not patch captured sources or relax scientific gates.
This offline correction comes before any new target-attempt proposal. Never
rerun the consumed helper or start the other workstation as a fallback.

Both university stations may idle. No new remote work, oracle, cleanup, tests
or paid compute occurred during review. No additional target grant was
requested. Financial caps, holds and the last reported $6.56 wallet remain
unchanged; known Runpod Pods remain terminated.
