# Graph-owned predictor EOT8: university handoff v1

The corrected [backward route](predictor_owned_backward_v1.md) is now wired into a
separate live worker and bounded university handoff. **51 distinct local checks
pass**, with zero failures, errors or skips. This prepares one additional
attended attempt on **linux10603**. No new authorization, attempt claim, SSH
connection, university work or paid compute has been created.

The active unarmed handoff is:
`runs/pretrained/diagnostic/predictor-owned-target-v1-20261106/linux10603-final`.
Its control SHA256 is
`90863e16153414e276219ddf6305660e9a1a9f515199137d2216749c68b88aa6`.
The earlier `linux10603` directory is a superseded preparation preview. Its
source binding is stale; never run that helper. The final client also binds the
exact local handoff directory, so copying a prepared helper cannot reset its
attempt claim under the same authorization.

## Scientific scope and ownership evidence

`models/predictor_eot8_owned_worker_v1.py` retains the frozen worker's original
training clip, checkpoint, teacher, shared FP64 projection/native FP32 backward,
eight attack-mask draws and both mask families. Only predictor attention uses
math SDPA in the candidate. Original encoder attention, tensor operations,
seeds, runtime/state checks and numerical bounds stay unchanged. The original
loss bounds are rtol 1e-5 / atol 2e-6; input-gradient bounds are rtol 2e-3 /
atol 2e-7. The existing study guard continues to reject this diagnostic candidate.

Four fresh workers run serially: original 0, predictor-math 0, original 1,
predictor-math 1. Each has five passes: full EOT8 before, first draw before,
first draw profiled, first draw after, full EOT8 after. Each actual rank-0 FP32
loss and FP32 input gradient is saved before its numerical checks. Full controls
must match the saved earlier full EOT8 reference. Candidates require a new
complete, passing original from the same host and replay; exact within-process
and fresh-process replay requirements remain.

The live worker uses the previously tested one-use token for the exact loss,
input, versions, root node and phase. Its root hook observes the actual autograd
GraphTask. Serial recomputation must belong to that graph; creator forwards
must be outside backward. No global autograd scheduling change or arbitrary
thread whitelist is introduced. Owned methods, backend contexts and hooks
must restore. Missing private API support refuses without a fallback.

A new stdlib-only receipt reader checks five complete scopes with removed
hooks, observed graph/thread/process identities, every declared forward draw,
family and block, and recomputation associations. It runs before accepting a
new reference and before the supervisor admits the next worker. It does not
prove a CUDA backend. The unchanged event reader separately requires actual
owned CUDA forward/backward evidence for the profiled first draw and both
families; it makes no claim about all encoder geometries or timing attribution.

Normal backward refusals retain the actual loss already saved, bounded call
rows, graph scopes and precise refusal predicates. The integration test injects
such a refusal and verifies that the scope fails, the hook is removed, methods
restore, no input gradient is invented and no worker is marked complete.
Timeouts or interrupted collection remain incomplete evidence, never passes.

## Exact upload and resource envelope

The destination is a new private `/var/tmp` directory on
`kdu@linux10603.dc.engr.scu.edu`. The upload consists of:

- The unchanged **17,192,095-byte / 99-member** packet: one original training
  clip, prior clean reference derivatives and frozen source.
- **16 overlay files / 207,662 bytes**, including the corrected route, receipt
  reader, live worker/operator, original capture/event readers, and bound plan
  metadata with the existing budget snapshot.
- A **640,444-byte** stdin bootstrap, below the 2 MiB ceiling, carrying source
  and control metadata. Authorization, if later approved, is a separate line.

The existing checkpoint and installed dependencies are verified in place.
There are no checkpoint-weight, dependency, private-key, password or held-out
confirmation/test-pixel uploads, and no new installations.

| Limit | Bound |
| --- | --- |
| Whole work, beginning before preflight | 40 minutes |
| Setup, included in work | 3 minutes |
| Full write/fsync/readback proof, before archive | 2.5 GiB, at most 60 seconds |
| Fresh serial scientific workers | Four, at most 8 minutes each |
| Same-SSH collection after work | At most 10 additional minutes / 3 GiB |
| Setup / capture allocated storage | 3 GiB / 2.25 GiB |
| Raw captures / Torch allocator | 1,792 MiB / 7 GiB |
| Call rows / graph scopes / threads per scope / refusals | 4,096 / 5 / 8 / 32 |

Lower tensor, trace, event, file-count, cache, inode and free-space bounds stay
unchanged. Category ceilings total 2,264,924,160 bytes, leaving 150,994,944 bytes
under the capture ceiling. Setup plus four worker limits plus final receipts
uses 2,220 of the 2,400 work seconds; actual continuing deadlines never reset.
The client has its own 50-minute transport bound, including authentication.

The expected GPU is the 8 GiB RTX 3070 with UUID
`GPU-e54fae90-feeb-8106-d8c0-1974fe48555f`, driver 580.178.04. A fresh status check
and live guards must pass. Other CUDA processes cause refusal and are left
alone. Sampling does not reserve exclusive use of the GPU or storage. Keep the
Mac awake and its attended Terminal open through calculation and collection.
There is no automatic retry, old-claim reset, standalone retrieval, cleanup,
parallel host attempt or paid compute in this scope.

## Validation and remaining limits

The 51 checks exercise authorization/domain rejection, exact source and public
host-key binding, unarmed refusal before any claim/SSH, copied-helper rejection,
cold 99-member imports without scientific dispatch, continuing deadlines,
serial stop-on-error, typed storage ceilings and same-connection collection
with byte hashes, modes and nanosecond modification times.

Four fresh local CPU executions of the new worker retain all five passes and
both families of all eight draws. All **40 actual FP32 loss/input tensors are
exactly equal to their corresponding saved v1 CPU returns**. Saved review
separately checks 24 within-process equalities, 20 original-bound candidate
comparisons, 20 fresh-process equalities, 20 complete graph scopes and the raw
CPU event ownership. The earlier 50-case run is retained as preparation history;
it is not added to the final 51 count. The prior 27 route checks remain unchanged.

Independent saved review rechecks all 4,910 predecessor pins and the consumed
03 claim, exact packet/overlay/bootstrap, final source bindings and CPU evidence
with model, backward, SDPA, CUDA, process creation and network creation blocked.
This is local arm64 / Python 3.11.16 / Torch 2.14.1 CPU evidence, not a qualification
of the target's PyTorch 2.8 CUDA engine, pretrained memory, complete derivatives,
backend behavior, performance or study cost.

The old failing target thread was never recorded. Historical numerical refusals,
predictor generation/fixed-loader gaps, adversarial/scoring, parameter/optimizer
recovery and broader research gates remain open. No research result or execution
backend is adopted by this preparation.

The prior linux10603 grant is consumed and its files remain unchanged. The old
linux10609 handoff remains unarmed and unselected. **One new exact scoped
university exception is required before arming this handoff.** After approval,
recheck the exact source/control bindings and fresh GPU status, record the
separate authorization, and provide the single attended Mac helper. Do not run
it beforehand.

Financial state is unchanged: $200 project / $160 compute / $40 noncompute
reserve, one $10 diagnostic allowance, conservative encumbrance $9.8116,
unreserved $0.1884, historical $2 hold and $0.0792 reconciliation gap. Last
reported wallet $6.56; account reconciliation remains incomplete. Known Runpod
Pods remain terminated. This proposed university attempt has no paid charge.
