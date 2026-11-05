# Prepared original attention pair, version 1

The separate operator is implemented and locally tested. Its linux10603 handoff
is **unarmed**: it has no `AUTHORIZATION.json`, no `ATTEMPT.json`, and no SSH or
university execution has started. Existing grants are consumed. A new explicit
GPU-execution resource exception must precede any upload or live invocation.
The [offline proposal](original_attention_pair_proposal.md) and all earlier
observers, sources, numerical results, claims and reports remain unchanged.

## Scientific scope and acceptance

One attended attempt uses only the fixed 400-token encoder-1 fixture on
linux10603's pinned RTX 3070. It runs two fresh processes serially:

| Order | Original worker | Reference |
| --- | --- | --- |
| 1 | efficient, replay 0 | None |
| 2 | math, replay 0 | Completed and passing efficient worker from this attempt |

The original worker and scientific functions are imported from the complete
unchanged packet. Source hashes and loaded function bytecode are bound before
observation. No scientific function, original gate, backend recognizer, original
parent or summarizer is patched. Any original refusal, capture error, contention,
source mismatch or resource failure stops the pair without a retry or next worker.

Each original worker calls its CPU FP64 oracle once and executes six attention
forward/VJP iterations if its gates pass: untimed, profiled, warmup, timed-1,
timed-2 and timed-3. The new observer saves the actual CPU return before its
original gate and all six actual attention returns before the caller's checks.
GPU returns stay FP32; CPU returns stay FP64. Source device, dtype, shape and
stride are recorded before copying. Nonfinite results are retained for the
original gate to reject. A failed write is a technical failure.

The CPU gate remains `rtol=1e-12, atol=1e-14`. Component and fresh-reference
bounds remain output `1e-4/2e-5`, dQ/dK/dV `2e-3/2e-7`. Original input,
transferred-input and `values.pt` snapshots stay intact. Saved returns are
checked against those snapshots, fixed canonical references and original replay
signatures. Original runtime/precision/flag restoration, finite/nonzero values,
exact within-process replay and actual forward/backward backend evidence are
still required. A task label cannot substitute for observed CUDA kernels.

Detailed library/thread metadata is collected after the whole original worker
completes or refuses. Hooks and the caller's remaining alarm are restored
without resetting elapsed time. The frozen original worker retains its own
completion/error record; new capture coverage and pair acceptance are separate.
The old schema's 1,200-second field is used solely to satisfy the original worker
validator. The original parent is never called, and the new outer 600-second
deadline governs the complete attempt.

## Resource exception to request

| Resource | Bound |
| --- | --- |
| Target | linux10603 only; UUID `GPU-e54fae90-feeb-8106-d8c0-1974fe48555f`, driver 580.178.04 |
| Environment | Existing Python 3.11.17 / PyTorch 2.8.0+cu128 / NumPy 2.4.6; no install or fallback |
| Additional attempts / retries | One / zero; all prior grants remain consumed |
| Total work | At most 600 seconds, including setup, proof and both workers |
| Original workers | At most two fresh processes, one at a time, at most 120 seconds each inside the continuing outer clock |
| Scientific calls | At most two CPU oracle calls and twelve GPU attention iterations |
| Scratch | Fresh private mode-0700 directory under local XFS `/var/tmp`, device 66307, separate from shared home |
| Initial free space / runtime floor | 8 GiB / 1 GiB; inode checks also required |
| Proof | Full 2.5 GiB write/fsync/readback, at most 60 seconds; setup freshness at most 110 seconds |
| Observed setup / capture ceiling | 3 GiB / 2.25 GiB, unchanged |
| Raw capture category | Explicit new limit 320 MiB, increased from the preceding 256 MiB single-return capture |
| Individual tensor file | 32 MiB; CPU-return NPZ at most 16 MiB and each GPU-return NPZ at most 8 MiB |
| Other categories | Code/input at most 128 MiB each; metadata at most 64 MiB; overlays at most 2 MiB; at most 1,024 files |
| Original Torch allocator cap | 4 GiB per worker, with no simultaneous scientific workers |
| Collection | Same attended SSH connection; at most 600 additional seconds and 3 GiB of existing files |
| Paid compute / financial changes | None |

The 320 MiB raw category follows the proposal's 240 MiB steady allowance,
32 MiB atomic-save allowance and 48 MiB margin. Math-worker PT extents remain
unmeasured on the target. Oversize files cause refusal rather than a larger cap.
The 2.5 GiB proof is performed only within the future approved attempt, never
replaced by the small local test. Polling and filesystem space are safeguards,
not an aggregate kernel quota, exclusive reservation or future-capacity proof.

Strict GPU checks run before, during and after workers; five-second observations
do not prove continuous exclusivity. Foreign CUDA processes are left alone.
Only launched owned children are stopped on failure. Unset CUDA visibility is
preserved through the CPU supervisor. Temporary/cache directories stay inside
the new scratch and are recorded after each worker; the CUDA cache request is
32 MiB and remains subject to observed metadata/capture limits.

## Exact payload and transport

The reviewed prospective payload is the unchanged **85,622,679-byte** synthetic
packet, containing 139 files / 90,901,652 logical bytes, plus **13 overlay files /
177,136 bytes** and bound control/authorization metadata. It contains all three
frozen fixtures because original verification requires them; only encoder-1 is
dispatched. No trained weights, research pixels, financial-ledger contents,
credentials or private SSH keys are included. Prior evidence is represented by
path/hash bindings, not its raw contents. The unarmed control binds 1,148 prior
files and 14 local operator source files. The stdin bootstrap is 477,574 bytes;
the new authorization's bytes depend on the future approval record.

The host public key is pinned, forwarding is disabled, and the password is
entered through the Mac's normal SSH prompt. One SSH connection handles work
and collection. The collector includes every regular saved file, including all
NPZ captures, authenticates sizes/hashes and restores exact modification times.
It preserves partial local downloads and never overwrites an existing extraction.
A later download-only helper requires the same consumed claim and a saved
scratch prelude; it can collect only a finished attempt and starts no model.

The new live helper lives under
`runs/pretrained/diagnostic/original-attention-pair-v1-20261104/linux10603/`.
Do not invoke it until the separate scoped approval is recorded. The actual
unarmed shell was exercised and refused before any SSH/session/claim.

## Local validation and limits

**163 checks pass with zero errors, failures or skips:** 84 new checks and 79
existing observer regression checks. Cold tiny CPU math replay-0/replay-1
processes each complete all six original phases, preserving all actual FP32
returns and fresh-reference comparison. Negative fixtures preserve the original
CPU and reference refusals before advancing. Capture-write errors, prior hooks,
changed functions, missing/tampered fresh baselines, old grants, resource changes,
oversize/aliased files, bad/truncated transport and consumed claims refuse.
The full unchanged packet also passes cold import with original dispatch blocked.
An isolated fake connection verifies the complete upload stream without SSH.

These local rehearsals use arm64 Python 3.11.16 / PyTorch 2.14.1 / NumPy 2.4.6,
not the university runtime. They never label CPU math as efficient CUDA. No new
CUDA kernel, full-size allocation proof or university pair has run. Independent
saved-file review rechecks original arrays, dtypes, gates, source hashes,
fresh-reference identity and modification times without evaluating an oracle
or SDPA. Its [preparation report](../report/checks/2026-11-04-original-attention-pair-v1-preparation.md)
is the current status record.

Instrumented copies/I/O appear inside the original profiler and affect timing,
cache and subsequent state. Retain every profiler row and timing observation;
**never use these timings for speed, study-cost or adoption claims**. Even a
successful target pair leaves the original replay-1 workers, full-model checks
and research admission pending. The historical approximately 1e-10 refusal
remains unresolved; no original rejection or missing historical array is waived.

The $200 project / $160 compute / single $10 diagnostic caps remain unchanged.
Conservative diagnostic encumbrance is $9.8116, unreserved $0.1884, with the old
$2 hold and $0.0792 gap retained. Last reported wallet is $6.56; reconciliation
is incomplete. All known Runpod Pods remain terminated.
