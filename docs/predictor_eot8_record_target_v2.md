# Predictor EOT8 with external records: university handoff v2

The [offline record correction](predictor_eot8_record_v2.md) is now integrated
into a separate live worker, supervisor, client and source-bound entry. The
handoff is **prepared and unarmed**. No new authorization, attempt, SSH/upload,
proof, university calculation, cleanup or paid compute has started.

The active handoff is
`runs/pretrained/diagnostic/predictor-record-target-v2-20261106/linux10603-final`.
Its control SHA256 is
`0895a8064abb4e745067350c4617e29dca9588858188c26603b05cd07e6d6bd9`.
The previously consumed helpers and their results remain frozen. Never rerun
one of them or treat the failed worker's saved values as a complete reference.

## What the new attempt would establish

The scientific calculation is the original one-clip clean EOT8 diagnostic.
Encoder/teacher, shared FP64 projection/native FP32 backward, checkpoint,
training pixels, masks/seeds, all eight draws and both mask families stay.
Only predictor attention selects math SDPA in the candidate. The original loss
bounds remain rtol 1e-5 / atol 2e-6; input-gradient bounds remain rtol 2e-3 /
atol 2e-7. Every actual scalar FP32 loss and FP32 input gradient is saved before
numerical gates. No extra oracle, warmup, optimizer, PGD or scoring work is added.

Four fresh workers run serially on linux10603: original 0, predictor-math 0,
original 1, predictor-math 1. Each runs full EOT8 before, first draw before,
first draw profiled, first draw after and full EOT8 after. Full controls must
pass against the saved prior full EOT8 reference. Candidates require new complete,
passing originals from the same host and replay; within-process and fresh-process
replay stay exact. The existing study guard still rejects this diagnostic
candidate. Historical numerical and broader research gates remain open.

The frozen graph-owned route, typed return writer, event reader and v2 record
format are reused unchanged. Call rows and graph scopes live in bounded files,
with a small worker header and final completion marker. Partial files, failure
markers, legacy records, wrong controls and missing final markers refuse.
The new worker revision must match before a reference can enter model setup.

Before the supervisor accepts a worker, and again before using earlier workers
for a later task, it verifies the completion marker, file hashes/ownership,
actual graph scopes, all ten typed capture descriptors and the saved raw
profiler events with the unchanged backend reader. Its receipt binds the header,
completion marker, calls, scopes and trace hashes. The child entry rechecks those
same earlier receipts within its continuing deadline before scientific imports.
A CPU-only analogue cannot pass the live supervisor's domain gate.

CUDA backend proof is limited to the profiled first draw and both families;
it does not cover every encoder geometry or qualify performance/cost.
No missing record from the earlier target attempt is reconstructed or adopted.

## Exact proposed upload

The destination is a fresh private `/var/tmp` directory on
`kdu@linux10603.dc.engr.scu.edu`. The payload consists of:

- The unchanged **17,192,095-byte / 99-member** packet: one original training
  clip, saved clean derivatives and frozen source.
- **17 overlay files / 225,423 bytes**, including the frozen corrected route,
  record writer/reader, new live worker/operator/entry, capture/event readers
  and bound metadata containing the existing financial snapshot.
- A **707,797-byte** stdin bootstrap, below the 2 MiB ceiling. An explicit
  authorization would be supplied separately after approval.

Checkpoint weights and dependencies are verified in place. No private SSH key,
password, held-out confirmation/test pixels, checkpoint weights, dependency
uploads or new installations are included.

The client binds the exact local handoff directory, source hashes, prior sealed
inputs, public host key and the previous consumed control/claim. Copies of a
helper cannot reset its claim under the same authorization. Both sides reject
old or expanded authorizations. The final helper refuses before a claim, folder
creation or SSH while unarmed. Standalone retrieval and automatic retries remain
disabled. The old pending linux10609 handoff stays unselected and unarmed.

## Bounds and operating conditions

| Limit | Bound |
| --- | --- |
| Whole work from before preflight | 40 minutes |
| Setup within that clock | 3 minutes |
| Full write/fsync/readback proof before archive storage | 2.5 GiB, at most 60 seconds |
| Serial workers, including their preflight checks | Four, at most 8 minutes each |
| Same-SSH collection after work | Up to 10 additional minutes / 3 GiB |
| Setup / capture allocated storage | 3 GiB / 2.25 GiB |
| Raw capture / Torch allocator | 1,792 MiB / 7 GiB |
| Metadata category, including all logs/plans | 64 MiB |
| Call file / row count / row bytes | 8 MiB / 4,096 / 2,048 |
| Scope file / worker header / final marker | 64 KiB / 256 KiB / 16 KiB |

All prior free-space, inode, event, tensor, trace, cache, process and file-count
limits remain. Each new record filename and its `.partial` counterpart has its
explicit lower file cap enforced by the supervisor inventory. Call/scope files
count as metadata; they do not borrow from the raw tensor/trace allowance.
The existing aggregate limits still apply to all other files and actual allocated
blocks. No storage or time allowance is increased or reset. Backend evidence
verification consumes the existing work/worker deadlines.

The expected host is linux10603, RTX 3070 UUID
`GPU-e54fae90-feeb-8106-d8c0-1974fe48555f`, driver 580.178.04. A fresh status
snapshot and the live hardware/storage/runtime guards must pass. Foreign CUDA
processes cause refusal and are left alone. Sampling does not reserve exclusive
use of the GPU or disk. Keep the Mac awake and its attended Terminal open through
work and collection. A failure stops the serial plan; no next worker or retry
is admitted. Once private scratch is created, collection follows work on that same
SSH connection after success or failure. A preflight refusal before scratch
creation stops without model work; retain and report the Terminal error.

## Verification and next action

**60 distinct local checks pass**, zero errors, failures or skips. They exercise:

- Four fresh local CPU executions of this live worker, all five passes/eight
  draws/two families, exact saved tensors and independently reread raw events.
- New revision/control/domain and typed-artifact checks, missing markers,
  altered evidence and altered raw traces or backend summaries.
- Exact source/payload/public-key binding, cold imports of the whole 99-member
  packet, unarmed and copied-helper refusal, continuing deadlines and serial
  stop-on-error behavior.
- Typed caps for every new record and partial filename, metadata-category
  accounting, and collection of a failed full-volume record with no retry.

All forty actual CPU returns match saved v1 CPU values exactly. New complete
references, 24 within-process comparisons, 20 original-bound comparisons,
20 fresh-process comparisons, twenty complete graph scopes and raw CPU event
ownership remain verified. These are local arm64 / Python 3.11.16 / Torch 2.14.1
analogues, not target CUDA/pretrained, memory, backend or performance qualification.
The earlier 33 writer tests remain unchanged and were not rerun or added to 60.

Independent saved review rechecks every one of the 6,759 predecessor pins,
consumed claims, exact packet members, source list, bootstrap and current CPU
artifacts with model/backward/SDPA/CUDA, process and network execution blocked.
Scaffold/YAML, helper shell syntax and diff checks pass.

**One new exact scoped university approval is required before arming this
additional attempt.** The previous one-attempt grant was consumed; no old claim
or clock can be reset. After approval and a fresh clear GPU check, independently
verify the armed state and give the single attended Mac helper.

Financial state stays: $200 project / $160 compute / $40 noncompute reserve,
one $10 diagnostic allowance, conservative encumbrance $9.8116, unreserved
$0.1884, prior $2 hold and $0.0792 reconciliation gap. Last wallet report $6.56;
reconciliation remains incomplete. Known Runpod Pods remain terminated. This
university proposal starts no paid compute and releases no financial holds.
