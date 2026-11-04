# CPU reference recovery with separate scratch storage

Latest result (November 4): the [target capture review](../report/checks/2026-11-04-cpu-oracle-scratch-target-review.md)
verifies all saved numerical values, but the original full run rejects stage
report reaggregation and the write proof exceeds the planned physical setup
ceiling. The target claim is consumed. Preserve the captured operator and keep
the control helper unarmed while separate offline revisions are prepared.
Earlier preparation and approval statements below are historical.

Current follow-up (November 4): the [new attended scratch operator](cpu_oracle_scratch_operator.md)
passes 60 local operational checks after the user-supplied XFS survey. It remains
unarmed pending a separate target-only CPU resource exception. No university
allocation proof or reference calculation has started. The earlier proposal and
outcome sections below are preserved as historical context; consumed helpers
must not be rerun.

Prepared November 4, 2026. This is an **unarmed design proposal**, not a runnable
operator or a new resource authorization. Both October 30 CPU probe claims remain
consumed. The [verified cleanup](../report/checks/2026-10-30-duplicate-archive-cleanup-review.md)
restored small-write access, but it did not establish a personal quota allowance
or authorize another reference attempt. The earlier numerical discrepancy remains
unresolved.

## Proposed execution and preserved science

Run the target linux10603 first, copy its evidence to the Mac, and independently
review it before considering control linux10609. Only one host runs at a time.
Use fresh private directories on verified local disk scratch, outside the shared
home filesystem. Never reuse or reset a consumed launcher or claim.

The proposed per-host scope remains 600 seconds, including allocation check,
extraction and export; at most 90 seconds per fresh worker; zero automatic retries.
The original runtime's initial-clock freshness check must also remain satisfied.
The four workers retain `(threads, replay)` order `(4,0), (1,0), (4,1), (1,1)`.
Each saves the original unobserved-before, observed and unobserved-after calls
before numerical gates. The same immutable `[1,16,400,64]` input fixture, original
oracle/observer, 64 observed chunks and `rtol=1e-12 / atol=1e-14` remain unchanged.
A captured numerical refusal prevents the next worker; storage/technical failure
also stops the attempt. Every failed or partial artifact stays evidence.

The packet is reused byte for byte. All 61 scientific pins, 33 upstream bindings,
eight CPU-probe source files, prior claims and old failed captures stay frozen.
The new outer operator will need its own identifier, provenance and authorization.
No GPU, SDPA, pretrained model, optimizer, dependency install or paid resource is
part of this proposal.

## Account for the whole working set

Four full workers save 668,467,200 bytes of array payload before container headers,
metadata, source/fixture copies or export. The prior Mac rehearsal's uncompressed
evidence plus compressed archive is 1,331,522,866 bytes (about 1.33 GB), already
larger than either individual file measure. That sum still omits retained input
archives, additional operational files and filesystem allocation overhead.
The earlier per-export cap alone did not bound total disk use.

The [machine-readable proposal](../configs/cpu_oracle_storage_recovery_v1.json)
therefore budgets each simultaneous category:

| Capture-stage category | Proposed allowance |
| --- | ---: |
| Retained input archive | 64 MiB |
| Unpacked code and fixture | 64 MiB |
| Raw result files | 1 GiB |
| Compressed export or its partial file | 1 GiB + 8 MiB |
| Logs, metadata and operational receipts | 64 MiB |
| Space for an emergency failure receipt | 16 MiB |
| One in-flight array-file write | 32 MiB |

These sum to 2,407,530,496 bytes, below a proposed 2.25 GiB capture ceiling.
Setup has a separate 3 GiB ceiling because of its full-size write test. These
are **design requirements, not implemented or proven runtime bounds**. The
operator must enforce them, including the compressed export and partial files,
before any launch can be approved.

## Storage gates before a new attempt

First inspect `/var/tmp` and `/tmp` on both hosts with read-only `df`, `stat` and
resolved-path checks. Prefer a supported local disk filesystem, initially
ext4/XFS/Btrfs. The selected parent must have a different `st_dev` from home;
network storage and memory-backed tmpfs/ramfs are refused. A different device
alone does not prove the absence of a different quota. The exact selected path
and mount must be reviewed, with no fallback to shared home or a new directory.

The future operator must then, within its separately approved time allowance:

1. Require at least 4 GiB available space and 2,048 available inodes on scratch,
   plus 8 GiB free on the Mac before collecting the two hosts serially.
2. Create a new private mode-0700 scratch directory. Write a new 2.5 GiB test file
   using actual fresh incompressible bytes, flush/fsync it, read back its hash,
   and record allocated blocks. This stage is capped at 60 seconds. It must
   remove only its own new test file after success. No reference call starts
   unless all checks pass; an error preserves the new failure evidence.
3. Record that this write test is a point-in-time writability check, **not a
   reservation** against other users' later disk or quota consumption.
4. Keep all new packet/capture/export files on that exact scratch filesystem.
   Recheck its identity and available space, enforce the category and aggregate
   ceilings, and stop if available space drops below 1 GiB. Reserve bounded space
   for an error receipt. The design calls for polling at most every 0.25 seconds
   and bounded in-flight writes; monitoring alone is not a hard allocation guard.
5. Keep foreground SSH attended through collection. Copy the archive, manifest,
   status and logs to the Mac and verify every regular member before independent
   review. Scratch can disappear on reboot or site cleanup; do not leave the only
   copy there or promise that a later reconnection will recover it.

An allocation/space/identity failure stops work without a different directory,
smaller capture, relaxed threshold or automatic retry. Capturing to local scratch
still may fail; retained partial files are never a passing result. No existing
remote evidence, environment, source or claim is in a cleanup scope. Any later
removal would cover only the newly identified private scratch directory after
verified Mac backup, explicitly included in the new authorization.

## Remaining work and authorization

The storage survey is pending. No scratch parent has been selected, no allocation
test performed, and no new operator implemented or armed. After the survey, build
and test the complete operator for low space, quota/write errors, path changes,
timeouts, capture/export size violations, interrupted collection, prior claims
and unapproved invocation. Only then request a concrete new CPU resource
exception. The generic instruction to continue and the exact old archive-cleanup
approval do not consume or extend any previous CPU grant.

The resulting finite CPU-only sample cannot establish host reliability or explain
the prior discrepancy by itself. The old GPU worker initialized CUDA; this probe
does not. Observation and storage I/O can perturb execution. A clean sample does
not waive historical refusals, qualify GPU math attention, justify full-model cost
claims or admit a research result.

All known Runpod Pods remain terminated. Financial state is unchanged: the single
$10 diagnostic envelope remains inside $160 compute / $200 project caps, with
$40 reserved for other costs; $9.8116 conservatively encumbered, $0.1884 unreserved,
the historical $2 hold and $0.0792 gap retained. Last reported wallet is $6.56;
account reconciliation is incomplete. This proposal changes no financial hold.

See the [offline planning audit](../report/checks/2026-11-04-cpu-oracle-storage-plan.md).
