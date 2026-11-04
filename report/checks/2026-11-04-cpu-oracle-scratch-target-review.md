# Target scratch capture review — November 4, 2026

**Saved numerical evidence passes independent checks; the original full run
remains rejected, and the setup allocation ceiling was not met.** No control
attempt is admitted. Preserve the target's consumed claim, captured source,
stored statistics, failure and authorization unchanged.

## Verified copy and numerical evidence

The single attended target run saved four fresh workers and finished its run
phase in 55.798 seconds, within the approved 600 seconds. The original parent
returns exit 1: `Stage comparisons do not reaggregate.` Its session remains
`capture_complete: false`, and it produced no summary. All four worker captures
are complete; their own unchanged numerical gates report success. This does
not turn the rejected parent session into an accepted run.

Independent local review verifies all 435 regular streamed members, their bytes,
hashes and original modification times, plus all 422 members of the original
export against the corresponding copied raw files. The response is
1,350,983,210 bytes with SHA256
`f5c367212fd0e927f9e17e9033a070d1888168f2152e799f12118c610c594898`.
The original 639,739,327-byte export has SHA256
`acfa4bc04cebefc2035496fa312b36a3016697ce73ffd5043ae1147385e256fb`.
The source packet, all 61 scientific pins, 33 upstream bindings, authorization,
new inner plan/claim and all four worker grants verify. Prior claims are unchanged.

The actual runtime is the expected linux10603 / Python 3.11.17 / Torch
2.8.0+cu128 / NumPy 2.4.6, with one/four CPU threads as planned and CUDA
uninitialized before and after every worker. Saved bytes cover all 12 original
before/observed/after calls and 256 observed chunks. Without generating a new
Torch oracle, independent saved-array checks verify:

- 2,112 logical array signatures and 2,048 NumPy intermediate comparisons.
- 48 final comparisons against the unchanged fixed references.
- 32 exact-byte comparisons within workers and 1,584 across workers.

All checked values pass the original `rtol=1e-12 / atol=1e-14`; maximum independent
absolute error is 2.498001805406602e-16, and maximum tolerance ratio is
0.007506069824549864. Actual converted inputs, chunk arrays and final tensors
are byte-identical across the one/four-thread workers and both fresh replays.
This finite CPU sample does not resolve the earlier approximately 1e-10 failure
or establish host reliability.

## Why the reported statistics do not reaggregate

The stored stage-audit reports are identical between the two four-thread workers
and between the two one-thread workers. Across those groups, 354 scalar statistics
differ: 178 relative-L2 errors, 82 maximum absolute errors and 94 maximum tolerance
ratios. All decision flags, bounds and captured array bytes agree. The differences
are in recomputed comparison statistics, not a change in the captured Torch arrays.

The unchanged `same_report` compares nested stage-comparison structures for exact
equality. Its special handling for `relative_l2_error` applies only at the top
level; maximum-error/ratio statistics also require exact equality. It rejects
the two observed report groups despite their agreeing numerical decisions.
The unchanged original Mac reader also rejects this capture with the same
stage-reaggregation error. No tolerance, stored report or original reader was
changed to obtain acceptance.

This supports a context-sensitive audit-statistic explanation for the **current
aggregation refusal**. The parent's recalculated report was not saved, so its
exact failed comparison pair is unobserved. It does not identify a specific
NumPy operation or explain the earlier unsaved 1e-10 reference discrepancy.

## Setup allocation guard failure

The write/fsync/readback test persisted and verified exactly 2,684,354,560 bytes
(2.5 GiB), then removed only its successful new test file. Its receipt reports
**4,294,901,760 allocated bytes**, already about 4 GiB, above the declared
3,221,225,472-byte (3 GiB) setup ceiling even before other files are counted.

The setup implementation checked write length and persistence but did not
enforce the recorded allocated-byte value against that ceiling. This is an
implementation gap. The filesystem reports local XFS and `noquota`; allocation
above logical file length is consistent with preallocation, but no lower-level
allocation trace was collected. Do not describe the overall storage protocol as
passing or retroactively expand the approved budget. The test's successful
cleanup does not remove the recorded peak-allocation violation.

The capture supervisor separately reports 183 samples, a maximum sampled
allocated footprint of 1,802,240,000 bytes below its 2.25 GiB capture ceiling,
and about 180.6 GB available at the final sample. These are recorded samples,
not continuous allocation proof, and do not repair the setup failure.

## State and next work

The target resource exception is consumed. Do not rerun its helper, erase the
claim, overwrite captured sources or relabel the failed session. Linux10609
has no grant or attempt claim, and no acceptance record has been written for
its launcher. Its current operator shares the setup/report issues, so leave it
idle while a separately identified revision is prepared.

Next work is offline: design and test allocation accounting that includes actual
filesystem blocks, and a saved-evidence audit contract that distinguishes numeric
acceptance from context-dependent diagnostic statistics. Use these preserved
arrays and failure cases; retain original refusals and unchanged scientific
bounds. Any future university attempt needs a separately reviewed operator and
resource scope. No remote connection, new university calculation, paid work,
financial change, scratch deletion or control-host launch occurred during review.

The remote private scratch directory remains at
`/var/tmp/vjepa-cpu-oracle-20261104-linux10603-qlx7eq8t`; no deletion is authorized.
The verified Mac copy is under the target's `attended-4qh6v3t1/saved` directory.
Detailed checks and the executed audit program are in the ignored
`independent-target-review-20261104` folder. No new implementation test suite ran
during this data review; the earlier 60 operational tests remain historical.

All financial caps and holds remain unchanged: $200 project / $160 compute,
$40 other-cost reserve, the single $10 diagnostic envelope with $9.8116
conservatively encumbered / $0.1884 unreserved, historical $2 hold and $0.0792
gap retained, last reported wallet $6.56, reconciliation incomplete. No optimization
or research result is admitted by this review.
