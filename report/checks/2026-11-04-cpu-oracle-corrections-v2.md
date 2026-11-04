# Offline v2 corrections verified — November 4, 2026

Two separately versioned components are implemented and verified locally:
physical setup-allocation accounting and a saved-array reader that preserves
context-dependent diagnostic statistics. **69 tests pass, with zero failures,
errors or skips.** The complete saved target capture and two fresh-process
reader verifications pass the original numerical bounds. Original v1 parent and
setup failures remain rejected; no new university job starts.

Implementation: [design and limits](../../docs/cpu_oracle_corrections_v2.md).
Evidence: [JSON review](2026-11-04-cpu-oracle-corrections-v2.json).
Frozen baseline: [original target review](2026-11-04-cpu-oracle-scratch-target-review.md).

## Allocation correction and its limits

The old guard let its successful 2.5 GiB write/fsync/readback proceed despite
recording 4,294,901,760 allocated bytes, above the 3 GiB setup ceiling. The new
library refuses that exact observed value. It requests a fixed extent and
counts actual allocated blocks for the complete setup root at allocation,
write, fsync, close and own-proof cleanup boundaries. Unsupported preallocation
refuses; no extending-write fallback exists. Its default free-space prerequisite
is 8 GiB, with the original 2.5 GiB request and 3 GiB ceiling unchanged.

Tests use a small real write/readback and an explicitly injected Mac allocator;
simulations cover physical excess at three boundaries, quota/full-disk errors,
unsupported allocation, corruption, partial writes, expiry and existing-file
preservation. No full-size v2 proof or live XFS allocation test has run. The
library is not an aggregate kernel quota or exclusive reservation, and polling
cannot prove that the ceiling was never exceeded between observations. The
future enclosing operator must supervise blocking calls and the full lifecycle.

## Saved-value reassessment

The offline reader reauthenticates the preserved response and **435 saved
members**, original packet/scientific dependencies and worker receipts. All
**2,048 stage comparisons and 48 fixed-final comparisons** pass unchanged
`rtol=1e-12 / atol=1e-14`. Freshly recomputed decisions, failed counts, shapes,
coverage and bounds agree with the producer. Numeric refusals, invalid/nonfinite
arrays, source/hash changes, missing data and inconsistent report labels still
reject. Two matching refusal labels never count as success.

The cross-group regression retains all **354** differences between the original
one-thread and four-thread diagnostic reports. Within the new local audit,
844/843/844/843 producer-stage statistic differences are retained for workers
0/1/2/3. These diagnostics are not numerical tolerance changes. The original
reader is rerun unchanged and still refuses with
`Stage comparisons do not reaggregate.` Its recalculated parent report was not
saved, so the exact failed pair remains unknown.

The new output verifies in two fresh local processes with requested thread
environments 1 and 4. Each repeats all saved-value checks and exits zero; local
decision and diagnostic replays agree. These environment requests do not prove
Mac Accelerate's actual thread count. The original producer and local audit
contexts remain separately recorded. This does not explain the historical
approximately 1e-10 failure or qualify GPU math attention or host reliability.

## Preserved provenance and resource state

The new driver verifies **90 frozen anchor files**, including 61 scientific
source bindings, eight original probe sources, 13 historical claim/plan records,
the current v1 operator sources, target/control plans, authorization and consumed
claim. Original copied members and the frozen target review also reverify.
No original source, packet, report, array or consumed claim is altered.
Linux10609's authorization, attempt and accepted-review bridge remain absent.

The raw local review, exact new-source copies, driver, test XML and CLI outputs
are saved under
`runs/pretrained/diagnostic/cpu-oracle-corrections-v2-20261104/`.
Reassessment summary SHA256:
`ea88a94a402a521b38ae51b828822c8dde08eb7f374d165b3632c271f0f30beb`.
Original response SHA256:
`f5c367212fd0e927f9e17e9033a070d1888168f2152e799f12118c610c594898`.
Raw tensors/logs remain outside Git; tracked reports contain compact metadata.

The test suite creates one tiny local CPU worker; the full saved capture review
starts no new original worker or oracle generation. There is no SSH connection,
new university allocation/capture, remote deletion, GPU or paid work. All known
Runpod Pods remain terminated. Conservative diagnostic encumbrance stays
$9.8116, leaving $0.1884 unreserved within the single $10 allowance. The $2 old
hold, $0.0792 gap, $6.56 last reported wallet and incomplete reconciliation stay
unchanged, as do the $160 compute/$200 project/$40 other-cost allocations.

## Next step

Integrate the corrected components into a separately versioned unarmed capture
operator, then review the full storage and runtime envelope before any new
resource exception or live qualification. Both university hosts can stay idle.
No old attempt is repeated and no failed gate is waived.
