# Live XFS publication mechanism reproduced; close-order candidate passes

The replacement completed on linux10609 with SSH exit 0, no stderr and a complete
4,121-byte saved response. Client elapsed time was 4.7573 seconds including login;
reported remote work/finalization took 0.09834 seconds. The one-use grant is now
consumed. No rerun or separate retrieval is needed or allowed under it.

For **both** inert files, actual allocation metadata changed at the owned fd close:

| Observation | Bytes |
| --- | ---: |
| Logical file length throughout | 12,583,040 |
| Allocated before close | 16,711,680 |
| Allocated after close | 12,587,008 |
| Allocation reduction at close | 4,124,672 |

Device/inode/owner/mode/link identity and modification time stayed unchanged for
each file. The read-back hashes match the fixed inert-byte hash in both cases:
`bfcd6fd60e8da4bae4eef2eac7655a718753b77226a10cf8d8a157b691fcb658`.
The target made no synthetic metadata substitutions.

The frozen writer pinned the allocation while its owned fd was still open. Its
subsequent full-row checks failed before and after read-back with the exact
original error: “Previously committed result metadata changed”, for
`results/numerical/objective-0/pixels.npy`. **Only allocated_bytes differed.**
The candidate closed the owned fd first, then pinned the settled metadata. Both
full-row checks and the byte hash passed. The allocation guard remains intact.

This establishes the close-order failure mechanism on the actual XFS host using
the unchanged frozen writer and reviewed candidate in a one-file byte inventory.
The prior model incident did not log its differing field; this matching native
mechanism is not a recovered old field log or a completed model/gradient replay.
No model, arrays, gradient, storage proof, GPU query or paid compute ran in the
probe. Scientific/training/performance/cost admission remains false.

Fresh independent stdlib saved-wire/hash/metadata review passes all **14,461 prior
hashes and 12,555 immutable fixture entries**, with the old FIFO lstat-only.
The four local attempt/client/wire/stderr records join the actual authorization,
control, nonce and program pins. The reviewer recomputes both metadata comparisons
and classifications directly from saved values, verifies the full wire hash and
bounds, and hashes only the known local constant to check the expected data hash.
It executes no project/test callbacks, processes, timers, network, GPU or
scientific code and repeats no test/cohort or remote work.

The reported footprint is 25,170,201 logical /25,182,208 allocated bytes, three
files and nine directories, within app caps. The third file is the remote status
snapshot. Remote data/status bytes were not separately downloaded; the complete
saved wire contains the pinned program's metadata and read-back-hash observations.
The reviewer does not claim a second independent remote read-back or byte-identical
copy of the server's earlier status snapshot. Free-space/cap observations are not
quota or reservations. The observed UTC timestamp is retained; its Pacific date
is November 16, 2026.

Next work is **local only**: integrate the exact full-module close-order repair
into fresh full-catalogue PGD runtime/manifest/contract/supervisor/reader bindings,
and separately fix/review the owned-child exit-monitor race. Keep all original
numerical/backend/resource/identity/time gates. Complete focused local checks and
review before proposing any new full-model scope. All prior university grants are
consumed; preserve every old result, source, authorization, claim and failure.
Financial caps, holds and reconciliation stay unchanged. The user commits/pushes.
