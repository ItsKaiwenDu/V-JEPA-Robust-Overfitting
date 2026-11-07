# Four-cell predictor attribution — November 6

All four fresh university captures and their saved evidence are independently
verified. **Attribution is complete; the native candidate remains rejected.**
This is one training clip, eight draws and two mask families on linux10603's
RTX 3070, with frozen targets, weights, pixels, seeds, runtime and original
numerical bounds. It is no study, kernel, optimizer or performance qualification.

## Observed four-cell comparison

Here `gij = Ji^T cj`: the first index identifies the original (`0`) or
predictor-math (`1`) propagation path, and the second identifies the actual
original or candidate predictor-output cotangent. Cross cells deliberately
replace that cotangent and are not native L1-loss gradient qualifications.

| Cell | Propagation path | Incoming predictor cotangent | Entries beyond original input bound versus g00 | Meaning |
| --- | --- | --- | ---: | --- |
| g00 | Original | Native original | 0 | Passing native values reproduced exactly |
| g11 | Predictor-math | Native candidate | 3,724 | Rejected native values reproduced exactly; rejection retained |
| g01 | Original | Replayed candidate | 3,739 | Controlled VJP only |
| g10 | Predictor-math | Replayed original | 0 | Controlled VJP only; not native acceptance |

Each input-gradient tensor contains 3,145,728 FP32 values. The original and
candidate diagonal loss/input values exactly reproduce their own saved anchors;
all four forward losses and clean targets are bitwise equal. The scalar loss
is 0.5617199540138245. The original loss and input gradient also exactly match
the historical reference. Candidate input rejection remains at rtol 0.002 /
atol 2e-7, with maximum absolute difference 1.4960765838623047e-5 and maximum
normalized error 8.641704438152846. Loss bounds remain rtol 1e-5 / atol 2e-6.

The cross cells preserve same-path predictions and actual incoming native
cotangents. Whole prediction/received/applied hashes match all sixteen saved
boundary pairs in every cell. At all eleven changed coordinates in nine pairs,
the exact observed source/destination uint32 values match the frozen sparse
metadata. Each cross cell actually substitutes those eleven values; diagonal
hooks substitute none. No native incoming tensor is modified in place.

## Both ordered decompositions, with interaction retained

Offline FP64 saved-array arithmetic verifies both identities over all elements:

`g11-g00 = (g10-g00) + (g11-g10)`

`g11-g00 = (g01-g00) + (g11-g01)`

Both reconstruction residuals are exactly zero. The total difference has L2
norm 2.1499490515081404e-4. The table retains both orderings; projection is
signed projection onto that total difference, not a unique causal percentage.
Term norms do not add, and cancellation can produce projections above 100%.

| Component | L2 norm / total-difference L2 | Signed projection / total difference |
| --- | ---: | ---: |
| Path at original cotangent: g10-g00 | 5.3825% | -2.3106% |
| Cotangent on candidate path: g11-g10 | 102.4260% | 102.3106% |
| Cotangent on original path: g01-g00 | 105.4864% | 105.2539% |
| Path at candidate cotangent: g11-g01 | 8.7519% | -5.2539% |
| Interaction: g11-g10-g01+g00 | 5.1101% | -2.9434% |

At the original 3,724 failure locations, cotangent terms have signed projections
100.6884% and 100.5350% in the two orders; path terms are -0.6884% and -0.5350%.
The interaction's L2 norm there is 0.7213% of the total difference at those
locations. Every original failure coordinate, four cell values, bound and
component value is retained in a verified 171,445-byte NPZ in the private raw
review. Full arrays remain in the original collected tensor files. The JSON
includes all metrics and exact file hashes; no worst-case subset replaces the
whole-array analysis.

This supports a conditional finding: in this observed sample, the eleven
external loss-cotangent changes dominate the measured difference in both
orders. Holding the original cotangent fixed leaves no original-bound
violations in the controlled comparison. Propagation-path differences and
interaction remain nonzero. Internal forward rounding still affects the
Jacobian, so these cells do not isolate a pure attention-backward arithmetic
bug. One capture per cross cell does not establish fresh-process cross replay.
No failed candidate or cross cell becomes a passing native reference.

## Evidence and approved limits

The 195,062,662-byte original response and all 186 collected files
(194,880,267 logical bytes) verify independently: raw and local hashes, exact
sizes, modes, nanosecond modification times, regular-file identity, member
coverage and complete stream extent. The complete 99-member input archive,
five-member value capsule, 21 exact overlays, sparse data, authorization,
control, public host-key binding and earlier consumed claims all match.
All 11,793 sealed preparation inputs reverify; the new immutable manifest
seals 11,998 files. Prior 68 focused checks were not rerun or added to any count.
The local review blocks model/autograd/SDPA/CUDA and process/network execution;
it uses saved CPU tensors and NumPy array arithmetic only.

Four distinct worker PIDs are 1519020, 1520984, 1522585 and 1524383. Each has
one complete native GraphTask, 1,152 actual call rows (576 forward / 576
backward), all 36 owners, sixteen boundaries, no ownership refusal and removed
hooks. Adapter/observer restoration, unchanged model state and absent parameter
gradients verify. All new records are refused by the frozen passing-native
loader. The candidate has its own hash-bound native-numerical-rejection marker.

Whole work is 137.390321 seconds inside the approved 2,400. The full 2.5 GiB
write/fsync/readback proof completed in 9.811373 seconds before either input
archive, with 1,926 allocation observations. Supervisor times are
31.078028 / 27.468737 / 29.327859 / 28.998643 seconds, all below 420.
Allocator peaks are 5,122,851,840 / 5,847,053,312 / 5,126,743,040 /
5,847,053,312 bytes, below 7 GiB. Peak supervised capture allocation is
195,522,560 bytes, below 2.25 GiB; all category and typed-file ceilings hold.
Seven/six/six/six GPU samples show only their own worker PID. This is sampled
evidence, not continuous exclusivity or reserved disk/GPU capacity. No resource,
time, storage or foreign-process refusal is recorded.

Collection used the one original attended SSH response; no download retry or
new model started. Local claim-to-last-response-write time is 204.997350 seconds
inside the 50-minute transport cap. There is no separately saved transport
exit-code receipt; complete stream evidence and supervisor statuses are
reported as observed, not reconstructed into a new exit receipt.

## Consumed attempt and next step

Claim SHA256:
`9066c0afb97cb9e92f02e047806124aab73fff36cc9cdb46b4c31e28d167e94c`.
The one approved attempt is consumed and frozen. Never rerun the helper,
reset authorization/claim/clocks, mutate captured source/results, reuse leftover
time, clean scratch or start standalone retrieval. linux10609 remains
unselected and unarmed. Both university SSH sessions may idle or close.

Next prepare an offline plan for the smallest same-internal-input and
same-cotangent comparison needed to distinguish propagation arithmetic from
internal forward rounding. Use both measured orders and their interaction;
preserve the rejected candidate and original bounds. No additional university
or paid work is authorized at this checkpoint. Any future request must follow
a concrete reviewed scope; do not ask for a speculative grant now.

Project $200 / compute $160 / single diagnostic $10 caps, $9.8116 conservative
encumbrance, $0.1884 unreserved diagnostic amount, historical $2 hold / $0.0792
unassigned gap and last reported $6.56 wallet remain unchanged. Reconciliation
is still pending; no paid compute occurred, and known Runpod Pods stay terminated.
