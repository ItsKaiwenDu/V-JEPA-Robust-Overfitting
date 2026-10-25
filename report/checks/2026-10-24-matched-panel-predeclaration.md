# Broader matched attack check: frozen predeclaration

October 24, 2026. The metadata-only specification is frozen and verified.
**The GPU panel runner and handoff are not implemented or executed.** No
university command or cloud deployment is required at this checkpoint.

Use ten clips: the first ten rows of the existing training-monitor manifest,
in their original order. Each is an actual member of the unchanged 100-clip,
20-class training set. The diagnostic covers ten classes; it does not reduce
the study's twenty classes. The previously inspected BreastStroke clip remains
in this declared prefix and is not described as unseen. Its historical result
is excluded from all new panel aggregates.

Use two distinct seeded attack banks per clip. Each bank has the same four
arms (20/50 steps crossed with four/eight nested draws). Both bank replicates
share clean targets, two random starts and the three independent four-draw
scoring banks for their clip. Fix step 2/255 and radius 4/255 for this diagnostic;
research step calibration still requires its declared 0.5/1/2 comparison.

The twenty case IDs, real video IDs, ordered clip metadata/pixel hashes and
all seed arrays are saved in the [specification](../../configs/matched_panel_diagnostic_v1.json).
There are 280 distinct mask draws after checking the actual upstream 31-bit
seed mapping, with no reuse of the historical one-clip draws/starts. CPU dry
sampling verifies 1,120 mask tensors and distinct realized mask draws without
pixel/weight deserialization or CUDA initialization. That audit uses local
Python3.11.16/PyTorch2.14.1; future university execution retains its declared
Python3.11.17/PyTorch2.8.0+cu128 runtime. The local hashes are a diagnostic
reference, not proof of bitwise equivalence or a new GPU acceptance gate.

## Work and review rules

The full collection has **20 cases, 80 matrix arms and 20 fresh A-only replays**
(100 PGD arm executions). Every case needs exact pairing, attack-bank selection,
bounds, model/target/RNG/mode preservation, observed native CUDA dispatch and
fresh A replay. Missing/rejected cases stop the queue and leave the scientific
collection incomplete. No automatic retries, backend fallback or wider tolerance.

Report B-A, D-C, C-A and D-B separately on the main and both audit banks.
Average the two attack-bank replicates within each clip, then give every clip
equal weight. Also retain all case/clip differences, medians/ranges, positive/
zero/negative counts and relative gaps. Treat clip as the analysis unit; the
paired bank replicates are not twenty independent video observations. Do not
pool or select scoring banks, choose an arm from audit loss, aggregate an
incomplete panel, or reuse the historical pilot in the new mean.

No new numerical attack-strength margin or automatic convergence/primary
choice is declared. Positive step gaps and bank disagreement are descriptive
review triggers. Technical integrity passing does not establish attack adequacy
or study admission. Future research choices remain training-only and must be
made before trajectories and held-out evaluation.

## Time and execution limits

Repeating the observed 40m57s one-case university session twenty times gives a
**13.65-hour scenario**, or **17.07 hours** with a 25% scheduling margin. Neither
is a prediction or a bound for the other clips/banks. Timings are instrumented
and remain outside A40 economics; the study budget has not been revised.

Require serial university-only cases, an 80-minute outer deadline per case,
60-minute matrix worker and 15-minute A-only replay, with the existing kill
graces and min(7 GiB, 90% VRAM) allocator policy. Summed per-case limits are
26h40m, or 26h45m including all outer kill graces. These do not bound calendar
time across pauses or unavailable hardware and are not implemented by a runner
yet. No paid compute is authorized by this plan.

Verify the first case before any remaining queue. Require a fresh host/GPU/
active-compute check and at least 12 GiB free disk before the panel. Retain the
same host, GPU UUID, driver and software across the collection. At most 256 MiB
uncompressed evidence per case gives a 5 GiB raw-evidence ceiling; the free-space
requirement also covers archives, staged inputs and atomic-write margin.
Only independently verified whole cases may be reused on resume, with exact
plan/source/runtime/data identities; no PGD-iteration resume or skipped failure.

## Verified preparation and preserved scope

All **67 focused checks** pass with zero skips, including **36 new** tests for
real training membership/order, seed independence/collisions, historical reuse,
changed limits/settings/admission flags, strict JSON identity, output immutability
and scheduling scenarios. Source/data metadata replay and the CPU mask dry run
pass. The earlier development draft is retained but was never executed; final
validation rejects even JSON Boolean/integer or integer/float substitutions.

All 198 original non-Markdown source pins and 49 prior bound source files remain
unchanged. The new planning identity binds 51 project files and 34 upstream files.
No canonical EvaluationSettings, PGD, study YAML, numerical threshold, held-out
pixel, trained weight or optimizer path changed. Full pytest was not repeated.
Existing $2 hold/$2.53 conservative unreserved allowance and the unassigned
$0.0792 billing gap are unchanged; no additional paid session has started.

Next is implementation and CPU testing of the explicit case runner, frozen
plan/receipt handling, first-case acceptance and complete-panel aggregation,
followed by a verified handoff. The old one-clip CLI cannot execute this new
plan. No masked data result or research calibration is claimed by predeclaration.
