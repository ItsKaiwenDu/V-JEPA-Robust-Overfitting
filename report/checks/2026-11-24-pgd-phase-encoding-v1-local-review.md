# Phase-row encoding byte prototype: local review

**28 new stdlib checks and saved audit v1 PASS FIRST RUN.** The unbound candidate preserves saved
prehash bytes/newline, row SHA256 and final bytes while using one serialization
instead of two. Parsing and copying add work; no timing or net benefit is known.
The existing runtime/client graph and budget gate remain unchanged and closed.

See [details](../../docs/predictor_pgd_phase_encoding_v1.md),
[design](../../configs/predictor_pgd_phase_encoding_v1.json) and
[machine-readable review](2026-11-24-pgd-phase-encoding-v1-local-review.json).

| Review item | Result |
| --- | --- |
| Frozen inputs | Eight pins |
| Cold definitions | One emission before checks; no operational graph/package/grant |
| Baseline/candidate/codec sizes | 23801/28765/5162B |
| Source derivation | One full-module byte inverse over two edits |
| Guard/timer/cleanup preservation | Every other original method AST-exact; FileSink unchanged |
| New controls | 28 PASS/28 executions, first run; sources/tests unchanged |
| Synthetic typed rows/final vectors | 123/122, plus4097B refusal |
| Synthetic chain | 32 metadata rows; no worker/clock observations |
| Encoder calls on successful paths | Baseline two; candidate one |
| FileSink/ledger/worker/scientific lifecycles | Zero |
| Workload clock/process/signal/timer/network attempts | Zero under explicit blockers |
| Timing/net cost/actual budget fit | Unqualified |
| Preservation basis | 69894 prior hashes/76570 prior fixtures/1080 new entries |

The helper finds top-level members by JSON structure, using an iterative
container walk and standard scalar/string decoding. It retains nested value
bytes, correctly sorts decoded root keys and replaces an existing hash member
without adding a duplicate. No unchecked substring key replacement is used.
Its input provenance is the unchanged canonical encoder on private string-keyed
rows. It is not a general canonicalizer or an external-input gate, and no claim
covers concurrent mutation or stateful custom encoders/containers.

Private test namespaces execute only extracted metadata `enc`/`record` functions
and the new helpers. An explicit append spy records requested bytes or raises a
synthetic error. Literal fake owner/timestamp fields are not observations of
processes or time. Guard preservation comes from full source inverse/AST joins;
no FileSink, durable journal, fsync or full phase lifecycle is executed.

The corpus covers nested/escaped key-like text, Unicode/surrogates, numeric keys,
deep containers, finite numeric spelling, nonfinite/unsupported/circular
refusals, row/count boundaries and append-spy error/counter preservation. A long
old hash value can produce a prehash over4096B yet a valid shorter final row;
the candidate matches that original behavior. Twenty malformed/order inputs are
saved; source drift refuses before emission. No passing or closed scientific
cohort is repeated, and there is no stopwatch or timing result.

The independent stdlib saved audit verifies old preservation, the full inverse,
guard/helper ASTs and all saved synthetic typed-row byte/SHA joins. It uses no
project/test/scientific callbacks or subprocess/timer/signal/GPU/proof/network
ports. Synthetic metadata is decoded/reencoded; tensor/scientific-array values
are not decoded or recalculated. Ordinary local metadata FS/JSON/AST/SHA256 work
is real. No actual Torch/NumPy/model/gradient/worker/target work occurs.

Active56-source/48-file runtime/cache47/nine-supervisor/270-path graph, science,
hashing/ownership/shutdown/proof/monitor/scanner/Guard methods, original clocks
and all resource caps remain unchanged. The v5 client still refuses admission
before protocol clocks, attempt creation or connection. Metadata byte equivalence
does not qualify complete-unit fit or an operational route.

Raw: `runs/pretrained/diagnostic/pgd-phase-encoding-v1-20261124`.
Seal: `sealed-phase-encoding-v1.json`. Mutable indexes excluded, exact snapshots
retained. Starting clean user commit `8416b66`; user alone stages/commits/pushes.
Device16777229 uses the prior frozen observation; no probe/remap/waiver. All old
target66307 gates/four directory exceptions/FIFO LSTAT-only rules remain. Actual
v3 proof/31-of40 partial failure and all consumed claims stay unchanged.

Next is a separate bounded local stdlib cost comparison of the complete metadata
path, including parser/copies/bookkeeping, before any integration decision. A
dedicated benchmark stopwatch would be local metadata timing only; original
workload clocks/guards stay untouched. Do not rerun this suite or scientific
cohorts for timing. Local results could reject this idea but cannot qualify
target/CUDA/complete-unit performance or training progress.

Budget remains **REJECT_CURRENT_ROUTE_FOR_NEW_SCOPE**. No scope, approval request,
launch command or remote/paid work follows. Mechanism closed; all policies,
study/financial caps/holds unchanged and research/training FALSE. Full-unit/
replay/memory/recovery/economics/calibration/data/protocol gates remain. No
percentage or time-to-training claim.
