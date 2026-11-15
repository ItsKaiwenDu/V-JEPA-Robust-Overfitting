# Split-profile metadata reader

The new reader verifies the two representative backend recipes from the
[split-profile design](predictor_pgd_split_profile_v1.md). It reads saved raw
metadata; it starts no model, gradient, profiler, process, GPU or SSH connection.
All checks in this stage use handwritten synthetic records. No old scientific
cohort or historical target trace is re-executed or promoted.

## Recipes and association

Both roles require one efficient forward in the declared forward phase. The
native role requires one efficient backward and no math recomputation. The
forward-preserving role requires no efficient backward and exactly one math
recomputation inside its producer-bound `_AttentionBackward` node. The original
same-family verifier remains unchanged and refuses the hybrid trace.

`models/predictor_pgd_split_backend_v1.py` uses the exact pinned raw projection
from the existing overhead-aware reader. It retains raw indices, CPU records,
linked runtime records, overhead signatures and GPU user annotations. Overhead
and annotations cannot provide kernel witnesses. Their existing source-supported
classification remains distinct from an authenticated native activity-kind
witness; this stage does not verify an installed binary's source commit.

The new association policy is explicit:

1. Require unique, ordered forward/backward CPU phase annotations on the same
   calling thread/process. Preserve integer nanosecond intervals.
2. Bind the native backward or custom outer backward to its actual forward
   producer using source thread **and** sequence number, a unique nested producer
   chain, and the direct matching engine/node parent. A CPU interval is an
   additional consistency check, never the identity by itself.
3. For a hybrid, require the math forward to descend from that actual custom
   backward. Bind each inner matrix/softmax backward separately to its producer
   inside the math forward. An outer SDPA sequence cannot replace those inner
   producer keys. Detached same-thread roots, unrelated autograd parents,
   repeated keys and ambiguous producer chains refuse.
4. Require positive CUDA kernels linked by recorded IDs to qualified frontend
   CPU operations. Every relevant matrix/softmax operator must have its own
   correlated kernel descendant. Kernel names establish family compatibility
   only after association. Names, durations, GPU intervals and globally unique
   individual GPU correlation IDs never establish launch identity.

The reader supports same-thread calls, separate outer/inner workers, and an inner
backward on the original caller thread. It leaves CPU parentage unchanged and
records separate producer anchors instead of reparenting foreign operations.
GPU launches may finish outside CPU phase intervals; their recorded links remain
the required evidence. Unassociated or additional attention/math operations,
unsupported backends, missing links and malformed records refuse without repair
or truncation.

The primary-source basis is explicit. PyTorch 2.8 records source thread and
sequence for backward nodes; a sequence alone is thread local.
[Node profiling and identity](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/autograd/function.h).
Custom forward profiling uses the Python type name and its sequence, while its
backward node reports the context type name.
[Custom function recording](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/autograd/python_function.cpp).
The collector transfers backward sequence/thread fields to the preceding engine
annotation. [Collection](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/profiler/collection.cpp).
These source facts support the policy; the expected `_Attention` runtime trace
has only synthetic coverage here, not a new target observation.

## Bounded saved-file use

`assess_bytes` accepts complete JSON bytes up to 4 MiB and 20,000 records, with
exact runtime `2.8.0+cu128`, the existing raw correlation schema and the pinned
split-design digest. Duplicate keys, nonfinite constants and trailing data refuse.
Ancestry has a 128-node ceiling, producer anchors a 16-entry ceiling and the
returned assessment a 128 KiB ceiling. The limits are explicit refusals, never
silent row or witness truncation. They do not preempt profiler collection or
materialization; a future component still needs its separate supervisor.

`read_private_trace` accepts only `events.json` in a canonical private 0700 parent,
with a regular 0600 single-link owned file and an expected SHA-256. It holds a
read-only descriptor, checks source/parent identities and modes plus file extent
and nanosecond mtime before/after reading, and refuses concurrent replacement,
linking, modification or digest drift. It reads no tensor/archive fallback.

The thin local command `scripts/check_predictor_pgd_split_backend_v1.py` writes
one new private assessment directory and 0600 JSON result. It preserves an
explicit refusal on malformed input and refuses an existing output directory.
It is a local metadata auditor, not a university launcher or retrieval helper.
The isolated CLI check blocks scientific imports, child process launches and
network calls inside the auditor.

External design/control/boundary/probe bindings are checked for exact role and
syntax, then carried into the assessment. Raw trace metadata cannot authenticate
those bindings. A matching assessment therefore keeps geometry, accuracy,
actual GraphTask/root/leaf ownership, CUDA execution, worker completion, full
EOT8 kernel coverage, fresh replay, performance/cost and training admission false.
Later component receipts must independently verify real source/layout lineage,
native execution/root/leaf proofs, resources and restoration before any result
can be admitted. A synthetic assessment can never close an old failed worker.

## Verification and next step

123 distinct new checks pass: 114 initial synthetic checks, then nine new source
race and owner-association regressions. No passing check is repeated. Coverage
includes both roles and multiple thread arrangements, outer/inner mismatches,
wrong source pools, duplicate producer/node/engine keys, unsupported or additional
backends, missing/foreign/ambiguous launch links, equal kernel names/durations,
reused individual GPU IDs, asynchronous launches, raw reordering, nested duplicate
operators, linked runtime threads, inherited overhead IDs, private-file races,
strict JSON/size limits, refusal persistence and the isolated metadata-only CLI.
Initial and later test sources and both JUnit/log results remain in
`runs/pretrained/diagnostic/predictor-pgd-split-backend-v1-20261115/`.

Next build the bounded component route using the frozen candidate arithmetic and
actual outer/inner root and three-leaf proof rules. Its lifecycle, layout/RNG/
flags/restoration, worker receipts and source package need cold/AST/simulated
integration checks before any target proposal. No new actual model, autograd,
SDPA, profiler, CPU gradient, GPU, proof, retrieval or paid work is authorized
by this local reader milestone. Existing failures, consumed claims, frozen
successful cohorts, numerical gates and financial caps/holds/reconciliation stay
unchanged. No university runner or authorization is ready.
