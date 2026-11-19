# Narrow loading-overhead metadata correction

The saved 1,535-record native trace now passes overhead classification under a
separate local policy, retaining all 12 negative-device records. The complete
backend reader still refuses the real trace at its unchanged engine/node identity
gate. This is a local parser correction and a precise next refusal, with no worker,
kernel, full-EOT8, performance or training admission.

## Source-supported signatures

The new `models/predictor_sdpa_launch_evidence_v3.py` extends the frozen v2 name
predicate to exactly three observed names. Every other signature/schema/type/
clock/correlation/ownership/ancestry/backend predicate remains exact.

| Observed name | Saved count |
| --- | ---: |
| Activity Buffer Request | 1 |
| Runtime Triggered Module Loading | 6 |
| Lazy Function Loading | 5 |

All three require exact `2.8.0+cu128`, CPU, device index -1, resource 0, thread and
end-thread 1, forward-thread 0, sequence -1, scope 0, zero linked correlation,
positive correlation, false async/annotation flags and legacy elapsed -1. Existing
field types, integer clocks, duplicate checks and size/event caps still apply.
Other overhead names, unsupported runtimes and malformed signatures still refuse.
Positive-process CPU events with these names remain ordinary frontend CPU rows.
This is not a name-only exclusion or a universal overhead allowlist.

PyTorch v2.8.0 pins Kineto `5e7501833f1021ce6f618572d3baf657b6319658`. Its
`overheadKindString` maps the two loading enum values under CUDA runtime 12.4 or
later; the buffer-request mapping applies at 12.7 or later. Its overhead wrapper
reports device -1 and an overhead activity type. PyTorch can inherit a parent's
correlation ID when a Kineto record has none. Five primary-source excerpts are
saved with their exact URL/ref, Git blob metadata and local content hashes.
[PyTorch gitlink](https://github.com/pytorch/pytorch/tree/v2.8.0/third_party/kineto),
[Kineto name mapping](https://github.com/pytorch/kineto/blob/5e7501833f1021ce6f618572d3baf657b6319658/libkineto/src/cupti_strings.cpp#L67),
[overhead wrapper](https://github.com/pytorch/kineto/blob/5e7501833f1021ce6f618572d3baf657b6319658/libkineto/src/CuptiActivity.h#L157),
[correlation inheritance](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/profiler/collection.cpp#L591),
[device accessor](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/autograd/profiler_kineto.cpp#L985).

These sources support the narrow metadata interpretation. The Python trace does
not expose a native overhead-kind field, and this review does not authenticate
the installed wheel's source commit or arbitrary fabricated metadata.

## Preservation and local binding

The cold recipe `models/predictor_pgd_profiler_loading_v1.py` derives two new files
with complete byte inverses to the frozen originals. Projection v3 is 19,602 bytes,
SHA-256 `d8ca7092b346d3e1ccd9c43a99a4379eba21914a08ced173b692148015010675`.
The local `models/predictor_pgd_split_backend_v2.py` is 21,614 bytes, SHA-256
`850084f8c74261a518ebb7a552d765276d6812d85a8bc735a4aba075334a5d1f`.

Only the projection policy/name declaration and negative-device name membership
change. Its complete `project` and backend verification functions have identical
ASTs. The split reader changes its local projection import, policy/source pins and
new policy-binding checks. Its parser, `_View`, `_assess`, producer/engine/node/
kernel association, private trace reader and assessment entry remain exact ASTs.
Frozen v1/v2 validators, target source bundles, old claims and failures stay exact.
The old reader still rejects the original trace at raw record 295.

No raw row is filtered, rewritten or renumbered. The projection has 1,535 rows,
with negative-device indices `[11,295,296,297,1016,1021,1022,1023,1040,1043,1044,1045]`
retained as overhead. Such rows cannot own launch links, parent CPU operations,
define phases, become GPU annotations or provide a kernel. Real CUDA activity
still needs its exact recorded frontend link and positive interval. IDs are never
generated or inferred from names, durations or proximity.

Bounds stay 20,000 records, 8 MiB for projection, 4 MiB for split trace input and
128 KiB for metadata proof. Original numerical, execution, resource and clock gates
remain unchanged. This local binding is not added to any university cache, worker,
control, grant or handoff. There is no target operator or launch command.

## Next strict refusal

After loading classification, the real trace reaches the old `_View.bind` test:
`Exact engine/node identity and enclosing CPU interval required.`

| Saved row | Index | Thread | Forward thread | Sequence | Parent |
| --- | ---: | ---: | ---: | ---: | ---: |
| Engine wrapper | 965 | 2 | 0 | -1 | none |
| ScaledDotProductEfficientAttentionBackward0 | 987 | 2 | 1 | 0 | 965 |

Both have recorded process/device index 927099 and nested CPU intervals. The
wrapper's name matches the child node's engine label, but its forward-thread /
sequence fields differ from the child. The existing reader requires those tags
on both wrapper and node. That predicate is retained. No tags are filled in,
rewritten or copied to make the real trace pass, and no parent/kernels are lifted
from an excluded event. This is an observed incompatibility with the current
association contract, not proof that names/timing alone establish ownership.

The original native worker remains failed, without a complete marker. The candidate
component remains uncompleted. The already verified eight gradient objectives and
four passing same-input comparisons remain preserved and are not rerun. The new
local reader grants no captured backend metadata assessment, component success,
fresh replay, actual CUDA execution, performance or training admission.

## Verification and limits

58 distinct new metadata checks pass in 61 executions. The first run passed 52
and failed three expectations: one incorrectly expected the real trace to pass
all downstream gates, and two used that same assumption as the baseline for kernel
anti-spoof tests. The unchanged engine/node gate caused all three failures. The
second run used the exact three failed node IDs plus three new cases; all six
passed. No earlier passing, historical or scientific case was repeated. Production
files did not change after the first run; every test revision/log/fixture remains.

The real-trace test now requires the continued strict engine/node refusal and
saves the exact diagnostic. A separate, explicitly handwritten 11-row metadata
fixture supplies the positive/native-kernel anti-spoof baseline with both loading
names. Its fabricated engine/node tags match the old contract; it is not captured
GPU evidence. Relabeling required positive kernels as loading overhead still
refuses. Positive-process names remain frontend rows, and unknown names, malformed
signatures, unsupported runtimes, overhead-only owners, altered producers, wrong
roles, source/policy/cap drift and wrong private trace hashes refuse.

Tests prohibit process/signal/timer/network entry points and import no torch/numpy.
No array, tensor, pixel or checkpoint is decoded, and no model, profiler, derivative,
CUDA, reference builder, storage proof or university operation runs. The separate
stdlib saved/hash/AST audit passes on its first run, verifying 20,883 prior hashes
and 21,480 prior fixture entries, including all completed numerical artifacts. The
old FIFO is lstat-only, never opened/read/hashed. It independently checks the byte/
AST inverses, five upstream snapshots, 1,535-row preservation, overhead non-ownership,
synthetic witness boundaries and the real engine/node refusal.

Next locally examine the pinned autograd/profiler implementation and recorded
producer/node/wrapper fields before proposing a separate strict association rule.
Preserve all raw records and old gates/failures; names or timestamp containment
alone cannot authenticate ownership. Do not repeat the completed eight-gradient
cohort. No new university scope, grant/claim, GPU work, retry, retrieval, cleanup,
installation, paid compute or financial hold release is prepared or authorized.
The $200 project / $160 compute / single $10 diagnostic caps and pending holds /
billing reconciliation remain unchanged. The user commits/pushes; the agent does not.
