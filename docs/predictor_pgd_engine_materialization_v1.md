# Recorded engine tags after PyTorch profiler materialization

The new local backend v4 reader accepts the saved candidate's four engine/node
associations and nine positive kernel-correlation witnesses. All 2,236 raw rows
remain intact. The saved native assessment is identical to its previous result
except for the policy label. No target worker is changed or rerun. The historical
candidate and session remain failed; saved-return accuracy is still unreviewed.

## The missing source step

The previous v3 policy checked generic engine-wrapper defaults at record creation.
It missed a later profiler step. In pinned PyTorch v2.8.0, profiler materialization
walks adjacent records in each thread-local operation buffer. When an engine
FUNCTION wrapper precedes a BACKWARD_FUNCTION node, it assigns the node's sequence
and forward-thread fields to that wrapper. Kineto's Python-facing accessors read
these materialized fields. A wrapper can therefore expose either its original
0/-1 defaults or the node's tags. The saved candidate's inner records have the
second signature.
[Profiler collation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/profiler/collection.cpp#L442),
[Kineto accessors](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/autograd/profiler_kineto.cpp#L1080).

Two full public-source snapshots are saved with exact v2.8.0 requests, Git blob
identities and content/snapshot SHA-256 hashes. The collection blob matches the
previous partial collection snapshot, whose range omitted this method. Existing
engine, node and default-record snapshots stay unchanged. This source supports a
version-specific metadata interpretation; it does not authenticate the installed
wheel's source commit or reconstruct native buffer order from the Python trace.

## Separate, conservative reader policy

The cold recipe `models/predictor_pgd_engine_materialization_v1.py` derives
`models/predictor_pgd_split_backend_v4.py` from frozen v3. Only the policy literal
and `_View.bind` change. A complete byte inverse reconstructs v3; every other AST
node is identical. Source pins, raw schema/projection, phase selection, typed
producer map, CPU interval ancestry, kernel correlations, file readers and bounds
remain intact. No scientific arithmetic or numerical threshold changes.

A supported node still needs its unique recorded forward-thread/sequence key,
one enclosing eligible engine with the matching name/process/backward thread,
a unique projected direct node, a bounded typed producer chain in its selected
scope, producer-before-engine order and the inherited backend witnesses. All
eligible enclosing wrappers are counted before their tag signature is selected;
an additional wrapper with different tags cannot disappear from ambiguity checks.

The wrapper must satisfy one of two recorded signatures:

- Generic defaults: exact recorded forward thread 0 and sequence -1. The previous
  native signature and its assessment fields remain unchanged.
- Materialized tags: both recorded wrapper fields exactly match the node. The
  next recorded synchronous frontend CPU row on that same process and thread
  must be this node, and the engine must start earlier. Same-thread intervening
  operations, partial/mismatched tags and reversed recorded order refuse.

Other-thread rows and separately classified profiler overhead do not enter that
same-thread frontend successor test. Every row remains in the raw trace and
projection. CPU parent fields continue to be interval-derived, not native parent
edges. Recorded order is a conservative source-shaped signature, not an observed
native operation-buffer order. The assessment explicitly leaves native storage
order, GraphTask ownership, installed-wheel source and execution authentication
unproven. The reader never fills, copies, normalizes or rewrites raw tags.

Bounds stay 20,000 records, 4 MiB trace, depth 128, 16 anchors and 128 KiB proof;
the frozen projection stays 8 MiB. The reader remains local-only. No target source
binding, upload package, live scope, grant, claim or sealed worker changes.

## Saved candidate result

| Engine / node indices | Node | Recorded wrapper tags | Node tags | Typed producer |
| --- | --- | --- | --- | ---: |
| 968 / 990 | `_AttentionBackward` | 0 / -1 | 1 / 0 | 272 |
| 1468 / 1469 | `BmmBackward0` | 2 / 14 | 2 / 14 | 1446 |
| 1528 / 1529 | `BmmBackward0` | 2 / 7 | 2 / 7 | 1357 |
| 1511 / 1512 | `SafeSoftmaxBackward0` | 2 / 9 | 2 / 9 | 1407 |

The outer association uses generic defaults; the three inner associations use
recorded adjacent-node tags. The unchanged backend checks retain nine positive
GPU/CPU correlation witnesses across efficient forward, math GEMM forward and
backward, and math softmax forward and backward. All 33 overhead rows and both
GPU annotations remain present and excluded from kernel qualification. Neither
kernel names, durations nor CPU containment alone establish ownership.

Candidate trace SHA-256 remains
`1e1a2687392e37941ba1ba6d306637f39c5474850602e3d777ff5f36e073d73f`.
The native trace retains all 1,535 records and its previous two kernel witnesses.
The generated v4 reader is 24,232 bytes, SHA-256
`0a5ef4a5dc99b55eb5df6a3af8d11f99dbb18352b4ffab55c310a3f5a0f62622`.

## Validation and preservation

All 29 distinct new metadata checks pass in 29 executions, with no failures or
repeated historical/scientific suites. They cover both saved traces, the retained
v3 refusal, source/blob semantics, exact derivation, two permitted interleavings,
two explicitly handwritten native cases and 20 refusing mutations. Negative cases
include mixed tags, wrong typed producers, same-thread gaps, ambiguous wrappers or
nodes, missing kernel links, foreign device/process and wrong runtime. Modified
and handwritten fixtures are explicitly labeled and carry no execution admission.
No PyTorch/NumPy or array/tensor value calculation runs.

The independent standard-library audit v4 verifies 25,884 prior hashes, 27,882
prior fixture entries, all 17 input pins, source materialization semantics, the
complete byte inverse, raw-to-projected identities/ancestry, all four associations,
positive correlations and preserved failure status. The old FIFO is lstat-only.
Auditor v1 omitted the existing linked-runtime category; v2 used the inner node
as the outer phase container; v3 expected an error message in a marker that stores
only its type. Their diagnostics are retained. These were audit-only corrections:
reader code and the 29 passing tests were not changed or rerun.

The eight saved component return files remain opaque values. Their earlier
header/hash checks and the completed eight-gradient / four-pair numerical cohort
are preserved. This correction establishes a saved metadata recipe match only;
it does not create missing candidate completion/resource/parent-success records,
compare component accuracy, or admit full EOT8, fresh replay, performance, cost
or training. All university grants remain consumed and financial limits/holds stay
unchanged. No remote connection, GPU query, capture, storage proof or paid work ran.

Next local work is to prepare a saved-return accuracy review against the original
bounds and provenance. Both traces and all eight returns are already local; this
parser correction needs no new candidate capture, native rerun or cohort repeat.
