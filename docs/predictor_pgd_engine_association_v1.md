# Source-shaped engine/node metadata association

The saved native component trace now matches a separately versioned metadata
recipe. All 1,535 raw records remain unchanged. The completed eight-gradient PGD
numerical cohort is preserved without rerunning it. The historical component
worker still failed, and this local review does not complete it or authorize
training, performance measurements or another university connection.

## Why the old association refused

PyTorch v2.8.0's engine wraps each evaluated node with a generic `RECORD_FUNCTION`.
The generic record defaults are forward-thread 0 and sequence -1. The node's
`BACKWARD_FUNCTION` record explicitly sets its forward thread and sequence. Thus
requiring both records to carry the node's tags rejects the saved native trace.
The dispatcher uses `sequence_number::peek()` for eligible forward records, so
unrelated preparation calls can also share a sequence value.
[Engine wrapper](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/autograd/engine.cpp#L570),
[record defaults](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/record_function.h#L519),
[node instrumentation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/autograd/function.h#L164),
[dispatcher sequence assignment](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/core/dispatch/Dispatcher.cpp#L533).

Seven saved public primary-source snapshots cover these definitions, node
construction, record assignment and profiler collection. Their request/ref,
Git blob, content hash and local snapshot hash are recorded in the config.
These support a version-specific interpretation; they do not authenticate the
installed wheel's source commit or arbitrary fabricated metadata.

## Separate association policy

`models/predictor_pgd_engine_association_v1.py` is a cold derivation of the new
`models/predictor_pgd_split_backend_v3.py`. It changes only the policy literal and
`_View.bind`. Complete byte inversion reconstructs frozen backend v2. All other
code, including source binding, projection, phase selection, CPU ancestry, kernel
correlation, trace/private-file readers, role selection and proof limits, stays
exact. No target worker, cache, control, claim or historical decision changes.

The new method requires all of these together:

- A unique supported backward node with recorded positive forward thread and
  nonnegative sequence, and no reused key or node.
- One matching generic engine wrapper with exact recorded defaults 0/-1, on the
  same process and backward thread, enclosing the node and enclosed by its
  selected backward scope. The projected direct child must be that unique node.
  Ambiguous enclosing wrappers, multiple direct nodes and wrapper reuse refuse.
- A forward producer of the inherited supported node type, with the node's
  recorded forward thread and sequence. All eligible producers must be in the
  selected scope and form one bounded nested chain before the engine. Inherited
  scope and cross-thread exclusions remain.
- The unchanged positive CUDA-event correlation and backend-name checks. Neither
  overhead nor GPU annotation records qualify. Names, durations and CPU intervals
  alone do not establish ownership.

CPU parent fields are derived by the unchanged interval projection. They are not
native parent edges. The assessment explicitly records that native parent and
GraphTask ownership were not observed. No wrapper tags are copied from the node,
filled in, normalized or rewritten. Node-tagged or mixed-tag wrappers still refuse
under this new policy; the historical policy remains available unchanged.

The closed node-to-operator type map avoids treating every shared peek-counter
record as a producer. Such rows remain in the projection and are listed as
`same_counter_nonproducer_indices`. They cannot substitute for a missing typed
producer, node or kernel. Trace depth, event and proof bounds still apply.

## Saved result and limits

| Saved entity | Raw index | Thread | Forward thread | Sequence |
| --- | ---: | ---: | ---: | ---: |
| Generic engine wrapper | 965 | 2 | 0 | -1 |
| Efficient backward node | 987 | 2 | 1 | 0 |
| Supported efficient forward | 273 | 1 | 0 | 0 |

The node projects directly under engine 965. Both have process/device index
927099. Their CPU intervals fit the backward phase. Producer 273 is the unique
supported producer with the node's forward thread/sequence; 82 other shared-counter
rows remain as nonproducers, including the enclosing SDPA API call.

The unchanged kernel checks join frontend row 280 to CUDA row 298 for the efficient
forward and frontend row 999 to CUDA row 1048 for its backward. These are saved
correlation records, not new execution or worker completion. All 12 overhead rows
and both GPU annotations remain excluded from kernel/ownership evidence.

The raw trace remains 566,539 bytes, SHA-256
`3db71b394c0bf0f251040445bca28f3f9b7c9a2bdeb265bb373b4a302054965d`.
Backend v3 is 23,260 bytes, SHA-256
`2673d507c18b1b09a0b269d8740252b217a591b11def52bcf9f22cdca7c5c2c8`.
Bounds remain 20,000 events, 4 MiB trace, depth 128, 16 anchors and 128 KiB proof;
projection remains 8 MiB. Original numerical and execution limits remain unchanged.

57 distinct new local metadata checks pass across 60 executions. Three first-run
expectations were corrected: the enclosing SDPA API also shares the peek counter;
legacy elapsed 0 remains legal metadata but cannot replace a missing kernel link;
and the source uses an explicit integer cast for the node sequence. Only those
three failed nodes were rerun. Production files stayed identical. Passing,
historical and scientific cases were not repeated. Handwritten hybrid fixtures
cover distinct default-tag wrappers on shared and different threads; they remain
explicitly synthetic, with all execution/training admission flags false.

Next work is local integration design for the remaining component checks using
the saved inputs and completed cohort. No target scope, handoff or approval request
is created here. Candidate captured backend evidence and runtime integration remain
outstanding; full-EOT8, replay, performance, study-cost and training gates stay shut.
