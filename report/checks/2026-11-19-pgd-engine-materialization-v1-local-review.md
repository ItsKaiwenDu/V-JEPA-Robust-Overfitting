# Saved candidate metadata matches the corrected local reader

Backend v4 now accepts the saved candidate's engine/node associations while
preserving all 2,236 raw rows. The native assessment keeps all 1,535 rows and is
identical to its previous result except for the policy label. The historical
candidate worker and overall attended session remain failed.

The missing source step was PyTorch v2.8.0 profiler materialization: adjacent
thread-local engine/node records can expose the same forward-thread and sequence
tags after collection. The old reader covered generic wrapper defaults only.
The separate new policy admits recorded defaults or exact node-matching tags with
recorded same-thread frontend adjacency. It keeps unique typed producers,
projected direct-node/scope checks and positive kernel correlations. Raw tags are
never copied or filled. Native buffer order, parent edges and GraphTask ownership
are not observed.
[Upstream materialization](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/profiler/collection.cpp#L442),
[exposed fields](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/autograd/profiler_kineto.cpp#L1080).

The saved candidate has one default-tag outer association and three exact-tag
inner associations. Their producer rows are 272, 1446, 1357 and 1407. Nine positive
recorded kernel witnesses cover efficient forward and inner math GEMM/softmax
forward/backward. All 33 overhead rows and both GPU annotations remain retained
and excluded from kernel qualification. The native result retains its previous
two witnesses. A metadata match does not complete the failed worker.

Only `_View.bind` and the policy literal differ from frozen backend v3. Complete
byte inversion and all-other-AST equality pass. Scientific arithmetic, source
bindings, phase/projection/kernel checks and numerical/resource bounds are
unchanged. The new reader is local-only and is not inserted into the consumed
attempt's source or control.

All **29 distinct new checks pass once / 29 executions**. Twenty mutations refuse;
permitted other-thread/overhead interleavings and explicitly handwritten native
cases remain labeled metadata fixtures. No historical or scientific suite reruns.
The independent standard-library audit **v4 passes**, preserving **25,884 prior
hashes / 27,882 prior fixture entries**, all 17 input pins and both source blob
identities. It independently checks raw projection identities/ancestry, associations,
correlations and failure status; the old FIFO is lstat-only.

Three audit-only assumptions were corrected and every version/log retained:
linked-runtime rows are a distinct existing category; the outer container is the
caller backward phase rather than the inner custom node; and the failure marker
stores the exception type rather than its message. No implementation or passing
test was changed or rerun for those corrections.

No SSH, GPU query, model/profiler/gradient, storage proof, array/tensor value decode,
paid work, new scope/handoff or approval request occurred. All grants remain
consumed. The original eight-gradient / four-pair cohort and eight saved component
returns stay preserved. Full EOT8, replay, performance, cost and training remain
gated; financial caps/holds/billing stay unchanged.

Next: prepare the local saved-return accuracy review using the original bounds
and provenance. No new candidate capture, native component rerun or completed
numerical-cohort repeat is needed for this parser repair. See
`docs/predictor_pgd_engine_materialization_v1.md`, both new materialization configs,
and the stage's `independent-saved-review-v4.json` for the evidence.
