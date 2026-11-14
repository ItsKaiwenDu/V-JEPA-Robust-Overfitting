# Small-upstream forward-preserving integration passes

The new instance-local integration preserves original predictions, loss, clean
targets and native L1 output cotangents bitwise in four actual small-upstream model
comparisons: two mask seeds, each on creator-thread and separate-thread backwards.
Each includes original-before/candidate/original-after controls with both mask
families and checkpointed two-block encoder/predictor. Input gradients pass the
original bounds with zero failed elements; largest error/tolerance ratio is
0.00392966345, below one. This covers one draw, not full eight-draw EOT8.

The unchanged native graph guard remains authoritative. Each evaluation has eight
primary and eight genuine recomputation rows; candidate math subgraphs add no
native call-thread evidence. All four predictor VJPs and their distinct nested
GraphTasks, native root identity and hook removal verify. Disabling checkpointing
still triggers the original missing-recomputation refusal; a foreign graph is
still rejected. Original methods, instance-private aliases, flags, model versions,
parameter-gradient absence and diagnostic identity restore.

Thirteen distinct checks are covered by twelve initial passes plus one corrected
targeted negative check. The initial negative-fixture failure accidentally combined
foreign graph and changed flags; the corrected injection isolates graph ownership.
Only the test fixture changes. Initial log/source/failure stay, production and
numerical gates are unchanged, and no successful model comparison or full suite
repeats. Counts are not summed.

Independent saved-only review verifies 120 FP32 arrays, reconstructs native L1
cotangents and original full-input comparisons, and checks all actual ownership
receipts. No model/autograd/SDPA/CUDA/process/network or new reference executes
in review. All 737 prior sealed files and 245 target descriptors stay unchanged;
new source/tests/data/review are sealed. Frozen prototype, native guard, study
and training code remain unchanged.

Next prepare a separately bounded local eight-draw/two-family scope with an
explicit 64-call requirement, leaving the frozen eight-call prototype bound and
native guard intact. No target handoff, manual SSH status, paid compute or training
is authorized by this result. CUDA/performance/optimizer and financial admission
remain pending. [Details](../../docs/predictor_forward_preserving_upstream_v1.md).
