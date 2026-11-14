# Original-forward/public-math-VJP local feasibility passes

The new separate tiny CPU operator preserves original attention forward values
while computing an explicit first-order public-math VJP. **29 final focused checks
pass**, with zero failures/errors/skips. Initial 25-pass and later 29-pass stages
are retained, including added state guards and a direct-reference layout check;
counts are not accumulated. No historical project or university suite repeats.

Four direct/checkpointed native L1 examples include exact-zero and adjacent-FP32
residuals. Actual old math-forward rounding changes a selected cotangent in each;
the new operator preserves original predictions, native cotangents and loss
bitwise. Before/after originals match. VJP returns exactly match direct public
math at original layouts. Two chained-attention examples also preserve their full
forward outputs and pass the original input-gradient bound. Maximum observed
input error/tolerance ratio is 0.000719138, below the acceptance threshold one.

Outer and distinct nested GraphTasks/threads, all root and leaf hooks, checkpoint
recomputation, source input layouts/aliases/versions and flag restoration verify.
Unowned/foreign execution, saved-input or leaf-hook mutation, higher derivatives,
reentrant checkpoint, incomplete/reused/consumed scopes and primary exceptions
refuse with cleanup. No frozen router, native ownership guard, study or training
policy is changed. Read-only GraphTask identity inspection is retained; no private
backend backward computation or fabricated backend state is used.

Independent review verifies 120 saved FP32 arrays across six final cases. Four
explicit additional NumPy FP64 analytical evaluations from the saved tiny inputs
produce 16 reference arrays. Original component bounds all pass with zero failed
elements; native L1 cotangents reconstruct from saved residuals and direct math
returns match bitwise. Review performs no model/autograd/SDPA/CUDA/process/network
execution. All earlier source/log/array stages remain, with the original four
native-L1 datasets numerically byte-identical. Frozen sources, 332 prior inputs
and 245 actual target descriptors remain unchanged; all new evidence is sealed.

This is local arm64 Torch 2.14.1 evidence for explicit tiny calls, not a full-model
or CUDA qualification. Original candidate rejection, budget/financial holds and
training gates stay. No remote, paid, installation or cleanup work occurs.
Next integrate locally with the small upstream predictor and unchanged native
GraphTask guard before preparing any target handoff. No manual GPU status is
needed. [Operator details](../../docs/predictor_forward_preserving_cpu_v1.md).
