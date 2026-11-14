# Local original-forward/public-math-VJP prototype

This separate CPU-only prototype answers the next feasibility question locally:
can original attention forward values and native L1 cotangents be retained while
a first-order public-math VJP passes the original derivative bounds? The small
examples pass. Full-model routing and Torch 2.8/CUDA behavior remain unqualified.
The earlier proposal remains a frozen historical plan, not a target grant.

## Operator and ownership

`LocalForwardPreservingAttention` exposes explicit tiny attention and gradient
calls. It patches no model, functional alias, training policy, native guard or
upstream source. It refuses linux10603/linux10609, non-CPU/FP32 inputs, target
geometry, inference/autocast/no-grad execution and unsupported scopes. Inputs are
limited to batch one, at most two heads, 32 tokens/features and 128 KiB backing
storage per operand; at most eight forwards are allowed. Duplicate input objects
refuse; distinct views sharing storage are supported and tested.

A custom autograd Function returns the original public SDPA result. It restores
grad-enabled forward eligibility explicitly inside Function.forward, whose default
context would otherwise disable grad. The unused original autograd graph is
discarded. Backward builds detached operands retaining actual layout/aliasing and
computes the public math SDPA VJP using the actual incoming gradient. No external
loss cotangent is substituted, no loss is smoothed, and no private backend backward
API or fabricated internal saved state is used. Math intermediates may still
differ from the original backend; this is an explicit diagnostic custom derivative.

One scalar-loss/input scope observes its actual outer GraphTask at the native
root. Each math VJP observes one distinct inner GraphTask at its root and all three
leaf hooks, with thread identities retained and hooks removed in finally blocks.
Checkpoint recomputation forwards must belong to that outer task. Existing
read-only GraphTask identity inspection is used as in the frozen native guard;
absence or wrong identities refuse without a thread-only fallback. The existing
native guard remains byte-identical and is not yet integrated with this operator.
Loss/input version drift and post-consumption forwards refuse. Higher derivatives,
external unowned backward, reentrant checkpoint and overlapping/reused scopes
refuse. Original flags restore on success and injected failure.

## Actual CPU evidence

The first focused suite passes 25 checks. Review then adds loss/input version and
closed-scope guards and four edge-case checks; the 29-check suite passes. A final
test-fixture improvement retains exact input layout/storage for the independent
direct-math comparison and asserts that identity; all 29 checks pass again. No
run has failures/errors/skips; counts are not added. All source versions, logs and
actual arrays remain. The custom forward/backward arithmetic class is unchanged
across these stages. Earlier project suites are not rerun.

Four final cases combine direct/nonreentrant-checkpoint execution with exact-zero
or adjacent-FP32 native L1 residuals, using a tiny 1x2x6x4 attention example. Actual
original and math outputs differ in this local fixture. The original math-forward
path changes the selected native L1 cotangent; the new route preserves original
predictions, targets, scalar loss and cotangents bitwise. Native before/after
controls match exactly. All returned derivatives match the independent direct
public-math VJP bitwise at the exact original operand layouts. Tiny full input
gradients meet the original rtol 0.002 / atol 2e-7 bound.

Two further cases chain two attention calls, with and without checkpointing. Both
original forwards remain bitwise exact; both inner GraphTasks are distinct and
owned by one observed outer task. Their complete tiny input gradients pass the
same original bound. Failure cases cover domain/context changes, foreign graph
or thread, unsupported higher derivatives, mutation, exception restoration,
incomplete/reused scopes and calls after the gradient scope is consumed.

Across six final examples, 120 actual FP32 NPY arrays and their receipts are saved.
Independent review loads those arrays with pickles disabled. It evaluates four
small vectorized NumPy FP64 analytical references directly from the saved Q/K/V
and actual cotangent, saving 16 FP64 arrays; it runs no model, autograd, SDPA, CUDA,
process or network operation. All original component bounds pass: output rtol
1e-4 / atol 2e-5, derivative rtol 0.002 / atol 2e-7. Native cotangents are independently
reconstructed from the actual L1 residuals, and direct-math returns match bitwise.
The largest tiny full-input error/tolerance ratio is 0.000719138, below one.
The review's four analytical reference evaluations are explicit additional NumPy
checks, not reused or fabricated producer references.

All four initial native-L1 numerical datasets remain byte-identical through the
later guard/layout stages. Frozen training/study/router/native ownership sources,
332 prior sealed inputs and 245 actual target descriptors remain unchanged.
Actual local environment is recorded after tests as arm64 Python 3.11.16,
Torch 2.14.1 and NumPy 2.4.6. No CUDA kernel profiling or target performance is
claimed. Model/parameter/optimizer behavior, second derivatives, full-model memory
and study throughput are not qualified by these tiny explicit-call examples.

## Next gate

Prepare separate local small-upstream integration under the unchanged native
GraphTask ownership guard before a target handoff. Validate the instance-local
predictor route, checkpoint recomputation, original predictions/native L1
cotangents and original full-input bounds. Do not loosen a native ownership
refusal or register a training policy. Stop this path if forward/context/ownership
cannot be preserved with the public computation APIs.

No university operator, upload/helper/authorization/attempt, held-out pixels,
cleanup, installation or paid compute is created. Old candidate rejection at
3,724 target input entries and all financial/training gates remain. No manual
hostname/GPU check is needed for this local stage.
