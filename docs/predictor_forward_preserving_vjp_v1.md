# Preserve original forward values before another full-model gradient trial

## Decision and evidence

The next question is whether a predictor-only public-math VJP can preserve the
original forward values and native L1 output cotangents while passing the original
full-EOT8 input-gradient bounds. A local-only prototype is the next implementation
step. Repeating the completed fixed-input attention cohort is unnecessary.
The [inert proposal](../configs/predictor_forward_preserving_vjp_v1.json) is no
execution grant, implemented candidate or research-policy change.

Existing verified results distinguish three observations:

| Observation | What it establishes |
| --- | --- |
| Native predictor-math route: 3,724 input-gradient violations | The old candidate is rejected, despite the equal scalar loss. |
| Eleven L1 output cotangent entries change in nine pairs | Small prediction rounding crosses the nonsmooth L1 boundary. |
| Four-cell attribution: g10 has zero violations; g01 has 3,739 | Holding the original external cotangent fixed removes the observed original-bound failures in that controlled comparison. |
| New frozen-Q/K/V/cotangent cohort: all four workers pass | At one predetermined attention call, both public paths meet original FP64 bounds and replay exactly. |

Both attribution orders and their nonzero interaction remain recorded. Signed
projections are not unique causal percentages. In particular, g10 keeps the
math path's internal forward states and explicitly replaces its external loss
cotangent. It is not the new forward-preserving route and cannot qualify it.
A passing component comparison also cannot qualify the whole model.

## Proposed mechanism and limits

Conceptually, return the original FP32 attention forward result `z` and compute
the first-order Q/K/V VJP through public math SDPA using actual incoming `bar_z`.
Apply this only to predictor attention. Preserve the encoder, projections, clean
targets, native masked L1 loss, weights, pixels, masks, seeds and dataset splits.
Do not patch/replay the external loss cotangent or smooth/change the objective.

This is an explicit diagnostic custom VJP. Math recomputation still constructs
its own internal forward intermediates; it does not prove identity with the
original kernel's backward arithmetic. Original forward dispatch can depend on
autograd/checkpoint context as well as flags, so exact values must be observed,
not assumed. Nested autograd and checkpoint ownership are implementation risks.
No private backward API, fabricated internal state or bypass of ownership guards
is allowed. The route is first-order only; higher derivatives are unqualified.

The hypothesis is that exact original predictions prevent the eleven observed
external L1 cotangent changes. Other accumulated derivative differences may still
fail the full-model gate. Preserving forward values is also no speed guarantee:
original forward plus math recomputation may erase the hoped-for saving.

## Smallest next local implementation

Create a separate local-only prototype, leaving every frozen source, helper,
training policy and prior result unchanged. It must refuse named university hosts,
CUDA and target-sized work before output creation. Use tiny direct and checkpointed
CPU examples to verify these properties:

1. Original forward values and observed execution context remain exact before,
   during and after the route. Layouts, aliasing, input versions and model state
   are preserved, with flags and hooks restored on success and exceptions.
2. The actual first-order public-math VJP meets an independent tiny FP64 analytical
   reference and reproduces a direct math VJP at identical inputs/cotangent.
3. Native L1 residuals near and at zero preserve original predictions, targets,
   scalar loss and actual output cotangents bitwise. A constant test cotangent
   alone would miss the failure that motivates this route.
4. Checkpoint recomputation and every introduced GraphTask, owner and thread are
   explicitly accounted for. Existing native guard failures stay failures.
5. Unsupported higher derivatives, contexts, mutations and partial failures
   refuse cleanly. No silent fallback, private API or parameter-gradient residue.

These are new operator checks, not a rerun of the accepted kernel comparison.
Local CPU behavior cannot qualify target CUDA compatibility, memory or speed.
If public APIs cannot preserve the required forward/context/ownership properties,
stop this path locally rather than launch another university diagnostic.

## Conditional full-model follow-up

Only after the local prototype passes should a separate concrete university
handoff be prepared and reviewed. Proposed initial work is one fresh original
baseline and one new-route native EOT8 loss/input-gradient pass. The baseline
must reproduce passing native anchors exactly. The new route must preserve full
original predictions, targets, native L1 cotangents and loss, and pass input
`abs(candidate-reference) <= 2e-7 + 0.002*abs(reference)` over every element.
Loss bounds remain rtol 1e-5 / atol 2e-6; component and bookkeeping bounds stay.
No controlled cross cell may stand in for a native new-route worker.

Any unexpected value, ownership, numerical or resource failure stops this route
before replay, profiling, attacks or optimizer work. Preserve the result and
return to the passing original policy for cost/design feasibility. Do not restart
the same comparison. Only if initial gates pass may a fresh original/new-route
pair establish exact own-route replay. Full execution/backend evidence, attacks,
optimizer/recovery, hardware measurements and complete-study budget admission
still follow before any separately reviewed training-policy decision.

## Work and user interaction at this checkpoint

This plan creates no prototype, target operator, helper, upload, authorization,
claim, storage proof, new scientific array or remote calculation. No prior grant,
clock or outcome changes. No tests or scientific cohorts are rerun for planning.
The original $200 project / $160 compute / single $10 diagnostic caps, existing
holds and pending billing reconciliation stay. Component timing is not used to
predict a training saving.

Reuse established host/GPU identities and already supplied status. A future
reviewed launcher must run live guards itself; request manual output only for an
access/identity change or unexpected failure. No SSH action is needed now.

[Completed component control](../report/checks/2026-11-14-predictor-sdpa-continuation-v6-target-review.md),
[four-cell attribution](../report/checks/2026-11-06-common-cotangent-v1-target-review.md),
[observed L1 boundary](../report/checks/2026-11-06-predictor-boundary-v1-target-review.md).
