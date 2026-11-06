# Owned backward wrapper passes local correction checks

A separately named route now binds one exact loss/input pair to an observed
autograd GraphTask. Creator forwards and serial checkpoint recomputation from
that graph are admitted; unrelated threads/graphs and overlapping calls refuse
before upstream attention begins. Precise bounded refusal metadata records
thread, graph, process, active state and predicate. Root hooks, private methods,
backend contexts and diagnostic policy restore across normal and error paths.
The original gradient operation, model objective and numerical bounds stay.

**27 distinct checks pass**, zero failures, errors or skips. The frozen route's
refusal is reproduced using a real CPU checkpoint backward from another Python
thread. Both corrected modes complete that case; saved records verify distinct
creator/runner IDs and the exact owned graph. Tests cover foreign/reentrant
graphs, reuse, overlapping attention, tensor/phase changes, premature exits,
error cleanup, evidence caps and inactive setup. The existing study guard and
owned event verifier remain in force.

Four fresh tiny CPU workers complete all eight draws, both mask families and
five passes. All **40 saved loss/input tensors exactly match the prior v1 CPU
captures**. Independent review verifies 24 within-process exact comparisons,
20 original-bound comparisons, 20 fresh-process exact replays, twenty completed
graph scopes and actual CPU profiler ownership. Sixteen additional unit-test
actual/reference tensors and two cross-thread cases verify. All 4,663 frozen
predecessor pins remain exact; 4,910 inputs are sealed. Earlier intermediate
logs/reviews are retained; they are not added to the 27-check total.

This is local arm64/Torch 2.14.1 CPU evidence. Target CUDA/thread behavior,
pretrained full derivatives, memory fit and performance remain unqualified.
The original target's failing thread ID was not captured; the CPU reproduction
supports the explanation without rewriting the failed evidence. No numerical
refusal, broader research gate or consumed claim is waived.

No remote operator, handoff, authorization or attempt is created. Both university
hosts may idle. Next prepare and independently review a new bounded unarmed
university handoff, retaining original scientific and resource limits, before
seeking another target attempt. Financial caps/holds remain unchanged.

[Implementation and evidence](../../docs/predictor_owned_backward_v1.md) ·
[Machine-readable review](2026-11-05-predictor-owned-backward-v1-offline-review.json).
