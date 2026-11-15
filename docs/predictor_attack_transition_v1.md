# From a clean EOT8 replay to an attack-loop integration

The next implementation is a separate, opt-in **per-iterate input-gradient
context for EOT8 PGD**. The completed clean numerical/replay comparison remains
closed. The model's parameter-update path, training PGD5/EOT1, routine EOT1,
execution-policy registry and existing study/evaluation code remain unchanged.

The [machine-readable local plan](../configs/predictor_attack_transition_v1.json)
records the source hashes, scope and remaining gates. It is an inert plan, not a
new model policy, target handoff, execution grant or training admission.

## Why one successful gradient is not the PGD interface

The verified `TargetRoute` owns one exact native loss/input token, one EOT8 loss
and one gradient. Its single-use predicates, per-objective call caps and native
ownership are part of the qualification. The frozen PGD loop instead calls
`torch.autograd.grad(losses.sum(), current)` directly on every iterate. Wrapping
the whole evaluator with a single diagnostic route would violate those contracts.

The evaluator also performs clean, final-iterate and independent-scoring calls
under `torch.no_grad()`. Those must retain original forward behavior outside the
custom gradient context. They are not gradient work and must not create a fake
native backward receipt. In particular, original no-grad eligibility should not
be replaced with the grad-enabled eligibility used by the custom Function.

The following are source-control-flow counts for positive epsilon. They are not
measurements, timing estimates or new executions. The four attack-bank no-grad
calls are evaluator clean attack loss, PGD clean baseline, and two final iterates.
Independent scoring uses separate banks and never chooses attack candidates.

| Scope | Input-gradient calls | Attack-bank no-grad calls | Total attack-bank forwards | Independent score forwards |
| --- | ---: | ---: | ---: | ---: |
| Proposed local PGD2 / two restarts | 4 | 4 | 8 | 2 |
| Primary PGD20 / two restarts | 40 | 4 | 44 | 2 |
| Stronger PGD50 / two restarts, two score audits | 100 | 4 | 104 | 6 |

A fresh bounded native/custom context is needed for each differentiation. Keep
all restart starts, mask banks, clean targets, projection, per-example best-
candidate selection and the original consider-before-backward ordering. Do not
patch `torch.autograd.grad`, widen the frozen one-objective guard, reuse its
consumed scope or pass an unrelated cotangent to obtain a receipt.

The public-math derivative remains an explicit different derivative. Elementwise
gradient agreement at a clean point does not imply the same `gradient.sign()`
at every adversarial iterate, identical PGD trajectories or equal attack strength.
Compare gradients at matched iterated inputs and record trajectory/selection
changes explicitly. Do not infer attack equivalence or choose a route by the
largest loss after observing results.

## Small next local prototype

Use a new diagnostic artifact identity and a tiny synthetic CPU model only:
`[1,3,4,16,16]` pixels, two encoder/two predictor blocks, eight draws and two mask
families. Reject CUDA and both university hosts before work. Use two steps and
two restarts at epsilon 4/255, with predetermined test step 1/255; that test value
is not a research calibration choice.

The bounded initial rollout permits four candidate differentiations and four
same-input original checks. Only after all gates pass may one fresh-process
candidate rollout add four differentiations for exact own-route replay: at most
12 new tiny objectives total. This is new iterated-input integration work; the
successful old clean cohort is not repeated.

Each gradient context must bind the real loss/input token, observe its native
root/checkpoints and actual nested math tasks, retain the existing tiny per-
objective caps, and restore hooks, private aliases, modes and flags before the
next iterate. Preserve the old loss/input numerical bounds. Verify the full
best-candidate logic and pixel bounds, parameter `.grad` absence, exception cleanup
and no silent fallback. Keep original forward calls outside the custom context
for all no-grad attack selection and independent scoring.

The first prototype adapts EOT8 primary/convergence evaluation only. Training's
one-draw attack and routine evaluation are different scopes, and optimizer backward
continues to use the existing parameter-gradient path. No training policy is
registered by this plan. Any initial failure stops the rollout before replay.

## Remaining admission gates

| Gate | Current evidence / required next result |
| --- | --- |
| One-clip clean EOT8 numerical and fresh replay | Passed: all 36 returns exact across candidate processes; 3,145,728 input entries within original bounds. |
| Repeated PGD gradient-context integration | Not implemented; the tiny CPU prototype above is next. |
| Iterated-input target values and ownership | A later separate reviewed target scope must validate the integrated path; clean-only receipts are not substitutes. |
| Backend and work-unit performance | No qualified timing for the integrated route. Measure complete matched original/candidate units and declare instrumentation/transfer/serialization overhead. |
| Research hardware and recovery | Use the selected research runtime and explicit new attack identity. The university 8 GiB GPU is not qualified for full AdamW training. |
| Calibration and stronger controls | Thirty primary evaluations on ten training clips for three step sizes; step remains unselected. Retain PGD50/EOT8 and independent scoring audits. |
| Full-study economics | A complete new-route estimate is absent; the historical RTX 3090 primary-only sample screen failed. Other units are not free or portable by assumption. |
| Training adoption | Requires the preceding evidence and a separate explicit adoption decision. |

The four diagnostic worker walls (original 103.607/100.100 s and candidate
108.265/109.668 s) include model/receipt/fingerprint/capture work. They are retained
as diagnostic metadata. No speedup ratio, PGD unit estimate or full-study cost is
derived from them. A future lean observation variant needs explicit validation;
guards must not simply be disabled to obtain an attractive time.

All $200 project / $160 compute / single $10 diagnostic caps, reservations and
pending reconciliation stay unchanged. No current cloud price is inferred from
historical quotes, and no paid session or financial-hold release is proposed now.
No SSH/GPU/status check is needed for this local transition.

## Separate local collection repair

`models/private_collection_v1.py` reconstructs an already saved same-SSH response
into a fresh private directory. It is not wired into any consumed client. It
reuses the exact frozen decoder code object, framing, file-count/byte caps,
member-hash/mode/nsmtime checks and refusals. A separately bound control and
response hash are required. The source stays open by descriptor through decoding;
a replaced pathname is not reopened and final source identity changes refuse.

The original read-only response may have mode 0644 inside its owned 0700 session;
it is not chmodded. New output directories are explicitly reconstructed as 0700
by a scoped process umask, restored on success and failure. File modes and
nanosecond timestamps retain their transported values. Future clients should use
this receiver after their sender has finished. No target directory-mode fidelity
is claimed: directory metadata was not in the original file-only stream.

All 242 files of the existing 599,851,959-byte response were reconstructed and
independently checked. The strict capsule reader accepts `reference/` directly,
without a second private copy. The raw download and original 0755 mirror
intermediates remain unchanged. No old marker or scientific array was regenerated.
Historical lowercase control/authorization files are copied only as evidence;
uppercase aliases on the local filesystem do not create a new grant or canonical
handoff, and the original claim remains consumed.

Twenty-six distinct local collection checks have passing final outcomes. The
first run passed 22 and preserved one failed input-mode assumption. The correction
accepts the original 0644 response only within a private parent; only the failed
case and two new boundaries ran next. One additional source-replacement check
passed. No passing earlier suite or scientific objective was rerun. Independent
standard-library review verifies all 3,237 prior sealed hashes, 902 old target
byte/mode/nsmtime descriptors, source snapshots, actual reconstruction and the
inert decision. Initial audit filename assertions and their correction are saved.
