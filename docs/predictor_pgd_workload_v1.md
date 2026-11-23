# Local whole-PGD input-gradient adapter

The opt-in local adapter preserves the original PGD selection and candidate
evaluation flow while supplying one fresh owned input-gradient route per iterate.
Primary PGD20/R2 uses 40 contexts; convergence PGD50/R2 uses 100. The entry accepts
only small FP32 CPU controls, with at most 3,072 pixel elements and two million
model parameter elements. `EXECUTION_BINDING=None`; it has no worker, SSH, CUDA,
training-policy or research dispatch.

The [review](../report/checks/2026-11-23-pgd-workload-v1-local-review.md) records
24 distinct checks passing across 26 executions, including two failed-case reruns.
The [contract](../configs/predictor_pgd_workload_v1.json) and
[input pins](../configs/predictor_pgd_workload_inputs_v1.json) preserve the exact
frozen scientific dependencies and study configuration. The completed
[mechanism diagnostic](predictor_pgd_mechanism_completion_20261123.md) stays closed.

## Preserved attack and evaluation behavior

The entire `pgd_linf` body has an AST inverse after removing the local domain
guard and reversing the explicit gradient-context bridge. The entire
`evaluate_clip` body likewise has an inverse after reversing target preparation,
observed original objectives and the PGD context argument. Original sources are
unchanged. The retained FGSM/random branches are unreachable through `LocalUnit`,
whose settings admit only primary and convergence controls.

The adapter keeps epsilon 4/255, two random starts, projection, strict per-example
improvement and consider-before-backward. The original seed schema pairs primary
and convergence banks/starts. Eight attack draws, four independent scoring draws,
two original mask families and zero/two additional scoring banks are preserved.
Clean scoring, attack-clean evaluation, the PGD clean baseline, both restart-final
evaluations and independent final/audit scores execute through the original
objective with gradients disabled and no gradient route active. Score banks never
select a candidate. The tested step sizes .5/1/2 per 255 are diagnostic settings;
no research calibration choice was made.

Each context owns exact input/root/loss-view tokens. It requires candidate
selection before its single derivative and checks unchanged fixed pixels,
targets, masks, model bytes/versions, parameter requirements and empty parameter
gradients. Process/thread, original APIs, precision/SDPA flags, modes, private
methods and hooks are observed. Context exit removes the route before the next
iterate or original no-grad call. Exclusive fsynced starts precede model work;
failures consume their unit, stop later steps and retain failed records.

The evaluator restores caller RNG and individual module modes in its original
`finally` block. Per-iterate/scoring checks require execution-time eval modes;
the final unit check separately compares exact caller modes with the pre-unit
snapshot. Data corruption is reported and retained as a failure; the adapter does
not silently repair mutated parameter or mask values.

## Evidence and limits

Complete units run on a new four-element analytic CPU model. Their selected
pixels, attack losses, independent scores, audits and seeds exactly match the
unchanged evaluator. Two separate `-I -B` interpreter processes replay the complete
primary rollout with exact logical hashes, metrics and selection. Analytic route
receipts explicitly deny native ownership; replay dictionaries cannot authenticate
process execution. The checked subprocess harness supplies the local process
observation. This is not perturbed target-model rollout replay.

One additional, newly seeded tiny upstream CPU objective exercises the concrete
frozen `LocalEOT8Route` through the new context, with one native root, 32 initial
public-math VJPs, 64 predictor forward/checkpoint calls and 128 native call records.
It completes with zero native refusals, a removed root hook, finite nonzero input
gradient and restored routes. Weight/pixel/mask seeds are
2026112302 / 2026112303 / 2026112304; local Torch is 2.14.1. This objective is not
an accuracy comparison, a pretrained objective or a complete native workload.
The existing tiny PGD2/R2 cohort and all completed target/native/clean cohorts
were not called again. No new FP64 oracle or full-model trace was added.

The initial 22 checks produced 20 passes and two final-mode failures. Both failed
units had finished their gradient loops, but the final guard compared restored
caller modes with the eval-mode snapshot. Only that final observation was
corrected; an AST inverse verifies the complete source difference. Only those two
failed nodes were rerun. Their old failed records remain failed and their step
hashes match the later successful controls. Two new timing checks were then added;
no previously passing test or the successful frozen CPU objective was repeated.
Across this retained history, complete analytic algorithm loops account for 920
outer input differentiations, plus up to four negative analytic derivative attempts
and the one tiny frozen owned objective. These are actual local Torch calculations.

The original evaluator's inner timer stops before restoration. A separate analytic
timing harness therefore measures from the caller through returned units,
restoration and RNG observation. Its observation and fsynced journal overhead is
included, never subtracted. Fixture construction is outside the measurement;
matched terminal JSON publication is after its endpoint. One fixed-order pair per
method is a timing-boundary control, not a statistical performance estimate:

| Local analytic control | Original caller seconds | Adapter caller seconds |
| --- | ---: | ---: |
| Primary, diagnostic step .5/255 | 0.046610 | 0.074514 |
| Convergence, diagnostic step 2/255, two audit banks | 0.080811 | 0.176116 |

These values do not estimate native/GPU throughput, isolated instrumentation
overhead, memory fit or economics. No reduced-observation scientific route was
introduced to obtain a faster number.

## Next local step

Derive a separate target-sized whole-unit runtime from the frozen scientific
sources, retaining each objective's ownership and numerical gates. Integrate
whole-unit clocks, parent/RSS observations, storage/collection and replay identities
with explicit fake OS/compute ports locally before preparing a new attended scope.
The small CPU entry must remain separate; its domain and the frozen old entries
must not be widened into a live worker.

Real current-route primary/convergence units, fresh perturbed target rollout
replay, matched target timing with explicit observation overhead, chosen-hardware
memory/exact recovery, complete economics, training-only calibration and
data/protocol/adoption locks remain outstanding. No new remote scope, handoff,
approval request, SSH/query/upload, GPU work or paid work was performed here.
All prior grants are consumed. Training PGD5/EOT1, routine EOT1, optimizer backward,
policy registry and study YAML are unchanged; research/training admission remains
false. Project $200 / compute $160 / noncompute $40 / single $10 caps and financial
holds remain unchanged; October 29 balances are historical.

The raw stage is `runs/pretrained/diagnostic/pgd-workload-v1-20261123`.
The standalone stdlib audit passes on 30,552 prior hashes, 33,850 prior fixture
entries, 1,154 new fixture entries, 20 frozen pins and all source/receipt joins.
It only reads saved hashes, JSON and source text: no project/test callbacks,
tensor value decoding, recomputation, process/timer/signal/GPU/proof/network work.
The old FIFO is lstat-only. Local device 16777233 matches the prior reviewed
conditional observation; no historical device record or target mapping is changed.
Before/after index snapshots, all source versions, failed logs and records remain.
The current seal is `sealed-pgd-workload-v1.json`; user stages/commits/pushes.
