# Projection reuse: CPU-only proof

The isolated `models/projection_reuse.py` prototype removes repeated FP64 patch
forward computations within one identical-input mask bank. It leaves every mask,
predictor call, loss reduction and first-order FP32 convolution backward separate.
It is **not integrated into evaluation/training and deliberately rejects CUDA**.
The original 61 scientific source/config/data pins, execution policies, masks,
seed rules, thresholds, targets, packets and historical reports remain unchanged.

`native_objective` evaluates every draw sequentially. `FixedTargetVJEPA.predict`
normalizes each draw's pixels and runs the encoder separately for each mask family.
The patch module receives the same normalized values before masks are applied.
The pinned recipe has two families: EOT8 therefore requests 16 identical context
projections per objective call. Independent scoring with four draws requests eight.
This identifies redundant work in the source; it does not measure its GPU-time share.

`BankProjectionReuse` is a plain, explicit context object. It computes the first
forward with the unchanged `_PrecisionProjection`, retaining its FP64 expression,
single FP32 cast and token layout. Every request receives its own cloned tokens
and custom-autograd node. Each node saves its own original input and uses the
unchanged first-order convolution VJP, so neither normalization gradients nor
draw gradients are combined earlier. Simply sharing one differentiable token
graph would change rounding/reduction order; that shortcut is not used here.

The bank validates bitwise-identical input (including signed zero), parameter
identity/version/value, geometry, mode, precision flags, grad mode and thread
count. It requires the declared request count and closes permanently on success
or error. Input/token/parameter snapshots are discarded on exit. A new bank is
required for every PGD objective invocation, scoring bank, candidate or optimizer
update. It is not a persistent feature cache and never reuses adversaries or targets.
The per-request checks and clones themselves cost memory and time; those costs
have not been measured on CUDA. Higher-order derivatives are unsupported.

Thirty-two new CPU checks verify exact token values/layout, raw-pixel and
parameter gradients under both backwards, unsafe-input/drift refusals, cleanup,
first-order limits and synthetic two-update AdamW/split recovery. Actual tiny
upstream transformers retain identical full input/parameter gradients and exact
PGD20/R2/EOT8 adversarial pixels, losses, RNG and weights. The attack has 43 objective
calls, each with 16 projection requests. The proof computes one forward per bank
and retains all independent backwards. This is a computation-count observation,
**not a 16× end-to-end speedup** or a pretrained/CUDA qualification.

The original candidate evaluator/runner is untouched. A test-only model adapter
is rejected by existing execution provenance. The prototype cannot be substituted
into old research artifacts under the existing policy/source identity. Any future
CUDA integration needs a separately identified diagnostic and its own bindings;
do not patch the old class in place or relax a guard to make it fit.

Final focused validation is **140 passing checks**: 32 new proof cases and 77
unchanged candidate checks together, plus 31 unchanged precision checks in a
separate process, with no final failures/errors/skips. The initial all-in-one run
retains 137 passes and one strict-runtime refusal: an older precision fixture's
parent-process settings did not match a fresh child's defaults. The new tests now
restore their own flags/RNG/threads. The original precision suite runs separately
to respect that legacy fixture boundary; no runtime guard or frozen fixture was
changed, and repository-wide fixture-order independence is not claimed.

The previous complete SE cost screen remains $243.71–249.62. If only ordinary
primary evaluation improves and every other measured cost and the unvalidated
eight-hour placeholder stay fixed, its primary work must improve by roughly
**1.46–1.53×** merely to reach that partial budget screen's cap. Under ideal
zero-overhead Amdahl arithmetic, 16-way projection reuse would require about
**33.6–36.7%** of current primary time to be the eligible repeated forward work;
eight-way reuse would require **36.0–39.3%**. The actual fraction and guard/copy
overhead are unknown. These are targets for profiling, not affordability claims.
The unchanged count/cost model credits no savings to other work in this scenario.
Unknown costs remain unknown and passing the partial screen would not admit a study.

Next prepare a separate source-bound correctness/profile diagnostic for the free
university GPU. It must compare original and reused forward values, input and
parameter derivatives under original tolerances, attack selection and independent
scoring, fresh-process replay, actual CUDA dispatch, memory and wall-clock timing.
Keep the target, all draws, steps, restarts, scoring seeds and loss identical.
Full pretrained optimizer/recovery qualification and a complete measured cost
projection are required before any integration or research trajectory. No SSH,
university job, paid benchmark, handoff upload or resource approval follows from
this CPU-only work.

The updated deleted-Pod charge is $1.1328 and the available wallet is $6.56.
Settling only its $1.30 reservation frees $0.1672 internally, leaving $0.1884
unreserved in the single $10 diagnostic envelope. The historical $2 hold and
unassigned gap remain; full-account reconciliation is still incomplete. The
$200 project/$160 compute caps remain unchanged. See the separate billing
settlement and machine-readable proof reports; historical ledgers remain intact.
