# Local eight-draw original-forward/public-math-VJP integration

The separate tiny CPU EOT8 integration passes the actual eight-draw native
objective. All nineteen new checks pass in the first and only suite run. The
existing native ownership guard and both earlier eight-call entries are unchanged.
This completes local EOT8 integration; target CUDA and training qualification remain.

## Explicit scope and frozen computation

`models/predictor_forward_preserving_eot8_cpu_v1.py` declares a separate local
identity, model marker, constructor and resource allowance. It requires exactly
eight draws, each containing the original complementary context/target families
8/24 and 24/8. Its scalar loss calls the unchanged `attacks.pgd.native_objective`:
two-family means, then the eight-draw mean, with no alternative objective or
incoming-gradient substitution. Only that exact loss/input pair can be differentiated.

The fixture uses pinned upstream architecture with locally seeded random weights,
not pretrained weights: two encoder and two predictor blocks, two heads, 32 tokens,
predictor head dimension 32 and output feature dimension 128. It retains the
shared-FP64/native execution policy and frozen clean target encoder. CPU FP32,
eval mode, small parameter/operand/storage bounds and local-host refusal remain.

The new allowance is explicit: 2 predictor blocks × 8 draws × 2 families ×
2 checkpoint passes = 64 forward calls. There are 32 initial calls requiring VJPs
and 32 actual nonreentrant checkpoint recomputations. The new state inherits the
frozen tensor validation, layout/alias/version checks, public math and native-root
observation; it initializes its own 64-call resource scope. The old eight-call
constructor and standalone attention entry are never widened or entered. The new
state refuses standalone attention/gradient calls. Its model entry enforces its
64-call cap and one objective/backward; no training policy is registered.

The actual autograd Function is reused unchanged from the earlier small integration.
Its forward calls the original native guarded SDPA; backward delegates verbatim
to the original prototype's first-order public math VJP on the saved Q/K/V with
the actual incoming cotangent. Internal math forward intermediates may differ;
this is a custom derivative, not a claim of original-kernel backward identity.
No loss smoothing, cotangent patch, private backend backward, fabricated state,
numerical-bound expansion or original-candidate adoption is involved.

Only each predictor instance's freshly copied private functional alias is composed.
The upstream global alias and existing native method identity remain unchanged.
The unchanged native EOT8 guard sees sixteen primary calls for every encoder and
predictor block. Its actual owned root authorizes backward; read-only bridge
observation adds neither an outer hook nor native call-thread evidence. Only real
upstream method/checkpoint executions append that evidence. Every math VJP retains
its own distinct observed inner GraphTask and three leaf hooks. Methods, aliases,
markers, flags and hooks restore on completion and failure.

## Actual comparisons and numerical gates

Four comparisons combine two fixed eight-draw banks (seeds 19–26 and 37–44) with
creator-thread and actual separate-thread backward execution. Each compares
original-before, candidate and original-after: twelve complete tiny EOT8 model
loss/input-gradient evaluations. These are local, same-process comparisons;
negative tests additionally exercise intentionally refused or incomplete scopes.
Neither old successful tests nor this new suite is repeated.

All sixteen predictions, selected clean targets, native L1 output cotangents and
scalar loss remain bitwise equal in every comparison. The original input gradients
before/after also replay bitwise. Candidate input gradients pass every original
rtol 0.002 / atol 2e-7 elementwise bound, with zero failed elements. Maximum absolute
difference is 3.4924596548080444e-10; maximum error/tolerance ratio is
0.0008573618110200237, below one. Model state and parameter/input/target versions
stay fixed; parameter and input `.grad` fields remain absent.

Each evaluation records 128 native attention calls: 64 primary plus 64 genuine
checkpoint calls, sixteen of each per encoder/predictor block. Every candidate
contains the declared 64 predictor calls and 32 completed math VJPs with distinct
inner graph IDs under one actual native EOT8 outer GraphTask. Every root and leaf
hook is observed and removed. The separate-thread cases observe a backward runner
different from the creator, with actual checkpoint thread evidence.

The nineteen checks cover those four comparisons; both university-host refusals;
five invalid draw/family/layout cases; unchanged native seven-draw coverage refusal;
unchanged native missing-checkpoint refusal; unchanged native foreign-GraphTask
refusal; exact loss/input identity; changed phase; closed standalone entry and the
old eight-call cap; primary-error restoration/single-use; and frozen-source binding.
Native negative receipts are preserved. In particular, all 32 math VJPs can finish
without checkpointing and the native guard still refuses empty recomputation
threads. Math admission does not fabricate that evidence.

## Independent saved review and scope limits

A separate NumPy-only reviewer verifies 624 saved FP32 arrays, hashes and complete
receipts, reconstructs all native L1 cotangents with the two-family/eight-draw
normalization, checks selected targets and loss arithmetic, and recomputes the
input-gradient bounds. Model/autograd/SDPA imports and process/network execution
are blocked during review; it builds no new model or FP64/VJP reference. The first
source snapshot, suite log/XML, returns and negative receipts are retained.

All 882 prior sealed inputs and 245 actual target byte/mode/nanosecond-timestamp
descriptors remain unchanged. Local runtime is arm64 Python 3.11.16, Torch 2.14.1,
NumPy 2.4.6. No Torch 2.8/CUDA, pretrained geometry/memory, optimizer, timing,
study-cost or research qualification follows from this CPU result. The previously
rejected candidate's 3,724 entries remain rejected. Financial caps, reservations
and pending reconciliation stay unchanged. No remote operator, authorization,
claim, upload, installation, cleanup, held-out pixels or paid work is created.

## Next stage

Prepare a concrete unarmed target handoff for the original-forward/public-math-VJP
route. It needs a fresh native original control and candidate at actual model/EOT8
scope, exact original predictions/native cotangents and the original full-input
bound. Any initial gate failure stops the candidate; fresh-process replay is
conditional on both initial gates passing. Ownership, backend, memory, time and
collection bounds need their own target validation and reviewed scope. Full
training still requires numerical, attack, optimizer, recovery, hardware and
budget admission. This local result authorizes no university execution or old-grant
reuse. Reuse established host/GPU information; no repeated manual check is needed
for the offline handoff preparation.
