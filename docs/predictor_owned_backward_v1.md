# Graph-owned predictor recomputation: offline correction

The [consumed university attempt](../report/checks/2026-11-05-predictor-eot8-10603-v1-target-review.md)
stopped at the original route's creator-thread guard during checkpoint backward.
A separate route now admits recomputation belonging to one specifically bound
backward graph. All captured v1 sources, data, results and claims stay frozen.
This is an offline correction; no target handoff or execution grant is created.

## Ownership and restoration

`models/predictor_eot8_owned_v1.py` keeps the frozen original RoPEAttention
forward bytecode, private per-instance functional alias, SDPA arguments,
geometry and actual-call predictor-only math context. It inherits the original
single-route registry and restoration checks. An outer owned method lock
checks admission before the upstream method or backend context opens and
rejects genuinely overlapping attention calls. Creator-thread forwards are
allowed only outside an owned backward scope and outside any GraphTask.

The creator prepares one token for the exact scalar FP32 loss, FP32 leaf input,
versions, root node and phase, after verifying every owned block's forward
coverage (all eight draws/two families for EOT8; both families for draw passes).
The token can run only once. It invokes the original `torch.autograd.grad(loss,
input)` with its original defaults. A temporary root-node prehook observes the
current real autograd GraphTask ID; it returns `None` and changes no gradient.
Only serial calls in that exact graph and phase are admitted during the bound
backward. No arbitrary thread whitelist or global autograd scheduling change
is used. Unknown/missing/repeated roots, foreign graphs, nested/reused tokens,
changed tensors, phases, methods, identities, flags or aliases cause refusal.

Each call records creator/current Python thread, PID, actual GraphTask ID,
backward-scope index and original tensor/backend metadata. Each scope records
its runner, root-hook thread, graph and observed call threads. Refusals retain
the specific predicate, active state, phase, current/expected graph IDs and
row/cap state before raising. One refusal poisons the route for further work.
Evidence is bounded: at most 4,096 call rows, five backward scopes, eight
observed threads per scope and 32 refusal records; invalid phase values are
not copied into records.

The hook is removed on success and failure; consumed tokens release tensor
references. Prepared but unused tokens are cancelled when the route exits.
An attempted exit during a running backward refuses while retaining owned
methods until that call unwinds. Private methods, diagnostic identity and
backend contexts restore on ordinary completion, exceptions and checkpoint
early-stop. The existing study guard still rejects the active candidate.

The mechanism uses the private `_current_graph_task_id` API which is present
in [PyTorch 2.8's graph implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/autograd/graph.py),
with the public loss-node prehook described there. The tagged
[autograd engine source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/autograd/engine.cpp#L89-L125)
describes the CUDA worker-thread distinction. A missing or unsupported API
refuses. These source checks do not establish target runtime behavior.

## CPU evidence

**27 distinct local checks pass**, zero errors, failures or skips. They include:

- A real upstream checkpoint backward from a different Python thread that
  reproduces the frozen route's refusal, followed by passing original and
  predictor-math cases through the new graph-bound route.
- Real foreign/reentrant graphs, overlapping attention, unscoped backward,
  token reuse, prepared-tensor changes, phase changes, concurrent misuse,
  call/evidence caps, setup errors and precise inactive-route refusal.
- Hook/method/policy restoration on errors, including an injected failure
  during actual checkpoint recomputation and an attempted mid-backward exit.
- The unchanged owner-aware event reader applied to actual CPU profiler
  events, plus continued rejection by the existing study execution guard.

Four creator/cross-thread unit cases save sixteen actual/reference FP32 tensors
before comparison. Their eight original-bound checks pass. The two cross-thread
records show different creator/runner threads in both modes, with every
recomputation call belonging to the observed root graph. These are real Python
thread and autograd/checkpoint executions; thread identities are not mocked.
They do not reproduce a CUDA engine thread on this Mac.

`models/predictor_eot8_owned_cpu_v1.py` is a separate local CPU analogue, guarded
against university hosts and every CUDA/live domain before model imports. It
retains the old five-pass/eight-draw/two-family objective and pre-gate return
writer while using the new owned gradient call. Four fresh workers share the
pinned earlier tiny initial state. All **40 actual FP32 loss/input returns are
bitwise equal to their corresponding saved v1 CPU returns**. Twenty graph
scopes complete with hooks removed, with 24 within-process exact comparisons,
20 original-bound candidate/reference comparisons and 20 fresh-process exact
replay comparisons. Parameters, targets and runtime contracts stay unchanged;
no parameter gradients accumulate. These checks use the existing tiny upstream
model, not pretrained research results.

The final independent saved-file review rereads all actual tensors and event
ownership while model/backward/SDPA/CUDA, process creation and network creation
are blocked. It rechecks 4,663 frozen predecessor inputs and seals 4,910 inputs.
Earlier 15/26-case logs, source snapshots and provisional review are retained as
preparation history. The active review is `independent-final-review`; counts
are 27 distinct checks, not a sum of repeated runs. Scaffold/YAML verification
and `git diff --check` pass.

## Remaining work

This Mac uses arm64, Python 3.11.16, Torch 2.14.1 and NumPy 2.4.6. CPU evidence
qualifies neither PyTorch 2.8 CUDA hook/thread behavior nor pretrained memory
fit, numerical/backend agreement or performance. The old target call's failing
thread was never captured; the CPU reproduction supports the threading
explanation without rewriting that record or resolving historical numerical
refusals.

Next prepare a separately named live worker and bounded operator/client,
source-bound metadata and unarmed handoff using this corrected route. Verify
cold imports, exact payload, continuing resource limits, error collection and
all original numerical/backend/reference/replay gates before requesting one
additional target attempt. Never patch or rerun the consumed v1 helper. No
new archive, authorization, attempt, SSH upload, scratch proof, university work,
cleanup or paid compute is included in this offline correction. Both university
hosts may idle; all financial caps and holds remain unchanged.
