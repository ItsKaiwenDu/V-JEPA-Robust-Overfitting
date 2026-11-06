# Predictor EOT8 attempt stopped at the backward ownership guard

The attended linux10603 attempt ended in a diagnostic refusal after about
85.3 seconds. Its same-SSH collection is complete: **131 saved files,
69,928,670 logical bytes**, with every hash, size, mode and exact nanosecond
mtime verified. The 70,054,000-byte response stream has the expected extent
and all 131 members were independently rehashed. The consumed claim and
captured source remain immutable. Linux10609 stayed unarmed and unselected.
Do not run either helper to retry this result.

## What completed

The full 2.5 GiB write/fsync/readback proof passed in about 9.60 seconds on
separate XFS `/var/tmp` storage. All 1,926 proof footprint records stayed
within the recorded bounds. The 99-member original packet and 14 overlay
sources match their exact approved bindings. All 4,526 predecessor pins
reverify. Independent review seals **4,663 inputs** without executing a model,
backward, SDPA, oracle, CUDA operation or remote process.

The first fresh original worker loaded the expected checkpoint, training clip,
weights, runtime, clean target, seeds and full masks. It recorded **576 returned
forward SDPA calls**: 24 encoder blocks plus 12 predictor blocks, each called
for all eight draws and both mask families. It preserved the actual rank-zero
FP32 EOT8 loss before the numerical gate. Saved CPU reread gives
`0.5617199540138245`, exactly equal to the prior original loss. This isolated
scalar agreement does not constitute a passing worker or input-gradient check.
The clean target capture also matches the bound target tensor identity.

## Where it stopped

The traceback enters `torch.autograd.grad`, then the original activation
checkpoint's recomputation, then the frozen route's same-thread/active/count
guard. That guard raises `Unowned/concurrent call or call count exceeded.`
The receipt has only 576 rows, below the 4,096-call cap, and no backward call
row. There is no saved input gradient, first-draw trace, accepted phase,
candidate worker or fresh replay worker. The route reports restoration on
exit; the supervisor records worker exit 1 and stops all further work.

The likely cause is the diagnostic wrapper's assumption that recomputation
runs on the Python thread which created the route. PyTorch 2.8's
[autograd engine source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/autograd/engine.cpp#L89-L125)
describes CUDA device worker threads and the caller thread's separate CPU
work. This explanation is an inference from the source and traceback: the
failed call's actual thread ID and active flag were not captured, so direct
thread attribution remains unproven. The local tiny CPU tests did not establish
this CUDA execution behavior. The original numerical gates were never relaxed.

All fifteen sampled GPU process lists contain only the owned worker when a
compute PID appears; none reports another compute PID. Sampling does not prove
continuous exclusivity. No memory-limit or foreign-process refusal is recorded.
The worker stopped after about 67.8 seconds; its supervisor finished in about
72.5 seconds. This diagnostic did not qualify backend selection, full
cross-host derivatives, predictor-math correctness, performance or research.

## Next offline task

Prepare a separately named route which can identify checkpoint recomputation
belonging to its one explicitly owned backward scope while still rejecting
unrelated threads, overlapping backward scopes and other model calls. Record
separate failure predicates plus caller/current-thread and graph identities
before refusal, with bounded evidence. Preserve actual-call backend selection,
private method ownership, restoration, numerical bounds and event attribution;
do not simply disable the thread guard or change global autograd scheduling.
Add genuine cross-thread checkpoint/autograd analogues and concurrency/error
checks before proposing another target attempt. All captured v1 sources and
this failed claim stay frozen.

The one replacement attempt is consumed; no retry, old-clock reset, host
fallback, cleanup or new remote work is authorized. Both university sessions
may idle. Financial caps, holds and last reported $6.56 wallet remain unchanged;
no paid compute ran. Historical numerical refusals and broader project gates
remain open.
