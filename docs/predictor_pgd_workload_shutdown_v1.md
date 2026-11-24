# Bounded whole-PGD shutdown adapter

The separate local adapter preserves the first workload exception while observing
bounded termination, child exit and handle cleanup. It addresses the error-masking
path found in the [consumed v3 attempt](../report/checks/2026-11-23-pgd-workload-handoff-v3-target-review.md).
It does not establish that the earlier attempt's missing primary cause was timeout.
The [local review](../report/checks/2026-11-23-pgd-workload-shutdown-v1-local-review.md)
passes 21 new checks and an independent saved audit.

## Cleanup and error preservation

The new cold factory is `models/predictor_pgd_workload_shutdown_v1.py`, with
explicit cleanup definitions in `models/predictor_pgd_workload_shutdown_logic_v1.py`.
It emits an unbound source graph, control and outer definitions into a new local
directory. There is no package, authorization, launch or arming API.

- The compute deadline stays absolute. Cleanup gets at most two seconds from the
  existing overhead, clamped to the original hard work deadline. TERM gets at
  most 0.5 seconds; KILL confirmation and reap use the remainder. This replaces
  the former two-second TERM grace followed by a short reap attempt.
- Held handles remain open until all observed owned identities are dead and the
  child terminal status is known. No numeric PID signal, descriptor reuse, guessed
  death or ownership substitution is introduced. If confirmation fails, handles
  remain held and the idle gate refuses finalization/collection.
- A cleanup or idle refusal adds bounded observations to the original exception.
  It cannot replace a timeout, RSS refusal or interruption with an idle error.
  Failure state retains sampled RSS/count, owned identities, exit status, cleanup
  bounds, errors and a bounded cause/context chain. Existing session and metadata
  caps apply; scientific and completion flags remain FALSE.
- The outer still requires confirmed idle before collection. It retains the
  primary error when that gate refuses. Re-arming finalization after confirmed
  cleanup cannot turn a failed compute role into success or extend compute time.

Five derived source files have full-byte inverses: unit monitor, bootstrap,
scratch catalogue bindings, coordinator and outer. The 46 scientific/preparation
runtime files are byte-exact to v3, including early device transfer, strict mapped
checkpoint loading, PGD/evaluator, numerical/native ownership, hashing and timers.
The unit monitor's observation body is AST-exact outside cleanup. The frozen proof
monitor source file is byte-exact; its role-adapted observation body keeps the
existing proof gates while delegating only cleanup to the new helper.

The graph contains 47 runtime files/457112 bytes, a 46-module executable cache,
nine supervisors and 55 uploads/905563 bytes. The original packet, 264 primary
result paths, scanner/assessment/Guard, proof reader, storage catalogue limits,
GPU gates and priority codec remain. Whole-unit 720s/objective 120s, soft/hard work
2280/2400s, collection 600s/connection 3000s, RSS 4GiB/allocator 7GiB and all other
compute/resource caps remain. Live execution and research/training admission are
unbound/FALSE. The old clients, packages, grants and sealed sources are unchanged.

## New local controls and retained history

Twenty-one distinct checks finally pass over 42 executions. The first 21 stopped
in fixture setup because a normal-returning fixture was treated as a generator.
Only that fixture line changed; production was not exercised or changed by the
failed run. All 21 failed nodes then passed; no passing case was repeated. Before
the checks, cold build v1 refused a shadowed helper name before emission. The
factory correction and successful v2 cold build, both source versions and logs
are retained. Production did not change after the first check run.

Direct fake controls cover delayed KILL/exit/reap, death/signal/wait/poll/close
failures, backward/nonfinite clocks, hard-work clamping, causal cycles, descendants
outliving the root, pre-monitor identity/release failure and unconfirmed exit.
Seven new fake OS lifecycles exercise source cache, release gate, worker/objective
timers and parent readers. One completes all four fake roles; three collect failed
work after timeout/RSS/interruption with the first exception retained; one refuses
collection with unconfirmed handles; two stop before worker release.

The four complete streams contain 432/171/171/171 files. The independent stdlib
audit parses their saved raw frames, payloads/references, hashes, modes, mtimes,
journal chains, prefix, terminal and EOF directly. Failed session/work/partial/
journal flags remain unchanged; transport TRUE does not authenticate computation.

All process, PIDFD, signal, clock, timer, GPU, proof and compute ports are explicit
FAKE. Proof records stay passed/complete FALSE with zero proof bytes; the real
proof reader refuses them. Seven ordinary 16MiB reserve fixtures and opaque
archive/source copies are real filesystem work. No Torch/NumPy, model, gradient,
new oracle/trace, scientific cohort, actual target process/proof/GPU or network
work occurs. The earlier real target's 31 completed gradients and proof remain
preserved, not rerun or promoted.

## Budget review and next local boundary

The saved budget review makes conditional arithmetic explicit. If all 40 contexts
equalled an observed endpoint, their recorded intervals alone would sum to
709.798–715.800 seconds, leaving 4.200–10.202 seconds inside 720. This is a scenario,
not a performance estimate or a bound on unseen contexts. Those intervals exclude
their own journal publication/timer exit and other preparation/scoring/caller
overhead. No completed original/candidate/replay cost is established.

Three full 720-second windows require 2160 seconds. Only 120 seconds remain before
the original 2280-second soft work deadline for setup, proof and intervening
overhead; the setup/proof design ceilings alone sum to 240 seconds. Actual earlier
setup may be shorter, but all design ceilings cannot be promised simultaneously.
No new timeout, shorter PGD/EOT, disabled guard, throughput/training percentage or
hardware-fit claim is adopted.

Next bind and test a separate client/bootstrap/reader integration for these exact
sources, and resolve complete-unit budget feasibility before any fresh exact
scope. Existing clients deliberately remain bound to old sources and cannot serve
as this adapter's operational client. All real grants are consumed. No new scope,
handoff, approval request, retry/resume, retrieval, remote cleanup or paid work
follows from this local stage. Mechanism remains closed; full units/replay/timing,
memory/recovery/economics/calibration/data/protocol/training gates remain.

Raw stage: `runs/pretrained/diagnostic/pgd-workload-shutdown-v1-20261123`.
Seal: `sealed-workload-shutdown-v1.json`. Prior preservation verifies 53571 hashes
and 58880 fixture entries; the saved audit additionally checks 2805 new entries.
The old FIFO is LSTAT only and never opened. Local device16777233 matches its prior
conditional observation; target/old devices are never remapped. Four narrow prior
directory lifecycle exceptions remain exact. Mutable index snapshots are retained
and live indexes excluded from the seal. User alone stages/commits/pushes.
