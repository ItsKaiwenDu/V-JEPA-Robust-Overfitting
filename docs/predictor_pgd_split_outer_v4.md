# V4 outer handoff: preserve preparation refusals

V4 integrates the reviewed correction for the secondary NoneType.layout error.
The last v3 attempt's setup journal detected another CUDA process before the parent
driver had a guard or durable session claim. Its outer error path then tried to
construct a collector with guard=None. V4 keeps the original setup refusal and
skips collection for an uninitialized parent.

The only new server branch is after the original owned-idle barrier and before
finalization/collection. An unclaimed parent or absent guard re-raises its original
work error. An inconsistent success without a parent session also refuses. The
existing bounded same-SSH refusal route closes the journal and returns the saved
last_guard_refusal. Claimed failures with valid guards keep their original
collection route. No GPU, process, ownership, numerical, timer or storage gate is
weakened, and no partial result is promoted.

The source preparer pins the reviewed correction recipe and the consumed v3 source,
then compares the entire integrated remote_session AST against the cold candidate.
Removing the one branch restores the complete v3 function AST. New source/control/
authorization/client/CLI identities and an exact new handoff path prevent reusing
the consumed v3 grant. Old files, claims and evidence are not edited or reused.

All 69 uploaded sources are byte-for-byte identical to v3: the same packet-aware
bootstrap, bounded parent diagnostics, 63 runtime sources, original monitor,
guards and collector. The full supervised control and source envelope are also
identical. The two dataset dependencies still use the original verified packet
members; no files, library installation or project search fallback is added.
The directory policy remains 26 setup directories with twelve exclusive worker
claims left fresh. Startup import and scientific behavior have not yet been
observed in a successful live v3/v4 worker.

Twelve distinct new checks pass on the first run, with no failures, errors, skips,
repeats or historical test/scientific cohort reruns. The important lifecycle tests
execute the integrated outer function and actual generated coordinator definitions,
private source installation/cache, original Layout/Guard scans, GPU guard logic,
bounded journal persistence/closure and canonical response frames. The original
saved setup GPU query texts reproduce the guard=None refusal. A second preparation
failure with a valid guard but no session claim is also returned directly. A third
case initializes the session, injects one inert proof-worker failure, and keeps the
collection route with the original saved-work receipt predicate.

Host/namespaces, GPU command responses, allocator, clocks/alarms, packet receive/
extract/content verification, proof admission, PID-handle API and worker launch
are explicit fakes. The emergency reserve fixture is zero bytes. Collection output
in the claimed-failure case is inert, not a scientific archive, and only its
original ended-session receipt predicate executes. No real child, timer, PID/proc,
signal, GPU query, proof, model, tensor, gradient, backend, profiler, CPU reference,
SSH or paid compute executes. Ordinary small local metadata writes/fsync do run.
These tests check supervisor integration, not live installation, fit or numerical
correctness. Existing v3 import/monitor checks are preserved by exact source hashes.

The final candidate is source-only and unarmed. Exact hashes/bytes are in
configs/predictor_pgd_split_outer_v4.json; the scope is in
report/checks/2026-11-16-pgd-split-outer-v4-scope.md. Full independent saved-file/AST
review must finish before a new exact approval can arm it. Earlier grants are all
consumed. No repeated manual GPU check is needed; a future approved launcher checks
live and refuses if another CUDA process remains active.

Original limits stay 40 minutes of work including the full 2.5 GiB storage proof,
eight matched gradients and two component VJPs/one inner VJP; then up to ten minutes
of collection through the same SSH, within 50 minutes total including login.
Setup/capture/full response remain 3 GiB/2.25 GiB/3 GiB; raw 1,792 MiB, metadata
including trace 64 MiB, trace 8 MiB, GPU allocator 7 GiB and observed component RSS
4 GiB / 120 seconds each. Bounded parent diagnostics retain their original 16 KiB
cap inside the 128 KiB session file. No quota/continuous-exclusivity/instantaneous-
stop guarantee is added.

No retries, separate retrieval, claim/clock reset, proof/result reuse, old-file
cleanup, installation, private-key upload, held-out pixels, new CPU reference,
warmup, replay/cohort repeat or paid compute is included. Scientific/training and
performance/cost adoption remain false; financial caps, holds and reconciliation
stay unchanged.
