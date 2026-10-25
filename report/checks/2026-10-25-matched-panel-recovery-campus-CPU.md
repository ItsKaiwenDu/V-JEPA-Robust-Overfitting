# Campus recovery CPU checks and logged attended launch

The university probe completed on linux10609 at 12:53 PM Pacific on October
25. All 35 unchanged operator tests passed in 2.625 seconds, with no failures,
errors or skips. JUnit contains 35 passing cases; probe, pytest and SSH exit
statuses are all zero. The saved hash checks confirm original operator/test,
proposal and authorization bytes, the unchanged journal and unused grant.
No real recovery case or GPU model was started. The earlier connection close
remains unexplained; no historical root cause is inferred from this pass.

The new, separately named launcher reuses that reviewed CPU evidence without
repeating the tests. It verifies its pins, the unused grant/attempt boundary,
original journal and fresh host/GPU/driver and idle compute status. It invokes
the unchanged operator, which performs the original full evidence preflight
before consuming the already approved grant. It runs only clip-01-bank-0,
attempt-02, for at most 4,800 seconds plus 15-second kill grace. Original unknown
time remains charged in full; no new grant is issued. No partial arms or
remaining queue are reused or dispatched.

Per-invocation shell trace, gate, process output and exit files are retained.
Operator stdio follows the campus-tested file-backed path; stdout is relayed
to the attended SSH console. Hangup/interrupt requests process-group cleanup.
Six ordinary CPU shell fixtures check early failure, immediate-zero output,
nonzero/deadline-code propagation and HUP/TERM cleanup. Eight metadata fixtures
check real pinned proof bytes and reject used grants, existing attempts,
changed journal/CPU XML/operator, busy compute and wrong GPU identity. Local
host/GPU and child fixtures are explicitly mocked: they qualify neither GNU
timeout nor neural/CUDA correctness. Syntax checks also pass. Raw validation
and source excerpts are retained separately from scientific evidence.

The Mac helper uploads only the pinned new starter, holds an attended awake
session, retains its transcript, and copies launcher evidence and, on success,
the new export/operator/journal. Independent export review remains required.
The helper has not been executed; both completed cases, original sources,
archive, proposal, authorization, study and budgets remain unchanged.

[Machine-readable verification](2026-10-25-matched-panel-recovery-campus-CPU.json).
