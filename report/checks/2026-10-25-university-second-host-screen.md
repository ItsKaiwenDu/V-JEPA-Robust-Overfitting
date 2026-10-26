# Second university GPU: metadata-compatible candidate

The user connected to linux10603 and reported RTX 3070 UUID
`GPU-e54fae90-feeb-8106-d8c0-1974fe48555f`, distinct from linux10609's
`GPU-4e2a7c77-19bd-689f-6c08-60eb774c0d52`. Both report 8 GiB VRAM and
NVIDIA driver 580.178.04. Python 3.11.17, PyTorch 2.8.0+cu128, NumPy 2.4.6
and PyTorch CUDA build 12.8 match the existing reference. Reported usage was
94 MiB; no compute-PID lines appeared in the supplied output. This is a
snapshot, not an exclusive reservation or a full numerical qualification.

The candidate passes an initial metadata screen only. No CUDA tensor test,
model, numerical oracle, fresh replay or panel case was run here by the agent.
All 198 original scientific pins and 58 execution sources remain unchanged.
Six cases remain independently accepted. Case clip-03-bank-0 was reported
running on linux10609; no completed export is reviewed in this screen.

## Constraint and qualification path

The frozen v1 plan explicitly specifies max_parallel_cases=1 and the same
host, GPU UUID, driver and software across the panel. Its worker verifies
actual hardware; its journal requires a completed ordered prefix and a single
supervisor lease. The current helper must not be repointed or its guard
bypassed to manufacture second-host v1 evidence.

A possible two-host extension would require the following separate work:

1. Predeclare a bounded shadow qualification using an existing training-only
   reference, fixed checkpoint/targets/starts/masks and the unchanged numerical,
   gradient, native-kernel and fresh-replay checks. Check actual CUDA access
   and full runtime/model identity first. Define its resource limit before
   launch; this metadata screen grants no additional attempt or GPU time.
2. Qualify numerical and replay behavior on the candidate. Matching model
   names and versions alone does not prove bitwise equivalence. Retain any
   failure; do not relax tolerances or relabel it as a panel pass.
3. If qualified, record an explicit execution-plan revision with both actual
   host/GPU identities, one worker per GPU and whole clip pairs assigned to
   the same host. Keep clip-03's two banks on linux10609. The existing v1 plan,
   handoff, receipts and partial evidence remain immutable.
4. Implement and test separate worker roots plus coordinated case ownership,
   bounded termination, unique exports and independent review. These hosts
   may share the same home filesystem: separate SSH sessions do not create
   isolated journals. Preserve each artifact's original plan/execution identity
   and explicitly verify compatibility before combining old/new evidence.

No plan revision, parallel runner, shadow qualification or second-host panel
case has been implemented or launched. No existing-case restart, partial-panel
aggregate, attack choice, precision adoption, research admission or paid
compute follows from this metadata result. The running case stays unchanged.

[Machine-readable screen](2026-10-25-university-second-host-screen.json).
