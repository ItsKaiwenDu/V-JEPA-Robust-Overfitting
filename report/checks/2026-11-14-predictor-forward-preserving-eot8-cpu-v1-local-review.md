# Local full EOT8 forward-preserving VJP: passed

The separate 64-call tiny CPU integration passes the actual native eight-draw,
two-family objective. Nineteen tests pass on the first and only suite run, with
zero failures, errors or skips. No old suite or successful comparison is repeated.

- Four comparisons: two fixed eight-draw banks × creator/separate-thread backward;
  original-before/candidate/original-after gives twelve complete local evaluations.
- All sixteen predictions, targets, scalar loss and native L1 cotangents remain
  bitwise original. Original input gradients replay exactly before/after.
- Candidate full input gradients pass the unchanged rtol 0.002 / atol 2e-7 bound:
  zero failed elements, max absolute error 3.4924596548080444e-10, max error/tolerance
  ratio 0.0008573618110200237 (pass requires ≤1).
- Each candidate has 64 predictor forward calls including actual checkpoint
  recomputation, 32 distinct completed public-math VJP GraphTasks and one actual
  native outer root. Each model evaluation records 64 primary + 64 genuine native
  checkpoint attention calls. Hooks, flags, aliases, methods and markers restore.
- Native incomplete-coverage, missing-recomputation and foreign-graph refusals
  remain enforced. Math admission never appends native recomputation evidence.
- A separate NumPy-only review verifies 624 actual saved FP32 arrays and receipts,
  reconstructs native cotangents and checks the full input bounds without executing
  any model, autograd, SDPA, CUDA, subprocess, network or new derivative reference.
- Both earlier eight-call entries, the native guard, 882 sealed prior inputs and
  245 actual target byte/mode/timestamp descriptors remain unchanged.

This qualifies local tiny EOT8 integration only. Target Torch 2.8/CUDA, pretrained
geometry/memory, optimizer, performance, research adoption and training remain
unqualified. The earlier 3,724-entry rejection and financial gates remain unchanged.
No remote operator, target authorization, upload, cleanup or paid work is created.

Next: prepare a concrete unarmed full-model university handoff with a fresh native
original control and exact-forward/native-cotangent/full-input gates. Fresh replay
is conditional on initial passes. Do not execute it or reuse consumed grants.

Implementation and scope: `docs/predictor_forward_preserving_eot8_cpu_v1.md`.
Exact source, first test log/XML, arrays, saved reviewer and seals:
`runs/pretrained/diagnostic/predictor-forward-preserving-eot8-cpu-v1-20261114/`.
The accompanying JSON pins the independent saved review and per-case receipts.
