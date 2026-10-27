# 10603 GPU-exclusivity failure — October 26, 2026

FloorGymnastics bank1 (`clip-04-bank-1`) on linux10603 stopped after the
A/B/C/D matrix returned, during the final hardware observation. The strict
guard reported another CUDA compute process. The fresh A replay was not
started, and this incomplete case is not accepted. All 122 campus CPU guards
had passed. The exact elapsed budget and saved partial contents still need
readback; no retry or new allowance is authorized by this incident review.

The user's later snapshot lists PID1162024, `/usr/NX/bin/nxnode.bin`, as a
compute process using175MiB. The whole GPU then used277MiB at0% utilization.
That is consistent with the guard condition, but the original failure trace
did not retain a competing PID; this later snapshot does not prove which PID
was active at the failure or who owns it. No process was killed, server
setting changed, or exclusivity/numerical gate relaxed.

The already-running linux10609 case was allowed to finish within its original
limit. It exited0; the original pinned reviewer and retained-evidence audit
verify Biking bank0. Ten of twenty diagnostic cases are now independently
verified, with four fully paired clips; its new independent marker has not
yet been synced to the nine-case university ledger. A new global stop was
observed in its saved progress snapshot, preserving the failed peer.

Copy the existing failed case from a **Mac Terminal prompt**:

```bash
bash /Users/kaiwendu/Documents/GitHub/V-JEPA-Robust-Overfitting/runs/pretrained/diagnostic/panel-gpu-contention-20261026/copy-existing-partial.sh
```

This reads only the existing numeric partial archive/sidecar, session,
envelope, claim/outcome/stop, fixed launcher/JUnit/source logs and a current
read-only `nvidia-smi` snapshot. Optional missing observations, replay,
receipt and export/review records are recorded explicitly. The helper starts
no model/test/GPU work, writes nothing remotely and retries only a failed copy
once. CPU byte fixtures verify the30required/11optional file inventory,
wrong-host rejection and preservation/rejection of corrupted copies. Those
fixtures admit no numerical case or CUDA qualification.

After copying, inspect the saved worker/session evidence and actual consumed
time before proposing recovery. Do not repeat the GPU launcher, fabricate a
missing observation, discard failed files, clear the new stop or stop another
process. Nine earlier cases and the valid new Biking case retain their original
execution identities. Scientific settings, budgets and held-out gates remain
unchanged. [Structured incident](2026-10-26-panel-GPU-contention-incident.json).
