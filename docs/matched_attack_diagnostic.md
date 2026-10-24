# Matched training-only attack diagnostic

October 23, 2026. The separate implementation has passed focused tests on actual
small upstream CPU transformers. The university CUDA run is **not started**;
current host/GPU availability and full-model 8 GB memory fit are unverified.
The user deferred their SSH check until tomorrow. All Runpod Pods remain
user-confirmed terminated; this handoff does not rent or resume anything.

The [design](matched_attack_diagnostic_plan.md) holds the original study methods
and configuration fixed. `models/matched_attack.py` calls the unchanged PGD
core in four diagnostic arms: 20/50 steps crossed with four/eight attack draws.
All use radius 4/255, step 2/255, two shared random starts, one nested eight-draw
attack bank, and common independent four-draw main plus two audit banks.
Actual upstream seed-space collisions reject. Independent scores never select
an iterate, restart, arm or mask bank.

Every objective input, attack loss and input gradient is traced by hash. Starts,
clean pixels, detached clean targets, masks and selected adversarial tensors are
retained. The first 20 gradient steps must match exactly for each EOT count;
the 20-step endpoint must equal the 50-step run's next input. Best attack-bank
loss cannot decrease. Independent scores may decrease. Model weights, fixed
target, parameter gradients, modes and Python/NumPy/PyTorch RNG are preserved.
The separate contract binds runtime, geometry/backend/precision, all project
model/attack/data modules, upstream code, checkpoint, exact training pixels and
the handoff manifest. Noncanonical arms never use `EvaluationSettings` or the
canonical evaluation journal.

A second fresh process reloads the same released checkpoint and replays **arm A
only** (20 steps/EOT4). Exact contract, trace, gradient, selected pixels and scores
must match. This bounds extra work while checking fresh-process execution; it
does not claim fresh replay of all four arms. Missing replay, timeout, OOM,
changed source/runtime or incomplete artifacts cannot pass. Failed numerical
or replay checks do not trigger a fallback or wider tolerance.

Instrumented times include per-step observation overhead. They are explicitly
excluded from A40 economic pooling, even on the same GPU model. One training
clip cannot establish study-wide convergence, choose the final primary attack,
admit the $200 study or support a robust-overfitting conclusion. Original failed
evidence, 20 classes, all seeds/YAML and held-out closure remain unchanged.

## Tomorrow: inspect the host first

In the existing university SSH session, run only:

```bash
hostname
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
```

Share that output before starting the diagnostic. There is no need to keep SSH
open overnight. Use the explicit host when reconnecting; the generic university
SSH alias can route to a different workstation.

## Prepared offline handoff

The builder `scripts/prepare_matched_attack_handoff.py` includes one verified
declared training-monitor tensor, complete metadata, source and a pinned Git
bundle. It includes no released checkpoint, dependency wheels, credentials or
held-out pixels. It checks that all 198 existing non-Markdown source pins still
match the accepted A40 base. New scripts use the already installed university
Python environment and previously verified checkpoint; no install or download.

The local handoff folder is
`runs/pretrained/diagnostic/matched-attack-handoff-20261023`. Its archive manifest
records every regular member, byte count and SHA-256. Verification of the final
package and extracted tests is recorded separately in
[the preparation report](../report/checks/2026-10-23-matched-attack-preparation.md).

After host review, copy the archive, archive manifest, `SHA256SUMS` and
`verify_extract.py` from that folder into the university folder
`$HOME/Desktop/VJEPA-Robust-Overfitting`. Extract into a fresh temporary
destination using the standard-library verifier, rather than merging into an
existing diagnostic directory. The extracted root is `matched-attack-20261023`.
The verifier rejects changed members, symlinks and escaped paths before writing.
From the university folder containing the four copied files, the extraction
commands are:

```bash
sha256sum -c SHA256SUMS
test ! -e matched-attack-unpacked-20261023
"$HOME/.venvs/vjepa/bin/python" verify_extract.py \
  --archive vjepa-matched-attack-20261023.tar.gz \
  --manifest vjepa-matched-attack-20261023.tar.gz.manifest.json \
  --destination matched-attack-unpacked-20261023
```

This writes `matched-attack-unpacked-20261023/matched-attack-20261023`.

The launcher requires the reviewed hostname and exact GPU name. For example,
**only after reviewing a fresh check that matches these values**:

```bash
bash scripts/university_matched_attack.sh linux10609 "NVIDIA GeForce RTX 3070"
```

Run it inside the newly extracted root. The default existing repository is
`$HOME/Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting` (with the
hyphen after V in the inner name); an optional third argument can point to
another verified existing repository. No result is assumed
from the example hardware. The launcher refuses other active CUDA compute
processes and uses an exclusive diagnostic lock.

## Bounds and evidence

The four-arm worker has at most 3,600 seconds; the fresh A-only replay at most
900 seconds. Parent scheduling ends within 4,620 seconds, including setup and a
30-second reserve, and the outer detached launcher has a 4,800-second deadline
plus 15-second kill grace. A timed-out process group gets TERM then KILL after
10 seconds. These are hard upper bounds, **not runtime predictions**.

CUDA allocation is capped at the smaller of 7 GiB and 90% of VRAM, including
model allocations. EOT graphs may exceed that cap on an 8 GB card; a rejection
is useful failure evidence and does not authorize changed precision or EOT.
A new observed native backward kernel probe is required. No optimizer runs.

The detached launcher prints a unique run directory, watch command and result
paths. SSH can be closed. `Ctrl-C` on `tail -f` stops watching, not the job.
Inspect `results/session.json` and the saved worker records; an exit code alone
does not prove numerical or replay acceptance. Completed or failed Python
sessions export an allowlisted `results/evidence.tar.gz` and member manifest,
bounded to 256 MiB before compression. Startup/outer-kill failures may have only
launcher logs and partial files; preserve them and never label them passes.

Review the raw paired losses and both audit banks before choosing any further
training-only checks. A protocol change still needs research calibration,
complete schedule/cost review and explicit admission before trajectories begin.
