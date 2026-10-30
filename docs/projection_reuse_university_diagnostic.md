# University projection-reuse diagnostic

The CPU proof now has a separately identified CUDA diagnostic and a verified
one-clip handoff. The prepared proposal is unarmed. No university connection,
model job or paid resource was started during preparation.

The next question is whether the original and reused projection paths preserve
the full pretrained model's results on the RTX 3070, and whether reuse helps
measured execution time. The prior CPU proof alone does not answer that question.
See the [preparation review](../report/checks/2026-10-29-projection-reuse-university-preparation.md).

## Run one attended diagnostic

Run this **once in a local Mac Terminal**, outside the university SSH windows:

```bash
bash /Users/kaiwendu/Documents/GitHub/V-JEPA-Robust-Overfitting/runs/pretrained/diagnostic/projection-reuse-university-20261029/run-attended.sh
```

Executing this command opts into one university attempt on **linux10609**. It
uploads the 7.1 MB verified packet, uses the existing Python environment,
checkpoint and upstream source, then runs in the foreground. Enter university
passwords locally when prompted. Keep the Terminal and Mac lid open until the
command ends; the helper uses `caffeinate` through execution and downloads.
There is no Runpod deployment or paid compute in this command.

The total diagnostic allowance is **80 minutes**, including remote verification
and extraction, with a five-second outer kill grace. Each fresh worker has an
18-minute ceiling within that total. Completion within those limits is unproved;
a timeout remains a failed diagnostic. There are no automatic model retries.
The allocator ceiling is 7 GiB. These limits do not grant a training run.

The helper pins the previously verified university SSH public host key. At
runtime, the diagnostic verifies the host, GPU UUID, driver, source bytes,
PyTorch 2.8.0/CUDA 12.8, and existing checkpoint. It refuses other CUDA compute
processes, including a NoMachine process, and monitors that condition while its
own worker runs. It never stops another user's process. No server settings change.

**linux10603 is a separate GPU and can support independent parallel work.**
This comparison uses four sequential workers on 10609 so hardware and competing
jobs do not confound the original/reuse timing comparison. Leave 10603 available
for another independently prepared diagnostic; check its GPU before that launch.

## Scope and evidence

The reference remains the existing shared FP64 forward / FP32 native backward
path. A temporary diagnostic adapter reuses one FP64 projection for each
byte-identical objective bank while retaining every normalization graph, mask
family, EOT loss and independent first-order backward request. Each request gets
its own token clone and backward node. The original CPU-only class still rejects
CUDA; existing scientific policies and their provenance checks are unchanged.
The new adapter has a distinct identity that those policies reject while active.

Four fresh processes run in order: original-0, reuse-0, original-1, reuse-1.
Each verifies the actual new native-backward delegator, the original FP64 oracle,
full-model input and patch-projection weight/bias derivatives under EOT8, and
full-geometry projection values/counts. It then times PGD20/R2/EOT8 plus three
independent scoring banks on the same original training clip. All original
numerical tolerances remain in force. Both mask families remain in every draw.

The attack uses epsilon 4/255 and **diagnostic-only** step 2/255. This does not
select the research step size. Every selected adversary and independent score
must replay exactly within each mode and match between modes. A completed
diagnostic reports observations rather than asserting a useful speedup.

Only `v_BreastStroke_g15_c01` is packaged, with its existing prepared pixel
checksum. The archive has 88 declared source/input files and one handoff
manifest, for **89 regular members**. All 61 original scientific pins and the
historical v4 source packet remain intact. No released weights, wheels, SSH
private keys or confirmation/test pixels are uploaded. Existing pretrained
weights and upstream files are rehashed before model work.

Evidence includes per-worker source/data/runtime identities, profiler events,
projection counts, saved tensors, numerical comparisons, scoring results,
timings, allocator observations, logs and the parent's GPU-process observations.
The original and reused timing paths include their diagnostic guard overhead.
The final export is capped at 1 GiB of uncompressed regular members; failed work
is preserved where an export can complete. Completion is reviewed from the saved
evidence, never inferred from utilization or a final log line alone.

This is not all-parameter optimizer/recovery qualification, research calibration,
a trajectory, or a full-study affordability result. The timing unit includes
three scoring banks, so it is not interchangeable with the earlier ordinary
primary-evaluation timing unit. No study cost projection is updated from CPU
timings or a single unreviewed GPU observation.

## Completion and download recovery

The Mac helper downloads the existing result only after the foreground launcher
has written its exit status. It checks the archive and every declared regular
member before extracting into a fresh local folder. Reply **Done** for independent
review and include any error or evidence path.

If the connection or download fails, preserve the reported folder and stop for
review. The helper may retry a failed upload once **before launch**, or a failed
download once into a fresh folder. It never repeats the attended model launch.
After review, this separate local command can retrieve existing completed files:

```bash
bash /Users/kaiwendu/Documents/GitHub/V-JEPA-Robust-Overfitting/runs/pretrained/diagnostic/projection-reuse-university-20261029/download-only.sh
```

Do not repeat `run-attended.sh`. Both local and remote exclusive claims prevent
an accidental second attempt. Closing SSH can interrupt the attended run; the
previous detached university interruption is not treated as resolved by this
helper. Keep the original failed evidence and never restart solely to copy files.

## Local verification

All **75 focused CPU checks pass, with zero skips**: the 32 original CPU proof
cases, nine execution-adapter cases and 34 diagnostic/handoff/CLI cases. Four
actual fresh tiny upstream workers reproduce all saved tensors, selected
adversaries and scoring results exactly. Tampered sources, receipts, settings,
seeds, scoring banks, runtime, domains and tensor evidence reject.

The tiny CPU reuse/original median ratio is about 0.993: no speed improvement was
observed in that small fixture. This is retained as CPU evidence and is not a
prediction of GPU throughput. Full pretrained CUDA execution remains pending.

The production packet matches the final tested packet byte for byte and passes
an independent restore. Four simulated transport scenarios verify a normal
transfer, an upload retry, a download-only retry and refusal of an existing
remote directory. Duplicate launch refuses; standalone download starts no model.
No real SSH connection occurs in those checks.

The project/compute caps remain $200/$160, with the single $10 diagnostic cap.
Conservative encumbrance stays $9.8116 and its unreserved balance $0.1884; no
financial hold is added or released. The older inclusive $2 hold and historical
accounting gap remain. All known Runpod Pods are terminated.
