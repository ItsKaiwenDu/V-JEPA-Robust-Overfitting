# Native backward probe correction — October 21

The first integration attempt stopped at the small native backward probe,
before loading the full pretrained model in the native worker. The user
reported exit status 1 and `Native backward CUDA kernel boundary was not
observed`. Both cuDNN worker logs contain only deprecation warnings; their
saved results and exact replay are not accepted until the complete export is
copied and verified. The fourth worker was not reached.

## Diagnosis and limits

The original probe matched `col2vol` in profiler names. The previously verified
university native Conv3d export instead records a positive-duration
`at::native::vol2im_kernel<float, float>` CUDA launch. `col2vol` is the helper
name, not the observed kernel name. The earlier worker is
`university-convolution-precision-results-20261021/results/workers/native/input-0.json`,
SHA-256 `0de375f98ff1c6d1c16091d78e0b5ac78cd7d3d07f3e655a0451236084f03028`;
see the [verified component outcome](../report/checks/2026-10-21-university-cuda-convolution-precision.md).

The old probe also discarded its observations when it raised, so the pasted
trace does not expose this attempt's actual profiler names. This identifies
an incorrect expectation in our diagnostic, not a newly verified CUDA pass.
The failed run must remain failed/preserved. Full pretrained native gradients,
attacks, exact replay and optimizer recovery remain pending.

## Corrected observer

The observer is now in `models/native_probe.py`. It retains raw events with
their CPU/CUDA device type and duration, requires an actual positive-duration
native `vol2im_kernel` CUDA event and a convolution-backward CPU operation,
and rejects cuDNN kernels or convolution operators. CPU names or aggregate
device time cannot establish a CUDA kernel launch. The profiler window still
contains only backward, after forward's cuDNN context has exited.

Flags must be restored after both directions, and input/parameter gradients
must be finite. Evidence is saved to worker JSON **before** a failed gate
raises. Export verification recomputes the gate from those retained events;
it does not trust a pass boolean. Worker/session and probe schemas are now 2.
The actual-CUDA regression uses the shared probe implementation.

The mathematical wrapper, official weights, normalization, clean target,
attacks, numerical tolerances, exact replay criteria, time/memory bounds and
study/proposal configuration are unchanged. This is an observer correction,
not permission to accept incomplete evidence or change scientific thresholds.
Fresh workers are required; old cuDNN results are not mixed into the new
source identity. Native workers run first to stop a rejected probe before
repeating expensive reference work.

## Separate retry handoff

Archive:
`runs/pretrained/diagnostic/cuda-native-model-fix-handoff-20261021/vjepa-native-model-fix.tar.gz`.
Extracted folder: `cuda-native-model-fix-20261021`. It uses the existing university
environment, upstream, checkpoint and first verified training clip read-only.
The original archive/folder/results remain unchanged. No downloads, installs,
full-model optimizer steps or paid cloud provisioning.

First, on the Mac, preserve the failed export and copy the new source archive:

```bash
cd /Users/kaiwendu/Documents/GitHub/V-JEPA-Robust-Overfitting
scp -r kdu@linux10609.dc.engr.scu.edu:Desktop/VJEPA-Robust-Overfitting/cuda-native-model-20261021/runs/native.kq8WhI \
  runs/pretrained/diagnostic/native-model-failure-20261021/
scp runs/pretrained/diagnostic/cuda-native-model-fix-handoff-20261021/vjepa-native-model-fix.tar.gz \
  kdu@linux10609.dc.engr.scu.edu:Desktop/VJEPA-Robust-Overfitting/
```

The local receiving directory is prepared. Then, inside university SSH:

```bash
(
  set -e
  cd "$HOME/Desktop/VJEPA-Robust-Overfitting"
  test ! -e cuda-native-model-fix-20261021
  tar -xzf vjepa-native-model-fix.tar.gz
  bash cuda-native-model-fix-20261021/scripts/university_native.sh
)
```

Follow the printed `tail -f` command. Ctrl-C exits the log viewer; it is
necessary to regain a shell prompt before typing another command. The
diagnostic runs detached, so SSH can be closed. Reconnect to linux10609 to
check the job. Four fresh workers, 180 seconds each, 840-second scheduling
budget and a 900-second outer supervisor plus 5-second cleanup remain in force.
The CUDA allocator cap is 7 GiB (at most 90% of visible VRAM), excluding driver
and other users' allocations. The new attempt has not run on CUDA yet.

When it finishes, preserve the entire new run directory and verify session,
worker, tensor and source hashes before interpreting the comparison. The
university 8 GB GPU still cannot establish full ViT-L AdamW recovery; a later
bounded sufficient-VRAM acceptance pilot and measured budget admission remain
required before research training. Reported spending remains $1.195; no new
paid compute was used in this repair.
