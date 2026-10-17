# Data preparation and attack checks

These implementations support the feasibility stage. Confirmatory training
and robust-overfitting measurements have not started.

## Official source and grouped manifests

Use the [official UCF101 dataset and split annotations](https://www.crcv.ucf.edu/data/UCF101.php).
The upstream documentation explains that clips within a source group come
from a common video, so these groups must remain separate across partitions.
Our study uses official split 1, then holds out entire training groups for
validation. It is a subset experiment, not the standard full three-split
UCF101 benchmark.

The prepared files are in `data/manifests/ucf101-split1-v1/`. They contain
2,020 training, 505 validation, and 505 test relative paths, with no shared
video IDs or source groups. Labels are zero-based. Manifest and annotation
hashes, the seed, and all held-out validation groups are in `metadata.json`.
Video content duplicates and decoding integrity across the entire dataset
have not been audited; group separation alone does not establish those.

Regenerate into a fresh directory from the extracted official lists:

```bash
python scripts/prepare_data.py \
  --lists-dir data/raw/ucfTrainTestlist \
  --output data/manifests/ucf101-split1-replay
```

Sampling sorts the classes/groups, shuffles group IDs with a class-specific
seed, reserves groups until at least five validation clips are available,
then samples 20/5/5 clips per class. Every unselected video from a reserved
validation group remains excluded from training. Input row order does not
change the result. Existing outputs are rejected. The parser preserves the
official `HandstandPushups/v_HandStandPushups...` case-only naming difference.

## Deterministic video preprocessing

The proposed recipe is `cfr-repair-center-4fps-16f-shortside256-centercrop-bilinear-aa-v1`:

1. Decode the primary video stream to RGB using FFmpeg, without automatic
   rotation. Read presentation timestamps with FFprobe. For missing timestamps
   in legacy AVI, fill only the missing entries using the declared average
   frame rate, and only if every observed timestamp agrees with that constant
   rate within 1e-5 seconds. Record every repaired index/value. Variable-rate
   streams with missing timestamps fail instead of receiving guessed timing.
   This handles a missing final B-frame timestamp in the real-data check.
   Require a matching decoded frame count and strictly increasing timestamps.
2. Center a 16-frame sampling window at 4 FPS, spanning 3.75 seconds. Select
   the closest decoded frame to each timestamp, choosing the earlier frame
   for a tie. Short clips start at zero and repeat the nearest final frame.
3. Convert RGB uint8 to float32 [0, 1]. Resize the shorter side to 256,
   flooring the proportional longer side. Use bilinear interpolation with
   antialiasing and `align_corners=False`, then center-crop 256 x 256.
4. Save `[3, 16, 256, 256]` raw pixels, selected timestamps/indices, source
   SHA-256, decoder versions, PyTorch version, and the resulting pixel hash.
   The model applies mean/std normalization afterward, inside the gradient path.

The decoder requires `ffmpeg` and `ffprobe` on PATH. Local checks use FFmpeg
9.0.1. Decoder or interpolation version changes can change pixels, so cache
identity includes those versions and source bytes. A decoding failure stops
preparation; it does not silently replace a selected video.

Prepare eight evenly spaced training-manifest rows for a pilot:

```bash
python scripts/prepare_clips.py \
  --manifest data/manifests/ucf101-split1-v1/train.csv \
  --data-root data/raw/UCF-101 \
  --output data/processed/pilot-train-v1 \
  --limit 8
```

This command accepts only the checksummed training or validation manifest.
Final-test clips are not decoded by this preparation path. A failed run
may leave partial clips without an index; use a fresh directory after
investigating it. A completed index identifies the exact clips and pixels.
Full dataset decoding and target-feature caching remain future work.

## Pixel attacks

`attacks/pgd.py` implements random-start L-infinity PGD and a one-step FGSM
variant. Epsilon is in raw [0, 1] units; use `4/255`, not `4`. Projection
enforces both the epsilon box and valid pixel bounds after every update.

The attack differentiates only with respect to input pixels and does not
populate or clear model parameter gradients. It preserves caller-controlled
model mode. Clean targets are computed once and detached. A fixed bank of
upstream mask draws defines the objective throughout the attack; averaging
this bank implements expectation over transformations (EOT).

Candidate selection is per example and includes clean input, random starts,
and all PGD iterates. More steps or restarts therefore cannot reduce the
recorded best attack-bank loss when the objective and initial random draws
are identical. This is not a guarantee about an independent scoring bank,
nor proof of convergence to the worst case. Independent masks are used only
for final scoring, not for choosing attack iterates. The FGSM wrapper also
retains the clean input if its loss is larger.

Small-fixture tests cover zero epsilon, exact linear-objective FGSM optima,
pixel bounds, deterministic replay, EOT gradient averaging, parameter-gradient
preservation, and native-loss ascent without changing model weights/targets.
PGD-50/EOT-8 convergence, embedding-distance baselines, downstream attacks,
and study-wide epsilon sweeps remain pending.
