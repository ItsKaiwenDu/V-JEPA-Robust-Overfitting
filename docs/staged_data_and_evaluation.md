# Staged data and native evaluation

October 20, 2026. The proposed identities are exported, and the **100 actual
training clips plus 20 exploration-validation clips** are prepared locally.
Native evaluation, evaluation recovery, saved-anchor loading, and primary
selection have CPU fixture coverage. This is preparation and implementation
evidence, not a frozen protocol, a research trajectory, or budget admission.

## Data actually prepared

The [staged manifests](../data/manifests/ucf101-staged-20x5-v1/metadata.json)
reproduce the earlier [proposal](budget_study_plan.md) exactly. They contain
100 training, 20 training-monitor, 20 exploration-validation, 20 confirmation-
validation, and 60 test rows. The monitor reuses actual training videos.
Class IDs retain their original zero-based UCF101 values.

The exporter checks the proposal's source hashes, recomputes the class/group/
video selections, and compares the result to the recorded proposal. Its reader
repeats this comparison, so updating a CSV and its adjacent checksum cannot
silently substitute a different subset. Parent manifests and protocol 0.3
are unchanged. Paths and checksums identify content; no absolute local paths
are embedded in the CSVs.

Only the 120 training/exploration sources were extracted from the existing
checksum-verified archive. Each video was decoded twice with the original
16-frame, 256-pixel recipe. **All 120 pixel tensors and preprocessing records
matched exactly**. No duplicate source-file SHA-256 or prepared-pixel SHA-256
was found among those videos. Every saved tensor was then loaded and its
identity, recipe, shape, range, and pixel checksum verified. The 20 monitor
entries read the same training cache rather than making separate copies.

This process took 54.73 seconds on the local CPU environment and produced
1,575,441,215 bytes including extracted sources and prepared caches. This is
a local preparation observation, not a GPU-throughput estimate. The archive
and all generated pixels remain ignored by Git. Small manifests and the
[preparation evidence](../report/checks/2026-10-20-staged-data-preparation.json)
are tracked.

**Limits:** Exact hashes do not detect visually similar or re-encoded videos.
Near-duplicate review remains pending. Confirmation/test metadata was exported,
but none of their videos was extracted, decoded, scored, or used for tuning.
The current preparation and pixel loader reject those panels. Full content
validation and final protocol freezing remain incomplete.

The commands used were:

```bash
.venv/bin/python scripts/export_staged_manifests.py \
  --output data/manifests/ucf101-staged-20x5-v1

.venv/bin/python scripts/prepare_staged_clips.py \
  --manifests data/manifests/ucf101-staged-20x5-v1 \
  --archive data/raw/UCF101.rar \
  --output data/processed/staged-exploration-v1
```

These output directories now exist. A replay requires new directory names;
the commands refuse to overwrite earlier preparation. `preparation.json` is
written only after every selected source passes. Partial output from a failure
is preserved and is not a usable completed cache.

## Evaluation contract

[evaluation.py](../models/evaluation.py) implements fresh, raw-pixel native
attacks with clean fixed targets, one clip at a time:

| Method | Steps | Restarts | Attack draws | Scoring draws |
| --- | ---: | ---: | ---: | ---: |
| Routine PGD | 5 | 1 | 1 | 1 |
| Primary PGD | 20 | 2 | 4 | 4 |
| Convergence PGD | 50 | 2 | 8 | 4 |
| FGSM control | 1 | 1 | 4 | 4 |
| Uniform random control | Sampled once | 1 | 4 | 4 |

Every draw includes both upstream mask families. The attack bank stays fixed
through all steps and restarts, matching the pilot's EOT convention. Candidate
selection uses only that attack objective, including clean input as a candidate
for gradient attacks. Scoring uses a separate bank. Random noise is scored as
sampled, without selecting it by loss. Held-out scoring loss is allowed to fall
below clean scoring loss; it is never clipped upward to manufacture attack
success.

The primary and convergence checks share the primary scoring bank. Two optional
additional scoring banks score the already selected perturbation without
influencing its selection. They record clean and adversarial losses separately.
Convergence uses a different attack bank, so a higher independent-scoring loss
is an observation to check, not a property assumed by the implementation.

Seed serialization is compact ASCII JSON of
`["evaluation-v1", base_seed, namespace, video_id, restart, draw]`, with
SHA-256's first eight bytes interpreted big-endian modulo `2**63`.
`base_seed` defaults to 20261020. Namespaces are method-specific attack/start
names, `routine_score`, `primary_score`, and `primary_score_audit_0/1`.
The fixed mask bank uses `restart=0`; one explicit start generator advances
through the requested restart sequence. Condition, training seed, and epoch
are absent so comparisons share random banks while re-optimizing every attack.

The pinned upstream sampler has a signed 32-bit counter. Mask seeds therefore
map the recorded 63-bit hash modulo `2**31 - 1`. Checks reject collisions between
attack/scoring/audit seeds within an evaluation. This compatibility mapping and
serialization are explicit implementation details, not a silent change to
training's separate seed rule. They still need inclusion in the eventual
frozen protocol. Model modes and Python/NumPy/PyTorch RNG states are restored,
and evaluation requires a gradient-free training boundary.

## Recovery and checkpoint selection

[evaluation_store.py](../models/evaluation_store.py) atomically commits the
ordered completed-clip records after each clip, with a checksum and a locked
contract. The contract binds actual context/predictor/target weight digests,
checkpoint metadata, ordered video/group IDs, input-cache identity, settings,
seed rule, sources, and runtime. Resuming can skip completed clips only when
that contract still matches. An interrupted active clip is recomputed. Changed
weights, panels, settings, source provenance, and corrupt records are rejected.

[evaluation_checkpoint.py](../models/evaluation_checkpoint.py) loads only an
anchor referenced by the training store's committed manifest. It checks the
file hash, training contract, original checkpoint identity, fixed target,
epoch/update alignment, and every tensor before changing context/predictor
weights. It never loads a new target from fine-tuned weights.

The selection function requires a complete primary-validation panel at every
declared anchor, including initialization. All evaluations must use the same
panel, settings, sources, runtime, and target, and the same training run for
nonzero epochs. It minimizes the mean independent-scoring loss and breaks
exact ties toward the earliest epoch. Routine PGD, training panels, test
panels, missing anchors, and mixed runs cannot select a checkpoint. The saved
loss-gap sign is `final - selected`. The output is a **provisional choice**;
it does not open test data, establish overfitting, or approve a run.

## Bounded command and verification

[evaluate.py](../scripts/evaluate.py) currently requires `--diagnostic` and
permits only one or two training-monitor clips. It supports routine, primary,
convergence, FGSM, random, and optional scoring-bank checks, with explicit
resume or saved-anchor loading. It does not yet run a full epoch schedule or
held-out study evaluation. For a future chosen device session:

```bash
.venv/bin/python scripts/evaluate.py --diagnostic \
  --checkpoint checkpoints/vitl.pt \
  --manifests data/manifests/ucf101-staged-20x5-v1 \
  --clips data/processed/staged-exploration-v1 \
  --method primary --device cuda --limit 1 \
  --output runs/pretrained/diagnostic/native-evaluation-01
```

This full-model command has **not** been run in this implementation step.
Add `--resume` with the identical arguments to recover an interrupted command.
For trained weights, add `--anchor-store <attempt>/checkpoints --anchor-epoch N`.
The bound is by clip count; there is no enforced wall-time cap. Completing or
interrupting Python does not stop cloud billing.

The full local suite passed **144 tests, 0 failures, 0 skips**, with 54 upstream
deprecation warnings. The 49 new tests exercise real small upstream transformers
and data fixtures: independent scoring, deterministic PGD replay, frozen targets,
RNG/mode preservation, interrupted evaluation, corrupt/changed contracts,
atomic-write failure, saved-anchor loading, and primary-selection safeguards.
See the [verification record](../report/checks/2026-10-20-evaluation-checks.json).

The cleared temporary Python environment was replaced with the repository's
ignored `.venv`, using the same pinned scientific package versions. The report
records the complete installed package set; unpinned transitive packages may
differ from the earlier session. The original GPU measurements remain unchanged.

**Update:** fixed probes, calibration records, and bounded epoch recovery are
now implemented and checked in [workflow diagnostics](workflow_diagnostics.md).
The latest suite has 197 passing tests; the 144-test evidence above records
the earlier milestone. Next, validate full-model CUDA recovery and measure
missing costs in a bounded pilot. Near-duplicate review, the novelty refresh, protocol freeze, and current
price/budget checks remain required before an exploratory research run. No
new cloud instance or paid compute was used for this step.
