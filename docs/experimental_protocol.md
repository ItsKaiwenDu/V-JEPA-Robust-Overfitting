# Experimental protocol

Version 0.1, drafted October 16, 2026. **Proposed; not preregistered or validated.**
Numerical defaults live in [configs/study.yaml](../configs/study.yaml).

## Question and scope

Does adversarial fine-tuning of V-JEPA 2's masked latent prediction objective
show robust overfitting under a fixed target encoder? Does any change in
prediction robustness also affect downstream action recognition?

Use the released V-JEPA 2 ViT-L/16 encoder and predictor. Load the context,
target, and predictor states from their actual checkpoint keys and audit all
weight shapes. Missing or reinitialized trained layers are a failed loading
check, not an acceptable silent fallback. Record the upstream commit and
checkpoint SHA-256 before results are interpreted.

The pretrained target encoder remains fixed for the entire study. Updating
the context encoder and predictor while freezing targets is a deliberate
fine-tuning variant. Findings do not establish the behavior of original EMA
pretraining, V-JEPA 2.1, causal future prediction, or robot planning.

## Data and preprocessing

Use [UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/) official split 1.
First hold out source-video groups from the official training partition for
validation. Then sample 20 training and 5 validation videos per class, and
5 test videos per class from the official test partition. No source group
may cross partitions. Unselected videos in a validation group cannot return
to training. Use manifest seed 20261016 and share the resulting manifests
across all training seeds.

Create one deterministic 16-frame, 256 x 256 clip per video and cache its clean
target features. Exact frame timestamps, short-video handling, resizing,
normalization, decoder version, and any corruption exclusions must be
specified and checked during the pilot. Freeze this preprocessing before
confirmatory runs. The initial controlled experiment does not use stochastic
image augmentation; masks remain stochastic.

Labels are used for balanced sampling and probes, never for the native
fine-tuning objective. A held-out adaptation split is not a guarantee that
the foundation model never encountered related content during pretraining.

## Native loss and threat model

For clip x and mask M, the predictor uses visible context from x + delta and
matches clean, normalized target features from x. Follow upstream layer
normalization, token indexing, mask distributions, and loss averaging.

```text
target = stop_gradient(frozen_target_encoder(clean_clip))
prediction = predictor(context_encoder(adversarial_clip, mask), mask)
loss = native_masked_L1(prediction, normalized_clean_target)
```

The adversary maximizes this loss under a raw-pixel L-infinity constraint:
every pixel stays in [0, 1], and abs(delta) <= epsilon / 255. Normalization
belongs inside the differentiable model path. Only pixels passed to the
visible context influence the masked prediction; the target branch always
receives the untouched clip.

The training attack uses random-start PGD-5, epsilon 4/255, step size 1/255,
and one restart. Keep the mask draw fixed inside each attack and its outer
update, then resample for the next batch. Compute input gradients without
updating model weights in the attack loop; perform the parameter update
afterward. Do not backpropagate through the sequence of attack updates.

## Training conditions

| Condition | Trainable components | Training data presented to context branch |
| --- | --- | --- |
| `pretrained` | None | No fine-tuning; common initialization reference |
| `clean-finetune` | Context encoder and predictor | Clean clips |
| `adversarial-finetune` | Context encoder and predictor | Native-loss PGD clips |

Run each fine-tuning condition with seeds 0, 1, and 2. Use 30 epochs and
effective batch size 8. AdamW uses encoder LR 1e-5, predictor LR 1e-4, and
weight decay 0.04. Warm up for one epoch, then reduce both learning rates by
0.1 after epochs 15 and 23. The epoch following each boundary uses the new
rate. Save epoch weights and a resumable state; use matched update counts
and examples for both fine-tuning conditions.

The pilot must resolve optimizer defaults, mixed-precision dtype,
microbatch/accumulation, gradient clipping policy, and checkpoint storage
cost. These are configuration decisions to record before the full study,
not post-result tuning opportunities.

## Evaluation

Track clean and adversarial native losses every epoch on fixed training and
validation subsets. Generate fresh attacks against every evaluated
checkpoint. Routine subset sizes and any cheaper monitoring attack must be
defined separately from the primary evaluation during the budget pilot.

The primary attack uses PGD-20, two random restarts, and expectation over
four masks. Score perturbations against an independent, reproducible mask
bank. Specify attack and scoring seeds, scoring-bank size, PGD step size,
and per-example candidate selection before confirmatory evaluation.
Do not infer robustness from failure to transfer between one attack mask
and a different scoring mask.

Check PGD-50 and eight-mask expectation on a fixed validation subset. Run an
epsilon sweep of 0, 1, 2, 4, and 8/255. Compare random perturbations, FGSM,
native-loss PGD, and random-start embedding-distance PGD. Use stronger
attacks when the convergence checks reveal an underestimated loss; account
for their cost before proceeding.

Fit the same clean-training-only linear-probe procedure separately for each
selected representation checkpoint, so a stale classification head is not
confused with representation damage. Freeze probe hyperparameters using
validation data. Measure clean accuracy, native-attack transfer, and direct
classifier attacks. Native-loss attacks alone do not establish worst-case
classification robustness.

## Checkpoint selection and interpretation

Select the minimum adversarial native validation-loss checkpoint within each
run; use the earliest epoch for an exact tie. Compare it with initialization
and the final epoch. Test data is opened only after the protocol and choices
are locked. Treat test results as a final estimate, not a tuning signal.

Evidence of robust overfitting requires initial held-out robustness
improvement followed by deterioration while robust training loss keeps
improving, with a consistent pattern across seeds and stronger attacks.
Compare against clean fine-tuning and inspect feature variance and downstream
accuracy to separate the effect from collapse, forgetting, or attack failure.
Report per-seed values and effect sizes. Use source-video groups for
resampling uncertainty; three seeds still limit generalization.

The loss gap has sign `final_loss - selected_best_loss`; larger positive
values mean worse final loss. For accuracy, use `selected_accuracy -
final_accuracy` in percentage points. Never select a different best epoch
from the test curve after seeing results.

Valid outcomes are: evidence of overfitting; no effect observed within the
tested regime; or an inconclusive study. Thirty epochs without an effect is
not evidence of immunity to robust overfitting.

## Feasibility gates

1. Verify data grouping, full checkpoint loading, native-loss alignment,
   nonzero finite gradients, projection, clean targets, and deterministic
   replay of an evaluation batch.
2. Measure full training and evaluation throughput, including data loading,
   checkpoint I/O, and restart/EOT costs.
3. Admit six 30-epoch fine-tuning runs plus required evaluation only if the
   remaining work fits within the $160 compute allowance and $200 total.
4. If needed, reduce training to 10 videos per class before confirmatory
   runs, preserving validation/test sizes and the three seeds. Document and
   freeze the change. If this still fails, stop at feasibility and revise
   scope instead of weakening the validity checks.
5. Recheck literature before the paid experiment and final report. If the
   matching native-loss/fine-tuning/overfitting study is found, halt and
   present the overlap to the researcher.
