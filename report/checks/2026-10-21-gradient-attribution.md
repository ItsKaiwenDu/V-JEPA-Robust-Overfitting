# Controlled derivative handoff — October 21

**Subsequent outcome:** the university run is now copied/verified. It completed
and rejected forward-oracle accuracy for both paths; all fixed-cotangent
backwards and exact saved-gradient reconstructions pass. See
[the verified outcome](2026-10-21-university-gradient-attribution.md). The
preparation-time record below remains historical.

Implemented a read-only derivative investigation after the verified pretrained
native/cuDNN input-gradient rejection. The old failure remains failed; backend
adoption, full-model optimizer recovery and study admission remain pending.
Pretrained CUDA attribution has **not run**.

The control captures each backend's actual forward patch tokens. Temporary
preserve-format leaves, one per mask family, provide incoming derivatives from
the unchanged transformer/predictor/native L1 loss. Both backward paths receive
the same family derivatives and share original normalization. CPU FP64 unfolded
patches supply independent trained-weight raw-pixel VJPs, including channel
standard deviation. The four VJPs quantify backward-path and incoming-derivative
effects; retained residuals expose sign/zero changes without asserting a cause.
An identical-token repeated tail must replay exactly within the worker.

Bind the original failed session and both replay-0 worker JSON/tensor hashes;
verify all 17 historical project/32 upstream source hashes, current data,
weights, runtime, targets and mask identities. Reconstruct the old input
gradients at unchanged bounds. Actual native CUDA observer evidence is saved
before rejection. No model/loss/attack or study YAML was changed for this
control. Original native code and rejected archives remain byte-identical.

**Local verification:** 345 passed, two actual-CUDA skips, 107 warnings in
119.55 seconds; 18 new tests. Meaningful checks cover direct-vs-chain-rule
upstream derivatives, independent FP64 normalization, hook/RNG restoration on
failure, unchanged parameter identity/modes/targets, residual sign accounting,
failed reconstruction, source/session/worker/tensor hash and runtime/mask/weight
mismatches, and bounded explicit CLI use. A fresh process from the extracted
source archive completes the tiny CPU control, retaining all nine artifacts,
in 0.767 worker seconds. This is a generated fixture, not pretrained/GPU evidence.
An initial unscoped pytest discovery included separate vendored upstream tests
and failed collection; the project suite is explicitly scoped to `tests/`.

**Handoff:** 51 checksummed files plus MANIFEST.json, 52 regular tar members,
119,896 bytes. SHA-256:
`1f932ba40b336f9101890b9fa82edb7b4bc8568bf87901ba9e3e21db6896bd51`.
The source-only archive and local raw evidence stay ignored by Git; curated
[hashes/limits](2026-10-21-gradient-attribution.json) are tracked for review.

**Next/manual action:** transfer the archive and launch the detached university
control using [the run guide](../../docs/native_gradient_control.md). Worker
limit 300 seconds; outer limit 330 seconds plus five-second kill grace; CUDA
allocator at most min(7 GiB, 90% visible VRAM). Partial/rejected controls cannot
pass. Existing training assets only; no optimizer, installation, download,
held-out pixels or paid instance. Reported spending remains $1.195, pilot
reserve $8.805. Review exported hashes, numerical checks and reconstruction
before any causal interpretation or next backend decision.
