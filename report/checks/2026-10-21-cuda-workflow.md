# CUDA workflow pilot: replay failure and deterministic runtime cost

October 21, 2026. **Diagnostic outcome; full-model CUDA recovery remains
unvalidated and the research study is not admitted.**
[Machine-readable evidence](2026-10-21-cuda-workflow.json).

## Observed outcomes

| Check | Outcome |
| --- | --- |
| Offline source/data restoration and released checkpoint hashes | Passed |
| Original suite on the RTX 5090 host | 219 passed, 0 failed/skipped, 81 warnings |
| Suite after explicit deterministic runtime policy | 225 passed, 0 failed/skipped, 82 warnings |
| Small CUDA gradients/updates across independent processes | Exact replay passed |
| Original full-model clean split/resume versus uninterrupted run | Exact comparison failed; execution stopped |
| Strict deterministic full-model retry | One calibration candidate completed; stopped during the second to preserve the session budget |
| Full-model adversarial recovery and trained features | Not reached |
| PGD-50 convergence and FGSM/random controls | Not reached |
| Exports | Both archives and all 42/15 members verified locally |

The setup used one RTX 5090 (32 GB), Python 3.12.3, PyTorch 2.8.0+cu128,
and torchvision 0.23.0+cu128. The 30 GB container and 80 GB `/workspace`
volume matched the selected configuration. Setup took 338 seconds, including
197.01 seconds for the original test suite and checkpoint download/verification.
No confirmation/test pixels were opened. Execution used eight training clips
and one training monitor; the package contained 100 training cache files.

## Why the first run stopped

The first attempt took 862.695 seconds. Calibration, initial features, and both
clean two-update trajectories completed. The comparison covered 4,307 logical
leaves and failed exact equality. Contracts, RNG, step, precision, and the fixed
target matched. Encoder/predictor weights, optimizer moments, and the second
optimizer loss differed.

Attacks also differed at the identical **initial** checkpoint, before any
restoration. The primary independent-mask loss was 0.6744728684 in the reference
and 0.6691417694 in the split run. Both had identical seeds, clean loss, weights,
and settings. This supports CUDA kernel nondeterminism as a cause; the pilot
does not isolate a particular operation or rule out every other defect.

The initial runtime recorded deterministic algorithms and cuDNN determinism as
disabled. The runner now enables strict deterministic algorithms, cuDNN
determinism, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization in
parent and worker processes. Nondeterministic operations raise errors. Runtime
contracts bind warning mode and the cuBLAS workspace. Precision flags are
unchanged: tensors are FP32, matmul TF32 is disabled, and cuDNN TF32 remains
allowed as in the original session. These flags remain explicit provenance,
not a claim that every operation uses full IEEE FP32 arithmetic.

The six new regressions passed on the Pod, including independent-process CUDA
convolution/gradient/update replay and rejection of incompatible resume policy.
The [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.8/notes/randomness.html)
explains why seeds alone do not ensure deterministic CUDA execution. A passing
small-kernel test did not predict full-model cost or establish full-model replay.

## Why the retry stopped

The retry used a fresh directory and calibration. Strict execution completed
one primary PGD-20/R2/EOT4 candidate, at step 0.5/255, in **1,261.208 seconds
(21.02 minutes)**. The corresponding original candidate took **78.769 seconds**.
The checkpoint, clip, attack settings, and seeds match. The observed ratio is
**16.011×** for this candidate.

The second candidate was unfinished when a controlled SIGTERM ended the retry
after 2,444.721 seconds. The parent terminated its worker and the wrapper
exported the partial evidence successfully. This was a budget decision, not a
completed calibration, failed deterministic operation, or successful recovery.
No new step-size choice is available from the incomplete retry. The original
one-clip choice of 2/255 is historical diagnostic evidence and cannot substitute
for the eventual ten-clip study calibration.

The specific slow kernel has **not** been established. Profiling must distinguish
convolution, attention, mask selection, and their backward paths rather than
assuming a cause. Disabling the guard or loosening exact equality would not
resolve the missing validation.

## Conditional measurements from the original runtime

These observations do not admit the study: full-model replay failed, and the
strict runtime is much more expensive in the measured attack.

| Operation | Samples | Observed seconds |
| --- | ---: | ---: |
| Clean batch-eight optimizer update | 4 | 4.067–4.180; median 4.106 |
| Cache fixed targets for eight clips | 3 | 0.829–0.839 |
| CPU copy of resumable engine state | 6 | 1.901–2.967 |
| Instrumented checkpoint save | 4 | 6.100–8.367 |
| Instrumented checkpoint load | 3 | 1.791–5.364 |
| Engine restore | 3 | 2.315–3.258 |
| Extract eight clean features | 1 | 5.851 |
| Fit fixed ridge probe | 1 | 0.002546 |
| Score training resubstitution | 1 | 0.001071 |

Nested spans overlap and must not be summed. The four updates are clean
fine-tuning samples across two trajectories, not adversarial update timings.
Checkpoint I/O and eight-example probes are small-panel measurements.

Peak observed CUDA allocator use was 8,091,180,032 bytes (about 7.54 GiB) in the
clean reference phase; allocator peak is not total device memory. Initial
features had centered rank 7 (sample bound 7), raw mean population variance
0.661727, and entropy effective rank 4.03824. The probe fit/scored the same eight
training examples and obtained 8/8; this is not held-out accuracy or a collapse
finding.

## Provenance, evidence, and spending

Execution began from committed milestone `4576aec`. The initial offline package
and failed run were preserved. The repaired run used a separate checksummed
four-file overlay. Source, checkpoint, upstream, archive, repair, and test-XML
hashes are recorded in the JSON report. Raw logs, tensors, archives, and the
session handoff remain ignored under
`runs/pretrained/diagnostic/workflow-session-20261021/`.

The first evidence archive SHA-256 is
`a40905ed36cbb6b2ef1de82d79331fe7b4ebdc7576e4a5ae538b8d54ebeebadf`.
The retry evidence archive SHA-256 is
`88010a361bbfee4696dbafb4fe9704fc692f5613529f14823b50f2aa972b1437`.
Both complete archive/member checks passed on the Mac. Model/optimizer tensors
are excluded. After local evidence verification, the user confirmed Pod
termination; the remote diagnostic checkpoints and volume were discarded.

The user corrected the session charge to **$0.603** after the console finished
updating, superseding the initial $0.501 report. Together with the prior $0.592,
user-reported spending is **$1.195**. No billing API or invoice was checked.
The user-reported ledger is reconciled to this correction, and the session
reservation is released; **$8.805** remains in the $10 pilot reserve. This does
not admit research spending. The user confirmed termination after the verified
exports, ending this Pod's retained volume; provider state was not queried.

## Next gate

Prepare a short, representative CUDA forward/backward profile before another
rental; the allocated university GPU can potentially run isolated operators
without full AdamW state. Validate any proposed equivalent implementation's
outputs and input/parameter gradients against the reference before crediting
speed or changing execution. Then repeat bounded full-model recovery and the
unreached diagnostics only if measured timings fit the remaining budget.

Near-duplicate review, novelty refresh, ten-clip study calibration, protocol
freeze, and complete measured budget admission still precede research runs.
No robust-overfitting result was produced by this session.
