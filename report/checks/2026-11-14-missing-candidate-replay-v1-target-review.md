# Missing candidate replay: independent target review

**PASS. The single-case numerical and fresh-process replay chain is complete.**
The new candidate index 3/replay 1 completed on linux10603 as PID 1352816 under
control `af2c495cb86ab78b48e20a700358f6964182f015939fa18c5f7566aa5cdf0842`.
It exactly reproduces all **36 saved candidate-0 returns**, including loss, full
input gradient, all 16 predictions and native L1 cotangents, targets and masks.
The three prior complete workers were reused without being reexecuted.

All **3,145,728 input-gradient entries** pass the original `rtol=.002, atol=2e-7`
bounds against the complete same-replay original and the independently decoded
saved anchors: **zero failures**, maximum absolute difference
`2.7939677238464355e-7`, maximum tolerance ratio `0.5058618267269619`.
Loss is exactly `0.5617199540138245`. Predictions, targets, masks and all native L1
cotangents are bitwise identical to the original. Native masked-L1 signs and loss
were also reconstructed from saved arrays with NumPy; no model or new gradient
reference was run during review.

The actual proofs verify 1,152 native calls: 576 primary and 576 checkpoint calls
across 24 encoder and 12 predictor owners. Candidate evidence contains 192 distinct
public-math inner GraphTasks / 384 predictor calls including checkpoints. Actual
native outer-root ownership, nested root/leaf hooks, original forward flags,
input layouts, hook removal and restoration pass the unchanged cold verifier.
This remains an explicitly different public-math derivative, without a claim of
original kernel-backward identity or kernel/performance qualification.

The new parent completed successfully in **161.828 seconds**. The scientific
objective took 109.668 seconds; supervised worker time was 129.805 seconds. Peak
GPU allocator was **5,897,503,744 bytes (5.492 GiB)**, below 7 GiB. The actual
2.5 GiB preallocation/write/fsync/readback proof passed in 9.745 seconds and removed
only its verified proof file. Proof peak allocated scratch was 2,702,159,872 bytes
under 3 GiB; worker peak scratch was 600,276,992 bytes under 2.25 GiB. The worker
exited 0 naturally, with no retry or forced descendant termination.

The same-SSH response is **599,851,959 bytes**, containing **242 regular files /
599,605,882 logical bytes**. Archive framing and every saved file's hash, mode and
nanosecond modification time independently verify. All **2,926 armed source and
evidence pins** remain unchanged. The original control and consumed authorization
remain historical reference lineage; the new grant and claim are now consumed.
The old parent's failure and partial candidate-1 are preserved, without promotion
or retrospective relabeling. The successful new one-worker parent is separate.

The file-only transport preserves file modes but omits directory metadata. Some
intermediate Mac mirror directories consequently have mode 0755, so the strict
reference reader initially refused that local mirror. The raw download and refusal
remain untouched. A new private local reference view contains only 55 verified
single-link regular copies with original bytes and timestamps; the unmodified cold
reader accepts that view. This representation issue does not alter the successful
target computation, its file receipts or numerical conclusion. A future transport
revision can address directory reconstruction locally without rerunning science.

Review used only saved NumPy/standard-library values and a separate cold receipt
check with scientific imports blocked. No new model, autograd, SDPA, oracle,
profile, optimizer, test suite, GPU proof, SSH, upload, installation, remote cleanup
or paid work occurred. The old 477.362277 seconds and consumed claims remain charged.

This closes the one-training-clip clean EOT8 numerical/replay gate for the new
forward-preserving candidate. The old 3,724-entry candidate remains rejected.
Broader integration, performance, optimizer/attack/study and full-training gates
remain open. Financial caps, holds and pending reconciliation are unchanged.
Next is a local decision on those remaining gates; this consumed grant authorizes
no further experiment or replay.
