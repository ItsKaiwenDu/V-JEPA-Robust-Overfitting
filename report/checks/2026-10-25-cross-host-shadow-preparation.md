# Candidate-host reference replay prepared

The separate linux10603 shadow packet is ready; no upload or GPU work has
started. It runs two fresh PGD20/EOT4 arm-A replays against the fixed accepted
Swing bank-1 reference. The expected attack alone was about 197 seconds on
linux10609, so roughly ten minutes overall is an estimate. The full remote
pipeline is bounded by twenty minutes plus fifteen seconds of kill grace.
Each worker has at most 510 seconds inside that allowance, with no GPU retry.

All 67 focused CPU guard/comparator tests pass locally and from the extracted
packet, with zero failures/errors/skips. Eight ordinary byte-transport fixtures
pass. Four CPU shell-wrapper checks verify actual success/failure/interruption
exit bytes and refusal to overwrite a prior launch. These are implementation
checks, not new CUDA or model pass evidence. Genuine reference read-back and
rejection of relabeling it as candidate execution also pass locally.

The nine-member source/metadata archive is 40,062 bytes,
SHA-256 `15e4caffa9e3d33cf523b94657522c5aded352f1c4cddcf648a848af84c4198e`.
Shadow plan payload SHA-256:
`66f8595088397e1ca0b56fd69c5bc9b82842feec9c3fe39c6f83a369ec40d1c4`.
The 198 original scientific pins and 58 qualified execution sources match;
the new module, CLI, CPU tests and genuine metadata fixture are separately
identified. The public host key matches the fingerprint supplied from the
candidate's SSH terminal. No private SSH keys or neural tensors are uploaded.

The candidate records its actual linux10603 UUID under a distinct shadow
execution identity. Exact retained targets, starts, masks, adversary, PGD
gradient traces and independent scores must match the accepted reference.
Native CUDA evidence, runtime/model/checkpoint identities and allocator limits
must verify. A fixed operation root and atomic worker claims prevent a copied
packet or repeated worker invocation from resetting this attempt.

The original linux10609 job is untouched. Its seventh case has reported
launcher exit zero; its download has not yet recorded completion at this
preparation checkpoint. Six cases remain independently accepted until that
export is downloaded and reviewed. The shadow writes no panel journal or
receipt and admits no additional panel case. Its successful execution would
still require independent export review and would qualify only this one
reference arm, not all clips/arms or parallel scheduling.

Next: run the prepared Mac run-shadow-replay.sh once in a separate Terminal,
keep the Mac and both attended sessions open, then independently review the
download. Copy failure may retry only copying once; no model restart.
No precision adoption, partial-panel mean, budget change, research admission
or paid compute follows from this preparation.

[Guide](../../docs/cross_host_replay.md) ·
[Machine-readable preparation](2026-10-25-cross-host-shadow-preparation.json).
