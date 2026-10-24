# CPU fixture inputs pinned; local validation passes

October 23, 2026 (Pacific). **495 local tests pass; only the two actual-CUDA
tests skip.** The repaired source snapshot also passes three extracted checks.
A source-only university handoff is verified and ready for its **128 required
CPU checks on the existing PyTorch 2.8 environment**. That Linux execution has
not occurred. No paid compute or new GPU job was started.
[Structured evidence](2026-10-23-cpu-fixture-repair.json) and
[run instructions](../../docs/cpu_fixture_repair.md).

## Findings and repair

The preserved A40 setup evidence shows different initial model/target hashes
between generated CPU workers despite the seed. On the Mac, twelve unmodified
fresh constructions across one/two/four threads and presence/absence of the
native sanity probe all have identical state bytes. Those saved state tensors
and their per-tensor signatures verify. This does not reproduce the Intel
PyTorch 2.8 discrepancy or identify its low-level cause.

The fragile input assumption is removed: each tiny diagnostic parent captures
one generated initial checkpoint, and every fresh worker loads its explicit
SHA-256 before executing the comparison. The helper audits the bounded file,
format, logical digest, all keys, shapes, dtypes and finite values before
mutating any model tensor. Context, predictor, target and normalization buffers
are restored together. Parent summaries check the checkpoint declaration and
the actual initial model digest. Missing pins, altered checkpoints and altered
worker declarations reject; no random fallback is allowed.

Fresh CPU recovery tests now deliberately use three different construction
seeds, load the same captured state, and compare weights, complete AdamW state,
RNG/history and continuation draws exactly. The target-contract rejection
guard remains intact. Capture preserves caller RNG/thread settings on success
and failure. The original small upstream factory, pixels, masks, numerical
tolerances and pretrained loading branches are unchanged. Generated fixture
override arguments are refused for pretrained CUDA commands.

## Verification and provenance

The initial affected checks passed (87 tests), and the initial loader checks
passed (16 tests). Final coverage adds malformed-checkpoint and parent/worker
binding cases, with 497 full-suite cases: 495 pass, two actual-CUDA cases skip,
zero failures/errors, in 268.47 seconds. The fresh extracted snapshot passes
the actual four-worker gradient/attack pipeline, adversarial shared-native
fresh-process recovery and malformed dtype rejection: three tests in 13.43s.

One preliminary full-suite run had 493 passes, two skips and a budget-builder
test failure. Its production source guard correctly noticed the changed CPU
diagnostic file. The positive unit case now uses its own explicitly isolated
source fixture; an added negative case requires the real historical builder
to reject the changed working tree. Historical variant arithmetic still
reproduces from the original bound plan/timing records. No historical pin,
plan, report or production guard was rewritten to pass the check. Reusing that
historical builder for the new source still requires a separate review.

Byte comparisons preserve the attack/training/execution/precision code,
preflight/evaluation, study/proposal YAML and previous result records. AST
comparisons preserve the original factory, pretrained loading branches,
precision oracle/probe checks and numerical limits. Raw local controls, tests,
verification code and the first failed suite remain ignored under
`runs/pretrained/diagnostic/cpu-fixture-repair-20261023/`.

## University handoff and remaining gate

The archive is 5,297,533 bytes with 264 verified members: 262 source files,
the original upstream Git bundle and a manifest. SHA-256:
`2ef3e437002f2e8e1e4575b0bb6386aec16afc4c2bdf3033033b52f963e03c95`.
Every archive/member/source pin verifies after extraction. The manifest
SHA-256 is `bedcf6e4da90e696f7649d240f0f97b1307d7cc4864fc3a08e5b7fcc145e9ccf`.
Later curated status documentation may follow this frozen snapshot.

The launcher uses the existing university Python environment, exposes no GPU,
installs/downloads nothing, caps testing at 25 minutes and the complete job at
28 minutes, and checks source hashes before/after testing. All 128 selected
fixture, gradient/precision, optimizer recovery and budget-provenance checks
must run with zero skips. A future GPU setup still needs the full suite,
including actual-CUDA checks; this free precheck does not replace that gate.

Portable launcher checks verify shell/Python syntax, acceptance of the intact
snapshot, rejection of modified sources and of the unreviewed local torch
version, and checksummed partial-JUnit/timeout failure export. These are local
guard checks, not an end-to-end Linux execution. No video pixels or pretrained
weights are included. The original A40 failure and terminated Pod remain
separate historical evidence; no new A40 speed, memory, recovery or budget
admission is claimed.

Next user action: transfer the archive plus checksum sidecar and start the
detached university CPU check using the run guide. Copy/verify its result
before another rental. Latest displayed aggregate remains $3.930 with per-Pod
allocation unresolved; both $3 holds and the conservative $2.805 unreserved
pilot balance remain. No new charge is inferred from this local work.
