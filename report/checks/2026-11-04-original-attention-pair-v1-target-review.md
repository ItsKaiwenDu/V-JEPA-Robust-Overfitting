# Original efficient/math GPU pair independently verified

**Both original encoder-1 GPU workers pass**, with complete typed captures and
all original numerical, actual-backend, runtime/flag and within-process replay
gates unchanged. The linux10603 attempt is consumed; never repeat its launcher
or reset its claim. Both university hosts can remain idle.

| Evidence | Verified result |
| --- | --- |
| Workers | Two fresh serial CUDA processes: efficient replay 0, then math replay 0 |
| CPU checks | Eight original FP64 checks pass at `rtol=1e-12, atol=1e-14` |
| GPU returns | All six FP32 phases per worker saved before checks; 12 captures total |
| GPU component checks | 48 saved-array comparisons pass at the original component bounds |
| Replay identity | 48 bitwise comparisons match each arm's original untimed snapshot and phase signatures |
| Fresh reference | All four math/efficient comparisons pass; original reference hashes/common contract match |
| Real backend | CUDA forward/backward kernels observed for both efficient and math |
| Transport | All 199 files / 374,125,075 logical bytes and exact modification times verify |
| Whole work | 36.780 seconds, including setup and the full 2.5 GiB proof |
| Allocation proof | Full write/fsync/readback passes in 9.614 seconds; 1,926 retained events |
| Observed storage peaks | Setup 2,702,053,376 bytes; capture 391,421,952 bytes, within 3 GiB/2.25 GiB |
| Final raw category | 196,605,878 bytes, within the explicitly approved 320 MiB |

Both actual 13,108,174-byte CPU-return files match the earlier accepted
post-CUDA-setup capture byte for byte. All eight corresponding arrays match it
bitwise. Current operands and recorded common contracts also match the historical
passing efficient worker and failing math worker. The old failure remains
unexplained; its missing actual computed values are not reconstructed or waived.

The efficient trace contains the expected efficient SDPA operators and CUDA
forward/backward attention kernels. The math trace contains math SDPA,
softmax-backward operators, GEMM and softmax-backward CUDA kernels. All profiler
rows are retained. The math worker references only this attempt's complete
passing efficient worker, with two distinct process IDs and serial monitor times.
Each worker's six actual FP32 returns are bitwise equal within that arm; the two
arms are compared under the original tolerances rather than assumed identical.

Independent review authenticates the 374,315,134-byte response, original 139-file
packet, 13 overlays, 1,148 prior bindings and all 2,028 preparation inputs, sealing
2,236 inputs. Process/network creation is denied after reader imports; no oracle,
SDPA, model, SSH or remote cleanup is started by review. Sources, original gates,
authorization, claim and captured files remain unchanged. The existing 163-check
operator preparation remains the test baseline; this evidence-only step adds no
unit tests or scientific rerun.

The first offline reader incorrectly required the CPU-exported V tensor's strides
to equal its live CUDA layout. That extra assertion was not an original gate.
The frozen exporter calls `detach().cpu()`: the saved V export is packed at
`[409600, 64, 1024, 1]`, while the original live layout record correctly remains
`[1228800, 64, 3072, 1]`. All exported logical values match the fixed inputs.
The corrected reader checks live layout and exported bytes separately, preserving
the first audit source/log and recording the correction. No tolerance or original
layout check is relaxed, and no model is restarted.

Control: `dabbd82a5d31c480d480fe314179683c9364984abde012f984656223231ffa3c`.
Consumed claim: `f7d663471c3e2afd4d22ced57fca02f6c2210cb08d0a7c9e234355d2dd213a4e`.
Raw independent review: `349e1d1225b92afaf8c75702c0bd03b607e55c026b1618cd9cf9e0fd5f45ecd8`.
Remote scratch remains intact at
`/var/tmp/vjepa-original-attention-pair-v1-20261104-linux10603-t40vx4zz`.

This is the first pair of the original four-worker session. **Replay-1,
full-model, performance/cost, optimization adoption and research qualification
remain pending.** Instrumented copies and I/O affect profiler/timing/cache state;
these timings cannot update the study budget or support speed claims. Five-second
sampling, nine persisted GPU observations and observed storage accounting do not
prove continuous exclusivity, an aggregate quota or a future reservation.

Next prepare an offline proposal for the remaining original fresh-process
replay-1 coverage. No next operator, grant, claim, upload, parallel job or GPU run
is prepared or authorized by this review. All previous grants remain consumed.
Financial caps/holds are unchanged: $200 project / $160 compute / single $10
diagnostic envelope; $9.8116 conservative encumbrance, $0.1884 unreserved, old $2
hold and $0.0792 gap retained, last wallet $6.56, reconciliation incomplete.
All known Runpod Pods remain terminated.
