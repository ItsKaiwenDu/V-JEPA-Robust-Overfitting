# Forward-preserving v2: initial CUDA numerical gate passed; final replay incomplete

Three complete workers are independently verified on linux10603's RTX 3070 using
the pretrained full-size V-JEPA model, original training clip, eight mask draws and
both original families. The candidate's original-forward/public-math-VJP route
passes the unchanged initial full-model numerical gate. The parent attempt still
fails: a filesystem inventory refusal interrupts the final candidate replay.
No research or training adoption follows from this partial cohort.

| Worker | Actual result |
| --- | --- |
| Original-0, PID 41586 | Complete; original loss/input anchors exact |
| Candidate-0, PID 48079 | Complete; original predictions/native cotangents exact; all input bounds pass |
| Original-1, PID 54863 | Complete; exact fresh-process replay of all 36 original captures |
| Candidate-1, PID 61329 | Interrupted during forward; six arrays, no loss/input gradient or completion marker |

The candidate loss remains exactly 0.5617199540138245. All sixteen predictions,
clean targets, masks and native L1 output cotangents match the new original
bitwise. Every one of the 3,145,728 input-gradient entries meets the original
rtol 0.002 / atol 2e-7 rule: zero failed entries, maximum absolute difference
2.7939677238464355e-7 and maximum error/tolerance ratio 0.5058618267269619 (<1).
The existing accepted prior FP32 loss/input anchor is decoded from its hash-bound
saved ZIP with a restricted storage-only reader; Torch/model code is not imported.
Both originals match those existing anchors exactly.

Each complete worker has 576 actual native primary and 576 checkpoint SDPA calls,
covering every encoder/predictor block in all eight draws/two families. Candidate-0
has 192 distinct completed public-math inner GraphTasks and 384 predictor forward
calls including checkpoint recomputation, under one real native outer root.
Native/inner/root/leaf identities, actual flags, original Q/K/V layout/domain,
restoration and hook removal verify. No native recomputation evidence is fabricated
by math admission. Candidate Torch allocator peak is 5,897,503,744 bytes
(5.492478 GiB), within 7 GiB. These captured diagnostic wall times do not qualify
training speed, complete kernel coverage, attack/optimizer behavior or study cost.

The final replay is not admitted. Only targets, masks and four predictions exist;
those partial arrays match original values, but there is no completed loss/input
return, native/VJP proof or final marker. The cold reader correctly rejects it.
Do not reconstruct missing evidence from the initial candidate or declare exact
candidate replay from a deterministic-code expectation.

The parent records `Unexpected special file or external hardlink.` during worker
3's live filesystem inventory. The record omits the offending pathname/stat.
Last recorded cache use was 2,355 bytes; no cache file remains in the final
collection. A transient cache or atomic-writer race is a hypothesis, not a proven
cause; an actual unsafe link/file is not ruled out. Keep the guard strict until
bounded path/inode/mode/link-count evidence identifies what it rejected. The
supervisor's frozen finally path terminates and waits for its owned child before
collection; the empty descendant list does not mean that parent child was left
running. No new manual GPU check or retrieval is needed for this saved review.

The 2.5 GiB proof passed. Total parent work is 477.362277 seconds. The complete
same-SSH response is 704,661,621 bytes, containing 280 files / 704,377,869 logical
bytes. Every member, source/archive/control/approval hash, private mode and original
nanosecond mtime verifies. Independent NumPy saved review verifies 114 actual arrays:
108 from complete workers and six partial arrays. It builds no model, autograd,
SDPA, profile or new derivative reference and performs no process/network/extraction
work. All 2,075 armed/preparation pins remain unchanged. No source/threshold or
historical outcome is rewritten. The distinct old 3,724-entry candidate remains
rejected; this is a different derivative with a new initial numerical pass.

The v2 claim and grant are consumed; never repeat its helper or reset its clock.
Preserve all three accepted workers and the partial fourth. Next local work is to
diagnose the inventory refusal with bounded stat/path evidence, then prepare a
separate continuation for only the missing candidate replay with exact immutable
original-1/candidate-0 reference lineage. A complete cohort rerun is unnecessary.
Any additional remote scope needs its own concrete review and authorization;
none is created here. Financial caps/holds/reconciliation and all broader
research/full-training gates stay unchanged. No new SSH, retrieval, cleanup,
installation, model execution or paid compute was performed during review.
