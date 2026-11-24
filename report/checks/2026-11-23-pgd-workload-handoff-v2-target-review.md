# Primary whole-PGD v2 attended attempt: saved evidence review

The approved attempt is **consumed and failed**. The actual full 2.5 GiB storage
proof passed. Original-unit model preparation then exceeded the existing 4 GiB
observed host-RSS limit while copying checkpoint weights. No whole-PGD unit
completed; candidate-unit and candidate-replay never started. Same-connection
collection completed and the attended command returned SSH exit status 1.

The standalone stdlib saved review **passed on its first execution**. This is a
successful review of a failed workload, without promoting its completion status.
The machine-readable review is [retained here](2026-11-23-pgd-workload-handoff-v2-target-review.json).

## Exact attempt and preservation

- Package, unchanged:
  `runs/pretrained/diagnostic/pgd-workload-handoff-v2-20261123/handoff-parent/linux10614-primary-v2-final-unarmed`.
- Outer control: `02d380a26ce80ed39f09c1bb6a1725065e37eaf4d16bdd5ba2dff4a212e03dfe`.
- Supervised control: `bc53eda6e44e8cfae82bf58e2b567461ede9c5a395d742d00f20944d6005df1c`.
- Existing authorization: 1728 bytes,
  `68757c11f5db48d0dbf630b0754bd20447bbf2c0727e7b4447787dc5e1f4d924`.
- User reported “Done.”; the bound local ATTEMPT/session/error records were
  inspected first. No additional connection, retrieval, retry, remote cleanup,
  installation, GPU query, proof, model calculation or manual paste was performed.
- All five previous package files retain their hashes and recorded identities.
  Directory device 16777233/inode 44930785/owner/mode remain fixed. Only the
  authorized ATTEMPT/session lifecycle extends tracked APFS nlink 7 to 9 and
  mtime; unarmed nlink was 6. The parent directory remains unchanged. Earlier
  component/candidate/v1 directory exceptions remain separate.
- The audit verified all 46,508 prior sealed hashes, 51,204 prior fixture entries,
  and 208 new attempt/session fixture entries. The historical FIFO was LSTAT
  only, never opened. Device 16777233 is the previously reviewed conditional
  local observation; no original device record or target mapping was rewritten.
- Prior uncommitted binding/scope/arming work, all source/test/auditor versions,
  old failures and unknown v1 remote progress remain preserved. User alone
  stages, commits and pushes.

## Observed execution

| Stage | Saved outcome |
| --- | --- |
| Workload claim and derived authorization | Correct workload prefix and whole-PGD authorization kind accepted; both claim copies retained |
| Full allocation proof | PASS: 2,684,354,560 bytes written, fsynced and read back; only its own proof file removed |
| Proof wall/allocation observations | 13.803068832 s; peak observed allocated 2,746,851,328 bytes; 1,926 events |
| Proof parent observation | Exit 0; 14.215986698 s; 105 RSS samples; peak observed RSS 43,171,840 bytes |
| Original-unit preparation | Parent MemoryError: observed owned-tree RSS exceeded 4,294,967,296 bytes |
| Original PGD workload | Not reached; no start/context/result files or parent success observation |
| Candidate-unit and candidate-replay | Not started after original failure |
| Whole workload | Session failed/complete FALSE, completed roles only `[proof]`; SSH exit 1 |
| Collection | 173 files, independently reconstructed bytes match recorded transport hash |

Proof result, event sequence, proof parent PID/start ticks, source identities,
session receipt, work receipt and collection receipt join. This review did not
recreate the removed random proof data or recompute its 2.5 GiB content digest.
Sampled storage/RSS observations do not establish an instantaneous quota,
exhaustive process enumeration, forced timeout coverage or a reservation against
other writers. Existing FALSE proof-authentication/qualification fields remain
unchanged; dictionaries alone do not authenticate execution.

The original-unit traceback reached `run_prepared` line 26, `load_checkpoint`,
and `load_components` line 209. PyTorch reported an interruption while copying
`blocks.21.mlp.fc2.weight`; the model and checkpoint dimensions are both
`[1024, 4096]`. The supervisor's primary error is the RSS guard at
`workload_watch_v1.py` line 167. Its cleanup sends SIGTERM to owned leases, and
the worker's signal handler uses the interruption text in this traceback. That
is consistent with termination after the resource refusal, rather than evidence
of a checkpoint shape mismatch or a VPN disconnect.

This target performed real Torch imports, CUDA context initialization, CPU model
construction and checkpoint deserialization/copying. Preparation failed before
the subsequent `model.to('cuda')` and before `TargetUnit.run`. Five saved GPU
journal rows include the original child PID 1681864 using 154 MiB on the final
sample; this does not establish full-model CUDA residency, GPU OOM, or workload
memory fit. The exact RSS value at refusal and original-unit peak/sample count
were not saved, so only the threshold breach is established.

## Sources, clocks and collection

All 31 input pins, 16 local source pins, 55 uploaded sources/895,657 bytes and
47 frozen runtime files/450,166 bytes join. All 99 members of the unchanged
17,192,095-byte original packet join their saved code/input copies (26,446,958
expanded bytes). Checkpoint tensor values and packet tensor payloads were not
decoded or recalculated locally.

Connection start 1061869.480013588 continues to work deadline
1064269.480013588 and connection deadline 1064869.480013588. Original launch
1061892.296402190 retains absolute deadline 1062612.296402190; collection starts
1061905.424440913 and retains deadline 1062505.424440913. No clock was reset.
These records do not qualify all three 720-second units, performance, timeout
enforcement, or later clock exhaustion behavior.

The independent stdlib audit reconstructed the priority header, ordered file
frames, data/reference payloads, prefix and terminal; checked every saved mode,
mtime and hash; and verified all 173 journal rows, offsets and hash links.

- Expanded bytes: 45,300,704; payload: 44,989,897; deduplicated: 310,807.
- Reconstructed wire bytes: 45,081,754.
- Recorded and independently reconstructed wire SHA256:
  `7894a738c7d08442d4bb94ad890e87ed53da967b328e798b2888445cdc964378`.
- Journal metadata plus progress: 182,381 bytes. Local sampled allocation:
  48,582,656 bytes, without a quota claim.

The raw wire/terminal was not separately saved, so this is reconstruction matched
to the client's saved full-wire digest, not a second direct scan of the original
network stream. Session complete FALSE, work collection FALSE, partial receipt
FALSE, per-file journal FALSE and original independent-review-pending flags stay
unchanged. Separate transport verification is TRUE; no failed record was promoted.

## Next local step and limits

Review a separate memory-aware preparation design under the same 4 GiB host-RSS
and 7 GiB allocator caps. The source constructs CPU destination parameters before
copying the checkpoint and moves the model to CUDA afterward. Coexistence of
destination storage and touched checkpoint pages is a plausible contributor,
not a measured allocation breakdown. `torch.load(..., mmap=True)` is already
enabled. No loader change or alternate allocation order was adopted in this review.
See the [preparation analysis](../../docs/predictor_pgd_workload_preparation_rss_review_v1.md).

Do not repeat the attended command, request another scope, raise or disable the
resource guard, or reopen closed mechanism/native/PGD/clean/tiny-gradient cohorts.
Derive and test any new preparation path locally before considering another exact
scope. Real primary/convergence/replay agreement and complete-unit performance
remain unqualified, followed by hardware recovery/economics/calibration/data and
protocol adoption gates. Research/training admission stays FALSE. No defensible
percentage or time-to-full-training has been established. Training PGD5/EOT1,
routine EOT1, optimizer backward, policy registry, study YAML, financial caps and
holds remain unchanged; old financial snapshots are historical.

Raw review:
`runs/pretrained/diagnostic/pgd-workload-handoff-v2-20261123/target-review-20261123`.
Current seal: `sealed-workload-v2-target-review-v1.json` in that directory.
The standalone audit invokes no project/test/science callbacks, process/timer/
signal/GPU/proof/network work, array decoder or numerical comparison. Mutable
index before/after snapshots are retained and live indexes excluded from the seal.
