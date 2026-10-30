# Interrupted CPU capture backup and saved-value review

Both backups are independently verified against their archive members and prior
status/trace hashes. All 145 existing files survive locally: 141 were bound to
previously recorded hashes; two unfinished fragments and two empty export files
are retained but remain unqualified. Neither fragment is parsed as a completed
calculation. No remote file was changed or removed during copying or review.

The 10603 archive has 69 members, 133,181,440 bytes, SHA256
`c698ed7e4615192535b3e168752a1677959b66e1683c675b0827ad53346eeee7`.
The 10609 archive has 76 members, 165,427,200 bytes, SHA256
`618bdf745fdf6f9175e5f63102a8ab8c4139d8003c979bfd745f6b309a87cd43`.
Raw archives, extracted bytes, full audit script and comparison reports remain
under `runs/pretrained/diagnostic/cpu-oracle-salvage-20261030/`.

All 996 complete logical tensor signatures verify. Independent NumPy calculations
check 976 intermediate arrays (58 saved chunks on 10603, 64 on 10609), and 12
saved final outputs/derivatives agree with the original fixed FP64 references.
All remain within unchanged 1e-12 relative / 1e-14 absolute bounds; largest
observed discrepancy is 2.4981e-16, maximum tolerance ratio 0.007507. A further
476 actual-byte comparisons pass across shared chunks/conversions/before outputs
and 10609's before/observed calls. Neither fragment is included in those counts.

10609's complete observed trace passes the unchanged trace reader. The same
reader still rejects 10603's incomplete trace; a separately labeled prefix
calculation audits its available bytes without waiving that gate. The unchanged
full-probe summarizer rejects BOTH sessions for incomplete coverage. No subsequent
one-thread/replay worker or complete third call is invented. These saved portions
do not reproduce the previous discrepancy and do not resolve it, qualify a host,
waive historical failures or admit GPU/backend/research work.

The current storage incident is separate: 10603 has explicit disk-quota evidence,
while 10609's empty error log leaves its exact exception unknown. Both share the
same home storage. The user reports 19G of research files plus a 6.6G environment;
the quota commands reveal no actual limit. All current and earlier sources,
claims, completed evidence and scientific thresholds remain unchanged.

Next review the exact duplicate-archive cleanup proposal. Its 24 Mac backups were
freshly rehashed and all 1,583 members verified; only matching university archive
copies are proposed for removal. No cleanup has been approved or started. No new
probe, paid work or financial release follows from the backup review.
