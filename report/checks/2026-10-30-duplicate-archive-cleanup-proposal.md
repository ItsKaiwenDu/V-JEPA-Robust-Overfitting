# Exact duplicate-archive cleanup proposal

Status: this exact proposal was explicitly approved, executed and independently
reviewed. All 24 archive copies were removed; their Mac backups remain verified.
See [the completion review](2026-10-30-duplicate-archive-cleanup-review.md).
The original preparation and approved scope below are retained for provenance;
the cleanup launcher is consumed and must not be repeated.

This proposal names exactly 24 compressed `.tar.gz` export archives, totaling
3,186,014,424 logical bytes (3.19 GB / 2.97 GiB). Each Mac backup was freshly
rehashed, its sidecar checked, and every archive member verified: 1,583 member
visits in total. These compressed exports duplicate preserved Mac archives and
the retained university unpacked results. Manifests, journals, claims, source,
Python environment, weights, current failed CPU captures and unpacked result
files are outside this list. No directory or glob deletion is permitted.

The legacy remote audit readers that directly require these archive paths would
need the archive restored byte for byte from its listed Mac backup before running
again. Current Mac evidence and registered analysis stay available. This proposal
is archival cleanup, not permission to rerun an experiment or waive any gate.

The helper is now armed by `CLEANUP-AUTHORIZATION.json`, bound to the exact
proposal SHA256. Approval authorizes only these archive removals and no new probe.
The Mac launcher rechecks every local backup. The remote program then verifies
all 24 current file hashes/sizes, regular-file status, single-link status and
path components before any removal. A missing/changed/symlinked/hardlinked file
stops the whole preflight without deleting a file. Each identity is checked again
before unlinking that exact basename through its held directory descriptor.
If removal is interrupted after starting, local progress is preserved and no
automatic retry occurs. A subsequent read-only inspection must determine which
named copies remain. It creates no remote audit/log/archive file, so the full
quota does not require extra scratch space. A receipt is saved on the Mac.

Eleven local check groups pass, including hash/missing/link/traversal/repeated
target refusal, mutation after preflight, approved deletion of only temporary
fixture files, and the actual unarmed launcher's refusal before SSH. These checks
removed no real university file and opened no real connection. Remote existence,
link counts and exact reclaimable quota bytes remain unmeasured until preflight;
the nominal 3.19 GB is not a guarantee of immediately available quota headroom.

Remote paths below are relative to `$HOME/Desktop/VJEPA-Robust-Overfitting`.
Full backup paths, sidecar digests, sizes and SHA256 values are recorded in
the linked JSON proposal and local `proposal.json`.

| Archive | University relative path | MiB | SHA256 |
| --- | --- | ---: | --- |
| clip-00-bank-0 | `matched-panel-transfer-20261024/unpacked/matched-panel-20261024/runs/panel.qIXeDn/exports/clip-00-bank-0-attempt-01.tar.gz` | 105.1 | `7aa64795e19e4b11208909435d03be140713ba7af8fee9ad232c24294597a2c8` |
| clip-00-bank-1 | `matched-panel-transfer-20261024/unpacked/matched-panel-20261024/runs/panel.qIXeDn/exports/clip-00-bank-1-attempt-01.tar.gz` | 104.8 | `230a8d99862b0f051b69cd091138c4b2c2105762c70af0f75b5957656360801c` |
| clip-01-bank-0 | `matched-panel-transfer-20261024/unpacked/matched-panel-20261024/runs/panel.qIXeDn/exports/clip-01-bank-0-attempt-02.tar.gz` | 111.9 | `bb9252edb0ce436fe5c758c3219c0a092ba5f5c2171e0570938175fa354ea7a0` |
| clip-01-bank-1 | `matched-panel-transfer-20261024/unpacked/matched-panel-20261024/runs/panel.qIXeDn/exports/clip-01-bank-1-attempt-01.tar.gz` | 112.6 | `df7bb4c5a7739ecebf95738309aca74568a0f26d2ec688c3d76b0312dd22def2` |
| clip-02-bank-0 | `matched-panel-transfer-20261024/unpacked/matched-panel-20261024/runs/panel.qIXeDn/exports/clip-02-bank-0-attempt-01.tar.gz` | 117.9 | `8df90c34d0937c2c956affeef67e16f4675697d957a32d47025abff428405f07` |
| clip-02-bank-1 | `matched-panel-transfer-20261024/unpacked/matched-panel-20261024/runs/panel.qIXeDn/exports/clip-02-bank-1-attempt-01.tar.gz` | 118.0 | `2697b71bed7c34ae799e55cbfe1680c2db95022bce9a09b50d5fffc5c13becc7` |
| clip-03-bank-0 | `matched-panel-transfer-20261024/unpacked/matched-panel-20261024/runs/panel.qIXeDn/exports/clip-03-bank-0-attempt-01.tar.gz` | 99.9 | `98c355f78984062cf9563f3ee024a59acc8fcbb4d88ab1d31abaf6b3ed87c16d` |
| clip-03-bank-1 | `matched-panel-transfer-20261024/two-host-panel-20261026/cases/clip-03-bank-1/attempt-01/revision-evidence.tar.gz` | 98.9 | `a4ff6ef36d28bd3eff7fb62ddce476f3d8cda958225ac150f928863a9f991756` |
| clip-04-bank-0 | `matched-panel-transfer-20261024/two-host-panel-20261026/cases/clip-04-bank-0/attempt-01/revision-evidence.tar.gz` | 102.5 | `a44aeb79fdfa6dbdab07aa454ac224fe31c0d9166526debda864fc0db3a25b0e` |
| clip-04-bank-1 | `matched-panel-transfer-20261024/two-host-panel-20261026/cases/clip-04-bank-1/attempt-02/recovery-evidence.tar.gz` | 103.2 | `4c5af7a00d74ba1f2282d0527769d4c1d7b50c9ac2aae51b571138d9264c1097` |
| clip-05-bank-0 | `matched-panel-transfer-20261024/two-host-panel-20261026/cases/clip-05-bank-0/attempt-01/revision-evidence.tar.gz` | 120.8 | `7efcb06752c40ce31a7788dc4032f0106d71660f30442483323d733b6cca33fa` |
| clip-05-bank-1 | `matched-panel-transfer-20261024/reviewed-execution-20261026/cases/clip-05-bank-1/attempt-01/continuation-evidence.tar.gz` | 120.8 | `57dcb92bab3cf2e701c17aa79692d841046104162080764589b177a485b27693` |
| clip-06-bank-0 | `matched-panel-transfer-20261024/reviewed-execution-20261026/cases/clip-06-bank-0/attempt-01/continuation-evidence.tar.gz` | 110.9 | `d4372de313717ba0d65cc0790e605eda0b2f15c6073df6278e6726d4b1a86295` |
| clip-06-bank-1 | `matched-panel-transfer-20261024/archery-recovery-execution-20261027/cases/clip-06-bank-1/recovery-01/continuation-evidence.tar.gz` | 111.3 | `2589bf4fc97cd14c262280b6bc65eecd9bd5de6bf13f02c57db0cd3385ca7ccb` |
| clip-07-bank-0 | `matched-panel-transfer-20261024/archery-yoyo-execution-20261027/cases/clip-07-bank-0/attempt-01/continuation-evidence.tar.gz` | 125.7 | `c7ac387cd1a87e9ca31be5460726ccc3b565a1c9e0a87c22b09eb3ab744871c3` |
| clip-07-bank-1 | `matched-panel-transfer-20261024/yoyo-bowling-execution-20261027/cases/clip-07-bank-1/attempt-01/continuation-evidence.tar.gz` | 125.9 | `7cf71401c979e67594fc8fe0312828b1645c9c52ed0a1f07c1ad0c27bc5607ee` |
| clip-08-bank-0 | `matched-panel-transfer-20261024/yoyo-bowling-execution-20261027/cases/clip-08-bank-0/attempt-01/continuation-evidence.tar.gz` | 122.4 | `13b579f6401278da552feb8f06dc3eaaaef75624cc600ae6e69d5d0c07b37246` |
| clip-08-bank-1 | `matched-panel-transfer-20261024/last-two-execution-20261028/cases/clip-08-bank-1/attempt-01/continuation-evidence.tar.gz` | 123.0 | `bd7ef17c41b747b2ac7a5a61241a93adcb383522c89eb4aedcdae3188ff96efc` |
| clip-09-bank-0 | `matched-panel-transfer-20261024/bowling-handstand-execution-20261028/cases/clip-09-bank-0/attempt-01/continuation-evidence.tar.gz` | 115.0 | `2de68fde0f75bfa5c3e288d6391a788e7167d4af9c81c8338289cdf4c71a938e` |
| clip-09-bank-1 | `matched-panel-transfer-20261024/last-two-execution-20261028/cases/clip-09-bank-1/attempt-01/continuation-evidence.tar.gz` | 116.3 | `3bbc7c9dc6beffde4ab0309dc8a600a47f12774c0026ba937bce013019b89916` |
| fixed-attention-university-20261030/linux10609 | `fixed-attention-transfer-20261030/linux10609/unpacked/fixed-attention-runs/fixed-attention.y4cv6d/results/evidence.tar.gz` | 313.5 | `4d4d577c40957d67eb010e18235e21c03307ea5359e5086867e26515a679ffad` |
| fixed-attention-university-20261030/linux10603 | `fixed-attention-transfer-20261030/linux10603/unpacked/fixed-attention-runs/fixed-attention.yVnT8b/results/evidence.tar.gz` | 122.9 | `1a71e16529ce258bec54cadccb8e50f2890ed150529f4a651ee9d179e418ffbc` |
| attention-components-university-20261030/linux10609 | `attention-components-transfer-20261030/linux10609/unpacked/attention-runs/attention.sAjUk6/results/evidence.tar.gz` | 204.3 | `1cede5502127193fa06816b5c6f0f42c2f4c407ed636e9d9cffed7a70e625be3` |
| attention-components-university-20261030/linux10603 | `attention-components-transfer-20261030/linux10603/unpacked/attention-runs/attention.RWeKzM/results/evidence.tar.gz` | 131.0 | `d27d74f1d83bd9bab8255ce54537de889b6451d2441e8a62e065b519a8200366` |

Proposal SHA256: `5dde205601c6d755d24a42450ba5e11a85dbd913e1eb70ed18e5ac081daa2118`.

[Full proposal and backup proof](2026-10-30-duplicate-archive-cleanup-proposal.json).
