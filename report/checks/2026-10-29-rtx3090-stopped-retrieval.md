# RTX 3090 stopped; final evidence retrieval pending — October 29

The user reports stopping `vjepa-eot8-rtx3090-08` at **4:00 PM Pacific**, before
the original 4:01 PM cutoff. This is a user report, not a provider-API check.
No further benchmark, model retry or resource restart is authorized by this update.

The last local read at 2:56 PM verifies 77 CPU checks, the numerical preflight
and calibration-bundle phase; primary-pretrained was active. The final exit status,
remaining phases and final evidence archive are still unknown. A stopped Pod cannot
serve an SSH download. Results were directed to the retained `/workspace` volume;
do not terminate the Pod before retrieving and verifying them.

The 119-minute running interval at the quoted $0.52/hour estimates $1.03133,
excluding stopped-volume storage and any other charges. This is not an invoice.
Retain the $1.30 reservation, the older hold/gap and $9.9788 conservative encumbrance
until actual billing is reconciled. The $200 project, $160 compute and single $10
diagnostic caps remain unchanged. Do not silently extend the original two-hour run.

A standard-library download-only helper is prepared in the raw session directory
as `retrieve_evidence_only.py`. It never deploys, starts or modifies a Pod, uploads
files, initializes CUDA, installs dependencies, regenerates exports or reruns a
model. Once a separately reviewed retrieval session is running, it uses the current
verified SSH endpoint/host key to copy the existing archive, manifest and exit
status. It retains partial copies, refuses automatic retries, limits download time
to seven minutes and archive size to 1 GiB, and reuses the hash-pinned original
archive verifier to validate every regular member before extraction. Byte-level
verification does not replace numerical/provenance review.

Next obtain the actual session cost/current balance and a CPU-only retrieval quote
if offered. Review a separate short retrieval allowance and explicit resource
decision before any restart. Recheck endpoint/public fingerprint after a restart;
the old container identity and agent lifetime may no longer apply. Copy and verify
the saved evidence, review full versus partial completion, then terminate. If an
archive is absent or invalid, preserve the files and review a metadata-export-only
fallback; never relaunch the expired benchmark operator.

Runpod documents [volume persistence until Pod deletion](https://docs.runpod.io/pods/storage/types)
and [CPU-only starts for data recovery](https://docs.runpod.io/pods/troubleshooting/zero-gpus).
Actual CPU-only availability and price for this Pod have not been checked.

The user subsequently reports a **provisional $1.1074** session cost, **$6.57**
account balance and **$0.022/hour** stopped storage. The restart menu offers
CPU-only data access, automatic migration or waiting; its CPU-only price remains
unknown. Preserve the entire reservation and request the quote without starting.
The local helper syntax, valid fixture archive round trip, nonzero model-status
preservation and tampered-archive rejection passed using mocked SSH backed only
by local `cat`; no real connection or model was started. Scaffold verification
also passed. These checks do not certify real downloads or scientific results.
