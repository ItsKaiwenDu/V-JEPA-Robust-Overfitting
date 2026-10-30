# Fixed-input attention preparation — October 30, 2026

**Prepared, unrun.** A new immutable-input diagnostic removes FP32 regeneration
from each backend worker while preserving the original failures and consumed
attempts. Encoder-0 is assigned to linux10609 and encoder-1 to linux10603;
these independent cases can run in parallel on their separate GPUs.

The 85,622,679-byte packet (SHA256
`51c4e4b71ec292afc65ee3bd53c5556c6b538056d607244aef4945ed5fcded9a`)
restores all 139 regular members. There are 138 source/metadata/fixture files
plus the handoff manifest. All 61 scientific pins and 33 upstream bindings are
unchanged. The nine new/current diagnostic sources match the tested CPU packet.

Predictor and encoder-0 inputs preserve the bytes of their reviewed prior
synthetic workers. Encoder-1 is a single new CPU capture at the original shape
and mask positions. Each fixture contains four FP32 input arrays and four fixed
FP64 oracle arrays, with original layouts and full origin/checksum metadata.
Runtime and review load the same immutable files; there is no generator fallback.
The predictor fixture is included for offline verification but no predictor GPU
rerun is planned. No weights, pixels, trained activations or installs are included.

Forty-six focused CPU checks pass with zero skips, including six actual fresh
tiny math workers and two deliberately rejected CPU workers. Numerical and
incomparable-reference failures retain inputs, device inputs and untimed values
before acceptance checks. Signed/tensor tampering, malformed NumPy headers,
wrong domains, wrong host plans, changed bounds, missing replay and invalid
clock/memory claims reject. CPU results do not qualify a CUDA backend.

Two additional fresh CPU-only loader processes verify every full-size fixture
with operand generation explicitly forbidden. All 24 logical tensor identities
match across processes, and eight reused input signatures match prior evidence.
Torch FP64 reconstruction agrees with the fixed NumPy references at the unchanged
1e-12 / 1e-14 bookkeeping bounds; the largest observed absolute difference is
1.388e-16. These checks execute no SDPA, model or CUDA workload. Eight local
transport/recovery scenarios pass across the two separate helpers.

The new plans bind both previous claims and permit only one newly invoked
attended case per host: 20 minutes including extraction/verification, four fresh
workers capped at 120 seconds each, 4 GiB allocator limit and five-second outer
kill grace. CPU/GPU input snapshots, actual backend evidence, exact replay and
same-input numerical checks remain required before interpreting timing. The
original component output and derivative tolerances are unchanged. Live
identity/foreign-compute checks repeat every five seconds; no other process is
stopped or modified. Old claims and launchers remain consumed and untouched.

Fresh user status reports both expected RTX 3070 UUIDs, 8192 MiB total, 94 MiB
used, driver 580.178.04 and empty compute-process lists. Both new proposals are
unarmed and neither new claim exists. No real SSH, upload, university GPU or
paid work occurred during preparation. Explicit user invocation of each new
helper activates that host's one new plan; it does not reuse the old plan.

The [machine-readable preparation record](2026-10-30-fixed-attention-preparation.json)
binds packet hashes, fixture origins, CPU exports, failure snapshots, full-size
loader checks, transport receipts and unchanged attempt/financial history.
See the [diagnostic design](../../docs/fixed_attention_diagnostic.md) for operation.

The previous input-generation and cross-platform reconstruction refusals remain
unqualified. This preparation does not identify their cause or certify full-model
equivalence, attention policy integration, PGD, optimizer memory/recovery,
training speed, study affordability or research admission. The $200/$160/single
$10 caps, $9.8116 encumbrance, $0.1884 unreserved balance, historical $2 hold
and $0.0792 reconciliation gap remain unchanged. All known Runpod Pods remain
terminated. The user continues to make their own commits.
