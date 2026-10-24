# Generated CPU fixture checkpoints

**Verified follow-up:** [the university retry](../report/checks/2026-10-23-university-cpu-recovery.md)
passes all 165 selected CPU checks with zero skips and exit 0. All saved
recovery comparisons and the actual NFS4 lock probe verify independently.
The instructions and rejected outcomes below are retained as history; no
additional university rerun is needed. GPU performance/research gates remain.

**Linux follow-up:** the university attempt is now verified: 99 passed,
29 failed, zero skips. Generated fixture/gradient/attack replay checks pass;
all failures report a checkpoint read-lock error on the university filesystem.
The [checkpoint-lock repair and fresh CPU retry](checkpoint_lock_repair.md)
replace the pending action below. Keep this first handoff and its rejected
result intact.

October 23, 2026. The A40 setup failed before pretrained execution because
fresh generated CPU fixtures did not start from identical model states.
See the [preserved rejection](../report/checks/2026-10-23-a40-setup-rejection.md).
The exact low-level cause on that Intel/PyTorch 2.8 host is still unproved.
The Mac's unmodified factory produced the same state in twelve fresh-process
controls across one/two/four threads and presence/absence of the native probe.
That local PyTorch 2.14.1 observation does not establish Intel 2.8 replay.

## Initial state is an input

Native and shared-forward tiny CPU diagnostics now construct one small
upstream model in their parent and save its initial state. Every fresh worker
receives the same explicit file SHA-256. The bounded, restricted loader checks
the file, format, logical state, complete keys, shapes, dtypes and finite
values before changing model tensors. It restores context, predictor, frozen
target and buffers. Missing/altered files or missing pins fail; there is no
fallback to a new random model. The parent also checks each worker's declared
checkpoint and actual initial model digest when summarizing evidence.

The original small-model factory, generated pixels and masks remain unchanged.
Its saved state is a generated test fixture, not pretrained model weights.
The capture restores the parent's CPU RNG and thread count on success/failure.
Training recovery workers deliberately use three different construction seeds
before loading the common state, then compare full AdamW state, model weights,
RNG, history and continuation draws exactly. Resume-contract mismatch guards
remain enforced. Numerical gates, attack strengths and scientific code are
unchanged; a successful fixture load is not itself a replay pass.

Pretrained CUDA commands reject generated fixture override arguments. Their
original loading branches remain unchanged. CPU artifact references and their
new helper are included in fresh source/worker provenance. Historical source
snapshots, failed evidence and reports retain their original checksums.

The October 22 budget builder therefore correctly refuses the changed working
diagnostic source. Its successful round-trip test now uses an explicitly
isolated source fixture; a separate test requires the real historical source
guard to reject. Stored historical arithmetic still reproduces from the bound
plan and timing data. No historical pin, report or YAML is rewritten, and no
new study admission follows from these CPU checks.

## Free university Linux check before another rental

The prepared source-only handoff lives under
`runs/pretrained/diagnostic/cpu-fixture-repair-20261023/`. It includes source,
tests and the original upstream Git bundle. It includes no video pixels,
pretrained checkpoint, dependency installation, credential or cloud lifecycle
operation. Its adjacent SHA-256 file verifies the archive before extraction;
the launcher checks every packaged source and bundle hash before execution.

Use the existing university environment at `~/.venvs/vjepa`, already verified
with Python 3.11 and PyTorch 2.8.0+cu128. The detached launcher sets
`CUDA_VISIBLE_DEVICES` empty for itself and all children. It runs all selected
fixture, gradient/precision replay, optimizer recovery and budget-provenance
checks with a 25-minute test cap and a 28-minute outer cap including export.
Every selected CPU check must execute; zero skips are allowed. The full Mac
suite is checked separately, and a future GPU setup still requires the complete
suite including actual CUDA tests. Missing dependencies, failures, errors,
skips, changed sources and timeouts
remain unsuccessful checks. It installs nothing and does not rent hardware.

On the Mac, from the repository root:

```bash
handoff_dir="runs/pretrained/diagnostic/cpu-fixture-repair-20261023"
scp "$handoff_dir/vjepa-cpu-fixtures.tar.gz" \
  "$handoff_dir/vjepa-cpu-fixtures.tar.gz.sha256" \
  kdu@linux10609.dc.engr.scu.edu:Desktop/VJEPA-Robust-Overfitting/
```

Then inside university SSH:

```bash
(
  set -e
  cd "$HOME/Desktop/VJEPA-Robust-Overfitting"
  sha256sum -c vjepa-cpu-fixtures.tar.gz.sha256
  test ! -e cpu-fixtures-20261023
  tar -xzf vjepa-cpu-fixtures.tar.gz
  bash cpu-fixtures-20261023/scripts/university_cpu_fixtures.sh
)
```

The launcher prints a fresh `runs/cpu-fixtures.XXXXXX` directory, log path,
session record and evidence export. SSH may be closed while it runs; reconnect
to the same host to inspect the process. Outputs remain in the shared home.
The export retains bounded generated fixture/checkpoint tensors needed to
check replay after copying. A failure preserves its log and partial artifacts.

Copy and independently verify the result before deciding on another paid
benchmark. This Linux CPU check cannot establish A40 pretrained performance,
full-model memory fit, GPU optimizer recovery or whole-study affordability.
The A40 quote, session deadline and SSH identity from the terminated Pod must
not be reused for a new deployment. Retain the pending billing attribution,
both conservative holds and the unchanged $200/$160 project caps.
