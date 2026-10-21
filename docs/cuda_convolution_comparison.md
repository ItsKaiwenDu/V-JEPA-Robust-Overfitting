# Strict CUDA convolution and precision comparison

Prepared October 21, 2026 after the [verified university component outcome](../report/checks/2026-10-21-university-cuda-components.md).
That result identifies a slow strict Conv3d input-backward kernel on RTX 3070;
it does not establish a faster alternative or the cause of the RTX 5090 slowdown.
This new diagnostic compares execution choices for the original upstream
`PatchEmbed3D` / `nn.Conv3d`. It never patches the research model.

**First CUDA comparison:** all 12 workers finished, but every option was rejected
before timing. Even the unchanged cuDNN reference failed the fixed FP64 accuracy
gate. Native output and gradients passed FP64 but failed compatibility with the
cuDNN reference. See the [preserved outcome](../report/checks/2026-10-21-university-cuda-convolution.md).
This motivated an explicit TF32-off control without changing the study or
relaxing tolerances. The [completed control](../report/checks/2026-10-21-university-cuda-convolution-precision.md)
now passes all 12 workers and observed replay: matching operands/oracles and
shared runtime, with cuDNN TF32 off, resolve the reference accuracy rejection.
Native forward/backward takes 5.897 / 7.043 ms in the input / parameter modes
versus 3.410 / 3.431 seconds for strict cuDNN. These isolated RTX 3070 results
qualify a candidate, not a production change or whole-model speedup. The next
gate is opt-in integration and full-model gradient/recovery validation.

## Options and unchanged settings

| Option | Explicit execution change |
| --- | --- |
| `contiguous` | Original contiguous strict reference |
| `channels-last-3d` | Original Conv3d with channels-last-3D weights; convert input inside every forward |
| `native` | Original Conv3d with cuDNN disabled around both forward and backward |

Every worker uses strict deterministic algorithms, warning mode off, cuDNN
determinism on, benchmarking off, and cuBLAS workspace `:4096:8`. Matmul TF32
remains off / highest precision. `--cudnn-tf32 on|off` explicitly declares
convolution precision for every fresh worker; the CLI default remains `on`
to preserve the original diagnostic policy. The **new handoff is launched
with `--cudnn-tf32 off`**. The native context retains this declared flag while
disabling cuDNN around forward and backward. It cannot silently restore TF32.
Unsupported deterministic operations raise errors; no fallback is enabled.

PyTorch documents separate matmul/cuDNN controls and reduced input precision
for TF32 in [CUDA semantics](https://docs.pytorch.org/docs/2.8/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).
FP32 tensor storage and strict determinism do not establish full FP32 arithmetic
accuracy. This is a diagnostic precision control, not an adopted study policy.
Any future study precision change requires a new recorded contract and pilot.

The same synthetic CPU seed, original random parameter values, and fixed
cotangent feed every option. Representative geometry is one
`[1,3,16,256,256]` input, 1024 output channels, kernel and stride `(2,16,16)`,
and `[1,2048,1024]` tokens. State keys and parameter-value hashes are checked.
No pretrained weights, optimizer, videos, or split pixels are loaded.

Two modes run separately: input gradients only, and input plus weight and bias
gradients. Each option/mode has two fresh workers: **12 workers** in total.
Workers perform correctness validation first, then one warmup, two timed
repetitions, and an optional profiler iteration excluded from timing medians.

## Independent numerical gate

[cuda_convolution.py](../models/cuda_convolution.py) constructs a separate
CPU FP64 reference by unfolding nonoverlapping patches. Matrix multiplication
computes predictions; analytic matrix derivatives compute input, weight, and
bias gradients, with an explicit inverse permutation to reconstruct pixels.
This reference uses neither convolution nor autograd. CPU double-precision tests
compare all four results against convolution/autograd on separate geometries.

Every worker compares its candidate against both this analytic reference and
the original contiguous strict Conv3d. The original Conv3d is also checked
against the analytic reference. Fixed elementwise tolerances are declared
before CUDA execution:

| Value | Relative tolerance | Absolute tolerance |
| --- | ---: | ---: |
| Output | 1e-4 | 2e-5 |
| Input gradient | 1e-4 | 2e-5 |
| Weight gradient | 1e-4 | 2e-4 |
| Bias gradient | 1e-4 | 1e-4 |

Reports retain the maximum absolute error, maximum ratio to the allowed
elementwise error, relative L2 error, and failed-element count. Near-zero
entries still have to pass the absolute tolerance. A rejected candidate is
not timed and cannot receive a speed ratio. Tolerances are not relaxed after
observing CUDA output. Different layouts/backends may fail despite computing
the same mathematical affine map.

Qualified isolated evidence also requires exact output/gradient hashes between
the correctness evaluation and both timed repetitions, and across the two
independent workers. Input-only success cannot qualify the parameter-gradient
mode. Source, operand, oracle, runtime, precision, and tolerance identities
must match; mismatched or incomplete evidence cannot qualify.
Schema 2 records the declared precision policy in parent and worker contracts
and checks it against the actual runtime. Mixing TF32-on and TF32-off evidence
is rejected. Validation tensor signatures are now saved even after numerical
rejection; rejected results still do not receive timing or replay qualification.
The original schema-1 attempt/archive remains unchanged.

## Timing, bounds, and limits

Input layout conversion is inside the timed candidate forward, and its
derivative is inside input backward. Separate spans record one-time weight
layout setup and standalone input conversion; these must not be added to
forward time again. Timed forward/backward spans synchronize the device and
exclude validation, CPU references, progress-file I/O, and profiler overhead.
Optional CPU/CUDA operator tables can identify the selected kernel. Event
times can overlap and cannot be summed into wall time.

Default limits: **45 seconds per worker, 480 seconds overall**, and **4 GiB
CUDA allocator limit**. The detached university launcher adds a 540-second
external supervisor with a five-second kill grace. Worker startup, validation,
warmup, timing, and optional profiling all consume the worker deadline. A slow
reference or parameter backward can time out; inspect its saved progress stage
instead of treating it as a pass. Driver memory is outside the allocator limit.

Atomic `session.json` records the active task, worker statuses, checksummed
result files, provenance, verified numerical/replay results, and ratios only
for qualified same-operand/runtime comparisons. `complete` means all evidence
pairs finished; `all_checks_passed` additionally requires every pair to qualify.
The CLI returns zero only for `all_checks_passed`, otherwise nonzero. A
validation rejection, error, timeout, or unavailable backend remains evidence
to inspect. No variant is selected automatically and `model_changed` is false.

Two independent processes and two timed repetitions provide a bounded
diagnostic, not a performance/reproducibility guarantee. CPU tiny fixtures
cannot establish CUDA speed or numerical agreement. RTX 3070 results cannot
replace representative RTX 5090 measurements or full-model recovery. Research
calibration and budget admission remain separate gates.

## Existing university GPU

No package install, model download, cloud rental, or old project modification
is required. The source-only handoff uses the existing Python 3.11 /
PyTorch 2.8.0+cu128 environment at `$HOME/.venvs/vjepa` and the existing pinned
upstream checkout. The old component profile remains intact.

In **Mac Terminal**, from the repository root:

```bash
scp runs/pretrained/diagnostic/cuda-convolution-precision-handoff-20261021/vjepa-cuda-convolution-precision.tar.gz \
  kdu@linux10609.dc.engr.scu.edu:Desktop/VJEPA-Robust-Overfitting/
```

Then in **university SSH on linux10609**:

```bash
(
  set -e
  cd "$HOME/Desktop/VJEPA-Robust-Overfitting"
  test ! -e cuda-convolution-precision-20261021
  tar -xzf vjepa-cuda-convolution-precision.tar.gz
  bash cuda-convolution-precision-20261021/scripts/university_convolution.sh --cudnn-tf32 off
)
```

The launcher verifies its source manifest, resolves shared-home aliases, and
prints the actual fresh run directory and watch command. You may exit SSH
after launch. Reconnect to the same host to inspect it; pressing Ctrl+C in
`tail -f` only exits the viewer.

Send the final launcher status even if nonzero. Copy the printed run folder
back to a fresh ignored local directory for review, including `results/`,
`launcher.log`, and `exit_status.txt`. A rerun uses a fresh output directory.
No production change or new paid rental follows automatically.
