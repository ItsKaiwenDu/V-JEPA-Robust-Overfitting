# Local integration checks — October 17, 2026

**Scope:** code correctness on small CPU fixtures and architecture/metadata
compatibility. These are not pretrained evaluations or research results.

- Source: `facebookresearch/vjepa2` at
  `204698b45b3712590f06245fbfba32d3be539812`.
- Environment: macOS ARM64, Python 3.11.16, torch 2.14.1, torchvision 0.29.1,
  timm 1.0.30, einops 0.8.2, NumPy 2.4.6, PyYAML 6.0.3, pytest 9.1.1.
- Test command: `python -m pytest -q tests --upstream <pinned-checkout>`.
- Result: **20 passed**, including native-loss agreement, finite differences,
  visibility-restricted input gradients, target freezing, context/predictor
  updates, activation checkpointing, strict loading, and deterministic masks.
- Upstream deprecation warnings: timm's old layers import and PyTorch's
  `torch.backends.cuda.sdp_kernel` context manager. No upstream patches applied.

The first test attempt used predictor attention heads smaller than the
released model and hit an upstream rotary-position shape error. The fixture
was corrected to preserve the released per-head dimensions (64 for the
encoder, 32 for the predictor), retaining fewer heads/layers/tokens. The
upstream rotary implementation was preserved for checkpoint compatibility.

The first 2 MiB of the [official checkpoint](https://dl.fbaipublicfiles.com/vjepa2/vitl.pt)
were fetched with an HTTP byte-range request. The first ZIP entry was
`vitl/data.pkl`, with 252,101 bytes of metadata. Its tensor names and shapes
were inspected without loading tensor data or executing checkpoint globals.
The full file advertised 5,127,726,842 bytes and a June 6, 2025 last-modified
date. A full checkpoint SHA-256 has not been computed.

The actual full ViT-L architecture was constructed locally without pretrained
weights, and all state names/shapes matched that metadata:

| Component | State tensors | Parameters |
| --- | ---: | ---: |
| Context encoder | 292 | 303,885,312 |
| Fixed target encoder | 292 | 303,885,312 |
| Predictor, including all ten mask tokens | 160 | 22,086,016 |

This supports architecture compatibility, not successful loading or
execution of the released weights. Full pretrained execution, real-video
decoding, adversarial attacks, training, CUDA memory/time, and rental cost
remain unverified. No Lambda instance was created and no paid compute was
used for these checks.
