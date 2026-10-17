"""Fixed-target V-JEPA 2 integration using the pinned Meta implementation.

This module does not download code, weights, or data. The original training
loss is in upstream app/vjepa/train.py; architecture details are in
src/models/{vision_transformer,predictor}.py. Upstream retains its MIT license.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import copy
import hashlib
import importlib
from pathlib import Path
import re
import subprocess
import sys

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import yaml


UPSTREAM_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
PRETRAIN_CONFIG = "configs/train/vitl16/pretrain-256px-16f.yaml"


def use_upstream(path: str | Path, expected_commit: str = UPSTREAM_COMMIT) -> Path:
    """Verify the source revision before importing its Python modules."""
    root = Path(path).expanduser().resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("The upstream pin must be a full Git commit SHA.")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.PIPE
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal",
             "--", "src", "app/vjepa", "configs/train/vitl16"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot inspect upstream Git checkout: {root}") from exc
    if revision != expected_commit:
        raise ValueError(f"Upstream revision mismatch: expected {expected_commit}, got {revision}.")
    if dirty:
        raise ValueError(f"Upstream model, training, or config files have local changes:\n{dirty}")
    # Do not accidentally reuse another project's generic `src` package.
    for name, module in list(sys.modules.items()):
        if name == "src" or name.startswith("src."):
            locations = list(getattr(module, "__path__", []))
            if getattr(module, "__file__", None):
                locations.append(module.__file__)
            if any(not Path(location).resolve().is_relative_to(root) for location in locations):
                raise RuntimeError("Another upstream `src` package is already imported; use a fresh process.")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


@dataclass(frozen=True)
class MaskPair:
    """Visible and prediction token indices, each with shape [batch, tokens]."""

    context: Tensor
    target: Tensor

    def to(self, device: torch.device | str) -> "MaskPair":
        return MaskPair(self.context.to(device), self.target.to(device))


class FixedTargetVJEPA(nn.Module):
    """Native masked prediction with an explicitly separate, frozen teacher."""

    def __init__(self, encoder: nn.Module, predictor: nn.Module, target_encoder: nn.Module):
        super().__init__()
        if encoder is target_encoder:
            raise ValueError("Context and target encoders must be separate modules.")
        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder = target_encoder.requires_grad_(False).eval()
        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1), persistent=False)

    def train(self, mode: bool = True) -> "FixedTargetVJEPA":
        super().train(mode)
        self.target_encoder.eval()
        return self

    def normalize(self, pixels: Tensor) -> Tensor:
        if pixels.ndim != 5 or pixels.shape[1] != 3 or not pixels.is_floating_point():
            raise ValueError("Expected floating-point raw RGB pixels with shape [B, 3, T, H, W].")
        expected = (self.encoder.num_frames, self.encoder.img_height, self.encoder.img_width)
        if pixels.shape[0] < 1 or tuple(pixels.shape[2:]) != expected:
            raise ValueError(f"Expected a nonempty batch of clips with (T, H, W)={expected}.")
        if not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
            raise ValueError("Pixels must be finite and in raw [0, 1] space.")
        return (pixels - self.pixel_mean.to(pixels)) / self.pixel_std.to(pixels)

    @torch.no_grad()
    def clean_targets(self, clean_pixels: Tensor) -> Tensor:
        """Return normalized full-token targets; callers must pass the clean clip."""
        self.target_encoder.eval()
        features = self.target_encoder(self.normalize(clean_pixels))
        return F.layer_norm(features, (features.shape[-1],)).detach()

    def _validate_masks(self, pixels: Tensor, masks: Sequence[MaskPair]) -> None:
        if not masks:
            raise ValueError("At least one context/target mask pair is required.")
        batch = pixels.shape[0]
        tokens = self.encoder.num_patches
        for pair in masks:
            for indices in (pair.context, pair.target):
                if indices.ndim != 2 or indices.shape[0] != batch or indices.shape[1] == 0:
                    raise ValueError("Every mask must be nonempty with shape [batch, tokens].")
                if indices.dtype != torch.long or indices.device != pixels.device:
                    raise ValueError("Mask indices must be int64 on the input device.")
                if indices.min() < 0 or indices.max() >= tokens:
                    raise ValueError("Mask token index is out of range.")
                ordered = indices.sort(dim=1).values
                if (ordered[:, 1:] == ordered[:, :-1]).any():
                    raise ValueError("Duplicate token indices are not allowed.")
            for visible, predicted in zip(pair.context, pair.target):
                if torch.isin(visible, predicted).any():
                    raise ValueError("Visible context and prediction masks must be disjoint.")

    def predict(self, pixels: Tensor, masks: Sequence[MaskPair]) -> list[Tensor]:
        normalized = self.normalize(pixels)
        self._validate_masks(pixels, masks)
        # Upstream PredictorMultiSeqWrapper uses the sequence-group index,
        # not the mask-family index. This study has a single sequence group.
        return [self.predictor(self.encoder(normalized, masks=pair.context),
                               pair.context, pair.target, mask_index=0)
                for pair in masks]

    def loss_per_example(self, pixels: Tensor, targets: Tensor, masks: Sequence[MaskPair]) -> Tensor:
        """Average L1 over features/tokens, then equally over mask families."""
        if targets.requires_grad:
            raise ValueError("Clean target features must be detached.")
        expected = (pixels.shape[0], self.encoder.num_patches, self.encoder.embed_dim)
        if tuple(targets.shape) != expected or targets.device != pixels.device:
            raise ValueError(f"Expected clean targets with shape {expected} on the input device.")
        losses = []
        for prediction, pair in zip(self.predict(pixels, masks), masks):
            indices = pair.target.unsqueeze(-1).expand(-1, -1, targets.shape[-1])
            selected = targets.gather(1, indices)
            losses.append((prediction - selected).abs().mean(dim=(1, 2)))
        return torch.stack(losses).mean(dim=0)

    def forward(self, pixels: Tensor, targets: Tensor, masks: Sequence[MaskPair]) -> Tensor:
        return self.loss_per_example(pixels, targets, masks).mean()


def build_vitl(upstream: str | Path, *, activation_checkpointing: bool = True) -> FixedTargetVJEPA:
    """Construct the released ViT-L architecture; no pretrained weights loaded yet."""
    use_upstream(upstream)
    vit = importlib.import_module("src.models.vision_transformer")
    pred = importlib.import_module("src.models.predictor")
    common = dict(img_size=256, patch_size=16, num_frames=16, tubelet_size=2,
                  use_rope=True, uniform_power=True, use_silu=False, wide_silu=True,
                  use_sdpa=True, use_activation_checkpointing=activation_checkpointing)
    encoder = vit.vit_large(**common)
    predictor = pred.vit_predictor(embed_dim=1024, predictor_embed_dim=384,
                                   depth=12, num_heads=12, use_mask_tokens=True,
                                   num_mask_tokens=10, zero_init_mask_tokens=True, **common)
    target = copy.deepcopy(encoder)
    target.use_activation_checkpointing = False
    return FixedTargetVJEPA(encoder, predictor, target)


def canonical_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Remove only known leading wrappers, rejecting ambiguous names."""
    cleaned = {}
    for original, tensor in state.items():
        if not isinstance(original, str) or not isinstance(tensor, Tensor):
            raise ValueError("A model state must contain only named tensors.")
        name = original
        for prefix in ("module.", "backbone."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name in cleaned:
            raise ValueError(f"Checkpoint wrapper prefixes collide at {name}.")
        cleaned[name] = tensor
    return cleaned


def load_components(model: FixedTargetVJEPA, checkpoint: Mapping) -> dict:
    """Audit all three states before loading any; never reinitialize missing layers."""
    audited = {}
    for key in ("encoder", "target_encoder", "predictor"):
        if key not in checkpoint or not isinstance(checkpoint[key], Mapping):
            raise ValueError(f"Checkpoint is missing the required {key} state.")
        state = canonical_state(checkpoint[key])
        expected = getattr(model, key).state_dict()
        missing, unexpected = set(expected) - set(state), set(state) - set(expected)
        mismatched = [k for k in set(expected) & set(state) if expected[k].shape != state[k].shape]
        if missing or unexpected or mismatched:
            raise ValueError(f"{key} audit failed: missing={sorted(missing)}, "
                             f"unexpected={sorted(unexpected)}, wrong_shape={sorted(mismatched)}")
        audited[key] = state
    for key, state in audited.items():
        getattr(model, key).load_state_dict(state, strict=True)
    model.target_encoder.requires_grad_(False).eval()
    return {key: {"state_tensors": len(state), "elements": sum(v.numel() for v in state.values())}
            for key, state in audited.items()}


def load_checkpoint(model: FixedTargetVJEPA, path: str | Path, expected_sha256: str | None = None) -> dict:
    """Hash a local checkpoint and load it with restricted PyTorch deserialization."""
    path = Path(path).expanduser().resolve()
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError(f"Checkpoint SHA-256 mismatch: got {digest}.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    components = load_components(model, checkpoint)
    return {"sha256": digest, "bytes": path.stat().st_size,
            "epoch": checkpoint.get("epoch"), "components": components}


def sample_masks(upstream: str | Path, *, batch_size: int, seed: int,
                 frames: int = 16, crop_size: int = 256, patch_size: int = 16) -> list[MaskPair]:
    """Replay upstream block-mask sampling without requiring a video decoder.

    This explicit seed controls both the upstream block-size counter and CPU
    placement RNG. It is a smoke-test sampler, not a training worker sampler.
    """
    root = use_upstream(upstream)
    if batch_size < 1 or not 0 <= seed < 2**31 - 1:
        raise ValueError("Use a positive batch size and a seed in [0, 2**31 - 1).")
    config = yaml.safe_load((root / PRETRAIN_CONFIG).read_text())
    collator_type = importlib.import_module("src.masks.multiseq_multiblock3d").MaskCollator
    collator = collator_type(config["mask"], [frames], crop_size=crop_size,
                             patch_size=patch_size, tubelet_size=2)
    masks = []
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for generator in collator.mask_generators[frames]:
            with generator._itr_counter.get_lock():
                generator._itr_counter.value = seed - 1
            context, target = generator(batch_size)
            masks.append(MaskPair(context, target))
    return masks
