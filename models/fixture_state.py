"""Pinned initial weights for generated CPU fixtures, never pretrained loading."""

import hashlib
import io
from pathlib import Path
import re

import torch

from models.diagnostics import state_digest

KIND = 'generated-cpu-fixture-initial-state-v1'
MAX_BYTES = 16 * 2**20


def _digest(state):
    checksum = hashlib.sha256()
    for name, tensor in state.items():
        checksum.update(name.encode())
        checksum.update(tensor.detach().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def _original_cpu_model(model):
    if (getattr(model, '_diagnostic_execution_policy', 'original') != 'original'
            or any(t.device.type != 'cpu' for t in model.state_dict().values())
            or any(p.requires_grad or p.grad is not None for p in model.target_encoder.parameters())
            or any(m.training for m in model.target_encoder.modules())):
        raise ValueError('Fixture checkpoints require an original CPU model with a frozen eval target.')


def save_fixture_state(model, path):
    """Freeze one generated initialization; refuse overwriting existing evidence."""
    _original_cpu_model(model)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if any(t.is_floating_point() and not torch.isfinite(t).all() for t in state.values()):
        raise ValueError('Generated fixture state contains nonfinite values.')
    payload = {'schema_version': 1, 'kind': KIND, 'state': state, 'state_sha256': _digest(state)}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    if len(data) > MAX_BYTES:
        raise ValueError('Generated fixture checkpoint exceeds the small-model limit.')
    path = Path(path)
    with path.open('xb') as handle:
        handle.write(data)
    return {'kind': KIND, 'sha256': hashlib.sha256(data).hexdigest(),
            'bytes': len(data), 'state_sha256': payload['state_sha256']}


def read_fixture_state(path, expected_sha256):
    """Hash exactly the bytes deserialized; never fall back to random weights."""
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('An explicit fixture checkpoint SHA-256 is required.')
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError('Require a bounded regular fixture checkpoint file.')
    with path.open('rb') as handle:
        data = handle.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Fixture checkpoint SHA-256 mismatch.')
    payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if (not isinstance(payload, dict) or set(payload) != {'schema_version', 'kind', 'state', 'state_sha256'}
            or payload['schema_version'] != 1 or payload['kind'] != KIND
            or not isinstance(payload['state'], dict) or not payload['state']):
        raise ValueError('Unsupported generated fixture checkpoint.')
    state = payload['state']
    if any(not isinstance(name, str) or not isinstance(t, torch.Tensor)
           or t.device.type != 'cpu' or t.layout != torch.strided
           or t.is_complex() or (t.is_floating_point() and not torch.isfinite(t).all())
           for name, t in state.items()):
        raise ValueError('Invalid generated fixture tensors.')
    if _digest(state) != payload['state_sha256']:
        raise ValueError('Generated fixture logical-state checksum mismatch.')
    return state, {'kind': KIND, 'sha256': expected_sha256, 'bytes': len(data),
                   'state_sha256': payload['state_sha256']}


def load_fixture_state(model, path, expected_sha256):
    """Audit all keys, shapes and dtypes before mutating any model tensor."""
    _original_cpu_model(model)
    state, identity = read_fixture_state(path, expected_sha256)
    expected = model.state_dict()
    if (list(state) != list(expected) or any(state[k].shape != v.shape or state[k].dtype != v.dtype
                                            for k, v in expected.items())):
        raise ValueError('Generated fixture keys, shapes or dtypes differ.')
    model.load_state_dict(state, strict=True)
    if state_digest(model) != identity['state_sha256']:
        raise RuntimeError('Loaded fixture does not match its pinned logical state.')
    return identity
