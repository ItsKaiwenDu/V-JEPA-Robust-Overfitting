"""Typed returns are saved before numerical gates, including failed values."""
import hashlib
import os
from pathlib import Path

import torch

MiB=2**20


def tensor_info(t):
    if not isinstance(t,torch.Tensor):raise TypeError('Actual tensor required.')
    raw=t.detach().cpu().contiguous()
    return {'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),'device':str(t.device),
            'sha256':hashlib.sha256(raw.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def save_return(path, value, kind, input_shape):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    # No cast/rank change. Record even wrong dtype/rank/nonfinite; then refuse.
    info=tensor_info(value);copy=value.detach().cpu().clone()
    limit=65536 if kind=='loss' else 16*MiB
    if copy.numel()*copy.element_size()+4096>limit:raise ValueError('Actual return too large; retain earlier evidence.')
    partial=Path(str(path)+'.partial')
    with partial.open('xb') as f:
        torch.save(copy,f);f.flush();os.fsync(f.fileno())
    if partial.stat().st_size>limit:raise ValueError('Serialized return over cap; preserve partial.')
    os.rename(partial,path)
    if kind not in ('loss','input') or value.dtype!=torch.float32 or list(value.shape)!=([] if kind=='loss' else list(input_shape)):
        raise ValueError('Saved actual return has wrong dtype/rank.')
    return {'file':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source':info,'saved_before_numerical_gate':True}


def compare(actual, expected, bounds):
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype or actual.dtype!=torch.float32:
        raise ValueError('Comparison dtype/rank drift.')
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():raise ValueError('Nonfinite actual/reference return.')
    delta=(actual.double()-expected.double()).abs();limit=bounds['atol']+bounds['rtol']*expected.double().abs()
    if not (delta<=limit).all():raise ValueError('Original numerical gate rejected; no tolerance change.')
    return {'passed':True,'max_absolute_error':float(delta.max()),**bounds}
