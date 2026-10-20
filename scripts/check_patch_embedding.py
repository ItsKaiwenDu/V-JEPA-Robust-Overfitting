#!/usr/bin/env python3
"""Isolate MPS input-gradient timing for equivalent nonoverlapping patch maps."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output filename.')
    import torch
    from models.patch_embed import LinearPatchEmbed3D
    from models.runtime import require_device, device_metadata
    require_device('mps')
    torch.set_num_threads(4)
    torch.manual_seed(20261018)
    native = torch.nn.Conv3d(3, 1024, kernel_size=(2,16,16), stride=(2,16,16))
    alternative = LinearPatchEmbed3D(SimpleNamespace(proj=copy.deepcopy(native), patch_size=16, tubelet_size=2))
    clean = torch.rand(1,3,16,256,256)
    upstream = torch.randn(1,2048,1024)
    modules = {'conv3d': native.to('mps'), 'linear': alternative.to('mps')}
    records, values = {}, {}
    for name, module in modules.items():
        timings = []
        for iteration in range(2):
            x = clean.to('mps').requires_grad_(True)
            cotangent = upstream.to('mps')
            torch.mps.synchronize()
            start = time.perf_counter()
            out = module(x)
            if name == 'conv3d':
                out = out.flatten(2).transpose(1,2)
            gradient, = torch.autograd.grad((out*cotangent).sum(), x)
            torch.mps.synchronize()
            elapsed = time.perf_counter()-start
            timings.append(elapsed)
            values[name] = (out.detach().cpu(), gradient.detach().cpu())
            print(f'{name}, pass {iteration}: forward + input gradient {elapsed:.6f}s', flush=True)
        records[name] = {'seconds': timings, 'measured_seconds': timings[1]}
    for reference, observed in zip(values['conv3d'], values['linear']):
        torch.testing.assert_close(reference, observed, rtol=1e-4, atol=2e-5)
    report = {'kind': 'mps_patch_projection_microbenchmark', 'passed': True, 'research_result': False,
              'runtime': device_metadata('mps'), 'torch': str(torch.__version__),
              'shape': [1,3,16,256,256], 'output_features': 1024,
              'records': records, 'seed': 20261018,
              'source_files_sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                                      for n in ('scripts/check_patch_embedding.py','models/patch_embed.py','models/runtime.py')},
              'limitations': ['Random patch weights; isolated operation, not full model throughput.',
                              'One warmup and one measured pass per implementation.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report,handle,indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
