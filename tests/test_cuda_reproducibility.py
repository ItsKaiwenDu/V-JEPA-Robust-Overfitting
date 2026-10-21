"""Runtime policy and fresh-process CUDA replay, separate from CPU recovery."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_vjepa2 import model

ROOT = Path(__file__).resolve().parents[1]


def isolated(program, **env):
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    environment.pop('CUBLAS_WORKSPACE_CONFIG', None)
    environment.update(env)
    return subprocess.run([sys.executable, '-c', program], cwd=ROOT, env=environment,
                          capture_output=True, text=True, timeout=90)


def test_policy_enforces_errors_without_changing_precision():
    result = isolated('''
import os, torch
from models.runtime import configure_cuda_reproducibility
before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
configure_cuda_reproducibility()
assert not torch.cuda.is_initialized()
assert torch.are_deterministic_algorithms_enabled()
assert not torch.is_deterministic_algorithms_warn_only_enabled()
assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
assert before == (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
''')
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('value', [':16:8', 'invalid'])
def test_conflicting_workspace_setting_is_rejected(value):
    result = isolated('from models.runtime import configure_cuda_reproducibility; configure_cuda_reproducibility()',
                      CUBLAS_WORKSPACE_CONFIG=value)
    assert result.returncode != 0 and 'requires CUBLAS_WORKSPACE_CONFIG' in result.stderr


def test_policy_refuses_a_late_configuration():
    result = isolated('''
import torch
from models.runtime import configure_cuda_reproducibility
torch.cuda.is_initialized = lambda: True
configure_cuda_reproducibility()
''')
    assert result.returncode != 0 and 'before CUDA initialization' in result.stderr


def test_resume_rejects_changed_determinism_warning_policy(model):
    from test_training import make_engine
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        original = make_engine(copy.deepcopy(model))
        saved = original.state()
        torch.use_deterministic_algorithms(True, warn_only=True)
        changed = make_engine(copy.deepcopy(model))
        with pytest.raises(ValueError, match='Resume contract differs'):
            changed.restore(saved)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual CUDA required for fresh-process kernel replay')
def test_cuda_gradients_and_updates_match_across_fresh_processes():
    program = '''
import hashlib, json, torch
from models.runtime import configure_cuda_reproducibility
configure_cuda_reproducibility()
torch.manual_seed(29)
torch.set_num_threads(4)
conv = torch.nn.Conv3d(3, 64, (2, 4, 4), stride=(2, 4, 4)).cuda()
head = torch.nn.Linear(64, 17).cuda()
optimizer = torch.optim.AdamW(list(conv.parameters()) + list(head.parameters()), lr=1e-4)
x = torch.rand(1, 3, 16, 64, 64, device='cuda', requires_grad=True)
values = []
for _ in range(2):
    optimizer.zero_grad(set_to_none=True)
    if x.grad is not None: x.grad = None
    loss = head(conv(x).mean((2,3,4))).square().mean()
    loss.backward()
    values.extend([loss.detach().cpu(), x.grad.detach().cpu()])
    optimizer.step()
values.extend(p.detach().cpu() for p in list(conv.parameters()) + list(head.parameters()))
print(json.dumps([hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest() for t in values]))
'''
    first, second = isolated(program), isolated(program)
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    assert json.loads(first.stdout) == json.loads(second.stdout)
