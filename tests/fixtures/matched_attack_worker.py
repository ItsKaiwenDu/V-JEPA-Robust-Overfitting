"""Generated upstream CPU fixture only; no production random-weight fallback."""

from pathlib import Path
import sys

root, output, role, initial, initial_sha = sys.argv[1:]
root, output = Path(root), Path(output)
sys.path.insert(0, str(root))

import torch

from models.execution import apply_model_policy, configure_execution
from models.fixture_state import load_fixture_state
from models.matched_attack import ARMS, POLICY, run_attacks
from models.native_diagnostic import tiny_fixture

configure_execution(POLICY, device='cpu')
torch.set_num_threads(2)
# Different construction RNGs must be overwritten by the same audited state.
torch.manual_seed(51 if role == 'matrix' else 97)
model, pixels, sampler = tiny_fixture(root / 'external/vjepa2')
identity = load_fixture_state(model, initial, initial_sha)
apply_model_policy(model, POLICY)
run_attacks(model, pixels, lambda seed: sampler(seed % (2**31 - 1)), video_id='generated-training-only',
            provenance={'fixture': identity}, output=output,
            arms=tuple(ARMS) if role == 'matrix' else ('A',))
