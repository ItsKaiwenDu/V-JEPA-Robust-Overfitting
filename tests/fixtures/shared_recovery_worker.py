"""Fresh-process test worker; generated pixels and small upstream model only."""

from pathlib import Path
import random
import sys

root, output, policy, condition, mode = sys.argv[1:]
root, output = Path(root), Path(output)
sys.path[:0] = [str(root), str(root / 'tests')]

import numpy as np
import torch

from models.checkpointing import CheckpointStore
from models.execution import apply_model_policy, configure_execution
from models.native_diagnostic import tiny_fixture
from test_training import make_engine

configure_execution(policy, device='cpu')
torch.set_num_threads(2)
torch.manual_seed(20261022)
model, _, _ = tiny_fixture(root / 'external/vjepa2')
apply_model_policy(model, policy)
engine = make_engine(model, condition=condition)
store = CheckpointStore(output / 'checkpoint')
if mode == 'resume':
    random.random(), np.random.rand(), torch.rand(8)
    engine.restore(store.load()[0])
if mode == 'split':
    for _ in range(2): engine.update()
    store.save(engine.state())
else:
    while not engine.finished: engine.update()
    state = engine.state()
    torch.save({'state': state, 'draws': (random.random(), float(np.random.rand()), torch.rand(3))}, output / (mode + '.pt'))
