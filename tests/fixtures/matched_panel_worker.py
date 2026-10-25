"""Only a generated CPU fixture; production CLI has no random-weight mode."""

import json
from pathlib import Path
import sys

root,output,plan_path,execution_path,case_id,initial,initial_sha=sys.argv[1:]
root,output=Path(root),Path(output)
sys.path.insert(0,str(root))

import torch

from models.execution import apply_model_policy,configure_execution
from models.fixture_state import load_fixture_state
from models.matched_panel_case import run_worker
from models.native_diagnostic import tiny_fixture

configure_execution('shared-fp64-native',device='cpu')
torch.set_num_threads(2)
torch.manual_seed(919 if case_id.endswith('0') else 713)
model,pixels,sampler=tiny_fixture(root/'external/vjepa2')
load_fixture_state(model,initial,initial_sha)
apply_model_policy(model,'shared-fp64-native')
run_worker(model,pixels,lambda s:sampler(s%(2**31-1)),
    plan=json.loads(Path(plan_path).read_text()),case_id=case_id,
    execution=json.loads(Path(execution_path).read_text()),output=output,role='replay',allow_fixture=True)
