"""Cold construction differs deliberately; only verified fixture bytes may align it."""

import copy
import hashlib

import pytest
import torch
from torch import nn

from models.diagnostics import state_digest
from models.fixture_state import load_fixture_state, read_fixture_state, save_fixture_state
from models.native_diagnostic import save_tiny_fixture
from models.training import rng_state


class SmallFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 3)
        self.predictor = nn.Linear(3, 3)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False).eval()
        self.register_buffer('pixel_mean', torch.tensor([.485,.456,.406]))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logical(state):
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode())
        h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def test_one_checkpoint_restores_all_components_and_buffers_after_different_construction(tmp_path):
    torch.manual_seed(3)
    source = SmallFixture()
    path = tmp_path/'initial.pt'
    identity = save_fixture_state(source,path)
    torch.manual_seed(91)
    destination = SmallFixture()
    destination.pixel_mean.add_(1)
    assert state_digest(source) != state_digest(destination)
    parameters = [id(p) for p in destination.parameters()]
    rng = torch.get_rng_state().clone()
    assert load_fixture_state(destination,path,identity['sha256']) == identity
    assert torch.equal(torch.get_rng_state(),rng)
    assert state_digest(destination) == state_digest(source) == identity['state_sha256']
    assert parameters == [id(p) for p in destination.parameters()]
    assert not destination.target_encoder.training and not any(p.requires_grad for p in destination.target_encoder.parameters())


@pytest.mark.parametrize('defect',['hash','missing_target','shape','dtype','nonfinite','logical_hash','unexpected','kind'])
def test_wrong_or_malformed_fixture_rejects_before_any_model_mutation(tmp_path,defect):
    path = tmp_path/'initial.pt'
    identity = save_fixture_state(SmallFixture(),path)
    destination = SmallFixture()
    before = state_digest(destination)
    if defect == 'hash':
        with path.open('ab') as out:
            out.write(b'changed')
        expected = identity['sha256']
    else:
        payload = torch.load(path,weights_only=True)
        state = payload['state']
        if defect == 'missing_target': del state['target_encoder.bias']
        elif defect == 'shape': state['encoder.weight'] = state['encoder.weight'][:1]
        elif defect == 'dtype': state['encoder.weight'] = state['encoder.weight'].double()
        elif defect == 'nonfinite': state['encoder.weight'][0,0] = float('nan')
        elif defect == 'unexpected': state['extra.weight'] = torch.ones(1)
        elif defect == 'kind': payload['kind'] = 'pretrained-checkpoint'
        payload['state_sha256'] = '0'*64 if defect == 'logical_hash' else logical(state)
        torch.save(payload,path)
        expected = digest(path)
    with pytest.raises(ValueError):
        load_fixture_state(destination,path,expected)
    assert state_digest(destination) == before


def test_missing_pin_missing_file_and_symlink_never_fall_back_to_random_weights(tmp_path):
    model = SmallFixture()
    before = state_digest(model)
    path = tmp_path/'initial.pt'
    identity = save_fixture_state(model,path)
    link = tmp_path/'link.pt'
    link.symlink_to(path)
    for filename,pin in [(path,None),(tmp_path/'missing.pt',identity['sha256']),(link,identity['sha256'])]:
        with pytest.raises(ValueError):
            load_fixture_state(model,filename,pin)
    assert state_digest(model) == before


def test_fixture_file_size_limit_and_existing_file_are_preserved(tmp_path):
    path = tmp_path/'initial.pt'
    identity = save_fixture_state(SmallFixture(),path)
    with pytest.raises(FileExistsError): save_fixture_state(SmallFixture(),path)
    assert digest(path) == identity['sha256']
    oversized = tmp_path/'oversized.pt'
    with oversized.open('wb') as out:
        out.truncate(16*2**20+1)
    with pytest.raises(ValueError,match='bounded'):
        read_fixture_state(oversized,'0'*64)


@pytest.mark.parametrize('defect',['target_grad','target_train','policy'])
def test_fixture_checkpoint_cannot_replace_a_changed_runtime_or_unfrozen_target(tmp_path,defect):
    source = SmallFixture()
    path = tmp_path/'initial.pt'
    identity = save_fixture_state(source,path)
    model = SmallFixture()
    if defect == 'target_grad': model.target_encoder.requires_grad_(True)
    elif defect == 'target_train': model.target_encoder.train()
    else: model._diagnostic_execution_policy = 'shared-fp64-native'
    before = state_digest(model)
    with pytest.raises(ValueError):load_fixture_state(model,path,identity['sha256'])
    assert state_digest(model) == before


@pytest.mark.parametrize('fail',[False,True])
def test_initial_fixture_capture_preserves_caller_rng_and_threads(upstream,tmp_path,monkeypatch,fail):
    torch.set_num_threads(2)
    before = rng_state('cpu')
    if fail:
        def reject(*args): raise RuntimeError('deliberate fixture write error')
        monkeypatch.setattr('models.native_diagnostic.save_fixture_state',reject)
        with pytest.raises(RuntimeError,match='deliberate'):
            save_tiny_fixture(upstream,tmp_path/'initial.pt')
    else:
        identity = save_tiny_fixture(upstream,tmp_path/'initial.pt')
        assert read_fixture_state(tmp_path/'initial.pt',identity['sha256'])[1] == identity
    after = rng_state('cpu')
    assert before['python'] == after['python'] and before['numpy'] == after['numpy']
    assert torch.equal(before['torch_cpu'],after['torch_cpu']) and torch.get_num_threads() == 2


@pytest.mark.parametrize('script',['check_native_model.py','check_precision.py'])
@pytest.mark.parametrize('misuse',['unbound_worker','pretrained_override'])
def test_cli_rejects_unbound_fixture_or_pretrained_override_before_any_result(tmp_path,script,misuse):
    from pathlib import Path
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    result_path = tmp_path/'must-not-exist.json'
    flags = ['--diagnostic']
    if misuse == 'unbound_worker':
        flags += ['--tiny','--device','cpu','--worker','--result',str(result_path)]
    else:
        checkpoint = tmp_path/'checkpoint.pt'
        checkpoint.write_bytes(b'not loaded')
        flags += ['--checkpoint',str(checkpoint),'--clips',str(tmp_path),'--manifest',str(checkpoint),
                  '--observed',str(checkpoint),'--reference-run',str(tmp_path)] if script == 'check_precision.py' else [
                  '--checkpoint',str(checkpoint),'--clips',str(tmp_path),'--manifest',str(checkpoint)]
        flags += ['--initial-fixture',str(checkpoint),'--initial-fixture-sha256','0'*64]
    result = subprocess.run([sys.executable,str(root/'scripts'/script),*flags],capture_output=True,text=True,timeout=15)
    assert result.returncode == 2 and 'fixture' in result.stderr.lower()
    assert not result_path.exists()
