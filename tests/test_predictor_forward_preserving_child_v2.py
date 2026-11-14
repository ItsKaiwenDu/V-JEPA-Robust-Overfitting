"""Exercise the exact cold child entry missed by v1; no model/proof/SSH work."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import pytest

from models import predictor_forward_preserving_operator_v1 as v1
from models import predictor_forward_preserving_operator_v2 as op
from models import predictor_forward_preserving_client_v2 as client

ROOT=Path(__file__).resolve().parents[1]
RAW=Path(os.environ['VJEPA_FORWARD_V2_EVIDENCE'])
HANDOFF=Path(os.environ['VJEPA_FORWARD_V2_HANDOFF'])
SAVED=ROOT/'runs/pretrained/diagnostic/predictor-forward-preserving-handoff-v1-20261114/linux10603-final/attended-elmh_fic/saved'


class BeforeScience(Exception):
    pass


def setup_child(tmp_path, monkeypatch, *, previous=False):
    operator=v1 if previous else op
    source=ROOT/'scripts'/('capture_predictor_forward_preserving_v1.py' if previous else 'capture_predictor_forward_preserving_v2.py')
    c=json.loads((SAVED/'control.json' if previous else HANDOFF/'CONTROL.json').read_text())
    # A distinct synthetic control cannot authorize the real final handoff.
    c['local_handoff_directory']=str(tmp_path/'SYNTHETIC-COLD-ENTRY-ONLY')
    sha=operator.hashlib.sha256(operator.old.json_bytes(c)).hexdigest()
    assert sha!=operator.old.digest(SAVED/'control.json')
    if not previous:assert sha!=operator.old.digest(HANDOFF/'CONTROL.json')
    scratch=tmp_path/(operator.PREFIX+'cold-fixture');overlay=scratch/'overlay'
    scratch.mkdir(mode=0o700);overlay.mkdir(mode=0o700)
    for name in operator.REMOTE_NAMES:
        data=operator.INIT if name=='models/__init__.py' else (ROOT/name).read_bytes()
        path=overlay/name;path.parent.mkdir(mode=0o700,parents=True,exist_ok=True);path.write_bytes(data)
    shutil.copytree(SAVED/'code',scratch/'code')
    operator.old.write_json(scratch/'control.json',c)
    auth=operator.authorization_template(c,sha,'Synthetic local cold-entry fixture only; not university authorization.')
    operator.old.write_json(scratch/'authorization.json',auth)
    operator.old.write_json(scratch/'session.json',{'complete':False,'tasks':[],'deadline_unix':3400.})
    operator.old.write_json(scratch/'worker-plan-0.json',{'index':0,'control_sha256':sha,'deadline_unix':1050.})
    spec=importlib.util.spec_from_file_location('local_child_fixture_'+('v1' if previous else 'v2'),source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.overlay=overlay
    def paths(*args):
        return tmp_path if args==('/var/tmp',) else Path(*args)
    module.Path=paths
    module.time=SimpleNamespace(time=lambda:1000.)
    timers=[]
    module.signal=SimpleNamespace(SIGALRM=1,SIG_DFL=0,ITIMER_REAL=0,signal=lambda *a:None,setitimer=lambda *a:timers.append(a))
    monkeypatch.setattr(sys,'argv',[str(source),'--scratch',str(scratch),'--control-sha256',sha,'--index','0'])
    calls=[]
    actual=operator.validate_authorization
    def authorize(a,h,control=None):
        calls.append({'passed_bound_control':control is not None,'control_sha256':h})
        return actual(a,h,control)
    # Patch only this test module's operator view, never frozen operator globals.
    view=SimpleNamespace(**{name:getattr(operator,name) for name in ('PREFIX','old','validate_control','check_worker')})
    view.validate_authorization=authorize
    def stop_before_science(*args):
        calls.append({'reached_live_GPU_boundary':True})
        raise BeforeScience('Intentional local stop before hardware/scientific imports.')
    view.base=SimpleNamespace(gpu_gate=stop_before_science)
    module.op=view
    return module,scratch,c,sha,calls,timers


def test_reproduce_real_v1_missing_control_refusal_before_science(tmp_path,monkeypatch):
    child,scratch,c,sha,calls,timers=setup_child(tmp_path,monkeypatch,previous=True)
    with pytest.raises(ValueError,match='Bound control is required'):
        child.main()
    assert calls==[{'passed_bound_control':False,'control_sha256':sha}] and not timers
    assert not (scratch/'results').exists()


def test_exact_v2_child_passes_control_then_reaches_live_guard_before_science(tmp_path,monkeypatch):
    child,scratch,c,sha,calls,timers=setup_child(tmp_path,monkeypatch)
    with pytest.raises(BeforeScience):
        child.main()
    assert calls==[{'passed_bound_control':True,'control_sha256':sha},{'reached_live_GPU_boundary':True}]
    assert timers==[(0,50.)] and not (scratch/'results').exists()
    (RAW/'exact-child-entry-check.json').write_text(json.dumps({'passed':True,'actual_child_entry_exercised':True,
        'bound_control_passed':True,'remaining_worker_seconds':50,'stopped_before_GPU_or_science':True,
        'new_proof_model_or_SSH_work':False},sort_keys=True,indent=2)+'\n')


@pytest.mark.parametrize('fault,expected',[
    ('consumed-v1-authorization','new exact scoped'),('changed-control','Control changed'),
    ('missing-authorization','authorization.json'),('changed-source','Operator source changed'),
    ('changed-packet','Original source/input changed'),('expired-clock','clock invalid'),
    ('wrong-order','Worker order/phase changed')])
def test_exact_child_still_refuses_invalid_scope_source_or_clock_before_science(tmp_path,monkeypatch,fault,expected):
    child,scratch,c,sha,calls,timers=setup_child(tmp_path,monkeypatch)
    if fault=='consumed-v1-authorization':
        (scratch/'authorization.json').write_bytes((SAVED/'authorization.json').read_bytes())
    elif fault=='changed-control':
        with (scratch/'control.json').open('ab') as f:f.write(b' ')
    elif fault=='missing-authorization':
        (scratch/'authorization.json').unlink()
    elif fault=='changed-source':
        with (scratch/'overlay/models/predictor_forward_preserving_operator_v2.py').open('ab') as f:f.write(b' ')
    elif fault=='changed-packet':
        with (scratch/'code/PROFILE-HANDOFF.json').open('ab') as f:f.write(b' ')
    elif fault=='expired-clock':
        plan=json.loads((scratch/'worker-plan-0.json').read_text());plan['deadline_unix']=999.
        (scratch/'worker-plan-0.json').write_bytes(op.old.json_bytes(plan))
    else:
        state=json.loads((scratch/'session.json').read_text());state['tasks']=[{}]
        (scratch/'session.json').write_bytes(op.old.json_bytes(state))
    with pytest.raises((ValueError,FileNotFoundError),match=expected):
        child.main()
    assert not any(x.get('reached_live_GPU_boundary') for x in calls)
    assert not (scratch/'results').exists()


def functions(path):
    return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}


def test_supervisor_changes_only_entry_path_and_keeps_frozen_bounds():
    a=functions(ROOT/'models/predictor_forward_preserving_operator_v1.py')['_execute']
    b=functions(ROOT/'models/predictor_forward_preserving_operator_v2.py')['_execute']
    class EntryPath(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='overlay/scripts/capture_predictor_forward_preserving_v1.py':
                return ast.copy_location(ast.Constant(value='overlay/scripts/capture_predictor_forward_preserving_v2.py'),n)
            return n
    assert ast.dump(EntryPath().visit(a),include_attributes=False)==ast.dump(b,include_attributes=False)
    for name in ('validate_control','authorization_template','validate_authorization','remote','main'):
        assert getattr(op,name).__code__ is getattr(v1,name).__code__
        assert getattr(op,name).__globals__ is not getattr(v1,name).__globals__
    assert op.inventory is v1.inventory and op.storage is v1.storage and op.verify_outputs is v1.verify_outputs


def test_final_v2_unarmed_and_old_claim_unchanged(monkeypatch):
    c,sha=client.check_local(HANDOFF)
    assert not (HANDOFF/'AUTHORIZATION.json').exists() and not (HANDOFF/'ATTEMPT.json').exists()
    claim=ROOT/'runs/pretrained/diagnostic/predictor-forward-preserving-handoff-v1-20261114/linux10603-final/ATTEMPT.json'
    assert json.loads(claim.read_text())['consumed'] is True
    def forbidden(*a,**k):raise AssertionError('Unarmed v2 reached SSH.')
    with pytest.raises(ValueError,match='Unarmed'):
        client.run(HANDOFF,popen=forbidden)
    assert not (HANDOFF/'ATTEMPT.json').exists()


def test_v1_authorization_cannot_admit_v2_control():
    c,sha=client.check_local(HANDOFF)
    old=json.loads((SAVED/'authorization.json').read_text())
    with pytest.raises(ValueError,match='new exact scoped'):
        op.validate_authorization(old,sha,c)
