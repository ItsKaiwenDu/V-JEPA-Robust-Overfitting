"""Cold packaged-entrypoint regression. No oracle, worker, SSH or allocation.

The original v2 files stay frozen. A candidate entrypoint exists only in each
temporary source-only package, to qualify the minimal path correction locally.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from models import cpu_oracle_capture_v2 as operator
from models.cpu_oracle_capture_client_v2 import sources

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/pretrained/diagnostic/cpu-oracle-capture-v2-20261104/linux10603'
CONTROL_SHA='64364f84133035c6fff3d5fbed87c64f7a8ec92203e202ec20d70456fe300484'

CANDIDATE_INSERTION=(
    "    if a.packet.is_symlink() or a.packet.resolve() != (ROOT.parent/'code').resolve():\n"
    "        p.error('Packet must be the original sibling code directory.')\n"
    "    sys.path.insert(1, str(a.packet.resolve()))\n")

HARNESS=r'''
import builtins,json,os,runpy,subprocess,sys,traceback
from pathlib import Path
entry,packet,scratch,sha=sys.argv[1:]
sys.argv=[entry,'--packet',packet,'--scratch',scratch,'--control-sha256',sha,
    '--launcher-started','0','--upstream',str(Path(scratch)/'unused-upstream')]
counts={'worker_dispatch_blocked':False,'subprocess_attempts':0,'oracle_calls':0}
def denied(*a,**k):
    counts['subprocess_attempts']+=1
    raise AssertionError('Import-only harness refuses subprocess creation.')
subprocess.Popen=denied
class ImportBoundaryReached(Exception):pass
def blocked_worker(*a,**k):
    counts['worker_dispatch_blocked']=True
    raise ImportBoundaryReached()
real_import=builtins.__import__
def guarded_import(name,globals=None,locals=None,fromlist=(),level=0):
    result=real_import(name,globals,locals,fromlist,level)
    if name=='models.cpu_oracle_capture_runtime_v2' and 'run_workers' in fromlist:
        sys.modules[name].run_workers=blocked_worker
    return result
builtins.__import__=guarded_import
def guard_calls(frame,event,arg):
    if event=='call' and frame.f_code.co_name in ('fp64_oracle','run_workers','worker'):
        if str(Path(scratch)) in frame.f_code.co_filename:
            counts['oracle_calls']+=1
            raise AssertionError('No original numerical entry point may execute.')
sys.setprofile(guard_calls)
try:
    runpy.run_path(entry,run_name='__main__')
except ImportBoundaryReached:
    import attacks,datasets,models,torch
    origins={n:str(Path(m.__file__).resolve()) for n,m in (('attacks',attacks),('datasets',datasets),('models',models))}
    assert origins['attacks']==str((Path(packet)/'attacks/__init__.py').resolve())
    assert origins['datasets']==str((Path(packet)/'datasets/__init__.py').resolve())
    assert origins['models']==str((Path(scratch)/'overlay/models/__init__.py').resolve())
    assert torch.cuda.is_initialized() is False
    print(json.dumps({'outcome':'runtime_imported_dispatch_blocked','origins':origins,
        'CUDA_initialized':False,'numerical_results':False,**counts},sort_keys=True))
except ModuleNotFoundError as exc:
    traceback.print_exc()
    print(json.dumps({'outcome':'missing_import','missing_module':exc.name,
        'numerical_results':False,**counts},sort_keys=True))
finally:
    sys.setprofile(None)
'''


@pytest.fixture
def packaged(tmp_path):
    control=json.loads((BASE/'CONTROL.json').read_text())
    assert operator.old.digest(BASE/'CONTROL.json')==CONTROL_SHA
    packet=tmp_path/'code';packet.mkdir()
    # Source-only restore. The regression imports definitions, never fixtures.
    for name,item in control['archive_manifest']['files'].items():
        if not name.endswith('.py'):continue
        original=ROOT/name
        assert operator.old.digest(original)==item['sha256']
        target=packet/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(original.read_bytes())
    overlay=tmp_path/'overlay'
    for name,data in sources(control).items():
        target=overlay/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    (tmp_path/'operator-control.json').write_bytes((BASE/'CONTROL.json').read_bytes())
    harness=tmp_path/'harness.py';harness.write_text(HARNESS)
    neutral=tmp_path/'neutral';neutral.mkdir()
    return packet,overlay,tmp_path,harness,neutral


def run_case(packaged,*,candidate=False,wrong_packet=False,symlink=False):
    packet,overlay,folder,harness,neutral=packaged
    entry=overlay/'scripts/capture_cpu_reference_v2.py'
    if candidate:
        text=entry.read_text();needle='    from models.cpu_oracle_capture_v2 import old\n'
        assert text.count(needle)==1
        entry=entry.with_name('capture_cpu_reference_candidate.py')
        entry.write_text(text.replace(needle,CANDIDATE_INSERTION+needle,1))
    if wrong_packet:packet=neutral
    if symlink:
        alias=folder/'code-alias';alias.symlink_to(packet,target_is_directory=True);packet=alias
    before={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}
    result=subprocess.run([sys.executable,'-I','-B',str(harness),str(entry),str(packet),str(folder),CONTROL_SHA],
        cwd=neutral,capture_output=True,text=True,timeout=30,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'})
    after={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}
    assert before==after
    assert not (folder/'results').exists() and not (folder/'AUTHORIZATION.json').exists()
    return result


def test_exact_v2_packaged_cli_reproduces_saved_missing_attacks(packaged):
    result=run_case(packaged)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout.strip().splitlines()[-1])
    assert value['outcome']=='missing_import' and value['missing_module']=='attacks'
    assert not value['worker_dispatch_blocked'] and value['oracle_calls']==value['subprocess_attempts']==0
    assert "ModuleNotFoundError: No module named 'attacks'" in result.stderr


def test_candidate_packet_root_resolves_imports_before_blocked_worker_dispatch(packaged):
    result=run_case(packaged,candidate=True)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout.strip().splitlines()[-1])
    assert value['outcome']=='runtime_imported_dispatch_blocked' and value['worker_dispatch_blocked']
    assert not value['CUDA_initialized'] and not value['numerical_results']
    assert value['oracle_calls']==value['subprocess_attempts']==0


@pytest.mark.parametrize('defect',['wrong_packet','symlink'])
def test_candidate_refuses_a_different_or_aliased_packet_root(packaged,defect):
    result=run_case(packaged,candidate=True,**{defect:True})
    assert result.returncode==2
    assert 'Packet must be the original sibling code directory.' in result.stderr
