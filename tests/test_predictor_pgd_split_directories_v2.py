"""New tiny filesystem integration of setup and frozen exclusive mkdir statements.

No scientific function/module, Torch/NumPy, model, tensor, GPU, subprocess, timer,
proof or remote operation executes. The only copied worker operation is mkdir.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_split_directories_v2 as candidate
from models import predictor_pgd_split_bootstrap_v1 as boot
from models import predictor_pgd_split_scratch_v1 as scratch
from models import predictor_pgd_split_store_v1 as store

ROOT=Path(__file__).resolve().parents[1]
SAVED=ROOT/'runs/pretrained/diagnostic/predictor-pgd-split-outer-v1-20261115/handoff-parent/linux10603-final-unarmed/attended-session/saved'

@pytest.fixture(scope='module')
def context():
    cb=(SAVED/'control.json').read_bytes();c,p,sha=boot.control(cb)
    ib=(SAVED/'installation.json').read_bytes()
    layout=scratch.Layout(p['scratch_contract_text'].encode(),p['base_text'].encode(),p['design_text'].encode(),ib,hashlib.sha256(ib).hexdigest())
    return NS(c=c,p=p,sha=sha,layout=layout)


def frozen_claim(role):
    if role in ('numerical','objective'):
        tree=ast.parse((SAVED/'overlay/models/_pgd_split_numerical_v1.py').read_bytes())
        name='folder.mkdir' if role=='numerical' else 'self.folder.mkdir'
    elif role=='boundary':
        tree=ast.parse((SAVED/'overlay/models/_pgd_split_observer_v1.py').read_bytes());name='folder.mkdir'
    else:
        tree=ast.parse((SAVED/'overlay/models/predictor_pgd_split_probe_job_v1.py').read_bytes());name='self.folder.mkdir'
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)==name]
    assert len(nodes)==1
    expression=copy.deepcopy(nodes[0])
    assert isinstance(expression.value.func,ast.Attribute) and expression.value.func.attr=='mkdir'
    return compile(ast.fix_missing_locations(ast.Module(body=[expression],type_ignores=[])),'<FROZEN-SINGLE-MKDIR-ONLY>','exec')


def claim(root,role,name):
    folder=root/name
    if role in ('native','forward-preserving'):
        # These two common ancestors are created by the frozen component job.
        for parent in (root/'capture/results',root/'capture/results/components'):
            try:parent.mkdir(mode=0o700)
            except FileExistsError:pass
    kind='objective' if role.startswith('objective-') else role
    exec(frozen_claim(kind),{'folder':folder,'self':NS(folder=folder)})


@pytest.mark.parametrize('role',['numerical','boundary','native','forward-preserving'])
def test_saved_v1_setup_reproduces_each_exclusive_worker_directory_collision(context,tmp_path,role):
    root=tmp_path/'EXPLICIT-V1-DIRECTORY-REPRODUCTION';root.mkdir(mode=0o700)
    tree=ast.parse((ROOT/'models/predictor_pgd_split_outer_v1.py').read_bytes())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_tree')
    loops=[n for n in fn.body if isinstance(n,ast.For) and 'layout.directories' in ast.unparse(n.iter)]
    assert len(loops)==1
    # Only the saved setup's mkdir loop executes, with no preflight, installer,
    # signal, worker, array or allocation call in this AST fragment.
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(loops[0])],type_ignores=[])),'<V1-SETUP-DIRECTORY-LOOP-ONLY>','exec'),
         {'root':root,'layout':context.layout,'PurePosixPath':PurePosixPath,'os':os,'stat':stat,'binding':scratch.identity_binding(root)})
    target=candidate.WORKER_CLAIMS[role]
    before=(root/target).stat()
    with pytest.raises(FileExistsError):claim(root,role,target)
    assert (root/target).stat().st_ino==before.st_ino


def test_minimal_plan_leaves_all_twelve_frozen_claims_fresh_and_preserves_original_store_catalogue(context,tmp_path):
    root=tmp_path/'EXPLICIT-V2-TINY-DIRECTORIES';root.mkdir(mode=0o700)
    plan=candidate.plan(context.layout)
    assert not plan['authorized'] and not plan['target_helper_ready']
    for name in plan['setup_directories']:
        if name!='.':(root/name).mkdir(mode=0o700)
    for name in context.layout.assets:
        assert (root/str(PurePosixPath(name).parent)).is_dir()
    for role,name in plan['worker_claim_directories'].items():
        assert not (root/name).exists()
        claim(root,role,name)
        assert stat.S_IMODE((root/name).stat().st_mode)==0o700
    inv=store.Inventory(root/'capture',base_bytes=context.p['base_text'].encode(),design_bytes=context.p['design_text'].encode())
    report=inv.scan();assert report['files']=={} and len(inv.spec)==419
    guard=scratch.Guard(root,scratch.identity_binding(root),context.layout)
    whole,files,dirs=guard.scan(phase='setup')
    assert files=={} and not whole['scientific_qualified']
    assert set(dirs)<=context.layout.directories
    (tmp_path/'CANDIDATE-DIRECTORY-PLAN.json').write_bytes(boot.encoded(dict(plan,
        origin='EXPLICIT TINY LOCAL DIRECTORIES; ONLY FROZEN MKDIR STATEMENTS EXECUTED',actual_worker_execution=False)))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


def test_existing_worker_owned_directory_is_still_preserved_and_refused(context,tmp_path):
    root=tmp_path/'EXPLICIT-PRESERVE';root.mkdir(mode=0o700)
    plan=candidate.plan(context.layout)
    for name in plan['setup_directories']:
        if name!='.':(root/name).mkdir(mode=0o700)
    folder=root/candidate.WORKER_CLAIMS['numerical'];folder.mkdir(mode=0o700)
    (folder/'PRESERVED.txt').write_text('INERT OLD EVIDENCE')
    with pytest.raises(FileExistsError):claim(root,'numerical',candidate.WORKER_CLAIMS['numerical'])
    assert (folder/'PRESERVED.txt').read_text()=='INERT OLD EVIDENCE'


@pytest.mark.parametrize('fault',['asset-count','result-count','capture-asset','absolute-asset','traversal-asset','unknown-parent','missing-worker'])
def test_directory_candidate_refuses_changed_original_catalogue(context,fault):
    layout=copy.deepcopy(context.layout)
    if fault=='asset-count':layout.assets.pop(next(iter(layout.assets)))
    elif fault=='result-count':layout.result.pop(next(iter(layout.result)))
    elif fault=='missing-worker':layout.directories.remove(candidate.WORKER_CLAIMS['boundary'])
    else:
        value=layout.assets.pop(next(iter(layout.assets)))
        key={'capture-asset':'capture/results/INJECTED.bin','absolute-asset':'/INJECTED.bin',
             'traversal-asset':'code/../INJECTED.bin','unknown-parent':'UNKNOWN/INJECTED.bin'}[fault]
        layout.assets[key]=value
    with pytest.raises(ValueError):candidate.plan(layout)
