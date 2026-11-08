"""Actual one-backward native CPU capture followed by four fresh components."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import pytest
from models import predictor_sdpa_target_v1 as target
from models import predictor_sdpa_handoff_operator_v1 as op
from models import predictor_sdpa_component_v1 as core
from models import predictor_sdpa_component_records_v1 as rec

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'runs/pretrained/diagnostic/predictor-sdpa-observer-v1-20261106/pytest-third/native-sdpa-observer0/creator-0'
INITIAL=ROOT/'runs/pretrained/diagnostic/predictor-eot8-10603-v1-20261105/cpu-four-worker/initial.pt'
INITIAL_SHA='989665a5a4f8ea90600214be5536b4600c37fc448108a5bd177b3416e5ae0338'
CONTROL='e'*64


def write(path,value):
    path.write_bytes(rec.encoded(value));path.chmod(0o600)


def capture_child(folder,fault='none'):
    script='''import sys,os
from pathlib import Path
from types import SimpleNamespace
os.umask(0o077)
if sys.argv[2]=='backward':
 from models.predictor_eot8_owned_v1 import OwnedBackward
 old=OwnedBackward._root
 def bad(self,g):old(self,g);raise RuntimeError('handoff-native-primary-backward-error')
 OwnedBackward._root=bad
elif sys.argv[2]=='one-ULP':
 import torch
 from models.predictor_eot8_owned_v1 import OwnedBackward
 old=OwnedBackward.run
 def bad(self):
  g,=old(self);g=g.clone();g.reshape(-1)[0]=torch.nextafter(g.reshape(-1)[0],torch.tensor(float('inf')));return (g,)
 OwnedBackward.run=bad
from models.predictor_sdpa_target_v1 import run_capture
run_capture(SimpleNamespace(output=Path(sys.argv[1]),tiny=True,device='cpu',control_sha256=sys.argv[3],
 upstream=Path(sys.argv[4]),initial=Path(sys.argv[5]),initial_sha=sys.argv[6],reference=Path(sys.argv[7])))'''
    p=subprocess.run([sys.executable,'-B','-c',script,str(folder),fault,CONTROL,str(ROOT/'external/vjepa2'),str(INITIAL),INITIAL_SHA,str(REFERENCE)],
        cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
    (folder.parent/(folder.name+'.log')).write_bytes(p.stdout);return p


@pytest.fixture(scope='module')
def cohort(tmp_path_factory):
    root=tmp_path_factory.mktemp('native-handoff');results=root/'results';results.mkdir(mode=0o700)
    p=capture_child(results/'capture');assert p.returncode==0,p.stdout.decode()[-7000:]
    components=results/'components';components.mkdir(mode=0o700)
    plan=target.component_plan(ROOT,results/'capture',components,control_sha256=CONTROL,local_tiny=True)
    path=components/'component-plan.json';write(path,plan);ps=rec.sha(path)
    for i in range(4):
        code='''import os,sys,time
from pathlib import Path
from types import SimpleNamespace
os.umask(0o077)
from models.predictor_sdpa_component_v1 import run
run(SimpleNamespace(plan=Path(sys.argv[1]),plan_sha256=sys.argv[2],index=int(sys.argv[3]),output=Path(sys.argv[4]),device='cpu',deadline_unix=time.time()+120))'''
        p=subprocess.run([sys.executable,'-B','-c',code,str(path),ps,str(i),str(components/str(i))],cwd=ROOT,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
        (components/(str(i)+'.log')).write_bytes(p.stdout);assert p.returncode==0,p.stdout.decode()[-7000:]
    w=json.loads((results/'capture/worker.json').read_text())
    return SimpleNamespace(root=root,context={'contract':{'approved_baseline':w['baseline'],
        'full_native_anchor_logical_sha256':w['full_native_anchor_logical_sha256']}},plan=plan,plan_sha256=ps)


def test_actual_single_native_backward_and_four_fresh_components_preserve_original_values(cohort):
    root=cohort.root;cap=op.verify_capture(root/'results/capture',CONTROL,cohort.context,tiny=True)
    results=[op.verify_task(root,r,CONTROL,cohort.context,tiny=True) for r in op.ROLES]
    assert len({r['pid'] for r in results})==5
    w=json.loads((root/'results/capture/worker.json').read_text());prior=json.loads((REFERENCE/'worker.json').read_text())
    assert not w['unobserved_before_after_work_performed'] and w['complete'] and w['original_value_replay_exact']
    assert w['baseline']==prior['baseline'] and set(w['artifacts'])=={'loss','input','targets'}
    assert w['route_restored'] and w['observer_restored'] and w['model_unchanged']
    for n in ('loss','input'):assert w['artifacts'][n]['source']['sha256']==prior['artifacts'][n]['source']['sha256']
    assert w['scope_check']['forward_calls']==w['scope_check']['backward_calls']==64
    assert w['scope_check']['owners']==4 and w['graph_scope']['hook_removed']
    _,_,_,identity=core.capture(cohort.plan)
    children=[core.load_worker(root/'results/components'/str(i),cohort.plan,cohort.plan_sha256,identity,i) for i in range(4)]
    assert [x['oracle_build_calls'] for x in children]==[1,0,0,0]
    assert all(x['profile_replay_exact'] and x['backend_evidence']['CPU_only'] for x in children)
    assert all(not x['backend_evidence']['CUDA_forward_and_backward_qualified'] for x in children)
    write(root/'five-process-CPU-summary.json',{'local_only':True,'new_university_work':False,'native':cap,'tasks':results,
        'source_original_exact':True,'full_native_backwards':1,'fresh_processes':5,'canonical_FP64_builds':1,
        'candidate_adopted':False,'kernel_or_performance_qualified':False})


@pytest.mark.parametrize('fault',('backward','one-ULP'))
def test_primary_failure_and_subtolerance_native_byte_change_stop_before_components(tmp_path,fault):
    p=capture_child(tmp_path/'capture',fault);assert p.returncode==1,p.stdout.decode()[-7000:]
    w=json.loads((tmp_path/'capture/worker.json').read_text())
    assert not w['complete'] and not (tmp_path/'capture/capture-complete.json').exists()
    assert (tmp_path/'capture/capture-failure.json').is_file() and w['route_restored'] and w['observer_restored']
    if fault=='backward':
        assert 'handoff-native-primary-backward-error' in w['error'] and set(w['artifacts'])=={'targets','loss'}
        cap=core.native.load(tmp_path/'capture/boundary',control_sha256=CONTROL,require_complete=False)
        assert not cap['capture_complete'] and set(cap['artifacts'])=={'q','k','v','output'}
    else:
        assert 'exactly' in w['error'] and w['full_native_checks']['input']['passed']
        assert set(w['artifacts'])=={'targets','loss','input'}
    assert not (tmp_path/'components').exists()


@pytest.mark.parametrize('fault',('marker','control','native-anchor','model','baseline','mask-indices','boundary-bytes','adopted','time'))
def test_capture_receipt_refuses_rehashed_or_missing_required_controls(cohort,tmp_path,fault):
    dest=tmp_path/'capture';shutil.copytree(cohort.root/'results/capture',dest)
    w=json.loads((dest/'worker.json').read_text())
    if fault=='marker':(dest/'capture-complete.json').unlink()
    elif fault=='control':w['control_sha256']='a'*64
    elif fault=='native-anchor':w['artifacts']['input']['source']['sha256']='a'*64
    elif fault=='model':w['model_unchanged']=False
    elif fault=='baseline':w['baseline']['seeds'][0]+=1
    elif fault=='mask-indices':
        path=dest/'mask-indices.json';v=json.loads(path.read_text());v[0][0]['context'][0][0]+=1;write(path,v)
        w['mask_indices'].update(bytes=path.stat().st_size,sha256=rec.sha(path))
    elif fault=='boundary-bytes':(dest/'boundary/q.pt').write_bytes(b'changed actual Q')
    elif fault=='adopted':w['native_candidate_adopted']=True
    else:w['wall_seconds']=421
    if fault not in ('marker','boundary-bytes'):
        write(dest/'worker.json',w)
        write(dest/'capture-complete.json',{'kind':w['kind'],'control_sha256':CONTROL,
            'header':{'file':'worker.json','bytes':(dest/'worker.json').stat().st_size,'sha256':rec.sha(dest/'worker.json')},'complete':True})
    with pytest.raises((ValueError,FileNotFoundError)):op.verify_capture(dest,CONTROL,cohort.context,tiny=True)


def test_local_native_capture_is_not_a_live_pretrained_reference(cohort):
    with pytest.raises(ValueError):op.verify_capture(cohort.root/'results/capture',CONTROL,cohort.context,tiny=False)
    with pytest.raises(ValueError):target.run_component(SimpleNamespace(device='cuda',output=Path('unused')))
    assert not Path('unused').exists()


def test_target_host_rejected_before_scientific_imports_or_output(tmp_path):
    script='''import importlib.abc,sys
from pathlib import Path
from types import SimpleNamespace
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','numpy','timm'):raise AssertionError('Scientific import before host guard')
sys.meta_path.insert(0,Deny())
from models.predictor_sdpa_target_v1 import run_capture
run_capture(SimpleNamespace(tiny=False,device='cuda',output=Path(sys.argv[1])))'''
    p=subprocess.run([sys.executable,'-B','-c',script,str(tmp_path/'capture')],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
    assert p.returncode==1 and b'Exact authorized target host/runtime required' in p.stderr and not (tmp_path/'capture').exists()


def test_target_component_configures_precision_before_any_CUDA_context_setup(tmp_path,monkeypatch):
    import torch
    import models.execution as execution
    calls=[];ref=tmp_path/'reference';plan={'local_tiny':False,'reference_dir':str(ref)}
    monkeypatch.setattr(target.platform,'node',lambda:'linux10603');monkeypatch.setattr(target.platform,'python_version',lambda:'3.11.17')
    monkeypatch.setattr(core,'read',lambda *a:plan);monkeypatch.setattr(core,'validate_plan',lambda p:p)
    monkeypatch.setattr(core,'capture',lambda *a:(None,None,None,None));monkeypatch.setattr(core,'previous',lambda *a:[])
    monkeypatch.setattr(rec,'sha',lambda *a:'bound')
    monkeypatch.setattr(torch,'__version__','2.8.0+cu128');monkeypatch.setattr(torch.version,'cuda','12.8')
    monkeypatch.setattr(execution,'configure_execution',lambda *a,**k:calls.append('configure'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:calls.append('probe') or True)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *a:SimpleNamespace(total_memory=8*2**30))
    def allocator(*a):assert calls[0]=='configure';calls.append('allocator')
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',allocator)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:calls.append('reset'))
    def stopped(*a,**k):raise RuntimeError('stop-before-scientific-body-local-fixture')
    monkeypatch.setattr(core,'_execute',stopped)
    with pytest.raises(RuntimeError,match='stop-before-scientific-body'):
        target.run_component(SimpleNamespace(device='cuda',plan=tmp_path/'plan.json',plan_sha256='bound',index=0,output=tmp_path/'0'))
    assert calls==['configure','probe','allocator','reset']


def copy_component(cohort,dest,index):
    """Rebind a COPY of actual CPU arrays for transport-only simulation."""
    source=cohort.root/'results/components';components=Path(dest)/'results/components'
    plan=json.loads((components/'component-plan.json').read_text());ps=rec.sha(components/'component-plan.json')
    def resign(folder,name,w):
        write(folder/name,w)
        write(folder/'complete.json',{'kind':w['kind'],'plan_sha256':ps,'header':{'file':name,'bytes':(folder/name).stat().st_size,'sha256':rec.sha(folder/name)},'complete':True})
    if index==0:
        shutil.copytree(source/'reference',components/'reference');w=json.loads((components/'reference/reference.json').read_text());w['plan_sha256']=ps;resign(components/'reference','reference.json',w)
    shutil.copytree(source/str(index),components/str(index));folder=components/str(index);w=json.loads((folder/'worker.json').read_text())
    w['plan_sha256']=ps;w['reference_manifest_sha256']=rec.sha(components/'reference/reference.json');w['reference_marker_sha256']=rec.sha(components/'reference/complete.json')
    w['previous_receipts']=[{'index':j,'worker_sha256':rec.sha(components/str(j)/'worker.json'),'complete_marker_sha256':rec.sha(components/str(j)/'complete.json')} for j in range(index)]
    if index>=2:w['fresh_replay_reference_sha256']=rec.sha(components/str(index-2)/'worker.json')
    resign(folder,'worker.json',w)
