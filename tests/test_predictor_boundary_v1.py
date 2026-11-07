"""Real fresh CPU native capture, immutable anchors and failure preservation."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
import pytest
import torch
from models import predictor_boundary_operator_v1 as op
from models import predictor_boundary_record_v1 as rec
from models.predictor_eot8_capture_v1 import tensor_info

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'runs/pretrained/diagnostic/predictor-eot8-10603-v1-20261105/cpu-four-worker'
CONTROL='b'*64


def tiny_anchors(folder):
    folder.mkdir(mode=0o700);anchors={}
    for i in (0,1):
        source=OLD/str(i);w=json.loads((source/'worker.json').read_text());(folder/str(i)).mkdir()
        artifacts={}
        for kind in ('loss','input'):
            item=w['artifacts']['EOT8-before-'+kind];name=f'{i}/{kind}.pt';shutil.copyfile(source/item['file'],folder/name)
            artifacts[kind]={**item,'file':name}
        anchors[str(i)]={'role':'local-CPU-values-only','value_only_not_executable_reference':True,
            'source_worker_passed':True,'source_worker_complete':True,'artifacts':artifacts}
    meta=op.anchor_metadata({},tiny=True,baseline=json.loads((OLD/'0/worker.json').read_text())['baseline'],anchors=anchors)
    (folder/'ANCHORS.json').write_bytes(op.old.json_bytes(meta))
    return folder


def child(folder,index,anchors,reference=None,fault='none'):
    script='''import os,sys
os.umask(0o077)
from pathlib import Path
from types import SimpleNamespace
from models.predictor_boundary_worker_v1 import run
import models.predictor_eot8_capture_v1 as capture
import torch
fault=sys.argv[5]
if fault=='numerical':
 original=capture.compare
 def fail(actual,expected,bounds):
  if actual.ndim>0:raise ValueError('Original numerical gate rejected; no tolerance change.')
  return original(actual,expected,bounds)
 capture.compare=fail
elif fault=='backward':
 def fail(*a,**kw):raise RuntimeError('boundary-test-primary-backward-error')
 torch.autograd.grad=fail
run(SimpleNamespace(output=Path(sys.argv[1]),index=int(sys.argv[2]),anchors=Path(sys.argv[3]),reference=Path(sys.argv[4]) if sys.argv[4]!='none' else None,
 tiny=True,device='cpu',upstream=Path(sys.argv[6]),initial=Path(sys.argv[7]),initial_sha='989665a5a4f8ea90600214be5536b4600c37fc448108a5bd177b3416e5ae0338',
 control_sha256='b'*64))'''
    p=subprocess.run([sys.executable,'-B','-c',script,str(folder),str(index),str(anchors),str(reference) if reference else 'none',fault,str(ROOT/'external/vjepa2'),str(OLD/'initial.pt')],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
    (folder.parent/(folder.name+'.log')).write_bytes(p.stdout)
    return p


@pytest.fixture(scope='module')
def captures(tmp_path_factory):
    folder=tmp_path_factory.mktemp('boundary-native');anchors=tiny_anchors(folder/'anchors')
    for i in (0,1):
        p=child(folder/str(i),i,anchors,folder/'0' if i else None)
        assert p.returncode==0,p.stdout.decode()[-5000:]
    p=child(folder/'failed-candidate',1,anchors,folder/'0',fault='numerical')
    assert p.returncode==1 and b'Original numerical gate rejected' in p.stdout
    p=child(folder/'backward-failed',0,anchors,fault='backward')
    assert p.returncode==1 and b'boundary-test-primary-backward-error' in p.stdout
    return folder


def test_two_fresh_native_workers_keep_values_and_all_actual_cotangents(captures):
    pids=[]
    for i in (0,1):
        folder=captures/str(i);w=rec.load(folder,control_sha256=CONTROL)
        pids.append(w['pid']);assert w['capture_complete'] and w['numerical_passed'] and w['passed'] and w['reproduces_anchor']
        assert w['observer_restored'] and w['adapter_restored'] and w['model_unchanged']
        proof=op.verify_capture(folder,i,control_sha256=CONTROL,live=False)
        assert proof['capture_complete'] and proof['attribution_only'] and not proof['CUDA_backend_qualified']
        (captures/('receipt-'+str(i)+'.json')).write_text(json.dumps(proof,sort_keys=True))
        for name in ('loss','input'):
            t=torch.load(folder/w['artifacts'][name]['file'],weights_only=True,map_location='cpu')
            anchor=torch.load(captures/'anchors'/str(i)/(name+'.pt'),weights_only=True,map_location='cpu')
            assert torch.equal(t,anchor)
        target=torch.load(folder/'targets.pt',weights_only=True,map_location='cpu');indices=json.loads((folder/'mask-indices.json').read_text())
        for row in w['observations']:
            values={}
            for name in ('prediction','cotangent'):
                item=row[name];path=folder/item['file'];t=torch.load(path,weights_only=True,map_location='cpu')
                assert tensor_info(t)==item['source'] and not t.requires_grad
                assert path.stat().st_size==item['bytes'] and op.old.digest(path)==item['sha256'];values[name]=t
            ids=torch.tensor(indices[row['draw']][row['family']]['target'],dtype=torch.int64)
            selected=target.gather(1,ids.unsqueeze(-1).expand(-1,-1,128))
            scale=torch.ones((),dtype=torch.float32)/8/2/(selected.shape[1]*128)
            assert torch.equal(values['cotangent'],(values['prediction']-selected).sign()*scale)
        assert len(w['observations'])==16 and len(w['backward_scopes'])==1 and not w['ownership_refusals']
        assert w['backward_scopes'][0]['hook_removed'] and not w['full_five_phase_qualified']
    assert len(set(pids))==2


def test_actual_rejected_candidate_has_complete_capture_but_cannot_be_a_passing_reference(captures):
    folder=captures/'failed-candidate';w=rec.load(folder,control_sha256=CONTROL,require_passed=False)
    assert not w['passed'] and not w['numerical_passed'] and w['capture_complete'] and w['reproduces_anchor']
    assert len(w['observations'])==16 and all(row['cotangent'] and row['hook_removed'] for row in w['observations'])
    assert w['observer_restored'] and w['adapter_restored'] and w['model_unchanged']
    proof=op.verify_capture(folder,1,control_sha256=CONTROL,live=False,require_passed=False)
    assert proof['attribution_only'] and not proof['numerical_passed']
    with pytest.raises(ValueError):rec.load(folder,control_sha256=CONTROL)
    assert (folder/rec.atomic.FAILURE).is_file() and not (folder/rec.atomic.COMPLETE).exists()


def test_real_backward_failure_keeps_primary_error_loss_predictions_and_removed_hooks(captures):
    folder=captures/'backward-failed';w=rec.load(folder,control_sha256=CONTROL,require_passed=False)
    assert w['error']=='boundary-test-primary-backward-error' and not w['capture_complete'] and not w['passed']
    assert set(w['artifacts'])=={'targets','loss'} and len(w['observations'])==16
    assert all(row['prediction'] and row['cotangent'] is None and row['hook_removed'] for row in w['observations'])
    assert w['observer_restored'] and w['adapter_restored'] and w['backward_scopes'][0]['state']=='failed' and w['backward_scopes'][0]['hook_removed']
    with pytest.raises(ValueError):op.verify_capture(folder,0,control_sha256=CONTROL,live=False,require_passed=False)


def resign_header(folder,fn):
    p=folder/rec.atomic.HEADER;w=json.loads(p.read_text());fn(w);p.write_bytes(rec.atomic._json(w))
    marker=folder/(rec.atomic.COMPLETE if w['passed'] else rec.atomic.FAILURE);m=json.loads(marker.read_text())
    m['header']={'file':rec.atomic.HEADER,'bytes':p.stat().st_size,'sha256':op.old.digest(p)};marker.write_bytes(rec.atomic._json(m))


@pytest.mark.parametrize('fault',['missing_pair','pair_index','artifact_shape','artifact_device','artifact_bytes','source_mask','observer_restore','numerical_status','backend_claim','five_phase_claim','loss_changed','missing_marker','scopes_changed','call_changed'])
def test_capture_rejects_corrupt_incomplete_resigned_or_expanded_evidence(captures,tmp_path,fault):
    folder=tmp_path/'copy';shutil.copytree(captures/'0',folder)
    if fault=='missing_pair':resign_header(folder,lambda w:w['observations'].pop())
    elif fault=='pair_index':resign_header(folder,lambda w:w['observations'][0].update(draw=1))
    elif fault=='artifact_shape':resign_header(folder,lambda w:w['observations'][0]['prediction']['source'].update(shape=[1,24,127]))
    elif fault=='artifact_device':resign_header(folder,lambda w:w['observations'][0]['cotangent']['source'].update(device='cuda:0'))
    elif fault=='artifact_bytes':resign_header(folder,lambda w:w['observations'][0]['cotangent'].update(bytes=1))
    elif fault=='observer_restore':resign_header(folder,lambda w:w.update(observer_restored=False))
    elif fault=='numerical_status':resign_header(folder,lambda w:w.update(numerical_passed=False))
    elif fault=='backend_claim':resign_header(folder,lambda w:w.update(backend_kernel_qualified=True))
    elif fault=='five_phase_claim':resign_header(folder,lambda w:w.update(full_five_phase_qualified=True))
    elif fault=='source_mask':
        p=folder/'mask-indices.json';m=json.loads(p.read_text());m[0][0]['target'][0][0]=m[0][0]['target'][0][1];p.write_bytes(rec.atomic._json(m))
        resign_header(folder,lambda w:w.update(mask_indices={'file':p.name,'bytes':p.stat().st_size,'sha256':op.old.digest(p)}))
    elif fault=='loss_changed':
        p=folder/'loss.pt';p.write_bytes(p.read_bytes()[:-1]+b'x')
    elif fault=='missing_marker':(folder/rec.atomic.COMPLETE).unlink()
    elif fault=='scopes_changed':
        p=folder/rec.atomic.SCOPES;m=json.loads(p.read_text());m['scopes'][0]['hook_removed']=False;p.write_bytes(rec.atomic._json(m))
        resign_header(folder,lambda w:w['evidence'].update(scopes={'file':p.name,'bytes':p.stat().st_size,'sha256':op.old.digest(p)}))
    else:
        p=folder/rec.atomic.CALLS;p.write_bytes(p.read_bytes()[:-1]+b'x')
    with pytest.raises((ValueError,FileNotFoundError)):op.verify_capture(folder,0,control_sha256=CONTROL,live=False)


def test_no_CPU_or_old_five_phase_qualification_or_wrong_control(captures):
    with pytest.raises(ValueError):op.verify_capture(captures/'0',0,control_sha256=CONTROL,live=True)
    with pytest.raises(ValueError):op.verify_capture(captures/'0',0,control_sha256='a'*64,live=False)
    from models.predictor_eot8_record_v2 import load_worker
    with pytest.raises(ValueError):load_worker(captures/'0',expected_control=CONTROL)


def test_anchor_change_refuses_before_model_configuration(captures,tmp_path,monkeypatch):
    import models.execution as execution
    from models.predictor_boundary_worker_v1 import run
    from types import SimpleNamespace
    anchors=tmp_path/'anchors';shutil.copytree(captures/'anchors',anchors)
    p=anchors/'0/input.pt';p.write_bytes(p.read_bytes()[:-1]+b'x')
    monkeypatch.setattr(execution,'configure_execution',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Model configuration reached')))
    folder=tmp_path/'bad';folder.parent.chmod(0o700)
    with pytest.raises(ValueError,match='Value anchor changed'):
        run(SimpleNamespace(output=folder,index=0,anchors=anchors,reference=None,tiny=True,device='cpu',control_sha256=CONTROL))
    w=rec.load(folder,control_sha256=CONTROL,require_passed=False);assert not w['artifacts'] and not w['capture_complete']


def test_observer_cleanup_cannot_replace_a_primary_exception(monkeypatch):
    from models.predictor_boundary_observer_v1 import BoundaryObserver
    from types import SimpleNamespace
    model=torch.nn.Module();model.predictor=torch.nn.Identity()
    banks=[[SimpleNamespace(context=torch.zeros(1,1,dtype=torch.int64),target=torch.ones(1,1,dtype=torch.int64))]*2 for _ in range(8)]
    def secondary():raise ValueError('secondary-publication')
    observer=BoundaryObserver(model,banks,Path('.'),tiny=True,publish=secondary)
    with pytest.raises(RuntimeError,match='original-primary'):
        with observer:raise RuntimeError('original-primary')
    assert observer.restored and not model.predictor._forward_hooks
