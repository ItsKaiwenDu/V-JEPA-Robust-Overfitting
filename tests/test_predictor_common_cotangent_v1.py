"""Actual fresh CPU cells, explicit interventions, distinct records and stop paths."""
from pathlib import Path
from types import SimpleNamespace
import copy,hashlib,json,shutil,subprocess,sys
import numpy as np
import pytest
import torch
from models import predictor_common_cotangent_record_v1 as rec
from models import predictor_common_cotangent_operator_v1 as op
from models.predictor_cotangent_attribution_v1 import build_bundle

ROOT=Path(__file__).resolve().parents[1]
CONTROL='c'*64
OLD=ROOT/'runs/pretrained/diagnostic/predictor-eot8-10603-v1-20261105/cpu-four-worker'
OBSERVED=ROOT/'runs/pretrained/diagnostic/predictor-boundary-v1-20261106/pytest-final/boundary-native0'


def local_inputs(folder):
    folder.mkdir(mode=0o700);(folder/'anchors').mkdir();oldmeta=json.loads((OBSERVED/'anchors/ANCHORS.json').read_text())
    for i in (0,1):
        (folder/'anchors'/str(i)).mkdir()
        for kind in ('loss','input'):
            item=oldmeta['anchors'][str(i)]['artifacts'][kind]
            shutil.copyfile(OBSERVED/'anchors'/item['file'],folder/'anchors'/item['file'])
    (folder/'anchors/ANCHORS.json').write_bytes(op.old.json_bytes(oldmeta))
    workers=[json.loads((OBSERVED/str(i)/'worker.json').read_text()) for i in (0,1)]
    pairs=[]
    for i,w in enumerate(workers):
        pairs.append([{key:torch.load(OBSERVED/str(i)/row[key]['file'],weights_only=True,map_location='cpu').numpy()
                       for key in ('prediction','cotangent')} for row in w['observations']])
    target=torch.load(OBSERVED/'0/targets.pt',weights_only=True,map_location='cpu').numpy()
    masks=json.loads((OBSERVED/'0/mask-indices.json').read_text())
    bundle=build_bundle(*pairs,target,masks,baseline_sha256=workers[0]['baseline']['weights_sha256'],domain='local-tiny-CPU')
    path=folder/'cotangent-values.json';path.write_bytes(op.old.json_bytes(bundle))
    expected={'sparse_data':bundle,'sparse_sha256':op.old.digest(path),
        'contract':{'approved_baseline':oldmeta['baseline'],'anchors_metadata':oldmeta}}
    return expected


def child(output,index,inputs,reference=None,fault='none'):
    script='''import os,sys
os.umask(0o077)
from pathlib import Path
from types import SimpleNamespace
import torch
import models.predictor_eot8_capture_v1 as capture
from models.predictor_common_cotangent_worker_v1 import run
fault=sys.argv[5]
if fault=='numerical':
 original=capture.compare
 def fail(a,b,bounds):
  if a.ndim>0:raise ValueError('Original numerical gate rejected; no tolerance change.')
  return original(a,b,bounds)
 capture.compare=fail
elif fault=='backward':
 def fail(*a,**kw):raise RuntimeError('controlled-primary-backward-error')
 torch.autograd.grad=fail
run(SimpleNamespace(output=Path(sys.argv[1]),index=int(sys.argv[2]),tiny=True,device='cpu',
 anchors=Path(sys.argv[3])/'anchors',sparse=Path(sys.argv[3])/'cotangent-values.json',
 reference=Path(sys.argv[4]) if sys.argv[4]!='none' else None,upstream=Path(sys.argv[6]),
 initial=Path(sys.argv[7]),initial_sha='989665a5a4f8ea90600214be5536b4600c37fc448108a5bd177b3416e5ae0338',control_sha256='c'*64))'''
    p=subprocess.run([sys.executable,'-B','-c',script,str(output),str(index),str(inputs),str(reference) if reference else 'none',fault,
                      str(ROOT/'external/vjepa2'),str(OLD/'initial.pt')],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
    (output.parent/(output.name+'.log')).write_bytes(p.stdout);return p


@pytest.fixture(scope='module')
def cells(tmp_path_factory):
    folder=tmp_path_factory.mktemp('common-native');expected=local_inputs(folder/'inputs')
    for i in range(4):
        p=child(folder/str(i),i,folder/'inputs',folder/'0' if i else None)
        assert p.returncode==0,p.stdout.decode()[-6000:]
    p=child(folder/'native-rejected',1,folder/'inputs',folder/'0',fault='numerical')
    assert p.returncode==0,p.stdout.decode()[-6000:]
    p=child(folder/'backward-failed',0,folder/'inputs',fault='backward')
    assert p.returncode==1 and b'controlled-primary-backward-error' in p.stdout
    return SimpleNamespace(folder=folder,expected=expected)


def test_four_actual_fresh_cells_preserve_all_values_and_sixteen_boundaries(cells):
    pids=[];values={}
    for i,(name,path,cotangent) in enumerate(op.CELLS):
        folder=cells.folder/str(i);w=rec.load(folder,control_sha256=CONTROL)
        receipt=op.verify_cell(folder,i,control_sha256=CONTROL,live=False,expected=cells.expected)
        (cells.folder/('receipt-'+str(i)+'.json')).write_bytes(op.old.json_bytes(receipt))
        assert w['attribution_complete'] and w['cell']==name and w['path']==path and w['cotangent']==cotangent
        assert w['native_loss_gradient'] is (i<2) and w['model_unchanged'] and w['adapter_restored'] and w['observer_restored']
        assert w['native_numerical_passed'] is (True if i<2 else None)
        assert not w['research_admitted'] and not w['backend_kernel_qualified'] and not w['full_five_phase_qualified']
        assert len(w['observations'])==16 and len(w['backward_scopes'])==1 and w['backward_scopes'][0]['hook_removed']
        assert not w['ownership_refusals'];pids.append(w['pid'])
        for kind in ('loss','input'):
            item=w['artifacts'][kind];actual=torch.load(folder/item['file'],weights_only=True,map_location='cpu')
            frozen=json.loads((OLD/str(path)/'worker.json').read_text())['artifacts']['EOT8-before-'+kind]
            expected=torch.load(OLD/str(path)/frozen['file'],weights_only=True,map_location='cpu')
            assert torch.equal(actual,expected);values[name+'-'+kind]=actual
        assert len(w['call_rows'])==128
        for row in w['observations']:
            assert row['hook_removed'] and row['replacement'] is (i>=2)
            pair=cells.expected['sparse_data']['pairs'][row['index']]
            assert row['received']['sha256']==pair['arms'][path]['cotangent_sha256']
            assert row['applied']['sha256']==pair['arms'][cotangent]['cotangent_sha256']
    assert len(set(pids))==4
    torch.save(values,cells.folder/'four-native-values.pt')


def test_native_rejection_retained_under_its_own_bound_marker_and_not_native_acceptance(cells):
    folder=cells.folder/'native-rejected';w=rec.load(folder,control_sha256=CONTROL)
    assert w['attribution_complete'] and w['native_loss_gradient'] and w['native_numerical_passed'] is False
    assert w['native_rejection']=={'type':'ValueError','error':'Original numerical gate rejected; no tolerance change.','kind':'input'}
    assert len(w['native_checks'])==4 and [r['passed'] for r in w['native_checks']]==[True,False,True,False]
    assert (folder/rec.REJECTION).is_file() and (folder/rec.COMPLETE).is_file() and not (folder/rec.FAILURE).exists()
    receipt=op.verify_cell(folder,1,control_sha256=CONTROL,live=False,expected=cells.expected)
    assert receipt['native_numerical_passed'] is False and not receipt['research_admitted']
    from models import predictor_boundary_record_v1 as old
    from models.predictor_eot8_record_v2 import load_worker
    for index in ('0','1','2','3','native-rejected'):
        with pytest.raises(ValueError):old.load(cells.folder/index,control_sha256=CONTROL)
        with pytest.raises(ValueError):load_worker(cells.folder/index,expected_control=CONTROL)


def test_primary_failure_preserves_loss_predictions_partial_rows_and_removed_hooks(cells):
    folder=cells.folder/'backward-failed';w=rec.load(folder,control_sha256=CONTROL,require_complete=False)
    assert w['error']=='controlled-primary-backward-error' and not w['attribution_complete'] and not w['capture_complete']
    assert set(w['artifacts'])=={'targets','loss'} and len(w['observations'])==16
    assert all(r['received'] is None and r['applied'] is None and r['hook_removed'] for r in w['observations'])
    assert w['observer_restored'] and w['adapter_restored'] and w['backward_scopes'][0]['hook_removed']
    with pytest.raises(ValueError):rec.load(folder,control_sha256=CONTROL)


def resign(folder,change):
    path=folder/rec.atomic.HEADER;w=json.loads(path.read_text());change(w);path.write_bytes(rec.atomic._json(w))
    for name in (rec.COMPLETE,rec.FAILURE,rec.REJECTION):
        p=folder/name
        if p.exists():
            m=json.loads(p.read_text());m['header']={'file':path.name,'bytes':path.stat().st_size,'sha256':op.old.digest(path)};p.write_bytes(rec.atomic._json(m))


@pytest.mark.parametrize('fault',['native_label','ambiguous_passed','backend','native_status','same_path_loss','target_hash','wrong_apply',
    'wrong_receive','missing_pair','sparse_values','scope','mask','input','marker','cross_native_checks','input_anchor_flag'])
def test_changed_or_relabelled_controlled_evidence_refuses_even_with_rehashed_header(cells,tmp_path,fault):
    folder=tmp_path/'cell';shutil.copytree(cells.folder/'2',folder)
    def mutate(w):
        if fault=='native_label':w['native_loss_gradient']=True
        elif fault=='ambiguous_passed':w['passed']=True
        elif fault=='backend':w['backend_kernel_qualified']=True
        elif fault=='native_status':w['native_numerical_passed']=True
        elif fault=='same_path_loss':w['artifacts']['loss']['source']['sha256']='f'*64
        elif fault=='target_hash':w['artifacts']['targets']['source']['sha256']='f'*64
        elif fault=='wrong_apply':w['observations'][0]['applied']['sha256']='f'*64
        elif fault=='wrong_receive':w['observations'][0]['received']['sha256']='f'*64
        elif fault=='missing_pair':w['observations'].pop()
        elif fault=='sparse_values':w['observations'][0]['sparse_entries']=[{'coordinate':[0,0,0],'received_bits':'00000000','applied_bits':'00000000'}]
        elif fault=='cross_native_checks':w['native_checks']=[{'passed':True}]
        elif fault=='input_anchor_flag':w['input_reproduces_anchor']=True
    resign(folder,mutate)
    if fault=='scope':(folder/rec.atomic.SCOPES).write_bytes(b'{}\n')
    elif fault=='mask':(folder/'mask-indices.json').write_bytes(b'[]\n')
    elif fault=='input':(folder/'input.pt').write_bytes(b'changed')
    elif fault=='marker':(folder/rec.COMPLETE).unlink()
    with pytest.raises((ValueError,FileNotFoundError)):op.verify_cell(folder,2,control_sha256=CONTROL,live=False,expected=cells.expected)


def test_rejection_marker_and_target_candidate_status_cannot_be_erased(cells,tmp_path):
    folder=tmp_path/'rejected';shutil.copytree(cells.folder/'native-rejected',folder)
    (folder/rec.REJECTION).unlink()
    with pytest.raises(FileNotFoundError):rec.load(folder,control_sha256=CONTROL)
    w=rec.load(cells.folder/'1',control_sha256=CONTROL);w['local_tiny']=False
    with pytest.raises(ValueError,match='remain rejected'):rec.semantics(w)


def test_source_and_destination_checks_apply_to_actual_nontrivial_backward_intervention():
    from test_predictor_cotangent_attribution_v1 import ToyModel,masks,mask_values
    from models.predictor_gradient_attribution_v1 import LocalPredictionObserver
    from models.predictor_common_cotangent_observer_v1 import ControlledCotangent
    from attacks.pgd import native_objective
    target=torch.zeros(1,32,128);banks=masks();pairs=[]
    for path in (0,1):
        model=ToyModel(path);x=torch.zeros_like(target,requires_grad=True)
        with LocalPredictionObserver(model,target,banks) as capture:
            loss=native_objective(model,target,banks)(x).sum();torch.autograd.grad(loss,x)
        pairs.append([{key:r[key].numpy() for key in ('prediction','cotangent')} for r in capture.records])
    bundle=build_bundle(*pairs,target.numpy(),mask_values(banks),baseline_sha256='a'*64,domain='local-tiny-CPU')
    proofs=[]
    for path,other in ((0,1),(1,0)):
        model=ToyModel(path);x=torch.zeros_like(target,requires_grad=True)
        with ControlledCotangent(model,target,banks,bundle,tiny=True,path=path,cotangent=other,publish=lambda:None) as observer:
            loss=native_objective(model,target,banks)(x).sum();g,=torch.autograd.grad(loss,x)
        expected=torch.zeros_like(x);expected[:,16:32,:]=(path+1)/2048
        expected[0,16,0]*=1 if other==0 else -1
        assert torch.equal(g,expected) and observer.restored and not model.predictor._forward_hooks
        assert all(r['hook_removed'] and r['replacement'] for r in observer.rows)
        assert all(r['sparse_entries'][0]['received_bits']!=r['sparse_entries'][0]['applied_bits'] for r in observer.rows)


@pytest.mark.parametrize('host,device',[('linux10603','cpu'),('linux10609','cpu'),('local','cuda')])
def test_local_analogue_refuses_university_or_cuda_before_scientific_import(monkeypatch,tmp_path,host,device):
    from models import predictor_common_cotangent_worker_v1 as worker
    monkeypatch.setattr(worker.platform,'node',lambda:host)
    with pytest.raises(ValueError):worker.run(SimpleNamespace(tiny=True,index=0,device=device,output=tmp_path/'no-output'))
    assert not (tmp_path/'no-output').exists()


@pytest.mark.parametrize('name',[rec.COMPLETE,rec.FAILURE,rec.REJECTION])
@pytest.mark.parametrize('fault',['oversize','symlink','hardlink','partial','old_reader'])
def test_distinct_markers_keep_bounds_file_integrity_and_frozen_reader_authority(tmp_path,name,fault):
    folder=tmp_path/'evidence';folder.mkdir(mode=0o700);path=folder/name
    if fault=='oversize':
        path.write_bytes(b'x'*(rec.atomic.MAX_MARKER_BYTES+1))
        with pytest.raises(ValueError):rec._read(folder,name,rec.atomic.MAX_MARKER_BYTES)
        with pytest.raises(ValueError):rec._atomic(folder,name,[b'{}\n'],rec.atomic.MAX_MARKER_BYTES)
        assert path.stat().st_size==rec.atomic.MAX_MARKER_BYTES+1
    elif fault=='symlink':
        target=folder/'other';target.write_bytes(b'{}\n');path.symlink_to(target)
        with pytest.raises(ValueError):rec._read(folder,name,rec.atomic.MAX_MARKER_BYTES)
    elif fault=='hardlink':
        path.write_bytes(b'{}\n');(folder/'other').hardlink_to(path)
        with pytest.raises(ValueError):rec._read(folder,name,rec.atomic.MAX_MARKER_BYTES)
    elif fault=='partial':
        partial=folder/(name+'.partial');partial.write_bytes(b'keep')
        with pytest.raises(FileExistsError):rec._atomic(folder,name,[b'{}\n'],rec.atomic.MAX_MARKER_BYTES)
        assert partial.read_bytes()==b'keep' and not path.exists()
    else:
        rec._atomic(folder,name,[b'{}\n'],rec.atomic.MAX_MARKER_BYTES)
        assert rec._read(folder,name,rec.atomic.MAX_MARKER_BYTES)==b'{}\n'
        with pytest.raises(ValueError):rec.atomic._read(folder,name,rec.atomic.MAX_MARKER_BYTES)
