"""Real upstream native checkpoint ownership, exact controls and bounded evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models import predictor_sdpa_capture_record_v1 as record
from models import predictor_sdpa_observer_v1 as observer_module

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'runs/pretrained/diagnostic/predictor-eot8-10603-v1-20261105/cpu-four-worker'
CONTROL='d'*64


def child(folder,runner='creator',fault='none'):
    script='''import sys
from pathlib import Path
from types import SimpleNamespace
if sys.argv[3]=='backward':
 from models.predictor_eot8_owned_v1 import OwnedBackward
 old=OwnedBackward._root
 def error(self,g):old(self,g);raise RuntimeError('cpu-primary-backward-error')
 OwnedBackward._root=error
elif sys.argv[3]=='save':
 from models.predictor_sdpa_capture_record_v1 import BoundaryStore
 old=BoundaryStore.save
 def error(self,name,value):
  if name=='k':raise OSError('cpu-capture-write-error')
  return old(self,name,value)
 BoundaryStore.save=error
from models.predictor_sdpa_observer_cpu_v1 import run
run(SimpleNamespace(output=Path(sys.argv[1]),runner=sys.argv[2],device='cpu',
 upstream=Path(sys.argv[4]),initial=Path(sys.argv[5]),reference=Path(sys.argv[6])))'''
    proc=subprocess.run([sys.executable,'-B','-c',script,str(folder),runner,fault,str(ROOT/'external/vjepa2'),str(OLD/'initial.pt'),str(OLD/'0')],
        cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
    (folder.parent/(folder.name+'.log')).write_bytes(proc.stdout)
    return proc


@pytest.fixture(scope='module')
def captures(tmp_path_factory):
    root=tmp_path_factory.mktemp('native-sdpa-observer')
    for name,runner in (('creator-0','creator'),('other-0','other-thread'),('creator-1','creator'),('other-1','other-thread')):
        p=child(root/name,runner);assert p.returncode==0,p.stdout.decode()[-7000:]
    for fault in ('backward','save'):
        p=child(root/fault,fault=fault);assert p.returncode==1,p.stdout.decode()[-7000:]
    return root


def actual(folder):return record.load(folder/'boundary',control_sha256=CONTROL)


def test_four_fresh_native_captures_match_exact_before_observed_after_and_prior_values(captures):
    pids=[];first={};source_groups=[]
    prior=json.loads((OLD/'0/worker.json').read_text())
    for name in ('creator-0','other-0','creator-1','other-1'):
        folder=captures/name;w=json.loads((folder/'worker.json').read_text());cap=actual(folder)
        assert w['complete'] and w['original_value_replay_exact'] and w['model_unchanged'] and w['route_restored'] and w['observer_restored']
        assert not w['CUDA_kernel_qualified'] and not w['performance_qualified'] and not w['research_admitted']
        assert w['boundary_header_sha256']==record.digest(folder/'boundary/boundary.json')
        for label in ('before','after',''):
            for kind in ('loss','input'):
                key=label+'-'+kind if label else kind
                tensor=torch.load(folder/w['artifacts'][key]['file'],weights_only=True,map_location='cpu')
                saved=torch.load(OLD/'0'/prior['artifacts']['EOT8-before-'+kind]['file'],weights_only=True,map_location='cpu')
                assert torch.equal(tensor,saved)
        pids.append(w['pid'])
        assert cap['capture_complete'] and cap['alias_restored'] and cap['tensor_versions_unchanged'] and cap['observer_restored']
        assert cap['selected']['forward_row_index']==2 and cap['selected_call']['actual_call']['owner']=='predictor.0'
        assert cap['scope_check']['owners']==4 and len(cap['call_rows'])==128
        assert cap['scope_check']['forward_calls']==cap['scope_check']['backward_calls']==64
        assert cap['ownership_refusals']==[] and cap['backward_scopes'][0]['hook_removed']
        assert len(cap['gradient_observations'])==4 and all(r['hook_removed'] for r in cap['gradient_observations'].values())
        if name.startswith('other'):
            assert w['graph_scope']['runner_thread']!=cap['creator_thread']
            assert all(r['thread']!=cap['creator_thread'] for r in cap['gradient_observations'].values())
        for n in record.TENSORS:
            value=record.load_tensor(folder/'boundary',n,cap)
            if n in first:assert torch.equal(value,first[n])
            else:first[n]=value
            assert cap['source_storage'][n]['positive_stride_span_bytes']<=16*record.MiB
        assert cap['source_storage']['v']['storage_offset_elements']>0
        assert cap['artifacts']['v']['source']['stride']!=cap['artifacts']['v']['stored_stride']
        assert not cap['artifacts']['v']['native_storage_alias_preserved_in_file']
        source_groups.append([cap['source_storage'][n]['alias_group'] for n in ('q','k','v')])
    assert len(set(pids))==4
    assert all(groups==source_groups[0] for groups in source_groups)


def test_saved_native_boundary_rebuilds_offsets_layouts_and_replays_standalone_CPU_values(captures):
    from models.attention_saved_evidence import oracle,agreement
    from models.predictor_eot8_capture_v1 import tensor_info
    from models.predictor_sdpa_fixed_input_v1 import TOLERANCES
    receipts=[]
    for name in ('creator-0','other-0'):
        folder=captures/name;cap=actual(folder)
        rebuilt=record.rebuild_cpu_inputs(folder/'boundary',cap)
        for n,t in rebuilt.items():
            assert t.storage_offset()==cap['source_storage'][n]['storage_offset_elements']
            assert list(t.stride())==cap['artifacts'][n]['source']['stride']
        q,k,v=[rebuilt[n].requires_grad_() for n in ('q','k','v')];g=rebuilt['grad_output']
        output=torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False,attn_mask=None)
        gradients=torch.autograd.grad(output,(q,k,v),g)
        values=dict(zip(('output','dq','dk','dv'),(output,*gradients)))
        analytic=oracle(*(rebuilt[n].detach().numpy() for n in ('q','k','v','grad_output')),chunk=32)
        checks={}
        for n,t in values.items():
            assert torch.equal(t.detach(),record.load_tensor(folder/'boundary',n,cap))
            checks[n]=agreement(t.detach().numpy(),analytic[n],TOLERANCES[n]);assert checks[n]['passed']
        item={'CPU_only':True,'pretrained_or_CUDA_qualified':False,'pure_backward_arithmetic_isolated':False,
            'captured_boundary':record.digest(folder/'boundary/boundary.json'),'exact_native_replay':True,
            'FP64_analytical_checks':checks,'rebuilt_inputs':{n:{**tensor_info(t),'storage_offset_elements':t.storage_offset(),
                'unused_native_padding_not_observed_filled_zero':True,'physical_alignment_not_claimed_identical':True} for n,t in rebuilt.items()}}
        (folder/'standalone-CPU-replay.json').write_bytes(record.encoded(item));receipts.append(item)
    assert all(r['exact_native_replay'] for r in receipts)


def test_primary_backward_and_capture_write_errors_preserve_real_partial_tensors(captures):
    for name,error,names in (('backward','cpu-primary-backward-error',{'q','k','v','output'}),('save','cpu-capture-write-error',{'q'})):
        folder=captures/name;w=json.loads((folder/'worker.json').read_text())
        assert not w['complete'] and error in w['error'] and w['route_restored'] and w['observer_restored']
        partial=record.load(folder/'boundary',control_sha256=CONTROL,require_complete=False)
        assert not partial['capture_complete'] and set(partial['artifacts'])==names and partial['observer_restored']
        assert all(not (folder/'boundary'/(n+'.pt')).exists() for n in set(record.TENSORS)-names)
        with pytest.raises(ValueError):actual(folder)


def test_observation_is_not_accepted_as_a_passing_native_or_controlled_worker(captures):
    from models import predictor_boundary_record_v1 as native
    from models import predictor_common_cotangent_record_v1 as controlled
    folder=captures/'creator-0/boundary'
    for old in (native,controlled):
        with pytest.raises((ValueError,FileNotFoundError)):old.load(folder,control_sha256=CONTROL)


def resign(folder,change):
    p=folder/'boundary.json';w=json.loads(p.read_text());change(w);p.write_bytes(record.encoded(w))
    marker=folder/'boundary-complete.json';m=json.loads(marker.read_text())
    m['header']={'file':p.name,'bytes':p.stat().st_size,'sha256':record.digest(p)};marker.write_bytes(record.encoded(m))


@pytest.mark.parametrize('fault',('replacement','forward-modified','full-model-admission','selected-call','selected-owner','hook-graph','hook-thread',
    'hook-removed','versions','source-offset','alias','alignment','logical-hash','stored-stride','missing-gradient','native-flags'))
def test_rehashed_or_relabelled_boundary_evidence_refuses(captures,tmp_path,fault):
    folder=tmp_path/'boundary';shutil.copytree(captures/'creator-0/boundary',folder)
    def change(w):
        if fault=='replacement':w['native_cotangent_replaced']=True
        elif fault=='forward-modified':w['native_forward_modified']=True
        elif fault=='full-model-admission':w['native_full_model_qualified']=True
        elif fault=='selected-call':w['selected']['forward_row_index']=6
        elif fault=='selected-owner':w['selected_call']['actual_call']['owner']='predictor.1'
        elif fault=='hook-graph':w['gradient_observations']['dq']['graph_id']+=1
        elif fault=='hook-thread':w['gradient_observations']['dq']['thread']=12345
        elif fault=='hook-removed':w['gradient_observations']['dq']['hook_removed']=False
        elif fault=='versions':w['tensor_versions_unchanged']=False
        elif fault=='source-offset':w['source_storage']['q']['storage_offset_elements']+=1
        elif fault=='alias':w['source_storage']['q']['alias_group']=12345
        elif fault=='alignment':w['source_storage']['q']['data_address_mod_256']=257
        elif fault=='logical-hash':
            w['artifacts']['dq']['source']['sha256']='f'*64;w['source_storage']['dq']['source_logical_hash']='f'*64
        elif fault=='stored-stride':w['artifacts']['dq']['stored_stride']=[1,1,1,1]
        elif fault=='missing-gradient':w['gradient_observations'].pop('dv')
        elif fault=='native-flags':w['mode']='predictor-math'
    resign(folder,change)
    with pytest.raises(ValueError):
        state=record.load(folder,control_sha256=CONTROL)
        for n in record.TENSORS:record.load_tensor(folder,n,state)


@pytest.mark.parametrize('fault',('oversize','symlink','hardlink','partial','marker','failed-and-complete'))
def test_private_typed_records_and_final_markers_preserve_file_integrity(captures,tmp_path,fault):
    folder=tmp_path/'boundary';shutil.copytree(captures/'creator-0/boundary',folder)
    if fault=='oversize':
        (folder/'boundary.json').write_bytes(b'x'*(record.LIMITS['boundary.json']+1))
    elif fault=='symlink':
        p=folder/'q.pt';p.unlink();p.symlink_to(folder/'k.pt')
    elif fault=='hardlink':(folder/'q-other.pt').hardlink_to(folder/'q.pt')
    elif fault=='partial':
        (folder/'boundary-complete.json').rename(folder/'boundary-complete.json.partial')
    elif fault=='marker':(folder/'boundary-complete.json').unlink()
    else:(folder/'boundary-failed.json').write_bytes(b'{}\n')
    with pytest.raises((ValueError,FileNotFoundError)):record.load(folder,control_sha256=CONTROL)


def local_fixture():
    from models.execution import configure_execution,apply_model_policy
    from models.native_diagnostic import tiny_fixture
    configure_execution('shared-fp64-native',device='cpu');torch.set_num_threads(4);torch.manual_seed(17)
    model,clean,masks=tiny_fixture(ROOT/'external/vjepa2');apply_model_policy(model,'shared-fp64-native');model.eval()
    target=model.clean_targets(clean)
    return model,clean,masks,target


def test_unowned_forward_and_a_real_foreign_GraphTask_are_not_whitelisted(tmp_path):
    from models.predictor_eot8_owned_v1 import PredictorRoute
    from attacks.pgd import native_objective
    model,clean,masks,target=local_fixture();banks=[masks(i) for i in range(8)]
    folder=tmp_path/'capture';folder.mkdir(mode=0o700)
    with pytest.raises(ValueError,match='outside its actual native GraphTask'):
        with PredictorRoute(model,'original',tiny=True) as route:
            route.phase='EOT8-before'
            with observer_module.NativeSDPAObserver(route,record.BoundaryStore(folder),control_sha256=CONTROL) as observer:
                x=clean.clone().requires_grad_(True);loss=native_objective(model,target,banks)(x).sum();task=route.prepare_backward(loss,x)
                original=task._root
                def root(gradients):
                    original(gradients)
                    class Foreign(torch.autograd.Function):
                        @staticmethod
                        def forward(ctx,value):return value.clone()
                        @staticmethod
                        def backward(ctx,g):observer._owner(backward=True);return g
                    with torch.enable_grad():v=torch.tensor(1.,requires_grad=True);y=Foreign.apply(v)
                    torch.autograd.grad(y,v)
                task._root=root;task.run()
    assert observer.state['observer_restored'] and not route.active
    assert not observer.state['capture_complete']


@pytest.mark.parametrize('fault',('candidate-route','duplicate-observer','alias-tamper','phase-change','early-exception'))
def test_instance_alias_is_restored_on_setup_and_primary_errors(tmp_path,fault):
    from models.predictor_eot8_owned_v1 import PredictorRoute
    model,clean,masks,target=local_fixture();folder=tmp_path/'capture';folder.mkdir(mode=0o700)
    route=PredictorRoute(model,'predictor-math' if fault=='candidate-route' else 'original',tiny=True)
    with pytest.raises((ValueError,RuntimeError)):
        with route:
            route.phase='EOT8-before'
            with observer_module.NativeSDPAObserver(route,record.BoundaryStore(folder),control_sha256=CONTROL) as observer:
                if fault=='duplicate-observer':
                    other=tmp_path/'other';other.mkdir(mode=0o700)
                    with observer_module.NativeSDPAObserver(route,record.BoundaryStore(other),control_sha256=CONTROL):pass
                elif fault=='alias-tamper':observer.view.route=lambda *a,**k:None
                elif fault=='phase-change':route.phase='draw-before';model.loss_per_example(clean.clone().requires_grad_(True),target,masks(3))
                else:raise RuntimeError('primary-local-error')
    assert not route.active and all('forward' not in b.attn.__dict__ for b in model.predictor.predictor_blocks)


@pytest.mark.parametrize('host,device',(('linux10603','cpu'),('linux10609','cpu'),('local','cuda')))
def test_CPU_entry_refuses_university_or_CUDA_before_model_and_output(tmp_path,monkeypatch,host,device):
    from models import predictor_sdpa_observer_cpu_v1 as entry
    monkeypatch.setattr(entry.platform,'node',lambda:host)
    with pytest.raises(ValueError):entry.run(SimpleNamespace(device=device,runner='creator',output=tmp_path/'none'))
    assert not (tmp_path/'none').exists()


def test_store_keeps_actual_partial_on_write_error_without_a_completion_marker(tmp_path):
    folder=tmp_path/'capture';folder.mkdir(mode=0o700);store=record.BoundaryStore(folder)
    def failure(f):f.write(b'actual-partial');raise OSError('simulated-storage-write-error')
    with pytest.raises(OSError):store._write('q.pt',failure)
    assert (folder/'q.pt.partial').read_bytes()==b'actual-partial' and not (folder/'q.pt').exists()
    assert store.poisoned and not (folder/'boundary-complete.json').exists()
    with pytest.raises(ValueError):store.publish({})


def test_store_preserves_an_unexpected_file_created_after_construction(tmp_path):
    folder=tmp_path/'capture';folder.mkdir(mode=0o700);store=record.BoundaryStore(folder)
    (folder/'q.pt').write_bytes(b'preserve-existing-file')
    with pytest.raises(FileExistsError):store.save('q',torch.ones(1,2,32,32))
    assert (folder/'q.pt').read_bytes()==b'preserve-existing-file'
    assert not (folder/'q.pt.partial').exists()
