"""Real CPU observer/replay plus refusal of malformed attribution and evidence."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from models.cuda_profile import tensor_signature
from models.eot_profile import summarize,validate_launch
from models.eot_profile_events import PREFIX,aggregate,annotate,union_us
from models.eot_profile_packet import build,verify
from models.execution import apply_model_policy,configure_execution,model_execution_contract
from models.native_diagnostic import tiny_fixture
from models.pilot_bundle import digest
from models.study_artifacts import read_record,write_record
from test_projection_reuse import restore_process_settings

ROOT=Path(__file__).resolve().parents[1]


def event(index,name,start,end,parent=None,device='cpu',self_cpu=0.,linked=()):
    return {'index':index,'name':name,'device':device,'device_index':0,'thread':1,'start_us':start,'end_us':end,
            'parent':parent,'self_cpu_us':self_cpu,'legacy':False,'async':False,'linked_kernels':list(linked)}


def write_events(path,rows):path.write_text(''.join(json.dumps(r)+'\n' for r in rows));return path


def test_exclusive_accounting_does_not_add_inclusive_parent_or_duplicate_device_event(tmp_path):
    rows=[event(0,PREFIX+'EOT8_forward',0,10,self_cpu=2),event(1,PREFIX+'context_encoder',1,9,0,self_cpu=3),
          event(2,'aten::mm',2,7,1,self_cpu=5,linked=[{'name':'actual-kernel','device':0,'duration_us':4}]),
          event(3,'actual-kernel',12,16,device='cuda')]
    result=aggregate(write_events(tmp_path/'events.jsonl',rows))
    categories=result['exclusive_CPU_and_linked_kernel_categories']
    assert sum(v['CPU_self_us'] for v in categories.values())==10
    assert result['CUDA_kernel_duration_sum_us']==result['CPU_event_linked_kernel_sum_us']==4
    assert categories['context_encoder']['linked_kernel_us']==4
    assert result['annotation_ranges']['EOT8_forward']['CPU_inclusive_us']==10
    assert not result['profiled_timings_are_throughput_or_cost_units']


def test_kernel_intervals_use_union_not_sum_for_overlapping_streams(tmp_path):
    assert union_us([(0,4),(2,6),(8,10)])==8
    rows=[event(0,'kernel1',0,4,device='cuda'),event(1,'kernel2',2,6,device='cuda')]
    result=aggregate(write_events(tmp_path/'events.jsonl',rows))
    assert result['CUDA_kernel_duration_sum_us']==8
    assert result['CUDA_busy_interval_union_us_by_device']=={'0':6}


@pytest.mark.parametrize('defect',['cycle','outside_parent','nonfinite','bad_index','negative_kernel','bad_device','self_time_claim','no_rows'])
def test_malformed_or_unsupported_profiler_records_reject(tmp_path,defect):
    rows=[event(0,PREFIX+'EOT8_forward',0,10,self_cpu=2),event(1,'aten::mm',1,9,0,self_cpu=8)]
    if defect=='cycle':rows[0]['parent']=0
    elif defect=='outside_parent':rows[1]['end_us']=11
    elif defect=='nonfinite':rows[1]['self_cpu_us']=float('nan')
    elif defect=='bad_index':rows[1]['index']=7
    elif defect=='negative_kernel':rows[1]['linked_kernels']=[{'name':'bad','duration_us':-1,'device':0}]
    elif defect=='bad_device':rows[1]['device']='cuda-fixture'
    elif defect=='self_time_claim':rows[0]['self_cpu_us']=10
    else:rows=[]
    with pytest.raises(ValueError):aggregate(write_events(tmp_path/'events.jsonl',rows))


def test_trace_size_and_count_limits_fail_instead_of_truncating(tmp_path,monkeypatch):
    import models.eot_profile_events as e
    path=write_events(tmp_path/'events.jsonl',[event(0,'x',0,1),event(1,'y',1,2)])
    monkeypatch.setattr(e,'MAX_EVENTS',1)
    with pytest.raises(ValueError,match='Too many'):aggregate(path)
    monkeypatch.setattr(e,'MAX_EVENT_BYTES',1)
    with pytest.raises(ValueError,match='too large'):aggregate(path)


def test_annotation_failure_restores_original_methods_and_scientific_identity(upstream):
    configure_execution('shared-fp64-native',device='cpu');model,_,_=tiny_fixture(upstream)
    apply_model_policy(model,'shared-fp64-native');model.eval();before=model_execution_contract(model)
    owners=[model,model.encoder,model.predictor,model.encoder.patch_embed,*model.encoder.blocks,*model.predictor.predictor_blocks]
    originals={id(owner):{k:v for k,v in owner.__dict__.items() if k in ('forward','normalize','_validate_masks','loss_per_example')} for owner in owners}
    with pytest.raises(RuntimeError,match='deliberate'):
        with annotate(model):raise RuntimeError('deliberate')
    assert before==model_execution_contract(model)
    assert originals=={id(owner):{k:v for k,v in owner.__dict__.items() if k in ('forward','normalize','_validate_masks','loss_per_example')} for owner in owners}


@pytest.fixture(scope='module')
def packet(tmp_path_factory):
    path=tmp_path_factory.mktemp('profile-packet')/'packet';metadata,spec=build(ROOT,path)
    return path/'handoff',metadata,digest(path/'handoff/PROFILE-HANDOFF.json')


def test_packet_preserves_one_training_clip_reference_and_unarmed_original_sources(packet):
    root,metadata,sha=packet;actual,parent=verify(root,sha)
    assert actual==metadata and actual['launch_authorized'] is False
    assert parent['packaged_training_clips']==1 and len(parent['parent_candidate_binding']['source_sha256'])==61
    assert [n for n in actual['files'] if n.endswith('.pt')]==['input/prior-EOT8-derivatives.pt','input/training-clip.pt']
    assert actual['paid_compute'] is False and actual['limits']['max_seconds']==1200
    reference=torch.load(root/'input/prior-EOT8-derivatives.pt',map_location='cpu',weights_only=True)
    assert reference['loss'].shape==torch.Size([])


@pytest.mark.parametrize('member',['models/eot_profile.py','REUSE-HANDOFF.json','input/prior-EOT8-derivatives.pt'])
def test_packet_source_or_reference_corruption_rejects_before_worker(packet,tmp_path,member):
    source,_,sha=packet;destination=tmp_path/'copy';shutil.copytree(source,destination)
    with (destination/member).open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError):verify(destination,sha)


def launch(sha):
    return {'kind':'eot8-work-profile-launch-v1','authorized':True,'attended':True,'paid_compute':False,
            'research_admitted':False,'diagnostic_only':True,'handoff_sha256':sha,'expected_host':'linux10609',
            'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':'GPU-4e2a7c77-19bd-689f-6c08-60eb774c0d52',
            'expected_driver':'580.178.04','max_seconds':1200,'worker_seconds':480,'allocator_gib':7,'automatic_retries':0}


@pytest.mark.parametrize('key,value',[('authorized',False),('paid_compute',True),('max_seconds',2400),
    ('worker_seconds',1080),('attended',False),('automatic_retries',1),('kind','study'),('allocator_gib',8)])
def test_profile_domain_and_resource_bounds_require_separate_plan(key,value):
    p=launch('fixture');p[key]=value
    with pytest.raises(ValueError):validate_launch(p,'fixture')


@pytest.fixture(scope='module')
def rehearsal(packet,tmp_path_factory):
    root,_,sha=packet;output=tmp_path_factory.mktemp('profile-workers')/'results'
    child=subprocess.run([sys.executable,'-B',str(root/'scripts/check_eot_profile.py'),'--diagnostic','--tiny','--device','cpu',
        '--handoff-sha256',sha,'--upstream',str(ROOT/'external/vjepa2'),'--output',str(output)],capture_output=True,text=True,timeout=310)
    assert child.returncode==0,child.stdout+child.stderr
    return root,output,sha


def test_two_actual_fresh_tiny_upstream_workers_and_profile_observer_are_exact(rehearsal):
    root,output,sha=rehearsal;summary=summarize(root,output,sha,True)
    assert summary==read_record(output/'summary.json') and summary['passed']
    assert summary['observer_loss_and_input_gradient_exact'] and summary['fresh_process_loss_and_input_gradient_replay_exact']
    assert summary['domain']=='CPU-fixture' and not summary['research_admitted'] and not summary['optimization_adopted']
    for replay in (0,1):
        w=read_record(output/'workers'/str(replay)/'worker.json')
        assert w['trace_summary']['annotation_ranges']['context_encoder']['count']==2
        assert w['trace_summary']['CUDA_events']==0
        values=torch.load(output/'workers'/str(replay)/'passes.pt',map_location='cpu',weights_only=True)
        assert all(values[phase+'_loss'].shape==torch.Size([]) for phase in ('before-0','before-1','draw-before','profiled-draw','draw-after','after'))


@pytest.mark.parametrize('defect',['missing_worker','receipt_hash','trace_change','trace_claim','domain','seed','options','observer_gradient','runtime','fixture','memory','clock','artifact_kind','draw_scope'])
def test_independent_reaggregation_rejects_incomplete_tampered_or_relabelled_evidence(rehearsal,tmp_path,defect):
    root,source,sha=rehearsal;output=tmp_path/'results';shutil.copytree(source,output)
    folder=output/'workers/1';path=folder/'worker.json';w=read_record(path)
    if defect=='missing_worker':path.unlink()
    elif defect=='receipt_hash':
        with path.open('ab') as f:f.write(b'changed')
    elif defect=='trace_change':
        with (folder/'profile-events.jsonl').open('a') as f:f.write('{}\n')
    else:
        if defect=='trace_claim':w['trace_summary']['events']+=1
        elif defect=='domain':w['device']='cuda';w['tiny_fixture']=False
        elif defect=='seed':w['contract']['seeds']['attack'][0]+=1
        elif defect=='options':w['contract']['profiler_options']['record_shapes']=True
        elif defect=='runtime':w['contract']['runtime']['matmul_tf32']=True
        elif defect=='fixture':w['contract']['data']['initial_fixture']['sha256']='changed'
        elif defect=='memory':w['timings']['before-0']['peak_allocated_bytes']=1
        elif defect=='clock':w['wall_seconds']=61
        elif defect=='artifact_kind':w['contract']['kind']='study'
        elif defect=='draw_scope':w['contract']['EOT_control_draws']=1
        else:
            artifact=folder/'passes.pt';values=torch.load(artifact,map_location='cpu',weights_only=True)
            values['profiled-draw_input'].flatten()[0]+=.1
            with artifact.open('wb') as f:torch.save(values,f)
            w['artifacts']['passes']['sha256']=digest(artifact);w['artifacts']['passes']['tensors']={k:tensor_signature(v) for k,v in values.items()}
        write_record(path,w,replace=True)
        s=read_record(output/'session.json');s['tasks'][1]['worker_sha256']=digest(path);write_record(output/'session.json',s,replace=True)
    with pytest.raises((ValueError,FileNotFoundError)):summarize(root,output,sha,True)


@pytest.mark.parametrize('flags',[['--tiny','--device','cpu'],['--diagnostic','--device','cpu'],
    ['--diagnostic','--tiny','--device','cpu','--max-seconds','0'],['--diagnostic','--tiny','--device','cpu','--worker-seconds','0']])
def test_cli_refuses_missing_opt_in_or_invalid_resource_scope(packet,tmp_path,flags):
    root,_,sha=packet;output=tmp_path/'result'
    child=subprocess.run([sys.executable,'-B',str(root/'scripts/check_eot_profile.py'),*flags,
       '--handoff-sha256',sha,'--upstream',str(ROOT/'external/vjepa2'),'--output',str(output)],capture_output=True,text=True)
    assert child.returncode!=0 and not output.exists()
