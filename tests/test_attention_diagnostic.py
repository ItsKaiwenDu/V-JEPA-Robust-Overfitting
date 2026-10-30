"""Source-only packets, real fresh CPU workers and independent evidence refusal."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from models.attention_component import TOLERANCES
from models.attention_diagnostic import summarize,tasks,validate_launch
from models.attention_packet import HOSTS,LIMITS,build,verify
from models.pilot_bundle import digest
from models.study_artifacts import read_record,write_record
from test_projection_reuse import restore_process_settings

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def packet(tmp_path_factory):
    path=tmp_path_factory.mktemp('attention-packet')/'packet';metadata,spec=build(ROOT,path)
    return path/'handoff',metadata,digest(path/'handoff/ATTENTION-HANDOFF.json')


def test_packet_is_source_only_binds_actual_shapes_and_all_original_science(packet):
    root,metadata,sha=packet;m,reference=verify(root,sha)
    assert m==metadata and not m['launch_authorized'] and not m['paid_compute']
    assert len(m['original_scientific_binding']['source_sha256'])==61
    assert not [n for n in m['files'] if n.endswith(('.pt','.pth','.whl'))]
    assert reference['shapes']=={'predictor':{'heads':12,'head_dim':32,'tokens':2048},
                                'encoder-0':{'heads':16,'head_dim':64,'tokens':656},
                                'encoder-1':{'heads':16,'head_dim':64,'tokens':400}}
    assert len(tasks('predictor',False))==4 and len(tasks('encoder',False))==8
    assert all(arm=='math' for _,arm,_ in tasks('all',True))


def test_full_profile_geometries_reconstruct_without_pixels_or_pretrained_model(packet,upstream):
    from models.attention_component import operands
    from models.cuda_profile import tensor_signature
    root,_,sha=packet;_,reference=verify(root,sha)
    for case,expected in reference['shapes'].items():
        tensors,shape=operands(upstream,case,reference,False)
        assert all(shape[k]==expected[k] for k in ('heads','head_dim','tokens'))
        assert all(t.shape==(1,expected['heads'],expected['tokens'],expected['head_dim']) for t in tensors)
        if case.startswith('encoder'):
            assert tensor_signature(shape['positions'])==reference['first_draw_mask_signatures'][int(case[-1])][0]


@pytest.mark.parametrize('member',['models/attention_component.py','attention-reference.json','configs/study.yaml'])
def test_packet_corruption_rejects(packet,tmp_path,member):
    root,_,sha=packet;target=tmp_path/'copy';shutil.copytree(root,target)
    # The selected scientific pin may have a different name; choose the first one.
    if not (target/member).is_file():member=next(iter(json.loads((target/'ATTENTION-HANDOFF.json').read_text())['original_scientific_binding']['source_sha256']))
    with (target/member).open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError):verify(target,sha)


def launch(host='linux10609'):
    return {'kind':'attention-component-launch-v1','authorized':True,'attended':True,'paid_compute':False,
            'research_admitted':False,'diagnostic_only':True,'handoff_sha256':'fixture','expected_host':host,
            'role':HOSTS[host]['role'],'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':HOSTS[host]['uuid'],
            'expected_driver':'580.178.04',**LIMITS}


@pytest.mark.parametrize('key,value',[('authorized',False),('paid_compute',True),('attended',False),('max_seconds',2400),
    ('worker_seconds',480),('allocator_gib',8),('automatic_retries',1),('expected_gpu_uuid','GPU-wrong'),
    ('role','encoder'),('repeats',4),('warmup',0),('kind','study')])
def test_host_role_and_bounds_cannot_be_relabelled(key,value):
    plan=launch();plan[key]=value
    with pytest.raises(ValueError):validate_launch(plan,'fixture','predictor')


@pytest.fixture(scope='module')
def rehearsal(packet,tmp_path_factory):
    root,_,sha=packet;out=tmp_path_factory.mktemp('attention-workers')/'results'
    child=subprocess.run([sys.executable,'-B',str(root/'scripts/check_attention_component.py'),'--diagnostic','--tiny','--device','cpu',
        '--role','all','--handoff-sha256',sha,'--upstream',str(ROOT/'external/vjepa2'),'--output',str(out)],capture_output=True,text=True,timeout=310)
    assert child.returncode==0,child.stdout+child.stderr+'\n'+('\n'.join(p.read_text() for p in out.rglob('worker.log')) if out.exists() else '')
    return root,out,sha


def test_six_actual_fresh_CPU_workers_qualify_only_math_fixture(rehearsal):
    root,out,sha=rehearsal;result=summarize(root,out,sha,'all',True)
    assert result==read_record(out/'summary.json') and result['passed']
    assert result['domain']=='CPU-math-fixture' and not result['CUDA_backend_comparison_qualified']
    assert not result['full_model_correctness_or_speed_qualified'] and not result['research_admitted']
    assert len(read_record(out/'session.json')['tasks'])==6
    for case in result['cases']:
        assert result['cases'][case]['fresh_process_tensor_replay_exact']
        assert result['cases'][case]['efficient_over_math_total_speedup'] is None


def test_downloaded_summary_rebinds_a_verified_checkout_without_editing_source_identity(rehearsal,tmp_path):
    root,source,sha=rehearsal;out=tmp_path/'results';shutil.copytree(source,out)
    session=read_record(out/'session.json');session['upstream']='/nonexistent/remote/linux/upstream'
    write_record(out/'session.json',session,replace=True)
    result=summarize(root,out,sha,'all',True,upstream=ROOT/'external/vjepa2')
    assert result==read_record(out/'summary.json')


@pytest.mark.parametrize('defect',['coverage','receipt_hash','tensor_bytes','domain','seed','geometry','strides','runtime','tolerance',
    'backend_evidence','timing_nan','timing_sum','timing_median','clock','memory','oracle','operands','operand_signature','phase'])
def test_independent_summary_rejects_resigned_tampering(rehearsal,tmp_path,defect):
    root,source,sha=rehearsal;out=tmp_path/'results';shutil.copytree(source,out)
    path=out/'workers/predictor-math-1/worker.json';w=read_record(path);session=read_record(out/'session.json')
    if defect=='coverage':session['tasks'].pop()
    elif defect=='receipt_hash':path.write_text(path.read_text()+'changed')
    elif defect=='tensor_bytes':
        with (path.parent/'values.pt').open('ab') as f:f.write(b'changed')
    elif defect=='domain':w['common']['research_admitted']=True
    elif defect=='seed':w['common']['shape']['seed']+=1
    elif defect=='geometry':w['common']['shape']['tokens']+=1
    elif defect=='strides':w['common']['shape']['strides']['v'][-2]+=1
    elif defect=='runtime':w['common']['runtime']['matmul_tf32']=True
    elif defect=='tolerance':w['common']['tolerances']=copy.deepcopy(TOLERANCES);w['common']['tolerances']['dq']['atol']=1
    elif defect=='backend_evidence':w['backend_profile_events']=[]
    elif defect=='timing_nan':w['timings'][0]['backward_seconds']=float('nan')
    elif defect=='timing_sum':w['timings'][0]['total_seconds']+=1
    elif defect=='timing_median':w['median_seconds']['forward_seconds']+=1
    elif defect=='clock':w['wall_seconds']=31
    elif defect=='memory':w['peak_cuda_allocated_bytes']=1
    elif defect in ('oracle','operands'):
        from models.cuda_profile import tensor_signature
        values=torch.load(path.parent/'values.pt',map_location='cpu',weights_only=True)
        values['oracle_output' if defect=='oracle' else 'q'].add_(.1)
        torch.save(values,path.parent/'values.pt');w['artifact']['sha256']=digest(path.parent/'values.pt')
        w['artifact']['tensor_signatures']={n:tensor_signature(t) for n,t in values.items()}
    elif defect=='operand_signature':w['common']['operand_signatures']['q']['sha256']='0'*64
    elif defect=='phase':w['stage']='untimed_numerical_gate'
    if defect=='timing_nan':
        # The atomic record layer refuses NaN even before independent summary.
        with pytest.raises(ValueError,match='JSON compliant'):write_record(path,w,replace=True)
        return
    if defect not in ('coverage','receipt_hash','tensor_bytes'):
        write_record(path,w,replace=True)
        session['tasks'][1]['worker_sha256']=digest(path)
    write_record(out/'session.json',session,replace=True)
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):summarize(root,out,sha,'all',True)


def test_CLI_requires_optin_and_refuses_unbound_GPU_before_worker(packet,tmp_path):
    root,_,sha=packet
    command=[sys.executable,'-B',str(root/'scripts/check_attention_component.py'),'--role','predictor',
             '--handoff-sha256',sha,'--upstream',str(ROOT/'external/vjepa2'),'--output',str(tmp_path/'run')]
    assert subprocess.run(command,capture_output=True).returncode==2
    assert subprocess.run(command+['--diagnostic'],capture_output=True).returncode==2
    assert not (tmp_path/'run').exists()
