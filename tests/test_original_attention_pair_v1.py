"""New-pair resource, typed-file and transport refusal checks; no remote work."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import time

import numpy as np
import pytest
import torch

from models import attention_return_capture_v1 as capture
from models import original_attention_pair_client_v1 as client
from models import original_attention_pair_operator_v1 as op
from models import original_attention_pair_runtime_v1 as runtime
from test_original_attention_pair_packaged_v1 import ROOT,control,authorization


def base(tmp_path,armed=False):
    folder=tmp_path/'client';folder.mkdir();c=control()
    key=ROOT/'runs/pretrained/diagnostic/oracle-boundary-v1-20261104/linux10603/known-hosts'
    (folder/'known-hosts').write_bytes(key.read_bytes());c['known_hosts_sha256']=op.base.old.digest(folder/'known-hosts')
    op.base.old.write_json(folder/'CONTROL.json',c);sha=op.base.old.digest(folder/'CONTROL.json')
    if armed:op.base.old.write_json(folder/'AUTHORIZATION.json',authorization(sha))
    return folder,c,sha


@pytest.mark.parametrize('key,value',[
    ('host','linux10609'),('GPU_attention_allowed',False),('GPU_input_transfer_allowed',False),('kind',op.base.KIND),
    ('reuse_old_grant',True),('old_refusals_waived',True),('paid_compute',True),('research_admitted',True),
    ('timings_usable_for_speed_or_budget',True),('original_four_worker_session_qualified',True),
    ('tasks',[runtime.task(1,False),runtime.task(0,False)])])
def test_changed_domain_refuses(key,value):
    c=control();c[key]=value
    with pytest.raises(ValueError):op.validate_control(c)


@pytest.mark.parametrize('key',['max_seconds','worker_seconds','max_raw_bytes','max_workers','max_CPU_oracle_calls',
    'max_attention_iterations','write_test_bytes','write_test_seconds','setup_ceiling','capture_ceiling','collection_seconds'])
def test_resource_bounds_cannot_expand(key):
    c=copy.deepcopy(control());c['policy'][key]+=1
    with pytest.raises(ValueError):op.validate_control(c)


@pytest.mark.parametrize('key',['max_seconds','worker_seconds','max_fresh_workers','max_CPU_oracle_calls',
    'max_attention_iterations','raw_capture_ceiling','write_test_bytes','setup_ceiling','capture_ceiling','maximum_new_attempts'])
def test_grant_scope_cannot_expand(key):
    a=authorization('a'*64);a[key]+=1
    with pytest.raises(ValueError):op.validate_authorization(a,'a'*64)


def test_consumed_pre_transfer_grant_cannot_authorize_GPU_pair():
    a=json.loads((ROOT/'runs/pretrained/diagnostic/oracle-boundary-v1-20261104/linux10603/AUTHORIZATION.json').read_text())
    with pytest.raises(ValueError):op.validate_authorization(a,a['control_sha256'])


def test_unarmed_refuses_before_any_session_claim_or_process(tmp_path,monkeypatch):
    folder,c,sha=base(tmp_path)
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('No process before grant.'))
    with pytest.raises(ValueError,match='UNARMED'):client.run_client(folder,sha)
    assert sorted(p.name for p in folder.iterdir())==['CONTROL.json','known-hosts']


def test_exact_payload_and_prior_bindings_verify_without_processes(tmp_path,monkeypatch):
    folder,c,sha=base(tmp_path,True)
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('No process in verification.'))
    assert client.verify_local(folder,sha)==c
    assert len(c['prior_bindings_sha256'])>=1131
    assert len(client.sources(c))==13 and sum(map(len,client.sources(c).values()))<2*2**20


def test_consumed_pair_claim_refuses_before_process(tmp_path,monkeypatch):
    folder,c,sha=base(tmp_path,True);op.base.old.write_json(folder/'ATTEMPT.json',{'consumed':True})
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('Never relaunch consumed claim.'))
    with pytest.raises(FileExistsError):client.run_client(folder,sha)


def test_full_bootstrap_imports_only_stdlib_and_never_initializes_cuda(tmp_path):
    folder,c,sha=base(tmp_path);command=client.command_for(folder,c)
    assert command[0]=='ssh' and command[-2]=='kdu@linux10603.dc.engr.scu.edu'
    assert all(n in command for n in ('ForwardAgent=no','StrictHostKeyChecking=yes','ControlMaster=no','ClearAllForwardings=yes'))
    code=shlex.split(command[-1])[-1]
    harness='''
import sys,json
code=sys.argv[1];code=code[:code.rindex('sys.modules["models.original_attention_pair_operator_v1"].remote_main')]
exec(code)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
from models import original_attention_pair_operator_v1 as op
op.validate_control(control)
assert op.KIND=='original-attention-pair-capture-v1'
print('stdlib bootstrap passed; no dispatch')
'''
    result=subprocess.run([sys.executable,'-I','-B','-c',harness,code],input=client.bootstrap_payload(c),capture_output=True,timeout=15)
    assert result.returncode==0,result.stderr.decode()
    assert len(command[-1])<8000 and len(client.bootstrap_payload(c))<2**20


@pytest.mark.parametrize('foreign',['123, /usr/NX/bin/nxnode.bin, 175','456, other, 800','999, self, 1\n456, other, 1'])
def test_foreign_compute_refuses_without_stopping_other_process(monkeypatch,foreign):
    c=control();answers=iter([f"{c['expected_gpu']}, {c['expected_gpu_uuid']}, 8192, 94, 580.178.04",foreign])
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:next(answers))
    monkeypatch.setattr(os,'kill',lambda *a:pytest.fail('Foreign processes stay untouched.'))
    with pytest.raises(ValueError,match='Another CUDA'):op.base.gpu_gate(c,allowed_pid=999)


def tensors(dtype=torch.float32):
    return {n:torch.arange(24,dtype=dtype).reshape(1,2,3,4).transpose(1,2) for n in capture.KEYS}


@pytest.mark.parametrize('dtype,label',[('float32',torch.float32),('float64',torch.float64)])
def test_typed_capture_preserves_values_dtype_and_original_noncontiguous_stride(tmp_path,dtype,label):
    values=tensors(label);shape=list(values['output'].shape)
    item=capture.save_return(tmp_path/'return.npz',values,dtype=dtype,device='cpu',shape=shape,limit=2**20)
    saved=capture.load_return(tmp_path,item,dtype=dtype,shape=shape,limit=2**20)
    for name,array in saved.items():
        assert array.tobytes()==values[name].contiguous().numpy().tobytes()
        assert item['source_tensors'][name]['stride']==list(values[name].stride())
        assert array.dtype==(np.float32 if label==torch.float32 else np.float64)


@pytest.mark.parametrize('defect',['dtype','device','shape','missing','oversize','existing'])
def test_writer_refuses_unapproved_cast_domain_extent_or_overwrite(tmp_path,defect):
    values=tensors();shape=list(values['output'].shape);device='cpu';limit=2**20
    if defect=='dtype':values['dq']=values['dq'].double()
    elif defect=='device':device='cuda'
    elif defect=='shape':shape=[1,2,3,4]
    elif defect=='missing':values.pop('dv')
    elif defect=='oversize':limit=10
    elif defect=='existing':(tmp_path/'return.npz').write_bytes(b'preserve')
    with pytest.raises((ValueError,FileExistsError)):capture.save_return(tmp_path/'return.npz',values,dtype='float32',device=device,shape=shape,limit=limit)
    if defect=='existing':assert (tmp_path/'return.npz').read_bytes()==b'preserve'


def test_nonfinite_result_is_preserved_before_numerical_acceptance(tmp_path):
    values=tensors();values['dq'][0,0,0,0]=float('nan')
    item=capture.save_return(tmp_path/'return.npz',values,dtype='float32',device='cpu',shape=list(values['output'].shape),limit=2**20)
    assert not item['arrays']['dq']['finite'] and item['arrays']['dq']['max_abs'] is None
    saved=capture.load_return(tmp_path,item,dtype='float32',shape=list(values['output'].shape),limit=2**20)
    assert np.isnan(saved['dq'][0,0,0,0])


@pytest.mark.parametrize('defect',['bytes','hash','path','dtype','shape','signature'])
def test_saved_return_tampering_refuses(tmp_path,defect):
    values=tensors();shape=list(values['output'].shape)
    item=capture.save_return(tmp_path/'return.npz',values,dtype='float32',device='cpu',shape=shape,limit=2**20)
    dtype='float32'
    if defect=='bytes':item['bytes']+=1
    elif defect=='hash':item['sha256']='0'*64
    elif defect=='path':item['file']='../return.npz'
    elif defect=='dtype':dtype='float64'
    elif defect=='shape':shape=[1,2,3,4]
    elif defect=='signature':item['arrays']['dv']['sha256']='0'*64
    with pytest.raises(ValueError):capture.load_return(tmp_path,item,dtype=dtype,shape=shape,limit=2**20)


def wire(c,sha,*,mutate=None):
    root={'operator-control.json':op.base.old.json_bytes(c),'operator-result.json':op.base.old.json_bytes({
        'kind':op.KIND,'host':c['host'],'control_sha256':sha,'work_finished':True,'GPU_attention_allowed':True,
        'paid_compute':False,'original_four_worker_session_qualified':False,'performance_qualified':False,
        'research_admitted':False,'old_refusals_waived':False,'scientific_review_complete':False,
        'pair_checks_passed':False,'completed_workers':0,'error':'LOCAL TEST: proof refused before model.'}),
        'results/retained-return.npz':b'LOCAL TEST NPZ bytes; transport checks identity, not numerical acceptance.'}
    files={n:{'bytes':len(b),'mtime_ns':1234567890123456789,'sha256':hashlib.sha256(b).hexdigest()} for n,b in root.items()}
    if mutate:mutate(files)
    out=io.BytesIO();out.write(op.base.old.json_bytes({'kind':op.KIND,'phase':'scratch-created','host':c['host'],'control_sha256':sha,
        'scratch_directory':'/var/tmp/'+op.PREFIX+'linux10603-fixture'}))
    out.write(op.base.old.json_bytes({'kind':op.KIND,'phase':'collection','control_sha256':sha,'files':files,
        'logical_bytes':sum(x['bytes'] for x in files.values()),'scientific_review_complete':False,'model_restarted_for_collection':False}))
    with tarfile.open(mode='w',fileobj=out) as archive:
        for n,data in root.items():
            m=tarfile.TarInfo(n);m.size=len(data);archive.addfile(m,io.BytesIO(data))
    return out.getvalue()


def test_transport_includes_NPZ_and_preserves_exact_modification_time_without_requalifying(tmp_path):
    c=control();sha=hashlib.sha256(op.base.old.json_bytes(c)).hexdigest();path=tmp_path/'response.bin';path.write_bytes(wire(c,sha))
    result=client.verify_response(path,tmp_path/'saved',c,sha)
    assert result['transport_verified'] and result['members_verified']==3 and not result['pair_checks_passed']
    assert not result['scientific_review_complete'] and not result['model_restarted_for_collection']
    assert (tmp_path/'saved/results/retained-return.npz').stat().st_mtime_ns==1234567890123456789
    with pytest.raises(FileExistsError):client.verify_response(path,tmp_path/'saved',c,sha)


@pytest.mark.parametrize('defect',['hash','bytes','unsafe','truncated','relabel'])
def test_bad_transport_preserves_partial_and_refuses(tmp_path,defect):
    c=control();sha=hashlib.sha256(op.base.old.json_bytes(c)).hexdigest()
    def mutate(files):
        if defect=='hash':files['results/retained-return.npz']['sha256']='0'*64
        elif defect=='bytes':files['results/retained-return.npz']['bytes']+=1
        elif defect=='unsafe':files['../outside']=files.pop('results/retained-return.npz')
    data=wire(c,sha,mutate=mutate)
    if defect=='truncated':data=data[:data.index(b'operator-control.json')+200]
    if defect=='relabel':data=data.replace(op.KIND.encode(),op.base.KIND.encode(),1)
    path=tmp_path/'response.bin';path.write_bytes(data)
    with pytest.raises((ValueError,tarfile.ReadError,EOFError)):client.verify_response(path,tmp_path/'saved',c,sha)
    assert path.exists() and not (tmp_path/'outside').exists()


def test_one_local_fake_connection_sends_full_bound_packet_and_retains_claim(tmp_path):
    folder,c,sha=base(tmp_path,True);response=tmp_path/'wire.bin';response.write_bytes(wire(c,sha))
    script=tmp_path/'fake-connection.py';script.write_text('''
import sys,json,hashlib
from pathlib import Path
bundle=json.loads(sys.stdin.buffer.readline());auth=json.loads(sys.stdin.buffer.readline())
assert auth['kind']=='original-attention-pair-resource-exception-v1'
assert len(bundle['sources'])==13
data=sys.stdin.buffer.read();assert len(data)==85622679
assert hashlib.sha256(data).hexdigest()==bundle['control']['archive_sha256']
sys.stderr.write('LOCAL fake connection; no SSH or computation.\\n')
sys.stdout.buffer.write(Path(sys.argv[1]).read_bytes())
''')
    result=client.run_client(folder,sha,ssh_override=[sys.executable,'-I','-B',str(script),str(response)])
    assert result['transport_verified'] and not result['pair_checks_passed']
    claim=op.base.old.digest(folder/'ATTEMPT.json');assert not any(folder.rglob('upload-stream.bin'))
    with pytest.raises(FileExistsError):client.run_client(folder,sha,ssh_override=[sys.executable,'-c','raise AssertionError()'])
    assert op.base.old.digest(folder/'ATTEMPT.json')==claim


def test_failed_full_extent_proof_stops_before_archive_or_any_scientific_child(tmp_path,monkeypatch):
    c=control();sha=hashlib.sha256(op.base.old.json_bytes(c)).hexdigest();calls=[]
    identity={'device':tmp_path.stat().st_dev,'inode':tmp_path.stat().st_ino}
    monkeypatch.setattr(op,'readiness',lambda c:identity)
    monkeypatch.setattr(op.base,'gpu_gate',lambda *a,**k:{'LOCAL_TEST':True})
    monkeypatch.setattr(op.tempfile,'mkdtemp',lambda **k:str(tmp_path/'scratch'))
    (tmp_path/'scratch').mkdir(mode=0o700)
    def failed(command,folder,**kw):
        calls.append(kw['role'])
        assert kw['role']=='proof' and 0<kw['deadline']-time.monotonic()<=60
        assert kw['file_limit']==3*2**30
        return {'exit_status':1}
    monkeypatch.setattr(op.base.primitives,'supervise',failed)
    class NoRead:
        def read(self,*a):pytest.fail('Archive cannot arrive after rejected proof.')
    output=io.BytesIO();state=op.run_remote(c,authorization(sha),NoRead(),output,client.sources(c))
    assert calls==['proof'] and state['work_finished'] and not state['pair_checks_passed']
    assert state['deadline_unix']-state['started_unix']==600 and 'workers_started' not in state
    assert not (tmp_path/'scratch/results').exists() and not (tmp_path/'scratch/emergency-space.bin').exists()
    assert not state['original_four_worker_session_qualified'] and state['error_type']=='RuntimeError'


@pytest.mark.parametrize('defect',['tensor_limit','raw_category','external_hardlink','symlink'])
def test_storage_refuses_oversize_or_unowned_file_aliases(tmp_path,defect):
    folder=tmp_path/'scratch';folder.mkdir(mode=0o700);(folder/'results').mkdir()
    parent=Path('/var/tmp').stat();identity={'device':folder.stat().st_dev,'inode':parent.st_ino}
    assert identity['device']==parent.st_dev
    if defect=='tensor_limit':
        with (folder/'results/oversize.npz').open('wb') as f:f.truncate(32*2**20+1)
    elif defect=='raw_category':
        for i in range(11):
            with (folder/f'results/return-{i}.npz').open('wb') as f:f.truncate(31*2**20)
    else:
        (tmp_path/'outside').write_bytes(b'preserved')
        if defect=='external_hardlink':os.link(tmp_path/'outside',folder/'results/alias.npz')
        else:(folder/'results/alias.npz').symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError):op.storage_gate(folder,identity)
    if (tmp_path/'outside').exists():assert (tmp_path/'outside').read_bytes()==b'preserved'


def test_atomic_capture_name_pair_is_allowed_but_collection_never_omits_NPZ(tmp_path):
    folder=tmp_path/'scratch';folder.mkdir(mode=0o700);(folder/'results').mkdir()
    p=folder/'results/return.npz';p.write_bytes(b'LOCAL TEST captured data');os.link(p,p.with_name(p.name+'.partial'))
    files,_=op.base.inventory(folder,running=True)
    assert len(files)==2
    p.with_name(p.name+'.partial').unlink()
    output=io.BytesIO();op.result_stream(folder,'a'*64,output);output.seek(0)
    header=json.loads(output.readline());assert 'results/return.npz' in header['files']
    with tarfile.open(mode='r|',fileobj=output) as archive:assert [x.name for x in archive]==['results/return.npz']


def test_outer_clock_never_resets_for_late_worker_issuance(tmp_path):
    (tmp_path/'code').mkdir();(tmp_path/'code/FIXED-ATTENTION-HANDOFF.json').write_text('{}')
    began=time.time()-118
    runtime.initialize(tmp_path,tmp_path/'code',local=True,control_sha=None,started=began,deadline=began+120)
    before=(tmp_path/'results/pair.json').read_bytes()
    with pytest.raises(TimeoutError):runtime.issue_task(tmp_path,tmp_path/'code',0,local=True,control_sha=None)
    assert (tmp_path/'results/pair.json').read_bytes()==before
    assert not (tmp_path/'results/workers/encoder-1-math-0/authorization.json').exists()


@pytest.mark.parametrize('visibility',[None,'0'])
def test_rejected_first_worker_never_starts_candidate_and_preserves_CUDA_visibility(tmp_path,monkeypatch,visibility):
    c=control();sha=hashlib.sha256(op.base.old.json_bytes(c)).hexdigest();calls=[];seen=[]
    if visibility is None:monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    else:monkeypatch.setenv('CUDA_VISIBLE_DEVICES',visibility)
    folder=tmp_path/'scratch';folder.mkdir(mode=0o700)
    monkeypatch.setattr(op,'readiness',lambda c:{'device':folder.stat().st_dev,'inode':folder.stat().st_ino})
    monkeypatch.setattr(op.tempfile,'mkdtemp',lambda **k:str(folder))
    monkeypatch.setattr(op.base,'gpu_gate',lambda *a,**k:{'LOCAL_TEST':True})
    monkeypatch.setattr(op,'storage_gate',lambda *a,**k:{'LOCAL_TEST':True})
    monkeypatch.setattr(op.base.primitives,'validate_proof',lambda *a:{'LOCAL_TEST':True})
    def supervised(command,path,**kw):
        calls.append(kw['role'])
        if kw['role']=='proof':return {'exit_status':0}
        seen.append((command,kw))
        assert kw['role']=='worker-0' and 0<kw['deadline']-time.monotonic()<=120
        return {'exit_status':1}
    monkeypatch.setattr(op.base.primitives,'supervise',supervised)
    with (ROOT/c['original_archive_path']).open('rb') as incoming:
        state=op.run_remote(c,authorization(sha),incoming,io.BytesIO(),client.sources(c))
    assert calls==['proof','worker-0'] and state['workers_started']==1 and state['completed_workers']==0
    assert not state['pair_checks_passed'] and state['deadline_unix']-state['started_unix']==600
    assert not (folder/'results/workers/encoder-1-math-0').exists()
    command,kw=seen[0]
    if visibility is None:
        assert command[:3]==['/usr/bin/env','-u','CUDA_VISIBLE_DEVICES'] and 'CUDA_VISIBLE_DEVICES' not in kw['env']
    else:assert command[0]==sys.executable and kw['env']['CUDA_VISIBLE_DEVICES']==visibility
    assert kw['env']['OMP_NUM_THREADS']==kw['env']['MKL_NUM_THREADS']==kw['env']['OPENBLAS_NUM_THREADS']=='4'
    assert kw['env']['TMPDIR']==str(folder/'tmp') and kw['env']['CUDA_CACHE_PATH']==str(folder/'cache/cuda')
    assert op.pair_status(folder)['pair_checks_passed'] is False
