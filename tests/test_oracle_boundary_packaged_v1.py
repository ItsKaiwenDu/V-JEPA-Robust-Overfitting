"""Cold extracted original worker and real tiny CPU oracle; no CUDA or SSH."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest
import torch
from models import oracle_boundary_operator_v1 as op
from models import oracle_boundary_observer_v1 as observer
from models import oracle_boundary_client_v1 as client
from models.fixed_attention_packet import build
from models.study_artifacts import read_record,write_record
from models.cuda_profile import tensor_signature
from test_oracle_boundary_v1 import control,authorization,ROOT,PACKET


def overlay(folder):
    c=control()
    for name,data in client.sources(c).items():
        path=folder/'overlay'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return c


def command(folder,*,tiny=True,sha=None):
    result=[sys.executable,'-I','-B',str(folder/'overlay/scripts/capture_oracle_boundary_v1.py'),
        '--packet',str(folder/'code'),'--scratch',str(folder),'--upstream',str(ROOT/'external/vjepa2'),
        '--launcher-started',str(time.time())]
    if tiny:result+=['--local-tiny-rehearsal']
    if sha:result+=['--control-sha256',sha]
    return result


def environment():
    return {**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1',
        'PYTHONPATH':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}


@pytest.fixture(scope='module')
def tiny_packet(tmp_path_factory,record_testsuite_property):
    root=tmp_path_factory.mktemp('boundary-cold-packet');manifest,spec=build(ROOT,root/'build',tiny=True)
    record_testsuite_property('tiny_original_packet_directory',str(root))
    return root,manifest,spec


def prepare_tiny(tmp_path,tiny_packet,*,refusal=False):
    root,manifest,spec=tiny_packet
    op.extract_packet(root/'build/vjepa-fixed-attention.tar.gz',spec,tmp_path/'code');overlay(tmp_path)
    if refusal:
        # Explicit local negative fixture, never a change to live/original files,
        # oracle code, gate, or tolerances. Update its own bound descriptors.
        packet=tmp_path/'code';p=packet/'fixtures/encoder-1/oracle_output.npy';array=np.load(p,allow_pickle=False)
        array=array+.01
        with p.open('wb') as f:np.save(f,array,allow_pickle=False)
        fpath=packet/'fixtures/encoder-1/fixture.json';f=read_record(fpath)
        f['files']['oracle_output'].update(sha256=op.old.digest(p),bytes=p.stat().st_size,tensor=tensor_signature(torch.from_numpy(array)))
        write_record(fpath,f,replace=True)
        mpath=packet/'FIXED-ATTENTION-HANDOFF.json';m=json.loads(mpath.read_text())
        m['fixtures']['encoder-1']['sha256']=op.old.digest(fpath)
        for changed in (p,fpath):m['files'][str(changed.relative_to(packet))].update(sha256=op.old.digest(changed),bytes=changed.stat().st_size)
        mpath.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
    neutral=tmp_path/'neutral';neutral.mkdir();return neutral


@pytest.fixture(scope='module')
def captures(tmp_path_factory,tiny_packet,record_testsuite_property):
    result={}
    for name,negative in [('passing',False),('refusal',True)]:
        folder=tmp_path_factory.mktemp('boundary-'+name);neutral=prepare_tiny(folder,tiny_packet,refusal=negative)
        done=subprocess.run(command(folder),cwd=neutral,env=environment(),capture_output=True,text=True,timeout=45)
        (folder/'cold.stdout').write_text(done.stdout);(folder/'cold.stderr').write_text(done.stderr)
        assert done.returncode==(1 if negative else 0),done.stdout+done.stderr
        result[name]=(folder,done);record_testsuite_property('cold_'+name+'_directory',str(folder))
    return result


@pytest.mark.parametrize('name',['passing','refusal'])
def test_real_original_oracle_return_is_saved_once_before_original_gate(captures,name):
    folder,done=captures[name];state=read_record(folder/'results/boundary.json')
    old=read_record(folder/'results/original-worker/worker.json')
    assert state['capture_complete'] and state['original_oracle_calls']==state['original_oracle_returns']==1
    assert state['events'][:2]==['original_oracle_call','actual_return_saved_before_original_gate']
    assert state['forbidden_attention_calls']==0 and state['profile_restored'] and state['trace_restored']
    assert state['original_gate_passed']==(name=='passing')
    assert state['outcome']==('captured-gate-pass' if name=='passing' else 'captured-original-refusal')
    assert not state['original_full_worker_accepted'] and not old['complete'] and not old['passed']
    assert old['error_type']==('BoundaryReached' if name=='passing' else 'ValueError')
    assert old['stage']=='independent_CPU_FP64_forward_and_analytical_QKV_oracle'
    assert 'device_input_strides' not in old and 'device_inputs' not in old['evidence']
    assert not (folder/'results/original-worker/device-inputs.pt').exists()
    assert not old['timings'] and not state['GPU_attention_executed'] and not state['GPU_input_transfer_completed']
    assert state['post_gate_context']['CUDA_initialized'] is False
    assert not state['detailed_introspection_before_first_oracle']
    if name=='passing':assert state['clone_call_intercepted_before_body']
    else:
        assert not state['clone_call_intercepted_before_body']
        assert old['error']=='Fixed oracle bookkeeping failed; preserve input snapshot.'


@pytest.mark.parametrize('name',['passing','refusal'])
def test_actual_arrays_reproduce_unmodified_gate_and_inputs(captures,name):
    folder,_=captures[name];state=read_record(folder/'results/boundary.json');old=read_record(folder/'results/original-worker/worker.json')
    item=state['computed_return_artifact'];path=folder/'results'/item['file']
    assert op.old.digest(path)==item['sha256'] and path.stat().st_size==item['bytes']
    assert item['source_dtypes']=={n:'torch.float64' for n in observer.OUTPUTS}
    with np.load(path,allow_pickle=False) as values:
        assert set(values.files)==set(observer.OUTPUTS)
        for name in observer.OUTPUTS:
            actual=values[name];reference=np.load(folder/f'code/fixtures/encoder-1/oracle_{name}.npy',allow_pickle=False)
            passed=bool(torch.allclose(torch.from_numpy(actual),torch.from_numpy(reference),rtol=1e-12,atol=1e-14))
            gate=old['oracle_bookkeeping'][name]
            assert passed is gate['passed'] and gate['rtol']==1e-12 and gate['atol']==1e-14
            assert float(np.max(np.abs(actual-reference)))==gate['max_absolute_error']
    inputs=torch.load(folder/'results/original-worker/inputs.pt',map_location='cpu',weights_only=True)
    for name,tensor in inputs.items():
        expected=np.load(folder/f'code/fixtures/encoder-1/{name}.npy',allow_pickle=False)
        assert np.array_equal(tensor.numpy(),expected)
    for bound in state['function_identities'].values():
        assert bound['bytecode_matches_bound_source'] and op.old.digest(folder/'code'/bound['source'])==bound['sha256']


def test_original_failure_is_retained_while_missing_actual_values_are_now_present(captures):
    folder,_=captures['refusal'];state=read_record(folder/'results/boundary.json')
    assert state['original_gate_checks']['output']['passed'] is False
    assert state['original_gate_checks']['output']['max_absolute_error']>.009
    assert state['capture_complete'] and not state['old_failure_waived'] and not state['research_admitted']


def test_rerunning_cold_capture_refuses_without_touching_original_files(captures):
    folder,_=captures['passing'];paths=list((folder/'results').rglob('*'))
    before={str(p):op.old.digest(p) for p in paths if p.is_file()}
    done=subprocess.run(command(folder),cwd=folder/'neutral',env=environment(),capture_output=True,text=True,timeout=30)
    assert done.returncode!=0 and 'never repeat' in done.stderr
    assert before=={p:op.old.digest(p) for p in before}


def harness(folder,body):
    script=folder/'harness.py'
    script.write_text('''
import sys,signal,time,runpy,json
from pathlib import Path
folder=Path(sys.argv[1]);sys.path.insert(0,str(folder/'overlay'));sys.path.insert(1,str(folder/'code'))
'''+body)
    args=command(folder)[4:]
    return subprocess.run([sys.executable,'-I','-B',str(script),str(folder),*args],cwd=folder/'neutral',
        env=environment(),capture_output=True,text=True,timeout=45)


def test_original_alarm_profile_and_trace_are_restored_without_time_reset(tmp_path,tiny_packet):
    prepare_tiny(tmp_path,tiny_packet)
    done=harness(tmp_path,'''
def previous(*a):pass
signal.signal(signal.SIGALRM,previous);signal.setitimer(signal.ITIMER_REAL,100,0)
start=time.monotonic();sys.argv=[str(folder/'overlay/scripts/capture_oracle_boundary_v1.py'),*sys.argv[2:]]
try:runpy.run_path(sys.argv[0],run_name='__main__')
except SystemExit as exc:assert exc.code==0
remaining,interval=signal.getitimer(signal.ITIMER_REAL)
assert signal.getsignal(signal.SIGALRM) is previous
assert 0<remaining<100 and abs(remaining-(100-(time.monotonic()-start)))<1
assert sys.getprofile() is None and sys.gettrace() is None
signal.setitimer(signal.ITIMER_REAL,0);print('RESTORED_WITHOUT_RESET')
''')
    assert done.returncode==0 and 'RESTORED_WITHOUT_RESET' in done.stdout,done.stdout+done.stderr


def test_capture_write_failure_preserves_original_error_and_never_retries(tmp_path,tiny_packet):
    prepare_tiny(tmp_path,tiny_packet)
    done=harness(tmp_path,'''
from models import oracle_trace
calls=[]
def fail(*a,**k):calls.append(1);raise OSError('injected capture write refusal')
oracle_trace.save_arrays=fail
sys.argv=[str(folder/'overlay/scripts/capture_oracle_boundary_v1.py'),*sys.argv[2:]]
try:runpy.run_path(sys.argv[0],run_name='__main__')
except SystemExit as exc:assert exc.code==2
assert calls==[1] and sys.getprofile() is None
''')
    assert done.returncode==0,done.stdout+done.stderr
    state=read_record(tmp_path/'results/boundary.json');old=read_record(tmp_path/'results/original-worker/worker.json')
    assert not state['capture_complete'] and state['outcome']=='technical-failure'
    assert old['error_type']=='OSError' and 'injected capture write refusal' in old['error']
    assert not (tmp_path/'results/computed-oracle.npz').exists()


@pytest.mark.parametrize('hook',['profile','trace'])
def test_existing_observation_hook_refuses_before_original_dispatch(tmp_path,tiny_packet,hook):
    prepare_tiny(tmp_path,tiny_packet)
    done=harness(tmp_path,'''
def existing(*a):return existing
getattr(sys,'set'''+hook+'''')(existing)
sys.argv=[str(folder/'overlay/scripts/capture_oracle_boundary_v1.py'),*sys.argv[2:]]
try:runpy.run_path(sys.argv[0],run_name='__main__')
except ValueError as exc:assert 'existing trace or profile' in str(exc)
else:raise AssertionError('Existing hook must refuse')
assert getattr(sys,'get'''+hook+'''')() is existing
getattr(sys,'set'''+hook+'''')(None)
''')
    assert done.returncode==0,done.stdout+done.stderr
    assert not (tmp_path/'results/original-worker/worker.json').exists()


@pytest.mark.parametrize('change',['alias','worker_code'])
def test_loaded_function_replacement_refuses_before_computation(tmp_path,tiny_packet,change):
    prepare_tiny(tmp_path,tiny_packet)
    body='''
from models import fixed_attention_diagnostic as original
def replaced(*a,**k):raise AssertionError('No replacement may run')
'''+("original.fp64_oracle=replaced\n" if change=='alias' else "original.worker=replaced\n")+'''
sys.argv=[str(folder/'overlay/scripts/capture_oracle_boundary_v1.py'),*sys.argv[2:]]
try:runpy.run_path(sys.argv[0],run_name='__main__')
except ValueError:pass
else:raise AssertionError('Replaced original must refuse')
'''
    done=harness(tmp_path,body);assert done.returncode==0,done.stdout+done.stderr
    assert not (tmp_path/'results/computed-oracle.npz').exists()


@pytest.fixture
def full_bundle(tmp_path):
    c=overlay(tmp_path);op.extract_packet(PACKET/'vjepa-fixed-attention.tar.gz',c['archive_manifest'],tmp_path/'code')
    op.old.write_json(tmp_path/'operator-control.json',c);sha=op.old.digest(tmp_path/'operator-control.json')
    op.old.write_json(tmp_path/'resource-authorization.json',authorization(sha))
    (tmp_path/'neutral').mkdir();return tmp_path,c,sha


def test_full_unchanged_bundle_cold_import_stops_before_dispatch(full_bundle):
    folder,c,sha=full_bundle
    script=folder/'cold-import.py'
    script.write_text('''
import sys,runpy,json,platform
from pathlib import Path
folder=Path(sys.argv[1]);sha=sys.argv[2]
sys.path.insert(0,str(folder/'overlay'));sys.path.insert(1,str(folder/'code'))
from models import oracle_boundary_observer_v1 as observer
calls=[]
class Halt(BaseException):pass
def blocked(original,args,packet,out,**kw):
    import torch
    assert not torch.cuda.is_initialized()
    assert original.worker.__name__=='worker' and original.fp64_oracle.__name__=='fp64_oracle'
    calls.append(1);raise Halt()
observer.observe=blocked
sys.argv=[str(folder/'overlay/scripts/capture_oracle_boundary_v1.py'),'--packet',str(folder/'code'),'--scratch',str(folder),
    '--upstream',str(folder/'no-upstream-needed'),'--control-sha256',sha,'--launcher-started',str(__import__('time').time())]
try:runpy.run_path(sys.argv[0],run_name='__main__')
except Halt:pass
assert calls==[1]
assert not (folder/'results/original-worker/worker.json').exists()
print(json.dumps({'cold_import_complete':True,'original_dispatch_blocked':True,'CUDA_initialized':False}))
''')
    done=subprocess.run([sys.executable,'-I','-B',str(script),str(folder),sha],cwd=folder/'neutral',
        env=environment(),capture_output=True,text=True,timeout=30)
    assert done.returncode==0,done.stdout+done.stderr
    assert json.loads(done.stdout)['original_dispatch_blocked'] is True


@pytest.mark.parametrize('defect',['local_full','old_grant','packet_change','overlay_change','alias_root'])
def test_full_source_or_domain_changes_refuse_before_original_dispatch(full_bundle,defect):
    folder,c,sha=full_bundle;tiny=False
    if defect=='local_full':tiny=True;sha=None
    elif defect=='old_grant':
        old=ROOT/'runs/pretrained/diagnostic/cpu-oracle-control-v1-20261104/linux10609/AUTHORIZATION.json'
        (folder/'resource-authorization.json').write_bytes(old.read_bytes())
    elif defect=='packet_change':
        p=folder/'code/attacks/__init__.py';p.write_bytes(p.read_bytes()+b'\n# altered private test copy\n')
    elif defect=='overlay_change':
        p=folder/'overlay/models/oracle_boundary_observer_v1.py';p.write_bytes(p.read_bytes()+b'\n# altered private test copy\n')
    elif defect=='alias_root':
        (folder/'code').rename(folder/'elsewhere');(folder/'code').symlink_to(folder/'elsewhere',target_is_directory=True)
    done=subprocess.run(command(folder,tiny=tiny,sha=sha),cwd=folder/'neutral',env=environment(),capture_output=True,text=True,timeout=30)
    assert done.returncode!=0,done.stdout+done.stderr
    assert not (folder/'results').exists()
