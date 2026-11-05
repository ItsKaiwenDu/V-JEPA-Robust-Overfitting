"""Cold, real original tiny CPU math workers; never CUDA qualification."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from models import original_attention_pair_operator_v1 as op
from models import original_attention_pair_runtime_v1 as runtime
from models import original_attention_pair_client_v1 as client
from models import attention_return_capture_v1 as capture
from models.fixed_attention_packet import build
from models.study_artifacts import read_record,write_record
from models.cuda_profile import tensor_signature

ROOT=Path(__file__).resolve().parents[1]
PACKET=ROOT/'runs/pretrained/diagnostic/fixed-attention-university-20261030/packet'


def control():
    c=json.loads((ROOT/'runs/pretrained/diagnostic/oracle-boundary-v1-20261104/linux10603/CONTROL.json').read_text())
    for name in ('CUDA_setup_allowed','GPU_SDPA_allowed','original_refusals_waived'):c.pop(name,None)
    c.update(kind=op.KIND,role='encoder-1-efficient-math-replay-0-pair',policy=op.POLICY,prerequisite=op.PREREQUISITE,
        tasks=[runtime.task(i,False) for i in (0,1)],GPU_input_transfer_allowed=True,GPU_attention_allowed=True,
        old_refusals_waived=False,timings_usable_for_speed_or_budget=False,original_four_worker_session_qualified=False)
    c['prior_bindings_sha256']={n:sha for n,sha in json.loads((ROOT/'runs/pretrained/diagnostic/original-attention-pair-proposal-20261104/verified-design/sealed-inputs.json').read_text()).items()}
    for name in ('configs/original_attention_pair_proposal_v1.json','report/checks/2026-11-04-oracle-boundary-v1-target-review.json',
            'runs/pretrained/diagnostic/oracle-boundary-v1-20261104/linux10603/ATTEMPT.json'):
        c['prior_bindings_sha256'][name]=op.base.old.digest(ROOT/name)
    c['local_source_sha256']={n:op.base.old.digest(ROOT/n) for n in op.LOCAL_NAMES}
    c['remote_source_sha256']={n:hashlib.sha256(op.OVERLAY_INIT).hexdigest() if n=='models/__init__.py'
        else op.base.old.digest(ROOT/n) for n in op.REMOTE_NAMES}
    return c


def authorization(sha):
    return {'kind':op.AUTH_KIND,'approved':True,'control_sha256':sha,'host':'linux10603','maximum_new_attempts':1,
        'max_seconds':600,'worker_seconds':120,'max_fresh_workers':2,'max_CPU_oracle_calls':2,'max_attention_iterations':12,
        'GPU_input_transfer_allowed':True,'GPU_attention_allowed':True,'write_test_bytes':2684354560,
        'setup_ceiling':3*2**30,'capture_ceiling':9*2**30//4,'raw_capture_ceiling':320*2**20,
        'collection_seconds':600,'paid_compute':False,'reset_old_claims':False,'automatic_retries':0,
        'user_basis':'LOCAL TEST FIXTURE ONLY; no university resources authorized.'}


def overlay(folder):
    c=control()
    for name,data in client.sources(c).items():
        p=folder/'overlay'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    return c


def environment():
    return {**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1',
        'PYTHONPATH':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}


@pytest.fixture(scope='module')
def tiny_packet(tmp_path_factory,record_testsuite_property):
    root=tmp_path_factory.mktemp('original-pair-packet');manifest,spec=build(ROOT,root/'build',tiny=True)
    record_testsuite_property('tiny_original_packet_directory',str(root));return root,manifest,spec


def prepare(folder,tiny_packet,*,negative=False):
    root,manifest,spec=tiny_packet
    op.base.extract_packet(root/'build/vjepa-fixed-attention.tar.gz',spec,folder/'code');overlay(folder)
    if negative:
        # Local fixture-only refusal. Original code, calculations and bounds stay unchanged.
        packet=folder/'code';p=packet/'fixtures/encoder-1/oracle_output.npy';array=np.load(p,allow_pickle=False)+.01
        with p.open('wb') as f:np.save(f,array,allow_pickle=False)
        fp=packet/'fixtures/encoder-1/fixture.json';f=read_record(fp)
        f['files']['oracle_output'].update(sha256=op.base.old.digest(p),bytes=p.stat().st_size,tensor=tensor_signature(torch.from_numpy(array)))
        write_record(fp,f,replace=True);mp=packet/'FIXED-ATTENTION-HANDOFF.json';m=json.loads(mp.read_text())
        m['fixtures']['encoder-1']['sha256']=op.base.old.digest(fp)
        for changed in (p,fp):m['files'][str(changed.relative_to(packet))].update(sha256=op.base.old.digest(changed),bytes=changed.stat().st_size)
        mp.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
    (folder/'neutral').mkdir()


def command(folder):
    return [sys.executable,'-I','-B',str(folder/'overlay/scripts/capture_original_attention_pair_v1.py'),
        '--packet',str(folder/'code'),'--scratch',str(folder),'--upstream',str(ROOT/'external/vjepa2'),'--local-tiny-rehearsal']


@pytest.fixture(scope='module')
def captures(tmp_path_factory,tiny_packet,record_testsuite_property):
    result={}
    for name,negative in [('passing',False),('CPU-refusal',True)]:
        folder=tmp_path_factory.mktemp('original-pair-'+name);prepare(folder,tiny_packet,negative=negative)
        done=subprocess.run(command(folder),cwd=folder/'neutral',env=environment(),capture_output=True,text=True,timeout=125)
        (folder/'cold.stdout').write_text(done.stdout);(folder/'cold.stderr').write_text(done.stderr)
        record_testsuite_property('cold_'+name+'_directory',str(folder))
        assert done.returncode==(1 if negative else 0),done.stdout+done.stderr+'\n'+str(folder)
        result[name]=(folder,done)
    return result


def test_two_real_original_fresh_math_processes_complete_every_phase(captures):
    folder,_=captures['passing'];pair=read_record(folder/'results/pair.json')
    assert pair['pair_checks_passed'] and pair['run_finished'] and len(pair['completed'])==2
    assert pair['planned']==[runtime.task(i,True) for i in (0,1)]
    assert pair['local_tiny_rehearsal'] and not pair['original_four_worker_session_qualified']
    assert not pair['performance_qualified'] and not pair['research_admitted']
    process_ids=[]
    for index,receipt in enumerate(pair['completed']):
        worker=folder/'results/workers'/runtime.task(index,True)['worker']
        state=read_record(worker/'capture.json');old=read_record(worker/'worker.json')
        assert runtime.worker_status(worker,local=True)==receipt
        assert state['outcome']=='original-worker-passed' and old['complete'] and old['passed']
        assert old['arm']=='math' and old['device']=='cpu' and old['tiny_fixture']
        assert state['CPU_oracle_calls']==state['CPU_oracle_returns']==1
        assert state['attention_calls']==state['SDPA_calls']==len(state['attention_returns'])==6
        assert [x['phase'] for x in state['attention_returns']]==list(runtime.PHASES)
        assert state['events'][:2]==['original_CPU_oracle_call','CPU_return_saved_before_original_gate']
        assert state['events'][-1]=='detailed_context_after_original_worker_exit'
        assert all(state[k] for k in ('profile_restored','trace_restored','profile_intact_until_original_exit','alarm_handler_restored'))
        assert state['post_worker_context']['CUDA_initialized'] is False
        process_ids.append(state['process_id'])
    assert len(set(process_ids))==2


def test_actual_FP32_returns_and_FP64_CPU_return_match_original_saved_bytes(captures):
    folder,_=captures['passing']
    for index in (0,1):
        worker=folder/'results/workers'/runtime.task(index,True)['worker'];state=read_record(worker/'capture.json')
        values=torch.load(worker/'values.pt',map_location='cpu',weights_only=True)
        shape=list(values['q'].shape)
        cpu=capture.load_return(worker,state['CPU_return'],dtype='float64',shape=shape,limit=16*2**20)
        assert all(a.dtype==np.float64 for a in cpu.values())
        for entry in state['attention_returns']:
            arrays=capture.load_return(worker,entry['artifact'],dtype='float32',shape=shape,limit=8*2**20)
            assert all(a.dtype==np.float32 for a in arrays.values())
            for name,array in arrays.items():assert array.tobytes()==values['actual_'+name].numpy().tobytes()
        audit=capture.audit_worker(worker,folder/'code',local=True)
        assert audit['CPU_gate_checks']==4 and audit['component_array_checks']==24
        assert audit['reference_array_checks']==(4 if index else 0)


def test_CPU_refusal_is_captured_and_stops_before_transfer_or_second_worker(captures):
    folder,_=captures['CPU-refusal'];pair=read_record(folder/'results/pair.json')
    assert not pair['pair_checks_passed'] and not pair['completed'] and 'error' in pair
    first=folder/'results/workers/encoder-1-math-0';state=read_record(first/'capture.json');old=read_record(first/'worker.json')
    assert state['CPU_return'] and state['CPU_oracle_returns']==1 and state['attention_calls']==state['SDPA_calls']==0
    assert state['outcome']=='original-worker-rejected' and old['error']=='Fixed oracle bookkeeping failed; preserve input snapshot.'
    assert not old['passed'] and not old['complete'] and not (first/'device-inputs.pt').exists()
    assert not (folder/'results/workers/encoder-1-math-1').exists()


def harness(folder,body):
    script=folder/'test-harness.py'
    script.write_text('''
import sys,runpy,json,time,signal
from pathlib import Path
folder=Path(sys.argv[1]);sys.path.insert(0,str(folder/'overlay'));sys.path.insert(1,str(folder/'code'))
from models import original_attention_pair_runtime_v1 as runtime
runtime.initialize(folder,folder/'code',local=True,control_sha=None,started=time.time(),deadline=time.time()+120)
worker,plan=runtime.issue_task(folder,folder/'code',0,local=True,control_sha=None)
from types import SimpleNamespace
args=SimpleNamespace(packet=folder/'code',scratch=folder,upstream=Path(sys.argv[2]),local_tiny_rehearsal=True,
    control_sha256=None,task_index=0,worker_plan_sha256=runtime.digest(worker/'worker-plan.json'))
'''.replace('started=time.time(),deadline=time.time()+120','started=(begin:=time.time()),deadline=begin+120')+body)
    return subprocess.run([sys.executable,'-I','-B',str(script),str(folder),str(ROOT/'external/vjepa2')],
        cwd=folder/'neutral',env=environment(),capture_output=True,text=True,timeout=40)


@pytest.mark.parametrize('hook',['profile','trace'])
def test_existing_hook_refuses_without_replacing_it(tmp_path,tiny_packet,hook):
    prepare(tmp_path,tiny_packet)
    done=harness(tmp_path,'''
existing=lambda *a:None
getattr(sys,'set'''+hook+'''')(existing)
try:runtime.dispatch(args,folder/'overlay')
except ValueError:pass
else:raise AssertionError('Existing hook must refuse')
assert getattr(sys,'get'''+hook+'''')() is existing
getattr(sys,'set'''+hook+'''')(None)
''')
    assert done.returncode==0,done.stdout+done.stderr
    assert not (tmp_path/'results/workers/encoder-1-math-0/worker.json').exists()


@pytest.mark.parametrize('change',['alias','worker_code'])
def test_changed_original_loaded_function_refuses_before_computation(tmp_path,tiny_packet,change):
    prepare(tmp_path,tiny_packet)
    done=harness(tmp_path,'''
from models import fixed_attention_diagnostic as original
def replacement(*a,**k):raise AssertionError('Replaced function must not run')
'''+('original.iteration=replacement\n' if change=='alias' else 'original.worker=replacement\n')+'''
try:runtime.dispatch(args,folder/'overlay')
except ValueError:pass
else:raise AssertionError('Changed original function must refuse')
''')
    assert done.returncode==0,done.stdout+done.stderr
    assert not (tmp_path/'results/workers/encoder-1-math-0/CPU-return.npz').exists()


def test_capture_error_fails_closed_and_restores_previous_alarm_without_reset(tmp_path,tiny_packet):
    prepare(tmp_path,tiny_packet)
    done=harness(tmp_path,'''
from models import attention_return_capture_v1 as capture
original_writer=capture.save_return
def fail_FP32(path,*a,**kw):
    if kw['dtype']=='float32':raise OSError('LOCAL TEST: capture disk error')
    return original_writer(path,*a,**kw)
capture.save_return=fail_FP32
def previous_alarm(*unused):raise AssertionError('Test timeout')
signal.signal(signal.SIGALRM,previous_alarm);signal.setitimer(signal.ITIMER_REAL,25)
begin=time.monotonic();status=runtime.dispatch(args,folder/'overlay')
remaining=signal.getitimer(signal.ITIMER_REAL)[0];signal.setitimer(signal.ITIMER_REAL,0)
assert status==1 and signal.getsignal(signal.SIGALRM) is previous_alarm
assert 0<remaining<=25-(time.monotonic()-begin)+.05
assert sys.getprofile() is None and sys.gettrace() is None
state=runtime.read_record(worker/'capture.json');old=runtime.read_record(worker/'worker.json')
assert state['outcome']=='technical-failure' and state['observer_error']['message']=='LOCAL TEST: capture disk error'
assert old['error_type']=='OSError' and old['error']=='LOCAL TEST: capture disk error'
assert state['CPU_oracle_returns']==state['attention_calls']==state['SDPA_calls']==1
assert len(state['attention_returns'])==0 and not old['complete'] and not old['passed']
''')
    assert done.returncode==0,done.stdout+done.stderr
    assert not (tmp_path/'results/workers/encoder-1-math-1').exists()


def test_original_reference_refusal_retains_actual_FP32_return_before_its_gate(tmp_path,tiny_packet):
    prepare(tmp_path,tiny_packet)
    done=harness(tmp_path,'''
# Negative original reference descriptor. No scientific function or gate is changed.
from models import fixed_attention_diagnostic as original
from models.original_attention_pair_observer_v1 import observe_worker
auth=runtime.read_record(worker/'authorization.json')
auth['reference']={'worker':'bad-reference','worker_sha256':'0'*64,'values_sha256':'0'*64}
runtime.record(worker/'authorization.json',auth,replace=True)
bad=worker.parent/'bad-reference';bad.mkdir()
runtime.record(bad/'worker.json',{'passed':True,'complete':True,'common':{},'arm':'math'})
original_args=SimpleNamespace(worker_dir=worker,authorization_sha256=runtime.digest(worker/'authorization.json'),
    case='encoder-1',arm='math',replay=0,handoff_sha256=runtime.digest(folder/'code/FIXED-ATTENTION-HANDOFF.json'),
    tiny=True,device='cpu',upstream=args.upstream)
state=observe_worker(original,original_args,folder/'code',worker,local=True)
old=runtime.read_record(worker/'worker.json')
assert state['outcome']=='original-worker-rejected' and not old['complete'] and not old['passed']
assert old['error']=='Reference worker is invalid or incomparable.'
assert state['attention_calls']==state['SDPA_calls']==len(state['attention_returns'])==1
assert state['attention_returns'][0]['phase']=='untimed' and (worker/'return-0-untimed.npz').is_file()
assert (worker/'values.pt').is_file() and not (worker/'return-1-profiled.npz').exists()
''')
    assert done.returncode==0,done.stdout+done.stderr


@pytest.mark.parametrize('defect',['incomplete','return_hash','values_hash'])
def test_fresh_candidate_never_advances_from_rejected_or_changed_baseline(tmp_path,captures,defect):
    original,_=captures['passing'];shutil.copytree(original/'results',tmp_path/'results')
    pair=read_record(tmp_path/'results/pair.json');pair.update(completed=pair['completed'][:1],run_finished=False,pair_checks_passed=False)
    write_record(tmp_path/'results/pair.json',pair,replace=True)
    first=tmp_path/'results/workers/encoder-1-math-0';shutil.rmtree(tmp_path/'results/workers/encoder-1-math-1')
    if defect=='incomplete':
        state=read_record(first/'capture.json');state['capture_coverage_complete']=False;write_record(first/'capture.json',state,replace=True)
    elif defect=='return_hash':
        p=first/'return-5-timed-3.npz';p.write_bytes(p.read_bytes()+b'bad')
    else:
        p=first/'values.pt';p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError):runtime.issue_task(tmp_path,tmp_path/'unused-code',1,local=True,control_sha=None)
    assert not (tmp_path/'results/workers/encoder-1-math-1').exists()


def test_full_unchanged_cold_import_is_complete_before_blocked_original_dispatch(tmp_path):
    c=overlay(tmp_path);op.base.extract_packet(PACKET/'vjepa-fixed-attention.tar.gz',c['archive_manifest'],tmp_path/'code')
    op.base.old.write_json(tmp_path/'operator-control.json',c);sha=op.base.old.digest(tmp_path/'operator-control.json')
    op.base.old.write_json(tmp_path/'resource-authorization.json',authorization(sha))
    script=tmp_path/'cold-import.py';script.write_text('''
import sys,time
from pathlib import Path
folder=Path(sys.argv[1]);sha=sys.argv[2];sys.path.insert(0,str(folder/'overlay'));sys.path.insert(1,str(folder/'code'))
from models import original_attention_pair_runtime_v1 as runtime
from models import original_attention_pair_observer_v1 as observer
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
begin=time.time();runtime.initialize(folder,folder/'code',local=False,control_sha=sha,started=begin,deadline=begin+600)
worker,plan=runtime.issue_task(folder,folder/'code',0,local=False,control_sha=sha)
class Halt(BaseException):pass
calls=[]
def blocked(original,args,packet,out,**kw):
    import torch
    assert not torch.cuda.is_initialized() and args.arm=='efficient' and args.device=='cuda' and args.tiny is False
    assert original.worker.__name__=='worker';calls.append(1);raise Halt()
observer.observe_worker=blocked
from types import SimpleNamespace
args=SimpleNamespace(packet=folder/'code',scratch=folder,control_sha256=sha,local_tiny_rehearsal=False,task_index=0,
    worker_plan_sha256=runtime.digest(worker/'worker-plan.json'),upstream=folder/'no-upstream-needed')
try:runtime.dispatch(args,folder/'overlay')
except Halt:pass
assert calls==[1] and not (worker/'worker.json').exists()
print('Full cold import verified; original dispatch blocked; no oracle/CUDA work.')
''')
    done=subprocess.run([sys.executable,'-I','-B',str(script),str(tmp_path),sha],cwd=tmp_path,env=environment(),capture_output=True,text=True,timeout=30)
    (tmp_path/'cold-import.stdout').write_text(done.stdout);(tmp_path/'cold-import.stderr').write_text(done.stderr)
    assert done.returncode==0,done.stdout+done.stderr
