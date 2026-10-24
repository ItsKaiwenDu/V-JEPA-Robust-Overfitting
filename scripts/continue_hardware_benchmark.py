#!/usr/bin/env python3
"""Operational continuation only; preserves frozen research sources and prior evidence."""
import argparse, hashlib, json, math, os, pathlib, shutil, subprocess, sys, tempfile, time
import xml.etree.ElementTree as ET

MANIFEST='e42164317e16fbf6e05f4143fd466f00105371284fc9e769d09d5c71ba82608a'
CHECKPOINT='5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012'
JUNIT='71a9a688cd743c5f11156a21d2a209d8e7cfc5a354dedfaa6c03f2584f769cec'
MIN_PRIOR_RUNNING_SECONDS=4260  #12:58 through conservatively rounded-up14:09 Pacific.

def sha(path):
    with pathlib.Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def save(path,value):
    pathlib.Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def plan_gate(plan,now):
    for key in ('restart_epoch','deadline_epoch','prior_running_seconds'):
        if type(plan.get(key)) is not int:raise ValueError('Require integral reviewed times: '+key)
    prior=plan['prior_running_seconds'];restart=plan['restart_epoch'];deadline=plan['deadline_epoch']
    if prior<MIN_PRIOR_RUNNING_SECONDS or prior>=7200:raise ValueError('Prior billed time cannot be reset.')
    if not restart<=now<deadline or deadline>restart+7200-prior:raise ValueError('Cumulative two-hour deadline rejected.')
    rate=plan.get('quoted_all_in_hourly_usd')
    if isinstance(rate,bool) or not isinstance(rate,(int,float)) or not math.isfinite(rate) or not 0<rate<=.70:
        raise ValueError('Fresh all-in quote must be positive and at most0.70/hour.')
    if plan.get('fresh_quote_confirmed') is not True or plan.get('new_hold_usd')!=0:
        raise ValueError('Require a fresh quote and the existing reservation only.')
    if plan.get('existing_hold_usd')!=2.8:raise ValueError('Existing2.80 hold must be retained.')
    if deadline-now<1440:raise ValueError('Insufficient remaining runtime and export reserve.')
    return int(deadline-now)
def source_gate(base):
    manifest=base/'handoff-manifest.json'
    if sha(manifest)!=MANIFEST:raise ValueError('Frozen manifest changed.')
    spec=json.loads(manifest.read_text())
    for name,row in spec['archives']['source']['files'].items():
        path=base/name
        if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()) or sha(path)!=row['sha256']:
            raise ValueError('Pinned source changed: '+name)
    return len(spec['archives']['source']['files'])
def input_gate(base,setup,checkpoint,receipt):
    sources=source_gate(base)
    if sha(setup/'tests.xml')!=JUNIT:raise ValueError('Full509-case JUnit changed.')
    suite=next(ET.parse(setup/'tests.xml').getroot().iter('testsuite')).attrib
    if suite['tests']!='509' or any(suite[k]!='0' for k in ('failures','errors','skipped')):
        raise ValueError('Full CUDA-host suite was not accepted.')
    value=json.loads(receipt.read_text())
    if (value['manifest_sha256']!=MANIFEST or value['junit_sha256']!=JUNIT
        or value['checkpoint_sha256']!=CHECKPOINT or value['original_supervisor_exit_status']!=124):
        raise ValueError('Prepared-input evidence changed.')
    if checkpoint.is_symlink() or checkpoint.stat().st_size!=5127726842 or sha(checkpoint)!=CHECKPOINT:
        raise ValueError('Released checkpoint differs.')
    return {'source_files_verified':sources,'full_suite':suite,'manifest_sha256':MANIFEST,
            'junit_sha256':JUNIT,'checkpoint_sha256':CHECKPOINT,'checkpoint_bytes':checkpoint.stat().st_size,
            'original_preparation_exit_status':124,'research_admitted':False}
def actual_gate(base,run):
    python=str(base/'.venv/bin/python')
    code="""import json,pathlib,platform,sys,torch,torchvision
assert platform.system()=='Linux' and platform.python_version()=='3.12.3'
limits=[]
for name in ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes'):
 pth=pathlib.Path(name)
 if pth.exists() and pth.read_text().strip().isdigit():limits.append(int(pth.read_text().strip()))
assert limits and min(limits)>=45_000_000_000, 'Review lower host RAM before execution.'
assert torch.__version__=='2.8.0+cu128' and torchvision.__version__=='0.23.0+cu128'
assert torch.cuda.is_available() and torch.cuda.device_count()==1
p=torch.cuda.get_device_properties(0)
assert p.name=='NVIDIA A40' and p.total_memory>=44*2**30
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'gpu':p.name,'vram_bytes':p.total_memory}))
"""
    result=subprocess.run([python,'-c',code],cwd=base,text=True,capture_output=True,timeout=30,check=True)
    runtime=json.loads(result.stdout)
    tests=['tests/test_cuda_reproducibility.py::test_cuda_gradients_and_updates_match_across_fresh_processes',
           'tests/test_native_execution.py::test_cuda_backward_outside_forward_context_uses_native_operator']
    from scripts.run_workflow_pilot import run_child
    run_child([python,'-m','pytest','-q','-p','no:cacheprovider',*tests,
          '--junitxml='+str(run/'restart-cuda-tests.xml'),'--basetemp='+str(run/'restart-fixtures')],
          run/'restart-cuda-tests.log',120)
    suite=next(ET.parse(run/'restart-cuda-tests.xml').getroot().iter('testsuite')).attrib
    if suite['tests']!='2' or any(suite[k]!='0' for k in ('errors','failures','skipped')):
        raise ValueError('Fresh actual-CUDA restart controls rejected.')
    return {'runtime':runtime,'actual_cuda_restart_tests':suite}
def worker(args,plan):
    base=args.base.resolve();run=args.run.resolve();status=1;error=None
    sys.path.insert(0,str(base));os.chdir(base)
    from scripts.run_workflow_pilot import run_child
    def remaining():return int(plan['deadline_epoch']-time.time())
    try:
        print('Stage: verify prepared inputs and fresh CUDA runtime',flush=True)
        run_child([sys.executable,str(pathlib.Path(__file__).resolve()),'--gate','--base',str(base),
             '--plan',str(args.plan.resolve()),'--run',str(run)],run/'input-gate.log',min(180,remaining()-300))
        plan_gate(plan,int(time.time()))
        pilot_seconds=remaining()-240
        print('Stage: original seven-phase pretrained hardware screen',flush=True)
        command=[str(base/'.venv/bin/python'),'-u','scripts/run_hardware_benchmark.py','--diagnostic',
          '--expected-gpu','NVIDIA A40','--hourly-rate-usd',str(plan['quoted_all_in_hourly_usd']),
          '--deadline-epoch',str(plan['deadline_epoch']),'--checkpoint','checkpoints/vitl.pt',
          '--upstream','external/vjepa2','--manifests','data/manifests/ucf101-staged-20x5-v1',
          '--clips','data/processed/staged-exploration-v1','--output',str(run/'pilot'),'--max-seconds',str(pilot_seconds)]
        run_child(command,run/'pilot-worker.log',pilot_seconds+15);status=0
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc)}
        print('Continuation failed:',error,flush=True)
    finally:
        (run/'exit_status.txt').write_text(str(status)+'\n')
        pilot=run/'pilot';pilot.mkdir(exist_ok=True)
        if not (pilot/'session.json').exists():
            value={'schema_version':1,'diagnostic_only':True,'research_result':False,'budget_approved':False,
              'complete':False,'execution_policy':'shared-fp64-native','phases':[],
              'pre_session_stage':'restart_input_gate','worker_exit_status':status,'error':error}
            checksum=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            save(pilot/'session.json',{'payload':value,'sha256':checksum})
        for name in ('launcher-metadata.json','resume-plan.json','prepared-inputs-verification.json',
                     'restart-cuda-tests.xml','restart-cuda-tests.log','exit_status.txt','input-gate.log','pilot-worker.log'):
            if (run/name).is_file():shutil.copyfile(run/name,pilot/name)
        save(pilot/'continuation-source.json',{'source':pathlib.Path(__file__).read_text(),
             'sha256':sha(__file__),'frozen_research_sources_changed':False})
        try:
            run_child([sys.executable,'scripts/export_workflow_pilot.py','--run',str(pilot),
              '--output',str(run/'evidence.tar.gz'),'--include-preflight-tensors'],run/'export.log',min(180,max(1,remaining()-20)))
        except BaseException as exc:print('Preserve volume; export incomplete:',str(exc),flush=True)
        print('Continuation exit status:',status,'Provider billing requires user Stop/Terminate.',flush=True)
    return status

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=pathlib.Path,required=True)
    parser.add_argument('--plan',type=pathlib.Path)
    parser.add_argument('--run',type=pathlib.Path)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--gate',action='store_true')
    parser.add_argument('--offline-check',action='store_true')
    parser.add_argument('--setup',type=pathlib.Path);parser.add_argument('--checkpoint',type=pathlib.Path)
    parser.add_argument('--receipt',type=pathlib.Path)
    args=parser.parse_args();base=args.base.resolve();sys.path.insert(0,str(base))
    if args.offline_check:
        value=input_gate(base,args.setup,args.checkpoint,args.receipt)
        value.update(offline_only=True,cuda_restart_controls_run=False,paid_job_started=False)
        print(json.dumps(value,indent=2));return 0
    if args.plan is None:parser.error('Supply a fresh reviewed resume plan.')
    plan=json.loads(args.plan.read_text());seconds=plan_gate(plan,int(time.time()))
    if args.gate:
        metadata=json.loads((args.run/'launcher-metadata.json').read_text())
        if sha(__file__)!=metadata['continuation_source_sha256']:raise ValueError('Operational launcher changed.')
        result=input_gate(base,base/'runs/pretrained/diagnostic/workflow-setup',base/'checkpoints/vitl.pt',
                base.parent/'hardware-benchmark-handoff-20261023b/prepared-inputs-receipt.json')
        result.update(actual_gate(base,args.run));result.update(resume_plan=plan,gate_passed=True)
        save(args.run/'prepared-inputs-verification.json',result);return 0
    if args.worker:return worker(args,plan)
    source_gate(base)
    if not shutil.which('timeout'):raise RuntimeError('Finite outer timeout is required.')
    parent=base/'runs/pretrained/diagnostic';parent.mkdir(parents=True,exist_ok=True)
    claim=parent/('resume-claimed-'+str(plan['restart_epoch'])+'.json')
    with claim.open('x') as handle:json.dump(plan,handle)
    run=pathlib.Path(tempfile.mkdtemp(prefix='hardware-resume.',dir=parent))
    save(run/'resume-plan.json',plan)
    save(run/'launcher-metadata.json',{**plan,'schema_version':1,'diagnostic_only':True,
      'expected_gpu':'NVIDIA A40','execution_policy':'shared-fp64-native','maximum_inclusive_session_usd':2.8,
      'additional_reservation_usd':0,'profile':'hardware-cost-screen-v1','exact_optimizer_recovery_checked':False,
      'export_reserve_seconds':240,'preflight_cap_seconds':1200,'cloud_billing_stopped_by_launcher':False,
      'preparation_exit_status_preserved':124,'continuation_source_sha256':sha(__file__)})
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    command=['timeout','--signal=TERM','--kill-after=15s',str(seconds),sys.executable,
      str(pathlib.Path(__file__).resolve()),'--worker','--base',str(base),'--plan',str((run/'resume-plan.json').resolve()),'--run',str(run)]
    with (run/'launcher.log').open('x') as log:
        process=subprocess.Popen(command,cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(run/'launcher-process.json',{'pid':process.pid,'command':command})
    print(json.dumps({'run_directory':str(run),'supervisor_pid':process.pid,'deadline_epoch':plan['deadline_epoch'],
      'source_sha256':sha(__file__),'billing_stopped_automatically':False},indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
