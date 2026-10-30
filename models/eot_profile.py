"""Separate attended EOT8 work attribution; no optimization or study execution."""
import gc
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

import torch

from attacks.pgd import native_objective
from models.candidate_evaluation import EvaluationSettings,seeds_for
from models.cuda_convolution import agreement
from models.cuda_profile import tensor_signature
from models.diagnostics import state_digest
from models.eot_profile_events import PREFIX,aggregate,annotate,save_events
from models.eot_profile_packet import verify
from models.execution import ExecutionGuard,apply_model_policy,configure_execution
from models.native_diagnostic import TOLERANCES,tiny_fixture,save_tiny_fixture
from models.precision_diagnostic import same_report
from models.fixture_state import load_fixture_state,read_fixture_state
from models.pilot_bundle import CHECKPOINT,digest,make_archive
from models.reuse_diagnostic import hardware_gate,upstream_gate
from models.reuse_handoff import training_input
from models.runtime import synchronize
from models.study_artifacts import read_record,write_record
from models.training import runtime_contract
from models.vjepa2 import build_vitl,load_checkpoint,sample_masks

KIND='eot8-work-attribution-diagnostic-v1'
EOT_PHASES=('before-0','before-1','after')
DRAW_PHASES=('draw-before','profiled-draw','draw-after')
PHASES=('before-0','before-1','draw-before','profiled-draw','draw-after','after')


def validate_launch(plan,handoff):
    if (plan.get('kind')!='eot8-work-profile-launch-v1' or plan.get('authorized') is not True
            or plan.get('attended') is not True or plan.get('paid_compute') is not False
            or plan.get('research_admitted') is not False or plan.get('diagnostic_only') is not True
            or plan.get('handoff_sha256')!=handoff or plan.get('expected_host') not in ('linux10609','linux10603')
            or plan.get('expected_gpu')!='NVIDIA GeForce RTX 3070'
            or not str(plan.get('expected_gpu_uuid','')).startswith('GPU-') or plan.get('expected_driver')!='580.178.04'
            or plan.get('max_seconds')!=1200 or plan.get('worker_seconds')!=480
            or plan.get('allocator_gib')!=7 or plan.get('automatic_retries')!=0):
        raise ValueError('Require the separately bound one-attempt attended profile plan.')


def measured_pass(model,clean,targets,banks,guard):
    x=clean.detach().clone().requires_grad_(True)
    with torch.profiler.record_function(PREFIX+'boundary_guard_before'):
        tick=time.monotonic();guard.check();synchronize(clean.device.type);guard_before=time.monotonic()-tick
    objective=native_objective(model,targets,banks);synchronize(clean.device.type)
    if clean.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    with torch.profiler.record_function(PREFIX+'objective_forward'):
        loss=objective(x).sum();synchronize(clean.device.type)
    forward=time.monotonic()-started;backward_started=time.monotonic()
    with torch.profiler.record_function(PREFIX+'input_backward'):
        gradient,=torch.autograd.grad(loss,x);synchronize(clean.device.type)
    backward=time.monotonic()-backward_started
    timings={'forward_seconds':forward,'input_backward_seconds':backward,'objective_seconds':forward+backward,
             'boundary_guard_before_seconds':guard_before,
             'peak_allocated_bytes':torch.cuda.max_memory_allocated() if clean.device.type=='cuda' else None}
    with torch.profiler.record_function(PREFIX+'boundary_guard_after'):
        tick=time.monotonic();guard.check();synchronize(clean.device.type);timings['boundary_guard_after_seconds']=time.monotonic()-tick
    if not torch.isfinite(loss).all() or not torch.isfinite(gradient).all() or not gradient.abs().max()>0:
        raise ValueError('Nonfinite or zero EOT8 derivative.')
    return {'loss':loss.detach().cpu(),'input':gradient.detach().cpu()},timings


def validate_trace(trace,device,encoder_depth,predictor_depth):
    ranges=trace['annotation_ranges']
    counts={'context_encoder':2,'patch_projection':2,'predictor':2,
            'normalize_and_pixel_validation':1,'mask_validation':1,'draw_loss':1,
            'objective_forward':1,'input_backward':1,'boundary_guard_before':1,'boundary_guard_after':1}
    if any(ranges.get(k,{}).get('count')!=n for k,n in counts.items()):
        raise ValueError('Detailed trace must cover the first original draw and both mask families.')
    # Checkpoint recomputation is included, not assumed to be a second full pass.
    if (ranges.get('encoder_block_forward_or_recompute',{}).get('count',0)<2*encoder_depth
            or ranges.get('predictor_block_forward_or_recompute',{}).get('count',0)<2*predictor_depth):
        raise ValueError('Missing actual block execution annotations.')
    if device=='cuda':
        if not trace['CUDA_events'] or not trace['native_backward_observed'] or trace['cudnn_convolution_kernel_observed'] or trace['legacy_events']:
            raise ValueError('Actual native CUDA profile is incomplete or uses a fallback.')
    elif trace['CUDA_events'] or trace['native_backward_observed']:
        raise ValueError('CPU fixture may not claim actual CUDA evidence.')


def worker(args,root):
    folder=Path(args.worker_dir);auth=read_record(folder/'authorization.json')
    if (digest(folder/'authorization.json')!=args.authorization_sha256 or auth['replay']!=args.replay
            or auth['handoff_sha256']!=args.handoff_sha256):raise ValueError('Worker authorization changed.')
    remaining=auth['deadline_unix']-time.time()
    if not 0<remaining<=(60 if args.tiny else 480):raise ValueError('Worker deadline invalid.')
    signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,remaining)
    metadata,parent=verify(root,args.handoff_sha256);started=time.monotonic()
    result={'kind':KIND,'device':args.device,'tiny_fixture':args.tiny,'replay':args.replay,
            'complete':False,'passed':False,'research_admitted':False,'artifacts':{}}
    def progress(stage):
        result['stage']=stage;write_record(folder/'worker.json',result,replace=(folder/'worker.json').exists())
        print('EOT8 profile:',args.replay,stage,flush=True)
    try:
        progress('bind_sources_and_original_runtime')
        configure_execution('shared-fp64-native',device=args.device);torch.set_num_threads(4)
        upstream_gate(args.upstream,parent)
        if not args.tiny:
            plan=json.loads(Path(args.plan).read_text());validate_launch(plan,args.handoff_sha256)
            if digest(Path(args.plan))!=args.plan_sha256:raise ValueError('Launch plan checksum differs.')
            hardware_gate(plan,allowed_pid=os.getpid())
            if torch.__version__.split('+')[0]!='2.8.0' or torch.version.cuda!='12.8' or not torch.cuda.is_available():
                raise ValueError('Use existing PyTorch2.8/CUDA12.8; no install or fallback.')
            torch.cuda.set_per_process_memory_fraction(min(.9,7*2**30/torch.cuda.get_device_properties(0).total_memory))
        torch.manual_seed(20261029)
        progress('load_model_and_original_training_input')
        if args.tiny:
            model,clean,masks_for=tiny_fixture(args.upstream)
            initial=load_fixture_state(model,args.initial_fixture,args.initial_fixture_sha256)
            data={'kind':'generated-tiny-CPU-fixture','video_id':'eot8-work-profile-fixture','initial_fixture':initial}
        else:
            clean=training_input(root,parent);model=build_vitl(args.upstream,activation_checkpointing=True)
            checkpoint=load_checkpoint(model,args.checkpoint,CHECKPOINT)
            data={'kind':'original-training','video_id':parent['training_row']['video_id'],
                  'pixel_sha256':parent['training_row']['pixel_sha256'],'checkpoint_sha256':checkpoint['sha256']}
            masks_for=lambda seed:sample_masks(args.upstream,batch_size=1,seed=seed)
        apply_model_policy(model,'shared-fp64-native');model.to(args.device).eval();clean=clean.to(args.device)
        guard=ExecutionGuard(model,args.device,target_versions=True);weights=state_digest(model)
        targets=model.clean_targets(clean);target=tensor_signature(targets)
        settings=EvaluationSettings.for_method('primary',2.);seeds=seeds_for(settings,data['video_id'])
        banks=[[m.to(args.device) for m in masks_for(seed%(2**31-1))] for seed in seeds['attack']]
        if len(banks)!=8 or any(len(v)!=2 for v in banks):raise ValueError('Require EOT8 and both mask families.')
        masks=[[[tensor_signature(m.context),tensor_signature(m.target)] for m in v] for v in banks]
        result['contract']={'handoff_sha256':args.handoff_sha256,'kind':KIND,'canonical_evaluation_artifact':False,
            'diagnostic_only':True,'research_admitted':False,'data':data,'runtime':runtime_contract(args.device,model=model),
            'weights_sha256':weights,'targets':target,'pixels':tensor_signature(clean),'seeds':seeds,'attack_masks':masks,
            'encoder_depth':len(model.encoder.blocks),'predictor_depth':len(model.predictor.predictor_blocks),
            'scope':'Full clean EOT8 timing/gradient controls; detailed profile of attack draw0 with both families. No PGD/scoring/optimizer or study cost unit.',
            'detailed_profile_draw':0,'detailed_profile_families':2,'EOT_control_draws':8,
            'observer':'Temporary record_function method-entry/exit wrappers; unchanged reference tensor operations.',
            'profiler_options':{'record_shapes':False,'with_stack':False,'profile_memory':False},
            'boundary_guards_outside_objective_timings':True,'synchronize_between_forward_and_backward':True}
        if not args.tiny:
            prior=json.loads((Path(root)/'input/prior-contract.json').read_text())
            if (weights!=prior['weights_sha256'] or target!=prior['targets'] or result['contract']['pixels']!=prior['pixels']
                    or seeds['attack']!=prior['seeds']['attack'] or masks!=prior['masks']['attack']
                    or result['contract']['runtime']!=prior['runtime']):
                raise ValueError('Reference model/data/masks/runtime differs from verified university result.')
        progress('unprofiled_warmup')
        measured_pass(model,clean,targets,banks,guard)
        values={};times={}
        for phase in PHASES:
            progress(phase)
            selected=banks[:1] if phase in DRAW_PHASES else banks
            if phase=='profiled-draw':
                activities=[torch.profiler.ProfilerActivity.CPU]
                if args.device=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(activities=activities,record_shapes=False,with_stack=False,profile_memory=False) as profile:
                    with annotate(model):current,timing=measured_pass(model,clean,targets,selected,guard)
                trace=save_events(profile,folder/'profile-events.jsonl');del profile;gc.collect()
                validate_trace(trace,args.device,len(model.encoder.blocks),len(model.predictor.predictor_blocks))
                result['trace_summary']=trace
                result['profile_events_sha256']=digest(folder/'profile-events.jsonl')
            else:current,timing=measured_pass(model,clean,targets,selected,guard)
            values.update({phase+'_'+name:v for name,v in current.items()});times[phase]=timing
            if any(p.grad is not None for p in model.parameters()):raise ValueError('Profiling accumulated parameter gradients.')
        for group in (EOT_PHASES,DRAW_PHASES):
            for phase in group[1:]:
                if any(not torch.equal(values[group[0]+'_'+n],values[phase+'_'+n]) for n in ('loss','input')):
                    raise ValueError('Profiler observer or repeated pass changed values.')
        if not args.tiny:
            prior=torch.load(Path(root)/'input/prior-EOT8-derivatives.pt',map_location='cpu',weights_only=True)
            result['prior_reference_checks']={n:agreement(values['before-0_'+n],prior[n],TOLERANCES['clean_loss' if n=='loss' else 'input_gradient']) for n in ('input','loss')}
            if not all(v['passed'] for v in result['prior_reference_checks'].values()):raise ValueError('Original derivative gate rejected; unchanged bounds.')
        with (folder/'passes.pt').open('xb') as f:torch.save(values,f)
        result['artifacts']['passes']={'file':'passes.pt','sha256':digest(folder/'passes.pt'),
                                      'tensors':{n:tensor_signature(v) for n,v in values.items()}}
        result['timings']=times
        guard.check()
        if state_digest(model)!=weights or tensor_signature(targets)!=target:raise ValueError('Model or clean target changed.')
        verify(root,args.handoff_sha256)
        result.update(complete=True,passed=True,stage='complete',model_and_targets_unchanged=True)
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result['wall_seconds']=time.monotonic()-started
        write_record(folder/'worker.json',result,replace=(folder/'worker.json').exists())


def summarize(root,output,handoff,tiny):
    metadata,parent=verify(root,handoff);output=Path(output);session=read_record(output/'session.json')
    if (session['handoff_sha256']!=handoff or session['tiny_fixture']!=tiny or [v['replay'] for v in session['tasks']]!=[0,1]
            or session['kind']!=KIND or session['paid_compute'] is not False or session['research_admitted'] is not False
            or not 0<session['deadline_unix']-session['started_unix']<=(300 if tiny else 1200)):
        raise ValueError('Require two bound fresh workers in the declared domain.')
    initial=None
    if tiny:_,initial=read_fixture_state(output/'initial-fixture.pt',session['initial_fixture']['sha256'])
    workers=[];saved=[];traces=[]
    for task in session['tasks']:
        folder=output/'workers'/str(task['replay']);w=read_record(folder/'worker.json')
        if (digest(folder/'worker.json')!=task['worker_sha256'] or not w['complete'] or not w['passed']
                or w['kind']!=KIND or w['tiny_fixture']!=tiny or w['device']!=('cpu' if tiny else 'cuda')
                or w['research_admitted'] is not False or w['replay']!=task['replay']):raise ValueError('Incomplete or changed profile worker.')
        c=w['contract']
        auth=read_record(folder/'authorization.json')
        if (auth['replay']!=task['replay'] or auth['handoff_sha256']!=handoff or auth['deadline_unix']>session['deadline_unix']
                or not 0<w['wall_seconds']<=(60 if tiny else 480)):
            raise ValueError('Changed worker authorization or time bound.')
        if (c['handoff_sha256']!=handoff or c['kind']!=KIND or c['canonical_evaluation_artifact'] is not False
                or c['research_admitted'] is not False or c['diagnostic_only'] is not True
                or c['boundary_guards_outside_objective_timings'] is not True
                or c['synchronize_between_forward_and_backward'] is not True):
            raise ValueError('Profile artifact domain changed.')
        if (c.get('detailed_profile_draw'),c.get('detailed_profile_families'),c.get('EOT_control_draws'))!=(0,2,8):
            raise ValueError('Changed detailed-profile or full EOT control scope.')
        runtime=c['runtime']
        if (runtime['device']!=w['device'] or runtime['threads']!=4 or not runtime['deterministic_algorithms']
                or runtime['deterministic_warn_only'] or runtime['matmul_precision']!='highest'
                or runtime['matmul_tf32'] or not runtime['cudnn_enabled'] or runtime['cudnn_tf32']
                or runtime['cudnn_benchmark'] or not runtime['cudnn_deterministic']
                or runtime['cublas_workspace_config']!=':4096:8' or runtime['model_execution']['policy']!='shared-fp64-native'):
            raise ValueError('Profile precision/runtime changed.')
        if tiny and (c['data'].get('initial_fixture')!=initial or c['data'].get('kind')!='generated-tiny-CPU-fixture'):
            raise ValueError('Generated CPU fixture identity changed.')
        if not tiny and (runtime['torch'].split('+')[0]!='2.8.0' or runtime['cuda']!='12.8'
                or (c['encoder_depth'],c['predictor_depth'])!=(24,12)
                or c['data'].get('checkpoint_sha256')!=CHECKPOINT
                or c['data'].get('video_id')!=parent['training_row']['video_id']
                or c['pixels']['sha256']!=parent['training_row']['pixel_sha256']):
            raise ValueError('Original full model/GPU identity changed.')
        if c['profiler_options']!={'record_shapes':False,'with_stack':False,'profile_memory':False}:
            raise ValueError('Profiler options changed.')
        if c['seeds']!=seeds_for(EvaluationSettings.for_method('primary',2.),c['data']['video_id']):raise ValueError('Changed mask seeds.')
        if digest(folder/'profile-events.jsonl')!=w['profile_events_sha256']:raise ValueError('Changed profiler events.')
        trace=aggregate(folder/'profile-events.jsonl')
        if trace!=w['trace_summary']:raise ValueError('Trace summary does not match its retained events.')
        validate_trace(trace,w['device'],c['encoder_depth'],c['predictor_depth']);traces.append(trace)
        item=w['artifacts']['passes']
        if item['file']!='passes.pt' or digest(folder/'passes.pt')!=item['sha256']:raise ValueError('Changed pass tensors.')
        values=torch.load(folder/'passes.pt',map_location='cpu',weights_only=True)
        if set(values)!={p+'_'+n for p in PHASES for n in ('loss','input')} or {n:tensor_signature(t) for n,t in values.items()}!=item['tensors']:
            raise ValueError('Changed tensor signatures.')
        if c['pixels']['shape']!=list(values['before-0_input'].shape) or not values['before-0_input'].abs().max()>0:
            raise ValueError('Invalid full input derivative.')
        for phase in PHASES:
            for name in ('loss','input'):
                reference='draw-before' if phase in DRAW_PHASES else 'before-0'
                if not torch.equal(values[reference+'_'+name],values[phase+'_'+name]):raise ValueError('Observer changed the derivative or loss.')
            t=w['timings'][phase]
            if any(not math.isfinite(t[k]) or t[k]<=0 for k in ('forward_seconds','input_backward_seconds','objective_seconds')):
                raise ValueError('Invalid timing observation.')
            if t['objective_seconds']!=t['forward_seconds']+t['input_backward_seconds']:raise ValueError('Phase timing sum changed.')
            if any(not math.isfinite(t[k]) or t[k]<0 for k in ('boundary_guard_before_seconds','boundary_guard_after_seconds')):
                raise ValueError('Invalid diagnostic guard timing.')
            peak=t['peak_allocated_bytes']
            if (tiny and peak is not None) or (not tiny and (type(peak) is not int or not 0<peak<=7*2**30)):
                raise ValueError('Profile memory observation differs from its domain or cap.')
        if not tiny:
            prior=json.loads((Path(root)/'input/prior-contract.json').read_text())
            if any(c[k]!=prior[k] for k in ('runtime','weights_sha256','targets','pixels')) or c['attack_masks']!=prior['masks']['attack']:
                raise ValueError('Changed original CUDA reference identity.')
            expected=torch.load(Path(root)/'input/prior-EOT8-derivatives.pt',map_location='cpu',weights_only=True)
            for name in ('loss','input'):
                actual=agreement(values['before-0_'+name],expected[name],TOLERANCES['clean_loss' if name=='loss' else 'input_gradient'])
                if not actual['passed'] or not same_report(actual,w['prior_reference_checks'][name]):raise ValueError('Prior numerical gate changed.')
        if workers and c!=workers[0]['contract']:raise ValueError('Fresh profile workers have different identities.')
        workers.append(w);saved.append(values)
    if any(not torch.equal(saved[0][k],saved[1][k]) for k in saved[0]):raise ValueError('Fresh-process profile replay differs.')
    unprofiled=[[w['timings'][p]['objective_seconds'] for p in ('before-0','before-1','after')] for w in workers]
    return {'kind':KIND,'passed':True,'tiny_fixture':tiny,'domain':'CPU-fixture' if tiny else 'university-CUDA',
            'fresh_process_loss_and_input_gradient_replay_exact':True,'observer_loss_and_input_gradient_exact':True,
            'unprofiled_objective_seconds':unprofiled,
            'profiled_draw_to_unprofiled_draw_time_ratios':[w['timings']['profiled-draw']['objective_seconds']/statistics.median(w['timings'][p]['objective_seconds'] for p in ('draw-before','draw-after')) for w in workers],
            'trace_events':[v['events'] for v in traces],
            'scope':'Full clean EOT8 timing controls; detailed profile covers draw0 with both original families. Not PGD or an evaluation cost unit.',
            'optimization_adopted':False,'full_study_cost_approved':False,'research_admitted':False}


def export(root,output):
    root=Path(root);output=Path(output);m=json.loads((root/'PROFILE-HANDOFF.json').read_text())
    files={'PROFILE-HANDOFF.json':root/'PROFILE-HANDOFF.json'}
    for name in m['files']:
        if name.endswith(('.py','.sh','.json')):files['source/'+name]=root/name
    for p in output.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt','.pt') and p.name!='evidence.tar.gz.manifest.json':
            if p.is_symlink():raise ValueError('Export refuses symbolic links.')
            files['results/'+str(p.relative_to(output))]=p
    if sum(p.stat().st_size for p in files.values())>2**30:raise ValueError('Profile evidence exceeds 1 GiB.')
    spec=make_archive(output/'evidence.tar.gz',files)
    (output/'evidence.tar.gz.manifest.json').write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n')
    return spec


def run(args,root):
    started=args.launcher_started;elapsed=time.time()-started
    if not math.isfinite(started) or not 0<=elapsed<=120:raise ValueError('Invalid original launcher clock.')
    metadata,parent=verify(root,args.handoff_sha256);upstream_gate(args.upstream,parent)
    if args.tiny:
        if not 0<args.max_seconds<=300 or not 0<args.worker_seconds<=60:raise ValueError('CPU rehearsal bounds exceeded.')
    else:
        plan=json.loads(Path(args.plan).read_text());validate_launch(plan,args.handoff_sha256)
        if digest(Path(args.plan))!=args.plan_sha256 or args.max_seconds!=1200 or args.worker_seconds!=480:raise ValueError('Changed launch plan or bounds.')
        hardware_gate(plan)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    if not args.tiny:
        claims=Path(root)/'profile-claims';claims.mkdir(exist_ok=True)
        with (claims/(args.plan_sha256+'.json')).open('x') as f:json.dump({'started_unix':started,'output':str(output)},f)
    initial=save_tiny_fixture(args.upstream,output/'initial-fixture.pt',seed=20261029) if args.tiny else None
    deadline=started+args.max_seconds;end=time.monotonic()+deadline-time.time()
    session={'kind':KIND,'handoff_sha256':args.handoff_sha256,'tiny_fixture':args.tiny,'initial_fixture':initial,
             'started_unix':started,'deadline_unix':deadline,'tasks':[],'passed':False,'complete':False,
             'hardware':None if args.tiny else hardware_gate(plan),'research_admitted':False,'paid_compute':False}
    write_record(output/'session.json',session);child=None
    def alarm(*unused):raise TimeoutError('Independent profile deadline reached.')
    def interrupted(*unused):raise RuntimeError('Attended session interrupted; no retry.')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.1,deadline-time.time()-5))
    signal.signal(signal.SIGHUP,interrupted);signal.signal(signal.SIGTERM,interrupted)
    def monitor(row):
        with (output/'gpu-monitor.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
    try:
        for replay in (0,1):
            remaining=min(deadline-time.time(),end-time.monotonic())-(15 if args.tiny else 120)
            if remaining<=0:raise TimeoutError('No time remains for a worker and evidence export.')
            seconds=min(args.worker_seconds,remaining);folder=output/'workers'/str(replay);folder.mkdir(parents=True)
            write_record(folder/'authorization.json',{'replay':replay,'handoff_sha256':args.handoff_sha256,'deadline_unix':time.time()+seconds})
            command=[sys.executable,'-B','-u',str(Path(root)/'scripts/check_eot_profile.py'),'--diagnostic','--device',args.device,
                     '--worker','--replay',str(replay),'--worker-dir',str(folder),'--authorization-sha256',digest(folder/'authorization.json'),
                     '--handoff-sha256',args.handoff_sha256,'--upstream',str(args.upstream)]
            if args.tiny:command+=['--tiny','--initial-fixture',str(output/'initial-fixture.pt'),'--initial-fixture-sha256',initial['sha256']]
            else:command+=['--plan',str(args.plan),'--plan-sha256',args.plan_sha256,'--checkpoint',str(args.checkpoint)]
            print('Fresh EOT8 profiling worker:',replay,flush=True)
            if not args.tiny:hardware_gate(plan,retain=monitor)
            with (folder/'worker.log').open('x') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'})
                tick=time.monotonic();last_gate=0.
                try:
                    while child.poll() is None:
                        if time.monotonic()-tick>=seconds:raise TimeoutError('Worker time limit; no retry.')
                        if not args.tiny and time.monotonic()-last_gate>=5:
                            hardware_gate(plan,allowed_pid=child.pid,retain=monitor);last_gate=time.monotonic()
                        time.sleep(.2)
                    if child.returncode:raise RuntimeError('Profile worker failed; preserve evidence, no retry.')
                finally:
                    if child.poll() is None:
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                    child.wait();child=None
            session['tasks'].append({'replay':replay,'worker_sha256':digest(folder/'worker.json')});write_record(output/'session.json',session,replace=True)
        summary=summarize(root,output,args.handoff_sha256,args.tiny);write_record(output/'summary.json',summary)
        session.update(complete=True,passed=True,summary_sha256=digest(output/'summary.json'))
    except BaseException as exc:
        session.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        if child is not None:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            child.wait()
        session['wall_seconds']=time.time()-started;write_record(output/'session.json',session,replace=True)
        try:
            verify(root,args.handoff_sha256);spec=export(root,output)
            print('Profile evidence:',json.dumps({'bytes':spec['bytes'],'sha256':spec['sha256']}),flush=True)
        except BaseException as exc:
            session.update(complete=False,passed=False,export_error=str(exc));write_record(output/'session.json',session,replace=True);raise
        finally:signal.setitimer(signal.ITIMER_REAL,0)
    return session
