"""Bounded, one-clip diagnostic of original versus reused projection execution."""
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
from types import SimpleNamespace

import torch
from torch import nn

from attacks.pgd import pgd_linf
from models.candidate_evaluation import EvaluationSettings, seeds_for
from models.cuda_convolution import agreement, TOLERANCES as PATCH_TOLERANCES
from models.cuda_profile import tensor_signature
from models.diagnostics import state_digest
from models.execution import ExecutionGuard, apply_model_policy, configure_execution, model_execution_contract, precision_contract
from models.fixture_state import load_fixture_state, read_fixture_state
from models.native_diagnostic import TOLERANCES, tiny_fixture, save_tiny_fixture
from models.native_probe import kernel_boundary_checks, profile_records, validate_probe
from models.pilot_bundle import CHECKPOINT, digest, make_archive
from models.precision_diagnostic import projection_check, same_report
from models.precision_patch import PrecisionPatchEmbed3D
from models.reuse_execution import KIND, MODES, DiagnosticProjectionBank, diagnostic_bank, diagnostic_objective
from models.reuse_handoff import verify, training_input
from models.runtime import synchronize
from models.study_artifacts import read_record, write_record
from models.training import runtime_contract
from models.vjepa2 import build_vitl, load_checkpoint, sample_masks, use_upstream

TASKS = (('recompute',0),('reuse',0),('recompute',1),('reuse',1))
MAX_EXPORT = 1024**3


def validate_launch(plan, handoff_sha):
    if (plan.get('kind')!='projection-reuse-university-launch-v1' or plan.get('authorized') is not True
            or plan.get('diagnostic_only') is not True or plan.get('paid_compute') is not False
            or plan.get('research_admitted') is not False or plan.get('handoff_sha256')!=handoff_sha
            or plan.get('expected_host') not in ('linux10609','linux10603')
            or plan.get('expected_gpu')!='NVIDIA GeForce RTX 3070'
            or not str(plan.get('expected_gpu_uuid','')).startswith('GPU-')
            or plan.get('expected_driver')!='580.178.04' or plan.get('allocator_gib')!=7
            or plan.get('max_seconds')!=4800 or plan.get('worker_seconds')!=1080
            or plan.get('automatic_retries')!=0 or plan.get('attended') is not True):
        raise ValueError('Require a separately bound, attended university diagnostic launch.')


def hardware_gate(plan, allowed_pid=None, retain=None):
    if platform.node()!=plan['expected_host']:
        raise ValueError('University host identity differs.')
    row=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,driver_version',
                                 '--format=csv,noheader,nounits'],text=True,timeout=10).strip().splitlines()
    if len(row)!=1:raise ValueError('Exactly one expected university GPU is required.')
    name,uuid,total,used,driver=[v.strip() for v in row[0].split(',')]
    if ((name,uuid,driver)!=(plan['expected_gpu'],plan['expected_gpu_uuid'],plan['expected_driver'])
            or float(total)!=8192 or allowed_pid is None and float(used)>512):
        raise ValueError('University GPU/driver/memory gate rejected.')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                                    '--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
    if retain is not None:retain({'observed_unix':time.time(),'host':platform.node(),'GPU_row':row[0],
                                 'allowed_compute_pid':allowed_pid,'compute_process_rows':active})
    if any(int(v.split(',')[0].strip())!=allowed_pid for v in active if v.strip()):
        raise ValueError('Another CUDA compute process is active: '+repr(active)+'; leave it alone and stop this diagnostic.')
    return {'host':platform.node(),'gpu':name,'uuid':uuid,'memory_mib':int(float(total)),'driver':driver}


def upstream_gate(path, metadata):
    path=use_upstream(path)
    for name,expected in metadata['upstream_sha256'].items():
        if digest(path/name)!=expected:raise ValueError('Pinned upstream source changed: '+name)


def native_reuse_probe(device):
    before=precision_contract()
    projection=nn.Conv3d(3,16,(2,4,4),stride=(2,4,4)).to(device).eval()
    patch=PrecisionPatchEmbed3D(SimpleNamespace(proj=projection,patch_size=4,tubelet_size=2,training=False),'native').eval()
    x=torch.ones(1,3,4,16,16,device=device,requires_grad=True)
    with DiagnosticProjectionBank(patch,expected_calls=2,device=device,diagnostic_kind=KIND) as bank:
        a,b=bank(x),bank(x)
    after=precision_contract()
    activities=[torch.profiler.ProfilerActivity.CPU]
    if device=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profile:
        (a.square().mean()+b.sin().mean()).backward();synchronize(device)
    rows=profile_records(profile)
    evidence={'schema_version':2,'device':device,'delegator':'models.projection_reuse._IndependentVJP',
        'diagnostic_kind':KIND,'projection_requests':2,'forward_computations':bank.forward_computations,
        'backward_outside_forward_context':True,'flags_before':before,'flags_after_forward':after,
        'flags_after_backward':precision_contract(),'profiler_events':rows,'kernel_boundary_checks':kernel_boundary_checks(rows),
        'parameter_and_input_gradients_finite':all(v.grad is not None and torch.isfinite(v.grad).all().item() for v in (x,*projection.parameters())),
        'actual_cuda_kernel_evidence':device=='cuda' and kernel_boundary_checks(rows)['passed']}
    validate_probe(evidence,device=device)
    if evidence['forward_computations']!=1:raise ValueError('The new delegator did not reuse its forward.')
    return evidence


def full_gradient(model,clean,targets,banks,mode):
    receipts=[];objective=diagnostic_objective(model,targets,banks,mode=mode,receipts=receipts)
    x=clean.detach().clone().requires_grad_(True);p=model.encoder.patch_embed.proj
    synchronize(clean.device.type);tick=time.monotonic()
    loss=objective(x).sum()
    gradients=torch.autograd.grad(loss,[x,p.weight,p.bias])
    synchronize(clean.device.type);seconds=time.monotonic()-tick
    values={'loss':loss.detach().cpu(),**{k:v.detach().cpu() for k,v in zip(('input','weight','bias'),gradients)}}
    if not values['input'].abs().max()>0:raise ValueError('Zero full-model input gradient.')
    return values,receipts,seconds


def projection_timings(model,clean,device):
    tensors={};rows={}
    with torch.no_grad():
        normalized=model.normalize(clean)
        expected=model.encoder.patch_embed(normalized).detach()
        tensors['tokens_reference']=expected.cpu()
        for mode in MODES:
            synchronize(device);tick=time.monotonic()
            with diagnostic_bank(model,mode,16,device) as bank:
                checks=[]
                for _ in range(16):
                    value=model.encoder.patch_embed(normalized)
                    checks.append(bool(torch.equal(value,expected)))
                tensors['tokens_'+mode]=value.cpu()
            synchronize(device)
            # Equality checks and output transfer are deliberately outside the
            # timed component loop below; their first pass is correctness only.
            if not all(checks):raise ValueError('Projection forward values changed across requests.')
            tick=time.monotonic()
            with diagnostic_bank(model,mode,16,device) as bank:
                for _ in range(16):value=model.encoder.patch_embed(normalized)
            synchronize(device)
            rows[mode]={'seconds':time.monotonic()-tick,'requests':bank.requests,'forward_computations':bank.forward_computations}
    return tensors,rows


def worker(args,root):
    started=time.monotonic();directory=Path(args.worker_dir)
    auth=read_record(directory/'authorization.json')
    if (auth['mode']!=args.mode or auth['replay']!=args.replay or auth['handoff_sha256']!=args.handoff_sha256
            or digest(directory/'authorization.json')!=args.authorization_sha256):
        raise ValueError('Worker authorization differs from its parent.')
    remaining=auth['deadline_unix']-time.time()
    if not 0<remaining<=(60 if args.tiny else 1080):raise ValueError('Worker deadline is expired or invalid.')
    signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,remaining)
    metadata=verify(root,args.handoff_sha256)
    result={'kind':KIND,'mode':args.mode,'replay':args.replay,'complete':False,'passed':False,
        'tiny_fixture':args.tiny,'device':args.device,'research_admitted':False,'artifacts':{}}
    def progress(stage):
        result['stage']=stage;write_record(directory/'worker.json',result,replace=(directory/'worker.json').exists())
        print('Projection reuse:',args.mode,args.replay,stage,flush=True)
    def save(group,values):
        path=directory/(group+'.pt')
        with path.open('xb') as f:torch.save(values,f)
        result['artifacts'][group]={'file':path.name,'sha256':digest(path),
            'tensors':{name:tensor_signature(t) for name,t in values.items()}}
        progress(result['stage'])
    try:
        progress('configure_and_bind_sources')
        configure_execution('shared-fp64-native',device=args.device);torch.set_num_threads(4)
        upstream_gate(args.upstream,metadata)
        if not args.tiny:
            plan=json.loads(Path(args.plan).read_text());validate_launch(plan,args.handoff_sha256)
            if digest(Path(args.plan))!=args.plan_sha256:raise ValueError('Launch plan changed.')
            hardware_gate(plan,allowed_pid=os.getpid())
            if str(torch.__version__).split('+')[0]!='2.8.0' or torch.version.cuda!='12.8' or not torch.cuda.is_available():
                raise ValueError('Use the existing PyTorch 2.8.0 / CUDA 12.8; no fallback or install.')
            total=torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(.9,7*2**30/total))
        torch.manual_seed(20261029)
        progress('common_new_native_reuse_probe');result['native_probe']=native_reuse_probe(args.device)
        gc.collect()
        if args.device=='cuda':torch.cuda.empty_cache()
        torch.manual_seed(20261029)
        progress('load_model_and_one_training_input')
        if args.tiny:
            model,clean,masks_for=tiny_fixture(args.upstream)
            initial=load_fixture_state(model,args.initial_fixture,args.initial_fixture_sha256)
            data={'kind':'generated-tiny-CPU-fixture','initial_fixture':initial,'video_id':'projection-reuse-fixture'}
        else:
            clean=training_input(root,metadata)
            model=build_vitl(args.upstream,activation_checkpointing=True)
            checkpoint=load_checkpoint(model,args.checkpoint,CHECKPOINT)
            data={'kind':'original-prepared-training','checkpoint_sha256':checkpoint['sha256'],
                'video_id':metadata['training_row']['video_id'],'pixel_sha256':metadata['training_row']['pixel_sha256'],
                'training_manifest_sha256':metadata['training_manifest_sha256']}
            masks_for=lambda seed:sample_masks(args.upstream,batch_size=1,seed=seed)
        apply_model_policy(model,'shared-fp64-native');model.to(args.device).eval();clean=clean.to(args.device)
        guard=ExecutionGuard(model,args.device,target_versions=True)
        weights=state_digest(model);targets=model.clean_targets(clean);target=tensor_signature(targets)
        settings=EvaluationSettings.for_method('primary',2.)
        seeds=seeds_for(settings,data['video_id'],sensitivity_banks=2)
        banks={name:[[m.to(args.device) for m in masks_for(seed % (2**31-1))] for seed in values]
               for name,values in seeds.items() if name!='start'}
        if any(len(v)!=2 for group in banks.values() for v in group):raise ValueError('Expected both original mask families.')
        result['contract']={'kind':KIND,'canonical_evaluation_artifact':False,'diagnostic_only':True,
            'research_admitted':False,'handoff_sha256':args.handoff_sha256,'data':data,
            'runtime':runtime_contract(args.device,model=model),'weights_sha256':weights,'targets':target,
            'pixels':tensor_signature(clean),'seeds':seeds,'settings':settings.__dict__,
            'diagnostic_step_255':2.,'research_step_255':None,
            'masks':{name:[[[tensor_signature(m.context),tensor_signature(m.target)] for m in families] for families in draws]
                     for name,draws in banks.items()},
            'input_and_loss_tolerances':TOLERANCES,'patch_tolerances':PATCH_TOLERANCES,
            'hardware':None if args.tiny else hardware_gate(plan,allowed_pid=os.getpid()),
            'all_parameter_gradients_or_optimizer_qualified':False,'allocator_cap_gib':None if args.tiny else 7}
        progress('original_projection_FP64_oracle')
        checks,values=projection_check(model,clean,targets,banks['attack'][0])
        result['oracle_checks']=checks;save('oracles',values)
        if not all(v['passed'] for v in checks.values()):raise ValueError('Original projection oracle rejected; no tolerance changed.')
        del values;gc.collect()
        progress('both_EOT8_input_and_projection_parameter_gradients')
        gradients={};gradient_receipts={};seconds={}
        for mode in MODES:
            values,receipts,elapsed=full_gradient(model,clean,targets,banks['attack'],mode)
            gradients.update({mode+'_'+k:v for k,v in values.items()});gradient_receipts[mode]=receipts;seconds[mode]=elapsed
            guard.check()
        result['gradient_checks']={name:agreement(gradients['reuse_'+name],gradients['recompute_'+name],
            TOLERANCES['clean_loss'] if name=='loss' else TOLERANCES['input_gradient'] if name=='input' else PATCH_TOLERANCES[name])
            for name in ('loss','input','weight','bias')}
        result['gradient_receipts']=gradient_receipts;result['gradient_seconds']=seconds
        clean_attack_loss=float(gradients['recompute_loss'])
        save('gradients',gradients)
        if not all(v['passed'] for v in result['gradient_checks'].values()):raise ValueError('Reused EOT derivatives rejected; unchanged bounds.')
        del gradients;gc.collect()
        progress('full_geometry_projection_counts_and_timings')
        values,rows=projection_timings(model,clean,args.device);result['component_timings']=rows;save('tokens',values)
        del values;gc.collect()
        progress('primary_PGD20_R2_EOT8_and_independent_scoring')
        if not args.tiny:hardware_gate(plan,allowed_pid=os.getpid())
        receipts=[];objective=diagnostic_objective(model,targets,banks['attack'],mode=args.mode,receipts=receipts)
        synchronize(args.device)
        if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
        tick=time.monotonic()
        attack=pgd_linf(objective,clean,epsilon=4/255,step_size=2/255,steps=20,restarts=2,seed=seeds['start'])
        with torch.no_grad():
            science={'clean_attack_bank_loss':clean_attack_loss,'attack_loss':float(attack.losses),'max_pixel_delta':float((attack.adversarial-clean).abs().max()),'scores':{}}
            for name in ('main','audit_0','audit_1'):
                key='score' if name=='main' else name
                scoring=diagnostic_objective(model,targets,banks[key],mode=args.mode,receipts=receipts)
                science['scores'][name]={'clean':float(scoring(clean)),'adversarial':float(scoring(attack.adversarial))}
        synchronize(args.device);result['primary_and_three_score_banks_seconds']=time.monotonic()-tick
        result['peak_primary_cuda_allocated_bytes']=torch.cuda.max_memory_allocated() if args.device=='cuda' else None
        result['attack_receipts']=receipts;result['science']=science
        if (len(receipts)!=49 or any(r['requests']!=16 for r in receipts[:43])
                or any(r['requests']!=8 for r in receipts[43:])
                or any(r['forward_computations']!=(1 if args.mode=='reuse' else r['requests']) for r in receipts)
                or science['max_pixel_delta']>4/255+1e-7 or attack.adversarial.min()<0 or attack.adversarial.max()>1
                or science['attack_loss']<clean_attack_loss):
            raise ValueError('Draw count, bounds or clean-candidate gate rejected.')
        save('attack',{'clean':clean.detach().cpu(),'adversarial':attack.adversarial.detach().cpu(),'attack_loss':attack.losses.detach().cpu()})
        guard.check()
        if state_digest(model)!=weights or tensor_signature(targets)!=target or any(p.grad is not None for p in model.parameters()):
            raise ValueError('Model/target changed or parameter gradients accumulated.')
        if not args.tiny:hardware_gate(plan,allowed_pid=os.getpid())
        verify(root,args.handoff_sha256)
        result.update(complete=True,passed=True,stage='complete',restored_original_model_identity=True)
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),passed=False)
        raise
    finally:
        result['wall_seconds']=time.monotonic()-started
        write_record(directory/'worker.json',result,replace=(directory/'worker.json').exists())


def summarize(directory,metadata,handoff_sha,tiny):
    directory=Path(directory);workers={};artifacts={};common=None
    parent=read_record(directory/'session.json')
    if (parent['handoff_sha256']!=handoff_sha or parent['tiny_fixture']!=tiny
            or [(v['mode'],v['replay']) for v in parent['tasks']]!=list(TASKS)):
        raise ValueError('Parent does not contain four bound worker receipts.')
    initial=None
    if tiny:
        _,initial=read_fixture_state(directory/'initial-fixture.pt',parent['initial_fixture']['sha256'])
    for mode,replay in TASKS:
        folder=directory/'workers'/f'{mode}-{replay}';v=read_record(folder/'worker.json')
        task=next(t for t in parent['tasks'] if (t['mode'],t['replay'])==(mode,replay))
        if digest(folder/'worker.json')!=task['worker_sha256']:raise ValueError('Worker receipt hash changed.')
        if (v['mode']!=mode or v['replay']!=replay or v['tiny_fixture']!=tiny or not v['complete'] or not v['passed']
                or v['kind']!=KIND or v['device']!=('cpu' if tiny else 'cuda') or v['research_admitted'] is not False
                or set(v['artifacts'])!={'oracles','gradients','tokens','attack'}):
            raise ValueError('Require all four complete workers in the correct domain.')
        validate_probe(v['native_probe'],device=v['device'])
        if (v['native_probe'].get('delegator')!='models.projection_reuse._IndependentVJP'
                or v['native_probe'].get('diagnostic_kind')!=KIND
                or v['native_probe'].get('projection_requests')!=2
                or v['native_probe'].get('forward_computations')!=1
                or not any('_IndependentVJPBackward' in r['name'] for r in v['native_probe']['profiler_events'])):
            raise ValueError('The actual new delegator probe is missing.')
        contract=v['contract']
        if contract['handoff_sha256']!=handoff_sha or contract['input_and_loss_tolerances']!=TOLERANCES or contract['patch_tolerances']!=PATCH_TOLERANCES:
            raise ValueError('Mixed source identity or changed numerical bounds.')
        runtime=contract['runtime']
        if (runtime['device']!=v['device'] or runtime['threads']!=4 or not runtime['deterministic_algorithms']
                or runtime['deterministic_warn_only'] or runtime['matmul_precision']!='highest'
                or runtime['matmul_tf32'] or not runtime['cudnn_enabled'] or runtime['cudnn_tf32']
                or runtime['cudnn_benchmark'] or not runtime['cudnn_deterministic']
                or runtime['cublas_workspace_config']!=':4096:8'
                or runtime['model_execution']['policy']!='shared-fp64-native'
                or not tiny and (runtime['torch'].split('+')[0]!='2.8.0' or runtime['cuda']!='12.8')):
            raise ValueError('Diagnostic runtime/precision differs from the specified reference.')
        settings=EvaluationSettings.for_method('primary',2.)
        if (contract['settings']!=settings.__dict__ or contract['seeds']!=seeds_for(settings,contract['data']['video_id'],sensitivity_banks=2)
                or contract['research_admitted'] is not False or contract['canonical_evaluation_artifact'] is not False):
            raise ValueError('Diagnostic attack settings, scoring seeds or artifact domain changed.')
        if tiny:
            if contract['data'].get('initial_fixture')!=initial or contract['hardware'] is not None:
                raise ValueError('Generated CPU initialization or domain changed.')
        elif (contract['data']['video_id']!=metadata['training_row']['video_id']
                or contract['data']['pixel_sha256']!=metadata['training_row']['pixel_sha256']
                or contract['pixels']['sha256']!=metadata['training_row']['pixel_sha256']
                or contract['pixels']['shape']!=[1,3,16,256,256]
                or contract['data']['checkpoint_sha256']!=CHECKPOINT
                or contract['hardware']!=parent['hardware']):
            raise ValueError('Original training/checkpoint/GPU identity changed.')
        if common is None:common=contract
        elif contract!=common:raise ValueError('Fresh workers have different weights/data/masks/runtime.')
        saved={}
        for group,item in v['artifacts'].items():
            path=folder/item['file']
            if path.parent!=folder or path.is_symlink() or digest(path)!=item['sha256']:
                raise ValueError('Worker artifact path/hash mismatch.')
            values=torch.load(path,map_location='cpu',weights_only=True)
            if set(values)!=set(item['tensors']) or any(tensor_signature(t)!=item['tensors'][n] for n,t in values.items()):
                raise ValueError('Worker tensor signature mismatch.')
            saved[group]=values
        o=saved['oracles']
        for name,a,b,tol in [('output','patch_tokens','fp64_tokens',PATCH_TOLERANCES['output']),
                ('input','patch_input_gradient','fp64_input_gradient',TOLERANCES['input_gradient']),
                ('weight','patch_weight_gradient','fp64_weight_gradient',PATCH_TOLERANCES['weight']),
                ('bias','patch_bias_gradient','fp64_bias_gradient',PATCH_TOLERANCES['bias'])]:
            actual=agreement(o[a],o[b],tol)
            if not actual['passed'] or not same_report(actual,v['oracle_checks'][name]):
                raise ValueError('Saved original FP64 oracle gate rejected.')
        for name in ('loss','input','weight','bias'):
            g=saved['gradients'];tol=TOLERANCES['clean_loss'] if name=='loss' else TOLERANCES['input_gradient'] if name=='input' else PATCH_TOLERANCES[name]
            actual=agreement(g['reuse_'+name],g['recompute_'+name],tol)
            if not actual['passed'] or not same_report(actual,v['gradient_checks'][name]):raise ValueError('Saved EOT gradient comparison rejected.')
        for gradient_mode in MODES:
            if v['gradient_receipts'][gradient_mode]!=[{'requests':16,'forward_computations':1 if gradient_mode=='reuse' else 16}]:
                raise ValueError('Incomplete EOT derivative projection requests.')
        t=saved['tokens']
        if not torch.equal(t['tokens_reference'],t['tokens_recompute']) or not torch.equal(t['tokens_reference'],t['tokens_reuse']):
            raise ValueError('Projection tokens are not bitwise equal.')
        if len(v['attack_receipts'])!=49:raise ValueError('Missing original draw/scoring requests.')
        attack=saved['attack'];clean=attack['clean'];adversarial=attack['adversarial'];science=v['science']
        if (tensor_signature(clean)!=contract['pixels'] or clean.shape!=adversarial.shape
                or clean.min()<0 or clean.max()>1 or adversarial.min()<0 or adversarial.max()>1
                or float((adversarial-clean).abs().max())!=science['max_pixel_delta']
                or science['max_pixel_delta']>4/255+1e-7
                or float(attack['attack_loss'])!=science['attack_loss']
                or science['attack_loss']<science['clean_attack_bank_loss']
                or science['clean_attack_bank_loss']!=float(saved['gradients']['recompute_loss'])
                or not torch.isfinite(saved['gradients']['recompute_input']).all()
                or not saved['gradients']['recompute_input'].abs().max()>0
                or set(science['scores'])!={'main','audit_0','audit_1'}
                or any(not math.isfinite(y) for x in science['scores'].values() for y in x.values())):
            raise ValueError('Saved input, pixel bounds, attack selection or independent scoring rejected.')
        for i,r in enumerate(v['attack_receipts']):
            n=16 if i<43 else 8
            if r!={'requests':n,'forward_computations':1 if mode=='reuse' else n}:raise ValueError('Incorrect projection reuse receipts.')
        workers[(mode,replay)]=v;artifacts[(mode,replay)]=saved
    for mode in MODES:
        for group in artifacts[(mode,0)]:
            for name,t in artifacts[(mode,0)][group].items():
                if not torch.equal(t,artifacts[(mode,1)][group][name]):raise ValueError('Fresh-process tensor replay is not exact.')
        if workers[(mode,0)]['science']!=workers[(mode,1)]['science']:raise ValueError('Fresh-process science replay differs.')
    for replay in (0,1):
        a,b=artifacts[('recompute',replay)]['attack'],artifacts[('reuse',replay)]['attack']
        if any(not torch.equal(a[k],b[k]) for k in a) or workers[('recompute',replay)]['science']!=workers[('reuse',replay)]['science']:
            raise ValueError('Reused attack/scoring is not bitwise equal to the original.')
    times={mode:[workers[(mode,i)]['primary_and_three_score_banks_seconds'] for i in (0,1)] for mode in MODES}
    if any(not math.isfinite(x) or x<=0 for values in times.values() for x in values):raise ValueError('Invalid timing observations.')
    return {'kind':KIND,'passed':True,'tiny_fixture':tiny,'domain':'CPU-fixture' if tiny else 'university-CUDA',
        'four_fresh_workers_complete':True,'fresh_process_tensor_and_science_replay_exact':True,
        'attack_and_scoring_bitwise_equal_between_modes':True,'original_tolerances_unchanged':True,
        'primary_with_three_scoring_banks_seconds':times,
        'observed_two_sample_median_speedup':statistics.median(times['recompute'])/statistics.median(times['reuse']),
        'observed_reuse_median_is_faster':statistics.median(times['reuse'])<statistics.median(times['recompute']),
        'timing_scope':'One clip, PGD20/R2/EOT8 and three independent scoring banks; diagnostic guard overhead included, not a full-study bound.',
        'full_parameter_optimizer_recovery_qualified':False,'full_study_cost_approved':False,'research_admitted':False}


def export(root,output):
    root,output=Path(root),Path(output)
    files={'REUSE-HANDOFF.json':root/'REUSE-HANDOFF.json'}
    m=json.loads((root/'REUSE-HANDOFF.json').read_text())
    for name in m['files']:
        if name.endswith('.py') or name in ('input/train-index.json',):files['source/'+name]=root/name
    for p in output.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt','.pt') and p.name not in ('evidence.tar.gz.manifest.json',):
            if p.is_symlink():raise ValueError('Export refuses symlinks.')
            files['results/'+str(p.relative_to(output))]=p
    if sum(p.stat().st_size for p in files.values())>MAX_EXPORT:raise ValueError('Evidence exceeds the 1 GiB export cap; preserve output.')
    result=make_archive(output/'evidence.tar.gz',files)
    (output/'evidence.tar.gz.manifest.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def run(args,root):
    started=args.launcher_started if args.launcher_started is not None else time.time()
    if not math.isfinite(started) or not 0<=time.time()-started<=120:raise ValueError('Invalid attended launcher clock.')
    root=Path(root);metadata=verify(root,args.handoff_sha256);upstream_gate(args.upstream,metadata)
    if args.tiny:
        if not 0<args.max_seconds<=300 or not 0<args.worker_seconds<=60:raise ValueError('CPU rehearsal limits exceeded.')
    else:
        plan=json.loads(Path(args.plan).read_text())
        validate_launch(plan,args.handoff_sha256)
        if digest(Path(args.plan))!=args.plan_sha256:raise ValueError('Launch plan changed.')
        hardware_gate(plan)
        if args.max_seconds!=plan['max_seconds'] or args.worker_seconds!=plan['worker_seconds']:raise ValueError('Resource limits differ.')
    deadline=started+args.max_seconds;end=time.monotonic()+deadline-time.time()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    if not args.tiny:
        claims=root/'claims';claims.mkdir(exist_ok=True)
        with (claims/(args.plan_sha256+'.json')).open('x') as f:json.dump({'started_unix':started,'output':str(output)},f)
    initial=None
    if args.tiny:
        initial=save_tiny_fixture(args.upstream,output/'initial-fixture.pt',seed=20261029)
    session={'kind':KIND,'tiny_fixture':args.tiny,'handoff_sha256':args.handoff_sha256,'started_unix':started,
        'deadline_unix':deadline,'tasks':[],'complete':False,'passed':False,'research_admitted':False,'paid_compute':False,
        'initial_fixture':initial,'hardware':None if args.tiny else hardware_gate(plan)}
    write_record(output/'session.json',session)
    child=None
    def monitor(observation):
        with (output/'gpu-monitor.jsonl').open('a') as f:
            f.write(json.dumps(observation,sort_keys=True,allow_nan=False)+'\n');f.flush()
    def alarm(*unused):raise TimeoutError('Independent overall university diagnostic deadline.')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.1,deadline-time.time()-5))
    def interrupted(*unused):raise RuntimeError('Attended SSH session interrupted; no retry.')
    signal.signal(signal.SIGHUP,interrupted);signal.signal(signal.SIGTERM,interrupted)
    try:
        for mode,replay in TASKS:
            remaining=min(deadline-time.time(),end-time.monotonic())-(15 if args.tiny else 120)
            if remaining<=0:raise TimeoutError('No time left for a worker and evidence export.')
            seconds=min(args.worker_seconds,remaining)
            folder=output/'workers'/f'{mode}-{replay}';folder.mkdir(parents=True)
            auth={'mode':mode,'replay':replay,'handoff_sha256':args.handoff_sha256,'deadline_unix':time.time()+seconds}
            write_record(folder/'authorization.json',auth)
            command=[sys.executable,'-B','-u',str(root/'scripts/check_projection_reuse.py'),'--diagnostic',
                '--device',args.device,'--worker','--mode',mode,'--replay',str(replay),'--worker-dir',str(folder),
                '--authorization-sha256',digest(folder/'authorization.json'),'--handoff-sha256',args.handoff_sha256,
                '--upstream',str(args.upstream)]
            if args.tiny:
                command+=['--tiny','--initial-fixture',str(output/'initial-fixture.pt'),'--initial-fixture-sha256',initial['sha256']]
            else:
                command+=['--plan',str(args.plan),'--plan-sha256',args.plan_sha256,'--checkpoint',str(args.checkpoint)]
            print('Fresh projection worker:',mode,replay,flush=True)
            if not args.tiny:hardware_gate(plan,retain=monitor)
            with (folder/'worker.log').open('x') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'})
                tick=time.monotonic();last_gate=0.
                try:
                    while child.poll() is None:
                        if time.monotonic()-tick>=seconds:raise TimeoutError('Worker time limit reached; no retry.')
                        if not args.tiny and time.monotonic()-last_gate>=5:
                            hardware_gate(plan,allowed_pid=child.pid,retain=monitor);last_gate=time.monotonic()
                        time.sleep(.2)
                    if child.returncode:raise RuntimeError('Worker failed; preserve partial evidence, no retry.')
                finally:
                    if child.poll() is None:
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                    child.wait();child=None
            session['tasks'].append({'mode':mode,'replay':replay,'worker_sha256':digest(folder/'worker.json')})
            write_record(output/'session.json',session,replace=True)
        summary=summarize(output,metadata,args.handoff_sha256,args.tiny)
        write_record(output/'summary.json',summary)
        session.update(complete=True,passed=True,summary_sha256=digest(output/'summary.json'))
    except BaseException as exc:
        session.update(error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        if child is not None:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            child.wait()
        session['wall_seconds']=time.time()-started
        write_record(output/'session.json',session,replace=True)
        try:
            verify(root,args.handoff_sha256)
            saved=export(root,output)
            print('Evidence:',json.dumps({'bytes':saved['bytes'],'sha256':saved['sha256']}),flush=True)
        except BaseException as exc:
            session.update(passed=False,complete=False,export_error=str(exc))
            write_record(output/'session.json',session,replace=True)
            raise
        finally:signal.setitimer(signal.ITIMER_REAL,0)
    return session
