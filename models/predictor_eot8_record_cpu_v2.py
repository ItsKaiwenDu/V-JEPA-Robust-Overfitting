"""Local-only five-pass CPU analogue using bounded external ownership records."""
import json
import os
from pathlib import Path
import platform
import sys
import time

PHASES=('EOT8-before','draw-before','draw-profiled','draw-after','EOT8-after')
TASKS=(('original',0),('predictor-math',0),('original',1),('predictor-math',1))
BOUNDS={'loss':{'rtol':1e-5,'atol':2e-6},'input':{'rtol':2e-3,'atol':2e-7}}


def run(args):
    if platform.node() in ('linux10603','linux10609') or not args.tiny or args.device!='cpu':
        raise ValueError('Only local CPU analogue execution is implemented; no target or CUDA grant.')
    import torch
    from attacks.pgd import native_objective
    from models.candidate_evaluation import EvaluationSettings,seeds_for
    from models.cuda_profile import tensor_signature
    from models.diagnostics import state_digest
    from models.execution import ExecutionGuard,configure_execution,apply_model_policy,precision_contract
    from models.fixture_state import load_fixture_state
    from models.native_diagnostic import tiny_fixture
    from models.training import runtime_contract
    from models.cpu_oracle_scratch import digest,write_json
    from models.predictor_eot8_owned_v1 import PredictorRoute
    from models.predictor_eot8_owned_receipt_v1 import verify_ownership
    from models.predictor_eot8_record_v2 import WorkerRecord,load_worker
    from models.predictor_eot8_capture_v1 import save_return,compare
    from models import predictor_eot8_events_v1 as events
    folder=Path(args.output); folder.mkdir(mode=0o700,exist_ok=False)
    if args.index not in range(4):raise ValueError('Unknown worker index.')
    mode,replay=TASKS[args.index]
    started=time.monotonic();state={'kind':'predictor-clean-eot8-owned-worker-v1','execution_host':platform.node(),'complete':False,'passed':False,
        'local_tiny_analogue':args.tiny,'mode':mode,'replay':replay,'index':args.index,'pid':os.getpid(),
        'control_sha256':getattr(args,'control_sha256',None),'artifacts':{},'phases':[],
        'performance_qualified':False,'research_admitted':False,'old_refusals_waived':False}
    store=WorkerRecord(folder);route=None
    def record(*,final=False):return store.write(state,route=route,final=final)
    record()
    try:
        def reference_receipt(path,expected_mode,expected_replay,expected_index):
            ref=Path(path);saved=load_worker(ref,expected_control=state['control_sha256'])
            if (saved['complete'] is not True or saved['passed'] is not True or saved['mode']!=expected_mode
                    or saved['replay']!=expected_replay or saved['index']!=expected_index or saved['local_tiny_analogue'] is not args.tiny
                    or saved['control_sha256']!=state['control_sha256'] or len(saved['artifacts'])!=10 or saved['phases']!=list(PHASES)):
                raise ValueError('New complete passing reference in the same domain is required before model work.')
            for item in saved['artifacts'].values():
                if Path(item['file']).name!=item['file'] or digest(ref/item['file'])!=item['sha256']:raise ValueError('Reference artifact changed.')
            if saved['execution_host']!=state['execution_host']:raise ValueError('Fresh reference host changed.')
            return saved
        if mode=='predictor-math':
            if not args.reference:raise ValueError('Candidate reference missing before model work.')
            reference_receipt(args.reference,'original',replay,args.index-1)
        if replay==1:
            if not args.replay_reference:raise ValueError('Fresh replay reference missing before model work.')
            reference_receipt(args.replay_reference,mode,0,args.index-2)
        configure_execution('shared-fp64-native',device=args.device);torch.set_num_threads(4);torch.manual_seed(20261029)
        model,clean,masks_for=tiny_fixture(args.upstream)
        initial=load_fixture_state(model,args.initial,args.initial_sha)
        data={'kind':'local-tiny-CPU','initial':initial};video='tiny-predictor-eot8'
        apply_model_policy(model,'shared-fp64-native');model.to(args.device).eval();clean=clean.to(args.device)
        guard=ExecutionGuard(model,args.device,target_versions=True);weights=state_digest(model)
        targets=model.clean_targets(clean);target=tensor_signature(targets)
        seeds=seeds_for(EvaluationSettings.for_method('primary',2.),video)['attack']
        banks=[[m.to(args.device) for m in masks_for(seed%(2**31-1))] for seed in seeds]
        masks=[[[tensor_signature(m.context),tensor_signature(m.target)] for m in draw] for draw in banks]
        if len(banks)!=8 or any(len(draw)!=2 for draw in banks):raise ValueError('EOT8/two-family scope changed.')
        baseline={'weights_sha256':weights,'targets':target,'pixels':tensor_signature(clean),'seeds':seeds,
                  'masks':masks,'runtime':runtime_contract(args.device,model=model),'data':data}
        prior=None
        state['baseline']=baseline;record()
        # Save target and mask identities through bounded regular files, no computed oracle.
        with (folder/'targets.pt').open('xb') as f:torch.save(targets.detach().cpu(),f)
        if (folder/'targets.pt').stat().st_size>12*2**20:raise ValueError('Target capture cap.')
        write_json(folder/'mask-signatures.json',masks)
        def versions():
            return tuple((n,id(t),t._version,str(t.dtype),str(t.device),t.requires_grad) for n,t in (*model.named_parameters(),*model.named_buffers()))
        original_versions=versions();modules=tuple((n,id(m),m.training) for n,m in model.named_modules());precision=precision_contract()
        def frozen():
            if versions()!=original_versions or tuple((n,id(m),m.training) for n,m in model.named_modules())!=modules or precision_contract()!=precision or tensor_signature(targets)!=target:
                raise ValueError('Model/target/runtime changed.')
            if any(p.grad is not None for p in model.parameters()):raise ValueError('Parameter gradients accumulated.')
        values={}
        route=PredictorRoute(model,mode,tiny=args.tiny)
        with route:
            if mode=='predictor-math':
                try:guard.check()
                except ValueError:state['existing_study_guard_rejected_candidate']=True
                else:raise ValueError('Existing study guard admitted unsupported candidate.')
            for phase in PHASES:
                route.phase=phase;route.check();frozen();state['phase']=phase;record()
                x=clean.detach().clone().requires_grad_(True)
                selected=banks if phase.startswith('EOT8') else banks[:1]
                objective=native_objective(model,targets,selected)
                def calculate():
                    loss=objective(x).sum()
                    item=save_return(folder/(phase+'-loss.pt'),loss,'loss',clean.shape)
                    state['artifacts'][phase+'-loss']=item;record()
                    gradient,=route.input_gradient(loss,x)
                    item=save_return(folder/(phase+'-input.pt'),gradient,'input',clean.shape)
                    state['artifacts'][phase+'-input']=item;record()
                    return {'loss':loss.detach().cpu(),'input':gradient.detach().cpu()}
                if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
                if phase=='draw-profiled':
                    activities=[torch.profiler.ProfilerActivity.CPU]
                    if args.device=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
                    with torch.profiler.profile(activities=activities,record_shapes=False,with_stack=False,profile_memory=False) as profile:current=calculate()
                    state['trace_capture']=events.save(profile,folder/'trace.jsonl')
                    state['backend']=events.verify(events.read(folder/'trace.jsonl'),route.rows,cuda=not args.tiny,mode=mode,owners=[r+'.'+str(i) for r,blocks in [('encoder',model.encoder.blocks),('predictor',model.predictor.predictor_blocks)] for i in range(len(blocks))])
                    del profile
                else:current=calculate()
                # Every original block must run for every declared mask family.
                # Recompute calls are observed separately, never assumed exactly two.
                primary=[r for r in route.rows if r['phase']==phase and not r['during_backward']]
                for role,blocks in [('encoder',model.encoder.blocks),('predictor',model.predictor.predictor_blocks)]:
                    for index in range(len(blocks)):
                        owned=[r for r in primary if r['owner']==role+'.'+str(index)]
                        if len(owned)!=len(selected)*2 or any(r['outcome']!='returned' for r in owned):
                            raise ValueError('Declared draw/family/block forward coverage changed.')
                if args.device=='cuda':
                    peak=torch.cuda.max_memory_allocated();state.setdefault('peaks',{})[phase]=peak
                    if not 0<peak<=7*2**30:raise ValueError('Full-model allocator cap.')
                for name,t in current.items():
                    if not torch.isfinite(t).all() or name=='input' and not t.abs().max()>0:raise ValueError('Saved nonfinite/zero return rejected.')
                    if prior is not None and phase.startswith('EOT8'):compare(t,prior[name],BOUNDS[name])
                if args.reference:
                    ref=Path(args.reference);accepted=load_worker(ref,expected_control=state['control_sha256'])
                    if not accepted['complete'] or not accepted['passed'] or accepted['index']!=args.index-1 or accepted['mode']!='original' or accepted['replay']!=replay or accepted['baseline']!=baseline:
                        raise ValueError('Candidate requires new complete passing same-replay original reference.')
                    for name,t in current.items():
                        item=accepted['artifacts'][phase+'-'+name];p=ref/item['file']
                        if digest(p)!=item['sha256']:raise ValueError('Fresh reference capture changed.')
                        compare(t,torch.load(p,weights_only=True,map_location='cpu'),BOUNDS[name])
                elif mode!='original':raise ValueError('Candidate reference missing.')
                key='EOT8-before' if phase.startswith('EOT8') else 'draw-before'
                if key in values and any(not torch.equal(t,values[key][name]) for name,t in current.items()):raise ValueError('Observer/control exact replay rejected.')
                values[phase]=current;state['phases'].append(phase);route.check();frozen();record()
        guard.check();frozen()
        if state_digest(model)!=weights:raise ValueError('Final model state changed.')
        if args.replay_reference:
            old=load_worker(Path(args.replay_reference),expected_control=state['control_sha256'])
            if not old['complete'] or not old['passed'] or old['mode']!=mode or old['replay']!=0 or replay!=1 or old['baseline']!=baseline:raise ValueError('Fresh complete replay-0 reference required.')
            if {k:v['source'] for k,v in state['artifacts'].items()}!={k:v['source'] for k,v in old['artifacts'].items()}:raise ValueError('Fresh-process complete tensor replay rejected.')
        elif replay!=0:raise ValueError('Replay-1 reference missing.')
        state['ownership_check']=verify_ownership({**state,'call_rows':route.rows,'backward_scopes':route.scopes,'ownership_refusals':route.refusals})
        state.update(complete=True,passed=True,adapter_restored=True,model_and_targets_unchanged=True)
    except BaseException as exc:
        state.update(complete=False,passed=False,error_type=type(exc).__name__,error=str(exc)[:2048]);raise
    finally:
        original_error=sys.exception()
        if route is not None:state['adapter_restored']=not route.active
        state['wall_seconds']=time.monotonic()-started
        try:record(final=True)
        except BaseException as error:
            state.update(complete=False,passed=False)
            try:store.failure(state,error)
            except BaseException as marker_error:
                print('Worker evidence writer and failure marker failed: '+type(marker_error).__name__,file=sys.stderr,flush=True)
            if original_error is None:raise
    return state
