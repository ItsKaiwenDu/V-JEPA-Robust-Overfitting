"""One original native capture and separately guarded fixed-input component body."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


def run_capture(args):
    if type(args.tiny) is not bool:raise ValueError('Exact capture domain required.')
    if args.tiny:
        if args.device!='cpu' or platform.node() in ('linux10603','linux10609'):raise ValueError('Local CPU analogue only.')
    elif args.device!='cuda' or platform.node()!='linux10603' or platform.python_version()!='3.11.17':
        raise ValueError('Exact authorized target host/runtime required before scientific imports.')
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
    from models.predictor_eot8_capture_v1 import save_return,compare,tensor_info
    from models.predictor_sdpa_observer_v1 import NativeSDPAObserver
    from models.predictor_sdpa_capture_record_v1 import BoundaryStore,load
    folder=Path(args.output);folder.mkdir(mode=0o700,exist_ok=False);(folder/'boundary').mkdir(mode=0o700)
    start=time.monotonic();route=observer=None
    state={'kind':'original-SDPA-observer-local-CPU-rehearsal-v1' if args.tiny else 'original-SDPA-native-target-capture-v1',
        'local_tiny':args.tiny,'device':args.device,'execution_host':platform.node(),'control_sha256':args.control_sha256,
        'pid':os.getpid(),'complete':False,'original_value_replay_exact':False,'artifacts':{},
        'unobserved_before_after_work_performed':False,'research_admitted':False,'performance_qualified':False,
        'native_candidate_adopted':False,'CUDA_kernel_qualified':False}
    def publish():write_json(folder/'worker.json',state,replace=(folder/'worker.json').exists())
    publish()
    try:
        configure_execution('shared-fp64-native',device=args.device);torch.set_num_threads(4);torch.manual_seed(20261029)
        if args.tiny:
            model,clean,masks_for=tiny_fixture(args.upstream)
            initial=load_fixture_state(model,args.initial,args.initial_sha)
            data={'kind':'local-tiny-CPU','initial':initial};video='tiny-predictor-eot8'
            before=json.loads((Path(args.reference)/'worker.json').read_text())
            expected={}
            for name in ('loss','input'):
                item=before['artifacts'][name];path=Path(args.reference)/item['file']
                if digest(path)!=item['sha256']:raise ValueError('Actual original CPU anchor changed.')
                expected[name]=torch.load(path,weights_only=True,map_location='cpu')
            approved=before['baseline'];anchors={n:tensor_info(t)['sha256'] for n,t in expected.items()}
        else:
            import numpy
            if torch.__version__!='2.8.0+cu128' or torch.version.cuda!='12.8' or numpy.__version__!='2.4.6' or not torch.cuda.is_available():
                raise ValueError('Exact installed target runtime required; no installation/fallback.')
            torch.cuda.set_per_process_memory_fraction(7*2**30/torch.cuda.get_device_properties(0).total_memory)
            torch.cuda.reset_peak_memory_stats()
            from models.eot_profile_packet import verify
            from models.reuse_diagnostic import upstream_gate
            from models.reuse_handoff import training_input
            from models.vjepa2 import build_vitl,load_checkpoint,sample_masks
            _,parent=verify(args.packet,'c95c0b21ee018dfd6c897756baf453dc7d69eb1d52b5c64d99e1ac4b4b945dbd')
            upstream_gate(args.upstream,parent)
            model=build_vitl(args.upstream,activation_checkpointing=True)
            checkpoint=load_checkpoint(model,args.checkpoint,'5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012')
            clean=training_input(args.packet,parent);masks_for=lambda seed:sample_masks(args.upstream,batch_size=1,seed=seed)
            video=parent['training_row']['video_id'];data={'kind':'original-training','video_id':video,'checkpoint_sha256':checkpoint['sha256']}
            expected=torch.load(Path(args.packet)/'input/prior-EOT8-derivatives.pt',weights_only=True,map_location='cpu')
            approved=args.contract['approved_baseline'];anchors=args.contract['full_native_anchor_logical_sha256']
        if set(expected)!={'loss','input'} or any(tensor_info(expected[n])['sha256']!=anchors[n] for n in expected):
            raise ValueError('Historical actual original anchors differ from independently reviewed native values.')
        apply_model_policy(model,'shared-fp64-native');model.to(args.device).eval();clean=clean.to(args.device)
        guard=ExecutionGuard(model,args.device,target_versions=True);weights=state_digest(model)
        targets=model.clean_targets(clean);target=tensor_signature(targets)
        seeds=seeds_for(EvaluationSettings.for_method('primary',2.),video)['attack']
        banks=[[p.to(args.device) for p in masks_for(seed%(2**31-1))] for seed in seeds]
        masks=[[[tensor_signature(p.context),tensor_signature(p.target)] for p in d] for d in banks]
        baseline={'weights_sha256':weights,'targets':target,'pixels':tensor_signature(clean),'seeds':seeds,
            'masks':masks,'runtime':runtime_contract(args.device,model=model),'data':data}
        if baseline!=approved or len(banks)!=8 or any(len(d)!=2 for d in banks):raise ValueError('Original baseline/runtime/masks changed.')
        state['baseline']=baseline;state['full_native_anchor_logical_sha256']=anchors;publish()
        state['artifacts']['targets']=save_return(folder/'targets.pt',targets,'input',targets.shape)
        indices=[[{'context':p.context.detach().cpu().tolist(),'target':p.target.detach().cpu().tolist()} for p in d] for d in banks]
        write_json(folder/'mask-indices.json',indices)
        state['mask_indices']={'file':'mask-indices.json','bytes':(folder/'mask-indices.json').stat().st_size,'sha256':digest(folder/'mask-indices.json')}
        def versions():return tuple((n,id(t),t._version,str(t.dtype),str(t.device),t.requires_grad) for n,t in (*model.named_parameters(),*model.named_buffers()))
        initial_versions=versions();modes=tuple((n,id(m),m.training) for n,m in model.named_modules());precision=precision_contract()
        def unchanged():
            if (versions()!=initial_versions or tuple((n,id(m),m.training) for n,m in model.named_modules())!=modes
                    or precision_contract()!=precision or tensor_signature(targets)!=target or any(p.grad is not None for p in model.parameters())):
                raise ValueError('Original model/targets/modes/flags or absent parameter gradients changed.')
        route=PredictorRoute(model,'original',tiny=args.tiny)
        with route:
            route.phase='EOT8-before';route.check();unchanged()
            observer=NativeSDPAObserver(route,BoundaryStore(folder/'boundary'),control_sha256=args.control_sha256)
            x=clean.detach().clone().requires_grad_(True)
            with observer:
                loss=native_objective(model,targets,banks)(x).sum()
                state['artifacts']['loss']=save_return(folder/'loss.pt',loss,'loss',clean.shape);publish()
                task=route.prepare_backward(loss,x);gradient,=task.run()
                state['artifacts']['input']=save_return(folder/'input.pt',gradient,'input',clean.shape);publish()
            route.check();unchanged()
        guard.check();unchanged()
        if state_digest(model)!=weights:raise ValueError('Original model bytes changed.')
        state['full_native_checks']={}
        for n,t in (('loss',loss),('input',gradient)):
            state['full_native_checks'][n]=compare(t.detach().cpu(),expected[n],{'rtol':1e-5,'atol':2e-6} if n=='loss' else {'rtol':.002,'atol':2e-7})
            if tensor_info(t)['sha256']!=anchors[n]:raise ValueError('Native loss/input did not replay original logical bytes exactly.')
        cap=load(folder/'boundary',control_sha256=args.control_sha256)
        if args.device=='cuda':
            peak=torch.cuda.max_memory_allocated()
            if not 0<peak<=7*2**30:raise ValueError('Native capture allocator ceiling exceeded.')
            state['peak_allocator_bytes']=peak
        state.update(complete=True,original_value_replay_exact=True,boundary_capture_complete=True,
            boundary_header_sha256=digest(folder/'boundary/boundary.json'),scope_check=cap['scope_check'],
            graph_scope=cap['backward_scopes'][0],model_unchanged=True,runtime=runtime_contract(args.device,model=model))
    except BaseException as e:state.update(error_type=type(e).__name__,error=str(e)[:2048]);raise
    finally:
        primary=sys.exception();state['route_restored']=route is not None and not route.active
        state['observer_restored']=observer is not None and observer.state['observer_restored'];state['wall_seconds']=time.monotonic()-start
        try:
            publish();header={'file':'worker.json','bytes':(folder/'worker.json').stat().st_size,'sha256':digest(folder/'worker.json')}
            write_json(folder/('capture-complete.json' if state['complete'] else 'capture-failure.json'),
                {'kind':state['kind'],'control_sha256':args.control_sha256,'header':header,'complete':state['complete']})
        except BaseException as e:
            if primary is None:raise
            primary.add_note('Capture final-publication error also retained: '+repr(e))
    return state


def component_plan(overlay,capture,components,*,control_sha256,local_tiny):
    from models import predictor_sdpa_component_v1 as core
    from models.predictor_sdpa_component_records_v1 import sha
    overlay=Path(overlay);capture=Path(capture);w=json.loads((capture/'worker.json').read_text())
    p={'kind':core.PLAN_KIND,'local_tiny':local_tiny,'repository':str(overlay.resolve()),'capture_root':str(capture.resolve()),
        'native_worker_sha256':sha(capture/'worker.json'),'boundary_sha256':w['boundary_header_sha256'],'capture_control_sha256':control_sha256,
        'full_native_anchor_logical_sha256':w['full_native_anchor_logical_sha256'],'reference_dir':str((Path(components)/'reference').resolve()),
        'cells':[list(x) for x in core.CELLS],'bounds':core.BOUNDS,'bookkeeping':core.BOOK,'reference_builder_calls':1,
        'worker_seconds':120,'untimed_iterations':1,'profiled_iterations':1,'research_admitted':False,'performance_qualified':False,
        'full_model_candidate_adopted':False,'pure_backward_arithmetic_isolated':False,'paid_compute':False,'automatic_retry':False,'private_backward_API':False,
        'source_sha256':{n:sha(overlay/n) for n in core.SOURCE_NAMES}}
    core.validate_plan(p);core.capture(p);return p


def run_component(args):
    if args.device!='cuda' or platform.node()!='linux10603' or platform.python_version()!='3.11.17':raise ValueError('Exact authorized target component entry required.')
    from models import predictor_sdpa_component_v1 as core
    from models import predictor_sdpa_component_records_v1 as rec
    p=core.validate_plan(core.read(args.plan))
    if rec.sha(args.plan)!=args.plan_sha256 or p['local_tiny'] is not False or args.index not in range(4):raise ValueError('Bound target component plan changed.')
    parent,source,cap,identity=core.capture(p);prior=core.previous(p,args.plan_sha256,identity,args.index)
    ref=Path(p['reference_dir'])
    if args.index:
        core.load_reference(ref,p,args.plan_sha256,identity)
        first=core.load_worker(ref.parent/'0',p,args.plan_sha256,identity,0)
        if first['reference_manifest_sha256']!=rec.sha(ref/'reference.json'):raise ValueError('Original canonical reference changed.')
    elif ref.exists():raise FileExistsError('Never rebuild the canonical reference.')
    if Path(args.output)!=ref.parent/str(args.index):raise ValueError('Exact component output required.')
    import torch,numpy
    from models.execution import configure_execution
    configure_execution('shared-fp64-native',device='cuda')
    if torch.__version__!='2.8.0+cu128' or torch.version.cuda!='12.8' or numpy.__version__!='2.4.6' or not torch.cuda.is_available():raise ValueError('Installed target runtime changed.')
    torch.cuda.set_per_process_memory_fraction(7*2**30/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats()
    start=time.monotonic();w=core._execute(args,p,parent,source,cap,identity,prior)
    peak=torch.cuda.max_memory_allocated()
    if not 0<peak<=7*2**30 or time.time()>args.deadline_unix:raise ValueError('Component resource limit rejected after returns; preserve evidence.')
    from models.cpu_oracle_scratch import write_json
    write_json(Path(args.output)/'component-resource.json',{'kind':'native-fixed-SDPA-component-resource-v1','plan_sha256':args.plan_sha256,
        'worker_sha256':rec.sha(Path(args.output)/'worker.json'),'pid':os.getpid(),'index':args.index,
        'peak_allocator_bytes':peak,'post_entry_wall_seconds':time.monotonic()-start,'research_admitted':False,'performance_qualified':False})
    return w
