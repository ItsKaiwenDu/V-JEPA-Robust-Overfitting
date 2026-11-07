"""One native EOT8 pass with actual predictor/cotangent observation; no profiling."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from models.predictor_boundary_contract_v1 import WORKER_KIND,CONTRACT_SHA
from models.predictor_boundary_record_v1 import BoundaryRecord,verify_scope,load

BOUNDS={'loss':{'rtol':1e-5,'atol':2e-6},'input':{'rtol':.002,'atol':2e-7}}


def run(args):
    if type(args.tiny) is not bool or type(args.index) is not int or args.index not in (0,1):raise ValueError('Exact boundary domain/index required.')
    if args.tiny:
        if platform.node() in ('linux10603','linux10609') or args.device!='cpu':raise ValueError('Local CPU analogue cannot use university/CUDA.')
    elif platform.node()!='linux10603' or platform.python_version()!='3.11.17' or args.device!='cuda':raise ValueError('Exact live host/Python/CUDA required before scientific imports.')
    import torch
    from attacks.pgd import native_objective
    from models.candidate_evaluation import EvaluationSettings,seeds_for
    from models.cuda_profile import tensor_signature
    from models.diagnostics import state_digest
    from models.execution import ExecutionGuard,configure_execution,apply_model_policy,precision_contract
    from models.fixture_state import load_fixture_state
    from models.native_diagnostic import tiny_fixture
    from models.eot_profile_packet import verify as verify_packet
    from models.reuse_diagnostic import upstream_gate
    from models.reuse_handoff import training_input
    from models.training import runtime_contract
    from models.vjepa2 import build_vitl,load_checkpoint,sample_masks
    from models.cpu_oracle_scratch import digest,write_json
    from models.predictor_eot8_owned_v1 import PredictorRoute
    from models.predictor_eot8_capture_v1 import save_return,compare
    from models.predictor_boundary_observer_v1 import BoundaryObserver
    folder=Path(args.output);folder.mkdir(mode=0o700,exist_ok=False)
    started=time.monotonic();mode='original' if args.index==0 else 'predictor-math'
    state={'kind':WORKER_KIND,'execution_host':platform.node(),'local_tiny':args.tiny,'mode':mode,'index':args.index,
        'pid':os.getpid(),'control_sha256':args.control_sha256,'capture_complete':False,'numerical_passed':False,
        'reproduces_anchor':False,'passed':False,'artifacts':{},'observations':[],
        'backend_kernel_qualified':False,'full_five_phase_qualified':False,'research_admitted':False,'performance_qualified':False}
    store=BoundaryRecord(folder);route=observer=None
    def record(*,final=False):return store.write(state,route,final=final)
    record()
    try:
        meta=json.loads((Path(args.anchors)/'ANCHORS.json').read_text())
        if meta.get('kind')!='predictor-boundary-value-anchors-v1' or meta.get('local_tiny') is not args.tiny or meta.get('contract_sha256')!=CONTRACT_SHA:
            raise ValueError('Exact value-only anchor domain required.')
        if not args.tiny:
            contract=json.loads(Path(args.contract).read_text())
            if hashlib.sha256(Path(args.contract).read_bytes()).hexdigest()!=CONTRACT_SHA or meta['anchors']!=contract['anchors'] or meta['baseline']!=contract['approved_target_source_baseline']:raise ValueError('Frozen value-anchor/source binding changed.')
        else:contract=None
        anchors=[]
        for i in (0,1):
            anchor=meta['anchors'][str(i)]
            if anchor.get('value_only_not_executable_reference') is not True:raise ValueError('Anchors are values, not passing worker references.')
            if not args.tiny and (anchor['source_worker_passed'] is not (i==0) or anchor['source_worker_complete'] is not (i==0)):
                raise ValueError('Failed candidate may not be relabelled as a passing reference.')
            values={}
            for name,item in anchor['artifacts'].items():
                p=Path(args.anchors)/item['file']
                if p.is_symlink() or digest(p)!=item['sha256'] or p.stat().st_size!=item['bytes']:raise ValueError('Value anchor changed.')
                values[name]=torch.load(p,weights_only=True,map_location='cpu')
                if values[name].dtype!=torch.float32 or not torch.isfinite(values[name]).all():raise ValueError('Actual FP32 anchor required.')
            if set(values)!={'loss','input'}:raise ValueError('Both actual anchor returns required.')
            anchors.append(values)
        if args.index==1:
            previous=load(args.reference,control_sha256=args.control_sha256)
            if previous['index']!=0 or previous['local_tiny'] is not args.tiny or previous['execution_host']!=state['execution_host']:
                raise ValueError('New passing same-host original capture required before candidate model work.')
        configure_execution('shared-fp64-native',device=args.device);torch.set_num_threads(4);torch.manual_seed(20261029)
        if args.tiny:
            model,clean,masks_for=tiny_fixture(args.upstream)
            initial=load_fixture_state(model,args.initial,args.initial_sha)
            data={'kind':'local-tiny-CPU','initial':initial};video='tiny-predictor-eot8'
        else:
            if (platform.node()!='linux10603' or torch.__version__!='2.8.0+cu128' or platform.python_version()!='3.11.17'):
                raise ValueError('Live host/Python/Torch changed; no installation/fallback.')
            import numpy
            if numpy.__version__!='2.4.6' or torch.version.cuda!='12.8' or not torch.cuda.is_available():raise ValueError('Live NumPy/CUDA runtime changed.')
            torch.cuda.set_per_process_memory_fraction(7*2**30/torch.cuda.get_device_properties(0).total_memory)
            metadata,parent=verify_packet(args.packet,'c95c0b21ee018dfd6c897756baf453dc7d69eb1d52b5c64d99e1ac4b4b945dbd')
            upstream_gate(args.upstream,parent)
            model=build_vitl(args.upstream,activation_checkpointing=True)
            checkpoint=load_checkpoint(model,args.checkpoint,'5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012')
            clean=training_input(args.packet,parent);masks_for=lambda seed:sample_masks(args.upstream,batch_size=1,seed=seed)
            video=parent['training_row']['video_id'];data={'kind':'original-training','video_id':video,'checkpoint_sha256':checkpoint['sha256']}
        apply_model_policy(model,'shared-fp64-native');model.to(args.device).eval();clean=clean.to(args.device)
        guard=ExecutionGuard(model,args.device,target_versions=True);weights=state_digest(model)
        targets=model.clean_targets(clean);target=tensor_signature(targets)
        seeds=seeds_for(EvaluationSettings.for_method('primary',2.),video)['attack']
        banks=[[m.to(args.device) for m in masks_for(seed%(2**31-1))] for seed in seeds]
        masks=[[[tensor_signature(m.context),tensor_signature(m.target)] for m in draw] for draw in banks]
        if len(banks)!=8 or any(len(draw)!=2 for draw in banks):raise ValueError('EOT8/two-family scope changed.')
        baseline={'weights_sha256':weights,'targets':target,'pixels':tensor_signature(clean),'seeds':seeds,
                  'masks':masks,'runtime':runtime_contract(args.device,model=model),'data':data}
        if not args.tiny:
            expected=json.loads(Path(args.proposal).read_text())['reference']
            if any(baseline[k]!=expected[v] for k,v in [('weights_sha256','weights_sha256'),('targets','clean_targets'),('pixels','clean_pixels'),('seeds','attack_seeds'),('masks','full_mask_tensor_identities')]):
                raise ValueError('Original weights/pixels/targets/masks changed.')
            prior=torch.load(Path(args.packet)/'input/prior-EOT8-derivatives.pt',weights_only=True,map_location='cpu')
            old=json.loads((Path(args.packet)/'input/prior-contract.json').read_text())
            if baseline['runtime']!=old['runtime']:raise ValueError('Original full-model runtime changed.')
        else:prior=None
        state['baseline']=baseline;record()
        if baseline!=meta['baseline']:raise ValueError('Value-anchor weights/pixels/targets/masks/runtime changed.')
        if args.tiny:prior=anchors[0]
        state['artifacts']['targets']=save_return(folder/'targets.pt',targets,'input',targets.shape)
        indices=[[{'context':pair.context.detach().cpu().tolist(),'target':pair.target.detach().cpu().tolist()} for pair in draw] for draw in banks]
        write_json(folder/'mask-indices.json',indices)
        state['mask_indices']={'file':'mask-indices.json','bytes':(folder/'mask-indices.json').stat().st_size,'sha256':digest(folder/'mask-indices.json')}
        def versions():return tuple((n,id(t),t._version,str(t.dtype),str(t.device),t.requires_grad) for n,t in (*model.named_parameters(),*model.named_buffers()))
        initial_versions=versions();modules=tuple((n,id(m),m.training) for n,m in model.named_modules());precision=precision_contract()
        def frozen():
            if (versions()!=initial_versions or tuple((n,id(m),m.training) for n,m in model.named_modules())!=modules
                    or precision_contract()!=precision or tensor_signature(targets)!=target or any(p.grad is not None for p in model.parameters())):
                raise ValueError('Original model/target/runtime or parameter gradients changed.')
        route=PredictorRoute(model,mode,tiny=args.tiny)
        with route:
            route.phase='EOT8-before';route.check();frozen()
            if mode=='predictor-math':
                try:guard.check()
                except ValueError:state['existing_study_guard_rejected_candidate']=True
                else:raise ValueError('Existing study guard admitted unsupported candidate.')
            observer=BoundaryObserver(model,banks,folder,tiny=args.tiny,publish=record)
            state['observations']=observer.rows
            x=clean.detach().clone().requires_grad_(True)
            objective=native_objective(model,targets,banks)
            if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
            with observer:
                loss=objective(x).sum()
                state['artifacts']['loss']=save_return(folder/'loss.pt',loss,'loss',clean.shape);record()
                gradient,=route.input_gradient(loss,x)
                state['artifacts']['input']=save_return(folder/'input.pt',gradient,'input',clean.shape);record()
            route.check();frozen()
        guard.check();frozen()
        if state_digest(model)!=weights:raise ValueError('Final model values changed.')
        state.update(observer_restored=observer.restored,adapter_restored=not route.active,model_unchanged=True)
        state['scope_check']=verify_scope(state,route.rows,route.scopes,route.refusals)
        if len(state['observations'])!=16 or any(r['cotangent'] is None or not r['hook_removed'] for r in state['observations']):raise ValueError('Incomplete actual native boundary capture.')
        state['capture_complete']=True
        values={'loss':loss.detach().cpu(),'input':gradient.detach().cpu()}
        # This exact check is descriptive reproduction, not acceptance of failed data.
        if any(not torch.equal(value,anchors[args.index][name]) for name,value in values.items()):raise ValueError('Observer did not exactly reproduce the preserved same-arm values.')
        state['reproduces_anchor']=True
        if args.device=='cuda':
            peak=torch.cuda.max_memory_allocated();state['peak_allocator_bytes']=peak
            if not 0<peak<=7*2**30:raise ValueError('Original allocator cap exceeded.')
        state['numerical_checks']=[];record()
        for name,value in values.items():
            comparison=compare(value,prior[name],BOUNDS[name]);state['numerical_checks'].append({'against':'historical-original','kind':name,**comparison});record()
        if args.index==1:
            previous=load(args.reference,control_sha256=args.control_sha256)
            if previous['baseline']!=baseline:raise ValueError('Fresh original baseline identity changed.')
            for name,value in values.items():
                item=previous['artifacts'][name];p=Path(args.reference)/item['file']
                if digest(p)!=item['sha256']:raise ValueError('Fresh original value changed.')
                comparison=compare(value,torch.load(p,weights_only=True,map_location='cpu'),BOUNDS[name]);state['numerical_checks'].append({'against':'fresh-original','kind':name,**comparison});record()
        state.update(numerical_passed=True,passed=True)
    except BaseException as error:
        state.update(passed=False,error_type=type(error).__name__,error=str(error)[:2048]);raise
    finally:
        original_error=sys.exception()
        state['adapter_restored']=route is not None and not route.active
        state['observer_restored']=observer is not None and observer.restored
        state['wall_seconds']=time.monotonic()-started
        try:record(final=True)
        except BaseException as error:
            state['passed']=False
            try:store.failure(state,error)
            except BaseException:print('Boundary evidence failure marker failed.',file=sys.stderr,flush=True)
            if original_error is None:raise
    return state
