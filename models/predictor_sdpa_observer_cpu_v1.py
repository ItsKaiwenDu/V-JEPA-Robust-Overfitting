"""One original native EOT8 local CPU rehearsal; no university/target entry."""
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time

INITIAL_SHA='989665a5a4f8ea90600214be5536b4600c37fc448108a5bd177b3416e5ae0338'


def run(args):
    if platform.node() in ('linux10603','linux10609') or args.device!='cpu' or args.runner not in ('creator','other-thread'):
        raise ValueError('Local native CPU observer rehearsal only; no target/CUDA grant.')
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
    from models.predictor_eot8_capture_v1 import save_return
    from models.predictor_sdpa_observer_v1 import NativeSDPAObserver
    from models.predictor_sdpa_capture_record_v1 import BoundaryStore,load
    folder=Path(args.output);folder.mkdir(mode=0o700,exist_ok=False);(folder/'boundary').mkdir(mode=0o700)
    started=time.monotonic();route=observer=None
    state={'kind':'original-SDPA-observer-local-CPU-rehearsal-v1','local_tiny':True,'device':'cpu','runner':args.runner,
        'complete':False,'original_value_replay_exact':False,'pid':os.getpid(),'artifacts':{},'research_admitted':False,
        'CUDA_kernel_qualified':False,'performance_qualified':False,'native_candidate_adopted':False}
    def publish():write_json(folder/'worker.json',state,replace=(folder/'worker.json').exists())
    publish()
    try:
        reference=Path(args.reference);prior=json.loads((reference/'worker.json').read_text())
        expected={}
        for name in ('loss','input'):
            item=prior['artifacts']['EOT8-before-'+name];path=reference/item['file']
            if digest(path)!=item['sha256']:raise ValueError('Frozen CPU native reference changed.')
            expected[name]=torch.load(path,weights_only=True,map_location='cpu')
        configure_execution('shared-fp64-native',device='cpu');torch.set_num_threads(4);torch.manual_seed(20261029)
        model,clean,masks_for=tiny_fixture(args.upstream)
        initial=load_fixture_state(model,args.initial,INITIAL_SHA)
        apply_model_policy(model,'shared-fp64-native');model.to('cpu').eval();guard=ExecutionGuard(model,'cpu',target_versions=True)
        weights=state_digest(model);targets=model.clean_targets(clean);target=tensor_signature(targets)
        seeds=seeds_for(EvaluationSettings.for_method('primary',2.),'tiny-predictor-eot8')['attack']
        banks=[[p.to('cpu') for p in masks_for(seed%(2**31-1))] for seed in seeds]
        masks=[[[tensor_signature(p.context),tensor_signature(p.target)] for p in d] for d in banks]
        baseline={'weights_sha256':weights,'targets':target,'pixels':tensor_signature(clean),'seeds':seeds,
            'masks':masks,'runtime':runtime_contract('cpu',model=model),'data':{'kind':'local-tiny-CPU','initial':initial}}
        if baseline!=prior['baseline']:raise ValueError('Original CPU baseline identity changed before observation.')
        state['baseline']=baseline;publish()
        def versions():return tuple((n,id(t),t._version,str(t.dtype),str(t.device),t.requires_grad) for n,t in (*model.named_parameters(),*model.named_buffers()))
        before=versions();modes=tuple((n,id(m),m.training) for n,m in model.named_modules());precision=precision_contract()
        def unchanged():
            if (versions()!=before or tuple((n,id(m),m.training) for n,m in model.named_modules())!=modes
                    or precision_contract()!=precision or tensor_signature(targets)!=target
                    or any(p.grad is not None for p in model.parameters())):raise ValueError('Native model/targets/flags or parameter gradients changed.')
        # Fresh native controls surround observation; all are local small-model work.
        def unobserved(label):
            x=clean.detach().clone().requires_grad_(True);loss=native_objective(model,targets,banks)(x).sum()
            g,=torch.autograd.grad(loss,x)
            for kind,t in (('loss',loss),('input',g)):
                state['artifacts'][label+'-'+kind]=save_return(folder/(label+'-'+kind+'.pt'),t,kind,clean.shape)
                if not torch.equal(t.detach(),expected[kind]):raise ValueError('Fresh unobserved native CPU value does not exactly replay frozen control.')
            unchanged();publish()
        unobserved('before')
        route=PredictorRoute(model,'original',tiny=True)
        with route:
            route.phase='EOT8-before';route.check();unchanged()
            store=BoundaryStore(folder/'boundary');observer=NativeSDPAObserver(route,store,control_sha256='d'*64)
            value=clean.detach().clone().requires_grad_(True)
            with observer:
                loss=native_objective(model,targets,banks)(value).sum()
                state['artifacts']['loss']=save_return(folder/'loss.pt',loss,'loss',clean.shape);publish()
                task=route.prepare_backward(loss,value)
                if args.runner=='creator':gradient,=task.run()
                else:
                    result=[];errors=[]
                    def body():
                        try:result.append(task.run())
                        except BaseException as e:errors.append(e)
                    worker=threading.Thread(target=body);worker.start();worker.join(30)
                    if worker.is_alive():raise RuntimeError('Local CPU native task did not unwind.')
                    if errors:raise errors[0]
                    gradient,=result[0]
                state['artifacts']['input']=save_return(folder/'input.pt',gradient,'input',clean.shape);publish()
            route.check();unchanged()
        guard.check();unchanged()
        if state_digest(model)!=weights:raise ValueError('Native model bytes changed.')
        if not torch.equal(loss.detach(),expected['loss']) or not torch.equal(gradient.detach(),expected['input']):
            raise ValueError('Observed native CPU loss/input do not exactly reproduce frozen values.')
        unobserved('after')
        checked=load(folder/'boundary',control_sha256='d'*64)
        state.update(complete=True,original_value_replay_exact=True,boundary_capture_complete=checked['capture_complete'],
            boundary_header_sha256=digest(folder/'boundary/boundary.json'),model_unchanged=True,
            hook_threads=sorted({r['thread'] for r in checked['gradient_observations'].values()}),
            graph_scope=checked['backward_scopes'][0],scope_check=checked['scope_check'],runtime=runtime_contract('cpu',model=model))
    except BaseException as e:state.update(error_type=type(e).__name__,error=str(e)[:2048]);raise
    finally:
        state['route_restored']=route is not None and not route.active
        state['observer_restored']=observer is not None and observer.state['observer_restored']
        state['wall_seconds']=time.monotonic()-started
        publish()
    return state
