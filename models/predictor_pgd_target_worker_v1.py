"""One new pretrained PGD2/R2 diagnostic, eight owned input gradients maximum."""
import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
from models import predictor_pgd_target_records_v1 as records


def run(args):
    if (platform.node().split('.')[0] != 'linux10603' or platform.system() != 'Linux'
            or platform.python_version() != '3.11.17' or args.device != 'cuda' or args.tiny is not False):
        raise ValueError('Only the separately bound pretrained linux10603 CUDA attempt; no local analogue.')
    import numpy as np
    import torch
    from attacks.pgd import native_objective
    from attacks.pgd_pretrained_v1 import pgd_linf_pretrained_v1, target_domain, START_SEED
    from models.candidate_evaluation import EvaluationSettings, seeds_for
    from models.cuda_profile import tensor_signature
    from models.diagnostics import state_digest
    from models.execution import ExecutionGuard, configure_execution, apply_model_policy, precision_contract
    from models.training import runtime_contract
    from models import predictor_eot8_owned_v1 as native
    from models import predictor_forward_preserving_target_v1 as target
    from models import predictor_eot8_adapter_v1 as adapter
    if torch.__version__ != '2.8.0+cu128' or torch.version.cuda != '12.8' or np.__version__ != '2.4.6' or not torch.cuda.is_available():
        raise ValueError('Exact runtime required; no installation/fallback.')
    proposal = json.loads(Path(args.proposal).read_bytes())
    folder = Path(args.output);folder.mkdir(mode=0o700,exist_ok=False)
    started = time.monotonic()
    state = {'kind':records.KIND,'algorithm':records.ALGORITHM,'execution_host':'linux10603',
             'control_sha256':args.control_sha256,'pid':os.getpid(),'complete':False,'passed':False,
             'research_admitted':False,'performance_qualified':False,'fresh_rollout_replay_qualified':False,
             'gradient_objectives_started':0,'candidate_points':0,'no_grad_calls':[],'artifacts':{},'numerical_bounds':records.BOUNDS}
    def progress():
        records.save_json(folder,'worker.json',state,replace=(folder/'worker.json').exists())
    progress()
    try:
        configure_execution('shared-fp64-native',device='cuda');torch.set_num_threads(4);torch.manual_seed(20261029)
        torch.cuda.set_per_process_memory_fraction(7*2**30/torch.cuda.get_device_properties(0).total_memory)
        from models.eot_profile_packet import verify
        from models.reuse_diagnostic import upstream_gate
        from models.reuse_handoff import training_input
        from models.vjepa2 import build_vitl, load_checkpoint, sample_masks
        _,parent = verify(args.packet,'c95c0b21ee018dfd6c897756baf453dc7d69eb1d52b5c64d99e1ac4b4b945dbd')
        upstream_gate(args.upstream,parent)
        model = build_vitl(args.upstream,activation_checkpointing=True)
        checkpoint = load_checkpoint(model,args.checkpoint,'5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012')
        clean = training_input(args.packet,parent)
        apply_model_policy(model,'shared-fp64-native');model.to('cuda').eval();clean=clean.to('cuda')
        target_domain(clean,4/255,1/255,2,2,START_SEED,True)
        guard=ExecutionGuard(model,'cuda',target_versions=True)
        weights=state_digest(model);targets=model.clean_targets(clean)
        video=parent['training_row']['video_id'];seeds=seeds_for(EvaluationSettings.for_method('primary',1.),video)
        if seeds!=proposal['workload']['seeds'] or seeds['start']!=START_SEED:
            raise ValueError('Declared unchanged primary seed schema required.')
        masks_for=lambda seed:[p.to('cuda') for p in sample_masks(args.upstream,batch_size=1,seed=seed % (2**31-1))]
        banks=[masks_for(seed) for seed in seeds['attack']];score_banks=[masks_for(seed) for seed in seeds['score']]
        packed=torch.stack([torch.stack([torch.cat((p.context,p.target),dim=1) for p in draw]) for draw in banks])
        score_packed=torch.stack([torch.stack([torch.cat((p.context,p.target),dim=1) for p in draw]) for draw in score_banks])
        masks=[[[tensor_signature(p.context),tensor_signature(p.target)] for p in draw] for draw in banks]
        baseline={'weights_sha256':weights,'targets':tensor_signature(targets),'pixels':tensor_signature(clean),
                  'seeds':seeds['attack'],'masks':masks,'runtime':runtime_contract('cuda',model=model),
                  'data':{'kind':'original-training','video_id':video,'checkpoint_sha256':checkpoint['sha256']}}
        expected=proposal['reference']
        for key,alias in [('weights_sha256','weights_sha256'),('targets','clean_targets'),('pixels','clean_pixels'),('seeds','attack_seeds'),('masks','full_mask_tensor_identities')]:
            if baseline[key]!=expected[alias]:raise ValueError('Frozen clean baseline changed: '+key)
        if baseline['runtime']!=json.loads((Path(args.packet)/'input/prior-contract.json').read_bytes())['runtime']:
            raise ValueError('Original runtime contract changed.')
        state['baseline']=baseline;state['baseline_sha256']=hashlib.sha256(records.frozen.atomic._json(baseline)).hexdigest()
        state['mask_context_lengths']=[[int(p.context.shape[1]) for p in draw] for draw in banks]
        state['score_mask_context_lengths']=[[int(p.context.shape[1]) for p in draw] for draw in score_banks]
        state['score_mask_signatures']=[[[tensor_signature(p.context),tensor_signature(p.target)] for p in draw] for draw in score_banks]
        precision=precision_contract();versions=[(p,p._version,p.requires_grad) for p in model.parameters()]
        modes=[(m,m.training) for m in model.modules()]
        fixed=[(t,t._version,tensor_signature(t)) for t in (clean,targets,packed,score_packed)]
        fixed += [(t,t._version,tensor_signature(t)) for bank in (banks,score_banks) for draw in bank for p in draw for t in (p.context,p.target)]
        original_grad=torch.autograd.grad;original_sdpa=torch.nn.functional.scaled_dot_product_attention
        creator=(os.getpid(),threading.get_ident());active=[None];step_count=[0]
        def common():
            if ((os.getpid(),threading.get_ident())!=creator or torch.is_inference_mode_enabled()
                    or torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda') or precision_contract()!=precision
                    or torch.autograd.grad is not original_grad or torch.nn.functional.scaled_dot_product_attention is not original_sdpa
                    or [id(p) for p in model.parameters()]!=[id(p) for p,_,_ in versions]
                    or any(p._version!=v or p.requires_grad!=r or p.grad is not None for p,v,r in versions)
                    or any(m.training!=mode for m,mode in modes)
                    or any(t._version!=v or tensor_signature(t)!=h for t,v,h in fixed)):
                raise ValueError('Exact first-order modes/parameters/targets/masks/flags/owner changed.')
        def idle():
            common();guard.check()
            if (active[0] is not None or adapter._ACTIVE or target.cpu._ACTIVE or hasattr(model,target.MARKER)
                    or state_digest(model)!=weights or model.predictor._forward_hooks
                    or any('forward' in b.attn.__dict__ for blocks in (model.encoder.blocks,model.predictor.predictor_blocks) for b in blocks)):
                raise ValueError('Model/hash/route/private methods/hooks not restored.')
        def no_grad(value,bank,label):
            idle()
            if torch.is_grad_enabled():raise ValueError('Original no-grad scoring required outside custom contexts.')
            result=native_objective(model,targets,bank)(value)
            if result.shape!=(1,) or not torch.isfinite(result).all():raise ValueError('Finite original no-grad score required.')
            state['no_grad_calls'].append({'label':label,'draws':len(bank),'pixels':tensor_signature(value)['sha256'],
                                          'loss':result.detach().cpu().tolist(),'custom_route_active':False,'grad_enabled':False})
            idle();progress();return result
        class Capture:
            def __init__(self,value,step,candidate,reference=None):
                self.value,self.step,self.candidate,self.reference=value,step,candidate,reference
                self.used=self.entered=self.closed=self.considered=False
                self.index=2*step+int(candidate);self.handles=[];self.arrays={};self.route=None;self.profile=None
                self.folder=folder/('objective-'+str(self.index))
                self.record={'kind':records.OBJECTIVE,'index':self.index,'step':step,'mode':'candidate' if candidate else 'original',
                             'execution_host':'linux10603','local_tiny':False,'pid':os.getpid(),'control_sha256':args.control_sha256,
                             'baseline_sha256':state['baseline_sha256'],'mask_context_lengths':state['mask_context_lengths'],
                             'artifacts':{},'complete':False,'passed':False,'research_admitted':False,'numerical_bounds':records.BOUNDS,'events':[]}
            def save(self,name,tensor):
                self.arrays[name]=records.save_tensor(self.folder,self.record,name,tensor)
                records.save_json(self.folder,'worker.json',self.record,replace=(self.folder/'worker.json').exists())
            def __enter__(self):
                idle()
                if self.entered or self.closed or self.index!=state['gradient_objectives_started'] or self.index>=8:
                    raise ValueError('One fresh exact ordered loss/input scope; no retry.')
                self.entered=True;self.folder.mkdir(mode=0o700)
                state['gradient_objectives_started']+=1;progress();active[0]=self
                self.stack=ExitStack();self.input_version=self.value._version;self.input_hash=tensor_signature(self.value)
                try:
                    if self.step==0:
                        self.profile=self.stack.enter_context(torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False,profile_memory=False))
                    self.route=target.TargetRoute(model) if self.candidate else native.PredictorRoute(model,'original',tiny=False,max_calls=4096)
                    self.stack.enter_context(self.route)
                    def remove():
                        for h in self.handles:h.remove()
                    self.stack.callback(remove)
                    self.save('pixels',self.value);self.save('targets',targets);self.save('masks',packed)
                    predictions=[]
                    def observe(module,inputs,output):
                        i=len(predictions);predictions.append(i)
                        if i>=16:raise ValueError('At most 16 original predictor outputs.')
                        self.save('prediction-'+str(i),output)
                        def incoming(g):
                            if 'cotangent-'+str(i) in self.arrays:raise ValueError('One actual native L1 cotangent only.')
                            self.save('cotangent-'+str(i),g);return None
                        self.handles.append(output.register_hook(incoming))
                    self.handles.append(model.predictor.register_forward_hook(observe))
                    if self.candidate:self.root=self.route.loss(self.value,targets,banks)
                    else:
                        self.route.phase='EOT8-before';self.root=native_objective(model,targets,banks)(self.value).sum()
                    self.losses=self.root.view(1);self.token=(self.root,self.value,self.losses,self.root._version)
                    self.save('loss',self.root);self.record['events'].append('forward');return self
                except BaseException:
                    self.__exit__(*sys.exc_info());raise
            def admitted(self):
                common()
                if (not self.entered or self.closed or self.used or active[0] is not self or self.root is not self.token[0]
                        or self.value is not self.token[1] or self.losses is not self.token[2] or self.root._version!=self.token[3]
                        or self.value._version!=self.input_version or tensor_signature(self.value)!=self.input_hash
                        or self.value.grad is not None or not torch.is_grad_enabled() or torch._C._current_graph_task_id()!=-1):
                    raise ValueError('One exact fresh first-order native loss/input token required.')
            def note_consider(self,value,losses):
                self.admitted()
                if not self.candidate or self.considered or value is not self.value or losses is not self.losses:
                    raise ValueError('Consider exact candidate loss before backward, once.')
                self.considered=True;self.record['events'].append('consider')
            def gradient(self,*,create_graph=False):
                self.admitted()
                if create_graph is not False or self.candidate and not self.considered:raise ValueError('First-order only; selection before backward required.')
                self.used=True;self.record['events'].append('backward')
                if self.candidate:gradient=self.route.gradient(self.root,self.value)
                else:gradient,=self.route.prepare_backward(self.root,self.value).run()
                self.save('input',gradient)
                if len(self.arrays)!=37 or not self.arrays['input'].any():raise ValueError('All actual returns and nonzero full gradient required.')
                if self.reference is not None:self.record['comparison']=records.compare(self.arrays,self.reference.arrays)
                return gradient
            def __exit__(self,kind,error,tb):
                failed=error is not None
                try:
                    self.stack.__exit__(kind,error,tb);self.record['events'].append('exit');active[0]=None;idle()
                    if self.value._version!=self.input_version or tensor_signature(self.value)!=self.input_hash or self.value.grad is not None:
                        raise ValueError('Input/version changed or input .grad populated.')
                    if not failed:
                        if not self.used or self.record['events']!=(['forward','consider','backward','exit'] if self.candidate else ['forward','backward','exit']):
                            raise ValueError('Exact consider-before-backward/exit ordering required.')
                        n=self.route.native if self.candidate else self.route
                        self.record['ownership']=records.check_scope(self.record,n.rows,n.scopes,n.refusals,
                            self.route.prototype.calls if self.candidate else [],self.route.prototype.scope if self.candidate else None)
                        if self.profile is not None:
                            observed=records.profile_summary(self.profile,self.record['mode'])
                            records.save_json(self.folder,'backend.json',observed,cap=4*2**20)
                            self.record['backend_sha256']=records.frozen.sha_file(self.folder/'backend.json')
                            records.validate_backend(observed,self.record['mode'])
                        self.record.update(complete=True,passed=True,model_unchanged=True,adapter_restored=True,
                            parameter_gradients_absent=True,input_dot_grad_absent=True)
                except BaseException as cleanup:
                    failed=True;self.record.update(error_type=type(cleanup).__name__,error=str(cleanup)[:2048]);raise
                finally:
                    active[0]=None;self.closed=True
                    if failed:self.record.update(complete=False,passed=False,error_type=self.record.get('error_type',getattr(kind,'__name__',None)),error=self.record.get('error',str(error)[:2048]))
                    n=None if self.route is None else self.route.native if self.candidate else self.route
                    records.write_final(self.folder,self.record,calls=[] if n is None else n.rows,scopes=[] if n is None else n.scopes,refusals=[] if n is None else n.refusals,
                        vjps=[] if self.route is None or not self.candidate else self.route.prototype.calls,borrowed=None if self.route is None or not self.candidate else self.route.prototype.scope)
        def context(value):
            idle()
            if (step_count[0]>=4 or not value.is_leaf or not value.requires_grad or value.grad is not None
                    or value.device!=clean.device or value.dtype!=clean.dtype or value.shape!=clean.shape
                    or not torch.isfinite(value).all() or not torch.all((value>=(clean-4/255).clamp(0,1))&(value<=(clean+4/255).clamp(0,1)))):
                raise ValueError('Four exact fresh projected candidate leaves only.')
            step=step_count[0];step_count[0]+=1;state['candidate_points']=step_count[0];progress()
            reference=Capture(value.detach().clone().requires_grad_(True),step,False)
            with reference:reference.gradient()
            return Capture(value,step,True,reference)
        idle();torch.cuda.reset_peak_memory_stats();rng_cpu=torch.get_rng_state().clone();rng_cuda=torch.cuda.get_rng_state().clone()
        with torch.no_grad():
            clean_attack=no_grad(clean,banks,'attack');clean_score=no_grad(clean,score_banks,'score-clean')
        result=pgd_linf_pretrained_v1(lambda value:no_grad(value,banks,'attack'),clean,epsilon=4/255,step_size=1/255,steps=2,restarts=2,seed=seeds['start'],random_start=True,gradient_context=context)
        with torch.no_grad():score=no_grad(result.adversarial,score_banks,'score-adversarial')
        idle()
        if not torch.equal(torch.get_rng_state(),rng_cpu) or not torch.equal(torch.cuda.get_rng_state(),rng_cuda):raise ValueError('Global RNG changed.')
        generator=torch.Generator(device='cpu').manual_seed(seeds['start'])
        output={'clean':clean,'adversarial':result.adversarial,'attack-loss':result.losses,'clean-attack-loss':clean_attack,'clean-score':clean_score,'adversarial-score':score,'scoring-masks':score_packed}
        for i in range(2):output['restart-noise-'+str(i)]=torch.rand(clean.shape,generator=generator,dtype=torch.float32)
        for name,tensor in output.items():records.save_tensor(folder,state,name,tensor);progress()
        peak=torch.cuda.max_memory_allocated();state['peak_allocator_bytes']=peak
        if state['gradient_objectives_started']!=8 or step_count[0]!=4 or not 0<peak<=7*2**30 or len(state['no_grad_calls'])!=6:
            raise ValueError('Original objective/allocator/no-grad limits changed.')
        state.update(complete=True,passed=True,model_unchanged=True,global_rng_unchanged=True,all_methods_hooks_flags_restored=True)
    except BaseException as error:
        state.update(complete=False,passed=False,error_type=type(error).__name__,error=str(error)[:2048]);raise
    finally:
        state['wall_seconds']=time.monotonic()-started;progress()
    return state
