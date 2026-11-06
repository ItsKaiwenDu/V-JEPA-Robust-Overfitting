"""Real upstream checkpoint tests across Python threads; no CUDA qualification."""
from pathlib import Path
import threading

import pytest
import torch

from models import predictor_eot8_adapter_v1 as frozen
from models.predictor_eot8_owned_v1 import PredictorRoute
from models.native_diagnostic import tiny_fixture
from models.execution import configure_execution, apply_model_policy, ExecutionGuard
from models import predictor_eot8_events_v1 as events

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def fixture():
    configure_execution('shared-fp64-native',device='cpu')
    torch.set_num_threads(2);torch.manual_seed(17)
    model,clean,masks=tiny_fixture(ROOT/'external/vjepa2')
    apply_model_policy(model,'shared-fp64-native');model.eval()
    return model,clean,masks


def other_thread(fn):
    result=[];error=[]
    def body():
        try:result.append(fn())
        except BaseException as exc:error.append(exc)
    thread=threading.Thread(target=body)
    thread.start();thread.join(15)
    assert not thread.is_alive(),'CPU analogue timed out'
    if error:raise error[0]
    return result[0]


def objective(fixture,seed=19):
    model,clean,masks=fixture
    target=model.clean_targets(clean)
    value=clean.detach().clone().requires_grad_(True)
    return value,target,masks(seed)


def test_frozen_creator_thread_guard_reproduces_failure_on_real_checkpoint_backward(fixture):
    model,_,_=fixture;value,target,masks=objective(fixture)
    with frozen.PredictorRoute(model,'original',tiny=True) as route:
        route.phase='draw-before';loss=model.loss_per_example(value,target,masks).sum()
        route.backward_active=True
        with pytest.raises(ValueError,match='Unowned/concurrent'):
            other_thread(lambda:torch.autograd.grad(loss,value))
    assert not route.active and all('forward' not in b.attn.__dict__ for b in model.encoder.blocks)


@pytest.mark.parametrize('mode',['original','predictor-math'])
@pytest.mark.parametrize('runner',['creator','other-thread'])
def test_graph_owned_real_checkpoint_backward_matches_frozen_baseline(fixture,mode,runner,tmp_path):
    model,_,_=fixture;value,target,masks=objective(fixture)
    original=model.predictor.predictor_blocks[0].attn.forward.__func__
    baseline=model.loss_per_example(value,target,masks).sum();baseline_grad,=torch.autograd.grad(baseline,value)
    value=value.detach().clone().requires_grad_(True)
    guard=ExecutionGuard(model,'cpu',target_versions=True)
    with PredictorRoute(model,mode,tiny=True) as route:
        route.phase='draw-before'
        if mode=='predictor-math':
            with pytest.raises(ValueError):guard.check()
        loss=model.loss_per_example(value,target,masks).sum()
        task=route.prepare_backward(loss,value)
        gradient,=task.run() if runner=='creator' else other_thread(task.run)
        # Preserve actual returns before the comparison, including any failing values.
        from models.predictor_eot8_capture_v1 import save_return
        saved={name:save_return(tmp_path/(name+'.pt'),tensor,'loss' if 'loss' in name else 'input',value.shape)
               for name,tensor in [('actual-loss',loss),('actual-input',gradient),('reference-loss',baseline),('reference-input',baseline_grad)]}
        import json
        (tmp_path/'thread-evidence.json').write_text(json.dumps({'CPU_analogue_only':True,'mode':mode,'runner':runner,'scopes':route.scopes,'calls':route.rows,'refusals':route.refusals,'artifacts':saved},sort_keys=True)+'\n')
        torch.testing.assert_close(loss,baseline,rtol=1e-5,atol=2e-6)
        torch.testing.assert_close(gradient,baseline_grad,rtol=2e-3,atol=2e-7)
        scope=route.scopes[0];assert scope['state']=='complete' and scope['hook_removed']
        backward=[r for r in route.rows if r['during_backward']]
        assert backward and all(r['graph_id']==scope['graph_id']>=0 for r in backward)
        assert all(r['thread'] in scope['call_threads'] for r in backward)
        if runner=='other-thread':assert scope['runner_thread']!=route.thread and all(r['thread']!=route.thread for r in backward)
        assert not route.refusals and all(r['entry_flags']==r['exit_flags'] for r in route.rows)
        route.check()
    guard.check();assert model.predictor.predictor_blocks[0].attn.forward.__func__ is original
    assert all(p.grad is None for p in model.parameters()) and not route.active


def test_same_thread_profile_retains_original_owner_event_verifier(fixture,tmp_path):
    model,_,_=fixture;value,target,masks=objective(fixture)
    with PredictorRoute(model,'predictor-math',tiny=True) as route:
        route.phase='draw-profiled'
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            loss=model.loss_per_example(value,target,masks).sum()
            route.input_gradient(loss,value)
        events.save(profile,tmp_path/'trace.jsonl')
        checked=events.verify(events.read(tmp_path/'trace.jsonl'),route.rows,cuda=False,mode='predictor-math',owners=['encoder.0','encoder.1','predictor.0','predictor.1'])
        assert checked['CPU_analogue_only'] and not checked['CUDA_qualified']


@pytest.mark.parametrize('action,predicate',[
 ('unscoped-backward','unowned-forward-thread-or-graph'),
 ('foreign-forward','unowned-forward-thread-or-graph'),
 ('task-reuse','backward-task-not-single-use'),
 ('overlapping-task','creator-check-overlap'),
 ('prepared-forward','backward-not-running-or-phase-changed'),
 ('changed-phase','backward-phase-or-method-overlap'),
 ('changed-input','prepared-tensor-or-version-changed'),
])
def test_refusals_are_precise_and_restoration_survives(fixture,action,predicate):
    model,_,_=fixture;value,target,masks=objective(fixture)
    with PredictorRoute(model,'original',tiny=True) as route:
        route.phase='draw-before';loss=model.loss_per_example(value,target,masks).sum()
        if action=='unscoped-backward':fn=lambda:torch.autograd.grad(loss,value)
        elif action=='foreign-forward':fn=lambda:other_thread(lambda:model.loss_per_example(value,target,masks))
        else:
            task=route.prepare_backward(loss,value)
            if action=='task-reuse':task.run();fn=task.run
            if action=='overlapping-task':fn=lambda:route.prepare_backward(loss,value)
            if action=='prepared-forward':fn=lambda:model.loss_per_example(value,target,masks)
            if action=='changed-phase':route.phase='draw-after';fn=task.run
            if action=='changed-input':
                with torch.no_grad():value.add_(.001)
                fn=task.run
        with pytest.raises(ValueError,match=predicate):fn()
        last=route.refusals[-1]
        assert last['predicate']==predicate and last['current_thread']>0
        assert last['current_pid']>0
        assert last['current_graph_id']>=-1
    assert not route.active and all('forward' not in b.attn.__dict__ for b in model.encoder.blocks)


def test_refuse_foreign_graph_even_on_admitted_runner_thread(fixture):
    model,_,_=fixture;value,target,masks=objective(fixture)
    with PredictorRoute(model,'original',tiny=True) as route:
        route.phase='draw-before';loss=model.loss_per_example(value,target,masks).sum()
        task=route.prepare_backward(loss,value)
        # A second real GraphTask is invoked from the root hook of the bound one.
        original_root=task._root
        def root(gradients):
            original_root(gradients)
            unrelated=torch.tensor(1.,requires_grad=True)
            class Foreign(torch.autograd.Function):
                @staticmethod
                def forward(ctx,x):return x.clone()
                @staticmethod
                def backward(ctx,g):
                    model.encoder.blocks[0].attn(torch.rand(1,32,128))
                    return g
            with torch.enable_grad():foreign=Foreign.apply(unrelated)
            torch.autograd.grad(foreign,unrelated)
        task._root=root
        with pytest.raises(ValueError,match='foreign-autograd-graph'):task.run()
        assert route.refusals[0]['current_graph_id']!=route.refusals[0]['expected_graph_id']


def test_foreign_call_during_actual_running_recomputation_stops_scope(fixture):
    model,_,_=fixture;value,target,masks=objective(fixture)
    entered=threading.Event();release=threading.Event();errors=[]
    def hold(block,args):
        if torch._C._current_graph_task_id()>=0:
            entered.set();assert release.wait(10)
    hook=model.encoder.blocks[-1].register_forward_pre_hook(hold)
    try:
        with PredictorRoute(model,'original',tiny=True) as route:
            route.phase='draw-before';loss=model.loss_per_example(value,target,masks).sum()
            task=route.prepare_backward(loss,value)
            def run():
                try:task.run()
                except BaseException as exc:errors.append(exc)
            thread=threading.Thread(target=run);thread.start()
            try:
                assert entered.wait(10)
                with pytest.raises(ValueError,match='foreign-autograd-graph'):
                    model.encoder.blocks[0].attn(torch.rand(1,32,128))
            finally:release.set();thread.join(15)
            assert not thread.is_alive() and errors and task.state=='failed'
            assert task.receipt['hook_removed'] and route._task is None
    finally:hook.remove()


@pytest.mark.parametrize('fault',['source','identity','method','exception'])
def test_restore_on_setup_and_call_errors(fixture,monkeypatch,fault):
    model,_,_=fixture;value,target,masks=objective(fixture)
    original=model.predictor.predictor_blocks[0].attn.forward.__func__
    if fault=='source':monkeypatch.setattr(frozen,'MODULE_SHA','0'*64)
    with pytest.raises((ValueError,RuntimeError)):
        with PredictorRoute(model,'predictor-math',tiny=True) as route:
            route.phase='draw-before'
            if fault=='identity':model._diagnostic_execution_policy='changed';route.check()
            elif fault=='method':model.encoder.blocks[0].attn.forward=lambda *a,**k:None;route.check()
            elif fault=='exception':raise RuntimeError('injected')
    assert model._diagnostic_execution_policy=='shared-fp64-native'
    assert model.predictor.predictor_blocks[0].attn.forward.__func__ is original
    assert not frozen._ACTIVE


def test_call_and_refusal_evidence_caps(fixture):
    model,_,_=fixture;value,target,masks=objective(fixture)
    with PredictorRoute(model,'original',tiny=True,max_calls=1) as route:
        route.phase='draw-before'
        with pytest.raises(ValueError,match='call-cap'):model.loss_per_example(value,target,masks)
        assert len(route.rows)==1
        for _ in range(40):
            with pytest.raises(ValueError):model.encoder.blocks[0].attn(torch.rand(1,32,128))
        assert len(route.refusals)==32 and route.refusals[0]['predicate']=='call-cap'


def test_completed_token_is_inert_after_route_exit(fixture):
    model,_,_=fixture;value,target,masks=objective(fixture)
    with PredictorRoute(model,'original',tiny=True) as route:
        route.phase='draw-before';loss=model.loss_per_example(value,target,masks).sum()
        task=route.prepare_backward(loss,value)
    assert task.state=='cancelled' and task.loss is None and task.receipt['hook_removed']
    with pytest.raises(ValueError,match='single-use'):other_thread(task.run)
    assert not route.active


def test_worker_failure_removes_root_hook_and_restores_methods(fixture):
    model,_,_=fixture;value,target,masks=objective(fixture)
    handle=None
    with PredictorRoute(model,'original',tiny=True) as route:
        route.phase='draw-before';loss=model.loss_per_example(value,target,masks).sum()
        task=route.prepare_backward(loss,value)
        def injected(block,args):
            if torch._C._current_graph_task_id()>=0:raise RuntimeError('injected backward failure')
        handle=model.encoder.blocks[0].register_forward_pre_hook(injected)
        try:
            with pytest.raises(RuntimeError,match='injected backward failure'):other_thread(task.run)
        finally:handle.remove()
        assert task.state=='failed' and task.receipt['hook_removed'] and route._task is None
    assert not route.active and not frozen._ACTIVE


def test_genuinely_overlapping_attention_call_is_refused_before_upstream_context(fixture):
    import inspect
    model,_,_=fixture;entered=threading.Event();release=threading.Event();errors=[]
    with PredictorRoute(model,'original',tiny=True) as route:
        route.phase='draw-before';owner=model.encoder.blocks[0].attn
        private=inspect.getclosurevars(owner.forward.__func__).nonlocals['private']
        view=private.__globals__['F'];original=view.route
        def pause(*a,**kw):
            entered.set();assert release.wait(10)
            return original(*a,**kw)
        view.route=pause
        def contender():
            try:
                assert entered.wait(10)
                owner(torch.rand(1,32,128))
            except BaseException as exc:errors.append(exc)
            finally:release.set()
        thread=threading.Thread(target=contender);thread.start()
        try:
            with pytest.raises(ValueError,match='prior-refusal'):owner(torch.rand(1,32,128))
        finally:release.set();thread.join(15)
        assert not thread.is_alive() and errors
        assert route.refusals[0]['predicate']=='concurrent-attention-method'
    assert not route.active and not frozen._ACTIVE


def test_exit_during_running_scope_cannot_remove_owned_methods(fixture):
    model,_,_=fixture;value,target,masks=objective(fixture)
    entered=threading.Event();release=threading.Event();errors=[]
    def hold(block,args):
        if torch._C._current_graph_task_id()>=0:
            entered.set();assert release.wait(10)
    hook=model.encoder.blocks[-1].register_forward_pre_hook(hold)
    try:
        with PredictorRoute(model,'original',tiny=True) as route:
            route.phase='draw-before';loss=model.loss_per_example(value,target,masks).sum()
            task=route.prepare_backward(loss,value)
            def run():
                try:task.run()
                except BaseException as exc:errors.append(exc)
            thread=threading.Thread(target=run);thread.start()
            try:
                assert entered.wait(10)
                with pytest.raises(ValueError,match='route-exit-during-backward'):route.__exit__(None,None,None)
                assert route.active and all('forward' in block.attn.__dict__ for block in model.encoder.blocks)
            finally:release.set();thread.join(15)
            assert not thread.is_alive() and errors and route._task is None
    finally:hook.remove()
    assert not route.active and not frozen._ACTIVE


def test_local_only_worker_refuses_every_target_domain_before_model_import(tmp_path):
    from types import SimpleNamespace
    from models.predictor_eot8_owned_cpu_v1 import run
    with pytest.raises(ValueError,match='Only local CPU analogue'):
        run(SimpleNamespace(tiny=False,device='cuda'))
    with pytest.raises(ValueError,match='Only local CPU analogue'):
        run(SimpleNamespace(tiny=True,device='cuda'))
    assert not list(tmp_path.iterdir())


def test_four_fresh_cpu_workers_all_eight_draws_match_saved_v1_and_replay(tmp_path):
    import hashlib,json,subprocess,sys
    baseline=ROOT/'runs/pretrained/diagnostic/predictor-eot8-10603-v1-20261105/cpu-four-worker'
    initial=baseline/'initial.pt';initial_sha=hashlib.sha256(initial.read_bytes()).hexdigest()
    code='''import sys
from pathlib import Path
from types import SimpleNamespace
from models.predictor_eot8_owned_cpu_v1 import run
b=Path(sys.argv[1]);i=int(sys.argv[2]);run(SimpleNamespace(output=b/str(i),index=i,tiny=True,device='cpu',upstream=Path(sys.argv[3]),initial=Path(sys.argv[4]),initial_sha=sys.argv[5],reference=b/str(i-1) if i in (1,3) else None,replay_reference=b/str(i-2) if i in (2,3) else None))
'''
    pids=[]
    for i in range(4):
        result=subprocess.run([sys.executable,'-B','-c',code,str(tmp_path),str(i),str(ROOT/'external/vjepa2'),str(initial),initial_sha],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        (tmp_path/('log-'+str(i))).write_bytes(result.stdout)
        assert result.returncode==0,result.stdout.decode()[-6000:]
        record=json.loads((tmp_path/str(i)/'worker.json').read_text())
        old=json.loads((baseline/str(i)/'worker.json').read_text())
        assert record['complete'] and record['passed'] and record['local_tiny_analogue']
        assert record['baseline']==old['baseline'] and not record['ownership_refusals']
        assert len(record['backward_scopes'])==5 and len(record['artifacts'])==10
        assert all(s['state']=='complete' and s['hook_removed'] and s['graph_id']>=0 for s in record['backward_scopes'])
        for name,item in record['artifacts'].items():
            assert item['source']==old['artifacts'][name]['source']
            current=torch.load(tmp_path/str(i)/item['file'],map_location='cpu',weights_only=True)
            prior=torch.load(baseline/str(i)/old['artifacts'][name]['file'],map_location='cpu',weights_only=True)
            assert torch.equal(current,prior),name
        pids.append(record['pid'])
    assert len(set(pids))==4


def test_inactive_route_retains_precise_refusal_before_context_entry(fixture):
    model,_,_=fixture;value,target,masks=objective(fixture)
    route=PredictorRoute(model,'original',tiny=True)
    with pytest.raises(ValueError,match='route-inactive'):route.prepare_backward(None,value)
    assert route.refusals[0]['predicate']=='route-inactive'
    assert route.refusals[0]['phase'] is None and route.refusals[0]['current_graph_id']==-1
    assert not route.active and not frozen._ACTIVE
