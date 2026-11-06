"""Separate graph-owned recomputation route; captured v1 source remains frozen."""
import os
import threading
from types import FunctionType, MethodType

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from models import predictor_eot8_adapter_v1 as base

_REGISTRY_LOCK = threading.RLock()
_PHASES = ('EOT8-before', 'draw-before', 'draw-profiled', 'draw-after', 'EOT8-after')


class OwnedBackward:
    """One exact loss/input call, prepared by the route's creator; never reusable."""
    def __init__(self, route, loss, value):
        self.route, self.loss, self.value = route, loss, value
        self.phase = route.phase
        self.state = 'prepared'
        self.graph_id = None
        self.runner_thread = self.hook_thread = None
        self.hook = None
        self.root = loss.grad_fn
        self.versions = (loss._version, value._version)
        self.identities = (id(loss), id(value))
        self.receipt = {'phase': self.phase, 'index': len(route.scopes), 'state': 'prepared',
                        'creator_thread': route.thread, 'pid': route.pid,
                        'loss_id': id(loss), 'input_id': id(value), 'graph_id': None,
                        'runner_thread': None, 'root_hook_thread': None,
                        'call_threads': [], 'hook_removed': False}
        route.scopes.append(self.receipt)

    def _root(self, gradients):
        route = self.route
        with route._state_lock:
            graph = route._graph_id()
            if route._task is not self or self.state != 'running' or self.phase != route.phase:
                route._fail('backward-root-outside-scope')
            if graph < 0 or self.graph_id is not None:
                route._fail('backward-root-missing-or-repeated')
            self.graph_id, self.hook_thread = graph, threading.get_ident()
            self.receipt.update(graph_id=graph, root_hook_thread=self.hook_thread)
        # Returning None leaves every actual incoming gradient unchanged.
        return None

    def run(self):
        route = self.route
        with route._state_lock:
            if route._task is not self or self.state != 'prepared':
                route._fail('backward-task-not-single-use')
            route._common()
            if self.phase != route.phase or route._method_owner is not None:
                route._fail('backward-phase-or-method-overlap')
            if route._graph_id() != -1:
                route._fail('nested-autograd-task')
            if (id(self.loss), id(self.value)) != self.identities or self.versions != (self.loss._version, self.value._version) or self.loss.grad_fn is not self.root:
                route._fail('prepared-tensor-or-version-changed')
            if base.flags() != route.original_flags:
                route._fail('backward-entry-flags')
            if torch.autograd.grad is not route._grad:
                route._fail('autograd-function-changed')
            self.state, self.runner_thread = 'running', threading.get_ident()
            self.receipt.update(state=self.state, runner_thread=self.runner_thread)
        try:
            self.hook = self.root.register_prehook(self._root)
            result = route._grad(self.loss, self.value)
            with route._state_lock:
                route._common()
                if base.flags() != route.original_flags:
                    route._fail('backward-exit-flags')
                if self.graph_id is None or route._method_owner is not None:
                    route._fail('backward-root-unobserved-or-method-open')
                if not self.receipt['call_threads']:
                    route._fail('backward-recomputation-unobserved')
                self.state = 'complete'
                self.receipt['state'] = self.state
            return result
        except BaseException as exc:
            self.state = 'failed'
            self.receipt.update(state='failed', error_type=type(exc).__name__)
            route._poisoned = True
            raise
        finally:
            if self.hook is not None:
                self.hook.remove()
            self.receipt['hook_removed'] = True
            with route._state_lock:
                if route._task is self:
                    route._task = None
            # Release references after this single call; receipts contain only metadata.
            self.loss = self.value = self.root = None


class PredictorRoute(base.PredictorRoute):
    """Admit creator forwards and serial recomputation in one observed GraphTask."""
    def __init__(self, model, mode, *, tiny=False, max_calls=4096):
        super().__init__(model, mode, tiny=tiny, max_calls=max_calls)
        if type(max_calls) is not int or not 0 < max_calls <= 4096:
            raise ValueError('Bounded call count required.')
        self.phase = None
        self.pid = os.getpid()
        self._state_lock = threading.RLock()
        self._method_lock = threading.Lock()
        self._method_owner = self._task = None
        self._poisoned = False
        self._grad = torch.autograd.grad
        self._graph_api = getattr(torch._C, '_current_graph_task_id', None)
        if not callable(self._graph_api):
            raise ValueError('Graph identity API unavailable; no thread fallback.')
        self.refusals, self.scopes = [], []

    def _graph_id(self):
        graph = self._graph_api()
        if type(graph) is not int or graph < -1:
            raise ValueError('Unsupported autograd graph identity.')
        return graph

    def _fail(self, predicate):
        self._poisoned = True
        task = self._task
        if len(self.refusals) < 32:
            self.refusals.append({'predicate': predicate, 'active': self.active,
                'creator_thread': self.thread, 'current_thread': threading.get_ident(),
                'creator_pid': self.pid, 'current_pid': os.getpid(),
                'current_graph_id': self._graph_id(),
                'expected_graph_id': None if task is None else task.graph_id,
                'task_state': None if task is None else task.state,
                'phase': self.phase if type(self.phase) is str and self.phase in _PHASES else None,
                'call_rows': len(self.rows), 'max_calls': self.max_calls})
        raise ValueError('Owned attention refused: ' + predicate)

    def _common(self):
        if not self.active:
            self._fail('route-inactive')
        if os.getpid() != self.pid:
            self._fail('process-changed')
        if self._poisoned:
            self._fail('prior-refusal')
        expected = base.IDENTITY if self.mode == 'predictor-math' else self.old_policy
        if getattr(self.model, '_diagnostic_execution_policy', 'original') != expected:
            self._fail('diagnostic-identity-changed')
        if any(obj.forward is not function for obj, function in self.expected):
            self._fail('owned-method-changed')
        if self.phase is not None and (type(self.phase) is not str or self.phase not in _PHASES):
            self._fail('unknown-phase')

    def check(self):
        with self._state_lock:
            self._common()
            if threading.get_ident() != self.thread or self._task is not None or self._method_owner is not None:
                self._fail('creator-check-overlap')
            if base.flags() != self.original_flags:
                self._fail('backend-flags-leaked')

    def _admit(self):
        with self._state_lock:
            self._common()
            if self.phase is None:
                self._fail('phase-missing')
            if len(self.rows) >= self.max_calls:
                self._fail('call-cap')
            graph = self._graph_id()
            task = self._task
            if task is None:
                if threading.get_ident() != self.thread or graph != -1:
                    self._fail('unowned-forward-thread-or-graph')
                return False
            if task.state != 'running' or task.phase != self.phase:
                self._fail('backward-not-running-or-phase-changed')
            if task.graph_id is None or graph != task.graph_id:
                self._fail('foreign-autograd-graph')
            thread = threading.get_ident()
            if thread not in task.receipt['call_threads']:
                if len(task.receipt['call_threads']) >= 8:
                    self._fail('backward-thread-cap')
                task.receipt['call_threads'].append(thread)
            return True

    def prepare_backward(self, loss, value):
        """Bind exact tensors on the creator thread before any worker may run them."""
        with self._state_lock:
            self.check()
            if self.phase is None or self._graph_id() != -1:
                self._fail('prepare-outside-forward-phase')
            if len(self.scopes) >= 5:
                self._fail('backward-scope-cap')
            if (not isinstance(loss, torch.Tensor) or not isinstance(value, torch.Tensor)
                    or loss.shape != torch.Size([]) or loss.dtype != torch.float32
                    or value.dtype != torch.float32 or not loss.requires_grad or not value.requires_grad
                    or not value.is_leaf or loss.grad_fn is None or loss.device != value.device
                    or value.device.type != ('cpu' if self.tiny else 'cuda')):
                self._fail('loss-or-input-contract')
            primary = [r for r in self.rows if r['phase'] == self.phase and not r['during_backward']]
            owners = {role + '.' + str(i) for role, blocks in [('encoder', self.model.encoder.blocks),
                ('predictor', self.model.predictor.predictor_blocks)] for i in range(len(blocks))}
            expected = 16 if self.phase.startswith('EOT8') else 2
            if (set(r['owner'] for r in primary) != owners or
                    any(sum(r['owner'] == owner and r['outcome'] == 'returned' for r in primary) != expected for owner in owners)):
                self._fail('forward-coverage-before-backward')
            task = OwnedBackward(self, loss, value)
            self._task = task
            return task

    def input_gradient(self, loss, value):
        return self.prepare_backward(loss, value).run()

    def _install(self, obj, name, role, index):
        # The frozen upstream operations/geometry/alias rules are retained verbatim.
        if 'forward' in obj.__dict__:
            raise ValueError('Refuse pre-wrapped attention instances.')
        method = obj.forward
        fn = method.__func__
        if type(obj).__name__ != 'RoPEAttention' or fn.__name__ != 'forward':
            raise ValueError('Require original owned RoPEAttention.')
        import hashlib, inspect
        from pathlib import Path
        path = Path(inspect.getsourcefile(fn)).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != base.MODULE_SHA or fn.__globals__.get('F') is not torch.nn.functional:
            raise ValueError('Upstream source or F alias changed.')
        if not obj.use_sdpa or obj.proj_drop_prob != 0 or obj.is_causal:
            raise ValueError('Require original noncausal zero-dropout SDPA configuration.')
        old_f = fn.__globals__['F']; old_sdpa = old_f.scaled_dot_product_attention
        if old_sdpa is not torch._C._nn.scaled_dot_product_attention:
            raise ValueError('Require original builtin SDPA alias.')

        def routed(q, k, v, **kwargs):
            during_backward = self._admit()
            if self._method_owner != (threading.get_ident(), id(obj)):
                self._fail('outside-owned-method')
            if old_f.scaled_dot_product_attention is not old_sdpa:
                self._fail('functional-alias-changed')
            if set(kwargs) != {'dropout_p', 'is_causal', 'attn_mask'} or kwargs['dropout_p'] != 0 or kwargs['is_causal'] is not False or kwargs['attn_mask'] is not None:
                self._fail('sdpa-arguments-changed')
            if any(t.dtype != torch.float32 or t.device.type != ('cpu' if self.tiny else 'cuda') or t.ndim != 4 for t in (q,k,v)):
                self._fail('sdpa-dtype-or-domain')
            if q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1 or q.shape[1] != obj.num_heads or q.shape[3] != obj.head_dim:
                self._fail('owned-geometry')
            if not self.tiny and (q.shape[1:] != (12,2048,32) if role == 'predictor' else q.shape[1] != 16 or q.shape[3] != 64):
                self._fail('pretrained-geometry')
            row = {'index': len(self.rows), 'owner': name, 'role': role, 'block': index,
                   'q': base.layout(q), 'k': base.layout(k), 'v': base.layout(v), 'entry_flags': base.flags(),
                   'outcome': 'entered', 'phase': self.phase, 'during_backward': during_backward,
                   'thread': threading.get_ident(), 'creator_thread': self.thread, 'pid': os.getpid(),
                   'graph_id': self._graph_id(),
                   'backward_scope': None if self._task is None else self._task.receipt['index']}
            self.rows.append(row)
            label = base.PREFIX + str(row['index']) + '/' + name
            ctx = sdpa_kernel(SDPBackend.MATH) if self.mode == 'predictor-math' and role == 'predictor' else base._unchanged()
            try:
                with ctx:
                    row['effective_flags'] = base.flags()
                    if self.mode == 'predictor-math' and role == 'predictor' and row['effective_flags'] != {'flash':False,'mem_efficient':False,'math':True,'cudnn':False}:
                        self._fail('actual-call-math-selection')
                    with torch.profiler.record_function(label):
                        result = old_sdpa(q,k,v,**kwargs)
                    row['outcome'] = 'returned'
                    return result
            except BaseException as exc:
                row.update(outcome='interrupted', error_type=type(exc).__name__)
                raise
            finally:
                row['exit_flags'] = base.flags()
                if row['exit_flags'] != row['entry_flags']:
                    self._fail('sdpa-context-restoration')

        env = dict(fn.__globals__); env['F'] = base.FunctionalView(old_f, routed)
        private = FunctionType(fn.__code__, env, fn.__name__, fn.__defaults__, fn.__closure__)
        private.__kwdefaults__ = fn.__kwdefaults__

        def owned_forward(instance, *args, **kwargs):
            # Guard before upstream work/context entry; serialize only owned attention calls.
            if not self._method_lock.acquire(blocking=False):
                self._fail('concurrent-attention-method')
            try:
                self._admit()
                if base.flags() != self.original_flags:
                    self._fail('method-entry-flags')
                self._method_owner = (threading.get_ident(), id(obj))
                return private(instance, *args, **kwargs)
            finally:
                self._method_owner = None
                self._method_lock.release()
                if base.flags() != self.original_flags:
                    self._fail('method-exit-flags')

        bound = MethodType(owned_forward, obj)
        obj.forward = bound
        self.changes.append(obj); self.expected.append((obj,bound))

    def __enter__(self):
        with _REGISTRY_LOCK:
            return super().__enter__()

    def __exit__(self, *args):
        if threading.get_ident() != self.thread:
            self._fail('route-exit-foreign-thread')
        task = self._task
        if task is not None and task.state == 'running':
            # Keep methods/hook owned until the admitted call unwinds.
            self._fail('route-exit-during-backward')
        try:
            if task is not None:
                if task.hook is not None:
                    task.hook.remove()
                task.state = 'cancelled'
                task.loss = task.value = task.root = None
                task.receipt.update(state='cancelled', hook_removed=True)
                self._task = None
        finally:
            with _REGISTRY_LOCK:
                super().__exit__(*args)
