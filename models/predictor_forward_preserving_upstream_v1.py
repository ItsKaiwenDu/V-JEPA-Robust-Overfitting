"""One local tiny upstream mask bank; frozen native ownership stays authoritative."""
from contextlib import ExitStack
import hashlib
import inspect
import os
from pathlib import Path
import platform
import threading
from types import FunctionType

import torch
from models import predictor_eot8_owned_v1 as native
from models import predictor_eot8_adapter_v1 as adapter
from models import predictor_forward_preserving_cpu_v1 as cpu

IDENTITY = 'local-upstream-original-forward-public-math-vjp-v1'
NATIVE_SHA = 'ef3675e69ef56b0b4ef4f543cab96b7f2bc757a64202d74500f3802ccd5dfc12'
PROTOTYPE_SHA = '6ba308e948c0ce346512f17ef4fc11eccb542954048c859ad6c6a751db1fbb66'
MARKER = '_local_forward_vjp_integration_identity'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _IntegratedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, route, index, guarded_forward, q, k, v):
        route._admit_forward()
        row = route.calls[index]
        with torch.enable_grad():
            output = guarded_forward(q, k, v, dropout_p=0., is_causal=False, attn_mask=None)
            row['original_forward_grad_enabled'] = torch.is_grad_enabled()
        ctx.route, ctx.index = route, index
        ctx.save_for_backward(q, k, v)
        row.update(forward_complete=True, output=cpu.fingerprint(output), exit_flags=cpu.flags())
        route._same_flags()
        return output.detach()

    @staticmethod
    def backward(ctx, incoming):
        result = cpu._OriginalForwardMathVJP.backward(ctx, incoming)
        # Additional guarded-forward callable is nondifferentiable metadata.
        return result[:2] + (None,) + result[2:]


class _BoundPrototype(cpu.LocalForwardPreservingAttention):
    """Read the actual native root identity without adding native call evidence."""
    def __init__(self, bridge):
        super().__init__(max_calls=8)
        self.bridge = bridge
        self.bound_task = None

    def _observe_owner(self):
        self._common()
        bridge = self.bridge
        bridge.check_binding()
        guard = bridge.native
        with guard._state_lock:
            guard._common()
            graph = guard._graph_id()
            task = guard._task
            if task is None:
                if self.bound_task is not None or graph != -1 or threading.get_ident() != guard.thread:
                    raise ValueError('Unowned local upstream forward.')
                return None
            if (task is not self.bound_task or task.state != 'running' or task.phase != guard.phase
                    or task.graph_id is None or graph != task.graph_id or self._outer is None):
                raise ValueError('Native root has not admitted this local upstream execution.')
            if self._outer['graph_id'] is None:
                self._outer.update(graph_id=task.graph_id, state='running', native_root_thread=task.hook_thread)
            if self._outer['graph_id'] != graph:
                raise ValueError('Foreign graph during upstream integration.')
            # Never call native._admit here: it appends actual method/recomputation
            # evidence. Only the unchanged native method and SDPA wrappers may do so.
            return self._outer

    def _admit_forward(self):
        outer = self._observe_owner()
        if self.gradient_used and outer is None:
            raise ValueError('Local upstream scope already consumed.')

    def _admit_backward(self):
        outer = self._observe_owner()
        if outer is None:
            raise ValueError('Math VJP requires the actual native outer GraphTask.')
        return outer


class LocalUpstreamRoute:
    """Explicit diagnostic identity, no training registration or CUDA entry."""
    def __init__(self, model):
        if platform.node().split('.')[0] in ('linux10603', 'linux10609'):
            raise ValueError('Local small-upstream integration only; university refused.')
        if digest(native.__file__) != NATIVE_SHA or digest(cpu.__file__) != PROTOTYPE_SHA:
            raise ValueError('Frozen native guard/prototype source changed.')
        parameters = list(model.parameters())
        if (not parameters or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in parameters)
                or sum(p.numel() for p in parameters) > 2_000_000
                or len(model.encoder.blocks) != 2 or len(model.predictor.predictor_blocks) != 2
                or any(m.training for m in model.modules())
                or any(p.requires_grad or p.grad is not None for p in model.target_encoder.parameters())
                or getattr(model, '_diagnostic_execution_policy', None) != 'shared-fp64-native'
                or hasattr(model, MARKER)):
            raise ValueError('Exact small eval CPU fixture with frozen clean target required.')
        for block in model.predictor.predictor_blocks:
            if block.attn.num_heads != 2 or block.attn.head_dim != 32:
                raise ValueError('Tiny predictor geometry changed.')
        self.model = model
        self.native = native.PredictorRoute(model, 'original', tiny=True, max_calls=128)
        self.prototype = _BoundPrototype(self)
        self.views = []
        self.stack = None
        self.active = self.entered = False
        self.identity = IDENTITY
        self.last_task = None

    def check_binding(self):
        if (not self.active or getattr(self.model, MARKER, None) != IDENTITY
                or any(function.__globals__['F'] is not new for function, old, new in self.views)):
            raise ValueError('Local upstream binding changed.')

    def _restore_views(self):
        for function, old, new in self.views:
            function.__globals__['F'] = old

    def _remove_marker(self):
        if hasattr(self.model, MARKER):
            delattr(self.model, MARKER)

    def __enter__(self):
        if self.entered:
            raise ValueError('Local upstream route is single use.')
        self.entered = self.active = True
        stack = self.stack = ExitStack()
        try:
            stack.enter_context(self.native)
            stack.enter_context(self.prototype)
            stack.callback(self._restore_views)
            setattr(self.model, MARKER, IDENTITY)
            stack.callback(self._remove_marker)
            for index, block in enumerate(self.model.predictor.predictor_blocks):
                wrapper = block.attn.forward.__func__
                private = inspect.getclosurevars(wrapper).nonlocals.get('private')
                if not isinstance(private, FunctionType):
                    raise ValueError('Expected private native wrapper unavailable.')
                old = private.__globals__['F']
                if not isinstance(old, adapter.FunctionalView) or old.original is not torch.nn.functional:
                    raise ValueError('Expected original instance-private F alias required.')
                guarded = old.route
                def routed(q, k, v, _guarded=guarded, _index=index, **kwargs):
                    if kwargs != {'dropout_p': 0., 'is_causal': False, 'attn_mask': None}:
                        raise ValueError('Original attention arguments required.')
                    proto = self.prototype
                    proto._admit_forward()
                    if (not torch.is_grad_enabled() or torch.is_autocast_enabled('cpu')
                            or torch.is_inference_mode_enabled() or len(proto.calls) >= 8):
                        raise ValueError('Bounded grad-enabled tiny upstream calls only.')
                    proto._validate((q, k, v))
                    row = {'index': len(proto.calls), 'owner': 'predictor.'+str(_index),
                           'forward_graph_id': proto._graph_id(), 'forward_thread': threading.get_ident(),
                           'inputs': [cpu.fingerprint(t) for t in (q, k, v)], 'aliases': cpu.aliases((q, k, v)),
                           'entry_flags': cpu.flags(), 'forward_complete': False, 'backward_complete': False}
                    proto.calls.append(row)
                    return _IntegratedAttention.apply(proto, row['index'], _guarded, q, k, v)
                new = adapter.FunctionalView(old.original, routed)
                self.views.append((private, old, new))
                private.__globals__['F'] = new
            self.native.phase = 'draw-before'
            self.check_binding()
            return self
        except BaseException:
            stack.__exit__(*__import__('sys').exc_info())
            self.active = False
            raise

    def _run_native(self, task):
        return task.run()

    def gradient(self, loss, value):
        self.check_binding()
        proto = self.prototype
        if proto.gradient_used:
            raise ValueError('Local upstream derivative scope consumed.')
        task = self.native.prepare_backward(loss, value)
        self.last_task = proto.bound_task = task
        proto.gradient_used = True
        row = {'state': 'prepared', 'graph_id': None, 'identity_source': 'unchanged-native-root',
               'new_outer_hook_added': False, 'loss_version': loss._version, 'input_version': value._version}
        proto.scope = proto._outer = row
        try:
            result, = self._run_native(task)
            if (task.state != 'complete' or not task.receipt['hook_removed'] or task.graph_id != row['graph_id']
                    or task.graph_id is None or self.native._task is not None
                    or (loss._version, value._version) != (row['loss_version'], row['input_version'])):
                raise ValueError('Unchanged native completion/versions required.')
            initial = [c for c in proto.calls if c['forward_graph_id'] == -1]
            if len(initial) != 4 or any(not c['backward_complete'] for c in initial):
                raise ValueError('All two-block/two-family predictor VJPs required.')
            if len({c['inner_graph_id'] for c in initial}) != 4:
                raise ValueError('Four distinct math GraphTasks required.')
            self.native.check()
            row.update(state='complete', native_root_hook_removed=task.receipt['hook_removed'])
            return result
        except BaseException as error:
            row.update(state='failed', error_type=type(error).__name__, error=str(error)[:2048])
            raise
        finally:
            proto._outer = None
            proto.bound_task = None

    def __exit__(self, kind, error, tb):
        try:
            return self.stack.__exit__(kind, error, tb)
        finally:
            self.active = False
