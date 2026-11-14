"""Explicit target port; frozen arithmetic/ownership, device-aware observation only."""
from contextlib import ExitStack
import inspect
import os
from pathlib import Path
import platform
import threading
from types import FunctionType

import torch
from attacks.pgd import native_objective
from models import predictor_forward_preserving_cpu_v1 as cpu
from models import predictor_forward_preserving_upstream_v1 as small
from models import predictor_eot8_owned_v1 as native
from models import predictor_eot8_adapter_v1 as adapter

IDENTITY = 'predictor-original-forward-public-math-vjp-diagnostic-v1'
MARKER = '_forward_preserving_target_diagnostic'
PINS = { 'models/predictor_forward_preserving_cpu_v1.py': small.PROTOTYPE_SHA,
         'models/predictor_eot8_owned_v1.py': small.NATIVE_SHA,
         'models/predictor_forward_preserving_upstream_v1.py': 'db71de4b6b8c3ba1532639258f23e7b4b5a850493625bf6ba1a7e2f8c9b13b88'}


def fingerprint(t):
    # Device transfer is observational only: no cast, replacement or saved-state fabrication.
    raw = t.detach().cpu().contiguous()
    return {'shape': list(t.shape), 'stride': list(t.stride()), 'dtype': str(t.dtype),
            'device': str(t.device), 'offset': t.storage_offset(),
            'storage_bytes': t.untyped_storage().nbytes(), 'version': t._version,
            'sha256': cpu.hashlib.sha256(raw.numpy().tobytes()).hexdigest()}


# Keep the exact frozen arithmetic code object. Only fingerprint observation gains
# an explicit device-to-CPU copy; no tensor fed into the VJP is substituted.
_math_globals = dict(cpu._OriginalForwardMathVJP.backward.__globals__, fingerprint=fingerprint)
_public_math_backward = FunctionType(cpu._OriginalForwardMathVJP.backward.__code__, _math_globals,
                                    'frozen_public_math_backward', cpu._OriginalForwardMathVJP.backward.__defaults__)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, route, index, guarded_forward, q, k, v):
        route._admit_forward()
        row = route.calls[index]
        with torch.enable_grad():
            output = guarded_forward(q, k, v, dropout_p=0., is_causal=False, attn_mask=None)
            row['original_forward_grad_enabled'] = torch.is_grad_enabled()
        ctx.route, ctx.index = route, index
        ctx.save_for_backward(q, k, v)
        row.update(forward_complete=True, output=fingerprint(output), exit_flags=cpu.flags())
        route._same_flags()
        return output.detach()

    @staticmethod
    def backward(ctx, incoming):
        result = _public_math_backward(ctx, incoming)
        return result[:2] + (None,) + result[2:]


class _State(small._BoundPrototype):
    def __init__(self, bridge):
        self.bridge = bridge
        self.max_calls = bridge.forward_cap
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.original_flags = cpu.flags()
        self._sdpa = torch.nn.functional.scaled_dot_product_attention
        self._grad = torch.autograd.grad
        self._graph_api = getattr(torch._C, '_current_graph_task_id', None)
        if not callable(self._graph_api) or self._sdpa is not torch._C._nn.scaled_dot_product_attention:
            raise ValueError('Original public computation and actual graph identity required.')
        self.calls, self.scope = [], None
        self.active = self.entered = self.gradient_used = False
        self._outer = self.bound_task = None

    def attention(self, *args, **kwargs):
        raise ValueError('No standalone target attention entry.')

    def gradient(self, *args, **kwargs):
        raise ValueError('Use the exact native loss/input token.')

    def _validate(self, values, *, incoming=None):
        bridge = self.bridge
        device = 'cpu' if bridge.tiny else 'cuda'
        shape = (1, 2, 32, 32) if bridge.tiny else (1, 12, 2048, 32)
        backing_cap = 128*1024 if bridge.tiny else 16*2**20
        if len({id(t) for t in values}) != 3:
            raise ValueError('Distinct original Q/K/V view objects required.')
        for t in values:
            if (t.device.type != device or t.dtype != torch.float32 or tuple(t.shape) != shape
                    or not t.requires_grad or any(s <= 0 for s in t.stride())
                    or t.untyped_storage().nbytes() > backing_cap or not torch.isfinite(t).all()):
                raise ValueError('Exact bounded native predictor Q/K/V domain required.')
        if incoming is not None and (incoming.device != values[0].device or incoming.dtype != torch.float32
                or tuple(incoming.shape) != shape or not torch.isfinite(incoming).all()):
            raise ValueError('Actual finite native incoming cotangent required.')


class TargetRoute:
    """One EOT8 diagnostic only; CPU analogue is refused on university hosts."""
    def __init__(self, model, *, tiny=False):
        host = platform.node().split('.')[0]
        if type(tiny) is not bool or (tiny and host in ('linux10603', 'linux10609')):
            raise ValueError('Local tiny CPU analogue is not a university entry.')
        if not tiny and (host != 'linux10603' or platform.system() != 'Linux'):
            raise ValueError('Target route requires linux10603; no local CUDA fallback.')
        root = Path(__file__).resolve().parents[1]
        for n, w in PINS.items():
            if small.digest(root/n) != w:
                raise ValueError('Frozen scientific/ownership source changed.')
        counts, device = ((2, 2), 'cpu') if tiny else ((24, 12), 'cuda')
        parameters = list(model.parameters())
        if (len(model.encoder.blocks), len(model.predictor.predictor_blocks)) != counts:
            raise ValueError('Exact declared upstream block counts required.')
        if (not parameters or any(p.device.type != device or p.dtype != torch.float32 for p in parameters)
                or any(m.training for m in model.modules()) or hasattr(model, MARKER)
                or getattr(model, '_diagnostic_execution_policy', None) != 'shared-fp64-native'
                or any(p.requires_grad or p.grad is not None for p in model.target_encoder.parameters())):
            raise ValueError('Fixed native FP32 eval model/teacher/domain required.')
        expected_heads = (2, 32) if tiny else (12, 32)
        if any((b.attn.num_heads, b.attn.head_dim) != expected_heads for b in model.predictor.predictor_blocks):
            raise ValueError('Predictor native attention geometry changed.')
        self.model, self.tiny = model, tiny
        self.initial_vjps = counts[1]*8*2
        self.forward_cap = self.initial_vjps*2
        self.native = native.PredictorRoute(model, 'original', tiny=tiny, max_calls=4096)
        self.prototype = _State(self)
        self.views = []
        self.active = self.entered = False
        self.stack = self.last_task = self._loss_input = None
        self._objective_used = False

    def check_binding(self):
        if (not self.active or getattr(self.model, MARKER, None) != IDENTITY
                or self.native.phase != 'EOT8-before'
                or any(fn.__globals__['F'] is not new for fn, old, new in self.views)):
            raise ValueError('Target diagnostic binding/phase changed.')

    def _restore(self):
        for fn, old, new in self.views:
            fn.__globals__['F'] = old
        if hasattr(self.model, MARKER):
            delattr(self.model, MARKER)
        self._loss_input = None

    def __enter__(self):
        if self.entered:
            raise ValueError('Target diagnostic scope is single use.')
        self.active = self.entered = True
        stack = self.stack = ExitStack()
        try:
            stack.enter_context(self.native)
            stack.enter_context(self.prototype)
            stack.callback(self._restore)
            setattr(self.model, MARKER, IDENTITY)
            for block, module in enumerate(self.model.predictor.predictor_blocks):
                private = inspect.getclosurevars(module.attn.forward.__func__).nonlocals.get('private')
                if not isinstance(private, FunctionType):
                    raise ValueError('Original instance-private native method required.')
                old = private.__globals__['F']
                if not isinstance(old, adapter.FunctionalView) or old.original is not torch.nn.functional:
                    raise ValueError('Original private functional alias required.')
                guarded = old.route
                def routed(q, k, v, _guarded=guarded, _block=block, **kwargs):
                    state = self.prototype
                    state._admit_forward()
                    if (kwargs != {'dropout_p': 0., 'is_causal': False, 'attn_mask': None}
                            or not torch.is_grad_enabled() or torch.is_inference_mode_enabled()
                            or torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda')
                            or len(state.calls) >= self.forward_cap):
                        raise ValueError('Bounded original-context target attention calls required.')
                    state._validate((q, k, v))
                    row = {'index': len(state.calls), 'owner': 'predictor.'+str(_block),
                           'forward_graph_id': state._graph_id(), 'forward_thread': threading.get_ident(),
                           'inputs': [fingerprint(t) for t in (q, k, v)], 'aliases': cpu.aliases((q, k, v)),
                           'entry_flags': cpu.flags(), 'forward_complete': False, 'backward_complete': False}
                    state.calls.append(row)
                    return _Attention.apply(state, row['index'], _guarded, q, k, v)
                new = adapter.FunctionalView(old.original, routed)
                self.views.append((private, old, new))
                private.__globals__['F'] = new
            self.native.phase = 'EOT8-before'
            self.check_binding()
            return self
        except BaseException:
            stack.__exit__(*__import__('sys').exc_info())
            self.active = False
            raise

    def loss(self, value, targets, banks):
        self.check_binding()
        if self._objective_used or self.prototype.calls:
            raise ValueError('One native EOT8 loss only; no retry.')
        self._objective_used = True
        pixels = (1, 3, 4, 16, 16) if self.tiny else (1, 3, 16, 256, 256)
        target = (1, 32, 128) if self.tiny else (1, 2048, 1024)
        if (tuple(value.shape) != pixels or not value.is_leaf or not value.requires_grad
                or tuple(targets.shape) != target or targets.requires_grad
                or targets.device != value.device or targets.dtype != torch.float32
                or len(banks) != 8 or any(len(draw) != 2 for draw in banks)):
            raise ValueError('Exact original full EOT8 loss/input/clean target scope required.')
        loss = native_objective(self.model, targets, banks)(value).sum()
        self._loss_input = (loss, value)
        return loss

    def gradient(self, loss, value):
        self.check_binding()
        state = self.prototype
        if (state.gradient_used or self._loss_input is None or loss is not self._loss_input[0]
                or value is not self._loss_input[1]):
            raise ValueError('One exact native loss/input derivative required.')
        task = self.native.prepare_backward(loss, value)
        self.last_task = state.bound_task = task
        state.gradient_used = True
        scope = {'state': 'prepared', 'graph_id': None, 'identity_source': 'unchanged-native-root',
                 'new_outer_hook_added': False, 'loss_version': loss._version, 'input_version': value._version,
                 'forward_cap': self.forward_cap, 'expected_initial_VJPs': self.initial_vjps}
        state.scope = state._outer = scope
        try:
            result, = task.run()
            initial = [r for r in state.calls if r['forward_graph_id'] == -1]
            if (task.state != 'complete' or not task.receipt['hook_removed'] or task.graph_id != scope['graph_id']
                    or task.graph_id is None or len(state.calls) != self.forward_cap or len(initial) != self.initial_vjps
                    or any(not r['backward_complete'] or not r['hooks_removed'] for r in initial)
                    or len({r['inner_graph_id'] for r in initial}) != self.initial_vjps
                    or (loss._version, value._version) != (scope['loss_version'], scope['input_version'])):
                raise ValueError('Complete original ownership, checkpoints and public math VJPs required.')
            self.native.check()
            scope.update(state='complete', native_root_hook_removed=True)
            return result
        except BaseException as error:
            scope.update(state='failed', error_type=type(error).__name__, error=str(error)[:2048])
            raise
        finally:
            state.bound_task = state._outer = self._loss_input = None

    def __exit__(self, kind, error, tb):
        try:
            return self.stack.__exit__(kind, error, tb)
        finally:
            self.active = False
