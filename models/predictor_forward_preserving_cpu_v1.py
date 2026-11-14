"""Tiny local first-order custom VJP. No model patching or target entry exists."""
import hashlib
import os
import platform
import threading

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

IDENTITY = 'local-original-forward-public-math-vjp-v1'
_ACTIVE = set()
_LOCK = threading.RLock()


def flags():
    return {name: bool(getattr(torch.backends.cuda, name + '_sdp_enabled')())
            for name in ('flash', 'mem_efficient', 'math', 'cudnn')}


def fingerprint(t):
    return {'shape': list(t.shape), 'stride': list(t.stride()), 'dtype': str(t.dtype),
            'offset': t.storage_offset(), 'storage_bytes': t.untyped_storage().nbytes(),
            'version': t._version,
            'sha256': hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()}


def aliases(values):
    # Storage identities are inspected only; no storage or input is fabricated.
    pointers = []
    result = []
    for t in values:
        pointer = t.untyped_storage().data_ptr()
        if pointer not in pointers:
            pointers.append(pointer)
        result.append(pointers.index(pointer))
    return result


class _OriginalForwardMathVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, route, index, q, k, v):
        row = route.calls[index]
        route._admit_forward()
        # Function.forward normally disables grad. Restore the observed original
        # forward eligibility explicitly, then discard its unused autograd graph.
        with torch.enable_grad():
            output = route._sdpa(q, k, v, dropout_p=0., is_causal=False, attn_mask=None)
            row['original_forward_grad_enabled'] = torch.is_grad_enabled()
        ctx.route, ctx.index = route, index
        ctx.save_for_backward(q, k, v)
        row.update(forward_complete=True, output=fingerprint(output), exit_flags=flags())
        route._same_flags()
        return output.detach()

    @staticmethod
    def backward(ctx, incoming):
        route, row = ctx.route, ctx.route.calls[ctx.index]
        if torch.is_grad_enabled():
            raise ValueError('First-order custom VJP only; create_graph is unsupported.')
        outer = route._admit_backward()
        if row.get('backward_complete'):
            raise ValueError('Custom VJP is single use.')
        # Unpacking may invoke nonreentrant checkpoint recomputation. That call
        # must belong to the already observed outer GraphTask.
        values = ctx.saved_tensors
        route._admit_backward()
        if [fingerprint(t) for t in values] != row['inputs'] or aliases(values) != row['aliases']:
            raise ValueError('Saved input values, layout, aliasing or version changed.')
        route._validate(values, incoming=incoming)
        before = fingerprint(incoming)
        row.update(backward_thread=threading.get_ident(), outer_graph_id=outer['graph_id'],
                   incoming=before, inner_graph_id=None, inner_root_calls=0, leaf_hooks=[], hooks_removed=False)
        hooks = []
        try:
            leaves = tuple(t.detach().requires_grad_(True) for t in values)
            if aliases(leaves) != row['aliases']:
                raise ValueError('Math operands changed alias groups.')
            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                row['math_flags'] = flags()
                if row['math_flags'] != {'flash': False, 'mem_efficient': False, 'math': True, 'cudnn': False}:
                    raise ValueError('Exclusive public math selection required.')
                output = route._math_call(*leaves)
                def root_hook(unused):
                    graph = route._graph_id()
                    if (route._outer is not outer or graph < 0 or graph == outer['graph_id']
                            or row['inner_graph_id'] is not None):
                        raise ValueError('Missing, repeated or foreign math GraphTask.')
                    row.update(inner_graph_id=graph, inner_root_thread=threading.get_ident(), inner_root_calls=1)
                hooks.append(output.grad_fn.register_prehook(root_hook))
                for index, t in enumerate(leaves):
                    def leaf_hook(gradient, index=index):
                        if (route._outer is not outer or route._graph_id() != row['inner_graph_id']
                                or any(item['index'] == index for item in row['leaf_hooks'])):
                            raise ValueError('Foreign or repeated inner leaf hook.')
                        row['leaf_hooks'].append({'index': index, 'graph_id': route._graph_id(),
                                                  'thread': threading.get_ident()})
                        return None
                    hooks.append(t.register_hook(leaf_hook))
                gradients = route._grad(output, leaves, incoming.detach(), create_graph=False, retain_graph=False)
            route._admit_backward()
            if row['inner_root_calls'] != 1 or sorted(x['index'] for x in row['leaf_hooks']) != [0, 1, 2]:
                raise ValueError('Every math GraphTask and leaf must be observed.')
            if fingerprint(incoming) != before:
                raise ValueError('Actual incoming cotangent changed.')
            if any(t.shape != values[0].shape or t.dtype != torch.float32 or not torch.isfinite(t).all()
                   for t in gradients):
                raise ValueError('Finite complete math VJP returns required.')
            route._same_flags()
            row.update(backward_complete=True, returned=[fingerprint(t) for t in gradients])
            return None, None, *gradients
        except BaseException as error:
            row.update(error_type=type(error).__name__, error=str(error)[:2048])
            raise
        finally:
            for hook in hooks:
                hook.remove()
            row['hooks_removed'] = True


class LocalForwardPreservingAttention:
    """Explicit attention/gradient calls in one local CPU scope, never a model policy."""
    def __init__(self, *, max_calls=8):
        if platform.node().split('.')[0] in ('linux10603', 'linux10609'):
            raise ValueError('Local tiny CPU prototype only; university execution refused.')
        if type(max_calls) is not int or not 1 <= max_calls <= 8:
            raise ValueError('At most eight tiny local forward calls.')
        self.max_calls = max_calls
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.original_flags = flags()
        self._sdpa = torch.nn.functional.scaled_dot_product_attention
        self._grad = torch.autograd.grad
        self._graph_api = getattr(torch._C, '_current_graph_task_id', None)
        if not callable(self._graph_api) or self._sdpa is not torch._C._nn.scaled_dot_product_attention:
            raise ValueError('Original public SDPA and graph identity inspection required.')
        self.calls, self.scope = [], None
        self.active = self.entered = self.gradient_used = False
        self._outer = None

    def _graph_id(self):
        value = self._graph_api()
        if type(value) is not int or value < -1:
            raise ValueError('Invalid GraphTask identity; no fallback.')
        return value

    def _same_flags(self):
        if flags() != self.original_flags:
            raise ValueError('Original backend flags changed.')

    def _common(self):
        if (not self.active or os.getpid() != self.pid
                or torch.nn.functional.scaled_dot_product_attention is not self._sdpa
                or torch.autograd.grad is not self._grad):
            raise ValueError('Inactive, foreign or changed execution scope.')
        self._same_flags()

    def _admit_forward(self):
        self._common()
        graph = self._graph_id()
        if self._outer is None:
            if graph != -1 or threading.get_ident() != self.thread:
                raise ValueError('Foreign forward GraphTask or thread.')
        elif self._outer['state'] != 'running' or graph != self._outer['graph_id']:
            raise ValueError('Foreign checkpoint GraphTask.')

    def _admit_backward(self):
        self._common()
        outer = self._outer
        if outer is None or outer['state'] != 'running' or self._graph_id() != outer['graph_id']:
            raise ValueError('Custom VJP requires its observed owned outer GraphTask.')
        return outer

    @staticmethod
    def _validate(values, *, incoming=None):
        if len({id(t) for t in values}) != 3:
            raise ValueError('Distinct Q/K/V view objects required.')
        q = values[0]
        for t in values:
            if (not isinstance(t, torch.Tensor) or t.device.type != 'cpu' or t.dtype != torch.float32
                    or t.ndim != 4 or t.shape != q.shape or t.shape[0] != 1
                    or not 1 <= t.shape[1] <= 2 or not 1 <= t.shape[2] <= 32 or not 1 <= t.shape[3] <= 32
                    or not t.requires_grad or any(s <= 0 for s in t.stride())
                    or t.untyped_storage().nbytes() > 128 * 1024 or not torch.isfinite(t).all()):
                raise ValueError('Finite tiny FP32 CPU operands with original layouts required.')
        if incoming is not None and (incoming.device.type != 'cpu' or incoming.dtype != torch.float32
                or incoming.shape != q.shape or not torch.isfinite(incoming).all()):
            raise ValueError('Actual finite matching incoming cotangent required.')

    def _math_call(self, q, k, v):
        return self._sdpa(q, k, v, dropout_p=0., is_causal=False, attn_mask=None)

    def attention(self, q, k, v):
        self._admit_forward()
        if self.gradient_used and self._outer is None:
            raise ValueError('Forward execution after a consumed gradient scope is refused.')
        if (not torch.is_grad_enabled() or torch.is_inference_mode_enabled()
                or torch.is_autocast_enabled('cpu') or len(self.calls) >= self.max_calls):
            raise ValueError('Grad-enabled bounded non-autocast local execution required.')
        self._validate((q, k, v))
        row = {'index': len(self.calls), 'forward_graph_id': self._graph_id(),
               'forward_thread': threading.get_ident(), 'inputs': [fingerprint(t) for t in (q, k, v)],
               'aliases': aliases((q, k, v)), 'entry_flags': flags(), 'forward_complete': False,
               'backward_complete': False}
        self.calls.append(row)
        return _OriginalForwardMathVJP.apply(self, row['index'], q, k, v)

    def gradient(self, loss, value, *, create_graph=False):
        self._admit_forward()
        if (create_graph is not False or self.gradient_used or self._outer is not None
                or self._graph_id() != -1 or threading.get_ident() != self.thread
                or not self.calls or loss.shape != torch.Size([]) or not loss.requires_grad
                or loss.grad_fn is None or loss.dtype != torch.float32 or not torch.isfinite(loss)
                or value.device.type != 'cpu' or not value.is_leaf or not value.requires_grad):
            raise ValueError('One fresh first-order local loss/input scope required.')
        self.gradient_used = True
        row = {'state': 'running', 'graph_id': None, 'root_calls': 0, 'runner_thread': threading.get_ident(),
               'root_hook_removed': False, 'loss_version': loss._version, 'input_version': value._version}
        self.scope = self._outer = row
        def root_hook(unused):
            graph = self._graph_id()
            if self._outer is not row or graph < 0 or row['graph_id'] is not None:
                raise ValueError('Outer root must identify exactly one owned GraphTask.')
            row.update(graph_id=graph, root_thread=threading.get_ident(), root_calls=1)
        hook = loss.grad_fn.register_prehook(root_hook)
        try:
            gradient, = self._grad(loss, value, create_graph=False, retain_graph=False)
            if (loss._version, value._version) != (row['loss_version'], row['input_version']):
                raise ValueError('Loss or input version changed during the owned derivative.')
            initial = [c for c in self.calls if c['forward_graph_id'] == -1]
            if (row['root_calls'] != 1 or not initial or any(not c['backward_complete'] for c in initial)
                    or self._graph_id() != -1 or not torch.isfinite(gradient).all()):
                raise ValueError('Incomplete owned local derivative.')
            self._same_flags()
            row['state'] = 'complete'
            return gradient
        except BaseException as error:
            row.update(state='failed', error_type=type(error).__name__, error=str(error)[:2048])
            raise
        finally:
            hook.remove()
            row['root_hook_removed'] = True
            self._outer = None

    def __enter__(self):
        with _LOCK:
            if self.entered or _ACTIVE or os.getpid() != self.pid or threading.get_ident() != self.thread:
                raise ValueError('One fresh serial local scope required.')
            self.entered = self.active = True
            _ACTIVE.add(id(self))
        return self

    def __exit__(self, kind, error, tb):
        try:
            if kind is None:
                self._common()
                if (self._outer is not None or self.scope is None or self.scope['state'] != 'complete'
                        or any(not c['backward_complete'] for c in self.calls if c['forward_graph_id'] == -1)):
                    raise ValueError('Incomplete local prototype scope.')
        finally:
            self.active = False
            with _LOCK:
                _ACTIVE.discard(id(self))
        return False
