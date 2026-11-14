"""Separate 64-call local CPU EOT8 entry; frozen eight-call entries stay unchanged."""
from contextlib import ExitStack
import inspect
import os
import platform
import threading
from types import FunctionType

import torch
from attacks.pgd import native_objective
from models import predictor_forward_preserving_upstream_v1 as small
from models import predictor_forward_preserving_cpu_v1 as cpu
from models import predictor_eot8_adapter_v1 as adapter

IDENTITY = 'local-eot8-original-forward-public-math-vjp-v1'
MARKER = '_local_eot8_forward_vjp_identity'
SMALL_SHA = 'db71de4b6b8c3ba1532639258f23e7b4b5a850493625bf6ba1a7e2f8c9b13b88'
OBJECTIVE_SHA = '947c3e4aa275fa98b5f84a2a7a89f74b8161d6ba498b948862f7d488712c2d81'
DRAW_COUNT = 8
FAMILY_COUNT = 2
PREDICTOR_BLOCKS = 2
INITIAL_VJPS = DRAW_COUNT * FAMILY_COUNT * PREDICTOR_BLOCKS
FORWARD_CAP = INITIAL_VJPS * 2  # Includes actual nonreentrant checkpoint calls.


class _EOT8VJPState(small._BoundPrototype):
    """A new explicit resource scope, not an expanded eight-call prototype instance.

    Inherit frozen tensor validation, public math and read-only native-root
    observation. The old constructor and its eight-call attention entry are not
    used. Only LocalEOT8Route may construct/apply the integrated attention calls.
    """
    def __init__(self, bridge):
        self.max_calls = FORWARD_CAP
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.original_flags = cpu.flags()
        self._sdpa = torch.nn.functional.scaled_dot_product_attention
        self._grad = torch.autograd.grad
        self._graph_api = getattr(torch._C, '_current_graph_task_id', None)
        if not callable(self._graph_api) or self._sdpa is not torch._C._nn.scaled_dot_product_attention:
            raise ValueError('Original public SDPA and graph identity inspection required.')
        self.calls, self.scope = [], None
        self.active = self.entered = self.gradient_used = False
        self._outer = self.bound_task = None
        self.bridge = bridge

    def attention(self, *args, **kwargs):
        raise ValueError('Only the explicitly bounded EOT8 model entry may apply attention.')

    def gradient(self, *args, **kwargs):
        raise ValueError('Only the explicitly bound native EOT8 loss/input may run backward.')


class LocalEOT8Route(small.LocalUpstreamRoute):
    """One synthetic tiny eight-draw/two-family loss and one native owned backward."""
    def __init__(self, model):
        if platform.node().split('.')[0] in ('linux10603', 'linux10609'):
            raise ValueError('Local CPU EOT8 integration only; university refused.')
        if (small.digest(small.__file__) != SMALL_SHA
                or small.digest(__import__('attacks.pgd', fromlist=['native_objective']).__file__) != OBJECTIVE_SHA):
            raise ValueError('Frozen small integration/native objective source changed.')
        # Reuse exact small-model/source-domain validation; no eight-call scope is entered.
        super().__init__(model)
        if hasattr(model, MARKER):
            raise ValueError('An EOT8 diagnostic identity is already installed.')
        self.prototype = _EOT8VJPState(self)
        self.identity = IDENTITY
        self._loss_input = None
        self._objective_used = False

    def check_binding(self):
        if (not self.active or getattr(self.model, MARKER, None) != IDENTITY
                or self.native.phase != 'EOT8-before'
                or any(function.__globals__['F'] is not new for function, old, new in self.views)):
            raise ValueError('Local EOT8 binding or phase changed.')

    def _remove_marker(self):
        if hasattr(self.model, MARKER):
            delattr(self.model, MARKER)

    def __enter__(self):
        if self.entered:
            raise ValueError('Local EOT8 route is single use.')
        self.entered = self.active = True
        stack = self.stack = ExitStack()
        try:
            stack.enter_context(self.native)
            stack.enter_context(self.prototype)
            stack.callback(self._restore_views)
            setattr(self.model, MARKER, IDENTITY)
            stack.callback(self._remove_marker)
            for index, block in enumerate(self.model.predictor.predictor_blocks):
                private = inspect.getclosurevars(block.attn.forward.__func__).nonlocals.get('private')
                if not isinstance(private, FunctionType):
                    raise ValueError('Expected private native wrapper unavailable.')
                old = private.__globals__['F']
                if not isinstance(old, adapter.FunctionalView) or old.original is not torch.nn.functional:
                    raise ValueError('Original instance-private F alias required.')
                guarded = old.route
                def routed(q, k, v, _guarded=guarded, _index=index, **kwargs):
                    if kwargs != {'dropout_p': 0., 'is_causal': False, 'attn_mask': None}:
                        raise ValueError('Original attention arguments required.')
                    proto = self.prototype
                    proto._admit_forward()
                    if (not torch.is_grad_enabled() or torch.is_autocast_enabled('cpu')
                            or torch.is_inference_mode_enabled() or len(proto.calls) >= FORWARD_CAP):
                        raise ValueError('At most 64 grad-enabled local CPU EOT8 predictor calls.')
                    proto._validate((q, k, v))
                    row = {'index': len(proto.calls), 'owner': 'predictor.'+str(_index),
                           'forward_graph_id': proto._graph_id(), 'forward_thread': threading.get_ident(),
                           'inputs': [cpu.fingerprint(t) for t in (q, k, v)],
                           'aliases': cpu.aliases((q, k, v)), 'entry_flags': cpu.flags(),
                           'forward_complete': False, 'backward_complete': False}
                    proto.calls.append(row)
                    # Exact frozen original-forward / public-math-VJP Function.
                    return small._IntegratedAttention.apply(proto, row['index'], _guarded, q, k, v)
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

    def loss(self, value, clean_targets, mask_bank):
        self.check_binding()
        if self._objective_used or self.prototype.calls:
            raise ValueError('One fresh native EOT8 objective only.')
        self._objective_used = True  # A refused objective cannot be retried in this scope.
        if (len(mask_bank) != DRAW_COUNT or any(len(draw) != FAMILY_COUNT for draw in mask_bank)
                or value.shape != (1, 3, 4, 16, 16) or value.device.type != 'cpu'
                or value.dtype != torch.float32 or not value.requires_grad or not value.is_leaf
                or clean_targets.shape != (1, 32, 128) or clean_targets.device.type != 'cpu'
                or clean_targets.dtype != torch.float32 or clean_targets.requires_grad
                or not torch.isfinite(clean_targets).all()):
            raise ValueError('Exactly eight tiny CPU draws with two original families required.')
        for draw in mask_bank:
            for family, pair in enumerate(draw):
                nc, nt = ((8, 24), (24, 8))[family]
                if (pair.context.device.type != 'cpu' or pair.target.device.type != 'cpu'
                        or pair.context.dtype != torch.int64 or pair.target.dtype != torch.int64
                        or pair.context.shape != (1, nc) or pair.target.shape != (1, nt)
                        or not torch.equal(torch.sort(torch.cat((pair.context, pair.target), dim=1)).values,
                                           torch.arange(32).reshape(1, 32))):
                    raise ValueError('Original disjoint complementary 8/24 and 24/8 mask families required.')
        loss = native_objective(self.model, clean_targets, mask_bank)(value).sum()
        self._loss_input = (loss, value)
        return loss

    def gradient(self, loss, value):
        self.check_binding()
        proto = self.prototype
        if (proto.gradient_used or self._loss_input is None
                or loss is not self._loss_input[0] or value is not self._loss_input[1]):
            raise ValueError('One exact fresh native EOT8 loss/input derivative required.')
        task = self.native.prepare_backward(loss, value)
        self.last_task = proto.bound_task = task
        proto.gradient_used = True
        row = {'state': 'prepared', 'graph_id': None, 'identity_source': 'unchanged-native-root',
               'new_outer_hook_added': False, 'loss_version': loss._version, 'input_version': value._version,
               'draw_count': DRAW_COUNT, 'families_per_draw': FAMILY_COUNT,
               'forward_cap': FORWARD_CAP, 'expected_initial_VJPs': INITIAL_VJPS}
        proto.scope = proto._outer = row
        try:
            result, = self._run_native(task)
            if (task.state != 'complete' or not task.receipt['hook_removed'] or task.graph_id != row['graph_id']
                    or task.graph_id is None or self.native._task is not None
                    or (loss._version, value._version) != (row['loss_version'], row['input_version'])):
                raise ValueError('Unchanged native EOT8 completion/versions required.')
            initial = [c for c in proto.calls if c['forward_graph_id'] == -1]
            recomputed = [c for c in proto.calls if c['forward_graph_id'] == task.graph_id]
            if (len(proto.calls) != FORWARD_CAP or len(initial) != INITIAL_VJPS
                    or len(recomputed) != INITIAL_VJPS
                    or any(not c['backward_complete'] or not c['hooks_removed'] for c in initial)
                    or any(not c['forward_complete'] for c in proto.calls)):
                raise ValueError('All 32 initial predictor VJPs and 32 actual checkpoint calls required.')
            if len({c['inner_graph_id'] for c in initial}) != INITIAL_VJPS:
                raise ValueError('32 distinct public math GraphTasks required.')
            self.native.check()
            row.update(state='complete', native_root_hook_removed=task.receipt['hook_removed'])
            return result
        except BaseException as error:
            row.update(state='failed', error_type=type(error).__name__, error=str(error)[:2048])
            raise
        finally:
            proto._outer = proto.bound_task = None
            self._loss_input = None
