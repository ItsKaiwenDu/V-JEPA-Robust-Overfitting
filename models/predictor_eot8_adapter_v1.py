"""Instance-local SDPA routing. Frozen upstream tensor operations are unchanged."""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import threading
from types import FunctionType, MethodType

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

IDENTITY = 'predictor-math-clean-eot8-diagnostic-v1'
PREFIX = 'predictor-eot8/call/'
MODULE_SHA = 'b93f6c7e0747deb216419c000c2878f11a9189024a9adeacfd437e172396dff0'
_ACTIVE = set()


def flags():
    return {k: bool(getattr(torch.backends.cuda, k + '_sdp_enabled')())
            for k in ('flash', 'mem_efficient', 'math', 'cudnn')}


def layout(t):
    return {'shape': list(t.shape), 'stride': list(t.stride()), 'dtype': str(t.dtype),
            'device': str(t.device), 'requires_grad': t.requires_grad}


class FunctionalView:
    """A private F alias; never replace torch.nn.functional or upstream globals."""
    def __init__(self, original, route):
        self.original, self.route = original, route

    def __getattr__(self, name):
        return self.route if name == 'scaled_dot_product_attention' else getattr(self.original, name)


class PredictorRoute:
    def __init__(self, model, mode, *, tiny=False, max_calls=4096):
        if mode not in ('original', 'predictor-math') or type(tiny) is not bool:
            raise ValueError('Unknown diagnostic route/domain.')
        self.model, self.mode, self.tiny, self.max_calls = model, mode, tiny, max_calls
        self.rows, self.changes, self.expected, self.active = [], [], [], False
        self.thread = threading.get_ident(); self.backward_active = False
        self.old_policy = getattr(model, '_diagnostic_execution_policy', 'original')
        self.policy_existed = '_diagnostic_execution_policy' in model.__dict__
        self.original_flags = flags()

    def check(self):
        if not self.active or threading.get_ident() != self.thread:
            raise ValueError('Route inactive or called from another thread.')
        expected = IDENTITY if self.mode == 'predictor-math' else self.old_policy
        if getattr(self.model, '_diagnostic_execution_policy', 'original') != expected:
            raise ValueError('Active diagnostic identity drifted.')
        if flags() != self.original_flags:
            raise ValueError('Backend flags leaked outside the SDPA boundary.')
        for obj, function in self.expected:
            if obj.forward is not function:
                raise ValueError('Owned attention method changed.')

    def _install(self, obj, name, role, index):
        if 'forward' in obj.__dict__:
            raise ValueError('Refuse pre-wrapped attention instances.')
        method = obj.forward
        fn = method.__func__
        if type(obj).__name__ != 'RoPEAttention' or fn.__name__ != 'forward':
            raise ValueError('Require original owned RoPEAttention.')
        path = Path(inspect.getsourcefile(fn)).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODULE_SHA or fn.__globals__.get('F') is not torch.nn.functional:
            raise ValueError('Upstream source or F alias changed.')
        if not obj.use_sdpa or obj.proj_drop_prob != 0 or obj.is_causal:
            raise ValueError('Require original noncausal zero-dropout SDPA configuration.')
        old_f = fn.__globals__['F']; old_sdpa = old_f.scaled_dot_product_attention
        if old_sdpa is not torch._C._nn.scaled_dot_product_attention:
            raise ValueError('Require original builtin SDPA alias.')
        owner_thread = self.thread

        def routed(q, k, v, **kwargs):
            if threading.get_ident() != owner_thread or not self.active or len(self.rows) >= self.max_calls:
                raise ValueError('Unowned/concurrent call or call count exceeded.')
            if old_f.scaled_dot_product_attention is not old_sdpa:
                raise ValueError('Functional SDPA alias changed.')
            if set(kwargs) != {'dropout_p', 'is_causal', 'attn_mask'} or kwargs['dropout_p'] != 0 or kwargs['is_causal'] is not False or kwargs['attn_mask'] is not None:
                raise ValueError('SDPA arguments changed.')
            if any(t.dtype != torch.float32 or t.device.type != ('cpu' if self.tiny else 'cuda') or t.ndim != 4 for t in (q,k,v)):
                raise ValueError('Require actual FP32 Q/K/V in the declared domain.')
            if q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1 or q.shape[1] != obj.num_heads or q.shape[3] != obj.head_dim:
                raise ValueError('Owned SDPA geometry changed.')
            if not self.tiny and (q.shape[1:] != (12,2048,32) if role == 'predictor' else q.shape[1] != 16 or q.shape[3] != 64):
                raise ValueError('Pretrained owner geometry changed.')
            row = {'index': len(self.rows), 'owner': name, 'role': role, 'block': index,
                   'q': layout(q), 'k': layout(k), 'v': layout(v), 'entry_flags': flags(),
                   'outcome': 'entered', 'phase': self.phase, 'during_backward': self.backward_active}
            self.rows.append(row)
            label = PREFIX + str(row['index']) + '/' + name
            ctx = sdpa_kernel(SDPBackend.MATH) if self.mode == 'predictor-math' and role == 'predictor' else _unchanged()
            try:
                with ctx:
                    row['effective_flags'] = flags()
                    if self.mode == 'predictor-math' and role == 'predictor' and row['effective_flags'] != {'flash':False,'mem_efficient':False,'math':True,'cudnn':False}:
                        raise ValueError('Actual-call math selection failed.')
                    with torch.profiler.record_function(label):
                        result = old_sdpa(q,k,v,**kwargs)
                    row['outcome'] = 'returned'
                    return result
            except BaseException as exc:
                row.update(outcome='interrupted', error_type=type(exc).__name__)
                raise
            finally:
                row['exit_flags'] = flags()
                if row['exit_flags'] != row['entry_flags']:
                    raise ValueError('SDPA context did not restore entry flags.')

        env = dict(fn.__globals__); env['F'] = FunctionalView(old_f, routed)
        private = FunctionType(fn.__code__, env, fn.__name__, fn.__defaults__, fn.__closure__)
        private.__kwdefaults__ = fn.__kwdefaults__
        bound = MethodType(private, obj)
        obj.forward = bound
        self.changes.append(obj); self.expected.append((obj,bound))

    def __enter__(self):
        if self.active or _ACTIVE or self.old_policy != 'shared-fp64-native':
            raise ValueError('Require one original shared-native model with no active route.')
        _ACTIVE.add(id(self)); self.active = True; self.phase = None
        try:
            for role, blocks in [('encoder',self.model.encoder.blocks), ('predictor',self.model.predictor.predictor_blocks)]:
                for i, block in enumerate(blocks):
                    self._install(block.attn, role+'.'+str(i), role, i)
            if self.mode == 'predictor-math':self.model._diagnostic_execution_policy = IDENTITY
            self.check(); return self
        except BaseException:
            self.__exit__(None,None,None); raise

    def __exit__(self, *unused):
        intact = all(obj.forward is fn for obj,fn in self.expected)
        for obj in reversed(self.changes):del obj.forward
        if self.policy_existed:self.model._diagnostic_execution_policy = self.old_policy
        else:self.model.__dict__.pop('_diagnostic_execution_policy',None)
        self.active = False; _ACTIVE.discard(id(self))
        if not intact or flags() != self.original_flags:
            raise ValueError('Route was modified or flags were not restored.')


@contextmanager
def _unchanged():
    yield
