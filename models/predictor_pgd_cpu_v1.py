"""Local, opt-in per-iterate tiny EOT8 native loss/input ownership.

Each context is single-use. Original no-grad scoring is kept outside all routes.
This is a synthetic first-order test entry, never a training execution policy.
"""
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

import numpy as np
import torch

from attacks.pgd import native_objective
from attacks.pgd_context_v1 import IDENTITY, pgd_linf_context_v1, tiny_domain
from models import predictor_forward_preserving_eot8_cpu_v1 as integration
from models import predictor_eot8_owned_v1 as native
from models import predictor_eot8_adapter_v1 as adapter
from models.attention_saved_evidence import agreement
from models.diagnostics import state_digest
from models.execution import precision_contract
from models.predictor_pgd_records_v1 import check_proof

INPUT_BOUND = {'rtol': .002, 'atol': 2e-7}
LOSS_BOUND = {'rtol': 1e-5, 'atol': 2e-6}
SEED = 73


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    # Callers provide a fresh private parent; exclusive creation preserves failures.
    with Path(path).open('x', encoding='utf8') as f:
        os.fchmod(f.fileno(), 0o600)
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def tensor_hash(t):
    a = t.detach().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def save_arrays(folder, arrays):
    files = {}
    for name, t in arrays.items():
        a = t.detach().contiguous().numpy()
        p = folder/(name+'.npy')
        with p.open('xb') as f:
            os.fchmod(f.fileno(), 0o600)
            np.save(f, a, allow_pickle=False)
        files[name] = {'file': p.name, 'sha256': digest(p), 'bytes': p.stat().st_size,
                       'dtype': str(a.dtype), 'shape': list(a.shape)}
    return files


class ObjectiveLedger:
    """Persistent start-before-work accounting; any started objective is consumed."""
    def __init__(self, folder, mode):
        if mode not in ('paired', 'replay'):
            raise ValueError('One initial paired or one conditional fresh replay only.')
        self.folder, self.mode, self.starts = Path(folder), mode, []
        self.limit = 8 if mode == 'paired' else 4

    def begin(self, role, step):
        expected = [('original', i//2) if i % 2 == 0 else ('candidate', i//2)
                    for i in range(8)] if self.mode == 'paired' else [('candidate', i) for i in range(4)]
        if len(self.starts) >= self.limit or (role, step) != expected[len(self.starts)]:
            raise ValueError('Tiny objective budget/order exhausted; no retry.')
        row = {'index': len(self.starts), 'role': role, 'step': step, 'pid': os.getpid()}
        write_json(self.folder/f"started-{row['index']:02}.json", row)
        self.starts.append(row)


class IteratedBinding:
    def __init__(self, model, clean, targets, attack_bank, score_bank, ledger):
        tiny_domain(clean, 4/255, 1/255, 2, 2, SEED, True)
        # Frozen constructor validates tiny geometry/source/eval/CPU, without execution.
        integration.LocalEOT8Route(model)
        if (targets.shape != (1, 32, 128) or targets.dtype != torch.float32
                or targets.device.type != 'cpu' or targets.requires_grad
                or not torch.isfinite(targets).all() or clean.requires_grad):
            raise ValueError('Detached original clean target and pixels required.')
        for bank, draws in ((attack_bank, 8), (score_bank, 4)):
            if len(bank) != draws or any(len(draw) != 2 for draw in bank):
                raise ValueError('Exactly eight attack/four independent scoring draws and two families.')
            for draw in bank:
                for i, p in enumerate(draw):
                    nc, nt = ((8, 24), (24, 8))[i]
                    if (p.context.device.type != 'cpu' or p.target.device.type != 'cpu'
                            or p.context.dtype != torch.int64 or p.target.dtype != torch.int64
                            or p.context.shape != (1, nc) or p.target.shape != (1, nt)
                            or not torch.equal(torch.cat((p.context, p.target), 1).sort().values,
                                               torch.arange(32).reshape(1, 32))):
                        raise ValueError('Original complementary mask geometry required.')
        if not model.encoder.use_activation_checkpointing or not model.predictor.use_activation_checkpointing:
            raise ValueError('Original encoder/predictor activation checkpointing required.')
        self.model, self.clean, self.targets = model, clean, targets
        self.attack_bank = tuple(tuple(d) for d in attack_bank)
        self.score_bank = tuple(tuple(d) for d in score_bank)
        self.ledger, self.mode = ledger, ledger.mode
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.active, self.step, self.poisoned = None, 0, False
        self.flags = integration.cpu.flags()
        self.precision = precision_contract()
        self.grad_api, self.sdpa = torch.autograd.grad, torch.nn.functional.scaled_dot_product_attention
        self.state = state_digest(model)
        self.parameters = [(p, p._version) for p in model.parameters()]
        self.parameter_require_grad = [p.requires_grad for p, _ in self.parameters]
        self.modes = [(m, m.training) for m in model.modules()]
        self.hooks = [(m, dict(m._forward_hooks), dict(m._forward_pre_hooks), dict(m._backward_hooks))
                      for m in model.modules()]
        self.fixed = [(t, t._version, tensor_hash(t)) for t in (clean, targets)]
        self.fixed += [(t, t._version, tensor_hash(t)) for bank in (self.attack_bank, self.score_bank)
                       for draw in bank for p in draw for t in (p.context, p.target)]
        self.completed, self.no_grad_calls = [], []
        self.check_idle()

    def common(self):
        if (self.poisoned or os.getpid() != self.pid or threading.get_ident() != self.thread
                or torch.is_inference_mode_enabled() or torch.is_autocast_enabled('cpu')
                or torch.autograd.grad is not self.grad_api
                or torch.nn.functional.scaled_dot_product_attention is not self.sdpa
                or integration.cpu.flags() != self.flags or precision_contract() != self.precision):
            raise ValueError('Foreign, poisoned or changed first-order CPU binding.')
        if ([id(p) for p in self.model.parameters()] != [id(p) for p, _ in self.parameters]
                or [p.requires_grad for p, _ in self.parameters] != self.parameter_require_grad
                or state_digest(self.model) != self.state
                or any(p._version != v or p.grad is not None for p, v in self.parameters)
                or any(m.training != training for m, training in self.modes)
                or any(t._version != v or tensor_hash(t) != h for t, v, h in self.fixed)):
            raise ValueError('Parameters, modes, fixed targets/masks/pixels or versions changed.')

    def check_idle(self):
        self.common()
        if (self.active is not None or adapter._ACTIVE or integration.cpu._ACTIVE
                or hasattr(self.model, integration.MARKER) or hasattr(self.model, integration.small.MARKER)
                or any('forward' in b.attn.__dict__ for blocks in
                       (self.model.encoder.blocks, self.model.predictor.predictor_blocks) for b in blocks)
                or any(dict(m._forward_hooks) != f or dict(m._forward_pre_hooks) != pre
                       or dict(m._backward_hooks) != back for m, f, pre, back in self.hooks)):
            raise ValueError('Routes, private methods, markers or hooks were not restored.')

    def no_grad(self, value, bank, label):
        self.check_idle()
        if torch.is_grad_enabled():
            raise ValueError('Clean/final/independent scoring must be original no-grad forward.')
        values = native_objective(self.model, self.targets, bank)(value)
        if values.shape != (1,) or not torch.isfinite(values).all():
            raise ValueError('Finite original per-example no-grad score required.')
        self.no_grad_calls.append({'index': len(self.no_grad_calls), 'label': label,
                                   'pixels': tensor_hash(value), 'loss': values.tolist(),
                                   'draws': len(bank), 'grad_enabled': False, 'custom_route_active': False})
        self.check_idle()
        return values

    def objective(self, value):
        return self.no_grad(value, self.attack_bank, 'attack')

    def context(self, value):
        self.check_idle()
        if (self.step >= 4 or not torch.is_grad_enabled() or value.shape != self.clean.shape
                or value.dtype != torch.float32 or value.device.type != 'cpu'
                or not value.is_leaf or not value.requires_grad or value.grad is not None
                or not torch.isfinite(value).all()
                or not torch.all((value >= (self.clean-4/255).clamp(0, 1))
                                 & (value <= (self.clean+4/255).clamp(0, 1)))):
            raise ValueError('Four fresh owned tiny PGD input leaves only; no retry.')
        index = self.step
        self.step += 1
        reference = None
        try:
            if self.mode == 'paired':
                reference = Capture(self, value.detach().clone().requires_grad_(True), index, False)
                with reference:
                    reference.gradient()
            return Capture(self, value, index, True, reference)
        except BaseException:
            self.poisoned = True
            raise


class Capture:
    def __init__(self, binding, value, index, candidate, reference=None):
        self.binding, self.value, self.index = binding, value, index
        self.candidate, self.reference = candidate, reference
        self.entered = self.used = self.considered = self.closed = False
        self.events, self.arrays, self.handles = [], {}, []
        self.proof, self.error, self.comparison = {}, None, None

    def __enter__(self):
        b = self.binding
        b.check_idle()
        if self.entered or self.closed:
            raise ValueError('Per-iterate contexts are single use.')
        self.entered = True
        b.ledger.begin('candidate' if self.candidate else 'original', self.index)
        b.active = self
        self.folder = b.ledger.folder/(('candidate' if self.candidate else 'original')+f'-{self.index}')
        self.folder.mkdir(mode=0o700)
        self.stack = ExitStack()
        self.input_version = self.value._version
        self.input_hash = tensor_hash(self.value)
        self.arrays = {'pixels': self.value.detach().clone(), 'targets': b.targets.detach().clone(),
                       'masks': torch.stack([torch.cat((p.context, p.target), 1).squeeze(0)
                                             for draw in b.attack_bank for p in draw])}
        self.route = integration.LocalEOT8Route(b.model) if self.candidate else native.PredictorRoute(
            b.model, 'original', tiny=True, max_calls=128)
        try:
            self.stack.callback(self.remove_hooks)
            self.stack.enter_context(self.route)
            def observe(module, args, output):
                i = sum(k.startswith('prediction-') for k in self.arrays)
                if i >= 16:
                    raise ValueError('At most 16 actual predictions per objective.')
                self.arrays[f'prediction-{i}'] = output.detach().clone()
                def incoming(g):
                    if f'cotangent-{i}' in self.arrays:
                        raise ValueError('One native L1 incoming hook per prediction.')
                    self.arrays[f'cotangent-{i}'] = g.detach().clone()
                    return None
                self.handles.append(output.register_hook(incoming))
            self.handles.append(b.model.predictor.register_forward_hook(observe))
            if self.candidate:
                self.root = self.route.loss(self.value, b.targets, b.attack_bank)
            else:
                self.route.phase = 'EOT8-before'
                self.root = native_objective(b.model, b.targets, b.attack_bank)(self.value).sum()
            self.losses = self.root.view(1)
            self._token = (self.root, self.value, self.losses, self.root._version)
            self.arrays['loss'] = self.root.detach().clone()
            self.events.append('forward')
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def remove_hooks(self):
        for h in self.handles:
            h.remove()

    def admitted(self):
        b = self.binding
        b.common()
        if (b.active is not self or not self.entered or self.closed or self.used
                or self.value._version != self.input_version or tensor_hash(self.value) != self.input_hash
                or self.root is not self._token[0] or self.value is not self._token[1]
                or self.losses is not self._token[2] or self.root._version != self._token[3]
                or self.value.grad is not None or not torch.is_grad_enabled()
                or torch._C._current_graph_task_id() != -1):
            raise ValueError('One fresh first-order exact loss/input token in its owning context required.')

    def note_consider(self, value, losses):
        self.admitted()
        if not self.candidate or self.considered or value is not self.value or losses is not self.losses:
            raise ValueError('Consider the exact per-example candidate before backward, once.')
        self.considered = True
        self.events.append('consider')

    def gradient(self, *, create_graph=False):
        self.admitted()
        if create_graph is not False or self.candidate and not self.considered:
            raise ValueError('First-order only; candidate selection must precede backward.')
        self.used = True  # A failed differentiation still consumes this context.
        self.events.append('backward')
        if self.candidate:
            g = self.route.gradient(self.root, self.value)
        else:
            g, = self.route.prepare_backward(self.root, self.value).run()
        self.arrays['input'] = g.detach().clone()
        r = self.route.native if self.candidate else self.route
        self.proof = {'native_scopes': copy.deepcopy(r.scopes), 'native_calls': copy.deepcopy(r.rows),
                      'native_refusals': copy.deepcopy(r.refusals)}
        if self.candidate:
            self.proof.update(prototype_calls=copy.deepcopy(self.route.prototype.calls),
                              borrowed_scope=copy.deepcopy(self.route.prototype.scope))
        self.proof_check = check_proof(self.proof, self.candidate, os.getpid())
        if len(self.arrays) != 37 or any(not torch.isfinite(t).all() for t in self.arrays.values()):
            raise ValueError('All 37 finite actual value/gradient/native-L1/mask returns required.')
        if self.reference is not None:
            expected = self.reference.arrays
            self.comparison = {'input': agreement(g.numpy(), expected['input'].numpy(), INPUT_BOUND),
                               'loss': agreement(self.root.detach().numpy(), expected['loss'].numpy(), LOSS_BOUND),
                               'forward_and_native_L1_bitwise': all(torch.equal(v, expected[k])
                                   for k, v in self.arrays.items() if k not in ('input', 'loss'))}
            if not all((self.comparison['input']['passed'], self.comparison['loss']['passed'],
                        self.comparison['forward_and_native_L1_bitwise'])):
                raise ValueError('Same-input original/candidate numerical gate failed.')
        return g

    def __exit__(self, kind, error, tb):
        b = self.binding
        failed = error is not None
        try:
            self.stack.__exit__(kind, error, tb)
            self.events.append('exit')
            b.active = None
            b.check_idle()
            if self.value._version != self.input_version or tensor_hash(self.value) != self.input_hash or self.value.grad is not None:
                raise ValueError('Input value/version changed or input .grad was populated.')
            if not failed and (not self.used or self.events != (
                    ['forward', 'consider', 'backward', 'exit'] if self.candidate
                    else ['forward', 'backward', 'exit'])):
                raise ValueError('Complete original forward/consider/backward/exit ordering required.')
        except BaseException as cleanup:
            failed = True
            self.error = repr(cleanup)
            raise
        finally:
            b.active = None
            self.closed = True
            if failed:
                b.poisoned = True
            self.error = self.error or (None if error is None else repr(error))
            r = self.route.native if self.candidate else self.route
            if not self.proof:
                self.proof = {'native_scopes': copy.deepcopy(r.scopes), 'native_calls': copy.deepcopy(r.rows),
                              'native_refusals': copy.deepcopy(r.refusals)}
            files = save_arrays(self.folder, self.arrays)
            receipt = {'kind': 'tiny-pgd-owned-objective-v1', 'algorithm': IDENTITY,
                       'candidate': self.candidate, 'step': self.index, 'pid': os.getpid(),
                       'complete': not failed, 'error': self.error, 'files': files,
                       'events': self.events, 'proof': self.proof,
                       'proof_check': getattr(self, 'proof_check', None), 'comparison': self.comparison,
                       'restoration_passed': not failed, 'model_state_sha256': b.state,
                       'parameter_gradients_absent': all(p.grad is None for p, _ in b.parameters),
                       'input_gradient_absent': self.value.grad is None,
                       'CPU_only': True, 'pretrained': False, 'CUDA_qualified': False,
                       'training_or_research_qualified': False}
            write_json(self.folder/'receipt.json', receipt)
            if not failed:
                b.completed.append({'role': 'candidate' if self.candidate else 'original',
                                    'step': self.index, 'receipt': self.folder.name+'/receipt.json'})


def rollout(binding):
    with torch.no_grad():
        clean_attack = binding.objective(binding.clean)
        clean_score = binding.no_grad(binding.clean, binding.score_bank, 'score-clean')
    result = pgd_linf_context_v1(binding.objective, binding.clean, epsilon=4/255,
                                step_size=1/255, steps=2, restarts=2, seed=SEED,
                                random_start=True, gradient_context=binding.context)
    with torch.no_grad():
        adversarial_score = binding.no_grad(result.adversarial, binding.score_bank, 'score-adversarial')
    binding.check_idle()
    if (binding.step != 4 or len(binding.completed) != binding.ledger.limit
            or len(binding.no_grad_calls) != 6
            or [r['draws'] for r in binding.no_grad_calls] != [8, 4, 8, 8, 8, 4]
            or result.adversarial.requires_grad or result.losses.requires_grad
            or (result.adversarial-binding.clean).abs().max() > 4/255+1e-7):
        raise ValueError('Exact four-step selection/projection/scoring/restore counts required.')
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    noise = torch.stack([torch.rand(binding.clean.shape, generator=generator, dtype=torch.float32) for _ in range(2)])
    return {'restart-noise': noise, 'clean': binding.clean, 'adversarial': result.adversarial, 'attack-loss': result.losses,
            'clean-attack-loss': clean_attack, 'clean-score': clean_score,
            'adversarial-score': adversarial_score}
