"""Opt-in local whole-unit input-gradient integration. No execution-policy registration.

The native scientific route is frozen. Analytic ports are explicitly controls;
neither port binds a university worker, CUDA runtime or research trajectory.
"""
from contextlib import AbstractContextManager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time

import torch

from attacks.pgd import native_objective
from attacks.pgd_workload_v1 import checked_gradient, local_domain
from models.candidate_evaluation import EvaluationSettings, seeds_for
from models.execution import precision_contract

IDENTITY = 'local-whole-pgd-input-gradient-workload-v1'
EXECUTION_BINDING = None
ROOT = Path(__file__).resolve().parents[1]
# Filled from preserved source bytes before the first check, never from a live route.
FROZEN_SOURCES = {'attacks/pgd.py': '947c3e4aa275fa98b5f84a2a7a89f74b8161d6ba498b948862f7d488712c2d81', 'models/candidate_evaluation.py': '26510d6bd5557421fc10bcdc28dd4b5c553b69c3a9914a3cd80457d5bc746a30', 'models/candidate_contract.py': '13f664f7146eb4816cbaf9fde15d1b6e572db0949e22c319eb2420d23c1e4572', 'models/execution.py': 'e665f7c842c0d26733bd6c4d16887b10e2590b76f15e4fb1981b304ff37903ba', 'models/runtime.py': '0764f5c63384882e06f1d409535e96b3423861866c16bc8ba5e3fa71955f83de', 'models/training.py': '15e280dfcc409a94f36defa42c59304e96dc198c8e4e063a9da6f4adf4f03679', 'models/vjepa2.py': '5501eb4ce79317d8ebab591d7fe4d8d0b74b1b5b692d2f7513d8c583d188debf', 'models/diagnostics.py': '12185169ca1ebe2564bc0b1815ec19b8599c08f625af056bc30c25fca6f390d0', 'models/native_diagnostic.py': 'c56b29496397d5516416a884ab591ec6de43e0fc4dca1bc05536234e10ea75d9', 'models/predictor_forward_preserving_eot8_cpu_v1.py': '256079682d27aa2be42b0c909298a5adb3632cc385e95aa3590d2724b75a1a9e', 'models/predictor_forward_preserving_upstream_v1.py': 'db71de4b6b8c3ba1532639258f23e7b4b5a850493625bf6ba1a7e2f8c9b13b88', 'models/predictor_forward_preserving_cpu_v1.py': '6ba308e948c0ce346512f17ef4fc11eccb542954048c859ad6c6a751db1fbb66', 'models/predictor_eot8_owned_v1.py': 'ef3675e69ef56b0b4ef4f543cab96b7f2bc757a64202d74500f3802ccd5dfc12', 'models/predictor_eot8_adapter_v1.py': '89b7e0c67e88f2f776bb0bb6ad786a2202115e64ffc6c6f9a962381cc2e9a6b8'}
ANALYTIC = 'analytic-local-control-v1'
FROZEN_CPU = 'tiny-frozen-forward-preserving-eot8-v1'


def file_hash(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def write_json(path, row):
    with Path(path).open('x', encoding='utf8') as out:
        os.fchmod(out.fileno(), 0o600)
        json.dump(row, out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())


def verify_sources():
    if not FROZEN_SOURCES or any(file_hash(ROOT/name) != want for name, want in FROZEN_SOURCES.items()):
        raise ValueError('Frozen workload dependencies changed.')


def local_boundary(clean, settings, video_id, sensitivity_banks):
    if (type(settings) is not EvaluationSettings or settings.method not in ('primary', 'convergence')
            or type(video_id) is not str or not video_id.startswith('local-workload-control/')
            or type(sensitivity_banks) is not int or sensitivity_banks not in (0, 2)
            or clean.requires_grad):
        raise ValueError('Explicit detached local primary/convergence control required.')
    settings.__post_init__()
    local_domain(clean, settings.epsilon_255/255, settings.step_size_255/255,
                 settings.steps, settings.restarts, seeds_for(settings, video_id)['start'], True)
    if not torch.isfinite(clean).all() or clean.min() < 0 or clean.max() > 1:
        raise ValueError('Finite local raw pixels inside [0, 1] required.')


class Journal:
    """Exclusive, durable starts precede model work. Any started unit is consumed."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(mode=0o700)  # Fresh directory only; no reset/retry.
        os.chmod(self.folder, 0o700)

    def put(self, name, row):
        write_json(self.folder/(name+'.json'), row)


class Snapshot:
    """Observed bytes/versions/objects and API state, independent of route receipts."""
    def __init__(self, model, fixed=()):
        self.model = model
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.grad_api, self.sdpa = torch.autograd.grad, torch.nn.functional.scaled_dot_product_attention
        self.graph_api = getattr(torch._C, '_current_graph_task_id', None)
        self.flags = (torch.is_inference_mode_enabled(), torch.is_autocast_enabled('cpu'),
                      torch.are_deterministic_algorithms_enabled(),
                      torch.is_deterministic_algorithms_warn_only_enabled(),
                      torch.get_num_threads(), torch.get_default_dtype())
        self.precision = precision_contract()
        self.sdpa_flags = {name: getattr(torch.backends.cuda, name+'_sdp_enabled')()
                           for name in ('flash', 'mem_efficient', 'math', 'cudnn')}
        self.modules = [(name, module, module.training,
                         {key: dict(getattr(module, key)) for key in
                          ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks', '_backward_pre_hooks')},
                         module.__dict__.get('forward')) for name, module in model.named_modules()]
        self.parameters = [(name, p, p.requires_grad) for name, p in model.named_parameters()]
        self.buffers = [(name, p) for name, p in model.named_buffers()]
        self.fixed = []
        for value in [p for _, p, _ in self.parameters] + [p for _, p in self.buffers] + list(fixed):
            self.add(value)

    def add(self, value):
        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
            raise ValueError('Local CPU tensor boundary required.')
        self.fixed.append((value, value._version, tuple(value.shape), value.dtype,
                           value.requires_grad, tensor_hash(value)))

    def check(self, *, modes=True, idle=True):
        if (os.getpid() != self.pid or threading.get_ident() != self.thread
                or torch.autograd.grad is not self.grad_api
                or torch.nn.functional.scaled_dot_product_attention is not self.sdpa
                or getattr(torch._C, '_current_graph_task_id', None) is not self.graph_api
                or self.flags != (torch.is_inference_mode_enabled(), torch.is_autocast_enabled('cpu'),
                    torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled(),
                    torch.get_num_threads(), torch.get_default_dtype())
                or self.precision != precision_contract()
                or self.sdpa_flags != {name: getattr(torch.backends.cuda, name+'_sdp_enabled')()
                                      for name in ('flash', 'mem_efficient', 'math', 'cudnn')}):
            raise ValueError('Workload process/thread/API/precision boundary changed.')
        if ([(n, id(p), p.requires_grad) for n, p in self.model.named_parameters()]
                != [(n, id(p), r) for n, p, r in self.parameters]
                or [(n, id(b)) for n, b in self.model.named_buffers()]
                != [(n, id(b)) for n, b in self.buffers]
                or any(p.grad is not None for _, p, _ in self.parameters)
                or [(n, id(m)) for n, m in self.model.named_modules()]
                != [(n, id(m)) for n, m, _, _, _ in self.modules]):
            raise ValueError('Workload module/parameter/buffer/gradient boundary changed.')
        for value, version, shape, dtype, required, want in self.fixed:
            if (value._version != version or tuple(value.shape) != shape or value.dtype != dtype
                    or value.device.type != 'cpu' or value.requires_grad != required or tensor_hash(value) != want):
                raise ValueError('Fixed model/pixel/target/mask bytes or versions changed.')
        for _, module, mode, hooks, forward in self.modules:
            if modes and module.training != mode:
                raise ValueError('Module mode changed.')
            if idle and (module.__dict__.get('forward') is not forward
                         or any(dict(getattr(module, key)) != want for key, want in hooks.items())
                         or any(key in module.__dict__ for key in
                                ('_local_eot8_forward_vjp_identity', '_local_forward_vjp_identity',
                                 '_native_eot8_adapter_identity'))):
                raise ValueError('Private methods, markers or hooks were not restored.')


class FrozenCPUFactory:
    kind = FROZEN_CPU

    def create(self, model):
        from models.predictor_forward_preserving_eot8_cpu_v1 import LocalEOT8Route
        return LocalEOT8Route(model)

    def observe(self, route):
        scope, calls = route.prototype.scope, route.prototype.calls
        if (scope['state'] != 'complete' or not scope['native_root_hook_removed']
                or len(calls) != 64 or route.native._task is not None
                or route.last_task.state != 'complete' or not route.last_task.receipt['hook_removed']):
            raise ValueError('Frozen CPU route did not complete its unchanged ownership gates.')
        return {'kind': self.kind, 'native_root_hook_removed': True, 'forward_calls': 64,
                'initial_predictor_VJPs': 32, 'native_scopes': len(route.native.scopes),
                'native_calls': len(route.native.rows), 'native_refusals': len(route.native.refusals)}

    def idle(self, model):
        from models import predictor_forward_preserving_eot8_cpu_v1 as integration
        from models import predictor_eot8_adapter_v1 as adapter
        if (adapter._ACTIVE or integration.cpu._ACTIVE or hasattr(model, integration.MARKER)
                or hasattr(model, integration.small.MARKER)):
            raise ValueError('Frozen native route was not removed.')


class Binding:
    def __init__(self, model, clean, settings, factory, journal):
        if type(factory) is not FrozenCPUFactory:
            # A separate named port makes analytic evidence impossible to label as native.
            from models.predictor_pgd_workload_controls_v1 import AnalyticFactory
            if type(factory) is not AnalyticFactory:
                raise ValueError('Only the concrete frozen CPU route or explicit analytic control port.')
        self.model, self.clean, self.settings = model, clean, settings
        self.factory, self.journal = factory, journal
        self.limit = settings.steps * settings.restarts
        self.targets = self.snapshot = self.attack_bank = None
        self.objectives, self.steps, self.scores = [], [], []
        self.started, self.active, self.poisoned = 0, None, False

    def prepare(self, model, clean, targets):
        if self.snapshot is not None or model is not self.model or clean is not self.clean:
            raise ValueError('One model/clean/target preparation only.')
        if targets.requires_grad or not torch.isfinite(targets).all():
            raise ValueError('Detached finite clean targets required.')
        self.targets = targets
        self.snapshot = Snapshot(model, (clean, targets))
        self.idle()

    def idle(self, *, modes=True):
        if self.poisoned or self.active is not None:
            raise ValueError('Poisoned or active workload cannot score/start/retry.')
        if self.snapshot is None:
            raise ValueError('Workload has no prepared targets.')
        self.factory.idle(self.model)
        self.snapshot.check(modes=modes)

    def objective(self, model, targets, bank):
        self.idle()
        index = len(self.objectives)
        expected = 8 if index == 0 else 4
        if (model is not self.model or targets is not self.targets or index >= 4
                or len(bank) != expected or any(len(draw) != 2 for draw in bank)):
            raise ValueError('Original eight/four draw banks with two mask families required.')
        bank = tuple(tuple(draw) for draw in bank)
        for draw in bank:
            for pair in draw:
                for value in (pair.context, pair.target):
                    if value.dtype != torch.int64 or value.requires_grad:
                        raise ValueError('Original detached integral masks required.')
                    self.snapshot.add(value)
        if index == 0:
            self.attack_bank = bank
        self.objectives.append(bank)
        original = native_objective(model, targets, bank)

        def score(value):
            self.idle()
            if torch.is_grad_enabled():
                raise ValueError('Original clean/final/independent scoring must be no-grad.')
            values = original(value)
            if values.shape != (1,) or not torch.isfinite(values).all():
                raise ValueError('Finite original per-example score required.')
            self.scores.append({'bank': index, 'draws': expected, 'pixels': tensor_hash(value),
                                'losses': values.tolist(), 'grad_enabled': False, 'route_active': False})
            self.idle()
            return values
        return score

    def context(self, value):
        self.idle()
        if (self.started >= self.limit or self.attack_bank is None or not torch.is_grad_enabled()
                or value.shape != self.clean.shape or value.dtype != self.clean.dtype
                or value.device.type != 'cpu' or not value.requires_grad or not value.is_leaf
                or value.grad is not None or not torch.isfinite(value).all()
                or not torch.all((value >= (self.clean-4/255).clamp(0, 1))
                                 & (value <= (self.clean+4/255).clamp(0, 1)))):
            raise ValueError('Fresh first-order input leaf inside the original projection bound required.')
        index = self.started
        self.journal.put(f'step-{index:03}-started', {'index': index, 'pixels': tensor_hash(value),
                                                   'factory': self.factory.kind, 'pid': os.getpid()})
        self.started += 1  # Consumed before route construction or model work.
        return OwnedStep(self, value, index)

    def finish(self, sensitivity_banks):
        # The frozen evaluator has restored caller modes in its finally block.
        # Per-iterate/scoring checks still require eval; LocalUnit independently
        # checks the exact caller modes against its pre-evaluation snapshot.
        self.idle(modes=False)
        counts = [sum(row['bank'] == i for row in self.scores) for i in range(len(self.objectives))]
        if (self.started != self.limit or len(self.steps) != self.limit
                or counts != [4, 2] + [2]*sensitivity_banks):
            raise ValueError('Incomplete gradient/clean/baseline/final/independent scoring unit.')


class OwnedStep(AbstractContextManager):
    def __init__(self, binding, value, index):
        self.binding, self.value, self.index = binding, value, index
        self.entered = self.considered = self.used = self.complete = False
        self.exited = False
        self.route = self.root = self.losses = None

    def __enter__(self):
        b = self.binding
        if self.entered:
            raise ValueError('Input-gradient context is single use.')
        self.entered = True
        b.idle()
        b.active = self
        self.started_at = time.perf_counter()
        self.input_version, self.input_hash = self.value._version, tensor_hash(self.value)
        try:
            self.route = b.factory.create(b.model)
            self.route.__enter__()
            self.root = self.route.loss(self.value, b.targets, b.attack_bank)
            if self.root.shape != () or not self.root.requires_grad or not torch.isfinite(self.root):
                raise ValueError('One finite exact scalar native objective root required.')
            self.losses = self.root.view(1)
            self.root_version, self.loss_version = self.root._version, self.losses._version
            return self
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise

    def tokens(self):
        b = self.binding
        if (not self.entered or b.active is not self or b.poisoned or self.root is None
                or self.value._version != self.input_version or tensor_hash(self.value) != self.input_hash
                or self.root._version != self.root_version or self.losses._version != self.loss_version):
            raise ValueError('Exact unchanged input/loss/owner tokens required.')
        b.snapshot.check(idle=False)

    def note_consider(self, value, losses):
        self.tokens()
        if self.considered or value is not self.value or losses is not self.losses:
            raise ValueError('Original consider-before-backward tokens required once.')
        self.considered = True

    def gradient(self):
        self.tokens()
        if not self.considered or self.used:
            raise ValueError('One gradient after original candidate selection required.')
        self.used = True
        gradient = checked_gradient(self.route.gradient(self.root, self.value), self.value)
        self.tokens()
        proof = self.binding.factory.observe(self.route)
        self.row = {'index': self.index, 'pixels': self.input_hash,
                    'losses': self.losses.detach().tolist(), 'gradient': tensor_hash(gradient),
                    'consider_before_backward': True, 'route': proof}
        self.complete = True
        return gradient

    def __exit__(self, kind, error, tb):
        b = self.binding
        if self.exited:
            raise ValueError('Owned context exit is single use.')
        self.exited = True
        try:
            if self.route is not None:
                self.route.__exit__(kind, error, tb)
            b.active = None
            b.idle()
            if kind is None and not self.complete:
                raise ValueError('Unfinished owned gradient cannot complete.')
            if kind is None:
                self.row['context_seconds_including_observations'] = time.perf_counter()-self.started_at
                b.journal.put(f'step-{self.index:03}-complete', self.row)
                b.steps.append(self.row)
        except BaseException:
            b.poisoned = True
            raise
        finally:
            if kind is not None:
                b.poisoned = True
            b.active = None
            self.root = self.losses = self.value = None
        return False


class LocalUnit:
    """Single-use complete primary/convergence unit; durable failures remain failures."""
    def __init__(self, *, factory):
        self.factory, self.state = factory, 'new'

    def run(self, model, clean, masks_for, settings, *, video_id, journal, sensitivity_banks=0):
        if self.state != 'new':
            raise ValueError('Unit has already been consumed; no retry.')
        local_boundary(clean, settings, video_id, sensitivity_banks)
        verify_sources()
        self.state = 'started'
        started = time.perf_counter()
        store = Journal(journal)
        store.put('unit-started', {'kind': IDENTITY, 'pid': os.getpid(), 'settings': asdict(settings),
                                 'video_id': video_id, 'sensitivity_banks': sensitivity_banks,
                                 'route': self.factory.kind, 'gradient_budget': settings.steps*settings.restarts,
                                 'execution_binding': None, 'research_admitted': False})
        baseline = binding = None
        try:
            if (sum(p.numel() for p in model.parameters()) > 2_000_000
                    or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters())):
                raise ValueError('Bounded local FP32 CPU model only.')
            baseline = Snapshot(model, (clean,))
            binding = self.binding = Binding(model, clean, settings, self.factory, store)
            from models.predictor_pgd_workload_evaluation_v1 import evaluate_local_unit
            metrics, adversarial = evaluate_local_unit(model, clean, masks_for, settings,
                video_id=video_id, sensitivity_banks=sensitivity_banks, binding=binding)
            binding.finish(sensitivity_banks)
            baseline.check()
            receipt = {'kind': IDENTITY, 'status': 'complete', 'route': self.factory.kind,
                       'metrics': metrics, 'steps': binding.steps, 'scores': binding.scores,
                       'adversarial': tensor_hash(adversarial), 'restored': True,
                       'execution_binding': None, 'research_admitted': False,
                       'full_unit_seconds_including_guards_journal_and_observations': time.perf_counter()-started,
                       'timing_endpoint': 'after restoration and observations; before terminal JSON publication',
                       'overhead_subtracted': False, 'GPU_performance_qualified': False}
            store.put('unit-complete', receipt)
            self.state = 'complete'
            return receipt, adversarial
        except BaseException as error:
            self.state = 'failed'
            if binding is not None:
                binding.poisoned = True
            restored = None
            if baseline is not None:
                try:
                    baseline.check()
                    if binding is not None and binding.snapshot is not None:
                        binding.factory.idle(model)
                        binding.snapshot.check(modes=False)
                    restored = True
                except BaseException:
                    restored = False
            store.put('unit-failed', {'status': 'failed', 'error_type': type(error).__name__,
                                    'error': str(error)[:2048], 'restored': restored,
                                    'gradient_starts': 0 if binding is None else binding.started,
                                    'gradient_completions': 0 if binding is None else len(binding.steps),
                                    'consumed': True, 'research_admitted': False})
            raise
