"""Resumable FP32 native-loss training core; research evaluation is separate."""

from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
import platform
import random

import numpy as np
import torch
import yaml

from models.checkpointing import cpu_tree
from models.diagnostics import state_digest
from models.execution import ExecutionGuard, model_execution_contract, precision_contract
from models.pilot import effective_update
from models.vjepa2 import MaskPair, PRETRAIN_CONFIG, use_upstream


def stable_seed(*fields):
    encoded = json.dumps(['training-v1', *fields], separators=(',', ':'), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big') % (2**31 - 1)


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 30
    effective_batch: int = 8
    warmup_epochs: int = 1
    decay_after_epochs: tuple = (15, 23)
    decay_factor: float = 0.1
    encoder_lr: float = 1e-5
    predictor_lr: float = 1e-4
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.04

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in (self.epochs, self.effective_batch)):
            raise ValueError('Epoch and effective batch counts must be positive integers.')
        if type(self.warmup_epochs) is not int or not 0 <= self.warmup_epochs <= self.epochs:
            raise ValueError('Invalid warmup duration.')
        if (tuple(sorted(set(self.decay_after_epochs))) != self.decay_after_epochs
                or any(type(e) is not int or not 0 < e < self.epochs for e in self.decay_after_epochs)):
            raise ValueError('Invalid learning-rate decay epochs.')
        if any(not math.isfinite(v) or v <= 0 for v in (self.encoder_lr, self.predictor_lr, self.eps)):
            raise ValueError('Learning rates and optimizer epsilon must be finite and positive.')
        if (not math.isfinite(self.decay_factor) or not 0 < self.decay_factor <= 1
                or not math.isfinite(self.weight_decay) or self.weight_decay < 0
                or len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas)):
            raise ValueError('Invalid decay or optimizer settings.')

    @classmethod
    def from_plan(cls, plan):
        cfg = plan['training']
        required = {'micro_batch_size': 1, 'precision': 'float32', 'context_patch_embedding': 'conv3d',
                    'activation_checkpointing': True, 'cache_fixed_clean_targets': True,
                    'final_partial_batch': 'retain_and_normalize_by_actual_example_count',
                    'optimizer': 'AdamW', 'gradient_clipping': None,
                    'training_attack': {'steps': 5, 'restarts': 1, 'attack_masks': 1,
                                        'epsilon_255': 4, 'step_size_255': 1}}
        if any(cfg.get(k) != v for k, v in required.items()):
            raise ValueError('Runner supports only the declared FP32 microbatch-one PGD-5 configuration.')
        return cls(epochs=cfg['epochs'], effective_batch=cfg['effective_batch_size'],
                   warmup_epochs=cfg['warmup_epochs'], decay_after_epochs=tuple(cfg['lr_decay_after_epochs']),
                   decay_factor=cfg['lr_decay_factor'], encoder_lr=cfg['encoder_lr'],
                   predictor_lr=cfg['predictor_lr'], betas=tuple(cfg['betas']),
                   eps=cfg['eps'], weight_decay=cfg['weight_decay'])


class TrainingMaskSampler:
    """Initialize the pinned upstream sampler once; replay each explicit seed."""

    def __init__(self, upstream, *, frames=16, crop_size=256, patch_size=16):
        root = use_upstream(upstream)
        config = yaml.safe_load((root / PRETRAIN_CONFIG).read_text())
        cls = importlib.import_module('src.masks.multiseq_multiblock3d').MaskCollator
        self.generators = cls(config['mask'], [frames], crop_size=crop_size,
                              patch_size=patch_size, tubelet_size=2).mask_generators[frames]

    def __call__(self, seed):
        if not 0 <= seed < 2**31 - 1:
            raise ValueError('Invalid upstream mask seed.')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            result = []
            for generator in self.generators:
                with generator._itr_counter.get_lock():
                    generator._itr_counter.value = seed - 1
                result.append(MaskPair(*generator(1)))
            return result


def seed_rng(seed, device):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.random.default_generator.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    elif device == 'mps':
        torch.mps.manual_seed(seed)


def rng_state(device):
    numpy = np.random.get_state()
    result = {'python': random.getstate(), 'torch_cpu': torch.get_rng_state(),
              'numpy': [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]]}
    if device == 'cuda':
        result['torch_cuda'] = torch.cuda.get_rng_state_all()
    elif device == 'mps':
        result['torch_mps'] = torch.mps.get_rng_state()
    return result


def restore_rng(state, device):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch_cpu'])
    if device == 'cuda':
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    elif device == 'mps':
        torch.mps.set_rng_state(state['torch_mps'])


def runtime_contract(device, *, model=None):
    result = {'torch': str(torch.__version__), 'numpy': np.__version__, 'python': platform.python_version(),
              'device': device, 'architecture': platform.machine(), 'threads': torch.get_num_threads(),
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
              'matmul_precision': torch.get_float32_matmul_precision()}
    result.update(precision_contract())
    if model is not None:
        result['model_execution'] = model_execution_contract(model)
    if device == 'cuda':
        result.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      gpu=torch.cuda.get_device_name(), devices=torch.cuda.device_count(),
                      cudnn_benchmark=torch.backends.cudnn.benchmark,
                      cudnn_deterministic=torch.backends.cudnn.deterministic,
                      matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                      cudnn_tf32=torch.backends.cudnn.allow_tf32,
                      cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    return result


class TrainingEngine:
    """One effective update at a time, resumable only at completed boundaries.

    The caller supplies clean detached targets bound to verified training clips.
    No worker RNG or sampler cursor is implicit: the seed, ordered IDs and
    completed update count determine every epoch order and mask/attack seed.
    """

    def __init__(self, model, settings, video_ids, load_example, masks_for, *,
                 condition, seed, provenance, device='cpu'):
        if condition not in ('clean-finetune', 'adversarial-finetune'):
            raise ValueError('Unknown training condition.')
        if not video_ids or len(set(video_ids)) != len(video_ids) or any(not isinstance(v, str) for v in video_ids):
            raise ValueError('Use a nonempty unique ordered list of training video IDs.')
        if type(seed) is not int or not 0 <= seed < 2**31 - 1:
            raise ValueError('Invalid training seed.')
        if device not in ('cpu', 'cuda', 'mps'):
            raise ValueError('Use a single supported device.')
        self.model, self.settings, self.ids = model, settings, list(video_ids)
        self.load_example, self.masks_for = load_example, masks_for
        self.condition, self.seed, self.device = condition, seed, device
        parameters = list(model.encoder.parameters()) + list(model.predictor.parameters())
        if (any(not p.requires_grad or p.dtype != torch.float32 or p.device.type != device for p in parameters)
                or any(p.requires_grad or p.grad is not None for p in model.target_encoder.parameters())):
            raise ValueError('Need FP32 trainable context/predictor and a frozen gradient-free target.')
        self.target_sha256 = state_digest(model.target_encoder)
        self.optimizer = torch.optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': settings.encoder_lr},
            {'params': model.predictor.parameters(), 'lr': settings.predictor_lr},
        ], betas=settings.betas, eps=settings.eps, weight_decay=settings.weight_decay, foreach=False)
        self.contract = json.loads(json.dumps({'settings': asdict(settings), 'ids': self.ids,
            'condition': condition, 'seed': seed, 'provenance': provenance,
            'runtime': runtime_contract(device, model=model), 'target_sha256': self.target_sha256}, sort_keys=True))
        self.contract_sha256 = hashlib.sha256(json.dumps(self.contract, sort_keys=True).encode()).hexdigest()
        self.execution_guard = ExecutionGuard(model, device, target_versions=True)
        self.step = 0
        self.history = []
        self.failed = False
        seed_rng(seed, device)

    @property
    def updates_per_epoch(self):
        return math.ceil(len(self.ids) / self.settings.effective_batch)

    @property
    def finished(self):
        return self.step == self.updates_per_epoch * self.settings.epochs

    def epoch_order(self, epoch):
        generator = torch.Generator().manual_seed(stable_seed(self.seed, 'order', epoch))
        return torch.randperm(len(self.ids), generator=generator).tolist()

    def learning_rates(self, step):
        epoch = step // self.updates_per_epoch
        warmup_steps = self.settings.warmup_epochs * self.updates_per_epoch
        warmup = min(1., (step + 1) / warmup_steps) if warmup_steps else 1.
        decay = self.settings.decay_factor ** sum(epoch >= e for e in self.settings.decay_after_epochs)
        return [self.settings.encoder_lr * warmup * decay, self.settings.predictor_lr * warmup * decay]

    def update(self):
        self.execution_guard.check()
        if self.failed or self.finished:
            raise RuntimeError('Cannot update a failed or completed engine.')
        epoch, batch = divmod(self.step, self.updates_per_epoch)
        start = batch * self.settings.effective_batch
        indices = self.epoch_order(epoch)[start:start + self.settings.effective_batch]
        mask_seeds = [stable_seed(self.seed, 'mask', epoch, self.ids[i]) for i in indices]
        attack_seed = stable_seed(self.seed, 'attack', self.step)
        rates = self.learning_rates(self.step)
        for group, lr in zip(self.optimizer.param_groups, rates):
            group['lr'] = lr

        def load(i):
            pixels, targets = self.load_example(indices[i])
            if (pixels.shape[0] != 1 or pixels.dtype != torch.float32
                    or targets.dtype != torch.float32 or targets.requires_grad):
                raise ValueError('Expected one FP32 clean clip and detached clean targets.')
            return pixels.to(self.device), targets.to(self.device)

        self.failed = True  # Any interruption before completion forbids a new save.
        loss = effective_update(self.model, self.optimizer, load,
            lambda i: [m.to(self.device) for m in self.masks_for(mask_seeds[i])],
            adversarial=self.condition == 'adversarial-finetune', accumulation=len(indices), seed=attack_seed)
        self.execution_guard.check()
        event = {'step': self.step + 1, 'epoch': epoch + 1, 'batch_in_epoch': batch + 1,
                 'video_ids': [self.ids[i] for i in indices], 'mask_seeds': mask_seeds,
                 'attack_seed': attack_seed, 'learning_rates': rates, 'optimizer_loss': loss}
        self.history.append(event)
        self.step += 1
        self.failed = False
        return event

    def state(self):
        self.execution_guard.check()
        if self.failed or any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError('Only a completed, gradient-free update boundary can be saved.')
        if (any(p.requires_grad for p in self.model.target_encoder.parameters())
                or state_digest(self.model.target_encoder) != self.target_sha256):
            raise RuntimeError('Fixed target weights changed.')
        return cpu_tree({'schema_version': 1, 'kind': 'native_training_resume',
            'contract': self.contract, 'contract_sha256': self.contract_sha256,
            'target_sha256': self.target_sha256, 'step': self.step, 'history': self.history,
            'encoder': self.model.encoder.state_dict(), 'predictor': self.model.predictor.state_dict(),
            'optimizer': self.optimizer.state_dict(), 'rng': rng_state(self.device),
            'precision': 'float32', 'scaler': None})

    def restore(self, state):
        self.execution_guard.check()
        # Reject changed run inputs before touching model or optimizer state.
        if (state.get('schema_version') != 1 or state.get('kind') != 'native_training_resume'
                or state['contract'] != self.contract or state['contract_sha256'] != self.contract_sha256
                or state['target_sha256'] != state_digest(self.model.target_encoder)
                or state['precision'] != 'float32' or state['scaler'] is not None):
            raise ValueError('Resume contract differs: check source, configuration, data, target and runtime.')
        step = state['step']
        if type(step) is not int or not 0 <= step <= self.updates_per_epoch * self.settings.epochs:
            raise ValueError('Invalid completed update count.')
        if len(state['history']) != step or [v['step'] for v in state['history']] != list(range(1, step + 1)):
            raise ValueError('Checkpoint history and completed update count disagree.')
        for name in ('encoder', 'predictor'):
            expected = getattr(self.model, name).state_dict()
            actual = state[name]
            if (set(expected) != set(actual) or any(expected[k].shape != actual[k].shape
                    or expected[k].dtype != actual[k].dtype or not torch.isfinite(actual[k]).all() for k in expected)):
                raise ValueError(f'Invalid {name} checkpoint tensors.')
        saved_groups = state['optimizer']['param_groups']
        if len(saved_groups) != len(self.optimizer.param_groups):
            raise ValueError('Optimizer parameter groups differ.')
        expected_rates = self.learning_rates(step - 1) if step else [self.settings.encoder_lr, self.settings.predictor_lr]
        for saved, current, rate in zip(saved_groups, self.optimizer.param_groups, expected_rates):
            if len(saved['params']) != len(current['params']):
                raise ValueError('Optimizer parameter count differs.')
            if saved['lr'] != rate or any(saved[k] != current[k] for k in
                    ('betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'capturable', 'differentiable', 'fused')):
                raise ValueError('Optimizer settings differ from the declared schedule.')
            if step and not any(number in state['optimizer']['state'] for number in saved['params']):
                raise ValueError('Missing optimizer moments for a trained parameter group.')
            for number, parameter in zip(saved['params'], current['params']):
                values = state['optimizer']['state'].get(number, {})
                if step and not values:
                    # Unused released predictor mask tokens legitimately have no moments.
                    continue
                if values and set(values) != {'step', 'exp_avg', 'exp_avg_sq'}:
                    raise ValueError('Incomplete AdamW moment state.')
                for key, value in values.items():
                    if isinstance(value, torch.Tensor) and (not torch.isfinite(value).all()
                            or (key != 'step' and value.shape != parameter.shape)):
                        raise ValueError('Invalid optimizer moment state.')
        self.failed = True
        for name in ('encoder', 'predictor'):
            getattr(self.model, name).load_state_dict(state[name], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'], self.device)
        self.step, self.history = step, list(state['history'])
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        self.failed = False
