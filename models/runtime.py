"""Explicit GPU timing and memory observations for small feasibility checks."""

import os
import platform
import subprocess
import threading

import torch



def configure_cuda_reproducibility():
    """Select strict deterministic kernels before this process initializes CUDA.

    Seeds alone do not make CUDA convolution/gradient execution repeatable.
    Keep precision flags unchanged; record those separately in the contract.
    """
    if torch.cuda.is_initialized():
        raise RuntimeError('Configure deterministic execution before CUDA initialization.')
    configured = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if configured not in (None, ':4096:8'):
        raise ValueError('This diagnostic requires CUBLAS_WORKSPACE_CONFIG=:4096:8.')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def require_device(device):
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable in this runtime.')
    elif device == 'mps':
        if not torch.backends.mps.is_available():
            raise ValueError('MPS is unavailable in this runtime. Check access from a normal macOS Terminal.')
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
            raise ValueError('Disable PYTORCH_ENABLE_MPS_FALLBACK for an explicit GPU feasibility check.')
    elif device != 'cpu':
        raise ValueError(f'Unsupported device: {device}')


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def empty_cache(device):
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()


def device_metadata(device):
    result = {'device': device, 'macos': platform.mac_ver()[0] or None,
              'architecture': platform.machine(), 'python': platform.python_version()}
    if device == 'cuda':
        result.update(gpu=torch.cuda.get_device_name(),
                      vram_bytes=torch.cuda.get_device_properties(0).total_memory)
    elif device == 'mps':
        result.update(gpu=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                      system_memory_bytes=int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)),
                      recommended_working_set_bytes=torch.mps.recommended_max_memory(),
                      mps_cpu_fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'unset'),
                      mps_fast_math=os.environ.get('PYTORCH_MPS_FAST_MATH', 'unset'),
                      mps_high_watermark_ratio=os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'default'))
    return result


class MemoryMonitor:
    """CUDA allocator peak, or sampled MPS allocations (not an exact peak).

    MPS driver allocation includes cache and other driver allocations, while
    live tensor bytes do not. Samples can miss transients; neither reports
    total system memory use. No memory limits or device settings are changed.
    """

    def __init__(self, device, interval=0.1):
        self.device = device
        self.interval = interval
        self.stop = threading.Event()
        self.allocated = self.driver = self.samples = 0
        self.errors = []

    def _sample(self):
        self.allocated = max(self.allocated, torch.mps.current_allocated_memory())
        self.driver = max(self.driver, torch.mps.driver_allocated_memory())
        self.samples += 1

    def _run(self):
        try:
            while not self.stop.wait(self.interval):
                self._sample()
        except Exception as exc:
            self.errors.append(f'{type(exc).__name__}: {exc}')

    def __enter__(self):
        synchronize(self.device)
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        elif self.device == 'mps':
            self._sample()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc):
        if self.device == 'mps':
            self.stop.set()
            self.thread.join()
            self._sample()

    def report(self):
        if self.device == 'cuda':
            return {'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        if self.device == 'mps':
            return {'mps_sampled_max_tensor_bytes': self.allocated,
                    'mps_sampled_max_driver_bytes': self.driver,
                    'sample_interval_seconds': self.interval, 'samples': self.samples,
                    'is_exact_peak': False, 'sampling_errors': self.errors}
        return {}
