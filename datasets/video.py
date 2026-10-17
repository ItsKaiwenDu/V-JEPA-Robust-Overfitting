"""Deterministic CPU preprocessing; pixel normalization remains in the model."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

RECIPE = 'cfr-repair-center-4fps-16f-shortside256-centercrop-bilinear-aa-v1'


def presentation_timestamps(frame_records, average_frame_rate):
    """Repair missing PTS only when all observed PTS agree with declared CFR."""
    known = {i: float(frame['best_effort_timestamp_time']) for i, frame in enumerate(frame_records)
             if 'best_effort_timestamp_time' in frame}
    missing = [i for i in range(len(frame_records)) if i not in known]
    if not known or not np.isfinite(list(known.values())).all():
        raise ValueError('No usable presentation timestamps.')
    repairs = {}
    if missing:
        fps = float(Fraction(average_frame_rate))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Cannot repair missing timestamps without a valid frame rate.')
        first = min(known)
        origin = known[first] - first / fps
        if any(abs(value - (origin + i / fps)) > 1e-5 for i, value in known.items()):
            raise ValueError('Missing timestamps in a stream that does not match its declared constant frame rate.')
        repairs = {i: origin + i / fps for i in missing}
    return [known[i] if i in known else repairs[i] for i in range(len(frame_records))], repairs


def temporal_indices(timestamps, frames=16, fps=4.0):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or not len(timestamps) or not np.isfinite(timestamps).all():
        raise ValueError('Need finite presentation timestamps for at least one frame.')
    if np.any(np.diff(timestamps) <= 0) or frames < 1 or fps <= 0:
        raise ValueError('Timestamps must increase strictly; frame count and FPS must be positive.')
    relative = timestamps - timestamps[0]
    start = max(0., (relative[-1] - (frames - 1) / fps) / 2)
    desired = start + np.arange(frames) / fps
    right = np.searchsorted(relative, desired).clip(0, len(relative) - 1)
    left = (right - 1).clip(0, len(relative) - 1)
    indices = np.where(np.abs(relative[left] - desired) <= np.abs(relative[right] - desired), left, right)
    return indices.tolist()


def preprocess_rgb(frames: np.ndarray, timestamps) -> tuple[torch.Tensor, list[int]]:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or len(frames) != len(timestamps):
        raise ValueError('Expected matching RGB uint8 frames [N, H, W, 3] and timestamps.')
    indices = temporal_indices(timestamps)
    pixels = torch.from_numpy(np.array(frames[indices], copy=True)).permute(0, 3, 1, 2).float() / 255
    height, width = pixels.shape[-2:]
    if min(height, width) < 1:
        raise ValueError('Empty spatial dimension.')
    scale = 256 / min(height, width)
    size = (max(256, int(height * scale)), max(256, int(width * scale)))
    resized = F.interpolate(pixels, size=size, mode='bilinear', align_corners=False, antialias=True)
    top, left = (size[0] - 256) // 2, (size[1] - 256) // 2
    clip = resized[:, :, top:top+256, left:left+256].permute(1, 0, 2, 3).contiguous().clamp(0, 1)
    return clip, indices


def decode_clip(path: Path) -> tuple[torch.Tensor, dict]:
    path = path.resolve(strict=True)
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time',
                            '-of', 'json', str(path)], check=True, capture_output=True, text=True, timeout=120)
    info = json.loads(probe.stdout)
    if len(info.get('streams', [])) != 1:
        raise ValueError('Expected a decodable primary video stream.')
    width, height = info['streams'][0]['width'], info['streams'][0]['height']
    timestamps, repairs = presentation_timestamps(info['frames'], info['streams'][0]['avg_frame_rate'])
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-noautorotate', '-i', str(path),
                              '-map', '0:v:0', '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24',
                              '-f', 'rawvideo', 'pipe:1'], check=True, capture_output=True, timeout=120)
    raw = np.frombuffer(decoded.stdout, dtype=np.uint8)
    if raw.size != len(timestamps) * height * width * 3:
        raise ValueError('Decoded frame count disagrees with presentation timestamps.')
    frames = raw.reshape(len(timestamps), height, width, 3)
    clip, indices = preprocess_rgb(frames, timestamps)
    versions = {program: subprocess.check_output([program, '-version'], text=True).splitlines()[0]
                for program in ('ffmpeg', 'ffprobe')}
    with path.open('rb') as handle:
        source_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    metadata = {'recipe': RECIPE, 'source_sha256': source_hash, 'decoded_frames': len(timestamps),
                'source_size': [height, width], 'frame_indices': indices,
                'selected_timestamps_seconds': [timestamps[i] for i in indices],
                'torch': str(torch.__version__), 'decoder': versions,
                'timestamp_repairs': {str(i): value for i, value in repairs.items()},
                'pixel_sha256': hashlib.sha256(clip.numpy().tobytes()).hexdigest()}
    metadata['preprocessing_key'] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return clip, metadata
