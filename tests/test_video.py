import numpy as np
import pytest
import shutil
import subprocess
import torch

from datasets.video import decode_clip, preprocess_rgb, presentation_timestamps, temporal_indices


def test_centered_sampling_uses_timestamps_and_earlier_tie():
    assert temporal_indices(np.arange(25) / 4) == list(range(4, 20))
    assert temporal_indices([0, 1], frames=1) == [0]


def test_short_video_repeats_last_frame():
    selected = temporal_indices([0, 0.1])
    assert selected[0] == 0 and selected[1:] == [1] * 15


def test_resize_crop_shape_range_and_replay():
    frames = np.full((3, 20, 30, 3), 128, dtype=np.uint8)
    first, a = preprocess_rgb(frames, [0, 0.1, 0.2])
    second, b = preprocess_rgb(frames, [0, 0.1, 0.2])
    assert first.shape == (3, 16, 256, 256) and first.dtype == torch.float32
    torch.testing.assert_close(first, torch.full_like(first, 128/255))
    assert torch.equal(first, second) and a == b


@pytest.mark.parametrize('timestamps', [[], [1, 0], [0, 0], [float('nan')]])
def test_bad_timestamps_fail(timestamps):
    with pytest.raises(ValueError):
        temporal_indices(timestamps)


def test_missing_pts_repaired_only_when_known_pts_confirm_cfr():
    values, repairs = presentation_timestamps([{'best_effort_timestamp_time': '.04'},
                                               {'best_effort_timestamp_time': '.08'}, {}], '25/1')
    assert values == pytest.approx([.04, .08, .12]) and repairs == {2: .12}
    with pytest.raises(ValueError, match='constant frame rate'):
        presentation_timestamps([{'best_effort_timestamp_time': '.04'},
                                 {'best_effort_timestamp_time': '.09'}, {}], '25/1')


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='FFmpeg integration check needs ffmpeg and ffprobe')
def test_avi_b_frames_decode_and_replay(tmp_path):
    video = tmp_path / 'bframes.avi'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=32x24:rate=25:duration=1', '-c:v', 'mpeg4', '-bf', '2', str(video)], check=True)
    first, metadata = decode_clip(video)
    second, repeated = decode_clip(video)
    assert first.shape == (3, 16, 256, 256) and metadata['decoded_frames'] == 25
    assert torch.equal(first, second) and metadata == repeated
