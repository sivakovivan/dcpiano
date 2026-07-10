from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from dcpiano.services.video import VideoService

import importlib

video_module = importlib.import_module(VideoService.__module__)


def make_capture(*, opened=True, fps=30.0, frames=None):
    """Build a mocked cv2.VideoCapture instance."""
    capture = MagicMock()
    capture.isOpened.return_value = opened
    capture.get.side_effect = lambda prop: {
        video_module.cv2.CAP_PROP_FRAME_WIDTH: 1920,
        video_module.cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        video_module.cv2.CAP_PROP_FPS: fps,
        video_module.cv2.CAP_PROP_FRAME_COUNT: 300,
    }[prop]

    frame_results = [(True, frame) for frame in (frames or [])]
    capture.read.side_effect = frame_results + [(False, None)]
    return capture


def test_get_video_metadata_returns_expected_values(monkeypatch):
    path = Path("example.mp4")
    capture = make_capture(fps=30.0)
    video_capture = MagicMock(return_value=capture)
    monkeypatch.setattr(video_module.cv2, "VideoCapture", video_capture)

    metadata = VideoService.get_video_metadata(path)

    video_capture.assert_called_once_with(str(path))
    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.path == path
    assert metadata.fps == 30.0
    assert metadata.duration == pytest.approx(10.0)
    capture.release.assert_called_once_with()


def test_get_video_metadata_uses_zero_duration_when_fps_is_zero(monkeypatch):
    capture = make_capture(fps=0.0)
    monkeypatch.setattr(video_module.cv2, "VideoCapture", MagicMock(return_value=capture))

    metadata = VideoService.get_video_metadata(Path("example.mp4"))

    assert metadata.duration == 0
    capture.release.assert_called_once_with()


def test_get_video_metadata_raises_when_video_cannot_be_opened(monkeypatch):
    path = Path("missing.mp4")
    capture = make_capture(opened=False)
    monkeypatch.setattr(video_module.cv2, "VideoCapture", MagicMock(return_value=capture))

    with pytest.raises(ValueError, match=r"Could not open video file: missing\.mp4"):
        VideoService.get_video_metadata(path)


def test_get_video_frames_returns_frames_with_indexes_and_timestamps(monkeypatch):
    path = Path("example.mp4")
    first = np.zeros((2, 3, 3), dtype=np.uint8)
    second = np.ones((2, 3, 3), dtype=np.uint8)
    capture = make_capture(fps=25.0, frames=[first, second])
    monkeypatch.setattr(video_module.cv2, "VideoCapture", MagicMock(return_value=capture))

    frames = VideoService.get_video_frames(path)

    assert len(frames) == 2
    assert frames[0].index == 0
    assert frames[0].timestamp == pytest.approx(0.0)
    assert frames[0].frame is first
    assert frames[1].index == 1
    assert frames[1].timestamp == pytest.approx(1 / 25.0)
    assert frames[1].frame is second
    assert capture.read.call_count == 3
    capture.release.assert_called_once_with()


def test_get_video_frames_uses_zero_timestamps_when_fps_is_zero(monkeypatch):
    first = np.zeros((1, 1, 3), dtype=np.uint8)
    second = np.ones((1, 1, 3), dtype=np.uint8)
    capture = make_capture(fps=0.0, frames=[first, second])
    monkeypatch.setattr(video_module.cv2, "VideoCapture", MagicMock(return_value=capture))

    frames = VideoService.get_video_frames(Path("example.mp4"))

    assert [frame.timestamp for frame in frames] == [0, 0]
    capture.release.assert_called_once_with()


def test_get_video_frames_raises_when_video_cannot_be_opened(monkeypatch):
    path = Path("missing.mp4")
    capture = make_capture(opened=False)
    monkeypatch.setattr(video_module.cv2, "VideoCapture", MagicMock(return_value=capture))

    with pytest.raises(ValueError, match=r"Could not open video file: missing\.mp4"):
        VideoService.get_video_frames(path)
