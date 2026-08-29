import json

import pytest

from dcpiano.services.stream import LandmarkStreamService
from dcpiano.types.landmark import LandmarkFrame, RawLandmark, Side
from dcpiano.types.video import FrameMetadata


def landmark(x: float, instance_id: int = 0) -> RawLandmark:
    return RawLandmark(
        name="wrist",
        source="hands",
        instance_id=instance_id,
        side=Side.LEFT,
        x=x,
        y=x + 0.1,
        z=None,
        confidence=0.9,
    )


def frame(index: int, landmarks: dict[str, RawLandmark]) -> LandmarkFrame:
    return LandmarkFrame(
        metadata=FrameMetadata(index=index, timestamp=index / 10),
        landmarks=landmarks,
    )


def test_generates_continuous_entries_for_missing_detections(tmp_path):
    detected = landmark(0.2)
    landmark_id = detected.generate_landmark_id()
    frames = [
        frame(0, {}),
        frame(1, {landmark_id: detected}),
        frame(2, {}),
        frame(3, {landmark_id: landmark(0.7)}),
    ]

    artifact = LandmarkStreamService.generate_landmark_streams(frames, tmp_path)

    stream = artifact.landmarks[landmark_id]
    assert [entry.exists for entry in stream.entries] == [False, True, False, True]
    assert [entry.metadata.index for entry in stream.entries] == [0, 1, 2, 3]
    assert stream.entries[0].x is None
    assert stream.entries[1].x == 0.2
    assert stream.entries[3].x == 0.7

    serialized = json.loads((tmp_path / "landmark_streams.json").read_text())
    assert serialized["landmarks"][landmark_id]["entries"][2]["exists"] is False


def test_rejects_unordered_frames():
    with pytest.raises(ValueError, match="ordered by increasing"):
        LandmarkStreamService.generate_landmark_streams([frame(2, {}), frame(1, {})])


def test_rejects_identity_change_for_same_id():
    first = landmark(0.2, instance_id=0)
    second = landmark(0.3, instance_id=1)
    landmark_id = first.generate_landmark_id()

    with pytest.raises(ValueError, match="changed identity"):
        LandmarkStreamService.generate_landmark_streams(
            [frame(0, {landmark_id: first}), frame(1, {landmark_id: second})]
        )