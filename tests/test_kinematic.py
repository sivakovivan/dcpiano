from __future__ import annotations

import json

import pytest

from dcpiano.kinematics.core import KinematicCalculator, KinematicContext
from dcpiano.kinematics.defaults import default_kinematic_calculators
from dcpiano.services.kinematic import KinematicService
from dcpiano.types.kinematic import Vector3
from dcpiano.types.landmark import LandmarkFrame, RawLandmark, Side
from dcpiano.types.video import FrameMetadata


def landmark(name: str, x: float, *, side: Side = Side.LEFT) -> RawLandmark:
    return RawLandmark(
        name=name,
        source="HandLandmarkDetector",
        instance_id=0,
        side=side,
        x=x,
        y=0.2,
        z=-0.1,
    )


def frame(index: int, timestamp: float, *landmarks: RawLandmark) -> LandmarkFrame:
    return LandmarkFrame(
        FrameMetadata(index=index, timestamp=timestamp),
        {item.generate_landmark_id(): item for item in landmarks},
    )


def test_default_registry_builds_derivative_hierarchy_and_nulls_missing_values(tmp_path):
    frames = [
        frame(0, 0.0, landmark("WRIST", 0.0)),
        frame(1, 0.5, landmark("WRIST", 1.0)),
        frame(2, 1.0, landmark("WRIST", 3.0)),
        frame(3, 1.5),
    ]

    generated = KinematicService(default_kinematic_calculators()).generate_kinematic_frames(frames, tmp_path)
    position = "hands.left.wrist.position"
    velocity = "hands.left.wrist.velocity"
    acceleration = "hands.left.wrist.acceleration"

    assert generated[0].get(position) == Vector3(0.0, 0.2, -0.1)
    assert generated[0].get(velocity) is None
    assert generated[1].get(velocity) == Vector3(2.0, 0.0, 0.0)
    assert generated[1].get(acceleration) is None
    assert generated[2].get(velocity) == Vector3(4.0, 0.0, 0.0)
    assert generated[2].get(acceleration) == Vector3(4.0, 0.0, 0.0)
    assert generated[3].get(position) is None
    assert generated[3].get(velocity) is None

    serialized = json.loads((tmp_path / "kinematic_frames.json").read_text())
    assert serialized[2]["values"][acceleration]["value"]["x"] == 4.0
    assert serialized[3]["values"][position]["value"] is None


class StubCalculator(KinematicCalculator):
    def __init__(self, path: str, prerequisites: tuple[str, ...] = ()) -> None:
        self.path = path
        self.prerequisites = prerequisites

    def calculate(self, context: KinematicContext) -> float:
        return 1.0


def test_registry_rejects_out_of_order_prerequisites():
    with pytest.raises(ValueError, match="registered after prerequisites"):
        KinematicService([StubCalculator("velocity", ("position",)), StubCalculator("position")])


def test_default_registry_contains_complete_frame_shape():
    paths = {calculator.path for calculator in default_kinematic_calculators()}
    assert len(paths) == 72
    assert "hands.right.hand_plane" in paths
    assert "hands.left.fingers.pinky.height_above_keyboard" in paths
    assert "hands.right.fingers.thumb.pip_angle" in paths


def test_default_registry_uses_mediapipe_pinky_landmark_names():
    calculators = {
        calculator.path: calculator
        for calculator in default_kinematic_calculators()
    }

    assert calculators["hands.left.fingers.pinky.tip.position"].landmark_inputs == (
        "HandLandmarkDetector:PINKY_TIP:left",
    )
    assert calculators["hands.right.fingers.pinky.mcp_angle"].landmark_inputs == (
        "HandLandmarkDetector:WRIST:right",
        "HandLandmarkDetector:PINKY_MCP:right",
        "HandLandmarkDetector:PINKY_PIP:right",
    )
    assert calculators["hands.right.fingers.pinky.pip_angle"].landmark_inputs == (
        "HandLandmarkDetector:PINKY_MCP:right",
        "HandLandmarkDetector:PINKY_PIP:right",
        "HandLandmarkDetector:PINKY_DIP:right",
    )
