from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, isfinite, sqrt
from typing import Callable

from dcpiano.kinematics.core import KinematicCalculator, KinematicContext
from dcpiano.types.kinematic import KinematicData, Vector3
from dcpiano.types.landmark import Side


HAND_SOURCE = "HandLandmarkDetector"
FOREARM_SOURCE = "ForearmLandmarkDetector"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _landmark_id(source: str, name: str, side: Side) -> str:
    return f"{source}:{name}:{side.value}"


def _point(context: KinematicContext, landmark_id: str) -> Vector3 | None:
    landmark = context.landmark(landmark_id)
    if landmark is None or landmark.z is None:
        return None
    coordinates = (landmark.x, landmark.y, landmark.z)
    if not all(isfinite(value) for value in coordinates):
        return None
    return Vector3(*coordinates)


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return Vector3(a.x - b.x, a.y - b.y, a.z - b.z)


def _scale(vector: Vector3, factor: float) -> Vector3:
    return Vector3(vector.x * factor, vector.y * factor, vector.z * factor)


def _norm(vector: Vector3) -> float:
    return sqrt(vector.x**2 + vector.y**2 + vector.z**2)


def _unit(vector: Vector3) -> Vector3 | None:
    length = _norm(vector)
    return None if length == 0 else _scale(vector, 1 / length)


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return Vector3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def _angle(a: Vector3, vertex: Vector3, b: Vector3) -> float | None:
    left, right = _sub(a, vertex), _sub(b, vertex)
    denominator = _norm(left) * _norm(right)
    if denominator == 0:
        return None
    cosine = max(-1.0, min(1.0, (left.x * right.x + left.y * right.y + left.z * right.z) / denominator))
    return degrees(acos(cosine))


@dataclass(frozen=True, slots=True)
class FunctionalCalculator(KinematicCalculator):
    path: str
    function: Callable[[KinematicContext], KinematicData | None]
    landmark_inputs: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()

    def calculate(self, context: KinematicContext) -> KinematicData | None:
        return self.function(context)


def _position(path: str, landmark_id: str) -> FunctionalCalculator:
    return FunctionalCalculator(path, lambda context: _point(context, landmark_id), (landmark_id,))


def _derivative(path: str, prerequisite: str) -> FunctionalCalculator:
    def calculate(context: KinematicContext) -> Vector3 | None:
        current = context.kinematic_frames[-1].get(prerequisite)
        if not isinstance(current, Vector3) or not context.kinematic_frames[:-1]:
            return None
        previous_frame = context.kinematic_frames[-2]
        previous = previous_frame.get(prerequisite)
        dt = context.current_landmark_frame.metadata.timestamp - previous_frame.timestamp
        if not isinstance(previous, Vector3) or dt <= 0 or not isfinite(dt):
            return None
        return _scale(_sub(current, previous), 1 / dt)

    return FunctionalCalculator(path, calculate, prerequisites=(prerequisite,))


def _axis(path: str, start_id: str, end_id: str) -> FunctionalCalculator:
    def calculate(context: KinematicContext) -> Vector3 | None:
        start, end = _point(context, start_id), _point(context, end_id)
        return None if start is None or end is None else _unit(_sub(end, start))
    return FunctionalCalculator(path, calculate, (start_id, end_id))


def _plane(path: str, origin_id: str, left_id: str, right_id: str) -> FunctionalCalculator:
    def calculate(context: KinematicContext) -> Vector3 | None:
        origin, left, right = (_point(context, item) for item in (origin_id, left_id, right_id))
        if origin is None or left is None or right is None:
            return None
        return _unit(_cross(_sub(left, origin), _sub(right, origin)))
    return FunctionalCalculator(path, calculate, (origin_id, left_id, right_id))


def _joint_angle(path: str, ids: tuple[str, str, str]) -> FunctionalCalculator:
    def calculate(context: KinematicContext) -> float | None:
        points = tuple(_point(context, item) for item in ids)
        if any(point is None for point in points):
            return None
        return _angle(points[0], points[1], points[2])  # type: ignore[arg-type]
    return FunctionalCalculator(path, calculate, ids)


def default_kinematic_calculators() -> list[KinematicCalculator]:
    """Build the ordered default registry. Derivatives follow their prerequisites."""
    calculators: list[KinematicCalculator] = []
    for side in (Side.LEFT, Side.RIGHT):
        root = f"hands.{side.value}"
        wrist_id = _landmark_id(HAND_SOURCE, "WRIST", side)
        wrist_position = f"{root}.wrist.position"
        calculators.extend((
            _position(wrist_position, wrist_id),
            _derivative(f"{root}.wrist.velocity", wrist_position),
            _derivative(f"{root}.wrist.acceleration", f"{root}.wrist.velocity"),
            _axis(f"{root}.hand_axis", wrist_id, _landmark_id(HAND_SOURCE, "MIDDLE_FINGER_MCP", side)),
            _plane(f"{root}.hand_plane", wrist_id, _landmark_id(HAND_SOURCE, "INDEX_FINGER_MCP", side), _landmark_id(HAND_SOURCE, "PINKY_MCP", side)),
            _axis(f"{root}.forearm_axis", wrist_id, _landmark_id(FOREARM_SOURCE, "FOREARM_FAR", side)),
        ))

        for finger in FINGERS:
            prefix = {
                "thumb": "THUMB",
                "pinky": "PINKY",
            }.get(finger, f"{finger.upper()}_FINGER")
            tip_id = _landmark_id(HAND_SOURCE, f"{prefix}_TIP", side)
            position = f"{root}.fingers.{finger}.tip.position"
            calculators.extend((
                _position(position, tip_id),
                _derivative(f"{root}.fingers.{finger}.tip.velocity", position),
                _derivative(f"{root}.fingers.{finger}.tip.acceleration", f"{root}.fingers.{finger}.tip.velocity"),
            ))
            if finger == "thumb":
                mcp_names = ("THUMB_CMC", "THUMB_MCP", "THUMB_IP")
                pip_names = ("THUMB_MCP", "THUMB_IP", "THUMB_TIP")
            else:
                mcp_names = ("WRIST", f"{prefix}_MCP", f"{prefix}_PIP")
                pip_names = (f"{prefix}_MCP", f"{prefix}_PIP", f"{prefix}_DIP")
            calculators.extend((
                _joint_angle(f"{root}.fingers.{finger}.mcp_angle", tuple(_landmark_id(HAND_SOURCE, name, side) for name in mcp_names)),
                _joint_angle(f"{root}.fingers.{finger}.pip_angle", tuple(_landmark_id(HAND_SOURCE, name, side) for name in pip_names)),
                FunctionalCalculator(
                    f"{root}.fingers.{finger}.height_above_keyboard",
                    lambda context, landmark_id=tip_id: (None if (point := _point(context, landmark_id)) is None else -point.z),
                    (tip_id,),
                ),
            ))
    return calculators
