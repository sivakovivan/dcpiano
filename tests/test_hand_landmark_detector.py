from types import SimpleNamespace

import pytest

from dcpiano.landmark_detectors.hand import HandLandmarkDetector, HandLandmarks
from dcpiano.types.landmark import Side


def _landmarks(x: float) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(x=x, y=float(index), z=0.0)
        for index, _ in enumerate(HandLandmarks)
    ]


def _category(side: str, score: float) -> list[SimpleNamespace]:
    return [SimpleNamespace(category_name=side, score=score)]


def _detector_without_model() -> HandLandmarkDetector:
    detector = object.__new__(HandLandmarkDetector)
    detector._landmark_filter = {landmark.value for landmark in HandLandmarks}
    return detector


def test_convert_results_keeps_highest_confidence_hand_for_duplicate_side() -> None:
    detector = _detector_without_model()
    results = SimpleNamespace(
        hand_landmarks=[_landmarks(0.1), _landmarks(0.9)],
        handedness=[_category("Right", 0.6), _category("Right", 0.95)],
    )

    converted = detector._convert_results(results)

    assert len(converted) == len(HandLandmarks)
    wrist = converted["HandLandmarkDetector:WRIST:right"]
    assert wrist.x == pytest.approx(0.9)
    assert wrist.confidence == pytest.approx(0.95)
    assert wrist.instance_id == 1


def test_convert_results_keeps_one_unknown_hand() -> None:
    detector = _detector_without_model()
    results = SimpleNamespace(
        hand_landmarks=[_landmarks(0.2), _landmarks(0.8)],
        handedness=[_category("", 0.4), _category("", 0.7)],
    )

    converted = detector._convert_results(results)

    assert len(converted) == len(HandLandmarks)
    wrist = converted["HandLandmarkDetector:WRIST"]
    assert wrist.side is Side.UNKNOWN
    assert wrist.x == pytest.approx(0.8)


def test_convert_results_preserves_one_hand_per_distinct_side() -> None:
    detector = _detector_without_model()
    results = SimpleNamespace(
        hand_landmarks=[_landmarks(0.2), _landmarks(0.8)],
        handedness=[_category("Left", 0.9), _category("Right", 0.8)],
    )

    converted = detector._convert_results(results)

    assert len(converted) == 2 * len(HandLandmarks)
    assert "HandLandmarkDetector:WRIST:left" in converted
    assert "HandLandmarkDetector:WRIST:right" in converted
