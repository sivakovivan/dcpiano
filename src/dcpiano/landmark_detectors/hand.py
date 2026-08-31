from __future__ import annotations

from typing import Any
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import mediapipe as mp
import numpy as np

from dcpiano.landmark_detectors.core import LandmarkDetector
from dcpiano.logger import logger
from dcpiano.types.render import Connection
from dcpiano.types.landmark import RawLandmark, Side
from dcpiano.types.video import VideoFrame


class HandLandmarks(StrEnum):
    WRIST = "WRIST"
    THUMB_CMC = "THUMB_CMC"
    THUMB_MCP = "THUMB_MCP"
    THUMB_IP = "THUMB_IP"
    THUMB_TIP = "THUMB_TIP"
    INDEX_FINGER_MCP = "INDEX_FINGER_MCP"
    INDEX_FINGER_PIP = "INDEX_FINGER_PIP"
    INDEX_FINGER_DIP = "INDEX_FINGER_DIP"
    INDEX_FINGER_TIP = "INDEX_FINGER_TIP"
    MIDDLE_FINGER_MCP = "MIDDLE_FINGER_MCP"
    MIDDLE_FINGER_PIP = "MIDDLE_FINGER_PIP"
    MIDDLE_FINGER_DIP = "MIDDLE_FINGER_DIP"
    MIDDLE_FINGER_TIP = "MIDDLE_FINGER_TIP"
    RING_FINGER_MCP = "RING_FINGER_MCP"
    RING_FINGER_PIP = "RING_FINGER_PIP"
    RING_FINGER_DIP = "RING_FINGER_DIP"
    RING_FINGER_TIP = "RING_FINGER_TIP"
    PINKY_MCP = "PINKY_MCP"
    PINKY_PIP = "PINKY_PIP"
    PINKY_DIP = "PINKY_DIP"
    PINKY_TIP = "PINKY_TIP"


@dataclass(slots=True)
class HandLandmarkConfig:
    model_path: Path = Path("models/hand_landmarker.task")
    max_hands: int = 2
    min_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


RENDER_CONNECTIONS: tuple[Connection, ...] = (
    ("WRIST", "THUMB_CMC"),
    ("THUMB_CMC", "THUMB_MCP"),
    ("THUMB_MCP", "THUMB_IP"),
    ("THUMB_IP", "THUMB_TIP"),
    ("WRIST", "INDEX_FINGER_MCP"),
    ("INDEX_FINGER_MCP", "INDEX_FINGER_PIP"),
    ("INDEX_FINGER_PIP", "INDEX_FINGER_DIP"),
    ("INDEX_FINGER_DIP", "INDEX_FINGER_TIP"),
    ("WRIST", "MIDDLE_FINGER_MCP"),
    ("MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP"),
    ("MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP"),
    ("MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP"),
    ("WRIST", "RING_FINGER_MCP"),
    ("RING_FINGER_MCP", "RING_FINGER_PIP"),
    ("RING_FINGER_PIP", "RING_FINGER_DIP"),
    ("RING_FINGER_DIP", "RING_FINGER_TIP"),
    ("WRIST", "PINKY_MCP"),
    ("PINKY_MCP", "PINKY_PIP"),
    ("PINKY_PIP", "PINKY_DIP"),
    ("PINKY_DIP", "PINKY_TIP"),
    ("INDEX_FINGER_MCP", "MIDDLE_FINGER_MCP"),
    ("MIDDLE_FINGER_MCP", "RING_FINGER_MCP"),
    ("RING_FINGER_MCP", "PINKY_MCP"),
)


class HandLandmarkDetector(LandmarkDetector):
    """MediaPipe Tasks hand detector operating in VIDEO mode.

    Keep one detector instance alive for all consecutive frames from a video.
    `FrameMetadata.timestamp` is assumed to be expressed in seconds.
    """

    name = "HandLandmarkDetector"
    render_connections = RENDER_CONNECTIONS
    
    
    _LANDMARK_INDEX_TO_NAME = tuple(HandLandmarks)

    def __init__(self, config: HandLandmarkConfig | None = None) -> None:
        self.config = config or HandLandmarkConfig()
        self.landmarks = tuple(HandLandmarks)
        self._landmark_filter = {landmark.value for landmark in self.landmarks}
        self._last_timestamp_ms = -1
        self._detector = self._create_detector()

    def _create_detector(self) -> Any:
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.config.model_path),
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=self.config.max_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_hand_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def reset_video(self) -> None:
        """Clear tracking state before processing a different video/clip."""
        self._detector.close()
        self._detector = self._create_detector()
        self._last_timestamp_ms = -1

    def detect(
        self,
        frame: VideoFrame,
        existing_landmarks: dict[str, RawLandmark] | None = None,
    ) -> dict[str, RawLandmark]:
        # ``existing_landmarks`` is accepted for the common detector protocol;
        # hand detection is the first stage and therefore does not consume it.
        timestamp_ms = self._timestamp_ms(frame)
        rgb_frame = np.ascontiguousarray(frame.frame[..., ::-1])

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        results = self._detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        return self._convert_results(results)

    def detect_video(
        self,
        frames: list[VideoFrame],
        reset: bool = True,
    ) -> list[dict[str, RawLandmark]]:
        """Process consecutive video frames while retaining tracking state."""
        if reset:
            self.reset_video()
        return [self.detect(frame) for frame in frames]

    def _timestamp_ms(self, frame: VideoFrame) -> int:
        timestamp_seconds = float(frame.metadata.timestamp)
        if timestamp_seconds < 0:
            raise ValueError(
                f"Frame {frame.metadata.index} has a negative timestamp: "
                f"{timestamp_seconds}"
            )

        timestamp_ms = round(timestamp_seconds * 1000.0)
        if timestamp_ms <= self._last_timestamp_ms:
            # Rounding or duplicate source timestamps must not violate the Tasks API.
            timestamp_ms = self._last_timestamp_ms + 1

        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _convert_results(self, results: object) -> dict[str, RawLandmark]:
        hand_landmarks = results.hand_landmarks
        if not hand_landmarks:
            return {}

        raw_landmarks: dict[str, RawLandmark] = {}
        for side, confidence, instance_id, landmarks in self._select_hand_candidates(
            hand_landmarks,
            results.handedness,
        ):

            for landmark_index, landmark in enumerate(landmarks):
                landmark_name = self._LANDMARK_INDEX_TO_NAME[landmark_index].value
                if landmark_name not in self._landmark_filter:
                    continue

                raw_landmark = RawLandmark(
                    name=landmark_name,
                    source=self.name,
                    instance_id=instance_id,
                    side=side,
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    confidence=confidence,
                )
                landmark_id = raw_landmark.generate_landmark_id()
                if landmark_id in raw_landmarks:
                    raise ValueError(f"Duplicate raw landmark id: {landmark_id}")
                raw_landmarks[landmark_id] = raw_landmark

        return raw_landmarks

    def _select_hand_candidates(
        self,
        hand_landmarks: object,
        handedness: object,
    ) -> list[tuple[Side, float | None, int, object]]:
        """Keep at most one complete detected hand for each side.

        MediaPipe can occasionally classify multiple detections as the same side.
        Stable landmark IDs contain the side but intentionally omit the transient
        detection index, so retaining both would create duplicate IDs. Resolve the
        ambiguity before converting individual landmarks, using handedness
        confidence as the tie-breaker and retaining the first detection on ties.
        """
        selected: dict[str, tuple[Side, float | None, int, object]] = {}

        for instance_id, landmarks in enumerate(hand_landmarks):
            side = self._resolve_side(handedness, instance_id)
            confidence = self._resolve_confidence(handedness, instance_id)
            side_key = side.value
            current = selected.get(side_key)

            if current is None:
                selected[side_key] = (side, confidence, instance_id, landmarks)
                continue

            current_confidence = current[1]
            candidate_score = confidence if confidence is not None else float("-inf")
            current_score = (
                current_confidence
                if current_confidence is not None
                else float("-inf")
            )
            if candidate_score > current_score:
                selected[side_key] = (side, confidence, instance_id, landmarks)

            kept_instance_id = selected[side_key][2]
            logger.warning(
                f"Multiple hands classified as {side.value}; keeping detection "
                f"{kept_instance_id} and discarding the other candidate."
            )

        return [
            candidate
            for candidate in selected.values()
        ]

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> HandLandmarkDetector:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @staticmethod
    def _resolve_side(handedness: object, instance_id: int) -> Side:
        if instance_id >= len(handedness):
            return Side.UNKNOWN

        categories = handedness[instance_id]
        if not categories:
            return Side.UNKNOWN

        label = (categories[0].category_name or "").lower()
        if label == Side.LEFT.value:
            return Side.LEFT
        if label == Side.RIGHT.value:
            return Side.RIGHT
        return Side.UNKNOWN

    @staticmethod
    def _resolve_confidence(handedness: object, instance_id: int) -> float | None:
        if instance_id >= len(handedness):
            return None

        categories = handedness[instance_id]
        if not categories:
            return None

        return categories[0].score
