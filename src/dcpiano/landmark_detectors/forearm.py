from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np
from armtrace import ArmTracer, HandLandmarks

from dcpiano.landmark_detectors.core import LandmarkDetector
from dcpiano.types.landmark import RawLandmark
from dcpiano.types.render import Connection
from dcpiano.types.video import VideoFrame


class ForearmLandmarks(StrEnum):
    """Points on the visible forearm centerline returned by ArmTrace."""

    FOREARM_MIDPOINT = "FOREARM_MIDPOINT"
    FOREARM_FAR = "FOREARM_FAR"


@dataclass(slots=True)
class ForearmLandmarkConfig:
    """Parameters used to configure :class:`armtrace.ArmTracer`."""

    patch_radius: int = 8
    threshold: float = 3.0
    morph_kernel: int = 7
    march_step: int = 8
    search_radius: int = 20
    direction_smoothing: float = 0.70
    minimum_radius: float = 5.0
    max_steps: int = 200
    roi_padding: int | None = None
    processing_scale: float = 0.5


RENDER_CONNECTIONS: tuple[Connection, ...] = (
    ("FOREARM_MIDPOINT", "FOREARM_FAR"),
)


class ForearmLandmarkDetector(LandmarkDetector):
    """Adapt stateful ArmTrace forearm detection to DCPiano landmarks."""

    name = "ForearmLandmarkDetector"
    render_connections = RENDER_CONNECTIONS

    _HAND_SOURCE = "HandLandmarkDetector"
    _FINGER_BASE_NAMES = (
        "INDEX_FINGER_MCP",
        "MIDDLE_FINGER_MCP",
        "RING_FINGER_MCP",
        "PINKY_MCP",
    )
    _REQUIRED_NAMES = ("WRIST", *_FINGER_BASE_NAMES)

    def __init__(self, config: ForearmLandmarkConfig | None = None) -> None:
        self.config = config or ForearmLandmarkConfig()
        self.landmarks = tuple(ForearmLandmarks)

        # Construct once so configuration validation, kernels, and working
        # buffers can be reused across every frame in the video.
        self._tracer = ArmTracer(**asdict(self.config))

    def detect(
        self,
        frame: VideoFrame,
        existing_landmarks: dict[str, RawLandmark],
    ) -> dict[str, RawLandmark]:
        image = frame.frame
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] < 3
            or image.size == 0
        ):
            return {}

        # This is zero-copy when the input is already contiguous three-channel
        # BGR. Four-channel or non-contiguous inputs are copied only as needed.
        image = np.ascontiguousarray(image[..., :3])
        height, width = image.shape[:2]

        hands: dict[int, dict[str, RawLandmark]] = {}
        for landmark in existing_landmarks.values():
            if (
                landmark.source == self._HAND_SOURCE
                and landmark.name in self._REQUIRED_NAMES
            ):
                hands.setdefault(landmark.instance_id, {})[
                    landmark.name
                ] = landmark

        detected: dict[str, RawLandmark] = {}

        for instance_id, hand in hands.items():
            if len(hand) < len(self._REQUIRED_NAMES):
                continue

            wrist_pixel = self._to_pixel(hand["WRIST"], width, height)
            if wrist_pixel is None:
                continue

            finger_base_pixels: list[tuple[int, int]] = []
            for name in self._FINGER_BASE_NAMES:
                point = self._to_pixel(hand[name], width, height)
                if point is None:
                    break
                finger_base_pixels.append(point)
            else:
                armtrace_hand = HandLandmarks(
                    wrist=wrist_pixel,
                    finger_bases=tuple(finger_base_pixels),
                )

                try:
                    midpoint, farpoint = self._tracer.detect(
                        image,
                        armtrace_hand,
                    )
                except (ValueError, np.linalg.LinAlgError):
                    # Occlusion, invalid segmentation, or the arm leaving the
                    # frame can legitimately prevent detection.
                    continue

                wrist = hand["WRIST"]
                self._add_landmark(
                    detected=detected,
                    name=ForearmLandmarks.FOREARM_MIDPOINT,
                    point=midpoint,
                    wrist=wrist,
                    instance_id=instance_id,
                    width=width,
                    height=height,
                )
                self._add_landmark(
                    detected=detected,
                    name=ForearmLandmarks.FOREARM_FAR,
                    point=farpoint,
                    wrist=wrist,
                    instance_id=instance_id,
                    width=width,
                    height=height,
                )

        return detected

    def _add_landmark(
        self,
        *,
        detected: dict[str, RawLandmark],
        name: ForearmLandmarks,
        point: tuple[int, int],
        wrist: RawLandmark,
        instance_id: int,
        width: int,
        height: int,
    ) -> None:
        x, y = point
        raw = RawLandmark(
            name=name.value,
            source=self.name,
            instance_id=instance_id,
            side=wrist.side,
            x=x / width,
            y=y / height,
            z=wrist.z,
            confidence=wrist.confidence,
        )

        landmark_id = raw.generate_landmark_id()
        if landmark_id in detected:
            raise ValueError(f"Duplicate raw landmark id: {landmark_id}")

        detected[landmark_id] = raw

    @staticmethod
    def _to_pixel(
        landmark: RawLandmark,
        width: int,
        height: int,
    ) -> tuple[int, int] | None:
        if not np.isfinite(landmark.x) or not np.isfinite(landmark.y):
            return None

        x = int(round(landmark.x * width))
        y = int(round(landmark.y * height))

        if not (0 <= x < width and 0 <= y < height):
            return None

        return x, y

    def reset_video(self) -> None:
        """Release buffers and state retained for the previous video."""

        self._tracer.reset()

    def close(self) -> None:
        """Release retained ArmTrace working memory."""

        self._tracer.reset()