from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from dcpiano.types.render import Connection
from dcpiano.types.landmark import LandmarkFrame, RawLandmark, Side

Color = tuple[int, int, int]  # OpenCV BGR


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """Visual and filtering configuration for landmark rendering."""

    connections_by_source: Mapping[str, Sequence[Connection]]

    min_confidence: float = 0.0

    min_joint_radius: int = 3
    max_joint_radius: int = 7
    line_thickness: int = 2

    # z is normalized into [0, 1] after being clamped to this interval.
    # MediaPipe-style z values are normally negative toward the camera, so
    # z_near_is_negative=True makes more-negative points appear slightly larger.
    z_min: float = -0.10
    z_max: float = 0.10
    z_near_is_negative: bool = True

    left_color: Color = (80, 220, 80)
    right_color: Color = (80, 140, 255)
    unknown_color: Color = (255, 210, 80)
    connection_color: Color = (220, 220, 220)

    codec: str = "mp4v"
    fallback_fps: float = 30.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.min_joint_radius <= 0:
            raise ValueError("min_joint_radius must be positive")
        if self.max_joint_radius < self.min_joint_radius:
            raise ValueError("max_joint_radius must be >= min_joint_radius")
        if self.line_thickness <= 0:
            raise ValueError("line_thickness must be positive")
        if self.z_max <= self.z_min:
            raise ValueError("z_max must be greater than z_min")
        if len(self.codec) != 4:
            raise ValueError("codec must be a four-character code")
        if self.fallback_fps <= 0:
            raise ValueError("fallback_fps must be positive")


class RenderService:
    """Overlay timestamped landmark frames onto their corresponding video frames."""

    def __init__(self, config: RenderConfig | None = None) -> None:
        self._config = config or RenderConfig()

    def render_video_landmarks(
        self,
        input_video: Path,
        landmark_frames: Sequence[LandmarkFrame],
        output_video: Path,
    ) -> None:
        """Render landmarks on ``input_video`` and write a new annotated video.

        Landmark frames are matched by ``metadata.index``. Missing landmark frames
        leave the source frame unchanged, and landmark frames beyond the end of the
        video are ignored.
        """
        input_video = Path(input_video)
        output_video = Path(output_video)

        if not input_video.is_file():
            raise FileNotFoundError(f"Input video does not exist: {input_video}")
        if input_video.resolve() == output_video.resolve():
            raise ValueError("input_video and output_video must be different paths")

        frames_by_index = self._index_landmark_frames(landmark_frames)
        output_video.parent.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(input_video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open input video: {input_video}")

        writer: cv2.VideoWriter | None = None
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if width <= 0 or height <= 0:
                raise RuntimeError("Input video reported an invalid frame size")
            if not np.isfinite(fps) or fps <= 0:
                fps = self._config.fallback_fps

            fourcc = cv2.VideoWriter_fourcc(*self._config.codec)
            writer = cv2.VideoWriter(
                str(output_video),
                fourcc,
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(
                    f"Could not open output video for writing: {output_video}"
                )

            frame_index = 0
            while True:
                ok, image = capture.read()
                if not ok:
                    break

                landmark_frame = frames_by_index.get(frame_index)
                if landmark_frame is not None:
                    self._draw_landmark_frame(image, landmark_frame)

                writer.write(image)
                frame_index += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()

    def _index_landmark_frames(
        self,
        landmark_frames: Sequence[LandmarkFrame],
    ) -> dict[int, LandmarkFrame]:
        indexed: dict[int, LandmarkFrame] = {}
        for landmark_frame in landmark_frames:
            frame_index = int(landmark_frame.metadata.index)
            if frame_index < 0:
                raise ValueError(f"Landmark frame index cannot be negative: {frame_index}")
            if frame_index in indexed:
                raise ValueError(f"Duplicate landmark frame index: {frame_index}")
            indexed[frame_index] = landmark_frame
        return indexed

    def _draw_landmark_frame(
        self,
        image: NDArray[np.uint8],
        landmark_frame: LandmarkFrame,
    ) -> None:
        height, width = image.shape[:2]

        grouped: dict[tuple[str, int], list[RawLandmark]] = defaultdict(list)
        for landmark in landmark_frame.landmarks:
            if self._is_visible(landmark):
                grouped[(landmark.source, landmark.instance_id)].append(landmark)

        for (source, _instance_id), landmarks in grouped.items():
            points_by_name = {
                landmark.name: self._to_pixel(landmark, width, height)
                for landmark in landmarks
            }

            # Lines first, then joints, so points remain visually prominent.
            for start_name, end_name in self._config.connections_by_source.get(
                source, ()
            ):
                start = points_by_name.get(start_name)
                end = points_by_name.get(end_name)
                if start is not None and end is not None:
                    cv2.line(
                        image,
                        start,
                        end,
                        self._config.connection_color,
                        self._config.line_thickness,
                        cv2.LINE_AA,
                    )

            for landmark in landmarks:
                cv2.circle(
                    image,
                    points_by_name[landmark.name],
                    self._joint_radius(landmark.z),
                    self._side_color(landmark.side),
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

    def _is_visible(self, landmark: RawLandmark) -> bool:
        if not np.isfinite(landmark.x) or not np.isfinite(landmark.y):
            return False
        if landmark.confidence is None:
            return True
        return (
            np.isfinite(landmark.confidence)
            and landmark.confidence >= self._config.min_confidence
        )

    @staticmethod
    def _to_pixel(
        landmark: RawLandmark,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        # Clamp normalized coordinates to the image so slightly out-of-range model
        # output cannot produce invalid or unexpectedly distant drawing positions.
        x = float(np.clip(landmark.x, 0.0, 1.0))
        y = float(np.clip(landmark.y, 0.0, 1.0))
        return (
            int(round(x * (width - 1))),
            int(round(y * (height - 1))),
        )

    def _joint_radius(self, z: float | None) -> int:
        if z is None or not np.isfinite(z):
            return self._config.min_joint_radius

        clamped_z = float(np.clip(z, self._config.z_min, self._config.z_max))
        normalized_z = (clamped_z - self._config.z_min) / (
            self._config.z_max - self._config.z_min
        )

        proximity = 1.0 - normalized_z if self._config.z_near_is_negative else normalized_z
        radius_range = self._config.max_joint_radius - self._config.min_joint_radius
        return int(round(self._config.min_joint_radius + proximity * radius_range))

    def _side_color(self, side: Side) -> Color:
        if side == Side.LEFT:
            return self._config.left_color
        if side == Side.RIGHT:
            return self._config.right_color
        return self._config.unknown_color
