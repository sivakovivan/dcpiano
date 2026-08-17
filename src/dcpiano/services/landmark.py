import json
from pathlib import Path
from enum import StrEnum

from dcpiano.landmark_detectors.core import LandmarkDetector
from dcpiano.types.landmark import LandmarkFrame, RawLandmark
from dcpiano.types.video import VideoFrame


def landmark_frame_json_encoder(obj):
    if isinstance(obj, StrEnum):
        return obj.value
    return obj.__dict__


class LandmarkService:
    def __init__(self, detectors: list[LandmarkDetector]) -> None:
        self.detectors = detectors

    def generate_landmark_frame(self, frame: VideoFrame) -> LandmarkFrame:
        """Process one frame through every configured detector."""
        landmarks: dict[str, RawLandmark] = {}
        for detector in self.detectors:
            detected_landmarks = detector.detect(frame)
            duplicate_ids = landmarks.keys() & detected_landmarks.keys()
            if duplicate_ids:
                duplicate_id = sorted(duplicate_ids)[0]
                raise ValueError(f"Duplicate raw landmark id: {duplicate_id}")
            landmarks.update(detected_landmarks)

        return LandmarkFrame(metadata=frame.metadata, landmarks=landmarks)

    def generate_landmark_frames(
        self,
        frames: list[VideoFrame],
        output_dir: str,
        reset_detectors: bool = True,
    ) -> list[LandmarkFrame]:
        """Process a video's consecutive frames and preserve per-frame metadata.

        Detectors are reused for the whole sequence. This is required for
        MediaPipe VIDEO-mode tracking to carry state from one frame to the next.
        """
        self._validate_frame_order(frames)

        frame_count = len(frames)

        if reset_detectors:
            [detector.reset_video() for detector in self.detectors]

        landmark_frames: list[LandmarkFrame] = []
        for index, frame in enumerate(frames):
            landmark_frames.append(self.generate_landmark_frame(frame))
            if index % 100 == 0:
                print(f"[LandmarkService] Processing frame {index} / {frame_count}")

        self.save_landmark_frames(landmark_frames, output_dir)

        return landmark_frames

    def save_landmark_frames(
        self, landmark_frames: list[LandmarkFrame], output_dir: str
    ) -> None:
        """Save landmark frames to a JSON file."""
        output_path = Path(output_dir) / "landmark_frames.json"
        with open(output_path, "w") as f:
            json.dump(
                [frame.__dict__ for frame in landmark_frames],
                f,
                indent=4,
                default=landmark_frame_json_encoder,
            )

    def close(self) -> None:
        [detector.close() for detector in self.detectors]


    @staticmethod
    def _validate_frame_order(frames: list[VideoFrame]) -> None:
        previous_index: int | None = None
        previous_timestamp: float | None = None

        for frame in frames:
            index = frame.metadata.index
            timestamp = float(frame.metadata.timestamp)

            if previous_index is not None and index <= previous_index:
                raise ValueError(
                    "VideoFrames must be ordered by increasing metadata.index; "
                    f"got {previous_index} followed by {index}."
                )
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(
                    "VideoFrames must be ordered by non-decreasing "
                    "metadata.timestamp; "
                    f"got {previous_timestamp} followed by {timestamp}."
                )

            previous_index = index
            previous_timestamp = timestamp
