import json
from enum import StrEnum
from pathlib import Path
from typing import Sequence

from dcpiano.types.landmark import (
    LandmarkFrame,
    LandmarkStream,
    LandmarkStreamEntry,
    LandmarkStreams,
    RawLandmark,
)


def _json_encoder(obj):
    if isinstance(obj, StrEnum):
        return obj.value
    return obj.__dict__


class LandmarkStreamService:
    """Convert frame-centric landmarks into continuous landmark timelines."""

    @classmethod
    def generate_landmark_streams(
        cls,
        landmark_frames: Sequence[LandmarkFrame],
        output_dir: str | Path | None = None,
    ) -> LandmarkStreams:
        cls._validate_frame_order(landmark_frames)

        streams: dict[str, LandmarkStream] = {}
        processed_metadata = []

        for landmark_frame in landmark_frames:
            metadata = landmark_frame.metadata
            processed_metadata.append(metadata)

            # Extend every known stream with an absent sample. Detections in this
            # frame replace that placeholder below.
            for stream in streams.values():
                stream.entries.append(
                    LandmarkStreamEntry(metadata=metadata, exists=False)
                )

            for landmark_id, landmark in landmark_frame.landmarks.items():
                cls._validate_landmark_id(landmark_id, landmark)
                stream = streams.get(landmark_id)

                if stream is None:
                    stream = LandmarkStream(
                        landmark_id=landmark_id,
                        name=landmark.name,
                        source=landmark.source,
                        instance_id=landmark.instance_id,
                        side=landmark.side,
                        entries=[
                            LandmarkStreamEntry(metadata=previous, exists=False)
                            for previous in processed_metadata[:-1]
                        ],
                    )
                    streams[landmark_id] = stream
                    stream.entries.append(cls._present_entry(metadata, landmark))
                else:
                    cls._validate_identity(stream, landmark)
                    stream.entries[-1] = cls._present_entry(metadata, landmark)

        artifact = LandmarkStreams(landmarks=streams)
        if output_dir is not None:
            cls.save_landmark_streams(artifact, output_dir)
        return artifact

    @staticmethod
    def save_landmark_streams(
        landmark_streams: LandmarkStreams, output_dir: str | Path
    ) -> Path:
        """Persist the stream artifact and return its path."""
        output_path = Path(output_dir) / "landmark_streams.json"
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                landmark_streams,
                output_file,
                indent=4,
                default=_json_encoder,
            )
        return output_path

    @staticmethod
    def _present_entry(metadata, landmark: RawLandmark) -> LandmarkStreamEntry:
        return LandmarkStreamEntry(
            metadata=metadata,
            exists=True,
            x=landmark.x,
            y=landmark.y,
            z=landmark.z,
            confidence=landmark.confidence,
        )

    @staticmethod
    def _validate_landmark_id(
        landmark_id: str, landmark: RawLandmark
    ) -> None:
        generated_id = landmark.generate_landmark_id()
        if landmark_id != generated_id:
            raise ValueError(
                f"Landmark key {landmark_id!r} does not match generated id "
                f"{generated_id!r}."
            )

    @staticmethod
    def _validate_identity(
        stream: LandmarkStream, landmark: RawLandmark
    ) -> None:
        identity = (
            landmark.name,
            landmark.source,
            landmark.instance_id,
            landmark.side,
        )
        stream_identity = (
            stream.name,
            stream.source,
            stream.instance_id,
            stream.side,
        )
        if identity != stream_identity:
            raise ValueError(
                f"Landmark id {stream.landmark_id!r} changed identity between frames."
            )

    @staticmethod
    def _validate_frame_order(
        landmark_frames: Sequence[LandmarkFrame],
    ) -> None:
        previous_index: int | None = None
        previous_timestamp: float | None = None
        for landmark_frame in landmark_frames:
            index = landmark_frame.metadata.index
            timestamp = float(landmark_frame.metadata.timestamp)
            if previous_index is not None and index <= previous_index:
                raise ValueError(
                    "LandmarkFrames must be ordered by increasing metadata.index; "
                    f"got {previous_index} followed by {index}."
                )
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(
                    "LandmarkFrames must be ordered by non-decreasing "
                    "metadata.timestamp; "
                    f"got {previous_timestamp} followed by {timestamp}."
                )
            previous_index = index
            previous_timestamp = timestamp
