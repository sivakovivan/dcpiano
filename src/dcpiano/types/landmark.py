from dataclasses import dataclass

from dcpiano.types.video import FrameMetadata


@dataclass
class RawLandmark:
    name: str
    source: str
    x_px: float
    y_px: float


@dataclass
class LandmarkFrame:
    metadata: FrameMetadata
    landmarks: list[RawLandmark]