from dataclasses import dataclass
from enum import StrEnum

from dcpiano.types.video import FrameMetadata


@dataclass
class Side(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


@dataclass
class RawLandmark:
    name: str
    source: str
    
    instance_id: int
    
    side: Side
    
    x: float
    y: float
    z: float | None
    
    confidence: float | None = None


@dataclass
class LandmarkFrame:
    metadata: FrameMetadata
    landmarks: list[RawLandmark]