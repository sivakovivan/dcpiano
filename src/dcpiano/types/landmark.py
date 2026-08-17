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

    def generate_landmark_id(self) -> str:
        """Return the stable identifier used to index this raw landmark."""
        parts = [self.source, self.name]
        if self.side != Side.UNKNOWN:
            parts.append(self.side.value)
        return ":".join(parts)


@dataclass
class LandmarkFrame:
    metadata: FrameMetadata
    landmarks: dict[str, RawLandmark]
