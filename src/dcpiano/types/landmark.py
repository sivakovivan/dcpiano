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


@dataclass
class LandmarkStreamEntry:
    """The state of one landmark at one point in the frame timeline."""

    metadata: FrameMetadata
    exists: bool
    x: float | None = None
    y: float | None = None
    z: float | None = None
    confidence: float | None = None


@dataclass
class LandmarkStream:
    """All time-ordered observations belonging to one stable landmark id."""

    landmark_id: str
    name: str
    source: str
    instance_id: int
    side: Side
    entries: list[LandmarkStreamEntry]


@dataclass
class LandmarkStreams:
    """Landmark-centric representation of a sequence of landmark frames."""

    landmarks: dict[str, LandmarkStream]
