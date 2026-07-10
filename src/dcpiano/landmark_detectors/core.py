from abc import ABC, abstractmethod

from dcpiano.types.video import VideoFrame
from dcpiano.types.landmark import RawLandmark


class LandmarkDetector(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def detect(self, frame: VideoFrame) -> list[RawLandmark]:
        ...