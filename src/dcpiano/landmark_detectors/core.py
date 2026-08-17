from abc import ABC, abstractmethod

from dcpiano.types.video import VideoFrame
from dcpiano.types.landmark import RawLandmark


class LandmarkDetector(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...
        
    @property
    @abstractmethod
    def render_connections(self) -> str:
        ...

    def init(self) -> None:
        ...
    
    @abstractmethod
    def reset_video(self) -> None:
        ...
    
    @abstractmethod
    def detect(self, frame: VideoFrame, existing_landmarks: dict[str, RawLandmark]) -> dict[str, RawLandmark]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
