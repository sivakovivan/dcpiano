from dcpiano.types.landmark import LandmarkFrame
from dcpiano.types.video import VideoFrame
from dcpiano.landmark_detectors.core import LandmarkDetector


class LandmarkService:
    def __init__(self, detectors: list[LandmarkDetector]):
        self.detectors = detectors
        
    
    def generate_landmark_frame(self, frame: VideoFrame) -> LandmarkFrame:
        pass