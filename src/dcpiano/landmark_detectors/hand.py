from enum import StrEnum
from dataclasses import dataclass

from dcpiano.landmark_detectors.core import LandmarkDetector
from dcpiano.types.landmark import RawLandmark
from dcpiano.types.video import VideoFrame


class HandLandmarks(StrEnum):
    WRIST = "WRIST"
    THUMB_CMC = "THUMB_CMC"
    THUMB_MCP = "THUMB_MCP"
    THUMB_IP = "THUMB_IP"
    THUMB_TIP = "THUMB_TIP"
    INDEX_FINGER_MCP = "INDEX_FINGER_MCP"
    INDEX_FINGER_PIP = "INDEX_FINGER_PIP"
    INDEX_FINGER_DIP = "INDEX_FINGER_DIP"
    INDEX_FINGER_TIP = "INDEX_FINGER_TIP"
    MIDDLE_FINGER_MCP = "MIDDLE_FINGER_MCP"
    MIDDLE_FINGER_PIP = "MIDDLE_FINGER_PIP"
    MIDDLE_FINGER_DIP = "MIDDLE_FINGER_DIP"
    MIDDLE_FINGER_TIP = "MIDDLE_FINGER_TIP"
    RING_FINGER_MCP = "RING_FINGER_MCP"
    RING_FINGER_PIP = "RING_FINGER_PIP"
    RING_FINGER_DIP = "RING_FINGER_DIP"
    RING_FINGER_TIP = "RING_FINGER_TIP"
    PINKY_MCP = "PINKY_MCP"
    PINKY_PIP = "PINKY_PIP"
    PINKY_DIP = "PINKY_DIP"
    PINKY_TIP = "PINKY_TIP"
    
    
@dataclass
class HandLandmarkConfig:
    max_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False
    

class HandLandmarkDetector(LandmarkDetector):
    name = "HandLandmarkDetector"
    
    def __init__(self, config: HandLandmarkConfig, landmarks: HandLandmarks) -> None:
        pass
    
    def detect(self, frame: VideoFrame) -> list[RawLandmark]:
        pass