import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class VideoMetadata:
    width: int
    height: int
    path: Path
    fps: float
    duration: float
    
    
@dataclass
class FrameMetadata:
    index: int
    timestamp: float
    
    
@dataclass
class VideoFrame:
    metadata: FrameMetadata
    frame: np.ndarray