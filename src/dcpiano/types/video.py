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
class VideoFrame:
    index: int
    timestamp: float
    image: np.ndarray