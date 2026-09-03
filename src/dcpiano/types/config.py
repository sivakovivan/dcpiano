import yaml
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Config:
    keyboard_width_mm: float
    keyboard_height_mm: float
    skip_calibration: bool = False
    tracking_only: bool = False
    create_rendered_video: bool = True
    
    def save_effective_config(self, path: Path):
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
