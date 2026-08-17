import yaml
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Config:
    keyboard_width_mm: float
    keyboard_height_mm: float
    
    def save_effective_config(self, path: Path):
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)